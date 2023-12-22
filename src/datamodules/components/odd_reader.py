"""Read csv files obtained from ACTS with the Open Data Detector.

The files are CSV files organized as follows:
acts/event000001000-hits.csv
acts/event000001000-measurements.csv
acts/event000001000-meas2hits.csv
acts/event000001000-spacepoints.csv
acts/event000001000-particles_final.csv
acts/event000001000-cells.csv
"""

import os
import re
import glob
import itertools
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.pylogger import get_pylogger
from src.datamodules.components.event_reader_base import EventReaderBase

logger = get_pylogger(__name__)

def make_true_edges(hits):
    hit_list = hits.groupby(['particle_id', 'geometry_id'], sort=False)['index'] \
        .agg(lambda x: list(x)).groupby(level=0) \
        .agg(lambda x: list(x))

    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    layerless_true_edges = np.array(e).T
    return layerless_true_edges


class ActsReader(EventReaderBase):
    def __init__(self,
                 overwrite: bool = False,
                 spname: str = "spacepoint",
                 *arg, **kwargs):
        """Initialize the reader"""
        super().__init__(*arg, **kwargs)

        self.overwrite = overwrite
        self.spname = spname

        # count how many events in the directory
        all_evts = glob.glob(os.path.join(
            self.inputdir, "event*-{}.csv".format(spname)))

        pattern = "event([0-9]*)-{}.csv".format(spname)
        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip())
            for x in all_evts])
        self.nevts = len(self.all_evtids)
        print("total {} events in directory: {}".format(
            self.nevts, self.inputdir))

        if self.nevts == 0:
            raise ValueError("No events found in {}".format(self.inputdir))

        detector_path = os.path.join(self.inputdir, "../detector.csv")
        # load detector info
        detector = pd.read_csv(detector_path)
        self.detector = detector
        self.build_detector_vocabulary(detector)

    def build_detector_vocabulary(self, detector: pd.DataFrame):
        """Build the detector vocabulary for the reader"""
        assert "geometry_id" in detector.columns, "geometry_id not in detector.csv"

        detector_umid = detector.geometry_id.unique()
        umid_dict = {}
        index = 1
        for i in detector_umid:
            umid_dict[i] = index
            index += 1
        self.umid_dict = umid_dict
        self.num_modules = len(detector_umid)
        # Inverting the umid_dict
        self.umid_dict_inv = {v: k for k, v in umid_dict.items()}

    def read_event(self, evt_idx: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Read one event from the input directory through the event index

        Return:
            hits: pd.DataFrame, hits information
        """
        if (evt_idx is None or evt_idx < 0) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print(f"read event {evtid}.")
        else:
            evtid = self.all_evtids[evt_idx]

        # construct file names for each csv file for this event
        prefix = os.path.join(self.inputdir, "event{:09d}".format(evtid))
        hit_fname = "{}-hits.csv".format(prefix)
        measurements_fname = "{}-measurements.csv".format(prefix)
        measurements2hits_fname = "{}-measurement-simhit-map.csv".format(prefix)
        sp_fname = '{}-{}.csv'.format(prefix, self.spname)
        p_name = '{}-particles_final.csv'.format(prefix)

        # read hit files
        hits = pd.read_csv(hit_fname)
        hits = hits[hits.columns[:-1]]
        hits = hits.reset_index().rename(columns={'index': 'hit_id'})

        # read measurements, maps to hits, and spacepoints
        measurements = pd.read_csv(measurements_fname)
        meas2hits = pd.read_csv(measurements2hits_fname)
        sp = pd.read_csv(sp_fname)

        # add geometry_id to space points
        vlid_groups = sp.groupby(["geometry_id"])
        sp = pd.concat([vlid_groups.get_group(x).assign(umid=self.umid_dict[x])
                        for x in vlid_groups.groups.keys()])
        logger.info(sp.columns)

        # read particles and add more variables for performance evaluation
        particles = pd.read_csv(p_name)
        pt = np.sqrt(particles.px**2 + particles.py**2)
        momentum = np.sqrt(pt**2 + particles.pz**2)
        theta = np.arccos(particles.pz / momentum)
        eta = -np.log(np.tan(0.5 * theta))
        radius = np.sqrt(particles.vx**2 + particles.vy**2)
        particles = particles.assign(p_pt=pt, p_radius=radius, p_eta=eta)

        # read cluster information
        cell_fname = '{}-cells.csv'.format(prefix)
        cells = pd.read_csv(cell_fname)
        if cells.shape[0] > 0:
            # calculate cluster shape information
            direction_count_u = cells.groupby(['hit_id']).channel0.agg(['min', 'max'])
            direction_count_v = cells.groupby(['hit_id']).channel1.agg(['min', 'max'])
            nb_u = direction_count_u['max'] - direction_count_u['min'] + 1
            nb_v = direction_count_v['max'] - direction_count_v['min'] + 1
            hit_cells = cells.groupby(['hit_id']).value.count().values
            hit_value = cells.groupby(['hit_id']).value.sum().values
            # as I don't access to the rotation matrix and the pixel pitches,
            # I can't calculate cluster's local/global position
            sp = sp.assign(len_u=nb_u, len_v=nb_v, cell_count=hit_cells, cell_val=hit_value)


        sp_hits = sp.merge(meas2hits, on='measurement_id', how='left').merge(
            hits[["hit_id", "particle_id"]], on='hit_id', how='left')
        sp_hits = sp_hits.merge(
            particles[['particle_id', 'vx', 'vy', 'vz', 'p_pt', 'p_eta']], on='particle_id', how='left')
        num_hits = sp_hits.groupby(['particle_id']).hit_id.count()
        sp_hits = sp_hits.merge(num_hits.to_frame(name='nhits'), on='particle_id', how='left')

        r = np.sqrt(sp_hits.x**2 + sp_hits.y**2)
        phi = np.arctan2(sp_hits.y, sp_hits.x)
        sp_hits = sp_hits.assign(r=r, phi=phi)

        sp_hits = sp_hits.assign(R=np.sqrt(
            (sp_hits.x - sp_hits.vx)**2
            + (sp_hits.y - sp_hits.vy)**2
            + (sp_hits.z - sp_hits.vz)**2))
        sp_hits = sp_hits.sort_values('R').reset_index(
            drop=True).reset_index(drop=False)

        true_edges = make_true_edges(sp_hits)
        self.particles = particles
        self.clusters = measurements
        self.spacepoints = sp_hits
        self.true_edges = true_edges
        self.evtid = evtid

        return sp_hits, particles, true_edges
