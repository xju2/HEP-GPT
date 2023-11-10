"""Read TrackML data.
https://competitions.codalab.org/competitions/20112

Assume the files are like:
detectors.csv
trackml/event000001000-hits.csv.gz
trackml/event000001000-cells.csv.gz
trackml/event000001000-particles.csv.gz
trackml/event000001000-truth.csv.gz
"""

import os
from typing import Union
from pathlib import Path
import glob
import re
import pickle
import concurrent.futures
import logging

import numpy as np
import pandas as pd

EVENT_START_TOKEN = 1
EVENT_END_TOKEN = 2
TRACK_START_TOKEN = 3
TRACK_END_TOKEN = 4
TRACK_HOLE_TOKEN = 5
MASK_TOKEN = 6
PAD_TOKEN = 7
UNKNOWN_TOKEN = 8
block_size = 18 + 2  # maximum number of hits for one track + START + END

class TrackMLReader(object):
    def __init__(self, inputdir: Union[str, Path],
                 outputdir: Union[str, Path],
                 name="TrackMLReader",
                 is_codalab_data: bool = True,
                 num_workers: int = 1,
                 min_truth_hits: int = 4,
                 with_padding: bool = False,
                 outname_prefix: str = "",
                 ) -> None:
        self.inputdir = Path(inputdir) if isinstance(inputdir, str) else inputdir
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(f"Input directory {self.inputdir} does not exist or is not a directory.")

        self.outputdir = Path(outputdir) if isinstance(outputdir, str) else outputdir
        if not self.outputdir.exists():
            self.outputdir.mkdir(parents=True)

        # check one file in the inputdir directory to see the file suffix
        for x in self.inputdir.iterdir():
            if x.is_file():
                filename = x
                break

        self.suffix = filename.suffix
        # count how many events in the directory
        all_evts = glob.glob(os.path.join(
            self.inputdir, "event*-hits.csv*"))

        self.nevts = len(all_evts)
        pattern = "event([0-9]*)-hits" + self.suffix

        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip())
            for x in all_evts])

        print("total {} events in directory: {}".format(
            self.nevts, self.inputdir))

        detector_path = os.path.join(self.inputdir, "../detector.csv")
        # load detector info
        detector = pd.read_csv(detector_path)
        if is_codalab_data:
            detector = detector.rename(columns={"pitchX": 'pitch_u',
                                                "pitchY": "pitch_v"})

        self.build_detector_vocabulary(detector)
        self.detector = detector

        self.name = name
        self.num_workers = num_workers
        self.min_truth_hits = min_truth_hits
        self.with_padding = with_padding
        self.outname_prefix = outname_prefix + "_" if outname_prefix else ""

    def build_detector_vocabulary(self, detector):
        """Build the detector vocabulary."""
        detector_umid = np.stack([detector.volume_id, detector.layer_id, detector.module_id], axis=1)
        umid_dict = {}
        index = 1
        for i in detector_umid:
            umid_dict[tuple(i)] = index
            index += 1
        self.umid_dict = umid_dict
        self.num_modules = len(detector_umid)
        pixel_moudels = [k for k in umid_dict.keys() if k[0] in [7, 8, 9]]
        self.num_pixel_modules = len(pixel_moudels)
        # Inverting the umid_dict
        self.umid_dict_inv = {v: k for k, v in umid_dict.items()}

    def prepare_data(self, start_evt: int, end_evt: int,
                     min_truth_hits: int = 4,
                     with_padding: bool = False,
                     offset_umid: int = UNKNOWN_TOKEN):
        """Prepare the data for training."""
        if end_evt <= start_evt or self.nevts < end_evt:
            logging.error("invalid start_evt {} or end_evt {}".format(
                start_evt, end_evt))
            return None

        if self.num_workers > 1:
            print("using {} workers to process the data".format(self.num_workers))

            # use the concurrent futures to speed up the processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                all_tracks = executor.map(self.convert_one_event, range(start_evt, end_evt))
        else:
            all_tracks = [self.convert_one_event(idx) for idx in range(start_evt, end_evt)]

        all_tracks = [item for sublist in all_tracks for item in sublist]
        all_tracks = np.array(all_tracks, dtype=np.uint16)
        return all_tracks

    def convert_one_event(self, idx: int):
        """Convert a given event index to a list of tokens."""

        hits = self.read_event(reader.all_evtids[idx])
        hits = hits[hits.nhits >= self.min_truth_hits]

        vlid_groups = hits.groupby("particle_id")
        tracks = [[TRACK_START_TOKEN] + vlid_groups.get_group(vlid).umid.map(
            lambda x: x + UNKNOWN_TOKEN).to_list() + [TRACK_END_TOKEN]
            for vlid in vlid_groups.groups.keys()]

        if self.with_padding:
            tracks = [track + [PAD_TOKEN] * (block_size - len(track)) for track in tracks]

        # flatten the list
        tracks = [item for sublist in tracks for item in sublist]
        return tracks

    def read_event(self, evtid: int = None) -> bool:
        """Read one event from the input directory"""

        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print("read event {}".format(evtid))

        prefix = os.path.join(self.inputdir, "event{:09d}".format(evtid))
        hit_fname = "{}-hits{}".format(prefix, self.suffix)
        cell_fname = "{}-cells{}".format(prefix, self.suffix)
        particle_fname = "{}-particles{}".format(prefix, self.suffix)
        truth_fname = "{}-truth{}".format(prefix, self.suffix)

        # read all files
        hits = pd.read_csv(hit_fname)

        # read truth info about hits and particles
        truth = pd.read_csv(truth_fname)
        particles = pd.read_csv(particle_fname)
        truth = truth.merge(particles, on='particle_id', how='left')

        hits = hits.merge(truth[['hit_id', 'particle_id', "nhits"]],
                          on='hit_id')

        # add detector unique module ID
        vlid_groups = hits.groupby(['volume_id', 'layer_id', 'module_id'])
        hits = pd.concat([vlid_groups.get_group(vlid).assign(umid=self.umid_dict[vlid])
                          for vlid in vlid_groups.groups.keys()])
        return hits

    def run(self, config: dict = {"train": (0, 2), "val": (2, 3)}):
        """Process tot_evts events in the input directory. Save the data in the output directory.
        config: {"train": (0, 2), "val": (2, 3)}
            train: (start_evt, end_evt)
            val: (start_evt, end_evt)
        """

        for k, v in config.items():
            num_evts = v[1] - v[0]
            print("processing {:,} events for {}".format(num_evts, k))
            data = self.prepare_data(v[0], v[1])
            if data is not None:
                data.tofile(self.outputdir / f"{self.outname_prefix}evt{num_evts}_{k}.bin")
                print("saved {:,} tokens for {}".format(data.shape[0], k))

        # save the meta information
        stoi = self.umid_dict
        itos = self.umid_dict_inv

        meta = {
            'vocab_size': self.num_modules + UNKNOWN_TOKEN + 1,
            'itos': itos,
            'stoi': stoi,
        }
        logging.info("vocab size: {}".format(meta['vocab_size']))
        with open(self.outputdir / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TrackML Reader')
    add_arg = parser.add_argument
    add_arg('inputdir', help='Input directory')
    add_arg('outputdir', help='Output directory')
    add_arg('-w', '--num_workers', type=int, default=1,)
    add_arg("--num-train", type=int, default=20, help="Number of training events")
    add_arg("--num-val", type=int, default=20, help="Number of validation events")
    add_arg("--num-test", type=int, default=0, help="Number of test events")
    add_arg("--padding", action="store_true", help="Whether to pad the tracks to the same length")
    add_arg("--prefix", type=str, default="", help="Prefix for the output files")
    args = parser.parse_args()

    reader = TrackMLReader(args.inputdir, args.outputdir,
                           num_workers=args.num_workers,
                           with_padding=args.padding,
                           outname_prefix=args.prefix
                           )
    reader.run(
        {
            "train": (0, args.num_train),
            "val": (args.num_train, args.num_train + args.num_val),
            "test": (args.num_train + args.num_val, args.num_train + args.num_val + args.num_test),
        }
    )
