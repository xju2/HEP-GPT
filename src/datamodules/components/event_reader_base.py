"""Base class for reading events from the input directory.
It provides multi-threading support to speed up the processing, and
functions to convert the hits to tokens.
"""

from typing import Union, Optional
from pathlib import Path
import itertools
import concurrent.futures
import pickle

import numpy as np
import pandas as pd

from src.datamodules import reserved_tokens as rt
from src.utils.pylogger import get_pylogger

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

def convert_hits_to_tokens(hits: pd.DataFrame,
                           min_truth_hits: str = 4,
                           with_padding: bool = False,
                           with_event_tokens: bool = False,
                           with_event_padding: bool = False,
                           with_seperate_event_pad_token: bool = False
                           ):
    """Convert hits to track tokens"""
    assert isinstance(hits, pd.DataFrame)
    needed_columns = ["nhits", "particle_id", "umid", ]
    contains_all_columns = all(
        col in hits.columns for col in ["nhits", "particle_id", "volume_id", "layer_id", "module_id", "x", "y", "z"])
    assert contains_all_columns, "hits must contain all columns: {}".format(
        ["nhits", "particle_id", "volume_id", "layer_id", "module_id", "x", "y", "z"])

    hits = hits[hits.nhits >= min_truth_hits]

    vlid_groups = hits.groupby("particle_id")
    tracks = [[rt.TRACK_START_TOKEN] + vlid_groups.get_group(vlid).umid.map(
        lambda x: x + rt.UNKNOWN_TOKEN).to_list() + [rt.TRACK_END_TOKEN]
        for vlid in vlid_groups.groups.keys()]

    if with_padding:
        tracks = [track + [rt.PAD_TOKEN] * (rt.BLOCK_SIZE - len(track)) for track in tracks if len(track) <= rt.BLOCK_SIZE]

    # flatten the list
    tracks = [item for sublist in tracks for item in sublist]

    if with_event_tokens:
        tracks = [rt.EVENT_START_TOKEN] + tracks + [rt.EVENT_END_TOKEN]

        if with_event_padding:
            event_pad_token = rt.EVENT_PAD_TOKEN if with_seperate_event_pad_token else rt.PAD_TOKEN
            tracks = tracks + [event_pad_token] * (rt.EVENT_BLOCK_SIZE - len(tracks)) \
                if len(tracks) <= rt.EVENT_BLOCK_SIZE else tracks[:rt.EVENT_BLOCK_SIZE]

    return tracks


class EventReaderBase(object):
    def __init__(self,
                 inputdir: Union[str, Path],
                 output_dir: Optional[str] = None,
                 outname_prefix: str = "",
                 num_workers: int = 1,
                 min_truth_hits: int = 4,
                 with_padding: bool = False,
                 with_event_tokens: bool = False,
                 with_event_padding: bool = False,
                 with_seperate_event_pad_token: bool = False,
                 name: str = "EventReaderBase",
                 *arg, **kwargs
                 ):
        """Initialize the reader"""
        self.input_dir = Path(inputdir) if isinstance(inputdir, str) else inputdir
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError("Input directory not found: {}".format(inputdir))

        self.outdir = Path(output_dir) if output_dir else self.input_dir / "processed_data"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.outname_prefix = outname_prefix + "_" if outname_prefix else ""

        self.num_workers = num_workers
        self.min_truth_hits = min_truth_hits
        self.with_padding = with_padding
        self.with_event_tokens = with_event_tokens
        self.with_event_padding = with_event_padding
        self.with_seperate_event_pad_token = with_seperate_event_pad_token
        self.name = name

        # attributes to be set in the derived class
        self.nevts = 0
        self.num_modules = 0

    def read_event(self, evt_idx: int = None) -> pd.DataFrame:
        """Read one event from the input directory through the event index

        Return:
            hits: pd.DataFrame
        """
        raise NotImplementedError

    def convert_one_evt(self, idx: int):
        hits = self.read_event(idx)

        tracks = convert_hits_to_tokens(
            hits, self.min_truth_hits, self.with_padding,
            self.with_event_tokens, self.with_event_padding,
            self.with_seperate_event_pad_token)
        return tracks

    def convert_events(self, start_evt: int, end_evt: int):
        if end_evt <= start_evt or self.nevts < end_evt:
            logger.error("invalid start_evt {} or end_evt {}".format(
                start_evt, end_evt))
            return None

        if self.config.num_workers > 1:
            print("using {} workers to process the data".format(self.config.num_workers))

            # use the concurrent futures to speed up the processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                all_tracks = executor.map(self.convert_one_event, range(start_evt, end_evt))
        else:
            all_tracks = [self.convert_one_event(idx) for idx in range(start_evt, end_evt)]

        all_tracks = [item for sublist in all_tracks for item in sublist]
        all_tracks = np.array(all_tracks, dtype=np.uint16)
        return all_tracks

    def run(self, config: dict = {"train": (0, 2), "val": (2, 3)}):
        """Process tot_evts events in the input directory. Save the data in the output directory.
        config: {"train": (0, 2), "val": (2, 3)}
            train: (start_evt, end_evt)
            val: (start_evt, end_evt)
        """
        for k, v in config.items():
            num_evts = v[1] - v[0]
            print("processing {:,} events for {}".format(num_evts, k))
            data = self.convert_events(v[0], v[1])
            if data is not None:
                data.tofile(self.outputdir / f"{self.outname_prefix}evt{num_evts}_{k}.bin")
                print("saved {:,} tokens for {}".format(data.shape[0], k))

        # save the meta information

        meta = {
            'vocab_size': self.num_modules + rt.UNKNOWN_TOKEN + 1,
        }
        logger.info("vocab size: {}".format(meta['vocab_size']))
        with open(self.outputdir / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
