#!/usr/bin/env python
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import logging
logging.basicConfig(
    filename='create_data.log', encoding='utf-8', level=logging.DEBUG,
    format='[%(asctime)s %(levelname)s %(name)s]: %(message)s',
    datefmt='%Y/%m/%d %I:%M:%S %p')
logger = logging.getLogger(__name__)

from src.datamodules.components.odd_reader import ActsReader

import h5pickle as h5py
from pathlib import Path
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def convert_to_hdf5(reader: ActsReader, idx: int, outdir: str, prefix: str = "v1"):
    outname = f"{outdir}/{idx:09d}.h5"
    if Path(outname).exists():
        return

    h5file = h5py.File(outname, "w")
    grp = h5file.create_group(prefix)

    evt = grp.create_group(f"{idx:09d}")
    spacepoints, particles, true_edges = reader.read_event(idx)
    particles = particles[(particles['q'] != 0) & (particles['p_eta'].abs() < 4)]
    evt.create_dataset("particles", data=particles.to_numpy())
    evt.create_dataset("spacepoints", data=spacepoints.to_numpy())
    evt.create_dataset("true_edges", data=true_edges)
    h5file.close()

def convert_to_parquet(reader: ActsReader, idx: int, outdir: str, prefix: str = "v1"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    outname = Path(f"{outdir}/spacepoints/{idx:09d}.parquet")
    if outname.exists():
        return

    spacepoints, particles, true_edges = reader.read_event(idx)
    particles = particles[(particles['q'] != 0) & (particles['p_eta'].abs() < 4)]
    particles = pa.Table.from_pandas(particles, preserve_index=False)
    particles = particles.append_column("evtid", pa.array([idx] * len(particles)))
    pq.write_table(particles, f"{outdir}/particles/{idx}.parquet")

    spacepoints = spacepoints[["measurement_id", "particle_id", "umid", "x", "y", "z", "p_eta", "nhits", "p_pt"]]
    spacepoints = pa.Table.from_pandas(spacepoints, preserve_index=False)
    spacepoints = spacepoints.append_column("evtid", pa.array([idx] * len(spacepoints)))
    pq.write_table(spacepoints, f"{outdir}/spacepoints/{idx}.parquet")

    true_edges = pa.table([pa.array(true_edges[0]), pa.array(true_edges[1])], names=["src", "dst"])
    true_edges = true_edges.append_column("evtid", pa.array([idx] * len(true_edges)))
    pq.write_table(true_edges, f"{outdir}/true_edges/{idx}.parquet")

def convert(inputdir: str, outputdir: str, prefix: str = "v1", **kwargs):
    print(f"inputdir: {inputdir}")
    print(f"outputdir: {outputdir}")
    reader = ActsReader(inputdir=inputdir,
                        outputdir=outputdir, **kwargs)
    num_workers = kwargs.get("num_workers", 1)

    tot_evts = reader.nevts
    Path(f"{outputdir}/particles").mkdir(parents=True, exist_ok=True)
    Path(f"{outputdir}/spacepoints").mkdir(parents=True, exist_ok=True)
    Path(f"{outputdir}/true_edges").mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Converting events", total=tot_evts)) as _:
        Parallel(n_jobs=num_workers)(
            delayed(convert_to_parquet)(reader, evt_idx, outputdir, prefix) for evt_idx in range(tot_evts)
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data Reader')
    add_arg = parser.add_argument
    add_arg('inputdir', help='Input directory')
    add_arg('outputdir', help='Output directory')
    add_arg('-w', '--num_workers', type=int, default=1)

    convert(**vars(parser.parse_args()))
