import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.pylogger import get_pylogger

from pathlib import Path
import numpy as np
import pandas as pd

import torch
from model import GPTConfig, GPT

TRACK_START_TOKEN = 3
TRACK_END_TOKEN = 4
UNKNOWN_TOKEN = 8

log = get_pylogger(__name__)

def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.ckpt_path
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

    gpt_config = GPTConfig(**cfg.model)
    model = GPT(gpt_config)
    optimizer = model.configure_optimizers(cfg.optimizer)

    fabric = hydra.utils.instantiate(cfg.fabric)
    fabric.launch()

    fabric.seed_everything(cfg.seed)
    model, optimizer = fabric.setup(model, optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
    }
    fabric.load(ckpt_path, state)

    data = np.memmap(cfg.data_path, dtype=np.int64, mode="r")
    track_start_indices = np.argwhere(data == TRACK_START_TOKEN).squeeze()
    track_end_indices = np.argwhere(data == TRACK_END_TOKEN).squeeze()
    num_of_hits = track_end_indices - track_start_indices

    sel_num_of_hits = num_of_hits >= cfg.min_num_of_hits

    # select tracks with more than min_num_of_hits
    track_start_indices = track_start_indices[sel_num_of_hits]
    track_end_indices = track_end_indices[sel_num_of_hits]
    num_of_hits = num_of_hits[sel_num_of_hits]

    all_valid_tracks = [data[x:y + 1] for x, y in zip(track_start_indices, track_end_indices)]

    # prepare the seed hits with num_of_seeds.
    seeds = np.array([track[:cfg.num_of_seeds] for track in all_valid_tracks])
    seeds = torch.from_numpy(seeds)

    # variable number of hits for each track
    truth_modules = [track[cfg.num_of_seeds:] for track in all_valid_tracks]

    # generate maximum-length modules for each seed modules
    next_modules = model.generate(seeds, cfg.max_new_tokens)

    # read detector information
    detector = pd.read_csv(cfg.detector_path)

    seed_positions = detector.iloc[seeds - UNKNOWN_TOKEN - 1][["cx", "cy", "cz"]].to_numpy()
    next_positions = detector.iloc[next_modules - UNKNOWN_TOKEN - 1][["cx", "cy", "cz"]].to_numpy()
    true_next_positions = detector.iloc[truth_modules - UNKNOWN_TOKEN - 1][["cx", "cy", "cz"]].to_numpy()


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def evaluation(cfg : DictConfig) -> None:
    if cfg.dry_run:
        log.info(OmegaConf.to_yaml(cfg))
        return

    main(cfg)

if __name__ == "__main__":
    evaluation()
