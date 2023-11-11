from typing import Optional
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl

# local imports
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class TrackMLDataSet(Dataset):
    def __init__(self, inputfile: str,
                 block_size: int,
                 do_randomize: bool = False,
                 transform=None,
                 target_transform=None,
                 name="TrackDataSet",
                 **kwargs):
        self.inputfile = inputfile
        self.block_size = block_size
        self.do_randomize = do_randomize

        self.transform = transform
        self.target_transform = target_transform
        self.name = name

        self.data = np.memmap(inputfile, dtype=np.uint16, mode='r')
        log.info(f"Total number of tokens in {self.name} dataset: {len(self.data):,d}")
        log.info("Total number of batches in {} dataset: {:,d}".format(self.name, len(self)))
        log.info("Do randomize: {}".format(self.do_randomize))

    def __len__(self):
        return self.data.shape[0] // self.block_size

    def __getitem__(self, idx: int):
        data, block_size = self.data, self.block_size

        if self.do_randomize:
            ix = torch.randint(len(data) - block_size, (1,)).item()
            x = torch.from_numpy((data[ix: ix + block_size]).astype(np.int64))
            y = torch.from_numpy((data[ix + 1 : ix + 1 + block_size]).astype(np.int64))
        else:
            start_idx = idx * block_size
            x = torch.from_numpy((data[start_idx : start_idx + block_size]).astype(np.int64))
            y = torch.from_numpy((data[start_idx + 1 : start_idx + 1 + block_size]).astype(np.int64))
        return x, y


class TrackMLDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: str,
                 val_data: str,
                 test_data: str,
                 meta_data: str,
                 batch_size: int,
                 block_size: int,
                 num_workers: int = 1,
                 do_randomize: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'shuffle': True,
            'drop_last': True,
        }
        self.read_meta_data()

    def read_meta_data(self):
        if Path(self.hparams.meta_data).exists():
            with open(self.hparams.meta_data, "rb") as f:
                meta_data = pickle.load(f)
            self.vocab_size = meta_data["vocab_size"]
        else:
            log.error(f"meta data file {self.hparams.meta_data} does not exist.")
            self.vocab_size = 0

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = TrackMLDataSet(self.hparams.train_data,
                                                self.hparams.block_size,
                                                self.hparams.do_randomize,
                                                name="train")
            self.val_dataset = TrackMLDataSet(self.hparams.val_data,
                                              self.hparams.block_size,
                                              self.hparams.do_randomize,
                                              name="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = TrackMLDataSet(self.hparams.test_data,
                                               self.hparams.block_size,
                                               self.hparams.do_randomize,
                                               name="test")
        if stage == "predict":
            raise NotImplementedError("predict stage is not implemented yet.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader_kwargs)
