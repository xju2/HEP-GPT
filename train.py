import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from lightning.fabric import Fabric

from model import GPTConfig, GPT

import torch
torch.set_float32_matmul_precision("medium")
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrackDataSet(Dataset):
    def __init__(self, inputfile: str,
                 batch_size: int, block_size: int,
                 do_randomize: bool = False,
                 transform=None, target_transform=None,
                 name="TrackDataSet"):
        self.inputfile = inputfile
        self.batch_size = batch_size
        self.block_size = block_size
        self.do_randomize = do_randomize

        self.transform = transform
        self.target_transform = target_transform
        self.name = name

        self.data = np.memmap(inputfile, dtype=np.uint16, mode='r')
        print(f"Total number of tokens in {self.name} dataset: {len(self.data)}")

    def __len__(self):
        return self.data.shape[0] // self.block_size

    def __getitem__(self, idx):
        data, block_size, batch_size = self.data, self.block_size, self.batch_size

        if self.do_randomize:
            ix = torch.randint(len(data) - block_size, (1,)).item()
            x = torch.from_numpy((data[ix: ix + block_size]).astype(np.int64))
            y = torch.from_numpy((data[ix + 1 : ix + 1 + block_size]).astype(np.int64))
        else:
            start_idx = idx * block_size
            x = torch.from_numpy((data[start_idx : start_idx + block_size]).astype(np.int64))
            y = torch.from_numpy((data[start_idx + 1 : start_idx + 1 + block_size]).astype(np.int64))
        return x, y


def main(cfg: DictConfig) -> None:
    fabric_args = cfg.fabric if cfg.fabric else {}
    fabric = Fabric(**fabric_args)
    fabric.launch()

    device_type = fabric.device
    print("device: ", device_type)

    gpt_config = GPTConfig(**cfg.model)
    model = GPT(gpt_config)
    optimizer = model.configure_optimizers(cfg.optimizer, device_type)
    train_dataset = TrackDataSet(cfg.train_data, cfg.training.batch_size,
                                 cfg.training.block_size, do_randomize=True,
                                 name="train"
                                )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.training.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers)

    val_dataset = TrackDataSet(cfg.val_data, cfg.training.batch_size,
                               cfg.training.block_size, do_randomize=True,
                               name="val")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.training.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers)

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(train_dataloader)

    iter_num = 0

    for epoch in range(cfg.max_epochs):
        for batch in dataloader:
            x, y = batch
            logits, loss = model(x, y)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % cfg.log_interval == 0:
                lossf = loss.item()
                print(f"epoch {epoch}, iter {iter_num}, loss {lossf:.4f}")

            if iter_num % cfg.validation.val_interval == 0:
                # evaluate the model on validation data and log the performance
                out = {}
                loader_dict = {
                    "train": train_dataloader,
                    "val": val_dataloader
                }

                model.eval()
                for loader_name, loader in loader_dict.items():
                    losses = torch.zeros(cfg.validation.num_batches)
                    for i, batch in enumerate(loader):
                        if i >= cfg.validation.num_batches:
                            break
                        x, y = batch
                        with torch.no_grad():
                            _, loss = model(x, y)

                        losses[i] = loss.item()

                    out[loader_name] = losses.mean().item()
                print(f"epoch {epoch}, iter {iter_num},", "train-loss {0[train]:.4f}, val-loss {0[val]:.4f}".format(out))
                model.train()


@hydra.main(version_base=None, config_path="config", config_name="config")
def training(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    main(cfg)

if __name__ == "__main__":
    training()
