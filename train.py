import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from lightning.fabric import Fabric

from model import GPTConfig, GPT

from torch.utils.data import Dataset


def TrackDataLoader(Dataset):
    def __init__(self, inputfile: str, batch_size: int, block_size: int,
                 do_randomize: bool = False,
                 transform=None, target_transform=None):
        self.inputfile = inputfile
        self.batch_size = batch_size
        self.block_size = block_size
        self.do_randomize = do_randomize

        self.transform = transform
        self.target_transform = target_transform

        self.data = np.memmap(inputfile, dtype=np.uint16, mode='r')

    def __len__(self):
        return self.data.shape[0] // self.block_size // self.batch_size

    def __getitem__(self, idx):
        data, block_size, batch_size = self.data, self.block_size, self.batch_size

        if self.do_randomize:
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        else:
            start_idx = idx * block_size * batch_size
            x = torch.stack([torch.from_numpy((
                             data[start_idx + i * block_size: start_idx + (i + 1) * block_size]).astype(np.int64))
                             for i in self.batch_size])

            y = torch.stack([torch.from_numpy((
                             data[start_idx + i * block_size + 1 : start_idx + (i + 1) * block_size + 1]).astype(np.int64))
                            for i in self.batch_size])
        return x, y


def main(cfg: DictConfig) -> None:
    fabric_args = cfg.fabric if cfg.fabric else {}
    fabric = Fabric(**fabric_args)
    fabric.launch()

    device_type = fabric.device
    print("device: ", device)

    gpt_config = GPTConfig(**cfg.model)
    model = GPT(gpt_config)
    optimizer = model.configure_optimizers(cfg.optimizer, device_type)
    train_dataloader = TrackDataLoader(cfg.train_data, cfg.training.batch_size,
                                       cfg.training.block_size, do_randomize=True)

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)


    for epoch in range(cfg.training.max_epochs):
        iter_num = 0
        for batch in dataloader:
            x, y = batch
            logits, loss = model(x, y)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % cfg.log_interval == 0:
                lossf = loss.item()
                print(f"epoch {epoch}, iter {iter_num}, loss {lossf}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def training(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    main(cfg)

if __name__ == "__main__":
    training()
