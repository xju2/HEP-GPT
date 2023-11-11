import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
from typing import List
import time
import signal

from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.fabric import Fabric
from lightning.fabric.loggers import Logger

import torch

# local imports
from model import GPTConfig, GPT
from src.utils.utils import instantiate_loggers
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def main(cfg: DictConfig) -> None:
    # torch precision settings
    torch.set_float32_matmul_precision("medium")
    loggers: List[Logger] = instantiate_loggers(cfg.loggers)

    # fabric plugins for slurm envionment
    # slurm_env = SLURMEnvironment(
    #     auto_requeue=cfg.slurm.auto_requeue,
    #     requeue_signal=signal.SIGUSR1)

    fabric: Fabric = hydra.utils.instantiate(
        cfg.fabric,
        loggers=loggers[0])
    fabric.launch()

    if cfg.get("seed"):
        fabric.seed_everything(cfg.seed)

    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

    device_type = fabric.device
    log.info("device: {}[{}]".format(device_type.type, device_type.index))

    gpt_config = GPTConfig(**cfg.model)
    model = GPT(gpt_config)

    optimizer = model.configure_optimizers(cfg.optimizer, device_type)

    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup("fit")

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # see issue #62 about compiling the model for FSDP strategy
    # https://github.com/Lightning-AI/lit-llama/issues/62
    if cfg.compile:
        log.info("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # require Pytorch 2.0
    model.train()
    model, optimizer = fabric.setup(model, optimizer)

    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    state = {
        "model": model,
        "optimizer": optimizer,
    }
    if cfg.init_from == "resume" and Path(cfg.ckpt_path).exists():
        log.info(f"Resume the training from the checkpoint: {cfg.ckpt_path}")
        fabric.load(cfg.ckpt_path, state)

    outdir = Path(cfg.paths.output_dir)
    log.info(f"output directionary: {outdir}")
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    iter_num = 0
    best_val_loss = 9999999
    best_val_step = 0
    log.info("epoch global_step train validation time[ms]")
    t0 = time.time()
    training_timer = time.time()
    for epoch in range(cfg.max_epochs):
        for batch in train_dataloader:
            x, y = batch
            logits, loss = model(x, y)
            fabric.backward(loss)

            # gradient clipping
            fabric.clip_gradients(model, optimizer, max_norm=cfg.optimizer.grad_clip_val)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1

            # timing info
            t1 = time.time()
            dt_tr = t1 - training_timer
            training_timer = t1

            # validation, if not, log the training loss
            if iter_num % cfg.validation.val_interval == 0:
                # evaluate the model on validation data and log the performance
                out = {"dt": dt_tr}
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
                model.train()

                fabric.log_dict(out, step=iter_num)

                if out["val"] < best_val_loss:
                    best_val_loss = out["val"]
                    best_val_step = iter_num
                    fabric.save(outdir / "best.ckpt", state)
                    fabric.save(ckpt_dir / f"ckpt-{iter_num}.ckpt", state)

                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    log.info(f"{epoch} {iter_num} {out['train']:.6f} {out['val']:.6f} {dt*1000:.2f}")
            elif iter_num % cfg.log_interval == 0:
                # log the training loss
                fabric.log_dict({"train": loss.item(), "val": -1.0, "dt": dt_tr}, step=iter_num)
            else:
                pass


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def training(cfg : DictConfig) -> None:
    if cfg.dry_run:
        log.info(OmegaConf.to_yaml(cfg))
        return

    main(cfg)

if __name__ == "__main__":
    training()
