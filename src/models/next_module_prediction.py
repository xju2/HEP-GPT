from typing import Any, Optional, Dict
import inspect

import torch
from torchmetrics import MinMetric

import lightning as L

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class NextModulePrediction(L.LightningModule):
    """Next Module Prediction Model
    It uses a Transformer model to predict the next module in a sequence of modules.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 metrics_fn: Optional[callable] = None,
                 label_smoothing: float = 0.05,
                 weight_decay: float = 0.01,
                 compile: bool = True
                 ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["model", "optimizer", "scheduler", "metrics_fn"]
        )

        if compile:
            log.info("Compiling model...")
            self.model = torch.compile(model)
        else:
            self.model = model

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_fn = metrics_fn

        # save evaluation outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.val_min_avg_loss = MinMetric()

    # @property
    # def example_input_array(self):
    #     return torch.LongTensor([[2] * self.model.block_size,
    #                              [4] * self.model.block_size])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        log.info(f"using fused AdamW: {use_fused}")

        opt = self.optimizer(params=optim_groups, **extra_args)

        if self.scheduler is None:
            return opt
        else:
            sch = self.scheduler(optimizer=opt)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": self.trainer.val_check_interval,
                    "strict": True,
                    "name": "LRScheduler",
                }
            }

    def cal_loss(self, logits: torch.Tensor, y: torch.Tensor):
        return self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    def on_train_start(self) -> None:
        self.val_min_avg_loss.reset()

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.cal_loss(logits, y)

        perf = {"loss": loss}
        self.training_step_outputs.append(perf)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return perf

    # def on_training_epoch_end(self) -> None:
    #     mean_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()

    #     # log metrics
    #     self.log("train/avg_loss", mean_loss, prog_bar=True)
    #     self.training_step_outputs.clear()

    def step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Common function for validation and test step"""
        x, y = batch
        logits = self(x)
        loss = self.cal_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        perf = self.step(batch, batch_idx)
        self.validation_step_outputs.append(perf)

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.val_min_avg_loss(mean_loss)

        # log metrics
        self.log("val/avg_loss", mean_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/min_avg_loss", self.val_min_avg_loss.compute(), prog_bar=True)

        # reset the outputs
        self.validation_step_outputs.clear()
