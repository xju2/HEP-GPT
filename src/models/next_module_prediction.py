from typing import Any, Optional, Dict
import inspect

import torch
from torchmetrics import MeanMetric, MinMetric

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
                 ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["model", "optimizer", "scheduler", "metrics_fn"]
        )

        self.model = model

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_fn = metrics_fn

        # save evaluation outputs
        self.validation_step_outputs = []

        # metrics to monitor
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # track best validation loss
        self.min_val_loss = MinMetric()

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

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.min_val_loss.reset()


    def cal_loss(self, logits: torch.Tensor, y: torch.Tensor):
        return self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self(x)

        loss = self.cal_loss(logits, y)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Common function for validation and test step"""
        x, y = batch
        logits = self(x)
        loss = self.cal_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        perf = self.step(batch, batch_idx)

        loss = perf["loss"]
        self.val_loss(loss)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.validation_step_outputs.append(perf)
        return perf

    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute()
        self.min_val_loss(avg_loss)
        log_args = {
            "prog_bar": True,
            "logger": True,
            "on_step": False,
            "on_epoch": True
        }
        self.log("val/avg_loss", avg_loss, **log_args)
        self.log("val/min_avg_loss", self.min_val_loss.compute(), **log_args)

        if avg_loss < self.min_val_loss.compute() and self.metrics_fn is not None:
            for perf in self.validation_step_outputs:
                self.metrics_fn(perf)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int):
        perf = self.step(batch, batch_idx)
        return perf
