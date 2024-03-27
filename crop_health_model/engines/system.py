import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
    ):
        super(LitModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, y):
        # If num_classes is 2, use sigmoid and binary cross entropy
        # Otherwise, use log_softmax and negative log likelihood
        if self.model.num_classes == 2:
            return F.binary_cross_entropy_with_logits(logits, y)
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_probs, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task, num_classes=self.model.num_classes)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )

        # log the learning rate
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task, num_classes=self.model.num_classes)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task, num_classes=self.model.num_classes)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
