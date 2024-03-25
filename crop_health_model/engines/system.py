import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module):
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.02)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     return {"loss": avg_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task, num_classes=self.model.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task, num_classes=self.model.num_classes)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x for x in outputs]).mean()
    #     return {"test_loss": avg_loss}
