import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, fbeta_score


class LitModel(pl.LightningModule):
    """PyTorch Lightning module for training the Crop Health model."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super(LitModel, self).__init__()
        self.model = model
        self.class_weights = None
        hyperparameters = self.model.get_hyperparameters()
        self.save_hyperparameters(hyperparameters)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """Set the class weights for the loss function."""
        self.class_weights = class_weights

    def _compute_loss(
        self, logits: torch.Tensor, y: torch.Tensor, weights=None
    ) -> torch.Tensor:
        """Compute the loss function."""
        return F.cross_entropy(logits, y, weight=weights)

    def step_wrapper(
        self, batch: tuple, batch_idx: int, prefix: str, use_weights: bool = False
    ) -> torch.Tensor:
        """Wrapper for the training/validation/test step."""
        x, y = batch
        logits = self.model(x)
        if use_weights:
            loss = self._compute_loss(logits, y, self.class_weights)
        else:
            loss = self._compute_loss(logits, y)

        # logging of metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task=task, num_classes=self.model.num_classes)
        f1 = fbeta_score(
            preds=preds,
            target=y,
            task=task,
            beta=1.0,
            num_classes=self.model.num_classes,
            average="macro",
        )
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.step_wrapper(batch, batch_idx, prefix="train", use_weights=True)

        # log the learning rate
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx, prefix="test")
