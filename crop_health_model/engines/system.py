import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitModel(pl.LightningModule):
    """PyTorch Lightning module for training the Crop Health model."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super(LitModel, self).__init__()
        self.model = model
        self.class_weights = None

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
        self, batch: tuple, batch_idx: int, use_weights: bool = False
    ) -> torch.Tensor:
        """Wrapper for the training/validation/test step."""
        x, y = batch
        logits = self.model(x)
        if use_weights:
            loss = self._compute_loss(logits, y, self.class_weights)
        else:
            loss = self._compute_loss(logits, y)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx, use_weights=True)

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx)
