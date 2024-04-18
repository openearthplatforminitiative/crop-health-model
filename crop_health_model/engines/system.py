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
        hyperparameters = self.model.get_hyperparameters()
        self.save_hyperparameters(hyperparameters)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss function."""
        return F.cross_entropy(logits, y)

    def step_wrapper(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Wrapper for the training/validation/test step."""
        x, y = batch
        logits = self.model(x)
        return self._compute_loss(logits, y)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx)

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx)
