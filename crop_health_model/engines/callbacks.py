import math

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class ImagePredictionLogger(Callback):
    """Callback to log image predictions during validation to TensorBoard."""

    def __init__(self, num_samples: int = 32) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the validation loop begins."""
        # Get the first batch of validation data
        val_samples = next(iter(trainer.datamodule.val_dataloader()))
        val_imgs = val_samples[0].to(device=pl_module.device)
        val_labels = val_samples[1].to(device=pl_module.device)

        # Get model prediction
        log_probs = pl_module(val_imgs)
        preds = torch.argmax(log_probs, dim=-1)

        # Determine grid size
        num_images = min(self.num_samples, len(val_imgs))
        num_cols = int(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows)
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                img = val_imgs[i].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                ax.imshow(img)
                ax.set_title(
                    f"Pred: {preds[i].item()}, Label: {val_labels[i].item()}",
                    fontsize=10,
                )
                ax.axis("off")
            else:
                ax.axis("off")  # Hide unused subplots

        # Use the logger tied to the Trainer
        if trainer.logger:
            # This assumes you have a logger that can log matplotlib figures directly.
            trainer.logger.experiment.add_figure(
                "Validation Predictions", fig, global_step=trainer.current_epoch
            )

        plt.close(fig)
