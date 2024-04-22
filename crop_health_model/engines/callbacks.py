import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class ClassWeightsCallback(Callback):
    """Callback to set class weights in the model."""

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the training starts."""
        weights = trainer.datamodule.compute_class_weights().to(pl_module.device)
        pl_module.set_class_weights(weights)
        print("Class weights set in the model.")
