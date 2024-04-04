import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI

from crop_health_model.datasets.datamodule import CropHealthDataModule
from crop_health_model.engines.system import LitModel

torch.set_float32_matmul_precision("medium")

# Set seeds and enable unique seeds for workers
# Doesn't seem possible to set workers=True in the
# YAML config so we set it here
seed_everything(42, workers=True)


def main():
    cli = LightningCLI(
        model_class=LitModel,
        datamodule_class=CropHealthDataModule,
        seed_everything_default=False,
    )


if __name__ == "__main__":
    main()
