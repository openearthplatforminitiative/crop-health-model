import torch
from lightning.pytorch.cli import LightningCLI

from crop_health_model.datasets.datamodule import CropHealthDataModule
from crop_health_model.engines.system import LitModel

torch.set_float32_matmul_precision("medium")


def main():
    cli = LightningCLI(model_class=LitModel, datamodule_class=CropHealthDataModule)


if __name__ == "__main__":
    main()
