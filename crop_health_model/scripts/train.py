import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI

from crop_health_model.datasets.datamodule import CropHealthDataModule
from crop_health_model.engines.system import LitModel
from crop_health_model.models.model import ResNet

crop_health_data_module = CropHealthDataModule


def main():
    cli = LightningCLI(
        model_class=LitModel, datamodule_class=CropHealthDataModule
    )  # , subclass_mode_model=True)


if __name__ == "__main__":
    main()
