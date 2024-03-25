import os

import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from crop_health_model.datasets.dataset import CustomImageDataset


class CropHealthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir: str = "crop_health_model/.data",
        annotations_file: str = "annotations.csv",
        crop_size: int | None = None,
        resize_size: int | None = None,
        interpolation: str | None = None,
        mean: tuple | None = None,
        std: tuple | None = None,
        data_split: tuple = (0.8, 0.2),
        limit: int | None = None,
    ):
        super(CropHealthDataModule, self).__init__()
        self.limit = limit
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.data_split = data_split

        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.interpolation = (
            getattr(transforms.InterpolationMode, interpolation.upper())
            if interpolation
            else None
        )

        resize = [transforms.Resize(self.resize_size)] if self.resize_size else []
        center_crop = [transforms.CenterCrop(self.crop_size)] if self.crop_size else []
        normalize = (
            [transforms.Normalize(mean=self.mean, std=self.std)]
            if self.mean and self.std
            else []
        )

        # Construct the transformations conditionally
        self.transform = transforms.Compose(
            [
                *resize,
                *center_crop,
                transforms.ToTensor(),
                *normalize,
            ]
        )

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        self.data = CustomImageDataset(
            annotations_file=os.path.join(self.data_dir, self.annotations_file),
            img_dir=self.data_dir,
            transform=self.transform,
            limit=self.limit
        )

        # Split dataset into train and test sets
        train_data, test_data = random_split(self.data, self.data_split)

        # Split train dataset into train and validation sets
        train_data, val_data = random_split(train_data, self.data_split)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = train_data
            self.val_data = val_data

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = test_data

        if stage == "predict" or stage is None:
            self.predict_data = self.test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size)
