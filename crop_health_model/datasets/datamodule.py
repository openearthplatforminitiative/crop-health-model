import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from crop_health_model.datasets.dataset import (
    CropHealthDataset,
    TransformWrapperDataset,
)


class CropHealthDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Crop Health Dataset."""

    def __init__(
        self,
        batch_size: int,
        task: str,
        data_dir: str = "crop_health_model/.data",
        annotations_file: str = "annotations.csv",
        data_split: tuple = (0.8, 0.2),
        limit: int | None = None,
        num_workers: int = 8,
        train_transforms: list[torch.nn.Module] | None = None,
        test_transforms: list[torch.nn.Module] | None = None,
        normalization: transforms.Normalize | None = None,
    ) -> None:
        super(CropHealthDataModule, self).__init__()
        self.task = task
        self.limit = limit
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.data_split = data_split
        self.num_workers = num_workers

        if train_transforms is not None:
            transformations = train_transforms + [transforms.ToTensor()]
            if normalization is not None:
                transformations += [normalization]
            self.train_transform = transforms.Compose(transformations)
        else:
            self.train_transform = None

        if test_transforms is not None:
            transformations = test_transforms + [transforms.ToTensor()]
            if normalization is not None:
                transformations += [normalization]
            self.test_transform = transforms.Compose(transformations)
            self.val_transform = transforms.Compose(transformations)
        else:
            self.test_transform = None
            self.val_transform = None

    def prepare_data(self) -> None:
        # assumes data has been downloaded using make_dataset.py
        # and prepared using make_annotations_file.py
        pass

    def setup(self, stage: str | None = None) -> None:
        self.data = CropHealthDataset(
            annotations_file=os.path.join(self.data_dir, self.annotations_file),
            img_dir=self.data_dir,
            task=self.task,
            transform=None,  # apply transforms later
            limit=self.limit,
        )

        # Split dataset into train and test sets
        train_data, test_data = random_split(self.data, self.data_split)

        # Split train dataset into train and validation sets
        train_data, val_data = random_split(train_data, self.data_split)

        # Wrap datasets with transforms
        train_data = TransformWrapperDataset(train_data, transform=self.train_transform)
        val_data = TransformWrapperDataset(val_data, transform=self.val_transform)
        test_data = TransformWrapperDataset(test_data, transform=self.test_transform)

        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.train_data = train_data
            self.val_data = val_data

        # Assign validation dataset
        if stage == "validate":
            self.val_data = val_data

        # Assign test dataset
        if stage == "test":
            self.test_data = test_data

        if stage == "predict":
            self.predict_data = self.test_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
