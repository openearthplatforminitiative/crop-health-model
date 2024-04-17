import os
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CropHealthDataset(Dataset):
    """A dataset for the Crop Health dataset."""

    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        task: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        limit: int | None = None,
    ) -> None:
        self.data_df = pd.read_csv(annotations_file).sample(frac=1)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # if limit is not None, use only the first `limit` rows
        if limit:
            self.data_df = self.data_df[:limit]
            print(f"Using only {limit} rows")

        if task == "binary":
            # binary classification with one healthy (HLT) class and one sick (NOT_HLT) class
            # map all non "HLT" classes to "NOT_HLT"
            mask = self.data_df["label"] != "HLT"
            self.data_df.loc[mask, "label"] = "NOT_HLT"
        elif task == "single-HLT":
            # keep everything as is, i.e., several sick classes and one healthy class
            pass
        elif task == "multi-HLT":
            # multiple healthy classes (one for each crop type) and multiple sick classes
            # combine "label" with "crop_type" to create a new label
            self.data_df["label"] = (
                self.data_df["label"] + "_" + self.data_df["crop_type"]
            )
        # print number of distinct classes
        print(f"Number of distinct classes: {len(self.data_df['label'].unique())}")

        # Define a mapping from the class labels to integers using the unique method from pandas
        self.class_map = {
            label: idx for idx, label in enumerate(self.data_df["label"].unique())
        }

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.data_df)

    def __getitem__(self, idx) -> tuple:
        """Return the image and its label at the given index."""
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])
        # open image with PIL
        image = Image.open(img_path)
        label_name = row["label"]
        label = self.class_map[label_name]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class TransformWrapperDataset(Dataset):
    """A dataset that wraps another dataset and applies a transform to the data.

    Useful when performing data splitting on original dataset and then applying transforms
    on the split datasets.
    """

    def __init__(self, dataset: Dataset, transform: Callable | None = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple:
        """Return the transformed data and its label at the given index."""
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target
