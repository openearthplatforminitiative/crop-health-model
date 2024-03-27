import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
        limit=None,
    ):
        self.img_labels = pd.read_csv(annotations_file).sample(frac=1)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # if limit is not None, use only the first `limit` rows
        if limit:
            self.img_labels = self.img_labels[:limit]
            print(f"Using only {limit} rows")
        # print number of distinct classes
        print(f"Number of distinct classes: {len(self.img_labels['label'].unique())}")

        # Define a mapping from the class labels to integers using the unique method from pandas
        self.class_map = {
            label: idx for idx, label in enumerate(self.img_labels["label"].unique())
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
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
