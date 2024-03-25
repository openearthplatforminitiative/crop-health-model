import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
        limit=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # if limit is not None, use only the first `limit` rows
        if limit:
            self.img_labels = self.img_labels[:limit]
            print(f"Using only {limit} rows")

        # Define a mapping from the class labels to integers using the unique method from pandas
        self.class_map = {
            label: idx for idx, label in enumerate(self.img_labels["label"].unique())
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = self.class_map[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
