# Script for generating annotations file of all images in a directory

import os

import pandas as pd
from tqdm import tqdm

from crop_health_model.data.metadata import all_datasets


def generate_annotations(
    img_dir, keywords_to_avoid, annotations_file_path, classes, total_image_count
):
    # List to store the image paths
    img_paths = []

    # List to store the image labels
    labels = []

    print(f"Generating annotations for {img_dir}")

    # Loop through all the subdirectories in the img_dir
    # But make sure to only add images to the img_paths list
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")) and not any(
                keyword in root for keyword in keywords_to_avoid
            ):
                path = os.path.join(root.lstrip(".data/"), file)
                img_paths.append(path)

                # Get the label from the directory name
                for class_ in classes:
                    if class_["raw"] in root:
                        labels.append(class_["clean"])
                        break

    # Verify that the number of labels matches the number of images
    # if not (len(img_paths) == len(labels) == total_image_count):
    print(f"Expected {total_image_count} images, found {len(img_paths)} images and {len(labels)} labels")

    # Verify that the number of class labels is correct
    for class_ in classes:
        if class_["count"] == -1:
            continue
        # if not labels.count(class_["clean"]) == class_["count"]:
        print(f"Expected {class_['count']} images for class {class_['clean']}, found {labels.count(class_['clean'])}")

    # Create a DataFrame from the img_paths and labels lists
    df = pd.DataFrame(list(zip(img_paths, labels)), columns=["image", "label"])

    # Save the DataFrame to a CSV file
    df.to_csv(annotations_file_path, index=False)

    return df


def generate_all_annotations():
    for dataset in all_datasets:
        img_dir = os.path.join(".data", dataset["folder"])
        keywords_to_avoid = []
        annotations_file_path = os.path.join(
            ".data", dataset["folder"], "annotations.csv"
        )
        classes = dataset["classes"]
        total_image_count = dataset["total_image_count"]
        generate_annotations(
            img_dir,
            keywords_to_avoid,
            annotations_file_path,
            classes,
            total_image_count,
        )
        print()

if __name__ == "__main__":
    generate_all_annotations()