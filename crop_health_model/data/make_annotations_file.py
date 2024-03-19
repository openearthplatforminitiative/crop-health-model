# Script for generating annotations file of all images in a directory

import os

import pandas as pd

def generate_annotations(img_dir, keywords_to_avoid, annotations_file_path, classes):
    # List to store the image paths
    img_paths = []

    # List to store the image labels
    labels = []

    # Loop through all the subdirectories in the img_dir
    # But make sure to only add images to the img_paths list
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")) and not any(
                keyword in root for keyword in keywords_to_avoid
            ):
                img_paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))

    # Create a DataFrame from the img_paths and labels lists
    df = pd.DataFrame(list(zip(img_paths, labels)), columns=["image", "label"])

    # Save the DataFrame to a CSV file
    df.to_csv(annotations_file_path, index=False)

    return df
