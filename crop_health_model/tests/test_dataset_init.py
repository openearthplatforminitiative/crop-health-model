import pandas as pd
import pytest

from crop_health_model.datasets.dataset import CropHealthDataset

TMP_DATA_PATH = "tmp_data.csv"


def create_dummy_dataset() -> None:
    # create a dummy dataset
    data = {
        "image": [
            "img1.jpg",
            "img2.jpg",
            "img3.jpg",
            "img4.jpg",
            "img5.jpg",
            "img6.jpg",
        ],
        "width": [100, 200, 300, 400, 500, 600],
        "height": [200, 300, 400, 500, 600, 700],
        "label": ["HLT", "SICKa", "SICKb", "SICKb", "HLT", "SICKm"],
        "crop_type": ["apple", "apple", "banana", "banana", "maize", "maize"],
    }
    df = pd.DataFrame(data)

    # write to a CSV file
    df.to_csv(TMP_DATA_PATH, index=False)


def cleanup_dummy_dataset() -> None:
    import os

    os.remove(TMP_DATA_PATH)


def test_dataset_multi_HLT():
    create_dummy_dataset()

    dataset = CropHealthDataset(
        annotations_file=TMP_DATA_PATH,
        img_dir=".",
        task="multi-HLT",
        transform=None,
        limit=None,
    )

    assert len(dataset) == 6
    assert dataset.data_df["label"].tolist() == [
        "HLT_apple",
        "SICKa_apple",
        "SICKb_banana",
        "SICKb_banana",
        "HLT_maize",
        "SICKm_maize",
    ]

    assert dataset.class_map == {
        "HLT_apple": 0,
        "SICKa_apple": 1,
        "SICKb_banana": 2,
        "HLT_maize": 3,
        "SICKm_maize": 4,
    }

    assert dataset.get_class_counts() == {
        0: 1,
        1: 1,
        2: 2,
        3: 1,
        4: 1,
    }

    # cleanup
    cleanup_dummy_dataset()


def test_dataset_single_HLT():
    create_dummy_dataset()

    dataset = CropHealthDataset(
        annotations_file=TMP_DATA_PATH,
        img_dir=".",
        task="single-HLT",
        transform=None,
        limit=None,
    )

    assert len(dataset) == 6
    assert dataset.data_df["label"].tolist() == [
        "HLT",
        "SICKa",
        "SICKb",
        "SICKb",
        "HLT",
        "SICKm",
    ]

    assert dataset.class_map == {"HLT": 0, "SICKa": 1, "SICKb": 2, "SICKm": 3}

    assert dataset.get_class_counts() == {
        0: 2,
        1: 1,
        2: 2,
        3: 1,
    }

    # cleanup
    cleanup_dummy_dataset()


def test_dataset_binary():
    create_dummy_dataset()

    dataset = CropHealthDataset(
        annotations_file=TMP_DATA_PATH,
        img_dir=".",
        task="binary",
        transform=None,
        limit=None,
    )

    assert len(dataset) == 6
    assert dataset.data_df["label"].tolist() == [
        "HLT",
        "NOT_HLT",
        "NOT_HLT",
        "NOT_HLT",
        "HLT",
        "NOT_HLT",
    ]

    assert dataset.class_map == {"HLT": 0, "NOT_HLT": 1}

    assert dataset.get_class_counts() == {
        0: 2,
        1: 4,
    }

    # cleanup
    cleanup_dummy_dataset()


def test_inavlid_task():
    create_dummy_dataset()

    with pytest.raises(ValueError):
        dataset = CropHealthDataset(
            annotations_file=TMP_DATA_PATH,
            img_dir=".",
            task="invalid",
            transform=None,
            limit=None,
        )

    # cleanup
    cleanup_dummy_dataset()
