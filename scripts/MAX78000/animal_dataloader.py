import os, sys

from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")

sys.path.append(os.path.join(ROOT, "scripts"))
from YOLODataset import YOLODataset

import ai8x  # type: ignore

IMG_SIZE = 224


def get_dataset(data, load_train, load_test):
    (data_dir, args) = data
    data_dir = os.path.join(ROOT, ".data", "animal")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            ai8x.normalize(args=args),
        ]
    )

    train_dataset = None
    if load_train:
        subset = "train"
        animal_directory = os.path.join(data_dir, subset, "animal")
        non_animal_directory = os.path.join(data_dir, subset, "non-animal")
        train_dataset = YOLODataset(
            animal_directory, non_animal_directory, transform=transform
        )

    test_dataset = None
    if load_test:
        subset = "test"
        animal_directory = os.path.join(data_dir, subset, "animal")
        non_animal_directory = os.path.join(data_dir, subset, "non-animal")
        test_dataset = YOLODataset(
            animal_directory, non_animal_directory, transform=transform
        )

    return train_dataset, test_dataset


datasets = [
    {
        "name": "animal_detection",
        "input": (3, IMG_SIZE, IMG_SIZE),
        "output": ("cat", "dog"),
        "loader": get_dataset,
    },
]
