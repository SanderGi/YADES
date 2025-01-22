import os
from PIL import Image
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, animal_directory, non_animal_directory, transform=lambda x: x):
        self.transform = transform

        self.classes = ["animal", "non-animal"]

        self.image_files = [
            (os.path.join(animal_directory, f), 0) for f in os.listdir(animal_directory)
        ] + [
            (os.path.join(non_animal_directory, f), 1)
            for f in os.listdir(non_animal_directory)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path, label = self.image_files[index]

        image = Image.open(image_path).convert("RGB")
        return (self.transform(image), label)
