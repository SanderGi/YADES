import os, sys

from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")

sys.path.append(os.path.join(ROOT, "scripts"))
from YOLODataset import YOLODataset

import ai8x  # type: ignore

IMG_SIZE = (180, 180)
# IMG_SIZE = (224, 244)


class simulate_rgb565:
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        import torch

        # Step 1: Scale float values (0.0-1.0) to 8-bit integers (0-255)
        image_8bit = (img * 255).to(torch.uint8)

        # Separate the 8-bit channels
        r_8bit = image_8bit[0, :, :]
        g_8bit = image_8bit[1, :, :]
        b_8bit = image_8bit[2, :, :]

        # Step 2: Quantize to 5, 6, and 5 bits
        # Shift right to truncate the least significant bits.
        # Red (5 bits): 8-bit value >> 3 (255 / 8 = 31.875)
        r_5bit = r_8bit >> 3
        # Green (6 bits): 8-bit value >> 2 (255 / 4 = 63.75)
        g_6bit = g_8bit >> 2
        # Blue (5 bits): 8-bit value >> 3 (255 / 8 = 31.875)
        b_5bit = b_8bit >> 3

        # Step 3: Pack into a single 16-bit integer
        # (r_5bit << 11) | (g_6bit << 5) | b_5bit
        # rgb565 = (r_5bit.astype(np.uint16) << 11) | (g_6bit.astype(np.uint16) << 5) | b_5bit.astype(np.uint16)

        # Instead keept as 3 separate channels
        if self.args.act_mode_8bit:
            image_8bit[0, :, :] = r_5bit
            image_8bit[1, :, :] = g_6bit
            image_8bit[2, :, :] = b_5bit

            return image_8bit.to(torch.float32)
        else:
            img[0, :, :] = (r_5bit / 32).to(torch.float32)
            img[1, :, :] = (g_6bit / 64).to(torch.float32)
            img[2, :, :] = (b_5bit / 32).to(torch.float32)

            return img


def get_dataset(data, load_train, load_test):
    (data_dir, args) = data
    data_dir = os.path.join(ROOT, ".data", "elk")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE),
            ai8x.normalize(args=args),
            # simulate_rgb565(args=args),
        ]
    )

    train_dataset = None
    if load_train:
        subset = "train"
        animal_directory = os.path.join(data_dir, subset, "elk")
        non_animal_directory = os.path.join(data_dir, subset, "other")
        train_dataset = YOLODataset(
            animal_directory, non_animal_directory, transform=transform
        )

    test_dataset = None
    if load_test:
        subset = "test"
        animal_directory = os.path.join(data_dir, subset, "elk")
        non_animal_directory = os.path.join(data_dir, subset, "other")
        test_dataset = YOLODataset(
            animal_directory, non_animal_directory, transform=transform
        )

    return train_dataset, test_dataset


datasets = [
    {
        "name": "elk_detection",
        "input": (3, *IMG_SIZE),
        "output": ("elk", "other"),
        "loader": get_dataset,
    },
]
