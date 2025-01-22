import sys
import time
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

torch.backends.quantized.engine = "qnnpack"


def main(args):
    if len(args) == 0 or args[0].lower() == "help":
        print(
            "Usage: python ./scripts/run_cnn.py <image_path> [<model_path>] [<IMG_RESIZE_TO>]"
        )
        return

    image_path = args[0]
    model_path = args[1] if len(args) > 1 else "data/animal_cnn_224_small.pt"
    img_size = int(args[2]) if len(args) > 2 else 224

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = plt.imread(image_path).copy()
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)  # type: ignore

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 6, 5)
            v = ((img_size - 5 + 1) // 2 - 5 + 1) // 2
            self.fc1 = nn.Linear(6 * v * v, 24)
            self.fc2 = nn.Linear(24, 24)
            self.fc3 = nn.Linear(24, 2)

        def forward(self, x):
            x = self.quant(x)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.dequant(x)
            return x

    sys.modules[__name__].Net = Net  # type: ignore

    cnn = torch.load(model_path, weights_only=False)
    cnn.conv1._backward_hooks = {}
    cnn.conv1._backward_pre_hooks = {}
    cnn.conv1._forward_hooks = {}
    cnn.conv1._forward_pre_hooks = {}
    cnn.conv2._backward_hooks = {}
    cnn.conv2._backward_pre_hooks = {}
    cnn.conv2._forward_hooks = {}
    cnn.conv2._forward_pre_hooks = {}

    start = time.perf_counter()
    output = cnn(image)
    end = time.perf_counter()
    print(
        ["animal", "non-animal"][output.argmax().item()],
        f"detected in {end - start:0.5f} seconds",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
