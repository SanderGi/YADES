# Minimal reimplementation of YOLOv8n-cls to make it easier to extend.

import torch
from torch import nn


def autopad(kernel_size):
    return (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else tuple(x // 2 for x in kernel_size)
    )


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels, add):
        super().__init__()
        self.cv1 = Conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        self.cv2 = Conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        self.add = add

    def forward(self, x):
        if not self.add:
            return self.cv2(self.cv1(x))
        return torch.add(x, self.cv2(self.cv1(x)))


class C2f(nn.Module):
    def __init__(self, channels, num_bottlenecks, bottleneck_channels):
        super().__init__()
        self.cv1 = Conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.cv2 = Conv(
            in_channels=channels + bottleneck_channels * num_bottlenecks,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.m = nn.ModuleList(
            [
                Bottleneck(channels=bottleneck_channels, add=True)
                for _ in range(num_bottlenecks)
            ]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Classify(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv = Conv(
            in_channels, out_channels=hidden_channels, kernel_size=(1, 1), stride=(1, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(
            in_features=hidden_channels, out_features=num_classes, bias=True
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        return x.softmax(1)


class YOLOMinimal(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.model = nn.Sequential(
            Conv(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
            Conv(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            C2f(channels=32, num_bottlenecks=1, bottleneck_channels=16),
            Conv(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            C2f(channels=64, num_bottlenecks=2, bottleneck_channels=32),
            Conv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2)),
            C2f(channels=128, num_bottlenecks=2, bottleneck_channels=64),
            Conv(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2)),
            C2f(channels=256, num_bottlenecks=1, bottleneck_channels=128),
            Classify(in_channels=256, hidden_channels=1280, num_classes=len(classes)),
        )

    @classmethod
    def load(cls, path):
        state_dict = torch.load(path, weights_only=True)
        classes = state_dict.pop("classes")
        m = cls(classes)
        m.load_state_dict(state_dict)
        return m

    @classmethod
    def from_YOLOv8n(cls, yolo):
        """Create a YOLOMinimal model from an Ultralytics YOLOv8n-cls model."""
        num_classes = yolo.model.model[-1].linear.out_features
        classes = [yolo.names[i] for i in range(num_classes)]
        m = cls(classes)
        m.load_state_dict(yolo.model.state_dict())
        return m

    def save(self, path):
        from copy import deepcopy

        half = deepcopy(self).half()
        torch.save({"classes": self.classes, **half.state_dict()}, path)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.classes[self.forward(x).argmax(1)]


# Example inference code
if __name__ == "__main__":
    import cv2
    import torchvision.transforms as transforms
    from ultralytics import YOLO

    mm = YOLOMinimal.from_YOLOv8n(
        YOLO("./data/yolo-finetune/10_epoch_224_imgsz/weights/best.pt")
    )
    mm.eval()

    IMG_SIZE = 224
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]
    )
    image = cv2.imread("./data/sample.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)  # type: ignore

    print(mm.predict(image))
