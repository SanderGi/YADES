# Implementation of YOLOPico model for MAX78000 that works with its hardware limitations and acceleration

from torch import nn, cat, Tensor

import ai8x  # type: ignore


# We can only use stride 1 with ai8x.FusedConv2dReLU so we'll simulate stride 2 with pooling
class Conv2dSimulatedStride2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        assert stride == 2
        self.conv = ai8x.FusedConv2dReLU(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=bias,
            **kwargs,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


# We cannot use AdaptiveAvgPool2d only ai8x.AvgPool2d, so we'll simulate using ai8x.AvgPool2d
class AdaptiveAvgPool2dSimulated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert 1 <= dim <= 16
        self.pool = nn.AvgPool2d(kernel_size=dim, stride=dim)
        self.dim = dim

    def forward(self, x):
        # asserts are useful for debugging, but should not be enabled in the forward pass while quantizing with pytorch
        # assert (
        #     x.shape[2] == x.shape[3] == self.dim
        # ), f"Expected input of shape (N, C, {self.dim}, {self.dim}), but got {x.shape}"
        return self.pool(x)


class Bottleneck(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.cv1 = ai8x.FusedConv2dReLU(
            channels, channels, kernel_size=3, padding=1, stride=1, bias=False, **kwargs
        )
        self.cv2 = ai8x.FusedConv2dReLU(
            channels, channels, kernel_size=3, padding=1, stride=1, bias=False, **kwargs
        )
        self.add = ai8x.Add()

    def forward(self, x):
        return self.add(x, self.cv2(self.cv1(x)))


class C2f(nn.Module):
    def __init__(self, channels, bottleneck_channels, **kwargs):
        super().__init__()
        self.cv1 = ai8x.FusedConv2dReLU(
            channels, channels, kernel_size=1, padding=1, stride=1, bias=False, **kwargs
        )
        self.cv2 = ai8x.FusedConv2dReLU(
            channels, channels, kernel_size=1, padding=1, stride=1, bias=False, **kwargs
        )
        self.bottleneck = Bottleneck(channels=bottleneck_channels, **kwargs)

    def forward(self, x):
        # Apply cv1 convolution
        x1 = self.cv1(x)

        # Manually split the tensor into two halves along the channel dimension
        chunk1 = x1[:, : x1.shape[1] // 2, :, :]
        chunk2 = x1[:, x1.shape[1] // 2 :, :, :]

        # Process chunk2 through bottleneck
        chunk2 = self.bottleneck(chunk2)

        # Concatenate chunk1 and chunk2 and pass through cv2
        return self.cv2(cat([chunk1, chunk2], dim=1))


class Classify(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dim, **kwargs):
        super().__init__()
        self.conv = ai8x.FusedConv2dReLU(
            in_channels,
            hidden_channels,
            kernel_size=1,
            padding=1,
            stride=1,
            bias=False,
            **kwargs,
        )
        dim += 2
        self.pool = AdaptiveAvgPool2dSimulated(dim=dim)
        self.linear = ai8x.Linear(
            hidden_channels, num_classes, bias=True, wide=True, **kwargs
        )

    def forward(self, x):
        return self.linear(self.pool(self.conv(x)).flatten(1))


class YOLOPico(nn.Module):
    def __init__(
        self, num_classes=2, num_channels=3, dimensions=(224, 224), bias=False, **kwargs
    ):
        super().__init__()
        assert bias is False

        layer1_channels = 16
        self.cv1 = Conv2dSimulatedStride2(
            in_channels=num_channels,
            out_channels=layer1_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            **kwargs,
        )

        layer2_channels = 32
        self.cv2 = Conv2dSimulatedStride2(
            in_channels=layer1_channels,
            out_channels=layer2_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            **kwargs,
        )

        self.cv3 = C2f(
            channels=layer2_channels, bottleneck_channels=layer2_channels // 2, **kwargs
        )

        layer3_channels = 32
        self.cv4 = Conv2dSimulatedStride2(
            in_channels=layer2_channels,
            out_channels=layer3_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            **kwargs,
        )

        self.cv5 = C2f(
            channels=layer3_channels, bottleneck_channels=layer3_channels // 2, **kwargs
        )

        layer4_channels = 64
        self.cv6 = Conv2dSimulatedStride2(
            in_channels=layer3_channels,
            out_channels=layer4_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            **kwargs,
        )

        self.cv7 = C2f(
            channels=layer4_channels, bottleneck_channels=layer4_channels // 2, **kwargs
        )

        layer5_channels = 128
        self.cv8 = Conv2dSimulatedStride2(
            in_channels=layer4_channels,
            out_channels=layer5_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            **kwargs,
        )

        self.cv9 = C2f(
            channels=layer5_channels, bottleneck_channels=layer5_channels // 2, **kwargs
        )

        test_input = Tensor(1, num_channels, *dimensions)
        test_input = self.cv1(test_input)
        test_input = self.cv2(test_input)
        test_input = self.cv3(test_input)
        test_input = self.cv4(test_input)
        test_input = self.cv5(test_input)
        test_input = self.cv6(test_input)
        test_input = self.cv7(test_input)
        test_input = self.cv8(test_input)
        test_input = self.cv9(test_input)
        assert test_input.shape[2] == test_input.shape[3]
        self.classify = Classify(
            in_channels=layer5_channels,
            hidden_channels=512,
            num_classes=num_classes,
            dim=test_input.shape[2],
            **kwargs,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.cv5(x)
        x = self.cv6(x)
        x = self.cv7(x)
        x = self.cv8(x)
        x = self.cv9(x)
        return self.classify(x)


def yolo_pico(pretrained=False, **kwargs):
    assert not pretrained
    return YOLOPico(**kwargs)


models = [
    {
        "name": "yolo_pico",
        "min_input": 1,
        "dim": 2,
    },
]
