# Implementation of YOLOv8n-cls that supports quantization.


import torch
from torch import nn

torch.backends.quantized.engine = "qnnpack"


def autopad(kernel_size):
    return (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else tuple(x // 2 for x in kernel_size)
    )


class SiLU(nn.Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        x = self.dequant(input)
        x = torch.nn.functional.silu(x)
        x = self.quant(x)

        if self.inplace:
            input.copy_(x)
            return input
        else:
            return x


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
        self.act = SiLU(inplace=False)

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
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if not self.add:
            return self.cv2(self.cv1(x))
        float_x = self.dequant(x)
        float_cv = self.dequant(self.cv2(self.cv1(x)))
        sum_x_cv = torch.add(float_x, float_cv)
        return self.quant(sum_x_cv)


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
        return x.dequantize().softmax(1)


class YOLOQuantized(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        model = nn.Sequential(
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
        self.model = torch.quantization.QuantWrapper(model)

    @classmethod
    def from_YOLOv8n(cls, yolo):
        """Create a YOLOQuantized model from an Ultralytics YOLOv8n-cls model."""
        num_classes = yolo.model.model[-1].linear.out_features
        classes = [yolo.names[i] for i in range(num_classes)]
        m = cls(classes)
        m.load_state_dict(
            {
                k.replace("model", "model.module"): v
                for k, v in yolo.model.state_dict().items()
            }
        )
        return m

    @classmethod
    def load(cls, path):
        from collections import OrderedDict

        state_dict = OrderedDict(torch.load(path, weights_only=True))
        classes = state_dict.pop("classes")
        m = cls(classes)
        m.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        torch.quantization.prepare_qat(m, inplace=True)
        torch.quantization.convert(m.eval(), inplace=True)
        setattr(
            state_dict,
            "_metadata",
            {
                "model.module.9.linear": {
                    "version": 3,
                },
                "model.module.9.linear._packed_params": {
                    "version": 3,
                },
            },
        )
        m.load_state_dict(state_dict)
        return m

    def save(self, path):
        from copy import deepcopy

        cpy = deepcopy(self)
        for param in cpy.parameters():
            if param.data.dtype == torch.float32:
                param.data = param.data.half()

        torch.save({"classes": self.classes, **cpy.state_dict()}, path)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.classes[self.forward(x).argmax(1)]

    def static_quantize(
        self,
        weight_observer=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine
        ),
        activation_observer=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        linear_weight_observer=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine
        ),
        linear_activation_observer=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        callibration_loader=None,
        callibration_transform=None,
    ):
        # Suggested observers: MinMaxObserver, HistogramObserver, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, PerChannelMinMaxObserver, RecordingObserver,
        # Suggested dtypes: (torch.qint8, torch.quint8, torch.quint4x2, torch.qint32, torch.int8, torch.uint8, torch.int16, torch.int32, torch.float8_e5m2, torch.float8_e4m3fn)
        # Suggested qschemes: torch.per_tensor_affine, torch.per_tensor_symmetric, torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams
        # Take a look at `torch.quantization.get_default_qconfig("qnnpack")` for inspiration

        self.train()
        self.qconfig = torch.quantization.QConfig(
            activation=activation_observer, weight=weight_observer
        )
        self.model.module[9].linear.qconfig = torch.quantization.QConfig(  # type: ignore
            activation=linear_activation_observer, weight=linear_weight_observer
        )
        torch.quantization.prepare(self, inplace=True)

        if callibration_loader is not None:
            with torch.no_grad():
                for data, _ in callibration_loader:
                    if callibration_transform:
                        data = callibration_transform(data)
                    self(data)

        torch.quantization.convert(self.eval(), inplace=True)

    def _get_optimizer(self, momentum=0.9, decay=1e-5):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        nc = len(self.classes)
        lr = round(0.002 * 5 / (4 + nc), 6)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizer = torch.optim.AdamW(
            g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

        return optimizer

    def qat(
        self,
        trainloader,
        valloader=None,
        epochs=10,
        show_stats_every_nth_batch=10,
        dropout=0.0,
        lrf=0.01,
        checkpoint_dir="data/yolo-finetune/10_epoch_224_imgsz/weights",
        data_transform=None,
    ):
        self.train()
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout
        for p in self.parameters():
            p.requires_grad = True

        lr = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf  # linear
        optimizer = self._get_optimizer()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
        criterion = nn.CrossEntropyLoss()

        self.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        torch.quantization.prepare_qat(self, inplace=True)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for i, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                if data_transform:
                    data = data_transform(data)
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % show_stats_every_nth_batch == show_stats_every_nth_batch - 1:
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / show_stats_every_nth_batch:.3f}"
                    )
                    running_loss = 0.0

            scheduler.step()

            quantized_model = torch.quantization.convert(self.eval(), inplace=False)
            quantized_model.save(f"{checkpoint_dir}/qat_epoch_{epoch + 1}.pt")
            if valloader is not None:
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in valloader:
                        if data_transform:
                            inputs = data_transform(inputs)
                        outputs = quantized_model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                val_loss /= len(valloader)
                print(f"Epoch {epoch + 1} Validation loss: {val_loss:.3f}")
            else:
                print(f"Epoch {epoch + 1} Completed")

        torch.quantization.convert(self.eval(), inplace=True)


# Example inference code
if __name__ == "__main__":
    import sys
    import cv2
    import torchvision.transforms as transforms
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="All inputs of this cat operator must share the same quantization parameters",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="must run observer before calling calculate_qparams",
    )

    args = sys.argv[1:]
    if len(args) == 0:
        print(
            "Usage: python ./scripts/yolo_quantized.py <path_to_image> [<path_to_model>]"
        )
        print("Example: python ./scripts/yolo_quantized.py ./data/sample.jpg")
        sys.exit(1)
    image_path = args[0]
    model_path = (
        args[1]
        if len(args) > 1
        else "./data/yolo-finetune/10_epoch_224_imgsz/weights/qat.pt"
    )

    pm = YOLOQuantized.load(model_path).eval()

    IMG_SIZE = 224
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)  # type: ignore

    print(pm.predict(image))
