# Implementation of YOLOv8n-cls that supports sparse tensors (otherwise pruned model won't actually require less parameters to store).

import torch
from torch import nn
import torch.nn.utils.prune as prune


def autopad(kernel_size):
    return (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else tuple(x // 2 for x in kernel_size)
    )


def create_sparse_zeros(size):
    return torch.empty(size, layout=torch.sparse_coo)


def encode(tensor: torch.Tensor):
    # if empty tensor, return empty tensor
    if tensor.numel() == 0:
        return None, 0

    # Save the original shape
    original_shape = tensor.shape

    # Ensure the tensor is 1D
    tensor = tensor.reshape(-1)

    # Find indices where the tensor changes value
    indices = torch.where(tensor[:-1] != tensor[1:])[0] + 1
    indices = torch.cat([torch.tensor([0]), indices, torch.tensor([tensor.numel()])])

    # Calculate the lengths of the runs
    lengths = indices[1:] - indices[:-1]
    # determine dtype of lengths
    viewl = 0
    if lengths.max() < 2**8:
        lengths = lengths.to(torch.uint8)
    elif lengths.max() < 2**16:
        lengths = lengths.to(torch.uint16)
        lengths = lengths.view(torch.uint8)
        viewl = 1
    elif lengths.max() < 2**32:
        lengths = lengths.to(torch.uint32)
        lengths = lengths.view(torch.uint8)
        viewl = 2

    # Get the values of the runs
    values = tensor[indices[:-1]]
    # determine dtype of values
    viewv = 0
    if values.max() < 256:
        values = values.to(torch.uint8)
    elif values.max() < 65536:
        values = values.to(torch.uint16)
        values = values.view(torch.uint8)
        viewv = 1
    elif values.max() < 2**32:
        values = values.to(torch.uint32)
        values = values.view(torch.uint8)
        viewv = 2

    size_bytes = (
        values.numel() * values.element_size()
        + lengths.numel() * lengths.element_size()
        + lengths.numel() * lengths.element_size()
    )

    return (values, lengths, original_shape, viewl, viewv), size_bytes


def decode(r):
    if r is None:
        return create_sparse_zeros((0,))
    if not isinstance(r, tuple):
        return r

    values, lengths, original_shape, viewl, viewv = r
    if viewl == 1:
        lengths = lengths.view(torch.uint16)
        lengths = lengths.to(torch.long)
    elif viewl == 2:
        lengths = lengths.view(torch.uint32)
        lengths = lengths.to(torch.long)
    if viewv == 1:
        values = values.view(torch.uint16)
        lengths = lengths.to(torch.long)
    elif viewv == 2:
        values = values.view(torch.uint32)
        lengths = lengths.to(torch.long)
    tensor = torch.zeros(original_shape, dtype=torch.long)
    flat_view = tensor.view(-1)
    start = 0
    for value, length in zip(values, lengths):
        flat_view[start : start + length] = value
        start += length

    return tensor


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
        self.conv.weight = nn.Parameter(create_sparse_zeros(self.conv.weight.size()))
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.bn.weight = nn.Parameter(create_sparse_zeros(self.bn.weight.size()))
        self.bn.bias = nn.Parameter(create_sparse_zeros(self.bn.bias.size()))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # convolutional layer has to be dense since no sparse PyTorch implementation is available on Mac
        x = x.to_dense()
        self.conv.weight = nn.Parameter(self.conv.weight.data.to_dense())
        self.bn.weight = nn.Parameter(self.bn.weight.data.to_dense())
        self.bn.bias = nn.Parameter(self.bn.bias.data.to_dense())
        bn = self.bn(self.conv(x))
        self.conv.weight = nn.Parameter(self.conv.weight.data.to_sparse())
        self.bn.weight = nn.Parameter(self.bn.weight.data.to_sparse())
        self.bn.bias = nn.Parameter(self.bn.bias.data.to_sparse())
        return self.act(bn).to_sparse()


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
        # chunk does not work with sparse tensors
        y = [y.to_sparse() for y in self.cv1(x).to_dense().chunk(2, 1)]
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
        # we end dense because pool and linear layers do not support sparse tensors
        x = self.linear(self.drop(self.pool(self.conv(x).to_dense()).flatten(1)))
        if self.training:
            return x
        return x.softmax(1)


class YOLOPruned(nn.Module):
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
        model = cls(classes)
        indices_and_values = state_dict.pop("indices_and_values")
        for name, parameter in model.named_parameters():
            if isinstance(indices_and_values[name], tuple):
                indices, values, size = indices_and_values[name]
                parameter.data = (
                    torch.sparse_coo_tensor(decode(indices), values, size)
                    .coalesce()
                    .to(parameter.data.dtype)
                )
            else:
                if parameter.data.is_sparse and not indices_and_values[name].is_sparse:
                    parameter.data = (
                        indices_and_values[name]
                        .to_sparse()
                        .coalesce()
                        .to(parameter.data.dtype)
                    )
                else:
                    parameter.data = indices_and_values[name].to(parameter.data.dtype)
        return model

    @classmethod
    def from_YOLOv8n(cls, yolo):
        """Create a YOLOPruned model from an Ultralytics YOLOv8n-cls model."""
        num_classes = yolo.model.model[-1].linear.out_features
        classes = [yolo.names[i] for i in range(num_classes)]
        m = cls(classes)
        for name, parameter in yolo.model.named_parameters():
            m_param = m.get_parameter(name)
            if m_param.data.is_sparse:
                m_param.data = parameter.data.to_sparse()
            else:
                m_param.data = parameter.data
        return m

    def save(self, path):
        indices_and_values = {}
        for name, parameter in self.named_parameters():
            if parameter.data.is_sparse:
                encoded, sparse_size_bytes = encode(parameter.data.indices())
                indices_and_values[name] = (
                    encoded,  # .to_dense()
                    parameter.data.values().to_dense().half(),
                    parameter.size(),
                )
                sparse_size_bytes += (
                    parameter.data.values().to_dense().half().numel()
                    * parameter.data.values().to_dense().half().element_size()
                    / 2
                )
                dense = parameter.data.to_dense().half()
                dense_size_bytes = dense.numel() * dense.element_size()
                if dense_size_bytes < sparse_size_bytes:
                    indices_and_values[name] = dense
            else:
                indices_and_values[name] = parameter.data.half()

        torch.save(
            {"classes": self.classes, "indices_and_values": indices_and_values}, path
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.classes[self.forward(x).argmax(1)]

    def to_dense(self):
        from yolo_minimal import YOLOMinimal

        mm = YOLOMinimal(self.classes)
        mm.load_state_dict(
            {
                k: v.to_dense()
                for k, v in self.state_dict().items()
                if k in mm.state_dict()
            }
        )
        return mm

    @classmethod
    def from_dense(cls, mm):
        m = cls(mm.classes)
        m.load_state_dict(
            {
                k: (
                    v.to_sparse()
                    if k in [n for n, _ in m.named_parameters()]
                    and m.get_parameter(k).is_sparse
                    else v
                )
                for k, v in mm.state_dict().items()
                if k in m.state_dict()
            }
        )
        return m

    @staticmethod
    def prune_dense(mm, amount=0.3, exclude=[nn.Linear, nn.BatchNorm2d]):
        names = [name for name, _ in mm.named_parameters()]
        for name in names:
            last_period = name.rfind(".")
            module_name, param_name = name[:last_period], name[last_period + 1 :]
            module = mm.get_submodule(module_name)
            if any(isinstance(module, e) for e in exclude):
                continue
            # prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=0)
            prune.l1_unstructured(module, name=param_name, amount=amount)
            prune.remove(module, name=param_name)

    def prune(self, amount=0.3, exclude=[nn.Linear, nn.BatchNorm2d]):
        mm = self.to_dense()
        self.prune_dense(mm, amount, exclude)

        for name, parameter in self.named_parameters():
            last_period = name.rfind(".")
            module_name = name[:last_period]
            module = mm.get_submodule(module_name)
            if any(isinstance(module, e) for e in exclude):
                continue

            parameter.data = mm.get_parameter(name).data.to_sparse()


# Example inference code
if __name__ == "__main__":
    import sys
    import cv2
    import torchvision.transforms as transforms

    args = sys.argv[1:]
    if len(args) == 0:
        print(
            "Usage: python ./scripts/yolo_pruned.py <path_to_image> [<path_to_model>]"
        )
        print("Example: python ./scripts/yolo_pruned.py ./data/sample.jpg")
        sys.exit(1)
    image_path = args[0]
    model_path = (
        args[1]
        if len(args) > 1
        else "./data/yolo-finetune/10_epoch_224_imgsz/weights/pruned.pt"
    )

    pm = YOLOPruned.load(model_path).eval()

    IMG_SIZE = 224
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]
    )
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)  # type: ignore

    print(pm.predict(image))
