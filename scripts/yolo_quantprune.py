# Implementation of YOLOv8n-cls that supports quantization and pruning

import os
import warnings

import torch
from torch import nn
import torch.nn.utils.prune as prune
from collections import OrderedDict

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


class YOLOQuantPrune(nn.Module):
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
        """Create a YOLOQuantPrune model from an Ultralytics YOLOv8n-cls model."""
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="must run observer before calling calculate_qparams",
            )
            try:
                # uncompress the file
                import gzip
                import tempfile

                with gzip.open(path, "rb") as f:
                    data = f.read()

                with tempfile.NamedTemporaryFile() as f:
                    old_data = f.read()
                    f.write(data)  # type: ignore
                    f.seek(0)
                    state_dict = OrderedDict(torch.load(f.name, weights_only=True))
                    f.seek(0)
                    f.write(old_data)
            except:
                # file is not compressed
                state_dict = OrderedDict(torch.load(path, weights_only=True))

            classes = state_dict.pop("classes")
            m = cls(classes)
            m.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
            torch.quantization.prepare_qat(m, inplace=True)
            torch.quantization.convert(m.eval(), inplace=True)
            state_dict = decompress_state_dict(state_dict)
            for name, param in m.named_parameters():
                if name in state_dict:
                    state_dict[name] = state_dict[name].to(param.data.dtype)
                    if param.data.is_sparse:
                        state_dict[name] = state_dict[name].to_sparse()
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

    def save(self, path, compress=True):
        if compress:
            torch.save(
                OrderedDict({"classes": self.classes, **self.state_dict()}),
                path,
            )
            # compress the file
            import gzip

            with open(path, "rb") as f:
                data = f.read()

            with gzip.open(path, "wb") as f:
                f.write(data)  # type: ignore
        else:
            # this stores in a sparse format but does not otherwise compress the file
            torch.save(
                OrderedDict(
                    {"classes": self.classes, **compress_state_dict(self.state_dict())}
                ),
                path,
            )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.classes[self.forward(x).argmax(1)]

    def sparsity(self):
        zero_params = 0
        total_params = 0
        for _, module in self.named_modules():
            if isinstance(module, torch.ao.nn.quantized.modules.conv.Conv2d):
                zero_params += torch.sum(module.weight() == 0).item()
                total_params += module.weight().numel()
            else:
                for _, param in module.named_parameters(recurse=False):
                    zero_params += torch.sum(param == 0).item()
                    total_params += param.numel()
        return zero_params / total_params

    def prune(
        self,
        amount=0.3,
        include=[
            torch.ao.nn.quantized.modules.conv.Conv2d,
            torch.ao.nn.qat.modules.conv.Conv2d,
        ],
        n=1,
        structured: None | int = None,
        remove=True,
        recompute_masks=True,
    ):
        for name, module in self.named_modules():
            if not any(isinstance(module, e) for e in include):
                continue

            weight_mask = module.weight_mask if hasattr(module, "weight_mask") else None

            is_module_quantized = isinstance(
                module, torch.ao.nn.quantized.modules.conv.Conv2d
            )
            if is_module_quantized:
                scale, zero_point, dtype = (
                    module.weight().q_scale(),
                    module.weight().q_zero_point(),
                    module.weight().dtype,
                )
                weight = module.weight().dequantize().contiguous()
                module = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=False,
                )
                module.weight.data = weight

            param_name = "weight"
            p = None
            if recompute_masks:
                if structured is None:
                    assert n == 1, "n must be 1 for unstructured pruning"
                    p = prune.l1_unstructured(module, name=param_name, amount=amount)
                    weight_mask = p.weight_mask
                else:
                    if isinstance(structured, tuple) and isinstance(amount, tuple):
                        for a, dim in zip(amount, structured):
                            p = prune.ln_structured(
                                module, name=param_name, amount=a, n=n, dim=dim
                            )
                            weight_mask = p.weight_mask
                    else:
                        p = prune.ln_structured(
                            module,
                            name=param_name,
                            amount=amount,
                            n=n,
                            dim=structured,
                        )
                        weight_mask = p.weight_mask
            elif weight_mask is not None:
                p = prune.custom_from_mask(module, name=param_name, mask=weight_mask)

            if remove:
                prune.remove(module, name=param_name)
                if p and not is_module_quantized:
                    p.weight = torch.nn.Parameter(p.weight)
                    module._forward_pre_hooks.clear()
                # if p:
                #     del p.weight_mask
                weight_mask = None

            if is_module_quantized:
                quantized_weights = torch.quantize_per_tensor(
                    module.weight.data,
                    scale=scale,
                    zero_point=zero_point,
                    dtype=dtype,
                )
                module = self.get_submodule(name)
                module.set_weight_bias(quantized_weights, None)
                if weight_mask is not None:
                    # set hook on forward to apply mask
                    module._forward_pre_hooks.clear()

                    def apply_mask(module, input):
                        module.set_weight_bias(module.weight * weight_mask, None)
                        return input

                    module.register_forward_pre_hook(apply_mask)

                    # save the mask
                    module.weight_mask = weight_mask

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
        name_fx=lambda epoch: f"qat_pruned_epoch_{epoch + 1}.pt",
        prune_kwargs=None,
        finalize=True,
        data_transform=None,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="All inputs of this cat operator must share the same quantization parameters",
            )

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
                if prune_kwargs:
                    self.prune(**prune_kwargs, remove=False, recompute_masks=True)
                running_loss = 0.0
                for i, (data, target) in enumerate(trainloader):
                    optimizer.zero_grad()
                    if data_transform:
                        data = data_transform(data)
                    output = self(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if prune_kwargs:
                        self.prune(**prune_kwargs, remove=False, recompute_masks=False)
                    running_loss += loss.item()

                    if i % show_stats_every_nth_batch == show_stats_every_nth_batch - 1:
                        print(
                            f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / show_stats_every_nth_batch:.3f}"
                        )
                        running_loss = 0.0

                scheduler.step()

                self.eval()
                if prune_kwargs:
                    self.prune(**prune_kwargs, remove=True, recompute_masks=False)
                quantized_model = torch.quantization.convert(self, inplace=False)
                print(f"Sparsity: {quantized_model.sparsity():.2%}")
                quantized_model.save(f"{checkpoint_dir}/{name_fx(epoch)}")
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
                print(
                    "Pytorch Model Size",
                    os.path.getsize(f"{checkpoint_dir}/{name_fx(epoch)}") / 1e3,
                    "KB",
                )

            if finalize:
                if prune_kwargs:
                    self.prune(**prune_kwargs, remove=True, recompute_masks=False)
                torch.quantization.convert(self.eval(), inplace=True)


# Compression utils
from collections import OrderedDict
from typing import Literal

COMPACT_T = tuple[torch.Tensor, torch.dtype | None]


def compact(tensor: torch.Tensor) -> COMPACT_T:
    if tensor.dim() == 0 or tensor.numel() == 0:
        return tensor, None

    if tensor.stride(-1) != 1:
        tensor = tensor.contiguous()

    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        return tensor.half(), None
    elif tensor.dtype in [torch.uint16, torch.uint32, torch.uint64]:
        if tensor.max() <= torch.iinfo(torch.uint8).max:
            return tensor.to(torch.uint8), None
        elif tensor.max() <= torch.iinfo(torch.uint16).max:
            return tensor.to(torch.uint16).view(torch.uint8), torch.uint16
        elif tensor.max() <= torch.iinfo(torch.uint32).max:
            return tensor.to(torch.uint32).view(torch.uint8), torch.uint32
        else:
            return tensor, None
    elif tensor.dtype in [torch.int16, torch.int32, torch.int64]:
        if (
            tensor.min() >= torch.iinfo(torch.int8).min
            and tensor.max() <= torch.iinfo(torch.int8).max
        ):
            return tensor.to(torch.int8), None
        elif (
            tensor.min() >= torch.iinfo(torch.int16).min
            and tensor.max() <= torch.iinfo(torch.int16).max
        ):
            return tensor.to(torch.int16).view(torch.uint8), torch.int16
        elif (
            tensor.min() >= torch.iinfo(torch.int32).min
            and tensor.max() <= torch.iinfo(torch.int32).max
        ):
            return tensor.to(torch.int32).view(torch.uint8), torch.int32
        else:
            return tensor, None
    else:
        return tensor, None


def undo_compact(rep: COMPACT_T) -> torch.Tensor:
    if rep[1] is None:
        return rep[0]
    else:
        return rep[0].view(rep[1])


ENCODED_T = tuple[COMPACT_T, COMPACT_T, torch.Size] | COMPACT_T


def rl_encode(tensor: torch.Tensor) -> tuple[ENCODED_T, int]:
    compact_tensor = compact(tensor)
    compact_tensor_size_bytes = (
        compact_tensor[0].numel() * compact_tensor[0].element_size()
    )
    if tensor.numel() == 0:
        return compact_tensor, compact_tensor_size_bytes

    original_shape = tensor.shape
    tensor = tensor.reshape(-1)

    indices = torch.where(tensor[:-1] != tensor[1:])[0] + 1
    indices = torch.cat([torch.tensor([0]), indices, torch.tensor([tensor.numel()])])
    lengths = compact(indices[1:] - indices[:-1])
    values = compact(tensor[indices[:-1]])

    encoded_size_bytes = (
        values[0].numel() * values[0].element_size()
        + lengths[0].numel() * lengths[0].element_size()
    )
    if encoded_size_bytes < compact_tensor_size_bytes:
        return (values, lengths, original_shape), encoded_size_bytes
    else:
        return compact_tensor, compact_tensor_size_bytes


def rl_decode(encoded: ENCODED_T) -> torch.Tensor:
    if not isinstance(encoded, tuple):
        return encoded
    if len(encoded) == 2:
        return undo_compact(encoded)
    values = undo_compact(encoded[0])
    lengths = undo_compact(encoded[1])
    original_shape = encoded[2]
    return torch.repeat_interleave(values, lengths).reshape(original_shape)


def to_sparse(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if tensor.dtype == torch.qint8:
        mask = tensor != 0
        indices = torch.nonzero(mask, as_tuple=False)
        values = tensor[mask]
        return indices, values, tensor.size()
    else:
        sparse = tensor.to_sparse()
        return sparse.indices(), sparse.values(), sparse.size()


COMPRESSED_T = tuple[Literal["c"], COMPACT_T | tuple[ENCODED_T, COMPACT_T, torch.Size]]


def compress_tensor(tensor: torch.Tensor) -> COMPRESSED_T:
    dense = compact(tensor)
    dense_size_bytes = dense[0].numel() * dense[0].element_size()

    indices, values, shape = to_sparse(tensor)
    encoded_indices, ix_size = rl_encode(indices)
    sparse = encoded_indices, compact(values), shape
    sparse_size_bytes = ix_size + sparse[1][0].numel() * sparse[1][0].element_size()

    if sparse_size_bytes < dense_size_bytes:
        return ("c", sparse)
    else:
        return ("c", dense)


def decompress_tensor(compressed: COMPRESSED_T) -> torch.Tensor:
    if len(compressed[1]) == 3:
        encoded_indices, compact_values, shape = compressed[1]
        indices = rl_decode(encoded_indices)
        values = undo_compact(compact_values)
        sparse = torch.sparse_coo_tensor(indices, values, shape)
        return sparse.to_dense()
    else:
        return undo_compact(compressed[1])


def compress_state_dict(state_dict: dict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, tuple):
            new_state_dict[k] = tuple(
                compress_tensor(x) if isinstance(x, torch.Tensor) else x for x in v
            )
        elif isinstance(v, torch.Tensor):
            new_state_dict[k] = compress_tensor(v)
        else:
            new_state_dict[k] = v
    return new_state_dict


def decompress_state_dict(state_dict: dict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, tuple) and v[0] == "c":
            new_state_dict[k] = decompress_tensor(v)
        elif isinstance(v, tuple):
            new_state_dict[k] = tuple(
                decompress_tensor(x) if x[0] == "c" else x for x in v
            )
        else:
            new_state_dict[k] = v
    return new_state_dict


# Example inference code
if __name__ == "__main__":
    import sys
    import cv2
    import torchvision.transforms as transforms

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
            "Usage: python ./scripts/yolo_quantprune.py <path_to_image> [<path_to_model>]"
        )
        print("Example: python ./scripts/yolo_quantprune.py ./data/sample.jpg")
        sys.exit(1)
    image_path = args[0]
    model_path = (
        args[1]
        if len(args) > 1
        else "./data/yolo-finetune/10_epoch_224_imgsz/weights/qat.pt"
    )

    pm = YOLOQuantPrune.load(model_path).eval()

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
