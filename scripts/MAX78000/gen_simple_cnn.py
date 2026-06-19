# fmt: off
import os, sys
import subprocess

import numpy as np

NUM_BITS = 8 # must be quantized to 8-bit for now
DATA_MEMORY_INSTANCE_MAX_SIZE = 32768 # max amount of data for each data memory instance (each processor)
MAX_WEIGHT_QUADRANT_KERNELS = 768 * 64 // 4

def usage():
    print("Usage: python scripts/MAX78000/gen_simple_cnn.py [--interactive] [--input-HWC] [-config <str:config_path>] [-l <int:num_layers>] [-s <int:target_size_in_bytes>] [-w <int:input_width>] [-h <int:input_height>] [-o <int:num_output_classes>] [-m <str:model_name>] [-d <str:dataset_name>]")

def gen_conv2d(input_channels, output_channels, kernel_size, padding=1, fused_pooling=None):
    assert input_channels <= 1024 and input_channels >= 1 and output_channels <= 1024 and output_channels >= 1, "channels must be between 1 and 1024"
    assert kernel_size in [1, 3], "only 1x1 and 3x3 kernels are supported"
    assert padding in [0, 1, 2] and padding < kernel_size, "only 0, 1, 2 padding is supported and it must be less than the kernel size"
    # stride, dilation, and groups is always fixed to 1

    if fused_pooling is None:
        code = f"ai8x.FusedConv2dReLU({input_channels}, {output_channels}, {kernel_size}, padding={padding}, bias=bias)"
        dim = f"dimx = (dimx - {kernel_size} + 2 * {padding}) // 1 + 1\n"
        dim += f"        dimy = (dimy - {kernel_size} + 2 * {padding}) // 1 + 1"
    else:
        pool_size, pool_stride = fused_pooling
        assert pool_size in range(1, 17), "pool kernel size must be between 1 and 16"
        assert pool_stride in range(1, 17), "pool stride must be between 1 and 16"
        code = f"ai8x.FusedMaxPoolConv2dReLU({input_channels}, {output_channels}, {kernel_size}, pool_size={pool_size}, pool_stride={pool_stride}, padding={padding}, bias=bias)"
        dim = f"dimx = (dimx - {pool_size}) // {pool_stride} + 1\n"
        dim += f"        dimx = (dimx - {kernel_size} + 2 * {padding}) // 1 + 1\n"
        dim += f"        dimy = (dimy - {pool_size}) // {pool_stride} + 1\n"
        dim += f"        dimy = (dimy - {kernel_size} + 2 * {padding}) // 1 + 1"

    return code, dim

def train(model_name, dataset_name, layers, input_dimensions, num_output_classes):
    print("Generating model file...")
    init_body = "\n"
    forward_body = "\n"
    for i, (input_channels, output_channels, kernel_size, padding, fused_pooling) in enumerate(layers):
        code, dim = gen_conv2d(input_channels, output_channels, kernel_size, padding, fused_pooling)
        init_body += f'        assert not bias or {output_channels} <= 512, "at most 512 output channels supported with bias"\n'
        init_body += f"        self.conv{i} = {code}\n"
        init_body += f"        {dim}\n"
        forward_body += f"        x = self.conv{i}(x)\n"


    model_code = f"""
import ai8x
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_channels={layers[0][0]}, dimensions={input_dimensions}, num_classes={num_output_classes}, bias=False,  **kwargs):
        super().__init__()
        assert num_channels == {layers[0][0]}
        dimx, dimy = dimensions
        {init_body}
        self.fc = ai8x.Linear({layers[-1][1]}*dimx*dimy, num_classes, bias=True)

    def forward(self, x):
        {forward_body}
        return self.fc(x.view(x.size(0), -1))

def {model_name}(pretrained=False, **kwargs):
    assert not pretrained
    return CNN(**kwargs)

models = [
    {{
        "name": "{model_name}",
        "min_input": 1,
        "dim": 2,
    }},
]
    """.strip()

    classes = ", ".join(map(lambda i: f'"{i}"', range(num_output_classes)))
    dataset_code = f"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x  # type: ignore

class Loader(Dataset):
    def __init__(self, image):
        self.image = image

    def __len__(self):
        return {num_output_classes}

    def __getitem__(self, index):
        return (self.image, index % {num_output_classes})

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
def get_dataset(data, load_train, load_test):
    (_, args) = data
    image_path = os.path.join(ROOT, "data", "sample.jpg")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize({input_dimensions}),
            ai8x.normalize(args=args),
        ]
    )
    image = transform(image)

    return Loader(image) if load_train else None, Loader(image) if load_test else None


datasets = [
    {{
        "name": "{dataset_name}",
        "input": ({layers[0][0]}, {input_dimensions[0]}, {input_dimensions[1]}),
        "output": ({classes}),
        "loader": get_dataset,
    }},
]
    """.strip()

    dataloader_path = os.path.join("MAX78000", "ai8x-training", "datasets", f"{dataset_name}.py")
    with open(dataloader_path, "w") as f:
        f.write(dataset_code)
    model_path = os.path.join("MAX78000", "ai8x-training", "models", f"{model_name}.py")
    with open(model_path, "w") as f:
        f.write(model_code)
    qat_policy_path = os.path.join("MAX78000", "ai8x-training", "policies", "test_policy.yaml")
    with open(qat_policy_path, "w") as f:
        f.write(f"""
---
start_epoch: 0
weight_bits: {NUM_BITS}
    """.strip())
    ret = subprocess.run(f"cd MAX78000/ai8x-training && . venv/bin/activate && python train.py --lr 0.01 --optimizer SGD --epochs 1 --deterministic --seed 1 --compress policies/schedule.yaml --model {model_name} --dataset {dataset_name} --confusion --param-hist --pr-curves --embedding --device MAX78000 --qat-policy policies/test_policy.yaml", shell=True, capture_output=True)
    if ret.returncode != 0:
        print(ret.stderr.decode())
        raise Exception("Failed")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "..", "MAX78000", "ai8x-training", "logs")
    checkpoint_path = os.path.join(checkpoint_path, next(reversed(sorted(os.listdir(checkpoint_path)))), "best.pth.tar")
    print("Model File:", checkpoint_path)
    return checkpoint_path

def generate_yaml(checkpoint_path: str, model_name, dataset_name, layers, input_dimensions, num_output_classes, input_mode):
    print("Writing yaml...")
    yaml_path = checkpoint_path.removesuffix(".pth.tar") + ".yaml"
    # ret = subprocess.run(f"cd MAX78000/ai8x-training && . venv/bin/activate && python train.py --lr 0.01 --optimizer SGD --epochs 1 --deterministic --seed 1 --compress policies/schedule.yaml --model {model_name} --dataset {dataset_name} --confusion --param-hist --pr-curves --embedding --device MAX78000 --qat-policy policies/test_policy.yaml --yaml-template {yaml_path}", shell=True, capture_output=True)
    # if ret.returncode != 0:
    #     print(ret.stderr.decode())
    #     raise Exception("Failed")

    streaming = [False] * len(layers)
    last_uses_padding = True
    dimx = input_dimensions[0]
    dimy = input_dimensions[1]
    for i, (input_channels, output_channels, kernel_size, padding, fused_pooling) in enumerate(layers):
        streaming_threshold = 32768 if i == 0 and input_mode == "CHW" else 8192
        input_channels = max(4, input_channels)
        if input_channels * dimx * dimy > streaming_threshold:
            for j in range(i+1):
                streaming[j] = True
            last_uses_padding = padding > 0
            assert dimx <= 1023 and dimy <= 1023, "streaming layers must not exceed 1023x1023 input"
            assert input_channels <= (4 if i == 0 and input_mode == "CHW" else 16), f"Layer {i} has {input_channels} input channels which exceeds the streaming mode limits"
        if fused_pooling is not None:
            dimx = (dimx - fused_pooling[0]) // fused_pooling[1] + 1
            dimy = (dimy - fused_pooling[0]) // fused_pooling[1] + 1
        dimx = (dimx - kernel_size + 2 * padding) // 1 + 1
        dimy = (dimy - kernel_size + 2 * padding) // 1 + 1
        # TODO: make sure bias is not used with streaming since MAX78000 is buggy in this case
        # TODO: linear layer
    assert not any(streaming) or last_uses_padding, f"with streaming enabled, the last streaming layer ({sum(streaming) - 1}) must use padding"
    assert sum(1 if s else 0 for s in streaming) <= 8, "at most 8 layers can be streamed"

    yaml = f"---\narch: {model_name}\ndataset: {dataset_name}\nlayers:\n"
    dimx = input_dimensions[0]
    dimy = input_dimensions[1]
    out_offset = "0x4000"
    processors = "0x0000000100010001"
    not_allowed = set(i for i, p in enumerate(processors.removeprefix("0x")) if p != "0")
    kernel_counts = [0] * (64 // 4)
    for ix, (input_channels, output_channels, kernel_size, padding, fused_pooling) in enumerate(layers):
        yaml += f"  # Layer {ix}\n"
        yaml += f"  - name: conv{ix}\n"
        yaml += f"    # input shape: ({input_channels}, {dimx}, {dimy}) = {dimx * dimy} px/channel\n"
        if ix == 0:
            yaml += f"    data_format: {input_mode}\n"

        yaml += f"    processors: {processors}\n"
        num_passes = 1
        if output_channels <= 64:
            num_processors = output_channels
        else:
            num_passes = 2
            while np.ceil(output_channels / num_passes) > 64:
                num_passes += 1
            num_processors = 4 * np.ceil(output_channels / num_passes / 4)
        buckets = [0] * (64 // 4)
        if streaming[ix]:
            for i in range(len(buckets)):
                if i not in not_allowed:
                    buckets[i] = 4 if num_processors >= 4 else num_processors % 4
                    num_processors -= buckets[i]
                    if num_processors == 0:
                        break
            assert num_processors == 0, f"Cannot allocate {num_processors} processors ({not_allowed}, {buckets})"
            # processors = f"0x{num_processors:016x}"
            # num_buckets = 64 // 4 - sum(1 if p != "0" else 0 for p in processors.removeprefix("0x"))
            # buckets = [num_processors // num_buckets] * (64 // 4)
            # for i, p in enumerate(processors.removeprefix("0x")):
            #     if p != "0":
            #         buckets[i] = 0
            # j = 0
            # for _ in range(num_processors % num_buckets):
            #     while j in set(i for i, p in enumerate(processors.removeprefix("0x")) if p != "0"):
            #         j += 1
            #         j %= 64 // 4
            #     buckets[j] += 1
            #     j += 1
        elif ix > 0 and streaming[ix - 1]:
            num_available = len(buckets) - len(not_allowed)
            assert num_processors <= num_available * 4, f"Not enough processors available at layer {ix}"
            per_bucket, left_over = num_processors // num_available, num_processors % num_available
            for i in sorted(range(len(buckets)), key=lambda i: kernel_counts[i]):
                if i not in not_allowed:
                    buckets[i] = per_bucket
                    if left_over > 0:
                        buckets[i] += 1
                        left_over -= 1
                    num_processors -= buckets[i]
                    if num_processors == 0:
                        break
            assert num_processors == 0, f"Cannot allocate {num_processors} processors ({not_allowed}, {buckets})"
        else:
            per_bucket, left_over = num_processors // len(buckets), num_processors % len(buckets)
            for i in sorted(range(len(buckets)), key=lambda i: kernel_counts[i]):
                buckets[i] = int(per_bucket)
                if left_over > 0:
                    buckets[i] += 1
                    left_over -= 1
        for i in range(len(buckets)):
            kernel_counts[i] += buckets[i] * num_passes * kernel_size * kernel_size
            assert kernel_counts[i] <= MAX_WEIGHT_QUADRANT_KERNELS, f"Too many kernels in quadrant {i} by layer {ix}"
        processors = "0x" + "".join(map(lambda c: ["0", "1", "3", "7", "f"][c], buckets))
        if streaming[ix]:
            not_allowed |= set(i for i, p in enumerate(processors.removeprefix("0x")) if p != "0")
        else:
            not_allowed = set(i for i, p in enumerate(processors.removeprefix("0x")) if p != "0")

        yaml += f"    out_offset: {out_offset}\n"
        out_offset = "0x4000" if out_offset == "0x0000" else "0x0000"

        yaml += "    op: Conv2d\n"
        yaml += f"    kernel_size: {kernel_size}x{kernel_size}\n"
        yaml += f"    pad: {padding}\n"
        yaml += "    activate: Relu\n"
        if streaming[ix]:
            yaml += "    streaming: true\n"
        if fused_pooling is not None:
            yaml += f"    max_pool: {fused_pooling[0]}\n"
            yaml += f"    pool_stride: {fused_pooling[1]}\n"
            yaml += "    pool_dilation: [1, 1]\n"
            dimx = (dimx - fused_pooling[0]) // fused_pooling[1] + 1
            dimy = (dimy - fused_pooling[0]) // fused_pooling[1] + 1
        dimx = (dimx - kernel_size + 2 * padding) // 1 + 1
        dimy = (dimy - kernel_size + 2 * padding) // 1 + 1
        yaml += f"    # output shape: ({output_channels}, {dimx}, {dimy}) = {dimx * dimy} px/channel\n\n"
            
    yaml += f"  # Layer {ix+1}\n"
    yaml += f"  - name: Linear\n"
    yaml += f"    # input shape: ({output_channels}, {dimx}, {dimy})\n"
    yaml += "    op: Linear\n"
    yaml += f"    processors: {processors}\n"
    yaml += f"    out_offset: {out_offset}\n"
    yaml += f"    flatten: true\n"
    yaml += f"    activate: None\n"
    yaml += f"    # output shape: ({num_output_classes},)\n"

    with open(yaml_path, "w") as f:
        f.write(yaml)
    print("YAML File:", yaml_path)
    return yaml_path, any(streaming)


def quantize(checkpoint_path: str):
    print("Quantizing...")
    quantized_path = checkpoint_path.removesuffix(".pth.tar") + "-q.pth.tar"
    ret = subprocess.run(f'cd MAX78000/ai8x-synthesis && . venv/bin/activate && python quantize.py "{checkpoint_path}" "{quantized_path}" --device MAX78000 -v', shell=True, capture_output=True)
    if ret.returncode != 0:
        print(ret.stderr.decode())
        raise Exception("Failed")
    print("Quantized File:", quantized_path)
    return quantized_path


def generate_sample(quantized_path: str, num_input_channels, input_dimensions):
    print("Generating sample input...")
    sample = np.zeros((num_input_channels, input_dimensions[0], input_dimensions[1]), np.int64)
    sample_path = quantized_path.removesuffix(".pth.tar") + ".npy"
    np.save(sample_path, sample)
    print("Sample File:", sample_path)
    return sample_path


def synthesize(sample_path: str, yaml_path: str, quantized_path: str, model_name, streaming=False):
    print("Synthesizing...")
    ret = subprocess.run(f'cd MAX78000/ai8x-synthesis && . venv/bin/activate && python ai8xize.py --verbose --log --test-dir sdk/Examples/MAX78000/CNN --prefix {model_name} --checkpoint-file {quantized_path} --config-file {yaml_path} --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --sample-input {sample_path} --overwrite --no-version-check' + (" --fifo" if streaming else ""), shell=True, capture_output=True)
    if ret.returncode != 0:
        print(ret.stderr.decode())
        raise Exception("Failed")
    print(ret.stdout.decode())
    synthesized_path = os.path.join(os.path.dirname(__file__), "..", "..", "MAX78000", "ai8x-synthesis", "sdk", "Examples", "MAX78000", "CNN", model_name)
    print("Synthesized Project:", synthesized_path)
    return synthesized_path

def parse_args(args: list[str], numbers: set[str]={"l", "s", "w", "h", "o"}, flags: set[str]={"interactive", "input-HWC"}, strings: set[str]={"m", "d", "config"}):
    try:
        parsed: dict[str, str | int | bool] = {f: False for f in flags} | {s: False for s in strings} | {n: False for n in numbers} # type: ignore
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                parsed[args[i].removeprefix("--")] = True
            elif args[i].startswith("-"):
                i += 1
                parsed[args[i-1].removeprefix("-") ] = args[i] if args[i-1].removeprefix("-") not in numbers else int(args[i])
            i += 1
        return parsed
    except:
        usage()
        sys.exit(1)

def main(args: list[str]):
    parsed = parse_args(args)
    model_name = parsed["m"] or "test_cnn"
    dataset_name =  parsed["d"] or "test_data"

    num_input_channels = 3 # fix RGB for now
    num_output_classes: int = parsed["o"] or 10 # type: ignore
    input_dimensions: tuple[int, int] = (parsed["w"] or 90, parsed["h"] or 90) # type: ignore
    input_mode = "HWC" if parsed["input-HWC"] else "CHW"
    # all internal layers must use HWC channel layout

    num_layers: int = parsed["l"] or 6 # type: ignore
    assert num_layers < 32, "at most 32 layers, hence at most 31 convolutional layers"
    if parsed["config"]:
        layers = []
        with open(parsed["config"], "r") as f:
            for line in f.readlines():
                if line.startswith("#") or not line.strip():
                    continue
                elif line.startswith("num_input_channels:"):
                    num_input_channels = int(line.removeprefix("num_input_channels:"))
                    if len(layers) > 0:
                        l = list(layers[0])
                        l[0] = num_input_channels
                        layers[0] = l
                elif line.startswith("input_width:"):
                    input_dimensions = (int(line.removeprefix("input_width:")), input_dimensions[1])
                elif line.startswith("input_height:"):
                    input_dimensions = (input_dimensions[0], int(line.removeprefix("input_height:")))
                elif line.startswith("num_output_classes:"):
                    num_output_classes = int(line.removeprefix("num_output_classes:"))
                else:
                    # layer definition
                    # format: output_channels kernel_size padding [pool_size pool_stride]
                    parts = line.split()
                    assert len(parts) in [3, 5], f"malformatted layer definition: {line}"
                    input_channels = layers[-1][1] if len(layers) > 0 else num_input_channels
                    output_channels = int(parts[0])
                    kernel_size = int(parts[1])
                    padding = int(parts[2])
                    pooling = None
                    if len(parts) == 5:
                        pooling = (int(parts[3]), int(parts[4]))
                    layers.append((input_channels, output_channels, kernel_size, padding, pooling))
    elif parsed["interactive"]:
        if not parsed["o"]:
            num_output_classes = int(input("Enter the number of output classes: "))
        if not parsed["w"] or not parsed["h"]:
            input_dimensions = tuple(map(int, input("Enter space separated input width and height: ").split())) # type: ignore
        if not parsed["l"]:
            num_layers = int(input("Enter the number of layers: "))
        layers = []
        channels = [3]
        for i in range(num_layers):
            print("Layer " + str(i))
            fused_pooling = None # type: ignore
            if i != 0:
                fused_pooling = tuple(map(int, input("  Enter space separated pooling size and stride (both in [1, 16]): ").split())) # type: ignore
            fused_pooling = fused_pooling or None # type: ignore
            output_channels = int(input("  Enter the number of output_channels: "))
            channels.append(output_channels)
            kernel_size = int(input("  Enter the kernel_size as a number (1 or 3): "))
            padding = int(input("  Enter the padding as a number (0, 1, ..., kernel_size-1): "))
            layers.append((channels[-2], channels[-1], kernel_size, padding, fused_pooling))
    else:
        target_size: int = parsed["s"] or 6_000 # type: ignore
        # channels = [3, ...]; each element must be in [1, 1024]
        # layer_bytes[i] = channels[i-1] * channels[i] * kernel_size * kernel_size
        # last_linear_layer_bytes = channels[-1] * dimx * dimy; channels[-1] can at most be 256 for flatten operation, dimx * dimy can also be at most 256
        required_pooling = int(np.ceil(np.sqrt(input_dimensions[0] * input_dimensions[1] / 256))) # how much pooling required by the linear layer
        second_layer_pooling = int(np.ceil(np.sqrt(input_dimensions[0] * input_dimensions[1] / 8192))) # how much pooling required by the second layer, dimx * dimy cannot exceed 8192
        linear_pooling = int(np.ceil(required_pooling / second_layer_pooling))
        linear_inputs = min(int((target_size - 4 * 9 * num_layers) / (num_output_classes * input_dimensions[0] * input_dimensions[1]) * required_pooling * required_pooling), 256, int(1024 / (input_dimensions[0] * input_dimensions[1]) * required_pooling * required_pooling))
        target_size -= int(num_output_classes * linear_inputs * input_dimensions[0] * input_dimensions[1] / required_pooling / required_pooling)
        average_channels = int(np.ceil(np.sqrt(target_size / num_layers / 9)))
        # target_size -= average_channels * average_channels * 9 * (num_layers - 2) + 3 * average_channels * 9 + average_channels * linear_inputs * 9
        
        kernel_size = 3 # 1 or 3
        padding = 1 # 0, 1, KERNEL_SIZE - 1
        fused_pooling: list[None | tuple[int, int]] = [None] * (num_layers - 1) # None or (pool_size in [1, 16], pool_stride in [1, 16])
        fused_pooling[0] = (second_layer_pooling, second_layer_pooling) if second_layer_pooling != 1 else None
        if num_layers > 8: # streaming limit
            non_streaming_pooling = int(np.ceil(np.sqrt(average_channels * input_dimensions[0] * input_dimensions[1] / 8192 / second_layer_pooling / second_layer_pooling)))
            linear_pooling = int(np.ceil(linear_pooling / non_streaming_pooling))
            fused_pooling[6] = (non_streaming_pooling, non_streaming_pooling) if non_streaming_pooling != 1 else None
        fused_pooling[-1] = (linear_pooling, linear_pooling) if linear_pooling != 1 else None
        inner_channels = [average_channels] * num_layers
        print(inner_channels)
        layers = [
            (num_input_channels, inner_channels[0], kernel_size, padding, None)
        ] + [
            (inner_channels[i], inner_channels[i+1], kernel_size, padding, fused_pooling[i]) for i in range(num_layers - 2)
        ] + [
            (inner_channels[-1], linear_inputs, kernel_size, padding, fused_pooling[-1])
        ]
        maximum = 16
        num_streaming = 0
        while True:
            dimx = input_dimensions[0]
            dimy = input_dimensions[1]
            num_streaming = 0
            for i, (ic, _, k, p, f) in enumerate(layers): # type: ignore
                streaming_threshold = 32768 if i == 0 and input_mode == "CHW" else 8192
                ic = max(4, ic)
                if ic * dimx * dimy > streaming_threshold:
                    num_streaming = i + 1
                if f is not None:
                    dimx = (dimx - f[0]) // f[1] + 1 # type: ignore
                    dimy = (dimy - f[0]) // f[1] + 1 # type: ignore
                dimx = (dimx - k + 2 * p) // 1 + 1
                dimy = (dimy - k + 2 * p) // 1 + 1
            if sum(np.ceil(ic / 4) for (ic, _, _, _, _) in layers[:num_streaming+2]) <= 16 and not any(ic > 16 for (ic, _, _, _, _) in layers[:num_streaming]):
                break
            for i in range(1, min(num_streaming+2, len(layers))):
                layers[i-1] = (layers[i-1][0], min(layers[i][0], maximum), layers[i-1][2], layers[i-1][3], layers[i-1][4]) # type: ignore
                layers[i] = (min(layers[i][0], maximum), layers[i][1], layers[i][2], layers[i][3], layers[i][4]) # type: ignore
            maximum -= 1
    print(layers)

    checkpoint_path = train(model_name, dataset_name, layers, input_dimensions, num_output_classes)
    yaml_path, streaming = generate_yaml(checkpoint_path, model_name, dataset_name, layers, input_dimensions, num_output_classes, input_mode)
    quantized_path = quantize(checkpoint_path)
    sample_path = generate_sample(quantized_path, num_input_channels, input_dimensions)
    synthesize(sample_path, yaml_path, quantized_path, model_name, streaming)

if __name__ == "__main__":
    main(sys.argv)
