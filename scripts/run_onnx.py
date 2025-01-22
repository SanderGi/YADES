import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def predict(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = np.expand_dims(image, axis=0).astype("float32") / 255.0
    image = np.transpose(image, [0, 3, 1, 2])
    outputs = model.run(None, {"images": image})
    predicted = "animal" if outputs[0][0][0] > outputs[0][0][1] else "non-animal"
    return predicted


def main(args):
    if len(args) == 0 or args[0].lower() == "help":
        print(
            "Usage: python ./scripts/run_onnx.py <image_path> [<model_path>] [<IMG_RESIZE_TO>]"
        )
        return

    image_path = args[0]
    model_path = (
        args[1]
        if len(args) > 1
        else "data/yolo-finetune/10_epoch_224_imgsz/weights/dynamic_quantized.onnx"
    )

    onnx_model = ort.InferenceSession(model_path)

    start = time.perf_counter()
    detection = predict(onnx_model, cv2.imread(image_path))
    end = time.perf_counter()

    print(detection, f"detected in {end - start:0.5f} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
