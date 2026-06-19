import sys
import time
from ultralytics import YOLO


def main(args):
    if len(args) == 0 or args[0].lower() == "help":
        print(
            "Usage: python ./scripts/run_yolo.py <image_path> [<model_path>] [<IMG_RESIZE_TO>]"
        )
        return

    image_path = args[0]
    model_path = (
        args[1]
        if len(args) > 1
        else "data/yolo-finetune/10_epoch_224_imgsz/weights/best.pt"
    )

    model = YOLO(model_path)

    start = time.perf_counter()
    detection = model(image_path, verbose=False)[0]
    end = time.perf_counter()

    print(
        detection.names[detection.probs.top1],
        f"detected in {end - start:0.5f} seconds",
        f"with confidence {detection.probs.top1conf:.2f}",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
