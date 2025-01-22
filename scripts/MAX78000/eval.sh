# Pydoc doesn't support symbolic links, so we'll manually copy files around
cp scripts/MAX78000/animal_dataloader.py MAX78000/ai8x-training/datasets/animal_dataloader.py
cp scripts/MAX78000/yolo_pico.py MAX78000/ai8x-training/models/yolo_pico.py

# Enter training environment
cd MAX78000/ai8x-training
. venv/bin/activate

# Eval
# python train.py --model yolo_pico --dataset animal_detection --confusion --evaluate --exp-load-weights-from "../../$1" --device MAX78000 --save-sample 1
python train.py --model yolo_pico --dataset animal_detection --confusion --evaluate --exp-load-weights-from "../../$1" -8 --device MAX78000 --save-sample 1

# Exit training environment
deactivate
cd ../..
