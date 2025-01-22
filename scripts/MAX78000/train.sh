# Pydoc doesn't support symbolic links, so we'll manually copy files around
cp scripts/MAX78000/animal_dataloader.py MAX78000/ai8x-training/datasets/animal_dataloader.py
cp scripts/MAX78000/yolo_pico.py MAX78000/ai8x-training/models/yolo_pico.py

# Enter training environment
cd MAX78000/ai8x-training
. venv/bin/activate

# # Train Normally
# python train.py --lr 0.1 --optimizer SGD --epochs 200 --deterministic --compress policies/schedule.yaml --model yolo_pico --dataset animal_detection --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"

# Train with QAT
python train.py --lr 0.01 --optimizer SGD --epochs 200 --deterministic --seed 1 --compress policies/schedule.yaml --model yolo_pico --dataset animal_detection --confusion --param-hist --pr-curves --embedding --device MAX78000 --qat-policy policies/qat_policy.yaml "$@"

# Exit training environment
deactivate
cd ../..
