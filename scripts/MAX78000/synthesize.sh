# Enter synthesis environment
cd MAX78000/ai8x-synthesis
. venv/bin/activate

# Synthesize
python ai8xize.py --verbose --log --test-dir sdk/Examples/MAX78000/CNN --prefix yolo-pico --checkpoint-file "../../$1" --config-file ../../scripts/MAX78000/yolo-pico.yaml --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --sample-input ../../data/sample_animal_detection.npy

# Exit training environment
deactivate
cd ../..
