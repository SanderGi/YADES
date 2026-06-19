# Pydoc doesn't support symbolic links, so we'll manually copy files around
cp scripts/MAX78000/animal_dataloader.py MAX78000/ai8x-training/datasets/animal_dataloader.py
cp scripts/MAX78000/elk_dataloader.py MAX78000/ai8x-training/datasets/elk_dataloader.py
cp scripts/MAX78000/yolo_pico.py MAX78000/ai8x-training/models/yolo_pico.py

# Inputs
YAML="../../scripts/MAX78000/yolo-pico-elk.yaml"
DATASET="elk_detection"
MODEL="yolo_pico"
DEVICE="MAX78000"

NUM_EPOCHS=200
LEARNING_RATE=0.01
QAT_POLICY="policies/qat_policy.yaml"
# NUM_EPOCHS=10
# LEARNING_RATE=0.01
# QAT_POLICY="policies/qat_policy_fast.yaml"
SCHEDULING_POLICY="policies/schedule.yaml"
OPTIMIZER="SGD"

# Outputs
CHKPNT="../../data/yolo-pico-max78000-qat-elk.pth.tar"
QUANT_CHKPNT="../../data/yolo-pico-max78000-qat-q-elk.pth.tar"
SAMPLE="../../data/sample_elk_detection.npy"
SYNTH_PREFIX="yolo-pico-elk"

# Enter training environment
cd MAX78000/ai8x-training
. venv/bin/activate
# Train with QAT
python train.py --lr $LEARNING_RATE --optimizer $OPTIMIZER --epochs $NUM_EPOCHS --deterministic --seed 1 --compress $SCHEDULING_POLICY --model $MODEL --dataset $DATASET --confusion --param-hist --pr-curves --embedding --device $DEVICE --qat-policy $QAT_POLICY
# Save checkpoint
cp ./latest_log_dir/qat_best.pth.tar $CHKPNT
# Exit training environment
deactivate
cd ../..

# Enter synthesis environment
cd MAX78000/ai8x-synthesis
. venv/bin/activate
# Quantize
python quantize.py $CHKPNT $QUANT_CHKPNT --device $DEVICE
# Exit synthesis environment
deactivate
cd ../..

# Enter training environment
cd MAX78000/ai8x-training
. venv/bin/activate
# Eval
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANT_CHKPNT -8 --device $DEVICE --save-sample 1
# Save the sample
cp ./sample_$DATASET.npy $SAMPLE
# Exit training environment
deactivate
cd ../..

# Enter synthesis environment
cd MAX78000/ai8x-synthesis
. venv/bin/activate
# Synthesize
python ai8xize.py --verbose --log --test-dir sdk/Examples/MAX78000/CNN --prefix $SYNTH_PREFIX --checkpoint-file $QUANT_CHKPNT --config-file $YAML --device $DEVICE --compact-data --mexpress --timer 0 --display-checkpoint --sample-input $SAMPLE --fifo --overwrite --no-version-check
# Exit synthesis environment
deactivate
cd ../..
