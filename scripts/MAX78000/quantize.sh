# Enter synthesis environment
cd MAX78000/ai8x-synthesis
. venv/bin/activate

# Quantize
python quantize.py ../../data/yolo-pico-max78000.pth.tar ../../data/yolo-pico-max78000-ptq.pth.tar --device MAX78000 -v "$@"
# python quantize.py ../../data/yolo-pico-max78000-qat.pth.tar ../../data/yolo-pico-max78000-qat-q.pth.tar --device MAX78000 -v "$@"

# Exit training environment
deactivate
cd ../..
