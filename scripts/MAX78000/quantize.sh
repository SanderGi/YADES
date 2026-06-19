# Enter synthesis environment
cd MAX78000/ai8x-synthesis
. venv/bin/activate

# Quantize
python quantize.py ../../data/test.pth.tar ../../data/test-q.pth.tar --device MAX78000 -v "$@"
python quantize.py ../../data/test-qat.pth.tar ../../data/test-qat-q.pth.tar --device MAX78000 -v "$@"
python quantize.py ../../data/yolo-pico-max78000.pth.tar ../../data/yolo-pico-max78000-ptq.pth.tar --device MAX78000 -v "$@"
python quantize.py ../../data/yolo-pico-max78000-qat.pth.tar ../../data/yolo-pico-max78000-qat-q.pth.tar --device MAX78000 -v "$@"

# Exit synthesis environment
deactivate
cd ../..
