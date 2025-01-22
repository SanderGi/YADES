#!/bin/bash

pip install -r requirements.txt

# download dataset from https://www.kaggle.com/datasets/tylerelvis/animal-vs-non-animal-image-recognition-dataset
curl -L -o ./.data/animal-no-animal-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/tylerelvis/animal-vs-non-animal-image-recognition-dataset

# download pretrained yolo weights
curl -L -o ./.data/yolov8n-cls.pt\
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt
