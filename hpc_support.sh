#!/bin/bash

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install lightning
pip install numpy pandas opencv-python matplotlib albumentations
pip install torchmetrics[detection]
pip install -U 'tensorboard'
