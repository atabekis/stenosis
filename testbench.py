#testbench.py

import torch
import random
import numpy as np
import pytorch_lightning as pl

from methods.reader import Reader
from methods.train import train_model

from config import (
    SEED,
    NUM_CLASSES,
    CADICA_DATASET_DIR,
)

from models.faster_rcnn import FasterRCNN, FasterRCNNLightningModule


if __name__ == '__main__':
    import os

    os.environ['PYTHONUNBUFFERED'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    pl.seed_everything(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('high')


    reader = Reader(dataset_dir=CADICA_DATASET_DIR)
    xca_images = reader.xca_images
    xca_videos = reader.construct_videos()

    config_single_gpu = {
        'batch_size': 8,
        # 'learning_rate': 1e-3,
        'max_epochs': 100,
        'use_augmentation': True,
        'num_workers': 8,
        'repeat_channels': True,
        'gpus': 1,
        'normalize_params': {'mean': [0.485],  'std': [0.229]},
    }

    configuration = config_single_gpu

    model = FasterRCNN(
        num_classes=NUM_CLASSES,
        pretrained=True,
        trainable_backbone_layers=3
    )

    # Create the Lightning module
    lightning_module = FasterRCNNLightningModule(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.0005
    )

    trained_model = train_model(
        data_list=xca_images,
        model = model,
        lightning_module=lightning_module,
        **config_single_gpu)