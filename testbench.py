#testbench.py
import os
import torch
import random
import argparse
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
from util import log

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train XCA detection model')
    parser.add_argument('--gpus', type=str, default='auto', help='GPU IDs to use (comma-separated, e.g. "0,1,2")')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['ddp', 'ddp_spawn', 'deepspeed', 'fsdp', None],
                        help='Distributed training strategy')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    args = parser.parse_args()


    os.environ['PYTHONUNBUFFERED'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    pl.seed_everything(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('high')

    if args.gpus and args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_ids = None  # Auto-detect

    strategy = args.strategy
    if strategy is None and gpu_ids and len(gpu_ids) > 1:
        strategy = 'ddp'  # default to DDP for multi-gpu

    effective_batch_size = 32
    num_gpus = len(gpu_ids) if gpu_ids else 1
    accumulate_grad_batches = max(1, effective_batch_size // (args.batch_size * num_gpus))

    config = {
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'use_augmentation': args.use_augmentation,
        'num_workers': args.num_workers,
        'repeat_channels': True,
        'gpus': gpu_ids,
        'strategy': strategy,
        'accumulate_grad_batches': accumulate_grad_batches,
        'normalize_params': {'mean': 0.485, 'std': 0.229},
    }

    log(f"Training configuration:")
    log(f"  GPUs: {gpu_ids}")
    log(f"  Strategy: {strategy}")
    log(f"  Batch size per GPU: {args.batch_size}")
    log(f"  Effective batch size: {args.batch_size * num_gpus * accumulate_grad_batches}")
    log(f"  Gradient accumulation steps: {accumulate_grad_batches}")


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