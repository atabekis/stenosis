#testbench.py
import os
from ast import parse

import torch
import random
import argparse
import warnings
import numpy as np

import pytorch_lightning as pl

from methods.reader import Reader
from methods.train import train_model
from methods.detector_module import DetectionLightningModule

from config import (
    SEED,
    NUM_CLASSES,
    CADICA_DATASET_DIR, DEBUG, NUM_WORKERS, POSITIVE_CLASS_ID,
    FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA
)
from models.faster_rcnn import FasterRCNN

from util import log
from models.retinanet_stage1 import Stage1RetinaNet



warnings.filterwarnings(
    "ignore",
    message=r".*Checkpoint directory .* exists and is not empty.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`training_step` returned `None`.*",
    category=UserWarning,
)


lr = 1e-4

config_single_gpu = {
    'batch_size': 32,
    'num_workers': NUM_WORKERS,
    'strategy': None,
    'learning_rate': lr,
    'weight_decay': 1e-4,
    'warmup_steps': 100,
    'normalize_params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'precision': '16-mixed',
    'repeat_channels': True
}

config_multi_gpu = {
    'batch_size': 32,  # per gpu
    'num_workers': NUM_WORKERS,
    'strategy': 'ddp',
    'learning_rate': lr,
    'weight_decay': 1e-4,
    'warmup_steps': 100,
    'normalize_params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}, # Use floats
    'precision': '16-mixed',
    'repeat_channels': True
}




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XCA detection model')
    # --- Arguments that override base configs or control training process ---
    parser.add_argument('--gpus', type=str, default='auto', help='GPU IDs (comma-separated, e.g., "0,1"), "auto" for all available, or 0 for CPU')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU (overrides base config)')
    parser.add_argument('--max_epochs', type=int, default=24, help='Maximum training epochs')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers (overrides base config)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['ddp', 'ddp_spawn', 'deepspeed', 'fsdp', None],
                        help='Distributed training strategy (overrides base config default)')
    parser.add_argument('--precision', type=str, default=None, choices=['16-mixed', 'bf16-mixed', '32-true', '64-true'], help='Training precision (overrides base config)')
    # --- Arguments that are generally fixed for the experiment ---
    parser.add_argument('--use_augmentation', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable data augmentation, default True')
    parser.add_argument('--effective_batch_size', type=int, default=32, help='Target effective batch size for gradient accumulation')
    # --- Model/Data specific ---
    # parser.add_argument('--detections_per_img', type=int, default=100, help='Max detections post-NMS')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True, help='Use pretrained backbone weights')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=DEBUG, help='Debug mode')

    args = parser.parse_args()


    os.environ['PYTHONUNBUFFERED'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    os.environ['SLURM_NTASKS_PER_NODE'] = '10'

    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if args.gpus == 'auto':
            num_target_gpus = num_available_gpus
            trainer_devices = 'auto'  # let PL handle it
        elif args.gpus and args.gpus != '0':
            try:
                gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
                if not all(0 <= g < num_available_gpus for g in gpu_ids):
                    raise ValueError(
                        f"Invalid GPU ID requested in {gpu_ids}. Available GPUs: {list(range(num_available_gpus))}")
                num_target_gpus = len(gpu_ids)
                trainer_devices = gpu_ids
            except ValueError as e:
                log(f"Error parsing GPU IDs: {e}. Exiting.")
                exit(1)
        else:  # if CPU
            num_target_gpus = 0
            trainer_devices = 0
    else:
        log("CUDA not available, running on CPU.")
        num_target_gpus = 0
        trainer_devices = 1

    if num_target_gpus == 0:
        log("Targeting CPU for training.")
    else:
        log(f"Targeting {num_target_gpus} GPU(s) for training. Device IDs for Trainer: {trainer_devices}")

    if num_target_gpus <= 1:
        log("Using single-device base configuration.")
        base_config = config_single_gpu.copy()
        # ensure strategy is None for single device unless explicitly overridden later
        if args.strategy is None:
            base_config['strategy'] = None
    else:
        log("Using multi-GPU base configuration.")
        base_config = config_multi_gpu.copy()
        if args.strategy is None:
            base_config['strategy'] = 'ddp'

    run_config = base_config

    if args.batch_size is not None: run_config['batch_size'] = args.batch_size
    if args.num_workers is not None: run_config['num_workers'] = args.num_workers
    if args.strategy is not None: run_config['strategy'] = args.strategy
    if args.precision is not None: run_config['precision'] = args.precision

    run_config['max_epochs'] = args.max_epochs
    run_config['use_augmentation'] = args.use_augmentation

    per_device_batch_size = run_config['batch_size']
    global_batch_size = per_device_batch_size * max(1, num_target_gpus)
    accumulate_grad_batches = max(1, args.effective_batch_size // global_batch_size)
    effective_batch_size_achieved = global_batch_size * accumulate_grad_batches

    run_config['accumulate_grad_batches'] = accumulate_grad_batches
    run_config['gpus'] = trainer_devices

    log(f'Debug mode?: {args.debug}')
    log("--- Run Configuration ---")
    for key, value in run_config.items():
        log(f"   {key}: {value}")
    log(f"   Calculated effective batch size: {effective_batch_size_achieved}")
    if effective_batch_size_achieved != args.effective_batch_size:
        log(f"   Warning: Could not achieve exact effective batch size {args.effective_batch_size}. Using {effective_batch_size_achieved}.")


    reader = Reader(dataset_dir=CADICA_DATASET_DIR, debug=args.debug)
    xca_images = reader.xca_images
    xca_videos = reader.construct_videos()

    model = Stage1RetinaNet(
        pretrained=args.pretrained,
    )
    #  model = FasterRCNN()

    lightning_module = DetectionLightningModule(
        model=model,
        model_stage=1,
        learning_rate=run_config['learning_rate'],
        weight_decay=run_config['weight_decay'],
        warmup_steps=run_config['warmup_steps'],
        max_epochs=run_config['max_epochs'],
        batch_size=per_device_batch_size * max(1, num_target_gpus),
        accumulate_grad_batches=run_config['accumulate_grad_batches'],
        focal_alpha=getattr(model, 'FOCAL_LOSS_ALPHA', FOCAL_LOSS_ALPHA),
        focal_gamma=getattr(model, 'FOCAL_LOSS_GAMMA', FOCAL_LOSS_GAMMA),
        positive_class_id=POSITIVE_CLASS_ID,
        normalize_params=run_config['normalize_params'],
    )

    trained_model = train_model(
        model=model,
        lightning_module=lightning_module,
        data_list=xca_images,

        max_epochs=run_config['max_epochs'],
        batch_size=run_config['batch_size'],
        num_workers=run_config['num_workers'],
        normalize_params=run_config['normalize_params'],
        accumulate_grad_batches=run_config['accumulate_grad_batches'],
        strategy=run_config['strategy'],
        gpus=run_config['gpus'],

        use_augmentation=run_config['use_augmentation'],
        repeat_channels=run_config['repeat_channels'],
    )
