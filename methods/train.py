# train.py
from pathlib import WindowsPath
# Pyton imports
from typing import List, Optional, Union

# Torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy

from methods.data_module import XCADataModule
from methods.reader import XCAImage, XCAVideo

from config import (
    MODEL_CHECKPOINTS_DIR, LOGS_DIR,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    NUM_WORKERS, POSITIVE_CLASS_ID

)




def train_model(
        data_list: list[Union['XCAImage', 'XCAVideo']],
        model,
        lightning_module,
        batch_size: int = 8,
        max_epochs: int = 50,
        use_augmentation: bool = True,
        num_workers: int = NUM_WORKERS,
        repeat_channels: bool = True,
        normalize_params: dict[str, Union[float, int]] = None,
        train_val_test_split: tuple[float, float, float] = (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
        gpus: Optional[Union[int, List[int]]] = None,
        precision: Optional[str] = None,
        accumulate_grad_batches: int = 1,
        strategy: Optional[str] = None,
        save_dir: WindowsPath | str = MODEL_CHECKPOINTS_DIR,
        log_dir: str = LOGS_DIR,
):
    """
    Train given model, supports single and multi-gpu
    :param model: model to be trained
    :param lightning_module the main training module using pl.LightningModule
    :param data_list: list of XCAImage or XCAVideo objects
    :param batch_size: batch size for training (per gpu)
    :param max_epochs: maximum number of training epochs
    :param use_augmentation: whether to use augmentation
    :param num_workers: number of cores/workers for data loaders
    :param repeat_channels: whether to repeat channels of grayscale image to 3-channel RGB
    :param normalize_params: normalization parameters, expected in the format {mean: x, std: y}
    :param train_val_test_split: ratio of train/val/test split
    :param gpus: number of GPUs to use or list of GPU indices
    :param precision: precision mode
    :param strategy: distributed training strategy {'ddp', 'ddp_spawn', etc.} passed onto Trainer
    :param save_dir: directory to save checkpoints
    :param log_dir: directory to save tensorboard logs
    :param accumulate_grad_batches: number of batches to accumulate for larger effective batch size
    :return: trained model
    """

    data_module = XCADataModule(
        data_list=data_list,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_test_split=train_val_test_split,
        use_augmentation=use_augmentation,
        repeat_channels=repeat_channels,
        normalize_params=normalize_params
    )


    save_dir = save_dir / ("augmented" if use_augmentation else "unaugmented")

    checkpoint_callback = ModelCheckpoint(
        dirpath= save_dir,
        filename= model.__class__.__name__ + '-v{version}-epoch{epoch:02d}-val_loss{val_loss:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min',
    )

    logger = TensorBoardLogger(log_dir, name=model.__class__.__name__)

    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': [checkpoint_callback, early_stop_callback, ],
        'logger': logger,
        'log_every_n_steps': 10,
        'deterministic': False,
        'accumulate_grad_batches': accumulate_grad_batches,
        'precision': precision,
    }

    if gpus == 0: # CPU
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = '1'
    elif isinstance(gpus, (int, list)) and gpus != 0:
        if not torch.cuda.is_available():
            print("Warning: GPUs requested but CUDA not available. Falling back to CPU.")
            trainer_kwargs['accelerator'] = 'cpu'
            trainer_kwargs['devices'] = 1
        else:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = gpus # Pass the int or list
    else: # Includes gpus='auto' or gpus=None (should not happen if testbench sends 'auto' or 0/1/[...])
        trainer_kwargs['accelerator'] = 'auto'
        trainer_kwargs['devices'] = 'auto'


    final_strategy = None
    if strategy:
        if strategy.lower() == 'ddp':
             final_strategy = DDPStrategy(find_unused_parameters=True)
        else:
            final_strategy = strategy
    elif isinstance(trainer_kwargs.get('devices'), list) and len(trainer_kwargs['devices']) > 1:
        print("Auto-configuring DDP strategy for multi-GPU.")
        final_strategy = DDPStrategy(find_unused_parameters=True)
    if final_strategy is not None:
        trainer_kwargs['strategy'] = final_strategy


    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, data_module)
    trainer.test(lightning_module, data_module)
    return lightning_module

