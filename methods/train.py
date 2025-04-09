# train.py
from pathlib import WindowsPath
# Pyton imports
from typing import List, Optional, Union

# Torch imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from methods.module import XCADataModule
from methods.reader import XCAImage, XCAVideo

from config import (
    MODEL_CHECKPOINTS_DIR, LOGS_DIR,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)

from lightning_fabric.plugins.environments import LightningEnvironment

def train_model(
        data_list: list[Union['XCAImage', 'XCAVideo']],
        model,
        lightning_module,
        batch_size: int = 8,
        max_epochs: int = 50,
        use_augmentation: bool = True,
        num_workers: int = 4,
        repeat_channels: bool = True,
        normalize_params: dict[str, Union[float, int]] = None,
        train_val_test_split: tuple[float, float, float] = (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
        gpus: Optional[Union[int, List[int]]] = None,
        accumulate_grad_batches: int = 1,
        strategy: Optional[str] = None,
        save_dir: WindowsPath | str = MODEL_CHECKPOINTS_DIR,
        log_dir: str = LOGS_DIR,
):
    """
    Train given model, supports single and multi-gpu
    :param data_list: list of XCAImage or XCAVideo objects
    :param batch_size: batch size for training (per gpu)
    :param max_epochs: maximum number of training epochs
    :param use_augmentation: whether to use augmentation
    :param num_workers: number of cores/workers for data loaders
    :param train_val_test_split: ratio of train/val/test split
    :param gpus: number of GPUs to use or list of GPU indices
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
        filename= model.__class__.__name__ + '-{epoch:02d}-{val_loss:.4f}',
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

    if strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': [checkpoint_callback, early_stop_callback, ],
        'logger': logger,
        'log_every_n_steps': 10,
        'deterministic': False,
        'accumulate_grad_batches': accumulate_grad_batches,
    }

    if gpus is None:
        trainer_kwargs['accelerator'], trainer_kwargs['devices'] = 'auto', 'auto'
    else:
        trainer_kwargs['accelerator'], trainer_kwargs['devices'] = 'gpu', gpus


    if strategy:
        trainer_kwargs['strategy'] = strategy
    elif isinstance(gpus, (list, int)) and (isinstance(gpus, list) and len(gpus) > 1 or isinstance(gpus, int) and gpus > 1):
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)


    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, data_module)
    trainer.test(lightning_module, data_module)
    return lightning_module

