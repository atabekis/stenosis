# train.py

# Pyton imports
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

# Torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import schedule, tensorboard_trace_handler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local imports
from util import log
from config import LOGS_DIR, TEST_MODEL_ON_KEYBOARD_INTERRUPT


def train_model(
        model,
        data_module,
        lightning_module,
        max_epochs: int = 50,
        patience: int = 10,
        gradient_clip_val: float = 1.0,
        use_augmentation: bool = True,
        gpus: Optional[Union[int, List[int]]] = None,
        precision: Optional[str] = None,
        accumulate_grad_batches: int = 1,
        strategy: Optional[str] = None,
        deterministic: bool = True,
        log_dir: str = LOGS_DIR,
        profiler_enabled: bool = False,
        profiler_scheduler_conf: Optional[dict] = None,

        resume_from_ckpt_path: Optional[str] = None,
        testing_ckpt_path:Optional[str] = None,
):
    """
    Train given model, supports single and multi-gpu
    :param model: model to be trained
    :param data_module: PyTorch Lightning data module
    :param lightning_module the main training module using pl.LightningModule
    :param max_epochs: maximum number of training epochs
    :param patience: number of epochs to wait before early stopping
    :param use_augmentation: whether to use augmentation
    :param gpus: number of GPUs to use or list of GPU indices
    :param precision: precision mode
    :param strategy: distributed training strategy {'ddp', 'ddp_spawn', etc.} passed onto Trainer
    :param deterministic: whether to enable deterministic training
    :param log_dir: directory to save tensorboard logs
    :param accumulate_grad_batches: number of batches to accumulate for larger effective batch size
    :param profiler_enabled: Whether PyTorch Lightning Profiler instance (e.g., PyTorchProfiler) activated
    :param profiler_scheduler_conf: config dictionary for steps/warmup/cycle to be used by the profiler scheduler
    :param testing_ckpt_path: If provided, skip training and only run testing using this checkpoint.

    :return: trained model
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    enable_pbar = not "SLURM_JOB_ID" in os.environ

    experiment_name_core = model.__class__.__name__

    if testing_ckpt_path: experiment_folder_suffix = 'test_only'
    elif resume_from_ckpt_path: experiment_folder_suffix = 'resumed_training'
    else: experiment_folder_suffix = 'augmented' if use_augmentation else 'unaugmented'


    experiment_name = f'{experiment_name_core}/{experiment_folder_suffix}'

    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    checkpoint_callback_map = ModelCheckpoint(
        filename=model.__class__.__name__ + '-{epoch:02d}-{val_mAP:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_mAP',
        mode='max',
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        save_last=True,
    )

    checkpoint_callback_val = ModelCheckpoint(
        filename=model.__class__.__name__ + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val/mAP',
        patience=patience,
        verbose=True,
        mode='max',
        log_rank_zero_only=True,
        check_finite=True,
    )


    profiler = None
    if profiler_enabled:
        try:
            prof_schedule = schedule(**profiler_scheduler_conf)
            profiler = PyTorchProfiler(
                schedule=prof_schedule,
                on_trace_ready=tensorboard_trace_handler(dir_name=logger.log_dir, use_gzip=True),
                profile_memory=True,
                record_shapes=True,
                with_stack=False,
            )
        except ImportError:
            log("Error: `torch.profiler` related imports failed. Is PyTorch version sufficient (>=1.8)? Disabling profiler.")
            profiler = None
        except Exception as e:
            log(f"Error configuring PyTorchProfiler: {e}. Disabling profiler.")
            profiler = None

    trainer_kwargs = {
        'max_epochs': max_epochs,
        'gradient_clip_val': gradient_clip_val,
        'callbacks': [checkpoint_callback_map, checkpoint_callback_val, early_stop_callback, ],
        'logger': logger,
        'log_every_n_steps': 10,
        'deterministic': deterministic,
        'accumulate_grad_batches': accumulate_grad_batches,
        'precision': precision,
        'enable_progress_bar': enable_pbar, # if SLURM env. do not print pbar
        'profiler': profiler
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
            trainer_kwargs['devices'] = gpus
    else:
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

    checkpoint_callback = checkpoint_callback_map  # TODO: add control between two callbacks

    results_dict =  {
        "trainer": trainer,
        "logger": logger,
        "checkpoint_callback": checkpoint_callback,
    }

    if testing_ckpt_path:
        log(f"Performing testing only using checkpoint: {testing_ckpt_path}")
        trainer.test(lightning_module, datamodule=data_module, ckpt_path=testing_ckpt_path)
        results_dict["tested_checkpoint_path"] = testing_ckpt_path
    else:

        try:
            log(f'Starting training. Resuming from checkpoint: {True if resume_from_ckpt_path else False}')
            trainer.fit(lightning_module, data_module, ckpt_path=resume_from_ckpt_path)
        except KeyboardInterrupt:
            if TEST_MODEL_ON_KEYBOARD_INTERRUPT:
                log(f'Keyboard interrupt received. Testing model (best).')
                trainer.test(lightning_module, datamodule=checkpoint_callback.best_model_path)
            else:
                sys.exit(1)

        ckpt_to_test = checkpoint_callback.best_model_path
        if not ckpt_to_test:
            log("No best model checkpoint found from this training run. Attempting to test 'last' checkpoint.")
            ckpt_to_test = 'last'

        log(f"Finished training. Testing with checkpoint: {ckpt_to_test}")
        trainer.test(lightning_module, datamodule=data_module, ckpt_path=ckpt_to_test)

    return results_dict



