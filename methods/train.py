# train.py

# Pyton imports
import os
import sys
from pathlib import Path
from typing import Optional, Union

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
from config import LOGS_DIR, DEBUG
from methods.callbacks import TestOnKeyboardInterruptCallback, RemoteTestTriggerCallback


def train_model(
        model,
        data_module,
        lightning_module,
        max_epochs: int = 50,
        patience: int = 10,
        gradient_clip_val: float = 1.0,
        use_augmentation: bool = True,
        gpus: Optional[Union[int, list[int]]] = None,
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
    else:
        experiment_folder_suffix = 'augmented' if use_augmentation else 'unaugmented'

    if DEBUG: experiment_folder_suffix = 'debug'


    experiment_name = f'{experiment_name_core}/{experiment_folder_suffix}'

    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    checkpoint_callback_val = ModelCheckpoint(
        filename=model.__class__.__name__ + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        save_last=True,
    )

    checkpoint_callback_map = ModelCheckpoint(
        filename=model.__class__.__name__ + '-{epoch:02d}-{val_mAP:.4f}',
        save_top_k=5,
        verbose=False,
        monitor='val_mAP',
        mode='max',
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

    test_on_interrupt_callback = TestOnKeyboardInterruptCallback()
    test_on_remote_trigger = RemoteTestTriggerCallback()

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

    accelerator_arg, devices_arg, estimated_num_devices = determine_device_config(gpus)

    checkpoint_callback = checkpoint_callback_val  # this is where we will go back to test the model from

    trainer_kwargs = {
        'max_epochs': max_epochs,
        'gradient_clip_val': gradient_clip_val,
        'callbacks':
            [checkpoint_callback_val,
             checkpoint_callback_map,

             early_stop_callback,

             test_on_interrupt_callback,
             test_on_remote_trigger,
             ],
        'logger': logger,
        'log_every_n_steps': 10,
        'deterministic': deterministic,
        'accumulate_grad_batches': accumulate_grad_batches,
        'precision': precision,
        'enable_progress_bar': enable_pbar, # if SLURM env. do not print pbar
        'profiler': profiler,
        'num_nodes': 1,  # will need to change this if using more than one node on HPC
        'accelerator': accelerator_arg,
        'devices': devices_arg,
    }


    ddp_config = {"find_unused_parameters": False}

    if strategy:
        strategy_lower = strategy.lower()
        if strategy_lower == "ddp":
            trainer_kwargs['strategy'] = DDPStrategy(**ddp_config)
        else:
            trainer_kwargs['strategy'] = strategy
            log(f"Using strategy: {strategy}")
    elif accelerator_arg == "gpu" and estimated_num_devices > 1:
        trainer_kwargs['strategy'] = DDPStrategy(**ddp_config)


    trainer = pl.Trainer(**trainer_kwargs)


    results_dict =  {
        "trainer": trainer,
        "logger": logger,
        "checkpoint_callback": checkpoint_callback,
        "final_test_metrics": None,
        "final_test_results": None,
    }

    if testing_ckpt_path:
        log(f"Performing testing only using checkpoint: {testing_ckpt_path}")
        ckpt_to_test = testing_ckpt_path
    else:
        log(f"Starting training. Resuming from checkpoint: {bool(resume_from_ckpt_path)}")
        trainer.fit(lightning_module, data_module, ckpt_path=resume_from_ckpt_path)

        if checkpoint_callback and checkpoint_callback.best_model_path:
            ckpt_to_test = checkpoint_callback.best_model_path
            log(f"Best model checkpoint found: {ckpt_to_test}")
        elif checkpoint_callback and checkpoint_callback.last_model_path:
            ckpt_to_test = checkpoint_callback.last_model_path
            log(f"No best checkpoint found; using last model checkpoint: {ckpt_to_test}")
        else:
            ckpt_to_test = "last"
            log("No checkpoint files found, falling back to 'last' for trainer.test().")

    log(f"Testing with checkpoint: {ckpt_to_test}")
    trainer.test(lightning_module, datamodule=data_module, ckpt_path=ckpt_to_test)

    results_dict.update({
        "tested_checkpoint_path": ckpt_to_test,
        "final_test_metrics": lightning_module.final_test_metrics,
        "final_test_results": lightning_module.final_test_results,
    })

    return results_dict



def determine_device_config(
    gpus_param: Optional[Union[int, list[int], str]]
) -> tuple[str, Union[int, list[int], str], int]:
    cuda_available = torch.cuda.is_available()
    num_visible = torch.cuda.device_count() if cuda_available else 0

    # --- SLURM multi-task logic ---
    if "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")) > 1:
        if num_visible > 0:
            log(f"SLURM multi-task: this task sees {num_visible} GPU(s).")
            return "gpu", num_visible, num_visible
        log("SLURM multi-task, but no GPUs visible or CUDA unavailable. Using CPU.")
        return "cpu", 1, 1

    # --- No gpus requested or available ---
    if gpus_param == 0 or not cuda_available:
        if gpus_param not in (0, None) and not cuda_available:
            log("Warning: GPUs requested but CUDA not available. Falling back to CPU.")
        return "cpu", 1, 1

    # --- auto gpu (None or auto) ---
    if gpus_param is None or (isinstance(gpus_param, str) and gpus_param.lower() == "auto"):
        return "gpu", "auto", num_visible

    # --- int gpu count ---
    if isinstance(gpus_param, int):
        count = min(gpus_param, num_visible)
        if gpus_param > num_visible:
            log(f"Warning: Requested {gpus_param} GPUs; only {num_visible} available. Using {count}.")
        return "gpu", count, count

    # --- parse list or comma-sep string into indices ---
    indices: list[int]
    if isinstance(gpus_param, str):
        try:
            indices = [int(x) for x in gpus_param.split(",") if x.strip() != ""]
            if not indices:
                raise ValueError
        except ValueError:
            log(f"Error parsing GPU string '{gpus_param}'. Defaulting to 'auto'.")
            return "gpu", "auto", num_visible

    elif isinstance(gpus_param, list):
        indices = gpus_param
    else:
        log(f"Warning: Unexpected type {type(gpus_param)} for `gpus`. Defaulting to 'auto'.")
        return "gpu", "auto", num_visible

    # validate indices
    valid = [i for i in indices if 0 <= i < num_visible]
    if not valid:
        if indices:
            log(f"Warning: No valid GPU indices in {indices} for {num_visible} available GPUs. Using 'auto'.")
        else:
            log("Warning: Empty GPU list provided. Using 'auto'.")
        return "gpu", "auto", num_visible

    if len(valid) != len(indices):
        log(f"Warning: Some indices invalid. Requested {indices}; using valid subset: {valid}.")

    return "gpu", valid, len(valid)



