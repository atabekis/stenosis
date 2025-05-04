# experiment.py

# Python imports
import sys
import argparse
import warnings
import traceback


# Torch + PL
import torch
import pytorch_lightning as pl

# Local imports - methods
from methods.reader import Reader
from methods.train import train_model
from methods.data_module import XCADataModule
from methods.detector_module import DetectionLightningModule

# Local imports - models
from models.stage1.retinanet import FPNRetinaNet
from models.stage2.tsm_retinanet import TSMRetinaNet

# Local imports - controls & utility
from util import log
from config import (
    SEED, DEBUG, NUM_WORKERS,
    CADICA_DATASET_DIR, LOGS_DIR,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    POSITIVE_CLASS_ID, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, CLASSES, T_CLIP
)

# --- Default base configs ---

DEFAULT_NORMALIZE_PARAMS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

BASE_CONFIG_SINGLE_GPU = {
    'batch_size': 32,
    'num_workers': NUM_WORKERS,
    'strategy': None,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'warmup_steps': 100,
    'patience': 10,
    'normalize_params': DEFAULT_NORMALIZE_PARAMS,
    'precision': '16-mixed',
    'repeat_channels': True,
    't_clip': T_CLIP,
    'seed': SEED,
    'debug': DEBUG,
    'dataset_dir': CADICA_DATASET_DIR,
    'log_dir': LOGS_DIR,
}

BASE_CONFIG_MULTI_GPU = {
    **BASE_CONFIG_SINGLE_GPU,
    'strategy': 'ddp',
}

DEFAULT_PROFILER_SCHEDULER = {
    'wait': 1, 'warmup': 1, 'active': 3, 'repeat': 2,
}

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", message=r".*Checkpoint directory .* exists and is not empty.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*`training_step` returned `None`.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*The epoch parameter in `scheduler.step\(\)` was not necessary.*", category=UserWarning)


def setup_reproducibility(seed, deterministic):
    """Seed everything and toggle CUDNN deterministic/benchmark modes."""
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic



class Experiment:
    """
    Setup and execution of the entire pipeline.
    """
    def __init__(self, config: dict[str, any]):
        """
        Initialize experiment with a config dictionary
        :param config: dict of configuration parameters
        """
        self.config = config
        self._validate_config()
        self.run_config = {}

        self.model = None
        self.data_module = None
        self.lightning_module = None


    def _validate_config(self):
        """Ensure configuration keys are present and valid."""
        cfg = self.config
        # Required keys
        for key in ('model_stage', 'max_epochs', 'effective_batch_size', 'gpus'):
            if key not in cfg:
                raise ValueError(f"Missing required configuration key: '{key}'")

        # model_stage must be 1, 2, or 3
        if cfg['model_stage'] not in (1, 2, 3):
            raise ValueError(f"Invalid model_stage: {cfg['model_stage']}. Must be 1, 2, or 3.")

        # if profiler is on but no schedule, fill in default
        if cfg.get('profiler_enabled') and 'profiler_scheduler_conf' not in cfg:
            cfg['profiler_scheduler_conf'] = DEFAULT_PROFILER_SCHEDULER


    def _setup_environment(self):
        """
        Initialize reproducibility and set float32 matmul precision.
        """
        cfg = self.config
        setup_reproducibility(
            cfg.get('seed', SEED),
            cfg.get('deterministic', False)
        )
        torch.set_float32_matmul_precision('high')


    def _determine_devices_and_strategy(self):
        """
        Decide how many GPUs (and which) to use, and pick a distributed strategy.
        Writes results into self.run_config['gpus'], ['num_target_gpus'], and ['strategy'].
        """
        cfg = self.config
        gpus_cfg = cfg.get('gpus', 'auto')
        cuda_available = torch.cuda.is_available()
        max_gpus = torch.cuda.device_count() if cuda_available else 0

        # 1. get number of GPUs and trainer 'devices' argument
        if gpus_cfg == 'auto':
            num_gpus = max_gpus
            devices = 'auto' if cuda_available else 1
        elif isinstance(gpus_cfg, int):
            num_gpus = min(gpus_cfg, max_gpus)
            devices = num_gpus or 1
        elif isinstance(gpus_cfg, str):
            normalized = gpus_cfg.strip().lower()
            if normalized in ('0', 'none'):
                num_gpus = 0
                devices = 1
            else:
                # parse comma-separated GPU indices
                ids = [int(x) for x in normalized.split(',') if x]
                if not ids or any(i < 0 or i >= max_gpus for i in ids):
                    raise ValueError(f"Invalid GPU IDs '{gpus_cfg}'. Available: 0..{max_gpus - 1}")
                num_gpus = len(ids)
                devices = ids
        else:
            raise ValueError(f"Unsupported type for 'gpus' config: {type(gpus_cfg)}")

        # 2. Choose strategy: user override or DDP if multi-GPU
        strategy = cfg.get('strategy')
        if strategy is None and num_gpus > 1:
            strategy = 'ddp'

        self.run_config['gpus'] = devices
        self.run_config['num_target_gpus'] = num_gpus
        self.run_config['strategy'] = strategy


    def _configure_run_params(self):
        """
        Merge in base configs (single‐ or multi‐GPU), apply user overrides,
        and compute gradient accumulation to hit the target effective batch size.
        """
        num_gpus = self.run_config.get('num_target_gpus', 0)

        # 1. Pick and change base config
        if num_gpus > 1:
            base = BASE_CONFIG_MULTI_GPU.copy()
            default_strategy = 'ddp'
        else:
            base = BASE_CONFIG_SINGLE_GPU.copy()
            default_strategy = None

        if self.run_config.get('strategy') is None:
            base['strategy'] = default_strategy

        # 2. Merge base & user config (user keys win)
        self.run_config.update({**base, **self.config})

        # 3. Compute accumulate_grad_batches
        per_device_bs = self.run_config['batch_size']
        target_bs = self.run_config['effective_batch_size']
        global_bs = per_device_bs * max(1, num_gpus)

        if global_bs: accum_steps = max(1, target_bs // global_bs)
        else: accum_steps = 1

        self.run_config['accumulate_grad_batches'] = accum_steps
        self.run_config['effective_batch_size_achieved'] = global_bs * accum_steps


    def _setup_data(self):
        """
        Initialize the Reader, load images or videos depending on stage, and build the DataModule.
        """
        rc = self.run_config
        reader = Reader(
            dataset_dir=rc['dataset_dir'],
            debug=rc['debug'],
            t_clip=rc['t_clip'],
        )

        stage = rc['model_stage']
        if stage == 1:
            data_list = reader.xca_images
            kind = "images"
        elif stage in (2, 3):
            data_list = reader.construct_videos()
            kind = "videos"
        else:
            raise ValueError(f"Invalid model_stage {stage} in run_config")

        if not data_list:
            raise ValueError(f"No {kind!r} found for stage {stage}")

        self.data_module = XCADataModule(
            data_list=data_list,
            batch_size=rc['batch_size'],
            num_workers=rc['num_workers'],
            train_val_test_split=rc.get(
                'train_val_test_split',
                (TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
            ),
            use_augmentation=rc['use_augmentation'],
            repeat_channels=rc['repeat_channels'],
            normalize_params=rc['normalize_params'],
            t_clip=rc['t_clip'],
            seed=rc['seed'],
        )

    def _setup_model(self):
        """Instantiate the appropriate model class based on `model_stage`."""
        stage = self.run_config['model_stage']
        pretrained = self.run_config.get('pretrained', True)

        if stage == 1:
            self.model = FPNRetinaNet(pretrained=pretrained)

        elif stage == 2:
            self.model = TSMRetinaNet(
                t_clip=self.run_config['t_clip'],
                tsm_div=self.run_config.get('tsm_div', 8),
                tsm_shift_mode=self.run_config.get('tsm_shift_mode', 'residual'),
                pretrained_backbone=pretrained,
            )

        elif stage == 3:
            raise NotImplementedError("Stage 3 model instantiation is not implemented yet.")

        else:
            raise ValueError(f"Invalid model_stage {stage!r} in run_config")


    def _setup_lightning_module(self):
        """Instantiate the LightningModule with all training hyperparameters."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before the LightningModule.")

        cfg = self.run_config
        effective_bs = cfg['batch_size'] * max(1, cfg['num_target_gpus'])

        self.lightning_module = DetectionLightningModule(
            model=self.model,
            model_stage=cfg['model_stage'],
            learning_rate=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
            warmup_steps=cfg['warmup_steps'],
            max_epochs=cfg['max_epochs'],
            batch_size=effective_bs,
            accumulate_grad_batches=cfg['accumulate_grad_batches'],
            stem_learning_rate=cfg.get('stem_learning_rate', 1e-5),
            focal_alpha=getattr(self.model, 'FOCAL_LOSS_ALPHA', FOCAL_LOSS_ALPHA),
            focal_gamma=getattr(self.model, 'FOCAL_LOSS_GAMMA', FOCAL_LOSS_GAMMA),
            positive_class_id=cfg.get('positive_class_id', POSITIVE_CLASS_ID),
            normalize_params=cfg['normalize_params'],
            num_log_val_images=cfg.get('num_log_val_images', 1),
            num_log_test_images=cfg.get('num_log_test_images', 'all'),
        )

    def _print_config_summary(self):
        log('---- Configuration Summary ----')
        for key in (
                'debug',
                'profiler_enabled',
                'model_stage',
                'batch_size',
                'effective_batch_size',
                'learning_rate',
                't_clip',
                'gpus',
                'num_workers',
                'strategy',
                'precision',
                'deterministic'
        ):
            log(f'   {key}: {self.run_config.get(key)}')


    def run(self):
        """Execute the full experiment pipeline end-to-end."""

        self._setup_environment()
        self._determine_devices_and_strategy()
        self._configure_run_params()

        self._print_config_summary()

        self._setup_data()
        self._setup_model()
        self._setup_lightning_module()


        rc = self.run_config

        return train_model(
            # core objects
            data_module=self.data_module,
            model=self.model,
            lightning_module=self.lightning_module,

            # data & augmentation settings
            use_augmentation=rc['use_augmentation'],

            # trainer hyperparameters
            max_epochs=rc['max_epochs'],
            patience=rc.get('patience'),
            gpus=rc['gpus'],
            precision=rc['precision'],
            accumulate_grad_batches=rc['accumulate_grad_batches'],
            strategy=rc['strategy'],
            deterministic=rc.get('deterministic', False),

            # logging & profiling
            log_dir=rc['log_dir'],
            profiler_enabled=rc.get('profiler_enabled', False),
            profiler_scheduler_conf=rc.get('profiler_scheduler_conf'),
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XCA detection model")
    # Core args
    parser.add_argument("--model_stage", type=int, choices=[1, 2, 3], required=True, help="Stage to train (1: RetinaNet, 2: TSM, 3: Transformer)")
    parser.add_argument("--max_epochs", type=int, default=24)
    parser.add_argument("--gpus", type=str, default="auto", help="GPU IDs comma-separated, 'auto', or 0 for CPU")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU batch size (overrides base config)")
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)

    # Secondary & flags
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--strategy", type=str, choices=["ddp","ddp_spawn","deepspeed","fsdp","none"], default=None)
    parser.add_argument("--precision", type=str, choices=["16-mixed","bf16-mixed","32-true","64-true"], default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--use_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=DEBUG)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeat_channels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)

    # Model-specific
    parser.add_argument("--t_clip", type=int, default=T_CLIP)

    # Model-specific: TSM
    parser.add_argument("--tsm_div", type=int, default=8)
    parser.add_argument("--tsm_shift_mode", type=str, choices=["residual","inplace"], default="residual")

    # Paths
    parser.add_argument("--dataset_dir", type=str, default=CADICA_DATASET_DIR)
    parser.add_argument("--log_dir", type=str, default=LOGS_DIR)

    args = parser.parse_args()

    config = {k: v for k, v in vars(args).items() if v is not None}
    config["profiler_enabled"] = args.profile
    if args.profile:
        config.setdefault("profiler_scheduler_conf", DEFAULT_PROFILER_SCHEDULER)

    try:
        experiment = Experiment(config=config)
        experiment.run()
    except Exception as e:
        log(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)

