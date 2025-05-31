# experiment.py

# Python imports
import os
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
from models.stage3.thanos_detector import THANOSDetector

# Local imports - controls & utility
from util import log
from config import (
    SEED, DEBUG, NUM_WORKERS,
    CADICA_DATASET_DIR, LOGS_DIR,
    DEFAULT_HEIGHT, DEFAULT_WIDTH,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    NUM_CLASSES, POSITIVE_CLASS_ID, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, T_CLIP,
    STAGE1_RETINANET_DEFAULT_CONFIG, STAGE2_TSM_RETINANET_DEFAULT_CONFIG, STAGE3_THANOS_DEFAULT_CONFIG,
    OPTIMIZER_CONFIG,
    TEST_MODEL_ON_KEYBOARD_INTERRUPT, DANILOV_DATASET_DIR,
)


# --- Default base configs ---

DEFAULT_NORMALIZE_PARAMS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

BASE_CONFIG_SINGLE_GPU = {
    'batch_size': 32,
    'num_workers': NUM_WORKERS,
    'strategy': None,
    'learning_rate': OPTIMIZER_CONFIG['base_lr'],  # now config.OPTIMIZER_CONFIG handles these ▼
    'weight_decay': OPTIMIZER_CONFIG['weight_decay'],
    'gradient_clip_val': 1.0,
    'warmup_steps': 100,
    'scheduler': 'cosine',
    'scheduler_patience_config': 5, # patience for mode 'reduce', used as patience in ReduceLROnPlateau
    'patience': 15,
    'normalize_params': DEFAULT_NORMALIZE_PARAMS,
    'precision': '16-mixed',
    'repeat_channels': True,
    't_clip': T_CLIP,
    'jitter': False,
    'seed': SEED,
    'debug': DEBUG,
    'dataset_dir': CADICA_DATASET_DIR,
    'log_dir': LOGS_DIR,
    'num_log_val_images': 2,
    'use_sca': False,
    'subsegment': False,
    'iou_split_thresh': 0.01,
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
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*A new version of Albumentations is available:.*", category=UserWarning)


def setup_reproducibility(seed, deterministic):
    """Seed everything and toggle CUDNN deterministic/benchmark modes."""
    pl.seed_everything(seed, workers=True)
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
        required_keys = ['model_stage', 'max_epochs', 'effective_batch_size']

        if not cfg.get('test_model_path'):
            for key in required_keys:
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

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # to suppress the annoying updates


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

        # 2. Choose strategy
        strategy = cfg.get('strategy')
        if strategy is None and num_gpus > 1:
            strategy = 'ddp'

        self.run_config['gpus'] = devices
        self.run_config['num_target_gpus'] = num_gpus
        self.run_config['strategy'] = strategy


    def _configure_run_params(self):
        """
        Merge in base configs (single‐ or multi‐GPU), then stage-specific model defaults,
        then apply user overrides from argparse (self.config),
        and compute gradient accumulation.
        The final resolved parameters are stored in self.run_config.
        """
        num_gpus = self.run_config.get('num_target_gpus', 0)
        current_run_config_keys = self.run_config.copy()

        self.run_config = {}

        # 1. base GPU config
        if num_gpus > 1:
            base_gpu_config = BASE_CONFIG_MULTI_GPU.copy()
            default_strategy_for_gpus = 'ddp'
        else:
            base_gpu_config = BASE_CONFIG_SINGLE_GPU.copy()
            default_strategy_for_gpus = None

        self.run_config.update(base_gpu_config)

        # 2. stage-specific model config
        stage = self.config.get('model_stage')
        if stage == 1: model_default_config = STAGE1_RETINANET_DEFAULT_CONFIG.copy()
        elif stage == 2: model_default_config = STAGE2_TSM_RETINANET_DEFAULT_CONFIG.copy()
        elif stage == 3:
            model_default_config = STAGE3_THANOS_DEFAULT_CONFIG.copy()
            _height = self.run_config.get('height', self.config.get('height', DEFAULT_HEIGHT))
            _width = self.run_config.get('width', self.config.get('width', DEFAULT_WIDTH))
            _t_clip_for_pe = self.run_config.get('t_clip', self.config.get('t_clip', T_CLIP))

            model_default_config["max_spatial_tokens_pe"] = (_height // 8) * (_width // 8)
            model_default_config["max_temporal_tokens_pe"] = _t_clip_for_pe
        else:
            raise ValueError(f"Invalid model_stage {stage} in _configure_run_params.")

        self.run_config.update(model_default_config)

        # 3. apply user overrides from args
        argparse_overrides = {k: v for k, v in self.config.items() if v is not None}
        self.run_config.update(argparse_overrides)

        if stage == 3:
            final_height = self.run_config.get('height', DEFAULT_HEIGHT)
            final_width = self.run_config.get('width', DEFAULT_WIDTH)
            final_t_clip = self.run_config['t_clip']

            self.run_config["max_spatial_tokens_pe"] = (final_height // 8) * (final_width // 8)
            self.run_config["max_temporal_tokens_pe"] = final_t_clip

            if "num_classes" not in self.run_config or self.run_config["num_classes"] is None:
                self.run_config["num_classes"] = NUM_CLASSES

            if "fpn_out_channels" not in self.run_config or self.run_config["fpn_out_channels"] is None:
                self.run_config["fpn_out_channels"] = self.run_config.get("transformer_d_model", 256)

        if self.run_config.get('strategy') is None and num_gpus > 1:
            self.run_config['strategy'] = default_strategy_for_gpus
        elif self.run_config.get('strategy') == "none":
            self.run_config['strategy'] = None

        self.run_config['gpus'] = current_run_config_keys.get('gpus')
        self.run_config['num_target_gpus'] = current_run_config_keys.get('num_target_gpus')

        if 'strategy' in self.config and self.config['strategy'] is not None:
            self.run_config['strategy'] = self.config['strategy']

        per_device_bs = self.run_config['batch_size']
        target_eff_bs = self.run_config['effective_batch_size']

        actual_num_gpus_for_calc = max(1, num_gpus)
        global_bs_without_accum = per_device_bs * actual_num_gpus_for_calc

        if global_bs_without_accum > 0:
            accum_steps = max(1, int(round(target_eff_bs / global_bs_without_accum)))
        else:
            accum_steps = 1
            log("Warning: global_bs_without_accum is 0. Setting accum_steps to 1.")

        self.run_config['accumulate_grad_batches'] = accum_steps
        self.run_config['effective_batch_size_achieved'] = global_bs_without_accum * accum_steps


    def _setup_data(self):
        """
        Initialize the Reader, load images or videos depending on stage, and build the DataModule.
        """
        dataset_map = {
            'danilov': DANILOV_DATASET_DIR,
            'cadica': CADICA_DATASET_DIR,
            'both': 'both'  # for consistency
        }

        rc = self.run_config

        if str(rc['dataset_dir']).lower() in dataset_map.keys():
            rc['dataset_dir'] = dataset_map[rc['dataset_dir']]


        reader = Reader(
            dataset_dir=rc['dataset_dir'],
            debug=rc['debug'],
            iou_split_thresh=rc['iou_split_thresh'],
            apply_gt_splitting=rc['subsegment']

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
            jitter=rc['jitter']
        )


    def _setup_model(self):
        """Instantiate the appropriate model class based on `model_stage`."""
        stage = self.run_config['model_stage']
        model_init_config = self.run_config

        if stage == 1:
            self.model = FPNRetinaNet(config=model_init_config)
        elif stage == 2:
            self.model = TSMRetinaNet(config=model_init_config)
        elif stage == 3:
            self.model = THANOSDetector(config=model_init_config)


    def _setup_lightning_module(self):
        """Instantiate the LightningModule with all training hyperparameters."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before the LightningModule.")

        cfg = self.run_config
        lightning_module_batch_size_arg = cfg['effective_batch_size_achieved']

        hparams_to_log = {
            'model_name': self.model.__class__.__name__,
            'model_stage': cfg['model_stage'],
            'backbone_variant': cfg.get('backbone_variant'), 'include_p2': cfg.get('include_p2_fpn'),
            'max_epochs': cfg['max_epochs'],
            'target_eff_batch_size': cfg['effective_batch_size'],
            'achieved_eff_batch_size': cfg['effective_batch_size_achieved'],
            'per_device_batch_size': cfg['batch_size'],
            'accumulate_grad_batches': cfg['accumulate_grad_batches'],
            'learning_rate': cfg['learning_rate'],
            'weight_decay': cfg['weight_decay'],
            'warmup_steps': cfg['warmup_steps'],
            'scheduler': cfg.get('scheduler', 'cosine'),
            'use_augmentation': cfg['use_augmentation'],
            'precision': cfg.get('precision', '32-true'),
            't_clip': cfg.get('t_clip', T_CLIP),
            'seed': cfg.get('seed'),
            'dataset_dir': os.path.basename(str(cfg.get('dataset_dir'))),
            'debug': cfg['debug'],
        }

        if cfg.get('resume_from_ckpt'):
            hparams_to_log['resumed_from_ckpt'] = os.path.basename(cfg['resume_from_ckpt'])
        if cfg.get('test_model_path'):
            hparams_to_log['tested_ckpt'] = os.path.basename(cfg['test_model_path'])


        if cfg['model_stage'] in [1, 2, 3]:
            hparams_to_log['focal_alpha'] = cfg.get('focal_loss_alpha')
            hparams_to_log['focal_gamma'] = cfg.get('focal_loss_gamma')
            hparams_to_log['anchor_sizes_p3_first'] = str(
                cfg.get('anchor_sizes', STAGE1_RETINANET_DEFAULT_CONFIG['anchor_sizes'])) # should always be passed to here

        if cfg['model_stage'] == 2:
            hparams_to_log['tsm_shift_fraction'] = cfg.get('tsm_shift_fraction')
            hparams_to_log['tsm_shift_mode'] = cfg.get('tsm_shift_mode')

        if cfg['model_stage'] == 3:
            hparams_to_log['stem_lr'] = cfg.get('stem_learning_rate')
            hparams_to_log['tf_d_model'] = cfg.get('transformer_d_model')
            hparams_to_log['tf_n_head'] = cfg.get('transformer_n_head')
            hparams_to_log['tf_spatial_layers'] = cfg.get('transformer_num_spatial_layers')
            hparams_to_log['tf_temporal_layers'] = cfg.get('transformer_num_temporal_layers')
            hparams_to_log['tf_fpn_levels_proc'] = str(cfg.get('fpn_levels_to_process_temporally'))

        for key, value in hparams_to_log.items():
            if value is None:
                hparams_to_log[key] = "None"
            elif not isinstance(value, (str, int, float, bool)):
                hparams_to_log[key] = str(value)

        self.lightning_module = DetectionLightningModule(
            model=self.model,
            model_stage=cfg['model_stage'],
            learning_rate=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
            warmup_steps=cfg['warmup_steps'],
            max_epochs=cfg['max_epochs'],
            batch_size=lightning_module_batch_size_arg,
            accumulate_grad_batches=cfg['accumulate_grad_batches'],
            # stem_learning_rate=cfg.get('stem_learning_rate', 1e-5),
            scheduler_type=cfg.get('scheduler', 'cosine'),
            use_sca=cfg.get('use_sca', False),
            focal_alpha=cfg.get('focal_loss_alpha'),
            focal_gamma=cfg.get('focal_loss_gamma'),
            positive_class_id=cfg.get('positive_class_id', POSITIVE_CLASS_ID),
            normalize_params=cfg['normalize_params'],
            num_log_val_images=cfg.get('num_log_val_images', 1),
            num_log_test_images=cfg.get('num_log_test_images', 'all'),
            hparams_to_log=hparams_to_log,
            hparam_primary_metric=cfg.get('hparam_primary_metric', 'F1_0.5')
        )

    def _print_config_summary(self):
        log('---- Configuration Summary ----')
        for key in (
            'model_stage', 'debug', 'profiler_enabled', 'pretrained', 'use_sca', 'subsegment',
            'max_epochs', 'patience',
            'batch_size',
            'effective_batch_size',
            'accumulate_grad_batches',
            'effective_batch_size_achieved',
            'learning_rate', 'weight_decay', 'warmup_steps', 'scheduler',
            'use_augmentation',
            'normalize_params', 'repeat_channels', 't_clip', 'jitter',
            'gpus_for_trainer', 'num_gpus_for_calc', 'num_workers', 'strategy_for_trainer',
            'precision', 'deterministic', 'seed',
            'dataset_dir', 'log_dir',
            'test_model_path', 'resume_from_ckpt'
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

        results = train_model(
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

            testing_ckpt_path=rc.get('test_model_path', None),
            resume_from_ckpt_path=rc.get('resume_from_ckpt')
        )


        return results



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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, choices=['off', 'cosine', 'reduce'], default='cosine')


    parser.add_argument("--subsegment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=DEBUG)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeat_channels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)

    # Model specific: general
    parser.add_argument("--backbone_variant", default=None, choices=['b0', 'v2_s', 'resnet18', 'resnet34'])

    # Model-specific: Stages 2, 3
    parser.add_argument('--use_sca', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--t_clip", type=int, default=T_CLIP)
    parser.add_argument("--tsm_shift_fraction", type=float, default=0.125)
    parser.add_argument("--tsm_shift_mode", type=str, choices=["residual","inplace"], default="residual")
    parser.add_argument("--use_grad_ckpt", action=argparse.BooleanOptionalAction, default=False)

    # Paths
    parser.add_argument("--dataset_dir", type=str, default=CADICA_DATASET_DIR)
    parser.add_argument("--log_dir", type=str, default=LOGS_DIR)

    parser.add_argument("--test_model_path", type=str, default=None)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)

    args = parser.parse_args()

    config = {k: v for k, v in vars(args).items() if v is not None}
    config["profiler_enabled"] = args.profile
    if args.profile:
        config.setdefault("profiler_scheduler_conf", DEFAULT_PROFILER_SCHEDULER)

    try:
        experiment = Experiment(config=config)
        experiment.run()

    except KeyboardInterrupt:  # the callback handles graceful shutdown, exit safely here.
        sys.exit(0)

    except Exception as e:
        log(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)

