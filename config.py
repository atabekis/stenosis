# config.py
import numpy as np
from pathlib import Path

from util import get_anchor_config

# ------------ RANDOM SEED -------------- #
SEED = 55

# --------------- PATHS ----------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

LOGS_DIR = PROJECT_ROOT / "logs"

# BACKBONE_MODEL_WEIGHTS = "logs/FPNRetinaNet/augmented/version_1/checkpoints/last.ckpt" # using pretrained model checkpoint
BACKBONE_MODEL_WEIGHTS = None

# ------------- CONTROLS --------------- #
DEBUG = False
DEBUG_SIZE = 0.05  # keep % (DEBUG_SIZE * 100) of data

CADICA_NEGATIVE_ONLY_ON_BOTH = True  # when loading both datasets, this will load only the negative frames from CADICA

# CALLBACK CONTROLS
TEST_MODEL_ON_KEYBOARD_INTERRUPT = True

# if in HPC, the command below will stop training and immediately test the model on available checkpoints:
# echo "TEST" | nc -v hostname 3131   → hostname printed when training starts, for HPC it's a compute node such as mcs-gpua001
REMOTE_TEST_PORT = 3131
REMOTE_TEST_COMMAND = 'TEST\n'


#  T_CLIP is used to set the length of a video sequence to T_CLIP frames, this is applied universally
#  if a video has > T_CLIP frames, T_CLIP of them used. If a video has < T_CLIP, then the video is
#  padded till it reaches T_CLIP frames
T_CLIP = 8

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
assert np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1), 'Train, Validation and Test sizes must sum to 1'

CLASSES = ['__background__', 'stenosis']  # binary task 1: stenosis, 0: background, one foreground class
NUM_CLASSES = len(CLASSES)
POSITIVE_CLASS_ID = 1
NUM_WORKERS = 8


# -------------- DATA MANIPULATION ---------- #
DEFAULT_WIDTH = DEFAULT_HEIGHT = 512  # pixels


APPLY_ADAPTIVE_CONTRAST = True
USE_STD_DEV_CHECK_FOR_CLAHE = True # if False, contrast will be applied to every image
ADAPTIVE_CONTRAST_LOW_STD_THRESH = 20.0  # if std < thresh, contrast is applied
CLAHE_CLIP_LIMIT = 5.0
CLAHE_TILE_GRID_SIZE = (8, 8)


MIN_SUBSEGMENT_LENGTH = 4 #  frames
IOU_THRESH_SUBSEGMENT = 1e-6  # basically bboxes next to each other


# -------------- MODEL-SPECIFIC CONTROLS ---------- #
FOCAL_LOSS_ALPHA = 0.25  # positive anchors weight=alpha, negative anchors weight = 1-alpha
FOCAL_LOSS_GAMMA = 2.0

DETECTIONS_PER_IMG_AFTER_NMS = 3  # give the model some flexibility

INFERENCE_SCORE_THRESH = 0.35  # passed onto score_threshold in RetinaNet
INFERENCE_NMS_THRESH = 0.4   # passed onto nms_thresh in RetinaNet
PRF1_THRESH = INFERENCE_SCORE_THRESH  # used in calculating the Precision Recall, F1 scores at a threshold
IOU_THRESH_METRIC = 0.5  # used in metric calculation, IoU@0.5


# DEFAULTS
DEFAULT_BACKBONE_VARIANT = 'b0'   # can be 'b0', 'v2_s', 'resnet18', 'resnet34' (implemented only for stage 1 for now)
INCLUDE_P2_FPN = True
FPN_OUT_CHANNELS = 256

DEFAULT_ANCHOR_SIZES, DEFAULT_ANCHOR_ASPECT_RATIOS = get_anchor_config(  # get auto-scaled anchor sizes based on:
    current_img_width=DEFAULT_WIDTH,  # image sizes
    current_img_height=DEFAULT_HEIGHT,
    include_p2_fpn=INCLUDE_P2_FPN, # whether to include extra P2 FPN layer
)


# -------- SHARED BASE CONFIG -------- #

OPTIMIZER_CONFIG = {
    'name': 'Adamw',
    'base_lr': 5e-4,
    'weight_decay': 5e-4,
    "differential_lr": {
        "enabled": True,  # false to use base_lr for all params
        "lr_backbone": 1e-4,
        "lr_fpn": 1e-4,
        "lr_transformer_thanos": 5e-5,
        "lr_regression_head": 1e-4,
        "lr_classification_head": 1e-5,
        "lr_other": 1e-5
    }
}


COMMON_BACKBONE_FPN_CONFIG = {
    "backbone_variant": DEFAULT_BACKBONE_VARIANT,
    "include_p2_fpn": INCLUDE_P2_FPN,
    "fpn_out_channels": FPN_OUT_CHANNELS,
    "pretrained_backbone": True,
    "load_weights_from_ckpt": BACKBONE_MODEL_WEIGHTS,
    "ckpt_model_key_prefix": "model."
}


COMMON_RETINANET_CONFIG = {
    **COMMON_BACKBONE_FPN_CONFIG,
    "num_classes": NUM_CLASSES,
    "anchor_sizes": DEFAULT_ANCHOR_SIZES,
    "anchor_aspect_ratios": DEFAULT_ANCHOR_ASPECT_RATIOS,
    "focal_loss_alpha": FOCAL_LOSS_ALPHA,
    "focal_loss_gamma": FOCAL_LOSS_GAMMA,
    "inference_score_thresh": INFERENCE_SCORE_THRESH,
    "inference_nms_thresh": INFERENCE_NMS_THRESH,
    "inference_detections_per_img": DETECTIONS_PER_IMG_AFTER_NMS,
}


CUSTOM_CLS_HEAD_CONFIG = {
    "custom_head":  True,

    "classification_head_dropout_p": 0.3,
    "classification_head_num_convs": 3,
    "classification_head_use_groupnorm": True,
    "classification_head_num_gn_groups": 32,
}


SCA_CONFIG = {
    "t_iou": 0.3,
    "t_frame": 3,
    "t_score_interp": 0.1,
    "max_frame_gap_for_linking": 1,
    "apply_sca_on_val": True,
    "apply_sca_on_test": True,
}

# -------- STAGE 1: EFFNET-B0 + FPN + RETINANET -------- #
STAGE1_RETINANET_DEFAULT_CONFIG = {
    **COMMON_RETINANET_CONFIG,
    **CUSTOM_CLS_HEAD_CONFIG
}

# -------- STAGE 2: EFFNET-B0 + FPN + TSM + RETINANET -------- #
STAGE2_TSM_RETINANET_DEFAULT_CONFIG = {
    **COMMON_RETINANET_CONFIG,
    "t_clip": T_CLIP,
    "tsm_shift_fraction": 0.125,
    "tsm_shift_mode": "residual",
    "tsm_effnet_stages_for_tsm": [3, 5, 6],
    "matcher_high_threshold": 0.5,
    "matcher_low_threshold": 0.3,
    "matcher_allow_low_quality": True,
    "use_gradient_checkpointing": False,
    **CUSTOM_CLS_HEAD_CONFIG
}

# -------- STAGE 3: THANOS + FPN + RETINANET -------- #
STAGE3_THANOS_DEFAULT_CONFIG = {
    **COMMON_RETINANET_CONFIG,
    "transformer_d_model": FPN_OUT_CHANNELS,
    "transformer_n_head": 8,
    "transformer_dim_feedforward": 1024,
    "transformer_num_spatial_layers": 2,
    "transformer_num_temporal_layers": 2,
    "transformer_dropout_rate": 0.1,
    "fpn_levels_to_process_temporally": ["P3", "P4", "P5"],
    "max_spatial_tokens_pe": (DEFAULT_HEIGHT // 8) * (DEFAULT_WIDTH // 8),
    "max_temporal_tokens_pe": T_CLIP,
}
