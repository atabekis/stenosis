# config.py
import numpy as np
from pathlib import Path

from util import get_optimal_workers

# ------------ RANDOM SEED -------------- #
SEED = 55

# --------------- PATHS ----------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

LOGS_DIR = PROJECT_ROOT / "logs"

# ------------- CONTROLS --------------- #
DEBUG = False
DEBUG_SIZE = 0.010  # keep % (DEBUG_SIZE * 100) of data


CADICA_NEGATIVE_ONLY_ON_BOTH = True  # when loading both datasets, this will load only the negative frames from CADICA

# CALLBACK CONTROLS
TEST_MODEL_ON_KEYBOARD_INTERRUPT = False

# if in HPC, the command below will stop training and immediately test the model on available checkpoints:
# echo "TEST" | nc -v localhost 3131
REMOTE_TEST_PORT = 3131
REMOTE_TEST_COMMAND = 'TEST\n'


#  T_CLIP is used to set the length of a video sequence to T_CLIP frames, this is applied universally
#  if a video has > T_CLIP frames, T_CLIP of them used. If a video has < T_CLIP, then the video is
#  padded till it reaches T_CLIP frames
T_CLIP = 24

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
assert np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1), 'Train, Validation and Test sizes must sum to 1'

CLASSES = ['__background__', 'stenosis']  # binary task 1: stenosis, 0: background, one foreground class
NUM_CLASSES = len(CLASSES)
POSITIVE_CLASS_ID = 1
NUM_WORKERS = 8


# -------------- DATA MANIPULATION ---------- #
APPLY_ADAPTIVE_CONTRAST = True
USE_STD_DEV_CHECK_FOR_CLAHE = True # if False, contrast will be applied to every image
ADAPTIVE_CONTRAST_LOW_STD_THRESH = 25.0  # if std < thresh, contrast is applied
CLAHE_CLIP_LIMIT = 5.0
CLAHE_TILE_GRID_SIZE = (8, 8)


MIN_SUBSEGMENT_LENGTH = 4 #  frames
IOU_THRESH_SUBSEGMENT = 1e-6  # basically videos next to each other





# -------------- MODEL-SPECIFIC CONTROLS ---------- #
FOCAL_LOSS_ALPHA = 0.25  # positive anchors weight=alpha, negative anchors weight = 1-alpha
FOCAL_LOSS_GAMMA = 2.0

DETECTIONS_PER_IMG_AFTER_NMS = 20

GIOU_LOSS_COEF = 2.0
L1_LOSS_COEF = 5.0
CLS_LOSS_COEF = 2.0

# DEFAULTS
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512  # pixels

# ANCHOR SIZES FOR 512x512 IMAGES
ANCHOR_SIZES_P345 = ((8, 11, 16), (22, 32, 45), (60, 80, 100))  # for P3, P4 and P5 levels of the PFN (include_p2_pfn = False)
ANCHOR_SIZES_P2345 = ((4, 5.5, 8), (11, 16, 22), (32, 45, 64), (90, 128, 180))  # add P2 level as well

# ANCHOR SIZES FOR 1024x1024 IMAGES
# ANCHOR_SIZES_P345 = ((24, 32, 48), (80, 96, 128), (160, 192, 224))
# ANCHOR_SIZES_P2345 = ((16, 22, 32), (48, 64, 80), (96, 128, 160), (192, 224, 256))


DEFAULT_BACKBONE_VARIANT = 'v2_s'   # can be either 'b0' or 'v2_s'
INCLUDE_P2_FPN = True
FPN_OUT_CHANNELS = 256

DEFAULT_ANCHOR_SIZES = ANCHOR_SIZES_P2345 if INCLUDE_P2_FPN else ANCHOR_SIZES_P345
DEFAULT_ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(DEFAULT_ANCHOR_SIZES)


INFERENCE_SCORE_THRESH = 0.1
INFERENCE_NMS_THRESH = 0.4
PRF1_THRESH = 0.1

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
        "lr_classification_head": 5e-5,
        "lr_other": 1e-5
    }
}


COMMON_BACKBONE_FPN_CONFIG = {
    "backbone_variant": DEFAULT_BACKBONE_VARIANT,
    "include_p2_fpn": INCLUDE_P2_FPN,
    "fpn_out_channels": FPN_OUT_CHANNELS,
    "pretrained_backbone": True,
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
    "classification_head_num_convs": 4,
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
