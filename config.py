# config.py
import numpy as np
from pathlib import Path

from util import get_optimal_workers

# ------------ RANDOM SEED -------------- #
SEED = 5

# --------------- PATHS ----------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

LOGS_DIR = PROJECT_ROOT / "logs"

# ------------- CONTROLS --------------- #

DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512  # pixels

# -- !Important
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


# ------------ RUN-SPECIFIC CONTROLS --------------#
DEBUG = False
DEBUG_SIZE = 0.25  # keep % (DEBUG_SIZE * 100) of data

NUM_WORKERS = get_optimal_workers() if not DEBUG else 4


# -------------- MODEL-SPECIFIC CONTROLS ---------- #
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
DETECTIONS_PER_IMG_AFTER_NMS = 1

GIOU_LOSS_COEF = 2.0
L1_LOSS_COEF = 5.0
CLS_LOSS_COEF = 2.0

# DEFAULTS
DEFAULT_ANCHOR_SIZES = ((8, 11, 16), (22, 32, 45), (64, 90, 128))
# DEFAULT_ANCHOR_SIZES = ((16, 24, 32), (48, 64, 96), (128, 192, 256))
DEFAULT_ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(DEFAULT_ANCHOR_SIZES)

FPN_OUT_CHANNELS = 256

INFERENCE_SCORE_THRESH = 0.1
INFERENCE_NMS_THRESH = 0.4

# -------- SHARED BASE CONFIG -------- #
COMMON_RETINANET_CONFIG = {
    "fpn_out_channels": FPN_OUT_CHANNELS,
    "num_classes": NUM_CLASSES,
    "anchor_sizes": DEFAULT_ANCHOR_SIZES,
    "anchor_aspect_ratios": DEFAULT_ANCHOR_ASPECT_RATIOS,
    "focal_loss_alpha": FOCAL_LOSS_ALPHA,
    "focal_loss_gamma": FOCAL_LOSS_GAMMA,
    "inference_score_thresh": INFERENCE_SCORE_THRESH,
    "inference_nms_thresh": INFERENCE_NMS_THRESH,
    "inference_detections_per_img": DETECTIONS_PER_IMG_AFTER_NMS,
    "pretrained_backbone": True,
}

# -------- STAGE 1: EFFNET-B0 + FPN + RETINANET -------- #
STAGE1_RETINANET_DEFAULT_CONFIG = {
    **COMMON_RETINANET_CONFIG,
}

# -------- STAGE 2: EFFNET-B0 + FPN + TSM + RETINANET -------- #
STAGE2_TSM_RETINANET_DEFAULT_CONFIG = {
    **COMMON_RETINANET_CONFIG,
    "t_clip": T_CLIP,
    "tsm_shift_fraction": 0.125,
    "tsm_shift_mode": "residual",
    "tsm_effnet_stages_for_tsm": [3, 5, 6],
    "matcher_high_threshold": 0.5,
    "matcher_low_threshold": 0.4,
    "matcher_allow_low_quality": True,
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
