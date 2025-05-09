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

DEBUG = False
DEBUG_SIZE = 0.10  # keep % (DEBUG_SIZE * 100) of data

NUM_WORKERS = get_optimal_workers() if not DEBUG else 4


# -------------- MODEL-SPECIFIC CONTROLS ---------- #
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
GIOU_LOSS_COEF = 2.0
L1_LOSS_COEF = 5.0
CLS_LOSS_COEF = 2.0
POSITIVE_CLASS_ID = 1