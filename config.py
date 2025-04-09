# config.py
import torch
import random
import numpy as np
from pathlib import Path
import pytorch_lightning as pl


# ------------ RANDOM SEED -------------- #
SEED = 5
# pl.seed_everything(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------- PATHS ----------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "model_checkpoints"

# ------------- CONTROLS --------------- #

DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512  # pixels

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
assert np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1), 'Train, Validation and Test sizes must sum to 1'

CLASSES = ['__background__', 'stenosis']
NUM_CLASSES = len(CLASSES)

DEBUG = False
DEBUG_SIZE = 0.1  # keep % (DEBUG_SIZE * 100) of data

