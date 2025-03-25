# config.py

import os
import torch
import random
import numpy as np
from pathlib import Path

import util

# ------------ RANDOM SEED -------------- #
SEED = 5


# --------------- PATHS ----------------- #

PROJECT_ROOT = Path(__file__).resolve().parent

# define dataset paths
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"

CACHE_DIR = PROJECT_ROOT / ".cache"
CACHED_MODELS_DIR = CACHE_DIR / "models"
CACHED_DATA_DIR = CACHE_DIR / "data"


# ------------- CONTROLS --------------- #
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1

DATA_LOADER_BS = 20
DATA_LOADER_SHUFFLE = False
DATA_LOADER_N_WORKERS = 0

# ------------- CUDA & PYTORCH ---------- #

if not globals().get("__CONFIG_INITIALIZED__"):  # prevent these changes from happening at every "import config"
    __CONFIG_INITIALIZED__ = True

    # set up cuda & initialize device
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

    # pass the random seed to everything & make deterministic
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # -------- CACHING --------


    if not CACHE_DIR.exists():
        CACHED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CACHED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        util.log(f"Cache directory not found, created at: {CACHE_DIR}")


# ------------- HELPER FUNCTIONS ---------- #

def reinitialize(seed):
    """Reinitialize the random seed settings, enables for changes during runtime"""
    global SEED
    SEED = seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(SEED)