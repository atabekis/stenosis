# config.py

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# define dataset paths
DANILOV_DATASET_DIR = PROJECT_ROOT / "data/DANILOV"
DANILOV_DATASET_PATH = DANILOV_DATASET_DIR / "dataset"

CADICA_DATASET_DIR = PROJECT_ROOT / "data/CADICA"

# TODO: ADD MODEL PATH AND LOGS(?) PATH
