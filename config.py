# config.py

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# define dataset path
DATASET_DIR = PROJECT_ROOT / "data/stenosis_detection"
DATASET_PATH = DATASET_DIR / "dataset"

# TODO: ADD MODEL PATH AND LOGS(?) PATH
