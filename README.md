# Spatiotemporal Networks for Automated Coronary Artery Stenosis Detection in X-ray Angiography


[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-792DE4.svg)](https://www.pytorchlightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


This repository contains the official implementation for a Bachelor's End Project thesis on stenosis detection in X-ray Coronary Angiography (XCA) videos. The framework is built with PyTorch and PyTorch Lightning.

## Features

-   **Command-Line Interface:** Run all experiments from the command line using `experiment.py`.
-   **Multi-Stage Models:** Supports different model architectures (single-frame, temporal, and transformer-based).
-   **Multi-Dataset Support:** Natively reads and processes the **CADICA** and **DANILOV** datasets.
-   **Rich Data Augmentation:** Uses `albumentations` for geometric and pixel-level augmentations.
-   **HPC/SLURM Ready:** Includes utilities for training on high-performance computing clusters.
-   **Logging:** Logs metrics and image samples to TensorBoard for visualization.

## Tech Stack

The core dependencies are listed below. For a full list of exact versions, please see `requirements.txt`.

-   `lightning` (PyTorch Lightning)
-   `torchvision`
-   `albumentations`
-   `opencv-python`
-   `pandas`
-   `tensorboard`
-   `pycocotools`


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/atabekis/stenosis.git
    cd stenosis
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set up Data Directories:**
    -   Download the required datasets from their sources:
        -   **DANILOV**: [data.mendeley.com/datasets/ydrm75xywg/2](https://data.mendeley.com/datasets/ydrm75xywg/2)
        -   **CADICA**: [data.mendeley.com/datasets/p9bpx9ctcv/2](https://data.mendeley.com/datasets/p9bpx9ctcv/2)
    -   Update the `DANILOV_DATASET_DIR` and `CADICA_DATASET_DIR` paths in `config.py`.

## Configuration

This project uses a layered configuration system:
1.  **`config.py`**: Contains all default parameters, paths, and model configurations.
2.  **Arguments**: Any parameter in `config.py` can be overridden by passing it as an argument to `experiment.py`. This is the recommended way to run experiments.

## Usage

All experiments are executed through the `experiment.py` script.

#### Running an Experiment
Specify the model stage and any other parameters you wish to change.

-   **Train a Stage 2 model with default settings:**
    ```bash
    python experiment.py --model_stage 2 --max_epochs 50
    ```

-   **Train a Stage 3 model with a custom learning rate:**
    ```bash
    python experiment.py --model_stage 3 --learning_rate 5e-5
    ```
    

#### Testing a Pre-trained Model
To run only a test loop on a saved checkpoint, use the `--test_model_path` argument.

```bash
python experiment.py --model_stage 2 --test_model_path "path/to/model.ckpt"
```

#### Resuming Training
To resume an interrupted training run, use the `--resume_from_ckpt` argument.
```bash
python experiment.py --model_stage 2 --resume_from_ckpt "path/to/last.ckpt"
```

## Advanced Features

-   **Remote Testing on HPC/SLURM:** For non-interactive jobs, trigger a final test run by sending a network command. After SSHing into the compute node, run:
    ```bash
    echo "TEST" | nc localhost 3131
    ```
-   **Video Sub-segmentation:** To handle significant pairwise IoU displacement, the data reader can split videos into more stable sub-segments. This is enabled by default for video models and can be toggled with the `--subsegment` / `--no-subsegment` flags.