# pipeline.py

# Python imports
import numpy as np

# Sklarn imports
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# PyTorch
import torch

# Local imports
from methods.reader import Reader
from methods.augment import DummyAugment
from methods.loader import DatasetConstructor
from methods.preprocess import PreprocessPipeline

from methods.train import train, predict, train_simple

from util import log
from config import CADICA_DATASET_DIR
from config import (DATA_LOADER_BS,
                    DATA_LOADER_SHUFFLE,
                    DATA_LOADER_N_WORKERS,
                    DATA_LOADER_PIN_MEM)



class ReaderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper around reader since Reader is not constructed based on the Sklearn transformer API"""
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        reader = Reader(dataset_dir=self.dataset_dir)
        return reader.xca_images



class PreprocessWrapper(BaseEstimator, TransformerMixin):
    """Wrapper around Preprocess for the ability to control train/val/test split or k-fold (?)"""
    def __init__(self, as_tensor=True, is_train=True, augmentor=None):
        self.as_tensor = as_tensor
        self.is_train = is_train
        self.augmentor = augmentor
        self.preprocess_pipeline_ = None

    def fit(self, X, y=None):
        self.preprocess_pipeline_ = PreprocessPipeline(
            as_tensor=self.as_tensor,
            is_train=self.is_train,
            augmentor=self.augmentor
        )
        self.preprocess_pipeline_.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.preprocess_pipeline_.split_transform(X)



class StenosisPipeline:
    """
    This is the main wrapper class to construct an integrated pipeline for data reading, preprocessing and modeling.
    """
    def __init__(self, dataset_dir, model: any = None, as_tensor=True, is_train=True, augmentor=None, mode=None):
        if model is None:
            raise TypeError("Model must be provided")

        if mode not in ['single_frame', 'sequence'] or mode is None:
            raise ValueError("Model must be either 'single_frame' or 'sequence'")

        self.dataset_dir = dataset_dir
        self.model = model
        self.mode = mode
        self.as_tensor = as_tensor
        self.is_train = is_train
        self.augmentor = augmentor if augmentor is not None else DummyAugment()

        self.pipeline = Pipeline([
            ('reader', ReaderWrapper(dataset_dir=dataset_dir)),
            ('preprocess', PreprocessWrapper(as_tensor=self.as_tensor,
                                             is_train=self.is_train,
                                             augmentor=self.augmentor)),
            ('loader', DatasetConstructor(mode=self.mode,
                                          batch_size=DATA_LOADER_BS,
                                          shuffle=DATA_LOADER_SHUFFLE,  # this will only affect the train loader as val & test should not be shuffled
                                          num_workers=DATA_LOADER_N_WORKERS,
                                          pin_memory=DATA_LOADER_PIN_MEM,
                                          repeat_channels=True)),
        ])

    def run(self, num_epochs=100, learning_rate=1e-3,  early_stop=50, verbosity=-1, use_pbar=True):
        loader_dict = self.pipeline.fit_transform(None)
        train_loader, val_loader, test_loader = loader_dict.values()

        # for i, (x_batch, y_batch, meta_batch) in enumerate(train_loader, 0):
        #
        #
        #     print(f"==== INSPECTING SAMPLE {i} ====")
        #     print(f"Video tensor shape:  {x_batch}")
        #     print(f"BBox shape:          {y_batch}")
        #     print(f"Metadata dictionary: {meta_batch}")


        log(f"Starting training for model: {self.model.__class__.__name__}")
        results = train(train_loader, val_loader, self.model,
                        num_epochs=num_epochs, learning_rate=learning_rate, early_stop=early_stop,
                        verbosity=verbosity, use_pbar=use_pbar)

        # log(f"Evaluating '{self.model.__class__.__name__}' on the test set")

        return results


if __name__ == "__main__":
    from models.dummy import DummyTorchModel

    pipeline = StenosisPipeline(
        dataset_dir=CADICA_DATASET_DIR,
        model=DummyTorchModel(),
        augmentor=DummyAugment(),

    )
    results = pipeline.run()
    print(results)