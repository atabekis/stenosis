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

from methods.train import train, predict

from util import log
from config import CADICA_DATASET_DIR
from config import DATA_LOADER_BS, DATA_LOADER_SHUFFLE, DATA_LOADER_N_WORKERS



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
    def __init__(self, dataset_dir, model: any = None, as_tensor=True, is_train=True, augmentor=None):
        if model is None:
            raise TypeError("Model must be provided")

        if model.mode not in ['single_frame', 'sequence']:
            raise ValueError("Model must be either 'single_frame' or 'sequence'")

        self.dataset_dir = dataset_dir
        self.model = model
        self.mode = model.mode
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
                                          num_workers=DATA_LOADER_N_WORKERS)),
        ])

    def run(self):
        loader_dict = self.pipeline.fit_transform(None)
        train_loader, val_loader, test_loader = loader_dict.values()

        for i, (x_batch, y_batch, meta_batch) in enumerate(train_loader, 0):


            print(f"==== INSPECTING SAMPLE {i} ====")
            print(f"Video tensor shape:  {x_batch.shape}")
            print(f"BBox shape:          {y_batch.shape}")
            print(f"Metadata dictionary: {meta_batch}")


        log(f"Starting training for model: {self.model.__class__.__name__}")
        self.model.train_model(train_loader, val_loader)

        log(f"Evaluating '{self.model.__class__.__name__}' on the test set")
        test_score = self.model.evaluate(test_loader)

        # return loader_dict
        return loader_dict, test_score



if __name__ == "__main__":
    from models.dummy import DummyTorchModel

    pipeline = StenosisPipeline(
        dataset_dir=CADICA_DATASET_DIR,
        model=DummyTorchModel(),
        augmentor=DummyAugment()
    )
    splits, score = pipeline.run()
    print(score)