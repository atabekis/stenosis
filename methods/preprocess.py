# preprocess.py

# Python imports

import numpy as np
import concurrent.futures
from collections import defaultdict

# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GroupKFold

# Torch imports
import torch

# Local imports
from util import log, cache_data
from methods.reader import Reader, XCAImage
from config import DEVICE, SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, CADICA_DATASET_DIR


class XCAVideo:
    """
    Represents a XCA video sequence constructed from XCAImage instances
    """
    def __init__(self, video_tensor: torch.Tensor, bboxes: torch.Tensor | None, meta: dict):
        self.video_tensor = video_tensor
        self.bboxes = bboxes
        self.meta = meta

    def __repr__(self):
        bbox_shape = tuple(self.bboxes.shape) if isinstance(self.bboxes, torch.Tensor) else None
        return (f"XCAVideo("
                f"meta={self.meta}, "
                f"tensor_shape={tuple(self.video_tensor.shape)}, "
                f"bboxes_shape={bbox_shape})")


class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Sklearn-enabled transformer to lead images from XCAImage objects, uses multi-threading for the massive I/O operations
    from the reader.get_image() method.
    """
    def __init__(self, as_tensor: bool = False):
        self.as_tensor = as_tensor
        self.device = DEVICE if as_tensor else None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def load_image(xca_obj: XCAImage):
            img = xca_obj.get_image()  # returns a numpy array
            if self.as_tensor:
                tensor = torch.from_numpy(img).float()  # we do not move to GPU yet since some of the next operations
                                                        # need to be in the CPU
                return tensor
            return img

        # multi-threading for I/O
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loaded_images = list(executor.map(load_image, X))
        return loaded_images



class Normalizer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer to normalize images, only for the training images to prevent data leak
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        self.mean_ = None  # fitted
        self.std_ = None


    def fit(self, X, y=None):
        if self.is_train:
            all_pixels = np.concatenate([
                (img.cpu().numpy() if isinstance(img, torch.Tensor) else img).ravel()  # turn tensor into np (if tensor)
                for img in X
            ])
            self.mean_ = float(all_pixels.mean())
            self.std_ = float(all_pixels.std())
        return self


    def transform(self, X):
        if self.is_train and (self.mean_ is None or self.std_ is None):
            raise ValueError("Must call fit() on training data first")

        def normalize(img):
            if isinstance(img, torch.Tensor):
                return (img - self.mean_) / self.std_
            return (img - self.mean_) / self.std_

        return [normalize(img) for img in X]



class VideoAggregator(BaseEstimator, TransformerMixin):
    """
    While the reader.get() method (semi-)aggregates the objects, I'll add this method as redundancy.

    Groups images by (patient_id, video_id) and sorts the frames by frame_nr to maintain temporal order.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        videos, meta_info = defaultdict(list), {}

        for xca, img_data in X:
            key = (xca.patient_id, xca.video_id)
            videos[key].append((xca.frame_nr, img_data, xca.bbox))
            if key not in meta_info:
                meta_info[key] = {
                    'patient_id': xca.patient_id,
                    'video_id': xca.video_id,
                    'severity': xca.stenosis_severity
                }

        output = []
        for key, frames in videos.items():
            frames_sorted = sorted(frames, key=lambda x: x[0])
            frame_list = [img for (_, img, _) in frames_sorted]
            bbox_list  = [bbox for (_, _, bbox) in frames_sorted]

            # if all bboxes are non-none, convert them to a torch.Tensor of shape (T, 4).
            if all(bbox is not None for bbox in bbox_list):
                bboxes_tensor = torch.tensor(bbox_list, dtype=torch.float32)
            else:
                bboxes_tensor = None

            output.append((frame_list, bboxes_tensor, meta_info[key]))
        return output

class Augmenter(BaseEstimator, TransformerMixin):
    """
    Applies augmentation techniques to the XCA video sequences, another augmenter is passed on via augmentor parameter.
    Expects each video to be a list of torch tensors representing the frames, expects frames to be grayscale.

    The actual augmentation logic is provided via an external augmentor instance.
    """
    def __init__(self, is_train: bool = True, augmentor=None):
        self.is_train = is_train
        self.augmentor = augmentor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        augmented_videos = []
        for frame_list, bboxes, meta in X:
            video_tensor = torch.stack( # stack frame list into a (T, C, H, W) tensor
                [frame.unsqueeze(0) if frame.dim() == 2 else frame for frame in frame_list],
                dim=0  # for each frame, if it's 2D (H x W), unsqueeze to get (1, H, W).
            # ).to(DEVICE)
            )

            # the augmentor should return both the augmented video tensor and bounding boxes.
            if self.is_train and self.augmentor is not None:
                video_tensor, bboxes = self.augmentor.augment(video_tensor, bboxes)
            augmented_videos.append(XCAVideo(video_tensor, bboxes, meta))
        return augmented_videos



class PreprocessPipeline(BaseEstimator, TransformerMixin):
    """
    Construct end-to-end preprocessing pipeline for XCAImage objects.

    The pipeline:
        1. Load images from XCAImage objects (optionally convert to tensor)
        2. Normalize images for training data
        3. Aggregate individual frames into video sequences
        4. Apply augmentation

    Provides the train/val/test split
    """

    def __init__(self, as_tensor: bool = False, is_train: bool = True, augmentor=None, verbose: bool = True):
        self.as_tensor = as_tensor
        self.is_train = is_train
        self.augmentor = augmentor
        self.verbose = verbose

        self.frame_pipeline = Pipeline([
            ('loader', ImageLoader(as_tensor=self.as_tensor)),
            ('normalizer', Normalizer(is_train=self.is_train)),
        ])

    def fit(self, X, y=None):
        self.frame_pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        try:
            check_is_fitted(self.frame_pipeline)
        except Exception:
            self.fit(X, y)

        # step 1
        log(f"Loading XCAImages as NumPy arrays to the disk", verbose=self.verbose)
        loaded_images = self.frame_pipeline.transform(X)
        paired = list(zip(X, loaded_images))
        log(f"Normalizing the XCAImages", verbose=self.verbose)

        # step 2
        log(f"Aggregating frames into video sequences", verbose=self.verbose)
        aggregator = VideoAggregator()
        video_sequences = aggregator.transform(paired)

        # step 3
        log(f"Augmenting video sequences using: {self.augmentor.__class__.__name__}", verbose=self.verbose)
        augmenter = Augmenter(is_train=self.is_train, augmentor=self.augmentor)
        video_sequences = augmenter.transform(video_sequences)

        self.verbose = False  # only print for training, not val and test
        return video_sequences


    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


    def split_transform(self, X, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=SEED):
        """
        Split the data into training, validation and test sets based on patient_id.
        Then fit the pipeline on the training set
        """

        # check for cached data:
        cache_filename = (f"regular_split_{train_size}_{val_size}_{test_size}_{random_state}"
                          f"_{self.augmentor.__class__.__name__}.pkl")
        cached_result = cache_data(filename=cache_filename, data=None, verbose=self.verbose)
        if cached_result is not None:
            return cached_result

        if not np.isclose(train_size + val_size + test_size, 1.0):  # redundancy
            raise ValueError("Train, validation, and test sizes must sum to 1.")

        patient_ids = sorted({img.patient_id for img in X})  # get all unique patient ids

        # temp_ids = (val + test) groups
        train_ids, temp_ids = train_test_split(patient_ids, train_size=train_size, random_state=random_state)

        # get the relative size for validation within the temporary set
        rel_val = val_size / (val_size + test_size)
        val_ids, test_ids = train_test_split(temp_ids, train_size=rel_val, random_state=random_state)

        split_ids = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
        splits = {
            split: [img for img in X if img.patient_id in ids]
            for split, ids in split_ids.items()
        }

        # fit on training data and compute normalization
        self.fit(splits["train"])
        result =  {
            "train": self.transform(splits["train"]),
            "val": self.transform(splits["val"]),
            "test": self.transform(splits["test"])
        }
        cache_data(filename=cache_filename, data=result, verbose=True)
        return result


    def kfold_transform(self, X, n_splits=5, random_state=SEED, holdout_test_size: float = 0.2):
        """
        Performs K-Fold Cross Validation based on patient_id.

        First splits the entire data into training and a final unseen test set (using holdout_test_size),
        then applies k-fold CV on the training set.
        """
        return DeprecationWarning




if __name__ == '__main__':
    reader = Reader(dataset_dir=CADICA_DATASET_DIR)
    images = reader.xca_images

    class DummyAugment:
        def augment(self, video_tensor, bboxes):
            return video_tensor, bboxes

    dummy = DummyAugment()

    pipeline = PreprocessPipeline(as_tensor=True, is_train=True, augmentor=dummy, verbose=True)


    splits = pipeline.split_transform(images)
    log(f"Processed {len(splits['train'])} training, {len(splits['val'])} validation, and {len(splits['test'])} test video sequences")

    train_img = splits['train'][0]
    print(train_img)

