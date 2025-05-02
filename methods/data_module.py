# data_module.py

# Python imports
import random
from typing import Optional, Union
from collections import defaultdict

# Torch imports
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Local imports
from util import log
from methods.dataset import XCADataset
from methods.reader import XCAImage, XCAVideo
from config import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED, NUM_WORKERS, T_CLIP



class XCADataModule(pl.LightningDataModule):
    """Pytorch Lightning DataModule for XCA Dataset, handles slitting, preprocessing, and dataloader creation"""
    def __init__(
            self,
            data_list : list[Union['XCAImage', 'XCAVideo']],
            batch_size: int = 8,
            num_workers: int = NUM_WORKERS,
            train_val_test_split: tuple[float, float, float] = (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
            use_augmentation: bool = False,
            repeat_channels: bool = True,
            normalize_params: dict[str, float] = None,

            t_clip: int = T_CLIP,

            seed: int = SEED,
    ):
        """
        Initialize the data module
        :param data_list: list of XCAImage or XCAVideo instances
        :param batch_size: batch size for data loaders
        :param num_workers: number of workers for data loaders
        :param train_val_test_split: ratio of train/val/test split
        :param use_augmentation: whether to use augmentation
        :param repeat_channels: whether to use repeat channels 1→3
        :param normalize_params: normalization parameters in the format {mean: X, std: Y}
        :param t_clip: sequence length of a clip for video data
        :param seed: random seed
        """
        super().__init__()

        self.data_list = data_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.use_augmentation = use_augmentation
        self.repeat_channels = repeat_channels
        self.normalize_params = normalize_params  # to be extracted from train dataset, or user supplied
        self.seed = seed


        self.using_video_format = isinstance(data_list[0], XCAVideo) if data_list else True
        self.t_clip = t_clip if self.using_video_format else 1  # also done in dataset.py, but can never be to sure lol

        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None


    def _split_data(self):
        """
        Split the data on video level to prevent data leakage.
        """
        # 1. group by source id
        items_by_source = defaultdict(list)
        for item in self.data_list:
            source_id = (item.patient_id, item.video_id)
            items_by_source[source_id].append(item)

        # 2. shuffle source keys
        source_keys = list(items_by_source)
        rng = random.Random(self.seed)
        rng.shuffle(source_keys)

        # 3. get split indices
        total_sources = len(source_keys)
        train_ratio, val_ratio, _ = self.train_val_test_split
        train_cutoff = int(total_sources * train_ratio)
        val_cutoff = train_cutoff + int(total_sources * val_ratio)

        # 4. partition sources
        train_keys = source_keys[:train_cutoff]
        val_keys = source_keys[train_cutoff:val_cutoff]
        test_keys = source_keys[val_cutoff:]

        # 5. flatten for each split
        train_data = [item for key in train_keys for item in items_by_source[key]]
        val_data = [item for key in val_keys for item in items_by_source[key]]
        test_data = [item for key in test_keys for item in items_by_source[key]]

        return train_data, val_data, test_data


    def setup(self, stage: Optional[str] = None):
        """
        set up the datasets for train, val and test
        :param stage: current stage ('fit', 'validate', 'test')
        """
        if self.train_dataset is not None:
            return

        log(f"Creating DataLoaders for training, validation and test sets... ({TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE})")

        self.train_data, self.val_data, self.test_data = self._split_data()

        # temporarily create train dataset to extract norm. params
        if self.normalize_params is None:
            if self.train_data:
                temp_ds = XCADataset(
                    data_list=self.train_data,
                    use_augmentation=False,
                    is_train=True,
                    repeat_channels=self.repeat_channels,
                    t_clip=self.t_clip
                )
                self.normalize_params = temp_ds.normalize_params
            else:
                self.normalize_params = {'mean': 0.5, 'std': 0.5}

        # print(self.train_data)
        # 3. create actual datasets
        self.train_dataset = XCADataset(
            data_list=self.train_data,
            use_augmentation=self.use_augmentation,
            is_train=True,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels,
            t_clip=self.t_clip
        )

        self._validate_dataset_samples(self.train_dataset, 'Training Set')

        self.val_dataset = XCADataset(
            data_list=self.val_data,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels,
            t_clip=self.t_clip
        )

        self._validate_dataset_samples(self.val_dataset, 'Validation Set')

        self.test_dataset = XCADataset(
            data_list=self.test_data,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels,
            t_clip=self.t_clip
        )

    def _make_dataloader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        self.setup('fit')
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        self.setup('validate')
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        self.setup('test')
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def on_train_epoch_end(self):
        """Callback to clear aug. cache after a training epoch"""
        if hasattr(self.train_dataset, 'on_epoch_end') and self.use_augmentation:
            self.train_dataset.on_epoch_end()


    @staticmethod
    def _validate_dataset_samples(dataset, dataset_name, num_samples=5):
        """Validate a few samples to catch potential issues, handling image and video."""
        if len(dataset) == 0:
            log(f"Skipping validation for {dataset_name}: Dataset is empty.", level='warning')
            return

        log(f"Validating {num_samples} samples from {dataset_name}...")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        is_video = dataset.using_video_format

        expected_channels = 3 if dataset.repeat_channels else 1

        for i, idx in enumerate(indices):
            try:
                item, target, metadata = dataset[idx]
                # --- Tensor Checks ---
                if is_video:
                    # Video Clip: (T, C, H, W)
                    assert item.dim() == 4, f"Video item should have 4 dimensions, got {item.dim()}"
                    assert item.shape[0] == dataset.t_clip, f"Video item T dim {item.shape[0]} != t_clip {dataset.t_clip}"
                    assert item.shape[1] == expected_channels, f"Video item C dim {item.shape[1]} != expected {expected_channels}"
                    assert item.dtype == torch.float32, f"Item tensor should be float32, got {item.dtype}"
                    # Check target (list[dict])
                    assert isinstance(target, list), f"Video target should be a list, got {type(target)}"
                    assert len(target) == dataset.t_clip, f"Video target list length {len(target)} != t_clip {dataset.t_clip}"
                    # val each frame's target dict
                    for frame_idx, frame_target in enumerate(target):
                         XCADataModule._validate_target_dict(frame_target, f"Frame {frame_idx}")
                else:
                    # Single Image: (C, H, W)
                    assert item.dim() == 3, f"Image item should have 3 dimensions, got {item.dim()}"
                    assert item.shape[0] == expected_channels, f"Image item C dim {item.shape[0]} != expected {expected_channels}"
                    assert item.dtype == torch.float32, f"Item tensor should be float32, got {item.dtype}"
                    # Check target (dict)
                    assert isinstance(target, dict), f"Image target should be a dict, got {type(target)}"
                    XCADataModule._validate_target_dict(target, "Image")

            except Exception as e:
                log(f"Validation failed for sample index {idx} in {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                raise

        log(f"Dataset validation successful for {dataset_name}!")

    @staticmethod
    def _validate_target_dict(target, context):
        """Validates the structure and content of a single target dictionary."""
        assert isinstance(target, dict), f"{context} target is not a dict: {type(target)}"
        assert 'boxes' in target, f"{context} target missing 'boxes' key"
        assert 'labels' in target, f"{context} target missing 'labels' key"
        boxes = target['boxes']
        labels = target['labels']
        assert isinstance(boxes, torch.Tensor), f"{context} boxes are not a tensor: {type(boxes)}"
        assert isinstance(labels, torch.Tensor), f"{context} labels are not a tensor: {type(labels)}"

        num_boxes = boxes.shape[0]
        num_labels = labels.shape[0]
        assert num_boxes == num_labels, f"{context} found {num_boxes} boxes but {num_labels} labels"

        if num_boxes > 0:
             assert boxes.dim() == 2, f"{context} boxes tensor should be 2D, got {boxes.dim()} dimensions"
             assert boxes.shape[1] == 4, f"{context} boxes should have 4 columns (coords), got {boxes.shape[1]}"
             assert boxes.dtype == torch.float32, f"{context} boxes should be float32, got {boxes.dtype}"
             assert labels.dim() == 1, f"{context} labels tensor should be 1D, got {labels.dim()} dimensions"
             assert labels.dtype == torch.int64, f"{context} labels should be int64, got {labels.dtype}"
             assert torch.all(boxes[:, 0] <= boxes[:, 2]), f"{context} found boxes where x1 > x2"
             assert torch.all(boxes[:, 1] <= boxes[:, 3]), f"{context} found boxes where y1 > y2"
             assert torch.all(labels >= 0) and torch.all(labels <= 1), f"{context} found labels outside [0, 1]: {labels.unique()}"
        else:
             assert boxes.shape == (0, 4), f"{context} empty boxes tensor has wrong shape: {boxes.shape}"
             assert labels.shape == (0,), f"{context} empty labels tensor has wrong shape: {labels.shape}"


    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function to handle variable-sized bounding boxes.
        """
        batch = [entry for entry in batch if entry is not None]
        if not batch:
            return torch.empty((0, 1, 1, 1)), [], []

        tensors, targets, metadata = zip(*batch)
        batch_tensors = torch.stack(tensors, dim=0)
        return batch_tensors, list(targets), list(metadata)




