# data_module.py

# Python imports
import random
import numpy as np
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
            jitter: bool = False,

            seed: int = SEED,

            verbose: bool = True
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

        self.jitter = jitter

        self.verbose = verbose

        self.using_video_format = isinstance(data_list[0], XCAVideo) if data_list else True
        self.t_clip = t_clip if self.using_video_format else 1  # also done in dataset.py, but can never be too sure lol

        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

        assert np.isclose(sum(train_val_test_split), 1), 'Train, Validation and Test sizes must sum to 1'

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

        log(f"Creating DataLoaders for training, validation and test sets... ({TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE})", verbose=self.verbose)

        self.train_data, self.val_data, self.test_data = self._split_data()

        # small logging
        add = 'videos' if self.using_video_format else 'images'
        is_video = self.using_video_format
        count = lambda dl: (
            sum(
                getattr(item, 'has_lesion', False)
                if is_video
                else getattr(item, 'bbox', None) is not None
                for item in dl
            ),
            len(dl)
        )
        if self.verbose:
            splits = {'Train Set': count(self.train_data), 'Validation Set': count(self.val_data), 'Test Set': count(self.test_data)}
            for name, (pos, total) in splits.items():
                neg = total - pos
                log(f"   {name:15}: {total} {add:2} ({pos} positive / {neg} negative)")

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
            t_clip=self.t_clip,
            jitter=self.jitter
        )

        self._validate_dataset_samples(self.train_dataset, 'Training Set', verbose=self.verbose)

        self.val_dataset = XCADataset(
            data_list=self.val_data,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels,
            t_clip=self.t_clip,
            jitter=False,
        )

        self._validate_dataset_samples(self.val_dataset, 'Validation Set', verbose=self.verbose)

        self.test_dataset = XCADataset(
            data_list=self.test_data,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels,
            t_clip=self.t_clip,
            jitter=False,
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
            drop_last=False
            # generator=torch.Generator().manual_seed(self.seed)
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
    def _validate_dataset_samples(dataset, name, num_samples=3, verbose=True):
        """
        Validate a few samples to catch potential dataset issues.
        """
        length = len(dataset) if dataset else 0
        if length == 0:
            log(f"Skipping validation for {name}: empty dataset.", verbose=verbose)
            return

        samples = min(num_samples, length)
        log(f"Validating {samples} samples from {name}...", verbose=verbose)
        indices = random.sample(range(length), samples)

        is_video = getattr(dataset, "using_video_format", False)
        expected_t = dataset.t_clip if is_video else 1
        expected_c = 3 if dataset.repeat_channels else 1

        for seq_idx, idx in enumerate(indices, start=1):
            try:
                item, targets, mask, meta = dataset[idx]

                # --- item tensor checks ---
                assert isinstance(item, torch.Tensor), f"Item not a Tensor: {type(item)}"
                assert item.ndim == 4, f"Expected 4D (T,C,H,W), got {item.ndim}D"
                assert item.shape[0] == expected_t, f"T dim {item.shape[0]} != {expected_t}"
                assert item.shape[1] == expected_c, f"C dim {item.shape[1]} != {expected_c}"
                assert item.dtype == torch.float32, f"Expected float32, got {item.dtype}"

                # --- mask checks ---
                assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                assert mask.ndim == 1 and mask.shape[0] == expected_t
                if is_video and "num_frames_in_clip" in meta: assert mask.sum().item() == meta["num_frames_in_clip"]

                # --- targets checks ---
                assert isinstance(targets, list) and len(targets) == expected_t
                for frame_i, (m, tgt) in enumerate(zip(mask.tolist(), targets)): # padded frames must be empty
                    if m: XCADataModule._validate_target_dict(tgt, f"Frame {frame_i}")
                    else: assert not tgt["boxes"].numel() and not tgt["labels"].numel(), f"Padded target not empty at frame {frame_i}"

                # --- meta checks ---
                assert isinstance(meta, dict), f"Metadata not a dict: {type(meta)}"
                assert "patient_id" in meta and "video_id" in meta

            except AssertionError as ae:
                log(f"Validation failed at index {idx}: {ae}")
                raise
            except Exception as e:
                log(f"Error validating index {idx}: {e}")
                raise

        log(f"Successfully validated {samples} samples from {name}.", verbose=verbose)

    @staticmethod
    def _validate_target_dict(target, context):
        """
        Validate a single target dict.
        """
        # ---- structure -----
        assert isinstance(target, dict), f"{context} target must be dict, got {type(target)}"
        assert 'boxes' in target and 'labels' in target, f"{context} missing keys"
        boxes, labels = target['boxes'], target['labels']
        assert isinstance(boxes, torch.Tensor), f"{context} boxes not Tensor: {type(boxes)}"
        assert isinstance(labels, torch.Tensor), f"{context} labels not Tensor: {type(labels)}"

        # ------ count consistency --------
        num_boxes = boxes.shape[0]
        assert labels.shape[0] == num_boxes, f"{context} mismatch: {num_boxes} boxes vs {labels.shape[0]} labels"

        # ------ empty case -------
        if num_boxes == 0:
            assert boxes.shape == (0, 4), f"{context} empty boxes shape: {boxes.shape}"
            assert labels.shape == (0,), f"{context} empty labels shape: {labels.shape}"
            return

        # -------- non‐empty: check dims & dtypes -------
        assert boxes.ndim == 2 and boxes.shape[1] == 4, f"{context} boxes must be [N,4], got {tuple(boxes.shape)}"
        assert boxes.dtype == torch.float32, f"{context} boxes dtype must be float32"
        assert labels.ndim == 1 and labels.dtype == torch.int64, f"{context} labels must be 1D int64, got {labels.shape}, {labels.dtype}"



    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function to handle variable-sized bounding boxes.
        """
        # drop any failed items
        batch = [b for b in batch if b is not None]
        if not batch:
            log("Warning: received empty batch in collate_fn")
            return (
                torch.empty((0, 1, 1, 1, 1), dtype=torch.float32), [],
                torch.empty((0, 1), dtype=torch.bool), []
            )

        # unzip into components
        tensors, targets, masks, metadata = zip(*batch)

        # stack into batched tensors
        batch_tensors = torch.stack(tensors, dim=0)  # [B, T, C, H, W]
        batch_masks = torch.stack(masks, dim=0)  # [B, T]

        return batch_tensors, list(targets), batch_masks, list(metadata)