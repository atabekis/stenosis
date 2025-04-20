# module.py

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
from config import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED



class XCADataModule(pl.LightningDataModule):
    """Pytorch Lightning DataModule for XCA Dataset, handles slitting, preprocessing, and dataloader creation"""
    def __init__(
            self,
            data_list : list[Union['XCAImage', 'XCAVideo']],
            batch_size: int = 8,
            num_workers: int = 4,
            train_val_test_split: tuple[float, float, float] = (TRAIN_SIZE, VAL_SIZE, TEST_SIZE),
            use_augmentation: bool = False,
            repeat_channels: bool = True,
            normalize_params: dict[str, float] = None,
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

        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None


    def _split_data(self):
        """
        Split the data on video level to prevent data leakage.
        """
        video_dict = defaultdict(list)
        for item in self.data_list:
            video_dict[item.video_id].append(item)

        # list of all unique ids and shuffle
        video_keys = list(video_dict.keys())
        random.shuffle(video_keys)

        # calc split sizes
        total = len(video_keys)
        train_size = int(total * self.train_val_test_split[0])
        val_size = int(total * self.train_val_test_split[1])

        # split keys for each
        train_keys = video_keys[:train_size]
        val_keys = video_keys[train_size:train_size + val_size]
        test_keys = video_keys[train_size + val_size:]

        # create data lists for each split
        train_data = [item for key in train_keys for item in video_dict[key]]
        val_data = [item for key in val_keys for item in video_dict[key]]
        test_data = [item for key in test_keys for item in video_dict[key]]

        return train_data, val_data, test_data


    def setup(self, stage: Optional[str] = None):
        """
        set up the datasets for train, val and test
        :param stage: current stage ('fit', 'validate', 'test')
        """
        log(f"Creating DataLoaders for training, validation and test sets... ({TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE})")
        train_data, val_data, test_data = self._split_data()

        # temporarily create train dataset to extract norm. params
        if self.normalize_params is None:
            temp_train_dataset = XCADataset(
                train_data,
                transform=None,
                use_augmentation=False,
                is_train=True
            )
            self.normalize_params = temp_train_dataset.normalize_params

        # Now actually create the datasets
        self.train_dataset = XCADataset(
            train_data,
            transform=None,
            use_augmentation=self.use_augmentation,
            is_train=True,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels
        )

        self._validate_dataset_samples(self.train_dataset)

        self.val_dataset = XCADataset(
            val_data,
            transform=None,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels
        )

        self.test_dataset = XCADataset(
            test_data,
            transform=None,
            use_augmentation=False,
            is_train=False,
            normalize_params=self.normalize_params,
            repeat_channels=self.repeat_channels
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(self.seed)
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(self.seed)

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers= True if self.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(self.seed)
        )

    @staticmethod
    def _validate_dataset_samples(dataset, num_samples=5):
        """Validate a few samples to catch potential issues."""
        import random

        log("Validating dataset samples...")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        for idx in indices:
            img, target, _ = dataset[idx]

            # Check image dimensions
            assert img.dim() == 3, f"Image should have 3 dimensions, got {img.dim()}"
            assert img.shape[0] == 3, f"Image should have 3 channels, got {img.shape[0]}"

            # Check target format
            assert 'boxes' in target, "Target missing 'boxes' key"
            assert 'labels' in target, "Target missing 'labels' key"

            # Check labels are valid integers
            if len(target['labels']) > 0:
                assert target['labels'].dtype == torch.int64, f"Labels should be int64, got {target['labels'].dtype}"
                assert torch.all(target['labels'] >= 0), "Labels should be >= 0"
                assert torch.all(target['labels'] < 2), "Labels should be < num_classes (2)"

            # Check boxes format
            if len(target['boxes']) > 0:
                assert target['boxes'].shape[1] == 4, f"Boxes should have 4 coordinates, got {target['boxes'].shape[1]}"
                # Check boxes have proper format (x1 < x2, y1 < y2)
                assert torch.all(target['boxes'][:, 0] < target['boxes'][:, 2]), "Boxes should have x1 < x2"
                assert torch.all(target['boxes'][:, 1] < target['boxes'][:, 3]), "Boxes should have y1 < y2"

            # Check labels and boxes match in count
            assert len(target['boxes']) == len(target[
                                                   'labels']), f"Boxes and labels should have same length, got {len(target['boxes'])} and {len(target['labels'])}"

        log("Dataset validation successful!")


    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function to handle variable-sized bounding boxes.
        """
        if not batch:
            return [], [], []
        images, targets, metadata = zip(*batch)
        return list(images), list(targets), list(metadata)



if __name__ == "__main__":
    from methods.reader import Reader
    from config import CADICA_DATASET_DIR

    r = Reader(CADICA_DATASET_DIR)
    vids = r.construct_videos()
    module = XCADataModule(vids)