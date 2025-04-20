# dataset.py

# Python imports
import numpy as np
from typing import Optional, Union

# Torch imports
import torch
from torch.utils.data import Dataset
# import torchvision.transforms.v2 as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from util import to_numpy
from config import CADICA_DATASET_DIR
from methods.reader import Reader, XCAImage, XCAVideo


class XCADataset(Dataset):
    """
    Constructs Dataset objects necessary for the Trainer,
    accepts both XCAImage and XCAVideo, for single and sequence models
    """

    def __init__(
            self,
            data_list: list[Union['XCAImage', 'XCAVideo']],
            transform: Optional[A.Compose] = None,
            use_augmentation: bool = False,
            is_train: bool = False,
            normalize_params: dict[str, float] = None,
            repeat_channels: bool = False
    ):
        """
        Initialize
        :param data_list: list of either XCAImage or XCAVideo instances
        :param transform: base transformations to apply to all data
        :param use_augmentation: whether to use augmentation (only in train)
        :param is_train: whether this is the training dataset
        :param normalize_params: dict containing 'mean' and 'std' for normalization example: {'mean': 0.5, 'std': 0.5}
        :param repeat_channels: whether to concatenate channel dimensions 1 channel grayscale to 3 channel rgb
        """

        self.data_list = data_list
        self.transform = transform
        self.use_augmentation = use_augmentation and is_train
        self.repeat_channels = repeat_channels

        self.using_video_format = isinstance(data_list[0], XCAVideo) if data_list else True

        if normalize_params is None and is_train:
            self.normalize_params = self._calculate_normalization_params()
        else:
            self.normalize_params = normalize_params or {'mean': 0.5, 'std': 0.5}


        # define base transforms
        self.base_transform = A.Compose([
            A.Normalize(mean=self.normalize_params['mean'], std=self.normalize_params['std']),
            ToTensorV2()
        ])

        if self.use_augmentation:
            self.augment_transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.3, limit=5),
                A.GaussNoise(p=0.5)
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # [x_min, y_min, x_max, y_max]
                label_fields=['labels']
            ))
        else:
            self.augment_transform = None

        self.transform_applied = False

        if self.using_video_format:
            self._create_frame_index()



    def _create_frame_index(self):
        """Create a flat index mapping dataset indices to (video_id, patient_id) pairs"""
        self.frame_index = [
            (video_idx, frame_idx)
            for video_idx, video in enumerate(self.data_list)
            for frame_idx in range(video.frame_count)
        ]

    def _calculate_normalization_params(self, min_sample_size=1000):
        """Calculate mean and std for normalization (from the training data only)"""
        if len(self.data_list) == 0:
            return {'mean': 0.5, 'std': 0.5}

        # create a generator for frames based on format
        if self.using_video_format:
            frame_generator = (frame for video in self.data_list for frame in video.frames)
        else:
            frame_generator = (img_obj.get_image() for img_obj in self.data_list)

        all_pixels = [   # process each frame, converting it to float, normalizing, and calculating the mean
            (frame.astype(np.float32) / 255.0).mean()
            for frame in frame_generator
            if isinstance(frame, np.ndarray)
        ]

        mean_val = np.mean(all_pixels)
        std_val = np.std(all_pixels) if np.std(all_pixels) > 0 else 1.0
        return {'mean': float(mean_val), 'std': float(std_val)}


    def __len__(self):
        if self.using_video_format:
            return len(self.frame_index)
        else:
            return len(self.data_list)


    def __getitem__(self, idx):
        self.transform_applied = False

        # select data source
        if self.using_video_format:
            vid_idx, frm_idx = self.frame_index[idx]
            item = self.data_list[vid_idx]
            arr = item.frames[frm_idx]
            metadata = {
                'patient_id': item.patient_id,
                'video_id': item.video_id,
                'frame_idx': frm_idx,
                'total_frames': item.frame_count
            }
            has_bbox = getattr(item, 'has_lesion', False)
            raw_bbox = (item.bboxes or [None])[frm_idx] if has_bbox else None
        else:
            item = self.data_list[idx]
            arr = item.get_image()
            metadata = {
                'patient_id': item.patient_id,
                'video_id': item.video_id,
                'frame_nr': item.frame_nr
            }
            raw_bbox = getattr(item, 'bbox', None)

        # prepare bounding boxes
        bboxes, labels = [], []
        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
            x1, y1, x2, y2 = raw_bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bboxes.append([x1, y1, x2, y2])
            labels.append(1)

        # process numpy arrays
        if isinstance(arr, np.ndarray):
            arr = arr.astype(np.float32)
            # ensure HWC
            if arr.ndim == 2:
                arr = arr[:, :, None]
            elif arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = arr.transpose(1, 2, 0)
            # to uint8
            if arr.dtype != np.uint8:
                arr = (arr * 255 if arr.max() <= 1.0 else arr).astype(np.uint8)
            # augment
            if self.augment_transform and not self.transform_applied:
                out = self.augment_transform(image=arr, bboxes=bboxes, labels=labels)
                arr, bboxes, labels = out['image'], out['bboxes'], out['labels']
                self.transform_applied = True
            # base transform
            arr = self.base_transform(image=arr)['image']
            # repeat channels if needed
            if self.repeat_channels and arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.repeat(3, 1, 1)
            # build target
            if bboxes:
                target = {
                    'boxes': torch.tensor(bboxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64)
                }
            else:
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64)
                }
        else:
            # non-array or no transform
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            }

        return arr, target, metadata



if __name__ == '__main__':
    r = Reader(dataset_dir=CADICA_DATASET_DIR)
    videos = r.construct_videos()
    images = r.xca_images

    x = XCADataset(data_list=videos, repeat_channels=True)





