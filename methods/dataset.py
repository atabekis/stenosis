# dataset.py

# Python imports
import numpy as np
from typing import Optional, Union

# Torch imports
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

# Local imports
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
            transform: Optional[T.Compose] = None,
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

        # make into tensor and normalize
        transform_list = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[self.normalize_params['mean']], std=[self.normalize_params['std']])
        ]

        if self.use_augmentation:
            augment_transforms = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.GaussianNoise()
            ]
            transform_list += augment_transforms

        self.transform = T.Compose(transform_list)

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
        """
        We will split this into two (1) if using video format and (2) is image format
        """
        if self.using_video_format:
            video_idx, frame_idx = self.frame_index[idx]
            video = self.data_list[video_idx]

            frame = video.frames[frame_idx]
            if isinstance(frame, np.ndarray):
                frame = frame.astype(np.float32)
                if len(frame.shape) == 2:  # add channel dimension if grayscale
                    frame = frame[np.newaxis, :, :]

            if self.transform:
                if isinstance(frame, np.ndarray) and frame.shape[0] == 1:
                    frame = frame[0]  # have to remove channel dimension for transforms
                frame = self.transform(frame)

            target = {}
            if video.has_lesion and video.bboxes is not None and frame_idx < len(video.bboxes):
                bbox = video.bboxes[frame_idx]
                if bbox is not None and len(bbox) > 0:
                    # Convert bbox to [x1, y1, x2, y2] format and torch tensor
                    boxes = torch.tensor([bbox], dtype=torch.float32)

                    # Ensure boxes are valid (non-empty and properly formed)
                    if boxes.shape[0] > 0 and boxes.shape[1] == 4:
                        # Make sure boxes are in correct format [x1, y1, x2, y2] with x1 < x2 and y1 < y2
                        boxes[:, 2] = torch.max(boxes[:, 0] + 1, boxes[:, 2])
                        boxes[:, 3] = torch.max(boxes[:, 1] + 1, boxes[:, 3])

                        target['boxes'] = boxes
                        # Important: Labels must be integers starting from 1 (0 is background)
                        target['labels'] = torch.ones(boxes.shape[0], dtype=torch.int64)
                    else:
                        # Empty or invalid boxes
                        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                        target['labels'] = torch.zeros(0, dtype=torch.int64)
                else:
                    # No bbox for this frame
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros(0, dtype=torch.int64)
            else:
                # No lesion/bbox in this video
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros(0, dtype=torch.int64)

            metadata = {
                'patient_id': video.patient_id,
                'video_id': video.video_id,
                'frame_idx': frame_idx,
                'total_frames': video.frame_count
            }

            if self.repeat_channels and frame.shape[0] == 1:
                frame = frame.repeat(3, 1, 1)

            return frame, target, metadata

        else: # using XCAImages
            image_obj = self.data_list[idx]
            image = image_obj.get_image()

            if self.transform:
                image = self.transform(image)

            target = {}
            if image_obj.bbox:
                target['boxes'] = torch.tensor([image_obj.bbox], dtype=torch.float32)
                target['labels'] = torch.tensor([1], dtype=torch.int64)  # stenosis
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros(0, dtype=torch.int64) # no stenosis

            metadata = {
                'patient_id': image_obj.patient_id,
                'video_id': image_obj.video_id,
                'frame_nr': image_obj.frame_nr
            }

            if self.repeat_channels and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)

            return image, target, metadata



if __name__ == '__main__':
    r = Reader(dataset_dir=CADICA_DATASET_DIR)
    videos = r.construct_videos()
    images = r.xca_images

    x = XCADataset(data_list=videos, repeat_channels=True)





