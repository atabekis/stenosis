# dataset.py

# Python imports
import random
import numpy as np
from typing import Optional, Union

# Torch imports
import torch
from torch.utils.data import Dataset
# import torchvision.transforms.v2 as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from util import log
from config import CADICA_DATASET_DIR, T_CLIP
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
            repeat_channels: bool = False,
            t_clip: int = T_CLIP
    ):
        """
        Initialize
        :param data_list: list of either XCAImage or XCAVideo instances
        :param transform: base transformations to apply to all data
        :param use_augmentation: whether to use augmentation (only in train)
        :param is_train: whether this is the training dataset
        :param normalize_params: dict containing 'mean' and 'std' for normalization example: {'mean': 0.5, 'std': 0.5}
        :param repeat_channels: whether to concatenate channel dimensions 1 channel grayscale to 3 channel rgb
        :param t_clip: sequence length for video clips
        """

        self.data_list = data_list
        self.transform = transform
        self.use_augmentation = use_augmentation and is_train
        self.repeat_channels = repeat_channels

        self.using_video_format = isinstance(data_list[0], XCAVideo) if data_list else True
        self.t_clip = t_clip if self.using_video_format else 1  # t_clip is 1 for images naturally

        if normalize_params is None and is_train:
            self.normalize_params = self._calculate_normalization_params()
        elif normalize_params is None and not is_train:
            self.normalize_params = {'mean': 0.5, 'std': 0.5}
        else:
            self.normalize_params = normalize_params

        # define base transforms
        self.base_transform = A.Compose([
            A.Normalize(mean=self.normalize_params['mean'], std=self.normalize_params['std']),
            ToTensorV2()
        ])

        if self.use_augmentation:
            self.augment_transform = A.ReplayCompose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                A.OneOf([ # will apply one of these augmentations with p=0.5 chance
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    A.Defocus(radius=(1, 3), alias_blur=0.3, p=0.3)
                ], p=0.5)
            ])
            self.video_aug_params = {}
        else:
            self.video_aug_params = None
            self.augment_transform = None

        self.transform_applied = False


        if self.using_video_format:
            original_count = len(self.data_list)
            self.data_list = [
                v for v in self.data_list
                if isinstance(v, XCAVideo) and v.frame_count >= self.t_clip
            ]
            filtered_count = len(self.data_list)
            if original_count != filtered_count:
                log(f"Filtered out {original_count - filtered_count} videos shorter than t_clip={self.t_clip}. Remaining videos: {filtered_count}")
            if not self.data_list and original_count > 0:
                raise ValueError(f"No videos remaining after filtering for t_clip={self.t_clip}. Check video lengths and t_clip setting.")

        self.epoch = 0   # used to reset the augmentation parameters per epoch



    def _calculate_normalization_params(self, min_sample_size=1000):
        """Calculate mean and std for normalization (from the training data only)"""

        default = {'mean': 0.5, 'std': 0.5}
        if not self.data_list:
            return default

        if self.using_video_format:
            frame_idx_list = getattr(self, 'frame_index', None)
            if not frame_idx_list:
                return default
            pool = frame_idx_list
            label = 'frames'
        else:
            pool = list(range(len(self.data_list)))
            label = 'images'

        sample_size = min(min_sample_size, len(pool))
        sampled = random.sample(pool, sample_size)

        pix_sum, pix_sq_sum, total_pix = 0.0, 0.0, 0

        for item in sampled:
            try:
                if self.using_video_format:
                    vid_idx, frm_idx = item
                    video = self.data_list[vid_idx]
                    arr = (video.frames[frm_idx]
                           if isinstance(video, XCAVideo)
                           and isinstance(video.frames, np.ndarray)
                           and 0 <= frm_idx < video.frame_count
                           else None)
                else:
                    img_obj = self.data_list[item]
                    arr = (img_obj.image
                           if isinstance(img_obj, XCAImage)
                           else None)
            except Exception as e:
                log(f"Could not retrieve {label[:-1]} {item}: {e}")
                continue

            if not isinstance(arr, np.ndarray):
                continue

            arr = arr.astype(np.float32) / 255.0
            pix_sum += arr.sum()
            pix_sq_sum += (arr ** 2).sum()
            total_pix += arr.size

            if total_pix == 0:
                return default

            mean = pix_sum / total_pix
            var = pix_sq_sum / total_pix - mean * mean
            std = float(max(np.sqrt(max(var, 0.0)), 1e-3))

            return {'mean': float(mean), 'std': std}


    def _apply_augment(self, arr, video_uid):
        """apply or replay augmentations, caching per video_uid"""
        if not self.use_augmentation or self.augment_transform is None:
            return arr

        try:
            if video_uid is not None:
                if video_uid in self.video_aug_params:
                    replay = self.video_aug_params[video_uid]
                    return A.ReplayCompose.replay(replay, image=arr)['image']

                out = self.augment_transform(image=arr)
                self.video_aug_params[video_uid] = out['replay']
                return out['image']

            else:  # image case
                return self.augment_transform(image=arr)['image']

        except Exception as e:
            log(f"Augmentation error for {video_uid or 'image'}: {e}")
            return arr


    def _process_frame(self, arr, video_uid):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'Non numpy array for {video_uid}')

        arr = arr.astype(np.float32)
        if arr.ndim == 2:  # grayscale (H, W) -> (H, W, 1)
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] in {1, 3}:  # CHW -> HWC
            arr = arr.transpose(1, 2, 0)

        max_val = arr.max()
        if 1.0 >= max_val > 0:
            arr = (arr * 255.0)
        arr = arr.clip(0, 255).astype(np.uint8)  #(H, W, C), dtype=uint8


        # apply Augmentations
        arr = self._apply_augment(arr, video_uid)

        # apply base transform
        try:
            # base_transform expects HWC, handles conversion to CHW Tensor
            processed_tensor = self.base_transform(image=arr)['image'] # (C, H, W), dtype=torch.float32

        except Exception as e:
            log(f'Base transform failed: {e}. Returning zero tensor.', level='error')
            h, w = arr.shape[:2]
            c = 3 if self.repeat_channels else arr.shape[2]  # use original channel
            processed_tensor = torch.zeros((c, h, w), dtype=torch.float32)

        # 5. repeat channels
        if self.repeat_channels and processed_tensor.shape[0] == 1:
            processed_tensor = processed_tensor.repeat(3, 1, 1)  # (3, H, W)

        return processed_tensor


    @staticmethod
    def _get_target_dict(raw_bbox):
        """Processes raw bounding box into a target dictionary."""
        bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
             try:
                  coords = [float(c) for c in raw_bbox]
                  x_coords = sorted(coords[0::2]) # [x_min, x_max]
                  y_coords = sorted(coords[1::2]) # [y_min, y_max]
                  x1, x2 = x_coords
                  y1, y2 = y_coords

                  if x2 > x1 and y2 > y1:
                       bboxes_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
                       labels_tensor = torch.tensor([1], dtype=torch.int64) # Assuming class 1 for lesion
             except (ValueError, TypeError) as e:
                  log(f"Error processing bbox {raw_bbox}: {e}", level='warning')

        return {'boxes': bboxes_tensor, 'labels': labels_tensor}


    def on_epoch_end(self):
        if self.video_aug_params is not None:
            self.video_aug_params.clear()
        self.epoch += 1


    def _fetch_random_clip(self, video):
        # 1. Type & length checks
        if not isinstance(video, XCAVideo):
            raise TypeError(f"Expected XCAVideo, got {type(video)}")
        frame_count = video.frame_count
        if frame_count < self.t_clip:
            raise ValueError(
                f"Video {video.patient_id}/{video.video_id} has only "
                f"{frame_count} frames, less than t_clip={self.t_clip}"
            )

        # 2. Sample start & end frames
        start = random.randint(0, frame_count - self.t_clip)
        end = start + self.t_clip

        # 3. Slice raw frames and verify length
        raw_frames = video.frames[start:end]
        if len(raw_frames) != self.t_clip:
            raise ValueError(
                f"Sliced {len(raw_frames)} frames, expected {self.t_clip} "
                f"for video {video.patient_id}/{video.video_id}"
            )

        # 4. Collect & pad bounding-boxes
        has_bbox = getattr(video, 'has_lesion', False) and getattr(video, 'bboxes', None) is not None
        if has_bbox:
            clip_bboxes = list(video.bboxes[start:end])
            if len(clip_bboxes) < self.t_clip:
                clip_bboxes += [None] * (self.t_clip - len(clip_bboxes))
        else:
            clip_bboxes = [None] * self.t_clip

        # 5. Process frames & build targets
        uid = (video.patient_id, video.video_id)
        processed_frames = [
            self._process_frame(frame, video_uid=uid)
            for frame in raw_frames
        ]
        targets = [
            self._get_target_dict(bbox)
            for bbox in clip_bboxes
        ]

        # 6. Stack into (T, C, H, W) and assemble metadata
        clip_tensor = torch.stack(processed_frames, dim=0)
        metadata = {
            'patient_id': video.patient_id,
            'video_id': video.video_id,
            'start_frame': start,
            'end_frame': end - 1,
            'total_frames_in_video': frame_count,
        }

        return clip_tensor, targets, metadata


    def _fetch_image(self, img):
        arr = img.image

        tensor = self._process_frame(arr, video_uid=None)
        target = self._get_target_dict(getattr(img, 'bbox', None))

        metadata = {
            'patient_id': img.patient_id,
            'video_id': img.video_id,
            'frane_nr': img.frame_nr
        }

        return tensor, target, metadata


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        if not (0 <= idx < len(self.data_list)):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self.data_list)}')

        item = self.data_list[idx]
        if self.using_video_format:
            return self._fetch_random_clip(item)
        else:
            return self._fetch_image(item)




if __name__ == '__main__':
    try:
        # welcome to ata's debugging nightmare
        r = Reader(dataset_dir=CADICA_DATASET_DIR)

        T_CLIP_TEST = 8

        log("\n--- Testing with Video Data ---")
        videos = r.construct_videos()
        if videos:
            log(f"Constructed {len(videos)} videos initially.")
            log(f"Initializing Training Video Dataset (Augmentation ON, t_clip={T_CLIP_TEST})...")
            train_video_dataset = XCADataset(
                data_list=list(videos),
                use_augmentation=True,
                is_train=True,
                repeat_channels=True,
                t_clip=T_CLIP_TEST
            )
            log(f"Train Video Dataset length (number of videos >= t_clip): {len(train_video_dataset)}")

            if len(train_video_dataset) > 0:
                 log("Getting first item (clip) from train video dataset...")
                 clip1, targets1, meta1 = train_video_dataset[0]
                 log(f"Item 0 - Meta: {meta1}, Clip shape: {clip1.shape}, Targets length: {len(targets1)}")
                 if targets1:
                      log(f"Target for first frame of clip 1: {targets1[0]}")

                 log("Getting first item again (should have same aug)...")
                 clip1_again, _, _ = train_video_dataset[0]
                 is_identical = torch.equal(clip1, clip1_again)
                 log(f"Is the clip identical when fetched again within same epoch? {is_identical}")


                 log("Simulating epoch end...")
                 train_video_dataset.on_epoch_end()
                 log("Getting first item again after epoch end (should have different aug)...")
                 clip1_ep2, _, meta1_ep2 = train_video_dataset[0]
                 is_identical_ep2 = torch.equal(clip1, clip1_ep2)
                 log(f"Is the clip identical after epoch end? {is_identical_ep2}")
                 log(f"Meta ep2: {meta1_ep2}")


            log("\nInitializing Validation Video Dataset (Augmentation OFF)...")
            val_video_dataset = XCADataset(
                data_list=list(videos),
                use_augmentation=False,
                is_train=False,
                normalize_params=train_video_dataset.normalize_params if train_video_dataset and hasattr(train_video_dataset, 'normalize_params') else None, # Reuse params
                repeat_channels=True,
                t_clip=T_CLIP_TEST
            )
            log(f"Validation Video Dataset length: {len(val_video_dataset)}")
            if len(val_video_dataset) > 0:
                log("Getting first item from validation video dataset...")
                clip_val, targets_val, meta_val = val_video_dataset[0]
                log(f"Item 0 - Meta: {meta_val}, Clip shape: {clip_val.shape}, Targets length: {len(targets_val)}")
                if targets_val:
                      log(f"Target for first frame of validation clip: {targets_val[0]}")
        else:
             log("No videos found by Reader.", level='warning')


        log("\n--- Testing with Image Data ---")
        images = r.xca_images
        if images:
             log(f"Found {len(images)} images.")
             log("Initializing Training Image Dataset (Augmentation ON)...")
             train_image_dataset = XCADataset(
                 data_list=list(images),
                 use_augmentation=True,
                 is_train=True,
                 repeat_channels=True,
                 t_clip=T_CLIP_TEST
             )
             log(f"Train Image Dataset length: {len(train_image_dataset)}")
             if len(train_image_dataset) > 0:
                  log("Getting first two items from train image dataset...")
                  img_i1, target_i1, meta_i1 = train_image_dataset[0]
                  log(f"Item 0 - Meta: {meta_i1}, Img shape: {img_i1.shape}, Target: {target_i1}")
                  if len(train_image_dataset) > 1:
                       img_i2, target_i2, meta_i2 = train_image_dataset[1]
                       log(f"Item 1 - Meta: {meta_i2}, Img shape: {img_i2.shape}, Target: {target_i2}")


             log("\nInitializing Validation Image Dataset (Augmentation OFF)...")
             val_image_dataset = XCADataset(
                 data_list=list(images),
                 use_augmentation=False,
                 is_train=False,
                 normalize_params=train_image_dataset.normalize_params if train_image_dataset and hasattr(train_image_dataset, 'normalize_params') else None, # Reuse params
                 repeat_channels=True,
                 t_clip=T_CLIP_TEST
             )
             log(f"Validation Image Dataset length: {len(val_image_dataset)}")
             if len(val_image_dataset) > 0:
                log("Getting first item from validation image dataset...")
                img_val_i, target_val_i, meta_val_i = val_image_dataset[0]
                log(f"Item 0 - Meta: {meta_val_i}, Img shape: {img_val_i.shape}, Target: {target_val_i}")

        else:
             log("No single images found by Reader.", level='warning')


    except ValueError as e:
         log(f"Value Error during dataset init or processing: {e}", level='error')
    except Exception as e:
        log(f"An unexpected error occurred in __main__: {e}", level='error')
        import traceback
        traceback.print_exc()