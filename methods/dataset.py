# dataset.py

# Python imports
import cv2
import random
import numpy as np
from typing import Optional, Union

# Torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from methods.reader import Reader, XCAImage, XCAVideo

from util import log
from config import CADICA_DATASET_DIR, T_CLIP, DEFAULT_HEIGHT, DEFAULT_WIDTH


class XCADataset(Dataset):
    """
    Constructs Dataset objects necessary for the Trainer,
    accepts both XCAImage and XCAVideo, for single and sequence models
    """

    def __init__(
            self,
            data_list: list[Union['XCAImage', 'XCAVideo']],
            use_augmentation: bool = False,
            is_train: bool = False,
            normalize_params: dict[str, float] = None,
            repeat_channels: bool = False,
            t_clip: int = T_CLIP
    ):
        """
        Initialize
        :param data_list: list of either XCAImage or XCAVideo instances
        :param use_augmentation: whether to use augmentation (only in train)
        :param is_train: whether this is the training dataset
        :param normalize_params: dict containing 'mean' and 'std' for normalization example: {'mean': 0.5, 'std': 0.5}
        :param repeat_channels: whether to concatenate channel dimensions 1 channel grayscale to 3 channel rgb
        :param t_clip: sequence length for video clips
        """

        self.data_list = data_list
        self.repeat_channels = repeat_channels
        self.use_augmentation = use_augmentation and is_train

        if not data_list:
            self.using_video_format = True
            log("Warning: Initializing XCADataset with an empty data_list.")
        else:
            self.using_video_format = isinstance(data_list[0], XCAVideo)

        self.t_clip = t_clip if self.using_video_format else 1

        if normalize_params is not None:
            self.normalize_params = normalize_params
        elif is_train:
            self.normalize_params = self._calculate_normalization_params()
        else:
            self.normalize_params = {'mean': 0.5, 'std': 0.5}

        self.base_transform = A.Compose([
            A.Normalize(mean=self.normalize_params['mean'], std=self.normalize_params['std']),
            ToTensorV2()
        ])

        if self.use_augmentation:
            bbox_params = A.BboxParams(
                format='pascal_voc',
                label_fields=['category_ids'],
                min_visibility=0.25,  # based on my observations, this should not happen (at least in CADICA)
                min_area=8
            )
            self.augment_transform = A.ReplayCompose([
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.5, p=0.75),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.ImageCompression(quality_range=(80, 99), p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                    A.MotionBlur(blur_limit=(3, 7), p=0.6),
                    A.MedianBlur(blur_limit=5, p=0.4),
                    A.Defocus(radius=(1, 3), alias_blur=0.3, p=0.3),
                ], p=0.7)
            ], bbox_params=bbox_params)
            self.video_aug_params = {}
        else:
            self.video_aug_params = None
            self.augment_transform = None

        self.epoch = 0

    def _calculate_normalization_params(self, min_sample_size=1000):
        default = {'mean': 0.5, 'std': 0.5}
        if not self.data_list:
            return default

        if self.using_video_format:
            pool = []
            for vid_idx, video in enumerate(self.data_list):
                if isinstance(video, XCAVideo):
                    for frm_idx in range(video.frame_count):
                        pool.append((vid_idx, frm_idx))
            label = 'frames'
            if not pool: return default
        else:
            pool = list(range(len(self.data_list)))
            label = 'images'

        if not pool:
            return default

        sample_size = min(min_sample_size, len(pool))
        sampled_indices = random.sample(pool, sample_size)

        pix_sum, pix_sq_sum, total_pix = 0.0, 0.0, 0

        for item_index in sampled_indices:
            try:
                if self.using_video_format:
                    vid_idx, frm_idx = item_index
                    video = self.data_list[vid_idx]
                    arr = (video.frames[frm_idx]
                           if isinstance(video, XCAVideo)
                              and hasattr(video, 'frames') and isinstance(video.frames, np.ndarray)
                              and 0 <= frm_idx < video.frames.shape[0]
                           else None)
                else:
                    img_obj = self.data_list[item_index]
                    arr = (img_obj.image
                           if isinstance(img_obj, XCAImage) and hasattr(img_obj, 'image')
                           else None)
            except Exception as e:
                log(f"Could not retrieve {label[:-1]} for item {item_index}: {e}")
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
        std = float(max(np.sqrt(max(var, 0.0)), 1e-5))

        return {'mean': float(mean), 'std': std}

    def _apply_augment(self, arr: np.ndarray, bboxes: list[list[float]], category_ids: list[int],
                       video_uid: Optional[tuple]):
        if not (self.use_augmentation and self.augment_transform):
            return arr, bboxes

        arr_uint8 = arr  # Assume arr is already HWC or compatible with below
        if arr.ndim == 2:
            arr_uint8 = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr_uint8 = arr.transpose(1, 2, 0)

        if arr_uint8.dtype != np.uint8:
            arr_uint8 = ((arr_uint8 * 255) if arr_uint8.max() <= 1.0 else arr_uint8).clip(0, 255).astype(np.uint8)

        if arr_uint8.ndim == 3 and arr_uint8.shape[2] != 1 and arr_uint8.shape[2] != 3:
            log(f"Warning: Unexpected channel count {arr_uint8.shape[2]} in _apply_augment for {video_uid}. Attempting to proceed.")
        elif arr_uint8.ndim == 2:
            arr_uint8 = arr_uint8[..., None]

        try:
            if video_uid is not None:
                self.video_aug_params = self.video_aug_params or {}
                replay_data = self.video_aug_params.get(video_uid)
                if replay_data is not None:
                    transformed = A.ReplayCompose.replay(replay_data, image=arr_uint8, bboxes=bboxes,
                                                         category_ids=category_ids)
                else:
                    transformed = self.augment_transform(image=arr_uint8, bboxes=bboxes, category_ids=category_ids)
                    self.video_aug_params[video_uid] = transformed.get('replay')
            else:
                transformed = self.augment_transform(image=arr_uint8, bboxes=bboxes, category_ids=category_ids)

            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            return aug_image, aug_bboxes
        except Exception as e:
            log(f"Augmentation error for {video_uid or 'image'}: {e}. Returning original array and bboxes.")
            return arr_uint8, bboxes

    def on_epoch_end(self):
        if self.video_aug_params is not None:
            self.video_aug_params.clear()
        self.epoch += 1

    def _process_frame(self, arr: np.ndarray, raw_bbox: Optional[np.ndarray], video_uid: Optional[tuple]):
        if not isinstance(arr, np.ndarray):
            log(f'Warning: Image array must be type np.ndarray, got {type(arr)}. Returning zero tensor and None bbox')
            c = 3 if self.repeat_channels else 1
            return torch.zeros((c, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=torch.float32), None

        processed_arr = arr.astype(np.float32)
        if processed_arr.ndim == 3 and processed_arr.shape[0] in (1, 3):
            processed_arr = processed_arr.transpose(1, 2, 0)
        elif processed_arr.ndim == 2:
            processed_arr = processed_arr[..., None]

        if processed_arr.ndim == 3 and processed_arr.shape[2] not in (1, 3):
            log(f"Warning: Unexpected shape {processed_arr.shape} for {video_uid}. Converting to grayscale.")
            code = cv2.COLOR_RGBA2GRAY if processed_arr.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            if processed_arr.dtype != np.uint8:
                processed_arr = (processed_arr.clip(0, 255) if processed_arr.max() > 1 else processed_arr * 255).astype(
                    np.uint8)
            processed_arr = cv2.cvtColor(processed_arr, code)[..., None]

        bboxes_for_aug = []
        category_ids_for_aug = []
        if raw_bbox is not None and isinstance(raw_bbox, np.ndarray) and raw_bbox.size == 4:
            x1, y1, x2, y2 = raw_bbox.astype(float)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            if x_max > x_min and y_max > y_min:
                img_h, img_w = processed_arr.shape[:2]
                x_min = np.clip(x_min, 0, img_w - 1)
                y_min = np.clip(y_min, 0, img_h - 1)
                x_max = np.clip(x_max, 0, img_w - 1)
                y_max = np.clip(y_max, 0, img_h - 1)
                if x_max > x_min and y_max > y_min:
                    bboxes_for_aug = [[x_min, y_min, x_max, y_max]]
                    category_ids_for_aug = [1]

        arr_aug, aug_bboxes_list = self._apply_augment(processed_arr, bboxes_for_aug, category_ids_for_aug, video_uid)
        final_bbox_for_target = aug_bboxes_list[0] if aug_bboxes_list else None

        try:
            tensor = self.base_transform(image=arr_aug)["image"]
        except Exception as e:
            log(f"Base transform failed for {video_uid or 'image'}: {e}. Returning zero tensor.")
            c = 3 if self.repeat_channels else 1
            h, w = arr_aug.shape[:2] if hasattr(arr_aug, 'shape') else (DEFAULT_HEIGHT, DEFAULT_WIDTH)
            return torch.zeros((c, h, w), dtype=torch.float32), None

        if self.repeat_channels and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif not self.repeat_channels and tensor.shape[0] == 3:
            log(f"Warning: repeat_channels is False but tensor has 3 channels for {video_uid or 'image'}. Taking first channel.")
            tensor = tensor[:1, :, :]

        expected_c = 3 if self.repeat_channels else 1
        if tensor.shape[0] != expected_c:
            log(f"Error: Expected {expected_c} channels but got {tensor.shape[0]} for {video_uid or 'image'}. Resetting tensor.")
            _, h, w = tensor.shape
            tensor = torch.zeros((expected_c, h, w), dtype=torch.float32)

        return tensor, final_bbox_for_target

    @staticmethod
    def _get_target_dict(processed_bbox: Optional[Union[list[float], np.ndarray]]):
        if processed_bbox is None:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        bbox_arr = np.asarray(processed_bbox, dtype=float).ravel()
        if bbox_arr.size != 4:
            log(f"Warning: _get_target_dict received invalid bbox_arr {bbox_arr}. Returning empty target.")
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        x_min, y_min, x_max, y_max = bbox_arr

        if x_max <= x_min or y_max <= y_min:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        box_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        label_tensor = torch.tensor([1], dtype=torch.int64)

        return {'boxes': box_tensor, 'labels': label_tensor}

    def _fetch_video_clip(self, video: 'XCAVideo'):
        if not isinstance(video, XCAVideo):
            raise TypeError(f"Expected XCAVideo, got {type(video)}")
        total_frames = video.frame_count
        if total_frames == 0:
            log(f"Warning: Video {video.patient_id}/{video.video_id} has no frames. Returning empty data.")
            c = 3 if self.repeat_channels else 1
            dummy_clip_tensor = torch.zeros((self.t_clip, c, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=torch.float32)
            dummy_targets = [self._get_target_dict(None) for _ in range(self.t_clip)]
            dummy_mask = torch.zeros((self.t_clip,), dtype=torch.bool)
            dummy_metadata = {
                "patient_id": video.patient_id, "video_id": video.video_id,
                "start_frame_orig": 0, "end_frame_orig": -1, "num_frames_in_clip": 0,
                "total_frames_in_video": 0, "is_full_video": True, "error": "No frames in video"
            }
            return dummy_clip_tensor, dummy_targets, dummy_mask, dummy_metadata

        uid = (video.patient_id, video.video_id)
        clip_len = min(total_frames, self.t_clip)
        start = 0
        if total_frames > self.t_clip:
            start = random.randint(0, total_frames - self.t_clip)
        elif total_frames < self.t_clip:
            clip_len = total_frames

        end = start + clip_len
        raw_frames_slice = video.frames[start:end]

        bboxes_slice = ([None] * clip_len)
        if video.bboxes is not None and len(video.bboxes) == total_frames:
            bboxes_slice = video.bboxes[start:end]
        elif video.bboxes is not None and len(video.bboxes) != total_frames:
            log(f"Warning: Mismatch in frame_count ({total_frames}) and bboxes length ({len(video.bboxes)}) for video {uid}. Using None for bboxes.")

        processed_frames_tensors = []
        processed_targets = []

        for i in range(clip_len):
            raw_frame = raw_frames_slice[i]
            raw_bbox_for_frame = bboxes_slice[i]

            if raw_bbox_for_frame is not None and not isinstance(raw_bbox_for_frame, np.ndarray):
                log(f"Warning: Bbox for frame {start + i} in video {uid} is not np.ndarray ({type(raw_bbox_for_frame)}). Treating as None.")
                raw_bbox_for_frame = None
            elif raw_bbox_for_frame is not None and raw_bbox_for_frame.size == 0:
                raw_bbox_for_frame = None

            frame_tensor, aug_bbox = self._process_frame(raw_frame, raw_bbox_for_frame, uid)
            processed_frames_tensors.append(frame_tensor)
            processed_targets.append(self._get_target_dict(aug_bbox))

        if not processed_frames_tensors:
            log(f"Error: No frames processed for video {uid} despite total_frames={total_frames}.")
            c = 3 if self.repeat_channels else 1
            dummy_clip_tensor = torch.zeros((self.t_clip, c, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=torch.float32)
            dummy_targets = [self._get_target_dict(None) for _ in range(self.t_clip)]
            dummy_mask = torch.zeros((self.t_clip,), dtype=torch.bool)
            dummy_metadata = {
                "patient_id": video.patient_id, "video_id": video.video_id,
                "start_frame_orig": start, "end_frame_orig": end - 1, "num_frames_in_clip": 0,
                "total_frames_in_video": total_frames, "is_full_video": clip_len < self.t_clip,
                "error": "No frames processed"
            }
            return dummy_clip_tensor, dummy_targets, dummy_mask, dummy_metadata

        clip_tensor = torch.stack(processed_frames_tensors, dim=0)

        pad_count = self.t_clip - clip_len
        if pad_count > 0:
            clip_tensor = F.pad(clip_tensor, (0, 0, 0, 0, 0, 0, 0, pad_count))
            empty_target = self._get_target_dict(None)
            processed_targets.extend([empty_target] * pad_count)

        mask = torch.arange(self.t_clip, device=clip_tensor.device) < clip_len

        metadata = {
            "patient_id": video.patient_id,
            "video_id": video.video_id,
            "start_frame_orig": start,
            "end_frame_orig": end - 1,
            "num_frames_in_clip": clip_len,
            "total_frames_in_video": total_frames,
            "is_full_video": clip_len == total_frames and pad_count > 0,
        }
        return clip_tensor, processed_targets, mask, metadata


    def _fetch_image(self, img: 'XCAImage'):

        raw_bbox_data = getattr(img, "bbox", None)

        if raw_bbox_data is not None and not isinstance(raw_bbox_data, np.ndarray):
            try:
                raw_bbox_data = np.array(raw_bbox_data, dtype=np.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to convert bbox data to np.ndarray {e}")

        elif raw_bbox_data is not None and raw_bbox_data.size == 0:
            raw_bbox_data = None

        frame_tensor, aug_bbox = self._process_frame(img.image, raw_bbox_data, video_uid=None)
        target = self._get_target_dict(aug_bbox)

        mask = torch.ones((1,), dtype=torch.bool)

        metadata = {
            'patient_id': img.patient_id,
            'video_id': img.video_id,
            'frame_nr': img.frame_nr
        }
        return frame_tensor.unsqueeze(0), [target], mask, metadata


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if self.using_video_format:
            return self._fetch_video_clip(item)
        else:
            return self._fetch_image(item)


