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

        self.using_video_format = isinstance(data_list[0], XCAVideo) if data_list else True
        self.t_clip = t_clip if self.using_video_format else 1  # t_clip is 1 for images naturally

        if normalize_params is not None:
            self.normalize_params = normalize_params
        elif is_train:
            self.normalize_params = self._calculate_normalization_params()
        else:
            self.normalize_params = {'mean': 0.5, 'std': 0.5}

        # define base transforms
        self.base_transform = A.Compose([
            A.Normalize(mean=self.normalize_params['mean'], std=self.normalize_params['std']),
            ToTensorV2()
        ])

        if self.use_augmentation:
            self.augment_transform = A.ReplayCompose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.5, p=0.75),
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
            std = float(max(np.sqrt(max(var, 0.0)), 1e-5))

            return {'mean': float(mean), 'std': std}


    def _apply_augment(self, arr, video_uid):
        """apply or replay augmentations, caching per video_uid"""
        if not (self.use_augmentation and self.augment_transform):
            return arr

        # -- ensure HWC uint8 for albumentations --
        arr = np.asarray(arr)
        if arr.ndim == 2:  # grayscale H×W → H×W×1
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW → HWC
            arr = arr.transpose(1, 2, 0)

        if arr.dtype != np.uint8:
            # if in [0,1], scale up; otherwise just clip & cast
            arr = ((arr * 255) if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)

        try:
            # -- video frames: replay or apply & cache --
            if video_uid is not None:
                # lazy-init cache dict
                self.video_aug_params = self.video_aug_params or {}
                replay = self.video_aug_params.get(video_uid)

                if replay is not None:
                    return A.ReplayCompose.replay(replay, image=arr)['image']

                result = self.augment_transform(image=arr)
                self.video_aug_params[video_uid] = result.get('replay')
                return result['image']

            # -- single image: always fresh --
            return self.augment_transform(image=arr)['image']

        except Exception as e:
            log(f"Augmentation error for {video_uid or 'image'}: {e}. Returning original array.")
            # if we transposed CHW→HWC, transpose back so callers see expected shape
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                return arr.transpose(1, 2, 0)
            return arr


    def on_epoch_end(self):
        if self.video_aug_params is not None:
            self.video_aug_params.clear()
        self.epoch += 1


    def _process_frame(self, arr, video_uid):
        """Processes a single frame (np.ndarray) → tensor [C, H, W]"""
        if not isinstance(arr, np.ndarray):
            log(f'Warning: Image array must be type np.ndarray, got {type(arr)}. Returning zero tensor')
            c = 3 if self.repeat_channels else 1
            return torch.zeros((c, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=torch.float32)

        # 1. convert to float32 HWC
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW → HWC
            arr = arr.transpose(1, 2, 0)

        if arr.ndim == 2:  # H×W → H×W×1
            arr = arr[..., None]

        if arr.ndim == 3 and arr.shape[2] not in (1, 3): # unexpected channel count → grayscale
            log(f"Warning: Unexpected shape {arr.shape} for {video_uid}. Converting to grayscale.")
            code = cv2.COLOR_RGBA2GRAY if arr.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            arr = cv2.cvtColor(arr, code)[..., None]

        # 2. apply augmentation
        arr_aug = self._apply_augment(arr, video_uid)

        # 3. base transform → tensor [c, h, w]
        try:
            tensor = self.base_transform(image=arr_aug)["image"]
        except Exception as e:
            log(f"Base transform failed for {video_uid}: {e}. Returning zero tensor.")
            c = 3 if self.repeat_channels else 1
            h, w = arr_aug.shape[:2]
            return torch.zeros((c, h, w), dtype=torch.float32)

        # 4. apply the self.repeat_channels flag
        if self.repeat_channels and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif not self.repeat_channels and tensor.shape[0] == 3:
            tensor = tensor[:1, :, :]

        # 5. final sanity check (remove after testing both cadica and danilov)
        expected_c = 3 if self.repeat_channels else 1
        if tensor.shape[0] != expected_c:
            log(f"Error: Expected {expected_c} channels but got {tensor.shape[0]}. Resetting tensor.")
            _, h, w = tensor.shape
            tensor = torch.zeros((expected_c, h, w), dtype=torch.float32)

        return tensor



    @staticmethod
    def _get_target_dict(raw_bbox):
        """Processes raw bounding box into a target dictionary."""
        if raw_bbox is None:  # if empty, negative image, return zeros and label 0
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        bbox_arr = np.asarray(raw_bbox, dtype=float).ravel()
        if bbox_arr.size != 4:  # does not happen in cadica, haven't checked for danilov
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        x1, y1, x2, y2 = bbox_arr
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))

        if x_max <= x_min or y_max <= y_min:  # again, for use with different datasets, will not happen with cadica
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
            }

        box_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        label_tensor = torch.tensor([1], dtype=torch.int64)

        return {'boxes': box_tensor, 'labels': label_tensor}



    def _fetch_video_clip(self, video):
        """
        Fetches frames for a video,
            if video length >= t_clip, samples a random clip of length t_clip
            if video length <= t_clip, takes all frames and pads the rest
        """

        if not isinstance(video, XCAVideo):
            raise TypeError(f"Expected XCAVideo, got {type(video)}")
        total_frames = video.frame_count
        if total_frames == 0:
            raise ValueError(f"Video {video.patient_id}/{video.video_id} has no frames.")

        uid = (video.patient_id, video.video_id)

        # 1. decide clip range
        clip_len = min(total_frames, self.t_clip)
        start = 0 if clip_len < self.t_clip else random.randint(0, total_frames - self.t_clip)
        end = start + clip_len

        # 2. slice raw frames & boxes
        raw_frames = video.frames[start:end]
        length = len(raw_frames)

        # pick & slice bboxes or fall back to None
        bboxes_slice = (
            video.bboxes[start:end]
            if video.bboxes is not None
            else [None] * length
        )

        # filter out empty boxes
        clip_bboxes = [
            b if (b is not None and b.any()) else None
            for b in bboxes_slice
        ]

        # 3. process frames
        processed = [self._process_frame(f, uid) for f in raw_frames]
        if not processed:
            raise RuntimeError(f"No frames processed for video {uid}")

        # 4. stack and pad tensor
        clip_tensor = torch.stack(processed, dim=0)  # [clip_len, C, H, W]
        pad = self.t_clip - clip_len
        if pad > 0:
            # pad=(T, C, H, W) -> (0,0 for W, 0,0 for H, 0,0 for C, 0, padding_size for T)
            clip_tensor = F.pad(clip_tensor, (0, 0, 0, 0, 0, 0, 0, pad))

        # 5. build mask
        mask = torch.arange(self.t_clip, device=clip_tensor.device) < clip_len

        # 6. build & pad targets
        targets = [self._get_target_dict(b) for b in clip_bboxes]
        if pad:
            empty = self._get_target_dict(None)
            targets.extend([empty] * pad)

        # 7. add metadata
        metadata = {
            "patient_id": video.patient_id,
            "video_id": video.video_id,
            "start_frame_orig": start,
            "end_frame_orig": end - 1,
            "num_frames_in_clip": clip_len,
            "total_frames_in_video": total_frames,
            "is_full_video": clip_len < self.t_clip,
        }

        return clip_tensor, targets, mask, metadata


    def _fetch_image(self, img):
        if not isinstance(img, XCAImage):
            raise TypeError(f"Expected XCAImage, got {type(img)}")

        frame = self._process_frame(img.image, video_uid=None)
        target = self._get_target_dict(getattr(img, "bbox", None))

        mask = torch.ones((1, ), dtype=torch.bool)

        metadata = {
            'patient_id': img.patient_id,
            'video_id': img.video_id,
            'frame_nr': img.frame_nr
        }

        return frame.unsqueeze(0), [target], mask, metadata  # unsqueeze the tensor for consistency [1, C, H, W]





    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        if not (0 <= idx < len(self.data_list)):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self.data_list)}')

        item = self.data_list[idx]
        if self.using_video_format:
            return self._fetch_video_clip(item)
        else:
            return self._fetch_image(item)




if __name__ == '__main__':
    try:
        # welcome to ata's debugging nightmare
        r = Reader(dataset_dir=CADICA_DATASET_DIR)

        T_CLIP_TEST = 42

        log("\n--- Testing with Video Data ---")
        videos = r.construct_videos()
        if videos:
            log(f"Constructed {len(videos)} videos.")
            log(f"Initializing Training Video Dataset (Augmentation ON, t_clip={T_CLIP_TEST})...")
            train_video_dataset = XCADataset(
                data_list=list(videos),
                use_augmentation=True,
                is_train=True,
                repeat_channels=True,
                t_clip=T_CLIP_TEST
            )
            log(f"Train Video Dataset length (using all videos): {len(train_video_dataset)}")

            if len(train_video_dataset) > 0:
                 log(f"Getting first item (clip, potentially padded) from train video dataset (t_clip={T_CLIP_TEST})...")
                 clip1, targets1, mask1, meta1 = train_video_dataset[0]
                 log(f"Item 0 - Meta: {meta1}")
                 log(f"Item 0 - Clip shape: {clip1.shape}, Mask shape: {mask1.shape}, Mask sum: {mask1.sum().item()}")
                 log(f"Item 0 - Targets length: {len(targets1)}")
                 first_real_frame_idx = mask1.nonzero(as_tuple=True)[0][0].item() if mask1.sum() > 0 else -1
                 if first_real_frame_idx != -1 and first_real_frame_idx < len(targets1):
                      log(f"Target for first real frame ({first_real_frame_idx}) of clip 1: {targets1[first_real_frame_idx]}")
                 else:
                      log("No real frames or targets found for clip 1.")


                 log("Getting first item again (should have same aug within epoch)...")
                 clip1_again, _, mask1_again, _ = train_video_dataset[0]
                 is_identical = torch.equal(clip1, clip1_again)
                 is_mask_identical = torch.equal(mask1, mask1_again)
                 log(f"Is the clip tensor identical when fetched again within same epoch? {is_identical} (May differ due to sampling)")
                 log(f"Is the mask identical when fetched again within same epoch? {is_mask_identical}")


                 log("Simulating epoch end...")
                 train_video_dataset.on_epoch_end()
                 log("Getting first item again after epoch end (should have different aug)...")
                 clip1_ep2, _, mask1_ep2, meta1_ep2 = train_video_dataset[0]
                 is_identical_ep2 = torch.equal(clip1, clip1_ep2)
                 log(f"Is the clip tensor identical after epoch end? {is_identical_ep2}")
                 log(f"Meta ep2: {meta1_ep2}")
                 log(f"Mask sum ep2: {mask1_ep2.sum().item()}")


            log("\nInitializing Validation Video Dataset (Augmentation OFF)...")
            norm_params_val = train_video_dataset.normalize_params if len(train_video_dataset) > 0 and hasattr(train_video_dataset, 'normalize_params') else {'mean':0.5, 'std':0.5}
            val_video_dataset = XCADataset(
                data_list=list(videos),
                use_augmentation=False,
                is_train=False,
                normalize_params=norm_params_val,
                repeat_channels=True,
                t_clip=T_CLIP_TEST
            )
            log(f"Validation Video Dataset length: {len(val_video_dataset)}")
            if len(val_video_dataset) > 0:
                log("Getting first item from validation video dataset...")
                clip_val, targets_val, mask_val, meta_val = val_video_dataset[0]
                log(f"Item 0 - Meta: {meta_val}")
                log(f"Item 0 - Clip shape: {clip_val.shape}, Mask shape: {mask_val.shape}, Mask sum: {mask_val.sum().item()}")
                log(f"Item 0 - Targets length: {len(targets_val)}")
                first_real_frame_idx_val = mask_val.nonzero(as_tuple=True)[0][0].item() if mask_val.sum() > 0 else -1
                if first_real_frame_idx_val != -1 and first_real_frame_idx_val < len(targets_val):
                      log(f"Target for first real frame ({first_real_frame_idx_val}) of validation clip: {targets_val[first_real_frame_idx_val]}")
                else:
                      log("No real frames or targets found for validation clip.")
        else:
             log("No videos found by Reader.", )


        log("\n--- Testing with Image Data ---")
        images = r.xca_images
        if images:
             log(f"Found {len(images)} images.")
             log("Initializing Training Image Dataset (Augmentation ON)...")
             norm_params_img = train_video_dataset.normalize_params if len(train_video_dataset) > 0 and hasattr(train_video_dataset, 'normalize_params') else {'mean':0.5, 'std':0.5}
             train_image_dataset = XCADataset(
                 data_list=list(images),
                 use_augmentation=True,
                 is_train=True,
                 normalize_params=norm_params_img,
                 repeat_channels=True,
                 t_clip=T_CLIP_TEST
             )
             log(f"Train Image Dataset length: {len(train_image_dataset)}")
             if len(train_image_dataset) > 0:
                  log("Getting first two items from train image dataset...")
                  img_i1, target_i1_list, mask_i1, meta_i1 = train_image_dataset[0]
                  log(f"Item 0 - Meta: {meta_i1}, Img shape: {img_i1.shape}, Mask shape: {mask_i1.shape}")
                  log(f"Item 0 - Target list (len {len(target_i1_list)}): {target_i1_list}")

                  if len(train_image_dataset) > 1:
                       img_i2, target_i2_list, mask_i2, meta_i2 = train_image_dataset[1]
                       log(f"Item 1 - Meta: {meta_i2}, Img shape: {img_i2.shape}, Mask shape: {mask_i2.shape}")
                       log(f"Item 1 - Target list (len {len(target_i2_list)}): {target_i2_list}")


             log("\nInitializing Validation Image Dataset (Augmentation OFF)...")
             val_image_dataset = XCADataset(
                 data_list=list(images),
                 use_augmentation=False,
                 is_train=False,
                 normalize_params=norm_params_img,
                 repeat_channels=True,
                 t_clip=T_CLIP_TEST
             )
             log(f"Validation Image Dataset length: {len(val_image_dataset)}")
             if len(val_image_dataset) > 0:
                log("Getting first item from validation image dataset...")
                img_val_i, target_val_i_list, mask_val_i, meta_val_i = val_image_dataset[0]
                log(f"Item 0 - Meta: {meta_val_i}, Img shape: {img_val_i.shape}, Mask shape: {mask_val_i.shape}")
                log(f"Item 0 - Target list (len {len(target_val_i_list)}): {target_val_i_list}")

        else:
             log("No single images found by Reader.", )


    except ValueError as e:
         log(f"Value Error during dataset init or processing: {e}", )
    except Exception as e:
        log(f"An unexpected error occurred in __main__: {e}", )
        import traceback
        traceback.print_exc()