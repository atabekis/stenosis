# loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin

# HPC
from HPC_config import get_world_size
from torch.utils.data.distributed import DistributedSampler

def video_collate_fn(batch):
    videos, bboxes, metas = zip(*batch)
    return list(videos), list(bboxes), list(metas)

def frame_collate_fn(batch):
    frames, bboxes, metas = zip(*batch)
    images = list(frames)

    targets = [
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32, device=bbox.device),
            "labels": torch.empty((0,), dtype=torch.int64, device=bbox.device)
        } if torch.sum(bbox) == 0.0 else {
            "boxes": bbox.unsqueeze(0).float(),
            "labels": torch.tensor([0], dtype=torch.int64, device=bbox.device)
        }
        for bbox in bboxes
    ]
    return images, targets, list(metas)




class StenosisFrameDataset(Dataset):
    """Dataset class for --single-- frame images, alongside its metadata"""
    def __init__(self, xca_videos, repeat_channels=False):
        super().__init__()
        self.repeat_channels = repeat_channels

        self.frames = [
            frame
            for vid in xca_videos
            for frame in vid.video_tensor
        ]
        self.bboxes = [
            bbox
            for vid in xca_videos
            for bbox in vid.bboxes
        ]

        self.meta = [
            {'patient_id': vid.meta.get('patient_id'), 'video_id': vid.meta.get('video_id')}
            for vid in xca_videos
            for _ in range(vid.video_tensor.shape[0])
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.repeat_channels and frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        return frame, self.bboxes[idx], self.meta[idx]





class StenosisVideoDataset(Dataset):
    """Wrap list of XCAVideo objects in PyTorch Dataset"""
    def __init__(self, xca_videos, repeat_channels=False):
        super().__init__()
        self.xca_videos = xca_videos
        self.repeat_channels = repeat_channels


    def __len__(self):
        return len(self.xca_videos)

    def __getitem__(self, idx):
        xca_video = self.xca_videos[idx]
        X = xca_video.video_tensor

        if self.repeat_channels and X.shape[1] == 1:
            X = X.repeat(3, 1, 1)

        y = xca_video.bboxes

        # X, y = X.to(DEVICE), y.to(DEVICE)

        meta = {
            "patient_id": xca_video.meta["patient_id"],
            "video_id": xca_video.meta["video_id"],
            "severity": xca_video.meta["severity"]
        }

        return X, y, meta




class DatasetConstructor(BaseEstimator, TransformerMixin):
    """A pipeline object that receives a dict of splits and creates a pytorch Dataset object"""
    def __init__(self, batch_size=4, shuffle=True, num_workers=4, pin_memory = False ,mode=None, repeat_channels=False):
        if mode is None:
            raise ValueError('Model must have a mode from [single_frame, sequence]')

        self.mode = mode
        self.repeat_channels = repeat_channels

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.mode == 'single_frame':
            dataset_cls = StenosisFrameDataset
            collate_fn = frame_collate_fn
        elif self.mode == 'sequence':
            dataset_cls = StenosisVideoDataset
            collate_fn = video_collate_fn
        else: # triple redundancy lol
            raise ValueError(f"Unknown mode: {self.mode}")

        loaders = {}
        world_size = get_world_size() # will return either a > 1 or 1
        for split in ['train', 'val', 'test']:
            dataset = dataset_cls(X[split], repeat_channels=self.repeat_channels)

            # ----- Need some logic here to handle distributed loading for the HPC cluster -------
            if split == 'train' and world_size > 1:
                sampler = DistributedSampler(dataset)
                shuffle_flag = False  # the distributed sampler will handle the shuffling
            else:
                sampler = None
                shuffle_flag = self.shuffle if split == 'train' else False
            # ---- ----------------------------------------------------------------------- -------

            loaders[f'{split}_loader'] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory
            )
        return loaders