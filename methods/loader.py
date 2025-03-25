# loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin

# Local imports
from util import log
from config import DEVICE


def video_collate_fn(batch):
    videos, bboxes, metas = zip(*batch)
    return list(videos), list(bboxes), list(metas)



class StenosisFrameDataset(Dataset):
    """Dataset class for --single-- frame images, alongside its metadata"""
    def __init__(self, xca_videos):
        super().__init__()

        self.frames = [
            frame.to(DEVICE)
            for vid in xca_videos
            for frame in vid.video_tensor
        ]
        self.bboxes = [
            bbox.to(DEVICE)
            for vid in xca_videos
            for bbox in (vid.bboxes if vid.bboxes is not None
                         else torch.zeros((vid.video_tensor.shape[0], 4)))
        ]
        self.meta = [
            {'patient_id': vid.meta.get('patient_id'), 'video_id': vid.meta.get('video_id')}
            for vid in xca_videos
            for _ in range(vid.video_tensor.shape[0])
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.bboxes[idx], self.meta[idx]





class StenosisVideoDataset(Dataset):
    """Wrap list of XCAVideo objects in PyTorch Dataset"""
    def __init__(self, xca_videos):
        super().__init__()
        self.xca_videos = xca_videos


    def __len__(self):
        return len(self.xca_videos)

    def __getitem__(self, idx):
        xca_video = self.xca_videos[idx]
        X = xca_video.video_tensor

        if xca_video.bboxes is None:
            frame_count = X.shape[0]
            y = torch.zeros((frame_count, 4), dtype=torch.float32)
        else:
            y = xca_video.bboxes

        X, y = X.to(DEVICE), y.to(DEVICE)

        meta = {
            "patient_id": xca_video.meta["patient_id"],
            "video_id": xca_video.meta["video_id"],
            "severity": xca_video.meta["severity"]
        }

        return X, y, meta




class DatasetConstructor(BaseEstimator, TransformerMixin):
    """A pipeline object that receives a dict of splits and creates a pytorch Dataset object"""
    def __init__(self, batch_size=4, shuffle=True, num_workers=4, mode=None):
        if mode is None:
            raise ValueError('Model must have a mode from [single_frame, sequence]')

        self.mode = mode

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers




    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.mode == 'single_frame':
            dataset_cls = StenosisFrameDataset
            collate_fn = None
        elif self.mode == 'sequence':
            dataset_cls = StenosisVideoDataset
            collate_fn = video_collate_fn
        else: # triple redundancy lol
            raise ValueError(f"Unknown mode: {self.mode}")

        loaders = {}
        for split in ['train', 'val', 'test']:
            dataset = dataset_cls(X[split])
            loaders[f'{split}_loader'] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle if split == 'train' else False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )
        return loaders





















        # train_dataset = StenosisDataset(X['train'])
        # val_dataset = StenosisDataset(X['val'])
        # test_dataset = StenosisDataset(X['test'])
        #
        # log(f"Constructing PyTorch dataset objects from XCA videos")
        #
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=self.shuffle,
        #     num_workers=self.num_workers,
        #     collate_fn=video_collate_fn
        # )
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     collate_fn=video_collate_fn
        # )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     collate_fn=video_collate_fn
        # )
        #
        # return {
        #     'train_loader': train_loader,
        #     'val_loader':   val_loader,
        #     'test_loader':  test_loader
        # }
