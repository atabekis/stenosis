# reader.py

# Python imports
import os
import re
from collections import defaultdict

import cv2
import random

import numpy as np
import pandas as pd
import concurrent.futures

from typing import Union
from pathlib import WindowsPath

import config
# Local imports
from util import log
from config import DEBUG, DEBUG_SIZE, DEFAULT_WIDTH, DEFAULT_HEIGHT, T_CLIP
from config import DANILOV_DATASET_DIR, DANILOV_DATASET_PATH, CADICA_DATASET_DIR


class Reader:
    """
    Reads the dataset directory, finds all .bmp + .xml pairs, and constructs XCAImage objects
    """
    def __init__(self, dataset_dir, debug=DEBUG) -> None:
        self.dataset_dir = dataset_dir
        self.xca_images = []

        self._which_dataset()

        if debug:
            log(f'Debug mode turned on, sampling {int(DEBUG_SIZE * 100)}% of the images')
            self.xca_images = random.sample(self.xca_images, int(len(self.xca_images) * DEBUG_SIZE))  # random.seed is set in pl.seed_everything

        log(f'Reader constructed {len(self.xca_images)} XCA images from the {self.dataset_type} dataset.')

    def _which_dataset(self):
        self.dataset_type = 'DANILOV' if 'DANILOV' in str(self.dataset_dir) else 'CADICA'
        self.dataset_path = self.dataset_dir / 'dataset' if self.dataset_type == 'DANILOV' else None

        if self.dataset_type == 'DANILOV':
            self._load_danilov()
        elif self.dataset_type == 'CADICA':
            self._load_cadica()

    def _load_danilov(self) -> None:
        """
        Loads the dataset from the merged CSV file, groups by filename to
        handle multiple bounding boxes per image, then constructs XCAImage objects.
        """
        df = self._merge_labels()

        # Group by filename, for possible multiple bounding boxes per image
        groups = df.groupby("filename")
        log("Building XCA images from CSV files using the Danilov dataset.")

        def build_xca_image(filename, group):
            return XCAImage.from_danilov(group, self.dataset_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda item: build_xca_image(item[0], item[1]), groups))
        self.xca_images = results


    def _load_cadica(self) -> None:
        """
        Load the CADICA dataset, only the selected frames are extracted from the data (i.e., only where the contrast
        agent is visible.
        Only the selectedVideos directory is used due to the same constraints.
        """
        base_dir = self.dataset_dir / 'selectedVideos'
        if not base_dir.is_dir():
            raise FileNotFoundError(f"No 'selectedVideos' folder in {self.dataset_dir}")

        tasks = []
        patients = sorted(base_dir.iterdir(), key=lambda p: int(p.name.lstrip('p')))
        for patient in patients:
            for list_file, has_gt in (('lesionVideos.txt', True),
                                      ('nonlesionVideos.txt', False)):
                list_path = patient / list_file
                if not list_path.exists():
                    continue
                video_dirs = [
                    patient / name
                    for name in list_path.read_text().splitlines()
                    if name
                ]
                for video_dir in video_dirs:
                    sel_frames = video_dir / f"{patient.name}_{video_dir.name}_selectedFrames.txt"
                    if not sel_frames.exists():
                        continue
                    frame_names = [ln for ln in sel_frames.read_text().splitlines() if ln]
                    input_dir = video_dir / 'input'
                    gt_dir = video_dir / 'groundtruth' if has_gt else None

                    for fn in frame_names:
                        img_path = input_dir / f"{fn}.png"
                        if not img_path.exists():
                            continue
                        ann = None
                        if has_gt:
                            gt_path = gt_dir / f"{fn}.txt"
                            if gt_path.exists():
                                ann = gt_path.read_text()
                        tasks.append((img_path, ann))

        def _make_image(task):
            path, ann = task
            try:
                return XCAImage.from_cadica(path, ann)
            except Exception:
                return None

        log('Building XCA images from the CADICA dataset...')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(_make_image, tasks)

        self.xca_images = [img for img in results if img is not None]



    def _merge_labels(self) -> pd.DataFrame:
        """
        Check if train_labels.csv & test_labels.csv are merged → labels.csv. Used for the DANILOV dataset
        """
        labels_csv = os.path.join(self.dataset_dir, 'labels.csv')
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            log(f"Merged dataset found at {labels_csv}. Loading merged dataset")
            return df
        else:
            train, test = os.path.join(self.dataset_dir, 'train_labels.csv'), os.path.join(self.dataset_dir, 'test_labels.csv')
            train_df, test_df = pd.read_csv(train), pd.read_csv(test)
            df = pd.concat([train_df, test_df], ignore_index=True)
            df.sort_values('filename', inplace=True)
            df.to_csv(labels_csv, index=False)
            log(f"Merged dataset created and saved to {labels_csv}")
            return df


    def get(self, patient_id: int =None, video_id: int =None, frame_nr: int =None,
            return_videos: bool =False, return_frames: bool =False,
            lesion: bool | None = None) -> Union['XCAImage', list['XCAImage']]:
        """
        Returns a XCAImage instance, if either "patient_id, video_id, frame_nr" is None,
        return a random image from the subset.
        :return:
        """

        subset = self.xca_images

        if lesion is not None and self.dataset_type == 'CADICA':  # we first need to filter lesion/nonlesion videos
            if lesion:
                subset = [x for x in subset if x.bbox is not None and x.stenosis_severity is not None] # lesion
            else:
                subset = [x for x in subset if x.bbox is None and x.stenosis_severity is None] # nonlesion

        criteria = [
            ("patient_id", patient_id, return_videos),
            ("video_id", video_id, return_frames),
            ("frame_nr", frame_nr, False)
        ]

        for attr, value, should_return in criteria:
            if value is not None:
                subset = [x for x in subset if getattr(x, attr) == value]
                if should_return:
                    if not subset:
                        raise ValueError(
                            f"No images found for the given criteria (patient_id={patient_id}, video_id={video_id}, frame_nr={frame_nr}),"
                            f"lesion={lesion}"
                        )
                    return subset

        if not subset:
            raise ValueError(
                f"No images found for the given criteria (patient_id={patient_id}, video_id={video_id}, frame_nr={frame_nr})."
            )

        if patient_id is not None and video_id is not None and frame_nr is not None:
            if len(subset) == 1:
                return subset[0]
            else:  # redundancy
                raise ValueError(
                    f"Expected exactly one image, found {len(subset)} "
                    f"(patient_id={patient_id}, video_id={video_id}, frame_nr={frame_nr})."
                )

        return random.choice(subset)



    def construct_videos(self, default_width=DEFAULT_WIDTH, default_height=DEFAULT_HEIGHT):
        """Group XCAImage instances by patient_id and video_id to construct XCAVideo objects"""

        log("Constructing XCAVideo sequences from XCA images...")

        videos_dict = defaultdict(list)
        # first we group images by pid, vid
        for image in self.xca_images:
            key = (image.patient_id, image.video_id)
            videos_dict[key].append(image)

        for key in videos_dict:
            videos_dict[key].sort(key=lambda x: x.frame_nr)  # sort frames by frame_nr

        # videos = []

        def process_video(key_frames):
            (patient_id, video_id), frames = key_frames
            frame_count = len(frames)

            frames_array = np.zeros((frame_count, 1, default_width, default_height), dtype=np.uint8)  # all images to be resized to 512x512
            original_dimensions = [(frame.width, frame.height) for frame in frames]

            has_bbox = all(frame.bbox is not None for frame in frames)
            bboxes_array = np.zeros((frame_count, 4), dtype=np.int32) if has_bbox else None

            # fill the arrays
            for i, frame in enumerate(frames):
                img = frame.image

                if img.ndim == 3:  # redundancy: check if RGB, if so convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if img.shape[0] != default_width or img.shape[1] != default_height:  # scale the images to 512x512 (for DANILOV only)
                    scale_factor = min(default_width / img.shape[1], default_height / img.shape[0])
                    width, height = int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)

                    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                    canvas = np.zeros((default_width, default_height), dtype=np.uint8) # create blank canvas
                    x_offset, y_offset = (default_width - width) // 2, (default_height - height) // 2  # calculate where to place resized img
                    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = img_resized
                    img = canvas

                    # finally, we also need to adjust the bounding box if resized
                    if has_bbox and frame.bbox:
                        # original_width, original_height = frame.width, frame.height
                        if isinstance(frame.bbox[0], tuple) and len(frame.bbox) > 0:  # for danilov there might be multiple bounding boxes (one image)
                            xmin, ymin, xmax, ymax = frame.bbox[0]
                        else:
                            xmin, ymin, xmax, ymax = frame.bbox

                        xmin_scaled = int(xmin * scale_factor) + x_offset
                        ymin_scaled = int(ymin * scale_factor) + y_offset
                        xmax_scaled = int(xmax * scale_factor) + x_offset
                        ymax_scaled = int(ymax * scale_factor) + y_offset
                        bboxes_array[i] = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]
                else:
                    if has_bbox and frame.bbox:

                        if isinstance(frame.bbox[0], tuple) and len(frame.bbox) > 0:
                            # danilov has multiple bboxes, take the first one
                            bboxes_array[i] = np.array(frame.bbox[0])
                        else:
                            bboxes_array[i] = np.array(frame.bbox)

                frames_array[i, 0] = img
            video = XCAVideo(patient_id, video_id, frames_array, bboxes_array, original_dimensions)

            for frame in frames:
                if frame.stenosis_severity is not None:
                    video.stenosis_severity = frame.stenosis_severity
                    break

            return video

        with concurrent.futures.ThreadPoolExecutor() as executor:
            videos = list(executor.map(process_video, videos_dict.items()))

        log(f'Reader constructed {len(videos)} total videos.')

        return sorted(videos, key=lambda v: (v.patient_id, v.video_id))


    def __repr__(self):
        return f"Reader(dataset_path='{self.dataset_path}', total_images={len(self.xca_images)})"



class XCAImage:
    """
    Represents a single coronary angiography image
    """

    STENOSIS_SEVERITY_MAP = {  # For CADICA
        "p0_20": (0, 20),
        "p20_50": (20, 50),
        "p50_70": (50, 70),
        "p70_90": (70, 90),
        "p90_98": (90, 98),
        "p99": (99, 99),
        "p100": (100, 100)
    }

    def __init__(self):
        """
        Initialize an XCAImage instance
        """
        self.filename = None
        self.patient_id, self.video_id, self.frame_nr = None, None, None
        self.width, self.height = None, None
        self.bbox = []
        self.stenosis_severity = None  # p0_20 = 0% to 20% stenosis, p20_50 = ...  only in CADICA

        self.path = None
        self.image = None

        self.dataset = None


    @classmethod
    def from_danilov(cls, bbox_rows: pd.DataFrame, dataset_path: str) -> 'XCAImage':
        """
        Constructs an XCAImage instance from CSV rows corresponding to a single image.
        """
        instance = cls()
        instance.dataset = "DANILOV"

        row = bbox_rows.iloc[0]
        instance.filename = row['filename']
        instance.width, instance.height = int(row['width']), int(row['height'])
        instance.stenosis_severity = None

        instance._parse_filename()

        instance.path = os.path.join(dataset_path, instance.filename)
        instance.image = cv2.imread(instance.path, cv2.IMREAD_UNCHANGED)
        if instance.image is None:
            raise IOError(f"Could not read image file: {instance.path}")
        for _, r in bbox_rows.iterrows():
            xmin, ymin, xmax, ymax = map(int, (r['xmin'], r['ymin'], r['xmax'], r['ymax']))
            instance.bbox.append((xmin, ymin, xmax, ymax))
        return instance

    @classmethod
    def from_cadica(cls, path: WindowsPath, annotation: str | None) -> 'XCAImage':
        """Constructs an XCAImage instance from a Reader directly."""
        instance = cls()
        instance.dataset = "CADICA"

        instance.filename = path.name
        instance.width, instance.height = 512, 512  # set for all images

        instance._parse_filename()
        instance._parse_annotation(annotation=annotation)  # for severity and bbox

        instance.path = path
        instance.image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if instance.image is None:
            raise IOError(f"Could not read image file: {path}")
        return instance

    def _parse_annotation(self, annotation: str | None):
        """
        For the CADICA dataset, if the file has annotation from the /groundtruth directory,
        parses the string for bounding box information and stenosis severity
        """
        if annotation is None:
            self.bbox = None
            self.stenosis_severity = None
        else:
            m = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(p\d+_\d+|p\d+)", annotation)
            if not m:  # check if there exists any multiple bounding boxes
                raise ValueError(f"Invalid annotation format: {annotation}, in file: {self.path}")

            xmin, ymin, w, h = map(int, m.groups()[:4])
            xmax, ymax = xmin + w, ymin + h  # in CADICA the format is x y w h
            self.bbox = [xmin, ymin, xmax, ymax]

            stenosis_severity = m.group(5)
            self.stenosis_severity = self.STENOSIS_SEVERITY_MAP.get(stenosis_severity, (None, None))


    def _parse_filename(self) -> None:
        """
        Extracts patient_id, video_id, and frame_nr from the filename.
        Expected format: 14_<patient>_<video>_<frame>.bmp or pX_vY_frameNr.png for cadica.
        """
        pattern = r"^14_(\d+)_(\d+)_(\d+)\.bmp$" if self.dataset == "DANILOV" else r"^p(\d+)_v(\d+)_(\d{5})\.png$"
        m = re.match(pattern, self.filename)
        if m:
            self.patient_id, self.video_id, self.frame_nr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            raise ValueError(f"filename {self.filename} does not match the expected pattern.")


    def get_image(self):
        """
        Lazy-loads and returns the image using cv2.imread."""
        return DeprecationWarning
        # if self.image is None:
        #     self.image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        #     if self.image is None:
        #         raise IOError(f"Could not read image file: {self.path}")
        # return self.image

    def __repr__(self):
        return (f"XCAImage(patient={self.patient_id}, video={self.video_id}, frame={self.frame_nr}, "
                f"width={self.width}, height={self.height}, bbox(es)={self.bbox}, "
                f"stenosis_severity={self.stenosis_severity}, dataset={self.dataset})")



class XCAVideo:
    """
    Represents a sequence of consequent XCAImage instances from the same patient and video
    """
    def __init__(self, patient_id, video_id, frames, bboxes, original_dimensions=None):
        self.patient_id, self.video_id = patient_id, video_id
        self.frames, self.bboxes = frames, bboxes  # frames=[frame_count, 1, width, height], bboxes=[frame_count, 4] or None

        self.frame_count = frames.shape[0]
        self.has_lesion = bboxes is not None
        self.original_dimensions = original_dimensions

        self.stenosis_severity = None  # will extract severity from the first frame (if any)

    def __repr__(self):
        shape_str = f"{self.frames.shape[2]}x{self.frames.shape[3]}"
        return (f"XCAVideo(patient_id={self.patient_id}, video_id={self.video_id}, "
                f"frame_count={self.frame_count}, shape={shape_str}, has_lesion={self.has_lesion}, "
                f"stenosis_severity={self.stenosis_severity})")





if __name__ == "__main__":
    reader = Reader(dataset_dir=CADICA_DATASET_DIR)
    # print(len(reader.xca_images))
    videos = reader.construct_videos()
    images = reader.xca_images
    print(len(images))



