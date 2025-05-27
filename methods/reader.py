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
from pathlib import WindowsPath, Path

import config
# Local imports
from util import log
from config import DEBUG, DEBUG_SIZE, DEFAULT_WIDTH, DEFAULT_HEIGHT, T_CLIP
from config import DANILOV_DATASET_DIR, DANILOV_DATASET_PATH, CADICA_DATASET_DIR
from config import CADICA_NEGATIVE_ONLY_ON_BOTH, MIN_SUBSEGMENT_LENGTH

from methods.video_utils import split_frames_into_subsegments


class Reader:
    """
    Reads the dataset directory, finds all .bmp + .xml pairs, and constructs XCAImage objects
    """
    def __init__(
            self,
            dataset_dir=None,
            debug=DEBUG,
            cadica_negative_only=CADICA_NEGATIVE_ONLY_ON_BOTH,
            iou_split_thresh = 0.01,
            apply_gt_splitting = True,) -> None:

        self.dataset_dir = dataset_dir # can be path to CADICA, DANILOV or BOTH
        self.cadica_negative_only = cadica_negative_only

        self.iou_split_thresh = iou_split_thresh
        self.apply_gt_splitting = apply_gt_splitting

        self.xca_images = []

        self._which_dataset()

        if debug:
            log(f'Debug mode turned on, sampling {int(DEBUG_SIZE * 100)}% of the images')
            self.xca_images = random.sample(self.xca_images, int(len(self.xca_images) * DEBUG_SIZE))  # random.seed is set in pl.seed_everything

        log(f'Reader constructed {len(self.xca_images)} XCA images from the {self.dataset_type} dataset.')

    def _which_dataset(self):
        self.dataset_type = 'DANILOV' if 'DANILOV' in str(self.dataset_dir) else 'CADICA'

        if 'DANILOV' in str(self.dataset_dir).upper():
            self.dataset_type = 'DANILOV'
            self.xca_images = self._load_danilov(self.dataset_dir)
        elif 'CADICA' in str(self.dataset_dir).upper():
            self.dataset_type = 'CADICA'
            self.xca_images = self._load_cadica(self.dataset_dir, negative_only=False)
        elif self.dataset_dir.upper() == 'BOTH':
            self._load_both(self.cadica_negative_only)


    def _load_danilov(self, dataset_dir) -> list['XCAImage']:
        """
        Loads the dataset from the merged CSV file, groups by filename to
        handle multiple bounding boxes per image, then constructs XCAImage objects.
        """
        df = self._merge_labels(dataset_dir)

        # Group by filename, for possible multiple bounding boxes per image
        groups = df.groupby("filename")
        log("Building XCA images from CSV files using the Danilov dataset.")

        data_path = Path(dataset_dir) / 'dataset' if 'DANILOV' in str(dataset_dir).upper() else dataset_dir

        def build_xca_image(filename, group):
            return XCAImage.from_danilov(group, data_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda item: build_xca_image(item[0], item[1]), groups))

        return [img for img in results if img is not None]


    def _load_cadica(self, dataset_dir, negative_only=False) -> list['XCAImage']:
        """
        Load the CADICA dataset, only the selected frames are extracted from the data (i.e., only where the contrast
        agent is visible.
        Only the selectedVideos directory is used due to the same constraints.
        """
        base_dir = dataset_dir / 'selectedVideos'
        if not base_dir.is_dir():
            raise FileNotFoundError(f"No 'selectedVideos' folder in {dataset_dir}")

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

                        is_positive = ann is not None and any(char.isdigit() for char in ann.split()[:4])
                        if negative_only and is_positive:
                            continue  # pass the positive examples

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

        return [img for img in results if img is not None]


    def _load_both(self, cadica_negative_only):
        self.xca_images = []  # fresh start
        self.dataset_type = 'BOTH'
        log(f'Loading both datasets. CADICA negative samples only: {self.cadica_negative_only}')

        danilov_images = self._load_danilov(DANILOV_DATASET_DIR)
        self.xca_images.extend(danilov_images)
        log(f'Loaded {len(danilov_images)} images from DANILOV.')

        cadica_images = self._load_cadica(CADICA_DATASET_DIR, negative_only=cadica_negative_only)
        self.xca_images.extend(cadica_images)
        log(f'Loaded {len(cadica_images)} images from CADICA.')


    def _merge_labels(self, dataset_dir) -> pd.DataFrame:
        """
        Check if train_labels.csv & test_labels.csv are merged → labels.csv. Used for the DANILOV dataset
        """
        labels_csv = os.path.join(dataset_dir, 'labels.csv')
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            log(f"Merged dataset found at {labels_csv}. Loading merged dataset")
            return df
        else:
            train, test = os.path.join(dataset_dir, 'train_labels.csv'), os.path.join(dataset_dir, 'test_labels.csv')
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



    def construct_videos(self, default_width=DEFAULT_WIDTH, default_height=DEFAULT_HEIGHT, min_subsegment_len=MIN_SUBSEGMENT_LENGTH):
        """Group XCAImage instances by patient_id and video_id to construct XCAVideo objects"""

        log("Constructing XCAVideo sequences from XCA images...")

        log(f'Sub-segmenting videos based on bounding box movement.', verbose=self.apply_gt_splitting)  # will only print if apply_gt_splitting
        log(f'   IoU threshold: {self.iou_split_thresh}', verbose=self.apply_gt_splitting)
        log(f'   Minimum subsegment length: {min_subsegment_len}', verbose=self.apply_gt_splitting)


        videos_dict = defaultdict(list)
        # first we group images by pid, vid
        for image in self.xca_images:
            key = (image.dataset ,image.patient_id, image.video_id)
            videos_dict[key].append(image)

        for key in videos_dict:
            videos_dict[key].sort(key=lambda x: x.frame_nr)  # sort frames by frame_nr

        videos = []

        def process_video(key_frames):
            (dataset_source, patient_id, video_id), frames = key_frames
            if not frames: return []

            is_group_lesion_type = any(f.bbox is not None for f in frames)

            if self.apply_gt_splitting and is_group_lesion_type:
                subsegments_of_frames = split_frames_into_subsegments(frames, self.iou_split_thresh)
            else:
                subsegments_of_frames = [frames]

            video_sub_list = []
            for sub_idx, frame_subsegment in enumerate(subsegments_of_frames):
                if not frame_subsegment or len(frame_subsegment) < min_subsegment_len:
                    continue

                frame_count = len(frame_subsegment)
                frames_array = np.zeros((frame_count, 1, default_height, default_width), dtype=np.uint8)
                original_dimensions_sub = [(fr.original_width, fr.original_height) for fr in frame_subsegment]

                all_frames_have_bbox = all(f.bbox is not None for f in frame_subsegment)
                bboxes_array = np.zeros((frame_count, 4), dtype=np.int32) if all_frames_have_bbox else None

                for i, frame_obj in enumerate(frame_subsegment):
                    # image is already resized, grayscale, and padded in XCAImage object
                    frames_array[i, 0] = frame_obj.image

                    if all_frames_have_bbox and frame_obj.bbox is not None:
                        current_bbox_to_use = frame_obj.bbox
                        if (
                            frame_obj.dataset == "DANILOV"
                            and isinstance(frame_obj.bbox, list)
                            and frame_obj.bbox and isinstance(frame_obj.bbox[0], tuple)):
                            current_bbox_to_use = frame_obj.bbox[0] # if danilov has multiple bboxes pick the first one

                        bboxes_array[i] = np.array(current_bbox_to_use, dtype=np.int32)

                video = XCAVideo(
                    patient_id=patient_id,
                    video_id=video_id,
                    subsegment_id=sub_idx,
                    frames=frames_array,
                    bboxes=bboxes_array,
                    original_dimensions=original_dimensions_sub,
                    dataset=dataset_source
                )


                if frame_subsegment and frame_subsegment[0].stenosis_severity is not None:
                    video.stenosis_severity = frame_subsegment[0].stenosis_severity
                video_sub_list.append(video)
            return video_sub_list

        with concurrent.futures.ThreadPoolExecutor() as executor:
             list_of_video_sub_lists = list(executor.map(process_video, videos_dict.items()))

        for sub_list in list_of_video_sub_lists:
            videos.extend(sub_list)

        log(f'Reader constructed {len(videos)} total XCAVideo sub-segments (after potential splitting).')
        return sorted(videos, key=lambda v: (v.dataset, v.patient_id, v.video_id, v.subsegment_id))


    def __repr__(self):
        return f"Reader(dataset='{self.dataset_type}', total_images={len(self.xca_images)})"



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
        self.path = None

        self.patient_id, self.video_id, self.frame_nr = None, None, None

        self.original_width, self.original_height = None, None
        self.width, self.height = DEFAULT_WIDTH, DEFAULT_HEIGHT

        self.bbox = []
        self.stenosis_severity = None  # p0_20 = 0% to 20% stenosis, p20_50 = ...  only in CADICA

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
        instance.stenosis_severity = None

        instance._parse_filename()

        instance.path = os.path.join(dataset_path, instance.filename)
        instance.image = cv2.imread(instance.path, cv2.IMREAD_UNCHANGED)
        if instance.image is None:
            raise IOError(f"Could not read image file: {instance.path}")

        if not bbox_rows.empty:
            r = bbox_rows.iloc[0]  # one image in danilov has 2 bboxes, we take one
            xmin, ymin, xmax, ymax = map(int, (r['xmin'], r['ymin'], r['xmax'], r['ymax']))
            instance.bbox = [xmin, ymin, xmax, ymax]

        instance._resize_to_default()
        return instance

    @classmethod
    def from_cadica(cls, path: WindowsPath, annotation: str | None) -> 'XCAImage':
        """Constructs an XCAImage instance from a Reader directly."""
        instance = cls()
        instance.dataset = "CADICA"

        instance.filename = path.name

        instance._parse_filename()
        instance._parse_annotation(annotation=annotation)  # for severity and bbox

        instance.path = path
        instance.image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if instance.image is None:
            raise IOError(f"Could not read image file: {path}")

        instance._resize_to_default()
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


    def _resize_to_default(self):
        """Resizes the loaded self.image to DEFAULT_WIDTH and DEFAULT_HEIGHT, adjusts the bboxes (if any) accordingly"""

        if self.image.ndim == 3:  # ensuring image is grayscale
            if self.image.shape[2] == 3:  #bgr
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            elif self.image.shape[2] == 4: #bgra
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        current_h, current_w = self.image.shape[:2]
        self.original_height, self.original_width = current_h, current_w

        if current_h == DEFAULT_HEIGHT and current_w == DEFAULT_WIDTH:  # already expected format
            self.width, self.height = DEFAULT_WIDTH, DEFAULT_HEIGHT
            return

        scale_factor = min(DEFAULT_HEIGHT / current_h, DEFAULT_WIDTH / current_w)  # scaling with padding
        new_h, new_w = int(current_h * scale_factor), int(current_w * scale_factor)

        img_resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # create canvas of the def. size and img original dtype
        canvas = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=self.image.dtype)
        y_offset = (DEFAULT_HEIGHT - new_h) // 2
        x_offset = (DEFAULT_WIDTH - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        self.image = canvas
        self.height, self.width = DEFAULT_HEIGHT, DEFAULT_WIDTH

        if self.bbox:
            xmin, ymin, xmax, ymax = self.bbox

            xmin_s = xmin * scale_factor + x_offset
            ymin_s = ymin * scale_factor + y_offset
            xmax_s = xmax * scale_factor + x_offset
            ymax_s = ymax * scale_factor + y_offset

            self.bbox = [int(xmin_s), int(ymin_s), int(xmax_s), int(ymax_s)]



    def __repr__(self):
        return (f"XCAImage(patient={self.patient_id}, video={self.video_id}, frame={self.frame_nr}, "
                f"width={self.width}, height={self.height}, bbox(es)={self.bbox}, "
                f"stenosis_severity={self.stenosis_severity}, dataset={self.dataset})")



class XCAVideo:
    """
    Represents a sequence of consequent XCAImage instances from the same patient and video
    """
    def __init__(self, patient_id, video_id, frames, bboxes, dataset,
                 original_dimensions=None, subsegment_id=0):
        self.patient_id, self.video_id = patient_id, video_id
        self.frames, self.bboxes = frames, bboxes  # frames=[frame_count, 1, width, height], bboxes=[frame_count, 4] or None
        self.dataset = dataset
        self.frame_count = frames.shape[0]
        self.has_lesion = bboxes is not None
        self.original_dimensions = original_dimensions
        self.subsegment_id = subsegment_id

        self.stenosis_severity = None  # will extract severity from the first frame (if any)

    def __repr__(self):
        shape_str = f"{self.frames.shape[2]}x{self.frames.shape[3]}"
        return (f"XCAVideo(patient_id={self.patient_id}, video_id={self.video_id}, subsegment={self.subsegment_id}, "
                f"frame_count={self.frame_count}, shape={shape_str}, has_lesion={self.has_lesion}, "
                f"stenosis_severity={self.stenosis_severity}, dataset={self.dataset})")





if __name__ == "__main__":
    reader = Reader(dataset_dir='both', cadica_negative_only=True)
    print(len(reader.xca_images))
    videos = reader.construct_videos()
    for v in videos:
        print(v)
    # images = reader.xca_images
    # print(len(images))



