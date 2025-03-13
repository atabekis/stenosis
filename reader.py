# reader.py

# Python imports
import os
import re

import cv2
import pandas as pd
import concurrent.futures

from typing import Union
from pathlib import WindowsPath

# Local imports
from util import log
from config import DANILOV_DATASET_DIR, DANILOV_DATASET_PATH, CADICA_DATASET_DIR


class Reader:
    """
    Reads the dataset directory, finds all .bmp + .xml pairs, and constructs XCAImage objects
    """
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        self.xca_images = []

        self._which_dataset()

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
        log("Building XCA images from CSV files")

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
        selected_videos = self.dataset_dir / 'selectedVideos'
        selected_patients = sorted(
            [v for v in selected_videos.iterdir() if v.is_dir()],
            key=lambda x: int(x.name.lstrip('p'))  # for getting patients in the format p1, p2, p3, as Path obj.
        )

        def classify_videos(patient_dir: WindowsPath):
            """lesion, non-lesion, extract"""
            lesion = (patient_dir / 'lesionVideos.txt').read_text().split('\n')[:-1]
            nonlesion = (patient_dir / 'nonlesionVideos.txt').read_text().split('\n')[:-1]
            return [patient_dir / v for v in lesion], [patient_dir / v for v in nonlesion]


        def construct_xca(video_dir: WindowsPath, lesion: bool):
            if not lesion:  # XCAImage.bbox = None
                selected_frames = ((video_dir / f'{video_dir.parent.name}_{video_dir.name}_SelectedFrames.txt')
                                   .read_text().split('\n')[:-1])  # ooga-booga way of reading :)
                selected_frames = [video_dir / "input" / f"{frame}.png" for frame in selected_frames]

                # Generate XCAImage instances from non-lesion videos meaning no bounding boxes / annotation
                for frame_path in selected_frames:
                    self.xca_images.append(
                        XCAImage.from_cadica(frame_path, annotation=None)
                    )

            if lesion:  # now we have the groundTruth to deal with as well
                selected_frames = ((video_dir / f'{video_dir.parent.name}_{video_dir.name}_SelectedFrames.txt')
                                   .read_text().split('\n')[:-1])
                selected_frames_paths = [video_dir / "input" / f"{frame}.png" for frame in selected_frames]
                ground_truth_files = [video_dir / "groundtruth" / f"{frame}.txt" for frame in selected_frames]

                for frame_path, truth in zip(selected_frames_paths, ground_truth_files):
                    annotation = truth.read_text()

                    # Generate XCAImage instances, now together with the ground truth annotations
                    self.xca_images.append(
                        XCAImage.from_cadica(frame_path, annotation=annotation)
                    )


        # Extract all lesion/nonlesion videos for each patient
        lesion_videos, nonlesion_videos = map(list, zip(*[classify_videos(patient) for patient in selected_patients]))

        # Construct all XCAImage instances
        [construct_xca(v, lesion=True) for video in lesion_videos for v in video]
        [construct_xca(v, lesion=False) for video in nonlesion_videos for v in video]



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
            log(f"Merged dataset created and saved to {labels_csv}.")
            return df


    def get(self, patient_id: int =None, video_id: int =None, frame_nr: int =None,
            return_videos: bool =False, return_frames: bool =False) -> Union['XCAImage', list['XCAImage']]:
        """
        Returns a XCAImage instance, if either "patient_id, video_id, frame_nr" is None,
        return a random image from the subset.
        :return:
        """
        import random

        subset = self.xca_images
        criteria = [
            ("patient_id", patient_id, return_videos),
            ("video_id", video_id, return_frames),
            ("frame_nr", frame_nr, False)
        ]

        for attr, value, should_return in criteria: # so when we want to return the subset directly, we can do with the two return args
            if value is not None:
                subset = [x for x in subset if getattr(x, attr) == value]  # systematically shrink based on pid, vid, frame_nr
                if should_return:
                    return subset

        if not subset:  # if our subset ended up empty, means something is wrong with the args
            raise ValueError(
                f"No images found for the given criteria (patient_id={patient_id}, video_id={video_id}, frame_nr={frame_nr})."
            )

        if patient_id is not None and video_id is not None and frame_nr is not None:  # extra redundancy, not exactly needed
            if len(subset) == 1:
                return subset[0]
            else:
                raise ValueError(
                    f"Expected exactly one image, found {len(subset)} "
                    f"(patient_id={patient_id}, video_id={video_id}, frame_nr={frame_nr})."
                )

        return random.choice(subset)


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
        self._image = None  # lazy loading

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
        instance._image = None

        for _, row in bbox_rows.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
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
        instance._image = None

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

            xmin, ymin, xmax, ymax = map(int, m.groups()[:4])
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
        if self._image is None:
            self._image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            if self._image is None:
                raise IOError(f"Could not read image file: {self.path}")
        return self._image

    def __repr__(self):
        return (f"XCAImage(patient={self.patient_id}, video={self.video_id}, frame={self.frame_nr}, "
                f"width={self.width}, height={self.height}, bbox(es)={self.bbox}), "
                f"stenosis_severity={self.stenosis_severity}, dataset={self.dataset})")




if __name__ == "__main__":
    reader = Reader(dataset_dir=CADICA_DATASET_DIR)


