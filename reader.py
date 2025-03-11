# reader.py

# Python imports
import os
import re
import cv2
import pandas as pd
import concurrent.futures
import xml.etree.ElementTree as ET

from typing import Union

# Local imports
from util import log
from config import DATASET_DIR, DATASET_PATH


class Reader:
    """
    Reads the dataset directory, finds all .bmp + .xml pairs, and constructs XCAImage objects
    """
    def __init__(self, dataset_path=DATASET_PATH, dataset_dir=DATASET_DIR) -> None:
        self.dataset_path, self.dataset_dir = dataset_path, dataset_dir
        self.xca_images = []

        self._load_dataset()

    def _merge_labels(self):
        """
        Check if train_labels.csv & test_labels.csv are merged → labels.csv
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

    def _load_dataset(self) -> None:
        """
        Loads the dataset from the merged CSV file, groups by filename to
        handle multiple bounding boxes per image, then constructs XCAImage objects.
        """
        df = self._merge_labels()

        # Group by filename, for possible multiple bounding boxes per image
        groups = df.groupby("filename")
        log("Building XCA images from CSV files")

        def build_xca_image(filename, group):
            return XCAImage.from_csv(group, self.dataset_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda item: build_xca_image(item[0], item[1]), groups))
        self.xca_images = results

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
    def __init__(self):
        """
        Initialize an XCAImage instance
        """
        self.filename = None
        self.patient_id, self.video_id, self.frame_nr = None, None, None
        self.width, self.height = None, None
        self.bbox = []

        self.bmp_path = None
        self._image = None  # lazy loading

    @classmethod
    def from_csv(cls, bbox_rows: pd.DataFrame, dataset_path: str):
        """
        Constructs an XCAImage instance from CSV rows corresponding to a single image.
        """
        instance = cls()
        row = bbox_rows.iloc[0]
        instance.filename = row['filename']
        instance.width, instance.height = int(row['width']), int(row['height'])

        instance._parse_filename()

        instance.bmp_path = os.path.join(dataset_path, instance.filename)
        instance._image = None

        for _, row in bbox_rows.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            instance.bbox.append((xmin, ymin, xmax, ymax))
        return instance

    def _parse_filename(self) -> None:
        """
        Extracts patient_id, video_id, and frame_nr from the filename.
        Expected format: 14_<patient>_<video>_<frame>.bmp
        """
        pattern = r"^14_(\d+)_(\d+)_(\d+)\.bmp$"
        m = re.match(pattern, self.filename)
        if m:
            self.patient_id, self.video_id, self.frame_nr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            raise ValueError(f"filename {self.filename} does not match the expected pattern.")


    def get_image(self):
        """
        Lazy-loads and returns the image using cv2.imread."""
        if self._image is None:
            self._image = cv2.imread(self.bmp_path, cv2.IMREAD_UNCHANGED)
            if self._image is None:
                raise IOError(f"Could not read image file: {self.bmp_path}")
        return self._image

    def __repr__(self):
        return (f"XCAImage(patient={self.patient_id}, video={self.video_id}, frame={self.frame_nr}, "
                f"width={self.width}, height={self.height}, bbox(es)={len(self.bbox)})")




if __name__ == "__main__":
    reader = Reader()
    img = reader.get(patient_id=2)
    print(img)
    image = img.get_image()
