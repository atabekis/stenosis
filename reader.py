# reader.py

# Python imports
import os
import re
import cv2
import concurrent.futures
import xml.etree.ElementTree as ET

# Local imports
from util import log
from config import DATASET_DIR, DATASET_PATH


class Reader:
    """
    Reads the dataset directory, finds all .bmp + .xml pairs, and constructs XCAImage objects
    """
    def __init__(self, dataset_path=DATASET_PATH) -> None:
        self.dataset_path = dataset_path
        self.xca_images = []

        # for use outside the class, we will keep information on the dataset with the following
        self.bmp_files, self.xml_files = [], []
        self.bmp_basenames, self.xml_basenames = {}, {}

        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        Gather all BMP files, match the XML file
        """
        self.bmp_files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith(".bmp")]
        self.xml_files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith(".xml")]

        self.bmp_basenames = {os.path.splitext(f)[0] for f in self.bmp_files}
        self.xml_basenames = {os.path.splitext(f)[0] for f in self.xml_files}

        matching = sorted(self.bmp_basenames & self.xml_basenames)  # all files are matching in the dataset, but, redundancy


        def build_xca_image(base_name: str) -> XCAImage:
            """Helper to build an XCAImage object, used by concurrent pool"""
            bmp_path = os.path.join(self.dataset_path, base_name + ".bmp")
            xml_path = os.path.join(self.dataset_path, base_name + ".xml")
            return XCAImage(bmp_path, xml_path)

        log("Building XCA images from the dataset")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(build_xca_image, matching))

        self.xca_images = results


    def __repr__(self):
        return f"Reader(dataset_path='{self.dataset_path}', total_images={len(self.xca_images)})"


class XCAImage:
    """
    Represents a single coronary angiography image
    """
    def __init__(self, bmp_path: str, xml_path: str):
        """
        Initialize an XCAImage instance
        """
        self.bmp_path = bmp_path
        self.xml_path = xml_path

        self.patient_id, self.video_id, self.frame_nr = None, None, None
        self.width, self.height = None, None
        self.bbox = []

        self._parse_filename()
        self._parse_image_size()
        self._parse_xml_annotation()

    def _parse_filename(self) -> None:
        """
        Extracts patient_id, video_id, and frame_nr from the BMP filename.
        e.g, 14_<patient>_<video>_<frame>.bmp
        """
        filename = os.path.basename(self.bmp_path)

        pattern = r"^14_(\d+)_(\d+)_(\d+)\.bmp$"
        m = re.match(pattern, filename)
        if m:
            self.patient_id, self.video_id, self.frame_nr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            raise ValueError(f"filename {filename} does not match the expected pattern.")


    def _parse_image_size(self) -> None:
        """
        Determine the image dimensions in pixels.
        """
        img = cv2.imread(self.bmp_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read image file: {self.bmp_path}")
        self.height, self.width = img.shape[:2]

    def _parse_xml_annotation(self) -> None:
        """
        Parses the XML annotation associated with the BMP image file, which includes the bounding box for the image
        There exists only one file with two bounding boxes, that file is still counted and the two bounding boxes are
        considered in the analysis.
        """

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Confirm that the XML size matches the actual image size (redundancy!)
        size_node = root.find('size')
        if size_node is not None:
            width, height = int(size_node.find('width').text), int(size_node.find('height').text)

            if width != self.width or height != self.height:
                log(f"Image {self.bmp_path} has different width/height then reported in the XML file "
                    f"([{self.width}x{self.height}] != [{width}x{height}])")


        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin, ymin, xmax, ymax = (int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                          int(bndbox.find('xmax').text), int(bndbox.find('ymax').text))
                box = {xmin, ymin, xmax, ymax}
                self.bbox.append(box)

    def __repr__(self):
        return (f"XCAImage(patient={self.patient_id}, video={self.video_id}, frame={self.frame_nr}, "
                f"width={self.width}, height={self.height}, bbox(es)={len(self.bbox)})")



if __name__ == "__main__":
    reader = Reader()
    print(reader)