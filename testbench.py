import argparse
import os
from config import CADICA_DATASET_DIR
from methods.augment import DummyAugment
from models.faster_rcnn import FasterRCNN
from methods.pipeline import StenosisPipeline
from util import log

def main():
    pipeline = StenosisPipeline(
        dataset_dir=CADICA_DATASET_DIR,
        model=FasterRCNN(),
        augmentor=DummyAugment(),
    )
    results = pipeline.run(verbosity=-1, use_pbar=True)
    print(results)


if __name__ == '__main__':
    main()
