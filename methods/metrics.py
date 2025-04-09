# metrics.py

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion

from config import NUM_CLASSES

class Metrics:
    def __init__(self, box_format="xyxy", num_classes=NUM_CLASSES):
        self.map = MeanAveragePrecision(box_format=box_format, class_metrics=True)
        self.iou_metric = IntersectionOverUnion(box_format=box_format)

        self.num_classes = num_classes

    def update(self, preds, targets):
        """Update metrics with predictions and targets."""
        self.map.update(preds, targets)
        self.iou_metric.update(preds, targets)

    def compute(self):
        """Compute and return the metrics."""
        map_result = self.map.compute()
        iou_result = self.iou_metric.compute()

        precision = map_result['map']  # mean Average Precision
        recall = map_result['mar_100']  # mean Average Recall
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = iou_result['iou']

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'map': map_result['map'],
            'map_50': map_result['map_50'],
            'map_75': map_result['map_75'],
        }
        # print(results)
        return results

    def reset(self):
        """Reset all metrics."""
        self.map.reset()
        self.iou_metric.reset()