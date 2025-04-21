# faster_rcnn.py

# Python imports
from typing import Dict, List, Optional

# Torch imports
import torch
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from util import log
from config import NUM_CLASSES


class FasterRCNN(nn.Module):
    """
    Faster R-CNN model with ResNet-50 backbone and Feature Pyramid Network (FPN)
    """

    def __init__(
            self,
            num_classes: int = NUM_CLASSES,
            pretrained: bool = True,
            trainable_backbone_layers: int = 3,
            min_size: int = 1024,
            max_size: int = 1536,
            image_mean: Optional[List[float]] = None,
            image_std: Optional[List[float]] = None,
    ):
        """
        Initialize Faster R-CNN model
        :param num_classes: number of classes (including background)
        :param pretrained: whether to use pretrained weights
        :param trainable_backbone_layers: number of trainable backbone layers
        :param min_size: minimum size of the image to be rescaled
        :param max_size: maximum size of the image to be rescaled
        :param image_mean: mean values for normalization
        :param image_std: std values for normalization
        """
        super().__init__()

        # default normalization values if not provided
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]


        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
        )

        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # smaller anchor sizes since i have very small bboxes
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        self.model.rpn.anchor_generator = anchor_generator

        # increase proposals to ensure small objects are captured
        self.model.rpn.pre_nms_top_n_train = 3000
        self.model.rpn.post_nms_top_n_train = 1500
        self.model.rpn.pre_nms_top_n_test = 1500
        self.model.rpn.post_nms_top_n_test = 1000

        # adjust nms threshold - better handling of small objects
        self.model.rpn.nms_thresh = 0.75

        self.model.roi_heads.score_thresh = 0.01  # very low threshold to ensure detection
        self.model.roi_heads.nms_thresh = 0.3  # higher nms to avoid duplicate predictions
        self.model.roi_heads.detections_per_img = 1  # only one object per image

        self.model.rpn.fg_bg_sampler.positive_fraction = 0.8 # increase positive fraction to focus more on the small object

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        log(f"Initialized Faster R-CNN model with {num_classes} classes")
        log(f"Trainable backbone layers: {trainable_backbone_layers}")

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Forward pass of the model
        :param images: list of image tensors [B, 3, H, W]
        :param targets: list of target dictionaries with 'boxes' and 'labels' keys
        :return: loss dict if targets are provided, else detections
        """
        if self.training:
            assert targets is not None, "Targets must be provided in training mode"
            return self.model(images, targets)  # Returns loss dictionary
        else:
            return self.model(images)  # Returns predictions only
