# retinanet.py

# Torch imports
import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Python import
from typing import Union, Optional

# Backbone model
from models.stage1.backbone import EfficientNetFPNBackbone

# Local imports
from util import log
from config import (
    NUM_CLASSES,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
)

class FPNRetinaNet(nn.Module):
    """
    Implements the stage 1 FPNRetinaNet model for single-frame detection
    Uses the efficient net + FPN backbone and standard FPNRetinaNet heads.
    """

    NUM_CLASSES = NUM_CLASSES
    FPN_OUT_CHANNELS = 256
    ANCHOR_SIZES = (
        (8, 11, 16),  #  smaller objects
        (22, 32, 45),  # medium objects
        (64, 90, 128)  # larger objects
    )
    ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)

    # inference params
    SCORE_THRESH = 0.3
    NMS_THRESH = 0.4

    FOCAL_LOSS_ALPHA = FOCAL_LOSS_ALPHA
    FOCAL_LOSS_GAMMA = FOCAL_LOSS_GAMMA

    DETECTIONS_PER_IMAGE_AFTER_NMS = 1

    def __init__(
            self,
            pretrained: bool = True,
    ):
        """
         Initialize FPNRetinaNet model
        """
        super().__init__()

        log("Initializing FPNRetinaNet with parameters:")
        log(f"  Num classes: {self.NUM_CLASSES}")
        log(f"  FPN out channels: {self.FPN_OUT_CHANNELS}")
        log(f"  Anchor sizes: {self.ANCHOR_SIZES}")
        log(f"  Anchor aspect ratios: {self.ANCHOR_ASPECT_RATIOS}")
        log(f"  Score threshold: {self.SCORE_THRESH}")
        log(f"  NMS threshold: {self.NMS_THRESH}")
        log(f"  Focal Loss alpha: {self.FOCAL_LOSS_ALPHA}")
        log(f"  Focal Loss gamma: {self.FOCAL_LOSS_GAMMA}")
        log(f"  Pretrained backbone: {pretrained}")
        log(f"  Detections per image: {self.DETECTIONS_PER_IMAGE_AFTER_NMS}")

        # 1. backbone & fpn
        self.backbone = EfficientNetFPNBackbone(
            out_channels=self.FPN_OUT_CHANNELS,
            pretrained=pretrained,
        )

        # 2. anchor generator
        self.anchor_generator = AnchorGenerator(
            self.ANCHOR_SIZES,
            aspect_ratios=self.ANCHOR_ASPECT_RATIOS,
        )

        # 3. FPNRetinaNet model
        self.retinanet = RetinaNet(
            backbone=self.backbone,
            num_classes=self.NUM_CLASSES,
            anchor_generator=self.anchor_generator,
            score_threshold=self.SCORE_THRESH,
            nms_threshold=self.NMS_THRESH,
            detections_per_img=self.DETECTIONS_PER_IMAGE_AFTER_NMS,
        )


        self.retinanet.head.classification_head.focal_loss_alpha = self.FOCAL_LOSS_ALPHA
        self.retinanet.head.classification_head.focal_loss_gamma = self.FOCAL_LOSS_GAMMA

    def forward(self,
            images: Union[list[torch.Tensor], torch.Tensor],
            targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Main forward pass"""

        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        return self.retinanet(images, targets)

