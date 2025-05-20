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
from models.common.retinanet_utils import GNDropoutRetinaNetClassificationHead

# Local imports
from util import log


class FPNRetinaNet(nn.Module):
    """
    Implements the stage 1 FPNRetinaNet model for single-frame detection
    Uses the efficient net + FPN backbone and standard FPNRetinaNet heads.
    """


    def __init__(self, config: dict):
        """
         Initialize FPNRetinaNet model
        """
        super().__init__()


        num_classes = config.get("num_classes")
        fpn_out_channels = config.get("fpn_out_channels", 256)
        anchor_sizes = config["anchor_sizes"]
        anchor_aspect_ratios = config["anchor_aspect_ratios"]
        score_thresh = config.get("inference_score_thresh", 0.3)
        nms_thresh = config.get("inference_nms_thresh", 0.4)
        focal_loss_alpha = config.get("focal_loss_alpha")
        focal_loss_gamma = config.get("focal_loss_gamma")
        pretrained_backbone = config.get("pretrained_backbone", True)
        detections_per_img = config.get("inference_detections_per_img")

        use_custom_classification_head = config.get("custom_head", False)
        classification_head_dropout_p = config.get("classification_head_dropout_p", 0.0)
        classification_head_num_convs = config.get("classification_head_num_convs", 4)
        classification_head_use_groupnorm = config.get("classification_head_use_groupnorm", False)
        classification_head_num_gn_groups = config.get("classification_head_num_gn_groups", 32)



        log("Initializing FPNRetinaNet with parameters:")
        log(f"  Num classes: {num_classes}")
        log(f"  FPN out channels: {fpn_out_channels}")
        log(f"  Anchor sizes: {anchor_sizes}")
        log(f"  Anchor aspect ratios: {anchor_aspect_ratios}")
        log(f"  Score threshold: {score_thresh}")
        log(f"  NMS threshold: {nms_thresh}")
        log(f"  Focal Loss alpha: {focal_loss_alpha}")
        log(f"  Focal Loss gamma: {focal_loss_gamma}")
        log(f"  Pretrained backbone: {pretrained_backbone}")
        log(f"  Detections per image: {detections_per_img}")
        log(f"  Use Custom Classification Head: {use_custom_classification_head}")

        # 1. backbone & fpn
        self.backbone = EfficientNetFPNBackbone(
            out_channels=fpn_out_channels,
            pretrained=pretrained_backbone,
        )

        # 2. anchor generator
        self.anchor_generator = AnchorGenerator(
            anchor_sizes,
            aspect_ratios=anchor_aspect_ratios,
        )

        # 3. FPNRetinaNet model
        self.retinanet = RetinaNet(
            backbone=self.backbone,
            num_classes=num_classes,
            anchor_generator=self.anchor_generator,
            score_threshold=score_thresh,
            nms_threshold=nms_thresh,
            detections_per_img=detections_per_img,
        )

        if use_custom_classification_head:
            self.retinanet.head.classification_head = GNDropoutRetinaNetClassificationHead(
                in_channels=fpn_out_channels,
                num_anchors=self.anchor_generator.num_anchors_per_location()[0],
                num_classes=num_classes,

                num_convs=classification_head_num_convs,
                dropout_p=classification_head_dropout_p,
                use_groupnorm=classification_head_use_groupnorm,
                num_gn_groups=classification_head_num_gn_groups,

                prior_probability=0.01,
            )


        self.retinanet.head.classification_head.focal_loss_alpha = focal_loss_alpha
        self.retinanet.head.classification_head.focal_loss_gamma = focal_loss_gamma

    def forward(self,
            images: Union[list[torch.Tensor], torch.Tensor],
            targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Main forward pass"""

        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        return self.retinanet(images, targets)