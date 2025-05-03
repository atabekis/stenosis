# tsm_retinanet.py

# Torch imports
import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet
from torchvision.ops import boxes as box_ops, batched_nms
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead

# Python imports
from itertools import chain
from typing import Optional, Union

# Local imports
from .tsm_backbone import TSMEfficientNetFPNBackbone

from util import log
from config import NUM_CLASSES, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, T_CLIP




class TSMRetinaNet(nn.Module):
    """
    Implements FPNRetinaNet with a TSM-EfficientNet-FPN backbone for object detection in temporal medium

    Processes video clips ([B, T, C, H, W]) by applying TSM in the backbone and running FPNRetinaNet heads on the temporal features
    """
    # same as retinanet_stage1
    NUM_CLASSES = NUM_CLASSES
    FPN_OUT_CHANNELS = 256
    ANCHOR_SIZES = ((16, 24, 32), (48, 64, 96), (128, 192, 256))
    ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
    SCORE_THRESH = 0.3
    NMS_THRESH = 0.4
    DETECTIONS_PER_IMAGE_AFTER_NMS = 20
    FOCAL_LOSS_ALPHA = FOCAL_LOSS_ALPHA
    FOCAL_LOSS_GAMMA = FOCAL_LOSS_GAMMA

    def __init__(
            self,
            t_clip: int = T_CLIP,
            tsm_div: int = 8,
            tsm_shift_mode: str = 'residual',
            pretrained_backbone: bool = True,
    ):
        """
        Initialize TSMRetinaNet model
        :param t_clip: number of frames processed together
        :param tsm_div: channel division factor for TSM
        :param tsm_shift_mode: 'residual' or 'inplace' for TSM
        :param pretrained_backbone: load pretrained backbone weights
        """
        super().__init__()

        self.t_clip = t_clip
        log("Initializing TSMRetinaNet with parameters:")
        log(f"  Num classes: {self.NUM_CLASSES}")
        log(f"  FPN out channels: {self.FPN_OUT_CHANNELS}")
        log(f"  TSM division: {tsm_div}")
        log(f"  TSM shift mode: {tsm_shift_mode}")
        log(f"  Backbone pretrained: {pretrained_backbone}")
        log(f"  Anchor sizes: {self.ANCHOR_SIZES}")
        log(f"  Anchor aspect ratios: {self.ANCHOR_ASPECT_RATIOS}")
        log(f"  Inference Score threshold: {self.SCORE_THRESH}")
        log(f"  Inference NMS threshold: {self.NMS_THRESH}")
        log(f"  Detections per frame: {self.DETECTIONS_PER_IMAGE_AFTER_NMS}")
        log(f"  Focal Loss alpha: {self.FOCAL_LOSS_ALPHA}")
        log(f"  Focal Loss gamma: {self.FOCAL_LOSS_GAMMA}")


        # 1. tsm backbone and fpn
        self.backbone = TSMEfficientNetFPNBackbone(
            n_segments=self.t_clip,
            out_channels=self.FPN_OUT_CHANNELS,
            tsm_div=tsm_div,
            tsm_shift_mode=tsm_shift_mode,
            pretrained=pretrained_backbone
        )

        # 2. anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=self.ANCHOR_SIZES,
            aspect_ratios=self.ANCHOR_ASPECT_RATIOS
        )

        # 3. retinanet heads (classification & regression)
        self.head = RetinaNetHead(
            in_channels=self.backbone.out_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
            num_classes=self.NUM_CLASSES
        )

        self.head.classification_head.focal_loss_alpha = self.FOCAL_LOSS_ALPHA
        self.head.classification_head.focal_loss_gamma = self.FOCAL_LOSS_GAMMA

        # 4. retinanet internals needed for loss/postprocess, we will 'borrow' params from a dummy retinanet instance
        _dummy_backbone = nn.Module()
        _dummy_backbone.out_channels = self.FPN_OUT_CHANNELS
        _dummy_retinanet = RetinaNet(
            backbone=_dummy_backbone,
            num_classes=self.NUM_CLASSES,
            anchor_generator=self.anchor_generator,
            score_thresh=self.SCORE_THRESH,
            nms_thresh=self.NMS_THRESH,
            detections_per_img=self.DETECTIONS_PER_IMAGE_AFTER_NMS
        )

        self.box_coder = _dummy_retinanet.box_coder
        self.compute_loss = _dummy_retinanet.compute_loss

        self.score_thresh = self.SCORE_THRESH
        self.nms_thresh = self.NMS_THRESH
        self.detections_per_img = self.DETECTIONS_PER_IMAGE_AFTER_NMS


    def set_inference_params(self, score_thresh, nms_thresh, detections_per_img):
        """Update parameters used during inference postprocessing"""
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img


    def forward(
            self,
            videos: torch.Tensor,
            targets: Optional[list[list[dict[str, torch.Tensor]]]] = None
    ) -> Union[dict[str, torch.Tensor], list[list[dict[str, torch.Tensor]]]]:
        """
        Main forward pass for TSMRetinaNet.
        :param videos: video clips of shape [B, T, C, H, W]
        :param targets: ground truth targets.
                        List of B elements, each element is a list of T dictionaries.
                        Each dictionary contains 'boxes' [N, 4] and 'labels' [N]
                        Required during training, optional during inference.
        :return:
            - Training: Dictionary of losses ('classification', 'bbox_regression').
            - Inference: List (B) of lists (T) of dictionaries. Each dict contains
                         'boxes', 'scores', 'labels' for detections in that frame.
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        B, T, C, H, W = videos.shape
        if T != self.t_clip:
            log(f"Warning: Input T dimension ({T}) != model t_clip ({self.t_clip}).")


        # 1. get features from tsm backbone
        features = self.backbone(videos)

        # 2. flatten batch & time dims
        feats_flat = {k: v.flatten(0, 1) for k, v in features.items()}  # [B*T, C', H', W']
        videos_flat = videos.flatten(0, 1)  # [B*T, C, H, W]

        # 3. generate anchors
        sizes_flat = [(H, W)] * (B * T)
        img_list = ImageList(videos_flat, sizes_flat)
        anchors = self.anchor_generator(img_list, list(feats_flat.values()))

        # 4. pass features thru retinanet head
        head_out = self.head(feats_flat)
        if self.training:
            targets_flat = list(chain.from_iterable(targets))
            return self.compute_loss(targets_flat, head_out, anchors)
        else: # inference
            cls_logits, bbox_reg = head_out['cls_logits'], head_out['bbox_regression']
            detections = []

            for i in range(B * T):
                # decode & clip
                props = self.box_coder.decode_single(bbox_reg[i], anchors[i])
                props = box_ops.clip_boxes_to_image(props, sizes_flat[i])

                # scores & threshold
                scores = cls_logits[i].sigmoid()  # [N_anchors, num_classes]
                mask = scores > self.score_thresh
                if not mask.any():
                    # no detections
                    detections.append({
                        "boxes": props.new_empty((0, 4)),
                        "scores": props.new_empty((0,)),
                        "labels": props.new_empty((0,), dtype=torch.int64)
                    })
                    continue

                # gather all (anchor_idx, class_idx) pairs above threshold
                idxs = mask.nonzero(as_tuple=False)  # [M, 2]
                anc_idx = idxs[:, 0]
                cls_idx = idxs[:, 1]
                scores_k = scores[anc_idx, cls_idx]
                boxes_k = props[anc_idx]

                keep = batched_nms(boxes_k, scores_k, cls_idx, self.nms_thresh)
                keep = keep[: self.detections_per_img]

                detections.append({
                    "boxes": boxes_k[keep],
                    "scores": scores_k[keep],
                    "labels": cls_idx[keep]
                })

            # 6. un-flatten: list[B*T] → list[B][T]
            return [
                detections[b * T: (b + 1) * T]
                for b in range(B)
            ]



class RetinaNetHead(nn.Module):
    """Implements the FPNRetinaNet classification and regression heads."""

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )

        self.regression_head = RetinaNetRegressionHead(
            in_channels, num_anchors
        )

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(x, dict):
            x = list(x.values())

        cls_logits = self.classification_head(x)
        bbox_regression = self.regression_head(x)

        return {
            'cls_logits': cls_logits,
            'bbox_regression': bbox_regression
        }