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
        _dummy_retinanet = self._get_dummy_torchvision_retinanet()
        self.box_coder = _dummy_retinanet.box_coder
        self.compute_loss = _dummy_retinanet.compute_loss

        self.score_thresh = _dummy_retinanet.score_thresh
        self.nms_thresh = _dummy_retinanet.nms_thresh
        self.detections_per_img = _dummy_retinanet.detections_per_img

        self.postprocess_detections = _dummy_retinanet.postprocess_detections


    def _get_dummy_torchvision_retinanet(self) -> RetinaNet:
        """Creates a minimal RetinaNet instance to borrow methods/attributes."""
        _dummy_backbone = nn.Module()
        _dummy_backbone.out_channels = self.FPN_OUT_CHANNELS
        return RetinaNet(
            backbone=_dummy_backbone,
            num_classes=self.NUM_CLASSES,
            anchor_generator=self.anchor_generator,
            score_thresh=self.SCORE_THRESH,
            nms_thresh=self.NMS_THRESH,
            detections_per_img=self.DETECTIONS_PER_IMAGE_AFTER_NMS,
        )


    def set_inference_params(self, score_thresh: float, nms_thresh: float, detections_per_img: int):
        """Update parameters used during inference postprocessing."""
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


        # 1. get features from tsm backbone & feature prep
        features = self.backbone(videos)

        # 2. prep inputs for head and anchors
        feature_maps = list(features.values())

        videos_flat = videos.contiguous().view(-1, C, H, W)  # [B*T, C, H, W]
        images_list = ImageList(videos_flat, [(H, W)] * (B * T))

        # 3. generate anchors
        anchors = self.anchor_generator(images_list, feature_maps)

        # 4. pass through retinanet head
        head_out = self.head(feature_maps) # {cls_logits: tensor[B*T, A, K], bbox_regression: tensor[B*T, A, 4]}

        # 5. compute loss
        detections: Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]

        if self.training:
            # flatten targets (B lists of length T) → flat list of len B*T
            device = videos.device
            flat_targets = [
                {k: v.to(device) for k, v in t.items()}
                if isinstance(t, dict) and 'boxes' in t and 'labels' in t
                else {
                    'boxes': torch.empty((0, 4), device=device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=device)
                }
                for t in chain.from_iterable(targets)
            ]
            return self.compute_loss(flat_targets, head_out, anchors)

        else:  # inference
            cls_logits = head_out['cls_logits']  # [B*T, A, K]
            bbox_reg = head_out['bbox_regression']  # [B*T, A, 4]

            detections = []
            # pre‐compute class labels tensor for K-1 real classes
            device = cls_logits.device
            num_classes = cls_logits.size(-1)
            class_ids = torch.arange(1, num_classes, device=device)

            for logits_i, reg_i, anchors_i, img_size in zip(
                    cls_logits, bbox_reg, anchors, images_list.image_sizes
            ):
                # decode & clip boxes
                boxes = self.box_coder.decode_single(reg_i, anchors_i)
                boxes = box_ops.clip_boxes_to_image(boxes, img_size)

                # compute scores and drop background class
                scores = logits_i.sigmoid()[:, 1:]  # [A, K-1]
                # expand boxes to match each class
                A = boxes.size(0)
                boxes = boxes[:, None, :].expand(A, num_classes - 1, 4).reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = class_ids.repeat_interleave(A)  # [A*(K-1)]

                # threshold + remove tiny boxes
                keep = scores > self.score_thresh
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                # batched NMS & top-k
                keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                detections.append({
                    'boxes': boxes[keep],
                    'scores': scores[keep],
                    'labels': labels[keep],
                })

            # reshape to [B][T]
            return [
                detections[b * T:(b + 1) * T]
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


    def forward(self, x: Union[list[torch.Tensor], dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]: # Accept dict or list
        """
        Processes features from FPN levels.
        :param x: List of tensors (one per FPN level) or Dict mapping level name to tensor.
        :return: Dictionary containing 'cls_logits' and 'bbox_regression' tensors,
                 where predictions from all levels are concatenated along the anchor dimension.
        """
        if isinstance(x, dict):
            x = [x[k] for k in sorted(x, key=int)]
        elif not isinstance(x, list):
            raise TypeError(f"Expected list or dict for features, got {type(x)}")

        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x),
        }

