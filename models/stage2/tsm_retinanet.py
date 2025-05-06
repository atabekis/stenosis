# tsm_video_retinanet.py

# Python import - typing
from typing import Optional, Union

# Torch imports
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import boxes as box_ops
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.image_list import ImageList
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Local imports
from util import log
from config import NUM_CLASSES

from models.stage1.retinanet import FPNRetinaNet as BaselineRetinaNet  # to get the base configurations
from models.stage2_v2.tsm_backbone import tsm_efficientnet_b0




class TSMRetinaNet(nn.Module):
    """
    Main wrapper around the EfficientNet-B0 + TSM + RetinaNet architecture
    """

    # We pass these from the Stage 1 baseline model so that they are comparable under the same settings
    NUM_CLASSES = NUM_CLASSES
    FPN_OUT_CHANNELS = BaselineRetinaNet.FPN_OUT_CHANNELS
    ANCHOR_SIZES = BaselineRetinaNet.ANCHOR_SIZES
    ANCHOR_ASPECT_RATIOS = BaselineRetinaNet.ANCHOR_ASPECT_RATIOS

    SCORE_THRESH = BaselineRetinaNet.SCORE_THRESH
    NMS_THRESH = BaselineRetinaNet.NMS_THRESH

    FOCAL_LOSS_ALPHA = BaselineRetinaNet.FOCAL_LOSS_ALPHA
    FOCAL_LOSS_GAMMA = BaselineRetinaNet.FOCAL_LOSS_GAMMA

    DETECTIONS_PER_IMAGE_AFTER_NMS = BaselineRetinaNet.DETECTIONS_PER_IMAGE_AFTERNMS


    def __init__(
        self,
        # TSM parameters
        t_clip: int,
        shift_fraction: float = 0.125,
        shift_mode: str = 'residual',
        tsm_stages_indices: Optional[list[int]] = [3, 5, 6],
        pretrained_backbone: bool = True,

        # matcher parameters ---
        matcher_high_threshold: float = 0.5, # RetinaNet default
        matcher_low_threshold: float = 0.4,  # default
        matcher_allow_low_quality: bool = True, # default
    ):
        super().__init__()

        self.t_clip = t_clip

        # 1. TSM backbone
        tsm_effnet = tsm_efficientnet_b0(
            pretrained=pretrained_backbone, time_dim=t_clip, shift_fraction=shift_fraction,
            shift_mode=shift_mode, tsm_stages_indices=tsm_stages_indices, num_classes=self.NUM_CLASSES
        )
        return_layers = {'stage3': '0', 'stage5': '1', 'stage6': '2'}
        self.backbone = IntermediateLayerGetter(tsm_effnet.features, return_layers=return_layers)

        # 2. FPN
        fpn_in_channels_list = [40, 112, 192]
        self.fpn = FeaturePyramidNetwork(in_channels_list=fpn_in_channels_list, out_channels=self.FPN_OUT_CHANNELS)

        # 3. anchor generator
        self.anchor_generator = AnchorGenerator(sizes=self.ANCHOR_SIZES, aspect_ratios=self.ANCHOR_ASPECT_RATIOS)
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # 4. RetinaNet head
        self.head = RetinaNetHead(
            in_channels=self.fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )

        self.head.classification_head.focal_loss_alpha = self.FOCAL_LOSS_ALPHA
        self.head.classification_head.focal_loss_gamma = self.FOCAL_LOSS_GAMMA

        # 5. box coder and inference params
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.score_thresh = self.SCORE_THRESH
        self.nms_thresh = self.NMS_THRESH
        self.detections_per_img = self.DETECTIONS_PER_IMAGE_AFTER_NMS

        # 6. matcher
        self.proposal_matcher = Matcher(
            high_threshold=matcher_high_threshold,
            low_threshold=matcher_low_threshold,
            allow_low_quality_matches=matcher_allow_low_quality,
        )

    def forward(
        self,
        videos: Tensor,
        targets: Optional[list[list[dict[str, Tensor]]]] = None
    ) -> Union[dict[str, Tensor], list[dict[str, Tensor]]]:
        """
        Main forward pass of the model.
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        B, T, C, H, W = videos.shape
        if T != self.t_clip:
             log(f"Warning: Input video time dimension ({T}) does not match model expected time dimension ({self.t_clip}).")

        # get image size
        image_sizes_list = [(H, W)] * (B * T)
        videos_reshaped = videos.view(B * T, C, H, W)

        backbone_features = self.backbone(videos_reshaped)

        fpn_features = self.fpn(backbone_features)
        fpn_features_list = list(fpn_features.values())
        image_list = ImageList(videos_reshaped, image_sizes_list)
        anchors = self.anchor_generator(image_list, fpn_features_list) # List[Tensor] length B*T
        head_outputs = self.head(fpn_features_list) # Dict {'cls_logits', 'bbox_regression'}

        detections = []
        losses = {}

        if self.training:
            if targets is None: raise ValueError("targets should not be None in training")

            flat_targets = [targets[i][j] for i in range(B) for j in range(T)]  # list[list[dict]] -> list[dict] length B*T

            # anchor matching ---
            matched_idxs_list = []

            for i in range(B * T):
                gt_boxes_i = flat_targets[i]["boxes"] # gt for frame i
                anchors_i = anchors[i] # anchors for frame i

                if gt_boxes_i.numel() == 0:  # no gt boxes for this frame, mark all anchors as background
                    matched_idxs = torch.full((anchors_i.shape[0],), -1, dtype=torch.int64, device=anchors_i.device)

                else: # iou btw gt and anchors
                    match_quality_matrix = box_ops.box_iou(gt_boxes_i, anchors_i)
                    matched_idxs = self.proposal_matcher(match_quality_matrix)

                matched_idxs_list.append(matched_idxs)

            return self.head.compute_loss(flat_targets, head_outputs, anchors, matched_idxs_list)  # compute & return loss

        else:
            # inference mode
            detections = self._postprocess_detections(head_outputs, anchors, image_sizes_list)
            return detections


    @torch.no_grad()
    def _postprocess_detections(
        self,
        head_outputs: dict[str, Tensor],
        anchors: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:

        """ Postprocessing detections, since the one from torchvision refuses to work!"""
        class_logits = head_outputs["cls_logits"] # [B*T, num_anchors, num_classes]
        box_regression = head_outputs["bbox_regression"] # [B*T, num_anchors, 4]
        num_images = len(anchors) # B*T

        detections = []

        for i in range(num_images):
            box_regression_per_image = box_regression[i]
            logits_per_image = class_logits[i]
            anchors_per_image = anchors[i]
            image_shape = image_shapes[i]

            scores_per_image = torch.sigmoid(logits_per_image)

            pred_boxes_per_image = self.box_coder.decode_single(box_regression_per_image, anchors_per_image)
            pred_boxes_per_image = box_ops.clip_boxes_to_image(pred_boxes_per_image, image_shape)

            labels_per_image = torch.arange(self.num_classes, device=scores_per_image.device)
            labels_per_image = labels_per_image.view(1, -1).expand_as(scores_per_image)

            scores_per_image = scores_per_image.flatten()
            labels_per_image = labels_per_image.flatten()

            pred_boxes_per_image = pred_boxes_per_image.unsqueeze(1).expand(-1, self.num_classes, -1)
            pred_boxes_per_image = pred_boxes_per_image.reshape(-1, 4)

            inds_to_keep = torch.where((labels_per_image != 0) & (scores_per_image > self.score_thresh))[0]

            pred_boxes_per_image = pred_boxes_per_image[inds_to_keep]
            scores_per_image = scores_per_image[inds_to_keep]
            labels_per_image = labels_per_image[inds_to_keep]

            keep = box_ops.batched_nms(pred_boxes_per_image, scores_per_image, labels_per_image, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            final_boxes = pred_boxes_per_image[keep]
            final_scores = scores_per_image[keep]
            final_labels = labels_per_image[keep]

            detections.append({"boxes": final_boxes, "scores": final_scores, "labels": final_labels})

        return detections
