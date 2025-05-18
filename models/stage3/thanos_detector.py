# thanos_detector.py

# Torch imports
import torch
import torch.nn as nn
from torchvision.ops import box_iou, nms, clip_boxes_to_image
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNetHead  # For the head structure
from torchvision.models.detection._utils import Matcher, BoxCoder  # For loss calculation
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Typing
from typing import Optional, Union

# Local imports - models
from models.stage1.backbone import EfficientNetFPNBackbone
from models.stage3.thanos_utils import LearnablePositionalEmbeddings
from models.stage3.thanos_transformer import SpatioTemporalAttentionBlock

# Logging
from util import log


class THANOSDetector(nn.Module):
    """
    Temporal Hybrid Attentive Network for Stenosis (THANOS).
    Combines a CNN-FPN backbone with a SpatioTemporalAttentionBlock to enhance
    FPN features before feeding them to a RetinaNet-style detection head.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.fpn_out_channels = config.get("fpn_out_channels", 256)
        pretrained_backbone = config.get("pretrained_backbone", True)

        self.num_classes = config.get("num_classes")

        transformer_d_model = config.get("transformer_d_model", self.fpn_out_channels)
        transformer_n_head = config.get("transformer_n_head", 8)
        transformer_dim_feedforward = config.get("transformer_dim_feedforward", transformer_d_model * 4)
        transformer_num_spatial_layers = config.get("transformer_num_spatial_layers", 2)
        transformer_num_temporal_layers = config.get("transformer_num_temporal_layers", 2)
        transformer_dropout_rate = config.get("transformer_dropout_rate", 0.1)
        self.fpn_levels_to_process_temporally = config.get("fpn_levels_to_process_temporally", ['P3', 'P4', 'P5'])
        max_spatial_tokens_pe = config["max_spatial_tokens_pe"]
        max_temporal_tokens_pe = config["max_temporal_tokens_pe"]

        anchor_sizes = config["anchor_sizes"]
        anchor_aspect_ratios = config["anchor_aspect_ratios"]

        focal_loss_alpha = config.get("focal_loss_alpha")
        focal_loss_gamma = config.get("focal_loss_gamma")

        matcher_high_threshold = config.get("matcher_high_threshold", 0.5)
        matcher_low_threshold = config.get("matcher_low_threshold", 0.4)
        matcher_allow_low_quality = config.get("matcher_allow_low_quality", True)


        self.inference_score_thresh = config.get("inference_score_thresh", 0.3)
        self.inference_nms_thresh = config.get("inference_nms_thresh", 0.4)
        self.inference_detections_per_img = config.get("inference_detections_per_img")

        log("Initializing THANOSDetector with parameters:")
        log(f"  Num classes: {self.num_classes}")
        log(f"  FPN out channels: {self.fpn_out_channels}")
        log(f"  Anchor sizes: {anchor_sizes}")
        log(f"  Anchor aspect ratios: {anchor_aspect_ratios}")
        log(f"  Score threshold: {self.inference_score_thresh}")
        log(f"  NMS threshold: {self.inference_nms_thresh}")
        log(f"  Focal Loss alpha: {focal_loss_alpha}")
        log(f"  Focal Loss gamma: {focal_loss_gamma}")
        log(f"  Pretrained backbone: {pretrained_backbone}")
        log(f"  Detections per image: {self.inference_detections_per_img}")
        log(f"  Transformer: d_model={transformer_d_model}, n_head={transformer_n_head}, ff_dim={transformer_dim_feedforward}")
        log(f"  Transformer Layers: Spatial={transformer_num_spatial_layers}, Temporal={transformer_num_temporal_layers}")
        log(f"  FPN Levels for Temporal Processing: {self.fpn_levels_to_process_temporally}")
        log(f"  PE Max Tokens: Spatial={max_spatial_tokens_pe}, Temporal={max_temporal_tokens_pe}")

        # set up the stage-1 backbone and FPN layers
        self.backbone = EfficientNetFPNBackbone(out_channels=self.fpn_out_channels, pretrained=pretrained_backbone)
        self.fpn_level_map = {'0': 'P3', '1': 'P4', '2': 'P5'}
        # going to check:  '3': 'P6', 'pool': 'P7' if your FPN or head uses them, but backbone currently only provides P3, P4, P5 outputs via keys '0','1','2'.

        # 2. positional embeddings
        if transformer_d_model != self.fpn_out_channels:
            log(f"Warning: THANOSDetector transformer_d_model ({transformer_d_model}) "
                f"differs from fpn_out_channels ({self.fpn_out_channels}). "
                "Ensure this is intended or add projection layers.")

        self.positional_embeddings = LearnablePositionalEmbeddings(
            d_model=transformer_d_model,
            max_spatial_tokens=max_spatial_tokens_pe,
            max_temporal_tokens=max_temporal_tokens_pe
        )

        # 3. spatiotemporal attention blocks
        self.temporal_attention_blocks = nn.ModuleDict()
        for level_name_p_style in self.fpn_levels_to_process_temporally:
            if level_name_p_style not in self.fpn_level_map.values():  # p-style names
                raise ValueError(f"Invalid FPN level '{level_name_p_style}' in fpn_levels_to_process_temporally. "
                                 f"Valid P-style names from fpn_level_map: {list(self.fpn_level_map.values())}")

            self.temporal_attention_blocks[level_name_p_style] = SpatioTemporalAttentionBlock(
                d_model=transformer_d_model,
                n_head=transformer_n_head,
                dim_feedforward=transformer_dim_feedforward,
                num_spatial_layers=transformer_num_spatial_layers,
                num_temporal_layers=transformer_num_temporal_layers,
                dropout_rate=transformer_dropout_rate
            )

        # 4. detection head
        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios)

        num_anchors_per_loc = self.anchor_generator.num_anchors_per_location() # returns a list, one int per FPN level
        if not num_anchors_per_loc:
            raise ValueError("AnchorGenerator.num_anchors_per_location() returned empty list.")

        self.head = RetinaNetHead(
            in_channels=self.fpn_out_channels,
            num_anchors=num_anchors_per_loc[0],  # use first anchor from per FPN list
            num_classes=self.num_classes
        )
        self.head.classification_head.focal_loss_alpha = focal_loss_alpha
        self.head.classification_head.focal_loss_gamma = focal_loss_gamma

        # 5. loss calculation
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))  # standard
        self.proposal_matcher = Matcher(
            high_threshold=matcher_high_threshold,
            low_threshold=matcher_low_threshold,
            allow_low_quality_matches=matcher_allow_low_quality,
        )

    def forward(self,
                videos_batch: torch.Tensor,
                targets: Optional[list[list[dict[str, torch.Tensor]]]] = None
                ) -> Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """
        :param videos_batch: Input video clips [B, T, Cin, Hin, Win].
        :param targets: Ground truth targets.
            - Outer list length B, inner list length T; each dict contains 'boxes' and 'labels'. Required during training for loss calculation.
        :returns: If training, a dictionary of losses; otherwise, a list of B*T detection dictionaries.

        """
        B, T_clip, C_in, H_in, W_in = videos_batch.shape

        # 1. reshape for backbone and get FPN feats
        videos_batch_flat = videos_batch.view(B * T_clip, C_in, H_in, W_in)
        fpn_features_flat_dict = self.backbone(videos_batch_flat)

        # 2. apply spatiotemporal attn.
        enhanced_fpn_features_for_head_input = {}
        for fpn_key, p_level_name in self.fpn_level_map.items():
            if fpn_key not in fpn_features_flat_dict:
                continue

            Px_flat = fpn_features_flat_dict[fpn_key]
            _BT_shape, C_level, H_level, W_level = Px_flat.shape
            Ns_level = H_level * W_level

            if p_level_name in self.fpn_levels_to_process_temporally:
                Px_temporal_view = Px_flat.view(B, T_clip, C_level, H_level, W_level)
                Px_tokens_bt_ns_c = Px_temporal_view.flatten(start_dim=3, end_dim=4).permute(0, 1, 3, 2).contiguous()
                spatial_pe_add, temporal_pe_add = self.positional_embeddings(
                    B=B, T=T_clip, Ns=Ns_level, device=Px_tokens_bt_ns_c.device
                )
                tokens_with_pe = Px_tokens_bt_ns_c + spatial_pe_add + temporal_pe_add

                enhanced_tokens_bt_ns_c = self.temporal_attention_blocks[p_level_name](tokens_with_pe)

                enhanced_Px_flat = enhanced_tokens_bt_ns_c.permute(0, 1, 3, 2).contiguous().view(
                    B * T_clip, C_level, H_level, W_level
                )
                enhanced_fpn_features_for_head_input[fpn_key] = enhanced_Px_flat
            else:
                enhanced_fpn_features_for_head_input[fpn_key] = Px_flat

        features_for_head = [enhanced_fpn_features_for_head_input[k] for k in
                             sorted(enhanced_fpn_features_for_head_input.keys()) if
                             k in enhanced_fpn_features_for_head_input]

        if not features_for_head:
            if self.training:
                return {"classification": torch.tensor(0.0, device=videos_batch.device, requires_grad=True),
                        "bbox_regression": torch.tensor(0.0, device=videos_batch.device, requires_grad=True)}
            else:
                return [{"boxes": torch.empty((0, 4), device=videos_batch.device),
                         "scores": torch.empty((0,), device=videos_batch.device),
                         "labels": torch.empty((0,), dtype=torch.int64, device=videos_batch.device)}
                        ] * (B * T_clip)

        # 3. generate anchors
        image_sizes_for_imagelist = [(H_in, W_in)] * (B * T_clip)
        image_list_for_anchors = ImageList(videos_batch_flat, image_sizes_for_imagelist)
        anchors_per_frame = self.anchor_generator(image_list_for_anchors, features_for_head)

        # 4. pass to detection head ---
        head_outputs_dict = self.head(features_for_head) # {cls_logits: [BT, SumAnchorsPerFrame, NumClasses], bbox_reg: [BT, SumAnchorsPerFrame, 4]}

        # 5. compute loss (if training) or postprocess (inference)
        if self.training:
            if targets is None:
                raise ValueError("Targets are required for training.")

            device = features_for_head[0].device
            targets_flat = []
            for video_targets_per_b in targets:
                for frame_target_per_t in video_targets_per_b:
                    targets_flat.append({k: v.to(device) for k, v in frame_target_per_t.items()})

            matched_idxs_list = []
            for i in range(B * T_clip):
                gt_boxes_frame_i = targets_flat[i]["boxes"]
                anchors_frame_i = anchors_per_frame[i]

                if gt_boxes_frame_i.numel() == 0:
                    matched_idxs_list.append(
                        torch.full_like(anchors_frame_i[:, 0], -1, dtype=torch.long, device=device)
                    )
                else:
                    match_quality_matrix = box_iou(gt_boxes_frame_i, anchors_frame_i)
                    matched_idxs_list.append(
                        self.proposal_matcher(match_quality_matrix)
                    )

            losses_from_head = self.head.compute_loss(
                targets_flat,
                head_outputs_dict,
                anchors_per_frame,
                matched_idxs_list
            )

            return {
                "classification": losses_from_head["classification"],
                "bbox_regression": losses_from_head["bbox_regression"],
            }
        else:  # inference
            image_shapes_flat_for_postprocess = [(H_in, W_in)] * (B * T_clip)

            detections = self.postprocess_detections(
                (head_outputs_dict['cls_logits'], head_outputs_dict['bbox_regression']),
                anchors_per_frame,
                image_shapes_flat_for_postprocess
            )
            return detections


    def postprocess_detections(self, head_outputs_tuple, anchors_per_frame_input: list[torch.Tensor],
                               image_shapes_flat):
        """
        Postprocesses detections: applies score threshold, NMS, limits detections.
        :param head_outputs_tuple:
            - cls_logits shape [B*T_clip, NumTotalAnchorsPerFrame, NumClasses]
            - bbox_regression shape [B*T_clip, NumTotalAnchorsPerFrame, 4].
        :param anchors_per_frame_input: List (length B*T_clip) of anchor tensors, each of shape [NumTotalAnchorsForFrame, 4].
        :param image_shapes_flat: Original height and width for each frame.
        """
        cls_logits_bt_na_nc, bbox_regression_bt_na_4 = head_outputs_tuple

        flat_anchors_per_frame = anchors_per_frame_input

        detections = []
        for i in range(cls_logits_bt_na_nc.size(0)):  # B*T_clip times
            pred_logits_this_image = cls_logits_bt_na_nc[i]  # [NumTotalAnchorsPerFrame, NumClasses]
            pred_regression_this_image = bbox_regression_bt_na_4[i]  # [NumTotalAnchorsPerFrame, 4]
            anchors_this_image = flat_anchors_per_frame[i]  # [NumTotalAnchorsPerFrame, 4]
            image_shape = image_shapes_flat[i]  # (H, W)

            boxes = self.box_coder.decode_single(pred_regression_this_image, anchors_this_image)
            scores_all_classes = torch.sigmoid(pred_logits_this_image)  # [N_anchors_this_image, num_classes]

            top_scores_per_anchor, top_labels_per_anchor = scores_all_classes.max(dim=1)

            keep_idxs = top_scores_per_anchor >= self.inference_score_thresh

            boxes = boxes[keep_idxs]
            final_scores = top_scores_per_anchor[keep_idxs]
            final_labels = top_labels_per_anchor[keep_idxs]

            non_bg_keep_idxs = final_labels != 0
            boxes = boxes[non_bg_keep_idxs]
            final_scores = final_scores[non_bg_keep_idxs]
            final_labels = final_labels[non_bg_keep_idxs]

            if boxes.numel() == 0:
                detections.append({
                    "boxes": torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype),
                    "scores": torch.empty((0,), device=final_scores.device, dtype=final_scores.dtype),
                    "labels": torch.empty((0,), device=final_labels.device, dtype=final_labels.dtype),
                })
                continue

            keep_after_nms = nms(boxes, final_scores, self.inference_nms_thresh)

            keep_after_nms = keep_after_nms[:self.inference_detections_per_img]

            boxes = boxes[keep_after_nms]
            final_scores = final_scores[keep_after_nms]
            final_labels = final_labels[keep_after_nms]

            current_image_size_tensor = torch.tensor(image_shape, device=boxes.device, dtype=torch.float32)
            boxes = clip_boxes_to_image(boxes, current_image_size_tensor)

            detections.append({
                "boxes": boxes,
                "scores": final_scores,
                "labels": final_labels,
            })
        return detections

