# thanos_detector.py

# Typing
from typing import Optional, Union

# Torch imports
import torch
import torch.nn as nn
from torchvision.ops import box_iou
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNetHead  # For the head structure
from torchvision.models.detection._utils import Matcher, BoxCoder  # For loss calculation, passed onto the common postproc.
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Local imports - models
from models.stage1.backbone_v2 import FPNBackbone
from models.stage3.thanos_utils import LearnablePositionalEmbeddings
from models.stage3.thanos_transformer import SpatioTemporalAttentionBlock

from models.common.params_helper import thanos_load_weights
from models.common.postprocess import postprocess_detections
from models.common.retinanet_utils import GNDropoutRetinaNetClassificationHead


# Logging
from util import log


class THANOS(nn.Module):
    """
    Temporal Hybrid Attentive Network for Stenosis (THANOS).
    Combines a CNN-FPN backbone with a SpatioTemporalAttentionBlock to enhance
    FPN features before feeding them to a RetinaNet-style detection head.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.fpn_out_channels = config.get("fpn_out_channels", 256)
        pretrained_backbone = config.get("pretrained_backbone", True)
        include_p2_fpn = config.get("include_p2_fpn", False)
        self.use_gradient_checkpointing = config.get("use_grad_ckpt", False)
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

        load_weights_ckpt_path = config.get("load_weights_from_ckpt", None)
        ckpt_model_key_prefix = config.get("ckpt_model_key_prefix", 'model.')

        backbone_use_gn = config.get("backbone_use_gn", False)
        backbone_num_gn_groups = config.get("backbone_num_gn_groups", 32)

        # Classification head specific parameters
        use_custom_classification_head = config.get("custom_head", False)
        classification_head_dropout_p = config.get("classification_head_dropout_p", 0.0)
        classification_head_num_convs = config.get("classification_head_num_convs", 4)
        classification_head_use_groupnorm = config.get("classification_head_use_groupnorm", False)
        classification_head_num_gn_groups = config.get("classification_head_num_gn_groups", 32)

        if load_weights_ckpt_path:  # disable imagenet weights if we give a checkpoint
            pretrained_backbone = False

        log("Initializing THANOS with parameters:")
        log(f"  Num classes: {self.num_classes}")
        log(f"  FPN out channels: {self.fpn_out_channels}")
        log(f"  Include P2 in FPN: {include_p2_fpn}")
        log(f"  Backbone GroupNorm: {backbone_use_gn}")
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
        log(f"  Gradient checkpointing: {self.use_gradient_checkpointing}")
        log(f"  Use Custom Classification Head: {use_custom_classification_head}")


        # set up the stage-1 backbone and FPN layers
        self.backbone = FPNBackbone(
            out_channels=self.fpn_out_channels,
            pretrained=pretrained_backbone,
            include_p2=include_p2_fpn,

            use_groupnorm=backbone_use_gn,
            num_gn_groups=backbone_num_gn_groups,
        )

        if include_p2_fpn:
            self.fpn_level_map = {'0': 'P2', '1': 'P3', '2': 'P4', '3': 'P5'}
        else:
            self.fpn_level_map = {'0': 'P3', '1': 'P4', '2': 'P5'}

        # 2. positional embeddings
        if transformer_d_model != self.fpn_out_channels:
            log(f"Warning: THANOS transformer_d_model ({transformer_d_model}) "
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
                dropout_rate=transformer_dropout_rate,
                use_gradient_checkpointing=self.use_gradient_checkpointing
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

        if use_custom_classification_head:
            self.head.classification_head = GNDropoutRetinaNetClassificationHead(
                in_channels=self.fpn_out_channels,
                num_anchors=num_anchors_per_loc[0],
                num_classes=self.num_classes,

                num_convs=classification_head_num_convs,
                dropout_p=classification_head_dropout_p,
                use_groupnorm=classification_head_use_groupnorm,
                num_gn_groups=classification_head_num_gn_groups,

                prior_probability=0.01,
            )

        self.head.classification_head.focal_loss_alpha = focal_loss_alpha
        self.head.classification_head.focal_loss_gamma = focal_loss_gamma


        # 5. box coder
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # 6. matcher
        self.proposal_matcher = Matcher(
            high_threshold=matcher_high_threshold,
            low_threshold=matcher_low_threshold,
            allow_low_quality_matches=matcher_allow_low_quality,
        )


        if load_weights_ckpt_path:
            thanos_load_weights(self, load_weights_ckpt_path, ckpt_model_key_prefix, load_head_weights=True)


    def _checkpointed_forward(self, module_to_ckpt: nn.Module, *args_for_module):
        """Helper to apply checkpointing to a given module's forward pass"""
        if self.use_gradient_checkpointing and self.training and not isinstance(module_to_ckpt, nn.Identity):
            return checkpoint(module_to_ckpt, *args_for_module, use_reentrant=False)
        else:
            return module_to_ckpt(*args_for_module)


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

        fpn_features_flat_dict = self._checkpointed_forward(self.backbone, videos_batch_flat)

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

        # 4. pass to detection head: {cls_logits: [BT, SumAnchorsPerFrame, NumClasses], bbox_reg: [BT, SumAnchorsPerFrame, 4]}
        head_outputs_dict = self._checkpointed_forward(self.head, features_for_head)

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
            image_shapes_flat = [(H_in, W_in)] * (B * T_clip)
            cls_logits = head_outputs_dict["cls_logits"]
            bbox_regression = head_outputs_dict["bbox_regression"]
            head_outputs_tuple = (cls_logits, bbox_regression)

            detections = postprocess_detections(
                head_outputs_tuple=head_outputs_tuple,
                anchors_per_frame_input=anchors_per_frame,
                image_shapes_flat=image_shapes_flat,
                box_coder=self.box_coder,
                num_classes=self.num_classes,
                score_thresh=self.inference_score_thresh,
                nms_thresh=self.inference_nms_thresh,
                detections_per_img=self.inference_detections_per_img,
            )

            return detections



if __name__ == "__main__":
    from config import STAGE3_THANOS_DEFAULT_CONFIG
    STAGE3_THANOS_DEFAULT_CONFIG['load_weights_from_ckpt'] = "../../.checkpoints/both.ckpt"
    STAGE3_THANOS_DEFAULT_CONFIG['custom_head'] = True
    model = THANOS(config=STAGE3_THANOS_DEFAULT_CONFIG)
