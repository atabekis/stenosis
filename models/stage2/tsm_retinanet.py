# tsm_retinanet.py

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


from models.stage2.tsm_backbone import tsm_efficientnet_b0
from models.stage2.tsm_load_weights import transfer_to_tsm_retinanet

from models.common.postprocess import postprocess_detections
from models.common.retinanet_utils import GNDropoutRetinaNetClassificationHead



class TSMRetinaNet(nn.Module):
    """
    Main wrapper around the EfficientNet-B0 + TSM + RetinaNet architecture
    """

    def __init__(self, config: dict):
        super().__init__()

        self.num_classes = config["num_classes"]
        self.score_thresh = config["inference_score_thresh"]
        self.nms_thresh = config["inference_nms_thresh"]
        self.detections_per_img = config["inference_detections_per_img"]
        fpn_out_channels = config["fpn_out_channels"]
        anchor_sizes = config["anchor_sizes"]
        anchor_aspect_ratios = config["anchor_aspect_ratios"]
        focal_loss_alpha = config["focal_loss_alpha"]
        focal_loss_gamma = config["focal_loss_gamma"]
        pretrained_backbone = config.get("pretrained_backbone", True)
        use_gradient_checkpointing = config.get("use_grad_ckpt", False)

        include_p2_fpn = config.get("include_p2_fpn", False)

        self.t_clip = config["t_clip"]
        shift_fraction = config.get("tsm_shift_fraction", 0.125)
        shift_mode = config.get("tsm_shift_mode", 'residual')
        tsm_effnet_stages = config.get("tsm_effnet_stages_for_tsm", [3, 5, 6])

        matcher_high_threshold = config.get("matcher_high_threshold", 0.5)
        matcher_low_threshold = config.get("matcher_low_threshold", 0.4)
        matcher_allow_low_quality = config.get("matcher_allow_low_quality", True)

        use_custom_classification_head = config.get("custom_head", False)
        classification_head_dropout_p = config.get("classification_head_dropout_p", 0.0)
        classification_head_num_convs = config.get("classification_head_num_convs", 4)
        classification_head_use_groupnorm = config.get("classification_head_use_groupnorm", False)
        classification_head_num_gn_groups = config.get("classification_head_num_gn_groups", 32)

        load_weights_ckpt_path = config.get("load_weights_from_ckpt", None)
        ckpt_model_key_prefix = config.get("ckpt_model_key_prefix", 'model.')

        freeze_bn_in_backbone = config.get("freeze_bn_in_backbone", True) # Add a config option

        if load_weights_ckpt_path:
            pretrained_backbone = False

        self.verbose = config.get('verbose', True)
        if self.verbose:
            log("Initializing TSMRetinaNet with parameters:")
            log(f"  Num classes: {self.num_classes}")
            log(f"  FPN out channels: {fpn_out_channels}")
            log(f"  Include P2 in FPN: {include_p2_fpn}")
            log(f"  Anchor sizes: {anchor_sizes}")
            log(f"  Anchor aspect ratios: {anchor_aspect_ratios}")
            log(f"  Score threshold: {self.score_thresh}")
            log(f"  NMS threshold: {self.nms_thresh}")
            log(f"  Focal Loss alpha: {focal_loss_alpha}")
            log(f"  Focal Loss gamma: {focal_loss_gamma}")
            log(f"  Pretrained backbone: {pretrained_backbone}")
            log(f"  Freeze backbone BN: {freeze_bn_in_backbone and load_weights_ckpt_path is not None}")
            log(f"  Detections per image: {self.detections_per_img}")
            log(f"  Inserting TSM to internal stages {tsm_effnet_stages}")
            log(f"  Gradient checkpointing: {use_gradient_checkpointing}")
            log(f"  Use Custom Classification Head: {use_custom_classification_head}")

        # 1. TSM backbone
        self.backbone = tsm_efficientnet_b0(
            pretrained=pretrained_backbone,
            time_dim=self.t_clip,
            shift_fraction=shift_fraction,
            shift_mode=shift_mode,
            tsm_stages_indices=tsm_effnet_stages,
            num_classes=self.num_classes,
            use_gradient_checkpoint=use_gradient_checkpointing,

            freeze_bn=freeze_bn_in_backbone
        )

        if include_p2_fpn:
            return_layers = {'stage2': '0', 'stage3': '1', 'stage5': '2', 'stage6': '3'} # p2, p3, p4, p5
            fpn_in_channels_list = [24, 40, 112, 192]
        else:
            return_layers = {'stage3': '0', 'stage5': '1', 'stage6': '2'} #p3, p4, p5
            fpn_in_channels_list = [40, 112, 192]


        self.fpn_feature_extractor = IntermediateLayerGetter(self.backbone.features, return_layers=return_layers)

        # 2. FPN
        self.fpn = FeaturePyramidNetwork(in_channels_list=fpn_in_channels_list, out_channels=fpn_out_channels)

        # 3. anchor generator
        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios)
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # 4. RetinaNet head - we have both default and the custom head with dropout and group norm

        self.head = RetinaNetHead(
            in_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )


        if use_custom_classification_head:
            self.head.classification_head = GNDropoutRetinaNetClassificationHead(
                in_channels=fpn_out_channels,
                num_anchors=num_anchors,
                num_classes=self.num_classes,

                num_convs=classification_head_num_convs,
                dropout_p=classification_head_dropout_p,
                use_groupnorm=classification_head_use_groupnorm,
                num_gn_groups=classification_head_num_gn_groups,

                prior_probability=0.01,

                use_grad_ckpt=use_gradient_checkpointing,

                verbose=self.verbose
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
            log(f'Loading weights from checkpoint: {load_weights_ckpt_path}')
            transfer_to_tsm_retinanet(self, load_weights_ckpt_path, ckpt_model_key_prefix)

        if freeze_bn_in_backbone:
            self.backbone.freeze_backbone_layers()


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

        backbone_features = self.fpn_feature_extractor(videos_reshaped)

        fpn_features = self.fpn(backbone_features)
        fpn_features_list = list(fpn_features.values())
        image_list = ImageList(videos_reshaped, image_sizes_list)
        anchors = self.anchor_generator(image_list, fpn_features_list) # List[Tensor] length B*T
        head_outputs = self.head(fpn_features_list) # Dict {'cls_logits', 'bbox_regression'}


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
            cls_logits = head_outputs["cls_logits"]
            bbox_regression = head_outputs["bbox_regression"]
            head_outputs_tuple = (cls_logits, bbox_regression)

            detections = postprocess_detections(
                head_outputs_tuple=head_outputs_tuple,
                anchors_per_frame_input=anchors,
                image_shapes_flat=image_sizes_list,
                box_coder=self.box_coder,
                num_classes=self.num_classes,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh,
                detections_per_img=self.detections_per_img
            )
            return detections