# backbone_v2.py

# Torch imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter

from functools import partial

from models.common.params_helper import get_adaptive_groupnorm_layer

from util import log

class FPNBackbone(nn.Module):
    """
    Creates an EfficientNet backbone, with a Feature Pyramid Network (FPN).

    Extracts features at strides and feeds them into the FPN.
    FPN produces P2, P3, P4, P5 (optional P2) feature maps.
    """
    def __init__(self,
                 variant: str = "b0",
                 out_channels: int = 256,
                 pretrained: bool = True,
                 include_p2: bool = False,
                 use_groupnorm: bool = True,
                 num_gn_groups: int = 32,
                 ):
        """
        :param variant: EfficientNet variant ('b0', 'v2_s', etc.)
        :param out_channels: Number of channels in the FPN output layers.
        :param pretrained: Whether to use pretrained ImageNet weights.
        :param include_p2: Whether to extract features for P2 level (stride 4).
        """
        super().__init__()
        self.variant = variant.lower()
        self.include_p2 = include_p2

        self.use_groupnorm = use_groupnorm
        self.num_gn_groups = num_gn_groups

        if self.use_groupnorm:
            log(f"Using Group Normalization instead of Batch Normalization. Num GN groups: {self.num_gn_groups}")
            norm_layer = get_adaptive_groupnorm_layer(default_num_gn_groups=num_gn_groups)
        else:
            norm_layer = nn.BatchNorm2d

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None


        if self.use_groupnorm and pretrained:
            effnet = models.efficientnet_b0(weights=None, norm_layer=norm_layer)

            state_dict_to_load = weights.get_state_dict(progress=True, check_hash=True)
            missing_keys, unexpected_keys = effnet.load_state_dict(state_dict_to_load, strict=False)

            if missing_keys: log(f"GN Load: Missing keys (expected for BN stats): {missing_keys[:5]}...")
            # if unexpected_keys: log(f"GN Load: Unexpected keys: {unexpected_keys[:5]}...")


        else:
            effnet = models.efficientnet_b0(weights=weights, norm_layer=norm_layer)  # use regular batchnorm


        base_model = effnet.features

        if self.include_p2:
            return_layers_map = {'2': '0', '3': '1', '5': '2', '6': '3'}
            in_channels_list_map = [24, 40, 112, 192]
        else:
            return_layers_map = {'3': '0', '5': '1', '6': '2'}
            in_channels_list_map = [40, 112, 192]

        self.body = IntermediateLayerGetter(base_model, return_layers=return_layers_map)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list_map,
            out_channels=out_channels,
        )
        self.out_channels = out_channels


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through Backbone and FPN.
        :param x: Input tensor of shape [B, 3, H, W].
        :returns: Dictionary of feature maps from FPN levels.
        """
        x_body = self.body(x)
        x_fpn = self.fpn(x_body)
        return x_fpn

