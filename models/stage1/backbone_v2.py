# backbone_v2.py

# Torch imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter

class EfficientNetFPNBackbone(nn.Module):
    """
    Creates an EfficientNet backbone, with a Feature Pyramid Network (FPN).

    Extracts features at strides and feeds them into the FPN.
    FPN produces P2, P3, P4, P5 (optional P2) feature maps.
    """
    def __init__(self,
                 variant: str = "b0",
                 out_channels: int = 256,
                 pretrained: bool = True,
                 include_p2: bool = False):
        """
        :param variant: EfficientNet variant ('b0', 'v2_s', etc.)
        :param out_channels: Number of channels in the FPN output layers.
        :param pretrained: Whether to use pretrained ImageNet weights.
        :param include_p2: Whether to extract features for P2 level (stride 4).
        """
        super().__init__()
        self.variant = variant.lower()
        self.include_p2 = include_p2

        # effnet.features[0] is the stem.
        # effnet.features[i] for i > 0 is the output of the (i-1)-th block configuration.

        if self.variant == "b0": # EfficientNet-B0
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            effnet = models.efficientnet_b0(weights=weights)
            # features[2]: out 24ch (effnet stride 4) -> P2
            # features[3]: out 40ch (effnet stride 8) -> P3
            # features[5]: out 112ch (effnet stride 16, deeper) -> P4
            # features[6]: out 192ch (effnet stride 32) -> P5

            if self.include_p2:
                return_layers_map = {'2': '0', '3': '1', '5': '2', '6': '3'} # P2, P3, P4, P5
                in_channels_list_map = [24, 40, 112, 192]
            else:
                return_layers_map = {'3': '0', '5': '1', '6': '2'} # P3, P4, P5
                in_channels_list_map = [40, 112, 192]

        elif self.variant == "v2_s":  # EfficientNet V2 Small
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            effnet = models.efficientnet_v2_s(weights=weights)
            # features[2]: out 48ch (effnet stride 4, from bneck_conf[1]) -> P2
            # features[3]: out 64ch (effnet stride 8, from bneck_conf[2]) -> P3
            # features[5]: out 160ch (effnet stride 16, from bneck_conf[4], deeper) -> P4
            # features[6]: out 256ch (effnet stride 32, from bneck_conf[5]) -> P5

            if self.include_p2:
                return_layers_map = {'2': '0', '3': '1', '5': '2', '6': '3'} # P2, P3, P4, P5
                in_channels_list_map = [48, 64, 160, 256]
            else:
                return_layers_map = {'3': '0', '5': '1', '6': '2'} # P3, P4, P5
                in_channels_list_map = [64, 160, 256]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        # backbone feature extractor
        self.body = IntermediateLayerGetter(effnet.features, return_layers=return_layers_map)

        # fpn
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

