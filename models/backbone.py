# backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter

class EfficientNetFPNBackbone(nn.Module):
    """
    Creates an EfficientNet-B0 backbone pretrained on ImageNet, combined with a Feature Pyramid Network (FPN).

    Extracts features at strides 8, 16, 32 to outputs of index 3, 5, and 6 of EfficientNet-B0',
    feeds them into the FPN. FPN produces P3, P4, P5 feature maps with a specified number of channels.
    """
    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        """
        :param out_channels: Number of channels in the FPN output layers (P3-P5).
        """
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # layers from which to extract features using their *index* in effnet.features
        # index 3: out stride 8 (40 ch.) -> fpn Level p3 input
        # index 5: out stride 16 (112 ch.) -> fpn Level p4 input
        # index 6: out stride 32 (192 ch.) -> fpn Level p5 input
        return_layers = {
            '3': '0',  # map output of effnet features 3 to key 0
            '5': '1',  # 5 to key 1
            '6': '2'   # 6 to key 2
        }

        # we need the number of output channels from these layers for the fpn
        in_channels_list = [
            40, # features[3]
            112, # features[5]
            192 # features[6]
        ]

        # create the backbone feature extractor
        self.body = IntermediateLayerGetter(effnet.features, return_layers=return_layers)

        # create fpn
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        self.out_channels = out_channels # fpn output channels


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through Backbone and FPN.
        :param x: Input tensor of shape [B, 3, H, W].
        :returns: dictionary of feature maps from fpn levels (keys 0, 1, 2 corresponding to P3, P4, P5).
        """
        x = self.body(x)
        x = self.fpn(x)
        return x



if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 512, 512)
    try:
        backbone = EfficientNetFPNBackbone(out_channels=256)
        features = backbone(dummy_input)
        print("Backbone instantiated successfully!")
        print("Output feature shapes:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}") # expected keys: 0, 1, 2

        # check output channels
        assert all(f.shape[1] == 256 for f in features.values())
        # check spatial dimensions (downsampling)
        print(f"P3 (key '0') spatial: {features['0'].shape[-2:]}")
        print(f"P4 (key '1') spatial: {features['1'].shape[-2:]}")
        print(f"P5 (key '2') spatial: {features['2'].shape[-2:]}")
        assert features['0'].shape[-2:] == (64, 64)
        assert features['1'].shape[-2:] == (32, 32)
        assert features['2'].shape[-2:] == (16, 16)
        print("checks passed")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()