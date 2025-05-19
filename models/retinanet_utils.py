# retinanet_utils.py
import torch
import torch.nn as nn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from collections import OrderedDict

from util import log

class GNDropoutRetinaNetClassificationHead(RetinaNetClassificationHead):
    """
    RetinaNet classification head that allows customization of:
    - Number of convolutional layers in the tower.
    - Optional Dropout2d after activations in the tower.
    - Optional GroupNorm after convolutions in the tower.
    """

    def __init__(self,
                 in_channels: int,
                 num_anchors: int,
                 num_classes: int,
                 num_convs: int = 4,
                 prior_probability: float = 0.01,
                 dropout_p: float = 0.0, # 0.0 to disable
                 use_groupnorm: bool = False,
                 num_gn_groups: int = 32):


        super().__init__(in_channels, num_anchors, num_classes, prior_probability)

        if not (0 <= dropout_p < 1):
            raise ValueError(f"dropout_p must be between 0 and 1, got {dropout_p}")
        if not (1 <= num_convs <= 4):
            raise ValueError(f"num_convs must be between 1 and 4 (inclusive), got {num_convs}")

        log(f"    Dropout Probability: {dropout_p}")
        log(f"    Number of Convs: {num_convs}")
        log(f"    Use GroupNorm: {use_groupnorm}")
        if use_groupnorm:
            log(f"      GN Groups: {num_gn_groups}")

        self.custom_dropout_p = dropout_p
        self.custom_num_convs = num_convs
        self.custom_use_groupnorm = use_groupnorm
        self.custom_num_gn_groups = num_gn_groups

        new_conv_tower_modules = OrderedDict()

        for i in range(self.custom_num_convs):
            new_conv_tower_modules[f'conv{i}'] = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            if self.custom_use_groupnorm:
                actual_num_gn_groups = self._get_actual_gn_groups(in_channels, self.custom_num_gn_groups, i)
                new_conv_tower_modules[f'gn{i}'] = nn.GroupNorm(
                    num_groups=actual_num_gn_groups,
                    num_channels=in_channels
                )

            new_conv_tower_modules[f'relu{i}'] = nn.ReLU()

            if self.custom_dropout_p > 0 and i < self.custom_num_convs - 1:
                new_conv_tower_modules[f'dropout{i}'] = nn.Dropout2d(p=self.custom_dropout_p)

        self.conv = nn.Sequential(new_conv_tower_modules)

        for layer_name, layer in self.conv.named_modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)


    def _get_actual_gn_groups(self, current_in_channels: int, config_num_groups: int, layer_idx: int) -> int:
        """ Helper to calculate actual GN groups, ensuring divisibility. """
        actual_num_gn_groups = config_num_groups
        if current_in_channels % actual_num_gn_groups != 0:
            possible_groups = [g for g in [32, 16, 8, 4, 2, 1] if current_in_channels % g == 0]
            if not possible_groups:
                raise ValueError(
                    f"Cannot find suitable num_groups for GroupNorm with in_channels={current_in_channels}."
                )
            actual_num_gn_groups = possible_groups[0]
            if actual_num_gn_groups != config_num_groups:
                print(f"Warning: For GroupNorm in layer {layer_idx} with in_channels={current_in_channels}, "
                      f"adjusted num_gn_groups from configured {config_num_groups} to {actual_num_gn_groups}.")
        return actual_num_gn_groups


