# tsm_backbone.py

# Python imports
import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Sequence

# Torch imports
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

# Local imports
from models.stage2.temporal_shift import TemporalShift
from models.stage2.tsm_load_weights import load_tsm_efficientnet_pretrained_weights, MBConvConfig, _efficientnet_conf

from util import log



class TemporalMBConvBlock(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
            # --- TSM specific args ---
            time_dim: int = 1,
            shift_fraction: float = 0.0,
            shift_mode: str = 'residual',
            use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError(f"Illegal stride value, got {cnf.stride}")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = OrderedDict()
        activation_layer = nn.SiLU

        self.time_dim = time_dim
        self.shift_fraction = shift_fraction
        self.shift_mode = shift_mode
        self.use_tsm = self.shift_fraction > 0.0 and self.time_dim > 1
        if self.use_tsm:
            self.temporal_shift = TemporalShift(self.shift_fraction, self.shift_mode)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        expanded_channels = MBConvConfig.adjust_channels(cnf.input_channels, cnf.expand_ratio)  # use staticmethod via class
        if expanded_channels != cnf.input_channels:
            layers["expand_conv"] = Conv2dNormActivation(
                cnf.input_channels,
                expanded_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        layers["dwconv"] = Conv2dNormActivation(
            expanded_channels,
            expanded_channels,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=expanded_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        squeeze_channels = max(1, cnf.input_channels // 4)
        layers["se"] = se_layer(expanded_channels, squeeze_channels, activation=activation_layer)

        layers["project_conv"] = Conv2dNormActivation(
            expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
        )

        self.block = nn.Sequential(layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def _block_forward_for_checkpoint(self, x: Tensor) -> Tensor:  # helper for checkpointing
        return self.block(x)

    def forward(self, current_input: Tensor) -> Tensor:
        input_to_mb_ops = current_input
        if self.use_tsm:
            B_T, C_local, H_local, W_local = current_input.shape
            if self.time_dim == 0:
                raise ValueError("time_dim cannot be 0 when TSM is active in TemporalMBConvBlock")
            B = B_T // self.time_dim
            if B_T % self.time_dim != 0:
                # this can happen if we pass input directly to TemporalMBConvBlock with B_T not a multiple of time_dim.
                # TSMEfficientNet itself (should) (insallah) handle the B*T reshape correctly for its own forward pass.
                pass

            input_reshaped = current_input.view(B, self.time_dim, C_local, H_local, W_local)
            shifted_input_btchw = self.temporal_shift(input_reshaped).view(B_T, C_local, H_local, W_local)
            input_to_mb_ops = shifted_input_btchw

        if self.training and self.use_gradient_checkpointing:  # for high t_clip values this is sorta necessary
            try:
                block_output = checkpoint(self._block_forward_for_checkpoint, input_to_mb_ops, use_reentrant=False)
            except TypeError:
                block_output = checkpoint(self._block_forward_for_checkpoint, input_to_mb_ops)
        else:
            block_output = self.block(input_to_mb_ops)

        if self.use_res_connect:
            final_output = self.stochastic_depth(block_output) + input_to_mb_ops
        else:
            final_output = block_output
        return final_output


class TSMEfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: Sequence[MBConvConfig],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
            time_dim: int = 1,
            shift_fraction: float = 0.0,
            shift_mode: str = 'residual',
            tsm_stages: Optional[list[int]] = None,
            use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        layers = OrderedDict()

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers["stem_conv"] = Conv2dNormActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        current_stage_idx = 1
        for i, cnf in enumerate(inverted_residual_setting):
            stage: OrderedDict[str, nn.Module] = OrderedDict()
            apply_tsm_in_stage = (
                    shift_fraction > 0.0 and
                    time_dim > 1 and
                    (tsm_stages is None or not tsm_stages or current_stage_idx in tsm_stages)
            # check not tsm_stages for empty list
            )
            stage_shift_fraction = shift_fraction if apply_tsm_in_stage else 0.0

            for j in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if j > 0:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = stochastic_depth_prob * float(
                    stage_block_id) / total_stage_blocks if total_stage_blocks > 0 else 0.0

                stage[f"block{j + 1}"] = TemporalMBConvBlock(
                    block_cnf,
                    sd_prob,
                    norm_layer,
                    time_dim=time_dim,
                    shift_fraction=stage_shift_fraction,
                    shift_mode=shift_mode,
                    use_gradient_checkpointing=use_gradient_checkpointing
                )
                stage_block_id += 1
            layers[f"stage{current_stage_idx}"] = nn.Sequential(stage)
            current_stage_idx += 1

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers["top_conv"] = Conv2dNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x  # features before avgpool and classifier for backbone

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def tsm_efficientnet_b0(
        pretrained: bool = False,
        time_dim: int = 1,
        shift_fraction: float = 0.0,
        shift_mode: str = 'residual',
        tsm_stages_indices: Optional[list[int]] = None,
        use_gradient_checkpoint: bool = False,
        num_classes: int = 1000,
        last_channel: Optional[int] = None,
        verbose: bool = False,

        use_groupnorm: bool = True,
        num_gn_groups: int = 32,

        **kwargs: any
) -> TSMEfficientNet:
    """
    Constructs a TSM-EfficientNet-B0 model.
    """

    if use_groupnorm:
        log(f'Using GroupNorm, number of groups for GN: {num_gn_groups}')
        curr_norm_layer = partial(nn.GroupNorm, num_gn_groups)
    else:
        curr_norm_layer = nn.BatchNorm2d

    tsm_stages_internal = None
    if tsm_stages_indices:
        # expected to be like [3,5,6] referring to the output stages
        # so, if tsm_stages_indices=[3,5,6], these are directly the stage numbers.
        tsm_stages_internal = [idx for idx in tsm_stages_indices if 1 <= idx <= 7]

    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    model = TSMEfficientNet(
        inverted_residual_setting,
        dropout=0.2,
        stochastic_depth_prob=0.2,
        time_dim=time_dim,
        shift_fraction=shift_fraction,
        shift_mode=shift_mode,
        tsm_stages=tsm_stages_internal,
        use_gradient_checkpointing=use_gradient_checkpoint,
        num_classes=num_classes,
        last_channel=last_channel,

        norm_layer=curr_norm_layer,
    )

    if pretrained:
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights_url = EfficientNet_B0_Weights.IMAGENET1K_V1.url
        except ImportError:
            weights_url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"

        loading_stats = load_tsm_efficientnet_pretrained_weights(model, weights_url)

        if verbose:
            if loading_stats.get("missing_keys") or loading_stats.get("unmapped_pretrained_keys"):
                log("WARNING: Issues encountered during pretrained weight loading for TSMEfficientNet.")
                log(f"  Missing Keys: {len(loading_stats.get('missing_keys', [])), loading_stats.get('missing_keys', [])}")
                log(f"  Unmapped Pretrained Keys: {len(loading_stats.get('unmapped_pretrained_keys', [])), loading_stats.get('unmapped_pretrained_keys', [])}")
            else:
                log("Pretrained weights loaded successfully into TSMEfficientNet.")
    return model