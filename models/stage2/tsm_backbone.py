# tsm_backbone.py

# Python imports
import copy
import math
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence


# Torch imports
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torch.hub import load_state_dict_from_url
from torchvision.models._utils import _make_divisible
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

# Local imports
from .temporal_shift import TemporalShift



class MBConvConfig:
    """
    Configuration for a Mobile Inverted Residual Bottleneck block, following EfficientNet specs.
    """
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        # these will be set by partial
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(expand_ratio={self.expand_ratio}, "
            f"kernel={self.kernel}, stride={self.stride}, "
            f"input_channels={self.input_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers})"
        )

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(int(channels * width_mult), 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float) -> int:
        return int(math.ceil(int(num_layers * depth_mult)))



class TemporalMBConvBlock(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
        # --- TSM Specific Args ---
        time_dim: int = 1, # Sequence length T
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

        # --- TSM specific ---
        self.time_dim = time_dim
        self.shift_fraction = shift_fraction
        self.shift_mode = shift_mode
        self.use_tsm = self.shift_fraction > 0.0 and self.time_dim > 1
        if self.use_tsm:
             self.temporal_shift = TemporalShift(self.shift_fraction, self.shift_mode)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # expansion phase
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers["expand_conv"] = Conv2dNormActivation(
                cnf.input_channels,
                expanded_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        # depthwise convolution
        layers["dwconv"] = Conv2dNormActivation(
            expanded_channels,
            expanded_channels,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=expanded_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers["se"] = se_layer(expanded_channels, squeeze_channels, activation=activation_layer)

        # projection phase
        layers["project_conv"] = Conv2dNormActivation(
            expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
        )

        # add the layers to model
        self.block = nn.Sequential(layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    # def forward(self, input: Tensor) -> Tensor:
    #     if self.use_tsm:
    #          B_T, C, H, W = input.shape
    #          B = B_T // self.time_dim
    #
    #          input_reshaped = input.view(B, self.time_dim, C, H, W)
    #          shifted_input = self.temporal_shift(input_reshaped)
    #          input = shifted_input.view(B_T, C, H, W)
    #
    #     result = self.block(input)
    #
    #     if self.use_res_connect:  # residual connection adds results back onto the input
    #         result = self.stochastic_depth(result)
    #         result += input
    #
    #     return result

    def forward(self, current_input: Tensor) -> Tensor:  # [B*T, C, H, W]
        input_to_mb_ops = current_input
        if self.use_tsm: # reshape -> apply TSM -> reshape back
            B_T, C_local, H_local, W_local = current_input.shape
            B = B_T // self.time_dim

            input_reshaped = current_input.view(B, self.time_dim, C_local, H_local, W_local) #[B, T, C, H, W]
            shifted_input_btchw = self.temporal_shift(input_reshaped).view(B_T, C_local, H_local, W_local) # [B*T, C, H, W]
            input_to_mb_ops = shifted_input_btchw

        if self.training and self.use_gradient_checkpointing:
            block_output = checkpoint(self._block_forward_for_checkpoint, input_to_mb_ops, use_reentrant=False)
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
        num_classes: int = 1000, # standard ImageNet default, not used for our detection backbone
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        # TSM related params
        time_dim: int = 1,
        shift_fraction: float = 0.0,
        shift_mode: str = 'inplace',
        tsm_stages: Optional[list[int]] = None,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """
        TSM_EfficientNet V1 architecture.

        :param inverted_residual_setting: Sequence of MBConvConfig specifying block layout.
        :param dropout: Dropout probability for the final classifier layer.
        :param stochastic_depth_prob: Stochastic depth probability.
        :param num_classes: Number of classes for the classifier head (not used for our detection).
        :param norm_layer: Normalization layer module. Defaults to BatchNorm2d.
        :param last_channel: The number of channels on the penultimate layer / classification head.
        :param time_dim: Sequence length T for TSM.
        :param shift_fraction: Fraction of channels to shift in TSM blocks.
        :param shift_mode: 'inplace' or 'residual' for TSM.
        :param tsm_stages: Optional list of stage indices (e.g., [1, 2, 3, 4, 5, 6]) where TSM should be applied.
                           If None or empty, TSM is applied based on shift_fraction only (all applicable blocks).
                           Note: Stage 0 is the initial Conv layer. Stages 1-7 correspond to MBConv blocks.
        """
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = OrderedDict()

        # build stem
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers["stem_conv"] = Conv2dNormActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        )

        # build MBConv blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        current_stage_idx = 1 # Start MBConv blocks at stage 1
        for i, cnf in enumerate(inverted_residual_setting):
            stage: OrderedDict[str, nn.Module] = OrderedDict()
            # determine if TSM should be applied in this stage
            apply_tsm_in_stage = (
                shift_fraction > 0.0 and
                time_dim > 1 and
                (tsm_stages is None or current_stage_idx in tsm_stages)
            )
            stage_shift_fraction = shift_fraction if apply_tsm_in_stage else 0.0

            for j in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if j > 0:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage[f"block{j+1}"] = TemporalMBConvBlock(
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


        # build top
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

        # init. weights
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
        # forward for classifier, expects [B*T, C, H, W]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_pretrained_weights(self, weights_url: str):
        """Loads pretrained weights from standard EfficientNet, attempting a detailed layer mapping."""
        stats = {
            "loaded": 0,
            "shape_mismatch": 0,
            "other_skipped": 0,
            "unmapped_pretrained_keys": [],
            "missing_keys": [],
            "unexpected_keys": []
        }

        # 1. download pretrained weights
        state_dict_pretrained = load_state_dict_from_url(weights_url, progress=True)
        own_state = self.state_dict()

        # 2. mapping tables
        simple_prefixes = {
            "features.0.": "features.stem_conv.",
            "features.8.": "features.top_conv.",
        }
        mbconv_inner_map = {
            "0.0.": "expand_conv.0.",
            "0.1.": "expand_conv.1.",
            "1.0.": "dwconv.0.",
            "1.1.": "dwconv.1.",
            "2.fc1.": "se.fc1.",
            "2.fc2.": "se.fc2.",
            "3.0.": "project_conv.0.",
            "3.1.": "project_conv.1.",
        }

        def map_pt_key(pt_key: str) -> Optional[str]:
            # a. stem & top conv
            for src, dst in simple_prefixes.items():
                if pt_key.startswith(src):
                    return pt_key.replace(src, dst, 1)

            # b. MBConv blocks
            if pt_key.startswith("features.") and not pt_key.startswith("features.8."):
                parts = pt_key.split(".")
                if len(parts) >= 7 and parts[3] == "block":
                    try:
                        stage = int(parts[1])
                        block = int(parts[2]) + 1
                        inner = ".".join(parts[4:])
                        for src_inner, dst_inner in mbconv_inner_map.items():
                            if inner.startswith(src_inner):
                                remapped = inner.replace(src_inner, dst_inner, 1)
                                return f"features.stage{stage}.block{block}.block.{remapped}"
                    except ValueError:
                        pass

            # c. classifier
            if pt_key.startswith("classifier.1."):
                return pt_key

            return None

        # 3. remap & filter shapes
        new_state = {}
        for pt_key, pt_tensor in state_dict_pretrained.items():
            model_key = map_pt_key(pt_key)
            if model_key is None or model_key not in own_state:
                stats["unmapped_pretrained_keys"].append(pt_key)
                continue

            target = own_state[model_key]
            if target.shape != pt_tensor.shape:
                stats["shape_mismatch"] += 1
                stats["unmapped_pretrained_keys"].append(pt_key)
            else:
                new_state[model_key] = pt_tensor
                stats["loaded"] += 1

        # 4. load into the model
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        stats["missing_keys"] = list(missing)
        stats["unexpected_keys"] = list(unexpected)

        # 5. Count any other skips
        total_params = len(own_state)
        stats["other_skipped"] = total_params - stats["loaded"] - stats["shape_mismatch"]

        return stats



def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> list[MBConvConfig]:
    """
    Build the MBConvConfig list for EfficientNet-B0 (or scaled variants).
    """
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    return [
        bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=32,  out_channels=16,  num_layers=1),  # stage 1 (idx=1 in features list)
        bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=16,  out_channels=24,  num_layers=2),  # (idx=2)
        bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=24,  out_channels=40,  num_layers=2),  # 3 <- stride 8 output
        bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=40,  out_channels=80,  num_layers=3),  # 4
        bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=80,  out_channels=112, num_layers=3), # 5 <- stride 16 output
        bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=112, out_channels=192, num_layers=4), # 6 <- stride 32 output
        bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=192, out_channels=320, num_layers=1),  # 7
    ]



def tsm_efficientnet_b0(
    pretrained: bool = False,
    time_dim: int = 1,
    shift_fraction: float = 0.0,
    shift_mode: str = 'residual',
    tsm_stages_indices: Optional[list[int]] = None, #[3, 5, 6]
    use_gradient_checkpoint: bool = False,
    **kwargs: Any
) -> TSMEfficientNet:
    """
    Constructs a TSM-EfficientNet-B0 model.
    """
    # map backbone features to stage indices
    tsm_stages_internal = None
    if tsm_stages_indices: tsm_stages_internal = [idx + 1 for idx in tsm_stages_indices if 1 <= idx+1 <= 7]

    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    model = TSMEfficientNet(
        inverted_residual_setting,
        dropout=0.2, # default B0 dropout
        stochastic_depth_prob=0.2, # default B0 SD prob
        time_dim=time_dim,
        shift_fraction=shift_fraction,
        shift_mode=shift_mode,
        tsm_stages=tsm_stages_internal,
        use_gradient_checkpointing=use_gradient_checkpoint,
        **kwargs
    )

    if pretrained:
        # some explanation as to why we load from the url and not use the weights directly:
        # we download the raw state dict of the standard pretrained model, then we customize
        # the backbone of the EffNet architecture to insert the TSM.
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights_url = EfficientNet_B0_Weights.IMAGENET1K_V1.url
        except ImportError: weights_url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"

        model.load_pretrained_weights(weights_url)

    return model


