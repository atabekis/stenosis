# tsm_backbone.py

# Torch imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.efficientnet import MBConv
from torchvision.models._utils import IntermediateLayerGetter


# Local imports
from util import log
from config import T_CLIP



class TemporalShift(nn.Module):
    """
    Shifts channels along the temporal dimension to enable temporal modeling in the 2D CNN backbone.

    Reference:
        Lin, Ji, Chuang Gan, and Song Han. "Tsm: Temporal shift module for efficient video understanding."
        Proceedings of the IEEE/CVF international conference on computer vision. 2019.
        https://arxiv.org/abs/1811.08383
    """

    def __init__(self, n_segments: int = T_CLIP, n_div: int = 8, shift_mode: str = 'residual'):
        """
        Initialize TemporalShift model

        :param n_segments: corresponds to T, the clip length during training.
        :param n_div: fraction of channels to shift (channels // n_div)
        :param shift_mode: 'residual' or 'inplace'
                            residual adds sifted channels to the original
                            inplace replaces  the channels with the shifted ones.
        """
        super().__init__()

        if shift_mode not in ['residual', 'inplace']:
            raise ValueError(f"shift_mode must be 'residual' or 'inplace', got {shift_mode}")
        if n_segments <= 1:
            # cannot shift if T=1, act as identity
            n_segments = 1
            n_div = 1

        self.n_segments = n_segments
        self.n_div = n_div
        self.shift_mode = shift_mode

        self.buffer = None

    def forward(self, x: torch.Tensor):
        """
        Main forward pass
        :param x: video sequence of n_segments length
            x.shape = [B * T, C, H, W] during training
                      [B, C, H, W] during inference
        """
        if self.n_segments <= 1:  # no shift possible if T=1
            return x

        N, C, H, W = x.size()
        B = N // self.n_segments  # calculate original batch size
        T = self.n_segments

        x_reshaped = x.view(B, T, C, H, W)

        # calc. number of channels to shift
        fold = C // self.n_div
        if fold == 0:  # case where C < n_div
            return x

        if self.shift_mode == 'residual':
            out = x_reshaped.clone()  # copy to add shifted features to
        else:
            out = x_reshaped  # directly modify in place

        # ------- Shift forward (channels 0 to fold-1) --------
        # slice channels to be shifted forward [B, T, fold, H, W]
        shifted_forward = out[:, :, :fold, :, :]
        shifted_forward = torch.roll(shifted_forward, shifts=-1,
                                     dims=1)  # roll among time dim: shift left by 1 step (t → t-1)
        shifted_forward[:, 0, :, :, :] = 0  # zero-pad the first frame T=0 since it doesnt have a prior frame
        out[:, :, :fold, :, :] = shifted_forward  # assign to the out tensor

        # ------- Shift backward (channels fold to 2*fold-1) --------
        shifted_backward = out[:, :, fold: 2 * fold, :, :]  # slice out [B, T, fold, H, W]
        shifted_backward = torch.roll(shifted_backward, shifts=1,
                                      dims=1)  # roll among time dim: shift right by 1 step (t → t+1)
        shifted_backward[:, T - 1, :, :,
        :] = 0  # zero-pad the last frame (T=T-1) which has no future frame to shift from
        out[:, :, fold: 2 * fold, :, :] = shifted_backward

        if self.shift_mode == 'residual':
            final_out = x_reshaped + out
        else:
            final_out = out

        return final_out.view(N, C, H, W)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_segments={self.n_segments}, n_div={self.n_div}, shift_mode='{self.shift_mode}')"


class TSMEfficientNetFPNBackbone(nn.Module):
    """
    Creates an EfficientNet-B0 backbone pretrained on ImageNet, combined with FPN and integrates Temporal Shift Module into specified MBConv blocks

    Extracts features at strides 8, 16, 32, feeds them into the FPN
    TSM modules are inserted before the main convolution within the MBConv blocks
    of the selected stages
    """

    def __init__(self,
                 n_segments: int,
                 out_channels: int = 256,
                 tsm_div: int = 8,
                 tsm_shift_mode: str = 'residual',
                 pretrained: bool = True,
                 ):
        """
        :param n_segments: number of temporal segments (T), analogous to T_CLIP
        :param out_channels: number of channels in the FPN output layers (P3-P5)
        :param tsm_div: channel division factor for TSM
        :param tsm_shift_mode: 'residual' or 'inplace' for TSM module
        :param pretrained: whether to load ImageNet pretrained weights for EfficientNet-B0
        """
        super().__init__()

        log(f"Initializing TSMEfficientNetFPNBackbone:")
        log(f"  n_segments (T): {n_segments}")
        log(f"  FPN out_channels: {out_channels}")
        log(f"  TSM division factor: {tsm_div}")
        log(f"  TSM shift mode: {tsm_shift_mode}")
        log(f"  Pretrained: {pretrained}")

        self.n_segments = n_segments
        self.tsm_div = tsm_div
        self.tsm_shift_mode = tsm_shift_mode

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # --- inserting the TSM modules ------
        # target states (indices in effnet.features) that feed into the FPN
        target_state_indices = [3, 5, 6]
        self._insert_tsm(effnet.features, target_state_indices)

        # --- define FPN layers ---
        # Map effnet features indices to FPN input keys
        # index 3: out stride 8 (40 ch.) -> fpn Level p3 input ('0')
        # index 5: out stride 16 (112 ch.) -> fpn Level p4 input ('1')
        # index 6: out stride 32 (192 ch.) -> fpn Level p5 input ('2')
        return_layers = {'3': '0', '5': '1', '6': '2'}

        # channel counts for the selected layers
        in_channels_list = [
            effnet.features[3][-1].out_channels,  # Last block of stage 3
            effnet.features[5][-1].out_channels,  # stage 5
            effnet.features[6][-1].out_channels  # stage 6
        ]

        self.body = IntermediateLayerGetter(effnet.features, return_layers=return_layers)

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def _insert_tsm(self, features: nn.Sequential, state_indices: list[int]):
        """Modifies MBConv blocks in specified stages to include TSM"""
        for stage_idx in state_indices:
            stage = features[stage_idx]
            if isinstance(stage, nn.Sequential):
                blocks_modified = 0

                for i, block in enumerate(stage):
                    if isinstance(block, MBConv):
                        # Wrap the main depthwise/pointwise conv blocks with tsm, we insert tsm before the first main conv op.
                        # something like:  MBConv(Input -> PW -> BN -> SiLU -> DW -> BN -> SiLU -> SE -> PW -> BN -> Dropout -> Add)

                        original_block_forward = block.forward  # the original forward method

                        tsm_module = TemporalShift(
                            n_segments=self.n_segments,
                            n_div=self.tsm_div,
                            shift_mode=self.tsm_shift_mode,
                        )

                        block.add_module(f'tsm_{stage_idx}_{i}', tsm_module)

                        # define new forward to replace pass for block.block
                        def make_new_forward(original_forward, tsm_mod):
                            def new_forward(x):
                                x_shifted = tsm_mod(x)
                                return original_forward(x_shifted)

                            return new_forward

                        # patch the forward method of MBConv block instance
                        block.forward = make_new_forward(original_block_forward, tsm_module)
                        blocks_modified += 1

                if blocks_modified == 0:
                    log(f'Warning: No MBConv blocks found/modified in stage {stage_idx}')
                else:
                    log(f'Inserted TSM into {blocks_modified} MBConv blocks in stage {stage_idx}', verbose=False)
            else:
                log(f'Warning: Stage {stage_idx} is not  nn.Sequential, skipping TSM insertion')

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through TSM-backbone and FPN
        :param x: input tensor [B, T, C, H, W]
        :return: dict of FPN feature maps. Keys are '0', '1', '2' (for P3, P4, P5). Values are tensors [B, T, FPN_C, H', W']
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)  # reshape for CNN and TSM

        features = self.body(x_reshaped)  # dict: {'0': [BT, C0, H0, W0], '1': [BT, C1, H1, W1], ...}
        fpn_features = self.fpn(features)  # dict: {'0': [BT, FPN_C, H0, W0], '1': [BT, FPN_C, H1, W1], ...}

        output_features = {}
        for key, value in fpn_features.items():
            _, FPN_C, H_prime, W_prime = value.shape
            output_features[key] = value.view(B, T, FPN_C, H_prime, W_prime)

        return output_features