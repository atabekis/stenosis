# tsm_utils.py

import torch
import torch.nn as nn

from util import log


class TemporalShift(nn.Module):
    """
    Shifts channels along the temporal dimension to enable temporal modeling in the 2D CNN backbone.

    Reference:
        Lin, Ji, Chuang Gan, and Song Han. "Tsm: Temporal shift module for efficient video understanding."
        Proceedings of the IEEE/CVF international conference on computer vision. 2019.
        https://arxiv.org/abs/1811.08383
    """
    def __init__(self, shift_fraction: float = 0.125, shift_mode: str = 'residual'):
        """
        Initialize TemporalShift model
        :param shift_fraction: fraction of channels to shift, should be 0 <= fraction <= 0.5
        :param shift_mode: 'residual' or 'inplace'
                            residual adds sifted channels to the original
                            inplace replaces  the channels with the shifted ones.
        """

        super().__init__()

        if shift_mode not in ['inplace', 'residual']:
            raise ValueError(f"shift_mode must be 'inplace' or 'residual', got {shift_mode}")

        if not (0.0 <= shift_fraction <= 0.5):
             raise ValueError(f"shift_fraction must be between 0.0 and 0.5, got {shift_fraction}")

        self.shift_fraction = shift_fraction
        self.shift_mode = shift_mode


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the temporal shift."""
        B, T, C, H, W = x.shape

        if T <= 1 or self.shift_fraction == 0.0:  # cannot perform shift if seq. len=1 or fraction is 0
            return x

        # number of channels to shift (even number for split)
        num_shifted_channels = max(1, int(C * self.shift_fraction))
        num_shifted_channels = (num_shifted_channels // 2) * 2 # even
        if num_shifted_channels == 0 and self.shift_fraction > 0.0:  # if not possible to shift
             log(f"Warning: TemporalShift with shift_fraction={self.shift_fraction} resulted in 0 channels being shifted for C={C}. Skipping shift.")
             return x

        num_fwd = num_shifted_channels // 2  # ch to be shifted forward
        num_bwd = num_shifted_channels // 2  # backward


        # part_stay: ch that remain unchanged (num_shifted_channels to C-1)
        part_shift_fwd, part_shift_bwd, part_stay = torch.split(
            x, [num_fwd, num_bwd, C - num_shifted_channels], dim=2
        )

        # 1. forward shift
        # we pad start with 0, take first T-1 frames
        zeros_fwd = torch.zeros_like(part_shift_fwd[:, :1, ...]) #  [B, 1, num_fwd, H, W]
        shifted_fwd = torch.cat((zeros_fwd, part_shift_fwd[:, :-1, ...]), dim=1) # [B, T, num_fwd, H, W]


        # 2. backward shift
        # pad end with 0, take last T-1 frames, create zero padding for the last time step
        zeros_bwd = torch.zeros_like(part_shift_bwd[:, -1:, ...]) # [B, 1, num_bwd, H, W]
        shifted_bwd = torch.cat((part_shift_bwd[:, 1:, ...], zeros_bwd), dim=1) # [B, T, num_bwd, H, W]


        # combine with features
        if self.shift_mode == 'inplace':  # shift replaces channels with shifted ones
            output = torch.cat((shifted_fwd, shifted_bwd, part_stay), dim=2)


        else: # residual
            # calculate differences caused by shift (delta)
            delta_fwd = shifted_fwd - part_shift_fwd
            delta_bwd = shifted_bwd - part_shift_bwd

            # we need to create zero delta for the non-shift part
            delta_stay = torch.zeros_like(part_stay)
            delta_all = torch.cat([delta_fwd, delta_bwd, delta_stay], dim=2)

            output = x + delta_all

        return output

