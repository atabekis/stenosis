# tsm_load_weights.py

# Torch imports
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models._utils import _make_divisible

# Python & local imports
import math
from functools import partial

from util import log


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
    def adjust_channels(channels: int, width_mult: float, min_value: int = None) -> int:
        return _make_divisible(int(channels * width_mult), 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float) -> int:
        return int(math.ceil(int(num_layers * depth_mult)))


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: any) -> list[MBConvConfig]:
    """
    Build the MBConvConfig list for EfficientNet-B0 (or scaled variants).
    """
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    return [
        bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=32, out_channels=16, num_layers=1),
        bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=16, out_channels=24, num_layers=2),
        bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=24, out_channels=40, num_layers=2),
        bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=40, out_channels=80, num_layers=3),
        bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=80, out_channels=112, num_layers=3),
        bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=112, out_channels=192, num_layers=4),
        bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=192, out_channels=320, num_layers=1),
    ]


# --- End of definitions needed for weight loading ---


def load_tsm_efficientnet_pretrained_weights(
        tsm_model: nn.Module,
        weights_url: str,
        width_mult_ref: float = 1.0,
        depth_mult_ref: float = 1.0
) -> dict[str, any]:
    """
    Loads pretrained weights from a standard EfficientNet state_dict URL into a TSMEfficientNet model.
    This function attempts a detailed layer mapping, accounting for blocks with expand_ratio=1.

    :param tsm_model: An instance of the TSMEfficientNet model to load weights into.
    :param weights_url: URL to the .pth file of the standard EfficientNet pretrained weights.
    :param width_mult_ref: Width multiplier of the reference EfficientNet model (e.g., 1.0 for B0).
    :param depth_mult_ref: Depth multiplier of the reference EfficientNet model (e.g., 1.0 for B0).
    :return: A dictionary containing statistics about the loading process.
    """
    stats = {
        "loaded": 0,
        "shape_mismatch": 0,
        "unmapped_pretrained_keys": [],
        "missing_keys": [],
        "unexpected_keys": []
    }

    try:
        state_dict_pretrained = load_state_dict_from_url(weights_url, progress=True)
    except Exception as e:
        log(f"Error loading weights from URL {weights_url}: {e}")
        return stats

    own_state = tsm_model.state_dict()

    ref_stage_configs = _efficientnet_conf(width_mult=width_mult_ref, depth_mult=depth_mult_ref)

    simple_prefixes = {
        "features.0.": "features.stem_conv.",
        "features.8.": "features.top_conv.",
        "classifier.1.": "classifier.1."
    }

    canonical_mbconv_inner_map = {
        "0.0.": "expand_conv.0.", "0.1.": "expand_conv.1.",
        "1.0.": "dwconv.0.", "1.1.": "dwconv.1.",
        "3.0.": "project_conv.0.", "3.1.": "project_conv.1.",
        "2.fc1.": "se.fc1.", "2.fc2.": "se.fc2."
    }

    new_state_to_load = {}

    for pt_key, pt_tensor in state_dict_pretrained.items():
        mapped_model_key = None
        key_processed_in_block_logic = False

        for src_prefix, dst_prefix in simple_prefixes.items():
            if pt_key.startswith(src_prefix):
                mapped_model_key = pt_key.replace(src_prefix, dst_prefix, 1)
                break

        if mapped_model_key:
            key_processed_in_block_logic = True
            if mapped_model_key in own_state:
                if own_state[mapped_model_key].shape == pt_tensor.shape:
                    new_state_to_load[mapped_model_key] = pt_tensor
                    stats["loaded"] += 1
                else:
                    stats["shape_mismatch"] += 1
                    stats["unmapped_pretrained_keys"].append(
                        f"{pt_key} (shape mismatch with simple prefix {mapped_model_key})")
            else:
                stats["unmapped_pretrained_keys"].append(
                    f"{pt_key} (target simple prefix key {mapped_model_key} not found in TSM model)")
            continue

        if pt_key.startswith("features.") and not pt_key.startswith("features.8.") and not pt_key.startswith(
                "features.0."):
            key_processed_in_block_logic = True
            parts = pt_key.split(".")
            if len(parts) >= 6 and parts[3] == "block":
                try:
                    tv_stage_idx = int(parts[1])
                    tv_block_idx_in_stage = int(parts[2])

                    if not (1 <= tv_stage_idx <= len(ref_stage_configs)):
                        stats["unmapped_pretrained_keys"].append(f"{pt_key} (invalid tv_stage_idx {tv_stage_idx})")
                        continue

                    cnf_for_this_stage_type = ref_stage_configs[tv_stage_idx - 1]
                    original_block_has_expand = (cnf_for_this_stage_type.expand_ratio != 1)
                    original_sequential_module_idx = int(parts[4])

                    canonical_op_type_idx = -1
                    if original_block_has_expand:
                        canonical_op_type_idx = original_sequential_module_idx
                    else:
                        if original_sequential_module_idx == 0:
                            canonical_op_type_idx = 1
                        elif original_sequential_module_idx == 1:
                            canonical_op_type_idx = 2
                        elif original_sequential_module_idx == 2:
                            canonical_op_type_idx = 3

                    if canonical_op_type_idx == -1:
                        stats["unmapped_pretrained_keys"].append(
                            f"{pt_key} (could not map original op index {original_sequential_module_idx} for expand_ratio={cnf_for_this_stage_type.expand_ratio})")
                        continue

                    original_op_path_segment = str(canonical_op_type_idx) + "." + ".".join(parts[5:])
                    if canonical_op_type_idx == 2 and parts[5].startswith("fc"):
                        original_op_path_segment = str(canonical_op_type_idx) + "." + ".".join(parts[5:])

                    destination_layer_prefix_in_tsm_mbconv = None
                    parameter_suffix = ""
                    for src_map_prefix, dst_map_prefix in canonical_mbconv_inner_map.items():
                        if original_op_path_segment.startswith(src_map_prefix):
                            destination_layer_prefix_in_tsm_mbconv = dst_map_prefix
                            parameter_suffix = original_op_path_segment[len(src_map_prefix):]
                            break

                    if destination_layer_prefix_in_tsm_mbconv:
                        tsm_model_stage_num = tv_stage_idx
                        tsm_model_block_num = tv_block_idx_in_stage + 1

                        your_tsm_block_cnf = ref_stage_configs[tsm_model_stage_num - 1]
                        your_tsm_block_has_expand = (your_tsm_block_cnf.expand_ratio != 1)

                        if destination_layer_prefix_in_tsm_mbconv.startswith(
                                "expand_conv") and not your_tsm_block_has_expand:
                            stats["unmapped_pretrained_keys"].append(
                                f"{pt_key} (original had expand, TSM model block does not for stage {tsm_model_stage_num})")
                            continue

                        mapped_model_key = f"features.stage{tsm_model_stage_num}.block{tsm_model_block_num}.block.{destination_layer_prefix_in_tsm_mbconv}{parameter_suffix}"

                        if mapped_model_key in own_state:
                            if own_state[mapped_model_key].shape == pt_tensor.shape:
                                new_state_to_load[mapped_model_key] = pt_tensor
                                stats["loaded"] += 1
                            else:
                                stats["shape_mismatch"] += 1
                                stats["unmapped_pretrained_keys"].append(
                                    f"{pt_key} (shape mismatch for TSM key {mapped_model_key})")
                        else:
                            stats["unmapped_pretrained_keys"].append(
                                f"{pt_key} (target TSM MBConv key {mapped_model_key} not found)")
                        continue
                    else:
                        stats["unmapped_pretrained_keys"].append(
                            f"{pt_key} (no match in canonical_mbconv_inner_map for op path {original_op_path_segment})")
                        continue
                except Exception as e_map:
                    stats["unmapped_pretrained_keys"].append(f"{pt_key} (error during MBConv mapping: {e_map})")
                    continue

        if not key_processed_in_block_logic:
            stats["unmapped_pretrained_keys"].append(f"{pt_key} (no mapping rule applied or fell through)")

    try:
        load_result = tsm_model.load_state_dict(new_state_to_load, strict=False)
        stats["missing_keys"] = list(load_result.missing_keys)
        stats["unexpected_keys"] = list(load_result.unexpected_keys)
    except Exception as e_load:
        log(f"Error during tsm_model.load_state_dict: {e_load}")
        stats["missing_keys"] = list(set(own_state.keys()) - set(new_state_to_load.keys()))

    return stats


if __name__ == '__main__':
    from tsm_backbone import tsm_efficientnet_b0
    test_model = tsm_efficientnet_b0(pretrained=False, time_dim=1, shift_fraction=0.0)
    try:
        from torchvision.models import EfficientNet_B0_Weights
        url = EfficientNet_B0_Weights.IMAGENET1K_V1.url
    except ImportError:
        url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
    results = load_tsm_efficientnet_pretrained_weights(test_model, url)
    log(results)
    pass