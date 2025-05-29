# util.py

# Python imports
import os
import math
import torch
import inspect
import datetime
import multiprocessing as mp

_DEFAULT_COLORS = {
    "timestamp": "\033[92m", # green
    "filename": "\033[94m", # blue
    "funcname": "\033[95m", # pink
    "reset": "\033[0m" # reset the color
}

_is_slurm_env = 'SLURM_JOB_ID' in os.environ
if _is_slurm_env:
    COLORS = {key: "" for key in _DEFAULT_COLORS}
else:
    COLORS = _DEFAULT_COLORS

def log(*args, verbose=True, show_func=True, omit_funcs=None, **kwargs):
    """
    Take regular print string as input, add timestamp, filename, function or class name where log() was called from.
    """
    if not verbose:
        return

    if omit_funcs is None:
        omit_funcs = {'<module>', '__init__'}


    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    class_name = frame.f_locals.get('self', None).__class__.__name__ if 'self' in frame.f_locals else None
    func_name = f'{class_name}.{caller}' if class_name else caller

    filename = os.path.basename(frame.f_code.co_filename).replace('.py', '')
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')

    # apply colors
    c_timestamp = f"{COLORS['timestamp']}[{timestamp}]{COLORS['reset']}"
    c_filename = f"{COLORS['filename']}{filename}{COLORS['reset']}"

    parts = [c_timestamp, f"[{c_filename}"]
    if show_func and caller not in omit_funcs:
        c_funcname = f"{COLORS['funcname']}{func_name}{COLORS['reset']}"
        parts.append(f".{c_funcname}")
    parts.append("]")

    message = ''.join(map(str, args))
    print("".join(parts), f"\"{message}\"", **kwargs)



_BASELINE_SHAPE= (512, 512)
_BASELINE_ANCHORS = {
    False: ((16, 26, 38), (48, 62, 80), (96, 112, 136)),  # P3 P4 P5
    True: ((10, 14, 18), (24, 32, 42), (56, 72, 88), (104, 128, 150)), #P2 P3 P4 P5
}
_ASPECT_RATIOS = (0.7, 1.0, 1.4)


def get_anchor_config(current_img_height: int, current_img_width: int, include_p2_fpn: bool):
    """
    Compute scaled anchor sizes and aspect ratios for a given image size.

    Scaling is done by the geometric mean of height and width factors relative
    to the baseline 512×512 image. Aspect ratios remain fixed per level.

    """
    base_h, base_w = _BASELINE_SHAPE

    # geometric-mean scale factor
    scale = math.sqrt((current_img_height / base_h) * (current_img_width / base_w))

    baseline_levels = _BASELINE_ANCHORS[include_p2_fpn]

    # scale and clamp to at least 1px
    anchor_sizes = tuple(
        tuple(max(1, int(round(size * scale))) for size in level)
        for level in baseline_levels
    )
    # repeat the aspect-ratio set for each level
    aspect_ratios = (_ASPECT_RATIOS,) * len(anchor_sizes)

    return anchor_sizes, aspect_ratios


def has_positive_gt(sample_targets, positive_class_id: int, model_stage: int) -> bool:
    """
    Return True if sample_targets contains a label equal to positive_class_id.
    """
    # stage 1: single dict of labels
    if model_stage == 1:
        labels = sample_targets.get("labels", None)
        if (
            isinstance(labels, torch.Tensor)
            and labels.numel() > 0
            and (labels == positive_class_id).any()
        ):
            return True
        return False

    # stage 2/3: sequence of frame‐dicts
    for frame_dict in sample_targets or ():
        labels = frame_dict.get("labels", None)
        if (
            isinstance(labels, torch.Tensor)
            and labels.numel() > 0
            and (labels == positive_class_id).any()
        ):
            return True
    return False
