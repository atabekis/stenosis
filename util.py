# util.py

# Python imports
import os
import math
import torch
import inspect
import logging
import datetime
import pandas as pd

from config import LOGS_DIR

LOG_LOCALLY = True

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


def _configure_logger(log_dir) -> logging.Logger:
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    logging.basicConfig(
        filename=log_dir / "project.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )
    return logger

_LOGGER = _configure_logger(LOGS_DIR)


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

    if LOG_LOCALLY:
        _LOGGER.log(
            level=logging.WARNING if "warning" in message.lower() else logging.INFO,
            msg=message,
            stacklevel=2
        )



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


def save_results_to_csv(results, csv_path, model_name):
    """
    Persist test-run metrics to *csv_path*.
    Rows are unique on (stage, t_clip); newer entries overwrite older ones.
    Accuracy metrics are rounded to 3 dp, fps to 1 dp, latency to 1 dp.
    """
    # --------------------------------------------------------------------- #
    _METRIC_MAP = {
        "test/mAP_0.5": "mAP",
        "test/mAR_100": "mAR_100",
        "test/AvgIoU_TP_0.5": "AvgIoU_TP_0.5",
        "test/AP_small": "AP_small",
        "test/Precision_0.5": "Precision",
        "test/Recall_0.5": "Recall",
        "test/F1_0.5": "F1",
    }
    _SPEED_KEYS = [
        "avg_time_ms_batch",       # kept for completeness (no rounding rule)
        "avg_latency_ms_batch",
        "avg_latency_ms_frame",
        "fps",
    ]

    metrics = results.get("test_results", {}).get("final_test_metrics", {})
    if not metrics:
        print(f"[save_results_to_csv] No test metrics for '{model_name}'.")
        return

    # ---------- build one-row DataFrame ----------------------------------- #
    row = {
        "model_name": model_name,
        "stage": results.get("model_stage"),
        "t_clip": results.get("run_config_metrics", {}).get("t_clip"),
        **{new: metrics.get(old) for old, new in _METRIC_MAP.items()},
        **{
            k: results.get("inference_speed_stats", {}).get(k)
            for k in _SPEED_KEYS
        },
    }
    df_new = pd.DataFrame([row])

    # ---------- merge with existing CSV (if any) -------------------------- #
    if csv_path.is_file():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
        df = (
            df.sort_values(["stage", "t_clip"])
              .drop_duplicates(subset=["stage", "t_clip"], keep="last")
              .reset_index(drop=True)
        )
        print(f"[save_results_to_csv] Updated '{csv_path}' for '{model_name}'.")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = df_new
        print(f"[save_results_to_csv] Created '{csv_path}' for '{model_name}'.")

    # ---------- apply required rounding ----------------------------------- #
    acc_cols = list(_METRIC_MAP.values())
    fps_cols = ["fps"]
    latency_cols = ["avg_latency_ms_batch", "avg_latency_ms_frame", "avg_time_ms_batch"]

    for col in acc_cols:
        if col in df:
            df[col] = df[col].round(3)

    for col in fps_cols + latency_cols:
        if col in df:
            df[col] = df[col].round(1)

    # ---------- write out -------------------------------------------------- #
    df.to_csv(csv_path, index=False)