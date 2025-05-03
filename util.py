# util.py

# Python imports
import os
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


def get_optimal_workers():
    cpu_count = os.cpu_count() or mp.cpu_count()
    # return max(1, cpu_count - 2)
    return 8


def to_numpy(*tensors):
    """this method is used to convert tensors back into numpy arrays"""
    return [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
