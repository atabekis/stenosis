# util.py

# Python imports
import os
import torch
import pickle
import inspect
import datetime

# Local imports
from config import CACHE_DIR, CACHED_DATA_DIR, CACHED_MODELS_DIR, USE_CACHE, DEVICE

COLORS = {
    "timestamp": "\033[92m", # green
    "filename": "\033[94m", # blue
    "funcname": "\033[95m", # pink
    "reset": "\033[0m" # reset the color
}


def log(*args, verbose=True, **kwargs):
    """
    Take regular print string as input, add timestamp, filename, function or class name where log() was called from.
    """
    frame = inspect.currentframe().f_back
    caller = frame.f_code.co_name
    class_name = frame.f_locals.get('self', None).__class__.__name__ if 'self' in frame.f_locals else None
    func_name = f'{class_name}.{caller}' if class_name else caller

    filename = os.path.basename(frame.f_code.co_filename).replace('.py', '')
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')

    # apply colors
    c_timestamp = f"{COLORS['timestamp']}[{timestamp}]{COLORS['reset']}"
    c_filename = f"{COLORS['filename']}{filename}{COLORS['reset']}"
    c_funcname = f"{COLORS['funcname']}{func_name}{COLORS['reset']}"


    message = ''.join(map(str, args))

    if verbose:
        print(f"{c_timestamp}[{c_filename}.{c_funcname}] \"{message}\"", **kwargs)


def cache_data(filename: str, data: any = None,
               cache_data_dir = CACHED_DATA_DIR, use_cache = USE_CACHE,
               verbose: bool = True) -> any:
    """
    If data is provided, save it to the cache file path and return the data
    otherwise load and return the data from the cached file
    This function uses the CACHED_DATA_PATH which points to /project_root/.cache/data
    """
    if data is None:
        if os.path.exists(cache_data_dir / filename) and use_cache:
            log(f"Preprocessed XCA Images found, loading from cache: {filename}", verbose=verbose)
            with open(cache_data_dir / filename, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    else:
        if use_cache:
            log(f"Saving data to cache: {filename}", verbose=verbose)
            with open(cache_data_dir / filename, 'wb') as f:
                pickle.dump(data, f)
            return data


def to_numpy(*tensors):
    """this method is used to convert tensors back into numpy arrays"""
    return [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]


def to_device(data, device=DEVICE, non_blocking=True):
    """
    Recursively moves tensors in data to the given device.
    Supports tensors, dicts, and lists.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking=non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device, non_blocking=non_blocking) for item in data]
    else:
        return data
