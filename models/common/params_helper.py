# params_helper.py
import re
import torch
import torch.nn as nn

from util import log

def get_optimizer_param_groups(model, config):
    """Creates parameter groups for the optimizer with differential learning rates"""

    if not config.get('differential_lr', {}).get('enabled', False):
        log("Differential learning rates disabled. Using base_lr for all parameters.")
        return model.parameters()

    base_lr = config['base_lr']
    lr_config = config['differential_lr']

    lr_backbone = lr_config.get('lr_backbone', base_lr * 0.1)
    lr_fpn = lr_config.get('lr_fpn', base_lr * 0.5)
    lr_transformer_thanos = lr_config.get('lr_transformer_thanos', base_lr * 0.5)
    lr_regression_head = lr_config.get('lr_regression_head', base_lr)
    lr_classification_head = lr_config.get('lr_classification_head', base_lr * 0.1)
    lr_other = lr_config.get('lr_other', base_lr * 0.1)

    param_groups = {
        "backbone": {"params": [], "lr": lr_backbone, "name": "backbone"},
        "fpn": {"params": [], "lr": lr_fpn, "name": "fpn"},
        "transformer_thanos": {"params": [], "lr": lr_transformer_thanos, "name": "transformer_thanos"},
        "regression_head": {"params": [], "lr": lr_regression_head, "name": "regression_head"},
        "classification_head": {"params": [], "lr": lr_classification_head, "name": "classification_head"},
        "other": {"params": [], "lr": lr_other, "name": "other"},
    }

    model_name = model.__class__.__name__
    log(f"Setting up differential LRs for model: {model_name}")

    head_prefix = ""
    if hasattr(model, 'retinanet') and hasattr(model.retinanet, 'head'):  # FPNRetinaNet
        head_prefix = "retinanet.head."
        log(f"Determined head_prefix: {head_prefix}")
    elif hasattr(model, 'head'):  # TSMRetinaNet, THANOS
        head_prefix = "head."
        log(f"Determined head_prefix: {head_prefix}")
    else:
        log("Warning: Could not determine head prefix for differential LRs. Head LRs might not be applied correctly.")

    # backbone structure: model.backbone.body or model.backbone (if it's the tsm_effnet.features)
    # FPN structure: model.backbone.fpn or model.fpn
    backbone_base_prefix = "backbone."  # Common prefix for most models


    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        assigned_to_group = False  # check if param was assigned

        # THANOS-specific layers first
        if model_name == "THANOS":
            if name.startswith("positional_embeddings.") or \
                    name.startswith("temporal_attention_blocks."):
                param_groups["transformer_thanos"]["params"].append(param)
                assigned_to_group = True

        if assigned_to_group:  # if assigned by THANOS specific, continue
            continue

        # head parameters
        if head_prefix and name.startswith(head_prefix + "classification_head."):
            param_groups["classification_head"]["params"].append(param)
            assigned_to_group = True
        elif head_prefix and name.startswith(head_prefix + "regression_head."):
            param_groups["regression_head"]["params"].append(param)
            assigned_to_group = True

        # FPN parameters
        elif name.startswith(
                backbone_base_prefix + "fpn."):  # model.backbone.fpn (FPNBackbone, THANOS)
            param_groups["fpn"]["params"].append(param)
            assigned_to_group = True
        # FPN as a direct attribute (TSMRetinaNet)
        elif name.startswith("fpn."):
            param_groups["fpn"]["params"].append(param)
            assigned_to_group = True

        # Backbone parameters
        elif name.startswith(backbone_base_prefix + "body."):
            param_groups["backbone"]["params"].append(param)
            assigned_to_group = True
        elif name.startswith(
                backbone_base_prefix) and not assigned_to_group:  # ensure not already assigned to FPN
            param_groups["backbone"]["params"].append(param)
            assigned_to_group = True

        # fallback unassigned parameters
        elif not assigned_to_group:
            log(f"  Parameter '{name}' assigned to 'other' group.")
            param_groups["other"]["params"].append(param)

    # filter groups with no params
    final_param_groups = []
    log("Optimizer parameter groups and learning rates:")
    total_params_in_groups = 0
    for group_name, group_data in param_groups.items():
        if group_data["params"]:
            final_param_groups.append(group_data)
            num_p = len(group_data['params'])
            total_params_in_groups += num_p
            log(f"  Group: {group_data['name']:<20}: Num Params: {num_p:<5}, LR: {group_data['lr']}")
        else:
            log(f"  Group: {group_data['name']:<20}: - No parameters assigned.")

    total_model_params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
    if total_params_in_groups != total_model_params_requiring_grad:
        log(f"Warning: Mismatch in parameter count. Total model params (req. grad): {total_model_params_requiring_grad}, Params in optimizer groups: {total_params_in_groups}")
        log("  This might indicate some parameters were missed or double-counted. Review prefix logic.")

    if not final_param_groups:
        log("Warning: No parameter groups were formed. Using all model parameters with base_lr.")
        return [{'params': model.parameters(), 'lr': base_lr}]

    return final_param_groups


def get_state_dict_from_ckpt(ckpt_path: str, model_key_prefix: str = 'model.') -> dict:
    """Loads a .ckpt checkpoint file and extracts the state_dict for the nn.Module"""
    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {ckpt_path}: {e}")

    # extract state_dict
    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' key not found in checkpoint: {ckpt_path}")

    state_dict = checkpoint["state_dict"]

    # return full state_dict if no prefix specified
    if not model_key_prefix:
        log(f"Returning full state_dict from {ckpt_path} with {len(state_dict)} initial keys.", verbose=True)
        return state_dict

    # filter and strip prefix
    filtered = {
        k[len(model_key_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(model_key_prefix)
    }

    if not filtered:
        sample = list(state_dict.keys())[:5]
        raise KeyError(
            f"Prefix '{model_key_prefix}' not found in keys of checkpoint {ckpt_path}."
            f" Sample keys: {sample}..."
        )

    log(f"Extracted {len(filtered)} entries with prefix '{model_key_prefix}' from {ckpt_path}.")
    return filtered



def thanos_load_weights(model: nn.Module, ckpt_path: str, ckpt_model_key_prefix: str = "model.", load_head_weights: bool = True):
    """
    Loads weights into THANOS model's backbone and optionally head from an FPNRetinaNet checkpoint.
    """
    def _load_module(module: nn.Module, source_weights: dict):
        tgt_state = module.state_dict()
        loaded = 0
        skipped_missing = 0
        skipped_shape = 0
        final_weights = {}

        for key, tgt_param in tgt_state.items():
            src_param = source_weights.get(key)
            if src_param is None:
                skipped_missing += 1
            else:
                if tgt_param.shape == src_param.shape:
                    final_weights[key] = src_param
                    loaded += 1
                else:
                    skipped_shape += 1

        module.load_state_dict(final_weights, strict=False)
        log(
            f"Module '{module.__class__.__name__}' | "
            f"Total keys: {len(tgt_state)}, "
            f"Loaded: {loaded}, "
            f"Skipped (missing): {skipped_missing}, "
            f"Skipped (shape): {skipped_shape}"
        )

    if not hasattr(model, "backbone"):
        log("THANOS Weight Load Error: model has no 'backbone' attribute.")
        return
    if load_head_weights and not hasattr(model, "head"):
        log("THANOS Weight Load Error: load_head_weights=True but model has no 'head' attribute.")
        return

    full_ckpt_state = get_state_dict_from_ckpt(ckpt_path, model_key_prefix=ckpt_model_key_prefix)
    if not full_ckpt_state:
        log(f"THANOS: Checkpoint '{ckpt_path}' yielded no usable weights.")
        return

    # 1. load the backbone
    backbone_prefixes = ["retinanet.backbone.", "backbone."]
    for prefix in backbone_prefixes:
        backbone_weights = {
            k[len(prefix):]: v
            for k, v in full_ckpt_state.items()
            if k.startswith(prefix)
        }
        if backbone_weights:
            _load_module(model.backbone, backbone_weights)
            break
    else:
        # no backbone weights found under any prefix
        _load_module(model.backbone, {})

    # 2. conditionally load head
    if load_head_weights:
        head_prefix = "retinanet.head."
        raw_head_weights = {
            k[len(head_prefix):]: v
            for k, v in full_ckpt_state.items()
            if k.startswith(head_prefix)
        }

        if raw_head_weights:
            remapped = {}
            pattern = re.compile(r"classification_head\.conv\.(conv|gn)(\d+)\.(.*)")  # need to do some ugly matching due
            for k_src, v_src in raw_head_weights.items():                             # to the custom cls head
                match = pattern.match(k_src)
                if match:
                    layer_type = match.group(1) # conv or gn
                    layer_idx = match.group(2) # e.g. '0'
                    suffix = match.group(3) # e.g. 'weight'
                    k_new = (
                        f"classification_head.conv_tower_layers."
                        f"{layer_idx}.{layer_type}{layer_idx}.{suffix}"
                    )
                    remapped[k_new] = v_src
                else:
                    remapped[k_src] = v_src

            _load_module(model.head, remapped)
        else:
            # no head weights found under prefix
            _load_module(model.head, {})


def get_adaptive_groupnorm_layer(default_num_gn_groups: int):
    """
    Returns a callable that creates a nn.GroupNorm layer with an adaptive number of groups
    """

    # common group numbers,
    _PREDEFINED_GROUP_OPTIONS = (32, 16, 8, 4, 2, 1)

    def _adaptive_group_norm_inner(num_channels: int):
        if num_channels <= 0:
            raise ValueError(f"Cannot apply GroupNorm to {num_channels} channels.")

        # if value is a divisor, use it
        if num_channels >= default_num_gn_groups > 0 == num_channels % default_num_gn_groups:
            groups = default_num_gn_groups
        else:
            # find the first largest option that divides num_channels.
            for g in _PREDEFINED_GROUP_OPTIONS:
                if g <= num_channels and num_channels % g == 0:
                    groups = g
                    break
            else:
                # fallback should only happen if num_channels < 1, but we guard above.
                groups = 1
                log(f"For {num_channels} channels, no predefined group option was suitable. Defaulting to 1 group.")

            # if we fell back to something other than the preferred, log it.
            if groups != default_num_gn_groups:
                log(f"For {num_channels} channels, using {groups} groups (default {default_num_gn_groups} was not suitable).", verbose=False)

        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)

    return _adaptive_group_norm_inner