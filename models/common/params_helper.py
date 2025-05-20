# params_helper.py
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
    elif hasattr(model, 'head'):  # TSMRetinaNet, THANOSDetector
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
        if model_name == "THANOSDetector":
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
                backbone_base_prefix + "fpn."):  # model.backbone.fpn (EfficientNetFPNBackbone, THANOS)
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