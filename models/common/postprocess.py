# detection_postprocess.py

import torch
from torchvision.ops import boxes as box_ops

@torch.no_grad()
def postprocess_detections(
    head_outputs_tuple: tuple[torch.Tensor, torch.Tensor],
    anchors_per_frame_input: list[torch.Tensor],
    image_shapes_flat: list[tuple[int, int]],
    box_coder: torch.nn.Module,
    num_classes: int,
    score_thresh: float,
    nms_thresh: float,
    detections_per_img: int,
) -> list[dict[str, torch.Tensor]]:
    """
    Universal postprocessing for detections from a RetinaNet-style head. Used in TSMRetinaNet and THANOS models.

    Combines:
    - Considers all anchor-class pairs for candidate generation.
    - Uses class-aware batched NMS.
    - Clips boxes to image boundaries before NMS.

    :param head_outputs_tuple: Classification logits [B*T_clip, #Anchors, #Classes] and bbox regressions [B*T_clip, #Anchors, 4]
    :param anchors_per_frame_input: List of anchor tensors per frame (length B*T_clip), each [#anchors, 4]
    :param image_shapes_flat: List of original image sizes (height, width) per frame (length B*T_clip)
    :param box_coder: Instance to decode bbox regressions
    :param num_classes: Total number of classes (including background)
    :param score_thresh: Minimum score to keep a detection
    :param nms_thresh: IoU threshold for NMS
    :param detections_per_img: Max detections per image

    :return: List of dicts (length B*T_clip) with:
        boxes: [N, 4]
        scores: [N]
        labels: [N]
    """
    cls_logits, bbox_regression = head_outputs_tuple
    num_images = cls_logits.size(0)  # B*t_clip

    detections = []

    for i in range(num_images):
        # Per-image data
        pred_logits_this_image = cls_logits[i]          # [#anchors, #classes]
        pred_regression_this_image = bbox_regression[i]  # [#anchors, 4]
        anchors_this_image = anchors_per_frame_input[i]  # same
        original_image_shape = image_shapes_flat[i]      # (H, W)

        # 1. decode boxes from reg and anchors
        decoded_boxes = box_coder.decode_single(pred_regression_this_image, anchors_this_image)

        # 2. clip boxes to image dims
        current_image_size_tensor = torch.tensor(original_image_shape, device=decoded_boxes.device, dtype=torch.float)
        decoded_boxes = box_ops.clip_boxes_to_image(decoded_boxes, current_image_size_tensor)

        # 3. get scores
        scores_all_classes = torch.sigmoid(pred_logits_this_image)  # [#anchors, #classes]

        # 4.multi-class nms: consider each class prediction for each anchor independently.
        # not entirely necessary for binary detection but is useul for any extension

        # expand boxes: [#anchors, 4] -> [#anchors, #classes, 4] -> [#anchors*#classes, 4]
        candidate_boxes = decoded_boxes.unsqueeze(1).expand(-1, num_classes, -1)
        candidate_boxes = candidate_boxes.reshape(-1, 4)

        # flatten scores: [#anchors, #classes] -> [#anchors*#classes]
        candidate_scores = scores_all_classes.reshape(-1)

        candidate_labels = torch.arange(num_classes, device=scores_all_classes.device)
        candidate_labels = candidate_labels.view(1, -1).expand_as(scores_all_classes).reshape(-1) # [#anchors*#classes]

        # 5. filter on score thresh and non-bg class
        keep_idxs = torch.where(
            (candidate_scores >= score_thresh) &
            (candidate_labels != 0)  # exclude background cls
        )[0]

        candidate_boxes = candidate_boxes[keep_idxs]
        candidate_scores = candidate_scores[keep_idxs]
        candidate_labels = candidate_labels[keep_idxs]

        # case with no detections after filtering
        if candidate_boxes.numel() == 0:
            detections.append({
                "boxes": torch.empty((0, 4), device=candidate_boxes.device, dtype=candidate_boxes.dtype),
                "scores": torch.empty((0,), device=candidate_scores.device, dtype=candidate_scores.dtype),
                "labels": torch.empty((0,), device=candidate_labels.device, dtype=candidate_labels.dtype),
            })
            continue

        # 6. apply class-aware nms
        keep_after_nms = box_ops.batched_nms(candidate_boxes, candidate_scores, candidate_labels, nms_thresh)

        # 7. keep top detections_per_img detections
        if detections_per_img > 0: # detections_per_img can be -1 to keep all
            keep_after_nms = keep_after_nms[:detections_per_img]

        final_boxes = candidate_boxes[keep_after_nms]
        final_scores = candidate_scores[keep_after_nms]
        final_labels = candidate_labels[keep_after_nms]

        detections.append({
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels,
        })

    return detections