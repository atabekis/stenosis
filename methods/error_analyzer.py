# error_analyzer.py

import torch
from collections import Counter

from config import IOU_THRESH_METRIC
from methods.visualize import get_tp_fp


def _extract_and_prepare_frame_tensors(preds: list[dict], targets: list[dict] ) -> tuple:
    """
    extract and concat boxes, scores, labels from preds and targets into flat tensors
    """
    device = None  # will hold inferred device

    # helper to collect and concatenate or create empty
    def _gather(dicts, keys, empty_shape, dtype):
        cols = {k: [] for k in keys}
        nonlocal device
        for d in dicts:
            boxes = d.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                device = device or boxes.device  # infer device from first non-empty
                for k in keys:
                    cols[k].append(d[k])
        device_final = device or torch.device("cpu")
        tensors = []
        for k in keys:
            lst = cols[k]
            if lst:
                tensors.append(torch.cat(lst, dim=0).to(device_final))
            else:
                tensors.append(torch.empty(empty_shape if k == "boxes" else empty_shape[:1],
                                           dtype=dtype, device=device_final))
        return tuple(tensors)

    # process predictions boxes (n,4), scores (n,), labels (n,)
    pred_boxes, pred_scores, pred_labels = _gather(
        preds,
        ("boxes", "scores", "labels"),
        empty_shape=(0, 4),
        dtype=torch.float32
    )

    # process targets  boxes (m,4), labels (m,)
    target_boxes, target_labels = _gather(
        targets,
        ("boxes", "labels"),
        empty_shape=(0, 4),
        dtype=torch.float32
    )

    return pred_boxes, pred_scores, pred_labels, target_boxes, target_labels



import torch

def _analyze_frame_errors(preds: list[dict], targets: list[dict], iou_threshold: float) -> dict[str, int]:
    """analyze one frame: count true positives, false positives, false negatives"""
    # extract tensors for preds and targets
    pred_boxes, pred_scores, pred_labels, target_boxes, target_labels = (
        _extract_and_prepare_frame_tensors(preds, targets)
    )
    # get per-box statuses
    pred_status_list, gt_status_list = get_tp_fp(
        pred_boxes, pred_scores, pred_labels,
        target_boxes, target_labels,
        iou_threshold
    )
    # count statuses
    pred_counts = Counter(pred_status_list)
    gt_counts   = Counter(gt_status_list)

    return {
        'num_tp': pred_counts['TP'],
        'num_fp': pred_counts['FP'],
        'num_fn': gt_counts['FN'],
        'num_gt': target_boxes.size(0),
        'num_preds': pred_boxes.size(0)
    }


def _get_top1_prediction(preds: list[dict]) -> tuple:
    """get top 1 pred dict, its score, label, and box area"""
    top_preds = []
    score = label = area = None
    if preds and preds[0]['scores'].numel() > 0:
        pd = preds[0]
        idx = torch.argmax(pd['scores']) # pick highest score
        box = pd['boxes'][idx]
        sc = pd['scores'][idx]
        lb = pd['labels'][idx]
        top_preds = [{
            'boxes': box.unsqueeze(0),
            'scores': sc.unsqueeze(0),
            'labels': lb.unsqueeze(0)
        }]
        score = sc.item() # scalar score
        label = lb.item() # scalar label
        area = ((box[2] - box[0]) * (box[3] - box[1])).item() # compute area
    return top_preds, score, label, area


def _get_area_from_first_box(targets: list[dict]) -> float | None:
    """get area of first gt box or none"""
    if targets and targets[0]['boxes'].numel() > 0:
        b = targets[0]['boxes'][0]
        return ((b[2] - b[0]) * (b[3] - b[1])).item() # compute area
    return None

def analyze_model_predictions(models_results_dict: dict[str, list[dict]], iou_threshold: float = IOU_THRESH_METRIC):
    """
    process model results: only top-1 per frame for tp/fp/fn, include pred/gt areas
    """
    all_records = []  # collect outputs

    for model_id, entries in models_results_dict.items():
        if not entries:
            print(f"warning: no result entries for model '{model_id}', skipping")
            continue

        for entry in entries:
            meta = entry.get('metadata', {}) # frame/clip info
            single = (
                'frame_nr' in meta
                and 'num_frames_in_clip' not in meta
                and 'start_frame_orig' not in meta
            )
            clip = 'num_frames_in_clip' in meta or 'start_frame_orig' in meta

            if single:
                frame_nr = meta.get('frame_nr')
                preds = entry.get('predictions', [])
                targets = entry.get('targets', [])
                top_preds, best_score, best_label, pred_area = _get_top1_prediction(preds)
                gt_area = _get_area_from_first_box(targets)
                errors = _analyze_frame_errors(top_preds, targets, iou_threshold) # count errors
                rec = {
                    'model_id': model_id,
                    'patient_id': meta.get('patient_id'),
                    'video_id': meta.get('video_id'),
                    'item_type': 'frame',
                    'frame_nr': frame_nr,
                    'clip_start_frame': None,
                    'clip_end_frame': None,
                    **errors,
                    'best_pred_score': best_score,
                    'best_pred_label': best_label,
                    'best_pred_frame_nr': frame_nr,
                    'pred_box_area': pred_area,
                    'gt_box_area': gt_area,
                }
                all_records.append(rec)

            elif clip:
                num_frames = meta.get('num_frames_in_clip', 0)
                start = meta.get('start_frame_orig', 0)
                preds_list = entry.get('predictions', [])
                targets_list = entry.get('targets', [])
                mask = entry.get('mask', torch.ones(num_frames, dtype=torch.bool))
                if isinstance(mask, list):
                    mask = torch.tensor(mask, dtype=torch.bool) # convert list mask

                agg = {'num_tp': 0, 'num_fp': 0, 'num_fn': 0, 'num_gt': 0, 'num_preds': 0}
                best_score = -1.0
                best_label = best_frame = None
                best_pred_area = best_gt_area = None

                for i in range(num_frames):
                    if (
                        i >= len(preds_list)
                        or i >= len(targets_list)
                        or i >= mask.size(0)
                        or not mask[i]
                    ):
                        continue # skip invalid or masked

                    f_preds = [preds_list[i]]
                    f_targets = [targets_list[i]]

                    top_preds, score, label, pred_area = _get_top1_prediction(f_preds)
                    gt_area = _get_area_from_first_box(f_targets)

                    if score is not None and score > best_score:
                        best_score, best_label = score, label
                        best_frame = start + i  # absolute frame no.
                        best_pred_area, best_gt_area = pred_area, gt_area

                    errs = _analyze_frame_errors(top_preds, f_targets, iou_threshold)

                    for k in agg:
                        agg[k] += errs[k]  # accumulate errors

                if best_score < 0:
                    best_score = None  # no preds found
                rec = {
                    'model_id': model_id,
                    'patient_id': meta.get('patient_id'),
                    'video_id': meta.get('video_id'),
                    'item_type': 'clip',
                    'frame_nr': None,
                    'clip_start_frame': meta.get('start_frame_orig'),
                    'clip_end_frame': meta.get('end_frame_orig'),
                    **agg,
                    'best_pred_score': best_score,
                    'best_pred_label': best_label,
                    'best_pred_frame_nr': best_frame,
                    'pred_box_area': best_pred_area,
                    'gt_box_area': best_gt_area,
                }
                all_records.append(rec)

            else:
                print(f"warning: unrecognized format for model '{model_id}', skipping")

    return all_records



