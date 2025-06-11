# visualize.py

# Python imports
import cv2
import imageio
import matplotlib.pyplot as plt

# Torch - helper
import torch
from torchvision.ops import box_iou


# Local imports
from methods.reader import Reader
from config import IOU_THRESH_METRIC


def get_tp_fp(
    pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
    target_boxes: torch.Tensor, target_labels: torch.Tensor,
    iou_threshold: float
) -> tuple[list[str], list[str]]:
    """
    Determine TP/FP status for each predicted box:
    - A prediction is a TP if its IoU with an unmatched GT box of the same class ≥ iou_threshold.
    - Each GT can be matched at most once by the highest‐scoring prediction.
    """
    num_preds = pred_boxes.size(0)
    num_targets = target_boxes.size(0)

    device = pred_boxes.device
    target_boxes = target_boxes.to(device)
    target_labels = target_labels.to(device)

    if num_preds == 0:
        return [], ["FN"] * num_targets

    if num_targets == 0:
        return ["FP"] * num_preds, []

    pred_status = ["FP"] * num_preds
    gt_status = ["FN"] * num_targets
    matched_gt = torch.zeros(num_targets, dtype=torch.bool, device=device)
    iou_matrix = box_iou(pred_boxes, target_boxes)

    for pred_idx in torch.argsort(pred_scores, descending=True):
        label = pred_labels[pred_idx]
        label_mask = (target_labels == label) & ~matched_gt
        if not label_mask.any():
            continue

        ious = iou_matrix[pred_idx]
        masked_ious = ious.masked_fill(~label_mask, -1.0)
        best_iou, best_gt_idx = masked_ious.max(dim=0)

        if best_iou >= iou_threshold:
            pred_status[pred_idx.item()] = "TP"
            gt_status[best_gt_idx.item()] = "TP"
            matched_gt[best_gt_idx] = True

    return pred_status, gt_status



def visualize_predictions(xca_image, result_entry: dict, iou_threshold: float = IOU_THRESH_METRIC,
                          figsize=(12, 12), ax=None, return_ax: bool = False):
    """
    Overlay ground‐truth (GT) and predicted boxes on the image:
    """
    img = xca_image.image.copy()
    meta = result_entry["metadata"]

    if img.ndim == 2:  # images coming in from reader are 1-ch, we need to create mock bgr
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # get gt data
    gt_boxes, gt_labels = [], []

    for tgt in result_entry.get("targets", []):
        boxes, labels = tgt.get("boxes"), tgt.get("labels")

        if boxes is not None and boxes.numel() > 0:
            gt_boxes.append(boxes); gt_labels.append(labels)


    if gt_boxes:
        device_tgt = gt_boxes[0].device
        gt_boxes = torch.cat([b.to(device_tgt) for b in gt_boxes], dim=0)
        gt_labels = torch.cat([l.to(device_tgt) for l in gt_labels], dim=0)
    else:
        device_tgt = torch.device("cpu")
        gt_boxes = torch.empty((0, 4), device=device_tgt)
        gt_labels = torch.empty((0,), dtype=torch.long, device=device_tgt)

    # get pred data
    pred_boxes, pred_scores, pred_labels = [], [], []
    for pred in result_entry.get("predictions", []):
        boxes, scores, labels = pred.get("boxes"), pred.get("scores"), pred.get("labels")

        if boxes is not None and boxes.numel() > 0:
            pred_boxes.append(boxes); pred_scores.append(scores); pred_labels.append(labels)

    if pred_boxes:
        device_pred = pred_boxes[0].device
        pred_boxes = torch.cat([b.to(device_pred) for b in pred_boxes], dim=0)
        pred_scores = torch.cat([s.to(device_pred) for s in pred_scores], dim=0)
        pred_labels = torch.cat([l.to(device_pred) for l in pred_labels], dim=0)
    else:
        device_pred = torch.device("cpu")
        pred_boxes = torch.empty((0, 4), device=device_pred)
        pred_scores = torch.empty((0,), device=device_pred)
        pred_labels = torch.empty((0,), dtype=torch.long, device=device_pred)

    # if there are multiple predictions, select only the one with the highest score
    if pred_boxes.size(0) > 1:
        best_score_idx = torch.argmax(pred_scores)

        pred_boxes = pred_boxes[best_score_idx].unsqueeze(0)
        pred_scores = pred_scores[best_score_idx].unsqueeze(0)
        pred_labels = pred_labels[best_score_idx].unsqueeze(0)

    # compute tp/fp for preds
    tp_fp_status =  get_tp_fp(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, iou_threshold
    )

    # settings for drawing
    COLORS = {
        # "GT": (92, 220, 249), # naples yellow
        "GT": (62, 162, 249), # naples yellow
        "TP": (80, 175, 76), # pigment green
        "FP": (53, 37, 228), # poppy - red
        "META": (255, 255, 255), # white
    }
    THICKNESS = {"GT": 3, "PRED": 2}
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FS_BOX = 0.5
    FT_BOX = 2
    FS_META = 0.8
    FT_META = 2

    #  draw gt boxes
    for box in gt_boxes.cpu().int().numpy():
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["GT"], THICKNESS["GT"])
        text = "GT"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FS_BOX, FT_BOX)
        text_x = x1
        text_y = y1 - 6  # 6 px above top edge

        # if text would go off-image, place inside
        if text_y < th + baseline:
            text_y = y1 + th + baseline + 2
        cv2.putText(img, text, (text_x, text_y), FONT, FS_BOX, COLORS["GT"], FT_BOX, cv2.LINE_AA)

    # draw pred boxes
    for i in range(pred_boxes.size(0)):
        box = pred_boxes[i].cpu().int().numpy()
        x1, y1, x2, y2 = box
        score = pred_scores[i].item()
        status = tp_fp_status[i][0]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[status], THICKNESS["PRED"])

        # score text
        text = f"{score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FS_BOX, FT_BOX)

        text_x = x1
        text_y = y2 + th + baseline + 2  # 2 px below the bottom edge

        cv2.putText(img, text, (text_x, text_y), FONT, FS_BOX, COLORS[status], FT_BOX, cv2.LINE_AA)

    # draw meta
    pid = meta.get("patient_id", "N/A")
    vid = meta.get("video_id", "N/A")
    fn = meta.get("frame_nr", "N/A")
    meta_text = f"Patient: {pid}, Video: {vid}, Frame: {fn}"
    cv2.putText(img, meta_text, (20, 780), FONT, FS_META, COLORS["META"], FT_META, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mock rgb

    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize)
    else:
        ax_plot = ax

    ax_plot.imshow(img_rgb)
    ax_plot.axis("off")


    return ax_plot if return_ax else plt


if __name__ == "__main__":
    from torch import tensor

    r = Reader(dataset_dir='both')
    res = {'metadata': {'patient_id': 39, 'video_id': 4, 'frame_nr': 27},
 'targets': [{'boxes': tensor([[240., 202., 267., 226.]]),
   'labels': tensor([1])}],
 'predictions': [{'boxes': tensor([[240.0085, 200.8250, 269.8293, 226.4715]]),
   'scores': tensor([0.8199]),
   'labels': tensor([1])}],
 'mask': tensor([True])}

    meta = res['metadata']
    res_img = r.get(patient_id=meta['patient_id'], video_id=meta['video_id'], frame_nr=meta['frame_nr'])
    yar = visualize_predictions(xca_image=res_img, result_entry=res)
    yar.show()
