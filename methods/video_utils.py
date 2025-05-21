# video_utils.py

import torch
from torchvision.ops import box_iou

def split_frames_into_subsegments(
        frames_group: list['XCAImage'],
        iou_threshold_for_split: float,
    ) -> list[list['XCAImage']]:
    """
    Splits frames (from a single video) into consistent subsegments based on GT bbox IoU.

    May 21 Note: buggy code, lost ~10 videos
    """
    if len(frames_group) < 2: # not enough frames
        return [frames_group]

    all_subsegments = []
    current_subsegment = [frames_group[0]]

    for i in range(len(frames_group) - 1):
        frame_t = frames_group[i]
        frame_tplus1 = frames_group[i+1]
        if frame_t.bbox is None or frame_tplus1.bbox is None:
            all_subsegments.append(current_subsegment)
            current_subsegment = [frame_tplus1]
            continue

        # ugly type checking
        gt_box_t_list = (frame_t.bbox if isinstance(frame_t.bbox, list) and len(frame_t.bbox) == 4 and not isinstance(frame_t.bbox[0], tuple) else None)
        gt_box_tplus1_list = frame_tplus1.bbox if isinstance(frame_tplus1.bbox, list) and len(frame_tplus1.bbox) == 4 and not isinstance(frame_tplus1.bbox[0], tuple) else None

        if gt_box_t_list is None or gt_box_tplus1_list is None: # problem with bbox format or it's actually a list of bboxes
            all_subsegments.append(current_subsegment)
            current_subsegment = [frame_tplus1]
            continue

        # check for valid box dim (area > 0)
        if not (gt_box_t_list[2] > gt_box_t_list[0] and gt_box_t_list[3] > gt_box_t_list[1] and \
                gt_box_tplus1_list[2] > gt_box_tplus1_list[0] and gt_box_tplus1_list[3] > gt_box_tplus1_list[1]):
            all_subsegments.append(current_subsegment)
            current_subsegment = [frame_tplus1]
            continue

        gt_box_t_tensor = torch.tensor(gt_box_t_list, dtype=torch.float).unsqueeze(0)
        gt_box_tplus1_tensor = torch.tensor(gt_box_tplus1_list, dtype=torch.float).unsqueeze(0)

        iou = 0.0
        try:
            iou_val = box_iou(gt_box_t_tensor, gt_box_tplus1_tensor)
            if iou_val.numel() > 0:
                iou = iou_val.item()
        except Exception: # should not happen
            iou = 0.0

        if iou < iou_threshold_for_split:
            all_subsegments.append(current_subsegment) # finalize current subsegment
            current_subsegment = [frame_tplus1] # start new one
        else:
            current_subsegment.append(frame_tplus1)

    if current_subsegment: # add the last sub
        all_subsegments.append(current_subsegment)

    # filter empty subsegments if any ()
    return [seg for seg in all_subsegments if seg]