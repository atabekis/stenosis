# sca_utils.py
import torch
from torchvision.ops import box_iou
from typing import Optional, List, Dict  # Added List, Dict for clarity


def calculate_iou_individual(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculates IoU for two individual bounding boxes.
    Args:
        box1 (torch.Tensor): A tensor of shape (4,) [x1, y1, x2, y2].
        box2 (torch.Tensor): A tensor of shape (4,) [x1, y1, x2, y2].
    Returns:
        float: The IoU value.
    """
    if box1.ndim == 1:
        box1 = box1.unsqueeze(0)
    if box2.ndim == 1:
        box2 = box2.unsqueeze(0)

    # Ensure boxes are valid (e.g., x2 > x1, y2 > y1) before IoU if necessary,
    # though box_iou might handle some degeneracies.
    # For simplicity, assuming valid boxes from model output.
    if box1.numel() != 4 or box2.numel() != 4:  # Basic check
        return 0.0

    iou_matrix = box_iou(box1, box2)
    if iou_matrix.numel() == 0: return 0.0
    return iou_matrix[0, 0].item()


def apply_sequence_consistency_alignment(
        sequence_detections: List[Dict[str, torch.Tensor]],
        t_iou: float = 0.3,
        t_frame: int = 3,
        t_score_interp: float = 0.1,
        max_frame_gap_for_linking: int = 1,
        debug: bool = False
) -> List[Dict[str, torch.Tensor]]:
    """
    Applies Sequence Consistency Alignment (SCA) to a sequence of detections.


    :param sequence_detections: list of detection dictionaries, one for each frame in the sequence.
                                Each dict contains 'boxes', 'scores', 'labels' tensors.
    :param t_iou: min. iou threshold to link detections into a track.
    :param t_frame: min number of frames a track must span to be kept.
    :param t_score_interp: score to assign to interpolated boxes.
    :param max_frame_gap_for_linking: Max number of consecutive frames with no match to skip when trying to extend a track.
                                         0 means only link consecutive frames.
                                         1 means can skip 1 frame, etc.
    :return (hopefully) refined predictions
    """
    num_total_frames = len(sequence_detections)
    if num_total_frames == 0:
        return []

    device = 'cpu'  # for some reason after epoch 4-5, torch gets mad at me for device type
    for frame_det_init in sequence_detections:
        if 'boxes' in frame_det_init and frame_det_init['boxes'].numel() > 0:
            device = frame_det_init['boxes'].device
            break
        elif 'scores' in frame_det_init and frame_det_init['scores'].numel() > 0:
            device = frame_det_init['scores'].device
            break
        elif 'labels' in frame_det_init and frame_det_init['labels'].numel() > 0:
            device = frame_det_init['labels'].device
            break

    if debug: print(
        f"\nSCA DEBUG (R): Determined device: {device}. Starting Track Building. Frames: {num_total_frames}, t_iou: {t_iou}, max_gap: {max_frame_gap_for_linking}")

    tracks: List[List[tuple[int, Dict[str, torch.Tensor]]]] = []
    used_detections_coords: set[tuple[int, int]] = set()  # (frame_idx, original_idx_in_frame_detections)

    for frame_idx in range(num_total_frames):
        current_frame_dets = sequence_detections[frame_idx]
        num_dets_in_current_frame = current_frame_dets['boxes'].size(0)

        for original_idx in range(num_dets_in_current_frame):
            if (frame_idx, original_idx) in used_detections_coords:
                continue

            # start a new track with the current det
            current_det_instance = {
                'box': current_frame_dets['boxes'][original_idx].clone().to(device),
                'score': current_frame_dets['scores'][original_idx].clone().to(device),
                'label': current_frame_dets['labels'][original_idx].clone().to(device),
            }

            new_track: List[tuple[int, Dict[str, torch.Tensor]]] = [(frame_idx, current_det_instance)]
            used_detections_coords.add((frame_idx, original_idx))
            if debug: print(
                f"SCA DEBUG (R): Frame {frame_idx}, Det {original_idx}: Starting new track: Label {current_det_instance['label'].item()}, Box {current_det_instance['box'].cpu().numpy().round(1)}")

            current_last_in_track_frame_idx = frame_idx
            current_last_in_track_instance = current_det_instance

            while True:  # track extension
                found_match_in_search_window = False
                best_overall_match_frame_idx = -1
                best_overall_match_instance = None
                best_overall_match_original_idx = -1

                # search window: from next frame up to max_frame_gap
                for frame_offset in range(1, max_frame_gap_for_linking + 2):  # +1 for gap, +1 for Python range
                    check_frame_idx = current_last_in_track_frame_idx + frame_offset
                    if check_frame_idx >= num_total_frames:
                        break

                    next_frame_dets_to_check = sequence_detections[check_frame_idx]
                    num_dets_in_next_to_check = next_frame_dets_to_check['boxes'].size(0)
                    if debug: print(
                        f"SCA DEBUG (R):   Track from F{current_last_in_track_frame_idx} (L{current_last_in_track_instance['label'].item()}). Checking F{check_frame_idx} (offset {frame_offset}, {num_dets_in_next_to_check} dets)")

                    # find the best iou match within this check_frame_idx
                    current_check_frame_best_iou = -1.0
                    current_check_frame_best_match_instance = None
                    current_check_frame_best_original_idx = -1

                    for cand_orig_idx in range(num_dets_in_next_to_check):
                        if (check_frame_idx, cand_orig_idx) in used_detections_coords:
                            continue

                        cand_det_instance = {
                            'box': next_frame_dets_to_check['boxes'][cand_orig_idx].to(device),
                            'score': next_frame_dets_to_check['scores'][cand_orig_idx].to(device),
                            'label': next_frame_dets_to_check['labels'][cand_orig_idx].to(device),
                        }

                        if current_last_in_track_instance['label'].item() == cand_det_instance['label'].item():
                            iou = calculate_iou_individual(current_last_in_track_instance['box'],
                                                           cand_det_instance['box'])
                            if debug: print(
                                f"SCA DEBUG (R):     Candidate in F{check_frame_idx}, Det {cand_orig_idx} (L{cand_det_instance['label'].item()}). IoU: {iou:.3f}")
                            if iou >= t_iou and iou > current_check_frame_best_iou:  # find best match in frame
                                current_check_frame_best_iou = iou
                                current_check_frame_best_match_instance = cand_det_instance
                                current_check_frame_best_original_idx = cand_orig_idx

                    if current_check_frame_best_match_instance is not None:
                        # match found in check_frame_idx → this is link for this step.
                        best_overall_match_frame_idx = check_frame_idx
                        best_overall_match_instance = current_check_frame_best_match_instance
                        best_overall_match_original_idx = current_check_frame_best_original_idx
                        found_match_in_search_window = True
                        if debug: print(
                            f"SCA DEBUG (R):     Found link in F{check_frame_idx}, Det {best_overall_match_original_idx} with IoU {current_check_frame_best_iou:.3f}. Will link to this.")
                        break

                if found_match_in_search_window and best_overall_match_instance is not None:
                    # clone the instance that will be added to the track
                    cloned_match = {
                        'box': best_overall_match_instance['box'].clone(),
                        'score': best_overall_match_instance['score'].clone(),
                        'label': best_overall_match_instance['label'].clone(),
                    }
                    new_track.append((best_overall_match_frame_idx, cloned_match))
                    used_detections_coords.add((best_overall_match_frame_idx, best_overall_match_original_idx))

                    current_last_in_track_instance = cloned_match
                    current_last_in_track_frame_idx = best_overall_match_frame_idx
                    if debug: print(
                        f"SCA DEBUG (R):   Successfully linked to F{best_overall_match_frame_idx}, Det {best_overall_match_original_idx}. Track len: {len(new_track)}.")
                else:

                    if debug: print(
                        f"SCA DEBUG (R):   No link found within max_gap+1 for F{current_last_in_track_frame_idx}. Ending current track extension.")
                    break

            tracks.append(new_track)
            if debug: print(
                f"SCA DEBUG (R): Finalized track: {[(f, d['label'].item()) for f, d in new_track]}. Length: {len(new_track)}")

    if debug:
        print(f"\nSCA DEBUG (R): --- Built Tracks ({len(tracks)} total) ---")
        for i, track in enumerate(tracks):
            print(
                f"SCA DEBUG (R): Track {i} (length {len(track)}): {[(f_idx, det_inst['label'].item(), det_inst['box'].cpu().numpy().round(1)) for f_idx, det_inst in track]}")

    # 2. filter tracks
    kept_tracks = [track for track in tracks if len(track) >= t_frame]
    if debug: print(f"\nSCA DEBUG (R): --- Kept Tracks ({len(kept_tracks)} after t_frame={t_frame}) ---")
    if debug:
        for i, track in enumerate(kept_tracks):
            print(
                f"SCA DEBUG (R): Kept Track {i} (length {len(track)}): {[(f_idx, det_inst['label'].item()) for f_idx, det_inst in track]}")

    #  3. interpolate missing dets in kept tracks
    final_processed_tracks: List[List[tuple[int, Dict[str, torch.Tensor]]]] = []
    if debug: print(f"\nSCA DEBUG (R): --- Interpolating Kept Tracks ---")
    for track_idx, track in enumerate(kept_tracks):
        if not track: continue

        interpolated_track_elements: List[tuple[int, Dict[str, torch.Tensor]]] = []
        track.sort(key=lambda x: x[0])

        existing_detections_map = {f_idx: det_inst for f_idx, det_inst in track}
        min_frame_in_track = track[0][0]
        max_frame_in_track = track[-1][0]

        if debug: print(
            f"SCA DEBUG (R):   Interpolating Track {track_idx}, spanning frames {min_frame_in_track}-{max_frame_in_track}")

        for target_frame_idx in range(min_frame_in_track, max_frame_in_track + 1):
            if target_frame_idx in existing_detections_map:
                if debug: print(f"SCA DEBUG (R):     Frame {target_frame_idx}: Using existing detection.")
                interpolated_track_elements.append(
                    (target_frame_idx, {k: v.to(device) for k, v in existing_detections_map[target_frame_idx].items()}))

            else:  # need to interpolate
                prev_det_tuple, succ_det_tuple = None, None
                # find closest PRECEDING detection IN THE ORIGINAL TRACK
                for f_idx_orig, det_inst_orig in reversed(track):
                    if f_idx_orig < target_frame_idx:
                        prev_det_tuple = (f_idx_orig, det_inst_orig)
                        break
                # find closest SUCCEEDING detection IN THE ORIGINAL TRACK
                for f_idx_orig, det_inst_orig in track:
                    if f_idx_orig > target_frame_idx:
                        succ_det_tuple = (f_idx_orig, det_inst_orig)
                        break

                if prev_det_tuple and succ_det_tuple:
                    f_prev, d_prev = prev_det_tuple
                    f_succ, d_succ = succ_det_tuple

                    d_prev_box = d_prev['box'].to(device)
                    d_succ_box = d_succ['box'].to(device)
                    d_prev_score_dtype = d_prev['score'].to(device).dtype  # dtype from a tensor already on device
                    d_prev_label = d_prev['label'].to(device)

                    if f_succ - f_prev <= 0:  # will not happen if track elements are unique by frame_idx
                        if debug: print(
                            f"SCA DEBUG (R):     Frame {target_frame_idx}: Skipping interpolation, f_succ <= f_prev ({f_succ} <= {f_prev})")
                        continue

                    factor = (target_frame_idx - f_prev) / (f_succ - f_prev)
                    interp_box = d_prev_box * (1.0 - factor) + d_succ_box * factor
                    interp_label = d_prev_label
                    interp_score = torch.tensor(t_score_interp, device=device, dtype=d_prev_score_dtype)

                    interpolated_det_instance = {'box': interp_box, 'score': interp_score, 'label': interp_label}
                    interpolated_track_elements.append((target_frame_idx, interpolated_det_instance))
                    if debug: print(
                        f"SCA DEBUG (R):     Frame {target_frame_idx}: Interpolated box {interp_box.cpu().numpy().round(1)}, score {interp_score.item()}")
                elif debug:
                    print(
                        f"SCA DEBUG (R):     Frame {target_frame_idx}: Could not interpolate. Prev: {prev_det_tuple is not None}, Succ: {succ_det_tuple is not None}")

        if interpolated_track_elements:
            final_processed_tracks.append(interpolated_track_elements)

    #  4. reconstruct output
    if debug: print(f"\nSCA DEBUG (R): --- Reconstructing Output ---")
    refined_sequence_detections_list_of_lists: List[Dict[str, list]] = [
        {'boxes': [], 'scores': [], 'labels': []} for _ in range(num_total_frames)
    ]

    for track in final_processed_tracks:
        for frame_idx, det_instance in track:
            if 0 <= frame_idx < num_total_frames:
                refined_sequence_detections_list_of_lists[frame_idx]['boxes'].append(det_instance['box'].to(device))
                refined_sequence_detections_list_of_lists[frame_idx]['scores'].append(det_instance['score'].to(device))
                refined_sequence_detections_list_of_lists[frame_idx]['labels'].append(det_instance['label'].to(device))

    final_output_sequence: List[Dict[str, torch.Tensor]] = []
    for frame_idx, frame_data in enumerate(refined_sequence_detections_list_of_lists):
        if frame_data['boxes']:
            try:
                boxes = torch.stack(frame_data['boxes'])
                scores = torch.stack(frame_data['scores'])
                labels = torch.stack(frame_data['labels'])
            except RuntimeError as e:
                if debug:
                    print(f"SCA DEBUG (R): ERROR during stack for frame {frame_idx}. Message: {e}")
                    for i_t, t_s in enumerate(frame_data['scores']): print(
                        f"  Score {i_t} device: {t_s.device}, dtype: {t_s.dtype}, shape: {t_s.shape}")
                    for i_t, t_b in enumerate(frame_data['boxes']): print(
                        f"  Box {i_t} device: {t_b.device}, dtype: {t_b.dtype}, shape: {t_b.shape}")
                    for i_t, t_l in enumerate(frame_data['labels']): print(
                        f"  Label {i_t} device: {t_l.device}, dtype: {t_l.dtype}, shape: {t_l.shape}")
                # Fallback to empty if stacking fails
                boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
                scores = torch.empty((0,), device=device, dtype=torch.float32)
                labels = torch.empty((0,), device=device, dtype=torch.int64)
        else:  # Lists were empty
            boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
            scores = torch.empty((0,), device=device, dtype=torch.float32)
            labels = torch.empty((0,), device=device, dtype=torch.int64)

        final_output_sequence.append({'boxes': boxes, 'scores': scores, 'labels': labels})
        if debug and boxes.numel() > 0:
            print(f"SCA DEBUG (R):   Final output frame {frame_idx} has {boxes.shape[0]} detections.")
        elif debug and not frame_data['boxes']:
            print(f"SCA DEBUG (R):   Final output frame {frame_idx} has 0 detections (source list was empty).")

    return final_output_sequence


