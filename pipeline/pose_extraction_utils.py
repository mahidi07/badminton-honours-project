"""
Shared helpers for the pose extraction pipeline.

Two-pass approach: pass 1 is YOLOv8 + ByteTrack over the full clip to get
player tracks, pass 2 feeds each track's per-frame box into RTMPose
(inference_topdown) to get the actual keypoints. No manual cropping needed -
inference_topdown handles that internally from the bbox.

Used by extract_keypoints.py (the full background run), reprocess_ambiguous.py,
resolve_wrist_tiebreak.py, and the QC/manual review notebooks.
"""

import numpy as np
import cv2


def _track_area(frames_dict):
    total = 0.0
    for box in frames_dict.values():
        x1, y1, x2, y2 = box
        total += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return total


def _track_peak_motion(frames_dict):
    """
    Peak single-frame centroid displacement across the track - a proxy for a
    swing/hit burst. The player actually playing the shot tends to show a
    sharp movement spike that a waiting player doesn't, which holds even
    when the two players are standing close together (unlike bbox area).
    """
    items = sorted(frames_dict.items())
    if len(items) < 2:
        return 0.0

    centroids = []
    for _, box in items:
        x1, y1, x2, y2 = box
        centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    peak = 0.0
    for (x0, y0), (x1_, y1_) in zip(centroids[:-1], centroids[1:]):
        d = ((x1_ - x0) ** 2 + (y1_ - y0) ** 2) ** 0.5
        peak = max(peak, d)
    return peak


def select_primary_track(track_boxes):
    """
    track_boxes: dict {track_id: {frame_idx: [x1, y1, x2, y2]}}

    Area-only selection - the player with the largest total bbox area across
    the clip. This is the DEFAULT used by extract_one_clip. Deliberately not
    blended with motion here: testing the blended score as a default (not
    just an escalation for already-hard cases) showed it makes far MORE
    clips ambiguous, not fewer - motion is noisy enough that blending it
    into every decision drags down cases area alone was already confident
    about. Motion stays as an escalation-only signal via
    select_primary_track_combined, used exclusively on clips area-only
    already flagged as genuinely hard.

    Returns (primary_track_id, runner_up_to_winner_area_ratio).
    """
    if not track_boxes:
        return None, 0.0

    areas = {tid: _track_area(f) for tid, f in track_boxes.items()}
    ranked = sorted(areas.items(), key=lambda kv: kv[1], reverse=True)
    primary_id = ranked[0][0]

    if len(ranked) > 1 and ranked[0][1] > 0:
        ratio = ranked[1][1] / ranked[0][1]
    else:
        ratio = 0.0

    return primary_id, ratio


def select_primary_track_combined(track_boxes, area_weight=0.5, motion_weight=0.5):
    """
    Area+motion blended score. ESCALATION ONLY - call this on clips
    select_primary_track already flagged ambiguous (ratio > threshold), not
    as a general-purpose replacement for it. See select_primary_track's
    docstring for why.

    Returns (primary_track_id, runner_up_to_winner_score_ratio).
    """
    if not track_boxes:
        return None, 0.0

    areas = {tid: _track_area(f) for tid, f in track_boxes.items()}
    motions = {tid: _track_peak_motion(f) for tid, f in track_boxes.items()}

    max_area = max(areas.values()) or 1.0
    max_motion = max(motions.values()) or 1.0

    scores = {
        tid: area_weight * (areas[tid] / max_area) + motion_weight * (motions[tid] / max_motion)
        for tid in track_boxes
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    primary_id = ranked[0][0]

    if len(ranked) > 1 and ranked[0][1] > 0:
        ratio = ranked[1][1] / ranked[0][1]
    else:
        ratio = 0.0

    return primary_id, ratio


def wrist_peak_velocity(pose_model, all_frames, frames_dict):
    """
    Runs pose estimation across one track's frames and returns the peak
    single-frame wrist displacement (COCO joints 9/10 = left/right wrist).

    More targeted swing signal than whole-body bbox motion - diagnosis on
    still-ambiguous clips showed genuinely two simultaneous players with
    near-identical bbox motion (0.98 mean frame overlap, both moving fast
    during net exchanges), so whole-body movement doesn't distinguish a
    racket swing from footwork, but wrist speed should.
    """
    from mmpose.apis import inference_topdown

    wrist_positions = []
    for frame_idx in sorted(frames_dict.keys()):
        frame = all_frames[frame_idx]
        box = frames_dict[frame_idx]
        pose_results = inference_topdown(pose_model, frame, bboxes=np.array([box]))
        kps = pose_results[0].pred_instances.keypoints[0]
        wrist_positions.append((kps[9].copy(), kps[10].copy()))

    peak = 0.0
    for (lw0, rw0), (lw1, rw1) in zip(wrist_positions[:-1], wrist_positions[1:]):
        peak = max(peak, float(np.linalg.norm(lw1 - lw0)), float(np.linalg.norm(rw1 - rw0)))
    return peak


def select_primary_track_with_wrist_tiebreak(
    track_boxes, all_frames, pose_model,
    area_weight=0.5, motion_weight=0.5, ambiguity_threshold=0.6,
):
    """
    Three-tier escalation: area-only first (cheap, confident for most
    clips), then combined area+motion (only for clips area-only already
    flagged ambiguous), then wrist velocity (only for clips still ambiguous
    after that). Each tier is strictly more expensive than the last, which
    is why cheaper/coarser signals are tried first.

    Returns (primary_id, ratio, method_used).
    """
    primary_id, ratio = select_primary_track(track_boxes)

    if primary_id is None or ratio <= ambiguity_threshold:
        return primary_id, ratio, "area"

    primary_id, ratio = select_primary_track_combined(track_boxes, area_weight, motion_weight)

    if ratio <= ambiguity_threshold:
        return primary_id, ratio, "area_motion"

    areas = {tid: _track_area(f) for tid, f in track_boxes.items()}
    motions = {tid: _track_peak_motion(f) for tid, f in track_boxes.items()}
    max_area = max(areas.values()) or 1.0
    max_motion = max(motions.values()) or 1.0
    scores = {
        tid: area_weight * (areas[tid] / max_area) + motion_weight * (motions[tid] / max_motion)
        for tid in track_boxes
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    id_a, id_b = ranked[0][0], ranked[1][0]

    wrist_a = wrist_peak_velocity(pose_model, all_frames, track_boxes[id_a])
    wrist_b = wrist_peak_velocity(pose_model, all_frames, track_boxes[id_b])

    winner = id_a if wrist_a >= wrist_b else id_b
    max_wrist = max(wrist_a, wrist_b)
    new_ratio = (min(wrist_a, wrist_b) / max_wrist) if max_wrist > 0 else 0.0

    return winner, new_ratio, "wrist_tiebreak"


def standardize_indices(n_available, n_target=30):
    """
    Maps however many valid frames a track actually has onto n_target evenly
    spaced indices. If n_available < n_target this naturally repeats some
    frames rather than failing.
    """
    if n_available <= 0:
        return np.zeros(n_target, dtype=int)
    return np.linspace(0, n_available - 1, n_target).round().astype(int)


def smooth_keypoint_sequence(keypoints, window=5, polyorder=2):
    """
    keypoints: (T, 17, 3) array, channel 2 is confidence.
    Savitzky-Golay filter over x/y channels only, per joint. Confidence left
    untouched.
    """
    from scipy.signal import savgol_filter

    T = keypoints.shape[0]
    if T < window:
        return keypoints.copy()

    smoothed = keypoints.copy()
    for joint in range(keypoints.shape[1]):
        for channel in range(2):
            smoothed[:, joint, channel] = savgol_filter(
                keypoints[:, joint, channel], window_length=window, polyorder=polyorder
            )
    return smoothed


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_bbox_overlay(frame, boxes_by_track, primary_id=None):
    """boxes_by_track: dict {track_id: [x1, y1, x2, y2]} for a single frame."""
    img = frame.copy()
    for tid, box in boxes_by_track.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        is_primary = (tid == primary_id)
        color = (0, 255, 0) if is_primary else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"id {tid}" + (" (primary)" if is_primary else "")
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_pose_overlay(frame, keypoints, conf_thresh=0.3):
    """keypoints: (17, 3) array for one frame - x, y, confidence per joint."""
    img = frame.copy()
    for x, y, c in keypoints:
        if c >= conf_thresh:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
    for a, b in COCO_SKELETON:
        if keypoints[a, 2] >= conf_thresh and keypoints[b, 2] >= conf_thresh:
            pt1 = tuple(keypoints[a, :2].astype(int))
            pt2 = tuple(keypoints[b, :2].astype(int))
            cv2.line(img, pt1, pt2, (255, 128, 0), 2)
    return img


def inference_topdown_safe(pose_model, frame, box):
    """Thin wrapper so this file stays importable even before mmpose is loaded."""
    from mmpose.apis import inference_topdown

    pose_results = inference_topdown(pose_model, frame, bboxes=np.array([box]))
    inst = pose_results[0].pred_instances
    keypoints = inst.keypoints[0]
    scores = inst.keypoint_scores[0]
    return keypoints, scores


def extract_one_clip(video_path, detector, pose_model, n_frames=30, return_debug=False):
    """
    Runs the full two-pass extraction on a single clip.
    Returns (keypoints_seq, meta) normally, or (keypoints_seq, meta, debug) if
    return_debug=True (used by the QC/manual-review notebooks for overlays).
    """
    all_frames = []
    track_boxes = {}

    results = detector.track(
        source=video_path, persist=False, tracker="bytetrack.yaml",
        classes=[0], verbose=False, stream=True,
    )

    for frame_idx, r in enumerate(results):
        all_frames.append(r.orig_img)
        if r.boxes is None or r.boxes.id is None:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy()
        for box, tid in zip(boxes_xyxy, track_ids):
            track_boxes.setdefault(int(tid), {})[frame_idx] = box

    primary_id, area_ratio = select_primary_track(track_boxes)
    if primary_id is None:
        raise RuntimeError("no player track detected in this clip")

    valid_frames = sorted(track_boxes[primary_id].keys())
    idx_map = standardize_indices(len(valid_frames), n_frames)

    keypoints_seq = np.zeros((n_frames, 17, 3), dtype=np.float32)

    for out_i, vf_i in enumerate(idx_map):
        frame_idx = valid_frames[vf_i]
        frame = all_frames[frame_idx]
        box = track_boxes[primary_id][frame_idx]

        kps, scores = inference_topdown_safe(pose_model, frame, box)
        keypoints_seq[out_i, :, 0:2] = kps
        keypoints_seq[out_i, :, 2] = scores

    meta = {
        "num_frames_total": len(all_frames),
        "num_valid_track_frames": len(valid_frames),
        "primary_track_id": primary_id,
        "track_area_ratio": area_ratio,
        "mean_confidence": float(keypoints_seq[:, :, 2].mean()),
        "min_confidence": float(keypoints_seq[:, :, 2].min()),
    }

    if return_debug:
        debug = {
            "all_frames": all_frames,
            "track_boxes": track_boxes,
            "primary_id": primary_id,
            "valid_frames": valid_frames,
            "idx_map": idx_map,
        }
        return keypoints_seq, meta, debug

    return keypoints_seq, meta
