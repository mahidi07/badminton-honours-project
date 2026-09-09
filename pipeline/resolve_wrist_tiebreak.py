"""
Resolves clips still flagged ambiguous (track_area_ratio > 0.6) after the
area+motion selection already baked into extract_keypoints.py. More
expensive per clip (runs pose estimation on both candidate tracks, not just
the winner), which is why this only targets the genuinely hard cases.

Run:
    nohup python resolve_wrist_tiebreak.py > /workspace/logs/wrist_tiebreak_stdout.log 2>&1 &
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_extraction_utils import (
    select_primary_track_with_wrist_tiebreak, standardize_indices,
    smooth_keypoint_sequence, inference_topdown_safe,
)

from ultralytics import YOLO
from mmpose.apis import init_model
from clearml import Task


RAW_ROOT = "/workspace/data/raw"
LOG_PATH = "/workspace/data/extraction_log.csv"
OUT_RAW = "/workspace/data/keypoints_raw"
OUT_SMOOTHED = "/workspace/data/keypoints_smoothed"
N_FRAMES_TARGET = 30
CHECKPOINT_DIR = "/workspace/checkpoints"
AMBIGUITY_THRESHOLD = 0.6


def find_pose_config_and_checkpoint():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*coco-256x192*.py"))
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*256x192*.pth"))
    if not configs or not checkpoints:
        raise FileNotFoundError(f"couldn't find rtmpose config/checkpoint under {CHECKPOINT_DIR}")
    return configs[0], checkpoints[0]


def extract_with_tiebreak(video_path, detector, pose_model):
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
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            track_boxes.setdefault(int(tid), {})[frame_idx] = box

    primary_id, ratio, method = select_primary_track_with_wrist_tiebreak(
        track_boxes, all_frames, pose_model, ambiguity_threshold=AMBIGUITY_THRESHOLD
    )
    if primary_id is None:
        raise RuntimeError("no player track detected")

    valid_frames = sorted(track_boxes[primary_id].keys())
    idx_map = standardize_indices(len(valid_frames), N_FRAMES_TARGET)

    keypoints_seq = np.zeros((N_FRAMES_TARGET, 17, 3), dtype=np.float32)
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
        "track_area_ratio": ratio,
        "selection_method": method,
        "mean_confidence": float(keypoints_seq[:, :, 2].mean()),
        "min_confidence": float(keypoints_seq[:, :, 2].min()),
    }
    return keypoints_seq, meta


def main():
    task = Task.init(project_name="BadmintonPoseAnalysis", task_name="pose_extraction_wrist_tiebreak")
    logger = task.get_logger()

    log = pd.read_csv(LOG_PATH)
    target_rows = log[log["track_area_ratio"] > AMBIGUITY_THRESHOLD].copy()
    print(f"resolving {len(target_rows)} clips with wrist-velocity tiebreak", flush=True)

    pose_config, pose_checkpoint = find_pose_config_and_checkpoint()
    detector = YOLO("yolov8m.pt")
    pose_model = init_model(pose_config, pose_checkpoint, device="cuda:0")

    resolved = 0
    still_ambiguous = 0
    failed = 0

    t_start = time.time()
    for i, (idx, row) in enumerate(target_rows.iterrows()):
        video_path = os.path.join(RAW_ROOT, row["filepath"])
        class_name = row["class_name"]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]

        try:
            keypoints_seq, meta = extract_with_tiebreak(video_path, detector, pose_model)
            smoothed = smooth_keypoint_sequence(keypoints_seq)

            np.save(os.path.join(OUT_RAW, class_name, stem + ".npy"), keypoints_seq)
            np.save(os.path.join(OUT_SMOOTHED, class_name, stem + ".npy"), smoothed)

            for key, value in meta.items():
                log.loc[idx, key] = value

            if meta["track_area_ratio"] > AMBIGUITY_THRESHOLD:
                still_ambiguous += 1
            resolved += 1

        except Exception as e:
            failed += 1
            print(f"failed on {row['filepath']}: {e}", flush=True)

        done = i + 1
        if done % 25 == 0 or done == len(target_rows):
            elapsed = time.time() - t_start
            print(
                f"[{done}/{len(target_rows)}] elapsed {elapsed/60:.1f} min, "
                f"{still_ambiguous} still ambiguous, {failed} failed",
                flush=True,
            )
            logger.report_scalar("wrist_tiebreak", "still_ambiguous", still_ambiguous, iteration=done)
            log.to_csv(LOG_PATH, index=False)

    log.to_csv(LOG_PATH, index=False)
    task.upload_artifact("updated_extraction_log", LOG_PATH)

    print("done.", flush=True)
    print(f"resolved: {resolved}, still ambiguous: {still_ambiguous}, failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
