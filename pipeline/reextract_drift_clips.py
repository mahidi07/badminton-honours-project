"""
Re-extracts only the clips the drift diagnostic flagged as genuine primary-
player disagreements (area-only vs the old always-on blended score actually
picked a different track, not just a lower confidence ratio). Everything
else in the dataset is left untouched - if area-only and blended agreed,
the already-saved keypoints from the original run are correct as-is.

Requires selection_drift.csv to exist (produced by diagnose_selection_drift.py).

Run:
    nohup python reextract_drift_clips.py > /workspace/logs/reextract_drift_stdout.log 2>&1 &
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_extraction_utils import extract_one_clip, smooth_keypoint_sequence

from ultralytics import YOLO
from mmpose.apis import init_model
from clearml import Task


RAW_ROOT = "/workspace/data/raw"
DRIFT_PATH = "/workspace/data/selection_drift.csv"
LOG_PATH = "/workspace/data/extraction_log.csv"
OUT_RAW = "/workspace/data/keypoints_raw"
OUT_SMOOTHED = "/workspace/data/keypoints_smoothed"
N_FRAMES_TARGET = 30
CHECKPOINT_DIR = "/workspace/checkpoints"


def find_pose_config_and_checkpoint():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*coco-256x192*.py"))
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*256x192*.pth"))
    if not configs or not checkpoints:
        raise FileNotFoundError(f"couldn't find rtmpose config/checkpoint under {CHECKPOINT_DIR}")
    return configs[0], checkpoints[0]


def main():
    if not os.path.exists(DRIFT_PATH):
        raise FileNotFoundError(
            f"{DRIFT_PATH} doesn't exist yet - run diagnose_selection_drift.py to completion first"
        )

    drift = pd.read_csv(DRIFT_PATH)
    target_rows = drift[drift["disagrees"] == True].copy()
    print(f"re-extracting {len(target_rows)} clips flagged as genuine primary-player disagreements", flush=True)

    if len(target_rows) == 0:
        print("nothing to do - area-only and blended agreed on every clip. done.", flush=True)
        return

    task = Task.init(project_name="BadmintonPoseAnalysis", task_name="pose_extraction_reextract_drift")
    logger = task.get_logger()

    log = pd.read_csv(LOG_PATH)

    pose_config, pose_checkpoint = find_pose_config_and_checkpoint()
    detector = YOLO("yolov8m.pt")
    pose_model = init_model(pose_config, pose_checkpoint, device="cuda:0")

    updated = 0
    failed = 0

    t_start = time.time()
    for i, (_, row) in enumerate(target_rows.iterrows()):
        video_path = os.path.join(RAW_ROOT, row["filepath"])
        class_name = row["class_name"]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]

        try:
            keypoints_seq, meta = extract_one_clip(
                video_path, detector, pose_model, n_frames=N_FRAMES_TARGET
            )
            smoothed = smooth_keypoint_sequence(keypoints_seq)

            os.makedirs(os.path.join(OUT_RAW, class_name), exist_ok=True)
            os.makedirs(os.path.join(OUT_SMOOTHED, class_name), exist_ok=True)
            np.save(os.path.join(OUT_RAW, class_name, stem + ".npy"), keypoints_seq)
            np.save(os.path.join(OUT_SMOOTHED, class_name, stem + ".npy"), smoothed)

            log_idx = log[log["filepath"] == row["filepath"]].index
            if len(log_idx) > 0:
                for key, value in meta.items():
                    log.loc[log_idx[0], key] = value
                log.loc[log_idx[0], "selection_method"] = "area_reextracted"

            updated += 1

        except Exception as e:
            failed += 1
            print(f"failed on {row['filepath']}: {e}", flush=True)

        done = i + 1
        if done % 25 == 0 or done == len(target_rows):
            elapsed = time.time() - t_start
            print(f"[{done}/{len(target_rows)}] elapsed {elapsed/60:.1f} min, {failed} failed", flush=True)
            logger.report_scalar("reextract_drift", "updated", updated, iteration=done)
            log.to_csv(LOG_PATH, index=False)

    log.to_csv(LOG_PATH, index=False)
    task.upload_artifact("updated_extraction_log", LOG_PATH)

    print("done.", flush=True)
    print(f"updated: {updated}, failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
