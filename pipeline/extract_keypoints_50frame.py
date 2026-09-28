"""
Re-extraction at N_FRAMES_TARGET=50, reusing corrected player selection.

Writes to entirely new parallel directories - does not touch the existing
30-frame data at all. Reuses primary_track_id from extraction_log.csv (the
already-corrected value, post drift-fix/wrist-tiebreak/manual-review) via
forced_primary_id, rather than re-running player disambiguation from scratch.

Meant to run detached:
    nohup python extract_keypoints_50frame.py > /workspace/logs/extraction_50frame_stdout.log 2>&1 &

Check on it later with:
    tail -20 /workspace/logs/extraction_50frame_stdout.log
    wc -l /workspace/data/extraction_log_50frame.csv
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
MANIFEST_PATH = "/workspace/data/split_manifest.csv"
PREV_LOG_PATH = "/workspace/data/extraction_log.csv"
OUT_RAW = "/workspace/data/keypoints_raw_50frame"
OUT_SMOOTHED = "/workspace/data/keypoints_smoothed_50frame"
LOG_PATH = "/workspace/data/extraction_log_50frame.csv"
FAIL_PATH = "/workspace/data/extraction_failures_50frame.csv"
N_FRAMES_TARGET = 50
CHECKPOINT_DIR = "/workspace/checkpoints"


def find_pose_config_and_checkpoint():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*coco-256x192*.py"))
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*256x192*.pth"))
    if not configs or not checkpoints:
        raise FileNotFoundError(
            f"couldn't find rtmpose config/checkpoint under {CHECKPOINT_DIR} - "
            "check the mim download actually landed there, or fix CHECKPOINT_DIR above"
        )
    return configs[0], checkpoints[0]


def main():
    os.makedirs(OUT_RAW, exist_ok=True)
    os.makedirs(OUT_SMOOTHED, exist_ok=True)
    os.makedirs("/workspace/logs", exist_ok=True)

    task = Task.init(project_name="BadmintonPoseAnalysis", task_name="pose_extraction_rtmpose_50frame")
    logger = task.get_logger()

    manifest = pd.read_csv(MANIFEST_PATH)
    prev_log = pd.read_csv(PREV_LOG_PATH)

    # merge in the already-corrected primary_track_id + excluded flag per clip,
    # rather than re-running player disambiguation from scratch
    merged = manifest.merge(
        prev_log[["filepath", "primary_track_id", "excluded"]],
        on="filepath", how="left",
    )

    missing_id = merged["primary_track_id"].isna().sum()
    if missing_id > 0:
        print(f"WARNING: {missing_id} clips in the manifest have no matching row in "
              f"extraction_log.csv - these will be skipped", flush=True)

    merged = merged[merged["primary_track_id"].notna()].copy()
    merged = merged[~merged["excluded"].astype(bool)].copy()
    merged["primary_track_id"] = merged["primary_track_id"].astype(int)
    merged = merged.reset_index(drop=True)

    print(f"loaded manifest: {len(manifest)} clips, "
          f"{len(merged)} after dropping excluded/unmatched", flush=True)

    pose_config, pose_checkpoint = find_pose_config_and_checkpoint()
    print(f"pose config: {pose_config}", flush=True)
    print(f"pose checkpoint: {pose_checkpoint}", flush=True)

    detector = YOLO("yolov8m.pt")
    pose_model = init_model(pose_config, pose_checkpoint, device="cuda:0")

    log_rows = []
    fail_rows = []

    t_start = time.time()
    for i, row in merged.iterrows():
        video_path = os.path.join(RAW_ROOT, row["filepath"])
        class_name = row["class_name"]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]

        raw_out_dir = os.path.join(OUT_RAW, class_name)
        smoothed_out_dir = os.path.join(OUT_SMOOTHED, class_name)
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(smoothed_out_dir, exist_ok=True)

        try:
            keypoints_seq, meta = extract_one_clip(
                video_path, detector, pose_model, n_frames=N_FRAMES_TARGET,
                forced_primary_id=int(row["primary_track_id"]),
            )
            smoothed = smooth_keypoint_sequence(keypoints_seq)

            np.save(os.path.join(raw_out_dir, stem + ".npy"), keypoints_seq)
            np.save(os.path.join(smoothed_out_dir, stem + ".npy"), smoothed)

            meta.update({
                "filepath": row["filepath"],
                "class_name": class_name,
                "split": row["split"],
                "status": "ok",
            })
            log_rows.append(meta)

        except Exception as e:
            fail_rows.append({
                "filepath": row["filepath"],
                "class_name": class_name,
                "error": str(e),
            })

        done = i + 1
        if done % 100 == 0 or done == len(merged):
            elapsed = time.time() - t_start
            print(
                f"[{done}/{len(merged)}] elapsed {elapsed/60:.1f} min, "
                f"{len(fail_rows)} failures so far",
                flush=True,
            )

            if log_rows:
                running_conf = float(np.mean([r["mean_confidence"] for r in log_rows]))
                logger.report_scalar("extraction", "mean_confidence_running", running_conf, iteration=done)
            logger.report_scalar("extraction", "failures_running", len(fail_rows), iteration=done)

            pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
            pd.DataFrame(fail_rows).to_csv(FAIL_PATH, index=False)

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    pd.DataFrame(fail_rows).to_csv(FAIL_PATH, index=False)

    task.upload_artifact("extraction_log", LOG_PATH)
    if fail_rows:
        task.upload_artifact("extraction_failures", FAIL_PATH)

    print("done.", flush=True)
    print(f"succeeded: {len(log_rows)}, failed: {len(fail_rows)}", flush=True)


if __name__ == "__main__":
    main()
