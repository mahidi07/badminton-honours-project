"""
Full pose extraction run over the frozen split manifest.

Meant to run detached, not inside a notebook cell:

    nohup python extract_keypoints.py > /workspace/logs/extraction_stdout.log 2>&1 &

Check on it later with:
    tail -20 /workspace/logs/extraction_stdout.log
    wc -l /workspace/data/extraction_log.csv
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
OUT_RAW = "/workspace/data/keypoints_raw"
OUT_SMOOTHED = "/workspace/data/keypoints_smoothed"
LOG_PATH = "/workspace/data/extraction_log.csv"
FAIL_PATH = "/workspace/data/extraction_failures.csv"
N_FRAMES_TARGET = 30
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

    task = Task.init(project_name="BadmintonPoseAnalysis", task_name="pose_extraction_rtmpose")
    logger = task.get_logger()

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"loaded manifest: {len(manifest)} clips", flush=True)

    pose_config, pose_checkpoint = find_pose_config_and_checkpoint()
    print(f"pose config: {pose_config}", flush=True)
    print(f"pose checkpoint: {pose_checkpoint}", flush=True)

    detector = YOLO("yolov8m.pt")
    pose_model = init_model(pose_config, pose_checkpoint, device="cuda:0")

    log_rows = []
    fail_rows = []

    t_start = time.time()
    for i, row in manifest.iterrows():
        video_path = os.path.join(RAW_ROOT, row["filepath"])
        class_name = row["class_name"]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]

        raw_out_dir = os.path.join(OUT_RAW, class_name)
        smoothed_out_dir = os.path.join(OUT_SMOOTHED, class_name)
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(smoothed_out_dir, exist_ok=True)

        try:
            keypoints_seq, meta = extract_one_clip(
                video_path, detector, pose_model, n_frames=N_FRAMES_TARGET
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
        if done % 100 == 0 or done == len(manifest):
            elapsed = time.time() - t_start
            print(
                f"[{done}/{len(manifest)}] elapsed {elapsed/60:.1f} min, "
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
