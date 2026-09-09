"""
Detection-only (no pose estimation) - checks every clip in the manifest for
whether area-only and the (now-deprecated-as-default) area+motion blended
score actually pick a DIFFERENT primary track, not just a lower confidence
ratio. A low ratio doesn't necessarily mean the argmax choice differs - this
measures the thing that actually matters: did the wrong player get saved.

Both metrics are computed from the SAME single detection+tracking pass per
clip, so there's no cross-run track-ID instability to worry about - this is
a direct, same-run comparison.

Run:
    nohup python diagnose_selection_drift.py > /workspace/logs/drift_stdout.log 2>&1 &

Output: /workspace/data/selection_drift.csv - one row per clip, flagged
wherever area-only and area+motion disagree on the primary track.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_extraction_utils import select_primary_track, select_primary_track_combined

from ultralytics import YOLO


RAW_ROOT = "/workspace/data/raw"
MANIFEST_PATH = "/workspace/data/split_manifest.csv"
OUT_PATH = "/workspace/data/selection_drift.csv"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"checking {len(manifest)} clips for selection drift", flush=True)

    detector = YOLO("yolov8m.pt")

    results = []
    t_start = time.time()

    for i, row in manifest.iterrows():
        video_path = os.path.join(RAW_ROOT, row["filepath"])
        track_boxes = {}

        try:
            det_results = detector.track(
                source=video_path, persist=False, tracker="bytetrack.yaml",
                classes=[0], verbose=False, stream=True,
            )
            for frame_idx, r in enumerate(det_results):
                if r.boxes is None or r.boxes.id is None:
                    continue
                for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
                    track_boxes.setdefault(int(tid), {})[frame_idx] = box

            area_id, area_ratio = select_primary_track(track_boxes)
            combined_id, combined_ratio = select_primary_track_combined(track_boxes)

            results.append({
                "filepath": row["filepath"],
                "class_name": row["class_name"],
                "area_only_primary": area_id,
                "area_only_ratio": area_ratio,
                "combined_primary": combined_id,
                "combined_ratio": combined_ratio,
                "disagrees": area_id != combined_id,
            })

        except Exception as e:
            print(f"failed on {row['filepath']}: {e}", flush=True)

        done = i + 1
        if done % 200 == 0 or done == len(manifest):
            elapsed = time.time() - t_start
            n_disagree = sum(r["disagrees"] for r in results)
            print(f"[{done}/{len(manifest)}] elapsed {elapsed/60:.1f} min, "
                  f"{n_disagree} disagreements so far", flush=True)
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)

    print("done.", flush=True)
    print(f"total clips checked: {len(df)}", flush=True)
    print(f"actual primary-player disagreements: {df['disagrees'].sum()}", flush=True)
    print(f"as percentage: {100 * df['disagrees'].sum() / len(df):.2f}%", flush=True)


if __name__ == "__main__":
    main()
