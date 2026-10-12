"""
Stage A of the two player re-extraction: save EVERY tracked person's boxes for every clip. No pose model here.

Why: picking the second player by box size picks bystanders, because the real opponent at the far end of the court is the smallest
person in the frame. With every track saved, the opponent can be chosen by where they stand (a court region per source video) and
selection rules can be tried offline without running detection again.

Detector and tracker are the same as the main pipeline (YOLOv8m, ByteTrack, person class). Two settings differ, to give small far
players a better chance of being tracked: a lower detection confidence (--conf, default 0.15) and a larger image size (--imgsz, default 960).

Output, one file per clip:
    /workspace/data/tracks_all/<class>/<stem>.npz
        rows      float32 (N, 7)   track_id, frame, x1, y1, x2, y2, detection_confidence
        n_frames  int              frames in the clip
        height, width              frame size in pixels
Log: /workspace/data/tracks_all_log[_shardIofN].csv

    python dump_tracks.py --only <part of a file name>        # one or a few clips (comma separated), for a quick look
    python dump_tracks.py --limit 20                          # smoke run
    python dump_tracks.py --shard 0/2   and   --shard 1/2    # the full run on two processes
Interrupted runs resume: clips whose output already exists are skipped.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

RAW_ROOT = "/workspace/data/raw"
MANIFEST_PATH = "/workspace/data/split_manifest.csv"
OUT_DIR = "/workspace/data/tracks_all"
LOG_DIR = "/workspace/data"


def dump_clip(video_path, detector, imgsz, conf):
    rows, n_frames, hw = [], 0, (0, 0)
    for f, r in enumerate(detector.track(source=video_path, persist=False, tracker="bytetrack.yaml", classes=[0],
                                         conf=conf, imgsz=imgsz, verbose=False, stream=True)):
        n_frames, hw = f + 1, r.orig_shape
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        for i, b, c in zip(ids, xyxy, cf):
            rows.append([i, f, b[0], b[1], b[2], b[3], c])
    return np.array(rows, dtype=np.float32).reshape(-1, 7), n_frames, hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="comma separated parts of file names; only matching clips")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    suffix = "" if shard_n == 1 else f"_shard{shard_i}of{shard_n}"
    log_path = f"{LOG_DIR}/tracks_all_log{suffix}.csv"

    from ultralytics import YOLO
    manifest = pd.read_csv(MANIFEST_PATH)
    if args.only:
        parts = args.only.split(",")
        manifest = manifest[manifest["filepath"].apply(lambda p: any(s in p for s in parts))]
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest = manifest.iloc[shard_i::shard_n].reset_index(drop=True)
    print(f"{len(manifest)} clips in this run (shard {args.shard}), conf {args.conf}, imgsz {args.imgsz}", flush=True)

    detector = YOLO("yolov8m.pt")
    log = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) and os.path.getsize(log_path) > 0 else []
    done = {r["filepath"] for r in log}
    t0, new, failures = time.time(), 0, 0
    for i, row in manifest.iterrows():
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        out = os.path.join(OUT_DIR, row["class_name"], stem + ".npz")
        if row["filepath"] in done and os.path.exists(out):
            continue
        try:
            rows, n_frames, hw = dump_clip(os.path.join(RAW_ROOT, row["filepath"]), detector, args.imgsz, args.conf)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, rows=rows, n_frames=n_frames, height=hw[0], width=hw[1])
            log.append({"filepath": row["filepath"], "class_name": row["class_name"], "split": row["split"],
                        "n_frames": n_frames, "n_tracks": int(len(np.unique(rows[:, 0]))), "n_boxes": len(rows),
                        "height": hw[0], "width": hw[1], "status": "ok"})
        except Exception as e:                                   # keep going; failures are listed in the log
            failures += 1
            log.append({"filepath": row["filepath"], "class_name": row["class_name"], "split": row["split"], "status": f"failed: {e}"})
        new += 1
        if new % 50 == 0:
            pd.DataFrame(log).to_csv(log_path, index=False)
            print(f"[{i + 1}/{len(manifest)}] {new} new clips in {(time.time() - t0) / 60:.1f} min "
                  f"({(time.time() - t0) / new:.1f} s per clip), {failures} failures", flush=True)
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"done: {new} new clips in {(time.time() - t0) / 60:.1f} min, {failures} failures", flush=True)


if __name__ == "__main__":
    main()
