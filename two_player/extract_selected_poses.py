"""
Stage B of the two player re-extraction: run RTMPose on the players chosen in notebook 31.

Input   /workspace/data/boxes_selected/<class>/<stem>.npy   float32 (2, n_frames, 4)   boxes of the near (0) and far (1) player, NaN where absent
Output  /workspace/data/keypoints_sel_raw/<class>/<stem>.npy float32 (2, 30, 17, 3)    x, y, confidence in source pixels (zeros where absent)
        /workspace/data/boxes_sel/<class>/<stem>.npy         float32 (2, 30, 4)        box used for each pose, NaN where absent
Log     /workspace/data/extraction_sel_log[_shardIofN].csv

The same 30 frame sampling as before: 30 frames spread evenly over the span in which either player is tracked; a player missing at a sampled frame
borrows the box of the nearest frame within MAX_GAP frames, otherwise stays absent. The pose model is the same RTMPose-m (256x192) as the main pipeline.

    python extract_selected_poses.py --limit 20
    python extract_selected_poses.py --shard 0/2      and      --shard 1/2
Interrupted runs resume. Run notebook 30's cleaning (smoothing, repair of repeated frames) on the output afterwards.
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

RAW_ROOT = "/workspace/data/raw"
MANIFEST_PATH = "/workspace/data/split_manifest.csv"
IN_BOX = "/workspace/data/boxes_selected"
OUT_KP = "/workspace/data/keypoints_sel_raw"
OUT_BOX = "/workspace/data/boxes_sel"
CHECKPOINT_DIR = "/workspace/checkpoints"
N_FRAMES = 30
MAX_GAP = 3


def find_pose_config_and_checkpoint():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*coco-256x192*.py"))
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*256x192*.pth"))
    if not configs or not checkpoints:
        raise FileNotFoundError(f"couldn't find rtmpose config/checkpoint under {CHECKPOINT_DIR}")
    return configs[0], checkpoints[0]


def nearest_frame(frames_sorted, f, max_gap=MAX_GAP):
    i = int(np.searchsorted(frames_sorted, f))
    cands = [frames_sorted[j] for j in (i - 1, i) if 0 <= j < len(frames_sorted)]
    best = min(cands, key=lambda c: abs(int(c) - f))
    return int(best) if abs(int(best) - f) <= max_gap else None


def poses_for_clip(video_path, sel_boxes, pose_model):
    import cv2
    from mmpose.apis import inference_topdown

    present = ~np.isnan(sel_boxes[:, :, 0])                      # (2, n_frames)
    if not present.any():
        raise RuntimeError("no player selected for this clip")
    first = int(np.where(present.any(axis=0))[0][0])
    last = int(np.where(present.any(axis=0))[0][-1])
    targets = np.linspace(first, last, N_FRAMES).round().astype(int)

    boxes = np.full((2, N_FRAMES, 4), np.nan, dtype=np.float32)
    need = {}                                                    # source frame -> [(person, slot, box)]
    for p in (0, 1):
        have = np.where(present[p])[0]
        if len(have) == 0:
            continue
        for t, f in enumerate(targets):
            used = nearest_frame(have, int(f))
            if used is not None:
                boxes[p, t] = sel_boxes[p, used]
                need.setdefault(used, []).append((p, t, sel_boxes[p, used]))

    wanted, frames, idx = set(need), {}, 0
    cap = cv2.VideoCapture(video_path)
    last_needed = max(wanted) if wanted else -1
    while idx <= last_needed:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames[idx] = bgr
        idx += 1
    cap.release()

    kps = np.zeros((2, N_FRAMES, 17, 3), dtype=np.float32)
    for frame_idx, items in need.items():
        if frame_idx not in frames:
            for p, t, _ in items:
                boxes[p, t] = np.nan                             # frame could not be decoded
            continue
        results = inference_topdown(pose_model, frames[frame_idx], bboxes=np.array([b for _, _, b in items]))
        for res, (p, t, _) in zip(results, items):
            inst = res.pred_instances
            kps[p, t, :, :2] = inst.keypoints[0]
            kps[p, t, :, 2] = inst.keypoint_scores[0]

    ok_slots = ~np.isnan(boxes[:, :, 0])
    meta = {"n_persons": int(present.any(axis=1).sum()), "span_first": first, "span_last": last, "span_frames": last - first + 1,
            "person0_coverage": float(ok_slots[0].mean()), "person1_coverage": float(ok_slots[1].mean()),
            "person0_mean_conf": float(kps[0, :, :, 2][ok_slots[0]].mean()) if ok_slots[0].any() else 0.0,
            "person1_mean_conf": float(kps[1, :, :, 2][ok_slots[1]].mean()) if ok_slots[1].any() else 0.0}
    return kps, boxes, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    suffix = "" if shard_n == 1 else f"_shard{shard_i}of{shard_n}"
    log_path = f"/workspace/data/extraction_sel_log{suffix}.csv"
    fail_path = f"/workspace/data/extraction_sel_failures{suffix}.csv"

    from mmpose.apis import init_model
    manifest = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest = manifest.iloc[shard_i::shard_n].reset_index(drop=True)
    print(f"{len(manifest)} clips in this run (shard {args.shard})", flush=True)

    cfg, ckpt = find_pose_config_and_checkpoint()
    pose_model = init_model(cfg, ckpt, device="cuda:0")
    log_rows = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) and os.path.getsize(log_path) > 0 else []
    done = {r["filepath"] for r in log_rows}
    fails, t0, n_new = [], time.time(), 0
    for i, row in manifest.iterrows():
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        kp_file = os.path.join(OUT_KP, row["class_name"], stem + ".npy")
        if row["filepath"] in done and os.path.exists(kp_file):
            continue
        try:
            sel = np.load(os.path.join(IN_BOX, row["class_name"], stem + ".npy"))
            kps, boxes, meta = poses_for_clip(os.path.join(RAW_ROOT, row["filepath"]), sel, pose_model)
            for root, arr in ((OUT_KP, kps), (OUT_BOX, boxes)):
                os.makedirs(os.path.join(root, row["class_name"]), exist_ok=True)
                np.save(os.path.join(root, row["class_name"], stem + ".npy"), arr)
            meta.update({"filepath": row["filepath"], "class_name": row["class_name"], "split": row["split"], "status": "ok"})
            log_rows.append(meta)
            n_new += 1
        except Exception as e:
            fails.append({"filepath": row["filepath"], "class_name": row["class_name"], "error": str(e)})
        if (i + 1) % 50 == 0 or i + 1 == len(manifest):
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            pd.DataFrame(fails).to_csv(fail_path, index=False)
            el = time.time() - t0
            print(f"[{i + 1}/{len(manifest)}] {n_new} new clips in {el / 60:.1f} min ({el / max(n_new, 1):.1f} s per clip), "
                  f"{len(fails)} failures", flush=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    pd.DataFrame(fails).to_csv(fail_path, index=False)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
