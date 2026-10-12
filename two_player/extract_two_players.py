"""
Two player pose extraction: keeps BOTH players of every clip instead of choosing one.

Why: the original run chose the single track with the largest total box area (select_primary_track). In a view where
one player is nearer the camera that is nearly always the near player, whoever hits the shot, so some clips carry the
wrong player's pose. This run keeps the two main players, aligned in time, so the actor can be chosen (or learned)
afterwards and a two person model can be trained.

Nothing existing is overwritten. Outputs:

    /workspace/data/keypoints_two_raw/<class>/<stem>.npy   float32 (2, 30, 17, 3)  x, y, confidence in source pixels
    /workspace/data/boxes_two/<class>/<stem>.npy            float32 (2, 30, 4)      box used for each pose, NaN where absent
    /workspace/data/keypoints_actor_raw/<class>/<stem>.npy  float32 (30, 17, 3)     the skeleton of the player named by the letter in the file name
                                                            (A -> person 0, B -> person 1); only written when that person was found
    /workspace/data/extraction_two_log.csv                  one row per clip (track ids, areas, coverage, legacy choice)
    /workspace/data/extraction_two_failures.csv

Person 0 is the player NEARER the camera (larger mean box bottom edge), person 1 the farther one. The order is a fixed
geometric rule, not a guess about who hits the shot. A missing person is all zeros with confidence 0.

Both players are sampled at the SAME 30 source frames (evenly spaced over the span where either player is tracked);
a player absent at a target frame uses the nearest frame within MAX_GAP frames, otherwise is left missing.

Run detached, not in a notebook cell. First a smoke run:

    python extract_two_players.py --limit 20
    nohup python extract_two_players.py > /workspace/logs/extraction_two_stdout.log 2>&1 &

Two processes can share the GPU:  --shard 0/2  and  --shard 1/2 (each writes its own log, merge afterwards).
Interrupted runs resume: clips whose output files already exist are skipped.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "pipeline"), os.path.join(_HERE, "pipeline"), _HERE):  # finds pose_extraction_utils from two_player/ or pipeline/
    sys.path.insert(0, os.path.abspath(_p))
from pose_extraction_utils import _track_area, select_primary_track

RAW_ROOT = "/workspace/data/raw"
MANIFEST_PATH = "/workspace/data/split_manifest.csv"
OUT_KP = "/workspace/data/keypoints_two_raw"
OUT_BOX = "/workspace/data/boxes_two"
OUT_ACTOR = "/workspace/data/keypoints_actor_raw"
CHECKPOINT_DIR = "/workspace/checkpoints"
N_FRAMES = 30
MAX_GAP = 3                 # frames: how far a player's box may be borrowed from to fill a target frame
MIN_COOCCUR = 0.3           # a second player's track must share this fraction of its own frames with the first
MAX_MEAN_IOU = 0.3          # ...and not overlap the first track (otherwise it is the same person, a fragmented track)
PERSISTENT = 0.3            # a track counts as a candidate player if present in this share of the clip's frames


def find_pose_config_and_checkpoint():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*coco-256x192*.py"))
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "rtmpose-m*256x192*.pth"))
    if not configs or not checkpoints:
        raise FileNotFoundError(f"couldn't find rtmpose config/checkpoint under {CHECKPOINT_DIR}")
    return configs[0], checkpoints[0]


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def choose_two_tracks(track_boxes):
    """Largest track by area, then the largest track that co-exists with it and is a different person (not a fragment)."""
    areas = {tid: _track_area(f) for tid, f in track_boxes.items()}
    ranked = sorted(areas, key=areas.get, reverse=True)
    if not ranked:
        return [], areas
    first = ranked[0]
    chosen = [first]
    for tid in ranked[1:]:
        shared = sorted(set(track_boxes[tid]) & set(track_boxes[first]))
        if len(shared) < 3 or len(shared) < MIN_COOCCUR * len(track_boxes[tid]):
            continue                                   # does not co-exist with the first: likely a fragment of it or a bystander
        if np.mean([box_iou(track_boxes[tid][f], track_boxes[first][f]) for f in shared]) < MAX_MEAN_IOU:
            chosen.append(tid)
            break
    return chosen, areas


def nearest_frame(frames_sorted, f, max_gap=MAX_GAP):
    i = int(np.searchsorted(frames_sorted, f))
    cands = [frames_sorted[j] for j in (i - 1, i) if 0 <= j < len(frames_sorted)]
    best = min(cands, key=lambda c: abs(int(c) - f))
    return int(best) if abs(int(best) - f) <= max_gap else None


def extract_two(video_path, detector, pose_model):
    from mmpose.apis import inference_topdown

    all_frames, track_boxes = [], {}
    for frame_idx, r in enumerate(detector.track(source=video_path, persist=False, tracker="bytetrack.yaml",
                                                 classes=[0], verbose=False, stream=True)):
        all_frames.append(r.orig_img)
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            track_boxes.setdefault(int(tid), {})[frame_idx] = box
    if not track_boxes:
        raise RuntimeError("no player track detected in this clip")

    n_total = len(all_frames)
    chosen, areas = choose_two_tracks(track_boxes)
    legacy_id, legacy_ratio = select_primary_track(track_boxes)
    # fixed geometric order: person 0 is the player nearer the camera (larger mean bottom edge of the box)
    chosen = sorted(chosen, key=lambda t: -np.mean([b[3] for b in track_boxes[t].values()]))
    n_persons = len(chosen)

    first = min(min(track_boxes[t]) for t in chosen)
    last = max(max(track_boxes[t]) for t in chosen)
    targets = np.linspace(first, last, N_FRAMES).round().astype(int)

    boxes = np.full((2, N_FRAMES, 4), np.nan, dtype=np.float32)
    need = {}                                          # source frame -> [(person, slot, box)]
    for p, tid in enumerate(chosen):
        have = np.array(sorted(track_boxes[tid]))
        for t, f in enumerate(targets):
            used = nearest_frame(have, int(f))
            if used is not None:
                boxes[p, t] = track_boxes[tid][used]
                need.setdefault(used, []).append((p, t, track_boxes[tid][used]))

    kps = np.zeros((2, N_FRAMES, 17, 3), dtype=np.float32)
    for frame_idx, items in need.items():              # one pose call per source frame, with both boxes when they share it
        results = inference_topdown(pose_model, all_frames[frame_idx], bboxes=np.array([b for _, _, b in items]))
        for res, (p, t, _) in zip(results, items):
            inst = res.pred_instances
            kps[p, t, :, :2] = inst.keypoints[0]
            kps[p, t, :, 2] = inst.keypoint_scores[0]

    present = ~np.isnan(boxes[:, :, 0])
    meta = {
        "num_frames_total": n_total, "n_tracks": len(track_boxes),
        "n_candidate_tracks": sum(len(f) >= PERSISTENT * n_total for f in track_boxes.values()),
        "n_persons": n_persons, "span_first": int(first), "span_last": int(last), "span_frames": int(last - first + 1),
        "person0_track_id": chosen[0], "person1_track_id": chosen[1] if n_persons > 1 else -1,
        "person0_area": areas[chosen[0]], "person1_area": areas[chosen[1]] if n_persons > 1 else 0.0,
        "area_ratio_second_first": (areas[chosen[1]] / areas[chosen[0]]) if n_persons > 1 and areas[chosen[0]] > 0 else 0.0,
        "person0_coverage": float(present[0].mean()), "person1_coverage": float(present[1].mean()),
        "person0_mean_conf": float(kps[0, :, :, 2][present[0]].mean()) if present[0].any() else 0.0,
        "person1_mean_conf": float(kps[1, :, :, 2][present[1]].mean()) if present[1].any() else 0.0,
        "legacy_primary_track_id": legacy_id, "legacy_area_ratio": legacy_ratio,
        "legacy_person": chosen.index(legacy_id) if legacy_id in chosen else -1,
    }
    return kps, boxes, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke run)")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th clip starting at i")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    suffix = "" if shard_n == 1 else f"_shard{shard_i}of{shard_n}"
    log_path = f"/workspace/data/extraction_two_log{suffix}.csv"
    fail_path = f"/workspace/data/extraction_two_failures{suffix}.csv"

    from ultralytics import YOLO
    from mmpose.apis import init_model

    os.makedirs("/workspace/logs", exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest = manifest.iloc[shard_i::shard_n].reset_index(drop=True)
    print(f"{len(manifest)} clips in this run (shard {args.shard})", flush=True)

    cfg, ckpt = find_pose_config_and_checkpoint()
    detector = YOLO("yolov8m.pt")
    pose_model = init_model(cfg, ckpt, device="cuda:0")

    log_rows = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) else []
    done_paths = {r["filepath"] for r in log_rows}
    fail_rows = []
    t0, n_new = time.time(), 0
    for i, row in manifest.iterrows():
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        kp_file = os.path.join(OUT_KP, row["class_name"], stem + ".npy")
        if row["filepath"] in done_paths and os.path.exists(kp_file):
            continue
        try:
            kps, boxes, meta = extract_two(os.path.join(RAW_ROOT, row["filepath"]), detector, pose_model)
            for root, arr in ((OUT_KP, kps), (OUT_BOX, boxes)):
                os.makedirs(os.path.join(root, row["class_name"]), exist_ok=True)
                np.save(os.path.join(root, row["class_name"], stem + ".npy"), arr)
            # the file name ends ..._<A or B>_<class number>; the letter is annotation of which player performs the shot
            # (working reading: A nearer the camera, B the opposite side). Recorded, not assumed: check it against the geometry.
            parts = stem.split("_")
            letter = parts[-2] if len(parts) >= 2 else ""
            actor_person = {"A": 0, "B": 1}.get(letter, -1)
            meta.update({"actor_letter": letter, "number_in_name": parts[-1],
                         "actor_person": actor_person,
                         "actor_certain": bool(actor_person >= 0 and meta["n_persons"] == 2)})
            if 0 <= actor_person < meta["n_persons"]:
                os.makedirs(os.path.join(OUT_ACTOR, row["class_name"]), exist_ok=True)
                np.save(os.path.join(OUT_ACTOR, row["class_name"], stem + ".npy"), kps[actor_person])
            meta.update({"filepath": row["filepath"], "class_name": row["class_name"], "split": row["split"], "status": "ok"})
            log_rows.append(meta)
            n_new += 1
        except Exception as e:
            fail_rows.append({"filepath": row["filepath"], "class_name": row["class_name"], "error": str(e)})
        if (i + 1) % 50 == 0 or i + 1 == len(manifest):
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            pd.DataFrame(fail_rows).to_csv(fail_path, index=False)
            el = time.time() - t0
            print(f"[{i + 1}/{len(manifest)}] {n_new} new clips in {el / 60:.1f} min "
                  f"({el / max(n_new, 1):.1f} s per clip), {len(fail_rows)} failures", flush=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
