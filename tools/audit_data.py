"""
Read-only audit of the extracted keypoint data, split manifest and extraction log.

Nothing in /workspace/data is modified. Summary tables are written to
/workspace/outputs/audit/ and a full text report is printed.

    python audit_data.py [--data-root /workspace/data] [--out /workspace/outputs/audit]
"""

import argparse
import hashlib
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

EXPECTED_SHAPE = (30, 17, 3)
JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
# 2022-08-30_18-00-09_dataset_set1_009_001424_001452_A_00
CLIP_PATTERN = re.compile(
    r"^(?P<match>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<source>.+?)_(?P<set>set\d+)_"
    r"(?P<rally>\d+)_(?P<start>\d+)_(?P<end>\d+)_(?P<player>[A-Za-z])_(?P<idx>\d+)$"
)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_of(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def audit_manifest(manifest, log, data_root):
    section("1. split manifest and extraction log")

    recorded = open(os.path.join(data_root, "split_manifest.sha256")).read().split()[0]
    actual = sha256(os.path.join(data_root, "split_manifest.csv"))
    print(f"manifest sha256 matches frozen hash: {recorded == actual}")

    print(f"manifest rows: {len(manifest)}, log rows: {len(log)}")
    print(f"duplicate filepaths in manifest: {manifest['filepath'].duplicated().sum()}")
    only_manifest = set(manifest["filepath"]) - set(log["filepath"])
    only_log = set(log["filepath"]) - set(manifest["filepath"])
    print(f"in manifest but not log: {len(only_manifest)}, in log but not manifest: {len(only_log)}")

    merged = manifest.merge(log, on="filepath", suffixes=("", "_log"))
    print(f"split disagreement manifest vs log: {(merged['split'] != merged['split_log']).sum()}")
    print(f"class disagreement manifest vs log: {(merged['class_name'] != merged['class_name_log']).sum()}")

    ids = manifest.groupby("class_name")["class_id"].nunique()
    print(f"classes with inconsistent class_id: {(ids > 1).sum()}")

    print("\nstatus counts:", dict(log["status"].value_counts(dropna=False)))
    print("selection_method counts:", dict(log["selection_method"].fillna("area_only(blank)").value_counts()))
    excluded = log[log["excluded"] == True]
    print(f"\nexcluded clips ({len(excluded)}):")
    for _, r in excluded.iterrows():
        print(f"  {r['split']:5s}  {r['filepath']}  valid_frames={r['num_valid_track_frames']}")

    kept = manifest[~manifest["filepath"].isin(excluded["filepath"])]
    table = pd.crosstab(kept["class_name"], kept["split"], margins=True)
    table["train_frac"] = (table["train"] / table["All"]).round(3)
    print("\nclass x split after exclusions:")
    print(table.to_string())
    counts = kept[kept["split"] == "train"]["class_name"].value_counts()
    print(f"\ntrain imbalance ratio (largest/smallest class): {counts.max() / counts.min():.1f}")

    section("2. clip length and tracking quality (from log)")
    for col in ["num_frames_total", "num_valid_track_frames", "track_area_ratio",
                "mean_confidence", "min_confidence"]:
        s = log[col]
        print(f"{col:24s} min={s.min():.3f} p5={s.quantile(.05):.3f} median={s.median():.3f} "
              f"p95={s.quantile(.95):.3f} max={s.max():.3f}")
    nft = log["num_frames_total"]
    print(f"\nclips shorter than 30 frames: {(nft < 30).sum()}, exactly 30: {(nft == 30).sum()}, "
          f"longer than 30: {(nft > 30).sum()}")
    gap = log["num_frames_total"] - log["num_valid_track_frames"]
    print(f"clips with untracked frames: {(gap > 0).sum()} (max missing {gap.max()})")
    print("\nmedian clip length by class:")
    print(log.groupby("class_name")["num_frames_total"].median().to_string())
    return kept


def audit_leakage(kept, out_dir):
    section("3. split leakage (parsed from clip filenames)")
    parsed = kept["filepath"].map(lambda p: CLIP_PATTERN.match(stem_of(p)))
    n_fail = parsed.isna().sum()
    print(f"filenames that did not parse: {n_fail}")
    if n_fail:
        print("  examples:", [stem_of(p) for p in kept["filepath"][parsed.isna()].head(5)])
    ok = kept[parsed.notna()].copy()
    fields = pd.DataFrame([m.groupdict() for m in parsed.dropna()], index=ok.index)
    ok = ok.join(fields)
    ok["start"] = ok["start"].astype(int)
    ok["end"] = ok["end"].astype(int)
    ok["rally_key"] = ok["match"] + "|" + ok["set"] + "|" + ok["rally"]

    print(f"distinct matches: {ok['match'].nunique()}, sources: {ok['source'].nunique()}, "
          f"rallies: {ok['rally_key'].nunique()}")
    per_match = pd.crosstab(ok["match"], ok["split"])
    print("\nclips per match per split:")
    print(per_match.to_string())

    rally_splits = ok.groupby("rally_key")["split"].nunique()
    print(f"\nrallies whose clips land in more than one split: {(rally_splits > 1).sum()} "
          f"of {len(rally_splits)}")

    # same player, same match/set/rally, overlapping frame ranges, different splits
    overlaps = []
    for _, g in ok.groupby(["rally_key", "player"]):
        g = g.sort_values("start")
        rows = g[["filepath", "split", "start", "end"]].values
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[j][2] > rows[i][3]:
                    break
                if rows[i][1] != rows[j][1]:
                    overlaps.append((rows[i][0], rows[i][1], rows[j][0], rows[j][1]))
    print(f"overlapping frame ranges across splits (same player, same rally): {len(overlaps)}")
    for o in overlaps[:10]:
        print("  ", o)
    pd.DataFrame(overlaps, columns=["clip_a", "split_a", "clip_b", "split_b"]).to_csv(
        os.path.join(out_dir, "cross_split_overlaps.csv"), index=False)

    exact = ok.duplicated(subset=["rally_key", "player", "start", "end"], keep=False)
    print(f"clips sharing identical match/rally/player/frame range: {exact.sum()}")


def audit_arrays(kept, keypoints_root, out_dir, label):
    section(f"4. keypoint arrays: {label}")
    records = []
    conf_all, x_all, y_all = [], [], []
    missing = 0
    for _, row in kept.iterrows():
        path = os.path.join(keypoints_root, row["class_name"], stem_of(row["filepath"]) + ".npy")
        if not os.path.exists(path):
            missing += 1
            continue
        a = np.load(path)
        rec = {"filepath": row["filepath"], "class_name": row["class_name"], "split": row["split"],
               "shape": str(a.shape), "dtype": str(a.dtype)}
        if a.shape == EXPECTED_SHAPE:
            finite = np.isfinite(a).all()
            zero_frame = (np.abs(a[:, :, :2]).sum(axis=(1, 2)) == 0)
            zero_joint = (np.abs(a[:, :, :2]).sum(axis=2) == 0)
            same_as_prev = np.r_[False, (np.abs(np.diff(a, axis=0)).sum(axis=(1, 2)) == 0)]
            trailing_repeat = 0
            for t in range(29, 0, -1):
                if same_as_prev[t]:
                    trailing_repeat += 1
                else:
                    break
            xy = a[:, :, :2][~zero_joint]
            span = xy.max(axis=0) - xy.min(axis=0) if len(xy) else np.array([0, 0])
            valid_frames = a[~zero_frame]
            if len(valid_frames):
                heights = valid_frames[:, :, 1].max(axis=1) - valid_frames[:, :, 1].min(axis=1)
            else:
                heights = np.array([0.0])
            rec.update({
                "finite": bool(finite), "zero_frames": int(zero_frame.sum()),
                "zero_joint_entries": int(zero_joint.sum()),
                "repeated_frames": int(same_as_prev.sum()), "trailing_repeats": trailing_repeat,
                "x_min": float(xy[:, 0].min()) if len(xy) else np.nan,
                "x_max": float(xy[:, 0].max()) if len(xy) else np.nan,
                "y_min": float(xy[:, 1].min()) if len(xy) else np.nan,
                "y_max": float(xy[:, 1].max()) if len(xy) else np.nan,
                "motion_span_x": float(span[0]), "motion_span_y": float(span[1]),
                "median_body_height": float(np.median(heights)),
                "conf_mean": float(a[:, :, 2].mean()), "conf_min": float(a[:, :, 2].min()),
                "conf_max": float(a[:, :, 2].max()),
            })
            conf_all.append(a[:, :, 2])
            x_all.append(a[:, :, 0][~zero_joint])
            y_all.append(a[:, :, 1][~zero_joint])
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, f"per_clip_{label}.csv"), index=False)
    print(f"files missing: {missing}")
    print("shapes:", dict(df["shape"].value_counts()))
    print("dtypes:", dict(df["dtype"].value_counts()))
    good = df[df["shape"] == str(EXPECTED_SHAPE)]
    print(f"non-finite clips: {(~good['finite']).sum()}")
    print(f"clips with any all-zero frame: {(good['zero_frames'] > 0).sum()} "
          f"(total zero frames {good['zero_frames'].sum()})")
    print(f"clips with any zeroed joint: {(good['zero_joint_entries'] > 0).sum()}")
    print(f"clips with repeated frames: {(good['repeated_frames'] > 0).sum()}, "
          f"with trailing repeats: {(good['trailing_repeats'] > 0).sum()} "
          f"(max {good['trailing_repeats'].max()})")

    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    conf = np.stack(conf_all)
    print(f"\nx range {x.min():.1f} to {x.max():.1f}, y range {y.min():.1f} to {y.max():.1f}")
    print(f"values outside [0, 1] present (i.e. pixel units): {bool((x > 1.5).any())}")
    for col in ["median_body_height", "motion_span_x", "motion_span_y"]:
        s = good[col]
        print(f"{col:20s} p5={s.quantile(.05):.1f} median={s.median():.1f} p95={s.quantile(.95):.1f}")
    print("\nper-joint mean confidence / fraction below 0.3:")
    for j, name in enumerate(JOINT_NAMES):
        c = conf[:, :, j]
        print(f"  {j:2d} {name:11s} {c.mean():.3f}  {(c < 0.3).mean():.3f}")
    print("\nmean confidence by class:")
    print(good.groupby("class_name")["conf_mean"].mean().round(3).to_string())
    return df


def compare_raw_smoothed(kept, data_root):
    section("5. raw vs smoothed consistency (random 200 clips)")
    rng = np.random.RandomState(0)
    sample = kept.sample(min(200, len(kept)), random_state=rng)
    diffs, conf_changed, missing = [], 0, 0
    for _, row in sample.iterrows():
        rel = os.path.join(row["class_name"], stem_of(row["filepath"]) + ".npy")
        pr, ps = os.path.join(data_root, "keypoints_raw", rel), os.path.join(data_root, "keypoints_smoothed", rel)
        if not (os.path.exists(pr) and os.path.exists(ps)):
            missing += 1
            continue
        r, s = np.load(pr), np.load(ps)
        if r.shape != s.shape:
            print("  shape mismatch:", rel, r.shape, s.shape)
            continue
        diffs.append(np.abs(r[:, :, :2] - s[:, :, :2]).mean())
        conf_changed += int(not np.allclose(r[:, :, 2], s[:, :, 2]))
    d = np.array(diffs)
    print(f"missing pairs: {missing}")
    print(f"mean abs xy change from smoothing: median={np.median(d):.2f}px max={d.max():.2f}px")
    print(f"clips where smoothing also changed confidence: {conf_changed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/data")
    parser.add_argument("--out", default="/workspace/outputs/audit")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = pd.read_csv(os.path.join(args.data_root, "split_manifest.csv"))
    log = pd.read_csv(os.path.join(args.data_root, "extraction_log.csv"))

    kept = audit_manifest(manifest, log, args.data_root)
    audit_leakage(kept, args.out)
    audit_arrays(kept, os.path.join(args.data_root, "keypoints_smoothed"), args.out, "smoothed")
    compare_raw_smoothed(kept, args.data_root)
    print(f"\nper-clip tables written to {args.out}")


if __name__ == "__main__":
    main()
