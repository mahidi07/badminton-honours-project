"""
Rebuilds the 30-frame keypoint arrays with linear interpolation instead of
frame duplication.

The original extraction mapped each clip's valid frames onto 30 slots with
standardize_indices (np.linspace(...).round()). For clips shorter than 30
frames that duplicates frames, and the Savitzky-Golay smoothing then ran on the
duplicated sequence. Because every native frame is hit at least once when
n < 30, the native frames can be recovered exactly from keypoints_raw without
re-running pose estimation.

The native frame count n is read from each raw array's own duplicate pattern,
not from extraction_log.csv: a handful of clips re-extracted during manual
review kept a stale num_valid_track_frames in the log.

Per clip:
    no duplicates    raw array is a plain subsample, smoothed exactly as before
                     (result must equal keypoints_smoothed)
    duplicates that  recover the n native frames, smooth them at the native rate
    match the map    (n >= 5 only, the filter window), then interpolate to 30
    anything else    left as the original smoothed array and flagged
Confidence is interpolated like x/y and clipped to [0, 1].

Reads keypoints_raw, writes a new folder, never modifies existing data.

    python repair_resampling.py [--data-root /workspace/data] [--out-name keypoints_interp]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

N_TARGET = 30
SG_WINDOW, SG_POLY = 5, 2


def standardize_indices(n_available, n_target=N_TARGET):
    # identical to pipeline/pose_extraction_utils.py, used here to invert it
    if n_available <= 0:
        return np.zeros(n_target, dtype=int)
    return np.linspace(0, n_available - 1, n_target).round().astype(int)


def smooth(seq):
    if seq.shape[0] < SG_WINDOW:
        return seq.copy()
    out = seq.copy()
    out[:, :, :2] = savgol_filter(seq[:, :, :2], SG_WINDOW, SG_POLY, axis=0)
    return out


def count_native_frames(raw):
    """Distinct consecutive frames in a raw array, and whether the duplicate
    pattern is exactly what standardize_indices produces for that count."""
    same = np.r_[False, np.abs(np.diff(raw, axis=0)).sum(axis=(1, 2)) == 0]
    n = int((~same).sum())
    expected = np.r_[False, np.diff(standardize_indices(n)) == 0]
    return n, bool(np.array_equal(same, expected))


def recover_native(raw, n):
    """Returns the n native frames and the largest disagreement between duplicates."""
    idx = standardize_indices(n)
    if len(np.unique(idx)) != n:
        raise ValueError(f"index map covers {len(np.unique(idx))} frames, expected {n}")
    native = np.zeros((n,) + raw.shape[1:], dtype=raw.dtype)
    worst = 0.0
    for k in range(n):
        slots = raw[idx == k]
        native[k] = slots[0]
        worst = max(worst, float(np.abs(slots - slots[0]).max()))
    return native, worst


def resample_linear(seq, n_target=N_TARGET):
    n = seq.shape[0]
    if n == 1:
        return np.repeat(seq, n_target, axis=0)
    src = np.arange(n)
    dst = np.linspace(0, n - 1, n_target)
    flat = seq.reshape(n, -1)
    out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((n_target,) + seq.shape[1:]).astype(seq.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/data")
    parser.add_argument("--out-name", default="keypoints_interp")
    args = parser.parse_args()

    raw_root = os.path.join(args.data_root, "keypoints_raw")
    old_root = os.path.join(args.data_root, "keypoints_smoothed")
    out_root = os.path.join(args.data_root, args.out_name)
    if os.path.exists(out_root):
        raise SystemExit(f"{out_root} already exists, remove it first if you mean to rebuild")

    log = pd.read_csv(os.path.join(args.data_root, "extraction_log.csv"))
    rows = []
    for _, r in log.iterrows():
        stem = os.path.splitext(os.path.basename(r["filepath"]))[0]
        rel = os.path.join(r["class_name"], stem + ".npy")
        raw = np.load(os.path.join(raw_root, rel)).astype(np.float32)
        old = np.load(os.path.join(old_root, rel))
        n_logged = int(r["num_valid_track_frames"])
        n, pattern_ok = count_native_frames(raw)

        dup_err = 0.0
        if n == N_TARGET:
            mode = "subsampled"
            new = smooth(raw)
        elif pattern_ok:
            mode = "interpolated"
            native, dup_err = recover_native(raw, n)
            new = resample_linear(smooth(native))
        else:
            mode = "unchanged_flagged"
            new = old.astype(np.float32).copy()
        new[:, :, 2] = np.clip(new[:, :, 2], 0.0, 1.0)

        repeats = int((np.abs(np.diff(new, axis=0)).sum(axis=(1, 2)) == 0).sum())
        rows.append({
            "filepath": r["filepath"], "n_native": n, "n_logged": n_logged, "mode": mode,
            "duplicate_disagreement": dup_err,
            "max_abs_diff_vs_old_xy": float(np.abs(new[:, :, :2] - old[:, :, :2]).max()),
            "repeated_frames_after": repeats,
        })
        os.makedirs(os.path.join(out_root, r["class_name"]), exist_ok=True)
        np.save(os.path.join(out_root, rel), new)

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(args.data_root, f"{args.out_name}_log.csv"), index=False)

    sub = report[report["mode"] == "subsampled"]
    inter = report[report["mode"] == "interpolated"]
    print(f"clips written: {len(report)} to {out_root}")
    print(f"subsampled (no duplicates): {len(sub)}, max diff vs keypoints_smoothed = "
          f"{sub['max_abs_diff_vs_old_xy'].max():.6f} (should be ~0)")
    print(f"interpolated (duplicates removed): {len(inter)}, max duplicate disagreement in raw = "
          f"{inter['duplicate_disagreement'].max():.6f} (should be 0)")
    print(f"interpolated clips still containing repeated frames: "
          f"{(inter['repeated_frames_after'] > 0).sum()} (only n = 1 clips expected)")
    print(f"n = 1 clips: {(report['n_native'] == 1).sum()}")
    print(f"flagged, left unchanged: {(report['mode'] == 'unchanged_flagged').sum()}")
    stale = report[report["n_native"] != report["n_logged"].clip(upper=N_TARGET)]
    print(f"clips where the log's frame count is stale: {len(stale)}")
    if len(stale):
        print(stale[["filepath", "n_logged", "n_native", "mode"]].to_string())
    print(f"median / max xy change on interpolated clips: "
          f"{inter['max_abs_diff_vs_old_xy'].median():.2f} / {inter['max_abs_diff_vs_old_xy'].max():.2f} px")


if __name__ == "__main__":
    main()
