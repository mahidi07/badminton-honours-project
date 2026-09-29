"""
Shared data module for every model notebook (LSTM, Transformer Encoder, ST-GCN, ST-TR and model 5).

One implementation of the split, cleaning rules, normalisation and augmentation, so every architecture
sees exactly the same data and the comparison between them stays fair.

Contents
    load_filtered_manifest   frozen split, manual review exclusions, short clip rule (train only)
    BadmintonPoseDataset     per clip arrays: keypoints, bone, velocity, court position, label
    augmentations            "basic" and "advanced" policies, train split only
    ClassBalancedSampler     oversamples minority classes to a floor per epoch

Units. Keypoints are stored in pixels (1280x960 source video). Augmentation strengths are defined
relative to the player's body height, so they behave the same for near and far players and do not
depend on whether normalisation is applied afterwards.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# COCO 17 joint skeleton, same 18 edges used for the extraction QC overlays
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
NUM_JOINTS = 17
JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
L_HIP, R_HIP = 11, 12
FRAME_W, FRAME_H = 1280.0, 960.0

# left/right pairs, swapped when mirroring so joint indices keep their meaning
LR_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# anatomical limb chains for rotation: pivot joint -> joints that move with it
LIMB_CHAINS = {
    5: [7, 9],     # left shoulder: elbow, wrist
    6: [8, 10],    # right shoulder
    7: [9],        # left elbow: wrist
    8: [10],       # right elbow
    11: [13, 15],  # left hip: knee, ankle
    12: [14, 16],  # right hip
    13: [15],      # left knee: ankle
    14: [16],      # right knee
}


def build_parent_array(root=0):
    """Parent of each joint in a BFS tree over COCO_SKELETON rooted at the nose (used for bone vectors)."""
    from collections import deque

    adj = {i: [] for i in range(NUM_JOINTS)}
    for a, b in COCO_SKELETON:
        adj[a].append(b)
        adj[b].append(a)

    parent = [-1] * NUM_JOINTS
    parent[root] = root  # root bone is zero
    visited = {root}
    q = deque([root])
    while q:
        node = q.popleft()
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                q.append(nbr)
    return np.array(parent, dtype=int)


PARENT = build_parent_array()


def build_adjacency_matrix():
    """Binary undirected adjacency with self loops."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for a, b in COCO_SKELETON:
        A[a, b] = A[b, a] = 1.0
    return A


def compute_bone(keypoints_xy):
    """(T, 17, 2) -> (T, 17, 2), each joint minus its parent."""
    return keypoints_xy - keypoints_xy[:, PARENT, :]


def compute_velocity(keypoints_xy):
    """(T, 17, 2) -> (T, 17, 2), frame to frame difference, first frame zero."""
    velocity = np.zeros_like(keypoints_xy)
    velocity[1:] = keypoints_xy[1:] - keypoints_xy[:-1]
    return velocity


def body_height(keypoints):
    """Median over frames of the vertical extent of the skeleton, in the array's own units."""
    y = keypoints[:, :, 1]
    return max(float(np.median(y.max(axis=1) - y.min(axis=1))), 1e-3)


def hip_centre(keypoints):
    """(T, 2) midpoint of the hips per frame."""
    return keypoints[:, [L_HIP, R_HIP], :2].mean(axis=1)


# ---------------------------------------------------------------------------
# advanced augmentations: (T, 17, 3) x, y, confidence in, same shape out
# ---------------------------------------------------------------------------

def aug_limb_rotation(keypoints, max_angle_deg=12.0, p=0.5):
    """
    Rotates one limb about its proximal joint (shoulder, elbow, hip or knee) by a fixed small angle
    across the whole clip. Only the joints further along that limb move, so bone lengths are kept and
    the rest of the body is untouched.
    """
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    pivot = int(np.random.choice(list(LIMB_CHAINS)))
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    moving = LIMB_CHAINS[pivot]
    origin = kp[:, pivot:pivot + 1, :2]
    kp[:, moving, :2] = origin + (kp[:, moving, :2] - origin) @ rot.T
    return kp


def aug_confidence_scaled_noise(keypoints, base_frac=0.04, p=0.5):
    """
    Gaussian noise on x/y with standard deviation base_frac * body height * (1 - confidence), so joints
    the pose model was unsure about move more. At body height 135 px this is about 3.8 px for a joint at
    confidence 0.3 and 1.6 px at 0.7.
    """
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    std = base_frac * body_height(kp) * (1.0 - kp[:, :, 2:3])
    kp[:, :, :2] += np.random.randn(*kp[:, :, :2].shape).astype(np.float32) * std
    return kp


def aug_joint_masking(keypoints, max_joints=2, max_span=5, p=0.5):
    """
    Simulates a missed or occluded joint the way the pose model actually reports one: for a short span
    the position is a rough guess (linear between the frames either side, plus noise of 5% of body height)
    and the confidence drops to near zero. The previous version zeroed the coordinates, which in pixel
    units sent the joint to the corner of the frame.
    """
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    T = kp.shape[0]
    h = body_height(kp)
    for j in np.random.choice(NUM_JOINTS, size=np.random.randint(1, max_joints + 1), replace=False):
        span = np.random.randint(1, max_span + 1)
        start = np.random.randint(0, T - span + 1)
        before, after = kp[max(start - 1, 0), j, :2], kp[min(start + span, T - 1), j, :2]
        w = np.linspace(0, 1, span + 2)[1:-1, None]
        kp[start:start + span, j, :2] = (1 - w) * before + w * after
        kp[start:start + span, j, :2] += np.random.randn(span, 2).astype(np.float32) * 0.05 * h
        kp[start:start + span, j, 2] = np.random.uniform(0.0, 0.2, size=span)
    return kp


def aug_temporal_warp(keypoints, max_warp=0.4, num_control_points=4, p=0.5):
    """
    Smooth non uniform resampling in time: a few interior control points are shifted, so one part of the
    shot is stretched and another compressed. Endpoints stay fixed and time never runs backwards.
    """
    if np.random.rand() > p:
        return keypoints
    T = keypoints.shape[0]
    control_x = np.linspace(0, T - 1, num_control_points + 2)
    spacing = (T - 1) / (num_control_points + 1)
    control_y = control_x.copy()
    control_y[1:-1] += np.random.uniform(-max_warp, max_warp, size=num_control_points) * spacing
    control_y = np.sort(control_y)
    control_y[0], control_y[-1] = 0, T - 1
    positions = np.interp(np.arange(T), control_x, control_y)

    flat = keypoints.reshape(T, -1)
    out = np.stack([np.interp(positions, np.arange(T), flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape(keypoints.shape).astype(keypoints.dtype)


ADVANCED_AUGMENTATIONS = [aug_limb_rotation, aug_confidence_scaled_noise, aug_joint_masking, aug_temporal_warp]


# ---------------------------------------------------------------------------
# basic augmentations: the conventional whole skeleton set, kept for comparison
# ---------------------------------------------------------------------------

def aug_basic_rotate(keypoints, max_angle_deg=15.0, p=0.5):
    """Whole skeleton rotation about each frame's centroid."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    centroid = kp[:, :, :2].mean(axis=1, keepdims=True)
    kp[:, :, :2] = centroid + (kp[:, :, :2] - centroid) @ rot.T
    return kp


def aug_basic_scale(keypoints, scale_range=(0.85, 1.15), p=0.5):
    """Uniform scale about each frame's centroid (undone by body normalisation, so only matters in pixels)."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    centroid = kp[:, :, :2].mean(axis=1, keepdims=True)
    kp[:, :, :2] = centroid + (kp[:, :, :2] - centroid) * np.random.uniform(*scale_range)
    return kp


def aug_basic_jitter(keypoints, frac=0.03, p=0.5):
    """Uniform Gaussian noise on x/y, 3% of body height, regardless of confidence."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    kp[:, :, :2] += np.random.randn(*kp[:, :, :2].shape).astype(np.float32) * frac * body_height(kp)
    return kp


def aug_basic_mirror(keypoints, p=0.5):
    """Horizontal flip about the clip's mean x, swapping left and right joint indices."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    kp[:, :, 0] = 2 * kp[:, :, 0].mean() - kp[:, :, 0]
    for left, right in LR_SWAP_PAIRS:
        kp[:, [left, right], :] = kp[:, [right, left], :]
    return kp


def aug_basic_crop(keypoints, min_frac=0.7, p=0.5):
    """Random contiguous temporal crop resampled back to full length."""
    if np.random.rand() > p:
        return keypoints
    T = keypoints.shape[0]
    crop_len = np.random.randint(int(T * min_frac), T)
    start = np.random.randint(0, T - crop_len + 1)
    cropped = keypoints[start:start + crop_len].reshape(crop_len, -1)
    positions = np.linspace(0, crop_len - 1, T)
    out = np.stack([np.interp(positions, np.arange(crop_len), cropped[:, i]) for i in range(cropped.shape[1])], axis=1)
    return out.reshape(keypoints.shape).astype(keypoints.dtype)


BASIC_AUGMENTATIONS = [aug_basic_rotate, aug_basic_scale, aug_basic_jitter, aug_basic_mirror, aug_basic_crop]

AUGMENTATION_POLICIES = {
    "none": [],
    "basic": BASIC_AUGMENTATIONS,
    "advanced": ADVANCED_AUGMENTATIONS,
}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

NORMALISATIONS = ("pixel", "body")


def normalise_body(keypoints):
    """
    Centres the clip on its mean hip position and divides by body height. Movement within the clip is
    kept (a player stepping forward still moves), only where the player stands in the frame and how
    large they appear are removed. Confidence is unchanged.
    """
    kp = keypoints.copy()
    centre = hip_centre(kp).mean(axis=0)
    kp[:, :, :2] = (kp[:, :, :2] - centre) / body_height(kp)
    return kp


def court_position(keypoints):
    """(T, 2) hip midpoint per frame in frame units (0 to 1), computed before any normalisation."""
    return hip_centre(keypoints) / np.array([FRAME_W, FRAME_H], dtype=np.float32)


# ---------------------------------------------------------------------------
# dataset, manifest, sampler
# ---------------------------------------------------------------------------

class BadmintonPoseDataset(Dataset):
    """
    manifest: rows for one split (filepath, class_name, split), cleaning already applied.
    keypoints_root: e.g. /workspace/data/keypoints_interp.
    augmentation_policy: "none", "basic" or "advanced"; only ever applied to train rows.
    normalisation: "pixel" (stored units) or "body" (hip centred, body height units).

    Each item:
        keypoints (T, 17, 3)  x, y, confidence
        bone      (T, 17, 2)
        velocity  (T, 17, 2)
        court     (T, 2)      hip position in frame units, for models that use it
        label, filepath
    """

    def __init__(self, manifest, keypoints_root, class_to_id, augmentation_policy="none", normalisation="pixel"):
        if augmentation_policy not in AUGMENTATION_POLICIES:
            raise ValueError(f"augmentation_policy must be one of {list(AUGMENTATION_POLICIES)}")
        if normalisation not in NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
        self.manifest = manifest.reset_index(drop=True)
        self.keypoints_root = keypoints_root
        self.class_to_id = class_to_id
        self.augmentation_policy = augmentation_policy
        self.normalisation = normalisation

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        keypoints = np.load(os.path.join(self.keypoints_root, row["class_name"], stem + ".npy")).astype(np.float32)

        if row["split"] == "train":
            for aug_fn in AUGMENTATION_POLICIES[self.augmentation_policy]:
                keypoints = aug_fn(keypoints)

        court = court_position(keypoints)
        if self.normalisation == "body":
            keypoints = normalise_body(keypoints)

        xy = keypoints[:, :, :2]
        return {
            "keypoints": torch.from_numpy(keypoints),
            "bone": torch.from_numpy(compute_bone(xy)),
            "velocity": torch.from_numpy(compute_velocity(xy)),
            "court": torch.from_numpy(court.astype(np.float32)),
            "label": torch.tensor(self.class_to_id[row["class_name"]], dtype=torch.long),
            "filepath": row["filepath"],
        }


class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Draws every clip of classes at or above target_per_class once per epoch, and samples minority classes
    with replacement up to target_per_class. Each repeated draw gets its own augmentation. Must be built
    fresh for each training run (it carries an epoch counter).
    """

    def __init__(self, manifest, target_per_class=500, seed=None):
        self.target_per_class = target_per_class
        self.seed = seed
        self.epoch = 0
        self.class_indices = {
            c: g.index.tolist() for c, g in manifest.reset_index(drop=True).groupby("class_name")
        }

    def __iter__(self):
        rng = np.random.RandomState(None if self.seed is None else self.seed + self.epoch)
        indices = []
        for idx_list in self.class_indices.values():
            if len(idx_list) >= self.target_per_class:
                indices.extend(idx_list)
            else:
                indices.extend(rng.choice(idx_list, size=self.target_per_class, replace=True).tolist())
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self):
        return sum(max(len(v), self.target_per_class) for v in self.class_indices.values())


def build_class_balanced_sampler(manifest, target_per_class=500, seed=None):
    """Sampler for DataLoader(..., sampler=...). Pass the same train manifest the Dataset was built from."""
    return ClassBalancedSampler(manifest, target_per_class=target_per_class, seed=seed)


def load_filtered_manifest(manifest_path, extraction_log_path, repair_log_path=None, min_native_frames=None):
    """
    Frozen split with the cleaning rules applied. Every notebook should load data through this.

    Drops clips marked excluded at manual review (all splits). If repair_log_path and min_native_frames
    are given, also drops train clips with fewer real frames than min_native_frames, using n_native from
    the resampling repair log (the extraction log's frame count is stale for a few clips). Val and test
    are never filtered by length.
    """
    manifest = pd.read_csv(manifest_path)
    log = pd.read_csv(extraction_log_path)

    excluded = set(log.loc[log["excluded"] == True, "filepath"]) if "excluded" in log else set()
    before = len(manifest)
    manifest = manifest[~manifest["filepath"].isin(excluded)]
    print(f"manual review: dropped {before - len(manifest)} clips")

    if repair_log_path is not None and min_native_frames is not None:
        n_native = pd.read_csv(repair_log_path).set_index("filepath")["n_native"]
        too_short = (manifest["split"] == "train") & (manifest["filepath"].map(n_native) < min_native_frames)
        manifest = manifest[~too_short]
        print(f"short clip rule: dropped {int(too_short.sum())} train clips with fewer than {min_native_frames} real frames")

    manifest = manifest.reset_index(drop=True)
    print(f"{len(manifest)} clips remain ({', '.join(f'{s} {n}' for s, n in manifest['split'].value_counts().items())})")
    return manifest
