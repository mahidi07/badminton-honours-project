"""
Shared dataset loader for all four model notebooks (LSTM, ST-GCN, Transformer,
ST-TR). Built once here so every architecture reads from the identical data
pipeline - same split, same derived streams, same augmentations - rather than
each notebook rolling its own and risking subtle inconsistencies between them.

Handles:
  - loading the raw or smoothed keypoint arrays against the frozen split manifest
  - dropping the 2 clips excluded during manual review (real hitter never tracked)
  - deriving bone vectors (joint - parent joint) and velocity (frame-to-frame delta)
  - the four augmentation techniques from the plan, train-split only:
      bone-segment rotation, confidence-scaled joint noise,
      random joint masking, temporal warping
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# same 18 edges used for drawing overlays during extraction - kept identical
# so the "skeleton" here and the one you were eyeballing in the QC images are
# literally the same graph, not two definitions that could quietly drift apart
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
NUM_JOINTS = 17


def build_parent_array(root=0):
    """
    Derives a parent-per-joint array from COCO_SKELETON via BFS from the
    root joint (nose). Doing it this way instead of hand-writing the tree
    means it's guaranteed consistent with the skeleton actually drawn in
    every overlay image from the extraction pipeline - one source of truth.
    """
    from collections import deque

    adj = {i: [] for i in range(NUM_JOINTS)}
    for a, b in COCO_SKELETON:
        adj[a].append(b)
        adj[b].append(a)

    parent = [-1] * NUM_JOINTS
    visited = [False] * NUM_JOINTS
    visited[root] = True
    parent[root] = root  # root is its own parent, bone vector will be zero

    q = deque([root])
    while q:
        node = q.popleft()
        for nbr in adj[node]:
            if not visited[nbr]:
                visited[nbr] = True
                parent[nbr] = node
                q.append(nbr)

    return np.array(parent, dtype=int)


PARENT = build_parent_array()


def build_adjacency_matrix():
    """Binary undirected adjacency + self-loops, the base graph ST-GCN needs."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for a, b in COCO_SKELETON:
        A[a, b] = 1.0
        A[b, a] = 1.0
    return A


def compute_bone(keypoints_xy):
    """keypoints_xy: (T, 17, 2). Returns (T, 17, 2) - each joint minus its parent."""
    return keypoints_xy - keypoints_xy[:, PARENT, :]


def compute_velocity(keypoints_xy):
    """keypoints_xy: (T, 17, 2). Frame-to-frame delta, first frame's velocity is zero."""
    velocity = np.zeros_like(keypoints_xy)
    velocity[1:] = keypoints_xy[1:] - keypoints_xy[:-1]
    return velocity


# ---------------------------------------------------------------------------
# augmentations - all operate on a (T, 17, 3) array (x, y, confidence) and
# only ever get called on the train split
# ---------------------------------------------------------------------------

def aug_bone_rotation(keypoints, max_angle_deg=12.0, p=0.5):
    """
    Rotates a random limb subtree around its parent joint by a small angle.
    More anatomically realistic than rotating the whole skeleton at once -
    a real shoulder rotation doesn't move the other arm.
    """
    if np.random.rand() > p:
        return keypoints

    kp = keypoints.copy()
    children = {i: [] for i in range(NUM_JOINTS)}
    for child, parent in enumerate(PARENT):
        if child != parent:
            children[parent].append(child)

    pivot = np.random.choice([j for j in range(NUM_JOINTS) if children[j]])

    # every joint in the pivot's subtree gets rotated together
    subtree = []
    stack = [pivot]
    while stack:
        node = stack.pop()
        for c in children[node]:
            subtree.append(c)
            stack.append(c)

    if not subtree:
        return kp

    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    for t in range(kp.shape[0]):
        origin = kp[t, pivot, :2]
        for j in subtree:
            offset = kp[t, j, :2] - origin
            kp[t, j, :2] = origin + offset @ rot.T

    return kp


def aug_confidence_scaled_noise(keypoints, base_std=3.0, p=0.5):
    """
    Adds Gaussian noise to x/y, scaled up where confidence is already low.
    Trains robustness specifically to the low-confidence joints the pose
    pipeline actually produces (occlusion, motion blur), rather than uniform
    noise everywhere regardless of how trustworthy a point already is.
    """
    if np.random.rand() > p:
        return keypoints

    kp = keypoints.copy()
    conf = kp[:, :, 2:3]
    noise_std = base_std * (1.0 - conf)
    noise = np.random.randn(*kp[:, :, :2].shape).astype(np.float32) * noise_std
    kp[:, :, :2] += noise
    return kp


def aug_joint_masking(keypoints, max_joints=2, max_span=5, p=0.5):
    """
    Zeroes out a random joint (or two) for a short span of frames, simulating
    a missed detection - the exact failure mode occlusion actually produces,
    rather than a generic dropout.
    """
    if np.random.rand() > p:
        return keypoints

    kp = keypoints.copy()
    T = kp.shape[0]
    n_joints = np.random.randint(1, max_joints + 1)
    joints = np.random.choice(NUM_JOINTS, size=n_joints, replace=False)

    for j in joints:
        span = np.random.randint(1, max_span + 1)
        start = np.random.randint(0, max(1, T - span))
        kp[start:start + span, j, :] = 0.0

    return kp


def aug_temporal_warp(keypoints, max_warp=0.4, num_control_points=4, p=0.5):
    """
    Non-uniform resampling along time using a small number of smoothly
    interpolated control points, rather than independent per-frame noise.

    The first version of this used per-step random multipliers and summed
    them - with 30 mostly-independent steps averaging together, the result
    collapsed back toward a straight line (basic CLT behaviour), so the
    "warp" was barely distinguishable from the original on inspection. This
    version perturbs a handful of control points instead, which keeps the
    warp coherent - genuinely stretching one part of the sequence and
    compressing another - rather than noise that cancels itself out.
    """
    if np.random.rand() > p:
        return keypoints

    T = keypoints.shape[0]
    control_x = np.linspace(0, T - 1, num_control_points + 2)
    spacing = (T - 1) / (num_control_points + 1)

    control_y = control_x.copy()
    control_y[1:-1] += np.random.uniform(-max_warp, max_warp, size=num_control_points) * spacing
    control_y = np.sort(control_y)  # time can't run backwards
    control_y[0], control_y[-1] = 0, T - 1  # endpoints pinned, only the interior warps

    warped_positions = np.interp(np.arange(T), control_x, control_y)

    src_indices = np.arange(T)
    kp_out = np.zeros_like(keypoints)
    for j in range(NUM_JOINTS):
        for c in range(keypoints.shape[2]):
            kp_out[:, j, c] = np.interp(warped_positions, src_indices, keypoints[:, j, c])

    return kp_out


AUGMENTATIONS = [aug_bone_rotation, aug_confidence_scaled_noise, aug_joint_masking, aug_temporal_warp]


# ---------------------------------------------------------------------------
# "basic" augmentations - the original rotate/scale/jitter/mirror/crop set,
# kept around so the paper can actually compare against them rather than
# just asserting the targeted set is better
# ---------------------------------------------------------------------------

# COCO left/right joint pairs - mirroring has to swap these indices, not just
# negate x, or a left wrist ends up sitting at the right wrist's index and
# every downstream joint-semantic (e.g. "joint 9 is always the left wrist")
# silently breaks
LR_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def aug_basic_rotate(keypoints, max_angle_deg=15.0, p=0.5):
    """Whole-skeleton rotation around its centroid - the original, coarser version."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    for t in range(kp.shape[0]):
        centroid = kp[t, :, :2].mean(axis=0)
        kp[t, :, :2] = centroid + (kp[t, :, :2] - centroid) @ rot.T
    return kp


def aug_basic_scale(keypoints, scale_range=(0.85, 1.15), p=0.5):
    """Uniform scale around the per-frame centroid."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    scale = np.random.uniform(*scale_range)
    for t in range(kp.shape[0]):
        centroid = kp[t, :, :2].mean(axis=0)
        kp[t, :, :2] = centroid + (kp[t, :, :2] - centroid) * scale
    return kp


def aug_basic_jitter(keypoints, std=4.0, p=0.5):
    """Uniform Gaussian noise on x/y - not confidence-scaled, same amount everywhere."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    kp[:, :, :2] += np.random.randn(*kp[:, :, :2].shape).astype(np.float32) * std
    return kp


def aug_basic_mirror(keypoints, p=0.5):
    """Horizontal flip - negates x AND swaps left/right joint indices."""
    if np.random.rand() > p:
        return keypoints
    kp = keypoints.copy()
    center_x = kp[:, :, 0].mean()
    kp[:, :, 0] = 2 * center_x - kp[:, :, 0]
    for left, right in LR_SWAP_PAIRS:
        kp[:, [left, right], :] = kp[:, [right, left], :]
    return kp


def aug_basic_crop(keypoints, min_frac=0.7, p=0.5):
    """Random contiguous temporal crop, resampled back to the full length."""
    if np.random.rand() > p:
        return keypoints
    T = keypoints.shape[0]
    crop_len = np.random.randint(int(T * min_frac), T)
    start = np.random.randint(0, T - crop_len + 1)
    cropped = keypoints[start:start + crop_len]

    src_indices = np.linspace(0, crop_len - 1, T)
    kp_out = np.zeros_like(keypoints)
    base_indices = np.arange(crop_len)
    for j in range(NUM_JOINTS):
        for c in range(keypoints.shape[2]):
            kp_out[:, j, c] = np.interp(src_indices, base_indices, cropped[:, j, c])
    return kp_out


BASIC_AUGMENTATIONS = [aug_basic_rotate, aug_basic_scale, aug_basic_jitter, aug_basic_mirror, aug_basic_crop]
ADVANCED_AUGMENTATIONS = [aug_bone_rotation, aug_confidence_scaled_noise, aug_joint_masking, aug_temporal_warp]

# kept for backwards compatibility with the QC notebook already built against this name
AUGMENTATIONS = ADVANCED_AUGMENTATIONS


AUGMENTATION_POLICIES = {
    "none": [],
    "basic": BASIC_AUGMENTATIONS,
    "advanced": ADVANCED_AUGMENTATIONS,
}


class BadmintonPoseDataset(Dataset):
    """
    manifest: dataframe with filepath, class_name, split columns (already
    filtered to the split you want and with excluded clips dropped).
    keypoints_root: /workspace/data/keypoints_raw or keypoints_smoothed.
    augmentation_policy: "none", "basic", or "advanced" - only ever actually
    applies anything when split == "train"; val/test rows pass through
    untouched regardless of what policy is set, by design.
    """

    def __init__(self, manifest, keypoints_root, class_to_id, augmentation_policy="none"):
        if augmentation_policy not in AUGMENTATION_POLICIES:
            raise ValueError(f"augmentation_policy must be one of {list(AUGMENTATION_POLICIES)}")
        self.manifest = manifest.reset_index(drop=True)
        self.keypoints_root = keypoints_root
        self.class_to_id = class_to_id
        self.augmentation_policy = augmentation_policy

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        npy_path = os.path.join(self.keypoints_root, row["class_name"], stem + ".npy")
        keypoints = np.load(npy_path).astype(np.float32)  # (30, 17, 3)

        if row["split"] == "train":
            for aug_fn in AUGMENTATION_POLICIES[self.augmentation_policy]:
                keypoints = aug_fn(keypoints)

        bone = compute_bone(keypoints[:, :, :2])
        velocity = compute_velocity(keypoints[:, :, :2])

        label = self.class_to_id[row["class_name"]]

        return {
            "keypoints": torch.from_numpy(keypoints),      # (T, 17, 3) - x, y, conf
            "bone": torch.from_numpy(bone),                # (T, 17, 2)
            "velocity": torch.from_numpy(velocity),        # (T, 17, 2)
            "label": torch.tensor(label, dtype=torch.long),
            "filepath": row["filepath"],
        }


class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Oversamples minority classes up to `target_per_class` draws per epoch.
    Majority classes (already at or above target) are seen at their natural
    count, once each - not squeezed down to match minority classes. This is
    the "floor minority classes to 500" strategy from the original run:
    touches only the classes that actually need more exposure, leaves
    majority classes untouched.

    This is a separate lever from the augmentation_policy on the Dataset
    itself - pair this sampler with whichever policy (none/basic/advanced)
    you're testing. Each oversampled draw of a minority clip still gets its
    own independent random augmentation, so it's not literally showing the
    model identical duplicates even when the same clip gets drawn twice.

    Worth remembering going in: this targets macro-F1 and Mean Class
    Accuracy specifically. It won't move top-1 much by itself, since top-1
    is dominated by majority-class volume which this deliberately doesn't
    touch - that's expected, not a sign it isn't working.
    """

    def __init__(self, manifest, target_per_class=500, seed=None):
        # manifest MUST be the same already-reset-index dataframe the paired
        # Dataset was built from, or the positional indices here won't line
        # up with what the Dataset's __getitem__ expects
        self.target_per_class = target_per_class
        self.seed = seed
        self.epoch = 0
        self.class_indices = {
            class_name: group.index.tolist()
            for class_name, group in manifest.reset_index(drop=True).groupby("class_name")
        }

    def __iter__(self):
        seed = None if self.seed is None else self.seed + self.epoch
        rng = np.random.RandomState(seed)

        indices = []
        for idx_list in self.class_indices.values():
            n = len(idx_list)
            if n >= self.target_per_class:
                indices.extend(idx_list)  # natural count, no boost
            else:
                sampled = rng.choice(idx_list, size=self.target_per_class, replace=True)
                indices.extend(sampled.tolist())

        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self):
        return sum(max(len(idx_list), self.target_per_class) for idx_list in self.class_indices.values())


def build_class_balanced_sampler(manifest, target_per_class=500, seed=None):
    """
    Convenience wrapper - takes the same manifest you'd pass to
    BadmintonPoseDataset (already filtered to split=="train") and returns a
    sampler ready to hand to DataLoader(..., sampler=...). Don't pass
    shuffle=True alongside a sampler - DataLoader doesn't allow both.
    """
    return ClassBalancedSampler(manifest, target_per_class=target_per_class, seed=seed)


def build_three_train_datasets(manifest, keypoints_root, class_to_id):
    """
    One shared underlying train split, three augmentation policies. Not
    three copies on disk - three Dataset objects, each applying a different
    live transform to the same .npy files.
    """
    train_manifest = manifest[manifest["split"] == "train"]
    return {
        policy: BadmintonPoseDataset(train_manifest, keypoints_root, class_to_id, augmentation_policy=policy)
        for policy in AUGMENTATION_POLICIES
    }


def load_filtered_manifest(manifest_path, extraction_log_path):
    """
    Joins the frozen split manifest against the extraction log and drops
    anything flagged excluded during manual review. This is the one place
    that logic lives - every model notebook should call this rather than
    reading split_manifest.csv directly, or the excluded clips creep back in.
    """
    manifest = pd.read_csv(manifest_path)
    log = pd.read_csv(extraction_log_path)

    excluded_paths = set(log[log.get("excluded", False) == True]["filepath"])
    before = len(manifest)
    manifest = manifest[~manifest["filepath"].isin(excluded_paths)].reset_index(drop=True)
    after = len(manifest)

    if before != after:
        print(f"dropped {before - after} manually-excluded clips from the manifest ({after} remain)")

    return manifest
