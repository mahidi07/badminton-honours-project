"""
Shared data and model code for the two player experiments (notebook 32).

Every arm, including the control on the old single player data, goes through the same dataset class, the same
ST-GCN backbone and the same head, so the only thing that differs between arms is which skeleton(s) the model sees.

PersonsDataset
    root       folder with <class>/<stem>.npy; each file is (30, 17, 3) (old single player data) or (2, 30, 17, 3)
               (person 0 = nearer the camera, person 1 = further away; an absent person is all zeros)
    persons    which persons to return, e.g. (0,), (1,) or (0, 1)
    Returns keypoints (M, T, 17, 3), bone (M, T, 17, 2), court (M, T, 2); an absent person is zeros throughout.
    donors     optional dict filepath -> (class_name, stem). When given, person 1 of each clip is replaced by person 1 of the
               donor clip (the shuffled far player control of notebook 33); person 0 and everything else are untouched.
    Augmentation (training clips only) draws the same random numbers for every person, so a rotation, a scale, a
    time crop or a time warp is the same for both players; their positions relative to each other are kept.

PersonsSTGCN
    The ST-GCN backbone of models/stgcn.py (9 blocks, COCO graph, spatial partitioning), applied to each person with shared
    weights; each person's feature map is averaged over time and joints (256 values) and the persons' vectors are
    concatenated, in the fixed person order, before one linear layer. With one person this is exactly mmaction2's
    GCNHead (global average, then a linear layer); with two it keeps who is who.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dataset as ds
import training as tr
from models import stgcn as stgcn_model


def augment_persons(kps, policy):
    """Apply the augmentation policy to each person with identical random draws. kps: list of (T, 17, 3)."""
    state = np.random.get_state()
    out = []
    for kp in kps:
        if not np.any(kp[:, :, 2] > 0):                    # absent person: nothing to augment
            out.append(kp)
            continue
        np.random.set_state(state)
        for fn in ds.AUGMENTATION_POLICIES[policy]:
            kp = fn(kp)
        out.append(kp)
    return out


class PersonsDataset(Dataset):
    def __init__(self, manifest, root, class_to_id, persons=(0,), augmentation_policy="none", normalisation="body", donors=None):
        self.donors = donors
        self.manifest = manifest.reset_index(drop=True)
        self.root, self.class_to_id, self.persons = root, class_to_id, tuple(persons)
        self.policy, self.normalisation = augmentation_policy, normalisation

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        arr = np.load(os.path.join(self.root, row["class_name"], stem + ".npy")).astype(np.float32)
        if arr.ndim == 3:                                   # old single player files
            arr = arr[None]
        if self.donors is not None:
            donor_class, donor_stem = self.donors[row["filepath"]]
            arr = arr.copy()
            arr[1] = np.load(os.path.join(self.root, donor_class, donor_stem + ".npy")).astype(np.float32)[1]
        kps = [arr[p] for p in self.persons]
        if row["split"] == "train" and self.policy != "none":
            kps = augment_persons(kps, self.policy)
        keypoints, bones, courts = [], [], []
        for kp in kps:
            if not np.any(kp[:, :, 2] > 0):
                keypoints.append(np.zeros_like(kp)); bones.append(np.zeros(kp.shape[:2] + (2,), np.float32))
                courts.append(np.zeros((kp.shape[0], 2), np.float32))
                continue
            court = ds.court_position(kp)
            if self.normalisation == "body":
                kp = ds.normalise_body(kp)
            keypoints.append(kp)
            bones.append(ds.compute_bone(kp[:, :, :2]))
            courts.append(court.astype(np.float32))
        return {
            "keypoints": torch.from_numpy(np.stack(keypoints)),
            "bone": torch.from_numpy(np.stack(bones)),
            "court": torch.from_numpy(np.stack(courts)),
            "label": torch.tensor(self.class_to_id[row["class_name"]], dtype=torch.long),
            "filepath": row["filepath"],
        }


def build_loaders(manifest, root, class_to_id, persons, augmentation, seed, batch_size=16, eval_batch_size=64,
                  num_workers=4, eval_workers=4, splits=("train", "val"), normalisation="body", donors=None, target_per_class=None):
    """train is augmented; train_eval, val (and test if asked) are not.
    target_per_class=None: natural class frequencies (no sampler). An integer: the class balanced sampler of dataset.py
    (classes with fewer clips are drawn with replacement up to that many per epoch), as in the single player baselines."""
    parts = {s: manifest[manifest["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    make = lambda split, aug: PersonsDataset(parts[split], root, class_to_id, persons, aug, normalisation, donors)
    generator = torch.Generator()
    generator.manual_seed(seed)
    eval_kwargs = dict(batch_size=eval_batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True,
                       persistent_workers=eval_workers > 0)
    if target_per_class is None:
        order = dict(shuffle=True, generator=generator)
    else:
        order = dict(sampler=ds.build_class_balanced_sampler(parts["train"], target_per_class=target_per_class, seed=seed))
    loaders = {
        "train": DataLoader(make("train", augmentation), batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0, worker_init_fn=tr.seed_worker, **order),
        "train_eval": DataLoader(make("train", "none"), **eval_kwargs),
        "val": DataLoader(make("val", "none"), **eval_kwargs),
    }
    if "test" in splits:
        loaders["test"] = DataLoader(make("test", "none"), **eval_kwargs)
    return loaders


class PersonsSTGCN(nn.Module):
    def __init__(self, in_channels, num_classes, n_persons):
        super().__init__()
        self.backbone = stgcn_model.build_stgcn(in_channels, num_classes).backbone
        self.fc = nn.Linear(256 * n_persons, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)           # the same initialisation as mmaction2's GCNHead
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):                                   # x: (N, M, T, V, C)
        f = self.backbone(x)                                # (N, M, 256, T', V)
        return self.fc(f.mean(dim=(3, 4)).flatten(1))


def make_forward(use_court=True):
    def forward(model, batch, device):
        parts = [batch["keypoints"], batch["bone"]]
        if use_court:
            parts.append(batch["court"][:, :, :, None, :].expand(-1, -1, -1, ds.NUM_JOINTS, -1))
        x = torch.cat(parts, dim=-1).float().to(device, non_blocking=True)
        return model(x)
    return forward


def n_channels(use_court=True):
    return stgcn_model.n_channels(use_court)
