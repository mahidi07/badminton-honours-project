"""
Transformer Encoder baseline: a standard pre-norm Transformer encoder over per-frame pose features, with a
learned [CLS] token and learned positional embeddings. No skeletal graph structure (like LSTM) but models
relationships between every pair of frames directly via self-attention, rather than LSTM's purely
sequential recurrence.

Per frame input is the same flattened (x, y, confidence, bone dx, bone dy[, court x, court y]) vector used
by the LSTM baseline, so the two are directly comparable on input representation; only the sequence model
differs.
"""

import math

import torch
import torch.nn as nn

import dataset as ds


class TransformerClassifier(nn.Module):
    def __init__(self, in_features, d_model=192, nhead=6, num_layers=4, dim_feedforward=None,
                num_classes=18, dropout=0.2, max_len=31):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.input_proj = nn.Linear(in_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (N, T, F)
        n = x.shape[0]
        x = self.input_proj(x)
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # (N, T+1, d_model), position 0 is [CLS]
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.encoder(x)
        cls_out = self.dropout(self.norm(x[:, 0]))
        return self.fc(cls_out)


def build_transformer(in_features, num_classes, d_model=192, nhead=6, num_layers=4, dropout=0.2):
    return TransformerClassifier(in_features, d_model, nhead, num_layers, num_classes=num_classes,
                                 dropout=dropout, max_len=31)


def in_features(use_court):
    return ds.NUM_JOINTS * (7 if use_court else 5)


def make_forward(use_court):
    """Returns forward(model, batch, device) -> logits, matching models.stgcn / models.lstm's interface."""
    def forward(model, batch, device):
        parts = [batch["keypoints"], batch["bone"]]
        if use_court:
            parts.append(batch["court"][:, :, None, :].expand(-1, -1, ds.NUM_JOINTS, -1))
        x = torch.cat(parts, dim=-1).float()          # (N, T, V, C)
        n, t, v, c = x.shape
        x = x.reshape(n, t, v * c).to(device, non_blocking=True)
        return model(x)
    return forward


def warmup_cosine_scheduler(optimizer, warmup_epochs, num_epochs):
    """Linear warmup from 10% to full lr, then cosine decay to 0, stepped once per epoch."""
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
