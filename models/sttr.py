"""
ST-TR style spatial temporal transformer (single stream), after Plizzari et al. (2021).

The original work trains a spatial self attention (SSA) stream and a temporal self attention (TSA) stream
separately and ensembles their scores. Here the two are combined in one model, the single stream variant
the authors also describe, so the result is one model rather than an ensemble. This is an implementation
following the documented design, not the authors' released code.

Each block applies SSA (joint to joint attention within every frame) and then TSA (frame to frame attention
within every joint), with a residual connection, in the place of ST-GCN's graph convolution and temporal
convolution. The backbone has the same shape as the ST-GCN used in this project: nine blocks, 64 to 128 to 256
channels, temporal stride 2 at blocks 4 and 7.

Self attention has no sense of order, so learned positional embeddings are added: one per joint inside SSA and
one per frame inside TSA, sized for the number of frames the block actually sees (30, then 15, then 7).
Without them the model cannot tell a wind up from a follow through, which is what limited the first version.
"""

import torch
import torch.nn as nn

import dataset as ds


class _Attention(nn.Module):
    def __init__(self, channels, num_tokens, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def attend(self, tokens):
        tokens = tokens + self.pos
        out, _ = self.mha(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + out)


class SpatialSelfAttention(_Attention):
    """Joint to joint attention within each frame; tokens are the V joints of one frame."""

    def forward(self, x):
        n, c, t, v = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(n * t, v, c)
        return self.attend(tokens).reshape(n, t, v, c).permute(0, 3, 1, 2)


class TemporalSelfAttention(_Attention):
    """Frame to frame attention within each joint; tokens are the T frames of one joint."""

    def forward(self, x):
        n, c, t, v = x.shape
        tokens = x.permute(0, 3, 2, 1).reshape(n * v, t, c)
        return self.attend(tokens).reshape(n, v, t, c).permute(0, 3, 2, 1)


class STTRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_frames, num_joints, num_heads=8, stride=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.ssa = SpatialSelfAttention(out_channels, num_joints, num_heads, dropout)
        self.tsa = TemporalSelfAttention(out_channels, num_frames, num_heads, dropout)
        pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1)) if stride > 1 else nn.Identity()
        self.pool = pool
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), pool if stride > 1 else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        out = self.tsa(self.ssa(self.in_proj(x)))
        return self.act(self.pool(out) + res)


class STTR(nn.Module):
    def __init__(self, in_channels=7, num_classes=18, num_frames=30, num_joints=17, base_channels=64,
                 num_stages=9, inflate_stages=(4, 7), down_stages=(4, 7), num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, base_channels, 1)
        blocks, channels, frames = [], base_channels, num_frames
        self.frames_per_block = []
        for stage in range(1, num_stages + 1):
            out_channels = channels * 2 if stage in inflate_stages else channels
            stride = 2 if stage in down_stages else 1
            self.frames_per_block.append(frames)
            blocks.append(STTRBlock(channels, out_channels, frames, num_joints, num_heads, stride, dropout))
            channels = out_channels
            if stride > 1:
                frames = (frames - stride) // stride + 1
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        # x: (N, C, T, V)
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return self.fc(self.dropout(out.mean(dim=(2, 3))))


def build_sttr(in_channels, num_classes, dropout=0.1, num_heads=8):
    return STTR(in_channels=in_channels, num_classes=num_classes, dropout=dropout, num_heads=num_heads)


def in_channels(use_court):
    return 7 if use_court else 5


def make_forward(use_court):
    """Returns forward(model, batch, device) -> logits, matching the other model modules."""
    def forward(model, batch, device):
        parts = [batch["keypoints"], batch["bone"]]
        if use_court:
            parts.append(batch["court"][:, :, None, :].expand(-1, -1, ds.NUM_JOINTS, -1))
        x = torch.cat(parts, dim=-1).float().permute(0, 3, 1, 2)      # (N, C, T, V)
        return model(x.to(device, non_blocking=True))
    return forward
