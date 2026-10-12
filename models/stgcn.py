"""
ST-GCN (Yan et al., 2018) built from mmaction2's backbone with the paper configuration.

    9 stages, channels 64 -> 128 -> 256, temporal stride 2 at stages 4 and 7, COCO graph with the
    spatial configuration partitioning, followed by mmaction2's GCNHead.

The configuration was checked against the mmaction2 backbone source; the toolbox's own default config is a
modernised variant and differs from the paper.
"""

import torch

import dataset as ds


POOLING_HEADS = {"avg": None, "attn_time": "time", "attn_joint_time": "joint_time"}


def _register_attentive_head():
    """Registers AttentiveGCNHead with mmaction2 (once). It replaces GCNHead's global average over time and joints
    with a learned weighted average, and is initialised so that it is exactly that average: the attention scores
    start at zero, the softmax over them is uniform, and the classifier is the same linear layer. Training can then
    move the weights away from uniform, and nothing else about the model differs from the plain head."""
    from mmaction.models.heads import GCNHead
    from mmaction.registry import MODELS

    if "AttentiveGCNHead" in MODELS.module_dict:
        return

    @MODELS.register_module()
    class AttentiveGCNHead(GCNHead):
        def __init__(self, *args, pooling="joint_time", **kwargs):
            super().__init__(*args, **kwargs)
            assert pooling in ("time", "joint_time"), pooling
            self.pooling = pooling
            self.attn = torch.nn.Conv2d(self.in_channels, 1, kernel_size=1)
            torch.nn.init.zeros_(self.attn.weight)
            torch.nn.init.zeros_(self.attn.bias)
            self.last_attn = None            # (N*M, T, V), sums to 1 over T and V, kept for inspection

        def forward(self, x, **kwargs):
            n, m, c, t, v = x.shape
            x = x.reshape(n * m, c, t, v)
            if self.pooling == "time":
                weights = torch.softmax(self.attn(x.mean(dim=3, keepdim=True)).flatten(1), dim=1)       # (N*M, T)
                pooled = (x.mean(dim=3) * weights[:, None, :]).sum(dim=2)
                self.last_attn = (weights[:, :, None] / v).expand(-1, -1, v).detach()
            else:
                weights = torch.softmax(self.attn(x).flatten(1), dim=1).view(n * m, 1, t, v)
                pooled = (x * weights).sum(dim=(2, 3))
                self.last_attn = weights[:, 0].detach()
            pooled = pooled.reshape(n, m, c).mean(dim=1)
            if self.dropout is not None:
                pooled = self.dropout(pooled)
            return self.fc(pooled)


STAGE_LAYOUTS = {   # blocks -> where the channels double (and the temporal stride is 2)
    9: dict(num_stages=9, inflate_stages=[4, 7], down_stages=[4, 7]),      # Yan et al. 2018, the paper configuration
    10: dict(num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8]),    # mmaction2 default, used by its pretrained weights
}


def build_stgcn(in_channels, num_classes, head="avg", stages=9, base_channels=64, **backbone_options):
    """
    stages: 9 is the paper configuration, 10 is mmaction2's default layout (one extra 64 channel block) which the
    NTU60 pretrained weights were trained with.
    head: "avg" is mmaction2's GCNHead (global average pooling, the paper configuration), "attn_time" and
    "attn_joint_time" are AttentiveGCNHead variants (see _register_attentive_head).
    base_channels: width of the first block (64 in the paper); the classification head width follows (4 times this).
    backbone_options go straight to mmaction2's STGCN backbone (e.g. gcn_adaptive="init", gcn_with_res=True);
    with none given this is the paper configuration.
    """
    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules

    register_all_modules()
    if head not in POOLING_HEADS:
        raise ValueError(f"unknown head {head!r}, expected one of {list(POOLING_HEADS)}")
    head_cfg = dict(type="GCNHead", num_classes=num_classes, in_channels=base_channels * 4, loss_cls=dict(type="CrossEntropyLoss"))
    if POOLING_HEADS[head]:
        _register_attentive_head()
        head_cfg.update(type="AttentiveGCNHead", pooling=POOLING_HEADS[head])
    cfg = dict(
        type="RecognizerGCN",
        backbone=dict(type="STGCN", graph_cfg=dict(layout="coco", mode="stgcn_spatial"),
                      in_channels=in_channels, base_channels=base_channels, ch_ratio=2,
                      **STAGE_LAYOUTS[stages], **backbone_options),
        cls_head=head_cfg,
    )
    return MODELS.build(cfg)


def smoothed_velocity(xy):
    """
    Per frame velocity of every joint from a 5 frame Savitzky-Golay derivative filter, (N, T, V, 2) -> (N, T, V, 2),
    in the input's units per frame. A plain frame to frame difference would turn the independent per frame noise
    of the noise augmentation into velocity (its size is comparable to real motion), so training and test
    clips would carry different velocity statistics. The derivative filter cuts that noise by about 4.5 times
    and is applied identically at training and test time. Edges use replicated frames.
    """
    n, t, v, c = xy.shape
    x = xy.permute(0, 2, 3, 1).reshape(n * v * c, 1, t)
    x = torch.nn.functional.pad(x, (2, 2), mode="replicate")
    kernel = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=x.dtype, device=x.device).view(1, 1, 5) / 10.0
    out = torch.nn.functional.conv1d(x, kernel)
    return out.reshape(n, v, c, t).permute(0, 3, 1, 2)


def n_channels(use_court, use_velocity=False):
    """Input channels per joint: 5 (x, y, confidence, bone dx, bone dy), +2 court position, +2 velocity."""
    return 5 + (2 if use_court else 0) + (2 if use_velocity else 0)


def make_forward(use_court, use_velocity=False, velocity_scale=1.0):
    """
    Returns forward(model, batch, device) -> logits for training.evaluate / train_model.
    Input channels per joint: x, y, confidence, bone dx, bone dy, plus the two court position channels when
    use_court is set (the same value for every joint of a frame), plus the two velocity channels (smoothed per frame
    velocity of each joint, see smoothed_velocity, in the batch's own units) when use_velocity is set.
    velocity_scale is a fixed constant that brings velocity to a magnitude comparable with the other channels;
    it is computed once from the training set and never learned. Shaped (N, clips, persons, T, V, C).
    """
    def forward(model, batch, device):
        parts = [batch["keypoints"], batch["bone"]]
        if use_court:
            parts.append(batch["court"][:, :, None, :].expand(-1, -1, ds.NUM_JOINTS, -1))
        if use_velocity:
            parts.append(smoothed_velocity(batch["keypoints"][..., :2].float()) * velocity_scale)
        x = torch.cat(parts, dim=-1).float().to(device, non_blocking=True)
        return model(x[:, None, None], mode="tensor", stage="head")
    return forward
