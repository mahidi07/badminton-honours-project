"""
Prints what a pretrained skeleton checkpoint contains, so the transfer into our ST-GCN can be planned from facts:
which keys and shapes it has, how many input channels and joints the first layers were built for, and which of
our layers would match.

    python inspect_checkpoint.py /workspace/checkpoints/stgcn_ntu60_pretrained/stgcn_ntu60.pth
"""

import sys

import torch


def main(path):
    ckpt = torch.load(path, map_location="cpu")
    print("top level keys:", list(ckpt.keys()))
    meta = ckpt.get("meta", {})
    if meta:
        print("meta keys:", list(meta.keys()))
        for k in ("mmengine_version", "mmaction_version", "epoch", "iter", "dataset_meta"):
            if k in meta:
                print(f"  meta[{k}] = {str(meta[k])[:200]}")
    state = ckpt.get("state_dict", ckpt)
    print(f"\n{len(state)} tensors, {sum(v.numel() for v in state.values() if hasattr(v, 'numel')):,} values")
    for k, v in state.items():
        if hasattr(v, "shape") and (k.startswith("backbone.data_bn") or ".0." in k[:20] or "cls_head" in k
                                    or "gcn.0." in k or "gcn.9" in k or k.endswith(".A")):
            print(f"  {k:60s} {tuple(v.shape)}")
    bn = state.get("backbone.data_bn.weight")
    if bn is not None:
        print(f"\ndata_bn has {bn.numel()} channels: in_channels x joints = {bn.numel()} "
              f"(17 joints -> {bn.numel() / 17:.2f} channels, 25 joints -> {bn.numel() / 25:.2f} channels)")
    a = [v for k, v in state.items() if k.endswith(".A") or k.endswith("gcn.A")]
    print("adjacency tensors stored as parameters:", len(a), "| shape of first:", tuple(a[0].shape) if a else None)
    head = state.get("cls_head.fc.weight")
    if head is not None:
        print("classifier:", tuple(head.shape), "(classes, features)")


if __name__ == "__main__":
    main(sys.argv[1])
