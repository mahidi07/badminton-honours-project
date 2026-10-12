"""
Loading mmaction2's NTU60 skeleton ST-GCN weights (stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d) into our model.

The checkpoint is a 10 block ST-GCN on the 17 joint COCO layout, trained on joint coordinates (x, y, confidence) with
the same graph definition as ours. Everything that depends only on the architecture is copied, matched by name and
shape. What cannot transfer is rebuilt:

    backbone.data_bn.*       batch norm over (joints x input channels): 51 values there, 153 here, and its statistics
                             describe NTU coordinates anyway, so it starts fresh and learns ours
    cls_head.*               60 NTU classes against our 18
    first graph convolution  its input weights are (192, 3, 1, 1) there and (192, 9, 1, 1) here. first_layer="xy"
                             copies the weights for x, y and confidence into our first three channels and sets the
                             weights for our extra channels (bone, court position, velocity) to zero, so the network
                             starts as the pretrained one looking at its own three inputs; "free" leaves the whole
                             layer at its fresh initialisation.

The adjacency matrices are loaded too, after checking that they equal ours, which confirms the two models use the
same graph definition.
"""

import torch

NOT_TRANSFERRED_PREFIXES = ("backbone.data_bn.", "cls_head.")
FIRST_CONV = "backbone.gcn.0.gcn.conv.weight"


def load_ntu60(model, path, first_layer="xy", copied_channels=3):
    """Loads the checkpoint into model (in place) and returns a report of what was and was not transferred."""
    if first_layer not in ("xy", "free"):
        raise ValueError(first_layer)
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    target = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    to_load, skipped = {}, []
    for key, value in source.items():
        if key.startswith(NOT_TRANSFERRED_PREFIXES):
            skipped.append((key, "restarts"))
        elif key not in target:
            skipped.append((key, "no such tensor in model"))
        elif key == FIRST_CONV:
            if first_layer == "xy":
                weight = torch.zeros_like(target[key])
                weight[:, :copied_channels] = value[:, :copied_channels]
                to_load[key] = weight
            else:
                skipped.append((key, "first layer re-initialised"))
        elif target[key].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} against {tuple(target[key].shape)}"))
        else:
            to_load[key] = value

    graph_keys = [k for k in source if k.endswith(".gcn.A") and k in target and target[k].shape == source[k].shape]
    graph_difference = max((target[k] - source[k]).abs().max().item() for k in graph_keys) if graph_keys else float("nan")

    result = model.load_state_dict(to_load, strict=False)
    return {
        "loaded_tensors": len(to_load),
        "loaded_values": int(sum(v.numel() for v in to_load.values())),
        "skipped": skipped,
        "unexpected_skips": [k for k, why in skipped if why not in ("restarts", "first layer re-initialised")],
        "not_loaded_model_keys": list(result.missing_keys),
        "graph_tensors_compared": len(graph_keys),
        "graph_max_difference": graph_difference,
    }
