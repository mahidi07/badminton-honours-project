"""
Shared training, evaluation and ClearML reporting for every model notebook.

Every model is trained through train_model() and reported through report_run(), so all of them log the same
curves, figures, tables and artifacts and are selected by the same rule.

Protocol
    - checkpoint selection and early stopping use the validation set only
    - train, val and test metrics are logged every epoch so the full curves can be inspected, but the test
      curve is never used to choose anything
    - train metrics per epoch are computed on the un-augmented training set in its natural class balance,
      so they are directly comparable with val and test; the loss on the augmented, balanced training batches
      is logged separately as "train_batches"

Set os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" before importing torch for deterministic cuBLAS.
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, top_k_accuracy_score,
)

import dataset as ds

SPLITS = ("train", "val", "test")
METRICS = ("top1", "top5", "mca", "macro_f1")


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoader workers do not inherit numpy/random state from torch.manual_seed; without this, augmentation
    # differs between runs with the same seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_loaders(manifest, keypoints_root, class_to_id, augmentation, normalisation, seed,
                  batch_size=16, eval_batch_size=64, target_per_class=500, num_workers=4):
    """
    Returns loaders for one run. The class balanced sampler and the worker generator are created here, fresh,
    for every run: reusing a sampler across runs carries its epoch counter over and breaks seeding.
        train       augmented, class balanced, used for gradient steps
        train_eval  un-augmented, natural balance, used for train metrics
        val, test   un-augmented
    """
    parts = {s: manifest[manifest["split"] == s].reset_index(drop=True) for s in SPLITS}
    make = lambda split, aug: ds.BadmintonPoseDataset(parts[split], keypoints_root, class_to_id,
                                                      augmentation_policy=aug, normalisation=normalisation)
    generator = torch.Generator()
    generator.manual_seed(seed)
    eval_kwargs = dict(batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                       persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(make("train", augmentation), batch_size=batch_size,
                            sampler=ds.build_class_balanced_sampler(parts["train"], target_per_class, seed=seed),
                            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
                            worker_init_fn=seed_worker, generator=generator),
        "train_eval": DataLoader(make("train", "none"), **eval_kwargs),
        "val": DataLoader(make("val", "none"), **eval_kwargs),
        "test": DataLoader(make("test", "none"), **eval_kwargs),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, forward, device):
    """forward(model, batch, device) -> logits. Returns metrics plus per clip labels, predictions and probabilities."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, labels, probs, paths = 0.0, [], [], []
    for batch in loader:
        logits = forward(model, batch, device)
        y = batch["label"].to(device)
        total_loss += criterion(logits, y).item()
        labels.append(y.cpu().numpy())
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        paths.extend(batch["filepath"])
    labels, probs = np.concatenate(labels), np.concatenate(probs)
    preds = probs.argmax(axis=1)
    classes = list(range(probs.shape[1]))
    return {
        "loss": total_loss / len(labels),
        "top1": accuracy_score(labels, preds),
        "top5": top_k_accuracy_score(labels, probs, k=5, labels=classes),
        "mca": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=classes, zero_division=0),
        "labels": labels, "preds": preds, "probs": probs, "filepaths": paths,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_model(model, forward, loaders, optimizer, scheduler, criterion, device, logger,
                num_epochs=60, patience=20, select_metric="top1", grad_clip=None, scheduler_per_step=False,
                print_every=5):
    """
    Trains with early stopping on the validation metric `select_metric` and returns (best_state, history,
    best_epoch). The scheduler steps once per epoch unless scheduler_per_step is True.
    """
    history, best_state, best_value, best_epoch, stale = [], None, -np.inf, -1, 0

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        batch_loss, n_seen = 0.0, 0
        for batch in loaders["train"]:
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(model, batch, device), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non finite loss at epoch {epoch}")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler_per_step:
                scheduler.step()
            batch_loss += loss.item() * len(y)
            n_seen += len(y)
        lr = optimizer.param_groups[0]["lr"]
        if not scheduler_per_step:
            scheduler.step()
        train_time = time.time() - start

        results = {s: evaluate(model, loaders["train_eval" if s == "train" else s], forward, device) for s in SPLITS}
        row = {"epoch": epoch, "lr": lr, "train_batches_loss": batch_loss / n_seen, "epoch_seconds": train_time}
        for s in SPLITS:
            for m in ("loss",) + METRICS:
                row[f"{s}_{m}"] = results[s][m]
        history.append(row)

        logger.report_scalar("loss", "train_batches", row["train_batches_loss"], iteration=epoch)
        for m in ("loss",) + METRICS:
            for s in SPLITS:
                logger.report_scalar(m, s, row[f"{s}_{m}"], iteration=epoch)
        logger.report_scalar("learning rate", "lr", lr, iteration=epoch)
        logger.report_scalar("time", "train seconds per epoch", train_time, iteration=epoch)

        value = row[f"val_{select_metric}"]
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % print_every == 0 or stale == 0:
            print(f"epoch {epoch:3d}  loss {row['train_batches_loss']:.3f}  "
                  f"train {row['train_top1']:.3f}  val {row['val_top1']:.3f} (mca {row['val_mca']:.3f})"
                  f"{'  *' if stale == 0 else ''}")
        if stale >= patience:
            print(f"early stop at epoch {epoch}, best val {select_metric} {best_value:.4f} at epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    return best_state, pd.DataFrame(history), best_epoch


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _learning_curves(history, best_epoch, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colours = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}
    axes[0].plot(history["epoch"], history["train_batches_loss"], color="grey", ls=":", label="train batches (augmented)")
    for s in SPLITS:
        axes[0].plot(history["epoch"], history[f"{s}_loss"], color=colours[s], label=s)
        axes[1].plot(history["epoch"], history[f"{s}_top1"], color=colours[s], label=s)
        axes[2].plot(history["epoch"], history[f"{s}_mca"], color=colours[s], label=s)
    for ax, name in zip(axes, ["cross entropy", "top 1 accuracy", "mean class accuracy"]):
        ax.axvline(best_epoch, color="black", lw=0.8, ls="--")
        ax.set_xlabel("epoch")
        ax.set_title(name)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title} (dashed: selected epoch, chosen on val)", fontsize=10)
    fig.tight_layout()
    return fig


def _confusion_figure(cm, class_names, title):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = [c.split("_", 1)[-1] for c in class_names]
    ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=7)
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}: test confusion (colour: row normalised, numbers: clips)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def _per_class_figure(per_class, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = per_class.sort_values("f1")
    ax.barh(order["class"], order["f1"], color="#4c72b0")
    for y, (f1, n) in enumerate(zip(order["f1"], order["support"])):
        ax.text(f1 + 0.01, y, f"n={n}", va="center", fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("test F1")
    ax.set_title(f"{title}: per class test F1", fontsize=9)
    fig.tight_layout()
    return fig


def report_run(task, model, forward, loaders, class_names, device, history, best_epoch, out_dir, run_name,
               extra=None, show=True):
    """
    Final evaluation of the selected checkpoint and full ClearML reporting. Returns a flat metrics dict.
    Logs: single values for every headline metric on train/val/test, learning curves, confusion matrix
    (interactive and figure), per class table and figure, and artifacts for metrics, history, per clip
    predictions and the checkpoint.
    """
    logger = task.get_logger()
    os.makedirs(out_dir, exist_ok=True)
    final = {s: evaluate(model, loaders["train_eval" if s == "train" else s], forward, device) for s in SPLITS}

    metrics = {"run": run_name, "best_epoch": int(best_epoch), "epochs_run": int(len(history)),
               "seconds_per_epoch": float(history["epoch_seconds"].mean()),
               "parameters": int(sum(p.numel() for p in model.parameters()))}
    for s in SPLITS:
        for m in ("loss",) + METRICS:
            metrics[f"{s}_{m}"] = float(final[s][m])
    metrics["train_test_gap"] = metrics["train_top1"] - metrics["test_top1"]
    metrics.update(extra or {})
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.report_single_value(k, v)

    figures = [
        ("learning curves", _learning_curves(history, best_epoch, run_name)),
    ]
    cm = confusion_matrix(final["test"]["labels"], final["test"]["preds"], labels=list(range(len(class_names))))
    short = [c.split("_", 1)[-1] for c in class_names]
    logger.report_confusion_matrix("test confusion matrix", "counts", matrix=cm, iteration=0,
                                   xaxis="predicted", yaxis="true", xlabels=short, ylabels=short)
    figures.append(("test confusion matrix (normalised)", _confusion_figure(cm, class_names, run_name)))

    p, r, f1, n = precision_recall_fscore_support(final["test"]["labels"], final["test"]["preds"],
                                                  labels=list(range(len(class_names))), zero_division=0)
    per_class = pd.DataFrame({"class": short, "precision": p, "recall": r, "f1": f1, "support": n}).round(4)
    logger.report_table("per class test metrics", "test", iteration=0, table_plot=per_class)
    figures.append(("per class test F1", _per_class_figure(per_class, run_name)))

    summary = pd.DataFrame([{"split": s, **{m: round(metrics[f"{s}_{m}"], 4) for m in ("loss",) + METRICS}} for s in SPLITS])
    logger.report_table("final metrics", "selected checkpoint", iteration=0, table_plot=summary)

    for title, fig in figures:
        logger.report_matplotlib_figure(title, run_name, figure=fig, iteration=0, report_interactive=False)
        if show:
            plt.show()
        plt.close(fig)

    preds = []
    for s in ("val", "test"):
        top5 = np.argsort(-final[s]["probs"], axis=1)[:, :5]
        preds.append(pd.DataFrame({
            "split": s, "filepath": final[s]["filepaths"],
            "label": [class_names[i] for i in final[s]["labels"]],
            "pred": [class_names[i] for i in final[s]["preds"]],
            "confidence": final[s]["probs"].max(axis=1).round(5),
            "top5": [";".join(class_names[i] for i in row) for row in top5],
        }))
    preds = pd.concat(preds, ignore_index=True)

    paths = {
        "metrics": os.path.join(out_dir, f"{run_name}_metrics.json"),
        "history": os.path.join(out_dir, f"{run_name}_history.csv"),
        "predictions": os.path.join(out_dir, f"{run_name}_predictions.csv"),
        "per_class": os.path.join(out_dir, f"{run_name}_per_class.csv"),
        "checkpoint": os.path.join(out_dir, f"{run_name}.pt"),
    }
    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    history.to_csv(paths["history"], index=False)
    preds.to_csv(paths["predictions"], index=False)
    per_class.to_csv(paths["per_class"], index=False)
    torch.save(model.state_dict(), paths["checkpoint"])
    for name, path in paths.items():
        task.upload_artifact(name, artifact_object=path)

    print(" | ".join(f"{s} top1 {metrics[f'{s}_top1']:.4f} mca {metrics[f'{s}_mca']:.4f}" for s in SPLITS))
    return metrics


def mean_std_table(results, group_cols, metric_cols):
    """Mean and std across seeds, formatted as percentages."""
    agg = results.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=agg.index)
    for m in metric_cols:
        out[m] = [f"{100 * mu:.2f} ± {100 * sd:.2f}" for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")].fillna(0))]
    out["seeds"] = agg[(metric_cols[0], "count")]
    return out.reset_index()
