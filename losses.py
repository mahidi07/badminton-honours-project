"""
Loss functions for the class imbalance experiments.

LogitAdjustedLoss is the loss side logit adjustment of Menon et al. (2021, "Long-tail learning via logit
adjustment", ICLR): cross entropy on logits shifted by tau * log(class prior). During training the shift makes
frequent classes cost more to predict, so the network is pushed to keep a margin for rare classes; at test time
the plain logits are used, which then approximate a classifier that is balanced across classes. tau = 1 is the
value the paper derives for balanced error, smaller values give a milder correction, tau = 0 is plain cross
entropy.

It is meant to be trained on the natural class frequencies (plain shuffling); the prior is the training set's
class frequency. Combining it with a class balanced sampler would correct for the imbalance twice.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_prior(labels, num_classes):
    """Training set class frequencies from an array of integer labels."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"classes without training clips: {np.flatnonzero(counts == 0).tolist()}")
    return counts / counts.sum()


class LogitAdjustedLoss(nn.Module):
    def __init__(self, prior, tau=1.0):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("log_prior", torch.log(torch.as_tensor(prior, dtype=torch.float32)))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.tau * self.log_prior.to(logits.device), target)


def posthoc_adjust(probs, prior, tau):
    """Evaluation time version: subtract tau * log(prior) from the log probabilities of a model trained with
    plain cross entropy on natural frequencies. Softmax outputs differ from logits by a per clip constant, so
    this is the same shift. Returns adjusted scores (N, K); take argmax for the prediction."""
    return np.log(np.clip(probs, 1e-12, None)) - tau * np.log(np.asarray(prior)[None, :])
