"""
LSTM baseline: a bidirectional LSTM over per-frame pose features, the simplest sequence model in the
project (no graph structure, no attention). Serves as the lower-capacity baseline against ST-GCN and the
attention-based models.

Per frame input is every joint's (x, y, confidence, bone dx, bone dy) flattened into one vector, optionally
with the court position channels appended per joint (matching ST-GCN's body_court input). The whole clip is
one (T, F) sequence; the model reads it in both time directions and classifies from the final hidden state.
"""

import torch
import torch.nn as nn

import dataset as ds


class LSTMClassifier(nn.Module):
    def __init__(self, in_features, hidden_size=256, num_layers=2, num_classes=18, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (N, T, F). Take the final layer's forward and backward hidden states, not a pooled average,
        # so the model has to carry the relevant information all the way to the end of the sequence.
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(self.norm(h))
        return self.fc(h)


def build_lstm(in_features, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
    return LSTMClassifier(in_features, hidden_size, num_layers, num_classes, dropout)


def in_features(use_court):
    return ds.NUM_JOINTS * (7 if use_court else 5)


def make_forward(use_court):
    """Returns forward(model, batch, device) -> logits, matching models.stgcn's interface."""
    def forward(model, batch, device):
        parts = [batch["keypoints"], batch["bone"]]
        if use_court:
            parts.append(batch["court"][:, :, None, :].expand(-1, -1, ds.NUM_JOINTS, -1))
        x = torch.cat(parts, dim=-1).float()          # (N, T, V, C)
        n, t, v, c = x.shape
        x = x.reshape(n, t, v * c).to(device, non_blocking=True)
        return model(x)
    return forward
