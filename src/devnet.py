"""DevNet (Deviation Networks), PyTorch port of the ADBench implementation.

Original: Pang, Shen and van den Hengel, "Deep Anomaly Detection with Deviation
Networks", KDD 2019. ADBench uses the authors' Keras/TensorFlow 1 code
(adbench/baseline/DevNet/run.py). That code does not run on current Python, so
this file reproduces it in PyTorch with the same architecture and settings.

This file currently holds the network and the deviation loss. Training and
scoring are added separately.
"""

import numpy as np
import torch
from torch import nn

# Settings taken from the ADBench DevNet code
HIDDEN_UNITS = 20          # network_depth=2: one hidden layer of 20 ReLU units
L2_WEIGHT = 0.01           # kernel_regularizer=l2(0.01) on the hidden layer
CONFIDENCE_MARGIN = 5.0    # anomalies should score at least 5 std above normal
N_REFERENCE = 5000         # number of reference scores drawn from N(0, 1)


class DevNetModel(nn.Module):
    """Input features -> 20 ReLU units -> 1 anomaly score (no activation)."""

    def __init__(self, n_features):
        super().__init__()
        self.hidden = nn.Linear(n_features, HIDDEN_UNITS)
        self.score = nn.Linear(HIDDEN_UNITS, 1)
        # Keras Dense defaults: Glorot (Xavier) uniform weights, zero biases
        for layer in (self.hidden, self.score):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.score(torch.relu(self.hidden(x))).squeeze(-1)

    def l2_penalty(self):
        """Keras l2(0.01) adds 0.01 * sum of squared hidden-layer weights (not biases)."""
        return L2_WEIGHT * torch.sum(self.hidden.weight ** 2)


def make_reference(seed):
    """Reference scores drawn once from a standard normal, as in the original code."""
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(0.0, 1.0, N_REFERENCE), dtype=torch.float32)


def deviation_loss(scores, labels, reference):
    """Z-score based deviation loss.

    dev = how many reference standard deviations a score is above the reference mean.
    Unlabeled samples (label 0) are pulled toward dev = 0.
    Labeled anomalies (label 1) are pushed to dev >= CONFIDENCE_MARGIN.
    """
    # Keras K.std is the population std (divide by n), hence correction=0
    dev = (scores - reference.mean()) / reference.std(correction=0)
    inlier_loss = torch.abs(dev)
    outlier_loss = torch.clamp(CONFIDENCE_MARGIN - dev, min=0.0)
    return torch.mean((1 - labels) * inlier_loss + labels * outlier_loss)


if __name__ == "__main__":
    # Sanity checks on the loss with hand-picked scores
    ref = make_reference(seed=1)
    print(f"reference mean {ref.mean():.4f}, std {ref.std(correction=0):.4f}")

    cases = [
        ("normal at score 0 (ideal)", [0.0], [0.0]),
        ("normal at score 3 (too high)", [3.0], [0.0]),
        ("anomaly at score 6 (past margin)", [6.0], [1.0]),
        ("anomaly at score 2 (short of margin)", [2.0], [1.0]),
    ]
    for name, s, y in cases:
        loss = deviation_loss(torch.tensor(s), torch.tensor(y), ref)
        print(f"{name:38s} loss = {loss.item():.4f}")

    model = DevNetModel(n_features=6)
    x = torch.zeros(4, 6)
    print("output shape for 4 samples:", tuple(model(x).shape))
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
