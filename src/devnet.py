"""DevNet (Deviation Networks), PyTorch port of the ADBench implementation.

Original: Pang, Shen and van den Hengel, "Deep Anomaly Detection with Deviation
Networks", KDD 2019. ADBench uses the authors' Keras/TensorFlow 1 code
(adbench/baseline/DevNet/run.py). That code does not run on current Python, so
this file reproduces it in PyTorch with the same architecture and settings.

This file currently holds the network, the deviation loss, the batch sampler
and the training loop. Scoring is added separately.
"""

import copy

import numpy as np
import torch
from torch import nn

# Settings taken from the ADBench DevNet code
HIDDEN_UNITS = 20          # network_depth=2: one hidden layer of 20 ReLU units
L2_WEIGHT = 0.01           # kernel_regularizer=l2(0.01) on the hidden layer
CONFIDENCE_MARGIN = 5.0    # anomalies should score at least 5 std above normal
N_REFERENCE = 5000         # number of reference scores drawn from N(0, 1)
BATCH_SIZE = 512           # samples per training batch
N_BATCHES = 20             # batches per epoch (nb_batch)
EPOCHS = 50                # training epochs
LEARNING_RATE = 0.001      # Keras RMSprop defaults: lr 0.001, rho 0.9, epsilon 1e-7
RMSPROP_RHO = 0.9
RMSPROP_EPS = 1e-7
CLIP_NORM = 1.0            # RMSprop(clipnorm=1.)


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


def sample_batch(X, y, rng, batch_size=BATCH_SIZE):
    """Build one balanced training batch, as in the original input_batch_generation_sup.

    Even positions take a random unlabeled sample (label 0), odd positions take a
    random labeled anomaly (label 1). Sampling is with replacement, so even a
    single labeled anomaly fills half of every batch.

    X: training features, y: training labels after hide_labels (1 = labeled anomaly)
    rng: numpy random Generator
    """
    inlier_idx = np.where(y == 0)[0]
    outlier_idx = np.where(y == 1)[0]

    idx = np.empty(batch_size, dtype=int)
    idx[0::2] = rng.choice(inlier_idx, size=len(idx[0::2]))
    idx[1::2] = rng.choice(outlier_idx, size=len(idx[1::2]))
    labels = np.tile([0.0, 1.0], batch_size // 2 + 1)[:batch_size]

    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), idx


def train_devnet(X_train, y_train, seed, epochs=EPOCHS, n_batches=N_BATCHES):
    """Train DevNet and return the model with the weights from its lowest-loss epoch.

    Mirrors the original fit(): balanced batches, deviation loss plus L2 penalty,
    RMSprop with gradient norm clipping, and a checkpoint that keeps the best
    weights by training loss (ModelCheckpoint with save_best_only=True).

    Returns the model and the list of per-epoch average losses.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reference = make_reference(seed)

    model = DevNetModel(X_train.shape[1])
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE,
                                    alpha=RMSPROP_RHO, eps=RMSPROP_EPS)

    best_loss, best_state, history = float("inf"), None, []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            xb, yb, _ = sample_batch(X_train, y_train, rng)
            loss = deviation_loss(model(xb), yb, reference) + model.l2_penalty()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss, best_state = epoch_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, history


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

    # Sanity checks on the batch sampler with real data: Annthyroid, 1% labeled
    from data import hide_labels, load_dataset, split_data

    X, y = load_dataset("annthyroid")
    X_train, _, y_train, _ = split_data(X, y, seed=1)
    y_semi, n_labeled = hide_labels(y_train, 0.01, seed=1)

    xb, yb, idx = sample_batch(X_train, y_semi, np.random.default_rng(1))
    print(f"\nbatch features {tuple(xb.shape)}, labels {tuple(yb.shape)}")
    print(f"labeled anomalies in batch: {int(yb.sum())} of {len(yb)} "
          f"(drawn from {n_labeled} distinct labeled patients: {len(set(idx[1::2]))} used)")
    print(f"unlabeled samples in batch: {int((yb == 0).sum())}, "
          f"all from the unlabeled pool: {bool((y_semi[idx[0::2]] == 0).all())}")
    print(f"hidden anomalies that landed in the unlabeled half: {int(y_train[idx[0::2]].sum())}")
    _, _, idx_again = sample_batch(X_train, y_semi, np.random.default_rng(1))
    print(f"same seed gives the same batch: {bool((idx == idx_again).all())}")

    # Sanity check on training: loss should fall, and the run should be repeatable
    import time
    from data import scale_features

    X_train_s, _ = scale_features(X_train, X_train)
    start = time.time()
    model, history = train_devnet(X_train_s, y_semi, seed=1)
    print(f"\ntrained {len(history)} epochs in {time.time() - start:.1f}s")
    for e in (0, 9, 19, 29, 39, 49):
        print(f"  epoch {e + 1:2d}  loss {history[e]:.4f}")
    best = int(np.argmin(history))
    print(f"  best epoch {best + 1} (loss {history[best]:.4f}) weights kept")
    _, history_again = train_devnet(X_train_s, y_semi, seed=1)
    print(f"same seed gives the same training: {history == history_again}")
