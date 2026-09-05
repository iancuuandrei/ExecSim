"""Frozen-representation probe input contracts."""

from __future__ import annotations

import numpy as np


def probe_feature_views(latents: np.ndarray) -> dict[str, np.ndarray]:
    """Return support bits, nonzero magnitudes, and full latent views."""
    values = np.asarray(latents, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Probe latents must be a finite matrix.")
    return {
        "support": (values != 0).astype(np.float32),
        "nonzero_magnitude": np.where(values != 0, np.abs(values), 0.0),
        "full_latent": values,
    }


def evaluate_frozen_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    """Fit a train-only logistic probe and report balanced accuracy and macro F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(train_features, train_labels)
    prediction = model.predict(test_features)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, prediction)),
        "macro_f1": float(f1_score(test_labels, prediction, average="macro")),
    }
