"""Dimension-matched target-free random representation control."""

from __future__ import annotations

import hashlib

import numpy as np


def random_projection_matrix(input_dim: int, *, output_dim: int = 640, seed: int) -> np.ndarray:
    """Create a deterministic column-normalized Gaussian projection."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Projection dimensions must be positive.")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, output_dim))
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True).clip(min=1e-12)
    return matrix.astype(np.float32)


def projection_hash(matrix: np.ndarray) -> str:
    """Hash projection shape, dtype, and canonical contiguous bytes."""
    values = np.ascontiguousarray(matrix)
    digest = hashlib.sha256(f"{values.shape}|{values.dtype}".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()
