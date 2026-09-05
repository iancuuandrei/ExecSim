"""Matched per-token encoder used by both JEPA branches."""

from __future__ import annotations

from torch import Tensor, nn


class TokenEncoder(nn.Module):
    """Map each 13-feature dynamic token to a 128-dimensional pre-link latent."""

    def __init__(self, feature_dim: int = 13, latent_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.network(tokens)
