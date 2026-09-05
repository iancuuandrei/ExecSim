"""Horizon-conditioned predictor-capacity ladder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearPredictor(nn.Module):
    """P0 horizon-specific affine dynamics over context, mask, and conditioning."""

    def __init__(
        self,
        context_length: int = 8,
        latent_dim: int = 128,
        horizons: int = 4,
        conditioning_dim: int = 5,
    ) -> None:
        super().__init__()
        input_dim = context_length * latent_dim + context_length + conditioning_dim
        self.horizon_linears = nn.ModuleList(
            [nn.Linear(input_dim, latent_dim) for _ in range(horizons)]
        )

    def forward(
        self,
        context: Tensor,
        mask: Tensor,
        conditioning: Tensor,
        horizon_index: Tensor | None = None,
    ) -> Tensor:
        if horizon_index is None:
            horizon_index = conditioning.long()
            conditioning = context.new_zeros((len(context), 5))
        inputs = torch.cat((context.flatten(1), mask.to(context.dtype), conditioning), dim=1)
        candidates = torch.stack([layer(inputs) for layer in self.horizon_linears], dim=1)
        rows = torch.arange(len(context), device=context.device)
        return candidates[rows, horizon_index]


class MLPPredictor(nn.Module):
    """P1 predictor with a learned 16-dimensional horizon embedding."""

    def __init__(
        self, context_length: int = 8, latent_dim: int = 128, conditioning_dim: int = 5
    ) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(4, 16)
        self.network = nn.Sequential(
            nn.Linear(context_length * latent_dim + context_length + conditioning_dim + 16, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self,
        context: Tensor,
        mask: Tensor,
        conditioning: Tensor,
        horizon_index: Tensor | None = None,
    ) -> Tensor:
        if horizon_index is None:
            horizon_index = conditioning.long()
            conditioning = context.new_zeros((len(context), 5))
        inputs = torch.cat(
            (
                context.flatten(1),
                mask.to(context.dtype),
                conditioning,
                self.horizon_embedding(horizon_index),
            ),
            dim=1,
        )
        return self.network(inputs)


class TransformerPredictor(nn.Module):
    """P2 tiny four-layer Transformer with a horizon-conditioned query."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(4, latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        self.output = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        context: Tensor,
        mask: Tensor,
        conditioning: Tensor,
        horizon_index: Tensor | None = None,
    ) -> Tensor:
        if horizon_index is None:
            horizon_index = conditioning.long()
        del conditioning
        query = self.horizon_embedding(horizon_index).unsqueeze(1)
        tokens = torch.cat((context, query), dim=1)
        padding = torch.cat(
            (~mask.bool(), torch.zeros((len(mask), 1), dtype=torch.bool, device=mask.device)), dim=1
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding)
        return self.output(encoded[:, -1])


def create_predictor(family: str, *, conditioning_dim: int = 5) -> nn.Module:
    """Construct one capacity-ladder predictor."""
    if family == "linear":
        return LinearPredictor(conditioning_dim=conditioning_dim)
    if family == "mlp":
        return MLPPredictor(conditioning_dim=conditioning_dim)
    if family == "transformer":
        return TransformerPredictor()
    raise ValueError(f"Unknown predictor family: {family}")


def trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
