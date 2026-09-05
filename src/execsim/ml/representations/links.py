"""Geometry-specific latent links."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class _ExactRepReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor) -> Tensor:
        ctx.save_for_backward(values)
        return functional.relu(values)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        (values,) = ctx.saved_tensors
        normal_cdf = 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
        normal_pdf = torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
        return gradient * (normal_cdf + values * normal_pdf)


def rep_relu(values: Tensor) -> Tensor:
    """Return exact ReLU values with GELU derivatives."""
    return _ExactRepReLU.apply(values)


class IdentityLink(nn.Module):
    """Keep dense pre-link values unchanged."""

    def forward(self, values: Tensor) -> Tensor:
        return values


class RepReLULink(nn.Module):
    """Apply the sparse exact-forward surrogate-gradient link."""

    def forward(self, values: Tensor) -> Tensor:
        return rep_relu(values)


def create_link(geometry: str) -> nn.Module:
    """Construct the locked geometry link."""
    if geometry == "dense":
        return IdentityLink()
    if geometry == "sparse":
        return RepReLULink()
    raise ValueError(f"Unknown representation geometry: {geometry}")
