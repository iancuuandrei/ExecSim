"""Matched shared-encoder dense and sparse JEPA objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from execsim.ml.representations.encoders import TokenEncoder
from execsim.ml.representations.links import create_link
from execsim.ml.representations.predictors import create_predictor
from execsim.ml.representations.rdmreg import sliced_wasserstein_distance
from execsim.ml.representations.schemas import RepresentationConfig
from execsim.ml.sequences.schemas import CONDITIONING_FEATURE_INDICES, ENCODER_FEATURE_INDICES


class PredictiveRepresentationModel(nn.Module):
    """Encode causal context and predict directly linked future token latents."""

    def __init__(self, config: RepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TokenEncoder(config.feature_dim, config.latent_dim)
        self.link = create_link(config.geometry)
        self.predictor = create_predictor(
            config.predictor_family, conditioning_dim=config.conditioning_dim
        )

    def encode(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return pre-link and linked latents without detaching target gradients."""
        pre_link = self.encoder(tokens)
        return pre_link, self.link(pre_link)

    def forward(
        self, context: Tensor, context_mask: Tensor, targets: Tensor, target_mask: Tensor
    ) -> dict[str, Tensor]:
        if (
            context.shape[-1] != self.config.observed_feature_dim
            or targets.shape[-1] != self.config.observed_feature_dim
        ):
            raise ValueError("JEPA inputs must use the complete 18-feature observation contract.")
        dynamic_context = context[..., ENCODER_FEATURE_INDICES]
        dynamic_targets = targets[..., ENCODER_FEATURE_INDICES]
        context_pre, context_latent = self.encode(dynamic_context)
        context_latent = context_latent * context_mask.unsqueeze(-1).to(context_latent.dtype)
        target_pre, target_latent = self.encode(dynamic_targets)
        token_indices = torch.arange(context.shape[1], device=context.device).expand_as(
            context_mask
        )
        last_indices = token_indices.masked_fill(~context_mask.bool(), -1).max(dim=1).values
        if bool((last_indices < 0).any()):
            raise ValueError("Every JEPA example requires at least one valid context token.")
        conditioning = context[torch.arange(len(context), device=context.device), last_indices][
            ..., CONDITIONING_FEATURE_INDICES
        ]
        predictions = []
        for horizon_index in range(len(self.config.horizons)):
            index = torch.full(
                (len(context),), horizon_index, dtype=torch.long, device=context.device
            )
            predictions.append(
                self.link(self.predictor(context_latent, context_mask, conditioning, index))
            )
        predicted = torch.stack(predictions, dim=1)
        squared = torch.mean((predicted - target_latent) ** 2, dim=-1)
        valid = target_mask.to(squared.dtype)
        per_example = (squared * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        prediction_loss = per_example.mean()
        return {
            "prediction_loss": prediction_loss,
            "prediction": predicted,
            "target": target_latent,
            "context_pre_link": context_pre,
            "target_pre_link": target_pre,
        }

    def loss(
        self,
        context: Tensor,
        context_mask: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        *,
        rdm_lambda: float,
        rdm_projections: int | None = None,
        rdm_seed: int | None = None,
        sample_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Combine equally weighted available-horizon prediction loss and RDMReg."""
        output = self(context, context_mask, targets, target_mask)
        if sample_weights is not None:
            error = torch.mean((output["prediction"] - output["target"]) ** 2, dim=-1)
            valid = target_mask.to(error.dtype)
            per_example = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            output["prediction_loss"] = (
                per_example * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
        pooled = torch.cat(
            (
                self.link(output["context_pre_link"])[context_mask.bool()],
                self.link(output["target_pre_link"])[target_mask.bool()],
            ),
            dim=0,
        )
        p, mu, sigma = self.config.target_parameters
        rdm = sliced_wasserstein_distance(
            pooled.float(),
            p=p,
            mu=mu,
            sigma=sigma,
            projections=rdm_projections or self.config.rdm_projections_train,
            seed=rdm_seed if rdm_seed is not None else self.config.seed + 2_000_003,
            rectify_target=self.config.geometry == "sparse",
            target_rms=self.config.target_rms,
        )
        output["rdm_loss"] = rdm
        output["loss"] = output["prediction_loss"] + rdm_lambda * rdm
        return output
