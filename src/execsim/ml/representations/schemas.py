"""Configuration and artifact schemas for predictive representations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Geometry = Literal["dense", "sparse"]
PredictorFamily = Literal["linear", "mlp", "transformer"]


@dataclass(frozen=True, slots=True)
class FutureScalingAxes:
    """Dormant axes for a later scaling study; never expands the current paper matrix."""

    encoder_capacity: tuple[int, ...] = (64, 128, 256)
    predictor_capacity: tuple[str, ...] = (
        "affine",
        "small_mlp",
        "medium_mlp",
        "larger_temporal_model",
    )
    latent_dimension: tuple[int, ...] = (64, 128, 256)
    target_sparsity: tuple[float, ...] = (0.5, 0.75, 0.875)
    training_data_scale: tuple[int, ...] = (25, 50, 100)
    market_complexity_stratum: tuple[str, ...] = ("ordinary", "unusual")


def sparse_location_for_zero_fraction(zero_fraction: float) -> float:
    """Return the unit-variance Laplace location for the Fold 1 sparsity sweep."""
    if zero_fraction not in {0.5, 0.75, 0.875}:
        raise ValueError("Sparsity sweep supports only 0.50, 0.75, and 0.875 zero fractions.")
    active_fraction = 1.0 - zero_fraction
    return (2**-0.5) * math.log(2.0 * active_fraction)


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Declare one matched JEPA training run."""

    geometry: Geometry
    predictor_family: PredictorFamily = "mlp"
    feature_dim: int = 13
    observed_feature_dim: int = 18
    conditioning_dim: int = 5
    latent_dim: int = 128
    context_length: int = 8
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    generalized_gaussian_p: float | None = None
    generalized_gaussian_mu: float | None = None
    generalized_gaussian_sigma: float | None = None
    rdm_projections_train: int = 512
    rdm_projections_evaluation: int = 2048
    seed: int = 13

    def __post_init__(self) -> None:
        if (
            self.feature_dim != 13
            or self.observed_feature_dim != 18
            or self.conditioning_dim != 5
            or self.latent_dim != 128
            or self.context_length != 8
        ):
            raise ValueError(
                "The primary paper representation is locked to 18/13/5/128/8 dimensions."
            )
        if self.horizons != (1, 2, 4, 8):
            raise ValueError("The primary paper horizons are locked to 1, 2, 4, and 8 tokens.")
        if self.rdm_projections_train <= 0 or self.rdm_projections_evaluation <= 0:
            raise ValueError("RDMReg projection counts must be positive.")

    @property
    def target_parameters(self) -> tuple[float, float, float]:
        """Return locked dense or sparse generalized-Gaussian parameters."""
        if self.geometry == "dense":
            return (
                self.generalized_gaussian_p or 2.0,
                self.generalized_gaussian_mu or 0.0,
                self.generalized_gaussian_sigma or 1.0,
            )
        return (
            self.generalized_gaussian_p or 2.0,
            self.generalized_gaussian_mu
            if self.generalized_gaussian_mu is not None
            else -0.6744897501960817,
            self.generalized_gaussian_sigma or 1.0,
        )

    @property
    def target_positive_moments(self) -> tuple[float, float]:
        """Return derived linked-target first and second moments."""
        from execsim.ml.representations.rdmreg import (
            generalized_gaussian_moments,
            rectified_generalized_gaussian_moments,
        )

        if self.geometry == "dense":
            p, mu, sigma = self.target_parameters
            mean, variance = generalized_gaussian_moments(p=p, mu=mu, sigma=sigma)
            return mean, variance + mean**2
        p, mu, sigma = self.target_parameters
        return rectified_generalized_gaussian_moments(p=p, mu=mu, sigma=sigma)

    @property
    def target_rms(self) -> float:
        """Return the derived root mean square after the configured link."""
        return math.sqrt(self.target_positive_moments[1])

    @property
    def target_zero_fraction(self) -> float:
        """Return the configured sparse zero probability for locked targets."""
        if self.geometry == "dense":
            return 0.0
        p, mu, sigma = self.target_parameters
        if p == 1.0 and mu <= 0:
            scale = p ** (1.0 / p) * sigma
            return 1.0 - 0.5 * math.exp(mu / scale)
        if p == 2.0:
            from scipy.special import ndtr

            return float(ndtr(-mu / sigma))
        raise ValueError("Target zero fraction is unavailable for this development target.")

    def target_manifest(self) -> dict[str, float]:
        """Persist target parameters and derived linked moments."""
        p, mu, sigma = self.target_parameters
        positive_mean, positive_second = self.target_positive_moments
        return {
            "p": p,
            "mu": mu,
            "sigma": sigma,
            "linked_mean": positive_mean,
            "linked_second_moment": positive_second,
            "target_rms": self.target_rms,
            "target_zero_fraction": self.target_zero_fraction,
        }


def rectified_gaussian_development_config(
    *, zero_fraction: float = 0.75, predictor_family: PredictorFamily = "mlp", seed: int = 13
) -> RepresentationConfig:
    """Create the single development-only rectified-Gaussian control."""
    if zero_fraction not in {0.5, 0.75, 0.875}:
        raise ValueError("Rectified-Gaussian control uses the locked sparsity sweep values.")
    from scipy.special import ndtri

    active_fraction = 1.0 - zero_fraction
    return RepresentationConfig(
        "sparse",
        predictor_family=predictor_family,
        generalized_gaussian_p=2.0,
        generalized_gaussian_mu=float(ndtri(active_fraction)),
        generalized_gaussian_sigma=1.0,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Describe safe representation weights and deterministic continuation state."""

    checkpoint_id: str
    geometry: Geometry
    predictor_family: PredictorFamily
    fold_id: str
    seed: int
    sequence_manifest_hash: str
    normalization_hash: str
    cutoff: str
    architecture_hash: str
    torch_version: str
    weights_sha256: str
    weights_format: Literal["safetensors"] = "safetensors"
    checkpoint_role: Literal["latest", "best", "final"] = "final"
    adaptation: str = "none"
    dataset_manifest_hash: str = ""
    universe_manifest_hash: str = ""
    architecture: str = "dynamic-token-encoder-13-128-conditioner-5"
    encoder: str = "linear-layernorm-gelu-linear"
    link: str = "identity"
    generalized_gaussian_p: float = 2.0
    generalized_gaussian_mu: float = 0.0
    generalized_gaussian_sigma: float = 1.0
    target_rms: float = 1.0
    target_positive_mean: float = 0.0
    target_positive_second_moment: float = 1.0
    target_zero_fraction: float = 0.0
    rdm_projections: int = 512
    calibrated_rdm_lambda: float = 0.0
    optimizer: str = "AdamW"
    scheduler: str = "linear-warmup-cosine"
    cuda_version: str | None = None
    cudnn_version: str | None = None
    code_commit: str = "UNSET"
    training_config_hash: str = ""
    paper_config_hash: str = ""
    validation_diagnostics: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    collapse_gate_status: Literal["PASS", "FAIL", "NOT RUN", "BLOCKED"] = "NOT RUN"
    collapse_gate_name: str = "paper-required-v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_manifest_hash", self.dataset_manifest_hash),
            ("universe_manifest_hash", self.universe_manifest_hash),
            ("sequence_manifest_hash", self.sequence_manifest_hash),
            ("normalization_hash", self.normalization_hash),
            ("architecture_hash", self.architecture_hash),
            ("training_config_hash", self.training_config_hash),
            ("paper_config_hash", self.paper_config_hash),
        ):
            if value and len(value) != 64:
                raise ValueError(f"Checkpoint {name} must contain a full SHA-256 digest.")
        if self.weights_sha256 and len(self.weights_sha256) != 64:
            raise ValueError(
                "Checkpoint weights_sha256 must be empty before save or a full digest."
            )
        if self.rdm_projections <= 0 or self.calibrated_rdm_lambda < 0:
            raise ValueError("Checkpoint RDMReg settings are invalid.")
        if not 0 <= self.target_zero_fraction <= 1 or self.target_rms <= 0:
            raise ValueError("Checkpoint target geometry is invalid.")


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Cover every input that can change a frozen embedding export."""

    raw_hash: str
    sequence_hash: str
    normalization_hash: str
    fold_id: str
    cutoff: str
    architecture_hash: str
    geometry: Geometry
    sparsity_target: float | None
    predictor_family: PredictorFamily
    seed: int
    checkpoint_hash: str
    torch_version: str


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    """Declare every identity that must match before checkpoint loading."""

    geometry: Geometry
    predictor_family: PredictorFamily
    fold_id: str
    cutoff: str
    universe_manifest_hash: str
    dataset_manifest_hash: str
    sequence_manifest_hash: str
    normalization_hash: str
    architecture_hash: str
    training_config_hash: str
    paper_config_hash: str
    generalized_gaussian_p: float
    generalized_gaussian_mu: float
    generalized_gaussian_sigma: float
    target_rms: float
    target_zero_fraction: float
    rdm_projections: int
    calibrated_rdm_lambda: float
    torch_version: str
    torch_compatibility: Literal["exact", "major-minor"] = "major-minor"
    adaptation: str = "none"
