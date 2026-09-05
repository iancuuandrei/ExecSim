"""Unified, fail-closed configuration for the locked paper experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from execsim.data.paper.manifests import file_sha256, stable_hash
from execsim.data.paper.partitions import PAPER_FOLDS

CONFIG_FILES = (
    "data.yaml",
    "sequences.yaml",
    "representation.yaml",
    "lightgbm.yaml",
    "evaluation.yaml",
    "tca.yaml",
)


@dataclass(frozen=True, slots=True)
class PaperRunConfig:
    """Hold all six authoritative paper configurations and their canonical hash."""

    root: Path
    sections: dict[str, dict[str, Any]]
    config_hash: str
    design_freeze: dict[str, Any]

    @property
    def data(self) -> dict[str, Any]:
        return self.sections["data"]

    @property
    def sequences(self) -> dict[str, Any]:
        return self.sections["sequences"]

    @property
    def representation(self) -> dict[str, Any]:
        return self.sections["representation"]

    @property
    def lightgbm(self) -> dict[str, Any]:
        return self.sections["lightgbm"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.sections["evaluation"]

    @property
    def tca(self) -> dict[str, Any]:
        return self.sections["tca"]

    @property
    def allow_network(self) -> bool:
        return bool(self.data["allow_network"])

    @property
    def allow_historical_training(self) -> bool:
        return bool(self.data["allow_historical_training"])

    @property
    def allow_full_paper_run(self) -> bool:
        return bool(self.data["allow_full_paper_run"])

    @property
    def paper_run_id(self) -> str:
        return str(self.data["paper_run_id"])

    @property
    def artifact_root(self) -> Path:
        return Path(self.data["artifact_root"])

    @property
    def report_root(self) -> Path:
        return Path(self.data["report_root"])

    def authorize(self, operation: str, *, cli_enabled: bool) -> None:
        """Require config and CLI opt-in for each privileged operation."""
        mapping = {
            "network": self.allow_network,
            "historical_training": self.allow_historical_training,
            "full_paper_run": self.allow_full_paper_run,
        }
        if operation not in mapping:
            raise ValueError(f"Unknown paper operation: {operation}")
        if not mapping[operation] or not cli_enabled:
            raise PermissionError(
                f"Paper operation {operation!r} requires explicit config and "
                "command-line authorization."
            )


def load_paper_config(path: Path) -> PaperRunConfig:
    """Load and cross-validate all six configuration files from one directory."""
    root = path if path.is_dir() else path.parent
    sections: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        file_path = root / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Paper configuration is incomplete: {file_path}")
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Paper YAML must contain a mapping: {file_path}")
        sections[file_path.stem] = payload
    freeze_path = root / "design-freeze-v1.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Paper design freeze is missing: {freeze_path}")
    import json

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise TypeError("Paper design freeze must contain a JSON object.")
    specification = root.parents[2] / str(freeze["source_specification"])
    if file_sha256(specification) != freeze.get("source_specification_sha256"):
        raise ValueError("Paper design freeze does not match its normative specification.")
    _validate_sections(sections)
    return PaperRunConfig(
        root, sections, stable_hash({"sections": sections, "freeze": freeze}), freeze
    )


def _validate_sections(sections: dict[str, dict[str, Any]]) -> None:
    data = sections["data"]
    sequence = sections["sequences"]
    representation = sections["representation"]
    lightgbm = sections["lightgbm"]
    evaluation = sections["evaluation"]
    tca = sections["tca"]
    expected = {
        "provider": "alpaca",
        "feed": "sip",
        "frequency": "1min",
        "adjustment": "raw",
        "timezone": "America/New_York",
        "extended_hours": False,
        "universe_size": 100,
    }
    contradictions = [name for name, value in expected.items() if data.get(name) != value]
    if contradictions:
        raise ValueError(f"Paper data configuration contradicts locked design: {contradictions}")
    if (
        sequence.get("token_minutes") != 15
        or sequence.get("session_tokens") != 26
        or sequence.get("feature_count") != representation.get("observed_feature_dim")
        or tuple(sequence.get("horizons", ())) != (1, 2, 4, 8)
        or sequence.get("context_length") != 8
    ):
        raise ValueError("Sequence and representation dimensions contradict the locked design.")
    configured_folds = evaluation.get("folds")
    expected_folds = [
        {
            "id": fold.fold_id,
            "train": [fold.train_start, fold.train_end],
            "validation": [fold.validation_start, fold.validation_end],
            "test": [fold.test_start, fold.test_end],
        }
        for fold in PAPER_FOLDS
    ]
    if configured_folds != expected_folds:
        raise ValueError("Evaluation folds do not match locked PAPER_FOLDS.")
    if lightgbm.get("main_rows") != ["ewma", "raw", "untrained_neural", "dense", "sparse"]:
        raise ValueError("LightGBM main rows contradict the locked comparison.")
    if (
        representation.get("geometries") != ["dense", "sparse"]
        or representation.get("predictor_family") != "mlp"
        or representation.get("observed_feature_dim") != 18
        or representation.get("feature_dim") != 13
        or representation.get("conditioning_dim") != 5
        or representation.get("seeds") != [13, 29, 47]
        or representation.get("max_epochs") != 40
        or representation.get("early_stopping_patience") != 6
        or representation.get("warmup_fraction") != 0.05
        or representation.get("rdm_lambda_candidates") != [0.1, 1.0, 10.0]
        or representation.get("probe_capacity_ladder") != ["affine_ridge", "mlp_64", "mlp_256"]
        or representation.get("probe_ridge_alphas") != [0.1, 1.0, 10.0]
        or representation.get("probe_mlp_epochs") != 20
        or representation.get("checkpoint_interval_steps") != 500
        or representation.get("rdm_diagnostic_sample_rows") != 2048
        or representation.get("safe_resource_bounds")
        != {
            "maximum_representation_runs": 29,
            "maximum_jepa_steps": 10_000_000,
            "maximum_shape_rows": 25_000_000,
            "maximum_embedding_bytes": 200_000_000_000,
        }
        or representation.get("future_difficulty_adaptation", {}).get("paper_matrix") is not False
    ):
        raise ValueError("Representation configuration contradicts the locked comparison.")
    sparse_target = representation.get("sparse_target", {})
    if (
        sparse_target.get("p") != 2.0
        or sparse_target.get("mu") != -0.6744897501960817
        or sparse_target.get("sigma") != 1.0
    ):
        raise ValueError("Primary sparse target must be the locked rectified Gaussian.")
    if (
        lightgbm.get("num_leaves") != [15, 31]
        or lightgbm.get("min_child_samples") != [50, 200]
        or lightgbm.get("reg_lambda") != [1.0, 10.0]
        or lightgbm.get("learning_rate") != 0.03
        or lightgbm.get("n_estimators") != 2000
        or lightgbm.get("early_stopping_rounds") != 100
    ):
        raise ValueError("LightGBM configuration contradicts the locked grid.")
    if (
        evaluation.get("bootstrap_block_dates") != 5
        or evaluation.get("bootstrap_block_sensitivity_dates") != [1, 10]
        or evaluation.get("bootstrap_repetitions") != 10_000
        or evaluation.get("confidence") != 0.95
        or evaluation.get("multiple_testing") != "holm"
        or len(evaluation.get("confirmatory_contrast_definitions", ())) != 5
    ):
        raise ValueError("Evaluation configuration contradicts locked inference.")
    if tca.get("forecast_update_minutes") != 15 or tca.get("fill_minutes") != 1:
        raise ValueError("TCA update/fill clocks contradict the locked design.")
    if (
        tca.get("window") != ["10:30", "15:30"]
        or tca.get("quantity_fraction_adv20") != 0.03
        or tca.get("planned_participation_rate") != 0.10
        or tca.get("hard_participation_rate") != 0.10
        or tca.get("risk_aversion") != 0.0
        or tca.get("tracking_penalty") != 0.0
    ):
        raise ValueError("TCA configuration contradicts the locked experiment.")
    if data.get("allow_full_paper_run") and not data.get("allow_historical_training"):
        raise ValueError("Full paper evaluation requires historical-training authorization.")
