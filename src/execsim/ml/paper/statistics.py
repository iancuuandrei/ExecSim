"""Date-clustered paired inference for locked paper endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PairedBlockResult:
    """Summarize one paired candidate-minus-baseline comparison."""

    paired_dates: int
    mean_difference: float
    median_difference: float
    confidence_interval: tuple[float, float]
    date_win_rate: float
    standardized_effect: float


@dataclass(frozen=True, slots=True)
class CompleteCaseResult:
    """Describe an exact paired intersection before date aggregation."""

    paired_rows: pd.DataFrame
    baseline_rows: int
    candidate_rows: int
    matched_rows: int
    dropped_baseline_rows: int
    dropped_candidate_rows: int


def construct_complete_case_differences(
    rows: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    value_column: str,
    identity_columns: tuple[str, ...],
    method_column: str = "method",
) -> CompleteCaseResult:
    """Intersect exact experiment identities before calculating paired differences."""
    required = {method_column, value_column, *identity_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Complete-case ledger missing columns: {sorted(missing)}")
    selected = rows.loc[rows[method_column].isin((baseline, candidate))].copy()
    if selected.duplicated([method_column, *identity_columns]).any():
        raise ValueError("Complete-case ledger contains duplicated method/case rows.")
    base = selected.loc[selected[method_column] == baseline, [*identity_columns, value_column]]
    other = selected.loc[selected[method_column] == candidate, [*identity_columns, value_column]]
    paired = base.merge(
        other,
        on=list(identity_columns),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["difference"] = paired[f"{value_column}_candidate"] - paired[f"{value_column}_baseline"]
    return CompleteCaseResult(
        paired_rows=paired,
        baseline_rows=len(base),
        candidate_rows=len(other),
        matched_rows=len(paired),
        dropped_baseline_rows=len(base) - len(paired),
        dropped_candidate_rows=len(other) - len(paired),
    )


def paper_forecast_metrics(
    actual_remaining: np.ndarray,
    predicted_remaining: np.ndarray,
    actual_shape: np.ndarray,
    predicted_shape: np.ndarray,
) -> dict[str, float]:
    """Compute primary log remaining-volume MAE and conditional-curve W1."""
    actual_total = np.asarray(actual_remaining, dtype=float)
    predicted_total = np.asarray(predicted_remaining, dtype=float)
    actual_curve = np.asarray(actual_shape, dtype=float)
    predicted_curve = np.asarray(predicted_shape, dtype=float)
    if actual_total.shape != predicted_total.shape or actual_curve.shape != predicted_curve.shape:
        raise ValueError("Paper forecast metric arrays must align.")
    if actual_curve.ndim != 2 or len(actual_curve) != len(actual_total):
        raise ValueError("Paper shape metrics require one curve per volume target.")
    if not all(
        np.isfinite(values).all()
        for values in (actual_total, predicted_total, actual_curve, predicted_curve)
    ):
        raise ValueError("Paper forecast metrics require finite values.")
    if (actual_total < 0).any() or (predicted_total < 0).any():
        raise ValueError("Remaining-volume metrics require non-negative values.")
    if (actual_curve < 0).any() or (predicted_curve < 0).any():
        raise ValueError("Conditional curves require non-negative shares.")
    if not np.allclose(actual_curve.sum(axis=1), 1.0) or not np.allclose(
        predicted_curve.sum(axis=1), 1.0
    ):
        raise ValueError("Conditional curves must be row-normalized.")
    return {
        "log_remaining_volume_mae": float(
            np.mean(np.abs(np.log1p(predicted_total) - np.log1p(actual_total)))
        ),
        "conditional_curve_wasserstein": float(
            np.mean(np.abs(np.cumsum(predicted_curve, axis=1) - np.cumsum(actual_curve, axis=1)))
        ),
    }


def average_cases_by_date(
    rows: pd.DataFrame,
    *,
    value_column: str,
    model_columns: tuple[str, ...] = ("model", "fold_id", "seed"),
) -> pd.DataFrame:
    """Give each date equal weight after averaging its symbol-as-of cases."""
    required = {"date", value_column, *model_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Date aggregation missing columns: {sorted(missing)}")
    return (
        rows.groupby([*model_columns, "date"], sort=True, as_index=False)[value_column]
        .mean()
        .sort_values([*model_columns, "date"], kind="stable")
        .reset_index(drop=True)
    )


def all_seed_claim_supported(seed_effects: pd.DataFrame) -> bool:
    """Require same-direction seed effects and an averaged interval excluding zero."""
    required = {"seed", "mean_difference", "average_ci_lower", "average_ci_upper"}
    missing = required.difference(seed_effects.columns)
    if missing or seed_effects.empty:
        raise ValueError(f"Seed claim evidence missing columns or rows: {sorted(missing)}")
    effects = seed_effects["mean_difference"].to_numpy(dtype=float)
    same_direction = bool(np.all(effects < 0) or np.all(effects > 0))
    lower = float(seed_effects["average_ci_lower"].iloc[0])
    upper = float(seed_effects["average_ci_upper"].iloc[0])
    return same_direction and (upper < 0 or lower > 0)


def moving_block_bootstrap(
    date_values: pd.DataFrame,
    *,
    value_column: str = "difference",
    fold_column: str = "fold_id",
    block_length: int = 5,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> PairedBlockResult:
    """Bootstrap contiguous dates without permitting blocks to cross folds."""
    required = {"date", value_column, fold_column}
    missing = required.difference(date_values.columns)
    if missing:
        raise ValueError(f"Block bootstrap missing columns: {sorted(missing)}")
    if block_length <= 0 or repetitions <= 0 or not 0 < confidence < 1:
        raise ValueError("Bootstrap parameters are invalid.")
    ordered = date_values.sort_values([fold_column, "date"], kind="stable")
    if ordered.duplicated([fold_column, "date"]).any():
        raise ValueError("Bootstrap input must contain one paired value per fold-date.")
    values = ordered[value_column].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be non-empty and finite.")
    fold_blocks: list[tuple[int, list[np.ndarray]]] = []
    for _, group in ordered.groupby(fold_column, sort=True):
        group_values = group[value_column].to_numpy(dtype=float)
        width = min(block_length, len(group_values))
        blocks = list(
            group_values[start : start + width] for start in range(len(group_values) - width + 1)
        )
        fold_blocks.append((len(group_values), blocks))
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled: list[float] = []
        for fold_size, blocks in fold_blocks:
            fold_sample: list[float] = []
            while len(fold_sample) < fold_size:
                fold_sample.extend(blocks[int(rng.integers(len(blocks)))].tolist())
            sampled.extend(fold_sample[:fold_size])
        means[repetition] = np.mean(sampled)
    alpha = (1 - confidence) / 2
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return PairedBlockResult(
        paired_dates=len(values),
        mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)),
        confidence_interval=(
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha)),
        ),
        date_win_rate=float(np.mean(values < 0)),
        standardized_effect=float(np.mean(values) / standard_deviation)
        if standard_deviation
        else 0.0,
    )


def holm_adjust_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Apply the predeclared Holm step-down family-wise adjustment."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires a finite one-dimensional family.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("P-values must lie in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
