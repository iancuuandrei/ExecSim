"""Formation diagnostics, resource plan, and human approval report for v2."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.acquisition import monthly_chunks
from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.data.paper.partitions import PAPER_FOLDS


def write_v2_formation_bundle(
    *,
    v1_candidates_path: Path,
    v2_candidates_path: Path,
    quality_path: Path,
    universe_path: Path,
    ticker_history_path: Path,
    daily_receipt_path: Path,
    quality_receipt_path: Path,
    output_directory: Path,
    report_path: Path,
    protocol_hash: str,
    observed_peak_rss_lower_bound_bytes: int | None = None,
) -> dict[str, Any]:
    """Write all-candidate diagnostics and the pre-target human approval bundle."""
    v1 = pd.read_parquet(v1_candidates_path)
    v2 = pd.read_parquet(v2_candidates_path)
    quality = pd.read_parquet(quality_path)
    universe = read_json(universe_path)
    daily_receipt = read_json(daily_receipt_path)
    quality_receipt = read_json(quality_receipt_path)
    _validate_inputs(v1, v2, quality, universe)
    comparison = _candidate_comparison(v1, v2, quality)
    selected_ids = {str(item["instrument_id"]) for item in universe["members"]}
    selected = comparison.loc[comparison["instrument_id"].isin(selected_ids)].copy()
    selected["token_completeness_band"] = np.select(
        [selected["token_completeness"] >= 0.95, selected["token_completeness"] >= 0.80],
        ["high", "medium"],
        default="low",
    )
    band_counts = {
        name: int((selected["token_completeness_band"] == name).sum())
        for name in ("high", "medium", "low")
    }
    correlations = _selection_bias_correlations(comparison)
    spy = quality.loc[
        (quality["instrument_id"] == "benchmark-spy") & (quality["session_date"] == "2021-05-05")
    ]
    if len(spy) != 1:
        raise ValueError("V2 report requires exactly one SPY 2021-05-05 quality row.")
    plan = _target_plan(
        universe,
        pd.read_parquet(ticker_history_path),
        selected_token_completeness=float(selected["token_completeness"].mean()),
        protocol_hash=protocol_hash,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = output_directory / "v1-v2-candidate-comparison.parquet"
    comparison.to_parquet(comparison_path, index=False)
    plan_path = output_directory / "target-acquisition-plan-v2.json"
    write_json_atomic(plan_path, plan)
    diagnostics_stable: dict[str, Any] = {
        "schema_version": "paper-v2-formation-diagnostics-v1",
        "protocol_id": "sparse-jepa-v2",
        "protocol_hash": protocol_hash,
        "candidate_count": len(comparison),
        "v1_eligible_count": int(comparison["v1_eligible"].sum()),
        "v2_daily_eligible_count": int(comparison["v2_daily_eligible"].sum()),
        "selected_count": len(selected),
        "selected_token_band_counts": band_counts,
        "correlations": correlations,
        "spy_2021_05_05": _json_record(spy.iloc[0]),
        "daily_receipt_sha256": file_sha256(daily_receipt_path),
        "quality_receipt_sha256": file_sha256(quality_receipt_path),
        "comparison_sha256": file_sha256(comparison_path),
        "universe_sha256": file_sha256(universe_path),
        "target_plan_sha256": file_sha256(plan_path),
        "resource_evidence": {
            "daily_rows": int(daily_receipt["rows"]),
            "minute_rows": int(quality_receipt["observed_minute_rows"]),
            "elapsed_seconds": float(quality_receipt["elapsed_seconds"]),
            "minute_rows_per_second": float(quality_receipt["minute_rows_per_second"]),
            "token_aggregation_attempts_per_second": float(
                quality_receipt["token_aggregation_attempts_per_second"]
            ),
            "scanner_peak_rss_bytes": quality_receipt.get("peak_rss_bytes"),
            "observed_peak_rss_lower_bound_bytes": observed_peak_rss_lower_bound_bytes,
            "formation_source_bytes": int(quality_receipt["source_response_bytes"]),
            "quality_parquet_bytes": int(quality_receipt["quality_parquet_bytes"]),
        },
        "target_data_status": "NOT RUN",
        "historical_training_status": "NOT RUN",
        "locked_test_status": "NOT RUN",
        "historical_tca_status": "NOT RUN",
        "terminal_status": "AWAITING V2 FORMATION APPROVAL",
    }
    diagnostics = {
        **diagnostics_stable,
        "diagnostics_sha256": stable_hash(diagnostics_stable),
    }
    diagnostics_path = output_directory / "formation-diagnostics-v2.json"
    write_json_atomic(diagnostics_path, diagnostics)
    report_path.write_text(
        _markdown_report(
            comparison,
            selected,
            universe,
            diagnostics,
            daily_receipt,
            quality_receipt,
            plan,
        ),
        encoding="utf-8",
    )
    return {
        **diagnostics,
        "diagnostics_manifest_sha256": file_sha256(diagnostics_path),
        "report_sha256": file_sha256(report_path),
    }


def _validate_inputs(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    quality: pd.DataFrame,
    universe: dict[str, Any],
) -> None:
    if len(v1) != 505 or len(v2) != 505 or set(v1["instrument_id"]) != set(v2["instrument_id"]):
        raise ValueError("V1 and v2 diagnostics require the same 505 stable candidates.")
    if universe.get("protocol_id") != "sparse-jepa-v2" or len(universe.get("members", ())) != 100:
        raise ValueError("V2 report requires a complete frozen 100-member universe.")
    required_quality = {
        "daily_valid",
        "minute_exact_full_session",
        "token_valid_full_session",
        "tca_window_exact",
        "early_close",
        "provider_gap_count",
        "observed_minute_count",
        "valid_token_count",
        "invalid_token_reason",
    }
    if missing := required_quality.difference(quality.columns):
        raise ValueError(f"V2 report quality data is incomplete: {sorted(missing)}")


def _candidate_comparison(
    v1: pd.DataFrame, v2: pd.DataFrame, quality: pd.DataFrame
) -> pd.DataFrame:
    standard = quality.loc[
        (~quality["early_close"]) & (quality["instrument_id"] != "benchmark-spy")
    ]
    aggregated = (
        standard.groupby("instrument_id", sort=True)
        .agg(
            token_completeness=("token_valid_full_session", "mean"),
            minute_exact_completeness_v2_scan=("minute_exact_full_session", "mean"),
            tca_window_completeness=("tca_window_exact", "mean"),
            average_observed_minute_count=("observed_minute_count", "mean"),
            formation_provider_gap_count=("provider_gap_count", "sum"),
        )
        .reset_index()
    )
    comparison = (
        v1.loc[:, ["instrument_id", "symbol", "session_completeness"]]
        .rename(
            columns={
                "symbol": "formation_symbol_v1",
                "session_completeness": "v1_exact_minute_completeness",
            }
        )
        .merge(v2, on="instrument_id", validate="one_to_one")
        .merge(aggregated, on="instrument_id", validate="one_to_one")
    )
    comparison["v1_eligible"] = comparison["v1_exact_minute_completeness"] >= 0.95
    comparison["v2_daily_eligible"] = (
        (comparison["security_type"] == "ordinary_common_stock")
        & comparison["in_sp500_on_formation_date"].astype(bool)
        & (comparison["median_daily_price"] >= 5.0)
        & (comparison["daily_completeness"] >= 0.95)
        & (comparison["median_daily_dollar_volume"] > 0)
    )
    comparison["liquidity_rank"] = (
        comparison["median_daily_dollar_volume"].rank(method="first", ascending=False).astype(int)
    )
    return comparison.sort_values("instrument_id", kind="stable").reset_index(drop=True)


def _selection_bias_correlations(comparison: pd.DataFrame) -> dict[str, dict[str, float]]:
    measures = {
        "log1p_median_daily_dollar_volume": np.log1p(
            comparison["median_daily_dollar_volume"].astype(float)
        ),
        "average_observed_minute_count": comparison["average_observed_minute_count"],
        "token_completeness": comparison["token_completeness"],
    }
    results: dict[str, dict[str, float]] = {}
    for name, values in measures.items():
        results[name] = {
            "v1_eligibility_pearson": float(comparison["v1_eligible"].astype(int).corr(values)),
            "v1_eligibility_spearman": float(
                comparison["v1_eligible"].astype(int).corr(values, method="spearman")
            ),
            "v1_completeness_pearson": float(
                comparison["v1_exact_minute_completeness"].corr(values)
            ),
            "v1_completeness_spearman": float(
                comparison["v1_exact_minute_completeness"].corr(values, method="spearman")
            ),
        }
    return results


def _target_plan(
    universe: dict[str, Any],
    ticker_history: pd.DataFrame,
    *,
    selected_token_completeness: float,
    protocol_hash: str,
) -> dict[str, Any]:
    import exchange_calendars as xcals

    start = date(2022, 1, 3)
    end = date(2025, 12, 31)
    calendar = xcals.get_calendar("XNYS")
    sessions = tuple(calendar.sessions_in_range(start, end))
    standard_sessions = sum(len(calendar.session_minutes(value)) == 390 for value in sessions)
    member_ids = {str(item["instrument_id"]) for item in universe["members"]}
    acquisition_ids = {*member_ids, "benchmark-spy"}
    history = ticker_history.loc[ticker_history["instrument_id"].astype(str).isin(acquisition_ids)]
    requests = 0
    for row in history.itertuples(index=False):
        interval_start = max(start, pd.Timestamp(row.start).date())
        interval_end = min(end, pd.Timestamp(row.end).date())
        if interval_start <= interval_end:
            requests += len(
                monthly_chunks(
                    str(row.instrument_id), str(row.symbol), interval_start, interval_end
                )
            )
    instruments = 101
    minute_rows_upper = instruments * len(sessions) * 390
    expected_token_sessions = round(standard_sessions * instruments * selected_token_completeness)
    fold_projection: dict[str, dict[str, int]] = {}
    training_origins = 0
    scale_rows = 0
    shape_train_rows = 0
    shape_held_out_rows = 0
    embedding_rows_upper = 0
    tca_case_identities = 0
    for fold in PAPER_FOLDS:
        counts = {
            "train_standard_sessions": _standard_session_count(
                calendar, fold.train_start, fold.train_end
            ),
            "validation_standard_sessions": _standard_session_count(
                calendar, fold.validation_start, fold.validation_end
            ),
            "test_standard_sessions": _standard_session_count(
                calendar, fold.test_start, fold.test_end
            ),
        }
        fold_projection[fold.fold_id] = counts
        training_origins += counts["train_standard_sessions"] * 100 * 2
        scale_rows += sum(counts.values()) * 100 * 22
        shape_train_rows += counts["train_standard_sessions"] * 100 * 88
        shape_held_out_rows += (
            (counts["validation_standard_sessions"] + counts["test_standard_sessions"]) * 100 * 253
        )
        embedding_rows_upper += sum(counts.values()) * 100 * 22 * 6
        tca_case_identities += counts["test_standard_sessions"] * 30
    stable: dict[str, Any] = {
        "schema_version": "paper-v2-target-acquisition-plan-v1",
        "protocol_id": "sparse-jepa-v2",
        "protocol_hash": protocol_hash,
        "status": "PLANNED_NOT_AUTHORIZED",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "instruments_including_spy": instruments,
        "xnys_sessions": len(sessions),
        "standard_390_minute_sessions": standard_sessions,
        "monthly_requests_from_sourced_symbol_intervals": requests,
        "expected_minute_rows_upper": minute_rows_upper,
        "expected_token_sessions_formation_rate_projection": expected_token_sessions,
        "projection_basis_selected_formation_token_completeness": selected_token_completeness,
        "fold_standard_session_projection": fold_projection,
        "training_origins_per_epoch_across_expanding_folds_upper": training_origins,
        "lightgbm_scale_rows_across_folds_upper": scale_rows,
        "lightgbm_shape_train_rows_across_folds_upper": shape_train_rows,
        "lightgbm_shape_validation_test_rows_across_folds_upper": shape_held_out_rows,
        "embedding_rows_upper": embedding_rows_upper,
        "embedding_bytes_upper": embedding_rows_upper * 644 * 4,
        "tca_case_identities_across_locked_tests_before_method_and_seed": (tca_case_identities),
        "raw_compressed_bytes_at_80_per_row": minute_rows_upper * 80,
        "processed_bytes_at_120_per_row": minute_rows_upper * 120,
        "target_acquisition": "NOT AUTHORIZED",
    }
    return {**stable, "plan_sha256": stable_hash(stable)}


def _standard_session_count(calendar: Any, start: date, end: date) -> int:
    return sum(
        len(calendar.session_minutes(value)) == 390
        for value in calendar.sessions_in_range(start, end)
    )


def _markdown_report(
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    universe: dict[str, Any],
    diagnostics: dict[str, Any],
    daily_receipt: dict[str, Any],
    quality_receipt: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    bands = diagnostics["selected_token_band_counts"]
    spy = diagnostics["spy_2021_05_05"]
    correlations = diagnostics["correlations"]
    exclusion_counts: Counter[str] = Counter()
    for raw in comparison["exclusion_reasons"]:
        exclusion_counts.update(json.loads(str(raw)))
    members = {str(item["instrument_id"]): item for item in universe["members"]}
    selected_rows = selected.sort_values("liquidity_rank", kind="stable")
    member_lines = [
        "| Rank | Symbol | Stable instrument | Liquidity group | Median price | "
        "Median daily dollar volume | Daily coverage | Token coverage | Band |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected_rows.itertuples(index=False):
        member = members[str(row.instrument_id)]
        member_lines.append(
            f"| {member['rank']} | {row.formation_symbol} | `{row.instrument_id}` | "
            f"{member['liquidity_group']} | {row.median_daily_price:.4f} | "
            f"{row.median_daily_dollar_volume:,.0f} | {row.daily_completeness:.3%} | "
            f"{row.token_completeness:.3%} | {row.token_completeness_band} |"
        )
    distribution = comparison[
        [
            "daily_completeness",
            "v1_exact_minute_completeness",
            "token_completeness",
            "average_observed_minute_count",
            "median_daily_dollar_volume",
        ]
    ].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    resource = diagnostics["resource_evidence"]
    liquidity_correlation = correlations["log1p_median_daily_dollar_volume"]
    activity_correlation = correlations["average_observed_minute_count"]
    peak_text = (
        f">= {resource['observed_peak_rss_lower_bound_bytes'] / 1024**2:.2f} MiB "
        "(OS lifetime-peak poll; in-process terminal counter unavailable in this run)"
        if resource["observed_peak_rss_lower_bound_bytes"]
        else "NOT MEASURED"
    )
    return (
        "# Sparse-JEPA v2 formation quality report\n\n"
        "## Outcome\n\n"
        "The daily-resolution formation protocol qualifies "
        f"{diagnostics['v2_daily_eligible_count']} of 505 candidates before ranking. "
        "Exactly 100 are frozen by median daily dollar volume with stable-ID tie-break. "
        "No target-period data, historical model, locked test, or historical TCA was run.\n\n"
        "**AWAITING V2 FORMATION APPROVAL**\n\n"
        "## Provider semantics\n\n"
        "Alpaca documents that stock minute and daily bars are trade aggregates with "
        "bar-type-specific condition rules and that a bar is emitted only when all OHLCV "
        "fields are nonzero. An absent minute is therefore not proof of zero activity or "
        "zero volume. V2 uses direct SIP `1Day` bars for formation and observed-only fixed "
        "15-minute aggregates for representation quality. It never inserts, interpolates, "
        "or zero-fills a missing minute. Sources are in `docs/RESEARCH_REFERENCES.md`.\n\n"
        "## V1 versus v2\n\n"
        f"V1 admitted {diagnostics['v1_eligible_count']} names under exact full-minute "
        f"formation completeness. V2 admits {diagnostics['v2_daily_eligible_count']} at the "
        "unchanged 95% concept measured over 252 expected daily sessions. Of the selected "
        f"100, token completeness is high for {bands['high']}, medium for {bands['medium']}, "
        f"and low for {bands['low']} under the pre-count 95%/80% bands and 251 standard-session "
        "denominator. This is a task-resolution correction, not a threshold relaxation.\n\n"
        "## Formation acquisition and exclusions\n\n"
        f"The direct daily corpus contains {daily_receipt['rows']:,} rows for "
        f"{daily_receipt['symbols_observed']} observed symbols including SPY, checksum "
        f"`{daily_receipt['content_sha256']}`. Daily exclusion reasons are "
        f"`{dict(sorted(exclusion_counts.items()))}`.\n\n"
        "Complete all-candidate distributions:\n\n"
        f"```text\n{distribution.to_string()}\n```\n\n"
        "## SPY resolution audit\n\n"
        f"On 2021-05-05 SPY has {spy['observed_minute_count']} observed minutes and "
        f"{spy['provider_gap_count']} provider gaps. Full-session minute exactness is "
        f"`{spy['minute_exact_full_session']}`, all {spy['valid_token_count']} tokens are valid "
        f"(`token_valid_full_session={spy['token_valid_full_session']}`), and exact TCA-window "
        f"quality is `{spy['tca_window_exact']}`. The five absences therefore do not invalidate "
        "SPY for JEPA context, but they do invalidate that date for exact-minute TCA.\n\n"
        "## Selection-bias diagnostic\n\n"
        "For all 505 candidates, the v1 eligibility indicator has Pearson/Spearman "
        f"correlations of {liquidity_correlation['v1_eligibility_pearson']:.3f}/"
        f"{liquidity_correlation['v1_eligibility_spearman']:.3f} "
        "with log median daily dollar volume and "
        f"{activity_correlation['v1_eligibility_pearson']:.3f}/"
        f"{activity_correlation['v1_eligibility_spearman']:.3f} "
        "with average observed minute count. V1 exact-minute eligibility therefore materially "
        "favored more liquid and more continuously emitting names. The diagnostic did not "
        "change v2 eligibility or thresholds.\n\n"
        "## Frozen top 100\n\n" + "\n".join(member_lines) + "\n\n## Resource evidence\n\n"
        f"The bounded scan read {quality_receipt['observed_minute_rows']:,} minute rows from "
        f"{quality_receipt['source_response_bytes'] / 1024**3:.3f} GiB of response files in "
        f"{quality_receipt['elapsed_seconds']:.2f} seconds: "
        f"{quality_receipt['minute_rows_per_second']:,.2f} minute rows/s and "
        f"{quality_receipt['token_aggregation_attempts_per_second']:,.2f} token attempts/s. "
        f"The quality Parquet is {quality_receipt['quality_parquet_bytes'] / 1024**2:.2f} MiB. "
        f"Peak RSS: {peak_text}.\n\n"
        "## Target acquisition plan\n\n"
        f"The frozen 100 plus SPY imply {plan['monthly_requests_from_sourced_symbol_intervals']:,} "
        f"monthly symbol-interval requests and at most {plan['expected_minute_rows_upper']:,} "
        f"minute rows over {plan['xnys_sessions']:,} target XNYS sessions. Projecting only for "
        f"capacity planning from formation token availability gives approximately "
        f"{plan['expected_token_sessions_formation_rate_projection']:,} stock/SPY token sessions. "
        f"The embedding upper bound is {plan['embedding_bytes_upper'] / 1024**3:.2f} GiB. "
        "These are planning quantities, not acquired data or empirical results.\n\n"
        "## Evidence boundary\n\n"
        "- DATA: formation only\n"
        "- TARGET DATA: NOT RUN\n"
        "- HISTORICAL TRAINING: NOT RUN\n"
        "- LOCKED TEST: NOT RUN\n"
        "- TCA: NOT RUN\n\n"
        "**AWAITING V2 FORMATION APPROVAL**\n"
    )


def _json_record(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (np.bool_, bool)):
            result[str(key)] = bool(value)
        elif isinstance(value, (np.integer, int)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[str(key)] = float(value) if math.isfinite(float(value)) else None
        else:
            result[str(key)] = str(value)
    return result
