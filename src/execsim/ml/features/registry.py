from __future__ import annotations

from dataclasses import dataclass, field

from execsim.ml.schemas import FeatureSpec


@dataclass(slots=True)
class FeatureRegistry:
    version: str
    _features: dict[str, FeatureSpec] = field(default_factory=dict)

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._features:
            raise ValueError(f"Feature is already registered: {spec.name}")
        self._features[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._features[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature: {name}") from exc

    def names(self, mode: str | None = None) -> tuple[str, ...]:
        return tuple(
            name
            for name, spec in sorted(self._features.items())
            if mode is None or spec.mode in {mode, "both"}
        )

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "description": spec.description,
                "source_fields": list(spec.source_fields),
                "lookback": spec.lookback,
                "transformation": spec.transformation,
                "earliest_availability": spec.earliest_availability,
                "mode": spec.mode,
                "missing_value_rule": spec.missing_value_rule,
                "version": spec.version,
                "leakage_notes": spec.leakage_notes,
                "rationale": spec.rationale,
            }
            for spec in sorted(self._features.values(), key=lambda value: value.name)
        ]


def _spec(
    name: str,
    description: str,
    sources: tuple[str, ...],
    lookback: str,
    transformation: str,
    availability: str,
    mode: str,
    rationale: str,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        dtype="float64",
        description=description,
        source_fields=sources,
        lookback=lookback,
        transformation=transformation,
        earliest_availability=availability,
        mode=mode,  # type: ignore[arg-type]
        missing_value_rule="exclude sample when unavailable; never backfill from the future",
        version="volume-features-v1",
        leakage_notes="source cutoff must be no later than sample as_of",
        rationale=rationale,
    )


DEFAULT_FEATURE_REGISTRY = FeatureRegistry("volume-features-v1")
for _feature in (
    _spec(
        "weekday",
        "Session weekday number.",
        ("timestamp",),
        "none",
        "calendar extraction",
        "before session",
        "both",
        "Intraday volume shape can vary by weekday.",
    ),
    _spec(
        "month",
        "Session month number.",
        ("timestamp",),
        "none",
        "calendar extraction",
        "before session",
        "both",
        "Seasonality may affect turnover.",
    ),
    _spec(
        "bucket_index",
        "Target bucket ordinal.",
        ("timestamp",),
        "none",
        "session ordinal",
        "before session",
        "both",
        "Volume has a stable time-of-day structure.",
    ),
    _spec(
        "previous_session_total_volume",
        "Prior session total volume.",
        ("volume",),
        "one session",
        "sum",
        "after previous close",
        "both",
        "Recent turnover anchors total-volume scale.",
    ),
    _spec(
        "rolling_adv",
        "Mean prior-session volume.",
        ("volume",),
        "20 sessions",
        "rolling mean",
        "after previous close",
        "both",
        "Average daily volume is a liquidity-scale baseline.",
    ),
    _spec(
        "rolling_median_volume",
        "Median prior-session volume.",
        ("volume",),
        "20 sessions",
        "rolling median",
        "after previous close",
        "both",
        "Median turnover is robust to volume spikes.",
    ),
    _spec(
        "rolling_volatility",
        "Prior close-to-close volatility.",
        ("close",),
        "20 sessions",
        "standard deviation of log returns",
        "after previous close",
        "both",
        "Turnover can vary with recent volatility.",
    ),
    _spec(
        "previous_session_return",
        "Prior session open-to-close log return.",
        ("open", "close"),
        "one session",
        "log close/open",
        "after previous close",
        "both",
        "Recent market activity can affect next-session volume.",
    ),
    _spec(
        "previous_session_range",
        "Prior session high-low range over open.",
        ("open", "high", "low"),
        "one session",
        "(high-low)/open",
        "after previous close",
        "both",
        "Range is a simple prior-session activity proxy.",
    ),
    _spec(
        "previous_session_trade_count",
        "Prior session trade count.",
        ("trade_count",),
        "one session",
        "sum",
        "after previous close",
        "both",
        "Trade count separates print frequency from share volume.",
    ),
    _spec(
        "elapsed_session_fraction",
        "Fraction of decision buckets elapsed.",
        ("timestamp",),
        "current session",
        "elapsed/total",
        "bucket start",
        "dynamic",
        "Forecast horizon shrinks through the session.",
    ),
    _spec(
        "observed_cumulative_volume",
        "Volume in completed target-session buckets.",
        ("volume",),
        "current session through k-1",
        "sum",
        "bucket start",
        "dynamic",
        "Observed turnover updates remaining-volume scale.",
    ),
    _spec(
        "recent_bucket_volume",
        "Most recently completed bucket volume.",
        ("volume",),
        "one bucket",
        "identity",
        "next bucket start",
        "dynamic",
        "Recent volume detects local deviations from profile.",
    ),
    _spec(
        "recent_realized_volatility",
        "Volatility of completed target-session returns.",
        ("close",),
        "up to five buckets",
        "standard deviation of log returns",
        "bucket start",
        "dynamic",
        "Intraday activity and volatility can co-move.",
    ),
    _spec(
        "required_future_participation",
        "Inventory divided by expected remaining volume.",
        ("remaining_inventory", "forecast_volume"),
        "current decision",
        "ratio",
        "bucket start",
        "dynamic",
        "Feasibility depends on required participation.",
    ),
):
    DEFAULT_FEATURE_REGISTRY.register(_feature)
