# ADR 0009: Separate formation, representation, and execution data quality by resolution

- Status: Accepted
- Date: 2026-09-05
- Owners: ExecSim maintainers
- Applies to: `sparse-jepa-v2`
- Amends: [ADR 0008](0008-redirect-paper-to-representation-accessibility.md)

## Context

The real Alpaca SIP formation corpus stopped `sparse-jepa-v1` before target acquisition. The locked exact-minute rule admitted 62 of 505 formation candidates, versus the required 100, and median exact-standard-session completeness was 12.749%. SPY also lacked one exact standard session because five one-minute aggregates were absent on 2021-05-05.

V1 treated absence of provider-emitted one-minute aggregates as equivalent to unusable market data and propagated minute-level exactness into a universe-selection step and a 15-minute representation-learning task. Real formation data demonstrated that this assumption was misaligned with both provider semantics and model resolution. V2 separates quality definitions by task resolution.

Alpaca documents that stock minute and daily bars are independently aggregated from trades, with bar-field updates governed by tape, trade condition, and bar type. A bar is emitted only when its OHLCV fields are nonzero. Therefore, no emitted minute bar can mean no trade that produced a complete eligible aggregate; it does not establish that the market had zero activity or zero volume. Alpaca also documents that higher non-daily intervals are aggregated from observed minute bars using first open, maximum high, minimum low, last close, summed volume and trade count, and volume-weighted VWAP.

## Evidence

- V1 terminal status: `BLOCKED — INSUFFICIENT FORMATION UNIVERSE UNDER LOCKED EXACT-MINUTE CRITERION`.
- V1 eligible candidates: 62 of 505.
- V1 median exact-session completeness: 12.749% after excluding the predeclared early close from the 251-session denominator.
- SPY 2021-05-05: 385 observed minute aggregates and therefore not exact at the full-session minute resolution. V2 separately determines whether all 26 model tokens remain computable.
- Provider evidence: the Alpaca Market Data FAQ defines emission and aggregation rules and explicitly distinguishes minute from daily condition handling. The provider semantics and source links are recorded in `docs/RESEARCH_REFERENCES.md`.
- Selection-bias evidence: the v1-versus-v2 activity correlations remain `NOT RUN` until the bounded formation token-quality scan completes. They are diagnostic and cannot change this decision or tune the 95% daily threshold.

## Decision

Adopt this quality hierarchy for `sparse-jepa-v2`:

```text
daily formation quality
  -> observed-only 15-minute JEPA token quality
  -> exact-minute TCA execution-window quality
```

The four and only four conceptual protocol changes from v1 are:

1. Formation completeness is measured from valid Alpaca SIP `1Day` observations over expected XNYS trading days, not from exact 390-row minute sessions. The unchanged threshold is 95%.
2. Representation-session completeness is measured at the actual 26-token resolution. Each fixed 15-minute interval aggregates only observed provider minute bars and requires at least two chronological valid observations so every OHLCV, trade-count, VWAP, return, and realized-volatility input is computable. All 26 tokens and the preceding valid close needed by cross-token returns are required for a primary standard session.
3. TCA retains strict minute-level quality over the simulator's actual start-inclusive, end-exclusive execution interval: 10:30 through 15:29 local time, exactly 300 unique valid rows. No price, capacity, or fill input is imputed.
4. Missing provider minute aggregates are never automatically interpreted as zero market activity. V2 inserts no fake bars, zero-fills no missing minute volume, and performs no interpolation to satisfy a quality gate.

For irregular but valid tokens, open and close are the first and last observed values, high and low are extrema, volume and trade count are sums, and VWAP is weighted by observed volume. Token realized volatility is the square root of the sum of squared consecutive observed-close log returns. A return spanning a provider gap remains one observed-grid return; V2 does not pretend it is a regular one-minute return or normalize it by elapsed time.

The formation universe is rebuilt from the complete 505-candidate source population using direct Alpaca SIP daily bars, the 2021-01-04 membership snapshot, ordinary/common-stock eligibility, stable sourced identities, median price of at least $5, positive median daily dollar volume, and stable-ID tie-breaking. If at least 100 qualify, exactly the top 100 by median daily dollar volume are frozen after the formation period. Target-period gaps exclude cases, never instruments, and no survivor replacement is allowed.

Every quality record stores `daily_valid`, `minute_exact_full_session`, `token_valid_full_session`, `tca_window_exact`, `early_close`, `provider_gap_count`, `observed_minute_count`, `valid_token_count`, and `invalid_token_reason` separately. SPY uses the same task-resolution contracts.

All research questions, geometries, encoder, latent width, sparsity, folds, seeds, metrics, LightGBM design, TCA order and cost assumptions, bootstrap, primary comparisons, causal boundaries, and authorization gates remain as frozen in v1.

## Rejected alternatives

- Lower the v1 exact-minute completeness threshold. This would tune a mismatched criterion to obtain a desired count.
- Keep only the 62 v1 stocks. This changes the declared universe size and preserves activity-based selection.
- Replace missing or later unavailable stocks. This introduces survivor replacement after formation.
- Zero-fill absent minutes. Alpaca's omission semantics do not establish zero market activity.
- Interpolate all one-minute bars. Fabricated prices, volumes, or capacity are not provider observations.
- Switch provider merely to preserve v1. That would change the data-generating system instead of correcting the resolution mismatch.
- Silently mutate v1. V1 is retained as a distinct immutable blocked protocol and evidence bundle.

## Consequences

V2 is a new protocol version with distinct configuration, freeze, candidate, universe, corpus, report, and artifact identities. Daily eligibility can admit securities that v1 rejected for sparse minute emission; this is the intended removal of a task-misaligned filter, not a threshold relaxation.

The JEPA/forecast sample can be larger than the exact-minute TCA complete-case subset. Reports and manifests must never collapse these populations into one `complete` flag. Early closes may be valid daily observations but remain excluded from the primary 26-token representation sample. Formation ranking is known only after 2021-12-31.

The v2 formation scan may retrieve daily formation bars under its separate network authorization, but it must stop before target acquisition. Historical model training, test inspection, and historical TCA remain `NOT RUN`. If daily formation produces at least 100 eligible names, the next terminal state is `AWAITING V2 FORMATION APPROVAL`; otherwise it is `BLOCKED — INSUFFICIENT DAILY-QUALITY FORMATION UNIVERSE`.

## Verification

Focused tests cover daily denominators, early closes, deterministic liquidity ranking, observed-only token aggregation, irregular realized volatility, exact-minute TCA windows, SPY parity, independent quality fields, and v1/v2 artifact separation. The final formation evidence additionally records bounded peak memory, scan throughput, token throughput, storage, all-candidate distributions, the frozen top 100 when available, and v1-versus-v2 selection-bias diagnostics.
