# Data leakage contract

Point-in-time integrity is a hard correctness property, not a reporting preference.

## Information clock

| Time | Permitted information |
|---|---|
| Before session | Prior-session bars and derived features whose `available_at` is no later than the pre-session cutoff; calendar and externally supplied assumptions |
| Start of bucket `k` | Completed buckets strictly before `k`; forecasts generated at or before this timestamp; remaining inventory and prior fills |
| While bucket `k` forms | For POV only, contemporaneously materializing volume may determine participation quantity; the eventual close/high/low/VWAP/total volume is not available as a decision feature |
| After bucket `k` closes | Final bar fields for `k` may update the next decision, forecast, and diagnostics |
| After session | Full-session targets and benchmarks may be constructed for evaluation, never retroactively supplied to deployable policies |

## Enforceable boundaries

- `DecisionContext` contains observations through a declared cutoff and forecasts for the future; it never contains the full target-day DataFrame.
- Static forecasts have `training_data_cutoff < session_date`.
- Dynamic samples state `as_of`; every feature record satisfies `available_at <= as_of`.
- Global chronological split dates apply across all symbols. No symbol/date belongs to more than one partition of a fold.
- Preprocessors fit on training partitions only. Validation may select hyperparameters. Test partitions remain locked until selection completes.
- Historical-profile VWAP receives prior sessions separately and rejects target/future dates.
- Oracle policies are type- and registry-separated, visibly labeled, and excluded from deployable defaults.

## Bar-level POV convention

The simulator treats POV quantity as flow that participates while trades occur inside a minute. It may use the volume realized during that minute only for the mechanical quantity `floor(rho*volume)`; it may not use that minute's eventual price path or any later bar to choose aggressiveness. The fill is summarized at the bar reference price after the bucket closes. This is an aggregation convention, not a claim that the closing bar was known beforehand.

## Failure behavior

Future-dated features, forecasts generated after their first forecast bucket, overlapping split memberships, missing `as_of`, ambiguous timezone, or target-day profile leakage cause validation errors. Missing buckets are excluded or masked with a recorded reason; they are never silently filled with zero.
