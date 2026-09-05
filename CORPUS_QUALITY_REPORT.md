# Sparse-JEPA v1 corpus quality report

Status: `BLOCKED — INSUFFICIENT FORMATION UNIVERSE UNDER LOCKED EXACT-MINUTE CRITERION`

Generated: 2026-09-05  
Protocol: `sparse-jepa-v1`  
Acquisition-time design-freeze SHA-256: `4b15e8b8d4c38792e9141846e133727c9246f336960ae039e0432662a4b00d05`  
Safe-default design-freeze SHA-256: `08a6fbf23bebbfc1fe4cc4ab13909c584fbbdd5b926c1b8601cd689a7573bc98`  
Acquisition-time paper-config hash: `1ef2fc274174b7c7229a5e72709aa4ec5818b2a32fc255e698540214d406a3cb`  
Empirical branch revision: `a3a73330354900f7de15e2a645015c14edd3ee90`

## Decision

The locked formation-universe gate failed. Only 59 of 505 formation constituents satisfy all frozen eligibility rules, while the protocol requires exactly 100. The dominant failure is the exact-minute session-completeness requirement: 446 candidates are below 95% complete sessions. The run stopped before freezing a universe, acquiring target-period bars, acquiring corporate actions for the selected universe, constructing folds, or training any model.

The network-authorization flag was returned to its safe tracked default after acquisition. That operational-only correction changed the tracked freeze checksum but no scientific protocol field. Acquisition receipts and this report retain the exact acquisition-time configuration identity.

The existing empty universe manifest remains `NOT RUN`. No substitute instrument was selected, no completeness threshold was changed, missing minutes were not imputed as zero activity, and no test-model result exists.

## Provider and acquisition

| Item | Result |
|---|---|
| Provider/feed | Alpaca SIP |
| Frequency/adjustment | One-minute, raw |
| Formation interval | 2021-01-04 through 2021-12-31 |
| Entitlement probe | `PASS` — AAPL, 390 exact regular-session rows, nine pages |
| Planned formation instruments | 505 constituents plus SPY |
| Planned formation requests | 6,074 monthly or partial-month chunks |
| Receipts | 6,028 complete; 46 explicit zero-row failures |
| Retained response rows | 45,464,276 |
| Retained response bytes | 1,545,199,849 |
| Expected session slots from receipts | 126,534 |
| Exact primary sessions from receipts | 41,040 |
| Complete chunks with no exact session | 2,452 |
| Target-period acquisition | `NOT RUN` |
| Retries | Retry-attempt totals are not exposed by the current receipt schema |

All 46 zero-row failures were retained as failure receipts rather than counted as complete data. They occur after the last returned data for `TIF`, `VAR`, `FLIR`, `MXIM`, `ALXN`, and `CXO`. Termination causes were not audited because the run stopped before corporate-action acquisition.

Historical bars may incorporate later vendor corrections. The intended study reconstructs causal feature availability from the downloaded corpus; it cannot guarantee the exact historical vendor-feed vintage.

## Formation source and identity

The formation snapshot contains 505 sourced share-class rows effective on 2021-01-04. Stable IDs combine SEC CIK with the formation symbol. Later aliases are linked automatically only when the CIK has one unambiguous formation share class.

| Artifact | SHA-256 or revision |
|---|---|
| Constituent source revision | `ed4cf46e5ec5bb02e709aa08ee8a3a218d1b7d19` |
| Constituent source content | `0c248c94e708f33a6235688c47aadfccc0d7779c545fd65c6c8b698dcf964c1b` |
| Snapshot | `24c3f06a3aa68c0df21d9567983c59139d827c24ae1029bee4d7bf204eaff513` |
| Ticker intervals | `323af75c2d85b7ffa02bd57b511518e0a3cc773d5b2e687e6bcead37d2fcd282` |
| Formation receipt ledger identity | `538fe2c315ee78d7d5cacada73a1cc3db05e677c2f6ba8002238d2c555a6b664` |
| Candidate table | `5d5af42ccc51dfc2f5ee8e63e6b5e90efc103cb133d99bb2a206f0b74c19d3de` |
| Exclusion receipts | `d4a9df8ca9923304dbba6152f80e48bead690143189e79e619ede4a1e5f057d1` |

## Locked universe eligibility

| Measure | Value |
|---|---:|
| Candidates | 505 |
| Required universe size | 100 |
| Eligible candidates | 59 |
| Candidates below 95% exact-session completeness | 446 |
| Candidates also below the $5 median-price floor | 107 |
| Candidates also lacking positive median daily dollar volume | 107 |
| Maximum observed candidate completeness | 99.6032% |
| Median observed candidate completeness | 12.6984% |
| 75th percentile candidate completeness | 66.2698% |

The 59 qualifying symbols, in formation median-dollar-volume rank order, are:

`TSLA, AAPL, MSFT, NVDA, FB, AMD, BA, BAC, JPM, PYPL, V, DIS, MU, C, INTC, XOM, WFC, PFE, T, CVX, WMT, QCOM, F, JNJ, GM, AMAT, VZ, TWTR, GE, CSCO, ORCL, CCL, NKE, CMCSA, FCX, AAL, KO, UAL, BMY, SBUX, ATVI, VIAC, DAL, NEE, COP, OXY, NCLH, CVS, RTX, LUV, NEM, SLB, LVS, MO, MPC, DVN, DOW, MRO, KMI`.

This list is not a frozen paper universe because it does not contain the required 100 instruments.

## SPY formation coverage

SPY has 250 exact primary sessions out of 252 expected XNYS session dates. One standard session, 2021-05-05, contains 385 rows and is missing five expected minutes. The 2021-11-26 early close is excluded from the 390-minute primary corpus by design. Target-period SPY coverage and alignment are `NOT RUN`.

## Exclusions and unavailable audits

At the formation gate, the available exclusion codes are:

| Reason | Candidates |
|---|---:|
| `formation_session_completeness_below_95_percent` | 446 |
| `median_price_below_5` | 107 |
| `nonpositive_median_daily_dollar_volume` | 107 |

The following required quality sections are `NOT RUN` because a compliant universe could not be frozen:

- target coverage by year, fold, symbol, and month;
- target-session exclusion reason counts;
- target ticker-change, acquisition, delisting, termination, and symbol-reuse audit;
- split and reverse-split known-at/effective-at checks;
- point-in-time unit-invariance checks;
- SPY target-date alignment;
- fold train/validation/test counts and row/origin estimates;
- corpus-derived JEPA, embedding, LightGBM, GPU-hour, and TCA estimates;
- rolling-history, normalization, training-cutoff, and fold leakage audits.

## Resource evidence

The pre-request upper bound was 89,237,850 minute rows and approximately 17.85 GB for combined raw and processed storage, with more than 14 times that estimate available at planning time. The initial universe implementation attempted an all-corpus concatenation and reached 6.63 GiB of process memory. A protocol-preserving implementation correction replaced it with per-instrument monthly streaming. The successful full formation scan held the real Python worker near 200 MiB and reproduced the original eligibility statistics on a deterministic test fixture.

## Leakage and protocol audit

| Check | Result |
|---|---|
| Protocol freeze verified before empirical inspection | `PASS` |
| Exact XNYS validation before formation statistics | `PASS` |
| Missing minutes treated as zero activity | `PASS` — prohibited behavior was not used |
| Eligibility threshold changed after observing coverage | `PASS` — unchanged |
| Substitute names introduced | `PASS` — none |
| Test-period model output exposed | `PASS` — no model was trained |
| Target corpus leakage checks | `NOT RUN` |
| Corporate-action clock checks | `NOT RUN` |
| Fold membership and normalization checks | `NOT RUN` |

## Required human decision

This is a Class C data limitation under the frozen bug-fix policy. Continuing requires an explicit decision outside `sparse-jepa-v1`, such as approving a new protocol version with a defensible data representation or coverage rule. The current v1 run cannot proceed to target acquisition or the human training-approval gate without silently changing its locked universe criterion.

Status: `BLOCKED — INSUFFICIENT FORMATION UNIVERSE UNDER LOCKED EXACT-MINUTE CRITERION`
