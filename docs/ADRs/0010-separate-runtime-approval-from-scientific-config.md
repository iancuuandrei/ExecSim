# ADR 0010: Separate runtime approval from scientific configuration

- Status: Accepted
- Date: 2026-09-06
- Owners: ExecSim maintainers
- Applies to: paper acquisition, historical fitting, and locked-result evaluation

## Context

The six paper YAML files are scientifically frozen and contribute to the canonical protocol hash. Their legacy `allow_*` fields are all false. Changing those fields to conduct an authorized operation would mutate the protocol identity and invalidate the v2 design freeze and formation evidence. The production `run` path also read the v1-only `formation_corpus_root` before checking the protocol, so the actual v2 configuration failed with `KeyError` instead of stopping at its target-data gate.

## Decision

Keep the six YAML files and design freeze unchanged. Represent operational permission in a separate, untracked JSON receipt bound to the exact `protocol_id` and canonical `paper_config_sha256`. Require a distinct boolean scope and the matching explicit CLI flag for target acquisition, historical training, and locked-result evaluation. Default to denial and reject malformed, incomplete, unknown-scope, or identity-mismatched receipts.

Dispatch formation readiness by protocol. V1 retains minute-corpus readiness. V2 reuses its compatible frozen universe or uses its direct daily formation artifact when a universe must be built. It never falls back to v1 exact-minute formation eligibility.

This refines the existing dual-authorization implementation direction without changing the scientific protocol or the direction itself, so the frozen implementation standard and paper specification are not rewritten. The operational schema is specified in [Paper runtime authorization](../PAPER_RUNTIME_AUTHORIZATION.md).

## Rejected alternatives

- Set `allow_network`, `allow_historical_training`, or `allow_full_paper_run` in frozen `data.yaml`. This would turn an operational act into a scientific configuration change.
- Treat the CLI flag as sufficient authorization. An accidental or copied command would cross the privileged boundary.
- Let one approval imply later stages. Acquisition, fitting, and locked-result reveal have distinct consequences and require independent intent.
- Read `formation_corpus_root` for v2 or restore that key. This would reintroduce the v1 formation contract and its exact-minute selection bias.

## Consequences

Operators must create a matching local receipt and pass its path together with the relevant CLI flag. Receipts become invalid when the protocol or canonical scientific configuration changes. Routine planning and unapproved `run` operations remain safe and can report the next unavailable stage. The v2 design freeze, six YAML files, frozen top-100 universe, and formation evidence remain byte-for-byte unchanged.

## Verification

Tests cover default denial, flag-only denial, approval-only denial, matching dual authorization, protocol/config mismatch, scope separation, real v2 CLI orchestration, and v1/v2 formation-readiness dispatch. No test performs a provider request or historical fit.
