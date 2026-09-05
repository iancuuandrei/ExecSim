# Paper runtime authorization

This operational specification controls privileged sparse-JEPA paper stages without changing the six scientifically frozen YAML files or their canonical configuration hash. The scientific configuration remains immutable and defaults every privileged operation to denied.

## Approval receipt

A runtime approval is an untracked JSON file below the repository-local `.runtime/paper-approvals/` directory and supplied with `--runtime-approval`. Canonical path validation rejects files outside that root, non-JSON files, non-regular files, and files larger than 64 KiB. The directory is ignored by Git. An approval has this exact schema:

```json
{
  "schema_version": "paper-runtime-approval-v1",
  "approval_id": "operator-chosen-nonempty-id",
  "approved_at_utc": "2026-09-06T00:00:00Z",
  "protocol_id": "sparse-jepa-v2",
  "paper_config_sha256": "current canonical paper config hash",
  "approvals": {
    "target_acquisition": false,
    "historical_training": false,
    "locked_result_evaluation": false
  }
}
```

All three approval fields are required booleans. Unknown, absent, malformed, or identity-mismatched receipts fail closed. Approval files are operational credentials and must not be committed.

## Dual authorization

Each privileged operation requires both its approval field and its corresponding CLI flag:

| Operation | Approval field | CLI flag |
|---|---|---|
| Target acquisition and provider network access | `target_acquisition` | `--enable-network` |
| Historical JEPA or LightGBM fitting | `historical_training` | `--enable-historical-training` |
| Locked-test evaluation, historical TCA, and historical reporting | `locked_result_evaluation` | `--enable-full-paper-run` |

Neither half authorizes work alone. Scopes do not imply one another. A target-acquisition approval cannot authorize training, and a training approval cannot reveal locked results. Synthetic fixtures remain outside these historical privileges and must stay visibly classified as synthetic.

## Protocol-aware orchestration

`execsim ml paper run` resolves formation readiness by protocol. V1 uses its existing minute formation corpus. V2 reuses a compatible frozen universe when present; otherwise it uses the direct daily formation corpus to build that universe. V2 never reads the v1-only `formation_corpus_root` and never reintroduces exact-minute formation eligibility.

With no runtime approval and no CLI opt-in, the v2 runner may reuse the frozen universe but stops at `DATA NOT ACQUIRED` when the target corpus is absent. It performs no network request, historical fit, locked-test evaluation, or TCA.
