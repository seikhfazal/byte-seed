---
name: byteseed-release-reviewer
description: Review ByteSeed commit, tag, release, and public-update readiness across tests, regressions, benchmarks, documentation, artifacts, reproducibility, secrets, and Git hygiene. Use before accepting a release candidate or making public claims about model quality or performance.
---

# ByteSeed Release Reviewer

Assess whether a ByteSeed state is safe, accurately documented, reproducible, and ready for a commit, tag, release, or public update.

## Safety boundary

- Review by default; do not push, merge, tag, release, train, regenerate data, change checkpoints, or download artifacts.
- Inspect only trusted local ByteSeed checkpoint metadata and prefer CPU mapping.
- Preserve the stable manual educational implementation and backward-compatible defaults unless an approved change explicitly says otherwise.

## Review workflow

1. Confirm branch, clean scope, `git diff --check`, status, tracked files, ignored artifacts, and the complete diff.
2. Run compilation and deterministic conventional tests. Require CPU CI for portable invariants.
3. Report anchor-retention regression separately from held-out evaluation. Flag verbatim train/eval overlap and do not call retention generalization.
4. Review benchmark evidence for fixed seeds, explicit sampling, warm-up handling, device/dtype, CUDA synchronization, token counting, variance, peak memory, and machine-readable output.
5. Cross-check README, architecture, run instructions, training notes, limitations, release checklist, model size, vocabulary, checkpoint name, and performance claims.
6. Confirm checkpoint exclusions, tokenizer artifact policy, release manifest expectations, hashes/checksums, provenance, licenses, and documented public reproducibility limitations.
7. Run documented secret/personal-data scans and review each match in context.

## Release decision

Classify the state as `Ready`, `Ready with documented limitations`, or `Not ready`. A release is not ready when required tests fail, artifacts are accidentally tracked, claims exceed evidence, secrets remain unresolved, compatibility is unknown, or required public artifacts/manifests are absent without disclosure.

## Expected output

Provide the decision, checks run, anchor-retention and held-out results, benchmark evidence, documentation inconsistencies, artifact and secret findings, compatibility/reproducibility limitations, exact blocking files, required corrections, and final Git status. Do not perform the release.
