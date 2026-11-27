# Decisions log — headless-ts-pipeline

Owner: Person B (ML & Evaluation Lead)
Status for all items below: **Proposed** — awaiting agreement with A and C.
Date: 24 Nov 2026 (catching up Week 1)

These proposals answer the open issues in the Person B playbook. Once agreed,
change the status to **Accepted** and record the date. Any later change to
`predict()` or `decide()` signatures must go through a PR that updates this file.

---

## D1 — Data pool for val/test (playbook issue #1)

**Problem.** `unitary/toxic-bert` was trained on the Jigsaw train data. Splitting
our val/test from `train.csv` would leak examples the model has already seen and
make F1 look better than reality.

**Proposal.** Build val and test from the **official `test.csv` + `test_labels.csv`**.
Drop rows where any of the 6 labels is `-1` (not scored). Roughly 64k rows remain.

**Why.** Avoids contamination; gives an honest estimate.

**Open for A/C.** Confirm we accept this trade-off (smaller pool, but unbiased).

---

## D2 — Split sizes and caching (playbook issue #3)

**Problem.** 1,000 val rows at ~10% positives is only ~100 positives, so a single
example moves recall by ~1%. Re-running inference for each experiment is slow.

**Proposal.**
- Stratified split: **3,000 val / 3,000 test**, fixed `seed = 42`.
- Save `data/processed/val.csv` and `data/processed/test.csv` with columns
  `id, comment_text, label` plus the 6 raw labels.
- Cache model scores once to `results/val_scores.csv` and `results/test_scores.csv`
  (columns: `id, label, score_raw, score_clean` and all 6 scores).
- Report 95% bootstrap confidence intervals for recall, precision and F1.

**Why.** Stable thresholds, fast re-runnable metrics, honest uncertainty.

---

## D3 — Label semantics in `predict()` (playbook issue #5)

**Problem.** The contract says `predict()` returns `(label, score)` but never
defines `label`. A score of 0.45 could produce `label = non_toxic` while
`decide()` returns `REVIEW` — confusing but intentional.

**Proposal.**
- `label` is **informational only**: `toxic` if `score >= 0.5`, else `non_toxic`.
- `action` is the real decision, produced by `app.policy.decide(score, allow_t, review_t)`.
- Document this in the README so nobody treats `label` as the moderation result.

**Open for A.** Confirm the API response exposes both fields clearly.

---

## D4 — Scoring rule: `toxic` vs max-of-6 (playbook issue #2)

**Problem.** Binary ground truth is `1` if **any** of the 6 Jigsaw columns is 1,
but the model wrapper currently reads only the `toxic` output. Rows flagged only
as insult/obscene/threat/identity_hate may get a low `toxic` score.

**Proposal.** On val, compare two scoring rules:
1. `toxic` score only (current default).
2. `max` of all 6 scores.

Pick the winner by **PR-AUC and recall**, then record it here. `predict()` still
returns one float, so whichever rule wins becomes that float.

**Owner.** B, Week 4.

---

## D5 — Threshold validation (playbook issue #6)

**Problem.** If someone sets `allow >= review` in `.env`, `decide()` silently
returns the wrong action.

**Proposal.** `decide()` raises `ValueError` unless `0 <= allow_t < review_t <= 1`
and `0 <= score <= 1` (already implemented in `app/policy.py`).
**A: mirror the same check as a Pydantic `Settings` validator.**

---

## Decision status

| ID | Proposal | Status | Agreed by | Date |
|----|----------|--------|-----------|------|
| D1 | Use official test_labels pool | Proposed | — | — |
| D2 | 3k/3k split + score caching + CIs | Proposed | — | — |
| D3 | label informational, action from policy | Proposed | — | — |
| D4 | toxic vs max-of-6 experiment | Proposed | — | — |
| D5 | Threshold validation in policy + Settings | Proposed | — | — |
