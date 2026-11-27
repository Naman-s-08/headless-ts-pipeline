# Model findings (Person B)

Evidence from running `unitary/toxic-bert` by hand in Google Colab on 18 Nov 2026,
before writing production code (playbook rule: run the model first). Additional
timing/offline/token measurements were taken on 27 Nov 2026 after the baseline run.

## Setup

- Model: `unitary/toxic-bert`
- Download: `model.safetensors` 438 MB (Hugging Face Hub)
- Warning seen: unauthenticated HF requests (harmless; only affects rate limits)
- Device: Colab default (CPU unless a GPU runtime is selected)
- `top_k=None`, `truncation=True`, `max_length=512`

Measured on Colab CPU with the pinned revision (`results/evidence.json`):

- Model load (cached weights, includes warm-up call): **2.3 s**. The very first
  download of `model.safetensors` took ~2 s for 438 MB on Colab's network.
- Inference on 100 val rows, `batch_size=16`, `max_length=512`: **88.98 s**
  (~**890 ms/row**, ~**12.7 s per batch of 16**). This is why the 3,000-row
  baseline takes ~10-15 min on CPU; a T4 GPU is recommended for full runs.

## Output format

- Six independent probabilities with labels:
  `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Scores are sigmoid outputs; they **do not sum to 1** (multi-label, not multi-class).
- Single string vs list input have different nesting; `predict_batch()` normalises both.
- We read the `toxic` score **by label name**, not by position.

For `you are an idiot` (single `top_k=None` call, pinned revision):

| Label | Score |
|---|---:|
| toxic | 0.986101 |
| insult | 0.959979 |
| obscene | 0.756419 |
| severe_toxic | 0.045554 |
| identity_hate | 0.014150 |
| threat | 0.001331 |
| **Sum** | **2.763533** |

The sum is ~2.76, not 1 — direct evidence that these are independent multi-label
probabilities. `insult` and `obscene` are also high, which supports testing the
max-of-6 scoring rule in D4.

## Smoke results

| Text | toxic score | label | action (0.40 / 0.80) |
|---|---:|---|---|
| `you are an idiot` | 0.986 | toxic | BLOCK |
| `have a nice day` | 0.003 | non_toxic | ALLOW |
| `I will kill this bug` | 0.919 | toxic | BLOCK |
| `` (empty string) | 0.002 | non_toxic | ALLOW |
| `HELLO    WORLD 😂` | 0.001 | non_toxic | ALLOW |
| `tu bahut bekaar hai` | 0.055 | non_toxic | ALLOW |

## Findings / limitations

1. **False positive:** `I will kill this bug` scores 0.919 and would be BLOCKed.
   The model reads "kill" literally and misses the benign software idiom.
   -> Use in error analysis (20 FP) and the limitations section.
2. **Code-mixed text:** Hinglish `tu bahut bekaar hai` scores only 0.055 and is
   ALLOWed. Confirms the model is English-only.
   -> Measured evidence for the Hinglish limitation; multilingual model is future work.
3. Clear toxic/polite separation is strong (0.986 vs 0.003), so the model is
   usable as a baseline despite the limitations above.

## Policy boundary check

`decide(score, allow_t=0.40, review_t=0.80)` produced:

```
decide(0.00) -> ALLOW
decide(0.39) -> ALLOW
decide(0.40) -> REVIEW   # lower boundary inclusive
decide(0.80) -> REVIEW   # upper boundary inclusive
decide(0.81) -> BLOCK
decide(1.00) -> BLOCK
```

## Week 3 baseline — val set

Source: `data/processed/val.csv` — 3,000 rows, toxic 9.77%, seed 42.
Split verified reproducible (identical SHA256 hashes on two runs).
Scored in Colab with:

```
!PYTHONPATH=. python scripts/evaluate.py score --input data/processed/val.csv --scores results/val_scores.csv
!PYTHONPATH=. python scripts/evaluate.py evaluate --scores results/val_scores.csv --allow 0.40 --review 0.80
```

| Metric | Value |
|---|---|
| Precision (toxic) | 0.5150 |
| Recall (toxic) | 0.9352 |
| F1 (toxic) | 0.6642 |
| Macro F1 | 0.8054 |
| Weighted F1 | 0.9189 |
| Accuracy | 0.9077 |
| PR-AUC (average precision) | 0.8108 |
| ROC-AUC | 0.9730 |
| ALLOW count | 2,468 (82.27%) |
| REVIEW count | 194 (6.47%) |
| BLOCK count | 338 (11.27%) |
| REVIEW fraction (human workload) | 6.47% |
| Confusion matrix (TN, FP, FN, TP) | 2,449 / 258 / 19 / 274 |

Artifacts: `results/metrics.json`, `results/confusion_matrix.png`, `results/pr_curve.png`
(downloaded from Colab; `metrics.json` used for this write-up).

### Interpretation

- PR-AUC 0.8108 is ~8.3x the 0.0977 random baseline — ranking works well.
- At the 0.40 operating threshold recall is high (0.9352) but precision is only
  0.5150: 258 false positives vs 274 true positives. The `I will kill this bug`
  idiom above is a likely contributor. Week 4 should compare raising the BLOCK
  threshold with moving the 0.40-0.80 band into REVIEW instead of auto-BLOCK.
- Only 19 toxic comments were missed (FN), so recall is not the weak point here.
- REVIEW workload is 6.47% of 3,000 = 194 comments — a realistic human queue.

### Run metadata (from `results/metrics.json`)

- `model_id`: `unitary/toxic-bert`; `revision` in this run: not pinned (`null`),
  but the weights were unchanged since 2024-03-13. We now pin that exact commit
  (`4d6c22e74ba2fdd26bc4f7238f50766b045a0d94`) in `app/model.py`, so future runs
  record and load the same revision.
- Thresholds: allow 0.40 / review 0.80; seed 42; 3,000 rows.
- Colab git commit: `55cf9e511bda235c8bb0b7c651f08c3152aef1e2` (develop at run time).
- Python 3.13.18; torch 2.14.1+cpu; transformers 5.22.1; pandas 2.2.3;
  scikit-learn 1.6.1; matplotlib 3.10.0.
- Token-length audit of all 3,000 val rows: **68 rows (2.27%) exceed 512 tokens**,
  longest is 2,351 tokens. The 588-token comment seen in the scoring log is one
  of these; all are truncated to `max_length=512` by the wrapper.

> This is the "before" number. Week 4 tunes thresholds on val and touches test exactly once.

## Offline reload check

With `HF_HUB_OFFLINE=1` set in Colab, a fresh `pipeline(...)` loaded from the local
HF cache in **0.35 s** and scored a sample sentence successfully (`OK`). So after
the first download, the model works with no network access — important for the
headless/offline deployment path.

## Next steps

- Confirm with A/C that pinning `REVISION` in `app/model.py` is acceptable.
- **Week 4:** settle D4 (`toxic` score vs max-of-6) using val PR-AUC/recall,
  then tune thresholds on val and touch `test.csv` exactly once.
