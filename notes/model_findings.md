# Model findings (Person B)

Evidence from running `unitary/toxic-bert` by hand in Google Colab on 18 Nov 2026,
before writing production code (playbook rule: run the model first).

## Setup

- Model: `unitary/toxic-bert`
- Download: `model.safetensors` 438 MB (Hugging Face Hub)
- Warning seen: unauthenticated HF requests (harmless; only affects rate limits)
- Device: Colab default (CPU unless a GPU runtime is selected)
- `top_k=None`, `truncation=True`, `max_length=512`

> TODO: record exact load time and per-batch inference time in Colab.

## Output format

- Six independent probabilities with labels:
  `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Scores are sigmoid outputs; they **do not sum to 1** (multi-label, not multi-class).
- Single string vs list input have different nesting; `predict_batch()` normalises both.
- We read the `toxic` score **by label name**, not by position.

> TODO: paste the 6 scores and their sum from the extra Colab check.

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

## Week 3 baseline — val set (fill from Colab)

Source: `data/processed/val.csv` — 3,000 rows, toxic 9.77%, seed 42.
Split verified reproducible (identical SHA256 hashes on two runs).
Command: `python scripts/evaluate.py evaluate --scores results/val_scores.csv --allow 0.40 --review 0.80`

| Metric | Value |
|---|---|
| Precision (toxic) | ___ |
| Recall (toxic) | ___ |
| F1 (toxic) | ___ |
| Macro F1 | ___ |
| Weighted F1 | ___ |
| PR-AUC (average precision) | ___ |
| ROC-AUC | ___ |
| ALLOW count | ___ |
| REVIEW count | ___ |
| BLOCK count | ___ |
| REVIEW fraction (human workload) | ___% |
| Confusion matrix (TN, FP, FN, TP) | ___ |

Artifacts: `results/metrics.json`, `results/confusion_matrix.png`, `results/pr_curve.png`.

> This is the "before" number. Week 4 tunes thresholds on val and touches test exactly once.

## Next steps

- Pin the model `revision` (commit hash) from the model card in `app/model.py`.
- Fill the two TODOs above (6 scores + sum; load/inference time).
- Record the `HF_HUB_OFFLINE=1` reload result.
- Fill the Week 3 baseline table with the numbers from Colab.
