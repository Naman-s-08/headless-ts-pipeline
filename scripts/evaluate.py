"""Week 3 evaluation starter.

Two commands, deliberately separated:

  score     slow: load the model once, write scores to CSV
  evaluate  fast: re-compute metrics/plots from the cached scores CSV

Usage:
    python scripts/evaluate.py score --input data/processed/val.csv --scores results/val_scores.csv
    python scripts/evaluate.py evaluate --scores results/val_scores.csv

The evaluation numbers use `allow_t` as the binary operating threshold:
score < allow_t -> non_toxic, score >= allow_t -> toxic. Actions come from
app.policy.decide() and are reported separately (including REVIEW workload).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in Docker/CI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from app.model import REVISION  # noqa: E402  (no torch import at module load)
from app.policy import decide  # noqa: E402

MODEL_ID = "unitary/toxic-bert"
DEFAULT_ALLOW = 0.40
DEFAULT_REVIEW = 0.80
SEED = 42


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def score_csv(
    input_csv: Path,
    scores_csv: Path,
    text_col: str = "comment_text",
    label_col: str = "label",
    model_id: str = MODEL_ID,
    batch_size: int = 16,
    max_length: int = 512,
) -> Path:
    """Run inference once and cache (id, label, score) to CSV."""
    from app.model import ToxicityModel  # lazy: metrics-only runs need no torch

    df = pd.read_csv(input_csv)
    texts = df[text_col].astype(str).tolist()

    # Sort by length so batches waste less padding; restore original order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    predictions: list[tuple[str, float] | None] = [None] * len(texts)

    model = ToxicityModel(model_id=model_id, max_length=max_length, batch_size=batch_size)
    model.load()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = [texts[i] for i in indices]
        for index, prediction in zip(indices, model.predict_batch(batch)):
            predictions[index] = prediction

    columns = ["id", label_col] if "id" in df.columns else [label_col]
    out = df[columns].copy()
    out["score"] = [prediction[1] for prediction in predictions]
    out["predicted_label"] = [prediction[0] for prediction in predictions]

    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(scores_csv, index=False)
    print(f"Wrote {len(out)} scores to {scores_csv}")
    return scores_csv


def evaluate_scores(
    scores_csv: Path,
    results_dir: Path = Path("results"),
    allow_t: float = DEFAULT_ALLOW,
    review_t: float = DEFAULT_REVIEW,
    model_id: str = MODEL_ID,
    revision: str | None = None,
) -> dict:
    """Compute metrics and plots from a cached scores CSV."""
    df = pd.read_csv(scores_csv)
    y_true = df["label"].astype(int)
    scores = df["score"].astype(float)
    y_pred = (scores >= allow_t).astype(int)
    actions = [decide(score, allow_t, review_t).value for score in scores]

    report = classification_report(
        y_true,
        y_pred,
        target_names=["non_toxic", "toxic"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    action_counts = {name: actions.count(name) for name in ("ALLOW", "REVIEW", "BLOCK")}

    results_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["non_toxic", "toxic"]
    ).plot(ax=ax, cmap="Blues", values_format="d")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name=model_id)
    ax.axhline(y_true.mean(), linestyle="--", color="grey", label="baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    payload = {
        "metrics": {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "action_counts": action_counts,
            "review_fraction": action_counts["REVIEW"] / len(actions),
            "binary_operating_threshold": allow_t,
            "n_rows": len(df),
        },
        "metadata": {
            "model_id": model_id,
            "revision": revision,
            "thresholds": {"allow": allow_t, "review": review_t},
            "seed": SEED,
            "scores_file": str(scores_csv),
            "git_hash": _git_hash(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "versions": {
                name: _version(name)
                for name in ("torch", "transformers", "pandas", "scikit-learn", "matplotlib")
            },
        },
    }

    (results_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Wrote metrics.json and plots to {results_dir}")
    print(f"PR-AUC={payload['metrics']['pr_auc']:.4f} "
          f"ROC-AUC={payload['metrics']['roc_auc']:.4f} "
          f"REVIEW={payload['metrics']['review_fraction']:.1%}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score", help="run inference and cache scores")
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--scores", type=Path, default=Path("results/val_scores.csv"))
    score_parser.add_argument("--model-id", default=MODEL_ID)
    score_parser.add_argument("--batch-size", type=int, default=16)
    score_parser.add_argument("--max-length", type=int, default=512)

    eval_parser = sub.add_parser("evaluate", help="compute metrics from cached scores")
    eval_parser.add_argument("--scores", type=Path, required=True)
    eval_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    eval_parser.add_argument("--allow", type=float, default=DEFAULT_ALLOW)
    eval_parser.add_argument("--review", type=float, default=DEFAULT_REVIEW)
    eval_parser.add_argument("--model-id", default=MODEL_ID)
    eval_parser.add_argument("--revision", default=REVISION)

    args = parser.parse_args()
    if args.command == "score":
        score_csv(
            args.input,
            args.scores,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        evaluate_scores(
            args.scores,
            args.results_dir,
            allow_t=args.allow,
            review_t=args.review,
            model_id=args.model_id,
            revision=args.revision,
        )


if __name__ == "__main__":
    main()
