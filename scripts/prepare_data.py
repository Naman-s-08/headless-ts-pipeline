"""Build reproducible val/test splits from the official Jigsaw test set.

Decision D1: use `test.csv` + `test_labels.csv`, not `train.csv`, because
toxic-bert was trained on the Jigsaw train data (contamination risk).

Decision D2: stratified 3,000 val / 3,000 test, seed 42. Outputs are written to
`data/processed/` so the raw downloads are never overwritten.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 3_000
TEST_SIZE = 3_000
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
OUTPUT_COLUMNS = ["id", "comment_text", "label", *LABELS]


def build_splits(
    raw_dir: Path,
    out_dir: Path,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (val, test) DataFrames and save them as CSV files."""
    comments = pd.read_csv(raw_dir / "test.csv")
    labels = pd.read_csv(raw_dir / "test_labels.csv")

    df = comments.merge(labels, on="id", how="inner")
    unscored = (df[LABELS] == -1).any(axis=1)
    dropped = int(unscored.sum())
    df = df.loc[~unscored].copy()
    df["label"] = (df[LABELS].max(axis=1) > 0).astype(int)

    print(f"Dropped {dropped} unscored rows (label = -1).")
    print(f"Scored pool: {len(df)} rows, toxic {df['label'].mean():.2%}")

    val, rest = train_test_split(
        df,
        train_size=val_size,
        stratify=df["label"],
        random_state=seed,
    )
    test, _ = train_test_split(
        rest,
        train_size=test_size,
        stratify=rest["label"],
        random_state=seed,
    )

    val = val[OUTPUT_COLUMNS].reset_index(drop=True)
    test = test[OUTPUT_COLUMNS].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"val  -> {val_path}  ({len(val)} rows, toxic {val['label'].mean():.2%})")
    print(f"test -> {test_path} ({len(test)} rows, toxic {test['label'].mean():.2%})")
    return val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing test.csv and test_labels.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder to write val.csv and test.csv",
    )
    args = parser.parse_args()

    build_splits(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
