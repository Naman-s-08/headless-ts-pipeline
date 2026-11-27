"""Fast check for the D4 scoring-rule comparison (no model needed)."""

import pytest

# scipy DLLs are blocked on the local machine; this test runs in Colab/CI.
try:
    import pandas as pd  # noqa: E402

    from scripts.evaluate import compare_scoring_rules  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"sklearn/scipy unavailable: {exc}", allow_module_level=True)


def test_compare_prefers_the_rule_with_better_pr_auc(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "score": [0.2, 0.1, 0.9, 0.3],
            "score_max6": [0.8, 0.1, 0.9, 0.3],
        }
    )
    scores_csv = tmp_path / "scores.csv"
    df.to_csv(scores_csv, index=False)

    payload = compare_scoring_rules(scores_csv, results_dir=tmp_path)

    assert payload["winner_by_pr_auc"] == "max6"
    assert payload["rules"]["max6"]["pr_auc"] > payload["rules"]["toxic"]["pr_auc"]
    assert (tmp_path / "d4_comparison.json").exists()
