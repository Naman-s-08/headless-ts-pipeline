"""Tests for the ALLOW / REVIEW / BLOCK decision boundaries."""

import pytest

from app.policy import decide
from app.schemas import Action

ALLOW_T = 0.40
REVIEW_T = 0.80


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.00, Action.ALLOW),
        (0.39, Action.ALLOW),
        (0.40, Action.REVIEW),  # lower boundary is inclusive
        (0.80, Action.REVIEW),  # upper boundary is inclusive
        (0.81, Action.BLOCK),
        (1.00, Action.BLOCK),
    ],
)
def test_decide_boundaries(score: float, expected: Action) -> None:
    assert decide(score, ALLOW_T, REVIEW_T) is expected


@pytest.mark.parametrize(
    ("allow_t", "review_t"),
    [
        (-0.01, 0.80),
        (0.40, 1.01),
        (0.40, 0.40),  # equal thresholds are invalid
        (0.80, 0.40),  # reversed thresholds are invalid
    ],
)
def test_decide_rejects_invalid_thresholds(allow_t: float, review_t: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        decide(0.5, allow_t, review_t)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_decide_rejects_invalid_score(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        decide(score, ALLOW_T, REVIEW_T)
