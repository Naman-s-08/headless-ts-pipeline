"""Decision policy for the trust & safety pipeline.

A toxicity score alone is not a decision. This module turns a score into an
ALLOW / REVIEW / BLOCK action using two thresholds supplied by configuration.
"""

from app.schemas import Action

# Boundaries (kept identical in code, tests and README):
#   score <  allow_t               -> ALLOW
#   allow_t <= score <= review_t   -> REVIEW
#   score >  review_t              -> BLOCK


def decide(score: float, allow_t: float, review_t: float) -> Action:
    """Return the moderation action for ``score``.

    Args:
        score: Toxicity score in [0, 1].
        allow_t: Scores below this are automatically allowed.
        review_t: Scores above this are automatically blocked.

    Raises:
        ValueError: If thresholds are not ``0 <= allow_t < review_t <= 1``
            or if ``score`` is outside ``[0, 1]``.

    The REVIEW band is inclusive at both ends, so a score exactly on a
    threshold is always sent to a human rather than auto-allowed/blocked.
    """
    if not (0.0 <= allow_t < review_t <= 1.0):
        raise ValueError("thresholds must satisfy 0 <= allow_t < review_t <= 1")
    if not (0.0 <= score <= 1.0):
        raise ValueError("score must be in [0, 1]")

    if score < allow_t:
        return Action.ALLOW
    if score <= review_t:
        return Action.REVIEW
    return Action.BLOCK
