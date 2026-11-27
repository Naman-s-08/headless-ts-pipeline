"""Shared schemas / enums used across the pipeline.

The moderation action is the system's decision, separate from the model's
informational label (toxic / non_toxic). See docs/decisions.md once agreed.
"""

from enum import Enum


class Action(str, Enum):
    """Outcome of applying the moderation policy to a toxicity score."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
