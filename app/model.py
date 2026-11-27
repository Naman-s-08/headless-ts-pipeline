"""Toxicity model wrapper for the headless trust & safety pipeline.

Load the model once (A calls this from the FastAPI lifespan), then score text
through predict() / predict_batch(). The model returns an informational label
and a score; app.policy.decide() turns that score into ALLOW/REVIEW/BLOCK.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

MODEL_ID = "unitary/toxic-bert"
# Pinned HF commit for MODEL_ID (last modified 2024-03-13). Keeps scoring
# reproducible even if the upstream model card is updated later.
REVISION = "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"
MAX_LENGTH = 512
BATCH_SIZE = 16


class ModelLoadError(RuntimeError):
    """Raised when the toxicity model cannot be loaded."""


class ToxicityModel:
    """Singleton wrapper around a Hugging Face multi-label toxicity pipeline."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        revision: str | None = REVISION,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.revision = revision
        self._pipe = None

    def load(self) -> None:
        """Load the pipeline once. Safe to call more than once."""
        if self._pipe is not None:
            return

        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            # Threads share CPU cores when A runs inference in a threadpool.
            torch.set_num_threads(2)

            self._pipe = pipeline(
                "text-classification",
                model=self.model_id,
                revision=self.revision,
                top_k=None,
                device=device,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot load model '{self.model_id}': {exc}"
            ) from exc

        self.predict("warm up")  # first inference is slow

    def _warn_if_too_long(self, text: str) -> None:
        token_ids = self._pipe.tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(token_ids) > self.max_length:
            log.warning(
                "Text has %d tokens > max_length %d; it will be truncated",
                len(token_ids),
                self.max_length,
            )

    @staticmethod
    def _toxic_from_mapping(scores: dict[str, float]) -> float:
        """Read the toxic probability by label name, never by list position."""
        try:
            return scores["toxic"]
        except KeyError as exc:
            raise ValueError(
                f"missing 'toxic' label in model output: {scores}"
            ) from exc

    @staticmethod
    def _toxic_score(scores: list[dict]) -> float:
        """Backward-compatible wrapper for raw pipeline output."""
        return ToxicityModel._toxic_from_mapping(
            {item["label"]: item["score"] for item in scores}
        )

    def predict_scores_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Return all six independent label probabilities per text (D4 input)."""
        if not texts:
            return []

        for text in texts:
            self._warn_if_too_long(text)

        outputs = self._pipe(
            texts,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )

        # A single string can come back flat; normalise to list-of-lists.
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        return [
            {item["label"]: float(item["score"]) for item in scores}
            for scores in outputs
        ]

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return (label, toxic score) for each text, preserving input order."""
        if not texts:
            return []

        results: list[tuple[str, float]] = []
        for scores in self.predict_scores_batch(texts):
            score = self._toxic_from_mapping(scores)
            results.append(("toxic" if score >= 0.5 else "non_toxic", score))
        return results

    def predict(self, text: str) -> tuple[str, float]:
        return self.predict_batch([text])[0]
