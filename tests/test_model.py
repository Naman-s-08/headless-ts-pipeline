"""Slow smoke tests for the real toxicity model.

Run explicitly with:  pytest -m slow
They are skipped by default so quick local/CI runs stay fast.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.model import ToxicityModel

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model() -> ToxicityModel:
    instance = ToxicityModel()
    instance.load()
    return instance


def test_clear_toxic_scores_high(model: ToxicityModel) -> None:
    label, score = model.predict("you are an idiot")
    assert label == "toxic"
    assert score > 0.8


def test_polite_scores_low(model: ToxicityModel) -> None:
    label, score = model.predict("have a nice day")
    assert label == "non_toxic"
    assert score < 0.2


def test_empty_batch_returns_empty_list(model: ToxicityModel) -> None:
    assert model.predict_batch([]) == []


def test_load_is_idempotent(model: ToxicityModel) -> None:
    pipeline = model._pipe
    model.load()
    assert model._pipe is pipeline


def test_long_text_logs_truncation_warning(
    model: ToxicityModel, caplog: pytest.LogCaptureFixture
) -> None:
    model.predict("word " * 600)
    assert "truncated" in caplog.text.lower()
