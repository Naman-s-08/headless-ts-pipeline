"""Fast checks for the pinned model configuration (no torch needed)."""

from app.model import MODEL_ID, REVISION, ToxicityModel


def test_revision_is_a_full_hf_commit_hash() -> None:
    assert len(REVISION) == 40
    assert set(REVISION) <= set("0123456789abcdef")


def test_wrapper_uses_the_pinned_revision_by_default() -> None:
    model = ToxicityModel()
    assert model.model_id == MODEL_ID
    assert model.revision == REVISION
