"""Placeholder tests for Week 1 skeleton."""


def test_app_exists():
    """Just confirms the test suite is working."""
    assert True


def test_clean_text_basic():
    """Test that clean_text lowercases and removes extra spaces."""
    from app.preprocess import clean_text

    result = clean_text("HELLO   WORLD")
    assert result == "hello world"


def test_clean_text_removes_urls():
    """Test that URLs are removed from text."""
    from app.preprocess import clean_text

    result = clean_text("Check this https://example.com out")
    assert "https://" not in result
    assert result == "check this out"


def test_clean_text_removes_html():
    """Test that HTML tags are removed."""
    from app.preprocess import clean_text

    result = clean_text("Hello <b>world</b>")
    assert "<b>" not in result
    assert result == "hello world"


def test_clean_text_removes_emojis():
    """Test that emojis are removed."""
    from app.preprocess import clean_text

    result = clean_text("Hello 🔥🔥")
    assert "🔥" not in result
    assert result == "hello"


def test_clean_text_empty_input():
    """Test that empty input returns empty string."""
    from app.preprocess import clean_text

    result = clean_text("")
    assert result == ""
