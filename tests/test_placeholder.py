"""Placeholder test for Week 1 skeleton."""

def test_app_exists():
    """Just confirms the test suite is working."""
    assert True


def test_import_app():
    """Confirm we can import the FastAPI app without crashing."""
    from app.main import app
    assert app is not None
