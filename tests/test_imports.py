"""Smoke test: the package imports cleanly."""


def test_package_imports() -> None:
    import agentic_patterns

    assert agentic_patterns is not None
