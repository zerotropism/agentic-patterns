"""The tool set is shared by the three variants, so its behaviour must be exact."""

import pytest

from agentic_patterns.tools import MAX_EXPONENT, calculate, today


def test_today_is_iso_formatted() -> None:
    from datetime import date

    assert today() == date.today().isoformat()


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("(4871 * 209) - 17", "1018022"),
        ("365 * 3", "1095"),
        ("2 ** 10", "1024"),
        ("-4 + 2.5", "-1.5"),
        ("7 // 2", "3"),
    ],
)
def test_arithmetic_is_exact(expression: str, expected: str) -> None:
    """Unlike an LLM doing the arithmetic, this is deterministic."""
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "[].__class__",
        "x + 1",
    ],
)
def test_anything_beyond_arithmetic_is_refused(expression: str) -> None:
    """This replaces PythonREPLTool, which executed whatever the model produced."""
    with pytest.raises(ValueError, match="Unsupported"):
        calculate(expression)


def test_huge_exponents_are_refused() -> None:
    with pytest.raises(ValueError, match=str(MAX_EXPONENT)):
        calculate("9 ** 9 ** 9")


def test_malformed_expressions_are_reported() -> None:
    with pytest.raises(ValueError, match="not a valid expression"):
        calculate("1 +")


def test_docstrings_exist_for_every_tool() -> None:
    """Docstrings become the tool descriptions sent to the model in all three variants."""
    from agentic_patterns.tools import TOOLS

    assert all(tool.__doc__ for tool in TOOLS)
