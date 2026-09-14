"""The tool set shared by every variant. Plain functions, no framework types."""

import ast
import operator
from datetime import date

MAX_EXPONENT = 64

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent above {MAX_EXPONENT} is refused.")
        return _BINARY[type(node.op)](left, right)
    raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")


def today() -> str:
    """Return today's date in ISO format. Takes no argument."""
    return date.today().isoformat()


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(4871 * 209) - 17'.

    Supports + - * / // % ** on numbers. No names, no calls, no attributes.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"'{expression}' is not a valid expression: {exc.msg}") from exc
    return str(_evaluate(tree.body))


TOOLS = (today, calculate)
