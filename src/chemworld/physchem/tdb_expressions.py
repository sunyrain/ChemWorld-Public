"""A finite declarative TDB expression interpreter; no eval or upstream imports."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from math import isfinite
from typing import ClassVar

from chemworld.physchem.caloric_math import exponential, logarithm
from chemworld.physchem.caloric_math import scalar_value as derivative_scalar_value

# The SGTE/TDB convention used by the selected pycalphad reference. This does
# not replace the gas constant of another declared native thermodynamic model.
TDB_GAS_CONSTANT_J_MOL_K = 8.3145
_FUNCTIONS = {"LN": logarithm, "LOG": logarithm, "EXP": exponential}
_BINARY = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow}


def scalar_value(value):
    """Extract only the primal for domain/branch checks, retaining jets elsewhere."""
    value = derivative_scalar_value(value)
    if isinstance(value, bool):
        raise ValueError("TDB scalars cannot be boolean")
    return float(value)


@dataclass(frozen=True)
class TdbExpression:
    _functions: ClassVar[dict] = _FUNCTIONS
    text: str
    symbols: tuple[str, ...] = field(init=False)
    _tree: ast.AST = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 16384:
            raise ValueError("TDB expression requires 1-16384 characters")
        text = self.text.upper().replace("#", "").replace("^", "**").strip()
        try:
            tree = ast.parse(text, mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise ValueError("invalid TDB arithmetic expression") from exc
        names = set()
        nodes = list(ast.walk(tree))
        if len(nodes) > 1024:
            raise ValueError("TDB expression exceeds 1024 syntax nodes")

        def validate(node, depth=0):
            if depth > 64:
                raise ValueError("TDB expression nesting exceeds 64")
            if isinstance(node, ast.Expression):
                validate(node.body, depth + 1)
            elif isinstance(node, ast.Constant):
                if type(node.value) not in (int, float):
                    raise ValueError("TDB expressions accept only finite real constants")
                try:
                    finite = isfinite(node.value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("TDB expressions accept only finite real constants")
            elif isinstance(node, ast.Name):
                if (not re.fullmatch(r"[A-Z][A-Z0-9_]*", node.id)
                        or node.id in {"TRUE", "FALSE", "NONE", "NAN", "INF"}):
                    raise ValueError("invalid TDB symbol")
                names.add(node.id)
            elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                validate(node.operand, depth + 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                validate(node.left, depth + 1)
                validate(node.right, depth + 1)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in self._functions and len(node.args) == 1
                  and not node.keywords):
                validate(node.args[0], depth + 1)
            else:
                raise ValueError("unsupported TDB expression syntax or function")

        validate(tree)
        object.__setattr__(self, "symbols", tuple(sorted(names)))
        object.__setattr__(self, "_tree", tree.body)

    def evaluate(self, values):
        """Interpret arithmetic over real scalars or the common analytic jets."""
        def visit(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                try:
                    return values[node.id]
                except KeyError as exc:
                    raise ValueError(f"unresolved TDB symbol {node.id}") from exc
            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.Call):
                return self._functions[node.func.id](visit(node.args[0]))
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(right, (int, float)):
                return left**right
            return exponential(right * logarithm(left))

        try:
            result = visit(self._tree)
            if isinstance(result, complex) or not isfinite(scalar_value(result)):
                raise ValueError("nonfinite or nonreal TDB expression result")
            return result
        except (ArithmeticError, TypeError) as exc:
            raise ValueError(f"TDB arithmetic is outside its real domain: {self.text}") from exc
