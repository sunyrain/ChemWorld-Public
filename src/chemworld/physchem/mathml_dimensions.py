"""Dimension inference for bounded scalar MathML with explicitly typed symbols.

Numeric literals are dimensionless. Source-specific unit annotations and function
definitions require their own binding and are rejected here, not guessed.
"""

from math import isfinite
from numbers import Real

from chemworld.foundation.dimensions import DimensionVector
from chemworld.physchem.analytic_functions import SMOOTH_FUNCTIONS
from chemworld.physchem.mathml import MathExpression

_ONE = DimensionVector()
_BOOLEAN = object()
_COMPARISONS = {"eq", "neq", "lt", "leq", "gt", "geq"}


def mathml_dimension(expression, symbol_dimensions, *, constants=None):
    """Require a real scalar result; check every branch before numerical use."""
    if not isinstance(expression, MathExpression):
        raise ValueError("dimension inference requires a parsed MathExpression")
    if expression.number_units or expression.function_names:
        raise ValueError(
            "typed MathML requires bound parameters instead of unit literals/functions"
        )
    if any(not isinstance(value, DimensionVector) for value in symbol_dimensions.values()):
        raise ValueError("MathML symbols require the common physical dimension algebra")
    constants = {} if constants is None else dict(constants)
    if not constants.keys() <= symbol_dimensions.keys() or any(
        isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v)
        for v in constants.values()
    ):
        raise ValueError("fixed MathML parameters require finite declared real values")

    def same(values):
        if not values or any(value != values[0] for value in values):
            raise ValueError("MathML operands have incompatible dimensions")
        return values[0]

    def scalar(value):
        if not isinstance(value, DimensionVector):
            raise ValueError("Boolean MathML cannot be used as a physical scalar")
        return value

    def fixed(node):
        try:
            value = expression.evaluate(constants, subtree=node)
        except ValueError as exc:
            raise ValueError("dimensionful MathML powers require a fixed exponent") from exc
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("MathML degree must be a finite fixed real")
        return float(value)

    def visit(node):
        operation, args = node.operation, node.arguments
        if operation in {"number", "constant"}:
            if isinstance(node.value, bool):
                return _BOOLEAN
            if not isinstance(node.value, Real) or not isfinite(node.value):
                raise ValueError("nonfinite MathML literal is not a physical declaration")
            return _ONE
        if operation == "symbol":
            if node.value not in symbol_dimensions:
                raise ValueError("unknown typed MathML symbol " + str(node.value))
            return symbol_dimensions[node.value]
        if operation == "piecewise":
            outputs = []
            for branch in args:
                outputs.append(visit(branch.arguments[0]))
                if branch.operation == "piece" and visit(branch.arguments[1]) is not _BOOLEAN:
                    raise ValueError("typed MathML piece conditions must be Boolean")
            return same(outputs)
        values = [visit(argument) for argument in args]
        if operation in _COMPARISONS:
            same(values)
            return _BOOLEAN
        if operation in {"and", "or", "xor", "not", "implies"}:
            if any(value is not _BOOLEAN for value in values):
                raise ValueError("typed logical MathML requires Boolean operands")
            return _BOOLEAN
        if operation in {"plus", "minus", "min", "max"}:
            return scalar(same(values)) if values else _ONE
        if operation == "times":
            result = _ONE
            for value in values:
                result = result * scalar(value)
            return result
        if operation == "divide":
            return scalar(values[0]) / scalar(values[1])
        if operation == "power":
            base, exponent = map(scalar, values)
            if exponent != _ONE:
                raise ValueError("MathML exponent must be dimensionless")
            return _ONE if base == _ONE else base ** fixed(args[1])
        if operation == "root":
            if scalar(values[0]) != _ONE:
                raise ValueError("MathML root degree must be dimensionless")
            degree = fixed(args[0])
            if degree == 0:
                raise ValueError("MathML root degree cannot be zero")
            return scalar(values[1]) ** (1.0 / degree)
        if operation == "abs":
            return scalar(values[0])
        if operation in SMOOTH_FUNCTIONS | {"log", "floor", "ceiling", "factorial"}:
            if any(scalar(value) != _ONE for value in values):
                raise ValueError("this MathML function requires dimensionless arguments")
            return _ONE
        raise ValueError("unsupported typed MathML operation " + operation)

    return scalar(visit(expression._root))
