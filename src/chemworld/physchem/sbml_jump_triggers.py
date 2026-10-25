"""Exact affine source-time boundaries between original stochastic jumps.

State-only Boolean guards are observed at each original reaction/tau boundary.
Time guards must prove affine comparison operands; no endpoint polling guesses
at a crossing. Numerical ties use the common coordinate queue's eight-ulp scale.
"""

from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite, ulp

from chemworld.physchem.mathml import _COMPARISONS, MathExpression


def tolerance(left, right):
    return 8 * max(ulp(left), ulp(right))


def _degree(node):
    if "$time" not in MathExpression._symbols(node):
        return 0
    if node.operation == "symbol" and node.value == "$time":
        return 1
    if node.operation in {"plus", "minus"}:
        return max((_degree(a) for a in node.arguments), default=0)
    if node.operation == "times":
        degree = sum(_degree(a) for a in node.arguments)
        if degree <= 1:
            return degree
    if node.operation == "divide" and _degree(node.arguments[1]) == 0:
        return _degree(node.arguments[0])
    if (node.operation == "power" and node.arguments[1].operation == "number"
            and node.arguments[1].value in (0.0, 1.0)):
        return 0 if node.arguments[1].value == 0 else _degree(node.arguments[0])
    raise ValueError("integer source-time guards require provably affine comparison operands")


def _linear(node, expression, scope, functions):
    if _degree(node) == 0:
        value = expression.evaluate(scope, functions=functions, subtree=node)
        return 0.0, float(value)
    if node.operation == "symbol":
        return 1.0, 0.0
    terms = [_linear(a, expression, scope, functions) for a in node.arguments]
    if node.operation == "plus":
        return sum(a for a, _ in terms), sum(b for _, b in terms)
    if node.operation == "minus":
        return (
            (-terms[0][0], -terms[0][1])
            if len(terms) == 1
            else (terms[0][0] - terms[1][0], terms[0][1] - terms[1][1])
        )
    if node.operation == "times":
        slope, intercept = 0.0, 1.0
        for a, b in terms:
            slope, intercept = slope * b + intercept * a, intercept * b
        return slope, intercept
    if node.operation == "divide":
        return terms[0][0] / terms[1][1], terms[0][1] / terms[1][1]
    if node.operation == "power":
        return terms[0]
    raise ValueError("uncompiled affine integer time expression")


@dataclass(frozen=True)
class SBMLJumpTriggers:
    frame: object
    rules: object
    _guards: tuple = field(init=False, repr=False)
    roots: tuple = ()

    def __post_init__(self):
        guards = []
        for index, event in enumerate(self.frame.model.declaration.items("listOfEvents")):
            expression = event.of("trigger")[0].mathematics
            node = self.rules.expand(expression._root)
            comparisons = []

            def boolean(current, comparisons=comparisons):
                if current.operation in _COMPARISONS:
                    for left, right in pairwise(current.arguments):
                        degree = max(_degree(left), _degree(right))
                        if degree and current.operation in {"eq", "neq"}:
                            raise ValueError(
                                "time equality pulses need an explicit pulse event profile"
                            )
                        comparisons.append((current, left, right, degree))
                elif current.operation in {"and", "or", "xor", "not", "implies"}:
                    for argument in current.arguments:
                        boolean(argument)
                elif current.operation != "constant" or type(current.value) is not bool:
                    raise ValueError("integer event guards require declared Boolean comparisons")

            boolean(node)
            guards.append(
                (event.get("id", f"@event:{index}"), expression, node, tuple(comparisons))
            )
        object.__setattr__(self, "_guards", tuple(guards))

    def _pairs(self, state):
        scope = self.frame.context(state.coordinates, state.time_s)
        for identity, expression, node, comparisons in self._guards:
            pairs = []
            for comparison, left, right, degree in comparisons:
                if degree:
                    a, b = _linear(left, expression, scope, self.frame._functions)
                    c, d = _linear(right, expression, scope, self.frame._functions)
                    slope, intercept = a - c, b - d
                    if not isfinite(slope) or not isfinite(intercept):
                        raise ValueError("integer guard affine coefficients must be finite")
                    crossing = (
                        None
                        if slope == 0
                        else (-intercept / slope - self.frame.initial_model_time)
                        * self.frame._time_scale
                    )
                    if crossing is not None and not isfinite(crossing):
                        raise ValueError("integer guard boundary exceeds finite native time")
                    value = slope * scope["$time"] + intercept
                    at_boundary = crossing is not None and abs(
                        state.time_s - crossing
                    ) <= tolerance(state.time_s, crossing)
                    if at_boundary:
                        truth = slope > 0 if comparison.operation in {"gt", "geq"} else slope < 0
                    else:
                        truth = _COMPARISONS[comparison.operation](value, 0.0)
                else:
                    crossing = None
                    values = [
                        expression.evaluate(scope, functions=self.frame._functions, subtree=n)
                        for n in (left, right)
                    ]
                    truth = _COMPARISONS[comparison.operation](*values)
                pairs.append((comparison, bool(truth), crossing))
            yield identity, expression, node, pairs, scope

    def trigger_values(self, state, _signs=None):
        result = {}
        for identity, expression, node, pairs, scope in self._pairs(state):
            overrides = {}
            for comparison, truth, _ in pairs:
                overrides[comparison] = overrides.get(comparison, True) and truth
            result[identity] = expression.evaluate(
                scope, functions=self.frame._functions, subtree=node, comparison_values=overrides
            )
        return result

    def next_boundary(self, state):
        future = [
            crossing
            for _, _, _, pairs, _ in self._pairs(state)
            for _, _, crossing in pairs
            if crossing is not None and crossing - state.time_s > tolerance(crossing, state.time_s)
        ]
        return min(future, default=None)
