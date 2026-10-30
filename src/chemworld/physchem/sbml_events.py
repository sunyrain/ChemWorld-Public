"""Pure SBML event adapters for the common atomic coordinate-event queue."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Any, cast

from chemworld.physchem.analytic_functions import SMOOTH_FUNCTIONS
from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.coordinate_events import CoordinateEventSpec
from chemworld.physchem.coordinate_hybrid import CoordinateRoot
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.mathml import _COMPARISONS, MathExpression
from chemworld.physchem.sbml_kinetics import SBMLKinetics


def sbml_coordinate_events(frame: SBMLKinetics) -> tuple[CoordinateEventSpec, ...]:
    """Compile declarations only; neither advance time nor own a live queue."""
    if not isinstance(frame, SBMLKinetics):
        raise ValueError("SBML event adapters require a compiled native quantity basis")

    def evaluate(expression: MathExpression, state: CoupledState) -> Any:
        return expression.evaluate(frame.context(state.coordinates, state.time_s),
                                   functions=frame._functions)

    def apply(state: CoupledState, payload: Mapping[str, float]) -> CoupledState:
        return replace(state, coordinates=frame.assigned_values(
            state.coordinates, state.time_s, payload))

    def expression_callback(expression: MathExpression) -> Callable[[CoupledState], Any]:
        def callback(state: CoupledState) -> Any:
            return evaluate(expression, state)

        return callback

    def delay_callback(expression: MathExpression) -> Callable[[CoupledState], float]:
        def callback(state: CoupledState) -> float:
            return evaluate(expression, state)*frame._time_scale

        return callback

    result = []
    for index, record in enumerate(frame.model.declaration.items("listOfEvents")):
        trigger = record.of("trigger")[0]
        assignments = {r.get("variable"): r.mathematics
                       for r in record.items("listOfEventAssignments")}
        delays, priorities = record.of("delay"), record.of("priority")

        def capture(
            state: CoupledState, expressions: Mapping[str, MathExpression] = assignments,
        ) -> dict[str, float]:
            scope = frame.context(state.coordinates, state.time_s)
            return {name: expression.evaluate(scope, functions=frame._functions)
                    for name, expression in expressions.items()}

        result.append(CoordinateEventSpec(record.get("id", f"@event:{index}"),
            tuple(assignments),
            trigger=expression_callback(cast(MathExpression, trigger.mathematics)),
            capture=capture, apply=apply,
            delay_s=(lambda _: 0.) if not delays else
                delay_callback(cast(MathExpression, delays[0].mathematics)),
            priority=None if not priorities else
                expression_callback(cast(MathExpression, priorities[0].mathematics)),
            initial_value=trigger.get("initialValue"), persistent=trigger.get("persistent"),
            use_trigger_values=record.get("useValuesFromTriggerTime")))
    return tuple(result)


@dataclass(frozen=True)
class SBMLTriggerRoots:
    """Immutable guard configuration for the shared transverse-root coordinator.

    Boolean combinations of continuous inequalities have outgoing root views.
    Point equalities, Boolean function bodies and discontinuous scalar guard
    expressions are explicitly outside this execution profile. They are never
    silently sampled at the endpoints of a long wait.
    """

    frame: SBMLKinetics
    roots: tuple
    _guards: tuple

    def __init__(self, frame):
        if not isinstance(frame, SBMLKinetics):
            raise ValueError("SBML roots require a compiled native quantity basis")
        object.__setattr__(self, "frame", frame)
        roots, shared, guards = [], {}, []
        smooth = SMOOTH_FUNCTIONS | {"number", "constant", "symbol", "plus", "times",
                                    "minus", "divide", "power", "root", "log", "call"}

        def scalar(node, locals_=frozenset(), visited=frozenset()):
            if node.operation not in smooth:
                raise ValueError("unsupported nonsmooth SBML event guard operand "+node.operation)
            dynamic = False
            if node.operation == "symbol" and node.value not in locals_:
                name = node.value
                if name == "$time":
                    return True
                if name not in visited:
                    recipe = frame._recipes[name]
                    if recipe.mode == "expression":
                        dynamic = scalar(cast(MathExpression, recipe.value)._root,
                                         frozenset(dict(recipe.local)),
                                         visited | {name})
                    elif recipe.mode in {"state", "concentration"}:
                        # Parameters without rate rules only change at an atomic
                        # event/control boundary. Reobserve them there; a held
                        # threshold is not a continuous zero-slope root.
                        dynamic = not (recipe.mode == "state"
                            and str(recipe.value).startswith("P:")
                            and name not in frame._rate_rules)
                    for dependency in recipe.dependencies:
                        # This also checks assigned volumes used by concentration views.
                        from chemworld.physchem.mathml import MathNode

                        dynamic = scalar(MathNode("symbol", value=dependency),
                                         visited=visited | {name}) or dynamic
            if node.operation == "call":
                function = frame._functions[node.value]._root
                dynamic = scalar(function.arguments[0], frozenset(function.bindings), visited)
            for argument in node.arguments:
                dynamic = scalar(argument, locals_, visited) or dynamic
            return dynamic

        def install(expression, left, right):
            # Root evaluation and derivatives consume the original common MathML
            # interpreter and FirstDerivative, including lexical function calls.
            key = (left, right)
            if key in shared:
                return shared[key]
            identity = f"sbml_guard:{len(roots)}"
            symbols = (expression._symbols(left) | expression._symbols(right)) - {"$time"}

            def difference(state, derivative=False):
                if derivative:
                    rates = frame.rates(state.coordinates, state.time_s).coordinate_rates
                    values = {k: FirstDerivative(v, rates[k]) for k, v in state.coordinates.items()}
                    time: float | FirstDerivative[float, float] = FirstDerivative(state.time_s, 1.)
                else:
                    values, time = state.coordinates, state.time_s
                scope = frame.context(values, time, targets=symbols)
                answer = (expression.evaluate(scope, functions=frame._functions, subtree=left)
                          - expression.evaluate(scope, functions=frame._functions, subtree=right))
                return answer.slope if derivative and isinstance(answer, FirstDerivative) else (
                    0. if derivative else answer)

            roots.append(CoordinateRoot(identity, difference, lambda s: difference(s, True)))
            shared[key] = identity
            return identity

        def boolean(expression, node, comparisons):
            if node.operation in {"lt", "leq", "gt", "geq"}:
                dynamic = [scalar(argument) for argument in node.arguments]
                if any(dynamic):
                    pairs = []
                    for i, (a, b) in enumerate(pairwise(node.arguments)):
                        if dynamic[i] or dynamic[i+1]:
                            pairs.append(install(expression, a, b))
                        else:
                            pairs.append((a, b))
                    comparisons[node] = tuple(pairs)
            elif node.operation in {"and", "or", "xor", "not", "implies"}:
                for child in node.arguments:
                    boolean(expression, child, comparisons)
            elif node.operation != "constant" or type(node.value) is not bool:
                raise ValueError("unsupported SBML event guard; use continuous inequalities")

        for index, record in enumerate(frame.model.declaration.items("listOfEvents")):
            expression = record.of("trigger")[0].mathematics
            comparisons = {}
            boolean(expression, expression._root, comparisons)
            guards.append((record.get("id", f"@event:{index}"), expression,
                           MappingProxyType(comparisons)))
        object.__setattr__(self, "roots", tuple(roots))
        object.__setattr__(self, "_guards", tuple(guards))

    def trigger_values(self, state, signs):
        scope = self.frame.context(state.coordinates, state.time_s)
        result = {}
        for name, expression, comparisons in self._guards:
            overrides = {}
            for node, identities in comparisons.items():
                truths = []
                for key in identities:
                    if isinstance(key, tuple):
                        left, right = (expression.evaluate(scope, functions=self.frame._functions,
                                       subtree=operand) for operand in key)
                        truth = _COMPARISONS[node.operation](
                            scalar_value(left), scalar_value(right))
                    else:
                        truth = (signs[key] > 0 if node.operation in {"gt", "geq"}
                                 else signs[key] < 0)
                    truths.append(truth)
                overrides[node] = all(truths)
            result[name] = expression.evaluate(scope, functions=self.frame._functions,
                                                comparison_values=overrides)
        return result
