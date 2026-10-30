"""Immutable source-rule proof for a piecewise constant integer process."""

from dataclasses import dataclass, field, replace
from types import MappingProxyType

from chemworld.physchem.mathml import MathExpression


def inline(node, assignments, functions, *, shadow=frozenset(), bindings=None, visited=frozenset()):
    """Expand actual source syntax with local shadowing and lexical function bindings."""
    bindings = {} if bindings is None else bindings
    if node.operation == "symbol":
        name = node.value
        if name in bindings:
            return bindings[name]
        if name in assignments and name not in shadow:
            if name in visited:
                raise ValueError("cyclic integer assignment rule")
            return inline(assignments[name]._root, assignments, functions, visited=visited | {name})
        return node
    arguments = tuple(
        inline(a, assignments, functions, shadow=shadow, bindings=bindings, visited=visited)
        for a in node.arguments
    )
    if node.operation == "call":
        function = functions[node.value]._root
        if len(arguments) != len(function.bindings):
            raise ValueError("integer source function argument count differs")
        # Source lambdas are lexical; caller globals and local reaction parameters
        # do not become a closure of the function body.
        return inline(
            function.arguments[0],
            {},
            functions,
            bindings=dict(zip(function.bindings, arguments, strict=True)),
        )
    return replace(node, arguments=arguments)


@dataclass(frozen=True)
class SBMLIntegerRules:
    frame: object
    assigned: object = field(init=False, repr=False)
    mutable_coordinates: frozenset = field(init=False)

    def __post_init__(self) -> None:
        frame, root = self.frame, self.frame.model.declaration
        rules = root.items("listOfRules")
        if any(rule.kind != "assignmentRule" for rule in rules):
            raise ValueError("integer rate/algebraic rules require continuous hazard coupling")
        assigned = dict(frame._assigned)
        if set(assigned) - frame._parameters.keys():
            raise ValueError("integer assignment rules currently require parameter targets")
        object.__setattr__(self, "assigned", MappingProxyType(assigned))
        for expression in assigned.values():
            if "$time" in MathExpression._symbols(self.expand(expression._root)):
                raise ValueError("integer assignment rules must stay constant between jumps")
        for constraint in root.items("listOfConstraints"):
            if "$time" in MathExpression._symbols(self.expand(constraint.mathematics._root)):
                raise ValueError(
                    "time-dependent integer constraints need continuous boundary coupling"
                )
        targets = {
            assignment.get("variable")
            for event in root.items("listOfEvents")
            for assignment in event.items("listOfEventAssignments")
        }
        if targets - set(frame._parameters) - set(frame._species) - set(frame._compartments):
            raise ValueError("integer events require species, parameter or compartment targets")
        factors = {
            r.get("conversionFactor", root.get("conversionFactor")) for r in frame._species.values()
        } - {None}
        if factors & (set(assigned) | targets | set(frame.controlled_parameters)):
            raise ValueError("integer conversion factors cannot vary molecular stoichiometry")
        mutable = {"P:" + name for name in targets if name in frame._parameters}
        mutable.update("V:" + name for name in targets if name in frame._compartments)
        object.__setattr__(self, "mutable_coordinates", frozenset(mutable))

    def expand(self, node, *, shadow=frozenset()):
        return inline(node, self.assigned, self.frame._functions, shadow=shadow)
