"""Bounded range declarations evaluated with the shared scalar MathML engine.

No model state is retained here. A functional range reads the caller's current
model view at the iteration boundary, not an initialized cached substitute.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from chemworld.data.sbml_parser import acyclic_order
from chemworld.data.sedml_declarations import finite_number, mathematics, records, step_count
from chemworld.data.sedml_models import calculation
from chemworld.data.sedml_xml import ExperimentXMLNode


def _range_reference(variable):
    target = variable.attributes.get("target", "")
    if not target.startswith("#"):
        return None
    if set(variable.attributes) & {"symbol", "modelReference", "taskReference"}:
        raise ValueError("range variables cannot also reference a model, task or symbol")
    return target[1:]


@dataclass(frozen=True)
class SEDMLRanges:
    task: ExperimentXMLNode
    maximum_values: int = 10000
    _nodes: object = field(init=False, repr=False, compare=False)
    _static: object = field(init=False, repr=False, compare=False)
    _order: tuple = field(init=False, repr=False, compare=False)
    count: int = field(init=False)

    def __post_init__(self) -> None:
        if (type(self.maximum_values) is not int or not 1 <= self.maximum_values <= 100000
                or self.task.local_name != "repeatedTask"):
            raise ValueError("range plan needs a repeated task and a finite positive budget")
        nodes = {n.attributes["id"]: n for n in records(self.task, "listOfRanges")}
        master = self.task.attributes["range"]
        if not nodes or len(nodes) > 128 or master not in nodes:
            raise ValueError("repeated task requires a known master range and bounded range count")
        graph, static = {}, {}
        for identity, node in nodes.items():
            if node.local_name == "functionalRange":
                reference = node.attributes["range"]
                dependencies = {reference}
                variables = records(node, "listOfVariables")
                for variable in variables:
                    if "taskReference" in variable.attributes:
                        raise ValueError("functional ranges cannot read task output arrays")
                    other = _range_reference(variable)
                    if other is not None:
                        dependencies.add(other)
                    elif "modelReference" not in variable.attributes:
                        raise ValueError("functional model variables require an explicit model")
                if dependencies-nodes.keys():
                    raise ValueError("functional range refers to an unknown range")
                scope = ({reference} | {v.attributes["id"] for v in variables}
                         | {p.attributes["id"] for p in records(node, "listOfParameters")})
                if set(mathematics(node).symbols)-scope:
                    raise ValueError("functional range has an undeclared mathematical input")
                graph[identity] = dependencies
                continue
            graph[identity] = set()
            if node.local_name == "vectorRange":
                values = tuple(finite_number(n.text.strip()) for n in node.children
                               if n.local_name == "value")
            else:
                count = step_count(node.attributes)+1
                if count > self.maximum_values:
                    raise ValueError("uniform range exceeds its declared finite expansion budget")
                start, end = (finite_number(node.attributes[key]) for key in ("start", "end"))
                if node.attributes["type"] == "log":
                    if start <= 0 or end <= 0:
                        raise ValueError("logarithmic range endpoints must be positive")
                    values = tuple(float(v) for v in np.geomspace(start, end, count))
                else:
                    values = tuple(float(v) for v in np.linspace(start, end, count))
            if (not values or len(values) > self.maximum_values
                    or not all(np.isfinite(v) for v in values)):
                raise ValueError("range values must be finite and fit their expansion budget")
            static[identity] = values
        order = acyclic_order(graph, label="SED-ML range")
        counts = {key: len(value) for key, value in static.items()}
        for identity in order:
            if identity not in counts:
                counts[identity] = counts[nodes[identity].attributes["range"]]
                if any(counts[other] < counts[identity] for other in graph[identity]):
                    raise ValueError("functional range input is shorter than its driving range")
        if (any(count < counts[master] for count in counts.values())
                or sum(counts.values()) > 200000):
            raise ValueError("auxiliary ranges must cover the master within the total range budget")
        object.__setattr__(self, "_nodes", MappingProxyType(nodes))
        object.__setattr__(self, "_static", MappingProxyType(static))
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "count", counts[master])

    def values_at(self, index, *, model_value=None):
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError("range index is outside the finite master grid")
        values = {}

        def variable_value(variable):
            reference = _range_reference(variable)
            if reference is not None:
                return values[reference]
            if model_value is None:
                raise ValueError("functional range requires a current model observation")
            return model_value(variable)

        for identity in self._order:
            if identity in self._static:
                values[identity] = self._static[identity][index]
            else:
                node = self._nodes[identity]
                reference = node.attributes["range"]
                values[identity] = calculation(node, variable_value=variable_value,
                                                extra={reference: values[reference]})
        return MappingProxyType(values)
