"""Immutable report arrays with explicit sample coordinates and shared mathematics."""

from dataclasses import dataclass, field
from math import isfinite, prod
from types import MappingProxyType

from chemworld.data.sedml_declarations import finite_number, mathematics, records
from chemworld.data.sedml_quantities import model_quantity
from chemworld.data.sedml_tasks import SEDMLTasks


@dataclass(frozen=True)
class SEDMLArray:
    values: tuple
    shape: tuple
    axes: tuple
    sample_keys: tuple
    unit: str | None = None

    def __post_init__(self):
        values, shape, axes, keys = map(tuple,
            (self.values, self.shape, self.axes, self.sample_keys))
        if (len(shape) > 16 or any(type(n) is not int or n < 1 for n in shape)
                or prod(shape) != len(values) or not 1 <= len(values) <= 2000000
                or len(axes) != len(shape) or any(not isinstance(a, str) or not a for a in axes)
                or len(keys) != len(values) or any(not isinstance(k, tuple) for k in keys)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                       for v in values)
                or (self.unit is not None and (not isinstance(self.unit, str) or not self.unit))):
            raise ValueError("report array needs finite values and complete shape/axis coordinates")
        # Coordinates are immutable scalar tuples; never live state or opaque objects.
        for key in keys:
            if any(type(v) not in (str, int, float) or (isinstance(v, float) and not isfinite(v))
                   for v in key):
                raise ValueError("report sample coordinates must be immutable finite scalars")
        for name, value in (("values", values), ("shape", shape), ("axes", axes),
                            ("sample_keys", keys)):
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {"values": list(self.values), "shape": list(self.shape), "axes": list(self.axes),
                "sample_keys": [list(key) for key in self.sample_keys], "unit": self.unit}


@dataclass(frozen=True)
class SEDMLReports:
    tasks: SEDMLTasks
    models: object
    bindings: object = field(init=False, repr=False)
    generators: object = field(init=False, repr=False)
    reports: object = field(init=False, repr=False)

    def __post_init__(self):
        bindings, generators, reports = {}, {}, {}
        models = dict(self.models)
        for node in records(self.tasks.document.declaration, "listOfDataGenerators"):
            identity = node.attributes["id"]
            variables = records(node, "listOfVariables")
            task_ids, model_ids = set(), set()
            for variable in variables:
                attributes = variable.attributes
                task_id = attributes.get("taskReference")
                if task_id not in self.tasks.nodes:
                    raise ValueError("report variables require a known task output")
                task_ids.add(task_id)
                involved = self.tasks.model_ids[task_id]
                model_id = attributes.get("modelReference")
                if model_id is None and len(involved) == 1:
                    model_id = next(iter(involved))
                if model_id not in involved or model_id not in models:
                    raise ValueError("multimodel task variables require a known explicit model")
                model_ids.add(model_id)
                quantity = model_quantity(models[model_id], variable)
                bindings[identity, attributes["id"]] = (task_id, model_id, quantity)
            if len(task_ids) > 1:
                raise ValueError("cross-task data generators require an explicit alignment profile")
            if len(model_ids) > 1:
                raise ValueError("cross-model generators need an explicit alignment profile")
            scope = ({v.attributes["id"] for v in variables}
                     | {p.attributes["id"] for p in records(node, "listOfParameters")})
            if set(mathematics(node).symbols)-scope:
                raise ValueError("data generator has undeclared mathematical inputs")
            generators[identity] = node
        for node in records(self.tasks.document.declaration, "listOfOutputs"):
            datasets = records(node, "listOfDataSets")
            if any(d.attributes["dataReference"] not in generators for d in datasets):
                raise ValueError("report dataset refers to an unknown data generator")
            reports[node.attributes["id"]] = datasets
        object.__setattr__(self, "models", MappingProxyType(models))
        object.__setattr__(self, "bindings", MappingProxyType(bindings))
        object.__setattr__(self, "generators", MappingProxyType(generators))
        object.__setattr__(self, "reports", MappingProxyType(reports))

    def evaluate(self, variable_arrays):
        if set(variable_arrays) != set(self.bindings):
            raise ValueError("report evaluation requires exactly the declared observation arrays")
        for key, array in variable_arrays.items():
            if (not isinstance(array, SEDMLArray)
                    or array.unit != self.bindings[key][2].source_unit):
                raise ValueError("report observation array differs from its declared source unit")
        generated = {}
        for identity, node in self.generators.items():
            variables = records(node, "listOfVariables")
            arrays = {v.attributes["id"]: variable_arrays[identity, v.attributes["id"]]
                      for v in variables}
            if any(not isinstance(a, SEDMLArray) for a in arrays.values()):
                raise ValueError("report inputs must be explicit immutable arrays")
            template = next(iter(arrays.values()), SEDMLArray((0.,), (), (), ((),)))
            expected_layout = (template.shape, template.axes, template.sample_keys)
            if any((a.shape, a.axes, a.sample_keys) != expected_layout for a in arrays.values()):
                raise ValueError("report variables have different sample axes or time grids")
            parameters = {p.attributes["id"]: finite_number(p.attributes["value"])
                          for p in records(node, "listOfParameters")}
            expression = mathematics(node)
            values = []
            for index in range(len(template.values)):
                result = expression.evaluate({**parameters,
                    **{key: array.values[index] for key, array in arrays.items()}})
                if (isinstance(result, bool) or not isinstance(result, (int, float))
                        or not isfinite(result)):
                    raise ValueError("data generator must produce finite scalar samples")
                values.append(float(result))
            # Only an unchanged variable carries its declared physical unit.
            # Derived mathematics remains unknown until dimensional inference is explicit.
            unit = (arrays[expression._root.value].unit
                    if expression._root.operation == "symbol" and expression._root.value in arrays
                    else None)
            generated[identity] = SEDMLArray(tuple(values), template.shape, template.axes,
                                             template.sample_keys, unit)
        return {identity: tuple({"id": d.attributes["id"], "label": d.attributes["label"],
                    "data_reference": d.attributes["dataReference"],
                    "data": generated[d.attributes["dataReference"]].to_dict()}
                for d in datasets) for identity, datasets in self.reports.items()}
