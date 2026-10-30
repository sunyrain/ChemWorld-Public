"""Constrained design of original multi-outlet flowsheets without a second solver.

Bindings change explicitly declared physical fields on a private graph copy.
Every objective/constraint evaluation solves that graph, including its original
tears and actual unit equations. Rate costs are caller-authored conversions of
observed quantities, not an inferred equipment-economic model. This tool owns no
experimental inventory, clock, ledger or executable physical session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite
from time import perf_counter
from typing import Any

from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    BoundedDesignResult,
    DesignConstraint,
    DesignEvaluationError,
    DesignPoint,
    DesignVariable,
    solve_bounded_design,
)
from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.process_flowsheet import (
    FlowsheetGraph,
    FlowsheetResult,
    unit_utility_rates,
)

FieldPath = tuple[str | int, ...]


def _physical_fields(operation: Mapping[str, Any]) -> dict[FieldPath, str]:
    """Only physical design fields, never a model identity or numerical policy."""
    kind = operation.get("kind")
    fields: dict[FieldPath, str] = {}
    if kind in {"mixer", "pressure", "heat"}:
        fields[("outlet_pressure_Pa",)] = "Pa"
    if kind == "pressure":
        fields[("isentropic_efficiency",)] = "1"
    if kind == "heat":
        fields.update({("heat_duty_W",): "W", ("boundary_temperature_K",): "K"})
    if kind == "splitter":
        fields.update({("fractions", i): "1" for i in range(len(operation["fractions"]))})
    if kind == "heat_exchanger":
        fields.update(
            {
                ("conductance_W_K",): "W/K",
                ("hot_outlet_pressure_Pa",): "Pa",
                ("cold_outlet_pressure_Pa",): "Pa",
            }
        )
    if kind == "pressure_recovery":
        fields[("transmission_efficiency",)] = "1"
        for name in ("expander", "compressor"):
            fields.update(
                {(name, *path): unit for path, unit in _physical_fields(operation[name]).items()}
            )
    if operation.get("model_type") in {
        "molecular_flash_separator_v1",
        "molecular_equilibrium_cascade_v1",
    }:
        fields[("pressure_Pa",)] = "Pa"
        prefixes: tuple[FieldPath, ...] = (
            (("device",),)
            if "device" in operation
            else tuple(("devices", i) for i in range(len(operation["devices"])))
        )
        for prefix in prefixes:
            fields.update(
                {
                    (*prefix, key): unit
                    for key, unit in (
                        ("setpoint_K", "K"),
                        ("maximum_power_W", "W"),
                        ("maximum_cooling_power_W", "W"),
                    )
                }
            )
    return fields


def _set_field(target: Any, path: FieldPath, value: float) -> None:
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _mutable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_mutable(v) for v in value]
    return deepcopy(value)


@dataclass(frozen=True)
class FlowsheetDesignBinding:
    variable_id: str
    target_kind: str
    target_id: str
    field_path: FieldPath
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_path", tuple(self.field_path))
        if (
            not self.variable_id
            or not self.target_id
            or self.target_kind not in {"unit", "source"}
            or not self.field_path
            or any(not isinstance(k, str | int) or isinstance(k, bool) for k in self.field_path)
            or any(isinstance(v, bool) or not isfinite(v) for v in (self.multiplier, self.offset))
            or self.multiplier == 0
        ):
            raise ValueError(
                "flowsheet binding needs an explicit physical target and finite affine map"
            )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "field_path": list(self.field_path)}


@dataclass(frozen=True)
class FlowsheetCostTerm:
    metric: str
    metric_unit: str
    coefficient: float

    def __post_init__(self) -> None:
        if (
            not self.metric
            or not self.metric_unit
            or isinstance(self.coefficient, bool)
            or not isfinite(self.coefficient)
        ):
            raise ValueError(
                "cost term needs a metric, native unit and finite authored coefficient"
            )


_STREAM_UNITS = {
    "present": "1",
    "molar_rate_mol_s": "mol/s",
    "temperature_K": "K",
    "pressure_Pa": "Pa",
    "enthalpy_rate_W": "W",
}
_TRANSFER_UNITS = {
    "heat_duty_W": "W",
    "shaft_work_W": "W",
    "heat_import_W": "W",
    "heat_export_W": "W",
    "shaft_import_W": "W",
    "shaft_export_W": "W",
}
_DIAGNOSTIC_UNITS = {
    "internal_heat_W": "W",
    "recovered_work_W": "W",
    "generated_work_W": "W",
    "compressor_work_W": "W",
    "transmission_loss_W": "W",
    "external_draw_W": "W",
    "external_export_W": "W",
    "UA_residual_W": "W",
    "entropy_generation_W_K": "W/K",
}


def flowsheet_metric_units(graph: FlowsheetGraph) -> dict[str, str]:
    units = {**_TRANSFER_UNITS, "energy_residual_W": "W", "material_residual_mol_s": "mol/s"}
    domains = graph.port_domains()
    streams = {f"port:{name}": d for name, d in domains.items()}
    streams.update({f"source:{s.port_id}": s.domain for s in graph.sources})
    streams.update({f"outlet:{name}": domains[port] for name, port in graph.outlets.items()})
    for prefix, domain in streams.items():
        units.update({f"{prefix}:{metric}": unit for metric, unit in _STREAM_UNITS.items()})
        for component in domain.model.component_ids:
            units[f"{prefix}:component_rate:{component}"] = "mol/s"
            units[f"{prefix}:mole_fraction:{component}"] = "1"
            if prefix.startswith("outlet:"):
                units[f"{prefix}:recovery:{component}"] = "1"
    for unit in graph.units:
        units.update(
            {
                f"unit:{unit.unit_id}:{metric}": symbol
                for metric, symbol in {
                    **_TRANSFER_UNITS,
                    **_DIAGNOSTIC_UNITS,
                }.items()
            }
        )
    return units


@dataclass(frozen=True)
class FlowsheetDesignSpec:
    graph: FlowsheetGraph
    variables: tuple[DesignVariable, ...]
    bindings: tuple[FlowsheetDesignBinding, ...]
    objective_unit: str
    costs: tuple[FlowsheetCostTerm, ...] = ()
    constraints: tuple[DesignConstraint, ...] = ()
    starts: tuple[Mapping[str, float], ...] = ()
    policy: BoundedDesignPolicy = field(default_factory=BoundedDesignPolicy)
    objective: Callable[[Mapping[str, float]], float] | None = None
    _frozen_graph: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = deepcopy(self.graph.to_dict())
        object.__setattr__(self, "_frozen_graph", frozen)
        object.__setattr__(self, "graph", FlowsheetGraph.from_dict(deepcopy(frozen)))
        for name in ("variables", "bindings", "costs", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "starts", tuple(deepcopy(dict(s)) for s in self.starts))
        variables = {v.parameter_id: v for v in self.variables}
        if not variables or len(variables) != len(self.variables):
            raise ValueError("flowsheet design needs distinct original bounded variables")
        if (
            not isinstance(self.objective_unit, str)
            or not self.objective_unit
            or bool(self.costs) == (self.objective is not None)
        ):
            raise ValueError("declare cost terms or one objective callback, and its objective unit")
        if self.objective is not None and not callable(self.objective):
            raise ValueError("flowsheet objective must be callable")
        if {b.variable_id for b in self.bindings} != set(variables):
            raise ValueError("each design variable must have an explicit original graph binding")
        targets = [(b.target_kind, b.target_id, b.field_path) for b in self.bindings]
        if len(set(targets)) != len(targets):
            raise ValueError("a physical field cannot have conflicting design bindings")
        for binding in self.bindings:
            unit = self._binding_unit(binding)
            if variables[binding.variable_id].unit != unit:
                raise ValueError("design coordinate units differ from the bound physical field")
        metrics = self.metric_units()
        for item, symbol in [
            *((c.metric, c.metric_unit) for c in self.costs),
            *((c.metric, c.unit) for c in self.constraints),
        ]:
            if item not in metrics or metrics[item] != symbol:
                raise ValueError(
                    "cost or constraint refers to an unknown metric or wrong native unit"
                )

    def _binding_unit(self, binding: FlowsheetDesignBinding) -> str:
        if binding.target_kind == "source":
            if binding.target_id not in {
                s["port_id"] for s in self._frozen_graph["sources"]
            } or binding.field_path != ("molar_rate_mol_s",):
                raise ValueError("source binding must name an original source molar rate")
            return "mol/s"
        operations = {u["unit_id"]: u["operation"] for u in self._frozen_graph["units"]}
        allowed = _physical_fields(operations.get(binding.target_id, {}))
        if binding.field_path not in allowed:
            raise ValueError("binding is not a supported original physical unit field")
        return allowed[binding.field_path]

    def metric_units(self) -> dict[str, str]:
        units = flowsheet_metric_units(FlowsheetGraph.from_dict(deepcopy(self._frozen_graph)))
        units.update({f"design:{v.parameter_id}": v.unit for v in self.variables})
        return units

    def compile(self, values: Mapping[str, float]) -> FlowsheetGraph:
        if set(values) != {v.parameter_id for v in self.variables}:
            raise ValueError("flowsheet design must supply exactly all declared variables")
        for variable in self.variables:
            variable.encode(values[variable.parameter_id])
        data = _mutable(self._frozen_graph)
        sources = {s["port_id"]: s["stream"] for s in data["sources"]}
        units = {u["unit_id"]: u["operation"] for u in data["units"]}
        for binding in self.bindings:
            target = (
                sources[binding.target_id]
                if binding.target_kind == "source"
                else units[binding.target_id]
            )
            value = binding.multiplier * values[binding.variable_id] + binding.offset
            if not isfinite(value):
                raise ValueError("bound physical design value is not finite")
            _set_field(target, binding.field_path, float(value))
        # Rebuild once: complementary split fractions are never partially applied.
        return FlowsheetGraph.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "original-flowsheet-design-v1",
            "graph": deepcopy(self._frozen_graph),
            "variables": [asdict(v) for v in self.variables],
            "bindings": [b.to_dict() for b in self.bindings],
            "objective_unit": self.objective_unit,
            "costs": [asdict(c) for c in self.costs],
            "constraints": [asdict(c) for c in self.constraints],
            "starts": [dict(s) for s in self.starts],
            "policy": asdict(self.policy),
            "objective_callback": None
            if self.objective is None
            else (
                f"{getattr(self.objective, '__module__', '')}."
                f"{getattr(self.objective, '__qualname__', type(self.objective).__name__)}"
            ),
            "callback_serialization": "description_only; explicit_callable_required",
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        objective: Callable[[Mapping[str, float]], float] | None = None,
    ) -> FlowsheetDesignSpec:
        if data.get("schema") != "original-flowsheet-design-v1":
            raise ValueError("unknown flowsheet design encoding")
        if (data.get("objective_callback") is not None) != (objective is not None):
            raise ValueError("callback design requires its explicitly supplied callable")
        return cls(
            FlowsheetGraph.from_dict(data["graph"]),
            tuple(DesignVariable(**v) for v in data["variables"]),
            tuple(FlowsheetDesignBinding(**b) for b in data["bindings"]),
            data["objective_unit"],
            tuple(FlowsheetCostTerm(**c) for c in data["costs"]),
            tuple(DesignConstraint(**c) for c in data["constraints"]),
            tuple(data["starts"]),
            BoundedDesignPolicy(**data["policy"]),
            objective,
        )


def _stream_observations(
    prefix: str, stream: MolecularStream | None, components: tuple[str, ...]
) -> dict[str, float]:
    result = {
        f"{prefix}:present": float(stream is not None),
        f"{prefix}:molar_rate_mol_s": 0.0 if stream is None else stream.molar_rate_mol_s,
        f"{prefix}:enthalpy_rate_W": 0.0 if stream is None else stream.enthalpy_rate_W,
    }
    for component in components:
        index = components.index(component)
        result[f"{prefix}:component_rate:{component}"] = (
            0.0 if stream is None else stream.molar_rate_mol_s * stream.state.mole_fractions[index]
        )
        if stream is not None:
            result[f"{prefix}:mole_fraction:{component}"] = stream.state.mole_fractions[index]
    if stream is not None:
        result.update(
            {
                f"{prefix}:temperature_K": stream.state.temperature_K,
                f"{prefix}:pressure_Pa": stream.state.pressure_Pa,
            }
        )
    return result


def flowsheet_observations(graph: FlowsheetGraph, result: FlowsheetResult) -> dict[str, float]:
    if not result.success:
        raise DesignEvaluationError(
            "unsuccessful_graph_has_no_design_observations", result.to_dict()
        )
    observations = {"heat_duty_W": result.heat_duty_W, "shaft_work_W": result.shaft_work_W}
    for key in ("heat_import_W", "heat_export_W", "shaft_import_W", "shaft_export_W"):
        observations[key] = result.diagnostics[key]
    observations["energy_residual_W"] = result.diagnostics["energy_residual_W"]
    observations["material_residual_mol_s"] = max(
        map(abs, result.diagnostics["component_balance_mol_s"].values()), default=0.0
    )
    feed_rates: dict[str, float] = {}
    for source in graph.sources:
        components = source.domain.model.component_ids
        observations.update(
            _stream_observations(f"source:{source.port_id}", source.stream, components)
        )
        for component, fraction in zip(components, source.stream.state.mole_fractions, strict=True):
            feed_rates[component] = (
                feed_rates.get(component, 0.0) + source.stream.molar_rate_mol_s * fraction
            )
    domains = graph.port_domains()
    for port, stream in result.port_streams.items():
        observations.update(
            _stream_observations(f"port:{port}", stream, domains[port].model.component_ids)
        )
    for name, port in graph.outlets.items():
        components = domains[port].model.component_ids
        observations.update(
            _stream_observations(f"outlet:{name}", result.outlets.get(name), components)
        )
        for component in components:
            if feed_rates.get(component, 0.0) > 0:
                observations[f"outlet:{name}:recovery:{component}"] = (
                    observations[f"outlet:{name}:component_rate:{component}"]
                    / feed_rates[component]
                )
    for declaration in graph.units:
        name = declaration.unit_id
        unit = result.unit_results[name]
        transfers = {"heat_duty_W": unit.heat_duty_W, "shaft_work_W": unit.shaft_work_W,
                     **unit_utility_rates(declaration.operation, unit)}
        observations.update(
            {
                f"unit:{name}:{key}": value
                for key, value in transfers.items()
            }
        )
        for key in _DIAGNOSTIC_UNITS:
            value = unit.diagnostics.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                observations[f"unit:{name}:{key}"] = float(value)
    return observations


def evaluate_flowsheet_design(
    specification: FlowsheetDesignSpec, values: Mapping[str, float]
) -> DesignPoint:
    start = perf_counter()
    result = None
    try:
        graph = specification.compile(values)
        result = graph.solve()
        if not result.success:
            raise DesignEvaluationError(result.failure_reason or "flowsheet_solve_failed")
        observations = flowsheet_observations(graph, result)
        observations.update({f"design:{name}": float(value) for name, value in values.items()})
        required = {c.metric for c in specification.costs} | {
            c.metric for c in specification.constraints
        }
        missing = sorted(required - set(observations))
        if missing:
            raise DesignEvaluationError("unavailable_physical_design_metrics:" + ",".join(missing))
        terms = {
            f"{index}:{term.metric}": term.coefficient * observations[term.metric]
            for index, term in enumerate(specification.costs)
        }
        objective = (
            fsum(terms.values())
            if specification.objective is None
            else specification.objective(deepcopy(observations))
        )
        if not isfinite(objective) or any(not isfinite(v) for v in observations.values()):
            raise DesignEvaluationError("nonfinite_flowsheet_objective_or_observation")
        return DesignPoint(
            float(objective),
            observations,
            {
                "original_flowsheet": result.to_dict(),
                "cost_terms": terms,
                "objective_unit": specification.objective_unit,
                "evaluation_wall_time_s": perf_counter() - start,
            },
        )
    except (ValueError, ArithmeticError, RuntimeError, KeyError) as exc:
        raise DesignEvaluationError(
            str(exc),
            {
                "original_flowsheet": None if result is None else result.to_dict(),
                "values": dict(values),
                "evaluation_wall_time_s": perf_counter() - start,
            },
        ) from exc


@dataclass(frozen=True)
class FlowsheetDesignResult:
    specification: FlowsheetDesignSpec
    optimization: BoundedDesignResult
    selected_graph: FlowsheetGraph | None
    wall_time_s: float

    @property
    def success(self) -> bool:
        return self.optimization.success and self.selected_graph is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "original-flowsheet-design-result-v1",
            "success": self.success,
            "specification": self.specification.to_dict(),
            "optimization": self.optimization.to_dict(),
            "selected_graph": None
            if self.selected_graph is None
            else self.selected_graph.to_dict(),
            "solve_wall_time_s": self.wall_time_s,
            "physical_time_s": 0.0,
            "experiment_resource_debit": None,
        }


def optimize_flowsheet(
    specification: FlowsheetDesignSpec, *, progress: Callable[[dict[str, Any]], None] | None = None
) -> FlowsheetDesignResult:
    start = perf_counter()
    result = solve_bounded_design(
        specification.variables,
        specification.constraints,
        lambda values: evaluate_flowsheet_design(specification, values),
        starts=specification.starts,
        policy=specification.policy,
        progress=progress,
    )
    selected = (
        specification.compile(result.values)
        if result.success and result.values is not None
        else None
    )
    return FlowsheetDesignResult(specification, result, selected, perf_counter() - start)
