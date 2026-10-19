"""Declarative world-composition entry point for the ChemWorld v1 surface."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isclose, isfinite
from typing import Any

from chemworld.foundation.units import convert_value, unit_spec
from chemworld.tasks import TaskSpec, default_kernel_maturity, get_task
from chemworld.world.component_registry import (
    COMPONENT_CAPABILITIES,
    MAX_CONNECTIONS,
    MAX_VESSELS,
    NETWORK_OPERATIONS,
    component_operations,
)
from chemworld.world.instruments import instrument_contracts
from chemworld.world.operations import (
    CRYSTALLIZATION_OPERATIONS,
    ELECTROCHEMISTRY_OPERATIONS,
    FLOW_OPERATIONS,
    INSTRUMENTS,
    SEPARATION_OPERATIONS,
)
from chemworld.world.parameters import SUPPORTED_SPLITS
from chemworld.world.process_time_budget import ProcessTimeBudgetPolicy
from chemworld.world.scenario import ScenarioSpec, get_scenario

WORLD_COMPOSITION_SCHEMA_VERSION = "chemworld-world-composition-0.1"
SUPPORTED_COMPONENT_KINDS = (
    "reaction",
    "thermal",
    "phase",
    "separation",
    "crystallization",
    "distillation",
    "continuous_flow",
    "electrochemistry",
    "observation",
)

_COMPONENT_ORDER = {kind: index for index, kind in enumerate(SUPPORTED_COMPONENT_KINDS)}
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "composition_id",
        "world_split",
        "components",
        "task",
        "vessels",
        "connections",
    }
)
_COMPONENT_KEYS = frozenset({"id", "vessel", "kind", "role", "parameters"})
_TASK_KEYS = frozenset(
    {
        "objective",
        "budget",
        "operations",
        "instruments",
        "observations",
        "resources",
        "termination",
        "evaluation",
        "seeds",
        "episode_mode",
        "safety_limit",
        "difficulty",
        "description",
        "tags",
    }
)
_EVALUATION_KEYS = frozenset({"metrics", "threshold"})
_RESOURCE_KEYS = frozenset(
    {
        "operation_budget",
        "sample_volume_L",
        "time_s",
        "instrument_uses",
        "final_assays",
        "process_time_policy",
    }
)

_INTERFACES_BY_COMPONENT = {
    "reaction": ("material", "reaction_event"),
    "thermal": ("temperature", "time", "energy", "safety"),
    "phase": ("phase", "volume", "composition", "state_identity"),
    "separation": ("material", "phase", "volume", "sample_identity"),
    "crystallization": ("temperature", "phase", "crystal_state", "sample"),
    "distillation": ("temperature", "phase", "volatility", "fraction"),
    "continuous_flow": ("flow_rate", "residence_time", "temperature", "material"),
    "electrochemistry": ("potential", "current", "charge", "energy", "material"),
    "observation": ("public_measurement", "missingness", "cost", "latency", "sample"),
}

_COMPONENT_DEPENDENCIES = {
    "separation": frozenset({"phase"}),
    "crystallization": frozenset({"reaction", "thermal"}),
    "distillation": frozenset({"reaction", "thermal"}),
    "continuous_flow": frozenset({"reaction", "thermal"}),
    "electrochemistry": frozenset({"reaction"}),
}
_STATE_OWNERS_BY_COMPONENT = {
    "reaction": ("material_transformation", "reaction_event"),
    "thermal": ("temperature_control",),
    "phase": ("phase_transition",),
    "separation": ("sample_routing",),
    "crystallization": ("phase_transition", "crystal_state"),
    "distillation": ("phase_transition", "fraction_state"),
    "continuous_flow": ("flow_residence_state",),
    "electrochemistry": ("electrical_state",),
    "observation": ("public_measurement",),
}
_REQUIRED_ALL_OPERATIONS = {
    "reaction": frozenset({"add_solvent", "add_reagent"}),
    "crystallization": frozenset(CRYSTALLIZATION_OPERATIONS),
    "distillation": frozenset({"distill", "collect_fraction"}),
    "continuous_flow": frozenset(FLOW_OPERATIONS),
    "electrochemistry": frozenset(ELECTROCHEMISTRY_OPERATIONS),
    "observation": frozenset({"measure"}),
}
_REQUIRED_ANY_OPERATIONS = {
    "thermal": frozenset({"heat", "wait", "cool_crystallize", "distill", "run_flow"}),
    "separation": frozenset(SEPARATION_OPERATIONS),
}


@dataclass(frozen=True)
class _ParameterRule:
    value_kind: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


_COMPONENT_PARAMETER_RULES: dict[str, dict[str, _ParameterRule]] = {
    "reaction": {
        "family": _ParameterRule("string"),
        "controls": _ParameterRule(
            "choices",
            choices=("reagent", "catalyst", "solvent", "temperature", "time"),
        ),
    },
    "thermal": {
        "temperature_range_K": _ParameterRule("range", "K", 250.0, 430.0),
        "duration_range_s": _ParameterRule("range", "s", 1.0, 14_400.0),
    },
    "phase": {
        "phases": _ParameterRule(
            "choices",
            choices=("reactor_liquid", "aqueous", "organic", "solid"),
        ),
    },
    "separation": {
        "operations": _ParameterRule("choices", choices=SEPARATION_OPERATIONS),
    },
    "crystallization": {
        "temperature_range_K": _ParameterRule("range", "K", 250.0, 330.0),
        "seed_mass_range_g": _ParameterRule("range", "g", 1.0e-6, 0.050),
    },
    "distillation": {
        "temperature_range_K": _ParameterRule("range", "K", 298.15, 430.0),
        "reflux_ratio_range": _ParameterRule("range", "dimensionless", 0.0, 10.0),
        "fraction_count": _ParameterRule("integer", minimum=1.0, maximum=24.0),
    },
    "continuous_flow": {
        "flow_rate_range_mL_min": _ParameterRule("range", "mL/min", 0.01, 20.0),
        "residence_time_range_s": _ParameterRule("range", "s", 1.0, 7200.0),
        "temperature_range_K": _ParameterRule("range", "K", 298.15, 430.0),
    },
    "electrochemistry": {
        "potential_range_V": _ParameterRule("range", "V", -2.5, 2.5),
        "current_range_mA": _ParameterRule("range", "mA", 1.0e-3, 500.0),
        "duration_range_s": _ParameterRule("range", "s", 1.0, 14_400.0),
    },
    "observation": {
        "instruments": _ParameterRule("choices", choices=INSTRUMENTS),
    },
}

_CUSTOM_UNIT_TABLE = {
    "mL/min": ("flow_rate", 1.0),
    "L/s": ("flow_rate", 60_000.0),
}


@dataclass(frozen=True)
class WorldCompositionDiagnostic:
    """One stable, location-aware pre-execution compatibility failure."""

    code: str
    path: str
    message: str
    components: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "components": list(self.components),
        }


@dataclass(frozen=True)
class WorldCompatibilityReport:
    """Deterministic compatibility decision produced before runtime creation."""

    pattern: str | None
    diagnostics: tuple[WorldCompositionDiagnostic, ...]
    state_owners: dict[str, str]
    minimum_resources: dict[str, float | int]
    operation_field_bounds: dict[tuple[str, str], tuple[float, float]]
    vessel_field_bounds: dict[str, dict[tuple[str, str], tuple[float, float]]] = field(
        default_factory=dict
    )

    @property
    def compatible(self) -> bool:
        return not self.diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "accepted" if self.compatible else "rejected",
            "pattern": self.pattern,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "state_owners": dict(sorted(self.state_owners.items())),
            "minimum_resources": deepcopy(self.minimum_resources),
        }


class WorldCompositionError(ValueError):
    """Raised when a declarative composition cannot be compiled."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[WorldCompositionDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - expected)
    if unknown:
        raise WorldCompositionError(f"{label} contains unsupported fields: {unknown}")


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise WorldCompositionError(f"{label} must be a non-empty list of strings")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise WorldCompositionError(f"{label} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise WorldCompositionError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True)
class WorldComponentRequest:
    """One reader-facing component declaration."""

    kind: str
    role: str
    parameters: dict[str, Any]
    id: str = ""
    vessel: str = "reactor"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorldComponentRequest:
        _require_exact_keys(payload, _COMPONENT_KEYS, "component")
        kind = str(payload.get("kind", "")).strip()
        if kind not in SUPPORTED_COMPONENT_KINDS:
            raise WorldCompositionError(f"unsupported component kind: {kind!r}")
        role = str(payload.get("role", kind)).strip()
        if not role:
            raise WorldCompositionError("component role must be non-empty")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise WorldCompositionError("component parameters must be an object")
        instance_id = str(payload.get("id", kind)).strip()
        vessel = str(payload.get("vessel", "reactor")).strip()
        if not instance_id or not vessel:
            raise WorldCompositionError("component id and vessel must be non-empty")
        return cls(
            kind=kind,
            role=role,
            parameters=deepcopy(dict(parameters)),
            id=instance_id,
            vessel=vessel,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id or self.kind,
            "vessel": self.vessel,
            "kind": self.kind,
            "role": self.role,
            "parameters": deepcopy(self.parameters),
        }


@dataclass(frozen=True)
class CompositionTaskRequest:
    """Reader-facing task surface attached to a composed world."""

    objective: str | None
    budget: int | None
    operations: tuple[str, ...] | None
    instruments: tuple[str, ...] | None
    observations: str | None
    resources: dict[str, Any]
    termination: str | None
    evaluation_metrics: tuple[str, ...] | None
    evaluation_threshold: float | None
    seeds: tuple[int, ...] | None
    episode_mode: str | None
    safety_limit: float | None
    difficulty: str | None
    description: str | None
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompositionTaskRequest:
        _require_exact_keys(payload, _TASK_KEYS, "task")
        budget_value = payload.get("budget")
        budget = None if budget_value is None else int(budget_value)
        if budget is not None and budget <= 0:
            raise WorldCompositionError("task budget must be positive")
        operations_value = payload.get("operations")
        operations = (
            None if operations_value is None else _string_tuple(operations_value, "task operations")
        )
        instruments_value = payload.get("instruments")
        instruments = (
            None
            if instruments_value is None
            else _string_tuple(instruments_value, "task instruments")
        )
        if instruments is not None:
            unknown_instruments = sorted(set(instruments) - set(INSTRUMENTS))
            if unknown_instruments:
                raise WorldCompositionError(
                    f"task instruments are unsupported: {unknown_instruments}"
                )
        resources_value = payload.get("resources", {})
        if not isinstance(resources_value, Mapping):
            raise WorldCompositionError("task resources must be an object")
        _require_exact_keys(resources_value, _RESOURCE_KEYS, "task resources")
        process_time_policy = resources_value.get("process_time_policy")
        if process_time_policy is not None:
            if not isinstance(process_time_policy, Mapping):
                raise WorldCompositionError("task resources.process_time_policy must be an object")
            try:
                normalized_policy = ProcessTimeBudgetPolicy.from_dict(process_time_policy)
            except (TypeError, ValueError) as exc:
                raise WorldCompositionError(
                    f"invalid task resources.process_time_policy: {exc}"
                ) from exc
            declared_time = resources_value.get("time_s")
            if declared_time is None or not isclose(
                float(declared_time),
                normalized_policy.process_time_limit_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise WorldCompositionError(
                    "task resources.time_s must match process_time_policy.process_time_limit_s"
                )
        evaluation_value = payload.get("evaluation", {})
        if not isinstance(evaluation_value, Mapping):
            raise WorldCompositionError("task evaluation must be an object")
        _require_exact_keys(evaluation_value, _EVALUATION_KEYS, "task evaluation")
        metrics_value = evaluation_value.get("metrics")
        evaluation_metrics = (
            None
            if metrics_value is None
            else _string_tuple(metrics_value, "task evaluation metrics")
        )
        threshold_value = evaluation_value.get("threshold")
        evaluation_threshold = None if threshold_value is None else float(threshold_value)
        if evaluation_threshold is not None and not 0.0 <= evaluation_threshold <= 1.0:
            raise WorldCompositionError("task evaluation threshold must be in [0, 1]")
        seeds_value = payload.get("seeds")
        seeds = None
        if seeds_value is not None:
            if not isinstance(seeds_value, (list, tuple)) or not seeds_value:
                raise WorldCompositionError("task seeds must be a non-empty list")
            seeds = tuple(int(seed) for seed in seeds_value)
            if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
                raise WorldCompositionError("task seeds must be unique non-negative integers")
        episode_mode_value = payload.get("episode_mode")
        episode_mode = None if episode_mode_value is None else str(episode_mode_value)
        if episode_mode not in {None, "single_experiment", "campaign"}:
            raise WorldCompositionError("task episode_mode must be single_experiment or campaign")
        safety_limit_value = payload.get("safety_limit")
        safety_limit = None if safety_limit_value is None else float(safety_limit_value)
        if safety_limit is not None and not 0.0 < safety_limit <= 1.0:
            raise WorldCompositionError("task safety_limit must be in (0, 1]")
        tags_value = payload.get("tags", ())
        tags = (
            ()
            if tags_value is None or tags_value == () or tags_value == []
            else _string_tuple(tags_value, "task tags")
        )
        return cls(
            objective=(None if payload.get("objective") is None else str(payload["objective"])),
            budget=budget,
            operations=operations,
            instruments=instruments,
            observations=(
                None if payload.get("observations") is None else str(payload["observations"])
            ),
            resources=deepcopy(dict(resources_value)),
            termination=(
                None if payload.get("termination") is None else str(payload["termination"])
            ),
            evaluation_metrics=evaluation_metrics,
            evaluation_threshold=evaluation_threshold,
            seeds=seeds,
            episode_mode=episode_mode,
            safety_limit=safety_limit,
            difficulty=(None if payload.get("difficulty") is None else str(payload["difficulty"])),
            description=(
                None if payload.get("description") is None else str(payload["description"])
            ),
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "budget": self.budget,
            "operations": None if self.operations is None else list(self.operations),
            "instruments": None if self.instruments is None else list(self.instruments),
            "observations": self.observations,
            "resources": deepcopy(self.resources),
            "termination": self.termination,
            "evaluation": {
                "metrics": (
                    None if self.evaluation_metrics is None else list(self.evaluation_metrics)
                ),
                "threshold": self.evaluation_threshold,
            },
            "seeds": None if self.seeds is None else list(self.seeds),
            "episode_mode": self.episode_mode,
            "safety_limit": self.safety_limit,
            "difficulty": self.difficulty,
            "description": self.description,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class WorldCompositionSpec:
    """Validated declarative request before runtime compilation."""

    schema_version: str
    composition_id: str
    world_split: str
    components: tuple[WorldComponentRequest, ...]
    task: CompositionTaskRequest
    vessels: tuple[dict[str, Any], ...] = ()
    connections: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorldCompositionSpec:
        _require_exact_keys(payload, _TOP_LEVEL_KEYS, "world composition")
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != WORLD_COMPOSITION_SCHEMA_VERSION:
            raise WorldCompositionError("unsupported world-composition schema")
        composition_id = str(payload.get("composition_id", "")).strip()
        if not composition_id:
            raise WorldCompositionError("composition_id must be non-empty")
        world_split = str(payload.get("world_split", "public-dev"))
        if world_split not in SUPPORTED_SPLITS:
            raise WorldCompositionError(f"unsupported world_split: {world_split!r}")
        components_value = payload.get("components")
        if not isinstance(components_value, list) or not components_value:
            raise WorldCompositionError("components must be a non-empty list")
        components = tuple(
            WorldComponentRequest.from_dict(component)
            for component in components_value
            if isinstance(component, Mapping)
        )
        if len(components) != len(components_value):
            raise WorldCompositionError("every component must be an object")
        identities = tuple(component.id for component in components)
        if len(set(identities)) != len(identities):
            raise WorldCompositionError(
                "component instance IDs must not be duplicated",
                diagnostics=(
                    _diagnostic(
                        "conflicting_state_owner",
                        "components",
                        "component instance IDs must be unique",
                    ),
                ),
            )
        components = tuple(
            sorted(components, key=lambda item: (_COMPONENT_ORDER[item.kind], item.id))
        )
        vessel_value = payload.get(
            "vessels",
            [{"id": name, "capacity_L": 0.1} for name in sorted({c.vessel for c in components})],
        )
        if not isinstance(vessel_value, list) or not 1 <= len(vessel_value) <= MAX_VESSELS:
            raise WorldCompositionError(f"vessels must contain 1..{MAX_VESSELS} declarations")
        vessels = []
        for vessel in vessel_value:
            if not isinstance(vessel, Mapping):
                raise WorldCompositionError("each vessel must be an object")
            _require_exact_keys(vessel, frozenset({"id", "capacity_L"}), "vessel")
            name, capacity = vessel.get("id"), vessel.get("capacity_L", 0.1)
            if not isinstance(name, str) or not name.strip():
                raise WorldCompositionError("vessel id must be non-empty")
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, (int, float))
                or not 0.0001 <= capacity <= 0.1
            ):
                raise WorldCompositionError("vessel capacity_L must be in [0.0001, 0.1]")
            vessels.append({"id": name, "capacity_L": float(capacity)})
        if len({v["id"] for v in vessels}) != len(vessels):
            raise WorldCompositionError("vessel IDs must be unique")
        connections = payload.get("connections", [])
        if not isinstance(connections, list) or len(connections) > MAX_CONNECTIONS:
            raise WorldCompositionError(f"connections must be a list of at most {MAX_CONNECTIONS}")
        if any(not isinstance(c, Mapping) for c in connections):
            raise WorldCompositionError("each connection must be an object")
        task_value = payload.get("task")
        if not isinstance(task_value, Mapping):
            raise WorldCompositionError("task must be an object")
        return cls(
            schema_version=schema_version,
            composition_id=composition_id,
            world_split=world_split,
            components=components,
            task=CompositionTaskRequest.from_dict(task_value),
            vessels=tuple(vessels),
            connections=tuple(deepcopy(dict(c)) for c in connections),
        )

    @property
    def component_kinds(self) -> tuple[str, ...]:
        return tuple(component.kind for component in self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "composition_id": self.composition_id,
            "world_split": self.world_split,
            "components": [component.to_dict() for component in self.components],
            "task": self.task.to_dict(),
            "vessels": deepcopy(list(self.vessels)),
            "connections": deepcopy(list(self.connections)),
        }


@dataclass(frozen=True)
class _CompositionPattern:
    pattern_id: str
    component_kinds: frozenset[str]
    template_task_id: str
    minimum_operation_budget: int
    minimum_nonfinal_measurements: int = 0
    minimum_process_time_s: float = 0.0


@dataclass(frozen=True)
class CompiledWorldComposition:
    """Runtime-ready task/scenario pair plus the reader-facing public surface."""

    spec: WorldCompositionSpec
    task_spec: TaskSpec
    scenario_spec: ScenarioSpec
    public_surface: dict[str, Any]
    runtime_task_profile_id: str
    compatibility: WorldCompatibilityReport

    def to_public_dict(self) -> dict[str, Any]:
        return deepcopy(self.public_surface)

    def env_kwargs(self, *, seed: int | None = None) -> dict[str, Any]:
        return {
            "composition": self,
            "seed": self.task_spec.seeds[0] if seed is None else int(seed),
        }


def _resolve_pattern(spec: WorldCompositionSpec) -> _CompositionPattern:
    capabilities = [COMPONENT_CAPABILITIES[kind] for kind in spec.component_kinds]
    selected = max(capabilities, key=lambda c: c.priority)
    template = (
        selected.reactive_template
        if "reaction" in spec.component_kinds and selected.reactive_template
        else selected.template
    )
    return _CompositionPattern(
        "component-network",
        frozenset(spec.component_kinds),
        template or "reaction-to-assay",
        max(c.minimum_operations for c in capabilities),
        max(c.minimum_measurements for c in capabilities),
        max(c.minimum_time_s for c in capabilities),
    )


def _surface_template(spec: WorldCompositionSpec, pattern: _CompositionPattern) -> TaskSpec:
    template = get_task(pattern.template_task_id)
    operations = component_operations(spec.component_kinds)
    if len(spec.vessels) > 1 or spec.connections:
        operations = (*operations, *NETWORK_OPERATIONS)
    instruments = tuple(
        dict.fromkeys(
            instrument
            for kind in spec.component_kinds
            if COMPONENT_CAPABILITIES[kind].template is not None
            for instrument in get_task(
                COMPONENT_CAPABILITIES[kind].template or pattern.template_task_id
            ).allowed_instruments
        )
    )
    return replace(
        template,
        allowed_operations=operations,
        allowed_instruments=instruments,
        episode_mode="single_experiment" if len(spec.vessels) > 1 else template.episode_mode,
        termination_policy="final-assay-or-budget"
        if len(spec.vessels) > 1
        else template.termination_policy,
    )


def _diagnostic(
    code: str,
    path: str,
    message: str,
    *components: str,
) -> WorldCompositionDiagnostic:
    return WorldCompositionDiagnostic(
        code=code,
        path=path,
        message=message,
        components=tuple(components),
    )


def _validate_connections(
    spec: WorldCompositionSpec, diagnostics: list[WorldCompositionDiagnostic]
) -> None:
    components = {c.id: c for c in spec.components}
    seen: set[str] = set()
    for index, connection in enumerate(spec.connections):
        path = f"connections[{index}]"
        name = connection.get("id")
        if not isinstance(name, str) or not name.strip() or name in seen:
            diagnostics.append(
                _diagnostic(
                    "invalid_connection", path, "connection IDs must be non-empty and unique"
                )
            )
        else:
            seen.add(name)
        if set(connection) != {"id", "source", "target", "unit"}:
            diagnostics.append(
                _diagnostic(
                    "invalid_connection",
                    path,
                    "connection requires exactly id, source, target, unit",
                )
            )
        if connection.get("unit") != "mol":
            diagnostics.append(
                _diagnostic("unit_mismatch", path, "material ports use mol; carrier volume is L")
            )
        endpoints = []
        for direction in ("source", "target"):
            endpoint = connection.get(direction)
            if not isinstance(endpoint, Mapping) or set(endpoint) != {"component", "port"}:
                diagnostics.append(
                    _diagnostic("invalid_port", path, "endpoint requires component and port")
                )
                continue
            component = components.get(str(endpoint["component"]))
            if component is None:
                diagnostics.append(_diagnostic("invalid_port", path, "unknown component instance"))
                continue
            endpoints.append(component)
            allowed = (
                COMPONENT_CAPABILITIES[component.kind].outputs if direction == "source" else ("in",)
            )
            if component.kind == "observation" or endpoint["port"] not in allowed:
                diagnostics.append(
                    _diagnostic(
                        "invalid_port",
                        path,
                        "port direction or material capability mismatch",
                        component.id,
                    )
                )
        if len(endpoints) == 2 and endpoints[0].vessel == endpoints[1].vessel:
            diagnostics.append(
                _diagnostic(
                    "invalid_connection", path, "material connections must join distinct vessels"
                )
            )


def _convert_authored_value(value: float, source_unit: str, target_unit: str) -> float:
    if source_unit in _CUSTOM_UNIT_TABLE or target_unit in _CUSTOM_UNIT_TABLE:
        try:
            source_dimension, source_scale = _CUSTOM_UNIT_TABLE[source_unit]
            target_dimension, target_scale = _CUSTOM_UNIT_TABLE[target_unit]
        except KeyError as exc:
            raise ValueError(f"cannot convert {source_unit!r} to {target_unit!r}") from exc
        if source_dimension != target_dimension:
            raise ValueError(f"cannot convert {source_unit!r} to {target_unit!r}")
        return value * source_scale / target_scale
    source = unit_spec(source_unit)
    target = unit_spec(target_unit)
    if source.dimension != target.dimension:
        raise ValueError(f"cannot convert {source_unit!r} to {target_unit!r}")
    return convert_value(value, source_unit, target_unit)


def _parameter_numeric_values(
    value: Any,
    *,
    rule: _ParameterRule,
    path: str,
    diagnostics: list[WorldCompositionDiagnostic],
    component_kind: str,
) -> tuple[float, ...] | None:
    authored_unit = rule.unit
    raw_value = value
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"value", "unit"})
        if unknown or "value" not in value or "unit" not in value:
            diagnostics.append(
                _diagnostic(
                    "invalid_parameter",
                    path,
                    "unit-bearing parameters require exactly 'value' and 'unit'",
                    component_kind,
                )
            )
            return None
        raw_value = value["value"]
        authored_unit = str(value["unit"])
    expected_count = 2 if rule.value_kind == "range" else 1
    raw_items = raw_value if isinstance(raw_value, (list, tuple)) else (raw_value,)
    if len(raw_items) != expected_count:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                f"parameter must contain {expected_count} numeric value(s)",
                component_kind,
            )
        )
        return None
    try:
        numbers = tuple(float(item) for item in raw_items)
    except (TypeError, ValueError):
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                "parameter values must be numeric",
                component_kind,
            )
        )
        return None
    if any(not isfinite(item) for item in numbers):
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                "parameter values must be finite",
                component_kind,
            )
        )
        return None
    if authored_unit is not None and rule.unit is not None:
        try:
            numbers = tuple(
                _convert_authored_value(item, authored_unit, rule.unit) for item in numbers
            )
        except ValueError:
            diagnostics.append(
                _diagnostic(
                    "unit_mismatch",
                    path,
                    f"unit {authored_unit!r} is incompatible with expected {rule.unit!r}",
                    component_kind,
                )
            )
            return None
    if rule.value_kind == "range" and numbers[0] >= numbers[1]:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                "range lower bound must be smaller than upper bound",
                component_kind,
            )
        )
        return None
    if rule.minimum is not None and min(numbers) < rule.minimum:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                f"parameter is below the supported lower bound {rule.minimum}",
                component_kind,
            )
        )
        return None
    if rule.maximum is not None and max(numbers) > rule.maximum:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                path,
                f"parameter exceeds the supported upper bound {rule.maximum}",
                component_kind,
            )
        )
        return None
    return numbers


def _bind_operation_bounds(
    component_kind: str,
    parameter_name: str,
    values: tuple[float, ...],
    operation_field_bounds: dict[tuple[str, str], tuple[float, float]],
) -> None:
    if len(values) != 2:
        return
    targets = {
        ("thermal", "temperature_range_K"): (("heat", "target_temperature_K"),),
        ("thermal", "duration_range_s"): (
            ("heat", "duration_s"),
            ("wait", "duration_s"),
        ),
        ("crystallization", "temperature_range_K"): (("cool_crystallize", "target_temperature_K"),),
        ("crystallization", "seed_mass_range_g"): (("seed_crystals", "seed_mass_g"),),
        ("distillation", "temperature_range_K"): (
            ("evaporate", "target_temperature_K"),
            ("distill", "target_temperature_K"),
        ),
        ("distillation", "reflux_ratio_range"): (("distill", "reflux_ratio"),),
        ("continuous_flow", "flow_rate_range_mL_min"): (("set_flow_rate", "flow_rate_mL_min"),),
        ("continuous_flow", "residence_time_range_s"): (("set_flow_rate", "residence_time_s"),),
        ("continuous_flow", "temperature_range_K"): (("run_flow", "target_temperature_K"),),
        ("electrochemistry", "potential_range_V"): (("set_potential", "potential_V"),),
        ("electrochemistry", "current_range_mA"): (("set_potential", "current_mA"),),
        ("electrochemistry", "duration_range_s"): (("electrolyze", "duration_s"),),
    }.get((component_kind, parameter_name), ())
    for target in targets:
        operation_field_bounds[target] = (values[0], values[1])


def _validate_component_parameters(
    spec: WorldCompositionSpec,
    diagnostics: list[WorldCompositionDiagnostic],
) -> dict[tuple[str, str], tuple[float, float]]:
    operation_field_bounds: dict[tuple[str, str], tuple[float, float]] = {}
    for component in spec.components:
        rules = _COMPONENT_PARAMETER_RULES[component.kind]
        for parameter_name in sorted(set(component.parameters) - set(rules)):
            diagnostics.append(
                _diagnostic(
                    "unsupported_parameter",
                    f"components.{component.kind}.parameters.{parameter_name}",
                    f"component {component.kind!r} does not own this parameter",
                    component.kind,
                )
            )
        for parameter_name, value in component.parameters.items():
            rule = rules.get(parameter_name)
            if rule is None:
                continue
            path = f"components.{component.kind}.parameters.{parameter_name}"
            if rule.value_kind == "string":
                if not isinstance(value, str) or not value.strip():
                    diagnostics.append(
                        _diagnostic(
                            "invalid_parameter",
                            path,
                            "parameter must be a non-empty string",
                            component.kind,
                        )
                    )
                continue
            if rule.value_kind == "choices":
                if not isinstance(value, (list, tuple)) or not value:
                    diagnostics.append(
                        _diagnostic(
                            "invalid_parameter",
                            path,
                            "parameter must be a non-empty list",
                            component.kind,
                        )
                    )
                    continue
                choices = tuple(str(item) for item in value)
                invalid = sorted(set(choices) - set(rule.choices))
                if invalid or len(set(choices)) != len(choices):
                    diagnostics.append(
                        _diagnostic(
                            "invalid_parameter",
                            path,
                            f"unsupported or duplicate choices: {invalid or list(choices)}",
                            component.kind,
                        )
                    )
                continue
            values = _parameter_numeric_values(
                value,
                rule=rule,
                path=path,
                diagnostics=diagnostics,
                component_kind=component.kind,
            )
            if values is None:
                continue
            if rule.value_kind == "integer" and any(not item.is_integer() for item in values):
                diagnostics.append(
                    _diagnostic(
                        "invalid_parameter",
                        path,
                        "parameter must be an integer",
                        component.kind,
                    )
                )
                continue
            _bind_operation_bounds(
                component.kind,
                parameter_name,
                values,
                operation_field_bounds,
            )
    return operation_field_bounds


def _resource_number(
    resources: Mapping[str, Any],
    name: str,
    diagnostics: list[WorldCompositionDiagnostic],
    *,
    integer: bool = False,
) -> float | int | None:
    if name not in resources:
        return None
    value = resources[name]
    if isinstance(value, bool):
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                f"task.resources.{name}",
                "resource value must be numeric",
            )
        )
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                f"task.resources.{name}",
                "resource value must be numeric",
            )
        )
        return None
    if not isfinite(numeric) or numeric < 0.0:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                f"task.resources.{name}",
                "resource value must be finite and non-negative",
            )
        )
        return None
    if integer:
        if not numeric.is_integer():
            diagnostics.append(
                _diagnostic(
                    "invalid_parameter",
                    f"task.resources.{name}",
                    "resource value must be an integer",
                )
            )
            return None
        return int(numeric)
    return numeric


def check_world_composition_compatibility(
    value: Mapping[str, Any] | WorldCompositionSpec | CompiledWorldComposition,
) -> WorldCompatibilityReport:
    """Return all deterministic compatibility diagnostics without starting an environment."""

    if isinstance(value, CompiledWorldComposition):
        return value.compatibility
    spec = (
        value if isinstance(value, WorldCompositionSpec) else WorldCompositionSpec.from_dict(value)
    )
    diagnostics: list[WorldCompositionDiagnostic] = []
    component_set = frozenset(spec.component_kinds)
    if "observation" not in component_set:
        diagnostics.append(
            _diagnostic(
                "missing_dependency",
                "components",
                "every executable v1 composition requires the observation component",
            )
        )
    if "reaction" in component_set and not component_set.intersection(
        {"thermal", "electrochemistry"}
    ):
        diagnostics.append(
            _diagnostic(
                "missing_dependency",
                "components",
                "reaction requires a thermal or electrochemical process driver",
                "reaction",
            )
        )
    for component_kind, required in _COMPONENT_DEPENDENCIES.items():
        if component_kind not in component_set:
            continue
        for missing in sorted(required - component_set):
            diagnostics.append(
                _diagnostic(
                    "missing_dependency",
                    "components",
                    f"component {component_kind!r} requires component {missing!r}",
                    component_kind,
                    missing,
                )
            )

    # Physical ownership and dependencies are local to a vessel. A second
    # reactor is not a second owner of the first reactor's material state.
    state_owners: dict[str, str] = {}
    reaction_families = {
        str(c.parameters["family"]) for c in spec.components
        if c.kind == "reaction" and "family" in c.parameters
    }
    if len(reaction_families) > 1:
        diagnostics.append(_diagnostic(
            "interface_mismatch", "components",
            "connected vessels require one shared reaction family and species mechanism",
        ))
    vessel_bounds: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    declared_vessels = {v["id"] for v in spec.vessels}
    for component in spec.components:
        if component.vessel not in declared_vessels:
            diagnostics.append(
                _diagnostic("unknown_vessel", "components", component.vessel, component.id)
            )
        for state_id in _STATE_OWNERS_BY_COMPONENT[component.kind]:
            key = f"{component.vessel}:{state_id}"
            previous = state_owners.get(key)
            if previous is not None:
                diagnostics.append(
                    _diagnostic(
                        "conflicting_state_owner",
                        "components",
                        f"state {key!r} is owned by {previous!r} and {component.id!r}",
                        previous,
                        component.id,
                    )
                )
            state_owners[key] = component.id
    for vessel in sorted(declared_vessels):
        local = tuple(c for c in spec.components if c.vessel == vessel)
        kinds = {c.kind for c in local}
        if not local:
            diagnostics.append(
                _diagnostic("missing_dependency", "vessels", f"vessel {vessel!r} has no components")
            )
        for c in local:
            for missing in sorted(_COMPONENT_DEPENDENCIES.get(c.kind, frozenset()) - kinds):
                diagnostics.append(
                    _diagnostic(
                        "missing_dependency",
                        "components",
                        f"{c.id!r} requires {missing!r} in vessel {vessel!r}",
                        c.id,
                    )
                )
        if "reaction" in kinds and not kinds.intersection({"thermal", "electrochemistry"}):
            diagnostics.append(
                _diagnostic(
                    "missing_dependency",
                    "components",
                    f"reaction in {vessel!r} needs a local driver",
                )
            )
        vessel_bounds[vessel] = _validate_component_parameters(
            replace(spec, components=local), diagnostics
        )
    _validate_connections(spec, diagnostics)
    operation_field_bounds = (
        next(iter(vessel_bounds.values()), {}) if len(vessel_bounds) == 1 else {}
    )
    pattern = _resolve_pattern(spec)
    minimum_resources: dict[str, float | int] = {}
    template = _surface_template(spec, pattern)
    request = spec.task
    operations = request.operations or template.allowed_operations
    instruments = request.instruments or template.allowed_instruments
    invalid_operations = sorted(set(operations) - set(template.allowed_operations))
    if invalid_operations:
        diagnostics.append(
            _diagnostic(
                "unsupported_operation",
                "task.operations",
                f"operations are outside the compiled component surface: {invalid_operations}",
            )
        )
    invalid_instruments = sorted(set(instruments) - set(template.allowed_instruments))
    if invalid_instruments:
        diagnostics.append(
            _diagnostic(
                "unsupported_instrument",
                "task.instruments",
                f"instruments are outside the compiled component surface: {invalid_instruments}",
            )
        )
    for component_kind in spec.component_kinds:
        missing_operations = sorted(
            _REQUIRED_ALL_OPERATIONS.get(component_kind, frozenset()) - set(operations)
        )
        if missing_operations:
            diagnostics.append(
                _diagnostic(
                    "lifecycle_hole",
                    "task.operations",
                    f"component {component_kind!r} has no complete executable path; missing "
                    f"{missing_operations}",
                    component_kind,
                )
            )
        required_any = _REQUIRED_ANY_OPERATIONS.get(component_kind)
        if required_any and not set(operations).intersection(required_any):
            diagnostics.append(
                _diagnostic(
                    "lifecycle_hole",
                    "task.operations",
                    f"component {component_kind!r} requires at least one of {sorted(required_any)}",
                    component_kind,
                )
            )
    if "separation" in component_set and "add_phase" not in operations:
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.operations",
                "phase separation requires add_phase to create a separable phase system",
                "phase",
                "separation",
            )
        )
    if not {"terminate", "measure"}.issubset(operations):
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.operations",
                "operations must include terminate and measure for lifecycle closure",
            )
        )
    if "final_assay" not in instruments:
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.instruments",
                "instruments must include final_assay for terminal evaluation",
            )
        )
    nonfinal_instruments = tuple(
        instrument for instrument in instruments if instrument != "final_assay"
    )
    if pattern.minimum_nonfinal_measurements and not nonfinal_instruments:
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.instruments",
                "this workflow requires at least one non-final process instrument",
            )
        )

    component_by_kind = {component.kind: component for component in spec.components}
    observation_instruments = (
        component_by_kind["observation"].parameters.get("instruments")
        if "observation" in component_by_kind
        else None
    )
    if len(spec.vessels) > 1:
        observation_instruments = None
    if isinstance(observation_instruments, (list, tuple)):
        outside_observation = sorted(set(instruments) - set(observation_instruments))
        if outside_observation:
            diagnostics.append(
                _diagnostic(
                    "interface_mismatch",
                    "task.instruments",
                    "task instruments are not provided by the observation component: "
                    f"{outside_observation}",
                    "observation",
                )
            )
    separation_operations = component_by_kind.get("separation")
    declared_separation_operations = (
        None
        if separation_operations is None
        else separation_operations.parameters.get("operations")
    )
    if isinstance(declared_separation_operations, (list, tuple)):
        task_separation_operations = set(operations).intersection(SEPARATION_OPERATIONS)
        outside_separation = sorted(
            task_separation_operations - set(declared_separation_operations)
        )
        if outside_separation:
            diagnostics.append(
                _diagnostic(
                    "interface_mismatch",
                    "task.operations",
                    "task operations are not provided by the separation component: "
                    f"{outside_separation}",
                    "separation",
                )
            )

    if request.termination is not None and request.termination != template.termination_policy:
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.termination",
                f"termination policy must be {template.termination_policy!r} for this pattern",
            )
        )
    if request.episode_mode is not None and request.episode_mode != template.episode_mode:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                "task.episode_mode",
                f"episode mode must be {template.episode_mode!r} for this pattern",
            )
        )
    if request.observations is not None and request.observations != template.observation_policy:
        diagnostics.append(
            _diagnostic(
                "invalid_parameter",
                "task.observations",
                f"observation policy must be {template.observation_policy!r} for this pattern",
            )
        )
    if request.evaluation_metrics is not None:
        unsupported_metrics = sorted(
            set(request.evaluation_metrics) - set(template.success_metrics)
        )
        if unsupported_metrics:
            diagnostics.append(
                _diagnostic(
                    "invalid_parameter",
                    "task.evaluation.metrics",
                    f"evaluation metrics are outside the pattern surface: {unsupported_metrics}",
                )
            )

    budget = request.budget or template.budget
    minimum_resources = {
        "operation_budget": pattern.minimum_operation_budget,
        "instrument_uses": 1 + pattern.minimum_nonfinal_measurements,
        "final_assays": 1,
        "time_s": pattern.minimum_process_time_s,
    }
    contracts = instrument_contracts()
    minimum_sample_volume = contracts["final_assay"].sample_consumption_L
    if pattern.minimum_nonfinal_measurements and nonfinal_instruments:
        minimum_nonfinal_sample = min(
            contracts[instrument].sample_consumption_L for instrument in nonfinal_instruments
        )
        minimum_sample_volume += pattern.minimum_nonfinal_measurements * minimum_nonfinal_sample
    minimum_resources["sample_volume_L"] = minimum_sample_volume
    if budget < pattern.minimum_operation_budget:
        diagnostics.append(
            _diagnostic(
                "resource_impossibility",
                "task.budget",
                f"budget {budget} cannot reach the minimum lifecycle path of "
                f"{pattern.minimum_operation_budget} operations",
            )
        )
    operation_budget = _resource_number(
        request.resources,
        "operation_budget",
        diagnostics,
        integer=True,
    )
    if operation_budget is not None and operation_budget != budget:
        diagnostics.append(
            _diagnostic(
                "resource_impossibility",
                "task.resources.operation_budget",
                "operation budget must match task budget",
            )
        )
    resource_checks = (
        ("sample_volume_L", minimum_sample_volume, False),
        ("time_s", pattern.minimum_process_time_s, False),
        ("instrument_uses", 1 + pattern.minimum_nonfinal_measurements, True),
        ("final_assays", 1, True),
    )
    for resource_name, minimum, integer in resource_checks:
        available = _resource_number(
            request.resources,
            resource_name,
            diagnostics,
            integer=integer,
        )
        if available is not None and available < minimum:
            diagnostics.append(
                _diagnostic(
                    "resource_impossibility",
                    f"task.resources.{resource_name}",
                    f"declared resource {available} is below the reachable minimum {minimum}",
                )
            )

    return WorldCompatibilityReport(
        pattern=pattern.pattern_id,
        diagnostics=tuple(diagnostics),
        state_owners=state_owners,
        minimum_resources=minimum_resources,
        operation_field_bounds=operation_field_bounds,
        vessel_field_bounds=vessel_bounds,
    )


def compile_world_composition(
    value: Mapping[str, Any] | WorldCompositionSpec | CompiledWorldComposition,
) -> CompiledWorldComposition:
    """Compile one declarative request into a runtime-ready public contract."""

    if isinstance(value, CompiledWorldComposition):
        return value
    spec = (
        value if isinstance(value, WorldCompositionSpec) else WorldCompositionSpec.from_dict(value)
    )
    compatibility = check_world_composition_compatibility(spec)
    if not compatibility.compatible:
        details = "; ".join(
            f"[{item.code}] {item.path}: {item.message}" for item in compatibility.diagnostics
        )
        raise WorldCompositionError(
            f"world composition compatibility failed: {details}",
            diagnostics=compatibility.diagnostics,
        )
    pattern = _resolve_pattern(spec)
    template = _surface_template(spec, pattern)
    request = spec.task
    operations = request.operations or template.allowed_operations
    invalid_operations = sorted(set(operations) - set(template.allowed_operations))
    if invalid_operations:
        raise WorldCompositionError(
            f"task operations are outside the compiled component surface: {invalid_operations}"
        )
    if not {"terminate", "measure"}.issubset(operations):
        raise WorldCompositionError(
            "task operations must include terminate and measure for lifecycle closure"
        )
    instruments = request.instruments or template.allowed_instruments
    invalid_instruments = sorted(set(instruments) - set(template.allowed_instruments))
    if invalid_instruments:
        raise WorldCompositionError(
            f"task instruments are outside the compiled component surface: {invalid_instruments}"
        )
    if "final_assay" not in instruments:
        raise WorldCompositionError("task instruments must include final_assay")
    budget = request.budget or template.budget
    resources = {
        "operation_budget": budget,
        "stock_accounting": "task-and-constitution",
        "sample_accounting": "per-instrument",
        "time_accounting": "per-operation-and-instrument",
        **deepcopy(request.resources),
    }
    if int(resources.get("operation_budget", budget)) != budget:
        raise WorldCompositionError("task resources.operation_budget must match task budget")
    metrics = request.evaluation_metrics or template.success_metrics
    threshold = (
        template.threshold if request.evaluation_threshold is None else request.evaluation_threshold
    )
    scenario_spec = get_scenario(template.scenario_id, split=spec.world_split)
    task_spec = TaskSpec(
        task_id=spec.composition_id,
        env_id=template.env_id,
        world_law_id=template.world_law_id,
        scenario_id=scenario_spec.scenario_id,
        initial_state_id=scenario_spec.initial_state_id,
        world_split=spec.world_split,
        objective=request.objective or template.objective,
        budget=budget,
        seeds=request.seeds or template.seeds,
        threshold=threshold,
        episode_mode=request.episode_mode or template.episode_mode,
        allowed_operations=operations,
        allowed_instruments=instruments,
        observation_policy=request.observations or template.observation_policy,
        termination_policy=request.termination or template.termination_policy,
        success_metrics=metrics,
        safety_limit=(
            template.safety_limit if request.safety_limit is None else request.safety_limit
        ),
        difficulty=request.difficulty or template.difficulty,
        description=(
            request.description or f"Declaratively composed ChemWorld world: {spec.composition_id}."
        ),
        tags=tuple(dict.fromkeys((*template.tags, "composed-world", *request.tags))),
        kernel_maturity=default_kernel_maturity(
            operations,
            allowed_instruments=instruments,
        ),
    )
    interfaces = tuple(
        sorted(
            {
                interface
                for component in spec.components
                for interface in _INTERFACES_BY_COMPONENT[component.kind]
            }
        )
    )
    public_surface = {
        "schema_version": WORLD_COMPOSITION_SCHEMA_VERSION,
        "composition_id": spec.composition_id,
        "compatibility": compatibility.to_dict(),
        "world": {
            "components": [component.to_dict() for component in spec.components],
            "vessels": deepcopy(list(spec.vessels)),
            "connections": deepcopy(list(spec.connections)),
            "execution": {
                "clock": "explicit local advancement; inactive vessels remain unchanged",
                "mechanism": "one shared species mechanism across connected vessels",
                "material_ports": (
                    "in: liquid or representative slurry; out: representative inventory; "
                    "named distillation outlets: one liquid phase"
                ),
                "transfer": "instantaneous; explicit mixing; heat-capacity weighted temperature",
            },
            "interfaces": list(interfaces),
            "family": scenario_spec.family,
            "split": spec.world_split,
            "initial_state": scenario_spec.initial_state_id,
        },
        "task": {
            "objective": task_spec.objective,
            "operations": list(task_spec.allowed_operations),
            "instruments": list(task_spec.allowed_instruments),
            "observations": task_spec.observation_policy,
            "resources": resources,
            "termination": task_spec.termination_policy,
            "evaluation": {
                "metrics": list(task_spec.success_metrics),
                "threshold": task_spec.threshold,
            },
            "episode_mode": task_spec.episode_mode,
            "safety_limit": task_spec.safety_limit,
        },
    }
    return CompiledWorldComposition(
        spec=spec,
        task_spec=task_spec,
        scenario_spec=scenario_spec,
        public_surface=public_surface,
        runtime_task_profile_id=pattern.template_task_id,
        compatibility=compatibility,
    )


__all__ = [
    "SUPPORTED_COMPONENT_KINDS",
    "WORLD_COMPOSITION_SCHEMA_VERSION",
    "CompiledWorldComposition",
    "CompositionTaskRequest",
    "WorldCompatibilityReport",
    "WorldComponentRequest",
    "WorldCompositionDiagnostic",
    "WorldCompositionError",
    "WorldCompositionSpec",
    "check_world_composition_compatibility",
    "compile_world_composition",
]
