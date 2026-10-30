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
    MAX_CONNECTIONS,
    MAX_VESSELS,
    NETWORK_OPERATIONS,
    component_operations,
)
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.domain_modules import ParameterRule as _ParameterRule
from chemworld.world.instruments import instrument_contracts
from chemworld.world.operations import (
    SEPARATION_OPERATIONS,
)
from chemworld.world.parameters import SUPPORTED_SPLITS
from chemworld.world.physical_world import CompositionPhysics
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
_REACTION_PROCESS_DRIVERS = frozenset({"thermal", "electrochemistry", "gas_reactor_network"})
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "composition_id",
        "world_split",
        "components",
        "task",
        "vessels",
        "connections",
        "physics",
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
    def from_dict(
        cls, payload: Mapping[str, Any], *, modules: DomainModuleRegistry | None = None
    ) -> WorldComponentRequest:
        _require_exact_keys(payload, _COMPONENT_KEYS, "component")
        kind = str(payload.get("kind", "")).strip()
        if kind not in (modules or DomainModuleRegistry.default()).components():
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
    def from_dict(cls, payload: Mapping[str, Any], *, modules=None) -> CompositionTaskRequest:
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
            else ()
            if instruments_value == []
            else _string_tuple(instruments_value, "task instruments")
        )
        if instruments is not None:
            unknown_instruments = sorted(
                set(instruments)
                - set((modules or DomainModuleRegistry.default()).instrument_specs())
            )
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
    physics: CompositionPhysics | None = None
    modules: DomainModuleRegistry = field(
        default_factory=DomainModuleRegistry.default, repr=False, compare=False
    )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, modules: DomainModuleRegistry | None = None
    ) -> WorldCompositionSpec:
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
        if modules is None:
            modules = DomainModuleRegistry.default()
            # This apparatus is opt-in: its operation layout and execution
            # declaration must not alter existing worlds or saved sessions.
            if any(isinstance(component, Mapping)
                   and component.get("kind") == "thermal_fixed_bed"
                   for component in components_value):
                from chemworld.world.thermal_fixed_bed_module import THERMAL_FIXED_BED_MODULE

                modules = DomainModuleRegistry((*modules.modules, THERMAL_FIXED_BED_MODULE))
        components = tuple(
            WorldComponentRequest.from_dict(component, modules=modules)
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
            sorted(
                components,
                key=lambda item: (
                    _COMPONENT_ORDER.get(item.kind, len(_COMPONENT_ORDER)),
                    item.kind,
                    item.id,
                ),
            )
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
                or not 0 < capacity < float("inf")
            ):
                raise WorldCompositionError("vessel capacity_L must be finite and positive")
            if payload.get("physics") is None and not 0.0001 <= capacity <= 0.1:
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
            task=CompositionTaskRequest.from_dict(task_value, modules=modules),
            vessels=tuple(vessels),
            connections=tuple(deepcopy(dict(c)) for c in connections),
            modules=modules,
            physics=(
                None
                if payload.get("physics") is None
                else CompositionPhysics.from_dict(payload["physics"], modules=modules)
            ),
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
            **({"physics": self.physics.to_dict()} if self.physics is not None else {}),
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
    runtime_task_profile_id: str | None
    compatibility: WorldCompatibilityReport

    def to_public_dict(self) -> dict[str, Any]:
        return deepcopy(self.public_surface)

    def to_execution_ir(self, *, backend: str = "auto") -> dict[str, Any]:
        """Lower this compiled authoring surface to the native execution IR.

        The registry remains the sole source of module capability metadata;
        this method only binds the already-validated component instances and
        ownership map to that generated contract.
        """
        return self.spec.modules.lower_execution_ir(self, backend=backend)

    def env_kwargs(self, *, seed: int | None = None) -> dict[str, Any]:
        return {
            "composition": self,
            "seed": self.task_spec.seeds[0] if seed is None else int(seed),
        }

    def bind_task(
        self, request: Mapping[str, Any] | CompositionTaskRequest
    ) -> CompiledWorldComposition:
        """Rebind policy without replacing physical entities, laws or initial state."""
        task = (
            request
            if isinstance(request, CompositionTaskRequest)
            else CompositionTaskRequest.from_dict(request, modules=self.spec.modules)
        )
        return compile_world_composition(replace(self.spec, task=task))


def _resolve_pattern(spec: WorldCompositionSpec) -> _CompositionPattern:
    # Old compositions are boundary presets. Explicit physics never selects a
    # legacy scenario by component priority.
    capabilities = [spec.modules.components()[kind] for kind in spec.component_kinds]
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
    operations = component_operations(spec.component_kinds, modules=spec.modules)
    if len(spec.vessels) > 1 or spec.connections:
        operations = (*operations, *NETWORK_OPERATIONS)
    instruments = tuple(
        dict.fromkeys(
            instrument
            for kind in spec.component_kinds
            if spec.modules.components()[kind].template is not None
            for instrument in get_task(
                spec.modules.components()[kind].template or pattern.template_task_id
            ).allowed_instruments
        )
    )
    return replace(
        template,
        domain_modules=spec.modules,
        allowed_operations=operations,
        allowed_instruments=instruments,
        episode_mode="single_experiment" if len(spec.vessels) > 1 else template.episode_mode,
        termination_policy="final-assay-or-budget"
        if len(spec.vessels) > 1
        else template.termination_policy,
    )


def physical_operations(spec: WorldCompositionSpec, vessel: str | None = None) -> tuple[str, ...]:
    if spec.physics is not None and vessel is None:
        return tuple(
            dict.fromkeys(
                [
                    op
                    for local in spec.physics.initial_states
                    for op in physical_operations(spec, local)
                ]
                + (list(NETWORK_OPERATIONS) if len(spec.vessels) > 1 or spec.connections else [])
            )
        )
    kinds = tuple(c.kind for c in spec.components if vessel is None or c.vessel == vessel)
    operations = component_operations(
        kinds, modules=spec.modules, include_common=spec.physics is None
    )
    if spec.physics is not None:
        operations = (
            *spec.physics.available_operations(operations, spec.modules, vessel),
            "terminate",
        )
    if vessel is None and (len(spec.vessels) > 1 or spec.connections):
        operations = (*operations, *NETWORK_OPERATIONS)
    return tuple(dict.fromkeys(operations))


def physical_instruments(spec: WorldCompositionSpec, vessel: str | None = None) -> tuple[str, ...]:
    """Installed instruments, independent of the task's permission subset."""
    return tuple(
        dict.fromkeys(
            name
            for component in spec.components
            if (vessel is None or component.vessel == vessel) and component.kind == "observation"
            for name in component.parameters.get("instruments", ())
        )
    )


def component_instruments(
    spec: WorldCompositionSpec,
    vessel: str,
    *,
    fallback: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the instrument surface bound to one generic composition vessel.

    An observation component with no explicit list inherits the compiled task
    surface. Keeping that fallback in one helper makes compatibility preflight
    and runtime network construction agree for multi-vessel graphs.
    """

    return tuple(
        dict.fromkeys(
            instrument
            for component in spec.components
            if component.vessel == vessel and component.kind == "observation"
            for instrument in component.parameters.get("instruments", fallback)
        )
    )


def _native_task_defaults(spec: WorldCompositionSpec) -> TaskSpec:
    operations = physical_operations(spec)
    instruments = physical_instruments(spec)
    scenario = spec.physics.scenario_spec(spec.composition_id, spec.world_split)
    return TaskSpec(
        task_id=spec.composition_id,
        env_id="ChemWorld",
        world_law_id=scenario.world_law_id,
        scenario_id=scenario.scenario_id,
        initial_state_id=scenario.initial_state_id,
        world_split=spec.world_split,
        objective="collect observations",
        budget=32,
        seeds=(0,),
        threshold=0.0,
        episode_mode="single_experiment",
        allowed_operations=operations,
        allowed_instruments=instruments,
        observation_policy="instrument-only",
        termination_policy="explicit-stop-or-budget",
        success_metrics=("trajectory_validity",),
        safety_limit=0.65,
        difficulty="authored",
        description="Explicit physical world.",
        tags=("composed-world",),
        domain_modules=spec.modules,
        kernel_maturity=default_kernel_maturity(operations, allowed_instruments=instruments),
    )


def _check_native_task(
    spec: WorldCompositionSpec,
    template: TaskSpec,
    diagnostics: list[WorldCompositionDiagnostic],
) -> dict[str, float | int]:
    request = spec.task
    operations = request.operations or template.allowed_operations
    instruments = (
        template.allowed_instruments if request.instruments is None else request.instruments
    )
    for field_name, requested, provided in (
        ("operations", operations, template.allowed_operations),
        ("instruments", instruments, template.allowed_instruments),
    ):
        missing = set(requested) - set(provided)
        if missing:
            diagnostics.append(
                _diagnostic(
                    "missing_capability",
                    f"task.{field_name}",
                    f"world does not provide {sorted(missing)}",
                )
            )
    termination = request.termination or template.termination_policy
    if termination not in {"explicit-stop-or-budget", "final-assay-or-budget"}:
        diagnostics.append(
            _diagnostic("lifecycle_hole", "task.termination", "unsupported termination policy")
        )
    if "terminate" not in operations:
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole", "task.operations", "explicit termination requires terminate"
            )
        )
    assay = termination == "final-assay-or-budget"
    terminal_instruments = tuple(
        name for name in instruments
        if spec.modules.instrument_specs()[name].definition.requires_terminated
    )
    if assay and (not terminal_instruments or "measure" not in operations):
        diagnostics.append(
            _diagnostic(
                "lifecycle_hole",
                "task.termination",
                "final assay requires its instrument and measure",
            )
        )
    if instruments and not {"measure", "queue_lab_measure"}.intersection(operations):
        diagnostics.append(
            _diagnostic("missing_capability", "task.operations", "instruments require measure")
        )
    for vessel in spec.physics.initial_states:
        for name in set(physical_instruments(spec, vessel)) & set(instruments):
            contract = spec.modules.instrument_specs()[name].contract
            required_phase = contract.input_state_schema.get("required_phase")
            if (
                required_phase is not None
                and spec.physics.initial_states[vessel].phase != required_phase
            ):
                diagnostics.append(_diagnostic(
                    "missing_capability", "task.instruments",
                    f"{name} in {vessel} requires phase {required_phase}",
                ))
            if contract.physical_duration_s and not spec.physics.supports_measurement_clock(
                spec.physics.initial_states[vessel], spec.modules
            ):
                diagnostics.append(_diagnostic(
                    "missing_capability", "task.instruments",
                    f"{name} in {vessel} requires a supported physical clock",
                ))
    metrics = request.evaluation_metrics or template.success_metrics
    supported = {"trajectory_validity", "final_assay_score"}
    supported.update(
        key
        for name in instruments
        for key in spec.modules.instrument_specs()[name].definition.observable_keys
    )
    if set(metrics) - supported:
        diagnostics.append(
            _diagnostic(
                "missing_capability",
                "task.evaluation.metrics",
                "metrics need an installed observation channel",
            )
        )
    if "final_assay_score" in metrics:
        roles = {role for s in spec.physics.network.species for role in s.observable_aliases}
        if "final_assay" not in instruments or not {"reactant", "target"} <= roles:
            diagnostics.append(
                _diagnostic(
                    "missing_capability",
                    "task.evaluation",
                    "assay scoring requires instrument and declared reactant/target roles",
                )
            )
    if set(metrics) - {"trajectory_validity"}:
        from chemworld.world.scoring import OBJECTIVES

        if request.objective not in OBJECTIVES:
            diagnostics.append(
                _diagnostic(
                    "invalid_parameter",
                    "task.objective",
                    "scored metrics require an explicit supported objective",
                )
            )
    minimum = {
        "operation_budget": 2 if assay else 1,
        "time_s": 0.0,
        "instrument_uses": int(assay),
        "final_assays": int(assay),
        "sample_volume_L": min((
            instrument_contracts(modules=spec.modules)[name].sample_consumption_L
            for name in terminal_instruments
        ), default=0.0) if assay
        else 0.0,
    }
    budget = request.budget or template.budget
    if budget < minimum["operation_budget"]:
        diagnostics.append(
            _diagnostic(
                "resource_impossibility",
                "task.budget",
                "budget cannot complete the declared ending",
            )
        )
    for name, lower in minimum.items():
        amount = _resource_number(
            request.resources,
            name,
            diagnostics,
            integer=name in {"operation_budget", "instrument_uses", "final_assays"},
        )
        if amount is not None and (
            amount < lower or (name == "operation_budget" and amount != budget)
        ):
            diagnostics.append(
                _diagnostic(
                    "resource_impossibility",
                    f"task.resources.{name}",
                    "resource conflicts with task budget or ending",
                )
            )
    return minimum


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
        required = {"id", "source", "target", "unit"}
        if (set(connection) - {"scheduling"} != required
                or ("scheduling" in connection and connection["scheduling"] != "manual")):
            diagnostics.append(
                _diagnostic(
                    "invalid_connection",
                    path,
                    "connection requires id, source, target, unit and optional scheduling=manual",
                )
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
                spec.modules.components()[component.kind].outputs
                if direction == "source"
                else spec.modules.components()[component.kind].input_ports
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
        if len(endpoints) == 2:
            try:
                spec.modules.validate_connection(
                    endpoints[0].kind,
                    connection["source"]["port"],
                    endpoints[1].kind,
                    connection["target"]["port"],
                    connection.get("unit"),
                )
            except ValueError as exc:
                diagnostics.append(
                    _diagnostic(
                        "unit_mismatch" if "units" in str(exc) else "invalid_port", path, str(exc)
                    )
                )
        if len(endpoints) == 2 and endpoints[0].vessel == endpoints[1].vessel:
            diagnostics.append(
                _diagnostic(
                    "invalid_connection", path, "material connections must join distinct vessels"
                )
            )


def _convert_authored_value(value: float, source_unit: str, target_unit: str) -> float:
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
        ("lab_schedule", "duration_range_s"): (("wait", "duration_s"),),
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
        ("finite_transport", "duration_range_s"): (("run_flow", "duration_s"),),
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
        rules = dict(spec.modules.components()[component.kind].parameters)
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
            # Native thermal laws declare their own validity (e.g. NASA7 gas
            # kinetics). The historical liquid range is not a universal limit.
            if (component.kind == "thermal" and spec.physics is not None
                    and parameter_name == "temperature_range_K"):
                try:
                    bounds = spec.physics.declared_temperature_bounds_k(
                        spec.physics.initial_states[component.vessel], spec.modules)
                except ValueError as exc:
                    diagnostics.append(_diagnostic(
                        "invalid_advance_binding", f"components.{component.id}", str(exc),
                        component.kind,
                    ))
                    continue
                if bounds is not None:
                    rule = replace(rule, minimum=bounds[0], maximum=bounds[1])
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
            if rule.value_kind in {"choices", "instrument_ids"}:
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
                allowed = (
                    spec.modules.instrument_specs().keys()
                    if rule.value_kind == "instrument_ids"
                    else rule.choices
                )
                invalid = sorted(set(choices) - set(allowed))
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
    *,
    modules: DomainModuleRegistry | None = None,
) -> WorldCompatibilityReport:
    """Return all deterministic compatibility diagnostics without starting an environment."""

    if isinstance(value, CompiledWorldComposition):
        return value.compatibility
    spec = (
        value
        if isinstance(value, WorldCompositionSpec)
        else WorldCompositionSpec.from_dict(value, modules=modules)
    )
    diagnostics: list[WorldCompositionDiagnostic] = []
    component_set = frozenset(spec.component_kinds)
    reactive_transport_vessels: set[str] = set()
    if spec.physics is not None:
        from chemworld.runtime.reactive_water_treatment import reactive_treatment_record

        for component in spec.components:
            state = spec.physics.initial_states.get(component.vessel)
            if component.kind != "finite_transport" or state is None:
                continue
            record = reactive_treatment_record(state)
            model = None if record is None else record.settings.get("physical_model")
            if isinstance(model, Mapping) and isinstance(model.get("reactor_vessel_id"), str):
                # The transport provider below validates the complete declaration,
                # original network and owned tanks. Other liquid paths gain no driver.
                reactive_transport_vessels.add(model["reactor_vessel_id"])
    if spec.physics is None and "observation" not in component_set:
        diagnostics.append(
            _diagnostic(
                "missing_dependency",
                "components",
                "every executable v1 composition requires the observation component",
            )
        )
    if spec.physics is None:
        for component in spec.components:
            capability = spec.modules.components()[component.kind]
            if capability.requires_physics:
                diagnostics.append(
                    _diagnostic(
                        "missing_physics_binding",
                        f"components.{component.id}",
                        f"component {component.kind!r} requires an explicit physics binding",
                        component.kind,
                    )
                )
    if ("reaction" in component_set
            and not component_set.intersection(_REACTION_PROCESS_DRIVERS)
            and not reactive_transport_vessels):
        diagnostics.append(
            _diagnostic(
                "missing_dependency",
                "components",
                "reaction requires a declared local or finite-system process driver",
                "reaction",
            )
        )
    for component_kind, capability in spec.modules.components().items():
        required = set(capability.dependencies)
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
        str(c.parameters["family"])
        for c in spec.components
        if c.kind == "reaction" and "family" in c.parameters
    }
    if len(reaction_families) > 1:
        diagnostics.append(
            _diagnostic(
                "interface_mismatch",
                "components",
                "connected vessels require one shared reaction family and species mechanism",
            )
        )
    vessel_bounds: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    declared_vessels = {v["id"] for v in spec.vessels}
    for component in spec.components:
        if component.vessel not in declared_vessels:
            diagnostics.append(
                _diagnostic("unknown_vessel", "components", component.vessel, component.id)
            )
        for state_id in spec.modules.components()[component.kind].state_owners:
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
            for missing in sorted(set(spec.modules.components()[c.kind].dependencies) - kinds):
                diagnostics.append(
                    _diagnostic(
                        "missing_dependency",
                        "components",
                        f"{c.id!r} requires {missing!r} in vessel {vessel!r}",
                        c.id,
                    )
                )
        if ("reaction" in kinds and not kinds.intersection(_REACTION_PROCESS_DRIVERS)
                and vessel not in reactive_transport_vessels):
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
    if spec.physics is not None:
        minimum = {}
        try:
            spec.physics.validate(spec)
        except (ValueError, KeyError, TypeError) as exc:
            diagnostics.append(_diagnostic("invalid_physics", "physics", str(exc)))
        else:
            minimum = _check_native_task(spec, _native_task_defaults(spec), diagnostics)
        return WorldCompatibilityReport(
            "native-components",
            tuple(diagnostics),
            state_owners,
            minimum,
            operation_field_bounds,
            vessel_bounds,
        )
    pattern = _resolve_pattern(spec)
    minimum_resources: dict[str, float | int] = {}
    template = _surface_template(spec, pattern)
    request = spec.task
    operations = request.operations or template.allowed_operations
    # Preserve an explicitly authored empty instrument subset. Compatibility
    # must inspect the same subset that compile_world_composition will bind;
    # otherwise preflight can accept a terminal workflow that the compiled
    # task cannot measure.
    instruments = (
        template.allowed_instruments if request.instruments is None else request.instruments
    )
    invalid_operations = sorted(set(operations) - set(template.allowed_operations))
    if invalid_operations:
        diagnostics.append(
            _diagnostic(
                "unsupported_operation",
                "task.operations",
                f"operations are outside the compiled component surface: {invalid_operations}",
            )
        )
    invalid_instruments = sorted(set(instruments) - set(spec.modules.instrument_specs()))
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
            set(spec.modules.components()[component_kind].required_all) - set(operations)
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
        required_any = set(spec.modules.components()[component_kind].required_any)
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
        available_by_vessel = {
            vessel["id"]: component_instruments(
                spec, vessel["id"], fallback=tuple(instruments)
            )
            for vessel in spec.vessels
        }
        available_instruments = set().union(*map(set, available_by_vessel.values()))
        outside_observation = sorted(set(instruments) - available_instruments)
        if outside_observation:
            diagnostics.append(
                _diagnostic(
                    "interface_mismatch",
                    "task.instruments",
                    "task instruments are not provided by any observation component: "
                    f"{outside_observation}",
                    "observation",
                )
            )
    elif isinstance(observation_instruments, (list, tuple)):
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
    contracts = instrument_contracts(modules=spec.modules)
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
    *,
    modules: DomainModuleRegistry | None = None,
) -> CompiledWorldComposition:
    """Compile one declarative request into a runtime-ready public contract."""

    if isinstance(value, CompiledWorldComposition):
        return value
    spec = (
        value
        if isinstance(value, WorldCompositionSpec)
        else WorldCompositionSpec.from_dict(value, modules=modules)
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
    pattern = _resolve_pattern(spec) if spec.physics is None else None
    template = (
        _surface_template(spec, pattern) if pattern is not None else _native_task_defaults(spec)
    )
    request = spec.task
    operations = request.operations or template.allowed_operations
    instruments = (
        template.allowed_instruments if request.instruments is None else request.instruments
    )
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
    scenario_spec = (
        get_scenario(template.scenario_id, split=spec.world_split)
        if spec.physics is None
        else spec.physics.scenario_spec(spec.composition_id, spec.world_split)
    )
    task_spec = TaskSpec(
        domain_modules=spec.modules,
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
                for interface in spec.modules.components()[component.kind].interfaces
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
        runtime_task_profile_id=pattern.template_task_id if pattern is not None else None,
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
