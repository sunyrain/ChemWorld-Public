"""Explicit, instance-scoped declarations for physical capabilities and actions.

Modules declare capabilities; they never own mutable world state. A registry is
constructed from a finite set of declarations, with no process-wide registration
or plugin discovery. Numeric compatibility layouts are separate, fixed contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Literal

from chemworld.foundation.ontology import Instrument, Operation
from chemworld.foundation.state_ownership import StateSlotSpec


def normalize_number_map(value):
    """Closed finite scalar bindings for one atomic multi-input action."""
    if not isinstance(value, Mapping) or not 1 <= len(value) <= 256:
        raise ValueError("numeric bindings need a nonempty mapping of at most 256 inputs")
    result = {}
    for key, number in value.items():
        if (not isinstance(key, str) or not key or key != key.strip()
                or isinstance(number, bool) or not isinstance(number, (int, float))):
            raise ValueError("numeric bindings need named finite numeric values")
        try:
            number = float(number)
        except OverflowError as exc:
            raise ValueError("numeric binding exceeds floating-point range") from exc
        if not isfinite(number):
            raise ValueError("numeric bindings need finite values")
        result[key] = number
    return result


@dataclass(frozen=True)
class ActionFieldSpec:
    name: str
    unit: str = "dimensionless"
    bounds: tuple[float, float] | None = None
    choices: tuple[Any, ...] = ()
    required: bool = True
    runtime_bounds: tuple[float, float] | None = None
    runtime_choices: tuple[Any, ...] = ()
    cumulative_limit: float | None = None
    choice_provider: Callable | None = None
    integer: bool = False
    value_kind: str = "number"

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(self.choices))
        if self.value_kind not in {"number", "number_map"}:
            raise ValueError("unknown native action value kind")
        if self.value_kind == "number_map" and (
                self.bounds or self.runtime_bounds or self.choices or self.runtime_choices
                or self.choice_provider is not None or self.integer
                or self.cumulative_limit is not None):
            raise ValueError("numeric maps cannot also declare scalar field policies")
        if self.choice_provider is not None and not callable(self.choice_provider):
            raise ValueError("dynamic field choices require an installed callable")
        if self.bounds is not None:
            object.__setattr__(self, "bounds", tuple(self.bounds))
            if (
                len(self.bounds) != 2
                or not all(isfinite(v) for v in self.bounds)
                or self.bounds[0] > self.bounds[1]
            ):
                raise ValueError("Action field bounds must be finite and ordered")

    def public_schema(self) -> dict[str, Any]:
        result: dict[str, Any] = {"field": self.name, "unit": self.unit, "required": self.required}
        if self.value_kind == "number_map":
            return {**result, "type": "object", "minProperties": 1, "maxProperties": 256,
                    "additionalProperties": {"type": "number"}}
        if self.bounds is not None:
            result.update(type="number", bounds={"low": self.bounds[0], "high": self.bounds[1]})
            if self.integer:
                result["type"] = "integer"
        if self.choices or self.choice_provider is not None:
            result["choices"] = list(self.choices)
        return result


@dataclass(frozen=True)
class OperationContract:
    operation_id: str
    module: str
    kind: str
    required_fields: tuple[str, ...]
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "module": self.module,
            "kind": self.kind,
            "required_fields": list(self.required_fields),
            "preconditions": list(self.preconditions),
        }


@dataclass(frozen=True)
class OperationSpec:
    operation: Operation
    capability: str
    service_id: str
    method: str
    kind: str = "primitive"
    fields: tuple[ActionFieldSpec, ...] = ()
    takes_action: bool = True
    method_kwargs: tuple[tuple[str, Any], ...] = ()
    function: Callable | None = None
    stock_field: tuple[str, str] | None = None
    precondition_checks: tuple[tuple[str, Callable], ...] = ()
    affected_ledgers: tuple[str, ...] = ("process",)

    def check_preconditions(self, state, constitution, action) -> dict[str, bool]:
        checks = {
            "not_terminated": not state.terminated,
            "has_volume": state.volume_L > constitution.tolerance,
            "has_material": any(amount > 0 for amount in (
                *state.species_amounts.values(), *state.state_slots.species_amounts().values())),
        }
        if self.operation.id in {"heat", "wait", "terminate"}:
            physical_thermal = state.vessel_id in state.metadata.get(
                "component_network", {}
            ).get("thermal_systems", ())
            checks["has_material"] |= physical_thermal
            # A declared thermal control volume can retain carrier/body heat
            # after its actual liquid stock is withdrawn. No fluid is added.
            checks["has_volume"] |= physical_thermal
        if self.operation.id in {"wait", "terminate"} and state.lab_clock is not None:
            # These acts concern the declared laboratory, including its filled
            # sources and independent systems, even when an empty collector is
            # selected. Sampling and heating retain their local stock checks.
            checks["has_volume"] = checks["has_material"] = True
        if state.process.stochastic_continuation is not None:
            # Volume is an explicit geometry; a mol-scale tolerance cannot
            # determine whether a femtolitre reaction chamber exists.
            checks["has_volume"] = state.volume_L > 0
        instrument = constitution.instruments.get(action.get("instrument"))
        if (self.operation.id == "measure" and state.phase == "formal"
                and instrument is not None and instrument.sample_volume_L == 0):
            checks["has_volume"] = True
        checks.update(
            {name: bool(check(state, action)) for name, check in self.precondition_checks}
        )
        return {name: checks[name] for name in self.operation.preconditions if name in checks}

    def contract(self) -> OperationContract:
        return OperationContract(
            self.operation.id,
            self.capability,
            self.kind,
            self.operation.required_fields,
            self.operation.preconditions,
        )

    def bind(self, services: Mapping[str, Any]) -> Callable:
        def execute(state, action):
            method = self.function or getattr(services[self.service_id], self.method)
            args = (state, action) if self.takes_action else (state,)
            return method(*args, **dict(self.method_kwargs))

        return execute


@dataclass(frozen=True)
class ServiceFactorySpec:
    service_id: str
    attribute: str
    factory: Callable
    service_class: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterRule:
    value_kind: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeclaredParameterRule(ParameterRule):
    """A rule with no execution effect until a composition names its parameter.

    Keep ordinary catalogue rules unchanged. This explicit opt-in lets a new
    optional parameter leave existing composition execution bindings exact.
    """


@dataclass(frozen=True)
class ComponentCapability:
    operations: tuple[str, ...]
    template: str | None = None
    priority: int = 0
    minimum_operations: int = 3
    minimum_measurements: int = 0
    minimum_time_s: float = 0.0
    outputs: tuple[str, ...] = ("out",)
    reactive_template: str | None = None
    interfaces: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    state_owners: tuple[str, ...] = ()
    required_all: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    input_ports: tuple[str, ...] = ("in",)
    port_unit: str = "mol"
    parameters: tuple[tuple[str, ParameterRule], ...] = ()
    processes: tuple[str, ...] = ()
    # Specialized native models must not silently fall back to generic
    # component-network services when their physical binding is absent.
    requires_physics: bool = False


@dataclass(frozen=True)
class InstrumentSpec:
    definition: Instrument
    contract: Any
    optional: bool = False
    native_observer: Callable | None = None
    # Physical sampling transform and independently resolved outlet must be
    # declared together. Pure plans may use explicitly owned sampling RNG state,
    # but never mutate inputs/global RNG or own cost/time accounting.
    sample_transform: Callable | None = None
    sample_outlet: Callable | None = None
    discrete_sampling: bool = False
    observer_factory: Callable | None = None

    def __post_init__(self) -> None:
        if self.definition.id != self.contract.instrument_id:
            raise ValueError("Instrument definition and contract IDs disagree")
        if self.native_observer is not None and not callable(self.native_observer):
            raise ValueError("Native instrument observer must be callable")
        if (self.observer_factory is not None and not callable(self.observer_factory)) or (
                self.observer_factory is not None and self.native_observer is not None):
            raise ValueError("Declare one native observer or its composition-bound factory")
        if (self.sample_transform is None) != (self.sample_outlet is None):
            raise ValueError("Instrument sample transform requires its boundary resolver")
        if self.discrete_sampling and (self.sample_transform is None
                                      or self.contract.sample_consumption_L != 0):
            raise ValueError("Discrete molecular capture needs a paired zero-carrier-volume plan")
        if self.sample_transform is not None and not all(
            callable(fn) for fn in (self.sample_transform, self.sample_outlet)
        ):
            raise ValueError("Instrument sampling hooks must be callable")

    def observer_for(self, composition=None):
        """Bind immutable model configuration without copying it into live state."""
        observer = (self.native_observer if self.observer_factory is None else
                    self.observer_factory(composition))
        if observer is not None and not callable(observer):
            raise ValueError("native observer factory must return a pure callable")
        return observer


@dataclass(frozen=True)
class PhysicsModelSpec:
    """An installed model codec and its physical binding, never a live state."""

    model_id: str
    version: str
    decode: Callable
    encode: Callable
    legacy_field: str | None = None
    binding: str = "vessel"
    component_kind: str | None = None
    exclusive_group: str | None = None
    validate_world: Callable | None = None
    validate_feed: Callable | None = None
    accepts_liquid_feed: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.model_id, self.version,
        )):
            raise ValueError("Physical models require explicit type and version")
        if self.binding not in {"vessel", "named"}:
            raise ValueError("Physical model binding must be vessel or named")
        if self.component_kind is not None and self.binding != "vessel":
            raise ValueError("Named model references are checked by their domain validator")
        if self.exclusive_group is not None and self.binding != "vessel":
            raise ValueError("Exclusive physical ownership requires a vessel binding")
        if self.legacy_field is not None and (
            not isinstance(self.legacy_field, str) or not self.legacy_field.endswith("_models")
        ):
            raise ValueError("Legacy model aliases must end in _models")
        if not all(callable(fn) for fn in (self.decode, self.encode)):
            raise ValueError("Physical model codecs must be callable")
        for fn in (self.validate_world, self.validate_feed):
            if fn is not None and not callable(fn):
                raise ValueError("Physical model validators must be callable")


@dataclass(frozen=True)
class AdvanceStrategySpec:
    """One solver owner and, where needed, a shared domain-service reference."""

    strategy_id: str
    factory: Callable
    service_attribute: str | None = None
    timed_measurements: bool = True
    legacy_input_default: bool = False
    shared_clock: bool = False

    def __post_init__(self):
        if (not isinstance(self.strategy_id, str) or not self.strategy_id
                or not callable(self.factory)):
            raise ValueError("Advance strategies require an explicit identity and factory")
        if type(self.shared_clock) is not bool:
            raise ValueError("shared-clock participation must be explicitly boolean")


_EXECUTION_FORMS = frozenset({"expression", "native_algorithm", "state_machine"})
_DERIVATIVE_MODES = frozenset(
    {"not_declared", "analytic", "automatic", "finite_difference", "not_applicable", "unsupported"}
)
_THREAD_SAFETY_MODES = frozenset(
    {"not_declared", "instance_local", "shared_read_only", "reentrant", "unsupported"}
)
_SNAPSHOT_MODES = frozenset({"not_declared", "supported", "not_applicable", "unsupported"})
_EFFECT_ACCESS = frozenset({"read", "write"})
_EFFECT_MIXING = frozenset({"sum", "equal", "weighted", "not_applicable"})
_EFFECT_RESET = frozenset({"clear", "preserve", "initial", "not_applicable"})
_NATIVE_UNIT_CONTRACTS = (
    {"dimension": "volume", "source": "L", "target": "m^3", "scale": 0.001},
    {"dimension": "volume", "source": "m^3", "target": "L", "scale": 1000.0},
    {
        "dimension": "concentration",
        "source": "mol/L",
        "target": "mol/m^3",
        "scale": 1000.0,
    },
    {
        "dimension": "concentration",
        "source": "mol/m^3",
        "target": "mol/L",
        "scale": 0.001,
    },
    {
        "dimension": "amount",
        "source": "mol",
        "target": "particle_count",
        "scale": 6.02214076e23,
        "reference": "exact_avogadro_constant",
    },
    {
        "dimension": "amount",
        "source": "particle_count",
        "target": "mol",
        "scale": 1.0 / 6.02214076e23,
        "reference": "exact_avogadro_constant",
    },
    {"dimension": "energy", "source": "J", "target": "kJ", "scale": 0.001},
    {"dimension": "energy", "source": "kJ", "target": "J", "scale": 1000.0},
)


@dataclass(frozen=True)
class NativeExecutionSpec:
    """Optional CPU execution contract attached to one Python module.

    The default deliberately remains undeclared.  A module must opt in with a
    lowering form and the three safety contracts before a CPU composition can
    be selected.  This keeps the Python author API source-compatible while
    making missing native capabilities visible at composition time.
    """

    lowering: str = "not_declared"
    derivatives: str = "not_declared"
    thread_safety: str = "not_declared"
    snapshot: str = "not_declared"
    public_result_type: str = "chemworld.public.Result"
    private_diagnostic_type: str = "chemworld.private.ExecutionDiagnostics"

    def __post_init__(self) -> None:
        if self.lowering != "not_declared" and self.lowering not in _EXECUTION_FORMS:
            raise ValueError(f"unknown native lowering form: {self.lowering!r}")
        if self.derivatives not in _DERIVATIVE_MODES:
            raise ValueError(f"unknown derivative capability: {self.derivatives!r}")
        if self.thread_safety not in _THREAD_SAFETY_MODES:
            raise ValueError(f"unknown thread-safety capability: {self.thread_safety!r}")
        if self.snapshot not in _SNAPSHOT_MODES:
            raise ValueError(f"unknown snapshot capability: {self.snapshot!r}")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.public_result_type, self.private_diagnostic_type)
        ):
            raise ValueError("native result and diagnostic types require stable names")

    @property
    def cpu_eligible(self) -> bool:
        return (
            self.lowering in _EXECUTION_FORMS
            and self.derivatives not in {"not_declared", "unsupported"}
            and self.thread_safety not in {"not_declared", "unsupported"}
            and self.snapshot not in {"not_declared", "unsupported"}
        )

    def cpu_diagnostics(self, module_id: str) -> tuple[dict[str, str], ...]:
        checks = (
            ("lowering", self.lowering, "not_declared"),
            ("derivatives", self.derivatives, "not_declared"),
            ("thread_safety", self.thread_safety, "not_declared"),
            ("snapshot", self.snapshot, "not_declared"),
        )
        diagnostics = []
        for field_name, value, undeclared in checks:
            if value == undeclared:
                diagnostics.append(
                    {
                        "code": "native_capability_undeclared",
                        "module": module_id,
                        "field": field_name,
                        "message": f"{module_id} does not declare {field_name}",
                    }
                )
            elif value == "unsupported":
                diagnostics.append(
                    {
                        "code": "native_capability_unsupported",
                        "module": module_id,
                        "field": field_name,
                        "message": f"{module_id} explicitly does not support {field_name}",
                    }
                )
        return tuple(diagnostics)

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "lowering": self.lowering,
            "derivatives": self.derivatives,
            "thread_safety": self.thread_safety,
            "snapshot": self.snapshot,
            "public_result_type": self.public_result_type,
            "private_diagnostic_type": self.private_diagnostic_type,
            "cpu_eligible": self.cpu_eligible,
        }


# The longer name is useful to authors who think in terms of a capability
# declaration; keep one canonical implementation and no second registry.
ExecutionCapabilitySpec = NativeExecutionSpec


@dataclass(frozen=True)
class ExecutionEffectSpec:
    """One explicit read/write effect consumed by a native lowering."""

    resource: str
    access: Literal["read", "write"]
    owner: str
    unit: str = "dimensionless"
    operation_id: str | None = None
    mixing: str = "not_applicable"
    reset: str = "not_applicable"
    conserved_quantity: str | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.resource, self.owner, self.unit)
        ):
            raise ValueError("native effects require stable resource, owner and unit names")
        if self.operation_id is not None and (
            not isinstance(self.operation_id, str) or not self.operation_id.strip()
        ):
            raise ValueError("native effect operation IDs must be non-empty")
        if self.access not in _EFFECT_ACCESS:
            raise ValueError(f"unknown native effect access: {self.access!r}")
        if self.mixing not in _EFFECT_MIXING:
            raise ValueError(f"unknown native effect mixing rule: {self.mixing!r}")
        if self.reset not in _EFFECT_RESET:
            raise ValueError(f"unknown native effect reset rule: {self.reset!r}")
        if self.conserved_quantity is not None and (
            not isinstance(self.conserved_quantity, str) or not self.conserved_quantity.strip()
        ):
            raise ValueError("conserved quantities require stable names")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "resource": self.resource,
            "access": self.access,
            "owner": self.owner,
            "unit": self.unit,
            "operation_id": self.operation_id,
            "mixing": self.mixing,
            "reset": self.reset,
            "conserved_quantity": self.conserved_quantity,
        }


SemanticEffectSpec = ExecutionEffectSpec


class NativeCapabilityError(ValueError):
    """Raised when a caller explicitly requests an ineligible CPU backend."""

    def __init__(self, diagnostics: tuple[dict[str, str], ...]):
        self.diagnostics = diagnostics
        message = "; ".join(item["message"] for item in diagnostics)
        super().__init__(f"CPU execution is not available for the selected modules: {message}")


@dataclass(frozen=True)
class DomainModuleSpec:
    module_id: str
    operations: tuple[OperationSpec, ...] = ()
    services: tuple[ServiceFactorySpec, ...] = ()
    components: tuple[tuple[str, ComponentCapability], ...] = ()
    instruments: tuple[InstrumentSpec, ...] = ()
    state_slots: tuple[StateSlotSpec, ...] = ()
    processes: tuple[tuple[str, Callable], ...] = ()
    dependencies: tuple[str, ...] = ()
    version: str = "1"
    advance_strategies: tuple[AdvanceStrategySpec, ...] = ()
    scope_notes: tuple[str, ...] = ()
    models: tuple[PhysicsModelSpec, ...] = ()
    execution: NativeExecutionSpec = field(default_factory=NativeExecutionSpec)
    effects: tuple[ExecutionEffectSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Domain modules require an explicit implementation version")
        for name in (
            "operations",
            "services",
            "components",
            "instruments",
            "state_slots",
            "processes",
            "dependencies",
            "advance_strategies",
            "scope_notes",
            "models",
            "effects",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        # Earlier local modules supplied (id, factory) pairs. Normalize at the
        # declaration boundary; every consumer below sees the same specification.
        object.__setattr__(self, "advance_strategies", tuple(
            value if isinstance(value, AdvanceStrategySpec) else AdvanceStrategySpec(*value)
            for value in self.advance_strategies
        ))
        if not isinstance(self.execution, NativeExecutionSpec):
            raise TypeError("module execution metadata must be a NativeExecutionSpec")
        if not all(isinstance(effect, ExecutionEffectSpec) for effect in self.effects):
            raise TypeError("module effects must be ExecutionEffectSpec declarations")


@dataclass(frozen=True)
class DomainModuleRegistry:
    modules: tuple[DomainModuleSpec, ...]

    def __post_init__(self) -> None:
        # Instrument contracts contain mapping-valued calibration data. Each
        # world owns its copy, even when it uses the built-in declarations.
        object.__setattr__(
            self,
            "modules",
            tuple(
                replace(m, instruments=deepcopy(m.instruments))
                for m in sorted(self.modules, key=lambda m: m.module_id)
            ),
        )
        ids = {m.module_id for m in self.modules}
        if len(ids) != len(self.modules):
            raise ValueError("Duplicate module owner")
        for module in self.modules:
            if not set(module.dependencies) <= ids:
                raise ValueError(f"Missing module dependencies for {module.module_id}")
            operation_ids = {spec.operation.id for spec in module.operations}
            effect_keys = set()
            for effect in module.effects:
                if effect.operation_id is not None and effect.operation_id not in operation_ids:
                    raise ValueError(
                        f"Missing operation owner for native effect {effect.operation_id}"
                    )
                key = (effect.operation_id, effect.resource, effect.access)
                if key in effect_keys:
                    raise ValueError(f"Duplicate native effect owner for {key!r}")
                effect_keys.add(key)
        for label, items in (
            ("operation", [op.operation.id for m in self.modules for op in m.operations]),
            ("service", [s.service_id for m in self.modules for s in m.services]),
            ("component", [key for m in self.modules for key, _ in m.components]),
            ("instrument", [s.definition.id for m in self.modules for s in m.instruments]),
            ("service attribute", [s.attribute for m in self.modules for s in m.services]),
            ("process", [key for m in self.modules for key, _ in m.processes]),
            ("advance strategy", [
                s.strategy_id for m in self.modules for s in m.advance_strategies
            ]),
            ("strategy service attribute", [
                s.service_attribute for m in self.modules for s in m.advance_strategies
                if s.service_attribute is not None
            ]),
            ("state slot", [s.slot_id for m in self.modules for s in m.state_slots]),
            ("physical model", [s.model_id for m in self.modules for s in m.models]),
            ("legacy model alias", [
                s.legacy_field for m in self.modules for s in m.models
                if s.legacy_field is not None
            ]),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"Duplicate {label} owner")
        services = self.service_specs()
        operations = self.operation_specs()
        for model in self.model_specs().values():
            if model.component_kind is not None and model.component_kind not in self.components():
                raise ValueError(f"Missing component owner for physical model {model.model_id}")
        for name, component in self.components().items():
            if not set(component.operations) <= operations.keys():
                raise ValueError(f"Missing operation owner for component {name}")
            if not set(component.dependencies) <= self.components().keys():
                raise ValueError(f"Missing component dependencies for {name}")
            if not set(component.processes) <= self.process_factories().keys():
                raise ValueError(f"Missing process factory for component {name}")
        for spec in self.operation_specs().values():
            if spec.service_id not in services:
                raise ValueError(f"Missing service for {spec.operation.id}")
            if {f.name for f in spec.fields if f.required} != set(spec.operation.required_fields):
                raise ValueError(f"Action fields disagree with {spec.operation.id}")
            if len({f.name for f in spec.fields}) != len(spec.fields):
                raise ValueError(f"Duplicate action field in {spec.operation.id}")
            from chemworld.world.action_layout import LEGACY_OPERATION_ORDER

            supported = {"not_terminated", "has_volume", "has_material"} | {
                name for name, _ in spec.precondition_checks
            }
            if (
                spec.operation.id not in LEGACY_OPERATION_ORDER
                and not set(spec.operation.preconditions) <= supported
            ):
                raise ValueError(f"Missing precondition implementation for {spec.operation.id}")
        self.service_order()
        remaining = {m.module_id: set(m.dependencies) for m in self.modules}
        while remaining:
            ready = {key for key, deps in remaining.items() if not deps & remaining.keys()}
            if not ready:
                raise ValueError("Cyclic module dependencies")
            remaining = {key: deps for key, deps in remaining.items() if key not in ready}

    @classmethod
    def default(cls) -> DomainModuleRegistry:
        from chemworld.world.builtin_modules import BUILTIN_MODULES

        return cls(BUILTIN_MODULES)

    def operation_specs(self) -> dict[str, OperationSpec]:
        return {op.operation.id: op for module in self.modules for op in module.operations}

    def contracts(self) -> dict[str, OperationContract]:
        return {key: value.contract() for key, value in self.operation_specs().items()}

    def service_specs(self) -> dict[str, ServiceFactorySpec]:
        return {s.service_id: s for module in self.modules for s in module.services}

    def components(self) -> dict[str, ComponentCapability]:
        return {key: value for module in self.modules for key, value in module.components}

    def execution_specs(self) -> dict[str, NativeExecutionSpec]:
        """Return the optional native contract keyed by the existing module ID."""
        return {module.module_id: module.execution for module in self.modules}

    def _modules_for_composition(self, composition) -> tuple[DomainModuleSpec, ...]:
        spec = getattr(composition, "spec", composition)
        if spec is None:
            return self.modules
        selected_ids: set[str] = set()
        component_kinds = {component.kind for component in getattr(spec, "components", ())}
        task = getattr(spec, "task", None)
        operations = set(getattr(task, "operations", None) or ())
        instruments = set(getattr(task, "instruments", None) or ())
        for module in self.modules:
            if component_kinds.intersection(name for name, _ in module.components):
                selected_ids.add(module.module_id)
            if operations.intersection(operation.operation.id for operation in module.operations):
                selected_ids.add(module.module_id)
            if instruments.intersection(item.definition.id for item in module.instruments):
                selected_ids.add(module.module_id)
        if not selected_ids:
            return self.modules
        module_by_id = {module.module_id: module for module in self.modules}
        pending = list(selected_ids)
        while pending:
            module = module_by_id[pending.pop()]
            for dependency in module.dependencies:
                if dependency not in selected_ids:
                    selected_ids.add(dependency)
                    pending.append(dependency)
        return tuple(module for module in self.modules if module.module_id in selected_ids)

    def cpu_diagnostics(self, composition=None) -> tuple[dict[str, str], ...]:
        """Report why an explicit CPU selection cannot be made.

        Missing metadata is intentionally a deterministic diagnostic rather than
        an inferred capability.  A composition may therefore select the whole
        reference backend without changing any existing Python module.
        """
        modules = self._modules_for_composition(composition)
        diagnostics = [
            diagnostic
            for module in modules
            for diagnostic in module.execution.cpu_diagnostics(module.module_id)
        ]
        if composition is not None:
            spec = getattr(composition, "spec", composition)
            for component in getattr(spec, "components", ()):
                if component.kind not in self.components():
                    diagnostics.append(
                        {
                            "code": "native_component_undeclared",
                            "module": component.kind,
                            "field": "components",
                            "message": f"component {component.kind} is not installed",
                        }
                    )
        return tuple(diagnostics)

    @staticmethod
    def _state_layout(module: DomainModuleSpec) -> list[dict[str, Any]]:
        return [
            {
                "id": slot.slot_id,
                "owner": {"kind": slot.owner_kind, "id": slot.owner_id},
                "state_type": slot.state_type,
                "mechanism": slot.mechanism_id,
                "phase": slot.phase_id,
                "batch_reset": slot.batch_reset,
                "time_evolution": slot.time_evolution,
                "fields": [
                    {
                        "id": field.name,
                        "unit": field.unit,
                        "mixing": field.mixing,
                        "weight_field": field.weight_field,
                        "minimum": field.minimum,
                        "conserved_quantity": field.conserved_quantity,
                        "species_id": field.species_id,
                    }
                    for field in slot.fields
                ],
            }
            for slot in module.state_slots
        ]

    @staticmethod
    def _port_layout(module: DomainModuleSpec) -> list[dict[str, Any]]:
        ports = []
        for component_id, component in module.components:
            ports.append(
                {
                    "component": component_id,
                    "inputs": [
                        {
                            "id": f"{component_id}.input.{name}",
                            "name": name,
                            "unit": component.port_unit,
                        }
                        for name in component.input_ports
                    ],
                    "outputs": [
                        {
                            "id": f"{component_id}.output.{name}",
                            "name": name,
                            "unit": component.port_unit,
                        }
                        for name in component.outputs
                    ],
                }
            )
        return ports

    def lower_execution_ir(self, composition=None, *, backend: str = "auto") -> dict[str, Any]:
        """Lower declarations into a JSON-safe native execution IR.

        The lowering is generated from the one module registry.  ``auto`` keeps
        the existing reference path when a module has no CPU contract;
        ``backend='cpu'`` fails at the boundary with all deterministic
        diagnostics instead of silently mixing executors.
        """
        if backend not in {"auto", "cpu", "reference"}:
            raise ValueError("backend must be auto, cpu or reference")
        diagnostics = self.cpu_diagnostics(composition)
        if backend == "cpu" and diagnostics:
            raise NativeCapabilityError(diagnostics)
        selected = (
            "cpu"
            if backend == "cpu" or (backend == "auto" and not diagnostics)
            else "reference"
        )
        modules = []
        for module in self.modules:
            operation_rows = []
            for operation in module.operations:
                operation_rows.append(
                    {
                        "id": operation.operation.id,
                        "kind": operation.kind,
                        "fields": [field.public_schema() for field in operation.fields],
                        "read_write_ledgers": list(operation.affected_ledgers),
                        "effects": [
                            effect.to_dict()
                            for effect in module.effects
                            if effect.operation_id in {None, operation.operation.id}
                        ],
                    }
                )
            modules.append(
                {
                    "id": module.module_id,
                    "version": module.version,
                    "dependencies": list(module.dependencies),
                    "execution": module.execution.to_dict(),
                    "state": self._state_layout(module),
                    "ports": self._port_layout(module),
                    "effects": [effect.to_dict() for effect in module.effects],
                    "operations": operation_rows,
                    "conservation": sorted(
                        {
                            field.conserved_quantity
                            for slot in module.state_slots
                            for field in slot.fields
                            if field.conserved_quantity is not None
                        }
                    ),
                }
            )
        spec = getattr(composition, "spec", composition)
        composition_row = None
        if spec is not None:
            compatibility = getattr(composition, "compatibility", None)
            composition_row = {
                "id": getattr(spec, "composition_id", None),
                "instances": [
                    {
                        "id": component.id,
                        "kind": component.kind,
                        "vessel": component.vessel,
                        "role": component.role,
                    }
                    for component in getattr(spec, "components", ())
                ],
                "state_owners": dict(
                    sorted(
                        getattr(compatibility, "state_owners", {})
                        if compatibility is not None
                        else {}
                    )
                ),
            }
        return {
            "schema": "chemworld-native-execution-ir-v1",
            "backend": selected,
            "diagnostics": [dict(item) for item in diagnostics],
            "types": {
                "public_result": "chemworld.public.Result",
                "private_diagnostics": "chemworld.private.ExecutionDiagnostics",
            },
            "units": {
                "canonical": {
                    "volume": "m^3",
                    "amount": "mol",
                    "concentration": "mol/m^3",
                    "particle_count": "integer_count",
                    "energy": "J",
                },
                "conversions": [dict(item) for item in _NATIVE_UNIT_CONTRACTS],
                "energy_reference": {
                    "absolute": "J",
                    "molar": "J/mol",
                    "zero": "model_declared",
                },
            },
            "composition": composition_row,
            "modules": modules,
        }

    def instrument_specs(self, *, include_optional: bool = False) -> dict[str, InstrumentSpec]:
        return {
            s.definition.id: s
            for m in self.modules
            for s in m.instruments
            if include_optional or not s.optional
        }

    def instrument_order(self, *, include_optional: bool = False) -> tuple[str, ...]:
        from chemworld.world.action_layout import LEGACY_INSTRUMENT_ORDER

        return (
            *LEGACY_INSTRUMENT_ORDER,
            *sorted(
                set(self.instrument_specs(include_optional=include_optional))
                - set(LEGACY_INSTRUMENT_ORDER)
            ),
        )

    def state_specs(self) -> dict[str, StateSlotSpec]:
        return {s.slot_id: s for m in self.modules for s in m.state_slots}

    def process_factories(self) -> dict[str, Callable]:
        return {key: factory for m in self.modules for key, factory in m.processes}

    def advance_factories(self) -> dict[str, Callable]:
        return {key: spec.factory for key, spec in self.advance_specs().items()}

    def advance_specs(self) -> dict[str, AdvanceStrategySpec]:
        return {s.strategy_id: s for m in self.modules for s in m.advance_strategies}

    def build_advance_strategies(self, composition=None, *, thermal=None) -> dict[str, Any]:
        result = {}
        for key, spec in self.advance_specs().items():
            strategy = spec.factory(composition)
            if thermal is not None:
                bind = getattr(strategy, "bind_runtime", None)
                if bind is not None:
                    bind(thermal)
                if spec.service_attribute is not None:
                    setattr(thermal, spec.service_attribute, strategy)
            result[key] = strategy
        return result

    def model_specs(self) -> dict[str, PhysicsModelSpec]:
        return {s.model_id: s for m in self.modules for s in m.models}

    def service_order(self) -> tuple[ServiceFactorySpec, ...]:
        remaining = self.service_specs()
        ordered = []
        resolved: set[str] = set()
        while remaining:
            ready = sorted(
                key for key, spec in remaining.items() if set(spec.dependencies) <= resolved
            )
            if not ready:
                raise ValueError("Missing or cyclic service dependencies")
            for key in ready:
                ordered.append(remaining.pop(key))
                resolved.add(key)
        return tuple(ordered)

    def operation_order(self) -> tuple[str, ...]:
        from chemworld.world.action_layout import LEGACY_OPERATION_ORDER

        available = self.operation_specs()
        # Retain reserved positions even when a subset of modules is installed.
        return (*LEGACY_OPERATION_ORDER, *sorted(set(available) - set(LEGACY_OPERATION_ORDER)))

    def validate_connection(
        self, source_kind: str, source_port: str, target_kind: str, target_port: str, unit: str
    ) -> None:
        source, target = self.components()[source_kind], self.components()[target_kind]
        if source_port not in source.outputs or target_port not in target.input_ports:
            raise ValueError("Incompatible component ports")
        if source.port_unit != unit or target.port_unit != unit:
            raise ValueError("Incompatible component port units")


@dataclass
class ServiceContext:
    world: Any
    constitution: Any
    species_view: Any
    partition_nominal_pair_contract: str | None = None
    services: dict[str, Any] = field(default_factory=dict)
    modules: DomainModuleRegistry | None = None
    composition: Any = None
