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
from typing import Any

from chemworld.foundation.ontology import Instrument, Operation
from chemworld.foundation.state_ownership import StateSlotSpec


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(self.choices))
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
        if self.bounds is not None:
            result.update(type="number", bounds={"low": self.bounds[0], "high": self.bounds[1]})
        if self.choices:
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
        if state.process.stochastic_continuation is not None:
            # Volume is an explicit geometry; a mol-scale tolerance cannot
            # determine whether a femtolitre reaction chamber exists.
            checks["has_volume"] = state.volume_L > 0
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

    def __post_init__(self) -> None:
        if self.definition.id != self.contract.instrument_id:
            raise ValueError("Instrument definition and contract IDs disagree")
        if self.native_observer is not None and not callable(self.native_observer):
            raise ValueError("Native instrument observer must be callable")
        if (self.sample_transform is None) != (self.sample_outlet is None):
            raise ValueError("Instrument sample transform requires its boundary resolver")
        if self.discrete_sampling and (self.sample_transform is None
                                      or self.contract.sample_consumption_L != 0):
            raise ValueError("Discrete molecular capture needs a paired zero-carrier-volume plan")
        if self.sample_transform is not None and not all(
            callable(fn) for fn in (self.sample_transform, self.sample_outlet)
        ):
            raise ValueError("Instrument sampling hooks must be callable")


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
    advance_strategies: tuple[tuple[str, Callable], ...] = ()

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
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


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
        for label, items in (
            ("operation", [op.operation.id for m in self.modules for op in m.operations]),
            ("service", [s.service_id for m in self.modules for s in m.services]),
            ("component", [key for m in self.modules for key, _ in m.components]),
            ("instrument", [s.definition.id for m in self.modules for s in m.instruments]),
            ("service attribute", [s.attribute for m in self.modules for s in m.services]),
            ("process", [key for m in self.modules for key, _ in m.processes]),
            ("advance strategy", [key for m in self.modules for key, _ in m.advance_strategies]),
            ("state slot", [s.slot_id for m in self.modules for s in m.state_slots]),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"Duplicate {label} owner")
        services = self.service_specs()
        operations = self.operation_specs()
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
        return {key: factory for m in self.modules for key, factory in m.advance_strategies}

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
