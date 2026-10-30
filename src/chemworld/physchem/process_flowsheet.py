"""Declared multi-tear molecular flowsheets on the common sparse root solver.

Ports have exactly one producer and one consumer (or export). Cutting declared
tear edges yields an execution DAG; *all* tear states are then closed together.
Trial streams are algebraic guesses and never an inventory or a resource charge.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from math import exp, fsum, isfinite
from typing import Any, TypeAlias

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

from chemworld.physchem.molecular_cascade import MolecularCascadeSpec, solve_molecular_cascade
from chemworld.physchem.molecular_eos import _molecular_density_root
from chemworld.physchem.molecular_flash_separator import (
    MolecularFlashSeparatorSpec,
    evaluate_molecular_flash_separator,
)
from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
    MolecularProcessResult,
)
from chemworld.physchem.process_flowsheet_units import (
    MolecularHeatExchanger,
    MolecularPressureRecovery,
    ProcessMixer,
    ProcessSplitter,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)

ProcessOperation: TypeAlias = (
    ProcessMixer
    | ProcessSplitter
    | MolecularHeatDuty
    | MolecularPressureStage
    | MolecularHeatExchanger
    | MolecularPressureRecovery
    | MolecularFlashSeparatorSpec
    | MolecularCascadeSpec
)


def unit_utility_rates(
    operation: ProcessOperation, result: MolecularProcessResult
) -> dict[str, float]:
    """External utility ports, preserving opposing duties inside composite units.

    Cascade stages have separate thermal boundaries. Heat exchanged internally
    by an HX is not an external utility. A pressure-recovery result already
    reports only external shaft draw/export and its transmission-loss cooling,
    so its internal expander/compressor powers must not be counted again.
    """
    if not result.success:
        raise ValueError("unsuccessful unit has no utility rates")
    heats = (
        tuple(float(stage["heat_duty_W"]) for stage in result.diagnostics["stages"])
        if isinstance(operation, MolecularCascadeSpec)
        else (result.heat_duty_W,)
    )
    return {
        "heat_import_W": fsum(max(q, 0.0) for q in heats),
        "heat_export_W": fsum(max(-q, 0.0) for q in heats),
        "shaft_import_W": max(result.shaft_work_W, 0.0),
        "shaft_export_W": max(-result.shaft_work_W, 0.0),
    }


def _stream_dict(stream: MolecularStream) -> dict[str, Any]:
    return {
        "temperature_K": stream.state.temperature_K,
        "molar_density_mol_m3": stream.state.molar_density_mol_m3,
        "mole_fractions": stream.state.mole_fractions,
        "molar_rate_mol_s": stream.molar_rate_mol_s,
    }


def _stream_from_dict(domain: MolecularProcessDomain, value: Mapping[str, Any]) -> MolecularStream:
    return MolecularStream(
        domain.state(
            value["temperature_K"], value["molar_density_mol_m3"], value["mole_fractions"]
        ),
        value["molar_rate_mol_s"],
    )


def _operation_from_dict(value: Mapping[str, Any]) -> ProcessOperation:
    if value.get("model_type") == "molecular_flash_separator_v1":
        return MolecularFlashSeparatorSpec.from_dict(value)
    if value.get("model_type") == "molecular_equilibrium_cascade_v1":
        return MolecularCascadeSpec.from_dict(value)
    decoders: dict[str, Callable[[Mapping[str, Any]], ProcessOperation]] = {
        "mixer": ProcessMixer.from_dict,
        "splitter": ProcessSplitter.from_dict,
        "heat": MolecularHeatDuty.from_dict,
        "pressure": MolecularPressureStage.from_dict,
        "heat_exchanger": MolecularHeatExchanger.from_dict,
        "pressure_recovery": MolecularPressureRecovery.from_dict,
    }
    if value.get("kind") not in decoders:
        raise ValueError("unknown flowsheet operation")
    return decoders[value["kind"]](value)


@dataclass(frozen=True)
class ProcessSource:
    port_id: str
    domain: MolecularProcessDomain
    stream: MolecularStream

    def __post_init__(self) -> None:
        if not isinstance(self.port_id, str) or not self.port_id:
            raise ValueError("source needs a port identity")
        self.domain.validate_stream(self.stream)

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "domain": self.domain.to_dict(),
            "stream": _stream_dict(self.stream),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessSource:
        domain = MolecularProcessDomain.from_dict(value["domain"])
        return cls(value["port_id"], domain, _stream_from_dict(domain, value["stream"]))


@dataclass(frozen=True)
class ProcessUnit:
    unit_id: str
    operation: ProcessOperation
    inlet_ports: tuple[str, ...]
    outlet_ports: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inlet_ports", tuple(self.inlet_ports))
        object.__setattr__(self, "outlet_ports", tuple(self.outlet_ports))
        if (
            not isinstance(self.unit_id, str)
            or not self.unit_id
            or any(not isinstance(p, str) or not p for p in (*self.inlet_ports, *self.outlet_ports))
        ):
            raise ValueError("unit and ports need explicit identities")
        ins, outs = operation_domains(self.operation, len(self.inlet_ports))
        if len(ins) != len(self.inlet_ports) or len(outs) != len(self.outlet_ports):
            raise ValueError("unit port arity disagrees with its operation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "operation": self.operation.to_dict(),
            "inlet_ports": self.inlet_ports,
            "outlet_ports": self.outlet_ports,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessUnit:
        return cls(
            value["unit_id"],
            _operation_from_dict(value["operation"]),
            tuple(value["inlet_ports"]),
            tuple(value["outlet_ports"]),
        )


def operation_domains(
    operation: ProcessOperation,
    inlet_count: int,
) -> tuple[tuple[MolecularProcessDomain, ...], tuple[MolecularProcessDomain, ...]]:
    """Declared inlet/output order; flash/cascade outputs are always gas, liquid."""
    domains: tuple[MolecularProcessDomain, ...]
    if isinstance(operation, ProcessMixer):
        if inlet_count < 1:
            raise ValueError("mixer needs at least one inlet")
        return (operation.domain,) * inlet_count, (operation.domain,)
    if isinstance(operation, ProcessSplitter):
        return (operation.domain,), (operation.domain,) * len(operation.fractions)
    if isinstance(operation, (MolecularHeatDuty, MolecularPressureStage)):
        return (operation.domain,), (operation.domain,)
    if isinstance(operation, MolecularHeatExchanger):
        domains = (operation.hot_domain, operation.cold_domain)
        return domains, domains
    if isinstance(operation, MolecularPressureRecovery):
        domains = (operation.expander.domain, operation.compressor.domain)
        return domains, domains
    if isinstance(operation, (MolecularFlashSeparatorSpec, MolecularCascadeSpec)):
        domains = tuple(
            MolecularProcessDomain(operation.model, operation.temperature_range_K, b)
            for b in operation.basins
        )
        return (
            (operation.feed_domain,)
            if isinstance(operation, MolecularFlashSeparatorSpec)
            else domains
        ), domains
    raise ValueError("unsupported flowsheet operation")


@dataclass(frozen=True)
class ProcessTear:
    port_id: str
    domain: MolecularProcessDomain
    initial_stream: MolecularStream
    component_flow_bounds_mol_s: tuple[float, float] = (0.0, 1.0)
    temperature_bounds_K: tuple[float, float] | None = None
    pressure_bounds_Pa: tuple[float, float] = (1.0, 1e9)
    component_flow_scale_mol_s: float = 0.1
    temperature_scale_K: float = 300.0
    pressure_scale_Pa: float = 1e5

    def __post_init__(self) -> None:
        self.domain.validate_stream(self.initial_stream)
        for name in ("component_flow_bounds_mol_s", "pressure_bounds_Pa"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "temperature_bounds_K",
            tuple(
                self.domain.temperature_range_K
                if self.temperature_bounds_K is None
                else self.temperature_bounds_K
            ),
        )
        if not isinstance(self.port_id, str) or not self.port_id:
            raise ValueError("tear needs an explicit port")
        for bounds in (
            self.component_flow_bounds_mol_s,
            self.temperature_bounds_K,
            self.pressure_bounds_Pa,
        ):
            if (
                bounds is None
                or len(bounds) != 2
                or any(not isfinite(x) for x in bounds)
                or not 0 <= bounds[0] < bounds[1]
            ):
                raise ValueError("tear coordinates need finite ordered physical bounds")
        if not (self.temperature_bounds_K is not None):
            raise ValueError("tear coordinates require resolved temperature bounds")
        if (
            self.temperature_bounds_K[0] < self.domain.temperature_range_K[0]
            or self.temperature_bounds_K[1] > self.domain.temperature_range_K[1]
            or self.pressure_bounds_Pa[0] <= 0
            or any(
                not isfinite(x) or x <= 0
                for x in (
                    self.component_flow_scale_mol_s,
                    self.temperature_scale_K,
                    self.pressure_scale_Pa,
                )
            )
        ):
            raise ValueError("tear scales must be positive and T/P bounds inside their domain")
        values = _coordinates(self.initial_stream)
        if any(x < v.lower or x > v.upper for x, v in zip(values, self.variables(), strict=True)):
            raise ValueError("tear initial stream exceeds its declared coordinate bounds")

    def variables(self) -> tuple[ResidualVariable, ...]:
        if not (self.temperature_bounds_K is not None):
            raise ValueError("tear coordinates require resolved temperature bounds")
        components = tuple(
            ResidualVariable(
                f"{self.port_id}.{c}",
                "mol/s",
                self.component_flow_scale_mol_s,
                "algebraic",
                *self.component_flow_bounds_mol_s,
            )
            for c in self.domain.model.component_ids
        )
        return (
            *components,
            ResidualVariable(
                f"{self.port_id}.T",
                "K",
                self.temperature_scale_K,
                "algebraic",
                *self.temperature_bounds_K,
            ),
            ResidualVariable(
                f"{self.port_id}.P",
                "Pa",
                self.pressure_scale_Pa,
                "algebraic",
                *self.pressure_bounds_Pa,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "domain": self.domain.to_dict(),
            "initial_stream": _stream_dict(self.initial_stream),
            "component_flow_bounds_mol_s": self.component_flow_bounds_mol_s,
            "temperature_bounds_K": self.temperature_bounds_K,
            "pressure_bounds_Pa": self.pressure_bounds_Pa,
            "component_flow_scale_mol_s": self.component_flow_scale_mol_s,
            "temperature_scale_K": self.temperature_scale_K,
            "pressure_scale_Pa": self.pressure_scale_Pa,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessTear:
        fields = dict(value)
        fields["domain"] = MolecularProcessDomain.from_dict(fields["domain"])
        fields["initial_stream"] = _stream_from_dict(fields["domain"], fields["initial_stream"])
        return cls(**fields)


@dataclass(frozen=True)
class ProcessConstraint:
    port_id: str
    coordinate: str
    value: float
    scale: float
    component_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.port_id
            or self.coordinate not in {"T", "P", "component_flow"}
            or not isfinite(self.value)
            or not isfinite(self.scale)
            or self.scale <= 0
            or (self.coordinate == "component_flow") != (self.component_id is not None)
        ):
            raise ValueError("fixed process constraint needs coordinate, target and scale")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessConstraint:
        return cls(**value)


@dataclass(frozen=True)
class FlowsheetPolicy:
    root: RootPolicy = field(default_factory=lambda: RootPolicy(1e-9, 60))
    relative_difference_step: float = 6e-6
    maximum_trial_evaluations: int = 1000
    maximum_unit_evaluations: int = 10000
    rank_relative_tolerance: float = 1e-8
    initialization_iterations: int = 0
    initialization_damping: float = 0.5
    balance_relative_tolerance: float = 1e-7
    component_balance_absolute_mol_s: float = 1e-9
    energy_balance_absolute_W: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, RootPolicy)
            or any(
                type(v) is not int or v < 1
                for v in (self.maximum_trial_evaluations, self.maximum_unit_evaluations)
            )
            or type(self.initialization_iterations) is not int
            or self.initialization_iterations < 0
            or any(
                not isfinite(v) or v <= 0
                for v in (
                    self.relative_difference_step,
                    self.rank_relative_tolerance,
                    self.balance_relative_tolerance,
                    self.component_balance_absolute_mol_s,
                    self.energy_balance_absolute_W,
                )
            )
            or not isfinite(self.initialization_damping)
            or not 0 < self.initialization_damping <= 1
        ):
            raise ValueError(
                "flowsheet requires finite explicit root, initialization and work budgets"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowsheetPolicy:
        fields = dict(value)
        fields["root"] = RootPolicy(**fields["root"])
        return cls(**fields)


@dataclass(frozen=True)
class FlowsheetResult:
    success: bool
    outlets: dict[str, MolecularStream] = field(default_factory=dict)
    absent_outlets: tuple[str, ...] = ()
    heat_duty_W: float = 0.0
    shaft_work_W: float = 0.0
    unit_results: dict[str, MolecularProcessResult] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    port_streams: dict[str, MolecularStream | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "outlets": {
                k: {"state": v.state.to_dict(), "molar_rate_mol_s": v.molar_rate_mol_s}
                for k, v in self.outlets.items()
            },
            "absent_outlets": self.absent_outlets,
            "heat_duty_W": self.heat_duty_W,
            "shaft_work_W": self.shaft_work_W,
            "unit_results": {k: v.to_dict() for k, v in self.unit_results.items()},
            "diagnostics": self.diagnostics,
            "failure_reason": self.failure_reason,
            "port_streams": {
                k: None
                if s is None
                else {"state": s.state.to_dict(), "molar_rate_mol_s": s.molar_rate_mol_s}
                for k, s in self.port_streams.items()
            },
        }


def _coordinates(stream: MolecularStream) -> np.ndarray:
    return np.array(
        [stream.molar_rate_mol_s * x for x in stream.state.mole_fractions]
        + [stream.state.temperature_K, stream.state.pressure_Pa]
    )


def _evaluate_unit(
    unit: ProcessUnit,
    inlets: tuple[MolecularStream | None, ...],
    policy: RootPolicy,
) -> tuple[MolecularProcessResult, tuple[MolecularStream | None, ...]]:
    op = unit.operation
    actual = tuple(s for s in inlets if s is not None)
    if isinstance(op, (MolecularCascadeSpec, MolecularPressureRecovery)) and len(actual) != 2:
        return MolecularProcessResult(False, failure_reason="required_inlet_phase_absent"), ()
    # Absent equilibrium phases remain absent: no synthetic zero-density phase.
    if not actual:
        result = MolecularProcessResult(True, diagnostics={"absent_inlet_propagation": True})
        return result, (None,) * len(unit.outlet_ports)
    if isinstance(op, ProcessMixer):
        result = op.evaluate(actual, policy=policy)
    elif isinstance(op, MolecularHeatExchanger) and len(actual) != len(inlets):
        index = 0 if inlets[0] is not None else 1
        domain = (op.hot_domain, op.cold_domain)[index]
        pressure = (op.hot_outlet_pressure_Pa, op.cold_outlet_pressure_Pa)[index]
        result = MolecularHeatDuty(domain, 0.0, pressure).evaluate(actual[0], policy=policy)
        result.diagnostics["absent_side_no_exchange"] = True
        return result, (result.outlet, None) if index == 0 else (None, result.outlet)
    elif any(s is None for s in inlets):
        return MolecularProcessResult(False, failure_reason="required_inlet_phase_absent"), ()
    elif isinstance(op, ProcessSplitter):
        result = op.evaluate(actual[0])
    elif isinstance(op, (MolecularHeatDuty, MolecularPressureStage)):
        result = op.evaluate(actual[0], policy=policy)
    elif isinstance(op, (MolecularHeatExchanger, MolecularPressureRecovery)):
        result = op.evaluate(actual[0], actual[1], policy=policy)
    elif isinstance(op, MolecularCascadeSpec):
        result = solve_molecular_cascade(op, actual[0], actual[1])
    elif isinstance(op, MolecularFlashSeparatorSpec):
        result = evaluate_molecular_flash_separator(op, actual[0])
        phases = dict(
            zip(result.diagnostics.get("outlet_phase_labels", ()), result.outlets, strict=True)
        )
        return result, tuple(phases.get(p) for p in ("gas", "liquid"))
    else:
        raise ValueError("unsupported operation")
    return result, result.outlets


@dataclass(frozen=True)
class FlowsheetGraph:
    sources: tuple[ProcessSource, ...]
    units: tuple[ProcessUnit, ...]
    outlets: Mapping[str, str]
    tears: tuple[ProcessTear, ...] = ()
    policy: FlowsheetPolicy = field(default_factory=FlowsheetPolicy)
    constraints: tuple[ProcessConstraint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("sources", "units", "tears", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "outlets", dict(self.outlets))
        if not isinstance(self.policy, FlowsheetPolicy) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in self.outlets.items()
        ):
            raise ValueError("flowsheet requires named exports and an explicit policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "molecular-flowsheet-v1",
            "sources": [s.to_dict() for s in self.sources],
            "units": [u.to_dict() for u in self.units],
            "outlets": dict(self.outlets),
            "tears": [t.to_dict() for t in self.tears],
            "policy": self.policy.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowsheetGraph:
        if value.get("model") != "molecular-flowsheet-v1":
            raise ValueError("unknown flowsheet encoding")
        return cls(
            tuple(ProcessSource.from_dict(v) for v in value["sources"]),
            tuple(ProcessUnit.from_dict(v) for v in value["units"]),
            value["outlets"],
            tuple(ProcessTear.from_dict(v) for v in value["tears"]),
            FlowsheetPolicy.from_dict(value["policy"]),
            tuple(ProcessConstraint.from_dict(v) for v in value.get("constraints", ())),
        )

    def port_domains(self) -> dict[str, MolecularProcessDomain]:
        domains = {s.port_id: s.domain for s in self.sources}
        for unit in self.units:
            ins, outs = operation_domains(unit.operation, len(unit.inlet_ports))
            for port, domain in zip(unit.outlet_ports, outs, strict=True):
                domains[port] = domain
            for port, domain in zip(unit.inlet_ports, ins, strict=True):
                domains.setdefault(port, domain)
        return domains

    def topology_diagnostics(self) -> dict[str, Any]:
        domains = self.port_domains()
        producers = Counter(
            [s.port_id for s in self.sources] + [p for u in self.units for p in u.outlet_ports]
        )
        consumers = Counter(
            [p for u in self.units for p in u.inlet_ports] + list(self.outlets.values())
        )
        errors: list[str] = []
        if not self.sources and not self.units:
            errors.append("empty_flowsheet")
        errors.extend(
            f"duplicate_unit:{k}"
            for k, n in Counter(u.unit_id for u in self.units).items()
            if n != 1
        )
        errors.extend(f"duplicate_producer:{k}" for k, n in producers.items() if n != 1)
        errors.extend(f"multiple_consumers:{k}" for k, n in consumers.items() if n != 1)
        errors.extend(f"missing_producer:{k}" for k in consumers if k not in producers)
        errors.extend(f"unused_port:{k}" for k in producers if k not in consumers)
        errors.extend(
            f"duplicate_tear:{k}"
            for k, n in Counter(t.port_id for t in self.tears).items()
            if n != 1
        )
        tear_ids = {t.port_id for t in self.tears}
        source_ids = {s.port_id for s in self.sources}
        for tear in self.tears:
            if tear.port_id not in producers or tear.port_id in source_ids:
                errors.append(f"invalid_tear_producer:{tear.port_id}")
            if tear.port_id in self.outlets.values() or tear.port_id not in consumers:
                errors.append(f"invalid_tear_consumer:{tear.port_id}")
            if domains.get(tear.port_id) != tear.domain:
                errors.append(f"tear_domain_mismatch:{tear.port_id}")
        for unit in self.units:
            ins, _ = operation_domains(unit.operation, len(unit.inlet_ports))
            for port, domain in zip(unit.inlet_ports, ins, strict=True):
                if domains[port] != domain:
                    errors.append(f"connection_domain_mismatch:{unit.unit_id}:{port}")
        known = source_ids | tear_ids
        pending = list(self.units)
        order: list[str] = []
        while pending:
            ready = [u for u in pending if set(u.inlet_ports) <= known]
            if not ready:
                errors.append(
                    "uncut_cycle_or_missing_boundary:" + ",".join(u.unit_id for u in pending)
                )
                break
            for unit in ready:
                order.append(unit.unit_id)
                known.update(unit.outlet_ports)
                pending.remove(unit)

        # Full stream-coordinate graph, including source boundary rows and explicit
        # fixed specs. This is not merely a count of square tear closure equations.
        offset: dict[str, tuple[int, int]] = {}
        columns = 0
        for port, domain in domains.items():
            size = len(domain.model.component_ids) + 2
            offset[port] = (columns, size)
            columns += size
        rows: list[int] = []
        cols: list[int] = []
        equation = 0
        for source in self.sources:
            start, size = offset[source.port_id]
            for j in range(size):
                rows.append(equation)
                cols.append(start + j)
                equation += 1
        for unit in self.units:
            input_columns = [
                j for p in unit.inlet_ports for j in range(offset[p][0], sum(offset[p]))
            ]
            for port in unit.outlet_ports:
                start, size = offset[port]
                for j in range(size):
                    dependency = set(input_columns) | {start + j}
                    rows.extend([equation] * len(dependency))
                    cols.extend(dependency)
                    equation += 1
        for constraint in self.constraints:
            if constraint.port_id not in offset:
                errors.append(f"constraint_unknown_port:{constraint.port_id}")
                equation += 1
                continue
            start, size = offset[constraint.port_id]
            if constraint.coordinate == "component_flow":
                ids = domains[constraint.port_id].model.component_ids
                if constraint.component_id not in ids:
                    errors.append(f"constraint_unknown_component:{constraint.component_id}")
                    equation += 1
                    continue
                index = ids.index(constraint.component_id)
            else:
                index = size - (2 if constraint.coordinate == "T" else 1)
            rows.append(equation)
            cols.append(start + index)
            equation += 1
        pattern = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(equation, columns))
        matching = maximum_bipartite_matching(pattern, perm_type="column") if equation else []
        rank = int(np.count_nonzero(np.asarray(matching) >= 0))
        if equation > columns:
            errors.append("overdetermined_fixed_specifications")
        if rank < columns:
            errors.append("structurally_underdetermined")
        return {
            "errors": errors,
            "execution_order": order,
            "degrees_of_freedom": columns - equation,
            "variable_count": columns,
            "equation_count": equation,
            "structural_rank": rank,
            "unmatched_variables": columns - rank,
            "redundant_equations": equation - rank,
            "tear_variable_count": sum(len(t.variables()) for t in self.tears),
            "structural_jacobian_shape": list(pattern.shape),
            "structural_jacobian_nnz": pattern.nnz,
            "scope": "full_port_n_T_P_equations_with_source_boundaries_and_fixed_specs",
        }

    def solve(
        self,
        source_streams: Mapping[str, MolecularStream] | None = None,
        initial_streams: Mapping[str, MolecularStream] | None = None,
    ) -> FlowsheetResult:
        topology = self.topology_diagnostics()
        diagnostics: dict[str, Any] = {
            "topology": topology,
            "trials": [],
            "initialization": [],
            "trial_evaluations": 0,
            "unit_evaluations": 0,
        }
        if topology["errors"]:
            return FlowsheetResult(
                False, diagnostics=diagnostics, failure_reason="invalid_flowsheet_topology"
            )
        sources = {s.port_id: s.stream for s in self.sources}
        seeds = {t.port_id: t.initial_stream for t in self.tears}
        for supplied, declared, name in (
            (source_streams, sources, "source"),
            (initial_streams, seeds, "tear"),
        ):
            if supplied is not None:
                if set(supplied) - set(declared):
                    raise ValueError(f"undeclared {name} override")
                declared.update(supplied)
        for source in self.sources:
            source.domain.validate_stream(sources[source.port_id])
        for tear in self.tears:
            tear.domain.validate_stream(seeds[tear.port_id])
        by_id = {u.unit_id: u for u in self.units}
        ordered = [by_id[k] for k in topology["execution_order"]]
        variables = tuple(v for t in self.tears for v in t.variables())
        scales = np.array([v.scale for v in variables])
        lower = np.array([v.lower for v in variables])
        upper = np.array([v.upper for v in variables])
        initial = (
            np.concatenate([_coordinates(seeds[t.port_id]) for t in self.tears])
            if self.tears
            else np.array([])
        )
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("override tear seed exceeds declared bounds")
        slices: dict[str, slice] = {}
        start = 0
        for tear in self.tears:
            size = len(tear.variables())
            slices[tear.port_id] = slice(start, start + size)
            start += size
        deps: dict[str, set[str]] = {k: set() for k in sources}
        for unit in ordered:
            upstream: set[str] = set()
            for port in unit.inlet_ports:
                upstream.update({port} if port in slices else deps[port])
            for port in unit.outlet_ports:
                deps[port] = upstream.copy()
        mask = np.zeros((start, start), dtype=bool)
        for tear in self.tears:
            for dependency in deps[tear.port_id] | {tear.port_id}:
                mask[slices[tear.port_id], slices[dependency]] = True
        diagnostics["tear_jacobian"] = {
            "format": "csr",
            "shape": [start, start],
            "structural_nnz": int(np.count_nonzero(mask)),
        }
        cache: dict[
            tuple[float, ...],
            tuple[np.ndarray, dict[str, MolecularStream | None], dict[str, MolecularProcessResult]],
        ] = {}
        # Exact native-unit reuse within one fixed graph solve. The existing
        # logical unit budget is charged on hits too, preserving refusal paths.
        unit_cache = {unit.unit_id: OrderedDict() for unit in ordered}
        native_operations = (ProcessMixer, ProcessSplitter, MolecularHeatDuty,
                             MolecularPressureStage, MolecularHeatExchanger,
                             MolecularPressureRecovery, MolecularFlashSeparatorSpec,
                             MolecularCascadeSpec)
        last_results: dict[str, MolecularProcessResult] = {}

        def trial(
            point: np.ndarray,
        ) -> tuple[
            np.ndarray, dict[str, MolecularStream | None], dict[str, MolecularProcessResult]
        ]:
            nonlocal last_results
            key = tuple(float(v) for v in point)
            if key in cache:
                return cache[key]
            if diagnostics["trial_evaluations"] >= self.policy.maximum_trial_evaluations:
                raise ResidualDomainError("flowsheet_trial_budget_exhausted")
            diagnostics["trial_evaluations"] += 1
            row: dict[str, Any] = {"tear_coordinates": list(key), "unit_failures": {}}
            diagnostics["trials"].append(row)
            guesses: dict[str, MolecularStream] = {}
            for tear in self.tears:
                values = point[slices[tear.port_id]]
                rate = fsum(values[:-2])
                if rate < 0:
                    row["failure_reason"] = f"negative_flow_tear:{tear.port_id}"
                    raise ResidualDomainError(row["failure_reason"])
                seed = seeds[tear.port_id]
                fractions = ((values[:-2] / rate).tolist() if rate > 0
                             else seed.state.mole_fractions)
                # T/P are already graph coordinates. Reuse the original scalar
                # density equation so a changed composition can leave the old
                # density seed's invalid-pressure region. Only EOS probes may
                # cross that region; accepted streams still satisfy the domain.
                try:
                    closed = _molecular_density_root(
                        tear.domain.model.residual_model, float(values[-2]), float(values[-1]),
                        fractions, initial_density_mol_m3=seed.state.molar_density_mol_m3,
                        density_bounds_mol_m3=tear.domain.density_basin.density_bounds_mol_m3,
                        policy=self.policy.root,
                    )
                    row.setdefault("tear_state_roots", {})[tear.port_id] = closed.diagnostics
                    if not closed.success or closed.value is None:
                        raise ResidualDomainError(closed.failure_reason)
                    state = tear.domain.state(float(values[-2]), exp(closed.value[0]), fractions)
                except (ValueError, ArithmeticError) as exc:
                    row["failure_reason"] = f"tear_state:{tear.port_id}:{exc}"
                    raise ResidualDomainError(row["failure_reason"]) from exc
                guesses[tear.port_id] = MolecularStream(state, rate)
            produced: dict[str, MolecularStream | None] = dict(sources)
            results: dict[str, MolecularProcessResult] = {}
            last_results = results
            for unit in ordered:
                if any(p not in guesses and p not in produced for p in unit.inlet_ports):
                    result = MolecularProcessResult(False, failure_reason="upstream_unit_failed")
                elif diagnostics["unit_evaluations"] >= self.policy.maximum_unit_evaluations:
                    result = MolecularProcessResult(
                        False, failure_reason="flowsheet_unit_budget_exhausted"
                    )
                else:
                    diagnostics["unit_evaluations"] += 1
                    inlets = tuple(
                        guesses[p] if p in guesses else produced[p] for p in unit.inlet_ports
                    )
                    try:
                        memo = unit_cache[unit.unit_id]
                        # Streams include every caloric/EOS output as well as
                        # T/rho/x/rate. No rounded or reduced-coordinate keys.
                        use_cache = type(unit.operation) in native_operations
                        try:
                            hash(inlets)
                        except TypeError:
                            use_cache = False
                        if use_cache and inlets in memo:
                            result, outputs = memo[inlets]
                            memo.move_to_end(inlets)
                        else:
                            result, outputs = _evaluate_unit(unit, inlets, self.policy.root)
                            if use_cache:
                                memo[inlets] = (result, outputs)
                                if len(memo) > 32:
                                    memo.popitem(last=False)
                        if result.success:
                            produced.update(zip(unit.outlet_ports, outputs, strict=True))
                    except (ValueError, ArithmeticError) as exc:
                        result = MolecularProcessResult(False, failure_reason=str(exc))
                results[unit.unit_id] = result
                if not result.success:
                    row["unit_failures"][unit.unit_id] = result.to_dict()
            if row["unit_failures"]:
                row["failure_reason"] = "flowsheet_unit_failures"
                raise ResidualDomainError(row["failure_reason"])
            residuals = []
            for tear in self.tears:
                actual = produced[tear.port_id]
                if actual is None:
                    row["failure_reason"] = f"absent_phase_tear:{tear.port_id}"
                    raise ResidualDomainError(row["failure_reason"])
                tear.domain.validate_stream(actual)
                residuals.extend(_coordinates(actual) - point[slices[tear.port_id]])
            residual = np.asarray(residuals)
            row["scaled_residual_norm"] = float(np.max(np.abs(residual / scales))) if start else 0.0
            cache[key] = residual, produced, results
            return cache[key]

        def function(point: np.ndarray) -> np.ndarray:
            return trial(point)[0]

        def jacobian(point: np.ndarray) -> sparse.csr_matrix:
            dense = finite_jacobian(
                function,
                point,
                scales,
                lower=lower,
                upper=upper,
                relative_step=self.policy.relative_difference_step,
            )
            return sparse.csr_matrix(np.where(mask, dense, 0.0))

        try:
            if start:
                for _ in range(self.policy.initialization_iterations):
                    residual = function(initial)
                    next_point = initial + self.policy.initialization_damping * residual
                    accepted = bool(np.all(next_point >= lower) and np.all(next_point <= upper))
                    diagnostics["initialization"].append(
                        {
                            "coordinates": initial.tolist(),
                            "scaled_residual_norm": float(np.max(np.abs(residual / scales))),
                            "next_in_bounds": accepted,
                        }
                    )
                    if not accepted:
                        break
                    initial = next_point
                root = solve_root(
                    function,
                    initial,
                    jacobian=jacobian,
                    variables=variables,
                    residual_scales=scales,
                    policy=self.policy.root,
                )
                diagnostics["root"] = root.diagnostics
                if not root.success:
                    raise ResidualDomainError(root.failure_reason)
                point = root.value
                reduced = jacobian(point).toarray() * scales[np.newaxis, :] / scales[:, np.newaxis]
                singular = np.linalg.svd(reduced, compute_uv=False)
                threshold = self.policy.rank_relative_tolerance * max(1.0, float(np.max(singular)))
                rank = int(np.count_nonzero(singular > threshold))
                topology.update(
                    numerical_rank=topology["variable_count"] - start + rank,
                    tear_numerical_rank=rank,
                    tear_singular_values=singular.tolist(),
                    local_rank_threshold=threshold,
                    numerical_degrees_of_freedom=start - rank,
                )
                if rank != start:
                    raise ResidualDomainError("numerically_underdetermined_flowsheet")
            else:
                point = initial
                topology.update(
                    numerical_rank=topology["variable_count"], numerical_degrees_of_freedom=0
                )
            residual, produced, results = trial(point)
            final_outlets = {
                k: stream for k, p in self.outlets.items() if (stream := produced[p]) is not None
            }
            absent = tuple(k for k, p in self.outlets.items() if produced[p] is None)
            heat = fsum(r.heat_duty_W for r in results.values())
            work = fsum(r.shaft_work_W for r in results.values())
            utilities = [unit_utility_rates(u.operation, results[u.unit_id]) for u in self.units]
            incoming = list(sources.values())
            outgoing = list(final_outlets.values())
            components = {c for s in incoming + outgoing for c in s.state.component_ids}

            def rate(streams: list[MolecularStream], component: str) -> float:
                return fsum(
                    s.molar_rate_mol_s
                    * s.state.mole_fractions[s.state.component_ids.index(component)]
                    for s in streams
                    if component in s.state.component_ids
                )

            diagnostics.update(
                final_tear_coordinates=point.tolist(),
                scaled_closure_norm=float(np.max(np.abs(residual / scales))) if start else 0.0,
                component_balance_mol_s={
                    c: rate(outgoing, c) - rate(incoming, c) for c in sorted(components)
                },
                energy_residual_W=fsum(s.enthalpy_rate_W for s in outgoing)
                - fsum(s.enthalpy_rate_W for s in incoming)
                - heat
                - work,
                heat_import_W=fsum(r["heat_import_W"] for r in utilities),
                heat_export_W=fsum(r["heat_export_W"] for r in utilities),
                shaft_import_W=fsum(r["shaft_import_W"] for r in utilities),
                shaft_export_W=fsum(r["shaft_export_W"] for r in utilities),
            )
            rate_scale = max(
                fsum(s.molar_rate_mol_s for s in incoming),
                fsum(s.molar_rate_mol_s for s in outgoing),
            )
            energy_scale = max(
                fsum(abs(s.enthalpy_rate_W) for s in incoming),
                fsum(abs(s.enthalpy_rate_W) for s in outgoing),
                fsum(abs(r.heat_duty_W) + abs(r.shaft_work_W) for r in results.values()),
            )
            component_tolerance = (
                self.policy.component_balance_absolute_mol_s
                + self.policy.balance_relative_tolerance * rate_scale
            )
            energy_tolerance = (
                self.policy.energy_balance_absolute_W
                + self.policy.balance_relative_tolerance * energy_scale
            )
            diagnostics["balance_tolerances"] = {
                "component_mol_s": component_tolerance,
                "energy_W": energy_tolerance,
            }
            if (
                any(
                    abs(v) > component_tolerance
                    for v in diagnostics["component_balance_mol_s"].values()
                )
                or abs(diagnostics["energy_residual_W"]) > energy_tolerance
            ):
                raise ResidualDomainError("flowsheet_conservation_failure")
            return FlowsheetResult(
                True, final_outlets, absent, heat, work, results, diagnostics, port_streams=produced
            )
        except (ValueError, ArithmeticError) as exc:
            return FlowsheetResult(
                False, unit_results=last_results, diagnostics=diagnostics, failure_reason=str(exc)
            )
