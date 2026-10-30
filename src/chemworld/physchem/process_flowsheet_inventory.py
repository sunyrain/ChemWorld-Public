"""One finite n/H system around an algebraic molecular process graph.

Every boundary port belongs to a declared tank. Internal recycle streams are
algebraic and carry no second inventory. The same ODE integrates signed heat
and shaft imports/exports; PH closures use the graph's original caloric model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.physchem.molecular_processes import MolecularProcessDomain
from chemworld.physchem.molecular_tanks import (
    molecular_tank_ph,
    molecular_tank_tp,
    validate_molecular_amounts,
)
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.process_flowsheet import FlowsheetGraph
from chemworld.physchem.residual_system import DEFAULT_ROOT_POLICY
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class FlowsheetTank:
    vessel_id: str
    domain: MolecularProcessDomain
    pressure_Pa: float
    phase_id: str
    capacity_mol: float
    capacity_L: float
    reserve_mol: float = 1e-8

    def __post_init__(self) -> None:
        if (not isinstance(self.vessel_id, str) or not self.vessel_id
                or self.phase_id not in {"gas", "liquid"}
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in
                       (self.pressure_Pa, self.capacity_mol, self.capacity_L, self.reserve_mol))
                or self.reserve_mol >= self.capacity_mol):
            raise ValueError("flowsheet tanks need explicit positive finite physical limits")

    def to_dict(self) -> dict[str, Any]:
        return {"vessel_id": self.vessel_id, "domain": self.domain.to_dict(),
                "pressure_Pa": self.pressure_Pa, "phase_id": self.phase_id,
                "capacity_mol": self.capacity_mol, "capacity_L": self.capacity_L,
                "reserve_mol": self.reserve_mol}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowsheetTank:
        fields = dict(value)
        fields["domain"] = MolecularProcessDomain.from_dict(fields["domain"])
        return cls(**fields)


@dataclass(frozen=True)
class ProcessFlowsheetInventorySpec:
    graph: FlowsheetGraph
    tanks: tuple[FlowsheetTank, ...]
    source_tanks: dict[str, str]
    outlet_tanks: dict[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tanks", tuple(self.tanks))
        object.__setattr__(self, "source_tanks", dict(self.source_tanks))
        object.__setattr__(self, "outlet_tanks", dict(self.outlet_tanks))
        ids = self.vessel_ids
        if (not ids or len(ids) != len(set(ids))
                or set(self.source_tanks) != {s.port_id for s in self.graph.sources}
                or set(self.outlet_tanks) != set(self.graph.outlets)
                or set(self.source_tanks.values()) | set(self.outlet_tanks.values()) != set(ids)):
            raise ValueError("each graph boundary needs exactly one declared finite tank owner")
        if not any(s.stream.molar_rate_mol_s > 0 for s in self.graph.sources):
            raise ValueError("finite flowsheet needs a declared positive feed")
        def same_reference(model: MolecularCaloricModel) -> bool:
            return model is self.properties or model.to_dict() == self.properties.to_dict()

        if any(not same_reference(t.domain.model) for t in self.tanks):
            raise ValueError("all finite tanks must share the original graph EOS/H reference")
        domains = self.graph.port_domains()
        for port in (*self.source_tanks, *self.graph.outlets.values()):
            if not same_reference(domains[port].model):
                raise ValueError("graph boundary caloric references disagree")
        for export, name in self.outlet_tanks.items():
            tank = self.tanks[ids.index(name)]
            basin = domains[self.graph.outlets[export]].density_basin
            if basin.name != tank.domain.density_basin.name:
                raise ValueError("a graph outlet must enter its declared homogeneous phase branch")

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(t.vessel_id for t in self.tanks)

    @property
    def properties(self) -> MolecularCaloricModel:
        return self.graph.sources[0].domain.model

    def at_tp(self, index: int, amounts: Mapping[str, float], temperature: float) -> Any:
        tank = self.tanks[index]
        return molecular_tank_tp(tank.domain, tank.pressure_Pa, amounts, temperature)

    def at_ph(self, index: int, amounts: Mapping[str, float], enthalpy: float) -> Any:
        tank = self.tanks[index]
        return molecular_tank_ph(tank.domain, tank.pressure_Pa, amounts,
                                 enthalpy, DEFAULT_ROOT_POLICY)

    def enthalpy_bounds(self, index: int, amounts: Mapping[str, float]) -> tuple[float, float]:
        total = validate_molecular_amounts(self.properties, amounts)
        low, high = self.tanks[index].domain.temperature_range_K
        return (total*self.at_tp(index, amounts, low).enthalpy_J_mol,
                total*self.at_tp(index, amounts, high).enthalpy_J_mol)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "finite_process_flowsheet_v1", "graph": self.graph.to_dict(),
                "tanks": [t.to_dict() for t in self.tanks], "source_tanks": self.source_tanks,
                "outlet_tanks": self.outlet_tanks}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessFlowsheetInventorySpec:
        if value.get("model_type") != "finite_process_flowsheet_v1":
            raise ValueError("unknown finite flowsheet encoding")
        return cls(FlowsheetGraph.from_dict(value["graph"]),
                   tuple(FlowsheetTank.from_dict(t) for t in value["tanks"]),
                   dict(value["source_tanks"]), dict(value["outlet_tanks"]))


@dataclass(frozen=True)
class ProcessFlowsheetInventory(TransportInventory):
    shaft_import_energy_J: float = 0.0
    shaft_export_energy_J: float = 0.0


def _boundary_tolerance(scale: float) -> float:
    return max(scale*1e-12, 1e-14)


def process_flowsheet_inventory(
    spec: ProcessFlowsheetInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
) -> ProcessFlowsheetInventory:
    if len(amounts) != len(spec.tanks) or len(temperatures_K) != len(spec.tanks):
        raise ValueError("flowsheet requires the actual initial state of every finite tank")
    enthalpies = []
    for i, (tank, n, t) in enumerate(zip(spec.tanks, amounts, temperatures_K, strict=True)):
        total = validate_molecular_amounts(spec.properties, n)
        phase = spec.at_tp(i, n, t)
        # Initialization and event detection share the same boundary roundoff
        # allowance. Keep the actual n/H; do not clip a barely full tank.
        if (total < tank.reserve_mol-_boundary_tolerance(tank.reserve_mol)
                or total > tank.capacity_mol+_boundary_tolerance(tank.capacity_mol)
                or 1000*total/phase.molar_density_mol_m3
                > tank.capacity_L+_boundary_tolerance(tank.capacity_L)):
            raise ValueError("initial inventory exceeds declared finite tank limits")
        enthalpies.append(total*phase.enthalpy_J_mol)
    return ProcessFlowsheetInventory(tuple(dict(n) for n in amounts), tuple(enthalpies))


# Package-level factory name must differ from the owning submodule: importing
# that submodule sets ``chemworld.physchem.process_flowsheet_inventory`` to it.
# Keep the module-local factory for existing explicit imports.
initialize_process_flowsheet_inventory = process_flowsheet_inventory


class ProcessFlowsheetInventoryError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _isobaric_tangent(model: MolecularCaloricModel, phase: Any,
                     amounts: Mapping[str, float], dn: Any, dh: float) -> tuple[float, float]:
    """Original Helmholtz tangent gives dT/dt and dV/dt without a trial time step."""
    total = sum(amounts.values())
    temperature, rho = phase.temperature_K, phase.molar_density_mol_m3
    volume = total/rho
    ids = model.component_ids
    point = {"V": volume, **{f"n{i}": amounts[k] for i, k in enumerate(ids)}}

    def extensive(values: Mapping[str, Any]) -> Any:
        ns = tuple(values[f"n{i}"] for i in range(len(ids)))
        n = sum(ns)
        return n*model.residual_model.alphar(temperature, n/values["V"], tuple(x/n for x in ns))

    rt = model.gas_constant_J_mol_K*temperature
    pn = tuple(rt/volume - rt*partial_derivative(extensive, point, ("V", f"n{i}"))
               for i in range(len(ids)))
    un = model.partial_inventory_energies(temperature, volume, amounts)
    pt = phase.dP_dT_rho_Pa_K
    pv = -rho/volume*phase.dP_drho_T_Pa_m3_mol
    # dH = N Cv dT + T P_T dV + U_n.dn at dP=0.
    tangent = np.linalg.solve(np.array([[total*phase.cv_J_mol_K, temperature*pt], [pt, pv]]),
                              np.array([dh-float(np.dot(un, dn)), -float(np.dot(pn, dn))]))
    return float(tangent[0]), float(tangent[1])


def advance_process_flowsheet(
    spec: ProcessFlowsheetInventorySpec,
    state: ProcessFlowsheetInventory,
    duration_s: float,
    *,
    rate_scale: float = 1.0,
    progress: Callable[[float, int], None] | None = None,
) -> tuple[PhaseInventoryAdvance, dict[str, Any]]:
    if (len(state.amounts_mol) != len(spec.tanks) or len(state.enthalpies_J) != len(spec.tanks)
            or isinstance(rate_scale, bool) or not isfinite(rate_scale) or rate_scale < 0):
        raise ValueError("finite flowsheet needs matching tanks and a nonnegative feed scale")
    indices = {name: i for i, name in enumerate(spec.vessel_ids)}
    before = tuple(spec.at_ph(i, n, h) for i, (n, h) in
                   enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True)))
    source_only = {indices[name] for name in set(spec.source_tanks.values())
                   - set(spec.outlet_tanks.values())}
    counters = {"graph_solves": 0, "unit_evaluations": 0, "boundary_tangent_evaluations": 0}
    last: dict[str, Any] = {}
    rate_cache: list[Any] = [None, None]

    def rates(candidate: Any, phases: Any) -> PhaseInventoryRates:
        feed_phases = {s.port_id: before[indices[spec.source_tanks[s.port_id]]]
                       if indices[spec.source_tanks[s.port_id]] in source_only
                       else phases[indices[spec.source_tanks[s.port_id]]]
                       for s in spec.graph.sources}
        key = tuple((s.port_id, feed_phases[s.port_id].temperature_K,
                     feed_phases[s.port_id].molar_density_mol_m3,
                     feed_phases[s.port_id].mole_fractions) for s in spec.graph.sources)
        if key == rate_cache[0]:
            return rate_cache[1]
        dn = np.zeros((len(spec.tanks), len(spec.properties.component_ids)))
        dh = np.zeros(len(spec.tanks))
        if rate_scale == 0:
            return PhaseInventoryRates(dn, dh, 0., 0., ())
        feeds = {}
        for source in spec.graph.sources:
            i = indices[spec.source_tanks[source.port_id]]
            stream = MolecularStream(feed_phases[source.port_id],
                                     source.stream.molar_rate_mol_s*rate_scale)
            source.domain.validate_stream(stream)
            feeds[source.port_id] = stream
            dn[i] -= stream.molar_rate_mol_s*np.array(stream.state.mole_fractions)
            dh[i] -= stream.enthalpy_rate_W
        solved = spec.graph.solve(source_streams=feeds)
        counters["graph_solves"] += 1
        counters["unit_evaluations"] += int(solved.diagnostics.get("unit_evaluations", 0))
        last.clear()
        last.update(solved.to_dict())
        if not solved.success:
            raise ProcessFlowsheetInventoryError("finite flowsheet algebraic solve failed",
                                                   {"flowsheet": last, **counters})
        for name, stream in solved.outlets.items():
            i = indices[spec.outlet_tanks[name]]
            tank = spec.tanks[i]
            # A pressure-changing outlet needs an explicit graph unit. Mixing
            # into a different declared tank pressure must not invent shaft/heat.
            if not isclose(stream.state.pressure_Pa, tank.pressure_Pa,
                           rel_tol=1e-7, abs_tol=1e-4):
                raise ValueError("flowsheet outlet requires an explicit tank-pressure transition")
            # The incoming stream can be hotter/colder than the tank's mixed
            # state limits. Its phase/EOS is declared by the graph; the actual
            # mixed n/H is bounded by the tank's own temperature events.
            dn[i] += stream.molar_rate_mol_s*np.array(stream.state.mole_fractions)
            dh[i] += stream.enthalpy_rate_W
        heat_in = solved.diagnostics["heat_import_W"]
        heat_out = solved.diagnostics["heat_export_W"]
        work_in = solved.diagnostics["shaft_import_W"]
        work_out = solved.diagnostics["shaft_export_W"]
        result = PhaseInventoryRates(dn, dh, heat_in, heat_out, (),
                                     shaft_import_W=work_in, shaft_export_W=work_out)
        rate_cache[:] = [key, result]
        return result

    def limit_margin(candidate: Any, phases: Any, index: int, kind: str) -> float:
        tank, n = spec.tanks[index], candidate.amounts_mol[index]
        total = sum(n.values())
        if kind == "reserve":
            margin, scale = total-tank.reserve_mol, tank.reserve_mol
        elif kind == "capacity":
            margin, scale = tank.capacity_mol-total, tank.capacity_mol
        elif kind == "volume":
            margin = tank.capacity_L-1000*total/phases[index].molar_density_mol_m3
            scale = tank.capacity_L
        else:
            low, high = spec.enthalpy_bounds(index, n)
            margin = candidate.enthalpies_J[index]-low if kind == "temperature_low" \
                else high-candidate.enthalpies_J[index]
            scale = max(abs(low), abs(high), 1.)
        # A full receiver with an absent outlet is not a physical stop. If a
        # phase later appears, the original event becomes active immediately.
        tolerance = _boundary_tolerance(scale)
        if -tolerance <= margin <= tolerance:
            contribution = rates(candidate, phases)
            net_matter = float(sum(contribution.amounts_mol_s[index]))
            inactive = (kind == "capacity" and net_matter <= 0) or (
                kind == "reserve" and net_matter >= 0) or (
                not np.any(contribution.amounts_mol_s[index])
                and contribution.enthalpy_W[index] == 0)
            if not inactive and kind in {"temperature_low", "temperature_high"}:
                # Uniform same-state advection is an exact isothermal direction.
                # Recognize it from physical ports instead of letting a rounded
                # 2x2 solve randomly label its mathematically zero dT as outward.
                phase = phases[index]
                dn = contribution.amounts_mol_s[index]
                expected_heat = net_matter*phase.enthalpy_J_mol
                eps = 64*np.finfo(float).eps
                inactive = bool(np.all(np.abs(dn-net_matter*np.array(phase.mole_fractions))
                                       <= eps*max(float(np.sum(np.abs(dn))), 1e-30))) and abs(
                    contribution.enthalpy_W[index]-expected_heat
                ) <= eps*max(abs(contribution.enthalpy_W[index]), abs(expected_heat), 1e-30)
            if not inactive and kind in {"volume", "temperature_low", "temperature_high"}:
                counters["boundary_tangent_evaluations"] += 1
                dt, dv = _isobaric_tangent(spec.properties, phases[index], n,
                                           contribution.amounts_mol_s[index],
                                           contribution.enthalpy_W[index])
                inward = {"volume": -dv, "temperature_low": dt, "temperature_high": -dt}[kind]
                inactive = inward > 0
            if inactive:
                return tolerance
        return margin

    stops: list[tuple[str, Callable[..., float]]] = []
    sources = set(spec.source_tanks.values())
    sinks = set(spec.outlet_tanks.values())
    for i, tank in enumerate(spec.tanks):
        if tank.vessel_id in sources:
            stops.append((f"reserve:{tank.vessel_id}",
                          lambda s, p, i=i: limit_margin(s, p, i, "reserve")))
        if tank.vessel_id in sinks:
            stops.extend(((f"capacity:{tank.vessel_id}",
                           lambda s, p, i=i: limit_margin(s, p, i, "capacity")),
                          (f"volume:{tank.vessel_id}",
                           lambda s, p, i=i: limit_margin(s, p, i, "volume"))))
        # Homogeneous source-only drainage preserves T/P/composition exactly.
        # Its n/H still integrates, but no PH trial at a fictitious negative
        # overshoot is needed before the real positive-reserve root is located.
        if i not in source_only:
            stops.extend(((f"temperature_low:{tank.vessel_id}",
                           lambda s, p, i=i: limit_margin(s, p, i, "temperature_low")),
                          (f"temperature_high:{tank.vessel_id}",
                           lambda s, p, i=i: limit_margin(s, p, i, "temperature_high"))))
    result = integrate_phase_inventory(
        spec, state, duration_s=duration_s, rates=rates,
        closure_indices=tuple(i for i in range(len(spec.tanks)) if i not in source_only),
        stop_functions=stops,
        enthalpy_bounds=spec.enthalpy_bounds, include_shaft_work=True,
        initial_boundary="stop", progress=progress, diagnostic_counters=counters,
        solver_id="finite_process_flowsheet_bdf_v1",
    )
    final = result.final_state
    after = tuple(spec.at_ph(i, n, h) for i, (n, h) in
                  enumerate(zip(final.amounts_mol, final.enthalpies_J, strict=True)))
    volumes = [tuple(sum(n.values())/p.molar_density_mol_m3 for n, p in
                     zip(s.amounts_mol, phases, strict=True))
               for s, phases in ((state, before), (final, after))]
    pressure_work = sum(t.pressure_Pa*(v1-v0) for t, v0, v1 in
                        zip(spec.tanks, *volumes, strict=True))
    heat_in = final.heater_energy_J-state.heater_energy_J
    heat_out = final.condenser_energy_J-state.condenser_energy_J
    work_in = final.shaft_import_energy_J-state.shaft_import_energy_J
    work_out = final.shaft_export_energy_J-state.shaft_export_energy_J
    du = sum(sum(n.values())*p.internal_energy_J_mol for n, p in
             zip(final.amounts_mol, after, strict=True)) - sum(
                 sum(n.values())*p.internal_energy_J_mol for n, p in
                 zip(state.amounts_mol, before, strict=True))
    return result, {
        "last_algebraic_evaluation": last, "rate_scale": rate_scale, **counters,
        "actual_duration_s": result.actual_duration_s,
        "heater_energy_J": heat_in, "condenser_energy_J": heat_out,
        "shaft_import_energy_J": work_in, "shaft_export_energy_J": work_out,
        "heat_J": heat_in-heat_out, "shaft_work_J": work_in-work_out,
        "pressure_work_J": pressure_work, "internal_energy_change_J": du,
        "internal_energy_residual_J": du-(heat_in-heat_out)-(work_in-work_out)+pressure_work,
        "initial_volumes_m3": volumes[0], "final_volumes_m3": volumes[1],
    }
