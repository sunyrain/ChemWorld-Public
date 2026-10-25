"""Passive electrical load on the original periodic ion-exchange membrane pair.

Original finite liquids own n/H. The shared phase integrator's two nonnegative
utility coordinates represent bath input and external load output in this slice;
the public receipt always names those physical ports, never a heater/condenser.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np

from chemworld.physchem.electrodialysis import (
    DEFAULT_ED_POLICY,
    ElectrodialysisSpec,
    electrodialysis_inventory,
    electrodialysis_point,
)
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class ReverseElectrodialysisSpec:
    cell_pair: ElectrodialysisSpec
    maximum_load_resistance_Ohm: float
    maximum_load_power_W: float
    maximum_bath_heating_W: float
    minimum_load_resistance_Ohm: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.cell_pair, ElectrodialysisSpec):
            raise ValueError("RED requires the original periodic membrane pair declaration")
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (
                self.maximum_load_resistance_Ohm,
                self.maximum_load_power_W,
                self.maximum_bath_heating_W,
            )
        ) or (
            isinstance(self.minimum_load_resistance_Ohm, bool)
            or not isfinite(self.minimum_load_resistance_Ohm)
            or not 0 <= self.minimum_load_resistance_Ohm < self.maximum_load_resistance_Ohm
        ):
            raise ValueError("passive RED needs finite load resistance, output and bath limits")

    @property
    def properties(self) -> MultiIonMembraneProperties:
        return self.cell_pair.properties

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return self.cell_pair.enthalpy(index, amounts, temperature)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "schema": "periodic-resistive-red-v1",
                    "cell_pair": self.cell_pair.to_dict(),
                    "minimum_load_resistance_Ohm": self.minimum_load_resistance_Ohm,
                    "maximum_load_resistance_Ohm": self.maximum_load_resistance_Ohm,
                    "maximum_load_power_W": self.maximum_load_power_W,
                    "maximum_bath_heating_W": self.maximum_bath_heating_W,
                }
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReverseElectrodialysisSpec:
        data = dict(payload)
        if data.pop("schema") != "periodic-resistive-red-v1":
            raise ValueError("unsupported passive reverse electrodialysis declaration")
        data["cell_pair"] = ElectrodialysisSpec.from_dict(data["cell_pair"])
        return cls(**data)


def validate_red_load_program(
    spec: ReverseElectrodialysisSpec, program: ScalarControlProgram
) -> None:
    if not isinstance(program, ScalarControlProgram) or program.unit not in {"A", "Ohm"}:
        raise ValueError("RED supports an explicit open circuit or passive resistance")
    for _, target in program.setpoints:
        if program.unit == "A":
            if target != 0:
                raise ValueError("open-circuit RED cannot impose a nonzero current")
        elif not spec.minimum_load_resistance_Ohm <= target <= spec.maximum_load_resistance_Ohm:
            raise ValueError("load resistance is outside its declared physical range")


def reverse_electrodialysis_point(
    spec: ReverseElectrodialysisSpec,
    amounts: tuple[Mapping[str, float], ...],
    resistance_Ohm: float | None,
    *,
    policy: RootPolicy = DEFAULT_ED_POLICY,
    numerical_trial: bool = False,
) -> dict[str, Any]:
    """None is an actual open circuit; zero Ohm is an actual short circuit."""
    if resistance_Ohm is not None and (
        isinstance(resistance_Ohm, bool)
        or not isfinite(resistance_Ohm)
        or not spec.minimum_load_resistance_Ohm
        <= resistance_Ohm
        <= spec.maximum_load_resistance_Ohm
    ):
        raise ValueError("passive load resistance is outside the declared finite range")
    unit, target = ("A", 0.0) if resistance_Ohm is None else ("Ohm", resistance_Ohm)
    row = electrodialysis_point(
        spec.cell_pair, amounts, target, unit, policy=policy, numerical_trial=numerical_trial
    )
    if row["status"] != "converged":
        return {**row, "load_mode": "open_circuit" if resistance_Ohm is None else "resistance"}
    current, voltage = row["current_A"], row["voltage_V"]
    power = 0.0 if resistance_Ohm is None else current * current * resistance_Ohm
    p, t = spec.cell_pair.reservoir_pressure_Pa, spec.cell_pair.temperature_K
    flows = (row["diluate_rates_mol_s"], row["concentrate_rates_mol_s"])
    hdot, gdot = [], []
    for n, flow in zip(amounts, flows, strict=True):
        hdot.append(spec.properties.enthalpy_rate(n, flow, t, p))
        mu = spec.properties.database.chemical_potentials(n, temperature_K=t, pressure_Pa=p)
        gdot.append(
            sum(mu.chemical_potentials_J_mol[k] * flow[k] for k in spec.properties.component_ids)
        )
    closure = row["electrical_power_W"] + power
    free_energy_residual = sum(gdot) + power + row["dissipation_W"]
    tolerance = 1e-10 + 1e-6 * max(power, abs(sum(gdot)), abs(row["dissipation_W"]))
    if abs(closure) > tolerance or abs(free_energy_residual) > tolerance:
        raise RuntimeError(
            "passive RED circuit or same-source chemical/electrical power is unclosed"
        )
    if row["dissipation_W"] < -tolerance:
        raise RuntimeError("passive RED has negative internal irreversible dissipation")
    # Ideal closed inventories have sum Hdot=0 exactly; no hidden heat of mixing
    # is substituted. Keep the actual same-table derivative in the heat balance.
    bath = float(sum(hdot)) + power
    if bath < -1e-12:
        raise RuntimeError("RED bath sign is inconsistent with its ideal closed isotherm")
    return {
        **row,
        "load_mode": "open_circuit" if resistance_Ohm is None else "resistance",
        "resistance_Ohm": resistance_Ohm,
        "load_power_W": power,
        "circuit_residual_V": voltage + current * resistance_Ohm
        if resistance_Ohm is not None
        else None,
        "terminal_load_power_residual_W": closure,
        "liquid_enthalpy_rates_W": hdot,
        "bath_heat_into_liquid_W": bath,
        "gibbs_energy_rate_W": float(sum(gdot)),
        "free_energy_power_residual_W": float(free_energy_residual),
        "load_boundary": (
            "external passive resistor; load heat is outside the liquid control volume"
        ),
    }


def reverse_electrodialysis_inventory(
    spec: ReverseElectrodialysisSpec, diluate: Mapping[str, float], concentrate: Mapping[str, float]
) -> TransportInventory:
    return electrodialysis_inventory(spec.cell_pair, diluate, concentrate)


def advance_reverse_electrodialysis(
    spec: ReverseElectrodialysisSpec,
    state: TransportInventory,
    program: ScalarControlProgram,
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[PhaseInventoryAdvance, dict[str, Any]]:
    validate_red_load_program(spec, program)
    pair, prop = spec.cell_pair, spec.properties
    t, p = pair.temperature_K, pair.reservoir_pressure_Pa
    if (
        len(state.amounts_mol) != 2
        or len(state.enthalpies_J) != 2
        or isinstance(duration_s, bool)
        or not isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError("finite RED needs two original n/H reservoirs and nonnegative seconds")
    for i, n in enumerate(state.amounts_mol):
        prop.validate_amounts(n)
        if not isclose(state.enthalpies_J[i], pair.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("RED current liquid H differs from its declared isotherm")
    counters = {"pore_solves": 0, "rhs_evaluations": 0, "off_neutral_numerical_probes": 0}
    cache: list[Any] = [None, None]

    def point(candidate: Any) -> dict[str, Any]:
        key = (
            candidate.time_s,
            tuple(tuple(n[k] for k in prop.component_ids) for n in candidate.amounts_mol),
        )
        if cache[0] == key:
            return cache[1]
        for n in candidate.amounts_mol:
            if abs(sum(prop.charges[k] * n[k] for k in prop.ion_ids)) > 1e-12:
                counters["off_neutral_numerical_probes"] += 1
        row = reverse_electrodialysis_point(
            spec,
            candidate.amounts_mol,
            program.value_at(candidate.time_s) if program.unit == "Ohm" else None,
            numerical_trial=True,
        )
        counters["pore_solves"] += 2
        if row["status"] != "converged":
            raise RuntimeError(f"passive RED coupled pore failed: {row['solver']}")
        cache[:] = [key, row]
        return row

    def rates(candidate: Any, flashes: Any) -> PhaseInventoryRates:
        row = point(candidate)
        counters["rhs_evaluations"] += 1
        return PhaseInventoryRates(
            np.array(
                [
                    [row[field][k] for k in prop.component_ids]
                    for field in ("diluate_rates_mol_s", "concentrate_rates_mol_s")
                ]
            ),
            np.array(row["liquid_enthalpy_rates_W"]),
            row["bath_heat_into_liquid_W"],
            row["load_power_W"],
            (),
        )

    stops: list[Any] = []
    for i in range(2):
        stops.extend(
            (
                (
                    f"water_minimum:{i}",
                    lambda s, _, i=i: s.amounts_mol[i][prop.water_id] - pair.minimum_water_mol,
                ),
                (
                    f"capacity:{i}",
                    lambda s, _, i=i: (
                        pair.capacities_L[i] - 1000 * prop.volume(s.amounts_mol[i], t, p)
                    ),
                ),
                (
                    f"ion_minimum:{i}",
                    lambda s, _, i=i: (
                        min(prop.concentrations(s.amounts_mol[i]).values())
                        - pair.minimum_ion_concentration_mol_m3
                    ),
                ),
            )
        )
    stops.extend(
        (
            (
                "voltage_compliance",
                lambda s, _: pair.maximum_voltage_V - abs(point(s)["voltage_V"]),
            ),
            (
                "current_compliance",
                lambda s, _: pair.maximum_current_A - abs(point(s)["current_A"]),
            ),
            ("load_power_limit", lambda s, _: spec.maximum_load_power_W - point(s)["load_power_W"]),
            (
                "bath_heating_limit",
                lambda s, _: spec.maximum_bath_heating_W - point(s)["bath_heat_into_liquid_W"],
            ),
        )
    )
    initial_point = point(state)
    finish = state.time_s + duration_s
    boundaries = sorted(
        {float(x) for x, _ in program.setpoints if state.time_s < x < finish} | {finish}
    )
    current, pieces, trajectory = state, [], [state]
    for end in boundaries:
        result = integrate_phase_inventory(
            spec,
            current,
            duration_s=end - current.time_s,
            rates=rates,
            closure_indices=(),
            stop_functions=stops,
            enthalpy_bounds=lambda i, n: (-np.inf, np.inf),
            initial_boundary="stop",
            progress=progress,
            solver_id="finite_periodic_resistive_red_nH_v1",
        )
        pieces.append(result)
        current = result.final_state
        trajectory.extend(result.trajectory[1:])
        if result.stopping_event:
            break
    for i, n in enumerate(current.amounts_mol):
        prop.validate_amounts(n)
        if not isclose(current.enthalpies_J[i], pair.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-5):
            raise RuntimeError("finite RED failed its original isothermal enthalpy closure")
    bath_in = current.heater_energy_J - state.heater_energy_J
    load_out = current.condenser_energy_J - state.condenser_energy_J
    delta_h = sum(current.enthalpies_J) - sum(state.enthalpies_J)
    initial_g = sum(prop.liquid(n, t, p).gibbs_energy_J for n in state.amounts_mol)
    final_g = sum(prop.liquid(n, t, p).gibbs_energy_J for n in current.amounts_mol)
    remaining = initial_g - final_g - load_out
    if remaining < -(1e-8 + 1e-6 * max(abs(initial_g - final_g), load_out)):
        raise RuntimeError("finite RED output exceeds the current salt-gradient free energy")
    combined = PhaseInventoryAdvance(
        state,
        current,
        tuple(trajectory),
        pieces[-1].stopping_event,
        {
            **counters,
            "intervals": [r.solver_diagnostic for r in pieces],
            "utility_coordinate_mapping": {
                "heater_energy_J": "bath_heat_into_liquid_J",
                "condenser_energy_J": "load_energy_delivered_J",
            },
        },
        {
            k: sum(n[k] for n in current.amounts_mol) - sum(n[k] for n in state.amounts_mol)
            for k in prop.component_ids
        },
        delta_h - bath_in + load_out,
    )
    return combined, {
        "initial": initial_point,
        "final": point(current),
        "load_energy_delivered_J": load_out,
        "electrical_work_into_liquid_J": -load_out,
        "bath_heat_into_liquid_J": bath_in,
        "liquid_enthalpy_change_J": delta_h,
        "initial_gibbs_energy_J": initial_g,
        "final_gibbs_energy_J": final_g,
        "free_energy_decrease_J": initial_g - final_g,
        "free_energy_decrease_minus_load_J": remaining,
        "energy_scope": (
            "signed electrical input + bath heat input = original liquid enthalpy change"
        ),
        "bath_allocation": "equal halves to the two explicitly isothermal reservoir baths",
        "load_boundary": "external load energy is not added as heat to the liquid a second time",
    }
