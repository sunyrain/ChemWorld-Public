"""Transient mixed-tank transport, reaction, heat and a finite galvanostatic cell.

Inventories include carrier components, passive reference tracers and integrated
outlets. The electrical model is an explicit idealized two-electron channel,
not a calibration of the separate batch electrochemical instrument.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.foundation.solvents import (
    DENSITY_RATIOS,
    HEAT_CAPACITY_RATIOS,
    VISCOSITY_RATIOS,
)
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledStateView,
    EnergyPort,
    HeatFlux,
    MaterialPort,
    ProcessContribution,
    assemble_coupled_rates,
)
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL, faradaic_extent_mol
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode

FLOW_MODEL = {
    "id": "finite-reservoir-mixed-tank-v1",
    "material": "single liquid; additive carrier volumes; well-mixed outlet; finite reservoirs",
    "clock": "one tank advances; reservoirs and collected liquid have no autonomous chemistry",
    "hydraulics": "0.5 m x 2 mm line per open port; Hagen-Poiseuille; Reynolds <= 2000",
    "pump_efficiency": 0.6,
    "pump_heat": "external drive/line bath; excluded from tank heat",
    "electrode": "ideal galvanostat; two-electron forward channel; no capacitance or Nernst model",
    "electrode_transfer_rate_s_inv": 0.005,
    "faradaic_efficiency": 0.9,
    "product_selectivity": 0.9,
    "reversible_cell_voltage_V": 0.25,
    "cell_resistance_ohm": 5.0,
    "cell_voltage_limit_V": 2.5,
    "validity": "finite surrogate equipment; not externally calibrated",
}
ENERGY_NAMES = (
    "jacket_J",
    "reaction_J",
    "loss_J",
    "outlet_enthalpy_J",
    "charge_C",
    "faradaic_charge_C",
    "electrical_work_J",
    "chemical_work_J",
    "cell_heat_J",
    "ohmic_heat_J",
    "pump_work_J",
)


@dataclass(frozen=True)
class FlowResult:
    tank: np.ndarray
    outlet: np.ndarray
    temperature_K: float
    outlet_temperature_K: float
    ledger: dict[str, float]
    diagnostics: dict[str, Any]
    actual_duration_s: float = 0.0
    exchanges: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _FlowSensibleThermodynamics:
    """Conservative variable-carrier sensible energy, with the old reaction ΔH.

    The packed strategy retains H=C(t)(T-Tref), volumes in L and reference shares
    in their original units. It does not claim a formation-energy U model.
    """

    inventory_keys: tuple[str, ...]
    capacity_J_K: float
    capacity_rate_W_K: float

    def heat_capacity_j_k(self, state):
        return self.capacity_J_K

    def internal_energy_j(self, state):
        return None

    def chemical_energy_rate_w(self, state, amount_rates, contributions):
        return sum(
            c.legacy_reaction_energy_W or 0.0 for c in contributions
        ) + self.capacity_rate_W_K * (state.temperature_K - 298.15)


def integrate_flow(
    *,
    inventory: np.ndarray,
    inlet_rates: np.ndarray,
    species_count: int,
    inlet_heat_W: float,
    temperature_K: float,
    duration_s: float,
    outlet_rate_L_s: float,
    inlet_pipes: tuple[tuple[float, tuple[float, ...]], ...],
    rho_cp_J_L_K: float,
    reaction: Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, float]],
    target_temperature_K: float,
    ua_W_K: float,
    environment_temperature_K: float,
    current_mA: float,
    electrode_indices: tuple[int, int, int],
) -> FlowResult:
    count = len(inventory)
    carriers = slice(species_count, species_count + 4)
    cp = np.asarray(HEAT_CAPACITY_RATIOS) * rho_cp_J_L_K
    initial_heat = float(np.dot(inventory[carriers], cp)) * (temperature_K - 298.15)
    y0 = np.zeros(2 * count + 1 + len(ENERGY_NAMES))
    y0[:count], y0[2 * count] = inventory, initial_heat
    max_reynolds, max_pressure = 0.0, 0.0
    keys = tuple(f"species:{i}" for i in range(species_count))
    ports = tuple(
        MaterialPort(
            f"{side}:{key}",
            key,
            "feed" if side == "in" else "tank",
            "tank" if side == "in" else "collector",
            side,
        )
        for side in ("in", "out")
        for key in keys
    )
    energy_ports = (
        EnergyPort("carrier_in", "feed", "tank", "in"),
        EnergyPort("carrier_out", "tank", "collector", "out"),
        EnergyPort("electrical", "power_supply", "tank", "in"),
        EnergyPort("chemical_work", "tank", "chemical_store", "out"),
    )

    def rhs(_time: float, y: np.ndarray) -> np.ndarray:
        nonlocal max_reynolds, max_pressure
        amounts = np.maximum(y[:count], 0)
        solvents = amounts[carriers]
        volume = float(solvents.sum())
        capacity = float(np.dot(solvents, cp))
        if volume <= 1e-9 or capacity <= 0:
            raise ValueError("Flow tank exhausted")
        temperature = 298.15 + y[2 * count] / capacity
        if not 250 <= temperature <= 470:
            raise ValueError("Flow temperature outside provider domain")
        out = amounts * outlet_rate_L_s / volume
        dn, reaction_heat = reaction(amounts[:species_count], solvents, temperature)
        current = min(
            current_mA / 1000,
            (2.5 - 0.25) / 5,
            2 * FARADAY_C_PER_MOL * 0.005 * amounts[electrode_indices[0]] / 0.9,
        )
        extent = faradaic_extent_mol(
            current_A=current, duration_s=1, electrons_transferred=2, faradaic_efficiency=0.9
        )
        a, p, b = electrode_indices
        dn[a] -= extent
        dn[p] += extent * 0.9
        dn[b] += extent * 0.1
        ohmic_heat = current**2 * 5
        electrical = current * 0.25 + ohmic_heat
        chemical = current * 0.9 * 0.25
        cell_heat = electrical - chemical
        jacket = float(np.clip(4 * (target_temperature_K - temperature), -70, 90))
        loss = ua_W_K * (temperature - environment_temperature_K)
        hout = float(np.dot(out[carriers], cp)) * (temperature - 298.15)
        pump = 0.0
        pipes = (*inlet_pipes, (outlet_rate_L_s, tuple(solvents / volume)))
        for flow, fractions in pipes:
            viscosity = 1.2e-3 * float(np.exp(np.dot(fractions, np.log(VISCOSITY_RATIOS))))
            density = 950 * float(np.dot(fractions, DENSITY_RATIOS))
            q = flow * 1e-3
            velocity = q / (np.pi * 0.002**2 / 4)
            reynolds = density * velocity * 0.002 / viscosity
            pressure = 128 * viscosity * 0.5 * q / (np.pi * 0.002**4)
            max_reynolds, max_pressure = max(max_reynolds, reynolds), max(max_pressure, pressure)
            if reynolds > 2000 or pressure > 400000:
                raise ValueError("Flow left laminar pipe pressure domain")
            pump += pressure * q / 0.6
        port_rates = {f"in:{key}": float(inlet_rates[i]) for i, key in enumerate(keys)}
        port_rates.update({f"out:{key}": float(out[i]) for i, key in enumerate(keys)})
        contribution = ProcessContribution(
            amount_rates_mol_s=dict(zip(keys, dn, strict=True)),
            port_rates_mol_s=port_rates,
            # This existing liquid model transports sensible carrier enthalpy;
            # reaction formation enthalpy is the separate legacy ΔH source.
            port_enthalpies_J_mol=dict.fromkeys(port_rates, 0.0),
            legacy_reaction_energy_W=reaction_heat,
            energy_port_rates_W={
                "carrier_in": inlet_heat_W,
                "carrier_out": hout,
                "electrical": electrical,
                "chemical_work": chemical,
            },
        )
        capacity_rate = float(np.dot(inlet_rates[carriers] - out[carriers], cp))
        view = CoupledStateView(
            dict(zip(keys, amounts[:species_count], strict=True)), temperature, volume
        )
        rates = assemble_coupled_rates(
            view,
            processes=(CoupledProcess("mixed-tank-reaction-cell", lambda _: contribution),),
            thermodynamics=_FlowSensibleThermodynamics(keys, capacity, capacity_rate),
            heat_flux=lambda _: HeatFlux(jacket, loss),
            ports=ports,
            energy_ports=energy_ports,
        )
        derivative = np.zeros_like(y)
        derivative[:count] = inlet_rates - out
        derivative[:species_count] = [rates.amount_rates_mol_s[key] for key in keys]
        derivative[count : 2 * count] = out
        derivative[2 * count] = capacity * rates.temperature_rate_K_s + capacity_rate * (
            temperature - 298.15
        )
        derivative[2 * count + 1 :] = (
            jacket,
            reaction_heat,
            loss,
            hout,
            current,
            current * 0.9,
            electrical,
            chemical,
            cell_heat,
            ohmic_heat,
            pump,
        )
        return derivative

    solution = solve_ode(
        rhs,
        y0,
        time_span_s=(0.0, duration_s),
        policy=ODESolverPolicy(
            "mixed-tank-conservative-enthalpy-v1", method="DOP853", rtol=1e-10, atol=1e-13
        ),
    )
    solution.raise_for_failure("mixed tank physical advance")
    if np.min(solution.y[: 2 * count, -1]) < -1e-10:
        raise ValueError("Coupled flow integration failed")
    final = solution.y[:, -1]
    tank, outlet = np.maximum(final[:count], 0), np.maximum(final[count : 2 * count], 0)
    ledger = dict(zip(ENERGY_NAMES, map(float, final[2 * count + 1 :]), strict=True))
    energy_residual = (
        final[2 * count]
        - initial_heat
        - inlet_heat_W * duration_s
        + ledger["outlet_enthalpy_J"]
        - ledger["jacket_J"]
        + ledger["reaction_J"]
        + ledger["loss_J"]
        - ledger["cell_heat_J"]
    )
    cell_residual = ledger["electrical_work_J"] - ledger["chemical_work_J"] - ledger["cell_heat_J"]
    final_capacity = float(np.dot(tank[carriers], cp))
    out_capacity = float(np.dot(outlet[carriers], cp))
    passive_residual = float(
        np.max(
            np.abs(
                inventory[species_count:]
                + inlet_rates[species_count:] * duration_s
                - tank[species_count:]
                - outlet[species_count:]
            ),
            initial=0,
        )
    )
    if (
        passive_residual > 1e-10
        or abs(energy_residual) > max(1e-6, abs(initial_heat) * 1e-7)
        or abs(cell_residual) > 1e-8
    ):
        raise ValueError("Coupled flow conservation failed")
    return FlowResult(
        tank,
        outlet,
        float(298.15 + final[2 * count] / final_capacity),
        float(298.15 + ledger["outlet_enthalpy_J"] / out_capacity)
        if out_capacity > 0
        else float(temperature_K),
        ledger,
        {
            "energy_residual_J": float(energy_residual),
            "cell_energy_residual_J": cell_residual,
            "passive_residual": passive_residual,
            "charge_residual_C": ledger["charge_C"]
            - ledger["faradaic_charge_C"]
            - ledger["charge_C"] * 0.1,
            "maximum_reynolds": max_reynolds,
            "maximum_line_pressure_drop_Pa": max_pressure,
            "solver_evaluations": solution.diagnostic.nfev,
            "solver": solution.diagnostic.to_dict(),
            "strategy": "mixed-tank-conservative-enthalpy",
            "inlet_enthalpy_J": inlet_heat_W * duration_s,
            "initial_sensible_J": initial_heat,
            "final_sensible_J": float(final[2 * count]),
        },
        float(solution.t[-1]),
        tuple(
            {
                "exchange_id": port.port_id,
                "source": port.source,
                "destination": port.destination,
                "unit": "mol",
                "amount": float(
                    inlet_rates[i] * solution.t[-1] if port.direction == "in" else outlet[i]
                ),
            }
            for port in ports
            for i in (keys.index(port.inventory_key),)
        )
        + tuple(
            {
                "exchange_id": f"{side}:passive:{i - species_count}",
                "source": "feed" if side == "in" else "tank",
                "destination": "tank" if side == "in" else "collector",
                "unit": "L" if i < species_count + 4 else "dimensionless",
                "amount": float(inlet_rates[i] * solution.t[-1] if side == "in" else outlet[i]),
            }
            for side in ("in", "out")
            for i in range(species_count, count)
        )
        + tuple(
            {
                "exchange_id": key,
                "source": source,
                "destination": destination,
                "unit": "J",
                "energy_J": float(energy),
            }
            for key, source, destination, energy in (
                ("carrier_in", "feed", "tank", inlet_heat_W * solution.t[-1]),
                ("carrier_out", "tank", "collector", ledger["outlet_enthalpy_J"]),
                ("electrical", "power_supply", "tank", ledger["electrical_work_J"]),
                ("chemical_work", "tank", "chemical_store", ledger["chemical_work_J"]),
                ("pump_work", "power_supply", "line_bath", ledger["pump_work_J"]),
                ("jacket", "thermal_boundary", "tank", ledger["jacket_J"]),
                ("ambient_loss", "tank", "ambient", ledger["loss_J"]),
            )
        ),
    )
