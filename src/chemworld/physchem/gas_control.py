"""Finite-flow ideal-gas pressure regulator with a well-mixed purge plenum.

The liquid is incompressible and does not dissolve/react with these gases. The
fixed-volume plenum has its own ideal thermostat following vessel temperature;
its heat and external pump work are accounted separately from liquid heating.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log

import numpy as np

from chemworld.foundation.gas import GAS_CV, GAS_FEEDS, GAS_R, GasBoundary
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledProcess,
    CoupledStateView,
    HeatFlux,
    MaterialPort,
    ProcessContribution,
    SpeciesInternalEnergy,
    assemble_coupled_rates,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode

GAS_CONTROL_MODEL = "ideal-gas-plenum-finite-flow-v1"
MAX_GAS_FLOW_MOL_S = 2e-5
PURGE_FLOW_MOL_S = 2e-6
REGULATOR_TIME_S = 20.0
PUMP_EFFICIENCY = 0.5
GAS_SUPPLY_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class GasAdvanceResult:
    boundary: GasBoundary
    actual_duration_s: float
    exchanges: tuple[dict, ...]
    solver_diagnostic: dict


def advance_gas_with_report(
    gas: GasBoundary,
    *,
    duration_s: float,
    final_temperature_K: float,
    pressure_setpoint_Pa: float,
    atmosphere: int,
    purge_flow_mol_s: float = PURGE_FLOW_MOL_S,
    maximum_flow_mol_s: float = MAX_GAS_FLOW_MOL_S,
) -> GasAdvanceResult:
    if not 0 < duration_s <= 14400 or not 250 <= final_temperature_K <= 470:
        raise ValueError("Gas advancement outside time/temperature domain")
    if not 50000 <= pressure_setpoint_Pa <= 400000 or atmosphere not in range(3):
        raise ValueError("Invalid gas pressure target or atmosphere")
    if not 0 <= purge_flow_mol_s <= maximum_flow_mol_s <= MAX_GAS_FLOW_MOL_S:
        raise ValueError("Gas flow limit outside provider domain")
    supply = np.asarray(GAS_FEEDS[atmosphere])
    cp = np.asarray(GAS_CV) + GAS_R
    initial = np.zeros(13)
    initial[:3] = gas.amounts_mol
    keys = ("nitrogen", "oxygen", "argon")
    thermo = ConstantCvThermodynamics(
        {key: SpeciesInternalEnergy(cv * 350.0, cv) for key, cv in zip(keys, GAS_CV, strict=True)},
        reference_temperature_K=350.0,
    )
    ports = tuple(
        MaterialPort(
            f"{side}:{key}",
            key,
            "gas_supply" if side == "in" else "plenum",
            "plenum" if side == "in" else "gas_exhaust",
            side,
        )
        for side in ("in", "out")
        for key in keys
    )

    def rhs(time: float, y: np.ndarray) -> np.ndarray:
        temperature = (
            gas.temperature_K + (final_temperature_K - gas.temperature_K) * time / duration_s
        )
        amounts = np.maximum(y[:3], 0)
        total = float(amounts.sum())
        pressure = total * GAS_R * temperature / (gas.volume_L * 0.001)
        desired = pressure_setpoint_Pa * gas.volume_L * 0.001 / (GAS_R * temperature)
        actuator_limit = maximum_flow_mol_s - purge_flow_mol_s
        regulator = float(
            np.clip((desired - total) / REGULATOR_TIME_S, -actuator_limit, actuator_limit)
        )
        incoming = supply * (purge_flow_mol_s + max(regulator, 0))
        outgoing = amounts / max(total, 1e-30) * (purge_flow_mol_s + max(-regulator, 0))
        input_h = float(np.dot(incoming, cp) * GAS_SUPPLY_TEMPERATURE_K)
        output_h = float(np.dot(outgoing, cp) * temperature)
        work = (
            GAS_R
            * temperature
            / PUMP_EFFICIENCY
            * (
                float(incoming.sum()) * log(max(pressure / 101325, 1))
                + float(outgoing.sum()) * log(max(101325 / max(pressure, 1), 1))
            )
        )
        port_rates = {f"in:{key}": float(incoming[i]) for i, key in enumerate(keys)}
        port_rates.update({f"out:{key}": float(outgoing[i]) for i, key in enumerate(keys)})
        enthalpies = {
            f"in:{key}": float(cp[i] * GAS_SUPPLY_TEMPERATURE_K) for i, key in enumerate(keys)
        }
        enthalpies.update({f"out:{key}": float(cp[i] * temperature) for i, key in enumerate(keys)})
        contribution = ProcessContribution(
            port_rates_mol_s=port_rates, port_enthalpies_J_mol=enthalpies
        )
        rates = assemble_coupled_rates(
            CoupledStateView(dict(zip(keys, amounts, strict=True)), temperature, gas.volume_L),
            processes=(CoupledProcess("finite-plenum-regulator", lambda _: contribution),),
            thermodynamics=thermo,
            heat_flux=lambda _: HeatFlux(),
            ports=ports,
            prescribed_temperature_rate_K_s=(final_temperature_K - gas.temperature_K) / duration_s,
        )
        return np.concatenate(
            (
                [rates.amount_rates_mol_s[key] for key in keys],
                incoming,
                outgoing,
                [input_h, output_h, work, rates.heater_W],
            )
        )

    result = solve_ode(
        rhs,
        initial,
        time_span_s=(0.0, duration_s),
        policy=ODESolverPolicy(
            "finite-plenum-prescribed-thermostat-v1", method="DOP853", rtol=1e-10, atol=1e-14
        ),
    )
    if not result.success or np.min(result.y[:3, -1]) < -1e-12:
        raise ValueError("Gas regulator integration failed")
    final = result.y[:, -1]
    amounts = tuple(max(float(v), 0.0) for v in final[:3])
    energy = sum(n * cv * final_temperature_K for n, cv in zip(amounts, GAS_CV, strict=True))
    residual = energy - gas.energy_j - float(final[9]) + float(final[10]) - float(final[12])
    if abs(residual) > 1e-8 + 1e-8 * max(abs(energy), abs(gas.energy_j)):
        raise ValueError("Gas thermostat energy failed to close")
    boundary = replace(
        gas,
        temperature_K=final_temperature_K,
        amounts_mol=amounts,
        added_mol=tuple(a + float(b) for a, b in zip(gas.added_mol, final[3:6], strict=True)),
        removed_mol=tuple(a + float(b) for a, b in zip(gas.removed_mol, final[6:9], strict=True)),
        enthalpy_in_J=gas.enthalpy_in_J + float(final[9]),
        enthalpy_out_J=gas.enthalpy_out_J + float(final[10]),
        bath_heat_J=gas.bath_heat_J + float(final[12]),
        pump_work_J=gas.pump_work_J + float(final[11]),
    )
    return GasAdvanceResult(
        boundary,
        float(result.t[-1]),
        tuple(
            {
                "exchange_id": port.port_id,
                "source": port.source,
                "destination": port.destination,
                "unit": "mol",
                "amount": float(amount),
            }
            for port, amount in zip(ports, final[3:9], strict=True)
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
                ("enthalpy_in", "gas_supply", "plenum", final[9]),
                ("enthalpy_out", "plenum", "gas_exhaust", final[10]),
                ("thermostat", "plenum_bath", "plenum", final[12]),
                ("pump_work", "power_supply", "external_pump", final[11]),
            )
        ),
        result.diagnostic.to_dict(),
    )


def advance_gas(
    gas: GasBoundary,
    *,
    duration_s: float,
    final_temperature_K: float,
    pressure_setpoint_Pa: float,
    atmosphere: int,
    purge_flow_mol_s: float = PURGE_FLOW_MOL_S,
    maximum_flow_mol_s: float = MAX_GAS_FLOW_MOL_S,
) -> GasBoundary:
    """Compatibility readout of the shared-contribution plenum advance."""
    return advance_gas_with_report(
        gas,
        duration_s=duration_s,
        final_temperature_K=final_temperature_K,
        pressure_setpoint_Pa=pressure_setpoint_Pa,
        atmosphere=atmosphere,
        purge_flow_mol_s=purge_flow_mol_s,
        maximum_flow_mol_s=maximum_flow_mol_s,
    ).boundary
