"""Finite-flow ideal-gas pressure regulator with a well-mixed purge plenum.

The liquid is incompressible and does not dissolve/react with these gases. The
fixed-volume plenum has its own ideal thermostat following vessel temperature;
its heat and external pump work are accounted separately from liquid heating.
"""

from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
from scipy.integrate import solve_ivp

from chemworld.foundation.gas import GAS_CV, GAS_FEEDS, GAS_R, GasBoundary

GAS_CONTROL_MODEL = "ideal-gas-plenum-finite-flow-v1"
MAX_GAS_FLOW_MOL_S = 2e-5
PURGE_FLOW_MOL_S = 2e-6
REGULATOR_TIME_S = 20.0
PUMP_EFFICIENCY = 0.5
GAS_SUPPLY_TEMPERATURE_K = 298.15


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
    if not 0 < duration_s <= 14400 or not 250 <= final_temperature_K <= 470:
        raise ValueError("Gas advancement outside time/temperature domain")
    if not 50000 <= pressure_setpoint_Pa <= 400000 or atmosphere not in range(3):
        raise ValueError("Invalid gas pressure target or atmosphere")
    if not 0 <= purge_flow_mol_s <= maximum_flow_mol_s <= MAX_GAS_FLOW_MOL_S:
        raise ValueError("Gas flow limit outside provider domain")
    supply = np.asarray(GAS_FEEDS[atmosphere])
    cp = np.asarray(GAS_CV) + GAS_R
    initial = np.zeros(12)
    initial[:3] = gas.amounts_mol

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
        return np.concatenate((incoming - outgoing, incoming, outgoing, [input_h, output_h, work]))

    result = solve_ivp(rhs, (0, duration_s), initial, method="DOP853", rtol=1e-10, atol=1e-14)
    if not result.success or np.min(result.y[:3, -1]) < -1e-12:
        raise ValueError("Gas regulator integration failed")
    final = result.y[:, -1]
    amounts = tuple(max(float(v), 0.0) for v in final[:3])
    energy = sum(n * cv * final_temperature_K for n, cv in zip(amounts, GAS_CV, strict=True))
    return replace(
        gas,
        temperature_K=final_temperature_K,
        amounts_mol=amounts,
        added_mol=tuple(a + float(b) for a, b in zip(gas.added_mol, final[3:6], strict=True)),
        removed_mol=tuple(a + float(b) for a, b in zip(gas.removed_mol, final[6:9], strict=True)),
        enthalpy_in_J=gas.enthalpy_in_J + float(final[9]),
        enthalpy_out_J=gas.enthalpy_out_J + float(final[10]),
        bath_heat_J=gas.bath_heat_J + energy - gas.energy_j - float(final[9]) + float(final[10]),
        pump_work_J=gas.pump_work_J + float(final[11]),
    )
