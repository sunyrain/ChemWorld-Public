"""Zero-holdup gas shaft machine on original thermochemistry, without stock or clock."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import fsum, isfinite, log
from typing import Any

from chemworld.physchem.coupled_system import CoupledCoordinate
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root


def shaft_key(connection_id: str, quantity: str) -> str:
    return json.dumps(["gas_shaft", connection_id, quantity], separators=(",", ":"))


@dataclass(frozen=True)
class GasShaftMachineSpec:
    connection_id: str
    mode: str
    isentropic_efficiency: float
    maximum_shaft_power_W: float
    maximum_pressure_ratio: float
    temperature_range_K: tuple[float, float] = (150.0, 2000.0)
    minimum_outlet_temperature_K: float = 180.0
    maximum_outlet_temperature_K: float = 1800.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        if (
            not isinstance(self.connection_id, str)
            or not self.connection_id
            or self.mode not in {"compressor", "expander"}
            or len(self.temperature_range_K) != 2
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (
                    self.isentropic_efficiency,
                    self.maximum_shaft_power_W,
                    self.maximum_pressure_ratio,
                    *self.temperature_range_K,
                    self.minimum_outlet_temperature_K,
                    self.maximum_outlet_temperature_K,
                )
            )
            or not 0 < self.isentropic_efficiency <= 1
            or self.maximum_shaft_power_W <= 0
            or self.maximum_pressure_ratio <= 1
            or not 0
            < self.temperature_range_K[0]
            < self.minimum_outlet_temperature_K
            < self.maximum_outlet_temperature_K
            < self.temperature_range_K[1]
        ):
            raise ValueError(
                "shaft machine needs explicit efficiency, power, pressure ratio and T bounds"
            )

    def coordinates(self) -> tuple[CoupledCoordinate, ...]:
        return tuple(
            CoupledCoordinate(shaft_key(self.connection_id, name), unit, 1.0)
            for name, unit in (
                ("delivered_enthalpy_J", "J"),
                ("shaft_work_J", "J"),
                ("entropy_generated_J_K", "J/K"),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "finite-gas-shaft-machine-v1",
            **asdict(self),
            "temperature_range_K": list(self.temperature_range_K),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasShaftMachineSpec:
        if (
            set(value) != {"model", *cls.__dataclass_fields__}
            or value["model"] != "finite-gas-shaft-machine-v1"
        ):
            raise ValueError("invalid gas shaft-machine declaration")
        return cls(**{k: v for k, v in value.items() if k != "model"})


def gas_shaft_operating_point(
    specification: GasShaftMachineSpec,
    thermochemistry: Mapping[str, Any],
    *,
    inlet_temperature_K: float,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
    mole_fractions: Mapping[str, float],
    requested_molar_rate_mol_s: float,
    enabled: bool = True,
    check_limits: bool = True,
) -> dict[str, Any]:
    """Frozen-composition adiabatic PS/eta/PH closure at actual endpoint pressures.

    The power limit is shaft power delivered to/from the gas, not electricity.
    Inactive or wrong-direction stages have zero flow; the network's declared
    direction event stops an active stage at equal pressures. Numerical trials
    may pass envelope events, but accepted states are checked separately.
    """
    spec = specification
    if (
        set(mole_fractions) != set(thermochemistry)
        or any(not isfinite(x) or x < 0 for x in mole_fractions.values())
        or abs(fsum(mole_fractions.values()) - 1.0) > 1e-10
        or any(
            not isfinite(v) or v <= 0
            for v in (inlet_temperature_K, inlet_pressure_Pa, outlet_pressure_Pa)
        )
        or not isfinite(requested_molar_rate_mol_s)
        or requested_molar_rate_mol_s < 0
        or type(enabled) is not bool
        or type(check_limits) is not bool
    ):
        raise ValueError("shaft inlet requires finite actual composition, T/P and requested flow")
    active = {k: x for k, x in mole_fractions.items() if x > 0}
    low, high = spec.temperature_range_K
    for key in active:
        item = thermochemistry[key]
        if item.min_temperature_K > low or item.max_temperature_K < high:
            raise ValueError("shaft temperature interval exceeds original species thermochemistry")
    if not low <= inlet_temperature_K <= high:
        raise ValueError("shaft inlet temperature lies outside declared thermochemistry domain")

    def properties(t: float) -> tuple[float, float, float]:
        values = [(x, thermochemistry[key].evaluate(t)) for key, x in active.items()]
        cp = fsum(x * p.cp_J_mol_K for x, p in values)
        if cp <= R or not isfinite(cp):
            raise ValueError("shaft ideal-gas calorics require positive Cv")
        return (
            fsum(x * p.enthalpy_J_mol for x, p in values),
            fsum(x * p.entropy_J_mol_K for x, p in values),
            cp,
        )

    hin, sin, _ = properties(inlet_temperature_K)
    ratio = (
        outlet_pressure_Pa / inlet_pressure_Pa
        if spec.mode == "compressor"
        else inlet_pressure_Pa / outlet_pressure_Pa
    )
    running = enabled and requested_molar_rate_mol_s > 0 and ratio >= 1.0
    base = {
        "connection_id": spec.connection_id,
        "mode": spec.mode,
        "inlet_pressure_Pa": inlet_pressure_Pa,
        "outlet_pressure_Pa": outlet_pressure_Pa,
        "pressure_ratio": ratio,
        "requested_molar_rate_mol_s": requested_molar_rate_mol_s,
        "inlet_temperature_K": inlet_temperature_K,
        "inlet_enthalpy_J_mol": hin,
        "isentropic_efficiency": spec.isentropic_efficiency,
        "maximum_shaft_power_W": spec.maximum_shaft_power_W,
        "mole_fractions": dict(mole_fractions),
        "enabled": enabled,
    }
    if not running:
        return {
            **base,
            "status": "disabled"
            if not enabled
            else "zero_request"
            if requested_molar_rate_mol_s == 0
            else "blocked_pressure_direction",
            "molar_rate_mol_s": 0.0,
            "outlet_temperature_K": inlet_temperature_K,
            "isentropic_temperature_K": inlet_temperature_K,
            "outlet_enthalpy_J_mol": hin,
            "isentropic_enthalpy_J_mol": hin,
            "shaft_work_J_mol": 0.0,
            "shaft_power_W": 0.0,
            "entropy_generation_W_K": 0.0,
            "entropy_generation_J_mol_K": 0.0,
            "power_limited": False,
            "solves": [],
        }
    if check_limits and ratio > spec.maximum_pressure_ratio * (1 + 1e-9):
        raise ValueError("shaft pressure ratio exceeds declared envelope")
    log_pressure = log(outlet_pressure_Pa / inlet_pressure_Pa)
    solves = []

    def temperature(target: float, coordinate: str, guess: float) -> float:
        idx, scale = (1, R) if coordinate == "S" else (0, R * inlet_temperature_K)
        result = solve_root(
            lambda z: [properties(float(z[0]))[idx] - target],
            [guess],
            jacobian=lambda z: [
                [
                    properties(float(z[0]))[2] / float(z[0])
                    if coordinate == "S"
                    else properties(float(z[0]))[2]
                ]
            ],
            variables=(ResidualVariable("shaft_temperature", "K", 500.0, lower=low, upper=high),),
            residual_scales=[scale],
            policy=RootPolicy(tolerance=1e-11, maximum_iterations=80),
        )
        solves.append(
            {
                "coordinate": coordinate,
                "success": result.success,
                "failure_reason": result.failure_reason,
                **result.diagnostics,
            }
        )
        if not result.success or result.value is None:
            raise ValueError(f"shaft {coordinate} closure unresolved: {result.failure_reason}")
        return float(result.value[0])

    ts = temperature(sin + R * log_pressure, "S", inlet_temperature_K)
    hs = properties(ts)[0]
    change = (
        (hs - hin) / spec.isentropic_efficiency
        if spec.mode == "compressor"
        else (hs - hin) * spec.isentropic_efficiency
    )
    tout = temperature(hin + change, "H", ts)
    hout, sout, _ = properties(tout)
    work = hout - hin
    entropy = sout - sin - R * log_pressure
    if (
        entropy < -1e-8
        or (spec.mode == "compressor" and work < -1e-7)
        or (spec.mode == "expander" and work > 1e-7)
    ):
        raise ValueError("shaft closure violates work direction or nonnegative entropy generation")
    if (
        check_limits
        and not spec.minimum_outlet_temperature_K - 1e-7
        <= tout
        <= spec.maximum_outlet_temperature_K + 1e-7
    ):
        raise ValueError("shaft outlet temperature exceeds declared operating limits")
    flow = (
        min(requested_molar_rate_mol_s, spec.maximum_shaft_power_W / abs(work))
        if work != 0
        else requested_molar_rate_mol_s
    )
    return {
        **base,
        "status": "running",
        "molar_rate_mol_s": flow,
        "outlet_temperature_K": tout,
        "isentropic_temperature_K": ts,
        "outlet_enthalpy_J_mol": hout,
        "isentropic_enthalpy_J_mol": hs,
        "shaft_work_J_mol": work,
        "shaft_power_W": flow * work,
        "entropy_generation_J_mol_K": entropy,
        "entropy_generation_W_K": flow * entropy,
        "power_limited": flow < requested_molar_rate_mol_s,
        "solves": solves,
    }
