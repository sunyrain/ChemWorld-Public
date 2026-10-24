"""Two finite circuits exchanging heat on the original shared n/H integrator."""

from __future__ import annotations

import warnings
from math import exp, expm1, isclose, log1p
from typing import Any

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.finite_heat_recovery_specs import FiniteHeatRecoverySpec, number
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)
from chemworld.physchem.transport import (
    counterflow_effectiveness,
    pipe_pressure_drop,
    pump_operating_point,
)
from chemworld.physchem.transport_devices import TransportInventory


class HeatRecoverySolveError(ValueError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def validate_amounts(spec: FiniteHeatRecoverySpec, amounts: dict) -> None:
    if set(amounts) != set(spec.properties.component_ids):
        raise ValueError("heat-recovery inventory must use exactly the original species")
    for value in amounts.values():
        number(value, "liquid amount")


def heat_recovery_inventory(
    spec: FiniteHeatRecoverySpec, amounts: tuple[dict, ...], temperatures_K: tuple[float, ...]
) -> TransportInventory:
    if len(amounts) != 4 or len(temperatures_K) != 4:
        raise ValueError("heat recovery requires four finite tank inventories/temperatures")
    rows = tuple(dict(n) for n in amounts)
    lo, hi = spec.properties.temperature_bounds_K
    for row, t in zip(rows, temperatures_K, strict=True):
        validate_amounts(spec, row)
        if not lo <= t <= hi:
            raise ValueError("initial tank temperature is outside declared properties")
    return TransportInventory(
        rows,
        tuple(
            spec.enthalpy(i, n, t)
            for i, (n, t) in enumerate(zip(rows, temperatures_K, strict=True))
        ),
    )


def _counterflow(spec: FiniteHeatRecoverySpec, sides: list[dict]) -> dict:
    ua = spec.conductance_W_K
    inlet = [s["pump_outlet_temperature_K"] for s in sides]
    zero = {
        "heat_left_to_right_W": 0.0,
        "outlet_temperatures_K": inlet,
        "conductance_residual_W_K": 0.0,
        "root": {"status": "zero_heat"},
    }
    if ua == 0 or any(s["molar_rate_mol_s"] == 0 for s in sides) or inlet[0] == inlet[1]:
        return zero
    hot, cold = (0, 1) if inlet[0] > inlet[1] else (1, 0)
    p = spec.properties
    correlations = [package.by_property("liquid_heat_capacity") for package in p.packages]
    constant_cp = all(
        len(rows) == 1
        and rows[0].equation_id == "cp_polynomial"
        and set(rows[0].coefficients) <= {"a", "b", "c", "d", "e"}
        and all(value == 0 for key, value in rows[0].coefficients.items() if key != "a")
        for rows in correlations
    )
    if constant_cp:
        capacities = [
            p.heat_capacity(s["mole_fractions"], t) * s["molar_rate_mol_s"]
            for s, t in zip(sides, inlet, strict=True)
        ]
        low, high = min(capacities), max(capacities)
        if low <= 0:
            raise ValueError("counterflow requires positive declared heat capacities")
        epsilon = counterflow_effectiveness(ntu=ua / low, capacity_ratio=low / high)
        q = epsilon * low * (inlet[hot] - inlet[cold])
        outlet = list(inlet)
        outlet[hot], outlet[cold] = (
            inlet[hot] - q / capacities[hot],
            inlet[cold] + q / capacities[cold],
        )
        return {
            "heat_left_to_right_W": q if hot == 0 else -q,
            "outlet_temperatures_K": outlet,
            "hot_circuit": ("left", "right")[hot],
            "conductance_residual_W_K": 0.0,
            "root": {
                "algorithm": "exact_declared_constant_cp_counterflow",
                "effectiveness": epsilon,
                "ntu": ua / low,
                "capacity_ratio": low / high,
                "floating_point_pinch_limit": epsilon == 1.0,
            },
        }

    def enthalpy(index: int, t: float) -> float:
        return p.enthalpy(sides[index]["mole_fractions"], t) * sides[index]["molar_rate_mol_s"]

    def capacity(index: int, t: float) -> float:
        c = p.heat_capacity(sides[index]["mole_fractions"], t) * sides[index]["molar_rate_mol_s"]
        if c <= 0:
            raise ResidualDomainError("counterflow requires positive local heat capacity")
        return c

    def temperature(index: int, h: float) -> float:
        return caloric_temperature(
            lambda t: enthalpy(index, t),
            h,
            p.temperature_bounds_K,
            energy_basis="enthalpy",
            algorithm="residual",
            heat_capacity_at_temperature=lambda t: capacity(index, t),
        )

    hh, hc = enthalpy(hot, inlet[hot]), enthalpy(cold, inlet[cold])
    limit = min(hh - enthalpy(hot, inlet[cold]), enthalpy(cold, inlet[hot]) - hc)
    if limit <= 0:
        raise ValueError("counterflow has no positive admissible enthalpy interval")
    cached: dict[float, tuple[float, float]] = {}

    def integral(fraction: float) -> tuple[float, float]:
        if fraction in cached:
            return cached[fraction]
        q = fraction * limit
        local: dict[float, tuple[float, float]] = {}

        def point(z: float) -> tuple[float, float]:
            if z not in local:
                th, tc = temperature(hot, hh - q * z), temperature(cold, hc + q * (1 - z))
                gap = th - tc
                if gap <= 0:
                    raise ResidualDomainError("counterflow local pinch or crossed temperature")
                local[z] = (gap, z / capacity(hot, th) + (1 - z) / capacity(cold, tc))
            return local[z]

        try:
            # Evaluate endpoints too; quadrature nodes alone cannot certify endpoint approach.
            point(0.0)
            point(1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                calculated, error = quad(
                    lambda z: q / point(z)[0],
                    0,
                    1,
                    epsabs=max(ua, 1e-4) * 1e-10,
                    epsrel=1e-10,
                    limit=100,
                )
                derivative, derivative_error = quad(
                    lambda z: limit * (1 / point(z)[0] + q * point(z)[1] / point(z)[0] ** 2),
                    0,
                    1,
                    epsabs=max(ua, 1e-4) * 1e-9,
                    epsrel=1e-9,
                    limit=100,
                )
        except (ValueError, IntegrationWarning) as exc:
            raise ResidualDomainError(str(exc)) from exc
        if error > 1e-8 * max(ua, 1e-4) or derivative_error > 1e-7 * max(derivative, 1e-4):
            raise ResidualDomainError("counterflow quadrature did not meet its declared accuracy")
        cached[fraction] = (calculated, derivative)
        return cached[fraction]

    # Logarithmic distance to the passive duty limit avoids repeatedly trying
    # the near-singular pinch when a Newton step exceeds the finite duty range.
    # This changes solver coordinates only; every trial still uses the full H
    # inverse and local gap, with no temperature or physical duty clipping.
    result = solve_root(
        lambda x: [integral(-expm1(-float(x[0])))[0] - ua],
        [-log1p(-min(0.5, ua * (inlet[hot] - inlet[cold]) / limit))],
        jacobian=lambda x: [[integral(-expm1(-float(x[0])))[1] * exp(-float(x[0]))]],
        variables=(ResidualVariable("log_distance_to_pinch", "1", 1, "algebraic", 0, 25),),
        residual_scales=[max(ua, 1e-4)],
        policy=RootPolicy(tolerance=1e-9, maximum_iterations=spec.maximum_root_iterations),
    )
    if not result.success:
        raise HeatRecoverySolveError(
            "counterflow root failed: " + result.failure_reason, result.diagnostics
        )
    fraction = -expm1(-float(result.value[0]))
    q = fraction * limit
    outlet = list(inlet)
    outlet[hot], outlet[cold] = temperature(hot, hh - q), temperature(cold, hc + q)
    return {
        "heat_left_to_right_W": q if hot == 0 else -q,
        "outlet_temperatures_K": outlet,
        "hot_circuit": ("left", "right")[hot],
        "conductance_residual_W_K": integral(fraction)[0] - ua,
        "root": result.diagnostics,
    }


def heat_recovery_profile(
    spec: FiniteHeatRecoverySpec,
    state: TransportInventory,
    requested_flows_m3_s: tuple[float, float],
) -> dict:
    if (
        len(state.amounts_mol) != 4
        or len(state.enthalpies_J) != 4
        or len(requested_flows_m3_s) != 2
    ):
        raise ValueError("heat-recovery profile requires four tanks and two independent flows")
    for n in state.amounts_mol:
        validate_amounts(spec, n)
    sides = []
    p = spec.properties
    for side, index in enumerate((0, 2)):
        requested = number(requested_flows_m3_s[side], "requested flow")
        amounts = state.amounts_mol[index]
        total = sum(amounts.values())
        if total <= 0:
            raise ValueError("finite feed requires positive initial liquid inventory")
        t = spec.at_ph(index, amounts, state.enthalpies_J[index])
        fluid, h, volume = p.evaluate(amounts, t)
        pump = pump_operating_point(
            spec.pumps[side],
            requested,
            lambda q, s=side, f=fluid: (
                pipe_pressure_drop(spec.pipes[s], f, volumetric_flow_m3_s=q).pressure_drop_total_Pa
            ),
        )
        flow = pump["actual_flow_m3_s"]
        rate = total * flow / volume
        fractions = {k: n / total for k, n in amounts.items()}
        outlet_t = t
        if rate:

            def stream_h(v: float, x: dict = fractions) -> float:
                return float(p.enthalpy(x, v))

            def stream_cp(v: float, x: dict = fractions) -> float:
                return float(p.heat_capacity(x, v))

            outlet_t = caloric_temperature(
                stream_h,
                h + pump["hydraulic_power_W"] / rate,
                p.temperature_bounds_K,
                energy_basis="enthalpy",
                algorithm="residual",
                heat_capacity_at_temperature=stream_cp,
            )
        sides.append(
            {
                **pump,
                "source_temperature_K": t,
                "molar_rate_mol_s": rate,
                "mole_fractions": fractions,
                "source_molar_enthalpy_J_mol": h,
                "pump_outlet_temperature_K": outlet_t,
                "component_rates_mol_s": {k: n * flow / volume for k, n in amounts.items()},
            }
        )
    return {
        "circuits": dict(zip(("left", "right"), sides, strict=True)),
        **_counterflow(spec, sides),
    }


def advance_finite_heat_recovery(
    spec: FiniteHeatRecoverySpec,
    state: TransportInventory,
    requested_flows_m3_s: tuple[float, float],
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[Any, dict]:
    profile = heat_recovery_profile(spec, state, requested_flows_m3_s)
    matrix: list[list[float]] = []
    hdot: list[float] = []
    shaft, loss = 0.0, 0.0
    heat = profile["heat_left_to_right_W"]
    for side, name in enumerate(("left", "right")):
        s = profile["circuits"][name]
        rate = s["component_rates_mol_s"]
        matrix.extend(
            (
                [-rate[k] for k in spec.properties.component_ids],
                [rate[k] for k in spec.properties.component_ids],
            )
        )
        inlet = s["molar_rate_mol_s"] * s["source_molar_enthalpy_J_mol"]
        hdot.extend((-inlet, inlet + s["hydraulic_power_W"] + (-heat if side == 0 else heat)))
        shaft += s["shaft_power_W"]
        loss += s["external_loss_W"]

    def bounds(index: int, n: dict) -> tuple[float, ...]:
        return tuple(spec.enthalpy(index, n, t) for t in spec.properties.temperature_bounds_K)

    def volume(index: int, s: Any, temperatures: Any) -> float:
        n = s.amounts_mol[index]
        return (
            spec.properties.evaluate(n, temperatures[index])[2] * 1000 if sum(n.values()) else 0.0
        )

    stops: list[tuple[str, Any]] = []
    initial_t = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    for side, name in enumerate(("left", "right")):
        source, receiver = 2 * side, 2 * side + 1
        if volume(source, state, initial_t) < spec.pickup_volumes_L[side] - 1e-10 or (
            volume(receiver, state, initial_t) > spec.receiver_capacities_L[side] + 1e-10
        ):
            raise ValueError("heat recovery starts outside finite source/receiver capacity")
        if profile["circuits"][name]["actual_flow_m3_s"]:
            stops.extend(
                (
                    (
                        name + "_pickup_exhausted",
                        lambda s, t, i=source, j=side: volume(i, s, t) - spec.pickup_volumes_L[j],
                    ),
                    (
                        name + "_receiver_full",
                        lambda s, t, i=receiver, j=side: (
                            spec.receiver_capacities_L[j] - volume(i, s, t)
                        ),
                    ),
                )
            )
    for index in range(4):
        stops.extend(
            (
                (
                    f"tank:{index}:minimum_temperature",
                    lambda s, t, i=index: s.enthalpies_J[i] - bounds(i, s.amounts_mol[i])[0],
                ),
                (
                    f"tank:{index}:maximum_temperature",
                    lambda s, t, i=index: bounds(i, s.amounts_mol[i])[1] - s.enthalpies_J[i],
                ),
            )
        )
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=lambda s, t: PhaseInventoryRates(
            np.asarray(matrix), np.asarray(hdot), shaft, loss, ()
        ),
        closure_indices=(0, 1, 2, 3),
        enthalpy_bounds=bounds,
        stop_functions=stops,
        initial_boundary="stop",
        progress=progress,
        solver_id="finite_two_stream_counterflow_nH_v1",
    )
    dt = result.actual_duration_s
    diagnostics = {
        **profile,
        "actual_duration_s": dt,
        "exchanged_heat_left_to_right_J": heat * dt,
        "shaft_work_J": shaft * dt,
        "motor_loss_J": loss * dt,
        "energy_residual_J": result.enthalpy_residual_J,
    }
    diagnostics["circuit_receipts"] = {
        name: {
            "transferred_mol": {
                k: v * dt for k, v in profile["circuits"][name]["component_rates_mol_s"].items()
            },
            "source_volume_m3": profile["circuits"][name]["actual_flow_m3_s"] * dt,
            "shaft_work_J": profile["circuits"][name]["shaft_power_W"] * dt,
            "motor_loss_J": profile["circuits"][name]["external_loss_W"] * dt,
            "hydraulic_work_J": profile["circuits"][name]["hydraulic_power_W"] * dt,
        }
        for name in ("left", "right")
    }
    for side, name in enumerate(("left", "right")):
        start = sum(state.enthalpies_J[2 * side : 2 * side + 2])
        end = sum(result.final_state.enthalpies_J[2 * side : 2 * side + 2])
        expected = diagnostics["circuit_receipts"][name]["hydraulic_work_J"] + (
            -heat * dt if side == 0 else heat * dt
        )
        if not isclose(end - start, expected, rel_tol=3e-7, abs_tol=1e-7):
            raise RuntimeError("heat-recovery circuit energy did not close")
    return result, diagnostics
