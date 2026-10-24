"""Authored fixtures and independent enthalpy/ODE references for fixed 31B cases."""

from dataclasses import replace
from math import exp, pi, sqrt

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from chemworld.physchem.finite_heat_recovery import heat_recovery_inventory
from chemworld.physchem.finite_heat_recovery_specs import FiniteHeatRecoverySpec
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties

IDS = ("water", "ethanol")
RHO = {"water": 1000.0, "ethanol": 800.0}
CP = {
    False: {"water": (75.0, 0.0), "ethanol": (110.0, 0.0)},
    True: {"water": (20.0, 0.2), "ethanol": (35.0, 0.25)},
}
COMPOSITIONS = (((1.0, 0.0), (0.0, 1.0)), ((0.8, 0.2), (0.2, 0.8)))
TREF = 298.15


def spec_case(*, variable=False, ua=0.2):
    packages = []
    for name, formula in (("water", "H2O"), ("ethanol", "C2H6O")):
        a, b = CP[variable][name]
        declarations = (
            ("liquid_heat_capacity", "cp_polynomial", {"a": a, "b": b}, "J/(mol*K)"),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": RHO[name], "T_ref": TREF, "alpha": 0.0},
                "kg/m^3",
            ),
            ("liquid_viscosity", "andrade_viscosity", {"A": 0.001, "B": 0.0}, "Pa*s"),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": 0.5, "T_ref": TREF, "slope": 0.0},
                "W/(m*K)",
            ),
        )
        packages.append(
            ComponentPropertyPackage(
                ComponentSpec(name, formula),
                tuple(
                    PropertyCorrelation(
                        correlation_id=f"synthetic31B-{name}-{quantity}",
                        property_id=quantity,
                        equation_id=equation,
                        coefficients=coefficients,
                        input_units={"temperature": "K"},
                        output_unit=unit,
                        validity_ranges={"temperature": (280.0, 340.0)},
                        metadata={"phase": "liquid"},
                        source_note="Synthetic heat-recovery fixture, not fitted equipment data",
                    )
                    for quantity, equation, coefficients, unit in declarations
                ),
            )
        )
    return FiniteHeatRecoverySpec(
        LiquidTransportProperties(tuple(packages)),
        ua,
        "Authored synthetic UA; no exchanger geometry claim",
        pumps=(PumpSpec(1e5, 10, 0.7), PumpSpec(1e5, 10, 0.7)),
        pipes=(PipeSpec(0.1, 0.002), PipeSpec(0.1, 0.002)),
    )


def molar_volumes(spec):
    return {
        p.component.identifier: p.component.molecular_weight_g_mol
        / 1000
        / RHO[p.component.identifier]
        for p in spec.properties.packages
    }


def setup(spec, composition=1, reverse=False):
    v = molar_volumes(spec)
    fractions = (COMPOSITIONS[composition][0], (0.5, 0.5), COMPOSITIONS[composition][1], (0.5, 0.5))
    amounts = tuple(
        {
            k: x * volume / 1000 / sum(y * v[s] for s, y in zip(IDS, xs, strict=True))
            for k, x in zip(IDS, xs, strict=True)
        }
        for xs, volume in zip(fractions, (0.03, 0.002, 0.04, 0.002), strict=True)
    )
    temperatures = (290.0, 300.0, 330.0, 300.0) if reverse else (330.0, 300.0, 290.0, 300.0)
    return amounts, temperatures


def initial_case(spec, composition=1, reverse=False):
    amounts, temperatures = setup(spec, composition, reverse)
    return heat_recovery_inventory(spec, amounts, temperatures)


def enthalpy(n, temperature, variable, body=0):
    delta = temperature - TREF
    return (
        sum(
            n[k]
            * (CP[variable][k][0] * delta + 0.5 * CP[variable][k][1] * (temperature**2 - TREF**2))
            for k in IDS
        )
        + body * delta
    )


def temperature(n, h, variable, body=0):
    c = body + sum(n[k] * (CP[variable][k][0] + CP[variable][k][1] * TREF) for k in IDS)
    b = sum(n[k] * CP[variable][k][1] for k in IDS)
    return TREF + (h / c if b == 0 else 2 * h / (c + sqrt(c * c + 2 * b * h)))


def reference(spec, amounts, temperatures, flows, duration, *, variable=False):
    v = molar_volumes(spec)
    rates, qactual, shaft, hydraulic, stream_h, tin = [], [], [], [], [], []
    for side, source in enumerate((0, 2)):
        pipe, pump = spec.pipes[side], spec.pumps[side]
        resistance = 128 * 0.001 * pipe.length_m / (pi * pipe.diameter_m**4)
        q = min(
            flows[side],
            pump.maximum_pressure_rise_Pa / resistance,
            sqrt(pump.maximum_shaft_power_W * pump.efficiency / resistance),
        )
        rate = {
            k: amounts[source][k] * q / sum(amounts[source][s] * v[s] for s in IDS) for k in IDS
        }
        rates.append(rate)
        qactual.append(q)
        hydraulic.append(resistance * q * q)
        shaft.append(hydraulic[-1] / pump.efficiency)
        h = enthalpy(rate, temperatures[source], variable)
        stream_h.append(h)
        tin.append(temperature(rate, h + hydraulic[-1], variable) if q else temperatures[source])
    duty, tout, shooting = 0.0, list(tin), {}
    if all(qactual) and tin[0] != tin[1] and spec.conductance_W_K:
        hot, cold = (0, 1) if tin[0] > tin[1] else (1, 0)

        def cp(i, t):
            return sum(rates[i][k] * (CP[variable][k][0] + CP[variable][k][1] * t) for k in IDS)

        if not variable:
            ch, cc = cp(hot, tin[hot]), cp(cold, tin[cold])
            cmin, cmax = min(ch, cc), max(ch, cc)
            ratio, ntu = cmin / cmax, spec.conductance_W_K / cmin
            epsilon = (
                ntu / (1 + ntu)
                if abs(ratio - 1) < 1e-12
                else ((1 - exp(-ntu * (1 - ratio))) / (1 - ratio * exp(-ntu * (1 - ratio))))
            )
            q = epsilon * cmin * (tin[hot] - tin[cold])
            tout[hot], tout[cold] = tin[hot] - q / ch, tin[cold] + q / cc
        else:
            trials = []

            def lower_domain(s, t):
                return t[1] - 280.0

            lower_domain.terminal = True
            lower_domain.direction = -1

            def shoot(cold_out):
                solution = solve_ivp(
                    lambda s, t: [-(t[0] - t[1]) / cp(hot, t[0]), -(t[0] - t[1]) / cp(cold, t[1])],
                    (0, spec.conductance_W_K),
                    [tin[hot], cold_out],
                    method="DOP853",
                    rtol=1e-11,
                    atol=1e-11,
                    events=lower_domain,
                )
                trials.append(
                    {
                        "cold_outlet_guess_K": cold_out,
                        "status": solution.status,
                        "final_conductance_W_K": float(solution.t[-1]),
                        "cold_endpoint_K": float(solution.y[1, -1]),
                        "nfev": solution.nfev,
                    }
                )
                if not solution.success:
                    raise ValueError("independent two-temperature shooting failed")
                return solution

            cold_out = brentq(
                lambda t: shoot(t).y[1, -1] - tin[cold], tin[cold], tin[hot], xtol=1e-10, rtol=1e-12
            )
            solution = shoot(cold_out)
            if solution.status != 0 or solution.t[-1] != spec.conductance_W_K:
                raise ValueError(
                    "final independent reference must complete within its property domain"
                )
            tout[hot], tout[cold] = float(solution.y[0, -1]), cold_out
            q = enthalpy(rates[hot], tin[hot], variable) - enthalpy(rates[hot], tout[hot], variable)
            shooting = {
                "nfev": solution.nfev,
                "cold_boundary_residual_K": float(solution.y[1, -1]) - tin[cold],
                "trials": trials,
            }
        duty = q if hot == 0 else -q
    limits = [(duration, None)]
    for side in range(2):
        if qactual[side]:
            source, receiver = 2 * side, 2 * side + 1
            vs = sum(amounts[source][k] * v[k] for k in IDS)
            vr = sum(amounts[receiver][k] * v[k] for k in IDS)
            name = ("left", "right")[side]
            limits.extend(
                (
                    (
                        (vs - spec.pickup_volumes_L[side] / 1000) / qactual[side],
                        name + "_pickup_exhausted",
                    ),
                    (
                        (spec.receiver_capacities_L[side] / 1000 - vr) / qactual[side],
                        name + "_receiver_full",
                    ),
                )
            )
    dt, stop = min(limits, key=lambda pair: pair[0])
    initial_h = [
        enthalpy(n, t, variable, body)
        for n, t, body in zip(amounts, temperatures, spec.capacities_J_K, strict=True)
    ]
    final, hfinal = [], []
    for side in range(2):
        for offset in (0, 1):
            index, sign = 2 * side + offset, -1 if offset == 0 else 1
            final.append({k: amounts[index][k] + sign * dt * rates[side][k] for k in IDS})
            rate_h = (
                -stream_h[side]
                if offset == 0
                else (stream_h[side] + hydraulic[side] + (-duty if side == 0 else duty))
            )
            hfinal.append(initial_h[index] + dt * rate_h)
    return {
        "amounts": final,
        "enthalpies_J": hfinal,
        "temperatures_K": [
            temperature(n, h, variable, body)
            for n, h, body in zip(final, hfinal, spec.capacities_J_K, strict=True)
        ],
        "source_flows_m3_s": qactual,
        "duty_left_to_right_W": duty,
        "outlet_temperatures_K": tout,
        "actual_duration_s": dt,
        "stop": stop,
        "shaft_work_J": sum(shaft) * dt,
        "motor_loss_J": (sum(shaft) - sum(hydraulic)) * dt,
        "circuit_motor_loss_J": [(a - b) * dt for a, b in zip(shaft, hydraulic, strict=True)],
        "shooting": shooting,
    }


def fixed_cases():
    cases = []
    base = spec_case()
    v = molar_volumes(base)
    right = 10 / 6e7
    for ratio in (0.25, 0.7, 1.0):
        left = ratio * right * 110 / v["ethanol"] / (75 / v["water"])
        for ntu in (0.1, 1.0, 4.0):
            for reverse in (False, True):
                spec = spec_case(ua=ntu * left / v["water"] * 75)
                cases.append(
                    (
                        f"constant-Cr{ratio}-NTU{ntu}-reverse{reverse}",
                        spec,
                        0,
                        reverse,
                        (left, right),
                        10,
                        False,
                    )
                )
    for composition in (0, 1):
        for ua in (0.02, 0.2, 1.0):
            cases.append(
                (
                    f"variable-composition{composition}-UA{ua}",
                    spec_case(variable=True, ua=ua),
                    composition,
                    False,
                    (8 / 6e7, 12 / 6e7),
                    10,
                    True,
                )
            )
    for kind in (
        "left-pickup",
        "right-pickup",
        "left-full",
        "right-full",
        "left-zero",
        "right-zero",
        "left-pressure",
        "both-zero",
    ):
        spec, flows = spec_case(), (8 / 6e7, 12 / 6e7)
        if kind == "left-pickup":
            spec = replace(spec, pickup_volumes_L=(0.02, 1e-6))
        elif kind == "right-pickup":
            spec = replace(spec, pickup_volumes_L=(1e-6, 0.035))
        elif kind == "left-full":
            spec = replace(spec, receiver_capacities_L=(0.002 + flows[0] * 3000, 1.0))
        elif kind == "right-full":
            spec = replace(spec, receiver_capacities_L=(1.0, 0.002 + flows[1] * 3000))
        elif kind == "left-zero":
            flows = (0.0, flows[1])
        elif kind == "right-zero":
            flows = (flows[0], 0.0)
        elif kind == "left-pressure":
            spec = replace(spec, pumps=(PumpSpec(1, 10, 0.7), spec.pumps[1]))
        else:
            flows = (0.0, 0.0)
        cases.append((kind, spec, 1, False, flows, 120, False))
    return cases
