"""Synthetic declared properties and independent split/utility equations."""

from dataclasses import replace
from math import pi, sqrt

from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties
from chemworld.physchem.water_treatment_economics import WaterTreatmentEconomicsSpec
from chemworld.physchem.zero_order_treatment import treatment_inventory
from chemworld.physchem.zero_order_treatment_specs import (
    ZeroOrderTreatmentSpec,
    ZeroOrderTreatmentStage,
)

DATA = {
    "water": ("H2O", 1000.0, 75.0),
    "ethanol": ("C2H6O", 800.0, 110.0),
    "acetone": ("C3H6O", 790.0, 125.0),
}
FRACTIONS = ((1.0, 0.0, 0.0), (0.999, 0.001, 0.0), (0.997, 0.002, 0.001))


def properties():
    packages = []
    for name, (formula, density, cp) in DATA.items():
        rows = (
            ("liquid_heat_capacity", "cp_polynomial", {"a": cp}, "J/(mol*K)"),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": density, "T_ref": 298.15, "alpha": 0.0},
                "kg/m^3",
            ),
            ("liquid_viscosity", "andrade_viscosity", {"A": 0.001, "B": 0.0}, "Pa*s"),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": 0.5, "T_ref": 298.15, "slope": 0.0},
                "W/(m*K)",
            ),
        )
        packages.append(
            ComponentPropertyPackage(
                ComponentSpec(name, formula),
                tuple(
                    PropertyCorrelation(
                        correlation_id=f"synthetic49-{name}-{quantity}",
                        property_id=quantity,
                        equation_id=equation,
                        coefficients=coefficients,
                        input_units={"temperature": "K"},
                        output_unit=unit,
                        validity_ranges={"temperature": (280.0, 340.0)},
                        metadata={"phase": "liquid"},
                        source_note=(
                            "Synthetic constant-property qualification data, not wastewater data"
                        ),
                    )
                    for quantity, equation, coefficients, unit in rows
                ),
            )
        )
    return LiquidTransportProperties(tuple(packages))


def spec_case(stages=3, temperature=298.15):
    return ZeroOrderTreatmentSpec(
        properties(),
        "water",
        tuple(
            ZeroOrderTreatmentStage(
                stage_id=f"stage{j + 1}",
                reject_vessel_id=f"concentrate{j + 1}",
                water_recovery=0.9 - 0.05 * j,
                removal_to_concentrate={"ethanol": 0.6 + 0.1 * j, "acetone": 0.4 + 0.1 * j},
                specific_electricity_kWh_m3=0.1 * (j + 1),
                source_note="Synthetic empirical split; not adsorption/clarification mechanism",
                maximum_inlet_concentration_mol_m3={"ethanol": 1000.0, "acetone": 1000.0},
            )
            for j in range(stages)
        ),
        temperature,
        pump=PumpSpec(1e5, 10.0, 0.7),
        pipe=PipeSpec(0.1, 0.002),
    )


def molar_volumes(spec):
    return {
        p.component.identifier: p.component.molecular_weight_g_mol
        / 1000
        / DATA[p.component.identifier][1]
        for p in spec.properties.packages
    }


def inventory_case(spec, composition=2, *, preload_L=0.001):
    v = molar_volumes(spec)
    fractions = dict(zip(DATA, FRACTIONS[composition], strict=True))
    total = 0.02 / 1000 / sum(v[k] * fractions[k] for k in v)
    feed = {k: fractions[k] * total for k in v}
    collector = {k: preload_L / 1000 / v[k] if k == "water" else 0.0 for k in v}
    return treatment_inventory(
        spec, feed, collectors=tuple(dict(collector) for _ in range(len(spec.stages) + 1))
    )


def analytic(spec, initial, requested=10 / 6e7, duration=10):
    v = molar_volumes(spec)
    h = {k: DATA[k][2] * (spec.temperature_K - 298.15) for k in v}

    def volume(n):
        return sum(n[k] * v[k] for k in v)

    resistance = 128 * 0.001 * spec.pipe.length_m / (pi * spec.pipe.diameter_m**4)
    flow = min(
        requested,
        spec.pump.maximum_pressure_rise_Pa / resistance,
        sqrt(spec.pump.maximum_shaft_power_W * spec.pump.efficiency / resistance),
    )
    feed = {k: initial.amounts_mol[0][k] * flow / volume(initial.amounts_mol[0]) for k in v}
    current, rejects, stages, extra = feed, [], {}, 0.0
    for j, stage in enumerate(spec.stages):
        stages[stage.stage_id] = volume(current)
        extra += 0.1 * (j + 1) * volume(current) * 3.6e6
        product = {
            k: current[k]
            * (
                0.9 - 0.05 * j
                if k == "water"
                else 1 - (0.6 + 0.1 * j if k == "ethanol" else 0.4 + 0.1 * j)
            )
            for k in v
        }
        rejects.append({k: current[k] - product[k] for k in v})
        current = product
    rows = [{k: -n for k, n in feed.items()}, current, *rejects]
    capacities = [spec.product_capacity_L, *(s.reject_capacity_L for s in spec.stages)]
    times = [(duration, None)]
    if flow:
        times.append(
            (
                (volume(initial.amounts_mol[0]) - spec.pickup_volume_L / 1000) / flow,
                "feed_pickup_exhausted",
            )
        )
        for i, cap in enumerate(capacities, 1):
            q = volume(rows[i])
            if q:
                times.append(
                    (
                        (cap / 1000 - volume(initial.amounts_mol[i])) / q,
                        "product_full" if i == 1 else f"reject:stage{i - 1}:full",
                    )
                )
    elif requested:
        times.append((0.0, "no_forward_pump_drive"))
    dt, stop = min(times, key=lambda x: x[0])
    pressure, hydraulic = resistance * flow, resistance * flow**2
    shaft = hydraulic / spec.pump.efficiency
    final = [{k: n[k] + dt * rows[i][k] for k in v} for i, n in enumerate(initial.amounts_mol)]
    return {
        "amounts": final,
        "enthalpies": [sum(n[k] * h[k] for k in v) for n in final],
        "dt": dt,
        "stop": stop,
        "flow": flow,
        "pressure": pressure,
        "receipt": {
            "schema": "native-treatment-throughput-v1",
            "elapsed_s": dt,
            "feed_volume_m3": flow * dt,
            "product_volume_m3": volume(current) * dt,
            "reject_volumes_m3": {
                s.stage_id: volume(n) * dt for s, n in zip(spec.stages, rejects, strict=True)
            },
            "stage_inlet_volumes_m3": {k: value * dt for k, value in stages.items()},
            "pump_work_J": shaft * dt,
            "hydraulic_work_J": hydraulic * dt,
            "additional_electrical_work_J": extra * dt,
            "motor_loss_J": (shaft - hydraulic) * dt,
            "bath_heat_into_liquid_J": -(hydraulic + extra) * dt,
        },
    }


def fixed_cases():
    cases = [
        (f"chain-{n}-T{t}-composition{c}", spec_case(n, t), c, 10 / 6e7, 10)
        for n in (1, 2, 3)
        for t in (288.15, 318.15)
        for c in range(3)
    ]
    original = spec_case()
    initial = inventory_case(original)
    out = analytic(original, initial)["receipt"]
    for kind in ("feed", "product", "first-reject", "last-reject", "pressure", "zero-flow"):
        spec, flow = original, 10 / 6e7
        if kind == "feed":
            spec = replace(spec, pickup_volume_L=0.005)
        elif kind == "product":
            spec = replace(
                spec, product_capacity_L=0.001 + out["product_volume_m3"] / 10 * 3 * 1000
            )
        elif kind in {"first-reject", "last-reject"}:
            index = 0 if kind == "first-reject" else 2
            stages = list(spec.stages)
            stages[index] = replace(
                stages[index],
                reject_capacity_L=0.001
                + out["reject_volumes_m3"][stages[index].stage_id] / 10 * 3 * 1000,
            )
            spec = replace(spec, stages=tuple(stages))
        elif kind == "pressure":
            spec = replace(spec, pump=PumpSpec(10, 10, 0.7))
        else:
            flow = 0.0
        cases.append((kind, spec, 2, flow, 120))
    return cases


def economic_policy(spec, discount=0.08):
    return WaterTreatmentEconomicsSpec(
        "USD",
        2026,
        {s.stage_id: 100.0 * (j + 1) for j, s in enumerate(spec.stages)},
        20.0,
        discount,
        0.03,
        40.0,
        0.12,
        2.0,
        4000.0,
        "Synthetic 2026 price scenario; not vendor quotation",
    )
