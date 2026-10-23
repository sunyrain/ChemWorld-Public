"""Fixed 22E5 source/process qualification. Reference imports are test-only."""

import argparse
import json
import math
import time
import traceback
from itertools import product
from pathlib import Path

import chemicals
from chemicals import dippr
from chemicals import interface as ref_interface
from chemicals import lennard_jones as ref_lj
from chemicals import thermal_conductivity as ref_k
from chemicals import viscosity as ref_mu
from transport_process_fixtures import synthetic_card

from chemworld.data.conductivity import ConductivityDatabase
from chemworld.data.interface import InterfaceDatabase
from chemworld.data.lennard_jones import LennardJonesDatabase
from chemworld.data.viscosity import ViscosityDatabase
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.interfaces import laplace_pressure
from chemworld.physchem.lennard_jones import lennard_jones_diffusivity_report
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.residual_solver import ResidualSolvePolicy
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_residual
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport import FluidState, PipeSpec, PumpSpec, pump_operating_point
from chemworld.physchem.transport_devices import (
    FiniteTransportSpec,
    LiquidTransportProperties,
    TransportPathSpec,
)
from chemworld.physchem.transport_properties import ideal_gas_transport_state
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import transport_world_spec

R = 8.31446261815324
KB, NA = 1.380649e-23, 6.02214076e23
COUNTS = (322, 75, 100, 345, 337, 274, 271, 14, 135, 432, 345, 340, 275, 271, 522, 115, 64, 64, 272)
DBS = (LennardJonesDatabase, ViscosityDatabase, ConductivityDatabase, InterfaceDatabase)
LUCAS = {"critical_temperature_K": 572.2, "critical_pressure_Pa": 3.47e6, "acentric_factor": 0.236}


def compare(actual, expected, *, rtol=5e-9, atol=1e-12):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)
    return {
        "actual": float(actual),
        "reference": float(expected),
        "absolute_error": float(abs(actual - expected)),
    }


def record_payload(record):
    return {
        "record_id": record.record_id,
        "source_id": record.source_id,
        "fields": dict(zip(record.field_names, record.raw_fields, strict=True)),
    }


def reference_value(source, record, temperature, molecular_weight):
    """Use original raw fields and reference formulas, not native curve coefficients."""
    raw = dict(zip(record.field_names, record.raw_fields, strict=True))

    def p(key):
        return float(raw[key])

    if source.endswith("_LJ"):
        omega = ref_lj.collision_integral_Kim_Monroe(temperature / p("Stockmayer"), l=2, s=2)
        return (
            5
            / 16
            * math.sqrt(molecular_weight / 1000 / NA * KB * temperature / math.pi)
            / ((p("molecular_diameter") * 1e-10) ** 2 * omega)
        )
    if source in {"PERRY_GAS_VISCOSITY", "PERRY_GAS_CONDUCTIVITY"}:
        return dippr.EQ102(temperature, *(p("C" + str(i)) for i in range(1, 5)))
    if source == "PERRY_LIQUID_VISCOSITY":
        return dippr.EQ101(temperature, *(p("C" + str(i)) for i in range(1, 6)))
    if source == "PERRY_LIQUID_CONDUCTIVITY":
        return dippr.EQ100(temperature, *(p("C" + str(i)) for i in range(1, 6)))
    if source.startswith("VDI_") and source != "VDI_SURFACE_TENSION":
        fn = ref_mu.PPDS9 if source == "VDI_LIQUID_VISCOSITY" else dippr.EQ100
        return fn(temperature, *(p(key) for key in "ABCDE"))
    if source == "VN2E_VISCOSITY":
        return ref_mu.Viswanath_Natarajan_2_exponential(temperature, p("C"), p("D"))
    if source == "VN2_VISCOSITY":
        return ref_mu.Viswanath_Natarajan_2(temperature, p("A") - math.log(100), p("B"))
    if source in {"DUTT_PRASAD_VISCOSITY", "VN3_VISCOSITY"}:
        return ref_mu.Viswanath_Natarajan_3(temperature, p("A") - 3, p("B"), p("C"))
    if source == "JASPER_LANGE":
        return ref_interface.Jasper(temperature, p("a"), p("b"))
    if source == "MULERO_CACHADINA":
        return ref_interface.REFPROP_sigma(
            temperature,
            **{key: p(key) for key in ("Tc", "sigma0", "n0", "sigma1", "n1", "sigma2", "n2")},
        )
    if source.startswith("SOMAYAJULU"):
        return ref_interface.Somayajulu(temperature, *(p(key) for key in ("Tc", "A", "B", "C")))
    if source == "VDI_SURFACE_TENSION":
        return dippr.EQ106(temperature, p("Tc"), *(p(key) for key in "ABCDE"))
    raise ValueError(source)


def resistance(mu, length, diameter):
    return 128 * mu * length / (math.pi * diameter**4)


def heat_reference(rho, cp_mass, k, flow, temperature, bath, diameter, *, area=0.001):
    capacity = flow * rho * cp_mass
    u = 1 / (diameter / (3.66 * k) + 1 / 500 + 0.001 / 15)
    return -math.expm1(-u * area / capacity) * capacity * (bath - temperature)


def source_input(db, source, *, identity_override=None):
    gas = source.endswith("_LJ") or "_GAS_" in source
    identity, formula = (
        ("7727-37-9", "N2")
        if source.endswith("_LJ")
        else {
            "VN2E_VISCOSITY": ("60-29-7", "C4H10O"),
            "VN2_VISCOSITY": ("71-36-3", "C4H10O"),
        }.get(source, ("64-17-5", "C2H6O"))
    )
    if identity_override is not None:
        identity, formula = identity_override
    component = ComponentSpec(
        identity, formula, cas_number=identity, default_phase="gas" if gas else "liquid"
    )
    bounds = (390, 460) if gas else (280, 330)
    if source.endswith("_LJ"):
        card = db.viscosity_input(
            identity,
            component,
            energy_method=source,
            diameter_method=source,
            collision_method="Kim_Monroe",
            temperature_range_K=bounds,
        )
    else:
        card = db.property_input(identity, component, method=source, temperature_bounds_K=bounds)
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict(), allow_nan=False))
    assert restored.to_dict() == card.to_dict()
    return restored, db.record(identity, method=source)


def gas_unit(db, source, temperature, pressure, *, identity_override=None):
    source_card, record = source_input(db, source, identity_override=identity_override)
    component = source_card.package.component
    companion = synthetic_card(
        component,
        cp=60,
        mu=1.5e-5,
        k=0.03,
        pressure=pressure,
        omit=(source_card.package.correlations[0].property_id,),
    )
    card = PropertyInputCard.combine((source_card, companion), reference_note="E5 selected sources")
    fluid, enthalpy, volume = ideal_gas_transport_state(
        (card.package,),
        {component.identifier: 0.01},
        temperature,
        pressure,
        {component.identifier: card.reference},
    )
    mw = component.molecular_weight_g_mol / 1000
    value = reference_value(source, record, temperature, mw * 1000)
    mu, conductivity = (1.5e-5, value) if "CONDUCTIVITY" in source else (value, 0.03)
    rho = pressure * mw / (R * temperature)
    path = TransportPathSpec(pipe=PipeSpec(0.5, 0.002), utility_temperature_K=temperature + 10)
    metrics = {
        "source_record": record_payload(record),
        "comparisons": {
            "density": compare(fluid.density_kg_m3, rho),
            "viscosity": compare(fluid.viscosity_Pa_s, mu),
            "conductivity": compare(fluid.thermal_conductivity_W_m_K, conductivity),
            "cp_mass": compare(fluid.heat_capacity_J_kg_K, 60 / mw),
            "volume": compare(volume, 0.01 * R * temperature / pressure),
            "enthalpy": compare(enthalpy, 60 * (temperature - 298.15)),
            "pressure_drop": compare(
                path.pressure_drop(fluid, 1e-6), resistance(mu, 0.5, 0.002) * 1e-6
            ),
            "heat_W": compare(
                path.heat_rate(fluid, 1e-6, temperature),
                heat_reference(
                    rho, 60 / mw, conductivity, 1e-6, temperature, temperature + 10, 0.002
                ),
            ),
        },
    }
    if source.endswith("_LJ"):
        oxygen = ComponentSpec("7782-44-7", "O2", cas_number="7782-44-7", default_phase="gas")
        a = db.parameters(
            component.identifier, component, energy_method=source, diameter_method=source
        )
        b = db.parameters(oxygen.identifier, oxygen, energy_method=source, diameter_method=source)
        diffusion = lennard_jones_diffusivity_report(
            component,
            a,
            oxygen,
            b,
            temperature_K=temperature,
            pressure_Pa=pressure,
            collision_method="Kim_Monroe",
            combining_rule="Lorentz_Berthelot",
        ).evaluation.value
        mass_a, mass_b = mw / NA, oxygen.molecular_weight_g_mol / 1000 / NA
        reduced_mass = mass_a * mass_b / (mass_a + mass_b)
        omega = ref_lj.collision_integral_Kim_Monroe(
            temperature / math.sqrt(a.epsilon_k_K * b.epsilon_k_K), l=1, s=1
        )
        dref = (
            3
            / 16
            * KB
            * temperature
            / pressure
            * math.sqrt(2 * KB * temperature / (math.pi * reduced_mass))
            / ((a.sigma_m + b.sigma_m) / 2) ** 2
            / omega
        )
        operator = ConservativeDiffusion(DiffusionMesh.planar(0.01, 2), diffusion)
        system = operator.residual_system(
            ("left", "right"),
            system_id="E5-LJ-diffusion",
            basis="amount",
            concentration_scale_mol_m3=2,
        )
        result = solve_residual(
            system,
            (0.01, 0.002),
            time_span_s=(0, 0.2),
            evaluation_times_s=(0, 0.2),
            policy=ResidualSolvePolicy(
                ode=ODESolverPolicy("E5-diffusion", method="BDF", rtol=1e-10, atol=1e-13)
            ),
        )
        assert result.success, result.failure_reason
        expected = 0.006 + 0.004 * math.exp(-2 * dref * 0.2 / 0.005**2)
        metrics["comparisons"].update(
            diffusivity=compare(diffusion, dref),
            left_amount=compare(result.points[-1].state[0], expected, rtol=3e-7, atol=1e-10),
            conserved_amount=compare(sum(result.points[-1].state), 0.012, atol=1e-10),
        )
    return metrics


def execute(runtime, operation, **arguments):
    result = runtime.execute({"operation": operation, **arguments})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def liquid_unit(db, source, temperature):
    source_card, record = source_input(db, source)
    component = source_card.package.component
    companion = synthetic_card(component, omit=(source_card.package.correlations[0].property_id,))
    card = PropertyInputCard.combine((source_card, companion), reference_note="E5 selected liquid")
    properties = LiquidTransportProperties(
        (card.package,), (280, 330), 101325, {component.identifier: card.reference}
    )
    spec = FiniteTransportSpec(
        properties, path=TransportPathSpec(utility_temperature_K=temperature + 5)
    )
    runtime = ChemWorldRuntime(
        composition=transport_world_spec(
            spec, feed_temperature_K=temperature, receiver_temperature_K=temperature
        )
    )
    restored = None
    try:
        runtime.reset(seed=24)
        _, _, initial = unpack_transport(runtime._state)
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
        execute(runtime, "run_flow", duration_s=1)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        )
        result = execute(runtime, "run_flow", duration_s=1)
        execute(restored, "run_flow", duration_s=1)
        assert runtime._state.to_dict() == restored._state.to_dict()
        _, _, final = unpack_transport(runtime._state)
        point = result.info["outcome_diagnostics"]["transport"]
        mw = component.molecular_weight_g_mol / 1000
        value = reference_value(source, record, temperature, mw * 1000)
        mu = 0.001 if "CONDUCTIVITY" in source else value
        load = resistance(mu, 2, 0.001)
        flow = min(20 / 6e7, 20000 / load, math.sqrt(0.005 * 0.7 / load))
        hydraulic = load * flow**2
        rate = flow * 800 / mw
        stream_temperature = temperature + hydraulic / (rate * 100)
        k_stream = (
            reference_value(source, record, stream_temperature, mw * 1000)
            if "CONDUCTIVITY" in source
            else 0.2
        )
        heat = heat_reference(
            800, 100 / mw, k_stream, flow, stream_temperature, temperature + 5, 0.001
        )
        target_t = temperature + 2 * (hydraulic + heat) / (
            (initial.amounts_mol[1][component.identifier] + rate * 2) * 100 + 5
        )
        comparisons = {
            "flow": compare(point["actual_flow_m3_s"], flow),
            "heat_W": compare(point["utility_heat_W"], heat),
            "transfer_mol": compare(
                final.amounts_mol[1][component.identifier]
                - initial.amounts_mol[1][component.identifier],
                rate * 2,
                rtol=2e-7,
                atol=1e-8,
            ),
            "receiver_temperature": compare(
                runtime._state.temperature_K, target_t, rtol=2e-7, atol=1e-8
            ),
            "enthalpy_change": compare(
                sum(final.enthalpies_J) - sum(initial.enthalpies_J),
                2 * (hydraulic + heat),
                rtol=2e-7,
                atol=1e-8,
            ),
            "shaft_work": compare(runtime._state.ledger.flow_pump_work_J, 2 * hydraulic / 0.7),
            "time": compare(runtime._state.ledger.time_s, 2),
        }
        for index in (0, 1):
            comparisons[f"amount_{index}"] = compare(
                final.amounts_mol[index][component.identifier],
                initial.amounts_mol[index][component.identifier] + (1 if index else -1) * rate * 2,
                rtol=2e-7,
                atol=1e-8,
            )
        return {
            "source_record": record_payload(record),
            "comparisons": comparisons,
            "snapshot_exact": True,
            "fluid_energy_scope": "synthetic companion calorics",
        }
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def interface_unit(db, source, temperature, *, curvature=20000):
    card, record = source_input(db, source)
    sigma = card.evaluate_state(
        temperature_K=temperature,
        pressure_Pa=101325,
        phase="liquid",
        quantities=("surface_tension",),
    ).properties["surface_tension"]
    pressure = laplace_pressure(
        sigma, curvature_1_per_m=curvature, curvature_2_per_m=curvature
    ).value
    fluid, path = FluidState(800, 0.001), TransportPathSpec()
    actual = pump_operating_point(
        PumpSpec(1500, 1, 0.7), 1e-6, lambda q: pressure + path.pressure_drop(fluid, q)
    )
    expected_sigma = reference_value(
        source, record, temperature, card.package.component.molecular_weight_g_mol
    )
    load, dp = resistance(0.001, 2, 0.001), 2 * curvature * expected_sigma
    flow = min(1e-6, max(0, (1500 - dp) / load))
    return {
        "source_record": record_payload(record),
        "comparisons": {
            "surface_tension": compare(sigma.value, expected_sigma),
            "laplace_pressure": compare(pressure, dp),
            "limited_flow": compare(actual["actual_flow_m3_s"], flow),
            "shaft_power": compare(actual["shaft_power_W"], flow * (dp + load * flow) / 0.7),
        },
        "scope": "fixed-curvature steady boundary; no interface energy transient",
    }


def source_units(root):
    counts, sources = [], []
    for cls in DBS:
        db = cls(root)
        for declaration in db.catalog:
            source = declaration.source_id
            counts.append(len(db.records(source)))
            sources.append(
                {
                    "source": source,
                    "records": counts[-1],
                    "provenance": declaration.provenance.to_dict(),
                }
            )
    assert tuple(counts) == COUNTS and sum(counts) == 4573
    units = []
    for cls in DBS:
        db = cls(root)
        for declaration in db.catalog:
            source = declaration.source_id
            if cls is InterfaceDatabase:
                settings = ((t, 101325) for t in (300, 320))
                def fn(s, t, p, db=db):
                    return interface_unit(db, s, t)
            elif source.endswith("_LJ") or "_GAS_" in source:
                settings = product((400, 440), (80000, 150000))
                def fn(s, t, p, db=db):
                    return gas_unit(db, s, t, p)
            else:
                settings = ((t, 101325) for t in (300, 320))
                def fn(s, t, p, db=db):
                    return liquid_unit(db, s, t)
            for t, p in settings:
                units.append(
                    (
                        f"{source}/{t}/{p}",
                        {"source": source, "T": t, "P": p},
                        lambda fn=fn, s=source, t=t, p=p: fn(s, t, p),
                    )
                )
    return units, sources


def mixture_unit(phase, temperature, fraction, pressure, *, area=0.001):
    components = tuple(
        ComponentSpec(name, formula, default_phase=phase)
        for name, formula in (("ethanol", "C2H6O"), ("butanol", "C4H10O"))
    )
    xs = [fraction, 1 - fraction]
    mw = [c.molecular_weight_g_mol for c in components]
    mbar = sum(x * m for x, m in zip(xs, mw, strict=True)) / 1000
    cps = [60, 85] if phase == "gas" else [100, 130]
    mus = [1.5e-5, 2.5e-5] if phase == "gas" else [0.001, 0.003]
    ks = [0.03, 0.045] if phase == "gas" else [0.2, 0.12]
    results, outputs = [], []
    for perturb in (False, True):
        cards = tuple(
            synthetic_card(
                c,
                cp=cp,
                rho=rho,
                mu=mu * (1.3 if perturb and i == 0 else 1),
                k=k * (1.2 if perturb and i == 0 else 1),
                pressure=pressure,
                variable=True,
                vapor_pressure=phase == "liquid" and i == 1,
            )
            for i, (c, cp, rho, mu, k) in enumerate(
                zip(components, cps, (800, 900), mus, ks, strict=True)
            )
        )
        packages = tuple(card.package for card in cards)
        references = {card.package.component.identifier: card.reference for card in cards}
        amounts = dict(zip((c.identifier for c in components), xs, strict=True))
        mu_pure = [
            mu * math.exp(300 / temperature - 1) * (1.3 if perturb and i == 0 else 1)
            for i, mu in enumerate(mus)
        ]
        k_pure = [
            k * (1.2 if perturb and i == 0 else 1) + 1e-4 * (temperature - 300)
            for i, k in enumerate(ks)
        ]
        if phase == "gas":
            fluid, _, volume = ideal_gas_transport_state(
                packages, amounts, temperature, pressure, references
            )
            mu = ref_mu.Wilke(xs, mu_pure, mw)
            conductivity = ref_k.Wassiljewa_Herning_Zipperer(xs, k_pure, mw)
            expected_volume = R * temperature / pressure
        else:
            props = LiquidTransportProperties(
                packages, (280, 330), pressure, references, {"butanol": LUCAS}
            )
            props = LiquidTransportProperties.from_dict(json.loads(json.dumps(props.to_dict())))
            fluid, _, volume = props.evaluate(amounts, temperature)
            mu_pure[1] = ref_mu.Lucas(
                temperature, pressure, 572.2, 3.47e6, 0.236, 20000, mu_pure[1]
            )
            mu = math.exp(sum(x * math.log(v) for x, v in zip(xs, mu_pure, strict=True)))
            ws = [x * m / (1000 * mbar) for x, m in zip(xs, mw, strict=True)]
            conductivity = ref_k.DIPPR9H(ws, k_pure)
            expected_volume = sum(
                x * m / 1000 / rho for x, m, rho in zip(xs, mw, (800, 900), strict=True)
            )
        cp_mass = sum(x * cp for x, cp in zip(xs, cps, strict=True)) / mbar
        rho = mbar / expected_volume
        path = TransportPathSpec(
            pipe=PipeSpec(0.5, 0.002),
            utility_temperature_K=temperature + 10,
            exchanger_area_m2=area,
        )
        actual_dp, heat = path.pressure_drop(fluid, 1e-6), path.heat_rate(fluid, 1e-6, temperature)
        results.append(
            {
                "perturbed": perturb,
                "comparisons": {
                    "mu": compare(fluid.viscosity_Pa_s, mu),
                    "k": compare(fluid.thermal_conductivity_W_m_K, conductivity),
                    "density": compare(fluid.density_kg_m3, rho),
                    "volume": compare(volume, expected_volume),
                    "pressure_drop": compare(actual_dp, resistance(mu, 0.5, 0.002) * 1e-6),
                    "heat": compare(
                        heat,
                        heat_reference(
                            rho,
                            cp_mass,
                            conductivity,
                            1e-6,
                            temperature,
                            temperature + 10,
                            0.002,
                            area=area,
                        ),
                    ),
                },
            }
        )
        outputs.append((actual_dp, heat))
    assert outputs[0][0] != outputs[1][0] and outputs[0][1] != outputs[1][1], outputs
    return {"responses": results, "material_perturbation_changes_process": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=(
            "sources",
            "mixtures",
            "lj-source-completion",
            "mixtures-response",
            "interface-blocked",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    if args.block == "sources":
        units, sources = source_units(Path(chemicals.__file__).parent)
    elif args.block == "lj-source-completion":
        sources, units = [], []
        db = LennardJonesDatabase(Path(chemicals.__file__).parent)
        for t, p in product((400, 440), (80000, 150000)):
            inputs = {
                "source": "MAGALHAES_LJ",
                "identity": "74-82-8",
                "formula": "CH4",
                "paired_identity": "7782-44-7",
                "T": t,
                "P": p,
            }
            units.append(
                (
                    f"MAGALHAES_LJ/methane/{t}/{p}",
                    inputs,
                    lambda t=t, p=p: gas_unit(
                        db, "MAGALHAES_LJ", t, p, identity_override=("74-82-8", "CH4")
                    ),
                )
            )
    elif args.block == "interface-blocked":
        sources, units = [], []
        db = InterfaceDatabase(Path(chemicals.__file__).parent)
        for source, t in product((s.source_id for s in db.catalog), (300, 320)):
            inputs = {
                "source": source,
                "T": t,
                "curvature_1_per_m": 40000,
                "curvature_2_per_m": 40000,
            }
            units.append(
                (
                    f"{source}/{t}/blocked",
                    inputs,
                    lambda s=source, t=t: interface_unit(db, s, t, curvature=40000),
                )
            )
    else:
        sources, units = [], []
        for phase in ("liquid", "gas"):
            temperatures = (300, 320) if phase == "liquid" else (400, 440)
            for t, x, p in product(temperatures, (0.2, 0.8), (101325, 200000)):
                area = 1e-5 if args.block == "mixtures-response" else 0.001
                inputs = {"phase": phase, "T": t, "x_ethanol": x, "P": p, "area_m2": area}
                units.append(
                    (
                        f"{phase}/{t}/{x}/{p}",
                        inputs,
                        lambda phase=phase, t=t, x=x, p=p, area=area: mixture_unit(
                            phase, t, x, p, area=area
                        ),
                    )
                )
    assert (
        len(units)
        == {
            "sources": 50,
            "mixtures": 16,
            "lj-source-completion": 4,
            "mixtures-response": 16,
            "interface-blocked": 10,
        }[args.block]
    )
    rows, start = [], time.perf_counter()
    for i, (name, inputs, run) in enumerate(units, 1):
        row = {"unit": name, "inputs": inputs}
        try:
            row.update(status="passed", metrics=run())
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        elapsed = time.perf_counter() - start
        print(
            f"{i}/{len(units)} {name}: {row['status']} {i / elapsed:.2f}/s "
            f"ETA {elapsed * (len(units) - i) / i:.1f}s",
            flush=True,
        )
    summary = {
        "task": "NATIVE-CONV-22E5",
        "mode": "development",
        "block": args.block,
        "reference_version": chemicals.__version__,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "sources": sources,
        "results": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )
    for row in rows:
        if row["status"] == "failed":
            print(row["unit"], row["failure"].splitlines()[-1])
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "sources"}}))
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
