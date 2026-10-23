"""Fixed 104-unit D4 development block; reference libraries remain test-only."""

import argparse
import json
import math
import time
import traceback
from collections.abc import Mapping
from dataclasses import fields, replace
from itertools import product
from pathlib import Path
from xml.etree import ElementTree as ET

import chemicals
import mpmath as mp
from chemicals import critical as refcritical
from chemicals import volume as refvolume
from tests.reference.run_virial_pure_block import independent
from tests.reference.run_virial_source_block import crc_formula
from tests.reference.run_volume_mixture_block import convert, mathematical
from tests.reference.run_volume_pure_block import CASES
from tests.reference.run_volume_source_block import SUPPLEMENT, reference, source_rows
from tests.reference.transport_process_fixtures import synthetic_card
from tests.test_virial_consumers import thermal_model

from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.data.volume import VolumeDatabase
from chemworld.examples.native_crystal_research import (
    synthetic_crystal_model,
    synthetic_crystal_thermodynamics,
)
from chemworld.foundation import equipment_settings
from chemworld.foundation.balances import MaterialPacket
from chemworld.physchem import (
    VIRIAL_METHODS,
    VOLUME_METHODS,
    ComponentSpec,
    VirialGasThermodynamics,
    critical_correlation_input,
    estimate_critical_property,
    evaluate_virial_gas_state,
    virial_gas_batch_system,
)
from chemworld.physchem.coupled_system import CoupledState, HeatFlux
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.transport_devices import (
    FiniteTransportSpec,
    LiquidTransportProperties,
    TransportPathSpec,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.crystal_world import crystal_world_spec
from chemworld.world.transport_world import transport_world_spec

R = 8.31446261815324
SOURCES = (
    ("PSRK", 995, "64-17-5", "C2H6O"),
    ("CRC_CRITICAL", 861, "64-17-5", "C2H6O"),
    ("DIPPR_PINA_MARTINES", 1799, "7782-50-5", "Cl2"),
    ("FEDORS", 57445, "64-17-5", "C2H6O"),
    ("IUPAC", 810, "64-17-5", "C2H6O"),
    ("MATHEWS", 120, "7732-18-5", "H2O"),
    ("OMEGA_PSAT", 29784, "64-17-5", "C2H6O"),
    ("PASSUT_DANNER", 192, "74-82-8", "CH4"),
    ("WILSON_JASPERSON", 29726, "64-17-5", "C2H6O"),
    ("YAWS", 7549, "64-17-5", "C2H6O"),
    ("HEOS", 145, "64-17-5", "C2H6O"),
    ("WEBBOOK", 13779, "108-88-3", "C7H8"),
    ("JOBACK", 52224, "64-17-5", "C2H6O"),
)


def compare(actual, expected, *, rtol=3e-9, atol=1e-12):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected)
    error = abs(actual - expected)
    assert error <= atol + rtol * abs(expected), (actual, expected, error, rtol, atol)
    return {"actual": actual, "reference": expected, "absolute_error": error}


def rejected(fn):
    try:
        fn()
    except (ValueError, TypeError) as error:
        return str(error)
    raise AssertionError("expected explicit rejection")


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            item
            for key in a.keys() | b.keys()
            for item in differences(a.get(key), b.get(key), path + "/" + key)
        ]
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        if type(a) is not type(b):
            return [(path, type(a).__name__, type(b).__name__)]
        return [
            item
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for item in differences(x, y, path + "/" + str(i))
        ]
    return [] if a == b else [(path, a, b)]


def source():
    return InputSource(
        DataSourceProvenance(
            "D4-explicit",
            0,
            "D4 fixed synthetic/application declaration; parameter sources retained",
            "1",
            "synthetic test",
        ),
        "synthetic",
    )


def sourced(value, key):
    return InputValue(
        value,
        {"Tc": "K", "Pc": "Pa", "Vc": "m^3/mol"}.get(key, "dimensionless"),
        None,
        "Explicit synthetic supplement; not a source measurement",
    )


def bridge(component, family, method, p, t, *, output="molar_volume", pressure=101325):
    options = {}
    if family == "volume" and VOLUME_METHODS[method].uses_pressure:
        options["pressure_range_Pa"] = (pressure * 0.5, pressure * 2)
    return critical_correlation_input(
        component,
        family=family,
        method=method,
        parameters=p,
        source=source(),
        temperature_range_K=(t - 5, t + 20),
        output=output,
        **options,
    )


def card_checks(card, temperature, pressure=101325):
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict(), allow_nan=False))
    # Canonical JSON normalizes tuple/list representations without dropping fields.
    assert json.loads(json.dumps(restored.to_dict())) == json.loads(json.dumps(card.to_dict()))
    curve = card.package.correlations[0]
    lo, hi = curve.validity_ranges["temperature"]
    result = {"card_json_exact": True, "declaration": card.to_dict()}
    for t in (lo - 1, hi + 1):
        result[f"reject_T_{t}"] = rejected(
            lambda t=t: card.package.evaluate(
                curve.property_id, temperature_K=t, pressure_Pa=pressure, validity_policy="raise"
            )
        )
    return result


def execute(runtime, operation, **arguments):
    result = runtime.execute({"operation": operation, **arguments})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def companion(component, *, pressure, omit=(), rho=800):
    card = synthetic_card(component, pressure=pressure, omit=omit, rho=rho)
    return replace(
        card,
        package=replace(
            card.package,
            correlations=tuple(
                replace(c, validity_ranges={"temperature": (50, 2000)})
                for c in card.package.correlations
            ),
        ),
    )


def liquid(
    cards,
    temperature,
    vm_reference,
    *,
    pressure=101325,
    fractions=None,
    receiver=None,
    mixing=None,
    parameters=None,
    pressure_parameters=None,
):
    completed = [
        PropertyInputCard.combine(
            (
                c,
                companion(
                    c.package.component,
                    pressure=pressure,
                    omit=("liquid_density", "liquid_molar_volume"),
                ),
            ),
            reference_note="D4 synthetic calorics",
        )
        for c in cards
    ]
    props = LiquidTransportProperties(
        tuple(c.package for c in completed),
        (temperature - 4, temperature + 15),
        pressure,
        {c.package.component.identifier: c.reference for c in completed},
        volume_mixing_method=mixing,
        volume_mixing_parameters=parameters,
        volume_pressure_parameters=pressure_parameters,
        volume_mixing_note="D4 explicit synthetic mixtures; additive liquid enthalpy",
    )
    spec = FiniteTransportSpec(
        props, path=TransportPathSpec(exchanger_area_m2=0, utility_temperature_K=temperature)
    )
    runtime = ChemWorldRuntime(
        composition=transport_world_spec(
            spec,
            feed_temperature_K=temperature,
            receiver_temperature_K=temperature,
            feed_mole_fractions=fractions,
            receiver_mole_fractions=receiver,
        )
    )
    restored = None
    try:
        runtime.reset(seed=24)
        _, _, initial = unpack_transport(runtime._state)
        initial_x = fractions or {props.component_ids[0]: 1.0}
        xs = tuple(initial_x[k] for k in props.component_ids)
        q = 1e-7
        hydraulic = 128 * 0.001 * 2 / (math.pi * 0.001**4) * q * q
        rate = q / vm_reference(temperature, xs)
        execute(runtime, "set_flow_rate", flow_rate_mL_min=6)
        execute(runtime, "run_flow", duration_s=1)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        result = execute(runtime, "run_flow", duration_s=1)
        execute(restored, "run_flow", duration_s=1)
        assert runtime._state.to_dict() == restored._state.to_dict(), differences(
            runtime._state.to_dict(), restored._state.to_dict()
        )[:20]
        _, _, final = unpack_transport(runtime._state)
        target = temperature + 2 * hydraulic / (100 * sum(final.amounts_mol[1].values()) + 5)
        metrics = {
            "receiver_T": compare(runtime._state.temperature_K, target, rtol=2e-7, atol=1e-10),
            "H_change": compare(
                sum(final.enthalpies_J) - sum(initial.enthalpies_J),
                2 * hydraulic,
                rtol=2e-7,
                atol=1e-8,
            ),
            "shaft_work": compare(runtime._state.ledger.flow_pump_work_J, 2 * hydraulic / 0.7),
            "time": compare(runtime._state.ledger.time_s, 2, rtol=0, atol=0),
            "flow": compare(result.info["outcome_diagnostics"]["transport"]["actual_flow_m3_s"], q),
        }
        for i in (0, 1):
            n = sum(final.amounts_mol[i].values())
            x = tuple(final.amounts_mol[i][k] / n for k in props.component_ids)
            t = temperature if i == 0 else target
            metrics[f"volume_{i}"] = compare(
                props.evaluate(final.amounts_mol[i], t)[2],
                n * vm_reference(t, x),
                rtol=2e-7,
                atol=1e-10,
            )
            for j, k in enumerate(props.component_ids):
                metrics[f"amount_{i}_{k}"] = compare(
                    final.amounts_mol[i][k],
                    initial.amounts_mol[i][k] + (1 if i else -1) * 2 * rate * xs[j],
                    rtol=2e-7,
                    atol=1e-10,
                )
        return {
            "comparisons": metrics,
            "snapshot_exact": True,
            "declaration": props.to_dict(),
            "final_x": x,
            "scope": "empirical liquid volume plus independent additive enthalpy",
        }
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def gas(card, temperature, b, c=lambda t: mp.mpf(0)):
    ideal = thermal_model()
    ideal = replace(
        ideal,
        heat_capacity=replace(ideal.heat_capacity, validity_ranges={"temperature": (50, 2000)}),
    )
    model = VirialGasThermodynamics(card.package, ideal)
    policy = ODESolverPolicy(solver_id="D4-development", method="DOP853", rtol=1e-10, atol=1e-12)
    system = virial_gas_batch_system(model, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy)
    identity = card.package.component.identifier
    start = CoupledState({identity: 2.0}, temperature, 20)
    v, r = mp.mpf(".01"), mp.mpf(str(R))

    def u(t):
        return 25 * (t - 300) - r * t * t * (mp.diff(b, t) / v + mp.diff(c, t) / (2 * v * v))

    result = system.advance(start, 60, evaluation_times_s=tuple(range(0, 61, 10)))
    metrics, trajectory = {}, []
    assert len(result.trajectory) == 7
    for point in result.trajectory:
        t = mp.findroot(
            lambda t, elapsed=point.time_s: 2 * (u(t) - u(mp.mpf(temperature))) - 5 * elapsed,
            (mp.mpf(temperature), mp.mpf(temperature + 10)),
        )
        p = r * t / v * (1 + b(t) / v + c(t) / v**2)
        metrics[f"T_{point.time_s}"] = compare(point.temperature_K, t, rtol=0, atol=2e-6)
        metrics[f"Q_{point.time_s}"] = compare(
            point.boundary_heat_J, 5 * point.time_s, rtol=0, atol=1e-7
        )
        metrics[f"P_{point.time_s}"] = compare(model.pressure_pa(point), p)
        trajectory.append(
            {"T": point.temperature_K, "P": model.pressure_pa(point), "t": point.time_s}
        )
    half = system.advance(start, 30).final_state
    # CoupledState owns arrays/mappings through its existing dataclass declaration.
    saved = json.loads(
        json.dumps(
            {
                f.name: dict(value) if isinstance(value, Mapping) else value
                for f in fields(half)
                for value in (getattr(half, f.name),)
            }
        )
    )
    restored_state = CoupledState(**saved)
    restored = VirialGasThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
    resumed = virial_gas_batch_system(
        restored, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy
    ).advance(restored_state, 30)
    metrics["resume_T"] = compare(
        resumed.final_state.temperature_K, result.final_state.temperature_K, rtol=0, atol=2e-6
    )
    metrics["energy_residual"] = compare(result.energy_balance_residual_J, 0, rtol=0, atol=1e-4)
    metrics["time"] = compare(result.final_state.time_s, 60, rtol=0, atol=0)
    final = result.final_state
    report = evaluate_virial_gas_state(
        card.package,
        temperature_K=final.temperature_K,
        pressure_Pa=model.pressure_pa(final),
        quantities=("internal_energy", "enthalpy", "cv"),
        ideal_caloric=ideal,
    )
    t = mp.mpf(final.temperature_K)
    for key, expected in {
        "internal_energy": u(t),
        "enthalpy": u(t) + model.pressure_pa(final) * v,
        "cv": mp.diff(u, t),
    }.items():
        metrics[key] = compare(report.properties[key].value, expected)
    return {
        "comparisons": metrics,
        "trajectory": trajectory,
        "model_json": model.to_dict(),
        "state_json": saved,
    }


def packet(card, temperature, expected, pressure=101325):
    prop = card.package.correlations[0].property_id
    vm = card.package.evaluate(prop, temperature_K=temperature, pressure_Pa=pressure).value
    identity = card.package.component.identifier
    receipt = MaterialPacket.representative(
        "source", "destination", {identity: 1.0}, amount_mol=1e-6 / vm
    )
    restored = MaterialPacket.from_dict(json.loads(json.dumps(receipt.to_dict())))
    a, b = restored.apply({identity: 1.0}, {identity: 0.0})
    restored.check_debit({identity: 1.0}, a)
    return {
        "comparisons": {
            "vm": compare(vm, expected),
            "transfer": compare(b[identity], 1e-6 / expected),
            "conserved": compare(a[identity] + b[identity], 1.0, rtol=0, atol=0),
        },
        "receipt": restored.to_dict(),
        "scope": "finite isothermal material transfer; no heating",
    }


def crystal(card, expected_vm):
    component = card.package.component
    vm = card.package.evaluate("solid_molar_volume", temperature_K=298.15).value
    mw = component.molecular_weight_g_mol / 1000
    model = synthetic_crystal_model(mw)
    model = replace(model, kinetics=replace(model.kinetics, crystal_density_kg_m3=mw / vm))
    runtime = ChemWorldRuntime(
        composition=crystal_world_spec(
            model, synthetic_crystal_thermodynamics(), {"P": component.formula, "I": "KCl"}
        )
    )
    restored = None
    try:
        runtime.reset(seed=23)
        initial = runtime._state.species_amounts["P"]
        execute(runtime, "seed_crystals", seed_mass_g=0.001)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        execute(runtime, "seed_crystals", seed_mass_g=0.001)
        execute(restored, "seed_crystals", seed_mass_g=0.001)
        assert runtime._state.to_dict() == restored._state.to_dict()
        cohorts = equipment_settings(runtime._state.equipment, "crystallizer")["population_cohorts"]
        expected_count = (0.001 + 2e-6 / mw) / (math.pi * (100e-6) ** 3 / (6 * expected_vm))
        return {
            "comparisons": {
                "Vm": compare(vm, expected_vm),
                "seed_moles": compare(runtime._state.species_amounts["P"] - initial, 2e-6 / mw),
                "particle_count": compare(sum(c[0] for c in cohorts), expected_count),
            },
            "snapshot_exact": True,
            "model": model.to_dict(),
            "scope": "isothermal seed only",
        }
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def pair_consumer(component, p, expected, family):
    t = (0.7 if family == "volume" else 1.5) * expected["Tc"]
    if family == "volume":
        zc = p.get("Zc") or estimate_critical_property(
            component, method="Zc_definition", parameters={k: p[k] for k in ("Tc", "Pc", "Vc")}
        )
        ez = expected.get("Zc", expected["Pc"] * expected["Vc"] / (R * expected["Tc"]))
        card = bridge(component, family, "Rackett", {"Tc": p["Tc"], "Pc": p["Pc"], "Zc": zc}, t)
        result = liquid(
            [card], t, lambda temp, xs: refvolume.Rackett(temp, expected["Tc"], expected["Pc"], ez)
        )
    else:
        card = bridge(component, family, "Abbott", {k: p[k] for k in ("Tc", "Pc", "omega")}, t)
        result = gas(card, t, lambda temp: independent("Abbott", temp, expected))
    return {
        **card_checks(card, t),
        **result,
        "input_parameters": {k: v.to_dict() for k, v in p.items()},
    }


def constant_unit(db, declaration, family):
    name, count, cas, formula = declaration
    assert len(db.records(name)) == count
    record = db.record(cas, method=name)
    component = ComponentSpec(cas, formula, cas_number=cas)
    p, expected = {}, {}
    for key, fallback in {"Tc": 600.0, "Pc": 4e6, "Vc": 0.0003, "omega": 0.2}.items():
        value = record.value(key) if key in record.field_names else None
        p[key] = value if value is not None and value.quality == "valid" else sourced(fallback, key)
        expected[key] = (
            float(record.raw(key)) if value is not None and value.quality == "valid" else fallback
        )
    if "Zc" in record.field_names and record.value("Zc").quality == "valid":
        p["Zc"] = record.value("Zc")
        expected["Zc"] = float(record.raw("Zc"))
    return {
        "source_count": count,
        "source": record.source.provenance.to_dict(),
        **pair_consumer(component, p, expected, family),
    }


def derived_unit(chain, variation, family):
    component = ComponentSpec("derived", "C10H22" if chain == "boiling" else "C7H8")
    if chain == "boiling":
        tb = (360, 380)[variation]
        tc = estimate_critical_property(
            component, method="Tb_Tc_relationship", parameters={"Tb": tb, "fit": "Perry8E"}
        )
        vc = estimate_critical_property(component, method="Mersmann_Kind", parameters={})
        pc = estimate_critical_property(component, method="Ihmels", parameters={"Tc": tc, "Vc": vc})
        etc = refcritical.Tb_Tc_relationship(Tb=tb, fit="Perry8E")
        evc = refcritical.Mersmann_Kind_predictor(dict(component.composition))
        epc = refcritical.Ihmels(Tc=etc, Vc=evc)
        p = {"Tc": tc, "Vc": vc, "Pc": pc, "omega": sourced(0.2, "omega")}
        expected = {"Tc": etc, "Vc": evc, "Pc": epc, "omega": 0.2}
    else:
        v = (0.000316, 0.000330)[variation]
        pc, vc = sourced(4108000.0, "Pc"), sourced(v, "Vc")
        tc = estimate_critical_property(
            component,
            method="Hekayati_Raeissi",
            parameters={"MW": component.molecular_weight_g_mol, "Pc": pc, "Vc": vc},
            output_property="Tc",
        )
        etc, epc, evc = refcritical.Hekayati_Raeissi(
            MW=component.molecular_weight_g_mol, Pc=4108000.0, Vc=v
        )
        p = {"Tc": tc, "Pc": pc, "Vc": vc, "omega": sourced(0.265, "omega")}
        expected = {"Tc": etc, "Pc": epc, "Vc": evc, "omega": 0.265}
    comparisons = {k: compare(p[k].value, v) for k, v in expected.items()}
    return {"derived_comparisons": comparisons, **pair_consumer(component, p, expected, family)}


def related_constant_unit(db, name, family):
    cas = "64-17-5"
    count = 431 if name == "CHEMSEP_XML" else 58
    assert len(db.records(name)) == count
    component = ComponentSpec(cas, "C2H6O", cas_number=cas)
    record = db.record(cas, method=name)
    expected = {"Tc": 600., "Pc": 4e6, "Vc": .0003, "omega": .2}
    p = {k: sourced(v, k) for k, v in expected.items()}
    path = db.source_directory / db.source(name).relative_path
    if name == "CHEMSEP_XML":
        raw = next(node for node in ET.parse(path).getroot()
                   if node.find("CAS").attrib["value"] == cas)
        for key, field in {"Tc": "CriticalTemperature", "Pc": "CriticalPressure",
                           "Vc": "CriticalVolume", "Zc": "CriticalCompressibility",
                           "omega": "AcentricityFactor"}.items():
            p[key] = record.value(key)
            item = raw.find(field)
            expected[key] = float(item.attrib["value"]) * (1e-3 if key == "Vc" else 1)
    else:
        p["Tc"] = record.value("Tc")
        expected["Tc"] = json.loads(path.read_text(encoding="utf8"))[cas]["Tc"]
    return {"source_count": count, "source": record.source.provenance.to_dict(),
            **pair_consumer(component, p, expected, family)}


def pure_unit(method, temperatures, raw):
    p = dict(raw)
    t = temperatures[0]
    if method == "Goodman":
        p, t = {"Tt": 350, "Vml": 0.0001}, 298.15
    spec = VOLUME_METHODS[method]
    component = ComponentSpec(
        "pure",
        "NaCl" if method == "Goodman" else "C3H8",
        default_phase=spec.phase,
        **({} if method == "Goodman" else {"molecular_weight_g_mol": 44.09562}),
    )
    pressure = p.pop("P", 101325)
    native_parameters = dict(p)
    if spec.native_output != "molar_volume":
        native_parameters["MW"] = component.molecular_weight_g_mol
    card = bridge(component, "volume", method, native_parameters, t, pressure=pressure)

    def vm(temp, xs=None):
        args = {**p, **({"P": pressure} if spec.uses_pressure else {})}
        raw_value = getattr(refvolume, method)(
            **args, **({} if method.startswith("Tait") else {"T": temp})
        )
        return (
            component.molecular_weight_g_mol / 1000 / raw_value
            if spec.native_output == "density"
            else raw_value * component.molecular_weight_g_mol / 1000
            if spec.native_output == "specific_volume"
            else raw_value
        )

    result = (
        crystal(card, vm(t))
        if spec.phase == "solid"
        else packet(card, t, vm(t), pressure)
        if spec.phase == "gas"
        else liquid([card], t, vm, pressure=pressure)
    )
    return {**card_checks(card, t, pressure), **result}


def volume_source_unit(db, name, curve_key):
    cas, formula = {
        "CRC_MOLTEN": ("7647-14-5", "NaCl"),
        "CRC_SOLID_VOLUME": ("7647-14-5", "NaCl"),
        "CRC_LIQUID_VOLUME": ("7732-18-5", "H2O"),
        "MCHAWEH_SN0": ("74-82-8", "CH4"),
        "CRC_VIRIAL": ("74-82-8", "CH4"),
    }.get(name, ("64-17-5", "C2H6O"))
    record = db.record(cas, method=name)
    options = (
        {"molecular_weight_g_mol": record.value("MW").value} if "MW" in record.field_names else {}
    )
    phase = "solid" if name == "CRC_SOLID_VOLUME" else "gas" if name == "CRC_VIRIAL" else "liquid"
    component = ComponentSpec(cas, formula, cas_number=cas, default_phase=phase, **options)
    candidates = db.volumes(cas, method=name)
    selected = (
        next(c for c in candidates if c.curve_key == curve_key) if curve_key else candidates[0]
    )
    t = (
        1150
        if name == "CRC_MOLTEN"
        else 310
        if name == "VDI"
        else 298.15
        if "_VOLUME" in name and name.startswith("CRC_")
        else 300
    )
    point = name in {"CRC_LIQUID_VOLUME", "CRC_SOLID_VOLUME"}
    build = (
        {
            "application_temperature_K": t,
            "application_note": "D4 explicit isothermal application; source temperature unknown",
        }
        if point
        else {"temperature_bounds_K": (t - 5, t + 15)}
    )
    if name == "VDI":
        build = {"temperature_bounds_K": (300, 320), "interpolation": "linear"}
    if selected.required_inputs:
        build["parameters"] = {
            key: sourced(SUPPLEMENT[key], key) for key in selected.required_inputs
        }
    card = db.property_input(cas, component, method=name, curve_id=selected.curve_id, **build)
    raw = (
        json.loads(
            (db.source_directory / db.source(name).relative_path).read_text(encoding="utf8")
        )[cas]
        if name == "VDI"
        else next(
            r
            for r in source_rows(db, name)
            if (
                r.get("CAS", "") == cas
                or (name == "CHEMSEP_XML" and r["CAS"].attrib["value"] == cas)
            )
        )
    )

    def vm(temp, xs=None):
        if name == "VDI":
            values = record.raw("Volume (l)")
            return values[0] + (values[1] - values[0]) * (temp - 300) / 20
        value = reference(name, raw, selected.curve_key, temp)
        return (
            component.molecular_weight_g_mol / 1000 / value
            if selected.output_unit == "kg/m^3"
            else value
        )

    result = (
        gas(card, t, crc_formula(raw))
        if name == "CRC_VIRIAL"
        else crystal(card, vm(t))
        if phase == "solid"
        else packet(card, t, vm(t))
        if point
        else liquid([card], t, vm)
    )
    return {**card_checks(card, t), **result, "source": name, "source_curve": selected.curve_key}


def mixture_unit(method, t, pressure, fractions):
    components = tuple(ComponentSpec(str(i), f) for i, f in enumerate(("C2H6O", "C7H8", "C10H22")))
    cards = [
        companion(c, pressure=pressure, rho=(800, 900, 1000)[i]) for i, c in enumerate(components)
    ]
    # Keep only the source volume declarations; liquid() supplies one caloric/transport companion.
    cards = [
        replace(
            c,
            package=replace(
                c.package,
                correlations=tuple(
                    q for q in c.package.correlations if q.property_id == "liquid_density"
                ),
            ),
            reference=None,
            uncertainties={k: v for k, v in c.uncertainties.items() if "liquid_density" in k},
        )
        for c in cards
    ]
    p = {
        "Tcs": (600, 650, 700),
        "Pcs": (4e6, 5e6, 6e6),
        "Vcs": (0.0002, 0.0003, 0.0004),
        "omegas": (0.2, 0.3, 0.4),
        "Zrs": (0.25, 0.27, 0.29),
        "MWs": tuple(c.molecular_weight_g_mol for c in components),
    }
    fields = (
        ()
        if method == "Amgat"
        else ("Tcs", "Pcs", "Zrs")
        if method == "Rackett_mixture"
        else ("Tcs", "Vcs", "omegas")
    )
    params = (
        {c.identifier: {key: p[key][i] for key in fields} for i, c in enumerate(components)}
        if fields
        else {}
    )
    pressure_parameters = {"Psat": 1e5} if method.endswith("compressed") else {}
    ref = (
        {
            "Vms": tuple(
                c.molecular_weight_g_mol / 1000 / rho
                for c, rho in zip(components, (800, 900, 1000), strict=True)
            )
        }
        if method == "Amgat"
        else {k: p[k] for k in (*fields, *(("MWs",) if method == "Rackett_mixture" else ()))}
    )
    ref = convert({**ref, **pressure_parameters, "P": pressure})
    return liquid(
        cards,
        t,
        lambda temp, xs: float(mathematical(method, mp.mpf(temp), ref, convert(xs))),
        pressure=pressure,
        fractions=dict(zip((c.identifier for c in components), fractions, strict=True)),
        receiver={"0": 0.1, "1": 0.8, "2": 0.1},
        mixing=method,
        parameters=params,
        pressure_parameters=pressure_parameters,
    )


def virial_unit(method):
    component = ComponentSpec("gas", "C7H8", default_phase="gas")
    p = {
        "Tc": 600,
        "Pc": 4e6,
        "Vc": 0.0003,
        "omega": 0.2,
        "a": -0.02 if method == "Meng" else 0.03,
        "b": 0.01,
    }
    keys = VIRIAL_METHODS[method].required.split()
    params = {k: p[k] for k in keys}
    if method == "Tsonopoulos_extended":
        params.update(a=p["a"], b=p["b"])
    if method == "Meng":
        params["a"] = p["a"]
    card = bridge(component, "virial", method, params, 900)
    b, c = (lambda t: independent(method, t, p)), (lambda t: mp.mpf(0))
    if VIRIAL_METHODS[method].third:
        bcard = bridge(component, "virial", "Abbott", {k: p[k] for k in ("Tc", "Pc", "omega")}, 900)
        card = PropertyInputCard.combine((bcard, card), reference_note="D4 explicit B/C pair")
        b, c = (lambda t: independent("Abbott", t, p)), b
    return {**card_checks(card, 900), **gas(card, 900, b, c)}


def units(root):
    cd, vd = ConstantDatabase(root), VolumeDatabase(root)
    rows = []

    def add(name, inputs, fn):
        rows.append((name, inputs, fn))

    for declaration, family in product(SOURCES, ("volume", "virial")):
        add(
            f"constant/{declaration[0]}/{family}",
            {"source": declaration, "family": family},
            lambda d=declaration, f=family: constant_unit(cd, d, f),
        )
    pure = {}
    for method, ts, p in CASES:
        pure.setdefault(method, (ts, p))
    pure.update(
        Tait=((300,), {"P": 1e7, "P_ref": 1e5, "rho_ref": 800, "B": 1e8, "C": 0.1}),
        Tait_molar=((300,), {"P": 1e7, "P_ref": 1e5, "V_ref": 1e-4, "B": 1e8, "C": 0.1}),
        COSTALD_compressed=(
            (300,),
            {"P": 1e7, "Tc": 500, "Pc": 4e6, "omega": 0.2, "Psat": 1e5, "Vs": 1e-4},
        ),
    )
    for method, (ts, p) in pure.items():
        add(
            f"pure/{method}",
            {"method": method, "temperatures": ts, "parameters": p},
            lambda m=method, ts=ts, p=p: pure_unit(m, ts, p),
        )
    sources = [(s.source_id, None) for s in vd.catalog if s.source_id != "COSTALD_PARAMETERS"]
    sources += [
        ("COSTALD_PARAMETERS", "COSTALD"),
        ("COSTALD_PARAMETERS", "Rackett"),
        ("CHEMSEP_XML", "LiquidDensity"),
        ("VDI", "Volume (l)"),
    ]
    for name, key in sources:
        add(
            f"volume-source/{name}/{key}",
            {"source": name, "curve": key},
            lambda n=name, k=key: volume_source_unit(vd, n, k),
        )
    for chain, variation, family in product(
        ("boiling", "Hekayati"), range(2), ("volume", "virial")
    ):
        add(
            f"derived/{chain}/{variation}/{family}",
            {"chain": chain, "variation": variation, "family": family},
            lambda c=chain, v=variation, f=family: derived_unit(c, v, f),
        )
    for method, t, p, x in product(
        ("Amgat", "Rackett_mixture", "COSTALD_mixture", "COSTALD_mixture_compressed"),
        (290, 320),
        (1e6, 1e7),
        ((0.2, 0.3, 0.5), (0.7, 0, 0.3)),
    ):
        add(
            f"mixture/{method}/{t}/{p}/{x}",
            {"method": method, "T": t, "P": p, "feed_x": x},
            lambda m=method, t=t, p=p, x=x: mixture_unit(m, t, p, x),
        )
    for method in VIRIAL_METHODS:
        add(f"virial/{method}", {"method": method, "T": 900}, lambda m=method: virial_unit(m))
    assert len(rows) == 104
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("core", "related-constants"), default="core")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 60
    rows, start = [], time.perf_counter()
    root = Path(chemicals.__file__).parent
    if args.block == "core":
        cases = units(root)
    else:
        db = ConstantDatabase(root)
        cases = [(f"related/{name}/{family}", {"source": name, "family": family},
                  lambda n=name, f=family: related_constant_unit(db, n, f))
                 for name, family in product(("CHEMSEP_XML", "VDI"), ("volume", "virial"))]
        assert len(cases) == 4
    total = len(cases)
    for i, (name, inputs, fn) in enumerate(cases, 1):
        row = {"unit": name, "inputs": inputs}
        try:
            row.update(status="passed", metrics=fn())
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        elapsed = time.perf_counter() - start
        print(
            f"{i}/{total} {name}: {row['status']} {i / elapsed:.2f}/s "
            f"ETA {elapsed * (total - i) / i:.1f}s",
            flush=True,
        )
    result = {
        "task": "NATIVE-CONV-22D4",
        "mode": "development",
        "reference_version": chemicals.__version__,
        "block": args.block,
        "total": total,
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )
    for row in rows:
        if row["status"] == "failed":
            print(row["unit"], row["failure"].splitlines()[-1])
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
