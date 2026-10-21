"""Fixed F1 development reference blocks; no reference production imports."""
# ruff: noqa: C408 -- compact scientific parameters

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import solubility as ref
from chemicals.rachford_rice import Rachford_Rice_solution
from pressure_reference import finite_difference

from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.equilibrium_chemistry import solid_solubility_mole_fraction
from chemworld.physchem.henry import (
    HENRY_SCALES,
    henry_convert,
    henry_matrix,
    henry_pressure_derivatives,
    henry_pressure_mixture,
    henry_raoult_flash,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.solubility import (
    HANSEN_SOURCES,
    HansenParameters,
    freezing_point_depression,
    hansen_distance,
    hildebrand_parameter,
    solid_solubility,
)
from chemworld.physchem.specs import (
    ComponentSpec,
    ComponentUncertainty,
    MixtureSpec,
    PropertyCorrelation,
)

TEMPERATURES = (260, 300, 380)
FITS = (
    (15, 300, 0.04, 0.001, 100, 0.00001),
    (20, -2000, 0, 0, 0, 0),
    (5, 0, 2, -0.003, 0, 0),
    (12, 0, 0, 0, 20000, -0.00001),
)
SOURCES = dict(
    zip(
        HANSEN_SOURCES,
        (
            ("alshehri", "ALSHERI_HANSEN", 888),
            ("hspipy", "HSPIPY", 248),
            ("ruben_manuel", "MANUEL_RUBEN_2022", 1211),
            ("schrier", "WDR_SCHRIER", 113),
        ),
        strict=True,
    )
)


def compare(actual, expected, *, atol=1e-12, rtol=3e-10):
    assert math.isfinite(actual) and math.isfinite(expected), (actual, expected)
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)
    return {"actual": actual, "reference": expected, "absolute_error": abs(actual - expected)}


def differences(actual, function, t, order=1):
    levels = finite_difference(function, t, order)
    compare(*levels, atol=1e-8, rtol=3e-5)
    return [compare(actual, level, atol=1e-8, rtol=3e-5) for level in levels]


def scale_unit(value, rho, mw, old, new):
    result = henry_convert(
        value, old, new, solvent_molar_density_mol_m3=rho, solvent_molecular_weight_g_mol=mw
    )
    return {
        "value": compare(result, ref.Henry_converter(value, old, new, rhom=rho, MW=mw)),
        "roundtrip": compare(
            henry_convert(
                result,
                new,
                old,
                solvent_molar_density_mol_m3=rho,
                solvent_molecular_weight_g_mol=mw,
            ),
            value,
        ),
    }


def scale_units():
    for i, (v, rho, mw) in enumerate(
        ((1.2e-5, 55344.59, 18.01528), (0.1, 12000, 78.11), (1e4, 9000, 100))
    ):
        for old in HENRY_SCALES:
            for new in HENRY_SCALES:
                yield (
                    f"scale/{i}/{old}/{new}",
                    dict(value=v, rho=rho, MW=mw),
                    (lambda v=v, r=rho, m=mw, a=old, b=new: scale_unit(v, r, m, a, b)),
                )


def henry_curve(p):
    return PropertyCorrelation(
        "synthetic-solute-in-solvent",
        "henry_pressure",
        p,
        {"temperature": "K"},
        "Pa",
        "henry_pressure",
        {"temperature": (250, 400)},
        "Synthetic Henry pair; not experimental data",
        {"solvent_id": "solvent"},
    )


def henry_unit(t, p):
    values = henry_pressure_derivatives(t, **p, order=3)

    def expression(T):
        a, b, c, d, e, f = (mp.mpf(str(p[k])) for k in "ABCDEF")
        return mp.exp(a + b / T + c * mp.log(T) + d * T + e / T**2 + f * T**2)

    with mp.workdps(50):
        derivatives = [
            compare(values[i], float(mp.diff(expression, mp.mpf(t), i))) for i in (1, 2, 3)
        ]
    curve = henry_curve(p)
    return {
        "value": compare(values[0], ref.Henry_pressure(t, **p)),
        "derivatives": derivatives,
        "registry": compare(evaluate_correlation(curve, temperature_K=t).value, values[0]),
        "registry_slope": compare(
            correlation_temperature_derivative(curve, temperature_K=t).value, values[1]
        ),
    }


def henry_units():
    for i, p in enumerate(FITS):
        for t in TEMPERATURES:
            params = dict(zip("ABCDEF", p, strict=True))
            yield f"Henry/{i}/{t}", dict(T=t, **params), lambda t=t, p=params: henry_unit(t, p)


def matrices(t, absent):
    logs, d1, d2 = ([[None] * 4 for _ in range(4)] for _ in range(3))
    for i, pairs in ((2, ((20, -300), (22, -600))), (3, ((21, -500), (23, -800)))):
        for j, (a, b) in enumerate(pairs):
            if not (absent and i == 3 and j == 1):
                logs[i][j], d1[i][j], d2[i][j] = a + b / t, -b / t**2, 2 * b / t**3
    return logs, d1, d2


def matrix_unit(t, z, absent, policy):
    logs, d1, d2 = matrices(t, absent)
    mask = (False, False, True, True)
    native = henry_matrix(logs, z, mask, dlog_dT=d1, d2log_dT2=d2, missing=policy, temperature_K=t)
    legacy = [[[0 if v is None else v for v in row] for row in m] for m in (logs, d1, d2)]
    skip = policy == "ignore"
    reference = (
        ref.Henry_constants(legacy[0], z, mask, skip_zero=skip),
        ref.dHenry_constants_dT(*legacy[:2], z, mask, skip_zero=skip),
        ref.d2Henry_constants_dT2(*legacy, z, mask, skip_zero=skip),
    )
    comparisons, derivatives = [], []
    assert native[:2] == (None, None)
    for i in (2, 3):
        for order in range(3):
            comparisons.append(compare(native[i][order].value, reference[order][i]))
        for order in (1, 2):

            def at(T, i=i):
                raw = matrices(T, absent)[0]
                adapted = [[0 if v is None else v for v in row] for row in raw]
                return ref.Henry_constants(adapted, z, mask, skip_zero=skip)[i]

            derivatives.extend(differences(native[i][order].value, at, t, order))
    hs = [math.exp(logs[2][j]) for j in (0, 1)] + [None, None]
    mix = henry_pressure_mixture(hs, mole_fractions=z, missing="ignore")
    comparisons.append(compare(mix.value, ref.Henry_pressure_mixture(hs, zs=z)))
    return {
        "values": comparisons,
        "finite_differences": derivatives,
        "coverage": [native[i][0].to_dict() for i in (2, 3)],
    }


def mixture_units():
    for t in TEMPERATURES:
        for z in ((0.6, 0.2, 0.15, 0.05), (0.1, 0.6, 0.1, 0.2)):
            for absent in (False, True):
                for policy in ("ignore", "zero_log"):
                    yield (
                        f"matrix/{t}/{z}/{absent}/{policy}",
                        dict(T=t, z=z, absent=absent, policy=policy),
                        lambda t=t, z=z, a=absent, p=policy: matrix_unit(t, z, a, p),
                    )


def solid_unit(t, cpl, cps, gamma):
    def reference(T):
        return ref.solubility_eutectic(T, 350, 12000, cpl, cps, gamma)

    v, d = derivative_values(lambda T: solid_solubility(T, 350, 12000, cpl, cps, gamma), t)
    old = solid_solubility_mole_fraction(
        temperature_K=t,
        melting_temperature_K=350,
        enthalpy_fusion_J_mol=12000,
        liquid_heat_capacity_J_mol_K=cpl,
        solid_heat_capacity_J_mol_K=cps,
        activity_coefficient=gamma,
    )
    return {
        "value": compare(v, reference(t)),
        "derivative": differences(d, reference, t),
        "old_consumer": compare(old, v),
    }


def freezing_unit(x, basis):
    native = (
        {"mole_fraction": x}
        if basis == "x"
        else {"molality_mol_kg": x * 1000 / 18.01528, "solvent_molecular_weight_g_mol": 18.01528}
    )
    upstream = {"x": x} if basis == "x" else {"M": x * 1000 / 18.01528, "MW": 18.01528}
    return {
        "value": compare(
            freezing_point_depression(353.35, 19110, **native),
            ref.Tm_depression_eutectic(353.35, 19110, **upstream),
        )
    }


def hildebrand_unit(t, volume):
    v, d = derivative_values(
        lambda T: hildebrand_parameter(
            T, vaporization_enthalpy_J_mol=40000, liquid_molar_volume_m3_mol=volume
        ),
        t,
    )

    def reference(T):
        return ref.solubility_parameter(T, 40000, volume)

    return {"value": compare(v, reference(t)), "derivative": differences(d, reference, t)}


def solubility_units():
    for t in (250, 300, 340):
        for cpl, cps in ((0, 0), (100, 70)):
            for gamma in (1, 3):
                yield (
                    f"solid/{t}/{cpl}/{gamma}",
                    dict(T=t, Cpl=cpl, Cps=cps, gamma=gamma),
                    (lambda t=t, a=cpl, b=cps, g=gamma: solid_unit(t, a, b, g)),
                )
    for x in (0.001, 0.01, 0.02):
        for basis in ("x", "molality"):
            yield (
                f"freezing/{x}/{basis}",
                dict(x=x, basis=basis),
                lambda x=x, b=basis: freezing_unit(x, b),
            )
    for volume in (0.00008, 0.00012):
        for t in (280, 300, 340):
            yield (
                f"Hildebrand/{volume}/{t}",
                dict(Vm=volume, T=t),
                lambda t=t, v=volume: hildebrand_unit(t, v),
            )


def flash_unit(t, pressure, solute):
    components = (ComponentSpec("solvent", "H2O"), ComponentSpec("solute", "CO2"))
    mixture = MixtureSpec.from_mole_fractions(
        components,
        (1 - solute, solute),
        phase_label="liquid",
        temperature_K=t,
        pressure_Pa=pressure,
    )
    curve = henry_curve(dict(A=math.log(200000) + 1, B=-300))
    source = InputSource(
        DataSourceProvenance(
            "synthetic", 0, "Declared synthetic pair", "1", "original synthetic data"
        ),
        "synthetic",
    )
    card = PropertyInputCard(
        ComponentPropertyPackage(components[1], (curve,)),
        (source,),
        {
            curve.correlation_id: ComponentUncertainty(
                curve.correlation_id,
                unit="Pa",
                coverage="not quantified",
                source_id="synthetic",
                note="Synthetic numerical check",
            )
        },
        None,
        "No caloric reference; only isothermal phase split",
    )
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    h = restored.package.evaluate("henry_pressure", temperature_K=t, validity_policy="raise")
    psat_curve = PropertyCorrelation(
        "solvent",
        "tde_pv_expansion",
        {"a1": math.log(20000) - 3, "a2": 0, "a3": 0, "a4": 0.01},
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": (250, 400)},
        "Synthetic solvent",
    )
    psat = evaluate_correlation(psat_curve, temperature_K=t)
    result = henry_raoult_flash(
        mixture, henry_evaluations={"solute": h}, solvent_vapor_pressures={"solvent": psat}
    )
    k = [
        20000 * math.exp(0.01 * (t - 300)) / pressure,
        ref.Henry_pressure(t, math.log(200000) + 1, -300) / pressure,
    ]
    z = [1 - solute, solute]
    at_zero = sum(x * (v - 1) for x, v in zip(z, k, strict=True))
    at_one = sum(x * (v - 1) / v for x, v in zip(z, k, strict=True))
    comparisons = []
    if at_zero <= 0:
        vf = 0
    elif at_one >= 0:
        vf = 1
    else:
        vf, xs, ys = Rachford_Rice_solution(z, k)
        for i, identity in enumerate(("solvent", "solute")):
            comparisons.extend(
                (
                    compare(result.liquid_composition[identity], xs[i], atol=1e-9, rtol=0),
                    compare(result.vapor_composition[identity], ys[i], atol=1e-9, rtol=0),
                )
            )
    comparisons.append(compare(result.vapor_fraction, vf, atol=1e-9, rtol=0))
    for i, identity in enumerate(("solvent", "solute")):
        amount = 2.5 * (
            (1 - result.vapor_fraction) * result.liquid_composition[identity]
            + result.vapor_fraction * result.vapor_composition[identity]
        )
        comparisons.append(compare(amount, 2.5 * z[i], atol=1e-9, rtol=0))
    return {"comparisons": comparisons, "result": result.to_dict(), "input_card": card.to_dict()}


def flash_units(*, phase_coverage=False):
    for t in (280, 300, 340):
        for pressure in (40000, 60000, 80000) if phase_coverage else (10000, 100000, 1000000):
            for x in (0.4, 0.6) if phase_coverage else (0.05, 0.3):
                yield (
                    f"flash/{t}/{pressure}/{x}",
                    dict(T=t, P=pressure, z_solute=x),
                    (lambda t=t, p=pressure, x=x: flash_unit(t, p, x)),
                )


def source_unit(database, source, index, raw, anchor_raw):
    record = database.records(source)[index]
    assert record.source_key == raw["CAS"]
    assert all(record.raw(k) == raw[k] for k in record.field_names)
    native = HansenParameters.from_record(record)
    restored = HansenParameters.from_dict(json.loads(json.dumps(native.to_dict())))
    assert restored == native
    comparisons = []
    for key in "DPH":
        field = "HANSEN_DELTA_" + key
        getter = getattr(ref, "hansen_delta_" + key.lower())
        methods = getattr(ref, "hansen_delta_" + key.lower() + "_methods")(raw["CAS"])
        available = database.methods(raw["CAS"], field, sources=HANSEN_SOURCES)
        assert {SOURCES[m][1] for m in available} == set(methods)
        value = record.value(field)
        if raw[field] == "":
            assert value.quality == "missing" and value.value is None
            assert getter(raw["CAS"], method=SOURCES[source][1]) is None
        else:
            assert value.unit == "Pa^0.5" and value.quality == "valid"
            comparisons.append(compare(value.value, float(raw[field])))
            comparisons.append(compare(value.value, getter(raw["CAS"], method=SOURCES[source][1])))
    if any(raw["HANSEN_DELTA_" + k] == "" for k in "DPH"):
        try:
            native.values_pa_half()
        except ValueError:
            return {"comparisons": comparisons, "incomplete_vector": 1}
        raise AssertionError("missing source value was fabricated")
    anchor = HansenParameters.from_record(database.records(source)[0])
    expected = (
        sum(
            w * (float(raw["HANSEN_DELTA_" + k]) - float(anchor_raw["HANSEN_DELTA_" + k])) ** 2
            for w, k in zip((4, 1, 1), "DPH", strict=True)
        )
        ** 0.5
    )
    comparisons.append(compare(hansen_distance(restored, anchor).value, expected))
    radius = InputValue(10, "MPa^0.5", None, "Synthetic explicit compatibility radius")
    comparisons.append(
        compare(
            hansen_distance(restored, anchor, interaction_radius=radius).value, expected / 10000
        )
    )
    return {"comparisons": comparisons, "incomplete_vector": 0}


def source_units(directory):
    database = ConstantDatabase(directory)
    for source, (prefix, _, count) in SOURCES.items():
        path = directory / "Misc" / (prefix + "_hansen_solubility_parameters.tsv")
        with path.open(encoding="utf8", newline="") as file:
            raw = list(csv.DictReader(file, delimiter="\t"))
        assert len(raw) == count and len(database.records(source)) == count
        for i, row in enumerate(raw):
            yield (
                f"{source}/{i}",
                dict(source=source, row=i, raw=row),
                (lambda s=source, i=i, r=row, a=raw[0]: source_unit(database, s, i, r, a)),
            )


def main():
    parser = argparse.ArgumentParser()
    counts = dict(scales=972, henry=12, mixtures=24, solubility=24, flash=18, sources=2460)
    counts["flash-phase"] = 18
    parser.add_argument("--block", choices=counts, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    blocks = dict(
        scales=scale_units,
        henry=henry_units,
        mixtures=mixture_units,
        solubility=solubility_units,
        flash=flash_units,
        sources=lambda: source_units(args.source_directory),
    )
    blocks["flash-phase"] = lambda: flash_units(phase_coverage=True)
    units = list(blocks[args.block]())
    assert len(units) == counts[args.block]
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    start, rows = time.perf_counter(), []
    for identity, inputs, run in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=run(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 200 == 0 or len(rows) == len(units):
            elapsed = time.perf_counter() - start
            print(
                f"{len(rows)}/{len(units)}; {len(rows) / elapsed:.1f}/s; "
                f"ETA {(len(units) - len(rows)) * elapsed / len(rows):.1f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "block": args.block,
        "reference_version": chemicals.__version__,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:3]:
        print(row["unit"], row["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
