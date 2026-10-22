"""Fixed D3C source/thermodynamic block; upstream source code is never inspected."""

import argparse
import csv
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import chemicals
import chemicals.virial as ref
import mpmath as mp
from tests.reference.run_virial_pure_block import independent_terms, reference
from tests.test_virial_consumers import thermal_model

from chemworld.data.constants import ConstantDatabase
from chemworld.data.identifiers import IdentifierDatabase
from chemworld.data.volume import VolumeDatabase
from chemworld.physchem import (
    VIRIAL_METHODS,
    ComponentPropertyPackage,
    ComponentSpec,
    VirialGasThermodynamics,
    evaluate_virial_gas_state,
    virial_gas_batch_system,
    virial_gas_molar_volume_report,
    virial_temperature_report,
)
from chemworld.physchem.coupled_system import CoupledState, HeatFlux, ProcessContribution
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.solver_backend import ODESolverPolicy

R = mp.mpf("8.31446261815324")
FLUIDS = (("74-82-8", "CH4"), ("74-84-0", "C2H6"), ("124-38-9", "CO2"), ("7727-37-9", "N2"))
QUANTITIES = (
    "molar_volume",
    "density",
    "compressibility_factor",
    "dV_dT_P",
    "dV_dP_T",
    "internal_energy",
    "enthalpy",
    "cv",
    "cp",
    "entropy",
    "chemical_potential",
)


def read_tsv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def crc_formula(row):
    a = [mp.mpf(row[f"a{i}"]) for i in range(1, 6)]
    return lambda t: (
        mp.mpf("1e-6") * sum(v * (mp.mpf("298.15") / t - 1) ** i for i, v in enumerate(a))
    )


def independent_coefficient(method, row):
    p = {key: mp.mpf(row[key]) for key in ("Tc", "Pc", "Vc", "omega")}
    terms = independent_terms(method, p)
    factor = R * p["Tc"] / p["Pc"]
    if method == "Liu_Xiang":
        factor = p["Vc"] ** 2
    elif method == "Orbey_Vera":
        factor *= factor
    return lambda t: factor * sum(a * (t / p["Tc"]) ** n for n, a in terms)


def independent_state(t, p, b, c, mw):
    t, p = mp.mpf(t), mp.mpf(p)
    b0, c0 = b(t), c(t)
    v = mp.findroot(
        lambda x: R * t / x * (1 + b0 / x + c0 / x**2) - p, (R * t / p * 0.98, R * t / p * 1.02)
    )

    # Independent differentiable Helmholtz potential, including ideal datum.
    def helmholtz(temp, vol):
        u = 25 * (temp - 300)
        s = 100 + 25 * mp.log(temp / 300) + R * mp.log(vol / (R * 300 / 100000))
        return u - temp * s + R * temp * (b(temp) / vol + c(temp) / (2 * vol**2))

    s = -mp.diff(lambda temp: helmholtz(temp, v), t)
    u = helmholtz(t, v) + t * s
    cv = -t * mp.diff(lambda temp: helmholtz(temp, v), t, 2)
    pt = -mp.diff(lambda temp: mp.diff(lambda vol: helmholtz(temp, vol), v), t)
    pv = -mp.diff(lambda vol: helmholtz(t, vol), v, 2)
    h = u + p * v
    return {
        "molar_volume": v,
        "density": mw / 1000 / v,
        "compressibility_factor": p * v / (R * t),
        "dV_dT_P": -pt / pv,
        "dV_dP_T": 1 / pv,
        "internal_energy": u,
        "enthalpy": h,
        "cv": cv,
        "cp": cv - t * pt * pt / pv,
        "entropy": s,
        "chemical_potential": h - t * s,
    }


def catalogue():
    """Explicit semantic mapping, including differences; no alias count inflation."""
    common = {
        "BVirial_mixture": "virial_mixture_response(kind='B').value",
        "B_from_Z": "second_virial_from_compressibility",
        "B_to_Z": "compressibility_from_second_virial",
        "CVirial_mixture_Orentlicher_Prausnitz": "virial_mixture_response(kind='C').value",
        "Lee_Kesler_virial_CSP_Vcijs": "lee_kesler_cross_volumes",
        "Meng_Duan_2005_virial_CSP_kijs": "meng_duan_interactions",
        "Tarakad_Danner_virial_CSP_kijs": "tarakad_danner_interactions",
        "Z_from_virial_density_form": "density_virial_state.compressibility_factor",
        "Z_from_virial_pressure_form": "pressure_virial_volume; P*V/(RT)",
        "dV_dzs_virial": "virial_volume_composition_response.gradient",
        "d2V_dzizjs_virial": "virial_volume_composition_response.hessian",
        "dBVirial_mixture_dzs": "virial_mixture_response(kind='B').gradient",
        "d2BVirial_mixture_dzizjs": "virial_mixture_response(kind='B').hessian",
        "d3BVirial_mixture_dzizjzks": "virial_mixture_response(kind='B').third",
        "dCVirial_mixture_Orentlicher_Prausnitz_dzs": "virial_mixture_response(kind='C').gradient",
        "d2CVirial_mixture_Orentlicher_Prausnitz_dzizjs": (
            "virial_mixture_response(kind='C').hessian"),
        "d3CVirial_mixture_Orentlicher_Prausnitz_dzizjzks": (
            "virial_mixture_response(kind='C').third"),
        "dCVirial_mixture_dT_Orentlicher_Prausnitz": (
            "virial_mixture_response(kind='C').temperature_derivatives[1]"),
        "d2CVirial_mixture_dT2_Orentlicher_Prausnitz": (
            "virial_mixture_response(kind='C').temperature_derivatives[2]"),
        "d3CVirial_mixture_dT3_Orentlicher_Prausnitz": (
            "virial_mixture_response(kind='C').temperature_derivatives[3]"),
        "d2CVirial_mixture_Orentlicher_Prausnitz_dTdzs": (
            "virial_mixture_response(kind='C').temperature_gradient"),
    }
    for name in ("Tc", "Pc", "omega"):
        common[f"Tarakad_Danner_virial_CSP_{name}ijs"] = f"virial_cross_parameters()['{name}']"
    rows = []
    for name in ref.__all__:
        match = None
        for method in VIRIAL_METHODS:
            base = ("C" if VIRIAL_METHODS[method].third else "B") + "Virial_" + method
            if name == base or name in {base + suffix for suffix in ("_fast", "_vec", "_mat")}:
                suffix = name[len(base) :]
                native = {
                    "": "virial_value / virial_derivatives",
                    "_fast": "virial_fast",
                    "_vec": "virial_vector",
                    "_mat": "virial_matrix",
                }[suffix]
                match = {
                    "public": name,
                    "native": native + f"('{method}')",
                    "evidence": "D3A",
                    "block": "batch" if suffix else "main",
                    "method": method,
                }
                break
        if match is None and name == "Meng_virial_a":
            match = {
                "public": name,
                "native": "meng_polar_parameter",
                "evidence": "D3A",
                "block": "main:meng_polar",
            }
        if match is None and name in common:
            match = {
                "public": name,
                "native": common[name],
                "evidence": "D3B",
                "block": "complete 92-unit block",
            }
        if match is None:
            match = {"public": name, "native": None, "evidence": None}
        match["semantics"] = (
            "signed real C cube root; upstream negative-entry clipping differences retained"
            if "CVirial_mixture" in name
            else "explicit class/carbon count; recovered constants and heldout evidence"
            if name.startswith("Meng_Duan")
            else "same documented mathematics; native contracts"
        )
        rows.append(match)
    return rows


def run(output, complete_identities=False):
    global R
    mp.mp.dps = 60
    R = mp.mpf("8.31446261815324")
    started = time.monotonic()
    root = Path(chemicals.__file__).parent
    vd, cd, ids = (
        VolumeDatabase(root),
        ConstantDatabase(root),
        IdentifierDatabase(root / "Identifiers"),
    )
    crc = read_tsv(root / vd.source("CRC_VIRIAL").relative_path)
    heos = {r["CAS"]: r for r in read_tsv(root / cd.source("HEOS").relative_path)}
    identity_rows = {}
    sources = ("pubchem-small", "inorganic", "pubchem-large")
    if complete_identities:
        sources += ("example-user",)
    for source in sources:
        with (ids.source_directory / ids.source(source).filename).open(encoding="utf-8") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) >= 4:
                    identity_rows.setdefault(row[1], (source, row))
    rows = []

    def compare(label, actual, expected, *, rtol=3e-8, atol=1e-9):
        actual, expected = float(actual), float(expected)
        passed = abs(actual - expected) <= atol + rtol * abs(expected)
        rows[-1]["checks"].append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "rtol": rtol,
                "atol": atol,
                "passed": passed,
            }
        )

    def unit(name, inputs, function):
        row = {"id": name, "inputs": inputs, "checks": []}
        rows.append(row)
        try:
            function(row)
        except Exception as exc:
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
        row["passed"] = (
            bool(row["checks"])
            and "error" not in row
            and all(check["passed"] for check in row["checks"])
        )
        output.with_suffix(".partial.json").write_text(
            json.dumps({"results": rows}, allow_nan=False), encoding="utf-8"
        )
        if len(rows) % 25 == 0 or len(rows) == 447:
            elapsed = time.monotonic() - started
            print(
                f"D3C {len(rows)}/447 units; {len(rows) / elapsed:.1f}/s; "
                f"ETA {(447 - len(rows)) * elapsed / len(rows):.1f}s",
                flush=True,
            )

    def get_crc(row):
        cas = row["CAS"]
        source, _ = identity_rows[cas]
        component = ids.component(cas, source_ids=(source,), identifier=cas, phase="gas")
        return vd.property_input(
            cas, component, method="CRC_VIRIAL", temperature_bounds_K=(200, 800)
        )

    def heos_card(cas, formula, method, bounds):
        extras = (
            {"a": 0, "b": 0}
            if method == "Tsonopoulos_extended"
            else ({"a": 0} if method == "Meng" else {})
        )
        return cd.virial_input(
            cas,
            ComponentSpec(cas, formula, cas_number=cas),
            method=method,
            constant_source="HEOS",
            temperature_range_K=bounds,
            parameters=extras,
        )

    def gas_package(cas, formula):
        cards = [
            heos_card(cas, formula, method, (200, 800)) for method in ("Tsonopoulos", "Liu_Xiang")
        ]
        return ComponentPropertyPackage(
            cards[0].package.component, tuple(card.package.correlations[0] for card in cards)
        )

    for raw in crc:
        for t in (250, 400, 700):

            def crc_check(record, raw=raw, t=t):
                card = get_crc(raw)
                card = PropertyInputCard.from_json(json.dumps(card.to_dict()))
                (curve,) = card.package.correlations
                b = crc_formula(raw)
                reports = virial_temperature_report(curve, temperature_K=t)
                for order, value in enumerate(reports):
                    compare(f"B-dT{order}", value.value, mp.diff(b, mp.mpf(t), order), atol=1e-17)
                source, id_row = identity_rows[raw["CAS"]]
                record["identity_source"] = source
                record["source_declaration"] = curve.to_dict()
                mw = float(id_row[3])
                compare("source-MW", card.package.component.molecular_weight_g_mol, mw)
                disc = (R * t) ** 2 + 4 * 100 * R * t * b(t)
                if disc <= 0:
                    try:
                        card.evaluate_state(
                            temperature_K=t,
                            pressure_Pa=100,
                            phase="gas",
                            gas_model="virial",
                            quantities=("molar_volume",),
                        )
                    except ValueError as exc:
                        record["root_rejection"] = str(exc)
                        record["root_candidates"] = getattr(exc, "candidates", ())
                        compare(
                            "expected-root-rejection",
                            bool(record["root_candidates"]),
                            True,
                            rtol=0,
                            atol=0,
                        )
                    else:
                        raise AssertionError("nonphysical root accepted")
                else:
                    v = (R * t + mp.sqrt(disc)) / 200
                    result = card.evaluate_state(
                        temperature_K=t,
                        pressure_Pa=100,
                        phase="gas",
                        gas_model="virial",
                        quantities=("molar_volume", "density"),
                    )
                    compare("volume", result.properties["molar_volume"].value, v)
                    compare("density", result.properties["density"].value, mw / 1000 / v)

            unit(f"crc-{raw['CAS']}-{t}", {"source_row": raw, "T": t, "P": 100}, crc_check)

    for cas, formula in FLUIDS:
        raw = heos[cas]
        tc = float(raw["Tc"])
        for method in VIRIAL_METHODS:
            for tr in (1.2, 2, 3):

                def source_check(
                    record, cas=cas, formula=formula, method=method, tr=tr, tc=tc, raw=raw
                ):
                    card = heos_card(cas, formula, method, (1.1 * tc, 3.1 * tc))
                    card = PropertyInputCard.from_json(json.dumps(card.to_dict()))
                    (curve,) = card.package.correlations
                    p = dict(curve.coefficients)
                    record["declaration"] = curve.to_dict()
                    for key in p.keys() & {"Tc", "Pc", "Vc", "omega"}:
                        compare(key, p[key], float(raw[key]), rtol=0, atol=0)
                    for order, value in enumerate(
                        virial_temperature_report(curve, temperature_K=tr * tc)
                    ):
                        compare(
                            f"public-dT{order}",
                            value.value,
                            reference(method, tr * tc, p, order),
                            atol=1e-17,
                        )

                unit(f"source-{cas}-{method}-{tr}", {"source_row": raw, "Tr": tr}, source_check)

    ideal = thermal_model()
    for cas, formula in FLUIDS:
        for t in (300, 450, 600):

            def gas_check(record, cas=cas, formula=formula, t=t):
                package = gas_package(cas, formula)
                report = evaluate_virial_gas_state(
                    package,
                    temperature_K=t,
                    pressure_Pa=1e5,
                    quantities=QUANTITIES,
                    ideal_caloric=ideal,
                )
                expected = independent_state(
                    t,
                    100000,
                    independent_coefficient("Tsonopoulos", heos[cas]),
                    independent_coefficient("Liu_Xiang", heos[cas]),
                    mp.mpf(str(package.component.molecular_weight_g_mol)),
                )
                for key, value in expected.items():
                    compare(key, report.properties[key].value, value)
                record["report"] = json.loads(json.dumps(report.to_dict()))
                q = report.properties
                compare(
                    "H-U=PV",
                    q["enthalpy"].value - q["internal_energy"].value,
                    1e5 * q["molar_volume"].value,
                )

            unit(f"gas-{cas}-{t}", {"T": t, "P": 1e5, "ideal_caloric": ideal.to_dict()}, gas_check)

    for kind in ("CRC", "HEOS_BC"):

        def process_check(record, kind=kind):
            raw = next(row for row in crc if row["CAS"] == "74-82-8")
            package = get_crc(raw).package if kind == "CRC" else gas_package("74-82-8", "CH4")
            b = (
                crc_formula(raw)
                if kind == "CRC"
                else independent_coefficient("Tsonopoulos", heos["74-82-8"])
            )
            c = (
                (lambda _: mp.mpf(0))
                if kind == "CRC"
                else independent_coefficient("Liu_Xiang", heos["74-82-8"])
            )

            def energy(t):
                return 2 * (
                    25 * (t - 300)
                    - R
                    * t
                    * t
                    * (mp.diff(b, t) / mp.mpf(".005") + mp.diff(c, t) / (2 * mp.mpf(".005") ** 2))
                )

            model = VirialGasThermodynamics(package, ideal)
            policy = ODESolverPolicy(
                solver_id="D3C-development", method="DOP853", rtol=1e-10, atol=1e-12
            )
            system = virial_gas_batch_system(
                model, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy
            )
            start = CoupledState({"74-82-8": 2.0}, 300, 10)
            result = system.advance(start, 60, evaluation_times_s=tuple(range(0, 61, 10)))
            record["trajectory"] = []
            compare("trajectory-count", len(result.trajectory), 7, rtol=0, atol=0)
            for point in result.trajectory:
                expected = mp.findroot(
                    lambda t, time_s=point.time_s: energy(t) - energy(mp.mpf(300)) - 5 * time_s,
                    (mp.mpf(300), mp.mpf(310)),
                )
                compare(f"T-{point.time_s}", point.temperature_K, expected, rtol=0, atol=2e-6)
                compare(
                    f"Q-{point.time_s}", point.boundary_heat_J, 5 * point.time_s, rtol=0, atol=1e-7
                )
                compare(f"n-{point.time_s}", point.amounts_mol["74-82-8"], 2, rtol=0, atol=0)
                record["trajectory"].append(
                    {
                        "time_s": point.time_s,
                        "T": point.temperature_K,
                        "Q": point.boundary_heat_J,
                        "pressure_Pa": model.pressure_pa(point),
                    }
                )
            half = system.advance(start, 30)
            restored = VirialGasThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
            resumed = virial_gas_batch_system(
                restored, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy
            ).advance(half.final_state, 30)
            compare(
                "resume-T",
                resumed.final_state.temperature_K,
                result.final_state.temperature_K,
                rtol=0,
                atol=2e-6,
            )
            compare("time", result.final_state.time_s, 60, rtol=0, atol=0)
            compare("energy-residual", result.energy_balance_residual_J, 0, rtol=0, atol=1e-4)
            record["solver"] = dict(result.solver_diagnostic)
            record["thermal_declaration"] = model.to_dict()

        unit(
            f"process-{kind}",
            {"n": 2, "V_L": 10, "T": 300, "heater_W": 5, "duration_s": 60},
            process_check,
        )

    # Boundary fixtures use the explicitly specified methane identity and are
    # independent of coverage/missing fields in the identifier-source block.
    base = vd.property_input(
        "74-82-8",
        ComponentSpec("74-82-8", "CH4", cas_number="74-82-8"),
        method="CRC_VIRIAL",
        temperature_bounds_K=(200, 800),
    ).package
    (curve,) = base.correlations
    boundaries = {
        "unknown_temperature_range": lambda: evaluate_virial_gas_state(
            replace(base, correlations=(replace(curve, validity_ranges={}),)),
            temperature_K=300,
            pressure_Pa=1e5,
            quantities=("density",),
        ),
        "outside_range": lambda: evaluate_virial_gas_state(
            base, temperature_K=900, pressure_Pa=1e5, quantities=("density",)
        ),
        "missing_coefficient": lambda: virial_temperature_report(
            replace(curve, coefficients={"a1": 0}), temperature_K=300
        ),
        "missing_heat_capacity": lambda: base.evaluate_state(
            temperature_K=300,
            pressure_Pa=1e5,
            phase="gas",
            gas_model="virial",
            quantities=("cv",),
            reference=ideal.reference,
        ),
        "missing_reference": lambda: replace(
            base, correlations=(curve, ideal.heat_capacity)
        ).evaluate_state(
            temperature_K=300, pressure_Pa=1e5, phase="gas", gas_model="virial", quantities=("cv",)
        ),
        "wrong_reference_phase": lambda: VirialGasThermodynamics(
            base, replace(ideal, reference=replace(ideal.reference, phase="liquid"))
        ),
        "no_stable_root": lambda: virial_gas_molar_volume_report(
            temperature_K=300, pressure_Pa=1e5, second_virial_m3_mol=-1
        ),
        "duplicate_B": lambda: evaluate_virial_gas_state(
            replace(base, correlations=(curve, replace(curve, correlation_id="other"))),
            temperature_K=300,
            pressure_Pa=1e5,
            quantities=("density",),
        ),
        "source_identity": lambda: cd.virial_input(
            "74-82-8",
            ComponentSpec("wrong", "N2"),
            method="Abbott",
            constant_source="HEOS",
            temperature_range_K=(200, 800),
        ),
        "material_change": lambda: VirialGasThermodynamics(base, ideal).chemical_energy_rate_w(
            CoupledState({"74-82-8": 2}, 300, 10),
            {"74-82-8": 0.1},
            (ProcessContribution(amount_rates_mol_s={"74-82-8": 0.1}),),
        ),
    }
    for name, function in boundaries.items():

        def reject(record, function=function):
            try:
                function()
            except ValueError as exc:
                record["rejection"] = str(exc)
                compare("rejected", True, True, rtol=0, atol=0)
            else:
                raise AssertionError("missing physics silently accepted")

        unit("boundary-" + name, {}, reject)
    mapping = catalogue()
    result = {
        "block": "NATIVE-CONV-22D3C",
        "mode": "development",
        "reference": chemicals.__version__,
        "identity_sources": sources,
        "input_block": "complete-identities" if complete_identities else "original",
        "units": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "failed_units": [r["id"] for r in rows if not r["passed"]],
        "comparisons": sum(len(r["checks"]) for r in rows),
        "results": rows,
        "catalogue": mapping,
        "catalogue_total": len(mapping),
        "catalogue_unmapped": [r["public"] for r in mapping if r["native"] is None],
        "CRC_source_rows": len(crc),
        "elapsed_seconds": time.monotonic() - started,
    }
    result["complete_and_passed"] = (
        len(rows) == 447
        and len(crc) == 105
        and len(mapping) == 56
        and not result["catalogue_unmapped"]
        and not result["failed_units"]
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"results", "catalogue"}}),
        flush=True,
    )
    return result["complete_and_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete-identities", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.complete_identities) else 1)
