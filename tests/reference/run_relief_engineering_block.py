"""Fixed 3593 inputs / 3760 comparisons. Imported numeric tables stay local."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose, log10
from pathlib import Path

import fluids
import fluids.safety_valve as reference
from scipy.interpolate import bisplev

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.relief_valves import (
    api520_critical_coefficient,
    api520_gas_backpressure_factor,
    api520_liquid_backpressure_factor,
    api520_steam_pressure_factor,
    api520_steam_superheat_factor,
    api520_viscosity_factor,
    api521_flare_noise,
    api521_noise_curve,
    api526_select_orifice,
    size_gas_relief_valve,
    size_liquid_relief_valve,
    size_steam_relief_valve,
    vdi3732_flare_noise,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar


def parameter_table(identity, axes, coordinates, responses):
    source = InputSource(
        DataSourceProvenance(
            "local-reference",
            0,
            "Locally installed fluids 1.3.1 safety_valve numeric API520 parameter tables",
            "1.3.1",
            "Software MIT; API numeric-table redistribution not established; local use only",
        ),
        "literature",
    )
    columns = tuple(
        QuantityColumn(name, unit, "parameter", (-1e20, 1e20), "explicit evaluation axis")
        for name, unit in axes
    )
    columns += (QuantityColumn("factor", "dimensionless", "parameter", (0, 2), "API correction"),)
    rows = []
    for index, (point, response) in enumerate(zip(coordinates, responses, strict=True)):
        values = {
            name: InputValue(value, unit, None, "Not quantified by source")
            for (name, unit), value in zip(axes, point, strict=True)
        }
        values["factor"] = InputValue(response, "dimensionless", None, "Not quantified by source")
        rows.append(
            ScientificRow(str(index), "local-reference", identity, str(index), "parameters", values)
        )
    table = ScientificTable(identity, "parameters", columns, (source,), tuple(rows))
    return TabulatedScalar(
        table, tuple((name, name, unit) for name, unit in axes), "factor", "dimensionless"
    )


def tables():
    data = {}
    for edition, temperatures in (("7E", reference._KSH_tempKs_7E), ("10E", reference._KSH_K_10E)):
        pressures = getattr(reference, "_KSH_Pa_" + edition)
        factors = getattr(reference, "_KSH_factors_" + edition)
        data[edition] = parameter_table(
            "steam-" + edition,
            (("pressure_Pa", "Pa"), ("temperature_K", "K")),
            product(pressures, temperatures),
            (v for row in factors for v in row),
        )
    for name, x, y in (
        ("W", reference.Kw_x, reference.Kw_y),
        ("B10", reference.Kb_10_over_x, reference.Kb_10_over_y),
        ("B16", reference.Kb_16_over_x, reference.Kb_16_over_y),
    ):
        data[name] = parameter_table(
            "backpressure-" + name,
            (("gauge_backpressure_fraction", "dimensionless"),),
            ((v / 100,) for v in x),
            y,
        )
    return data


def inputs():
    for gamma, ratio, mass in product((1.1, 1.4, 1.67), (0.1, 0.8), (1, 10)):
        yield {"family": "gas", "gamma": gamma, "ratio": ratio, "mass": mass}
    for edition, viscosity, backpressure in product(("7E", "10E"), (0.005, 0.2), (0.1, 0.4)):
        yield {
            "family": "liquid",
            "edition": edition,
            "viscosity": viscosity,
            "backpressure": backpressure,
        }
    for edition, pressure, temperature in product(("7E", "10E"), (1e6, 1e7, 18e6), (550, 700, 850)):
        yield {
            "family": "steam",
            "edition": edition,
            "pressure": pressure,
            "temperature": temperature,
        }
    for overpressure, fraction in product((0.1, 0.16, 0.21), (0.2, 0.4, 0.5)):
        yield {"family": "backpressure", "overpressure": overpressure, "fraction": fraction}
    for edition, reynolds in product(("7E", "10E"), (100, 1000, 1e6)):
        yield {"family": "viscosity", "edition": edition, "reynolds": reynolds}
    for ratio, mass in product((1.1, 2.92, 2.925, 2.93, 10), (0.5, 10)):
        yield {"family": "noise", "ratio": ratio, "mass": mass}
    for letter, area in zip(reference.API526_letters, reference.API526_A, strict=True):
        yield {"family": "orifice", "letter": letter, "area": area}
    for edition, temperatures in (("7E", reference._KSH_tempKs_7E), ("10E", reference._KSH_K_10E)):
        pressures = getattr(reference, "_KSH_Pa_" + edition)
        for pressure, temperature in product(pressures, temperatures):
            yield {
                "family": "table",
                "edition": edition,
                "kind": "node",
                "pressure": pressure,
                "temperature": temperature,
            }
        mid_p = [(p + q) / 2 for p, q in pairwise(pressures)]
        mid_t = [(t + u) / 2 for t, u in pairwise(temperatures)]
        for pressure, temperature in product(mid_p, mid_t):
            yield {
                "family": "table",
                "edition": edition,
                "kind": "midpoint",
                "pressure": pressure,
                "temperature": temperature,
            }


def compare(name, native, expected, tolerance=2e-11):
    return {
        "quantity": name,
        "native": native,
        "reference": expected,
        "relative_error": abs(native - expected) / max(abs(expected), 1e-300),
        "relative_tolerance": tolerance,
        "absolute_tolerance": 1e-10,
        "passed": isclose(native, expected, rel_tol=tolerance, abs_tol=1e-10),
    }


def run(case, values, data):
    family = case["family"]
    if family == "gas":
        args = {
            "temperature_K": 400,
            "compressibility_factor": 0.95,
            "molar_mass_g_mol": 28.97,
            "heat_capacity_ratio": case["gamma"],
            "upstream_pressure_Pa": 1e6,
            "downstream_pressure_Pa": 1e6 * case["ratio"],
        }
        expected = reference.API520_A_g(
            m=case["mass"],
            T=400,
            Z=0.95,
            MW=28.97,
            k=case["gamma"],
            P1=1e6,
            P2=args["downstream_pressure_Pa"],
        )
        values.append(
            compare(
                "C", api520_critical_coefficient(case["gamma"]), reference.API520_C(case["gamma"])
            )
        )
        values.append(
            compare(
                "area", size_gas_relief_valve(**args, mass_flow_kg_s=case["mass"]).area_m2, expected
            )
        )
        values.append(
            compare(
                "inverse_mass",
                size_gas_relief_valve(**args, area_m2=expected).mass_flow_kg_s,
                case["mass"],
                5e-8,
            )
        )
    elif family == "liquid":
        p1, p2 = 1.1 * 1.5e6 + 101325, case["backpressure"] * 1.5e6 + 101325
        args = {
            "density_kg_m3": 900,
            "upstream_pressure_Pa": p1,
            "downstream_pressure_Pa": p2,
            "overpressure": 0.1,
            "viscosity_Pa_s": case["viscosity"],
            "edition": case["edition"],
            "backpressure_table": data["W"],
        }
        expected = reference.API520_A_l(
            m=30,
            rho=900,
            P1=p1,
            P2=p2,
            overpressure=0.1,
            mu=case["viscosity"],
            edition=case["edition"],
        )
        result = size_liquid_relief_valve(**args, mass_flow_kg_s=30)
        values.append(
            compare("Kw", result.backpressure_factor, reference.API520_W(1.5e6 + 101325, p2))
        )
        values.append(compare("area", result.area_m2, expected))
        values.append(
            compare(
                "inverse_mass",
                size_liquid_relief_valve(**args, area_m2=expected).mass_flow_kg_s,
                30,
                5e-8,
            )
        )
    elif family == "steam":
        p, t, edition = case["pressure"], case["temperature"], case["edition"]
        args = {
            "temperature_K": t,
            "upstream_pressure_Pa": p,
            "edition": edition,
            "steam_phase": "superheated",
            "superheat_table": data[edition],
        }
        expected = reference.API520_A_steam(10, t, p, edition=edition)
        values.append(
            compare(
                "superheat",
                api520_steam_superheat_factor(
                    temperature_K=t, upstream_pressure_Pa=p, edition=edition, table=data[edition]
                ),
                reference.API520_SH(t, p, edition),
            )
        )
        values.append(compare("pressure", api520_steam_pressure_factor(p), reference.API520_N(p)))
        values.append(
            compare("area", size_steam_relief_valve(**args, mass_flow_kg_s=10).area_m2, expected)
        )
        values.append(
            compare(
                "inverse_mass",
                size_steam_relief_valve(**args, area_m2=expected).mass_flow_kg_s,
                10,
                5e-8,
            )
        )
    elif family == "backpressure":
        pset, pback = 1101325, 101325 + 1e6 * case["fraction"]
        table = data["B10" if case["overpressure"] == 0.1 else "B16"]
        values.append(
            compare(
                "Kb",
                api520_gas_backpressure_factor(
                    set_pressure_Pa=pset,
                    backpressure_Pa=pback,
                    overpressure=case["overpressure"],
                    table=table,
                    table_boundary_policy="clip",
                ),
                reference.API520_B(pset, pback, case["overpressure"]),
            )
        )
        values.append(
            compare(
                "Kw",
                api520_liquid_backpressure_factor(
                    set_pressure_Pa=pset, backpressure_Pa=pback, table=data["W"]
                ),
                reference.API520_W(pset, pback),
            )
        )
    elif family == "viscosity":
        values.append(
            compare(
                "Kv",
                api520_viscosity_factor(case["reynolds"], edition=case["edition"]),
                reference.API520_Kv(case["reynolds"], case["edition"]),
            )
        )
    elif family == "noise":
        ratio, mass = case["ratio"], case["mass"]
        values.append(
            compare("API521_curve", api521_noise_curve(ratio), reference.API521_noise_graph(ratio))
        )
        values.append(
            compare(
                "API521_noise",
                api521_flare_noise(
                    mass_flow_kg_s=mass,
                    upstream_pressure_Pa=ratio * 101325,
                    ambient_pressure_Pa=101325,
                    sound_speed_m_s=350,
                    distance_m=60,
                ),
                reference.API521_noise(mass, ratio * 101325, 101325, 350, 60),
            )
        )
        for configuration, ref in (
            ("elevated", reference.VDI_3732_noise_elevated_flare),
            ("ground", reference.VDI_3732_noise_ground_flare),
        ):
            values.append(
                compare(
                    "VDI_" + configuration + "_upstream",
                    vdi3732_flare_noise(mass, configuration=configuration),
                    ref(mass),
                )
            )
        for configuration, intercept, slope in (("elevated", 112, 17), ("ground", 100, 15)):
            values.append(
                compare(
                    "VDI_" + configuration + "_unit_equation",
                    vdi3732_flare_noise(mass, configuration=configuration),
                    intercept + slope * log10(mass * 3600 / 1000),
                )
            )
    elif family == "orifice":
        for fraction in (1, 0.999):
            values.append(
                compare(
                    "round_size_" + str(fraction),
                    api526_select_orifice(fraction * case["area"])[1],
                    reference.API520_round_size(fraction * case["area"]),
                )
            )
    else:
        point = {"pressure_Pa": case["pressure"], "temperature_K": case["temperature"]}
        values.append(
            compare(
                "table_interpolation",
                data[case["edition"]].value(point),
                float(
                    bisplev(
                        case["temperature"],
                        case["pressure"],
                        getattr(reference, "API520_KSH_tck_" + case["edition"]),
                    )
                ),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("requires fluids reference 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = tables()
    records = []
    for index, case in enumerate(inputs(), 1):
        row = {"unit": index, "input": case, "measurements": []}
        try:
            run(case, row["measurements"], data)
            row["passed"] = all(m["passed"] for m in row["measurements"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
        if index % 100 == 0 or not row["passed"]:
            print(f"Relief {index}/3593; elapsed={time.perf_counter() - start:.2f}s", flush=True)
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 3593,
        "expected_comparisons": 3760,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "records"}}))
    return 0 if summary["passed_inputs"] == 3593 else 1


if __name__ == "__main__":
    raise SystemExit(main())
