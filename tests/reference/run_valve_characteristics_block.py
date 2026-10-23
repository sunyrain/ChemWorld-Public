"""Fixed 134 inputs / 164 comparisons; no vendor curve arrays are redistributed."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose
from pathlib import Path

import fluids
import fluids.control_valve as reference

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.valve_characteristics import (
    ValveOpeningCurve,
    convert_valve_coefficient,
    gas_valve_choke_pressure,
    gas_valve_choked,
    liquid_valve_choke_pressure,
    liquid_valve_choked,
    valve_cavitation_index,
)
from chemworld.physchem.valve_noise import (
    ValveAcousticGeometry,
    gas_control_valve_noise,
    liquid_control_valve_noise,
)


def curves():
    result = {"linear": ValveOpeningCurve("linear")}
    source = InputSource(
        DataSourceProvenance(
            "local-emerson-curve",
            0,
            "Emerson Valve Handbook 5E curve digitized in locally installed fluids control_valve",
            "fluids-1.3.1",
            "Vendor numeric-curve redistribution not established; local reference only",
        ),
        "literature",
    )
    for characteristic, xs, ys in (
        ("quick_opening", reference.opening_quick, reference.frac_CV_quick),
        ("equal_percentage", reference.opening_equal, reference.frac_CV_equal),
    ):
        columns = tuple(
            QuantityColumn(
                name,
                "dimensionless",
                "parameter",
                (0, 1),
                "Inherent normalized valve characteristic",
            )
            for name in ("opening", "coefficient")
        )
        rows = tuple(
            ScientificRow(
                str(i),
                "local-emerson-curve",
                characteristic,
                str(i),
                "parameters",
                {
                    "opening": InputValue(x, "dimensionless", None, "Uncertainty not supplied"),
                    "coefficient": InputValue(y, "dimensionless", None, "Uncertainty not supplied"),
                },
            )
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        )
        table = ScientificTable(characteristic, "parameters", columns, (source,), rows)
        result[characteristic] = ValveOpeningCurve(
            characteristic,
            TabulatedScalar(
                table, (("opening", "opening", "dimensionless"),), "coefficient", "dimensionless"
            ),
        )
    return result


def inputs():
    for value, from_scale, to_scale in product(
        (0.01, 1, 100), ("Kv", "Cv", "Av"), ("Kv", "Cv", "Av")
    ):
        yield {
            "family": "conversion",
            "value": value,
            "from_scale": from_scale,
            "to_scale": to_scale,
        }
    for gamma, xt in product((1.1, 1.4, 1.67), (0.3, 0.6)):
        yield {"family": "gas_choke", "gamma": gamma, "xt": xt}
    for fl, psat in product((0.5, 0.8), (1000, 10000, 100000)):
        yield {"family": "liquid_choke", "fl": fl, "psat": psat}
    for name, i in product(("linear", "quick_opening", "equal_percentage"), range(21)):
        yield {"family": "curve", "name": name, "opening": i / 20}
    for mass, p2, onset in product((10, 40), (0.95e6, 0.65e6, 0.2e6), (None, 0.3)):
        yield {"family": "liquid_noise", "m": mass, "p2": p2, "onset": onset}
    for fl, mass in product((0.6, 0.8), (0.1, 2.22)):
        gamma = 1.22
        vena = 1 - (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        critical = fl**2 * vena
        alpha = (1 - vena) / (1 - critical)
        boundary_b = 1 - gamma ** (-gamma / (gamma - 1)) / alpha
        boundary_ce = 1 - 1 / (22 * alpha)
        for regime, (left, right) in enumerate(
            pairwise((0, critical, vena, boundary_b, boundary_ce, 1)), 1
        ):
            yield {
                "family": "gas_noise",
                "fl": fl,
                "m": mass,
                "regime": regime,
                "p2": 1e6 * (1 - (left + right) / 2),
            }


def compare(name, native, expected, noise=False):
    tolerance = 2e-9 if noise else 1e-10
    return {
        "quantity": name,
        "native": native,
        "reference": expected,
        "relative_tolerance": 0 if noise else 2e-10,
        "absolute_tolerance": tolerance,
        "passed": isclose(native, expected, rel_tol=0 if noise else 2e-10, abs_tol=tolerance),
    }


def run(case, values, opening_curves):
    family = case["family"]
    if family == "conversion":
        values.append(
            compare(
                "coefficient",
                convert_valve_coefficient(
                    case["value"], from_scale=case["from_scale"], to_scale=case["to_scale"]
                ),
                reference.convert_flow_coefficient(
                    case["value"], case["from_scale"], case["to_scale"]
                ),
            )
        )
    elif family == "gas_choke":
        args = {"pressure_ratio_factor": case["xt"], "heat_capacity_ratio": case["gamma"]}
        p2 = reference.control_valve_choke_P_g(case["xt"], case["gamma"], P1=1e6)
        values.append(compare("P2", gas_valve_choke_pressure(**args, upstream_pressure_Pa=1e6), p2))
        values.append(
            compare(
                "P1",
                gas_valve_choke_pressure(**args, downstream_pressure_Pa=p2),
                reference.control_valve_choke_P_g(case["xt"], case["gamma"], P2=p2),
            )
        )
        values.append(
            compare(
                "choked",
                gas_valve_choked(**args, pressure_drop_ratio=0.5),
                reference.is_choked_turbulent_g(0.5, case["gamma"] / 1.4, case["xt"]),
            )
        )
    elif family == "liquid_choke":
        args = {
            "saturation_pressure_Pa": case["psat"],
            "critical_pressure_Pa": 22e6,
            "effective_recovery_factor": case["fl"],
        }
        p2 = reference.control_valve_choke_P_l(case["psat"], 22e6, case["fl"], P1=1e6)
        values.append(
            compare("P2", liquid_valve_choke_pressure(**args, upstream_pressure_Pa=1e6), p2)
        )
        values.append(
            compare(
                "P1",
                liquid_valve_choke_pressure(**args, downstream_pressure_Pa=p2),
                reference.control_valve_choke_P_l(case["psat"], 22e6, case["fl"], P2=p2),
            )
        )
        ff = reference.FF_critical_pressure_ratio_l(case["psat"], 22e6)
        values.append(
            compare(
                "choked",
                liquid_valve_choked(
                    pressure_drop_Pa=2e5,
                    upstream_pressure_Pa=1e6,
                    saturation_pressure_Pa=case["psat"],
                    critical_pressure_factor=ff,
                    effective_recovery_factor=case["fl"],
                ),
                reference.is_choked_turbulent_l(2e5, 1e6, case["psat"], ff, case["fl"]),
            )
        )
        values.append(
            compare(
                "cavitation_index",
                valve_cavitation_index(
                    upstream_pressure_Pa=1e6,
                    downstream_pressure_Pa=7e5,
                    saturation_pressure_Pa=case["psat"],
                ),
                reference.cavitation_index(1e6, 7e5, case["psat"]),
            )
        )
    elif family == "curve":
        values.append(
            compare(
                "normalized_Cv",
                opening_curves[case["name"]].value(case["opening"]),
                getattr(reference, "Cv_char_" + case["name"])(case["opening"]),
            )
        )
    elif family == "liquid_noise":
        geometry = ValveAcousticGeometry(
            77.848,
            0.1,
            0.1071,
            0.0036,
            0.92,
            0.42,
            "Fixed IEC hydrodynamic example",
            air_density_kg_m3=1.293,
        )
        native = liquid_control_valve_noise(
            geometry,
            mass_flow_kg_s=case["m"],
            upstream_pressure_Pa=1e6,
            downstream_pressure_Pa=case["p2"],
            saturation_pressure_Pa=2320,
            density_kg_m3=997,
            sound_speed_m_s=1400,
            cavitation_onset_ratio=case["onset"],
        )
        expected = reference.control_valve_noise_l_2015(
            m=case["m"],
            P1=1e6,
            P2=case["p2"],
            Psat=2320,
            rho=997,
            c=1400,
            Kv=77.848,
            d=0.1,
            Di=0.1071,
            FL=0.92,
            Fd=0.42,
            t_pipe=0.0036,
            rho_pipe=7800,
            c_pipe=5000,
            rho_air=1.293,
            c_air=343,
            xFz=case["onset"],
            An=-4.6,
        )
        values.append(compare("a_weighted_dB", native.a_weighted_level_dB, expected, True))
    else:
        geometry = ValveAcousticGeometry(
            77.85,
            0.1,
            0.2031,
            0.008,
            case["fl"],
            0.296,
            "Fixed IEC aerodynamic regimes",
            pipe_density_kg_m3=8000,
            air_density_kg_m3=1.293,
        )
        native = gas_control_valve_noise(
            geometry,
            mass_flow_kg_s=case["m"],
            upstream_pressure_Pa=1e6,
            downstream_pressure_Pa=case["p2"],
            inlet_temperature_K=450,
            inlet_density_kg_m3=5.3,
            heat_capacity_ratio=1.22,
            molar_mass_g_mol=19.8,
        )
        expected = reference.control_valve_noise_g_2011(
            m=case["m"],
            P1=1e6,
            P2=case["p2"],
            T1=450,
            rho=5.3,
            gamma=1.22,
            MW=19.8,
            Kv=77.85,
            d=0.1,
            Di=0.2031,
            FL=case["fl"],
            Fd=0.296,
            t_pipe=0.008,
            rho_pipe=8000,
            c_pipe=5000,
            rho_air=1.293,
            c_air=343,
            An=-3.8,
            Stp=0.2,
        )
        values.append(compare("a_weighted_dB", native.a_weighted_level_dB, expected, True))
        if native.regime != "aerodynamic_" + str(case["regime"]):
            raise ValueError("expected acoustic regime not reached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("requires fluids 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    opening_curves = curves()
    records = []
    for index, case in enumerate(inputs(), 1):
        row = {"unit": index, "input": case, "measurements": []}
        try:
            run(case, row["measurements"], opening_curves)
            row["passed"] = all(m["passed"] for m in row["measurements"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
        if index % 25 == 0 or not row["passed"]:
            print(f"Valve characteristics {index}/134", flush=True)
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 134,
        "expected_comparisons": 164,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))
    return 0 if summary["passed_inputs"] == 134 else 1


if __name__ == "__main__":
    raise SystemExit(main())
