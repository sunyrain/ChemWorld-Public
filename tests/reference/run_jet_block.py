"""Fixed jet-pump inverse families and vacuum leakage references."""

import argparse
import inspect
import json
import re
import time
from itertools import combinations, product
from math import isclose
from pathlib import Path

import fluids
import fluids.jet_pump as ref
import mpmath as mp

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.jet_pumps import (
    JET_VARIABLE_UNITS,
    LiquidJetPumpModel,
    jet_pump_report,
    solve_jet_pump_inlet,
    solve_liquid_jet_pump,
    vacuum_air_leakage,
)
from chemworld.physchem.transport import FluidState

NAMES = dict(
    zip(JET_VARIABLE_UNITS, ("d_nozzle", "d_mixing", "Qp", "Qs", "P1", "P2", "P5"), strict=True)
)
VACUUM_TORR = (0.5, 1, 3, 5, 9.99, 10, 10.01, 19.99, 20, 89.99, 90, 99.99, 100, 700)


def tables():
    source = InputSource(
        DataSourceProvenance(
            "local-jet",
            0,
            "Installed fluids 1.3.1 HEI fits and Worthington source lookup",
            "1.3.1",
            "Local data, redistribution not asserted",
        ),
        "literature",
    )
    # Read the locally installed source fit constants, never bundle those tables.
    coefficients = re.findall(
        r"c0, c1 = ([0-9.]+), (-[0-9.]+)", inspect.getsource(ref.vacuum_air_leakage_HEI2633)
    )
    assert len(coefficients) == 5
    hei = [
        {"upper_drop": p / ref.mmHg_inv, "exponent": float(c0), "intercept": float(c1)}
        for p, (c0, c1) in zip((1, 3, 20, 90, 760), coefficients, strict=True)
    ]
    coker = [
        {
            "upper_drop": upper * ref.inchHg,
            "low": ref.vacuum_air_leakage_Coker_Worthington(
                101325 - probe * ref.inchHg, conservative=False
            ),
            "high": ref.vacuum_air_leakage_Coker_Worthington(
                101325 - probe * ref.inchHg, conservative=True
            ),
        }
        for upper, probe in ((3, 2), (5, 4), (8, 6), (30, 10))
    ]
    result = {}
    for name, values, units in (
        (
            "HEI2633",
            hei,
            {"upper_drop": "Pa", "exponent": "dimensionless", "intercept": "dimensionless"},
        ),
        ("Coker-Worthington", coker, {"upper_drop": "Pa", "low": "kg/s", "high": "kg/s"}),
    ):
        result[name] = ScientificTable(
            name,
            "parameters",
            tuple(
                QuantityColumn(k, u, "parameter", (-1e9, 1e9), "Source parameter")
                for k, u in units.items()
            ),
            (source,),
            tuple(
                ScientificRow(
                    str(i),
                    "local-jet",
                    name,
                    str(i),
                    "parameters",
                    {
                        k: InputValue(v, units[k], None, "Source published fit/lookup")
                        for k, v in row.items()
                    },
                )
                for i, row in enumerate(values)
            ),
        )
    return result


def base(density_ratio, diameter_ratio, flow_ratio, retracted):
    model = LiquidJetPumpModel(
        FluidState(1000, 0.001),
        FluidState(1000 * density_ratio, 0.001),
        0.04,
        0.11,
        0.186,
        0.12,
        retracted,
        "reference_code",
    )
    solution = ref.liquid_jet_pump(
        rhop=1000,
        rhos=1000 * density_ratio,
        Kp=0.04,
        Ks=0.11,
        Km=0.186,
        Kd=0.12,
        d_nozzle=0.045 * diameter_ratio,
        d_mixing=0.045,
        d_diffuser=0.1,
        Qp=0.002,
        Qs=0.002 * flow_ratio,
        P2=1e5,
        nozzle_retracted=retracted,
    )
    return model, {k: solution[v] for k, v in NAMES.items()}


def inputs():
    rows = []
    for parameters in product((0.8, 1.2), (0.35, 0.45, 0.55), (0.25, 0.75, 1.25), (False, True)):
        rows.append({"family": "forward", "parameters": parameters})
        for key in tuple(JET_VARIABLE_UNITS)[:6]:
            rows.append({"family": "inlet", "parameters": parameters, "missing": key})
        for keys in combinations(JET_VARIABLE_UNITS, 2):
            rows.append({"family": "inverse", "parameters": parameters, "missing": keys})
        rows.append({"family": "documented", "parameters": parameters})
    for method in ("Ryans-Croll", "Seider", "HEI2633"):
        volumes = (0.1, 0.3048**3 * 10, 10, 200, 400) if method == "HEI2633" else (0.1, 10, 1000)
        for volume, drop in product(volumes, VACUUM_TORR):
            rows.append(
                {
                    "family": "leak",
                    "method": method,
                    "volume_m3": volume,
                    "pressure_Pa": 101325 - drop / ref.torr_inv,
                    "conservative": True,
                }
            )
    for conservative, drop in product(
        (False, True), (2.9, 3, 3.1, 4.9, 5, 5.1, 7.9, 8, 8.1, 10, 20, 29)
    ):
        rows.append(
            {
                "family": "leak",
                "method": "Coker-Worthington",
                "volume_m3": None,
                "pressure_Pa": 101325 - drop * ref.inchHg,
                "conservative": conservative,
            }
        )
    assert len(rows) == 1222
    return rows


def independent_ratio(model, values, convention):
    m = mp.mpf
    dn, dm, qp, qs = (m(values[k]) for k in tuple(JET_VARIABLE_UNITS)[:4])
    r = (dn / dm) ** 2
    alpha = (dm / m(0.1)) ** 2
    ratio = qs / qp
    c = m(model.suction_fluid.density_kg_m3) / m(model.motive_fluid.density_kg_m3)
    kp, ks, km, kd = (
        m(getattr(model, key))
        for key in ("nozzle_loss", "suction_loss", "mixing_loss", "diffuser_loss")
    )
    numerator = (
        2 * r
        + 2 * c * ratio**2 * r**2 / (1 - r)
        - r**2 * (1 + c * ratio) * (1 + ratio) * (1 + km + kd + alpha**2)
    )
    numerator -= c * ratio**2 * r**2 * (1 + ks) / (1 - r) ** 2
    denominator = 1 + kp - 2 * r - 2 * c * ratio**2 * r**2 / (1 - r)
    denominator += r**2 * (1 + c * ratio) * (1 + ratio) * (1 + km + kd + alpha**2)
    if model.nozzle_retracted:
        denominator += (
            c * ratio**2 * r**2 * (1 + (ks if convention == "documented" else -ks)) / (1 - r) ** 2
        )
    return numerator / denominator


def independent_leak(row, table):
    m = mp.mpf
    drop = m(101325) - m(row["pressure_Pa"])
    torr = drop * 760 / 101325
    volume = None if row["volume_m3"] is None else m(row["volume_m3"]) / m("0.3048") ** 3
    method = row["method"]
    if method == "Ryans-Croll":
        coefficient = (
            m("0.026") * torr ** m("0.34")
            if torr < 10
            else (m("0.032") * torr ** m("0.26") if torr < 100 else m("0.106"))
        )
        return coefficient * volume ** m("0.6") * m("0.45359237") / 3600
    if method == "Seider":
        x = mp.log(torr)
        return (
            (5 + (m("0.0289") + m("0.03088") * x - m("0.0005733") * x * x) * volume ** m("0.66"))
            * m("0.45359237")
            / 3600
        )
    selected = next(
        (r for r in table.rows if drop <= m(r.values["upper_drop"].value)), table.rows[-1]
    )
    if method == "Coker-Worthington":
        return m(selected.values["high" if row["conservative"] else "low"].value)
    volume = max(m(10), volume)
    return (
        mp.exp(
            m(selected.values["intercept"].value)
            + m(selected.values["exponent"].value) * mp.log(volume)
        )
        * m("0.45359237")
        / 3600
    )


def evaluate(row, source_tables):
    family = row["family"]
    if family == "leak":
        method = row["method"]
        native = vacuum_air_leakage(
            method,
            pressure_Pa=row["pressure_Pa"],
            volume_m3=row["volume_m3"],
            parameters=source_tables.get(method),
            conservative=row["conservative"],
            outside="reference",
        )["mass_flow_kg_s"]
        fn = getattr(
            ref,
            "vacuum_air_leakage_"
            + {
                "Ryans-Croll": "Ryans_Croll",
                "Coker-Worthington": "Coker_Worthington",
            }.get(method, method),
        )
        value = (
            fn(row["pressure_Pa"], conservative=row["conservative"])
            if method == "Coker-Worthington"
            else fn(row["volume_m3"], row["pressure_Pa"])
        )
        return [
            (native, value, "source"),
            (native, float(independent_leak(row, source_tables.get(method))), "independent80"),
        ]
    model, values = base(*row["parameters"])
    if family == "forward":
        args = [values[k] for k in tuple(JET_VARIABLE_UNITS)[:4]]
        p1 = values["suction_pressure_Pa"] + model.inlet_pressure_difference(*args)
        num, den = model.pressure_ratio_terms(*args, diffuser_diameter_m=0.1)
        p5 = (num * p1 + den * values["suction_pressure_Pa"]) / (num + den)
        result = jet_pump_report(
            model,
            {**values, "motive_pressure_Pa": p1, "discharge_pressure_Pa": p5},
            diffuser_diameter_m=0.1,
        )
        source_ratio = (values["discharge_pressure_Pa"] - values["suction_pressure_Pa"]) / (
            values["motive_pressure_Pa"] - values["discharge_pressure_Pa"]
        )
        return [
            (p1, values["motive_pressure_Pa"], "source"),
            (p5, values["discharge_pressure_Pa"], "source"),
            (result["efficiency"], row["parameters"][2] * source_ratio, "source"),
        ]
    if family == "inlet":
        key = row["missing"]
        inlet = {
            k: (None if k == key else v) for k, v in values.items() if k != "discharge_pressure_Pa"
        }
        native = solve_jet_pump_inlet(model, **inlet)
        source = ref.liquid_jet_pump_ancillary(
            rhop=model.motive_fluid.density_kg_m3,
            rhos=model.suction_fluid.density_kg_m3,
            Kp=model.nozzle_loss,
            Ks=model.suction_loss,
            **{NAMES[k]: v for k, v in inlet.items()},
        )
        return [(native, values[key], "inverse"), (native, source, "inverse")]
    if family == "inverse":
        missing = row["missing"]
        result = solve_liquid_jet_pump(
            model,
            {k: v for k, v in values.items() if k not in missing},
            initial={
                k: values[k] * factor for k, factor in zip(missing, (0.97, 1.03), strict=True)
            },
            bounds={k: (values[k] * 0.75, values[k] * 1.25) for k in missing},
            diffuser_diameter_m=0.1,
        )
        return [(result[k], values[k], "inverse") for k in missing] + [
            (result["inlet_residual_Pa"] / values["motive_pressure_Pa"], 0, "residual"),
            (result["pressure_ratio_residual"], 0, "residual"),
            (
                result["outlet_mass_flow_kg_s"],
                1000 * result["motive_flow_m3_s"]
                + model.suction_fluid.density_kg_m3 * result["suction_flow_m3_s"],
                "source",
            ),
            (
                result["efficiency"],
                values["suction_flow_m3_s"]
                / values["motive_flow_m3_s"]
                * (values["discharge_pressure_Pa"] - values["suction_pressure_Pa"])
                / (values["motive_pressure_Pa"] - values["discharge_pressure_Pa"]),
                "inverse",
            ),
        ]
    if family == "documented":
        documented = LiquidJetPumpModel.from_dict(
            {**model.to_dict(), "retracted_loss_convention": "documented"}
        )
        args = [values[k] for k in tuple(JET_VARIABLE_UNITS)[:4]]
        num, den = documented.pressure_ratio_terms(*args, diffuser_diameter_m=0.1)
        source_num, source_den = model.pressure_ratio_terms(*args, diffuser_diameter_m=0.1)
        ratio = independent_ratio(model, values, "documented")
        reference = independent_ratio(model, values, "reference_code")
        return [
            (num / den, float(ratio), "independent80"),
            (num / den - source_num / source_den, float(ratio - reference), "independent80"),
        ]
    raise AssertionError(family)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 80
    started = time.perf_counter()
    rows, source_tables = inputs(), tables()
    results, counts = [], {}
    passed = completed = 0
    for index, row in enumerate(rows):
        result = {"input": row, "comparisons": []}
        try:
            for native, reference, kind in evaluate(row, source_tables):
                rtol, atol = (2e-7, 1e-10) if kind == "inverse" else (3e-11, 1e-12)
                if kind == "residual":
                    rtol, atol = 0, 2e-10
                good = isclose(native, reference, rel_tol=rtol, abs_tol=atol)
                result["comparisons"].append(
                    {"kind": kind, "native": native, "reference": reference, "passed": good}
                )
                completed += 1
                passed += good
                count = counts.setdefault(kind, {"completed": 0, "passed": 0})
                count["completed"] += 1
                count["passed"] += good
        except Exception as error:
            result["exception"] = f"{type(error).__name__}: {error}"
        results.append(result)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(rows) - index - 1) / (index + 1)
            print(
                f"jet {index + 1}/{len(rows)}, {passed}/{completed} checks, "
                f"{(index + 1) / elapsed:.1f} inputs/s, ETA {eta:.1f}s",
                flush=True,
            )
    failures = [
        r for r in results if "exception" in r or not all(c["passed"] for c in r["comparisons"])
    ]
    report = {
        "reference_version": fluids.__version__,
        "mode": "development",
        "inputs": len(rows),
        "expected_comparisons": 5504,
        "completed_comparisons": completed,
        "passed": passed,
        "failed": completed - passed,
        "exceptions": sum("exception" in r for r in results),
        "counts": counts,
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "failures")}))
    return 0 if not failures and completed == 5504 else 1


if __name__ == "__main__":
    raise SystemExit(main())
