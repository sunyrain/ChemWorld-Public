"""Fixed development reference block; source import stays outside production."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose, sqrt
from pathlib import Path

import fluids
from fluids import fittings as reference
from fluids.numerics import interp

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.valve_characteristics import convert_valve_coefficient
from chemworld.physchem.valve_fittings import (
    CRANE_VALVE_TYPES,
    VALVE_LIFT_TYPES,
    FittingCoefficientCatalog,
    change_loss_velocity_basis,
    crane_valve_coefficient,
    darby_resistance,
    junction_loss_coefficients,
    minimum_valve_lift_velocity,
    nominal_size_from_bore,
    resistance_from_valve_coefficient,
    valve_coefficient_from_resistance,
)


def table(identity, columns, records):
    source = InputSource(
        DataSourceProvenance(
            "fluids",
            0,
            "Public fitting reference parameters",
            fluids.__version__,
            "Public source; development comparison",
        ),
        "literature",
    )
    return ScientificTable(
        identity,
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (0, 1e8), "Published fitting parameter")
            for k, u in columns
        ),
        (source,),
        tuple(
            ScientificRow(
                str(i),
                "fluids",
                identity,
                name,
                "parameters",
                {
                    k: InputValue(float(v), u, None, "Published fitting parameter")
                    for (k, u), v in zip(columns, values, strict=True)
                },
            )
            for i, (name, values) in enumerate(records)
        ),
    )


def source_tables():
    darby = FittingCoefficientCatalog(
        table("darby", [(k, "dimensionless") for k in ("k1", "ki", "kd")], reference.Darby.items()),
        model="Darby",
    )
    hooper = FittingCoefficientCatalog(
        table(
            "hooper", [(k, "dimensionless") for k in ("k1", "k_infinity")], reference.Hooper.items()
        ),
        model="Hooper",
    )
    nominal = table(
        "nominal",
        [("inner_diameter", "m"), ("nominal_size", "dimensionless")],
        [
            (str(i), (d * 0.001, n))
            for i, (d, n) in enumerate(zip(reference.S40i, reference.NPS40, strict=True))
        ],
    )
    return {
        "Darby": FittingCoefficientCatalog.from_dict(json.loads(json.dumps(darby.to_dict()))),
        "Hooper": FittingCoefficientCatalog.from_dict(json.loads(json.dumps(hooper.to_dict()))),
        "nominal": nominal,
    }


def inputs():
    for variant, d, beta, angle, fd in product(
        range(28), (0.05, 0.2286, 0.381, 0.5), (0.6, 1), (0, 30, 45, 60, 90), (None, 0.015)
    ):
        yield {
            "family": "crane",
            "variant": variant,
            "D": d,
            "beta": beta,
            "angle": angle,
            "fd": fd,
        }
    for variant, rho, beta in product(range(15), (1.2, 1000), (0.6, 1)):
        yield {"family": "lift", "variant": variant, "rho": rho, "beta": beta}
    for mode, beta, q, angle in product(
        ("converging", "diverging"),
        (0.4, sqrt(0.35), 2 / 3, sqrt(2 / 3), 1),
        (0.1, 0.4, 0.5, 0.6, 0.9),
        (30, 45, 59.9, 60, 74.9, 75, 90),
    ):
        yield {"family": "junction", "mode": mode, "beta": beta, "q": q, "angle": angle}
    for model, records in (("Darby", reference.Darby), ("Hooper", reference.Hooper)):
        for name, size, re in product(records, (0.5, 2, 8), (100, 1e4, 1e6)):
            yield {"family": model, "name": name, "size": size, "Re": re}
        for name in records:
            yield {"family": "cells", "model": model, "name": name}
    for d, value in product((0.005, 0.05, 0.5), (0.1, 2, 200)):
        yield {"family": "conversion", "D": d, "value": value}
    diameters = [d * 0.001 for d in reference.S40i]
    for d in (
        *diameters,
        *((a + b) / 2 for a, b in pairwise(diameters)),
        0.8 * diameters[0],
        1.2 * diameters[-1],
    ):
        yield {"family": "nominal", "D": d}


def compare(quantity, native, expected, *, exact=False):
    rtol, atol = (0, 0) if exact else (3e-8, 1e-10)
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "rtol": rtol,
        "atol": atol,
        "passed": isclose(native, expected, rel_tol=rtol, abs_tol=atol),
    }


def evaluate(case, tables):
    family = case["family"]
    if family == "crane":
        kind = case["variant"]
        native = crane_valve_coefficient(
            CRANE_VALVE_TYPES[kind],
            pipe_diameter_m=case["D"],
            throat_diameter_m=case["D"] * case["beta"],
            angle_degrees=case["angle"],
            friction_factor=case["fd"],
        )
        expected = reference.Crane_loss_coefficient(
            case["D"] * case["beta"],
            case["D"],
            case["angle"],
            reference.CRANE_VALVES[kind],
            case["fd"],
        )
        return [compare("K", native, expected)]
    if family == "lift":
        style = VALVE_LIFT_TYPES[case["variant"]]
        source_style = (
            style.replace("tilting check 5", "tilting check 5\u00b0")
            .replace("tilting check 15", "tilting check 15\u00b0")
            .replace("foot poppet", "foot valve poppet disc")
            .replace("foot hinged", "foot valve hinged disc")
        )
        return [
            compare(
                "velocity",
                minimum_valve_lift_velocity(
                    style, density_kg_m3=case["rho"], throat_to_pipe_ratio=case["beta"]
                ),
                reference.v_lift_valve_Crane(
                    case["rho"], D1=case["beta"] * 0.1, D2=0.1, style=source_style
                ),
            )
        ]
    if family == "junction":
        run, branch = 0.01 * (1 - case["q"]), 0.01 * case["q"]
        native = junction_loss_coefficients(
            case["mode"],
            pipe_diameter_m=0.1,
            branch_diameter_m=0.1 * case["beta"],
            run_flow_m3_s=run,
            branch_flow_m3_s=branch,
            angle_degrees=case["angle"],
        )
        common = {
            "D_run": 0.1,
            "D_branch": 0.1 * case["beta"],
            "Q_run": run,
            "Q_branch": branch,
            "angle": case["angle"],
        }
        return [
            compare(
                part,
                getattr(native, part),
                getattr(reference, f"K_{part}_{case['mode']}_Crane")(**common),
            )
            for part in ("branch", "run")
        ]
    if family in ("Darby", "Hooper"):
        native = tables[family].resistance(
            case["name"],
            reynolds=case["Re"],
            size=case["size"] if family == "Darby" else case["size"] * 0.0254,
        )
        expected = (
            reference.Darby3K(NPS=case["size"], Re=case["Re"], name=case["name"])
            if family == "Darby"
            else reference.Hooper2K(case["size"], case["Re"], name=case["name"])
        )
        return [compare("K", native, expected)]
    if family == "cells":
        values = tables[case["model"]].parameters(case["name"])
        expected = getattr(reference, case["model"])[case["name"]]
        return [
            compare(k, v, e, exact=True) for (k, v), e in zip(values.items(), expected, strict=True)
        ]
    if family == "conversion":
        value, d = case["value"], case["D"]
        pairs = [
            (
                "Kv_to_K",
                resistance_from_valve_coefficient(value, diameter_m=d),
                reference.Kv_to_K(value, d),
            ),
            (
                "K_to_Kv",
                valve_coefficient_from_resistance(value, diameter_m=d),
                reference.K_to_Kv(value, d),
            ),
            (
                "Cv_to_K",
                resistance_from_valve_coefficient(value, diameter_m=d, scale="Cv"),
                reference.Cv_to_K(value, d),
            ),
            (
                "K_to_Cv",
                valve_coefficient_from_resistance(value, diameter_m=d, scale="Cv"),
                reference.K_to_Cv(value, d),
            ),
            (
                "Kv_to_Cv",
                convert_valve_coefficient(value, from_scale="Kv", to_scale="Cv"),
                reference.Kv_to_Cv(value),
            ),
            (
                "Cv_to_Kv",
                convert_valve_coefficient(value, from_scale="Cv", to_scale="Kv"),
                reference.Cv_to_Kv(value),
            ),
            (
                "basis",
                change_loss_velocity_basis(value, from_diameter_m=d, to_diameter_m=2 * d),
                reference.change_K_basis(value, d, 2 * d),
            ),
        ]
        return [compare(*pair) for pair in pairs]
    native = nominal_size_from_bore(tables["nominal"], case["D"], extrapolate=True)
    expected = interp(case["D"] * 1000, reference.S40i, reference.NPS40, extrapolate=True)
    values = next(iter(reference.Darby.values()))
    return [
        compare("nominal", native, expected),
        compare(
            "Darby_Di",
            darby_resistance(1e5, native, k1=values[0], ki=values[1], kd=values[2]),
            reference.Darby3K(Di=case["D"], Re=1e5, K1=values[0], Ki=values[1], Kd=values[2]),
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve all attempts")
    cases = list(inputs())
    assert len(cases) == 3372, len(cases)
    tables = source_tables()
    started = time.perf_counter()
    records = []
    for index, case in enumerate(cases, 1):
        try:
            records.append({"input": case, "comparisons": evaluate(case, tables)})
        except Exception as error:
            records.append({"input": case, "exception": f"{type(error).__name__}: {error}"})
        if index % 500 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"fitting valves {index}/{len(cases)} {index / elapsed:.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [c for r in records for c in r.get("comparisons", [])]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": len(cases),
        "expected_comparisons": 3929,
        "completed_comparisons": len(comparisons),
        "passed": sum(c["passed"] for c in comparisons),
        "failed": sum(not c["passed"] for c in comparisons),
        "exceptions": sum("exception" in r for r in records),
        "failures": [
            r
            for r in records
            if "exception" in r or any(not c["passed"] for c in r.get("comparisons", []))
        ],
        "records": records,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
