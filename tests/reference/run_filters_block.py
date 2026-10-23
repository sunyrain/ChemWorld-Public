"""Fixed 249 inputs/408 comparisons, local source curves only."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose
from pathlib import Path

import fluids
import fluids.filters as reference
from scipy.interpolate import splev

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.flow_filters import (
    MESH_PATTERNS,
    FilterCurve,
    filter_loss_coefficient,
    filter_pressure_drop,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState


def source_curves():
    source = InputSource(DataSourceProvenance(
        "local-blevins-filters", 0, "Blevins 1984 curves in locally installed fluids1.3.1",
        "fluids-1.3.1", "Numeric tables not redistributed; explicit local reference input"),
        "literature")
    curves, grids = {}, {}
    for name, axis, xs, ys, interpolation in (
        ("beta", "reynolds", reference.round_Res, reference.round_betas, "linear"),
        ("angle", "angle_degrees", reference.round_thetas, reference.round_gammas, "linear"),
        ("square", "open_fraction", reference.square_alphas, reference.square_Ks, "linear"),
        ("round", "open_fraction", reference.grills_rounded_alphas,
         reference.grills_rounded_Ks, "quadratic"),
    ):
        columns = tuple(QuantityColumn(key, "dimensionless", "parameter", (0, 1e20),
                                       "Explicit source interpolation coordinate/value")
                        for key in (axis, "response"))
        rows = tuple(ScientificRow(str(i), source.provenance.source_id, name, str(i), "parameters",
                     {axis: InputValue(float(x), "dimensionless", None, "Not supplied"),
                      "response": InputValue(float(y), "dimensionless", None, "Not supplied")})
                     for i, (x, y) in enumerate(zip(xs, ys, strict=True)))
        table = ScientificTable(name, "parameters", columns, (source,), rows)
        curves[name] = FilterCurve(TabulatedScalar(
            table, ((axis, axis, "dimensionless"),), "response", "dimensionless"),
            interpolation, "clip")
        grids[name] = list(xs)
    return curves, grids


def inputs(grids):
    for alpha, re, angle in product((.1, .7), (10, 20, 60, 400, 1000), (0, 45, 60, 90)):
        yield {"method": "round_screen", "alpha": alpha, "re": re, "angle": angle}
    for pattern, alpha, angle in product(MESH_PATTERNS, (.85, .95, 1), (0, 44.9, 45, 90)):
        yield {"method": "open_mesh", "alpha": alpha, "angle": angle, "pattern": pattern}
    for method, alphas in (("square_grill", (.2, .5, .8)), ("round_grill", (.3, .5, .7))):
        for alpha, ratio in product(alphas, (None, 1, 50, 75)):
            yield {"method": method, "alpha": alpha, "ratio": ratio}
    for alpha in [*grids["square"], *((a+b)/2 for a, b in pairwise(grids["square"]))]:
        yield {"method": "square_screen", "alpha": alpha}
    for name, xs in grids.items():
        for coordinate in [*xs, *((a+b)/2 for a, b in pairwise(xs))]:
            yield {"method": "curve", "name": name, "coordinate": coordinate}


def evaluate(case, curves):
    method = case["method"]
    if method == "curve":
        name, x = case["name"], case["coordinate"]
        if name == "round":
            expected = float(splev(x, reference.grills_rounded_tck))
        else:
            xs, ys = {"beta": (reference.round_Res, reference.round_betas),
                      "angle": (reference.round_thetas, reference.round_gammas),
                      "square": (reference.square_alphas, reference.square_Ks)}[name]
            expected = reference.interp(x, xs, ys)
        return [(curves[name].value(x), expected)]
    alpha, options, re = case["alpha"], {}, None
    if method in {"round_screen", "open_mesh"}:
        options.update(inclination_degrees=case["angle"], angle_curve=curves["angle"])
        re = case.get("re", 1000)
        if method == "round_screen":
            options["primary_curve"] = curves["beta"]
            expected = reference.round_edge_screen(alpha, re, case["angle"])
        else:
            options["mesh_pattern"] = case["pattern"]
            expected = reference.round_edge_open_mesh(alpha, case["pattern"], case["angle"])
    elif method == "square_screen":
        options["primary_curve"] = curves["square"]
        expected = reference.square_edge_screen(alpha)
    else:
        ratio = case["ratio"]
        if ratio is not None:
            options.update(thickness_m=.002*ratio, hydraulic_diameter_m=.002,
                           darcy_friction_factor=.0185)
        if method == "round_grill":
            options["primary_curve"] = curves["round"]
        function = (reference.round_edge_grill if method == "round_grill"
                    else reference.square_edge_grill)
        expected = function(alpha, l=.002*ratio if ratio is not None else None,
                            Dh=.002 if ratio is not None else None,
                            fd=.0185 if ratio is not None else None)
    actual = filter_loss_coefficient(method, open_fraction=alpha, reynolds=re, **options)
    pressure = filter_pressure_drop(method, FluidState(998, .001), approach_velocity_m_s=.2,
                                    open_fraction=alpha, characteristic_length_m=(
                                        re*.001/(998*.2) if re is not None else None), **options)
    return [(actual, expected), (pressure.pressure_drop_Pa, .5*998*.2**2*expected)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    curves, grids = source_curves()
    cases = list(inputs(grids))
    assert len(cases) == 249
    started, rows, failures, comparisons, passed = time.perf_counter(), [], [], 0, 0
    for i, case in enumerate(cases, 1):
        try:
            values = evaluate(case, curves)
            checks = [{"native": a, "reference": b,
                       "passed": isclose(a, b, rel_tol=2e-11, abs_tol=1e-10)} for a, b in values]
            comparisons += len(checks)
            passed += sum(row["passed"] for row in checks)
            row = {"input": case, "checks": checks, "passed": all(v["passed"] for v in checks)}
        except Exception as error:
            row = {"input": case, "passed": False,
                   "exception": type(error).__name__, "message": str(error)}
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
        if i % 50 == 0:
            print(f"filters {i}/249", flush=True)
    summary = {"mode": "development", "reference_version": fluids.__version__,
               "expected_inputs": 249, "completed_inputs": len(rows),
               "passed_inputs": len(rows)-len(failures), "expected_comparisons": 408,
               "completed_comparisons": comparisons, "passed_comparisons": passed,
               "failures": failures, "elapsed_s": time.perf_counter()-started,
               "source_table_nodes": {name: len(xs) for name, xs in grids.items()}}
    (args.output/"rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
