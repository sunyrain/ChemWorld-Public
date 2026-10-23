"""Fixed 213-input same-source packed-bed development block."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
import fluids.packed_bed as reference

from chemworld.physchem.packed_beds import (
    PACKED_BED_METHODS,
    PackedBedSpec,
    available_packed_bed_methods,
    packed_bed_pressure,
    packed_bed_voidage,
)
from chemworld.physchem.transport import FluidState


def inputs():
    for method, velocity, voidage, phi in product(
        PACKED_BED_METHODS, (.002, .02, .2), (.38, .45), (.8, 1.),
    ):
        ratio = (2.5 if method == "Guo_Sun" else 10.)
        tube = .001*phi*ratio if method in {
            "Guo_Sun", "Harrison_Brunner_Hecker", "Montillet_Akkari_Comiti"} else None
        yield {"kind": "pressure", "method": method, "velocity": velocity,
               "voidage": voidage, "phi": phi, "tube": tube}
    for method, ratio in (("Harrison_Brunner_Hecker", None),
                          ("Montillet_Akkari_Comiti", None),
                          ("Montillet_Akkari_Comiti", 50.),
                          ("Montillet_Akkari_Comiti", 51.)):
        for voidage in (.38, .45):
            yield {"kind": "pressure", "method": method, "velocity": .02,
                   "voidage": voidage, "phi": 1.,
                   "tube": None if ratio is None else ratio*.001}
    for method in PACKED_BED_METHODS:
        yield {"kind": "zero", "method": method, "velocity": 0., "voidage": .4,
               "phi": 1., "tube": .0025 if method == "Guo_Sun" else None}
    for shape, phi in (("general", .5), ("general", .75), ("general", 1.),
                       ("cylindrical", .5), ("cylindrical", .75), ("cylindrical", 1.),
                       ("spherical", None)):
        for ratio in (2., 10., 25.):
            yield {"kind": "voidage", "shape": shape, "phi": phi, "ratio": ratio}
    for tube in (None, .01):
        yield {"kind": "availability", "tube": tube}


def evaluate(case):
    if case["kind"] in {"pressure", "zero"}:
        spec = PackedBedSpec(case["method"], .2, .001, case["voidage"], case["phi"], case["tube"])
        native = packed_bed_pressure(spec, FluidState(998, .001),
                                     superficial_velocity_m_s=case["velocity"])
        kwargs = {"dp": .001*case["phi"], "voidage": case["voidage"],
                  "vs": case["velocity"], "rho": 998., "mu": .001, "L": .2}
        if case["method"] in {"Guo_Sun", "Harrison_Brunner_Hecker", "Montillet_Akkari_Comiti"}:
            kwargs["Dt"] = case["tube"]
        expected = (0. if case["kind"] == "zero" else getattr(reference, case["method"])(**kwargs))
        return native.pressure_drop_Pa, expected, native.range_notes
    if case["kind"] == "voidage":
        shape = case["shape"]
        phi = case["phi"] if shape == "general" else None
        native = packed_bed_voidage(particle_diameter_m=.001, tube_diameter_m=case["ratio"]*.001,
                                    particle_shape=shape, sphericity=phi)
        if shape == "general":
            expected = reference.voidage_Benyahia_Oneil(.001, case["ratio"]*.001, case["phi"])
        elif shape == "cylindrical":
            expected = reference.voidage_Benyahia_Oneil_cylindrical(
                .001, case["ratio"]*.001, case["phi"])
        else:
            expected = reference.voidage_Benyahia_Oneil_spherical(.001, case["ratio"]*.001)
        return native, expected, ()
    return len(available_packed_bed_methods(tube_diameter_m=case["tube"])), (
        13 if case["tube"] is None else 14), ()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(inputs())
    assert len(cases) == 213
    start, rows, failures = time.monotonic(), [], []
    for i, case in enumerate(cases, 1):
        try:
            native, expected, notes = evaluate(case)
            passed = isclose(native, expected, rel_tol=2e-11, abs_tol=1e-8)
            row = {"input": case, "native": native, "reference": expected,
                   "range_notes": notes, "passed": passed}
        except Exception as error:
            row = {"input": case, "passed": False,
                   "exception": type(error).__name__, "message": str(error)}
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
        if i % 50 == 0:
            print(f"packed-bed {i}/{len(cases)}", flush=True)
    summary = {"mode": "development", "reference_version": fluids.__version__,
               "expected_inputs": 213, "completed_inputs": len(rows),
               "passed_inputs": len(rows)-len(failures), "expected_comparisons": 213,
               "completed_comparisons": sum("native" in row for row in rows),
               "passed_comparisons": len(rows)-len(failures), "failures": failures,
               "elapsed_s": time.monotonic()-start}
    (args.output/"rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
