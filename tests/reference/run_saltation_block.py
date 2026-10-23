"""Fixed 146 input saltation development comparison, no runtime reference import."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi
from pathlib import Path

import fluids
import fluids.saltation as reference

from chemworld.physchem.pneumatic_conveying import SALTATION_METHODS, saltation_velocity
from chemworld.physchem.transport import FluidState


def inputs():
    for method, mass, dp, diameter in product(
        SALTATION_METHODS, (.01, .1, 1.), (.0001, .0005, .001), (.02, .05),
    ):
        yield {"method": method, "mass": mass, "dp": dp, "diameter": diameter, "terminal": 2.}
    for ratio, diameter in product((46000, 47000, 48000), (.02, .05)):
        yield {"method": "Geldart_Ling", "mass": ratio*diameter*(pi*diameter*diameter/4),
               "dp": .0005, "diameter": diameter, "terminal": 2.}
    for ratio, terminal in product((.99, 1., 1.01), (.5, 4.)):
        yield {"method": "Matsumoto_1977", "mass": .1,
               "dp": ratio*1.39*.05*(2500/1.2)**-.74, "diameter": .05, "terminal": terminal}
    for terminal, mass in product((2.99, 3., 3.01, 5.), (.01, .1)):
        yield {"method": "Weber_saltation", "mass": mass,
               "dp": .0005, "diameter": .05, "terminal": terminal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(inputs())
    assert len(cases) == 146
    rows, failures, start = [], [], time.perf_counter()
    for i, case in enumerate(cases, 1):
        try:
            method = case["method"]
            native = saltation_velocity(method, FluidState(1.2, 1.8e-5),
                solids_mass_flow_kg_s=case["mass"], pipe_diameter_m=case["diameter"],
                particle_diameter_m=case["dp"], particle_density_kg_m3=2500,
                terminal_velocity_m_s=case["terminal"])
            options = {"mp": case["mass"], "rhog": 1.2, "D": case["diameter"]}
            if method == "Geldart_Ling":
                options["mug"] = 1.8e-5
            else:
                options["dp"] = case["dp"]
                if method != "Rizk":
                    options["rhop"] = 2500
                if method not in {"Rizk", "Schade"}:
                    options["Vterminal"] = case["terminal"]
            expected = getattr(reference, method)(**options)
            row = {"input": case, "native": native.saltation_velocity_m_s,
                   "reference": expected, "branch": native.branch, "warnings": native.warnings,
                   "passed": isclose(native.saltation_velocity_m_s, expected,
                                      rel_tol=2e-11, abs_tol=1e-10)}
        except Exception as error:
            row = {"input": case, "passed": False, "exception": type(error).__name__,
                   "message": str(error)}
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
        if i % 50 == 0:
            print(f"saltation {i}/146", flush=True)
    summary = {"mode": "development", "reference_version": fluids.__version__,
               "expected_inputs": 146, "expected_comparisons": 146,
               "completed_inputs": len(rows), "passed_inputs": len(rows)-len(failures),
               "completed_comparisons": sum("native" in r for r in rows),
               "passed_comparisons": len(rows)-len(failures), "failures": failures,
               "elapsed_s": time.perf_counter()-start}
    (args.output/"rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
