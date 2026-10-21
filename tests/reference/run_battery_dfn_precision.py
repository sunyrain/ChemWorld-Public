"""Fixed three-input reference precision diagnostic; not the main qualification."""
import argparse
import json
import time
import traceback
from functools import partial

import numpy as np
from battery_dfn_pybamm import reference_dfn
from run_battery_dfn_block import units

refined = partial(reference_dfn, rtol=1e-8, atol=5e-9)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = []
start = time.monotonic()
for identity, spec, segments, _, _ in units():
    if identity not in {"D01", "D03", "D08"}:
        continue
    print(f"precision {identity}: {len(rows)}/3 complete", flush=True)
    try:
        clock = [np.linspace(0, segments[0][0], 31)]
        a = reference_dfn(spec, segments, evaluation_times=clock)[0]
        b = refined(spec, segments, evaluation_times=clock)[0]
        times = clock[0][clock[0] <= min(a[1], b[1])]
        e = spec.electrolyte
        checks = [("Voltage [V]", None, .0005), ("Current [A]", None, .0005),
                  ("Electrolyte concentration [mol.m-3]", e.mesh.centers_m, 2),
                  ("Electrolyte potential [V]", e.mesh.centers_m, .0005)]
        for name in ("negative", "positive"):
            x = e.mesh.centers_m[e.region_slices[name]]
            checks.extend([(f"{name.capitalize()} particle surface concentration", x, .0005),
                           (f"{name.capitalize()} electrode potential [V]", x, .0005),
                           (f"{name.capitalize()} electrode interfacial current density [A.m-2]",
                            x, None)])
        errors = {}
        for key, x, bound in checks:
            kwargs = {} if x is None else {"x": x}
            av, bv = (np.asarray(r[2][key](times, **kwargs)) for r in (a, b))
            if bound is None:
                bound = .1 * (.005 + .05 * float(np.max(abs(bv))))
            errors[key] = {"max_error": float(np.max(abs(av - bv))), "bound": bound}
        dt = abs(a[1] - b[1])
        rows.append({"unit": identity, "passed": bool(
            all(v["max_error"] <= v["bound"] for v in errors.values())
            and dt <= .1 * max(1e-4, .01 * b[1])), "errors": errors,
            "time_error_s": dt, "baseline_end_s": a[1], "refined_end_s": b[1]})
    except Exception:
        rows.append({"unit": identity, "passed": False, "exception": traceback.format_exc()})
    print(f"precision {identity}: passed={rows[-1]['passed']}", flush=True)
result = {"mode": "development-reference-precision", "total": 3,
          "passed": sum(r["passed"] for r in rows), "units": rows,
          "elapsed_s": time.monotonic() - start}
with open(args.output, "x") as output:
    json.dump(result, output, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
raise SystemExit(0 if result["passed"] == 3 else 1)
