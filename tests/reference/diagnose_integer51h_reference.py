"""All saved generators: independent exponential action and ODE diagnostics."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from run_sde51e_statistics import progress, write_json
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.input / "summary.json").read_text(encoding="utf-8"))
    rows, started = [], time.monotonic()
    for index, unit in enumerate(summary["units"], 1):
        row = {"id": unit["id"], "passed": False}
        try:
            data = json.loads(
                (args.input / unit["id"] / "reference/law.json").read_text(encoding="utf-8")
            )
            q = csr_matrix(np.array(data["generator"]).T)
            origin = np.zeros(q.shape[0])
            origin[0] = 1
            action = expm_multiply(q * data["duration_s"], origin)
            ode = solve_ivp(
                lambda _t, p, q=q: q @ p,
                (0, data["duration_s"]),
                origin,
                method="DOP853",
                rtol=2e-13,
                atol=1e-15,
            )
            expected, dense = np.array(data["probability"]), np.array(data["expm_probability"])
            ode_result = ode.y[:, -1]
            action_error = float(max(abs(action - expected)))
            ode_error = float(max(abs(ode_result - expected)))
            row.update(
                scipy=scipy.__version__,
                state_count=len(expected),
                action_probability=action.tolist(),
                ode_probability=ode_result.tolist(),
                action_vs_uniformization=action_error,
                ode_vs_uniformization=ode_error,
                action_vs_dense=float(max(abs(action - dense))),
                ode_success=bool(ode.success),
                ode_final_time=float(ode.t[-1]),
                ode_rhs_evaluations=ode.nfev,
                action_mass_residual=abs(float(sum(action)) - 1),
                ode_mass_residual=abs(float(sum(ode_result)) - 1),
                passed=bool(ode.success and action_error <= 1e-11 and ode_error <= 1e-11),
            )
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        write_json(args.output / (unit["id"] + ".json"), row)
        rows.append(row)
        progress("reference-diagnostic", index, 14, started, unit=unit["id"], passed=row["passed"])
    result = {
        "schema": "chemworld-integer51h-reference-diagnostic-v1",
        "input": str(args.input),
        "units": rows,
        "elapsed_s": time.monotonic() - started,
        "passed": len(rows) == 14 and all(r["passed"] for r in rows),
    }
    write_json(args.output / "summary.json", result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
