"""Actual pycalphad 0.11.2 reference; zero elements are absent, not dilute stock."""

import json
import sys
from importlib.metadata import version
from time import monotonic

import numpy as np
from pycalphad import Database, equilibrium
from pycalphad import variables as v


def main():
    if version("pycalphad") != "0.11.2":
        raise ValueError("reference requires pycalphad 0.11.2")
    cases = json.load(sys.stdin)
    records = []
    start = monotonic()
    for case in cases:
        elements = sorted(e for e, x in case["feed"].items() if x > 0)
        conditions = {v.T: case["T"], v.P: case["P"], v.N: 1}
        conditions.update({v.X(e): case["feed"][e] for e in elements[:-1]})
        try:
            result = equilibrium(
                Database(case["tdb"]),
                elements,
                case["phases"],
                conditions,
                output=["HM"],
                calc_opts={"pdens": 2000},
            )
            phase = np.asarray(result.Phase).ravel()
            amounts = np.asarray(result.NP).ravel()
            comps = list(map(str, np.asarray(result.component)))
            fractions = np.asarray(result.X).reshape(-1, len(comps))
            selected = [
                {
                    "phase_name": str(name),
                    "fraction": float(n),
                    "element_fractions": dict(zip(comps, map(float, x), strict=True)),
                }
                for name, n, x in zip(phase, amounts, fractions, strict=True)
                if name and np.isfinite(n) and n > 1e-6
            ]
            if not selected:
                raise ValueError("reference returned no positive stable phase")
            record = {
                "id": case["id"],
                "success": True,
                "active_elements": elements,
                "phases": selected,
                "G": float(np.asarray(result.GM).ravel()[0]),
                "H": float(np.asarray(result.HM).ravel()[0]),
                "mu": dict(zip(comps, map(float, np.asarray(result.MU).ravel()), strict=True)),
                "absent_element_mu": None,
            }
            json.dumps(record, allow_nan=False)
        except Exception as exc:
            record = {"id": case["id"], "success": False, "error": repr(exc)}
        records.append(record)
        elapsed = max(monotonic() - start, 1e-9)
        eta = (len(cases) - len(records)) * elapsed / len(records)
        print(
            f"pycalphad {len(records)}/{len(cases)} {case['id']} "
            f"rate={len(records) / elapsed:.3f}/s ETA={eta:.1f}s "
            f"success={record['success']}",
            file=sys.stderr,
            flush=True,
        )
    print(
        json.dumps(
            {
                "version": version("pycalphad"),
                "api": "pycalphad.Database/equilibrium",
                "numpy_version": version("numpy"),
                "scipy_version": version("scipy"),
                "pdens": 2000,
                "cases": records,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
