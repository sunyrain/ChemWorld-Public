"""Isolated reference adapter: public pycalphad 0.11.2 equilibrium API only.

Run by the locked native driver using a separate reference interpreter.
Input is the fixed case JSON on stdin; output is JSON on stdout.
"""

import json
import sys
from importlib.metadata import version

import numpy as np
from pycalphad import Database, equilibrium
from pycalphad import variables as v

records = []
for case in json.load(sys.stdin):
    elements = sorted(case["feed"])
    conditions = {v.T: case["T"], v.P: 101325, v.N: 1}
    conditions.update({v.X(e): case["feed"][e] for e in elements[:-1]})
    try:
        result = equilibrium(Database(case["tdb"]), elements, case["phases"], conditions,
                             output=["HM"], calc_opts={"pdens": 2000})
        phase = np.asarray(result.Phase).ravel()
        amounts = np.asarray(result.NP).ravel()
        comps = list(map(str, np.asarray(result.component)))
        fractions = np.asarray(result.X).reshape(-1, len(comps))
        sites = np.asarray(result.Y).reshape(len(phase), -1)
        selected = [{"phase_name": str(name), "fraction": float(n),
                     "element_fractions": dict(zip(comps, map(float, x), strict=True)),
                     "site_fractions": [float(y) for y in s if np.isfinite(y)]}
                    for name, n, x, s in zip(phase, amounts, fractions, sites, strict=True)
                    if name and np.isfinite(n) and n > 1e-6]
        records.append({"id": case["id"], "success": True, "phases": selected,
            "G": float(np.asarray(result.GM).ravel()[0]),
            "H": float(np.asarray(result.HM).ravel()[0]),
            "mu": dict(zip(comps, map(float, np.asarray(result.MU).ravel()), strict=True))})
    except Exception as exc:
        records.append({"id": case["id"], "success": False, "error": repr(exc)})
    print(f"pycalphad reference {len(records)}/12: {case['id']}", file=sys.stderr, flush=True)
print(json.dumps({"version": version("pycalphad"), "api": "pycalphad.Database/equilibrium",
                  "pdens": 2000, "cases": records}, allow_nan=False))
