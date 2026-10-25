"""Independent closed-form/Brent references for the frozen parent28 S block.

No native physics imports. Synthetic same-equation interface coverage, not
measured material data or a coupled precipitation-extent model.
"""

from itertools import pairwise, product
from math import exp, fsum, log

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

R, TREF, PREF = 8.3145, 300.0, 101325.0
SHIFTS = {"FE": (1000.0, 2.0, 1e-6), "NI": (-250.0, 0.5, 2e-6), "CR": (500.0, -0.2, 0.0)}


def cases():
    rows = []
    for t, factor, xs in product(
        (250.0, 300.0, 500.0), (0.5, 1.0, 2.0), ((0.2, 0.8), (0.2, 0.3, 0.5))
    ):
        rows.append(
            {
                "id": f"S-IDEAL-{len(rows) + 1:02d}",
                "kind": "ideal",
                "T": t,
                "P": factor * PREF,
                "amounts": dict(zip("abc"[: len(xs)], xs, strict=True)),
                "shifted": True,
            }
        )
    for t, factor in product((250.0, 500.0), (0.5, 2.0)):
        rows.append(
            {
                "id": f"S-ATOM-{t}-{factor}",
                "kind": "atom",
                "T": t,
                "P": factor * PREF,
                "amounts": {"fe2": 3.0},
                "shifted": True,
            }
        )
    for coordinate in ("T", "P"):
        rows.append(
            {
                "id": f"S-PATH-{coordinate}",
                "kind": "path",
                "coordinate": coordinate,
                "T": 300.0,
                "P": PREF,
                "amounts": {"a": 1.0},
            }
        )
    for n in (0.0, 1e-8):
        rows.append(
            {
                "id": f"S-SUPPORT-{n}",
                "kind": "support",
                "T": 300.0,
                "P": PREF,
                "amounts": {"a": 1 - n, "b": n},
            }
        )
    for n in (1.0, 0.5):
        rows.append(
            {
                "id": f"S-REACTION-{n}",
                "kind": "reaction",
                "T": 300.0,
                "P": 3 * PREF,
                "amounts": {"a": 1.0, "b": 1.0, "ab": n},
                "compatible": n == 1.0,
            }
        )
    for xs in ((0.2, 0.8), (0.2, 0.3, 0.5)):
        rows.append(
            {
                "id": f"S-CONSUMER-{len(xs)}",
                "kind": "consumer",
                "T": 300.0,
                "P": PREF,
                "amounts": dict(zip("abc"[: len(xs)], xs, strict=True)),
            }
        )
    for x in (0.3, 0.5):
        rows.append(
            {
                "id": f"S-NONIDEAL-{x}",
                "kind": "nonideal",
                "T": 300.0,
                "P": PREF,
                "amounts": {"a": x, "b": 1 - x},
                "omega": 8000.0,
            }
        )
    assert len(rows) == 32
    return rows


def mapping(case):
    rows = {
        "a": {"FE": 1.0},
        "b": {"NI": 1.0},
        "c": {"CR": 1.0},
        "ab": {"FE": 1.0, "NI": 1.0},
        "fe2": {"FE": 2.0},
    }
    return {k: rows[k] for k in case["amounts"]}


def reference(case):
    t, p, amounts, atoms = case["T"], case["P"], case["amounts"], mapping(case)
    total, rt = fsum(amounts.values()), R * t
    molecular = {
        k: rt * (log(n / total * p / PREF) + 2.5 * log(TREF / t))
        for k, n in amounts.items()
        if n > 0
    }
    stock = {
        e: fsum(n * atoms[k].get(e, 0.0) for k, n in amounts.items())
        for e in sorted({e for row in atoms.values() for e in row})
    }
    if case["kind"] == "reaction":
        target = {"FE": 0.0, "NI": 0.0} if case["compatible"] else None
    else:
        target = {
            e: molecular[k] / count
            for k, row in atoms.items()
            for e, count in row.items()
            if amounts[k] > 0
        }
    if target is None:
        return {
            "molecular": molecular,
            "stock": stock,
            "potentials": None,
            "affinity": molecular["ab"] - molecular["a"] - molecular["b"],
        }
    if case.get("shifted"):
        target = {
            e: value + SHIFTS[e][0] - t * SHIFTS[e][1] + SHIFTS[e][2] * (p - PREF)
            for e, value in target.items()
        }
    elements = tuple(target)
    lse = float(logsumexp([target[e] / rt for e in elements]))
    minimum = -rt * lse
    candidates = [{e: exp(target[e] / rt - lse) for e in elements}]
    source_roots = []
    if case["kind"] == "nonideal":
        omega = case["omega"]

        def derivative(x):
            return rt * log(x / (1 - x)) + omega * (1 - 2 * x) - target["FE"] + target["NI"]

        def distance(x):
            return (
                rt * (x * log(x) + (1 - x) * log(1 - x))
                + omega * x * (1 - x)
                - x * target["FE"]
                - (1 - x) * target["NI"]
            )

        grid = np.linspace(1e-12, 1 - 1e-12, 1001)
        roots = [float(grid[0]), float(grid[-1])]
        for a, b in pairwise(grid):
            da, db = derivative(a), derivative(b)
            if da == 0:
                roots.append(float(a))
            if da * db < 0:
                roots.append(float(brentq(derivative, a, b, xtol=1e-14)))
        source_roots = [{"x_FE": x, "distance": distance(x)} for x in roots]
        minimum = min(r["distance"] for r in source_roots)
        candidates = [
            {"FE": r["x_FE"], "NI": 1 - r["x_FE"]}
            for r in source_roots
            if abs(r["distance"] - minimum) < 1e-8
        ]
    return {
        "molecular": molecular,
        "stock": stock,
        "potentials": target,
        "minimum": minimum,
        "candidate_compositions": candidates,
        "source_roots": source_roots,
    }
