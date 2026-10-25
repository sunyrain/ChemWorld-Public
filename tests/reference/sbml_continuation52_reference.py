"""Independent polynomial/pool references; no production RHS, AD or roots reused."""

import json
from itertools import pairwise
from math import sqrt

import numpy as np


def numeric(name, actual, expected, *, atol=1e-8, rtol=2e-7):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return {
        "name": name,
        "passed": bool(
            np.all(np.isfinite(actual)) and np.allclose(actual, expected, atol=atol, rtol=rtol)
        ),
        "actual": actual.tolist(),
        "expected": expected.tolist(),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": float(np.max(np.abs(actual - expected), initial=0)),
    }


def exact(name, conditions):
    return {"name": name, "passed": all(conditions.values()), "conditions": conditions}


def reference_checks(report, *, closed=False, scaled=False, expect_folds=True):
    rows = report["records"]
    seconds, amount = (60.0, 0.001) if scaled else (1.0, 1.0)
    initial = report["initial_coordinates"]
    p = np.asarray([r["parameter_value"] for r in rows])
    if closed:
        total = initial["N:A"] + initial["N:B"]
        k2 = initial.get("P:k2", 0.2 * seconds)
        expected_amounts = np.array([total * k2 / (p + k2), total * p / (p + k2)]).T
        actual_amounts = [[r["coordinates"]["N:A"], r["coordinates"]["N:B"]] for r in rows]
        curve = numeric("independent-curve", actual_amounts, expected_amounts)
        expected_spectra = [[-(v + k2) / seconds] for v in p]
    else:
        x = np.asarray([r["coordinates"]["N:X"] for r in rows])
        feed = p * amount / seconds * initial["N:H"] / 2
        curve = numeric("independent-curve", feed, x**3 - 3 * x**2 + 2.5 * x)
        # Each reported point must lie near an independently computed real root,
        # including the middle unstable branch that forward time integration loses.
        root_error = [
            min(abs(z.real - value) for z in np.roots([1, -3, 2.5, -mu]) if abs(z.imag) < 1e-7)
            for value, mu in zip(x, feed, strict=True)
        ]
        curve["polynomial_root_max_error"] = max(root_error)
        curve["passed"] &= bool(max(root_error) < 1e-6)
        expected_spectra = [[6 * value - 2.5 - 3 * value**2] for value in x]
        if "Y" in report["species_ids"]:
            k = initial.get("P:k", 0.7 * seconds) / seconds
            gamma = initial.get("P:gamma", 0.4 * seconds) / seconds
            curve_y = numeric(
                "independent-output-pool", [r["coordinates"]["N:Y"] for r in rows], k / gamma * x
            )
            curve["output_pool"] = curve_y
            curve["passed"] &= curve_y["passed"]
            expected_spectra = [sorted([*v, -gamma]) for v in expected_spectra]
    actual_spectra = [sorted(v[0] for v in r["eigenvalues_s_inverse"]) for r in rows]
    spectrum = numeric("independent-spectrum", actual_spectra, expected_spectra)
    spectrum["passed"] &= all(abs(v[1]) < 1e-8 for r in rows for v in r["eigenvalues_s_inverse"])
    spectrum["passed"] &= all(
        r["stability"]
        == (
            "asymptotically_stable"
            if max(v) < -1e-9
            else "unstable"
            if max(v) > 1e-9
            else "marginal_or_unresolved"
        )
        for r, v in zip(rows, expected_spectra, strict=True)
    )
    held_keys = (
        set(initial) - {"N:" + s for s in report["species_ids"]} - {"P:" + report["parameter"]}
    )
    invariants = exact(
        "conservation-and-held-coordinates",
        {
            "held_inputs": all(
                all(r["coordinates"][k] == initial[k] for k in held_keys) for r in rows
            ),
            "conserved_totals": all(r["conservation_error"] < 1e-10 for r in rows),
            "correct_rank": report["reachable_dimension"]
            == (1 if closed else len(report["species_ids"])),
            "full_rhs": all(r["full_scaled_rhs_norm"] < 1e-8 for r in rows),
            "strict_positive": all(
                r["coordinates"]["N:" + s] > 0 for r in rows for s in report["species_ids"]
            ),
        },
    )
    if not closed and expect_folds:
        known = [1 - sqrt(1.5) / 3, 1 + sqrt(1.5) / 3]
        brackets = report["fold_brackets"]
        enclosed = []
        for fold in known:
            enclosed.append(
                any(
                    min(x[a], x[b]) <= fold <= max(x[a], x[b])
                    for a, b in (item["indices"] for item in brackets)
                )
            )
        behavior = exact(
            "both-folds-and-unstable-middle",
            {
                "two_brackets": len(brackets) == 2,
                "known_fold_locations": all(enclosed),
                "stable_and_unstable": {r["stability"] for r in rows}
                == {"asymptotically_stable", "unstable"},
                "x_monotone": bool(np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)),
                "parameter_turns": bool(np.any(np.diff(p) > 0) and np.any(np.diff(p) < 0)),
            },
        )
    else:
        behavior = exact(
            "finite-continuous-branch",
            {
                "multiple_points": len(rows) > 1,
                "oriented_tangents": all(
                    np.dot(a["tangent"], b["tangent"]) > 0 for a, b in pairwise(rows)
                ),
                "tangent_nullspace": all(r["tangent_residual"] < 1e-8 for r in rows),
            },
        )
    return [
        exact(
            "requested-finite-branch",
            {
                "complete": report["status"] == "complete",
                "points": len(rows) == report["requested_points"],
            },
        ),
        curve,
        spectrum,
        invariants,
        behavior,
        exact(
            "report-json-and-budget",
            {
                "json": json.loads(json.dumps(report, allow_nan=False)) == report,
                "cost_limit": report["costs"]["variational_rate_calls"]
                <= report["policy"]["maximum_evaluations"],
                "attempt_accounting": len(report["attempts"])
                == len(rows) + report["rejected_attempts"],
                "hypothetical_only": report["world_advanced"] is False,
            },
        ),
    ]
