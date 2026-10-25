"""Fixed independent default-quadrature and multicomponent GenericSAFT blocks."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from functools import cache
from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np
import teqp
from mie_reference_options_dense import mixture_expression
from run_dufal import CASES, PARAMS
from run_dufal import models as dufal_models
from run_dufal_high_precision import Oracle
from run_generic_saft import ORDERS, source_specification
from run_saftvrmie import PARAMETERS

from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.generic_saft_import import generic_saft_from_teqp
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL


def grid(block):
    if block == "original-independent":
        return [
            {"base": b, "association": a, "case": 0, "t": t, "rho": r, "original": True}
            for b, a, (t, r) in product(
                ("PCSAFT", "SAFTVRMie", "softSAFT", "2CLJ"),
                ("none", "CS", "KG", "Dufal"),
                ((450.0, 10.0), (600.0, 1000.0)),
            )
        ]
    rows = [
        {"base": b, "association": a, "case": c, "t": t, "rho": r, "original": False}
        for b, c, a, t, r in product(
            ("PCSAFT", "SAFTVRMie", "softSAFT"),
            range(9),
            ("none", "CS", "KG", "Dufal"),
            (450.0, 600.0),
            (10.0, 1000.0),
        )
    ]
    rows += [
        {
            "base": "2CLJ",
            "association": a,
            "case": 0,
            "t": t,
            "rho": r,
            "original": False,
            "author": author,
            "elongation": ell,
        }
        for author, ell, a, t, r in product(
            ("2CLJF_Mecke", "2CLJF_Lisal"),
            (0.0, 0.3),
            ("none", "CS", "KG", "Dufal"),
            (450.0, 600.0),
            (10.0, 1000.0),
        )
    ]
    return rows


def specification(row):
    if row["original"]:
        return source_specification(row["base"], row["association"]), ("A",)
    ids, _ = CASES[row["case"]]
    params = [PARAMETERS[i] for i in ids]
    kind = row["base"]
    model = {
        "coeffs": [
            {
                "name": p[0],
                "m": p[1],
                "sigma_m": p[2],
                "sigma_Angstrom": p[2] * 1e10,
                "epsilon_over_k": p[3],
                "lambda_r": p[4],
                "lambda_a": 6.0,
                "BibTeXKey": "Fixed generic completion",
            }
            for p in params
        ],
        "kmat": [[0.02 if i != j else 0.0 for j in ids] for i in ids],
    }
    if kind == "softSAFT":
        model = {
            "m": [p[1] for p in params],
            "sigma / m": [p[2] for p in params],
            "epsilon/kB / K": [p[3] for p in params],
        }
    elif kind == "2CLJ":
        model = {"author": row["author"], "L^*": row["elongation"]}
    spec = {"kind": "genericSAFT", "model": {"nonpolar": {"kind": kind, "model": model}}}
    association = row["association"]
    if association in ("CS", "KG"):
        spec["model"]["association"] = {
            "kind": "canonical",
            "model": {
                "Delta_rule": "CR1",
                "b / m^3/mol": [(3e-5, 4e-5, 5e-5)[i] for i in ids],
                "beta": [(0.01, 0.015, 0.02)[i] for i in ids],
                "epsilon / J/mol": [(1000.0, 1200.0, 900.0)[i] * R for i in ids],
                "molecule_sites": [
                    (["e", "H"], ["e", "H", "H"], ["e", "e", "H", "H"])[i] for i in ids
                ],
            },
        }
    elif association == "Dufal":
        ps = [PARAMS[i] for i in ids]
        spec["model"]["association"] = {
            "kind": "Dufal",
            "model": {
                "Delta_rule": "Dufal",
                "sigma / m": [p[2] for p in ps],
                "epsilon/kB / K": [p[3] for p in ps],
                "lambda_r": [p[4] for p in ps],
                "kmat": [[0.0] * len(ids) for _ in ids],
                "epsilon_HB / J/mol": [p[5] * R for p in ps],
                "K_HB / m^3": [p[6] for p in ps],
                "molecule_sites": [list(p[7]) for p in ps],
                "options": {"rtol": 1e-13, "atol": 1e-13, "max_iters": 1000},
            },
        }
    return spec, tuple(p[0] for p in params)


def mp_targets(expression, t, rho, fractions):
    t, rho = mp.mpf(t), mp.mpf(rho)
    x = tuple(mp.mpf(str(v)) for v in fractions)
    expected = {
        f"Ar{i}{j}": float(
            mp.diff(
                lambda a, b: expression(t / (1 + a), rho * (1 + b), x),
                (mp.mpf(0), mp.mpf(0)),
                (i, j),
            )
        )
        for i, j in ORDERS
    }
    for i in range(len(x)):

        def extensive(z, index=i):
            amounts = [v + z if k == index else v for k, v in enumerate(x)]
            total = mp.fsum(amounts)
            return total * expression(t, rho * total, [v / total for v in amounts])

        expected[f"mu_{i}"] = float(mp.mpf(str(R)) * t * mp.diff(extensive, mp.mpf(0)))
    return expected


def canonical_expression(spec, contact):
    b, beta, eps = [
        [mp.mpf(str(v)) for v in spec[key]] for key in ("b / m^3/mol", "beta", "epsilon / J/mol")
    ]
    sites = [(i, label) for i, labels in enumerate(spec["molecule_sites"]) for label in labels]

    def expression(t, rho, x):
        eta = rho * mp.fsum(xx * bb for xx, bb in zip(x, b, strict=True)) / 4
        g = (1 - eta / 2) / (1 - eta) ** 3 if contact == "CS" else 1 / (1 - mp.mpf("1.9") * eta)
        delta = [
            [
                rho
                * x[j]
                * g
                * (b[i] + b[j])
                / 2
                * mp.sqrt(beta[i] * beta[j])
                * mp.expm1((eps[i] + eps[j]) / (2 * mp.mpf(str(R)) * t))
                if a != z
                else mp.mpf(0)
                for j, z in sites
            ]
            for i, a in sites
        ]
        n = len(sites)

        def residual(*v):
            return tuple(
                v[i] * (1 + mp.fsum(delta[i][j] * v[j] for j in range(n))) - 1 for i in range(n)
            )

        guess = tuple(2 / (1 + mp.sqrt(1 + 4 * mp.fsum(d))) for d in delta)
        root = mp.findroot(residual, guess, solver="mdnewton", tol=mp.eps * 100, maxsteps=80)
        return mp.fsum(
            x[i] * (mp.log(root[k]) - root[k] / 2 + mp.mpf(".5")) for k, (i, _) in enumerate(sites)
        )

    return expression


@cache
def base_targets(spec_json, case, t, rho, original):
    spec = json.loads(spec_json)
    kind = spec["kind"]
    x = (1.0,) if original else CASES[case][1]
    if kind == "SAFTVRMie":
        parameters = (("A", 1.0, 3.7e-10, 180.0, 12.0),) if original else PARAMETERS
        cases = [((0,), (1.0,))] if original else CASES
        f = mixture_expression(
            case, t, cases=cases, parameters=parameters, kij=0.0 if original else 0.02
        )
        return mp_targets(f, t, rho, x)
    ref = teqp.make_model({"kind": "genericSAFT", "model": {"nonpolar": spec}}, validate=False)
    tt, rr = (t / 180.0, rho * AVOGADRO_PER_MOL * (3.7e-10) ** 3) if kind == "2CLJ" else (t, rho)
    xx = np.asarray(x, dtype=float)
    result = {f"Ar{i}{j}": ref.get_Arxy(i, j, tt, rr, xx) for i, j in ORDERS}
    mu = (
        [R * t * (result["Ar00"] + result["Ar01"])]
        if kind == "2CLJ"
        else ref.build_Psir_gradient_autodiff(t, rho * xx)
    )
    result.update({f"mu_{i}": float(v) for i, v in enumerate(mu)})
    return result


@cache
def association_targets(spec_json, association, case, t, rho, original, coefficient_source=None):
    x = (1.0,) if original else CASES[case][1]
    if association == "none":
        return {**{f"Ar{i}{j}": 0.0 for i, j in ORDERS}, **{f"mu_{i}": 0.0 for i in range(len(x))}}
    if association in ("CS", "KG"):
        return mp_targets(canonical_expression(json.loads(spec_json), association), t, rho, x)
    if coefficient_source is not None:
        oracle = Oracle(coefficient_source, case, "molecular")
        return mp_targets(oracle.alpha, t, rho, x)
    _, ref, _ = dufal_models(case, "teqp")
    xx = np.asarray(x, dtype=float)
    return {
        **{f"Ar{i}{j}": ref.get_Arxy(i, j, t, rho, xx) for i, j in ORDERS},
        **{f"mu_{i}": float(v) for i, v in enumerate(ref.chemical_potentials(t, rho, xx))},
    }


def run(block, output, coefficient_source=None):
    output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 80
    cases = grid(block)
    assert len(cases) == (32 if block == "original-independent" else 496)
    failures, count, metadata = [], 0, {}
    started = time.perf_counter()
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, row in enumerate(cases, 1):
            t, rho, case = row["t"], row["rho"], row["case"]
            spec, ids = specification(row)
            key = json.dumps(spec, sort_keys=True)
            imported = generic_saft_from_teqp(
                copy.deepcopy(spec),
                component_ids=ids,
                temperature_range_K=(100.0, 1000.0),
                source="Fixed completion inputs",
                canonical_radial_distribution="Kontogeorgis"
                if row["association"] == "KG"
                else "Carnahan-Starling",
                two_center_scales=(3.7e-10, 180.0),
            )
            native = imported.model
            metadata[key] = {
                "specification": spec,
                "component_ids": ids,
                "import_notes": imported.notes,
            }
            x = (1.0,) if row["original"] else CASES[case][1]
            actual, expected, errors = {}, {}, {}
            try:
                base = base_targets(
                    json.dumps(spec["model"]["nonpolar"], sort_keys=True),
                    case,
                    t,
                    rho,
                    row["original"],
                )
                association = association_targets(
                    json.dumps(
                        spec["model"].get("association", {}).get("model", {}), sort_keys=True
                    ),
                    row["association"],
                    case,
                    t,
                    rho,
                    row["original"],
                    coefficient_source,
                )
                expected = {q: value + association[q] for q, value in base.items()}
                expected["density_inverse"] = rho
            except Exception as exc:
                errors["reference_exception"] = f"{type(exc).__name__}: {exc}"
            try:
                actual = {
                    f"Ar{i}{j}": molecular_scaled_derivative(
                        native, t, rho, x, inverse_temperature_order=i, density_order=j
                    )
                    for i, j in ORDERS
                }
                actual.update(
                    {
                        f"mu_{i}": v
                        for i, v in enumerate(
                            molecular_residual_chemical_potentials(native, t, rho, x)
                        )
                    }
                )
                if "Ar01" in expected:
                    p = rho * R * t * (1 + expected["Ar01"])
                    actual["density_inverse"] = molecular_density_from_pressure(
                        native,
                        t,
                        p,
                        x,
                        initial_density_mol_m3=0.95 * rho,
                        density_bounds_mol_m3=(0.1 * rho, 2 * rho),
                    ).molar_density_mol_m3
            except Exception as exc:
                errors["native_exception"] = f"{type(exc).__name__}: {exc}"
            comparisons = []
            for q in (
                *(f"Ar{i}{j}" for i, j in ORDERS),
                *(f"mu_{i}" for i in range(len(x))),
                "density_inverse",
            ):
                a, b = actual.get(q), expected.get(q)
                atol = 2e-5 if q.startswith("mu_") else 2e-7 if q == "density_inverse" else 2e-9
                a = float(a) if a is not None and math.isfinite(a) else None
                b = float(b) if b is not None and math.isfinite(b) else None
                passed = (
                    a is not None
                    and b is not None
                    and math.isclose(a, b, rel_tol=2e-7, abs_tol=atol)
                )
                record = dict(
                    quantity=q, native=a, reference=b, atol=atol, rtol=2e-7, passed=passed, **errors
                )
                comparisons.append(record)
                count += 1
                if not passed:
                    failures.append(dict(unit=unit, **row, **record))
            stream.write(json.dumps(dict(unit=unit, **row, comparisons=comparisons)) + "\n")
            stream.flush()
            elapsed = time.perf_counter() - started
            print(
                f"Generic {block} {unit}/{len(cases)}, {unit / elapsed:.2f} units/s, "
                f"ETA {(len(cases) - unit) * elapsed / unit:.1f}s, failures={len(failures)}",
                flush=True,
            )
    expected_count = 384 if block == "original-independent" else 6336
    summary = {
        "mode": "development",
        "block": block,
        "units": len(cases),
        "comparisons": count,
        "expected_comparisons": expected_count,
        "passed": count - len(failures),
        "failed": len(failures),
        "all_failures": failures,
        "inputs": list(metadata.values()),
        "seconds": time.perf_counter() - started,
        "decimal_precision": 80,
        "dufal_reference": "independent source-table polynomial and mass-action root"
        if coefficient_source
        else "teqp0.23.2 isolated association",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert count == expected_count
    print(json.dumps({k: v for k, v in summary.items() if k not in ("all_failures", "inputs")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("original-independent", "mixtures"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient-source", type=Path)
    args = parser.parse_args()
    run(args.block, args.output, args.coefficient_source)
