"""Fixed 23C six-variant equations and actual shared parameter regression."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import thermo
from activity_reference_cases import COMPOSITIONS, TEMPERATURES
from run_activity_thermodynamics_block import compare, new_record, report_values
from run_regular_solution_block import closures, expected
from thermo.unifac import UNIFAC, UNIFAC_subgroup

from chemworld.data.activity_calibration import ActivityCalibration, ActivityFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.activity_thermodynamics import (
    activity_infinite_dilution,
    activity_thermodynamic_report,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.unifac import UNIFACSubgroup, unifac_activity_model


def case(version, n):
    groups = {
        i: UNIFACSubgroup(i, main, 0.0 if version == 3 else r, q)
        for i, main, r, q in (
            (1, 1, 0.9, 0.8),
            (2, 1, 0.4, 0.0),
            (3, 2, 1.5, 1.2),
            (4, 3, 2.1, 1.7),
        )
    }
    assignments = dict(list({"A": {1: 2, 2: 1}, "B": {1: 1, 3: 1}, "C": {3: 1, 4: 2}}.items())[:n])
    interactions = {
        (i, j): (
            120.0 * i - 75.0 * j,
            0.0 if version == 0 else 0.03 * i - 0.02 * j,
            0.0 if version in {0, 5} else 0.0001 * (i - j),
        )
        for i in (1, 2, 3)
        for j in (1, 2, 3)
        if i != j
    }
    return unifac_activity_model(
        f"23C-v{version}-{n}",
        assignments,
        subgroups=groups,
        interactions=interactions,
        version=version,
        provenance={"source": "fixed authored 23C"},
    )


def upstream(spec, T, xs):
    p = spec.parameters
    groups = {
        int(key[2:]): UNIFAC_subgroup(
            int(key[2:]), key, int(p["main:" + key[2:]]), "main", p["R:" + key[2:]], value
        )
        for key, value in p.items()
        if key.startswith("Q:")
    }
    assignments = [
        {int(g): int(p[f"nu:{key}|{g}"]) for g in (str(i) for i in groups) if p[f"nu:{key}|{g}"]}
        for key in spec.component_ids
    ]
    interactions = {}
    for key, a in p.items():
        if key.startswith("psi_a:"):
            i, j = map(int, key[6:].split("|"))
            value = (
                a
                if p["version"] == 0
                else (a, p.get(f"psi_b:{i}|{j}", 0.0), p.get(f"psi_c:{i}|{j}", 0.0))
            )
            interactions.setdefault(i, {})[j] = value
    return UNIFAC.from_subgroups(
        T=T,
        xs=list(xs),
        chemgroups=assignments,
        subgroups=groups,
        interaction_data=interactions,
        version=int(p["version"]),
    )


def fit(version):
    truth = case(version, 2)
    name = "psi_a:1|2"
    seed = replace(truth, parameters={**truth.parameters, name: 10.0})
    source = InputSource(
        DataSourceProvenance(
            "23C", 0, "thermo0.6 synthetic UNIFAC", version="0.6.0", license="MIT"
        ),
        "synthetic",
    )
    columns = (
        QuantityColumn("temperature", "K", "condition", (200, 500), "absolute T"),
        *(
            QuantityColumn("x:" + k, "dimensionless", "condition", (0, 1), "mole fraction")
            for k in "AB"
        ),
        *(
            QuantityColumn(
                "ln_gamma:" + k, "dimensionless", "observation", (-100, 100), "pure liquid"
            )
            for k in "AB"
        ),
        QuantityColumn("GE", "J/mol", "observation", (-1e6, 1e6), "excess Gibbs"),
    )
    rows = []
    for split, ts, fractions in (
        ("train", (280.0, 320.0), (0.2, 0.55)),
        ("validation", (370.0,), (0.35, 0.8)),
    ):
        for T in ts:
            for x in fractions:
                ref = upstream(truth, T, (x, 1 - x))
                values = {
                    "temperature": InputValue(T, "K", 0, "exact"),
                    "x:A": InputValue(x, "dimensionless", 0, "exact"),
                    "x:B": InputValue(1 - x, "dimensionless", 0, "exact"),
                    "GE": InputValue(ref.GE(), "J/mol", 1, "declared sigma"),
                    **{
                        "ln_gamma:" + k: InputValue(v, "dimensionless", 0.01, "declared sigma")
                        for k, v in zip("AB", ref.lngammas(), strict=True)
                    },
                }
                index = str(len(rows))
                rows.append(ScientificRow(index, "23C", "g" + index, "s" + index, split, values))
    table = ScientificTable(f"23C-v{version}", "measurements", columns, (source,), tuple(rows))
    return ActivityCalibration(
        table,
        seed,
        (ActivityFitParameter(name, (-100.0, 100.0), 10.0),),
        fitted_model_id=f"fitted-unifac-{version}",
    ).fit(max_nfev=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {
        "block": "23C-UNIFAC-kernels",
        "mode": "development",
        "reference_version": thermo.__version__,
        "units": [],
        "fits": [],
    }
    for version in range(6):
        for n in (2, 3):
            spec = case(version, n)
            for T in TEMPERATURES:
                for xs in COMPOSITIONS[n]:
                    rec = new_record("state", version, n, temperature_K=T, composition=xs)
                    result["units"].append(rec)
                    try:
                        report = activity_thermodynamic_report(
                            spec, dict(zip(spec.component_ids, xs, strict=True)), temperature_K=T
                        )
                        ref = upstream(spec, T, xs)
                        target = expected(ref)
                        target["d3GE_dT3"] = ref.d3GE_dT3()
                        compare(rec, report_values(report), target)
                        checks = closures(report, xs)
                        compare(rec, checks, dict.fromkeys(checks, 0.0), absolute=True)
                    except Exception as exc:
                        rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
            for index, key in enumerate(spec.component_ids):
                rec = new_record("pure", version, n, component=key)
                result["units"].append(rec)
                try:
                    r = activity_thermodynamic_report(
                        spec, {k: float(k == key) for k in spec.component_ids}, temperature_K=320.0
                    )
                    checks = {
                        "GE": r.gibbs_excess_J_mol,
                        "HE": r.enthalpy_excess_J_mol,
                        "SE": r.entropy_excess_J_mol_K,
                        "CpE": r.heat_capacity_excess_J_mol_K,
                        "ln_gamma": r.log_activity_coefficients[key],
                    }
                    compare(rec, checks, dict.fromkeys(checks, 0.0), absolute=True)
                except Exception as exc:
                    rec["failures"].append({"exception": str(exc)})
                rec = new_record("dilution", version, n, component=key)
                result["units"].append(rec)
                try:
                    r = activity_infinite_dilution(
                        spec,
                        dict.fromkeys(spec.component_ids, 1.0),
                        temperature_K=320.0,
                        solute_id=key,
                    )
                    xs = [0.0 if i == index else 1 / (n - 1) for i in range(n)]
                    ref = upstream(spec, 320.0, xs)
                    compare(
                        rec,
                        {
                            "gamma": r.activity_coefficients[key],
                            "ln_gamma": r.log_activity_coefficients[key],
                        },
                        {"gamma": ref.gammas()[index], "ln_gamma": ref.lngammas()[index]},
                    )
                except Exception as exc:
                    rec["failures"].append({"exception": str(exc)})
            done = len(result["units"])
            elapsed = max(time.monotonic() - start, 1e-9)
            print(
                f"UNIFAC equations {done}/168; {done / elapsed:.2f}/s; "
                f"ETA {(168 - done) * elapsed / done:.1f}s",
                flush=True,
            )
        rec = {"model": version, "criteria": 0, "failures": []}
        result["fits"].append(rec)
        try:
            fitted = fit(version)
            (args.output / f"fit-{version}.json").write_text(
                json.dumps(fitted, indent=2, allow_nan=False), encoding="utf-8"
            )
            checks = {
                "completed": fitted["status"] == "fitted",
                "parameter": abs(fitted["estimate"]["psi_a:1|2"] + 30) < 1e-5,
                "train": fitted["training"]["weighted_rmse"] < 1e-7,
                "heldout": fitted["validation"]["weighted_rmse"] < 1e-7,
                "rank": fitted["identifiability"]["rank"] == 1,
                "holdout_isolation": all(r["split"] == "train" for r in fitted["jacobian_runs"])
                and all(r["split"] == "train" for r in fitted["runs"] if r["stage"] != "heldout"),
            }
            rec.update(
                criteria=len(checks), estimate=fitted["estimate"], resources=fitted["resources"]
            )
            rec["failures"].extend({"criterion": k} for k, v in checks.items() if not v)
        except Exception as exc:
            rec["failures"].append({"exception": str(exc)})
        print(f"UNIFAC fits {len(result['fits'])}/6; failures={len(rec['failures'])}", flush=True)
        (args.output / "partial.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    result["comparisons"] = sum(r["comparisons"] for r in result["units"])
    result["fit_criteria"] = sum(r["criteria"] for r in result["fits"])
    result["failures"] = [
        {"unit": i, "model": r["model"], **f}
        for i, r in enumerate([*result["units"], *result["fits"]])
        for f in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = (
        not result["failures"] and len(result["units"]) == 168 and result["fit_criteria"] == 36
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: result[k] for k in ("comparisons", "fit_criteria", "passed", "elapsed_seconds")}
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
