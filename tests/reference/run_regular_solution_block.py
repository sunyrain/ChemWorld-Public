"""Fixed 23B regular-solution equations, all Hansen sources and real regression."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import chemicals
import thermo
from activity_reference_cases import COMPOSITIONS, TEMPERATURES
from regular_solution_high_precision import expected as high_precision_expected
from run_activity_thermodynamics_block import compare, new_record, report_values
from thermo.regular_solution import FloryHuggins, Hansen, RegularSolution

from chemworld.data.activity_calibration import ActivityCalibration, ActivityFitParameter
from chemworld.data.constants import ConstantDatabase
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
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.regular_solution import hansen_activity_model, regular_solution_model
from chemworld.physchem.solubility import HANSEN_SOURCES, HansenParameters

MODELS = ("regular_solution", "flory_huggins", "hansen")


def model_case(model, n):
    ids = tuple("ABC"[:n])
    volumes = {key: (80 + 30 * i) * 1e-6 for i, key in enumerate(ids)}
    if model != "hansen":
        return regular_solution_model(
            "23B-" + model,
            ids,
            model=model,
            molar_volumes_m3_mol=volumes,
            solubility_parameters_Pa_half={key: 16000 + 5000 * i for i, key in enumerate(ids)},
            interaction_parameters={
                (left, right): 0.02 + 0.01 * i + 0.02 * j
                for i, left in enumerate(ids)
                for j, right in enumerate(ids)
                if i != j
            },
        )
    provenance = DataSourceProvenance(
        "23B-synthetic", 0, "Authored fixed parameter vectors", version="1", license="MIT"
    )
    vectors = tuple(
        HansenParameters(
            key,
            *(
                InputValue(value, "Pa^0.5", None, "exact input")
                for value in (15000 + 2000 * i, 5000 + 3000 * i, 4000 + 4000 * i)
            ),
            provenance,
            "row-" + key,
        )
        for i, key in enumerate(ids)
    )
    return hansen_activity_model(
        "23B-hansen",
        vectors,
        molar_volumes={
            key: InputValue(v, "m^3/mol", None, "authored fixed volume")
            for key, v in volumes.items()
        },
        alpha=0.6,
        volume_reference="fixed synthetic reference volume",
    )


def upstream(spec, T, xs):
    ids, p = spec.component_ids, spec.parameters
    kwargs = {"T": T, "xs": list(xs), "Vs": [p[f"V:{key}"] for key in ids]}
    if spec.model == "hansen":
        kwargs.update(
            {
                name: [p[f"{name}:{key}"] for key in ids]
                for name in ("delta_d", "delta_p", "delta_h")
            },
            alpha=p["alpha"],
        )
    else:
        kwargs.update(
            SPs=[p[f"delta:{key}"] for key in ids],
            lambda_coeffs=[
                [0.0 if left == right else p[f"lambda:{left}|{right}"] for right in ids]
                for left in ids
            ],
        )
    return {"regular_solution": RegularSolution, "flory_huggins": FloryHuggins, "hansen": Hansen}[
        spec.model
    ](**kwargs)


def expected(ref):
    gammas = ref.gammas()
    return {
        "GE": ref.GE(),
        "HE": ref.HE(),
        "SE": ref.SE(),
        "CpE": ref.CpE(),
        "dGE_dT": ref.dGE_dT(),
        "d2GE_dT2": ref.d2GE_dT2(),
        "d3GE_dT3": 0.0,
        "dGE_dxs": ref.dGE_dxs(),
        "d2GE_dxixjs": ref.d2GE_dxixjs(),
        "d2GE_dTdxs": ref.d2GE_dTdxs(),
        "gamma": gammas,
        "ln_gamma": ref.lngammas(),
        "dln_gamma_dT": ref.dlngammas_dT(),
        "dln_gamma_dns": [[v / gammas[i] for v in row] for i, row in enumerate(ref.dgammas_dns())],
        "partial_GE": ref.dnGE_dns(),
        "partial_HE": ref.dnHE_dns(),
        "partial_SE": [-v for v in ref.d2nGE_dTdns()],
    }


def closures(r, xs):
    ids, mat, T = r.component_ids, r.dln_gamma_dns, r.temperature_K
    n = len(ids)
    return {
        "partial_GE": sum(
            x * r.partial_molar_gibbs_excess_J_mol[k] for x, k in zip(xs, ids, strict=True)
        )
        - r.gibbs_excess_J_mol,
        "partial_HE": sum(
            x * r.partial_molar_enthalpy_excess_J_mol[k] for x, k in zip(xs, ids, strict=True)
        )
        - r.enthalpy_excess_J_mol,
        "partial_SE": sum(
            x * r.partial_molar_entropy_excess_J_mol_K[k] for x, k in zip(xs, ids, strict=True)
        )
        - r.entropy_excess_J_mol_K,
        "gibbs_duhem": max(abs(sum(xs[i] * mat[i][j] for i in range(n))) for j in range(n)),
        "symmetry": max(abs(mat[i][j] - mat[j][i]) for i in range(n) for j in range(n)),
        "thermal_gibbs_duhem": sum(x * r.dln_gamma_dT[k] for x, k in zip(xs, ids, strict=True))
        + r.enthalpy_excess_J_mol / (8.31446261815324 * T * T),
    }


def fit_inputs(model, deficient):
    truth = model_case(model, 2)
    names = (
        ("alpha",)
        if model == "hansen"
        else (("lambda:A|B", "lambda:B|A") if deficient else ("lambda:A|B",))
    )
    initials = {
        name: truth.parameters[name] + (0.2 if name == "alpha" else 0.025 + 0.01 * i)
        for i, name in enumerate(names)
    }
    parameters = tuple(
        ActivityFitParameter(name, (0.0, 1.0) if name == "alpha" else (-0.2, 0.2), value)
        for name, value in initials.items()
    )
    initial = replace(
        truth, model_id=truth.model_id + "-seed", parameters={**truth.parameters, **initials}
    )
    source = InputSource(
        DataSourceProvenance(
            "23B-reference",
            0,
            "thermo 0.6.0 authored synthetic responses",
            version="0.6.0",
            license="MIT upstream software; authored synthetic points",
        ),
        "synthetic",
    )
    columns = (
        QuantityColumn("temperature", "K", "condition", (200, 500), "absolute temperature"),
        *(
            QuantityColumn("x:" + key, "dimensionless", "condition", (0, 1), "mole fraction")
            for key in "AB"
        ),
        *(
            QuantityColumn(
                "ln_gamma:" + key,
                "dimensionless",
                "observation",
                (-100, 100),
                "pure liquid reference",
            )
            for key in "AB"
        ),
    )
    rows = []
    for split, Ts, fractions in (
        ("train", (280.0, 340.0), (0.15, 0.35, 0.65, 0.85)),
        ("validation", (310.0, 370.0), (0.25, 0.75)),
    ):
        for T in Ts:
            for x in fractions:
                index = len(rows)
                logs = upstream(truth, T, (x, 1 - x)).lngammas()
                values = {
                    "temperature": InputValue(T, "K", 0, "exact synthetic condition"),
                    "x:A": InputValue(x, "dimensionless", 0, "exact synthetic condition"),
                    "x:B": InputValue(1 - x, "dimensionless", 0, "exact synthetic condition"),
                    **{
                        "ln_gamma:" + key: InputValue(
                            value, "dimensionless", 0.01, "declared sigma"
                        )
                        for key, value in zip("AB", logs, strict=True)
                    },
                }
                rows.append(
                    ScientificRow(
                        str(index), "23B-reference", f"g{index}", f"s{index}", split, values
                    )
                )
    return (
        truth,
        ScientificTable("23B-" + model, "measurements", columns, (source,), tuple(rows)),
        initial,
        parameters,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=("thermo", "high_precision"), default="thermo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {
        "block": "native-conv23-regular-solutions",
        "evidence_mode": "development",
        "thermo_version": thermo.__version__,
        "chemicals_version": chemicals.__version__,
        "state_reference": args.reference,
        "expected_scientific_comparisons": 34614,
        "expected_fit_criteria": 23,
        "units": [],
        "source_units": [],
        "fits": [],
    }
    for model in MODELS:
        for n in (2, 3):
            spec = model_case(model, n)
            for T in TEMPERATURES:
                for xs in COMPOSITIONS[n]:
                    rec = new_record("state", model, n, temperature_K=T, composition=list(xs))
                    result["units"].append(rec)
                    try:
                        r = activity_thermodynamic_report(
                            spec, dict(zip(spec.component_ids, xs, strict=True)), temperature_K=T
                        )
                        reference = (
                            high_precision_expected(spec, T, xs)
                            if args.reference == "high_precision"
                            else expected(upstream(spec, T, xs))
                        )
                        compare(rec, report_values(r), reference)
                        checks = closures(r, xs)
                        compare(rec, checks, dict.fromkeys(checks, 0.0), absolute=True)
                    except Exception as exc:
                        rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
            for index, key in enumerate(spec.component_ids):
                xs = [float(i == index) for i in range(n)]
                rec = new_record("pure_endpoint", model, n, component=key)
                result["units"].append(rec)
                try:
                    r = activity_thermodynamic_report(
                        spec, dict(zip(spec.component_ids, xs, strict=True)), temperature_K=320.0
                    )
                    tests = {
                        "GE": r.gibbs_excess_J_mol,
                        "HE": r.enthalpy_excess_J_mol,
                        "SE": r.entropy_excess_J_mol_K,
                        "CpE": r.heat_capacity_excess_J_mol_K,
                        "pure_ln_gamma": r.log_activity_coefficients[key],
                        "nonfinite_gradient": float(any(not math.isfinite(v) for v in r.dGE_dxs)),
                        "nonfinite_hessian": float(
                            any(not math.isfinite(v) for row in r.d2GE_dxixjs for v in row)
                        ),
                        "nonfinite_gamma": float(
                            any(not math.isfinite(v) for v in r.activity_coefficients.values())
                        ),
                    }
                    compare(rec, tests, dict.fromkeys(tests, 0.0), absolute=True)
                except Exception as exc:
                    rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
                rec = new_record("infinite_dilution", model, n, component=key)
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
                    rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
            print(
                f"Regular equation units {model}/{n} complete; {len(result['units'])}/84 units",
                flush=True,
            )
    database = ConstantDatabase(Path(chemicals.__file__).parent)
    solvent = HansenParameters(
        "synthetic-solvent",
        *(InputValue(v, "Pa^0.5", None, "explicit synthetic") for v in (15000.0, 8000.0, 5000.0)),
        DataSourceProvenance(
            "synthetic-solvent",
            0,
            "Fixed synthetic solvent, not a material qualification",
            version="1",
            license="MIT",
        ),
        "synthetic",
    )
    for source in HANSEN_SOURCES:
        for record in database.records(source):
            rec = new_record(
                "hansen_source", "hansen", 2, source_id=source, record_id=record.record_id
            )
            result["source_units"].append(rec)
            try:
                vector = HansenParameters.from_record(record)
                model = hansen_activity_model(
                    "source-consumer",
                    (vector, solvent),
                    alpha=0.6,
                    molar_volumes={
                        vector.component_id: InputValue(
                            8e-5, "m^3/mol", None, "synthetic fixed volume"
                        ),
                        solvent.component_id: InputValue(
                            1.1e-4, "m^3/mol", None, "synthetic fixed volume"
                        ),
                    },
                    volume_reference="authored development constants; not material densities",
                )
                payload = json.loads(json.dumps(model.to_dict()))
                restored = ActivityModelSpec(**payload)
                r = activity_thermodynamic_report(
                    restored,
                    {vector.component_id: 0.2, solvent.component_id: 0.8},
                    temperature_K=298.15,
                )
                ref = upstream(restored, 298.15, (0.2, 0.8))
                compare(
                    rec,
                    {
                        "GE": r.gibbs_excess_J_mol,
                        "HE": r.enthalpy_excess_J_mol,
                        "SE": r.entropy_excess_J_mol_K,
                        "CpE": r.heat_capacity_excess_J_mol_K,
                        "dGE_dT": r.dGE_dT,
                        "d2GE_dT2": r.d2GE_dT2,
                        "d3GE_dT3": r.d3GE_dT3,
                        "gamma": list(r.activity_coefficients.values()),
                    },
                    {
                        "GE": ref.GE(),
                        "HE": ref.HE(),
                        "SE": ref.SE(),
                        "CpE": ref.CpE(),
                        "dGE_dT": ref.dGE_dT(),
                        "d2GE_dT2": ref.d2GE_dT2(),
                        "d3GE_dT3": 0.0,
                        "gamma": ref.gammas(),
                    },
                )
                compare(
                    rec,
                    {
                        "source_values": [
                            restored.parameters[f"{p}:{vector.component_id}"]
                            for p in ("delta_d", "delta_p", "delta_h")
                        ],
                        "source_roundtrip": float(
                            restored.provenance["hansen_sources"][0] == vector.to_dict()
                            and restored.provenance == model.provenance
                        ),
                    },
                    {"source_values": list(vector.values_pa_half()), "source_roundtrip": 1.0},
                )
            except Exception as exc:
                rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
            done = len(result["source_units"])
            if done % 100 == 0:
                elapsed = max(time.monotonic() - start, 1e-9)
                print(
                    f"Hansen sources {done}/2460; {done / elapsed:.2f}/s; "
                    f"ETA {(2460 - done) * elapsed / done:.1f}s",
                    flush=True,
                )
                (args.output / "partial.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
                )
    for model, deficient in (
        ("regular_solution", False),
        ("flory_huggins", False),
        ("hansen", False),
        ("regular_solution", True),
        ("flory_huggins", True),
    ):
        rec = {"model": model, "rank_deficient_design": deficient, "criteria": 0, "failures": []}
        result["fits"].append(rec)
        try:
            truth, table, initial, parameters = fit_inputs(model, deficient)
            fitted = ActivityCalibration(
                table, initial, parameters, fitted_model_id="fitted-" + model
            ).fit(max_nfev=100)
            (args.output / f"fit-{model}-{deficient}.json").write_text(
                json.dumps(fitted, indent=2, allow_nan=False), encoding="utf-8"
            )
            rec.update(
                status=fitted["status"],
                estimate=fitted.get("estimate"),
                resources=fitted["resources"],
            )
            if fitted["status"] != "fitted":
                raise ValueError(f"fit not complete: {fitted['failures']}")
            rec.update(
                training_weighted_rmse=fitted["training"]["weighted_rmse"],
                validation_weighted_rmse=fitted["validation"]["weighted_rmse"],
                rank=fitted["identifiability"]["rank"],
            )
            if deficient:
                checks = {
                    "completed": fitted["status"] == "fitted",
                    "rank": fitted["identifiability"]["rank"] == 1,
                    "predictions": max(
                        fitted["training"]["weighted_rmse"], fitted["validation"]["weighted_rmse"]
                    )
                    < 1e-6,
                    "symmetric_sum": math.isclose(
                        sum(fitted["estimate"].values()),
                        sum(truth.parameters[p.name] for p in parameters),
                        rel_tol=1e-6,
                        abs_tol=1e-5,
                    ),
                }
            else:
                checks = {
                    "parameters": all(
                        math.isclose(
                            fitted["estimate"][p.name],
                            truth.parameters[p.name],
                            rel_tol=1e-6,
                            abs_tol=1e-5,
                        )
                        for p in parameters
                    ),
                    "training": fitted["training"]["weighted_rmse"] < 1e-6,
                    "validation": fitted["validation"]["weighted_rmse"] < 1e-6,
                    "rank": fitted["identifiability"]["rank"] == 1,
                    "holdout_isolation": all(r["split"] == "train" for r in fitted["jacobian_runs"])
                    and all(
                        r["split"] == "train" for r in fitted["runs"] if r["stage"] != "heldout"
                    ),
                }
            rec["criteria"] = len(checks)
            rec["failures"].extend({"criterion": k} for k, value in checks.items() if not value)
        except Exception as exc:
            rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
        print(
            f"Regular fits {len(result['fits'])}/5 {model} deficient={deficient}: "
            f"{len(rec['failures'])} failures",
            flush=True,
        )
    result["scientific_comparisons"] = sum(
        r["comparisons"] for r in [*result["units"], *result["source_units"]]
    )
    result["fit_criteria"] = sum(r["criteria"] for r in result["fits"])
    result["failures"] = [
        {"unit_index": i, "model": r["model"], **failure}
        for i, r in enumerate([*result["units"], *result["source_units"], *result["fits"]])
        for failure in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = (
        not result["failures"]
        and result["scientific_comparisons"] == 34614
        and result["fit_criteria"] == 23
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("scientific_comparisons", "fit_criteria", "passed", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
