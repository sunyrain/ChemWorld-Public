"""Fixed 28-unit independent analytic development qualification for 52A."""

from __future__ import annotations

import argparse
import json
import time
from math import isclose
from pathlib import Path

import numpy as np
from biochemical_analysis52_cases import analytic_case, build_model, fixed_cases, frame_case

from chemworld.physchem.biochemical_analysis import analyze_biochemical_steady_state
from chemworld.runtime.sbml_analysis import analyze_sbml_world_state


class Record:
    def __init__(self, unit):
        self.unit, self.checks, self.evidence = unit, [], {}

    def check(self, label, actual, expected, *, noise=False, numeric=False):
        actual = actual.item() if hasattr(actual, "item") else actual
        expected = expected.item() if hasattr(expected, "item") else expected
        passed = bool(
            isclose(
                actual, expected, rel_tol=3e-7 if noise else 2e-7, abs_tol=1e-32 if noise else 2e-10
            )
            if numeric or noise
            else actual == expected
        )
        self.checks.append(
            {"label": label, "actual": actual, "expected": expected, "passed": passed}
        )

    def array(self, label, actual, expected, *, noise=False):
        a, e = np.asarray(actual), np.asarray(expected)
        self.check(label + "/shape", list(a.shape), list(e.shape))
        if a.shape != e.shape:
            return
        for index in np.ndindex(a.shape):
            self.check(
                label + "/" + ",".join(str(i) for i in index),
                a[index],
                e[index],
                numeric=True,
                noise=noise,
            )


def verify_stable(record, report, kind, parameters):
    expected = analytic_case(kind, **parameters)
    record.evidence.update({"kind": kind, "parameters": parameters, "analysis": report})
    record.check("steady_status", report["status"], "steady_state_found")
    record.check("stability", report.get("stability"), "asymptotically_stable")
    if report["status"] != "steady_state_found":
        return
    record.array(
        "steady_amounts_mol", list(report["steady_amounts_mol"].values()), expected["amounts"]
    )
    record.array("jacobian_s_inv", report["jacobian_s_inv"], expected["jacobian"])
    record.array(
        "reduced_eigenvalues_s_inv",
        [z["real"] for z in report["reduced_eigenvalues_s_inv"]],
        expected["eigenvalues"],
    )
    record.array(
        "reduced_eigenvalue_imaginary",
        [z["imag"] for z in report["reduced_eigenvalues_s_inv"]],
        [0] * len(expected["eigenvalues"]),
    )
    record.array(
        "relaxation_times_s",
        report["relaxation_times_s"],
        [-1 / v for v in expected["eigenvalues"]],
    )
    record.check(
        "removed_conservation_modes",
        report["removed_conservation_modes"],
        1 if kind == "closed" else 0,
    )
    record.array(
        "steady_residual_mol_s",
        report["steady_rate_residual_mol_s"],
        np.zeros(len(expected["amounts"])),
    )
    record.array(
        "same_original_rhs_mol_s",
        report["stoichiometric_rhs_residual_mol_s"],
        np.zeros(len(expected["amounts"])),
    )
    control = report["control"]
    record.check("control_available", control["status"], "available")
    if control["status"] == "available":
        for name in ("scaled_amount_control", "scaled_flux_control"):
            record.array(name, control[name], expected[name])
        cs, cj = (
            np.asarray(control["scaled_amount_control"]),
            np.asarray(control["scaled_flux_control"]),
        )
        record.array("amount_control_summation", cs.sum(axis=1), np.zeros(cs.shape[0]))
        record.array("flux_control_summation", cj.sum(axis=1), np.ones(cj.shape[0]))
        if kind != "closed":
            elasticity = np.asarray(report["scaled_flux_elasticity"])
            record.array("unconserved_connectivity", cs @ elasticity, -np.eye(cs.shape[0]))
        else:
            record.array(
                "fixed_total_control",
                np.sum(control["amount_response_mol"], axis=0),
                np.zeros(cs.shape[1]),
            )
    noise = report["linear_noise"]
    record.check("noise_available", noise["status"], "available")
    if noise["status"] == "available":
        record.array(
            "covariance_mol2", noise["covariance_mol2"], expected["covariance"], noise=True
        )
        record.check("covariance_symmetry", noise["symmetry_error_mol2"], 0, noise=True)
        record.check(
            "covariance_positive_semidefinite",
            noise["minimum_covariance_eigenvalue_mol2"] >= -1e-32,
            True,
        )
        record.check(
            "lyapunov_residual_scaled_to_noise", noise["lyapunov_relative_residual"] <= 3e-7, True
        )
        conservation = np.asarray(noise["conservation_covariance_residual_mol2"])
        record.array(
            "conservation_covariance_mol2", conservation, np.zeros_like(conservation), noise=True
        )
    record.check(
        "unknown_chemistry_preserved",
        report["chemical_composition_scope"],
        "unknown_formal_entities",
    )
    record.check("unknown_energy_preserved", report["energy_scope"], "not_declared_formal_kinetics")


def run_block(output, summary):
    if output.exists() or summary.exists():
        raise ValueError("choose new full/summary filenames; retain earlier evidence")
    records, started = [], time.perf_counter()

    def unit(name, operation):
        record = Record(name)
        try:
            operation(record)
        except Exception as error:
            record.evidence["exception"] = {"type": type(error).__name__, "message": str(error)}
            record.check("execution_completed", False, True)
        records.append(record)
        elapsed = time.perf_counter() - started
        print(
            f"stage=analytic-reference units={len(records)}/28 "
            f"throughput={len(records) / elapsed:.2f}/s "
            f"eta_s={(28 - len(records)) * elapsed / len(records):.1f} unit={name}",
            flush=True,
        )

    def stable(record, kind, parameters, scaled=False):
        frame, values = frame_case(kind, **parameters, scaled=scaled)
        record.evidence["original_model"] = frame.model.to_dict()
        report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
        verify_stable(record, report, kind, parameters)

    for i, (kind, parameters) in enumerate(fixed_cases()):
        unit(f"{kind}-{i + 1:02d}", lambda r, k=kind, p=parameters: stable(r, k, p))

    def exceptional(record, marginal=False):
        kind, options = ("birth", {"j": 0, "k1": 0, "zero": True}) if marginal else ("unstable", {})
        frame, values = frame_case(kind, **options)
        record.evidence["original_model"] = frame.model.to_dict()
        report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
        record.evidence["analysis"] = report
        record.check("root_found", report["status"], "steady_state_found")
        record.check(
            "stability", report["stability"], "marginal_or_unresolved" if marginal else "unstable"
        )
        record.check(
            "eigenvalue",
            report["reduced_eigenvalues_s_inv"][0]["real"],
            0 if marginal else 0.5,
            numeric=True,
        )
        record.check("noise_unavailable", report["linear_noise"]["status"], "unavailable")
        record.check("no_decay_time", report["relaxation_times_s"], [None])
        if marginal:
            record.check("singular_control_unavailable", report["control"]["status"], "unavailable")
        else:
            record.check("positive_root", report["steady_amounts_mol"]["X"], 0.25, numeric=True)

    unit("unstable-positive-root", exceptional)
    unit("marginal-zero-flux", lambda r: exceptional(r, True))
    unit("mmol-ml-minute", lambda r: stable(r, "birth", {"j": 0.2, "k1": 0.5, "volume": 0.5}, True))

    def world(record):
        from chemworld.runtime import ChemWorldRuntime
        from chemworld.world.sbml_world import sbml_world

        model = build_model("closed", k1=0.5, k2=1.5, total=0.8)
        composition = sbml_world(model)
        runtime = ChemWorldRuntime(composition=composition)
        try:
            runtime.reset(seed=52)
            action = runtime.execute({"operation": "wait", "duration_s": 0.5})
            record.check("actual_wait_committed", action.info["transaction_status"], "committed")
            before = runtime.snapshot().to_dict()
            spec = composition.physics.models_of("sbml")[runtime._state.vessel_id]
            report = analyze_sbml_world_state(runtime._state, spec, include_lna=True).to_dict()
            record.check(
                "all_snapshot_fields_unchanged", runtime.snapshot().to_dict() == before, True
            )
            record.check("uses_world_time", report["time_s"], 0.5, numeric=True)
            verify_stable(record, report, "closed", {"k1": 0.5, "k2": 1.5, "total": 0.8})
        finally:
            runtime.close()

    unit("original-world-read-only", world)
    checks = [c for record in records for c in record.checks]
    failures = [
        {"unit": record.unit, **c} for record in records for c in record.checks if not c["passed"]
    ]
    report = {
        "mode": "development",
        "design": "BIOCHEMICAL_ANALYSIS_EXPERIMENT.md",
        "unit_count": 28,
        "completed_units": len(records),
        "passed_units": sum(all(c["passed"] for c in r.checks) for r in records),
        "comparison_count": len(checks),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "units": [
            {"unit": r.unit, "comparisons": r.checks, "evidence": r.evidence} for r in records
        ],
    }
    compact = {
        **{k: v for k, v in report.items() if k != "units"},
        "evidence_kind": "summary; complete comparison tables and source XML in external artifact",
        "full_artifact_path": str(output.resolve()),
        "units": [
            {
                "unit": r.unit,
                "comparison_count": len(r.checks),
                "passed_comparisons": sum(c["passed"] for c in r.checks),
                "failures": [c for c in r.checks if not c["passed"]],
                "evidence": {k: v for k, v in r.evidence.items() if k != "original_model"},
            }
            for r in records
        ],
    }
    encoded, concise = (
        json.dumps(report, indent=2, allow_nan=False),
        json.dumps(compact, indent=2, allow_nan=False),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    with summary.open("x", encoding="utf-8") as stream:
        stream.write(concise)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"units", "failures"}}), flush=True
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run_block(arguments.output, arguments.summary))
