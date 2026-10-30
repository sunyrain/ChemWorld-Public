"""Fixed 24-unit SDE block with isolated sequential engines and complete failures."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sde51e_cases import AVOGADRO
from sde51e_scalar_formula import scalar_endpoint
from sde51e_statistics import (
    CDF_LIMIT,
    EPSILON,
    amplitudes,
    cases,
    cdf_difference,
    continuous_moments,
    fluxes,
    job,
    moment_checks,
    moments,
    native_model,
    pools,
    r3,
)
from source_store import SourceStore


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "id": "invalid-line-" + str(index),
                    "status": "error",
                    "stage": "read_output",
                    "error": repr(exc),
                    "raw": line,
                }
            )
    return rows


def progress(stage, done, total, started, **extra):
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "stage": stage,
                "completed": done,
                "total": total,
                "elapsed_s": elapsed,
                "units_per_second": done / elapsed if elapsed > 0 else None,
                "eta_s": (total - done) * elapsed / done if done else None,
                **extra,
            }
        ),
        flush=True,
    )


def native_gate(case, network, spec):
    from chemworld.physchem import ReactionSDE, RI5Draw, ri5_step

    model = ReactionSDE(network, spec)
    values = np.array([case["initial"][k] for k in network.species_ids])
    matrix = np.array(
        [[r["stoichiometry"].get(k, 0) for r in case["reactions"]] for k in network.species_ids]
    )
    np.testing.assert_allclose(model.drift(0, values), matrix @ fluxes(case), atol=1e-8, rtol=1e-10)
    expected_diffusion = matrix * amplitudes(case)
    if case["kind"] == "zero":
        expected_diffusion = np.zeros((len(values), 0))
    np.testing.assert_allclose(
        model.diffusion(0, values), expected_diffusion, atol=1e-8, rtol=1e-10
    )
    data = {"scalar_gate": True}
    if case["kind"].startswith("ou") or case["kind"] == "gbm":
        h = case["step_s"]
        results = []
        for xi in (-math.sqrt(3), 0.0, math.sqrt(3)):
            for eta in (-1, 1):
                expected = scalar_endpoint(case, xi)
                actual = ri5_step(
                    model.drift, model.diffusion, 0, values, h, RI5Draw((xi,), (eta,))
                )
                value = actual.values[network.species_ids.index("A")]
                if not math.isclose(value, expected, abs_tol=1e-10, rel_tol=1e-10):
                    raise ValueError(
                        "independent scalar RI5 polynomial disagrees: "
                        + repr((xi, eta, value, expected))
                    )
                results.append({"hat": xi, "tilde": eta, "actual": value, "expected": expected})
        theory = moments(case, "native")
        if theory["low"] < 0 or theory["high"] > sum(case["initial"].values()):
            raise ValueError("analytic support intersects physical boundary")
        data.update(
            polynomial_checks=results,
            finite_moments=theory,
            continuous_moments=continuous_moments(case),
        )
    return data


def native_worker(case_path, output):
    from chemworld.physchem import advance_sde, initial_sde_continuation

    case = json.loads(case_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    start = last = time.monotonic()
    stage = "setup"
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
        try:
            network, spec = native_model(case)
            write_json(output / "gate.json", native_gate(case, network, spec))
            amounts = {k: v / AVOGADRO for k, v in case["initial"].items()}
        except Exception as exc:
            stream.write(
                json.dumps(
                    {
                        "id": f"{case['id']}-0001",
                        "status": "error",
                        "stage": stage,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "simulation_attempted": False,
                        "simulation_completed": False,
                    }
                )
                + "\n"
            )
            return
        for seed in range(1, case["replicates"] + 1):
            stage, attempted = "initialization", False
            try:
                memory = initial_sde_continuation(network, spec, amounts, seed=seed)
                stage, attempted = "time_course", True
                with (output / "events.jsonl").open("a", encoding="utf-8") as events:
                    events.write(
                        json.dumps(
                            {"id": f"{case['id']}-{seed:04d}", "event": "time_course_started"}
                        )
                        + "\n"
                    )
                result = advance_sde(network, spec, amounts, memory, case["duration_s"])
                row = {
                    "id": f"{case['id']}-{seed:04d}",
                    "seed": seed,
                    "status": "ok",
                    "simulation_attempted": True,
                    "simulation_completed": True,
                    "time_s": result.continuation["time_s"],
                    "populations": {k: v * AVOGADRO for k, v in result.amounts_mol.items()},
                    "diagnostics": result.diagnostics,
                }
            except Exception as exc:
                row = {
                    "id": f"{case['id']}-{seed:04d}",
                    "seed": seed,
                    "status": "error",
                    "stage": stage,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "simulation_attempted": attempted,
                    "simulation_completed": False,
                }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if time.monotonic() - last >= 30 or seed == case["replicates"]:
                progress("native-" + case["id"], seed, case["replicates"], start)
                last = time.monotonic()


def run_child(command, directory, output, total, env, *, timeout_s=600):
    started = time.monotonic()
    status = "running"
    termination = None
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        while True:
            try:
                code = process.wait(timeout=30)
                status = "exited"
                break
            except subprocess.TimeoutExpired:
                done = len(read_rows(output / "trajectories.jsonl"))
                progress(directory.name, done, total, started, pid=process.pid)
                if time.monotonic() - started >= timeout_s:
                    if os.name == "nt":
                        killed = subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            text=True,
                            errors="replace",
                            check=False,
                        )
                        termination = {
                            "returncode": killed.returncode,
                            "stdout": killed.stdout,
                            "stderr": killed.stderr,
                        }
                    if process.poll() is None:
                        process.terminate()
                    code = process.wait(timeout=30)
                    status = "timeout"
                    break
    result = {
        "exit_code": code,
        "status": status,
        "elapsed_s": time.monotonic() - started,
        "process_tree_termination": termination,
    }
    write_json(directory / "process.json", result)
    return result


def endpoint(row, engine):
    if engine == "native":
        return row["populations"], row["time_s"]
    return row["rows"][-1]["populations"], row["rows"][-1]["time_s"]


def unit_summary(case, directory, processes):
    from scipy.stats import binom

    result = {
        "id": case["id"],
        "kind": case["kind"],
        "step_s": case["step_s"],
        "duration_s": case["duration_s"],
        "checks": [],
        "engines": {},
        "processes": processes,
    }
    good = {}
    for engine in ("native", "copasi"):
        rows = read_rows(directory / engine / "trajectories.jsonl")
        valid = [r for r in rows if r["status"] == "ok"]
        good[engine] = [endpoint(r, engine)[0] for r in valid]
        count = case["replicates"]
        journal = directory / engine / "events.jsonl"
        started_ids = {
            r["id"] for r in read_rows(journal) if r.get("event") == "time_course_started"
        }
        attempted_exact = journal.is_file() or (
            processes[engine]["status"] == "exited"
            and (len(rows) == count or (rows and rows[0].get("stage") == "setup"))
        )
        attempted = (
            len(started_ids)
            if journal.is_file()
            else sum(r.get("simulation_attempted", False) for r in rows)
        )
        complete = len(rows) == count and len(valid) == count
        failures = [r for r in rows if r["status"] != "ok"]
        invariant_failures = []
        for row in valid:
            end, clock = endpoint(row, engine)
            if clock != case["duration_s"] or not all(
                math.isfinite(v) and v >= 0 for v in end.values()
            ):
                invariant_failures.append(
                    {"id": row["id"], "reason": "clock_or_finite_nonnegative"}
                )
            for pool in pools(case):
                actual = sum(end[k] * v for k, v in pool.items())
                target = sum(case["initial"][k] * v for k, v in pool.items())
                if not math.isclose(actual, target, abs_tol=1e-8, rel_tol=1e-10):
                    invariant_failures.append(
                        {
                            "id": row["id"],
                            "reason": "pool",
                            "pool": pool,
                            "actual": actual,
                            "target": target,
                        }
                    )
        result["engines"][engine] = {
            "planned": count,
            "recorded": len(rows),
            "completed": len(valid),
            "simulations_attempted": attempted,
            "simulation_attempt_count_exact": bool(attempted_exact),
            "unstarted": count - attempted if attempted_exact else None,
            "interrupted_started_jobs": sorted(started_ids - {r["id"] for r in rows}),
            "unrecorded": count - len(rows),
            "failures": failures,
            "invariant_failures": invariant_failures,
        }
        result["checks"].append(
            {
                "name": engine + ":complete_and_invariants",
                "passed": complete and not invariant_failures,
            }
        )
        theory = moments(case, engine)
        if theory is not None:
            data = moment_checks([v["A"] for v in good[engine]], theory) if complete else None
            result["engines"][engine]["moments"] = data
            result["checks"].append(
                {"name": engine + ":analytic_moments", "passed": complete and data["passed"]}
            )
        if case["kind"] == "zero":
            expected = case["initial"]["A"] * r3(-case["step_s"]) ** round(1 / case["step_s"])
            result["engines"][engine]["zero_R3_expected"] = expected
            result["checks"].append(
                {
                    "name": engine + ":zero_R3",
                    "passed": complete
                    and all(abs(end["A"] - expected) <= EPSILON for end in good[engine]),
                }
            )
        if case["kind"] in {"death", "birth_death"} and complete:
            size = 1 if case["kind"] == "death" else 4
            p = math.exp(-1) if size == 1 else (0.2 / 0.6) * (1 - math.exp(-0.6))
            samples = np.sort([end["A"] for end in good[engine]])
            points = np.union1d(samples, np.arange(size + 1))
            difference = max(
                np.max(
                    np.abs(
                        np.searchsorted(samples, points, side=side) / len(samples)
                        - binom.cdf(
                            points if side == "right" else np.nextafter(points, -np.inf), size, p
                        )
                    )
                )
                for side in ("left", "right")
            )
            result["engines"][engine]["cme_cdf_diagnostic_only"] = float(difference)
        if engine == "native":
            result["engines"][engine]["work_totals"] = {
                key: sum(r["diagnostics"][key] for r in valid)
                for key in (valid[0]["diagnostics"] if valid else [])
            }
    result["cdf"] = {}
    complete = all(
        result["engines"][engine]["completed"] == case["replicates"]
        for engine in ("native", "copasi")
    )
    if case["replicates"] > 1:
        for name in case["initial"]:
            left = [end[name] for end in good["native"]]
            right = [end[name] for end in good["copasi"]]
            metric = {"evaluated": complete, "limit": CDF_LIMIT}
            if complete:
                metric.update(
                    raw_ks=cdf_difference(left, right),
                    shifted_cdf=cdf_difference(left, right, EPSILON),
                )
            result["cdf"][name] = metric
            result["checks"].append(
                {"name": "cdf:" + name, "passed": complete and metric["shifted_cdf"] <= CDF_LIMIT}
            )
    result["passed"] = all(check["passed"] for check in result["checks"])
    return result


def convergence_checks(all_cases):
    checks, details = [], {}
    for kind in ("ou", "gbm", "ou_dense"):
        selected = [c for c in all_cases if c["kind"] == kind]
        for engine in ("native", "copasi"):
            errors = []
            for case in selected:
                finite, exact = moments(case, engine), continuous_moments(case)
                errors.append(
                    {
                        "step_s": case["step_s"],
                        "finite": finite,
                        "continuous": exact,
                        "mean_relative_error": abs(finite["mean"] - exact["mean"])
                        / abs(exact["mean"]),
                        "variance_relative_error": abs(finite["variance"] - exact["variance"])
                        / exact["variance"],
                    }
                )
            fine = errors[-1]
            passed = fine["variance_relative_error"] <= 0.03
            if kind != "ou_dense":
                ratio = errors[-2]["variance_relative_error"] / fine["variance_relative_error"]
                passed = passed and fine["mean_relative_error"] <= 0.03 and ratio >= 2.5
            else:
                ratio = None
            checks.append(
                {"name": kind + ":" + engine, "passed": passed, "last_error_ratio": ratio}
            )
            details[kind + ":" + engine] = errors
    return {"checks": checks, "details": details, "passed": all(c["passed"] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-case", type=Path)
    parser.add_argument(
        "--output-policy", choices=("linear", "ri5_fractional", "ri5_truncated"), default="linear"
    )
    args = parser.parse_args()
    if args.native_case:
        native_worker(args.native_case, args.output)
        return
    if args.sdk_python is None:
        parser.error("--sdk-python is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    note = {
        "linear": "NATIVE_CONV51E_STATISTICS_EXPERIMENT.md",
        "ri5_fractional": "NATIVE_CONV51E_FRACTIONAL_STATISTICS_EXPERIMENT.md",
        "ri5_truncated": "NATIVE_CONV51E_TRUNCATED_STATISTICS_EXPERIMENT.md",
    }[args.output_policy]
    shutil.copyfile(
        Path("workstreams/world_foundation") / note,
        output / "experiment-note.md",
    )
    store = SourceStore(output)
    for name in (
        "sde51e_cases.py",
        "sde51e_scalar_formula.py",
        "sde51e_statistics.py",
        "copasi_sde51e_worker.py",
        "run_sde51e_statistics.py",
    ):
        store.capture(Path(__file__).with_name(name))
    for source in sorted(Path("src/chemworld/physchem").rglob("*.py")):
        store.capture(source)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(TEMP=str(output), TMP=str(output), USER="chemworld_reference")
    all_cases = cases()
    for case in all_cases:
        case["output_policy"] = args.output_policy
    write_json(output / "cases.json", all_cases)
    started = time.monotonic()
    units = []
    for index, case in enumerate(all_cases, 1):
        directory = output / case["id"]
        directory.mkdir()
        write_json(directory / "case.json", case)
        jobs = [job(case, seed) for seed in range(1, case["replicates"] + 1)]
        write_json(directory / "jobs.json", jobs)
        processes = {}
        for engine in ("native", "copasi"):
            wrapper = directory / (engine + "-process")
            wrapper.mkdir()
            if engine == "native":
                command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(Path(__file__).resolve()),
                    "--native-case",
                    str(directory / "case.json"),
                ]
            else:
                worker = Path(__file__).with_name("copasi_sde51e_worker.py").resolve()
                command = [
                    str(args.sdk_python),
                    "-I",
                    "-X",
                    "utf8",
                    str(worker),
                    "--jobs",
                    str(directory / "jobs.json"),
                ]
            command += ["--output", str(directory / engine)]
            processes[engine] = run_child(
                command, wrapper, directory / engine, case["replicates"], env
            )
        summary = unit_summary(case, directory, processes)
        write_json(directory / "summary.json", summary)
        units.append(summary)
        progress("unit", index, len(all_cases), started, unit=case["id"], passed=summary["passed"])
    convergence = convergence_checks(all_cases)
    result = {
        "schema": "chemworld-sde51e-statistics-v1",
        "output_policy": args.output_policy,
        "units": units,
        "convergence": convergence,
        "planned_per_engine": 22530,
        "passed": all(u["passed"] for u in units) and convergence["passed"],
        "elapsed_s": time.monotonic() - started,
        "cdf_limit": CDF_LIMIT,
        "cdf_horizontal_tolerance_particles": EPSILON,
    }
    write_json(output / "summary.json", result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "units": len(units),
                "passed_units": sum(u["passed"] for u in units),
                "elapsed_s": result["elapsed_s"],
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
