"""Boundary-only CPU migration baseline; fixed blocks in NATIVE_CPU01_EXPERIMENT.md.

Run with ``uv run --no-sync python -m scripts.benchmark_cpu_baseline --output <new-dir>``.
Timing, profiling and allocation samples are separate. No production solver is patched.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
import traceback
import tracemalloc
from collections import Counter
from pathlib import Path

REPEATS = 3
TIMEOUT_S = 600
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def block_a():
    return (
        [
            {"id": f"ode-{size}-{kind}", "kind": "ode", "size": size, "rates": kind}
            for size in (2, 8, 32)
            for kind in ("regular", "stiff")
        ]
        + [
            {"id": "dae-" + mode, "kind": "dae", "derivatives": mode}
            for mode in ("analytic", "finite_difference")
        ]
        + [
            {"id": f"ledger-{length}", "kind": "ledger", "length": length}
            for length in (16, 64, 256, 1024, 4096)
        ]
    )


def write(path, value):
    def encode(other):
        if hasattr(other, "tolist"):
            return other.tolist()
        if isinstance(other, Path):
            return str(other)
        raise TypeError(f"Not JSON serializable: {type(other).__name__}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=encode) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "n": len(values),
        "values_s": values,
        "median_s": statistics.median(values),
        "p95_s": percentile(values, 0.95),
        "p95_method": "linear; only three repetitions",
    }


def environment():
    import numpy as np
    import psutil
    import scipy
    from threadpoolctl import threadpool_info

    memory = psutil.Process().memory_info()
    peak_field = next((name for name in ("peak_wset", "peak_rss") if hasattr(memory, name)), None)
    try:
        from numpy._core._multiarray_umath import __cpu_features__

        features = {key: bool(value) for key, value in __cpu_features__.items()}
    except ImportError:
        features = {"unavailable": True}
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_cpu_features": features,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
        "threadpools": threadpool_info(),
        "rss_bytes": memory.rss,
        "process_peak_rss_bytes": getattr(memory, peak_field) if peak_field else None,
        "peak_rss_method": peak_field or "unavailable; not substituted with current RSS",
    }


def make_workload(case, output):
    """Return prepare, operation, validator; prepare is excluded from warm timing."""
    if case["kind"] in {"world", "sbml", "dfn", "task_binding"}:
        from scripts.cpu_baseline_workloads import make_workload as existing_world_workload

        return existing_world_workload(case, output, write)
    if case["kind"] in {"clone", "cohort", "batch", "properties"}:
        from scripts.cpu_baseline_scaling import make_workload as scaling_workload

        return scaling_workload(case, output, write)
    import numpy as np

    if case["kind"] == "ode":
        from scipy import sparse

        from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode

        n = case["size"]
        rates = np.linspace(0.1, 1, n) if case["rates"] == "regular" else np.geomspace(0.1, 1e4, n)
        jac = sparse.diags(-rates, format="csc") if n == 32 else np.diag(-rates)
        policy = ODESolverPolicy("cpu01-fixed-bdf", method="BDF", rtol=1e-8, atol=1e-11)
        case["physical_inputs"] = {
            "rates": rates.tolist(),
            "initial": [1.0] * n,
            "time_span_s": [0, 1],
            "policy": policy.to_dict(),
        }

        def run(_):
            return solve_ode(
                lambda t, y: -rates * y,
                np.ones(n),
                time_span_s=(0, 1),
                evaluation_times_s=[1],
                policy=policy,
                jac=lambda t, y: jac,
            )

        def validate(report):
            actual, expected = report.y[:, -1], np.exp(-rates)
            return {
                "passed": bool(
                    report.success and np.allclose(actual, expected, atol=1e-7, rtol=2e-6)
                ),
                "state": actual.tolist(),
                "reference": expected.tolist(),
                "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
                "solver_diagnostic": report.diagnostic.to_dict(),
            }

        return lambda: None, run, validate
    if case["kind"] == "dae":
        from tests.test_residual_foundation import quadratic_dae

        from chemworld.physchem.residual_solver import ResidualSolvePolicy
        from chemworld.physchem.solver_backend import ODESolverPolicy, solve_residual

        system = quadratic_dae(case["derivatives"])
        ode = ODESolverPolicy("cpu01-fixed-dae", method="BDF", rtol=1e-8, atol=1e-11)
        case["physical_inputs"] = {
            "source": "tests.test_residual_foundation:quadratic_dae",
            "initial": [1, 0.4],
            "k": 0.7,
            "time_span_s": [0, 1],
            "ode_policy": ode.to_dict(),
        }

        def run(_):
            return solve_residual(
                system,
                [1, 0.4],
                time_span_s=(0, 1),
                evaluation_times_s=[1],
                policy=ResidualSolvePolicy(ode=ode),
            )

        def validate(report):
            actual = report.final_point.state
            expected = np.exp([-0.7, -1.4])
            constraint = float(actual[1] - actual[0] ** 2)
            return {
                "passed": bool(
                    report.success
                    and abs(constraint) <= 1e-8
                    and np.allclose(actual, expected, atol=1e-7, rtol=2e-6)
                ),
                "state": actual.tolist(),
                "reference": expected.tolist(),
                "constraint_residual": constraint,
                "solver_diagnostic": report.diagnostics,
            }

        return lambda: None, run, validate
    if case["kind"] == "ledger":
        from chemworld.campaign_resources import (
            CampaignResourceCard,
            CampaignResourceLedger,
            campaign_resource_event_id,
        )

        length = case["length"]
        card = CampaignResourceCard("cpu01-fixed-history", length + 2, 0, 0, 0, {})
        action = {"operation": "pause_control"}
        outcome = {"transaction_status": "committed", "operation_committed": True}
        case["physical_inputs"] = {
            "card": card.to_dict(),
            "action": action,
            "outcome": outcome,
            "campaign_id": "cpu01",
            "first_event_ordinal": 1,
        }
        ledger = CampaignResourceLedger(card)
        for index in range(1, length + 1):
            event_id = campaign_resource_event_id("cpu01", index)
            assert ledger.preflight(event_id, action).allowed
            ledger.record_outcome(event_id, action, outcome)
            if index % 64 == 0 or index == length:
                print(f"history built {index}/{length}", flush=True)
        initial = ledger.snapshot()
        write(output / "initial-ledger.json", initial)
        event_id = campaign_resource_event_id("cpu01", length + 1)

        def run(current):
            timings = {}
            values = {}
            operations = (
                ("preflight", lambda: current.preflight(event_id, action)),
                ("full_snapshot", current.snapshot),
                ("full_integrity", current.verify_integrity),
                ("record_outcome", lambda: current.record_outcome(event_id, action, outcome)),
            )
            for name, function in operations:
                started = time.perf_counter()
                values[name] = function()
                timings[name] = time.perf_counter() - started
            return current, timings, values

        def validate(result):
            current, timings, values = result
            assert values["preflight"].allowed and values["full_integrity"]
            final = current.snapshot()
            assert final["state"]["operation_attempts"] == length + 1
            assert final["state"]["operation_committed_counts"] == {"pause_control": length + 1}
            assert final["events"][:length] == initial["events"]
            assert final["state"]["stocks_used"] == initial["state"]["stocks_used"]
            assert len(final["events"]) == length + 1
            return {
                "operation_timings_s": timings,
                "state": final["state"],
                "final_ledger_sha256": final["ledger_sha256"],
                "original_prefix_equal": True,
                "snapshot_bytes": len(json.dumps(values["full_snapshot"]).encode()),
            }

        return lambda: CampaignResourceLedger.from_snapshot(initial), run, validate
    raise ValueError("Unknown fixed workload: " + case["id"])


def worker(case, output):
    record = {
        "case": case,
        "status": "running",
        "samples": [],
        "timing_samples": REPEATS,
        "timing_scope": "operation only; validation and per-sample prepare excluded",
    }
    started = time.perf_counter()
    write(output / "result.json", record)
    try:
        prepare, operation, validate = make_workload(case, output)
        record["cold_setup_including_late_imports_s"] = time.perf_counter() - started
        write(output / "result.json", record)
        for phase, repeat in [
            ("warmup", 0),
            *(("timing", i) for i in range(REPEATS)),
            ("profile", 0),
            ("allocation", 0),
        ]:
            print(f"sample {case['id']} {phase} {repeat}", flush=True)
            gc.collect()
            setup = time.perf_counter()
            context = prepare()
            row = {"phase": phase, "repeat": repeat, "prepare_s": time.perf_counter() - setup}
            record["samples"].append(row)
            write(output / "result.json", record)
            profile = cProfile.Profile() if phase == "profile" else None
            if phase == "allocation":
                tracemalloc.start()
                before = tracemalloc.take_snapshot()
            call_start = time.perf_counter()
            result = profile.runcall(operation, context) if profile else operation(context)
            row["operation_s"] = time.perf_counter() - call_start
            if phase == "allocation":
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                changes = tracemalloc.take_snapshot().compare_to(before, "lineno")
                tracemalloc.stop()
                row["allocation"] = {
                    "current_bytes": current_bytes,
                    "peak_bytes": peak_bytes,
                    "net_live_blocks": sum(x.count_diff for x in changes),
                    "net_live_bytes": sum(x.size_diff for x in changes),
                    "scope": "net live Python allocations; excludes freed events and native memory",
                }
            if profile:
                profile.dump_stats(str(output / "profile.pstats"))
                stats = pstats.Stats(profile)
                row["profile"] = sorted(
                    [
                        {
                            "file": key[0],
                            "line": key[1],
                            "function": key[2],
                            "primitive_calls": value[0],
                            "total_calls": value[1],
                            "exclusive_s": value[2],
                            "inclusive_s": value[3],
                        }
                        for key, value in stats.stats.items()
                    ],
                    key=lambda item: item["exclusive_s"],
                    reverse=True,
                )
            row["validation"] = validate(result)
            write(output / "result.json", record)
            if not row["validation"].get("passed", True):
                raise ValueError("fixed correctness criterion failed; complete values retained")
        measured = [r for r in record["samples"] if r["phase"] == "timing"]
        record["operation_distribution"] = distribution([r["operation_s"] for r in measured])
        if case["kind"] == "ledger":
            record["ledger_operation_distributions"] = {
                name: distribution([r["validation"]["operation_timings_s"][name] for r in measured])
                for name in measured[0]["validation"]["operation_timings_s"]
            }
        record["status"] = "passed"
    except Exception:
        record["status"] = "failed"
        record["failure"] = traceback.format_exc()
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        record["worker_wall_s"] = time.perf_counter() - started
        try:
            record["environment"] = environment()
        except Exception:
            record["environment_failure"] = traceback.format_exc()
            record["status"] = "failed"
        write(output / "result.json", record)
    return record["status"] == "passed"


def main(output, block="A"):
    output.mkdir(parents=True, exist_ok=False)
    from chemworld.examples.native_capabilities import execution_inventory
    from scripts.cpu_baseline_scaling import block_c, property_inputs
    from scripts.cpu_baseline_workloads import block_b

    write(output / "inventory.json", execution_inventory())
    plan = {
        "block": "CPU01-" + block,
        "units": {"A": block_a, "B": block_b, "C": block_c}[block](),
        "repeats": REPEATS,
        "timeout_s": TIMEOUT_S if block == "A" else 1800,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "interpretation": "Current Python baseline; no compiled CPU candidate or adoption claim",
    }
    write(output / "plan.json", plan)
    if block == "C":
        try:
            property_inputs(output, write)
        except Exception:
            plan["property_input_failure"] = traceback.format_exc()
            write(output / "plan.json", plan)
    rows, started = [], time.perf_counter()
    env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1"), "PYTHONIOENCODING": "utf-8"}
    for index, case in enumerate(plan["units"]):
        folder = output / case["id"]
        folder.mkdir()
        write(folder / "case.json", case)
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_cpu_baseline",
            "--worker",
            str(folder.resolve()),
        ]
        unit_start = time.perf_counter()
        with (folder / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            timed_out = False
            while process.poll() is None:
                try:
                    remaining = plan["timeout_s"] - (time.perf_counter() - unit_start)
                    process.wait(timeout=max(0.01, min(30, remaining)))
                except subprocess.TimeoutExpired:
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "active": case["id"],
                                "completed": index,
                                "total": len(plan["units"]),
                                "elapsed_s": elapsed,
                                "units_per_min": index * 60 / elapsed,
                                "eta_s": elapsed * (len(plan["units"]) - index) / index
                                if index
                                else None,
                            }
                        ),
                        flush=True,
                    )
                if process.poll() is None and time.perf_counter() - unit_start >= plan["timeout_s"]:
                    process.kill()
                    process.wait()
                    timed_out = True
        path = folder / "result.json"
        row = json.loads(path.read_text()) if path.exists() else {"case": case, "samples": []}
        row["process_wall_s"] = time.perf_counter() - unit_start
        row["command"] = command
        row["exit_code"] = process.returncode
        if timed_out:
            row["status"] = "timeout"
        elif process.returncode != 0 and row.get("status") != "failed":
            row["status"] = "worker_error"
        row["worker_record"] = str(path)
        if row.get("status") != "passed":
            row["stdout_tail"] = (folder / "stdout.log").read_text(encoding="utf-8")[-12000:]
        rows.append(row)
        write(
            output / "summary.json",
            {
                **plan,
                "completed_units": len(rows),
                "rows": rows,
                "counts": dict(Counter(r["status"] for r in rows)),
            },
        )
        print(
            f"completed {index + 1}/{len(plan['units'])}: {case['id']} {row['status']}", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--block", choices=("A", "B", "C"), default="A")
    parser.add_argument("--coverage", action="store_true", help="Generate source/task projection")
    parser.add_argument("--installation", action="store_true", help="Run cold distribution units")
    parser.add_argument("--uv", type=Path, help="Explicit uv executable for installation units")
    args = parser.parse_args()
    if args.worker:
        passed = worker(json.loads((args.worker / "case.json").read_text()), args.worker)
        raise SystemExit(0 if passed else 1)
    if args.output is None:
        parser.error("--output or --worker is required")
    if args.coverage:
        from scripts.cpu_baseline_coverage import coverage_inventory

        args.output.mkdir(parents=True, exist_ok=False)
        write(
            args.output / "coverage.json", coverage_inventory(Path(__file__).resolve().parents[1])
        )
    elif args.installation:
        from scripts.cpu_baseline_scaling import installation_baseline

        if args.uv is None or not args.uv.is_file():
            parser.error("--installation requires --uv pointing to an existing executable")
        installation_baseline(args.output, write, str(args.uv.resolve()))
    else:
        main(args.output, args.block)
