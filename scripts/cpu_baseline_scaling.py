"""Fixed CPU-01 scaling workloads; all computation uses existing public models."""

import json
import time
import traceback
from dataclasses import asdict, replace

from scripts.cpu_baseline_workloads import canonical


def block_c():
    return (
        [
            {"id": f"clone-{n}-{writes}", "kind": "clone", "fanout": n, "writes": writes}
            for n in (1, 10, 100)
            for writes in (1, 5, 10)
        ]
        + [{"id": f"cohort-{n}", "kind": "cohort", "cohorts": n} for n in (16, 64, 256)]
        + [
            {"id": "batch-" + group, "kind": "batch", "group": group}
            for group in ("single", "homogeneous", "heterogeneous", "event-stiff")
        ]
        + [
            {"id": "properties-" + backend, "kind": "properties", "backend": backend}
            for backend in ("fused_native", "coolprop_heos")
        ]
    )


def sbml_spec(rate=0.3, delay=0.5):
    from tests.test_sbml import MODEL, model

    from chemworld.world.sbml_world import sbml_world

    xml = MODEL.replace(
        'localParameter id="k" value="0.3"', f'localParameter id="k" value="{rate}"'
    )
    xml = xml.replace("<cn>0.5</cn>", f"<cn>{delay}</cn>")
    return sbml_world(model(xml), maximum_step_s=0.02)


def make_workload(case, output, write):
    return {
        "clone": make_clones,
        "cohort": make_cohorts,
        "batch": make_batch,
        "properties": make_properties,
    }[case["kind"]](case, output, write)


def make_clones(case, output, write):
    import psutil

    from chemworld.runtime import ChemWorldRuntime

    base = ChemWorldRuntime(composition=sbml_spec(), observation_seed_override=12)
    base.reset(seed=7)
    initial_step = base.execute({"operation": "wait", "duration_s": 2.25})
    assert initial_step.info["transaction_status"] == "committed"
    snapshot = canonical(base.snapshot().to_dict())
    base.close()
    encoded = json.dumps(snapshot, allow_nan=False).encode("utf-8")
    write(output / "initial-snapshot.json", snapshot)
    actions = [{"operation": "wait", "duration_s": 0.05}] * case["writes"]
    reference = ChemWorldRuntime.from_snapshot(snapshot)
    for action in actions:
        step = reference.execute(action)
        assert step.info["transaction_status"] == "committed"
    for _ in range(10 - case["writes"]):
        reference.observe()
    expected = canonical(reference._state.to_dict())
    reference.close()
    case["physical_inputs"] = {
        "snapshot": str(output / "initial-snapshot.json"),
        "snapshot_bytes": len(encoded),
        "actions": actions,
        "readonly_calls": 10 - case["writes"],
        "rng_policy": "clone",
        "fork_implementation": "original session-v2 from_snapshot; no COW or split API",
        "actual_copy_bytes": "not observable; JSON bytes are a separate scale proxy",
    }
    sample = 0

    def run(_):
        process = psutil.Process()
        rss_before = process.memory_info().rss
        start = time.perf_counter()
        clones = [ChemWorldRuntime.from_snapshot(snapshot) for _ in range(case["fanout"])]
        restored_s = time.perf_counter() - start
        rss_live = process.memory_info().rss
        rows = []
        try:
            for index, clone in enumerate(clones):
                calls = []
                for action in actions:
                    start = time.perf_counter()
                    step = clone.execute(action)
                    calls.append(
                        {
                            "kind": "execute",
                            "seconds": time.perf_counter() - start,
                            "info": step.info,
                        }
                    )
                for _ in range(10 - case["writes"]):
                    start = time.perf_counter()
                    clone.observe()
                    calls.append({"kind": "observe", "seconds": time.perf_counter() - start})
                rows.append(
                    {"index": index, "calls": calls, "state": canonical(clone._state.to_dict())}
                )
                if (index + 1) % 10 == 0:
                    print(f"clones {index + 1}/{len(clones)}", flush=True)
            return {
                "rows": rows,
                "restore_s": restored_s,
                "rss_before": rss_before,
                "rss_clones_live": rss_live,
                "rss_after_writes": process.memory_info().rss,
            }
        finally:
            for clone in clones:
                clone.close()

    def validate(result):
        nonlocal sample
        checks = {
            "original_snapshot_unchanged": json.dumps(snapshot, allow_nan=False).encode()
            == encoded,
            "independent_clones": all(row["state"] == expected for row in result["rows"]),
            "committed": all(
                call["info"]["transaction_status"] == "committed"
                for row in result["rows"]
                for call in row["calls"]
                if call["kind"] == "execute"
            ),
        }
        path = output / f"clones-{sample}.json"
        sample += 1
        write(path, {**result, "checks": checks})
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "raw_result": str(path),
            "restore_s": result["restore_s"],
            "execute_s": sum(
                call["seconds"]
                for row in result["rows"]
                for call in row["calls"]
                if call["kind"] == "execute"
            ),
            "observe_s": sum(
                call["seconds"]
                for row in result["rows"]
                for call in row["calls"]
                if call["kind"] == "observe"
            ),
            "rss_before": result["rss_before"],
            "rss_clones_live": result["rss_clones_live"],
            "rss_after_writes": result["rss_after_writes"],
            "cloned_snapshot_json_bytes": case["fanout"] * len(encoded),
        }

    return lambda: None, run, validate


def make_cohorts(case, output, write):
    import numpy as np
    from tests.test_crystal_thermal_kinetics import crystal_fixture, quantities

    from chemworld.physchem.coupled_system import HeatFlux
    from chemworld.physchem.crystal_thermal_kinetics import (
        ThermalCrystalCohort,
        advance_crystal_population,
    )

    model, thermal, state = crystal_fixture(saturation=0.5)
    count = case["cohorts"]
    state = replace(
        state,
        cohorts=tuple(
            ThermalCrystalCohort(
                0.001 / count / model.particle_moles(float(d)),
                float(d),
                0.0001 / count,
                0.0003 / count,
                0.001 / count,
            )
            for d in np.linspace(20e-6, 40e-6, count)
        ),
    )
    totals, seed = quantities(model, state)
    case["physical_inputs"] = {
        "model": model.to_dict(),
        "initial": asdict(state),
        "duration_s": 10,
        "heat_W": 5,
        "thermal_fixture": "tests.test_crystal_thermal_kinetics.crystal_fixture",
    }
    sample = 0

    def run(_):
        return advance_crystal_population(
            model, state, thermal, 10, heat_flux=lambda _: HeatFlux(5)
        )

    def validate(result):
        nonlocal sample
        final, final_seed = quantities(model, result.final_state)
        errors = {k: final[k] - v for k, v in totals.items()}
        errors["seed"] = final_seed - seed
        checks = {
            "material": all(abs(v) <= 1e-10 for v in errors.values()),
            "energy": abs(result.energy_balance_residual_J) <= 1e-6,
            "full_interval": result.actual_duration_s == 10 and result.stopping_event is None,
        }
        path = output / f"cohorts-{sample}.json"
        sample += 1
        write(path, {"result": asdict(result), "checks": checks, "material_errors_mol": errors})
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "raw_result": str(path),
            "material_errors_mol": errors,
            "energy_residual_J": result.energy_balance_residual_J,
            "final_cohorts": len(result.final_state.cohorts),
            "solver_diagnostics": result.solver_diagnostics,
        }

    return lambda: None, run, validate


def make_batch(case, output, write):
    import numpy as np

    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.biochemical_worlds import biochemical_world

    group = case["group"]
    count = 1 if group == "single" else 8
    specs = {}
    formal_parameters = {}
    for index in range(count):
        if group == "heterogeneous" and index >= 4:
            specs[index] = biochemical_world(
                "reversible" if index % 2 == 0 else "enzyme", seed=7 + index
            )
        else:
            rates = np.geomspace(0.01, 1000, 8) if group == "event-stiff" else np.arange(1, 9) / 10
            delay = 0.5 if group != "event-stiff" or index % 2 == 0 else 0.75
            specs[index] = sbml_spec(float(rates[index]), delay)
            formal_parameters[index] = (float(rates[index]), delay)
    formal_actions = [{"operation": "wait", "duration_s": value} for value in (2.25, 0.75)]
    stochastic_actions = [{"operation": "wait", "duration_s": value} for value in (2, 1)]

    def actions_for(key):
        return formal_actions if key in formal_parameters else stochastic_actions

    case["physical_inputs"] = {
        "worlds": {str(k): v.to_dict() for k, v in specs.items()},
        "actions_by_world": {str(k): actions_for(k) for k in specs},
        "extra_member_actions": stochastic_actions,
        "world_seeds": {str(k): 7 + k for k in specs},
        "formal_rate_and_delay": formal_parameters,
        "execution": "sequential independent sessions; not native SIMD/threading",
    }
    sample = 0
    reference = None

    def prepare_order(order, extra=False):
        active = dict(specs)
        if extra:
            active[8] = biochemical_world("reversible", seed=15)
        sessions = {}
        for key in order:
            session = ChemWorldRuntime(composition=active[key], observation_seed_override=12 + key)
            session.reset(seed=7 + key)
            sessions[key] = session
        return sessions

    def prepare():
        return prepare_order(list(specs))

    def run(sessions):
        rows = {}
        try:
            for key, session in sessions.items():
                start = time.perf_counter()
                trace = [session.execute(action).info for action in actions_for(key)]
                elapsed = time.perf_counter() - start
                rows[key] = {
                    "state": canonical(session._state.to_dict()),
                    "trace": trace,
                    "execute_s": elapsed,
                }
                print(f"batch {key}: {len(rows)}/{len(sessions)}", flush=True)
            return rows
        finally:
            for session in sessions.values():
                session.close()

    def validate(result):
        nonlocal sample, reference
        from math import exp

        from chemworld.foundation import WorldState
        from chemworld.physchem.stochastic_kinetics import mol_to_counts
        from chemworld.runtime.sbml_models import model_record

        checks = {
            "committed": all(
                info["transaction_status"] == "committed"
                for row in result.values()
                for info in row["trace"]
            )
        }
        science = {}
        for key, row in result.items():
            state = WorldState.from_dict(row["state"])
            checks[f"clock_{key}"] = abs(state.ledger.time_s - 3) <= 1e-12
            if key in formal_parameters:
                rate, delay = formal_parameters[key]
                expected = {
                    "N:A": 0.02 * exp(-rate * (1 - delay)),
                    "N:B": 0.01 * (1 - exp(-rate * (2 + delay)))
                    + 0.02 * (1 - exp(-rate * (1 - delay))),
                    "P:p": 5 + 0.1 * (1 - delay),
                }
                values = model_record(state).settings["coordinates"]
                errors = {name: abs(values[name] - value) for name, value in expected.items()}
                checks[f"analytic_{key}"] = all(v <= 1e-8 for v in errors.values())
                science[key] = {"expected": expected, "absolute_errors": errors}
            else:
                counts = mol_to_counts(state.species_amounts)
                conserved = (
                    counts["A"] + counts["B"] == 100
                    if key % 2 == 0
                    else counts["E"] + counts["ES"] == 10
                    and counts["S"] + counts["ES"] + counts["P"] == 100
                )
                checks[f"integer_pools_{key}"] = conserved
                science[key] = {"counts": counts}
        states = {key: row["state"] for key, row in result.items()}
        alternatives = {}
        if reference is None and checks["committed"]:
            reference = states
            for label, order, extra in (
                ("reverse", list(reversed(specs)), False),
                ("extra_member", [8, *specs], True),
            ):
                alternative = run(prepare_order(order, extra))
                alternatives[label] = alternative
                checks[label + "_same_by_id"] = all(
                    alternative[key]["state"] == value for key, value in states.items()
                )
                checks[label + "_committed"] = all(
                    info["transaction_status"] == "committed"
                    for row in alternative.values()
                    for info in row["trace"]
                )
        elif reference is not None:
            checks["exact_repeat"] = states == reference
        path = output / f"batch-{sample}.json"
        sample += 1
        write(
            path,
            {"worlds": result, "alternatives": alternatives, "checks": checks, "science": science},
        )
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "raw_result": str(path),
            "science": science,
            "world_execute_s": {str(key): row["execute_s"] for key, row in result.items()},
            "failed_transactions": [
                {
                    "world": key,
                    "action_index": index,
                    "status": info["transaction_status"],
                    "actual_duration_s": info.get("actual_duration_s"),
                    "diagnostics": info.get("outcome_diagnostics"),
                    "error_message": info.get("error_message"),
                }
                for key, row in result.items()
                for index, info in enumerate(row["trace"])
                if info["transaction_status"] != "committed"
            ],
        }

    return prepare, run, validate


def property_inputs(output, write):
    """Reference data preparation is outside both timed workers and is not a native dependency."""
    from math import isfinite

    import CoolProp.CoolProp as CP  # noqa: N817

    from scripts.benchmark_helmholtz import FIELDS, FLUIDS, reference_reader

    start = time.perf_counter()
    sources = {name: json.loads(CP.get_fluid_param_string(name, "JSON"))[0] for name in FLUIDS}
    read = reference_reader(sources, enforce_domain=True)
    points = []
    for name, raw in sources.items():
        eos = raw["EOS"][0]
        reducing = eos["STATES"]["reducing"]
        for tf in (0.02, 0.4):
            for df in (0.02, 0.4, 1.2):
                temperature = reducing["T"] + tf * (eos["T_max"] - reducing["T"])
                density = reducing["rhomolar"] * df
                row = {"fluid": name, "temperature_K": temperature, "density_mol_m3": density}
                try:
                    values = read(name, temperature, density)
                    assert all(isfinite(v) for v in values)
                    assert 0 < values[0] <= eos["p_max"] and values[5] > 0 and values[9] > 0
                    row.update(values=list(values), status="passed")
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                points.append(row)
    payload = {
        "sources": sources,
        "points": points,
        "fields": FIELDS,
        "input_generation_s": time.perf_counter() - start,
        "reference": "CoolProp 8.0.0 HEOS",
    }
    write(output / "property-inputs.json", payload)


def make_properties(case, output, write):
    from scripts.benchmark_helmholtz import native_reader, reference_reader

    data = json.loads((output.parent / "property-inputs.json").read_text(encoding="utf-8"))
    assert len(data["points"]) == 36
    read = (
        native_reader(data["sources"], legacy=False)
        if case["backend"] == "fused_native"
        else reference_reader(data["sources"], enforce_domain=True)
    )
    case["physical_inputs"] = {
        "fields": data["fields"],
        "points": data["points"],
        "sweeps": 10,
        "calls_per_sample": 360,
        "reference": data["reference"],
        "source_file": str(output.parent / "property-inputs.json"),
    }
    sample = 0

    def run(_):
        return [
            [read(p["fluid"], p["temperature_K"], p["density_mol_m3"]) for p in data["points"]]
            for _ in range(10)
        ]

    def validate(result):
        nonlocal sample
        differences = []
        for sweep, values in enumerate(result):
            for index, (row, point) in enumerate(zip(values, data["points"], strict=True)):
                if point["status"] != "passed":
                    differences.append({"sweep": sweep, "point": index, "reference_failure": point})
                    continue
                for field, value, expected in zip(
                    data["fields"], row, point["values"], strict=True
                ):
                    error, limit = abs(value - expected), 2e-7 + 2e-8 * abs(expected)
                    differences.append(
                        {
                            "sweep": sweep,
                            "point": index,
                            "field": field,
                            "absolute_error": error,
                            "allowed_error": limit,
                            "passed": error <= limit,
                        }
                    )
        checks = {
            "all_3600_fields": len(differences) == 3600,
            "reference_and_values": all(row.get("passed", False) for row in differences),
        }
        path = output / f"properties-{sample}.json"
        sample += 1
        write(path, {"values": result, "comparisons": differences, "checks": checks})
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "raw_result": str(path),
            "comparisons": len(differences),
            "failed": [row for row in differences if not row.get("passed", False)],
            "maximum_scaled_error": max(
                (
                    row["absolute_error"] / row["allowed_error"]
                    for row in differences
                    if "absolute_error" in row
                ),
                default=None,
            ),
        }

    return lambda: None, run, validate


def installation_baseline(output, write, uv):
    """Two planned interpreter units, never replacing a missing interpreter with another."""
    import importlib.metadata
    import os
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    root = Path(__file__).resolve().parents[1]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    plan = {
        "units": ["3.11", "3.12"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cpu_wheel": "not_implemented",
        "platforms": "Windows x86-64 only",
        "interpretation": "Cold commands include packaging/dependencies; separate from kernel time",
    }
    write(output / "plan.json", plan)
    rows = []

    def command(args, stage, cwd, row):
        start = time.perf_counter()
        log = output / f"{row['python']}-{stage}.log"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(
                        f"installation {row['python']} {stage}: {time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
                    if time.perf_counter() - start >= 1800:
                        process.kill()
                        process.wait()
                        break
        entry = {
            "stage": stage,
            "args": [str(v) for v in args],
            "cwd": str(cwd),
            "seconds": time.perf_counter() - start,
            "exit_code": process.returncode,
            "log": str(log),
        }
        if process.returncode:
            entry["error_tail"] = log.read_text(encoding="utf-8")[-12000:]
        row["commands"].append(entry)
        write(output / "summary.json", {**plan, "rows": rows})
        if process.returncode:
            raise RuntimeError(f"installation stage {stage} returned {process.returncode}")

    installed_probe = """import importlib.abc
import importlib.metadata
import json
import sys
import time
from pathlib import Path
class NoReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'CoolProp', 'teqp', 'thermopack', 'pybamm', 'cantera', 'tests'}
        if fullname.split('.')[0] in forbidden:
            raise RuntimeError('unexpected reference/test dependency: ' + fullname)
        return None
sys.meta_path.insert(0, NoReference())
started = time.perf_counter()
import chemworld
from chemworld.examples.native_capabilities import execution_inventory
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
import_s = time.perf_counter()-started
assert Path(chemworld.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
spec = WorldCompositionSpec.from_dict(json.loads(Path(sys.argv[1]).read_text()))
started = time.perf_counter()
session = ChemWorldRuntime(composition=spec, observation_seed_override=12)
session.reset(seed=7)
initialize_s = time.perf_counter()-started
started = time.perf_counter()
first = session.execute({'operation':'wait', 'duration_s':2.25})
first_s = time.perf_counter()-started
assert first.info['transaction_status'] == 'committed'
snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
copy = ChemWorldRuntime.from_snapshot(snapshot)
for runtime in (session, copy):
    step = runtime.execute({'operation':'wait', 'duration_s':.75})
    assert step.info['transaction_status']=='committed'
assert session._state.to_dict() == copy._state.to_dict()
assert session._state.ledger.time_s == 3
view = execution_inventory()
result = dict(passed=True, installed_from=chemworld.__file__, python=sys.version,
    import_s=import_s, initialize_s=initialize_s, first_advance_s=first_s,
    exact_pending_resume=True, modules=len(view['modules']),
    public_symbols=sum(len(m['symbols']) for m in view['public_physchem']),
    dependencies={name:importlib.metadata.version(name) for name in
    ('numpy','scipy','gymnasium','pandas','pyyaml','scikit-learn')})
Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
print(json.dumps(result))
session.close()
copy.close()
"""
    for target in plan["units"]:
        row = {"python": target, "commands": [], "status": "running"}
        rows.append(row)
        target_python = sys.executable
        if target != version:
            query = [uv, "python", "find", "--no-python-downloads", target]
            found = subprocess.run(query, capture_output=True, text=True, check=False)
            row["interpreter_query"] = {
                "args": query,
                "exit_code": found.returncode,
                "stdout": found.stdout,
                "stderr": found.stderr,
            }
            if found.returncode:
                row.update(status="unavailable", reason="no installed matching interpreter")
                write(output / "summary.json", {**plan, "rows": rows})
                continue
            target_python = found.stdout.strip()
        try:
            folder = output / target
            folder.mkdir()
            sdist_folder, wheel_folder = folder / "sdist", folder / "wheel"
            command([uv, "build", "--sdist", "--out-dir", str(sdist_folder)], "sdist", root, row)
            (archive,) = sdist_folder.glob("*.tar.gz")
            command(
                [uv, "build", str(archive), "--wheel", "--out-dir", str(wheel_folder)],
                "sdist-to-wheel",
                folder,
                row,
            )
            (wheel,) = wheel_folder.glob("*.whl")
            with zipfile.ZipFile(wheel) as zipped:
                (metadata,) = (
                    name for name in zipped.namelist() if name.endswith(".dist-info/WHEEL")
                )
                row["wheel_metadata"] = zipped.read(metadata).decode()
                row["contains_native_extension"] = any(
                    name.endswith((".pyd", ".dll", ".so")) for name in zipped.namelist()
                )
            environment = folder / "environment"
            command([uv, "venv", "--python", target_python, str(environment)], "venv", folder, row)
            interpreter = environment / "Scripts" / "python.exe"
            constraints = folder / "constraints.txt"
            constraints.write_text(
                "\n".join(
                    name + "==" + importlib.metadata.version(name)
                    for name in ("gymnasium", "numpy", "pandas", "pyyaml", "scipy", "scikit-learn")
                ),
                encoding="utf-8",
            )
            command(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(interpreter),
                    "--constraint",
                    str(constraints),
                    str(wheel),
                ],
                "install",
                folder,
                row,
            )
            composition = folder / "composition.json"
            write(composition, sbml_spec().to_dict())
            script = folder / "probe.py"
            script.write_text(installed_probe, encoding="utf-8")
            result_path = folder / "probe-result.json"
            command(
                [
                    uv,
                    "run",
                    "--no-sync",
                    "--no-project",
                    "--python",
                    str(interpreter),
                    "python",
                    "-I",
                    str(script),
                    str(composition),
                    str(result_path),
                ],
                "installed-first-call",
                folder,
                row,
            )
            row["probe"] = json.loads(result_path.read_text())
            row["status"] = "passed" if row["probe"]["passed"] else "failed"
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        write(output / "summary.json", {**plan, "rows": rows})
        print(f"installation {target}: {row['status']}", flush=True)
