"""Fixed 36 actual native worlds; independent parity is reported separately."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_sbml_semantic_block import ANALYTICAL, comparisons
from run_sbml_trajectory_block import TIMES, cases


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def execute(runtime, action):
    result = runtime.execute(action)
    if result.info["operation_outcome"] != "success":
        raise ValueError({"action": action, "info": result.info})
    return result


def measure(runtime):
    from chemworld.runtime.sbml_models import model_record

    state = runtime._state
    owned, clock = model_record(state).to_dict(), state.ledger.time_s
    result = execute(runtime, {"operation": "measure", "instrument": "formal_quantity_meter"})
    reading = runtime.observe()
    if (runtime._state.ledger.time_s != clock
            or model_record(runtime._state).to_dict() != owned
            or runtime._state.ledger.sample_consumed_L != 0
            or "N:A" in json.dumps(result.info)
            or "coordinates" in runtime._state.to_dict(include_hidden=False)["equipment"][
                "formal_model"]["settings"]):
        raise ValueError("formal gauge changed hidden truth/clock or exposed coordinates")
    expected = {"amount_A": "mol", "amount_B": "mol", "concentration_A": "mol/L",
                "concentration_B": "mol/L", "volume_c1": "L", "volume_c2": "L"}
    if {c.channel_id: c.unit for c in reading.channels} != expected:
        raise ValueError("formal native channel names/units differ")
    spec = runtime.compiled_composition.spec.physics.models_of("sbml")[state.vessel_id]
    truth = spec.readout(owned["settings"]["coordinates"], clock)
    floors = {"mol": 1e-9, "mol/L": 1e-6, "L": 1e-9}
    if any(c.uncertainty != spec.readout_relative_sd*max(abs(truth[c.channel_id]), floors[c.unit])
           for c in reading.channels):
        raise ValueError("formal readout uncertainty differs from its declared detector")
    return reading.native_record()


def resume(snapshot, output, scales):
    from chemworld.runtime import ChemWorldRuntime

    runtime = ChemWorldRuntime.from_snapshot(json.loads(snapshot.read_text(encoding="utf-8")))
    for target in TIMES[8:]:
        execute(runtime, {"operation": "wait",
                          "duration_s": target*scales-runtime._state.ledger.time_s})
        measure(runtime)
    write(output, {"state": runtime._state.to_dict(), "reading": runtime.observe().native_record()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--time-scale", type=float)
    args = parser.parse_args()
    if args.resume is not None:
        resume(args.resume, args.output, args.time_scale)
        return 0
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.physchem.sbml_system import SBMLObservable
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.sbml_models import model_record
    from chemworld.world.sbml_world import sbml_world

    root = Path(__file__).resolve().parents[2]
    if any(p.resolve().is_relative_to(root) for p in (args.source_directory, args.output)):
        raise ValueError("raw source and evidence must stay outside Git")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((args.baseline/"native.json").read_text(encoding="utf-8"))
    hashes = json.loads((args.baseline/"reference.json").read_text(encoding="utf-8"))
    reference = json.loads((args.semantic/"reference.json").read_text(encoding="utf-8"))
    if baseline["executed_inputs"] != 18 or baseline["comparison_count"] != 1386:
        raise ValueError("world regression needs the complete prior 18-source baseline")
    source = InputSource(DataSourceProvenance("synthetic-sbml-trajectory-18", 0,
        "Self-authored fixed 18 complete source models", "1", "CC0-1.0"), "synthetic")
    report = {"expected_worlds": 36, "expected_runtime_comparisons": 2772,
              "expected_independent_comparisons": 2772, "expected_closed_form_comparisons": 770,
              "expected_readout_channels": 2376, "units": [], "elapsed_s": 0.}
    started = time.perf_counter()
    for form in ("ode", "residual"):
        for name, scales, index, _ in cases():
            unit = {"input": name, "form": form, "runtime_passed": False,
                    "independent_passed": False, "failures": [], "samples": [], "steps": []}
            report["units"].append(unit)
            identity = form+"-"+Path(name).stem
            try:
                text = (args.source_directory/name).read_text(encoding="utf-8")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ref = next(u for u in reference["units"] if u["input"] == name)
                original = next(u for u in hashes["units"] if u["input"] == name)
                prior = next(u for u in baseline["units"] if u["input"] == name)
                if not ref["passed"] or digest != ref["sha256"] or digest != original["sha256"]:
                    raise ValueError("source differs from the original independent model")
                (args.output/(identity+".xml")).write_text(text, encoding="utf-8")
                unit.update(sha256=digest, relative_sd=(.01, .05)[index % 2],
                    noise_mode=("sequential", "keyed")[(index//2+(form == "residual")) % 2])
                declaration = sbml_world(SBMLModel(text, source), budget=60,
                    maximum_step_s=.02*scales["time_s"], rtol=1e-10, atol=1e-12,
                    numerical_form=form, readout_relative_sd=unit["relative_sd"],
                    observables=tuple(SBMLObservable(kind, key) for kind, names in (
                        ("amount", ("A", "B")), ("concentration", ("A", "B")),
                        ("volume", ("c1", "c2"))) for key in names))
                runtime = ChemWorldRuntime(composition=declaration,
                                           observation_noise_mode=unit["noise_mode"])
                runtime.reset(seed=19)
                frame = runtime.compiled_composition.spec.physics.models_of("sbml")[
                    "formal_workspace"].frame
                snapshot_path = args.output/(identity+"-snapshot.json")
                for node, target in enumerate(TIMES):
                    if target:
                        step = execute(runtime, {"operation": "wait", "duration_s":
                            target*scales["time_s"]-runtime._state.ledger.time_s})
                        unit["steps"].append(step.info)
                    record = model_record(runtime._state)
                    coordinates = record.settings["coordinates"]
                    clock = runtime._state.ledger.time_s
                    amounts, volumes = frame.quantities(coordinates, clock)
                    unit["samples"].append({"time_source": target,
                        "values": [amounts["A"], amounts["B"], volumes["c1"], volumes["c2"],
                                   amounts["A"]/volumes["c1"], amounts["B"]/volumes["c2"],
                                   frame.context(coordinates, clock)["p"]],
                        "readout": measure(runtime), "owned_state": record.to_dict()})
                    if node == 7:
                        write(snapshot_path, runtime.snapshot().to_dict())
                unit["runtime_comparisons"] = comparisons(unit["samples"], prior["samples"],
                                                          "world_pure_layer_regression")
                unit["independent_comparisons"] = comparisons(unit["samples"], ref["samples"],
                                                              "world_RoadRunner")
                unit["closed_form_comparisons"] = (comparisons(unit["samples"],
                    ref["analytical_samples"], "world_closed_form") if index in ANALYTICAL else [])
                child_path = args.output/(identity+"-child.json")
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume",
                    str(snapshot_path), "--output", str(child_path), "--time-scale",
                    str(scales["time_s"])], capture_output=True, text=True, timeout=60)
                write(args.output/(identity+"-child-log.json"), {
                    "exit": child.returncode, "stdout": child.stdout, "stderr": child.stderr})
                if child.returncode:
                    raise ValueError("cross-process continuation failed: "+child.stderr)
                expected = {"state": runtime._state.to_dict(),
                            "reading": runtime.observe().native_record()}
                actual = json.loads(child_path.read_text(encoding="utf-8"))
                unit["cross_process_exact"] = actual == json.loads(json.dumps(expected))
                if not unit["cross_process_exact"]:
                    raise ValueError("cross-process state or detector continuation differs")
                unit["runtime_passed"] = all(c["passed"] for c in (
                    *unit["runtime_comparisons"], *unit["closed_form_comparisons"]))
                unit["independent_passed"] = all(c["passed"]
                    for c in unit["independent_comparisons"])
            except Exception:
                unit["failures"].append(traceback.format_exc())
            elapsed = time.perf_counter()-started
            report["elapsed_s"] = elapsed
            write(args.output/"report.json", report)
            count = len(report["units"])
            passed = sum(u["runtime_passed"] for u in report["units"])
            print(f"native worlds {count}/36 runtime={passed}"
                  f" independent={sum(u['independent_passed'] for u in report['units'])} "
                  f"elapsed={elapsed:.1f}s ETA={elapsed/count*(36-count):.1f}s", flush=True)
    summary = {k: v for k, v in report.items() if k != "units"}
    for name in ("runtime_comparisons", "independent_comparisons", "closed_form_comparisons"):
        rows = [c for u in report["units"] for c in u.get(name, [])]
        summary[name] = {"executed": len(rows), "passed": sum(c["passed"] for c in rows)}
    summary.update(runtime_passed_worlds=sum(u["runtime_passed"] for u in report["units"]),
        independent_passed_worlds=sum(u["independent_passed"] for u in report["units"]),
        cross_process_exact=sum(u.get("cross_process_exact", False) for u in report["units"]),
        readout_channels=sum(len(s["readout"]["channels"]) for u in report["units"]
                             for s in u["samples"]))
    summary["runtime_contract_passed"] = (summary["runtime_passed_worlds"] == 36
        and summary["runtime_comparisons"]["executed"] == 2772
        and summary["closed_form_comparisons"]["executed"] == 770
        and summary["cross_process_exact"] == 36 and summary["readout_channels"] == 2376)
    summary["independent_parity_passed"] = summary["independent_passed_worlds"] == 36
    write(args.output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["runtime_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
