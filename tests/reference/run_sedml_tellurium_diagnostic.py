"""Execute the six unchanged initialization experiments with isolated Tellurium."""

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
import warnings
from pathlib import Path


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def reference(sources, output):
    import libsedml
    import numpy as np
    import roadrunner
    import tellurium
    from tellurium.sedml.tesedml import SEDMLCodeFactory

    report = {"expected_experiments": 6, "expected_rows": 90, "expected_values": 270,
              "versions": {"tellurium": tellurium.__version__,
                           "roadrunner": roadrunner.__version__,
                           "libsedml": libsedml.getLibSEDMLDottedVersion()}, "units": []}
    started = time.perf_counter()
    for kind in ("I", "E", "IE"):
        for reset in (True, False):
            identity = kind+"-reset-"+str(reset).lower()
            unit = {"id": identity, "rows": [], "failures": [], "warnings": [],
                    "task_completed": False}
            report["units"].append(unit)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    text = (sources/(identity+".sedml")).read_text(encoding="utf-8")
                    factory = SEDMLCodeFactory(text, workingDir=str(sources),
                                               createOutputs=False, saveOutputs=False)
                    unit["sedml_messages"] = [
                        factory.doc.getError(i).getMessage()
                        for i in range(factory.doc.getNumErrors())]
                    code = factory.toPython()
                    (output/(identity+"-generated.py")).write_text(code, encoding="utf-8")
                    result = factory.executePython()
                    (output/(identity+"-executed.py")).write_text(result["code"], encoding="utf-8")
                    arrays = {key: np.asarray(value, dtype=float)
                              for key, value in result["dataGenerators"].items()}
                    unit["task_completed"] = True
                    unit["data_generators"] = {
                        key: {"shape": list(value.shape), "values": value.tolist()}
                        for key, value in arrays.items()}
                    selected = []
                    for key in ("dg_t", "dg_a", "dg_kin"):
                        value = arrays[key]
                        if value.shape == (5, 3):
                            value = value.T
                        elif value.shape != (15, 1):
                            raise ValueError("unexpected reference time/repeat axes")
                        selected.append(value.ravel())
                    unit["row_order"] = "repeat_then_time"
                    if any(value.size != 15 for value in selected):
                        raise ValueError("each data generator must contain all fifteen samples")
                    rows = np.column_stack(selected)
                    if not np.all(np.isfinite(rows)):
                        raise ValueError("nonfinite reference output")
                    unit["rows"] = rows.tolist()
                except Exception:
                    unit["failures"].append(traceback.format_exc())
                finally:
                    unit["warnings"] = [str(item.message) for item in caught]
            unit["complete"] = (unit["task_completed"] and len(unit["rows"]) == 15
                                and not unit["failures"])
            write(output/"reference.json", report)
            print(f"Tellurium initialization {len(report['units'])}/6 "
                  f"rows={len(unit['rows'])}/15 complete={unit['complete']} "
                  f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
    report["row_count"] = sum(len(unit["rows"]) for unit in report["units"])
    report["value_count"] = 3*report["row_count"]
    report["complete_experiments"] = sum(unit["complete"] for unit in report["units"])
    report["collection_complete"] = (report["complete_experiments"] == 6
                                     and report["row_count"] == 90)
    write(output/"reference.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("parent", "reference"), default="parent")
    args = parser.parse_args()
    if args.stage == "reference":
        reference(args.source_directory, args.output)
        return 0
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root) or args.reference_python is None:
        raise ValueError("isolated interpreter and outside-Git evidence directory are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/"inputs"
    shutil.copytree(args.source_directory, sources)
    shutil.copy2(__file__, args.output/"executed_source.py")
    note = root/"workstreams/world_foundation/NATIVE_CONV50C_EXPERIMENT.md"
    shutil.copy2(note, args.output/note.name)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["MPLBACKEND"] = "Agg"
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()), "--stage",
               "reference", "--source-directory", str(sources), "--output", str(args.output)]
    with ((args.output/"reference.log").open("x", encoding="utf-8") as log,
          subprocess.Popen(command, stdout=log, stderr=log, env=environment) as child):
        while True:
            try:
                child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"Tellurium diagnostic alive pid={child.pid} log_bytes={log.tell()}",
                      flush=True)
    write(args.output/"process.json", {"exit": child.returncode, "command": command})
    path = args.output/"reference.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary = {key: value for key, value in data.items() if key != "units"}
    summary["exit"] = child.returncode
    write(args.output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary.get("collection_complete") and child.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
