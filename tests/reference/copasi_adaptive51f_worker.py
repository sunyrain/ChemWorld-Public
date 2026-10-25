"""Isolated actual adaptive RK4/LSODA SDK worker, preserving source material policy."""

import argparse
import json
import math
import traceback
from pathlib import Path


def output_rows(task, names):
    series = task.getTimeSeries()
    indices = {str(series.getTitle(i)): i for i in range(series.getNumVariables())}
    if not set(names) <= indices.keys():
        raise ValueError("source particle columns missing")
    return [
        {
            "time_s": float(series.getData(i, 0)),
            "populations": {name: float(series.getData(i, indices[name])) for name in names},
        }
        for i in range(series.getRecordedSteps())
    ]


def simulate(api, job, output):
    dm = api.CRootContainer.addDatamodel()
    stage, attempted, completed = "setup", False, False
    row = {
        "id": job["id"],
        "seed": job["seed"],
        "status": "error",
        "version": api.CVersion.VERSION.getVersion(),
        "method_enum": 29 if job["method"] == "RK4" else 30,
    }
    try:
        (output / (job["id"] + ".xml")).write_text(job["sbml"], encoding="utf-8")
        if not dm.importSBMLFromString(job["sbml"]):
            raise RuntimeError(api.CCopasiMessage.getAllMessageText())
        model = dm.getModel()
        if not model.compileIfNecessary():
            raise ValueError("source compilation failed")
        model.applyInitialValues()
        container = model.getMathContainer()
        container.updateSimulatedValues(False)
        container.pushAllTransientValues()
        initial = {str(m.getObjectName()): float(m.getValue()) for m in model.getMetabolites()}
        fluxes = {
            str(model.getReactions()[i].getObjectName()): float(
                model.getReactions()[i].getParticleFlux()
            )
            for i in range(model.getReactions().size())
        }
        volume = float(model.getCompartments()[0].getValue())
        factor = float(model.getQuantity2NumberFactor())
        row.update(
            initial=initial, initial_fluxes=fluxes, volume_L=volume, quantity_to_number=factor
        )
        if factor != 1 or not math.isclose(volume, job["volume_L"], rel_tol=1e-10, abs_tol=0):
            raise ValueError("source volume or item particle basis changed")
        for actual, expected in ((initial, job["initial"]), (fluxes, job["initial_fluxes"])):
            if set(actual) != set(expected) or not all(
                math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-8)
                for key, value in expected.items()
            ):
                raise ValueError("source initial population/flux gate failed")
        task = dm.getTask("Time-Course")
        selected = (
            api.CTaskEnum.Method_hybrid
            if job["method"] == "RK4"
            else api.CTaskEnum.Method_hybridLSODA
        )
        if not task.setMethodType(selected):
            raise ValueError("actual source adaptive method selection failed")
        problem = task.getProblem()
        problem.setStepNumber(job["output_steps"])
        problem.setDuration(job["duration_s"])
        problem.setTimeSeriesRequested(True)
        problem.setAutomaticStepSize(False)
        problem.setOutputEvent(False)
        parameters = {
            "Max Internal Steps": 1000000,
            "Lower Limit": job["lower"],
            "Upper Limit": job["upper"],
            "Partitioning Interval": 1,
            "Use Random Seed": True,
            "Random Seed": job["seed"],
        }
        parameters.update(
            {"Runge Kutta Stepsize": job.get("step_s", 0.125)}
            if job["method"] == "RK4"
            else {
                "Relative Tolerance": 1e-10,
                "Absolute Tolerance": 1e-12,
                "Max Internal Step Size": job.get("step_s", 0.125),
            }
        )
        method = task.getMethod()
        for name, value in parameters.items():
            parameter = method.getParameter(name)
            if parameter.setValue(value) is False or parameter.getValue() != value:
                raise ValueError("source method parameter did not bind: " + name)
        row.update(method_name=str(method.getObjectName()), parameters=parameters)
        if not dm.saveModel(str(output / (job["id"] + ".cps")), True):
            raise ValueError("cannot preserve source CPS")
        random = container.getRandomGenerator()
        random.initialize(job["seed"])
        row["rng_prefix"] = [int(random.getRandomU()) for _ in range(16)]
        random.initialize(job["seed"])
        stage, attempted = "time_course", True
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": job["id"], "event": "time_course_started"}) + "\n")
        processed = task.process(True)
        row["process_returned"] = bool(processed)
        if not processed:
            row["source_messages"] = str(api.CCopasiMessage.getAllMessageText())
            try:
                row["partial_rows"] = output_rows(task, job["initial"])
                row["partial_recorded_steps"] = len(row["partial_rows"])
                row["last_partial_time_s"] = (
                    row["partial_rows"][-1]["time_s"] if row["partial_rows"] else None
                )
            except Exception as exc:
                row["partial_capture_error"] = repr(exc)
            raise RuntimeError("source Time-Course returned False: " + row["source_messages"])
        stage, completed = "output", True
        row["rows"] = output_rows(task, job["initial"])
        if any(not math.isfinite(v) for p in row["rows"] for v in p["populations"].values()):
            raise ValueError("source output contains nonfinite populations")
        row.update(status="ok", messages=str(api.CCopasiMessage.getAllMessageText()))
    except Exception as exc:
        row.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        row.update(stage=stage, simulation_attempted=attempted, simulation_completed=completed)
        api.CRootContainer.removeDatamodel(dm)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import COPASI

    args.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))

    def portable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": repr(value)}
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value

    with (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(
                json.dumps(portable(simulate(COPASI, job, args.output)), allow_nan=False) + "\n"
            )
            stream.flush()


if __name__ == "__main__":
    main()
