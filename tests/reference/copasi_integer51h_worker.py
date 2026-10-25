"""Actual SDK-only integer chemistry; strict imported stoichiometry and controls."""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def capture(task, job, output, row):
    series = task.getTimeSeries()
    titles = [str(series.getTitle(i)) for i in range(series.getNumVariables())]
    table = [
        [float(series.getData(i, j)) for j in range(series.getNumVariables())]
        for i in range(series.getRecordedSteps())
    ]
    row.update(raw_series_titles=titles, raw_series_table=table)
    path = output / (job["id"] + ".tsv")
    if not path.is_file():
        raise ValueError("source complete particle report missing")
    values = [
        [float(v) for v in line.split("\t")]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    names = list(job["initial"])
    if any(len(v) != len(names) + 1 for v in values):
        raise ValueError("source particle report width disagrees")
    rows = [{"time_s": v[0], "populations": dict(zip(names, v[1:], strict=True))} for v in values]
    row["report_rows"] = rows
    if len(table) != len(rows):
        raise ValueError("source report and time-series lengths disagree")
    for raw, point in zip(table, rows, strict=True):
        if not math.isclose(raw[0], point["time_s"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("source report and time-series time disagree")
        for name in set(names) & set(titles):
            if not math.isclose(
                raw[titles.index(name)], point["populations"][name], abs_tol=1e-8, rel_tol=1e-10
            ):
                raise ValueError("source report and time-series particle stocks disagree")
    return rows


def simulate(api, job, output):
    row = {
        "id": job["id"],
        "seed": job["seed"],
        "status": "error",
        "version": api.CVersion.VERSION.getVersion(),
        "method_enum": job["method_enum"],
        "simulation_attempted": False,
        "simulation_completed": False,
        "stage": "setup",
    }
    dm = None
    try:
        api.CCopasiMessage.clearDeque()
        dm = api.CRootContainer.addDatamodel()
        (output / (job["id"] + ".xml")).write_text(job["sbml"], encoding="utf-8")
        if not dm.importSBMLFromString(job["sbml"]):
            raise ValueError("source import failed: " + str(api.CCopasiMessage.getAllMessageText()))
        model = dm.getModel()
        if not model.compileIfNecessary():
            raise ValueError("source compilation failed")
        if "configure_avogadro" in job:
            row.update(imported_avogadro=float(model.getAvogadro()),
                       imported_quantity_to_number=float(model.getQuantity2NumberFactor()),
                       source_quantity_policy="explicit_SI_NA_preserve_source_concentrations")
            model.setAvogadro(job["configure_avogadro"], api.CCore.Framework_Concentration)
            if not model.updateInitialValues(api.CCore.Framework_Concentration):
                raise ValueError("source quantity-policy initial synchronization failed")
        model.applyInitialValues()
        container = model.getMathContainer()
        container.updateSimulatedValues(False)
        container.pushAllTransientValues()
        reactions = {str(r.getObjectName()): r for r in model.getReactions()}
        chemistry = {}
        functions = {}
        for name, reaction in reactions.items():
            equation = reaction.getChemEq()
            chemistry[name] = {
                key: {
                    str(e.getMetabolite().getObjectName()): float(e.getMultiplicity())
                    for e in terms
                }
                for key, terms in (
                    ("substrates", equation.getSubstrates()),
                    ("products", equation.getProducts()),
                )
            }
            function = reaction.getFunction()
            functions[name] = {
                "name": str(function.getObjectName()),
                "type": int(function.getType()),
            }
        initial = {str(m.getObjectName()): float(m.getValue()) for m in model.getMetabolites()}
        fluxes = {name: float(r.getParticleFlux()) for name, r in reactions.items()}
        factor, volume = (
            float(model.getQuantity2NumberFactor()),
            float(model.getCompartments()[0].getValue()),
        )
        row.update(
            initial=initial,
            initial_fluxes=fluxes,
            chemistry=chemistry,
            functions=functions,
            quantity_to_number=factor,
            volume_L=volume,
            avogadro=float(model.getAvogadro()),
        )
        if chemistry != job["chemistry"]:
            raise ValueError("source full substrate/product multiplicities changed")
        if (factor != job.get("expected_quantity_to_number", 1)
                or not math.isclose(volume, job["volume_L"], rel_tol=1e-10, abs_tol=0)
                or ("expected_avogadro" in job
                    and model.getAvogadro() != job["expected_avogadro"])):
            raise ValueError("source quantity basis, Avogadro or physical volume changed")
        for actual, expected in ((initial, job["initial"]), (fluxes, job["initial_fluxes"])):
            if set(actual) != set(expected) or any(
                not math.isclose(actual[k], v, abs_tol=1e-8, rel_tol=1e-10)
                for k, v in expected.items()
            ):
                raise ValueError("source initial stocks or macroscopic particle flux changed")
        try:
            row["prestart_propensities_unqualified"] = [
                float(v) for v in container.getPropensities()
            ]
        except Exception as exc:
            row["prestart_propensity_capture_error"] = repr(exc)
        task = dm.getTask("Time-Course")
        if not task.setMethodType(job["method_enum"]):
            raise ValueError("actual source method selection failed")
        problem = task.getProblem()
        problem.setStepNumber(job["output_steps"])
        problem.setDuration(job["duration_s"])
        problem.setTimeSeriesRequested(True)
        problem.setAutomaticStepSize(False)
        problem.setOutputEvent(False)
        method = task.getMethod()
        row["method_name"] = str(method.getObjectName())
        expected_name = {
            25: "Stochastic (Direct method)",
            26: "Stochastic (Gibson + Bruck)",
            27: "Stochastic (\u03c4-Leap)",
        }[job["method_enum"]]
        if row["method_name"] != expected_name:
            raise ValueError("selected source method name disagrees")
        parameters = {
            "Max Internal Steps": 1000000,
            "Use Random Seed": True,
            "Random Seed": job["seed"],
        }
        if job["epsilon"] is not None:
            parameters["Epsilon"] = job["epsilon"]
        row["parameter_readback"] = {}
        for name, value in parameters.items():
            p = method.getParameter(name)
            if p.setValue(value) is False or p.getValue() != value:
                raise ValueError("source parameter did not bind: " + name)
            row["parameter_readback"][name] = p.getValue()
        if job["method_enum"] == 26:
            row["parameter_readback"]["Subtype"] = method.getParameter("Subtype").getValue()
            if row["parameter_readback"]["Subtype"] != 2:
                raise ValueError("source next-reaction subtype changed")
        report = dm.getReportDefinitionList().createReportDefinition(
            "Particles", "Actual complete particle inventory"
        )
        report.setTaskType(api.CTaskEnum.Task_timeCourse)
        report.setIsTable(False)
        report.setSeparator(api.CCopasiReportSeparator("\t"))
        report.setPrecision(17)
        body = report.getBodyAddr()
        cns = {
            "time_s": model.getCN().getString() + ",Reference=Time",
            **{
                str(m.getObjectName()): str(m.getValueReference().getCN().getString())
                for m in model.getMetabolites()
            },
        }
        ordered = [cns["time_s"], *(cns[k] for k in job["initial"])]
        for index, cn in enumerate(ordered):
            if index:
                body.push_back(api.CRegisteredCommonName(report.getSeparator().getCN().getString()))
            body.push_back(api.CRegisteredCommonName(cn))
        task.getReport().setReportDefinition(report)
        task.getReport().setTarget(str(output / (job["id"] + ".tsv")))
        task.getReport().setAppend(False)
        row["actual_report_cns"] = cns
        if not dm.saveModel(str(output / (job["id"] + ".cps")), True):
            raise ValueError("cannot retain source CPS")
        random = container.getRandomGenerator()
        random.initialize(job["seed"])
        row["rng_prefix"] = [int(random.getRandomU()) for _ in range(16)]
        random.initialize(job["seed"])
        row.update(stage="time_course", simulation_attempted=True)
        with (output / "events.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps({"id": job["id"], "event": "time_course_started"}) + "\n")
        processed = task.process(True)
        row["process_returned"] = bool(processed)
        if not processed:
            try:
                row["partial_rows"] = capture(task, job, output, row)
            except Exception as exc:
                row["partial_capture_error"] = repr(exc)
            raise RuntimeError(
                "source Time-Course returned False: " + str(api.CCopasiMessage.getAllMessageText())
            )
        row.update(stage="output", simulation_completed=True)
        row["rows"] = capture(task, job, output, row)
        row.update(status="ok", messages=str(api.CCopasiMessage.getAllMessageText()))
    except Exception as exc:
        row.update(
            error=repr(exc),
            traceback=traceback.format_exc(),
            source_messages=str(api.CCopasiMessage.getAllMessageText()),
        )
    finally:
        if dm is not None:
            api.CRootContainer.removeDatamodel(dm)
    return row


def portable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import COPASI

    assert not any(k == "chemworld" or k.startswith("chemworld.") for k in sys.modules)
    planned = json.loads(args.jobs.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    started = last = time.monotonic()
    with (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream:
        for index, job in enumerate(planned, 1):
            with (args.output / "events.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps({"id": job["id"], "event": "trajectory_started"}) + "\n")
            row = simulate(COPASI, job, args.output)
            stream.write(json.dumps(portable(row), allow_nan=False) + "\n")
            stream.flush()
            now = time.monotonic()
            if now - last >= 30 or index == len(planned):
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(planned),
                            "trajectories_per_second": index / elapsed if elapsed else None,
                            "eta_s": (len(planned) - index) * elapsed / index,
                        }
                    ),
                    flush=True,
                )
                last = now


if __name__ == "__main__":
    main()
