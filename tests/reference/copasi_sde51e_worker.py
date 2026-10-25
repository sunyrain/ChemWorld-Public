"""Isolated reference-only COPASI worker; no ChemWorld or NumPy imports."""

import argparse
import json
import math
import time
import traceback
from pathlib import Path


def simulate(api, job, output):
    dm = api.CRootContainer.addDatamodel()
    identity = job["id"]
    xml = job["sbml"]
    preserve = job.get("preserve_model", True)
    if preserve:
        (output / (identity + ".xml")).write_text(xml, encoding="utf-8")
    stage, attempted, completed = "setup", False, False
    try:
        if not dm.importSBMLFromString(xml):
            raise RuntimeError(api.CCopasiMessage.getAllMessageText())
        model = dm.getModel()
        reactions = {
            str(model.getReactions()[i].getObjectName()): model.getReactions()[i]
            for i in range(model.getReactions().size())
        }
        for name, amplitude in job["noise"].items():
            reaction = reactions[name]
            reaction.setHasNoise(True)
            if isinstance(amplitude, dict):
                species = {str(m.getObjectName()): m for m in model.getMetabolites()}
                cn = str(species[amplitude["population"]].getValueReference().getCN().getString())
                if not cn.endswith("Reference=ParticleNumber"):
                    raise ValueError("noise reference is not a particle number")
                expression = format(amplitude["coefficient"], ".17g") + "*<" + cn + ">"
            else:
                expression = (
                    reaction.getDefaultNoiseExpression()
                    if amplitude is None
                    else format(amplitude, ".17g")
                )
            if not reaction.setNoiseExpression(expression) or not reaction.hasNoise():
                raise ValueError("failed to bind source noise on " + name)
        if not model.compileIfNecessary():
            raise ValueError("source model compilation failed")
        model.applyInitialValues()
        container = model.getMathContainer()
        container.updateSimulatedValues(False)
        container.pushAllTransientValues()
        factor = float(model.getQuantity2NumberFactor())
        if factor != 1.0:
            raise ValueError("source item basis is not particle-scaled: " + str(factor))
        initial_fluxes = {name: float(r.getParticleFlux()) for name, r in reactions.items()}
        initial_noise = {
            name: float(r.getNoiseExpressionPtr().calcValue()) for name, r in reactions.items()
        }
        expressions = {name: str(r.getNoiseExpression()) for name, r in reactions.items()}
        if "gate_expected" in job:
            expected = job["gate_expected"]
            populations = {
                str(m.getObjectName()): float(m.getValue()) for m in model.getMetabolites()
            }
            for actual, target in (
                (populations, expected["initial"]),
                (initial_fluxes, expected["fluxes"]),
                (initial_noise, expected["noise"]),
            ):
                if not all(
                    math.isclose(actual[k], v, abs_tol=1e-8, rel_tol=1e-10)
                    for k, v in target.items()
                ):
                    raise ValueError("source scalar gate failed: " + repr((actual, target)))
        task = dm.getTask("Time-Course")
        if not task.setMethodType(api.CTaskEnum.Method_stochasticRunkeKuttaRI5):
            raise ValueError("source RI5 method selection failed")
        problem = task.getProblem()
        problem.setStepNumber(job.get("output_steps", round(job["duration_s"] / job["step_s"])))
        problem.setDuration(job["duration_s"])
        problem.setTimeSeriesRequested(True)
        problem.setAutomaticStepSize(False)
        problem.setOutputEvent(False)
        method = task.getMethod()
        for name, value in [
            ("Internal Steps Size", job["step_s"]),
            ("Max Internal Steps", 100000),
            ("Force Physical Correctness", True),
            ("Absolute Tolerance", job.get("absolute_tolerance", 1e-12)),
            ("Tolerance for Root Finder", 1e-12),
        ]:
            method.getParameter(name).setValue(value)
        absolute_tolerance = float(method.getParameter("Absolute Tolerance").getValue())
        if absolute_tolerance != job.get("absolute_tolerance", 1e-12):
            raise ValueError("source absolute physical tolerance was not retained")
        if preserve and not dm.saveModel(str(output / (identity + ".cps")), True):
            raise ValueError("failed to preserve source CPS")
        random = container.getRandomGenerator()
        random.initialize(job["seed"])
        prefix = [int(random.getRandomU()) for _ in range(16)]
        random.initialize(job["seed"])
        stage, attempted = "time_course", True
        with (output / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps({"id": identity, "event": "time_course_started"}) + "\n")
        if not task.process(True):
            raise RuntimeError(api.CCopasiMessage.getAllMessageText())
        stage, completed = "output", True
        series = task.getTimeSeries()
        indices = {str(series.getTitle(i)): i for i in range(series.getNumVariables())}
        if not set(job["species"]) <= indices.keys():
            raise ValueError("source particle output columns are missing")
        rows = [
            {
                "time_s": float(series.getData(i, 0)),
                "populations": {
                    name: float(series.getData(i, indices[name])) for name in job["species"]
                },
            }
            for i in range(series.getRecordedSteps())
        ]
        return {
            "id": identity,
            "case": job["case"],
            "seed": job["seed"],
            "status": "ok",
            "simulation_attempted": attempted,
            "simulation_completed": completed,
            "version": api.CVersion.VERSION.getVersion(),
            "method_enum": 33,
            "absolute_tolerance": absolute_tolerance,
            "quantity_to_number": factor,
            "quantity_unit": str(model.getQuantityUnit()),
            "initial_fluxes": initial_fluxes,
            "initial_noise": initial_noise,
            "noise_expressions": expressions,
            "rng_type": int(random.getType()),
            "rng_prefix": prefix,
            "rows": rows,
            "messages": str(api.CCopasiMessage.getAllMessageText()),
        }
    except Exception as exc:
        exc.sde_stage, exc.sde_attempted, exc.sde_completed = stage, attempted, completed
        raise
    finally:
        api.CRootContainer.removeDatamodel(dm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import COPASI

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    start = time.monotonic()
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "version": COPASI.CVersion.VERSION.getVersion(),
                "method_enum": 33,
                "python_source": __file__,
                "sdk_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for index, job in enumerate(jobs, 1):
            try:
                row = simulate(COPASI, job, args.output)
            except Exception as exc:
                row = {
                    "id": job["id"],
                    "case": job["case"],
                    "seed": job["seed"],
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "stage": getattr(exc, "sde_stage", "setup"),
                    "simulation_attempted": getattr(exc, "sde_attempted", False),
                    "simulation_completed": getattr(exc, "sde_completed", False),
                }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "stage": "copasi",
                        "completed": index,
                        "total": len(jobs),
                        "unit": job["id"],
                        "elapsed_s": elapsed,
                        "units_per_second": index / elapsed if elapsed > 0 else None,
                        "eta_s": (len(jobs) - index) * elapsed / index,
                    }
                ),
                flush=True,
            )
            if index == 1 and "gate_expected" in job and row.get("stage") == "setup":
                break


if __name__ == "__main__":
    main()
