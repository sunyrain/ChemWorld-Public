"""Isolated actual COPASI 4.48.309 task adapter; no native package imports."""

from __future__ import annotations

import importlib.metadata
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import COPASI


def parameters(group):
    result = []
    for i in range(group.size()):
        p = group.getParameter(i)
        kind = COPASI.CCopasiParameterGroup.getType(group, i)
        result.append(
            (
                p.getObjectName(),
                str(kind),
                "group" if kind == COPASI.CCopasiParameter.Type_GROUP else str(p.getValue()),
            )
        )
    return result


def probe():
    dm = COPASI.CRootContainer.addDatamodel()
    try:
        result = {
            "distribution": importlib.metadata.version("python-copasi"),
            "sdk": COPASI.CVersion.VERSION.getVersion(),
            "tasks": {},
        }
        for name in (
            "Steady-State",
            "Metabolic Control Analysis",
            "Linear Noise Approximation",
            "Lyapunov Exponents",
            "Time Scale Separation Analysis",
            "Optimization",
            "Parameter Estimation",
        ):
            task = dm.getTask(name)
            result["tasks"][name] = {
                "task_type": str(type(task)),
                "problem": parameters(task.getProblem()),
                "method": parameters(task.getMethod()),
                "task_api": [s for s in dir(task) if not s.startswith("_")],
                "method_api": [s for s in dir(task.getMethod()) if not s.startswith("_")],
                "problem_api": [s for s in dir(task.getProblem()) if not s.startswith("_")],
            }
        return result
    finally:
        COPASI.CRootContainer.removeDatamodel(dm)


def matrix(value):
    if not hasattr(value, "numRows"):
        shape = list(value.size())
        return [[float(value.array().get(i, j)) for j in range(shape[1])] for i in range(shape[0])]
    return [
        [float(value.get(i, j)) for j in range(value.numCols())] for i in range(value.numRows())
    ]


def task_run(request):
    dm = COPASI.CRootContainer.addDatamodel()
    temporary = tempfile.TemporaryDirectory(prefix="chemworld52-copasi-")
    try:
        if not dm.importSBMLFromString(request["xml"]):
            raise ValueError(
                "reference SBML import failed: " + COPASI.CCopasiMessage.getAllMessageText()
            )
        dm.getModel().compileIfNecessary()
        model = dm.getModel()
        task = dm.getTask(request["task"])
        values = {p.getSBMLId(): p for p in model.getModelValues()}
        species = {m.getSBMLId(): m for m in model.getMetabolites()}
        if request.get("fixed"):
            changed = COPASI.ObjectStdVector()
            for name, value in request["fixed"].items():
                values[name].setInitialValue(value)
                changed.push_back(values[name].getInitialValueReference())
            model.updateInitialValues(changed)
        if request["task"] == "Time-Course":
            task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
            task.getMethod().getParameter("Relative Tolerance").setValue(1e-10)
            task.getMethod().getParameter("Absolute Tolerance").setValue(1e-14)
            records = []
            for time_s in request["times"]:
                if time_s == 0:
                    records.append(
                        {
                            "time": 0.0,
                            "concentrations": {
                                name: s.getInitialConcentration() for name, s in species.items()
                            },
                            "processed": True,
                        }
                    )
                else:
                    task.getProblem().setDuration(time_s)
                    task.getProblem().setStepNumber(1)
                    initialized = task.initialize(COPASI.CCopasiTask.OUTPUT_UI)
                    processed = task.process(True) if initialized else False
                    records.append(
                        {
                            "time": model.getTime(),
                            "concentrations": {
                                name: s.getConcentration() for name, s in species.items()
                            },
                            "processed": processed,
                        }
                    )
            return {
                "records": records,
                "messages": COPASI.CCopasiMessage.getAllMessageText(),
                "quantity_unit": model.getQuantityUnit(),
                "time_unit": model.getTimeUnit(),
            }
        if request["task"] in {"Parameter Estimation", "Optimization"}:
            tc = dm.getTask("Time-Course")
            tc.setMethodType(COPASI.CTaskEnum.Method_deterministic)
            tc.getProblem().setDuration(request.get("duration", 2.0))
            tc.getProblem().setStepNumber(10)
            tc.getProblem().setTimeSeriesRequested(True)
            tc.getProblem().getParameter("OutputStartTime").setValue(request.get("duration", 2.0))
            tc.getMethod().getParameter("Relative Tolerance").setValue(1e-10)
            tc.getMethod().getParameter("Absolute Tolerance").setValue(1e-14)
            problem = task.getProblem()
            task.setMethodType(
                COPASI.CTaskEnum.Method_LevenbergMarquardt
                if request["task"] == "Parameter Estimation"
                else COPASI.CTaskEnum.Method_HookeJeeves
            )
            task.getMethod().getParameter("Iteration Limit").setValue(2000)
            task.getMethod().getParameter("Tolerance").setValue(1e-10)
            if request["task"] == "Parameter Estimation":
                task.getMethod().getParameter("Modulation").setValue(1e-6)
            problem.setRandomizeStartValues(False)
            for declaration in request["variables"]:
                parameter = values[declaration["id"]]
                if request["task"] == "Parameter Estimation":
                    item = problem.addFitItem(parameter.getInitialValueReference().getCN())
                else:
                    item = problem.addOptItem(parameter.getInitialValueReference().getCN())
                item.setStartValue(declaration["initial"])
                item.setLowerBound(float(declaration["bounds"][0]))
                item.setUpperBound(float(declaration["bounds"][1]))
            if request["task"] == "Parameter Estimation":
                file = Path(temporary.name) / "observations.tsv"
                file.write_text(
                    "time\tA\n" + "".join(f"{t}\t{x}\n" for t, x in request["data"]),
                    encoding="utf-8",
                )
                experiment = COPASI.CExperiment(dm)
                experiment.setFileName(str(file))
                experiment.setSeparator("\t")
                experiment.setFirstRow(1)
                experiment.setLastRow(len(request["data"]) + 1)
                experiment.setHeaderRow(1)
                experiment.setExperimentType(COPASI.CTaskEnum.Task_timeCourse)
                experiment.setNumColumns(2)
                experiment.setNormalizeWeightsPerExperiment(False)
                mapping = experiment.getObjectMap()
                mapping.setNumCols(2)
                mapping.setRole(0, COPASI.CExperiment.time)
                mapping.setObjectCN(0, model.getValueReference().getCN().getString())
                mapping.setRole(1, COPASI.CExperiment.dependent)
                mapping.setObjectCN(1, species["A"].getConcentrationReference().getCN().getString())
                mapping.setScale(1, 1.0 / request["sigma_concentration"] ** 2)
                problem.getParameter("Experiment Set").addExperiment(experiment)
                # addExperiment copies; destroy the authored parent-owned input
                # while its datamodel still exists, before removing the model.
                del mapping
                del experiment
                problem.setResidualsRequired(True)
            else:
                problem.setSubtaskType(
                    COPASI.CTaskEnum.Task_steadyState
                    if request["mode"] == "steady"
                    else COPASI.CTaskEnum.Task_timeCourse
                )
                objective = request["objective"]
                obj = (
                    species[objective["id"]].getConcentrationReference()
                    if objective["kind"] == "concentration"
                    else values[objective["id"]].getValueReference()
                )
                problem.setObjectiveFunction("<" + obj.getCN().getString() + ">")
                problem.setMaximize(request["maximize"])
                for declaration in request["constraints"]:
                    obj = (
                        species[declaration["id"]].getConcentrationReference()
                        if declaration["kind"] == "concentration"
                        else values[declaration["id"]].getValueReference()
                    )
                    item = problem.addOptConstraint(obj.getCN())
                    if declaration.get("lower") is not None:
                        item.setLowerBound(float(declaration["lower"]))
                    if declaration.get("upper") is not None:
                        item.setUpperBound(float(declaration["upper"]))
        for group, settings in (
            (task.getProblem(), request.get("problem", {})),
            (task.getMethod(), request.get("method", {})),
        ):
            for key, value in settings.items():
                if not group.getParameter(key).setValue(value):
                    raise ValueError("reference parameter assignment failed: " + key)
        initialized = task.initialize(COPASI.CCopasiTask.OUTPUT_UI)
        processed = task.process(True) if initialized else False
        result = {
            "initialized": initialized,
            "processed": processed,
            "messages": COPASI.CCopasiMessage.getAllMessageText(),
            "model_time": model.getTime(),
            "quantity_unit": model.getQuantityUnit(),
            "time_unit": model.getTimeUnit(),
            "volume_unit": model.getVolumeUnit(),
            "avogadro": model.getAvogadro(),
            "quantity_to_number_factor": model.getQuantity2NumberFactor(),
            "species": [
                {
                    "id": m.getSBMLId(),
                    "name": m.getObjectName(),
                    "particles": m.getValue(),
                    "concentration": m.getConcentration(),
                    "volume": m.getCompartment().getValue(),
                }
                for m in model.getMetabolites()
            ],
            "reactions": [r.getSBMLId() for r in model.getReactions()],
            "problem": parameters(task.getProblem()),
            "method": parameters(task.getMethod()),
        }
        if request["task"] == "Steady-State":
            eig = task.getEigenValuesReduced()
            result.update(
                jacobian=matrix(task.getJacobian()),
                reduced_jacobian=matrix(task.getJacobianReduced()),
                steady_result=int(task.getResult()),
                eigen_api=[x for x in dir(eig) if not x.startswith("_")],
            )
            result["eigenvalues"] = [
                [float(eig.getR().get(i)), float(eig.getI().get(i))]
                for i in range(eig.getR().size())
            ]
        elif request["task"] == "Metabolic Control Analysis":
            method = task.getMethod()
            result.update(
                scaled_concentration_control=matrix(method.getScaledConcentrationCC()),
                scaled_flux_control=matrix(method.getScaledFluxCC()),
                scaled_elasticities=matrix(method.getScaledElasticities()),
                unscaled_concentration_control=matrix(method.getUnscaledConcentrationCC()),
                unscaled_flux_control=matrix(method.getUnscaledFluxCC()),
                unscaled_elasticities=matrix(method.getUnscaledElasticities()),
            )
        elif request["task"] == "Linear Noise Approximation":
            method = task.getMethod()
            result.update(
                covariance=matrix(method.getCovarianceMatrix()),
                reduced_covariance=matrix(method.getCovarianceMatrixReduced()),
            )
        elif request["task"] == "Lyapunov Exponents":
            result.update(
                exponents=[
                    float(task.exponents().get(i))
                    for i in range(task.numberOfExponentsCalculated())
                ],
                divergence=task.averageDivergence(),
                sum_exponents=task.sumOfExponents(),
                result_available=task.resultAvailable(),
            )
        elif request["task"] in {"Parameter Estimation", "Optimization"}:
            problem = task.getProblem()
            result.update(
                solution={
                    p["id"]: problem.getSolutionVariables().get(i)
                    for i, p in enumerate(request["variables"])
                },
                objective=problem.getSolutionValue(),
                function_evaluations=problem.getFunctionEvaluations(),
                constraint_evaluations=problem.getConstraintEvaluations(),
                functional_violation=problem.getFunctionalConstraintsViolation(),
                parametric_violation=problem.getParametricConstraintsViolation(),
            )
            if request["task"] == "Parameter Estimation":
                result.update(
                    residuals=list(problem.getResiduals()),
                    fisher=matrix(problem.getFisherInformation()),
                    weight=problem.getParameter("Experiment Set")
                    .getExperiment(0)
                    .getObjectMap()
                    .getScale(1),
                )
        return result
    finally:
        COPASI.CRootContainer.removeDatamodel(dm)
        temporary.cleanup()


if __name__ == "__main__":
    result = {
        "distribution": importlib.metadata.version("python-copasi"),
        "sdk": COPASI.CVersion.VERSION.getVersion(),
    }
    try:
        result.update(task_run(json.load(sys.stdin)) if "--run" in sys.argv else probe())
    except Exception:
        result["error"] = traceback.format_exc()

    def serializable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": repr(value)}
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        return value

    json.dump(serializable(result), sys.stdout, allow_nan=False)
