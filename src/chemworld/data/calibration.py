"""Bounded calibration by replaying public measurements in fresh native sessions.

This is a boundary analysis tool. It owns no physical state, reward objective,
hidden readout or optimizer inside the runtime.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from math import exp, isclose, isfinite, log

import numpy as np
from scipy.optimize import least_squares

from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable
from chemworld.foundation.units import convert_value
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec


@dataclass(frozen=True)
class CalibrationParameter:
    column: QuantityColumn
    path: tuple[str, ...]
    initial: float
    coordinate: str = "log"

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(self.path))
        if self.column.role != "parameter" or self.coordinate not in {"linear", "log"}:
            raise ValueError("calibration requires parameter columns and linear/log coordinates")
        lower, upper = self.column.valid_range
        if isinstance(self.initial, bool) or not isfinite(self.initial) \
                or not lower <= self.initial <= upper:
            raise ValueError("initial parameter must lie inside its finite declared bounds")
        if self.coordinate == "log" and lower <= 0:
            raise ValueError("log parameters require strictly positive bounds")
        path = self.path
        registered = (
            len(path) >= 6 and path[:2] == ("physics", "models") and path[3] == "bindings"
        )
        equipment = (
            len(path) >= 8 and path[:2] == ("physics", "initial_states")
            and path[3] == "equipment" and path[5:7] == ("settings", "physical_model")
        )
        if not (registered or equipment):
            raise ValueError("only declared model or initial equipment model parameters may vary")

    def encode(self, value):
        lower, upper = self.column.valid_range
        if not isfinite(value) or not lower <= value <= upper:
            raise ValueError("parameter is outside its finite declared domain")
        if self.coordinate == "log":
            value, lower, upper = log(value), log(lower), log(upper)
        return (value - lower) / (upper - lower)

    def decode(self, coordinate):
        if not isfinite(coordinate) or not 0 <= coordinate <= 1:
            raise ValueError("normalized parameter coordinate must be finite and inside [0, 1]")
        lower, upper = self.column.valid_range
        if self.coordinate == "log":
            value = exp(log(lower) + coordinate * (log(upper) - log(lower)))
        else:
            value = lower + coordinate * (upper - lower)
        # Floating-point endpoint reconstruction must not extend the domain.
        return min(upper, max(lower, value))

    def to_dict(self):
        return {
            **self.column.to_dict(), "configuration_path": list(self.path),
            "initial": self.initial, "coordinate": self.coordinate,
        }


@dataclass(frozen=True)
class CalibrationReadout:
    row_id: str
    quantity_id: str
    step: int
    channel_id: str
    time_column: str | None = "time"


@dataclass(frozen=True)
class NativeCalibrationExperiment:
    group_id: str
    composition: WorldCompositionSpec
    actions: tuple[dict, ...]
    readouts: tuple[CalibrationReadout, ...]
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "actions", tuple(deepcopy(self.actions)))
        object.__setattr__(self, "readouts", tuple(self.readouts))
        if not self.group_id or not self.actions or not self.readouts:
            raise ValueError("a native calibration experiment requires a fixed measured history")
        if any(
            isinstance(r.step, bool) or not isinstance(r.step, int)
            or not 0 <= r.step < len(self.actions)
            or self.actions[r.step].get("operation") != "measure"
            for r in self.readouts
        ):
            raise ValueError("each readout must name an actual measurement step")


def parameterized_world(experiment, parameters, values):
    """Change only explicit numeric configuration leaves, then use normal codecs."""
    expected = {parameter.column.quantity_id for parameter in parameters}
    if set(values) != expected:
        raise ValueError("candidate must contain exactly the parameter whitelist")
    payload = experiment.composition.to_dict()
    for parameter in parameters:
        value = values[parameter.column.quantity_id]
        parameter.encode(value)
        node = payload
        for part in parameter.path[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError("parameter path is absent from the fixed composition")
            node = node[part]
        leaf = parameter.path[-1]
        if not isinstance(node, dict) or leaf not in node or not isinstance(node[leaf], float):
            raise ValueError("calibration targets must be existing continuous numeric model leaves")
        node[leaf] = float(value)
    return WorldCompositionSpec.from_dict(payload, modules=experiment.composition.modules)


class ForwardEvaluationError(ValueError):
    """An executed candidate could not provide the declared public measurements."""


class NativeCalibration:
    def __init__(self, table, parameters, experiments, *, max_forward_groups=300, progress=None):
        if not isinstance(table, ScientificTable) or table.kind != "measurements":
            raise ValueError("calibration requires a scientific measurement table")
        self.table = table
        self.parameters = tuple(parameters)
        self.experiments = tuple(
            NativeCalibrationExperiment(
                e.group_id,
                WorldCompositionSpec.from_dict(
                    e.composition.to_dict(), modules=e.composition.modules,
                ),
                e.actions, e.readouts, e.seed,
            ) for e in experiments
        )
        if (
            not self.parameters
            or len({p.column.quantity_id for p in parameters}) != len(parameters)
        ):
            raise ValueError("calibration parameters must be nonempty and unique")
        if len({p.path for p in parameters}) != len(parameters):
            raise ValueError("two parameters cannot write the same physical model field")
        if isinstance(max_forward_groups, bool) or not isinstance(max_forward_groups, int) \
                or max_forward_groups <= 0:
            raise ValueError("forward execution limit must be a positive integer")
        self.max_forward_groups = max_forward_groups
        self.progress = progress
        self.runs = []
        self.columns = {c.quantity_id: c for c in table.columns}
        self.rows = {row.row_id: row for row in table.rows}
        self.excluded = []
        self.pairs = {}
        groups = {row.group_id for row in table.rows}
        if (
            len({e.group_id for e in experiments}) != len(experiments)
            or {e.group_id for e in experiments} != groups
        ):
            raise ValueError("fixed experiments must cover each measurement group exactly once")
        seen = set()
        for experiment in self.experiments:
            # Check all paths and ordinary model constructors before optimization.
            parameterized_world(experiment, parameters, self.initial_values)
            for readout in experiment.readouts:
                key = (readout.row_id, readout.quantity_id)
                row = self.rows.get(readout.row_id)
                column = self.columns.get(readout.quantity_id)
                if (
                    key in seen or row is None or row.group_id != experiment.group_id
                    or column is None or column.role != "observation"
                ):
                    raise ValueError("readouts must map distinct observations in their own group")
                seen.add(key)
                if readout.time_column is not None:
                    time = self.columns.get(readout.time_column)
                    if time is None or time.role != "condition":
                        raise ValueError("readout time requires a declared condition column")
                    convert_value(row.values[time.quantity_id].value, time.unit, "s")
        expected = {
            (row.row_id, column.quantity_id)
            for row in table.rows for column in table.columns if column.role == "observation"
        }
        if seen != expected:
            raise ValueError("fixed readouts must cover every observation, including invalid data")
        for row in table.rows:
            for column in table.columns:
                if column.role != "observation":
                    continue
                value = row.values[column.quantity_id]
                if value.quality != "valid":
                    self.excluded.append({
                        "row_id": row.row_id, "quantity_id": column.quantity_id,
                        "split": row.split, "quality": value.quality,
                    })
                    continue
                if value.uncertainty is None or value.uncertainty <= 0:
                    raise ValueError(
                        "weighted fitting requires positive declared standard uncertainty"
                    )
                self.pairs.setdefault(row.split, []).append((row.row_id, column.quantity_id))
        if not self.pairs.get("train"):
            raise ValueError("at least one valid training measurement is required")

    @property
    def initial_values(self):
        return {p.column.quantity_id: p.initial for p in self.parameters}

    def values(self, coordinates):
        return {p.column.quantity_id: p.decode(x) for p, x in zip(
            self.parameters, coordinates, strict=True,
        )}

    def forward(self, values, split, *, stage="prediction"):
        predictions = {}
        for experiment in self.experiments:
            group_rows = [r for r in self.table.rows if r.group_id == experiment.group_id]
            if group_rows[0].split != split:
                continue
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared forward group execution budget exhausted")
            record = {
                "group_id": experiment.group_id, "split": split, "stage": stage,
                "parameters": dict(values), "seed": experiment.seed, "steps": [],
                "readouts": [], "failure": None,
            }
            self.runs.append(record)
            session = None
            try:
                spec = parameterized_world(experiment, self.parameters, values)
                session = ChemWorldRuntime(composition=spec, debug_truth=False)
                session.reset(seed=experiment.seed)
                for index, action in enumerate(experiment.actions):
                    result = session.execute(deepcopy(action))
                    info = result.info
                    acquisition = info.get("outcome_diagnostics", {}).get("acquisition", {})
                    record["steps"].append({
                        "index": index, "action": deepcopy(action),
                        "transaction_status": info["transaction_status"],
                        "operation_outcome": info["operation_outcome"],
                        "measurement_status": info.get("measurement_status"),
                        "actual_duration_s": info["actual_duration_s"],
                        "analysis_effort_s": acquisition.get("analysis_effort_s", 0.0),
                        "cost_delta": info["cost_delta"], "sample_delta_L": info["sample_delta"],
                    })
                    if info["transaction_status"] != "committed" \
                            or info["operation_outcome"] != "success":
                        raise ForwardEvaluationError(
                            "native operation did not complete the fixed history"
                        )
                    if (
                        (result.truncated or result.terminated)
                        and index < len(experiment.actions) - 1
                    ):
                        raise ForwardEvaluationError(
                            "native session ended before the fixed history"
                        )
                    for readout in (r for r in experiment.readouts if r.step == index):
                        channels = {c.channel_id: c for c in result.observation.channels}
                        channel = channels.get(readout.channel_id)
                        column = self.columns[readout.quantity_id]
                        reading = {
                            "row_id": readout.row_id, "quantity_id": readout.quantity_id,
                            "value": None, "unit": column.unit,
                            "native": None if channel is None else channel.to_dict(),
                        }
                        # Preserve the actual returned packet even when its
                        # unit, quality or time cannot enter the objective.
                        record["readouts"].append(reading)
                        if info["measurement_status"] != "valid":
                            raise ForwardEvaluationError(
                                "native acquisition did not return a valid packet"
                            )
                        if channel is None or channel.quality != "valid" \
                                or channel.value is None or channel.shape:
                            raise ForwardEvaluationError(
                                "expected finite scalar readout is unavailable"
                            )
                        if readout.time_column is not None:
                            time = self.rows[readout.row_id].values[readout.time_column].to("s")
                            if not isclose(channel.sampled_at.time_s, time.value,
                                           rel_tol=1e-10, abs_tol=1e-8):
                                raise ForwardEvaluationError(
                                    "public sampling time differs from fixed data"
                                )
                        predicted = convert_value(channel.value, channel.unit, column.unit)
                        if not isfinite(predicted):
                            raise ForwardEvaluationError("public prediction must be finite")
                        predictions[(readout.row_id, readout.quantity_id)] = predicted
                        reading["value"] = predicted
            except (ValueError, ArithmeticError) as exc:
                record["failure"] = f"{type(exc).__name__}: {exc}"
                raise ForwardEvaluationError(record["failure"]) from exc
            finally:
                if session is not None:
                    session.close()
                if self.progress is not None:
                    self.progress({
                        "stage": stage, "completed_forward_groups": len(self.runs),
                        "forward_limit": self.max_forward_groups,
                        "group_id": experiment.group_id, "failure": record["failure"],
                    })
        return predictions

    def residuals(self, predictions, split):
        return np.asarray([
            (predictions[key] - self.rows[key[0]].values[key[1]].value)
            / self.rows[key[0]].values[key[1]].uncertainty
            for key in self.pairs.get(split, ())
        ])

    def metrics(self, predictions, split):
        residuals = self.residuals(predictions, split)
        details = []
        for key in self.pairs.get(split, ()):
            measured = self.rows[key[0]].values[key[1]]
            details.append({
                "row_id": key[0], "quantity_id": key[1], "unit": measured.unit,
                "measured": measured.value, "predicted": predictions[key],
                "error": predictions[key] - measured.value, "uncertainty": measured.uncertainty,
            })
        return {
            "n_valid": len(details),
            "n_excluded": sum(e["split"] == split for e in self.excluded),
            "weighted_rmse": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
            "sum_squared_weighted_residuals": float(residuals @ residuals),
            "measurements": details,
        }

    def fit(self, *, max_nfev=30, diff_step=1e-3, rank_rtol=1e-5, rank_atol=1e-8):
        if self.runs:
            raise ValueError("use a fresh calibration object for each fixed fitting block")
        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("optimizer limit must be a positive integer")
        if not all(isfinite(v) and v > 0 for v in (diff_step, rank_rtol, rank_atol)):
            raise ValueError("derivative and rank tolerances must be positive and finite")
        report = {
            "schema_version": "chemworld-native-calibration-1",
            "dataset": self.table.summary(),
            "sources": [source.to_dict() for source in self.table.sources],
            "parameters": [p.to_dict() for p in self.parameters],
            "optimization_split": "train", "heldout_split": "validation",
            "excluded_observations": self.excluded,
            "optimizer": {
                "method": "TRF weighted least squares", "jacobian": "3-point",
                "diff_step": diff_step, "max_nfev": max_nfev,
                "max_forward_groups": self.max_forward_groups,
                "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8,
            },
            "runs": self.runs, "status": "not_started", "failures": [],
        }
        try:
            def training_residual(x):
                return self.residuals(self.forward(self.values(x), "train", stage="fit"), "train")

            result = least_squares(
                training_residual,
                [p.encode(p.initial) for p in self.parameters], bounds=(0.0, 1.0),
                method="trf", jac="3-point", diff_step=diff_step, max_nfev=max_nfev,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
            report["optimizer"].update(
                success=bool(result.success), message=result.message, nfev=result.nfev,
                njev=result.njev, optimality=float(result.optimality),
            )
            report["estimate"] = self.values(result.x)
            report["active_bounds"] = {
                p.column.quantity_id: int(active)
                for p, active in zip(self.parameters, result.active_mask, strict=True)
            }
            singular_values = np.linalg.svd(result.jac, compute_uv=False)
            threshold = max(rank_atol, rank_rtol * float(singular_values[0]))
            rank = int(np.count_nonzero(singular_values > threshold))
            report["identifiability"] = {
                "rank": rank, "parameter_count": len(self.parameters),
                "singular_values": singular_values.tolist(), "rank_threshold": threshold,
                "status": "full_rank_local" if rank == len(self.parameters) else "rank_deficient",
                "weighted_jacobian": result.jac.tolist(),
                "coordinate": "dimensionless bounded linear/log parameters",
                "boundary": "local sensitivity; no global uniqueness or material qualification",
            }
            if not result.success:
                report.update(status="optimizer_not_converged")
                report["failures"].append(result.message)
            else:
                report["training"] = self.metrics(
                    self.forward(report["estimate"], "train", stage="training_final"), "train",
                )
                report["validation"] = self.metrics(
                    self.forward(report["estimate"], "validation", stage="heldout"), "validation",
                )
                report["status"] = "fitted"
        except ForwardEvaluationError as exc:
            report.update(status="forward_failed")
            report["failures"].append(str(exc))
        steps = [step for run in self.runs for step in run["steps"]]
        report["resources"] = {
            "forward_groups": len(self.runs), "attempted_operations": len(steps),
            "committed_operations": sum(s["transaction_status"] == "committed" for s in steps),
            **{key: sum(s[key] for s in steps) for key in (
                "actual_duration_s", "analysis_effort_s", "cost_delta", "sample_delta_L",
            )},
        }
        # Results must be usable as strict, readable JSON without hidden objects.
        json.dumps(report, allow_nan=False)
        return report
