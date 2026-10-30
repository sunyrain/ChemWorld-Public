"""Bounded calibration by replaying public measurements in fresh native sessions.

This is a boundary analysis tool. It owns no physical state, reward objective,
hidden readout or optimizer inside the runtime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isclose, isfinite

from chemworld.data.calibration_core import (
    BoundedParameterCoordinates,
    ForwardEvaluationError,
    MeasurementCalibration,
)
from chemworld.data.scientific_inputs import QuantityColumn
from chemworld.foundation.units import convert_value
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec


@dataclass(frozen=True)
class CalibrationParameter(BoundedParameterCoordinates):
    column: QuantityColumn
    path: tuple[str, ...]
    initial: float
    coordinate: str = "log"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        if self.column.role != "parameter" or self.coordinate not in {"linear", "log"}:
            raise ValueError("calibration requires parameter columns and linear/log coordinates")
        self._validate_coordinates()
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

    @property
    def bounds(self):
        return self.column.valid_range

    @property
    def parameter_id(self):
        return self.column.quantity_id

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

    def __post_init__(self) -> None:
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


class NativeCalibration(MeasurementCalibration):
    def __init__(self, table, parameters, experiments, *, max_forward_groups=300, progress=None):
        super().__init__(table, parameters, max_forward_groups=max_forward_groups,
                         progress=progress)
        parameters = self.parameters
        self.experiments = tuple(
            NativeCalibrationExperiment(
                e.group_id,
                WorldCompositionSpec.from_dict(
                    e.composition.to_dict(), modules=e.composition.modules,
                ),
                e.actions, e.readouts, e.seed,
            ) for e in experiments
        )
        if len({self._binding_identity(p) for p in parameters}) != len(parameters):
            raise ValueError("two parameters cannot write the same physical model field")
        groups = {row.group_id for row in table.rows}
        if (
            len({e.group_id for e in experiments}) != len(experiments)
            or {e.group_id for e in experiments} != groups
        ):
            raise ValueError("fixed experiments must cover each measurement group exactly once")
        seen = set()
        for experiment in self.experiments:
            # Check all paths and ordinary model constructors before optimization.
            self._candidate(experiment, self.initial_values)
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

    def _binding_identity(self, parameter):
        """The unique declared model field, specialized by named-control adapters."""
        return parameter.path

    def _candidate(self, experiment, values):
        """Prepare a candidate without changing the one public-history executor."""
        return parameterized_world(experiment, self.parameters, values), experiment.actions

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
                spec, actions = self._candidate(experiment, values)
                session = ChemWorldRuntime(composition=spec, debug_truth=False)
                session.reset(seed=experiment.seed)
                for index, action in enumerate(actions):
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
                        and index < len(actions) - 1
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
