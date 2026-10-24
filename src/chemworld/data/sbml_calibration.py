"""Named SBML control calibration through the original public-history executor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isclose
from typing import Any

import numpy as np

from chemworld.data.calibration import NativeCalibration, NativeCalibrationExperiment
from chemworld.data.calibration_core import BoundedParameterCoordinates, ForwardEvaluationError
from chemworld.data.scientific_inputs import QuantityColumn
from chemworld.foundation.units import convert_value
from chemworld.physchem.sbml_sensitivity import (
    check_sensitivity_scope,
    sbml_trajectory_sensitivities,
)
from chemworld.physchem.sbml_system import _parameter_signature, parameter_observable
from chemworld.runtime.sbml_models import validate_model_state


@dataclass(frozen=True)
class SBMLCalibrationParameter(BoundedParameterCoordinates):
    column: QuantityColumn
    source_id: str
    initial: float
    coordinate: str = "log"

    def __post_init__(self) -> None:
        if (
            self.column.role != "parameter"
            or not isinstance(self.source_id, str)
            or (not self.source_id or self.source_id != self.source_id.strip())
        ):
            raise ValueError("SBML calibration needs an explicit named parameter column")
        self._validate_coordinates()

    @property
    def parameter_id(self) -> str:
        return self.column.quantity_id

    @property
    def bounds(self) -> tuple[float, float]:
        return self.column.valid_range

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.column.to_dict(),
            "source_id": self.source_id,
            "initial": self.initial,
            "coordinate": self.coordinate,
            "binding": "time_zero_set_model_parameters_after_initialization",
        }


class SBMLCalibration(NativeCalibration):
    """Fit declared controls while preserving fixed initial recipes and histories.

    Every objective evaluation still executes actual set/wait/measure actions in
    a fresh original runtime and reads public packets. Tangents are an analysis
    of that deterministic prediction function, not a substitute observation.
    """

    schema_version = "chemworld-sbml-control-calibration-1"
    jacobian_description = "original MathML analytic forward tangents"

    def __init__(
        self,
        table: Any,
        parameters: Any,
        experiments: Any,
        *,
        max_forward_groups: int = 300,
        progress: Any = None,
    ) -> None:
        parameters = tuple(parameters)
        if not parameters or any(not isinstance(p, SBMLCalibrationParameter) for p in parameters):
            raise ValueError("SBML calibration requires named SBMLCalibrationParameter bindings")
        self.jacobian_runs: list[dict[str, Any]] = []
        super().__init__(
            table,
            parameters,
            tuple(experiments),
            max_forward_groups=max_forward_groups,
            progress=progress,
        )
        for experiment in self.experiments:
            model, _ = self._model_initial(experiment)
            channels = {o.channel_id: o for o in model.observables}
            times = self._times(experiment)
            for readout in experiment.readouts:
                if readout.channel_id not in channels or readout.time_column is None:
                    raise ValueError("SBML readouts require a declared channel and sampling time")
                convert_value(
                    1.0, channels[readout.channel_id].unit, self.columns[readout.quantity_id].unit
                )
                expected = self.rows[readout.row_id].values[readout.time_column].to("s").value
                if not isclose(times[readout.step], expected, rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("SBML data time differs from its fixed native history")

    def _binding_identity(self, parameter: SBMLCalibrationParameter) -> str:
        return parameter.source_id

    def _model_initial(self, experiment: NativeCalibrationExperiment) -> tuple[Any, Any]:
        physics = experiment.composition.physics
        if physics is None or len(physics.models_of("sbml")) != 1:
            raise ValueError("SBML calibration requires one original formal model workspace")
        vessel, model = next(iter(physics.models_of("sbml").items()))
        if set(physics.initial_states) != {vessel}:
            raise ValueError("SBML calibration supports one fixed formal workspace")
        state = physics.initial_states[vessel]
        if state.ledger.time_s != 0:
            raise ValueError("control calibration starts at the authored time-zero state")
        _, initial, _ = validate_model_state(state, model)
        check_sensitivity_scope(model, tuple(p.source_id for p in self.parameters))
        if model.readout_relative_sd != 0:
            raise ValueError("analytic prediction calibration requires a zero-noise formal gauge")
        return model, initial

    def _source_values(self, model: Any, values: dict[str, float]) -> dict[str, float]:
        if set(values) != {p.parameter_id for p in self.parameters}:
            raise ValueError("candidate must contain exactly the named parameter whitelist")
        result = {}
        for parameter in self.parameters:
            value = values[parameter.parameter_id]
            parameter.encode(value)
            coherent = parameter_observable(model.model, parameter.source_id).unit
            result[parameter.source_id] = convert_value(value, parameter.column.unit, coherent) / (
                _parameter_signature(model.model, parameter.source_id).si_factor
            )
        return result

    def _candidate(
        self, experiment: NativeCalibrationExperiment, values: dict[str, float]
    ) -> tuple[Any, tuple[dict, ...]]:
        model, initial = self._model_initial(experiment)
        actions = deepcopy(experiment.actions)
        names = {p.source_id for p in self.parameters}
        if (
            actions[0].get("operation") != "set_model_parameters"
            or (set(actions[0].get("values", {})) != names)
            or any(a.get("operation") not in {"wait", "measure"} for a in actions[1:])
        ):
            raise ValueError(
                "history requires one initial parameter control then fixed wait/measure"
            )
        controls = self._source_values(model, values)
        model.frame.controlled_values(initial, 0.0, controls)
        actions[0]["values"] = controls
        self._times(experiment)
        return experiment.composition, actions

    @staticmethod
    def _times(experiment: NativeCalibrationExperiment) -> dict[int, float]:
        times, elapsed = {}, 0.0
        for index, action in enumerate(experiment.actions):
            if action.get("operation") == "wait":
                duration = action.get("duration_s")
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (float, int))
                    or (not np.isfinite(duration) or duration <= 0)
                ):
                    raise ValueError("SBML history waits need finite positive durations")
                elapsed += duration
            elif action.get("operation") == "measure":
                if action.get("instrument") != "formal_quantity_meter":
                    raise ValueError("SBML history requires the declared formal quantity meter")
                times[index] = elapsed
        return times

    def sensitivity(
        self, experiment: NativeCalibrationExperiment, values: dict[str, float]
    ) -> dict[str, Any]:
        """Analyze a fixed history without mutating it or charging live resources."""
        self._candidate(experiment, values)
        model, initial = self._model_initial(experiment)
        return sbml_trajectory_sensitivities(
            model,
            initial,
            self._source_values(model, values),
            sorted(set(self._times(experiment).values())),
        )

    def training_jacobian(self, coordinates: Any) -> np.ndarray:
        values = self.values(coordinates)
        record: dict[str, Any] = {"parameters": values, "groups": [], "rows": [], "failure": None}
        self.jacobian_runs.append(record)
        entries: dict[tuple[str, str], list[float]] = {}
        try:
            for experiment in self.experiments:
                group_rows = [r for r in self.table.rows if r.group_id == experiment.group_id]
                if group_rows[0].split != "train":
                    continue
                analysis = self.sensitivity(experiment, values)
                record["groups"].append({"group_id": experiment.group_id, "analysis": analysis})
                if analysis["status"] != "complete":
                    raise ValueError(
                        "original SBML tangent analysis failed: " + analysis["failure"]
                    )
                outputs = {row["time_s"]: row["readouts"] for row in analysis["records"]}
                times = self._times(experiment)
                model, _ = self._model_initial(experiment)
                for readout in experiment.readouts:
                    key = (readout.row_id, readout.quantity_id)
                    if key not in self.pairs["train"]:
                        continue
                    reading = outputs[times[readout.step]][readout.channel_id]
                    observed = self.rows[readout.row_id].values[readout.quantity_id]
                    response_scale = convert_value(1.0, reading["unit"], observed.unit) - (
                        convert_value(0.0, reading["unit"], observed.unit)
                    )
                    entries[key] = []
                    for parameter, coordinate in zip(self.parameters, coordinates, strict=True):
                        coherent = parameter_observable(model.model, parameter.source_id).unit
                        source_scale = (
                            convert_value(1.0, parameter.column.unit, coherent)
                            - convert_value(0.0, parameter.column.unit, coherent)
                        ) / _parameter_signature(model.model, parameter.source_id).si_factor
                        derivative = reading["derivative"][parameter.source_id]
                        entries[key].append(
                            derivative
                            * source_scale
                            * response_scale
                            * parameter.decode_derivative(coordinate)
                            / observed.uncertainty
                        )
            matrix = np.asarray([entries[key] for key in self.pairs["train"]])
            if not np.all(np.isfinite(matrix)):
                raise ValueError("nonfinite weighted SBML trajectory Jacobian")
            record["rows"] = [
                {"row_id": key[0], "quantity": key[1], "weighted_jacobian": entries[key]}
                for key in self.pairs["train"]
            ]
            return matrix
        except (ValueError, ArithmeticError, RuntimeError) as error:
            record["failure"] = f"{type(error).__name__}: {error}"
            raise ForwardEvaluationError(record["failure"]) from error

    def _optimizer_jacobian(self) -> Any:
        return self.training_jacobian

    def _resource_report(self) -> dict[str, Any]:
        return {
            **super()._resource_report(),
            "jacobian_calls": len(self.jacobian_runs),
            "tangent_groups": sum(len(r["groups"]) for r in self.jacobian_runs),
            "tangent_rhs_evaluations": sum(
                g["analysis"]["rhs_evaluations"] for r in self.jacobian_runs for g in r["groups"]
            ),
        }

    def fit(self, **options: Any) -> dict[str, Any]:
        result = super().fit(**options)
        result.update(
            jacobian_runs=self.jacobian_runs,
            initialization="author_initial_state_then_time_zero_control",
            prediction="zero_noise_public_native_measurement_packets",
        )
        return result


__all__ = ["SBMLCalibration", "SBMLCalibrationParameter"]
