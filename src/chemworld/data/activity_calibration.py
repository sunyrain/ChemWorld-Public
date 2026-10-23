"""ScientificTable adapter for the shared, training-only calibration driver."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

import numpy as np

from chemworld.data.calibration_core import (
    BoundedParameterCoordinates,
    ForwardEvaluationError,
    MeasurementCalibration,
)
from chemworld.foundation.units import convert_value
from chemworld.physchem.activity_kernels import observable
from chemworld.physchem.activity_thermodynamics import (
    active_activity_parameters,
    activity_parameter_jacobian,
    validated_state,
)
from chemworld.physchem.equilibrium import ActivityModelSpec


@dataclass(frozen=True)
class ActivityFitParameter(BoundedParameterCoordinates):
    name: str
    bounds: tuple[float, float]
    initial: float
    coordinate: str = "linear"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
        ):
            raise ValueError("activity fit parameter requires an explicit parameter name")
        object.__setattr__(self, "bounds", tuple(self.bounds))
        if len(self.bounds) != 2:
            raise ValueError("parameter bounds require two endpoints")
        self._validate_coordinates()

    @property
    def parameter_id(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.name,
            "bounds": list(self.bounds),
            "initial": self.initial,
            "coordinate": self.coordinate,
            "basis": "declared activity equation coefficient",
        }


class ActivityCalibration(MeasurementCalibration):
    """Fit explicit existing parameters; holdout responses never enter the optimizer.

    Conditions are exactly ``temperature`` and ``x:<component>`` for every
    component. Observation quantity IDs are GE/HE/SE/CpE or ln_gamma:<component>.
    Complete composition is normalized by the same existing activity consumer.
    """

    schema_version = "chemworld-activity-calibration-1"
    jacobian_description = "analytic activity parameters"

    def __init__(
        self,
        table: Any,
        model: ActivityModelSpec,
        parameters: Any,
        *,
        fitted_model_id: str,
        max_forward_groups: int = 20000,
        progress: Any = None,
    ) -> None:
        super().__init__(
            table, parameters, max_forward_groups=max_forward_groups, progress=progress
        )
        if not isinstance(model, ActivityModelSpec):
            raise TypeError("activity calibration requires ActivityModelSpec")
        if (
            not isinstance(fitted_model_id, str)
            or not fitted_model_id.strip()
            or fitted_model_id != fitted_model_id.strip()
            or fitted_model_id == model.model_id
        ):
            raise ValueError("fitted model needs a new explicit model identity")
        self.model = replace(model, parameters=dict(model.parameters))
        self.fitted_model_id = fitted_model_id
        active = active_activity_parameters(model)
        if any(
            not isinstance(p, ActivityFitParameter) or p.name not in active for p in self.parameters
        ):
            raise ValueError("fit parameters must be active explicit activity coefficients")
        condition_names = {"temperature", *(f"x:{key}" for key in model.component_ids)}
        if {
            name for name, col in self.columns.items() if col.role == "condition"
        } != condition_names:
            raise ValueError("activity conditions must be temperature and all component fractions")
        convert_value(1, self.columns["temperature"].unit, "K")
        for key in model.component_ids:
            convert_value(1, self.columns[f"x:{key}"].unit, "dimensionless")
        self.observation_units = {
            "GE": "J/mol",
            "HE": "J/mol",
            "SE": "J/(mol*K)",
            "CpE": "J/(mol*K)",
            **{f"ln_gamma:{key}": "dimensionless" for key in model.component_ids},
        }
        self.observations = tuple(
            name for name, col in self.columns.items() if col.role == "observation"
        )
        if set(self.observations) - self.observation_units.keys():
            raise ValueError("unsupported activity observation")
        self.scales = {
            name: convert_value(1, self.observation_units[name], self.columns[name].unit)
            for name in self.observations
        }
        self.jacobian_runs: list[dict[str, Any]] = []
        self.candidate(self.initial_values)

    def candidate(self, values: dict[str, float]) -> ActivityModelSpec:
        if set(values) != {p.name for p in self.parameters}:
            raise ValueError("candidate must match the explicit fitted parameter whitelist")
        for parameter in self.parameters:
            parameter.encode(values[parameter.name])
        return replace(self.model, parameters={**self.model.parameters, **values})

    def _conditions(self, row: Any) -> tuple[float, dict[str, float]]:
        return row.values["temperature"].to("K").value, {
            key: row.values[f"x:{key}"].to("dimensionless").value
            for key in self.model.component_ids
        }

    def forward(
        self, values: dict[str, float], split: str, *, stage: str = "prediction"
    ) -> dict[tuple[str, str], float]:
        predictions: dict[tuple[str, str], float] = {}
        groups: dict[str, list[Any]] = {}
        for row_id in dict.fromkeys(row_id for row_id, _ in self.pairs.get(split, ())):
            row = self.rows[row_id]
            groups.setdefault(row.group_id, []).append(row)
        for group, rows in groups.items():
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared forward group execution budget exhausted")
            record: dict[str, Any] = {
                "group_id": group,
                "split": split,
                "stage": stage,
                "parameters": dict(values),
                "readouts": [],
                "failure": None,
            }
            self.runs.append(record)
            try:
                model = self.candidate(values)
                for row in rows:
                    temperature, composition = self._conditions(row)
                    xs = validated_state(model, composition, temperature)
                    for quantity in self.observations:
                        if (row.row_id, quantity) not in self.pairs.get(split, ()):
                            continue
                        reading = {
                            "row_id": row.row_id,
                            "quantity": quantity,
                            "temperature_K": temperature,
                            "composition": composition,
                            "value": None,
                            "unit": self.columns[quantity].unit,
                        }
                        record["readouts"].append(reading)
                        value = observable(model, xs, temperature, quantity) * self.scales[quantity]
                        if not isfinite(value):
                            raise ValueError("nonfinite activity prediction")
                        reading["value"] = value
                        predictions[(row.row_id, quantity)] = value
            except (ValueError, ArithmeticError) as exc:
                record["failure"] = f"{type(exc).__name__}: {exc}"
                raise ForwardEvaluationError(record["failure"]) from exc
            finally:
                if self.progress is not None:
                    self.progress(
                        {
                            "stage": stage,
                            "completed_forward_groups": len(self.runs),
                            "forward_limit": self.max_forward_groups,
                            "group_id": group,
                            "failure": record["failure"],
                        }
                    )
        return predictions

    def training_jacobian(self, coordinates: Any) -> Any:
        values = self.values(coordinates)
        record = {"split": "train", "parameters": values, "rows": [], "failure": None}
        self.jacobian_runs.append(record)
        try:
            model = self.candidate(values)
            names = tuple(p.name for p in self.parameters)
            scales = [
                p.decode_derivative(x) for p, x in zip(self.parameters, coordinates, strict=True)
            ]
            matrix = []
            for row_id, quantity in self.pairs["train"]:
                row = self.rows[row_id]
                temperature, composition = self._conditions(row)
                source = activity_parameter_jacobian(
                    model,
                    composition,
                    temperature_K=temperature,
                    quantities=(quantity,),
                    parameters=names,
                )[quantity]
                weighted = [
                    source[name] * scale * self.scales[quantity] / row.values[quantity].uncertainty
                    for name, scale in zip(names, scales, strict=True)
                ]
                if not all(isfinite(v) for v in weighted):
                    raise ValueError("nonfinite weighted analytic activity Jacobian")
                record["rows"].append(
                    {"row_id": row_id, "quantity": quantity, "weighted_jacobian": weighted}
                )
                matrix.append(weighted)
            return np.asarray(matrix)
        except (ValueError, ArithmeticError) as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise ForwardEvaluationError(record["failure"]) from exc

    def _optimizer_jacobian(self) -> Any:
        return self.training_jacobian

    def _resource_report(self) -> dict[str, int]:
        return {
            "forward_groups": len(self.runs),
            "runtime_sessions": 0,
            "property_evaluations": sum(len(r["readouts"]) for r in self.runs),
            "jacobian_calls": len(self.jacobian_runs),
            "jacobian_rows": sum(len(r["rows"]) for r in self.jacobian_runs),
        }

    def fit(self, **options: Any) -> dict[str, Any]:
        result = super().fit(**options)
        result.update(input_model=self.model.to_dict(), jacobian_runs=self.jacobian_runs)
        if result["status"] == "fitted":
            result["fitted_model"] = replace(
                self.candidate(result["estimate"]), model_id=self.fitted_model_id
            ).to_dict()
        return result


__all__ = ["ActivityCalibration", "ActivityFitParameter"]
