"""Sourced electronic scaling and free-energy BEP fits on the shared fitter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from chemworld.data.calibration_core import (
    BoundedParameterCoordinates,
    ForwardEvaluationError,
    MeasurementCalibration,
)
from chemworld.data.calibration_uncertainty import known_sigma_parameter_uncertainty
from chemworld.data.scientific_inputs import ScientificTable
from chemworld.foundation.units import convert_value
from chemworld.physchem.catalysis_descriptors import (
    FreeEnergyBEP,
    LinearEnergyScaling,
    validate_source,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance


@dataclass(frozen=True)
class CatalyticEnergyFitParameter(BoundedParameterCoordinates):
    name: str
    bounds: tuple[float, float]
    initial: float
    unit: str
    coordinate: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(self.bounds))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name.strip() != self.name
        ):
            raise ValueError("energy fit parameter requires an explicit name")
        if self.unit not in {"J/mol", "dimensionless"}:
            raise ValueError("energy fit parameters use J/mol or dimensionless coordinates")
        self._validate_coordinates()

    @property
    def parameter_id(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds": list(self.bounds),
            "initial": self.initial,
            "unit": self.unit,
            "coordinate": self.coordinate,
        }


class CatalyticEnergyCalibration(MeasurementCalibration):
    """Fit only declared coefficients; energies remain the original native laws.

    Electronic-energy data bind to LinearEnergyScaling. BEP data are activation
    *free* energies against reaction free energies in an explicitly shared
    reference frame. No electronic-to-free-energy conversion is inferred.
    """

    schema_version = "chemworld-catalytic-energy-calibration-1"
    jacobian_description = "analytic native energy-law coefficients"

    def __init__(
        self,
        table: ScientificTable,
        model: LinearEnergyScaling | FreeEnergyBEP,
        parameters: Sequence[CatalyticEnergyFitParameter],
        *,
        fitted_source: DataSourceProvenance,
        reference_frame: str,
        energy_kind: str,
        noise_model: str,
        observation_column: str = "energy",
        descriptor_columns: Mapping[str, str] | None = None,
        delta_g_column: str = "delta_g",
        max_forward_groups: int = 6000,
        progress: Any = None,
    ) -> None:
        super().__init__(
            table, parameters, max_forward_groups=max_forward_groups, progress=progress
        )
        from chemworld.data.calibration_uncertainty import KNOWN_INDEPENDENT_SIGMA

        if noise_model != KNOWN_INDEPENDENT_SIGMA:
            raise ValueError("energy table fitting requires explicit independent known sigma")
        if not isinstance(reference_frame, str) or not reference_frame.strip():
            raise ValueError("energy fitting requires an explicit common reference frame")
        validate_source(fitted_source)
        if not isinstance(model, (LinearEnergyScaling, FreeEnergyBEP)):
            raise ValueError("choose the original electronic scaling or free-energy BEP law")
        expected_kind = (
            "electronic_energy"
            if isinstance(model, LinearEnergyScaling)
            else "activation_free_energy"
        )
        if energy_kind != expected_kind:
            raise ValueError("electronic energies and free-energy BEP data are distinct semantics")
        self.model = type(model).from_dict(model.to_dict())
        self.fitted_source = fitted_source
        self.reference_frame = reference_frame
        self.energy_kind = energy_kind
        self.noise_model = noise_model
        self.observation_column = observation_column
        self.delta_g_column = delta_g_column
        self.descriptor_columns = dict(descriptor_columns or {})
        if isinstance(model, LinearEnergyScaling):
            axes = {a.descriptor_id for a in model.axes}
            if descriptor_columns is None:
                self.descriptor_columns = {name: name for name in axes}
            if set(self.descriptor_columns) != axes or len(
                set(self.descriptor_columns.values())
            ) != len(axes):
                raise ValueError("descriptor columns must map each declared axis exactly once")
            conditions = set(self.descriptor_columns.values())
            units = {
                "intercept_J_mol": "J/mol",
                **{"coefficient:" + name: "dimensionless" for name in axes},
            }
        else:
            if descriptor_columns:
                raise ValueError("free-energy BEP uses a declared reaction delta_g column")
            conditions = {delta_g_column}
            units = {"intercept_J_mol": "J/mol", "alpha": "dimensionless"}
        if any(
            not isinstance(p, CatalyticEnergyFitParameter)
            or p.name not in units
            or p.unit != units[p.name]
            for p in self.parameters
        ):
            raise ValueError(
                "fit parameters must name the original energy-law coefficients and units"
            )
        if isinstance(model, FreeEnergyBEP) and any(
            (p.name == "alpha" and not 0 <= p.bounds[0] < p.bounds[1] <= 1)
            or (p.name == "intercept_J_mol" and p.bounds[0] < 0)
            for p in self.parameters
        ):
            raise ValueError(
                "BEP parameter bounds must preserve nonnegative intercept and 0<=alpha<=1"
            )
        if (
            observation_column in conditions
            or set(self.columns) != conditions | {observation_column}
            or self.columns[observation_column].role != "observation"
            or any(self.columns[name].role != "condition" for name in conditions)
            or any(c.reference_state != reference_frame for c in table.columns)
        ):
            raise ValueError(
                "energy columns require exact roles, mapping and shared reference frame"
            )
        for column in self.columns.values():
            convert_value(1, column.unit, "J/mol")
        if any(row.values[name].uncertainty != 0 for row in table.rows for name in conditions):
            raise ValueError(
                "this fit conditions on explicitly exact descriptors/delta_g (sigma=0)"
            )
        self.output_scale = convert_value(1, "J/mol", self.columns[observation_column].unit)
        self.jacobian_runs: list[dict[str, Any]] = []
        self.candidate(self.initial_values)

    def candidate(self, values: Mapping[str, float]) -> LinearEnergyScaling | FreeEnergyBEP:
        if set(values) != {p.parameter_id for p in self.parameters}:
            raise ValueError("candidate must contain exactly the energy coefficient whitelist")
        for parameter in self.parameters:
            parameter.encode(values[parameter.name])
        if isinstance(self.model, LinearEnergyScaling):
            coefficients = dict(self.model.coefficients)
            for name, value in values.items():
                if name.startswith("coefficient:"):
                    coefficients[name.removeprefix("coefficient:")] = value
            return replace(
                self.model,
                coefficients=coefficients,
                intercept_J_mol=values.get("intercept_J_mol", self.model.intercept_J_mol),
                source=self.fitted_source,
            )
        return replace(
            self.model,
            intercept_J_mol=values.get("intercept_J_mol", self.model.intercept_J_mol),
            alpha=values.get("alpha", self.model.alpha),
            source=self.fitted_source,
        )

    def _evaluate(self, model: Any, row: Any) -> tuple[float, dict[str, float]]:
        if isinstance(model, LinearEnergyScaling):
            point = {
                name: row.values[column].to("J/mol").value
                for name, column in self.descriptor_columns.items()
            }
            value = model.evaluate(point)
            basis = {
                "intercept_J_mol": 1.0,
                **{"coefficient:" + name: number for name, number in point.items()},
            }
        else:
            delta = row.values[self.delta_g_column].to("J/mol").value
            value = model.barriers(delta)[0]
            basis = {"intercept_J_mol": 1.0, "alpha": delta}
        return value * self.output_scale, basis

    def forward(self, values: Mapping[str, float], split: str, *, stage: str = "prediction") -> Any:
        predictions = {}
        groups: dict[str, list[Any]] = {}
        for row_id, _ in self.pairs.get(split, ()):
            row = self.rows[row_id]
            groups.setdefault(row.group_id, []).append(row)
        for group_id, rows in groups.items():
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared energy forward group budget exhausted")
            record: dict[str, Any] = {
                "group_id": group_id,
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
                    reading = {"row_id": row.row_id, "value": None}
                    record["readouts"].append(reading)
                    value, _ = self._evaluate(model, row)
                    reading["value"] = value
                    predictions[(row.row_id, self.observation_column)] = value
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
                            "group_id": group_id,
                            "failure": record["failure"],
                        }
                    )
        return predictions

    def training_jacobian(self, coordinates: Any) -> Any:
        values = self.values(coordinates)
        record: dict[str, Any] = {
            "parameters": values,
            "split": "train",
            "rows": [],
            "failure": None,
        }
        self.jacobian_runs.append(record)
        try:
            model = self.candidate(values)
            derivatives = [
                p.decode_derivative(x) for p, x in zip(self.parameters, coordinates, strict=True)
            ]
            matrix = []
            for row_id, quantity in self.pairs["train"]:
                row = self.rows[row_id]
                _, basis = self._evaluate(model, row)
                weighted = [
                    basis[p.name] * scale * self.output_scale / row.values[quantity].uncertainty
                    for p, scale in zip(self.parameters, derivatives, strict=True)
                ]
                record["rows"].append({"row_id": row_id, "weighted_jacobian": weighted})
                matrix.append(weighted)
            return np.asarray(matrix)
        except (ValueError, ArithmeticError) as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise ForwardEvaluationError(record["failure"]) from exc

    def _optimizer_jacobian(self) -> Any:
        return self.training_jacobian

    def _resource_report(self) -> dict[str, Any]:
        return {
            "forward_groups": len(self.runs),
            "runtime_sessions": 0,
            "energy_evaluations": sum(len(r["readouts"]) for r in self.runs),
            "jacobian_calls": len(self.jacobian_runs),
            "jacobian_rows": sum(len(r["rows"]) for r in self.jacobian_runs),
        }

    def fit(self, **options: Any) -> dict[str, Any]:
        result = super().fit(**options)
        result.update(
            energy_kind=self.energy_kind,
            reference_frame=self.reference_frame,
            input_table=self.table.to_dict(),
            input_model=self.model.to_dict(),
            fitted_source=self.fitted_source.to_dict(),
            descriptor_columns=dict(self.descriptor_columns),
            delta_g_column=self.delta_g_column,
            observation_column=self.observation_column,
            jacobian_runs=self.jacobian_runs,
        )
        result["parameter_uncertainty"] = known_sigma_parameter_uncertainty(
            self,
            result,
            noise_model=self.noise_model,
            parameter_units={p.name: p.unit for p in self.parameters},
        )
        if result["status"] == "fitted":
            result["fitted_model"] = self.candidate(result["estimate"]).to_dict()
        return result
