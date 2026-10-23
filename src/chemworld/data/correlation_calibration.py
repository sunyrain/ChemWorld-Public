"""Fit declared property correlations to ScientificTable observations.

The optimizer, normalized coordinates, weighted metrics and train/holdout policy
are shared with native-session calibration. This adapter owns no runtime state.
"""

from dataclasses import dataclass, replace

import numpy as np

from chemworld.data.calibration_core import (
    BoundedParameterCoordinates,
    ForwardEvaluationError,
    MeasurementCalibration,
)
from chemworld.foundation.units import convert_value
from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.dippr import dippr_parameter_jacobian
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.vapor_pressure import vapor_pressure_parameter_jacobian
from chemworld.physchem.viscosity_kernels import viscosity_yaws_parameter_jacobian
from chemworld.physchem.volume_kernels import volume_parameter_jacobian


@dataclass(frozen=True)
class CorrelationFitParameter(BoundedParameterCoordinates):
    """Bounds and initial value in the unmodified source coefficient coordinate.

    A coefficient's dimensional meaning is defined by its equation and the
    correlation's input/output units, not by the normalization used by the solver.
    No unit conversion is applied independently to a coefficient.
    """

    name: str
    bounds: tuple[float, float]
    initial: float
    coordinate: str = "linear"

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
        ):
            raise ValueError("fitted coefficient requires a name")
        if len(self.bounds) != 2:
            raise ValueError("coefficient bounds require two values")
        object.__setattr__(self, "bounds", tuple(self.bounds))
        self._validate_coordinates()

    @property
    def parameter_id(self):
        return self.name

    def to_dict(self):
        return {
            "coefficient": self.name,
            "bounds": list(self.bounds),
            "initial": self.initial,
            "coordinate": self.coordinate,
            "basis": "declared source coefficient",
        }


class CorrelationCalibration(MeasurementCalibration):
    schema_version = "chemworld-correlation-calibration-1"

    def __init__(
        self,
        table,
        correlation,
        parameters,
        *,
        fitted_correlation_id,
        temperature_column="temperature",
        observation_column="value",
        pressure_column=None,
        molecular_weight_g_mol=None,
        derivative_mode="analytic",
        max_forward_groups=60000,
        progress=None,
    ):
        super().__init__(
            table, parameters, max_forward_groups=max_forward_groups, progress=progress
        )
        if (
            not isinstance(correlation, PropertyCorrelation)
            or not isinstance(fitted_correlation_id, str)
            or not fitted_correlation_id.strip()
            or fitted_correlation_id != fitted_correlation_id.strip()
            or fitted_correlation_id == correlation.correlation_id
        ):
            raise ValueError("a fitted correlation needs a new explicit identity")
        self.correlation = PropertyCorrelation.from_dict(correlation.to_dict())
        self.fitted_correlation_id = fitted_correlation_id
        self.temperature_column = temperature_column
        self.pressure_column = pressure_column
        self.observation_column = observation_column
        self.molecular_weight_g_mol = molecular_weight_g_mol
        if derivative_mode not in {"analytic", "3-point"}:
            raise ValueError("choose analytic or explicit 3-point differentiation")
        self.jacobian_description = (
            "analytic source coefficients" if (derivative_mode == "analytic") else "3-point"
        )
        discrete = {
            "has_hydroxyl",
            "hydroxyl_polar_term",
            "estimate_saturation",
            "low_temperature_limit",
        }
        if any(
            not isinstance(p, CorrelationFitParameter)
            or p.name in discrete
            or p.name not in self.correlation.coefficients
            for p in self.parameters
        ):
            raise ValueError("fit parameters must name explicit continuous source coefficients")
        conditions = {temperature_column, *(() if pressure_column is None else (pressure_column,))}
        if (
            set(self.columns) != conditions | {observation_column}
            or observation_column in conditions
            or self.columns[observation_column].role != "observation"
            or any(self.columns[name].role != "condition" for name in conditions)
            or ("pressure" in correlation.input_units) != (pressure_column is not None)
        ):
            raise ValueError(
                "every table column must map exactly to a property input or observation"
            )
        convert_value(1, self.columns[temperature_column].unit, "K")
        if pressure_column is not None:
            convert_value(1, self.columns[pressure_column].unit, "Pa")
        output = self.columns[observation_column].unit
        self.output_scale = convert_value(1, correlation.output_unit, output) - convert_value(
            0, correlation.output_unit, output
        )
        self.jacobian_runs = []
        self.analytic_family = self._analytic_family()
        if derivative_mode == "analytic" and self.analytic_family is None:
            raise ValueError(
                "no analytic coefficient Jacobian for this family; "
                "3-point mode must be explicitly requested"
            )
        # Constructor validation does not consume held-out response values.
        self.candidate(self.initial_values)

    def _analytic_family(self):
        equation = self.correlation.equation_id
        family = CORRELATION_FAMILIES.get(equation)
        if equation.startswith("dippr"):
            return "dippr"
        if (family is not None and family.pressure_method) or equation in {"antoine", "wagner"}:
            return "pressure"
        if family is not None and family.volume_method:
            return "volume"
        if family is not None and family.viscosity_method == "mu_Yaws":
            return "viscosity_yaws"
        return None

    def candidate(self, values):
        if set(values) != {p.name for p in self.parameters}:
            raise ValueError("candidate must contain exactly the fitted coefficient whitelist")
        for p in self.parameters:
            p.encode(values[p.name])
        return replace(self.correlation, coefficients={**self.correlation.coefficients, **values})

    def _conditions(self, row):
        return {
            "temperature_K": row.values[self.temperature_column].to("K").value,
            "pressure_Pa": None
            if self.pressure_column is None
            else row.values[self.pressure_column].to("Pa").value,
        }

    def forward(self, values, split, *, stage="prediction"):
        predictions = {}
        groups = {}
        for row_id, _ in self.pairs.get(split, ()):
            row = self.rows[row_id]
            groups.setdefault(row.group_id, []).append(row)
        for group, rows in groups.items():
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared forward group execution budget exhausted")
            record = {
                "group_id": group,
                "split": split,
                "stage": stage,
                "parameters": dict(values),
                "readouts": [],
                "failure": None,
            }
            self.runs.append(record)
            try:
                card = self.candidate(values)
                for row in rows:
                    reading = {
                        "row_id": row.row_id,
                        **self._conditions(row),
                        "value": None,
                        "unit": self.columns[self.observation_column].unit,
                    }
                    record["readouts"].append(reading)
                    result = evaluate_correlation(
                        card,
                        **self._conditions(row),
                        molecular_weight_g_mol=self.molecular_weight_g_mol,
                        validity_policy="raise",
                    )
                    predicted = result.to(reading["unit"]).value
                    reading["value"] = predicted
                    predictions[(row.row_id, self.observation_column)] = predicted
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

    def _source_jacobian(self, card, conditions):
        T = conditions["temperature_K"]
        names = tuple(p.name for p in self.parameters)
        if self.analytic_family == "dippr":
            return dippr_parameter_jacobian(card, [T], parameters=names).values[0]
        if self.analytic_family == "pressure":
            return vapor_pressure_parameter_jacobian(card, [T], parameters=names).values[0]
        if self.analytic_family == "volume":
            family = CORRELATION_FAMILIES[card.equation_id]
            parameters = {**card.coefficients}
            if conditions["pressure_Pa"] is not None:
                parameters["P"] = conditions["pressure_Pa"]
            jac = volume_parameter_jacobian(
                family.volume_method, T, parameters, output=family.volume_output, names=names
            )
            return tuple(jac[name] for name in names)
        if self.analytic_family == "viscosity_yaws":
            row = viscosity_yaws_parameter_jacobian([T], **card.coefficients)[0]
            return tuple(row["ABCD".index(name)] for name in names)
        raise ValueError("no analytic Jacobian declared")

    def training_jacobian(self, coordinates):
        values = self.values(coordinates)
        record = {"split": "train", "parameters": values, "rows": [], "failure": None}
        self.jacobian_runs.append(record)
        try:
            card = self.candidate(values)
            scales = np.asarray(
                [p.decode_derivative(x) for p, x in zip(self.parameters, coordinates, strict=True)]
            )
            matrix = []
            for row_id, quantity_id in self.pairs["train"]:
                row = self.rows[row_id]
                # Check the same model/application domain as the forward consumer.
                evaluate_correlation(
                    card,
                    **self._conditions(row),
                    validity_policy="raise",
                    molecular_weight_g_mol=self.molecular_weight_g_mol,
                )
                source = np.asarray(self._source_jacobian(card, self._conditions(row)))
                weighted = source * self.output_scale * scales / row.values[quantity_id].uncertainty
                if not np.all(np.isfinite(weighted)):
                    raise ValueError("nonfinite weighted analytic Jacobian")
                record["rows"].append({"row_id": row_id, "weighted_jacobian": weighted.tolist()})
                matrix.append(weighted)
            return np.asarray(matrix)
        except (ValueError, ArithmeticError) as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise ForwardEvaluationError(record["failure"]) from exc

    def _optimizer_jacobian(self):
        return "3-point" if self.jacobian_description == "3-point" else self.training_jacobian

    def _resource_report(self):
        return {
            "forward_groups": len(self.runs),
            "runtime_sessions": 0,
            "property_evaluations": sum(len(r["readouts"]) for r in self.runs),
            "jacobian_calls": len(self.jacobian_runs),
            "jacobian_rows": sum(len(r["rows"]) for r in self.jacobian_runs),
        }

    def fit(self, **options):
        result = super().fit(**options)
        result.update(
            input_correlation=self.correlation.to_dict(),
            jacobian_runs=self.jacobian_runs,
            coefficient_basis={
                "equation_id": self.correlation.equation_id,
                "input_units": dict(self.correlation.input_units),
                "output_unit": self.correlation.output_unit,
                "meaning": "unmodified source coefficients; normalized only inside the optimizer",
            },
        )
        if result["status"] == "fitted":
            card = self.candidate(result["estimate"])
            metadata = {
                "source": {"source_id": "fit:" + self.fitted_correlation_id},
                "parent_correlation": self.correlation.to_dict(),
                "calibration": {
                    "dataset": self.table.summary(),
                    "sources": result["sources"],
                    "parameter_basis": result["coefficient_basis"],
                    "selected_coefficients": [p.name for p in self.parameters],
                    "training": result["training"],
                    "validation": result["validation"],
                    "identifiability": result["identifiability"],
                    "boundary": (
                        "local regression; no automatic material or full-domain qualification"
                    ),
                },
            }
            result["fitted_correlation"] = replace(
                card,
                correlation_id=self.fitted_correlation_id,
                metadata=metadata,
                source_note=f"Bounded fit to {self.table.dataset_id}; "
                "training rows only; original model and all sources retained in metadata",
            ).to_dict()
        return result
