"""PSD CDF fitting through the common training-only calibration driver."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from math import log
from typing import Any, cast

import numpy as np

from chemworld.data.calibration_core import (
    FitParameter,
    FloatArray,
    ForwardEvaluationError,
    Jacobian,
    MeasurementCalibration,
)
from chemworld.data.scientific_inputs import ScientificRow, ScientificTable
from chemworld.physchem.particle_distributions import AnalyticDiameterDistribution


class DiameterDistributionCalibration(MeasurementCalibration[FitParameter]):
    """Use existing bounded fit parameters named scale_m and/or shape.

    Input ScientificTable supplies diameter conditions and CDF observations,
    uncertainty, source and immutable train/validation group assignments.
    """

    schema_version = "chemworld-diameter-distribution-calibration-1"

    def __init__(
        self,
        table: ScientificTable,
        distribution: AnalyticDiameterDistribution,
        parameters: Sequence[FitParameter],
        *,
        diameter_column: str = "diameter",
        cdf_column: str = "cdf",
        max_forward_groups: int = 10000,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(
            table, parameters, max_forward_groups=max_forward_groups, progress=progress
        )
        if not isinstance(distribution, AnalyticDiameterDistribution):
            raise ValueError("an analytic diameter distribution is required")
        if any(p.parameter_id not in {"scale_m", "shape"} for p in self.parameters):
            raise ValueError("fit parameters must name scale_m or shape")
        if any(p.bounds[0] <= 0 for p in self.parameters):
            raise ValueError("distribution fit bounds must remain positive")
        if (
            self.columns[diameter_column].role != "condition"
            or self.columns[cdf_column].role != "observation"
        ):
            raise ValueError("diameter condition and CDF observation columns required")
        if any(quantity != cdf_column for pairs in self.pairs.values() for _, quantity in pairs):
            raise ValueError("this calibration fits only CDF observations")
        for row in table.rows:
            row.values[diameter_column].to("m")
            observed = row.values[cdf_column].to("dimensionless")
            if observed.quality == "valid" and not 0 <= cast(float, observed.value) <= 1:
                raise ValueError("CDF observations must lie in [0,1]")
        self.distribution, self.diameter_column, self.cdf_column = (
            distribution,
            diameter_column,
            cdf_column,
        )
        self.analytic_ggs = (
            distribution.family == "GGS"
            and distribution.lower_m == 0
            and distribution.upper_m == distribution.scale_m
        )
        self.jacobian_calls = 0
        if self.analytic_ggs:
            self.jacobian_description = "analytic GGS CDF with saturated branch derivative zero"

    def _optimizer_jacobian(self) -> Jacobian:
        return self.training_jacobian if self.analytic_ggs else "3-point"

    def training_jacobian(self, coordinates: FloatArray) -> FloatArray:
        values = self.values(coordinates)
        distribution = self.candidate(values)
        matrix = []
        for row_id, quantity in self.pairs["train"]:
            row = self.rows[row_id]
            diameter = cast(float, row.values[self.diameter_column].to("m").value)
            cdf = distribution.cdf(diameter)
            derivatives = {"scale_m": 0.0, "shape": 0.0}
            if 0 < diameter < distribution.scale_m:
                derivatives = {
                    "scale_m": -distribution.shape * cdf / distribution.scale_m,
                    "shape": cdf * log(diameter / distribution.scale_m),
                }
            matrix.append(
                [
                    derivatives[p.parameter_id]
                    * p.decode_derivative(x)
                    / cast(float, row.values[quantity].uncertainty)
                    for p, x in zip(self.parameters, coordinates, strict=True)
                ]
            )
        self.jacobian_calls += 1
        return np.asarray(matrix)

    def candidate(self, values: dict[str, float]) -> AnalyticDiameterDistribution:
        # GGS natural upper support follows the fitted scale unless the caller
        # explicitly truncated it below the original characteristic diameter.
        if (
            self.distribution.family == "GGS"
            and self.distribution.upper_m == self.distribution.scale_m
        ):
            values = {**values, "upper_m": values.get("scale_m", self.distribution.scale_m)}
        return replace(
            self.distribution,
            scale_m=values.get("scale_m", self.distribution.scale_m),
            shape=values.get("shape", self.distribution.shape),
            upper_m=values.get("upper_m", self.distribution.upper_m),
        )

    def forward(
        self, values: dict[str, float], split: str, *, stage: str
    ) -> dict[tuple[str, str], float]:
        predictions = {}
        groups: dict[str, list[ScientificRow]] = {}
        for row_id, _ in self.pairs.get(split, ()):
            row = self.rows[row_id]
            groups.setdefault(row.group_id, []).append(row)
        for group, rows in groups.items():
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared forward group budget exhausted")
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
                distribution = self.candidate(values)
                for row in rows:
                    diameter = cast(float, row.values[self.diameter_column].to("m").value)
                    value = distribution.cdf(diameter)
                    predictions[(row.row_id, self.cdf_column)] = value
                    record["readouts"].append(
                        {"row_id": row.row_id, "diameter_m": diameter, "cdf": value}
                    )
            except (ValueError, ArithmeticError) as error:
                record["failure"] = f"{type(error).__name__}: {error}"
                raise ForwardEvaluationError(record["failure"]) from error
            finally:
                if self.progress:
                    self.progress(
                        {
                            "stage": stage,
                            "completed_forward_groups": len(self.runs),
                            "forward_limit": self.max_forward_groups,
                        }
                    )
        return predictions

    def _resource_report(self) -> dict[str, int]:
        return {
            "forward_groups": len(self.runs),
            "runtime_sessions": 0,
            "cdf_evaluations": sum(len(run["readouts"]) for run in self.runs),
            "jacobian_calls": self.jacobian_calls,
        }

    def fit(self, **options: Any) -> dict[str, Any]:
        report = super().fit(**options)
        report["input_distribution"] = self.distribution.to_dict()
        if "estimate" in report:
            report["fitted_distribution"] = self.candidate(report["estimate"]).to_dict()
        return report
