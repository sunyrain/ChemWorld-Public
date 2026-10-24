"""Shared bounded coordinates, weighted residuals and one training-only fit driver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import exp, isfinite, log

import numpy as np
from scipy.optimize import least_squares

from chemworld.data.scientific_inputs import ScientificTable


class BoundedParameterCoordinates:
    """Shared linear/log normalization; concrete parameters own units and identity."""

    def _validate_coordinates(self):
        lower, upper = self.bounds
        if (
            self.coordinate not in {"linear", "log"}
            or any(isinstance(v, bool) or not isfinite(v) for v in (lower, upper, self.initial))
            or lower >= upper
        ):
            raise ValueError(
                "parameter requires finite increasing bounds and linear/log coordinates"
            )
        if not lower <= self.initial <= upper:
            raise ValueError("initial parameter must lie inside its finite declared bounds")
        if self.coordinate == "log" and lower <= 0:
            raise ValueError("log parameters require strictly positive bounds")

    def encode(self, value):
        lower, upper = self.bounds
        if not isfinite(value) or not lower <= value <= upper:
            raise ValueError("parameter is outside its finite declared domain")
        if self.coordinate == "log":
            value, lower, upper = log(value), log(lower), log(upper)
        return (value - lower) / (upper - lower)

    def decode(self, coordinate):
        if not isfinite(coordinate) or not 0 <= coordinate <= 1:
            raise ValueError("normalized parameter coordinate must be finite and inside [0, 1]")
        lower, upper = self.bounds
        if self.coordinate == "log":
            value = exp(log(lower) + coordinate * (log(upper) - log(lower)))
        else:
            value = lower + coordinate * (upper - lower)
        # Floating-point endpoint reconstruction must not extend the domain.
        return min(upper, max(lower, value))

    def decode_derivative(self, coordinate):
        value = self.decode(coordinate)
        lower, upper = self.bounds
        return value * (log(upper) - log(lower)) if self.coordinate == "log" else upper - lower


class ForwardEvaluationError(ValueError):
    """An executed candidate could not provide the declared measurements."""


class MeasurementCalibration:
    """Shared boundary calibration; forward consumers own their execution and resources."""

    schema_version = "chemworld-native-calibration-1"
    jacobian_description = "3-point"

    def __init__(self, table, parameters, *, max_forward_groups=300, progress=None):
        if not isinstance(table, ScientificTable) or table.kind != "measurements":
            raise ValueError("calibration requires a scientific measurement table")
        self.table = table
        self.parameters = tuple(parameters)
        if not self.parameters or len({p.parameter_id for p in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("calibration parameters must be nonempty and unique")
        if (
            isinstance(max_forward_groups, bool)
            or not isinstance(max_forward_groups, int)
            or max_forward_groups <= 0
        ):
            raise ValueError("forward execution limit must be a positive integer")
        self.max_forward_groups = max_forward_groups
        self.progress = progress
        self.runs = []
        self._fit_started = False
        self.columns = {c.quantity_id: c for c in table.columns}
        self.rows = {row.row_id: row for row in table.rows}
        self.excluded = []
        self.pairs = {}
        for row in table.rows:
            for column in table.columns:
                if column.role != "observation":
                    continue
                value = row.values[column.quantity_id]
                if value.quality != "valid":
                    self.excluded.append(
                        {
                            "row_id": row.row_id,
                            "quantity_id": column.quantity_id,
                            "split": row.split,
                            "quality": value.quality,
                        }
                    )
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
        return {p.parameter_id: p.initial for p in self.parameters}

    def values(self, coordinates):
        return {
            p.parameter_id: p.decode(x)
            for p, x in zip(
                self.parameters,
                coordinates,
                strict=True,
            )
        }

    def _optimizer_jacobian(self):
        return "3-point"

    def residuals(self, predictions, split):
        return np.asarray(
            [
                (predictions[key] - self.rows[key[0]].values[key[1]].value)
                / self.rows[key[0]].values[key[1]].uncertainty
                for key in self.pairs.get(split, ())
            ]
        )

    def metrics(self, predictions, split):
        residuals = self.residuals(predictions, split)
        details = []
        for key in self.pairs.get(split, ()):
            measured = self.rows[key[0]].values[key[1]]
            details.append(
                {
                    "row_id": key[0],
                    "quantity_id": key[1],
                    "unit": measured.unit,
                    "measured": measured.value,
                    "predicted": predictions[key],
                    "error": predictions[key] - measured.value,
                    "uncertainty": measured.uncertainty,
                }
            )
        return {
            "n_valid": len(details),
            "n_excluded": sum(e["split"] == split for e in self.excluded),
            "weighted_rmse": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
            "sum_squared_weighted_residuals": float(residuals @ residuals),
            "measurements": details,
        }

    def fit(
        self,
        *,
        max_nfev=30,
        diff_step=1e-3,
        rank_rtol=1e-5,
        rank_atol=1e-8,
        fixed_parameters=None,
        evaluate_validation=True,
    ):
        if self._fit_started or self.runs:
            raise ValueError("use a fresh calibration object for each fixed fitting block")
        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("optimizer limit must be a positive integer")
        if not all(isfinite(v) and v > 0 for v in (diff_step, rank_rtol, rank_atol)):
            raise ValueError("derivative and rank tolerances must be positive and finite")
        if not isinstance(evaluate_validation, bool):
            raise ValueError("evaluate_validation must be a boolean")
        if fixed_parameters is not None and not isinstance(fixed_parameters, Mapping):
            raise ValueError("fixed_parameters must be a named mapping")
        fixed = dict(fixed_parameters or {})
        known = {p.parameter_id: p for p in self.parameters}
        if set(fixed) - set(known):
            raise ValueError("fixed parameters must belong to the declared parameter whitelist")
        for name, value in fixed.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("fixed parameters must be finite numeric values")
            known[name].encode(value)
        free = [i for i, p in enumerate(self.parameters) if p.parameter_id not in fixed]
        initial = np.asarray(
            [p.encode(fixed.get(p.parameter_id, p.initial)) for p in self.parameters]
        )

        def full_coordinates(x):
            coordinates = initial.copy()
            coordinates[free] = x
            return coordinates

        def candidate_values(x):
            # Exact declared physical fixed values survive log coordinate round trips.
            return {**self.values(full_coordinates(x)), **fixed}

        self._fit_started = True
        report = {
            "schema_version": self.schema_version,
            "dataset": self.table.summary(),
            "sources": [source.to_dict() for source in self.table.sources],
            "parameters": [p.to_dict() for p in self.parameters],
            "optimization_split": "train",
            "heldout_split": "validation",
            "excluded_observations": self.excluded,
            "optimizer": {
                "method": "TRF weighted least squares",
                "jacobian": self.jacobian_description,
                "diff_step": diff_step if self.jacobian_description == "3-point" else None,
                "max_nfev": max_nfev,
                "max_forward_groups": self.max_forward_groups,
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
            },
            "runs": self.runs,
            "status": "not_started",
            "failures": [],
        }
        if fixed_parameters is not None:
            report.update(
                fixed_parameters=fixed,
                free_parameters=[self.parameters[i].parameter_id for i in free],
            )
        if not evaluate_validation:
            report["validation_evaluation"] = "not_requested"
        try:

            def training_residual(x):
                return self.residuals(
                    self.forward(candidate_values(x), "train", stage="fit"), "train"
                )

            if not free:
                report["estimate"] = candidate_values([])
                report["training"] = self.metrics(
                    self.forward(report["estimate"], "train", stage="fixed_evaluation"),
                    "train",
                )
                report["objective"] = report["training"]["sum_squared_weighted_residuals"]
                report["optimizer"].update(
                    method="fixed evaluation; no optimizer", success=True, nfev=0, njev=0
                )
                report["active_bounds"] = {}
                report["identifiability"] = {
                    "rank": 0,
                    "parameter_count": 0,
                    "status": "no_free_parameters",
                    "singular_values": [],
                    "weighted_jacobian": [],
                    "boundary": "fixed evaluation does not establish parameter identifiability",
                }
                if evaluate_validation:
                    report["validation"] = self.metrics(
                        self.forward(report["estimate"], "validation", stage="heldout"),
                        "validation",
                    )
                report["status"] = "fitted"
                report["resources"] = self._resource_report()
                json.dumps(report, allow_nan=False)
                return report

            original_jacobian = self._optimizer_jacobian()
            jacobian = original_jacobian
            if callable(original_jacobian) and fixed:

                def jacobian(x):
                    return original_jacobian(full_coordinates(x))[:, free]

            result = least_squares(
                training_residual,
                initial[free],
                bounds=(0.0, 1.0),
                method="trf",
                jac=jacobian,
                diff_step=diff_step,
                max_nfev=max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            report["optimizer"].update(
                success=bool(result.success),
                message=result.message,
                nfev=result.nfev,
                njev=result.njev,
                optimality=float(result.optimality),
            )
            report["estimate"] = candidate_values(result.x)
            # This is the objective of the actual returned candidate, even when
            # max_nfev terminates the optimizer. Never substitute a favorable fit.
            report["objective"] = float(result.fun @ result.fun)
            report["active_bounds"] = {
                self.parameters[i].parameter_id: int(active)
                for i, active in zip(free, result.active_mask, strict=True)
            }
            singular_values = np.linalg.svd(result.jac, compute_uv=False)
            threshold = max(rank_atol, rank_rtol * float(singular_values[0]))
            rank = int(np.count_nonzero(singular_values > threshold))
            report["identifiability"] = {
                "rank": rank,
                "parameter_count": len(free),
                "singular_values": singular_values.tolist(),
                "rank_threshold": threshold,
                "status": "full_rank_local" if rank == len(free) else "rank_deficient",
                "weighted_jacobian": result.jac.tolist(),
                "coordinate": "dimensionless bounded linear/log parameters",
                "boundary": "local sensitivity; no global uniqueness or material qualification",
            }
            if not result.success:
                report.update(status="optimizer_not_converged")
                report["failures"].append(result.message)
            else:
                report["training"] = self.metrics(
                    self.forward(report["estimate"], "train", stage="training_final"),
                    "train",
                )
                if evaluate_validation:
                    report["validation"] = self.metrics(
                        self.forward(report["estimate"], "validation", stage="heldout"),
                        "validation",
                    )
                report["status"] = "fitted"
        except ForwardEvaluationError as exc:
            report.update(status="forward_failed")
            report["failures"].append(str(exc))
        report["resources"] = self._resource_report()
        # Results must be usable as strict, readable JSON without hidden objects.
        json.dumps(report, allow_nan=False)
        return report

    def _resource_report(self):
        steps = [step for run in self.runs for step in run["steps"]]
        return {
            "forward_groups": len(self.runs),
            "attempted_operations": len(steps),
            "committed_operations": sum(s["transaction_status"] == "committed" for s in steps),
            **{
                key: sum(s[key] for s in steps)
                for key in (
                    "actual_duration_s",
                    "analysis_effort_s",
                    "cost_delta",
                    "sample_delta_L",
                )
            },
        }
