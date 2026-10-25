"""Bounded source-parameter regression through the original multifluid consumers.

Only declared continuous coefficients are fitted. Component identities, reducing
scales of pure fluids, gas constants, discrete term structure and source domains
are immutable. Training/holdout, uncertainty weighting and identifiability use
the same driver as the other ScientificTable calibration adapters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, cast

from chemworld.data.calibration_core import (
    BoundedParameterCoordinates,
    ForwardEvaluationError,
    MeasurementCalibration,
)
from chemworld.data.scientific_inputs import ScientificRow, ScientificTable
from chemworld.foundation.units import convert_value
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.multifluid import MultifluidModel

_BINARY = frozenset(("beta_T", "gamma_T", "beta_v", "gamma_v", "F"))
_DEPARTURE = frozenset(("n", "t", "eta", "epsilon", "beta", "gamma"))
_RESIDUAL_UNITS = {
    "pressure_Pa": "Pa",
    "compressibility_factor": "dimensionless",
    "residual_helmholtz_J_mol": "J/mol",
    "residual_internal_energy_J_mol": "J/mol",
    "residual_enthalpy_J_mol": "J/mol",
    "residual_entropy_J_mol_K": "J/(mol*K)",
    "residual_cv_J_mol_K": "J/(mol*K)",
    "residual_cp_J_mol_K": "J/(mol*K)",
}
_ABSOLUTE_UNITS = {
    "internal_energy_J_mol": "J/mol",
    "enthalpy_J_mol": "J/mol",
    "entropy_J_mol_K": "J/(mol*K)",
    "helmholtz_energy_J_mol": "J/mol",
    "gibbs_energy_J_mol": "J/mol",
    "cv_J_mol_K": "J/(mol*K)",
    "cp_J_mol_K": "J/(mol*K)",
    "mass_density_kg_m3": "kg/m^3",
    "speed_of_sound_m_s": "m/s",
}


@dataclass(frozen=True)
class MultifluidFitParameter(BoundedParameterCoordinates):
    """An oriented address in the existing source model, with physical bounds.

    ``binary`` addresses a reducing parameter or F on an exact oriented pair;
    ``departure`` addresses one continuous coefficient of a declared departure term;
    ``pure_residual`` addresses one n amplitude of a pure-fluid residual term.
    No arbitrary JSON paths, implicit reversed beta, or new terms are accepted.
    """

    scope: str
    component_ids: tuple[str, ...]
    coefficient: str
    bounds: tuple[float, float]
    initial: float
    term_index: int | None = None
    coefficient_index: int | None = None
    coordinate: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(self, "bounds", tuple(self.bounds))
        expected = 1 if self.scope == "pure_residual" else 2
        if (
            self.scope not in {"binary", "departure", "pure_residual"}
            or len(self.component_ids) != expected
            or len(set(self.component_ids)) != expected
            or any(not isinstance(k, str) or not k.strip() for k in self.component_ids)
            or self.coefficient
            not in (
                _BINARY
                if self.scope == "binary"
                else _DEPARTURE
                if self.scope == "departure"
                else {"n"}
            )
        ):
            raise ValueError("invalid explicit continuous multifluid parameter address")
        for value, required in (
            (self.term_index, self.scope != "binary"),
            (self.coefficient_index, self.scope == "pure_residual"),
        ):
            if (required and (type(value) is not int or value < 0)) or (
                not required and value is not None
            ):
                raise ValueError("parameter indices must match the declared source scope")
        if len(self.bounds) != 2:
            raise ValueError("parameter bounds require two values")
        self._validate_coordinates()
        if self.scope == "binary" and self.coefficient != "F" and self.bounds[0] <= 0:
            raise ValueError("reducing-parameter bounds must be strictly positive")

    @property
    def parameter_id(self) -> str:
        # JSON encoding is unambiguous even for component ids containing separators.
        return json.dumps(
            [
                self.scope,
                list(self.component_ids),
                self.term_index,
                self.coefficient,
                self.coefficient_index,
            ],
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "component_ids": list(self.component_ids),
            "coefficient": self.coefficient,
            "term_index": self.term_index,
            "coefficient_index": self.coefficient_index,
            "bounds": list(self.bounds),
            "initial": self.initial,
            "coordinate": self.coordinate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultifluidFitParameter:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("multifluid fit parameter requires its exact declared schema")
        return cls(**data)


class MultifluidCalibration(MeasurementCalibration):
    """Fit homogeneous T/rho/composition measurements; return the native model.

    Default conditions are ``temperature``, ``molar_density`` and ``x:<component>``.
    Every observation column maps to an original state field, ``mu:<component>``
    (residual chemical potential), or ``lnphi:<component>``. Units are converted by
    ScientificTable and the shared registry. Absolute outputs keep the source ideal
    reference. Local mechanical/thermal checks are never disabled for regression.
    """

    schema_version = "chemworld-multifluid-calibration-1"
    parameters: tuple[MultifluidFitParameter, ...]

    def __init__(
        self,
        table: ScientificTable,
        model: MultifluidModel,
        parameters: Sequence[MultifluidFitParameter],
        *,
        fitted_source: str,
        temperature_column: str = "temperature",
        density_column: str = "molar_density",
        composition_columns: Mapping[str, str] | None = None,
        observations: Mapping[str, str] | None = None,
        max_forward_groups: int = 60000,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        parameters = tuple(parameters)
        if any(not isinstance(p, MultifluidFitParameter) for p in parameters):
            raise ValueError("use explicit MultifluidFitParameter addresses")
        super().__init__(
            table, parameters, max_forward_groups=max_forward_groups, progress=progress
        )
        if not isinstance(model, MultifluidModel):
            raise ValueError("multifluid calibration requires the original MultifluidModel")
        if (
            not isinstance(fitted_source, str)
            or not fitted_source.strip()
            or fitted_source == model.source
        ):
            raise ValueError("a fitted model needs a new explicit source description")
        self.model = MultifluidModel.from_dict(model.to_dict())
        self.fitted_source = fitted_source
        self.temperature_column = temperature_column
        self.density_column = density_column
        self.composition_columns = (
            dict(composition_columns)
            if composition_columns is not None
            else {k: "x:" + k for k in model.component_ids}
        )
        self.observations = (
            dict(observations)
            if observations is not None
            else {c.quantity_id: c.quantity_id for c in table.columns if c.role == "observation"}
        )
        conditions = [temperature_column, density_column, *self.composition_columns.values()]
        if (
            set(self.composition_columns) != set(model.component_ids)
            or len(set(conditions)) != len(conditions)
            or set(conditions) & set(self.observations)
            or set(self.columns) != set(conditions) | set(self.observations)
            or any(self.columns[k].role != "condition" for k in conditions)
            or any(self.columns[k].role != "observation" for k in self.observations)
        ):
            raise ValueError(
                "every table column must map exactly to a model condition or observation"
            )
        for column, unit in (
            (temperature_column, "K"),
            (density_column, "mol/m^3"),
            *((k, "dimensionless") for k in self.composition_columns.values()),
        ):
            convert_value(1, self.columns[column].unit, unit)
        self.observation_units = {}
        for column, quantity in self.observations.items():
            native_unit = {**_RESIDUAL_UNITS, **_ABSOLUTE_UNITS}.get(quantity)
            if native_unit is None:
                prefix, _, component = quantity.partition(":")
                if prefix not in {"mu", "lnphi"} or component not in model.component_ids:
                    raise ValueError("unknown native multifluid observable")
                native_unit = "J/mol" if prefix == "mu" else "dimensionless"
            convert_value(1, native_unit, self.columns[column].unit)
            self.observation_units[column] = native_unit
        # Validate addresses without inspecting any held-out response values.
        self.candidate(self.initial_values)

    def candidate(self, values: Mapping[str, float]) -> MultifluidModel:
        """Detached native candidate; all changes are restricted to the whitelist."""
        if set(values) != {p.parameter_id for p in self.parameters}:
            raise ValueError("candidate must contain exactly the fitted parameter whitelist")
        fluids, pairs = list(self.model.fluids), list(self.model.binary_pairs)
        for p in self.parameters:
            value = values[p.parameter_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("candidate coefficients must be finite real numbers")
            p.encode(value)
            if p.scope == "pure_residual":
                if p.component_ids[0] not in self.model.component_ids:
                    raise ValueError("pure parameter component is absent from the source model")
                index = self.model.component_ids.index(p.component_ids[0])
                fluid = fluids[index]
                assert p.term_index is not None and p.coefficient_index is not None
                if p.term_index >= len(fluid.residual_terms):
                    raise ValueError("pure residual term index is outside the source model")
                terms = list(fluid.residual_terms)
                term = terms[p.term_index]
                coefficients = dict(term.coefficients)
                if "n" not in coefficients or p.coefficient_index >= len(coefficients["n"]):
                    raise ValueError("pure residual amplitude index is absent from the source term")
                amplitudes = list(coefficients["n"])
                amplitudes[p.coefficient_index] = value
                terms[p.term_index] = HelmholtzTerm(
                    term.type, {**coefficients, "n": amplitudes}, term.reference
                )
                fluids[index] = replace(fluid, residual_terms=tuple(terms))
            else:
                indices = [
                    i for i, pair in enumerate(pairs) if pair.component_ids == p.component_ids
                ]
                if not indices:
                    raise ValueError("parameter requires an exact source pair orientation")
                index = indices[0]
                pair = pairs[index]
                update: dict[str, Any] = {p.coefficient: value}
                if p.scope == "binary":
                    pairs[index] = replace(pair, **update)
                else:
                    assert p.term_index is not None
                    if p.term_index >= len(pair.departure_terms):
                        raise ValueError("departure term index is outside the source model")
                    departures = list(pair.departure_terms)
                    departure = departures[p.term_index]
                    if departure.kind == "power" and p.coefficient not in {"n", "t"}:
                        raise ValueError("power departure has no GERG shape coefficients")
                    departures[p.term_index] = replace(departure, **update)
                    pairs[index] = replace(pair, departure_terms=tuple(departures))
        return replace(self.model, fluids=tuple(fluids), binary_pairs=tuple(pairs))

    def _conditions(self, row: ScientificRow) -> tuple[float, float, tuple[float, ...]]:
        return (
            cast(float, row.values[self.temperature_column].to("K").value),
            cast(float, row.values[self.density_column].to("mol/m^3").value),
            tuple(
                cast(float, row.values[self.composition_columns[k]].to("dimensionless").value)
                for k in self.model.component_ids
            ),
        )

    def forward(
        self, values: Mapping[str, float], split: str, *, stage: str = "prediction"
    ) -> dict[tuple[str, str], float]:
        predictions: dict[tuple[str, str], float] = {}
        rows: dict[str, list[str]] = {}
        for row_id, column in self.pairs.get(split, ()):
            rows.setdefault(row_id, []).append(column)
        for row_id, columns in rows.items():
            if len(self.runs) >= self.max_forward_groups:
                raise ForwardEvaluationError("declared forward group execution budget exhausted")
            row = self.rows[row_id]
            record: dict[str, Any] = {
                "group_id": row.group_id,
                "row_id": row_id,
                "split": split,
                "stage": stage,
                "parameters": dict(values),
                "readouts": [],
                "failure": None,
                "state_attempts": 0,
                "chemical_potential_attempts": 0,
            }
            self.runs.append(record)
            try:
                model = self.candidate(values)
                conditions = self._conditions(row)
                record["conditions"] = {
                    "temperature_K": conditions[0],
                    "molar_density_mol_m3": conditions[1],
                    "mole_fractions": list(conditions[2]),
                }
                quantities = [self.observations[k] for k in columns]
                record["state_attempts"] += 1
                state = (
                    model.caloric_state(*conditions)
                    if any(k in _ABSOLUTE_UNITS for k in quantities)
                    else model.state(*conditions)
                )
                mu = None
                if any(k.startswith("mu:") for k in quantities):
                    record["chemical_potential_attempts"] += 1
                    mu = model.residual_chemical_potentials(*conditions)
                for column in columns:
                    quantity = self.observations[column]
                    prefix, _, component = quantity.partition(":")
                    if prefix in {"mu", "lnphi"}:
                        vector = mu if prefix == "mu" else state.log_fugacity_coefficients
                        assert vector is not None
                        value = vector[model.component_ids.index(component)]
                    else:
                        value = getattr(state, quantity)
                    predicted = convert_value(
                        value, self.observation_units[column], self.columns[column].unit
                    )
                    if not isfinite(predicted):
                        raise ValueError("nonfinite multifluid observable")
                    record["readouts"].append(
                        {
                            "quantity_id": column,
                            "observable": quantity,
                            "value": predicted,
                            "unit": self.columns[column].unit,
                        }
                    )
                    predictions[(row_id, column)] = predicted
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
                            "row_id": row_id,
                            "failure": record["failure"],
                        }
                    )
        return predictions

    def _resource_report(self) -> dict[str, Any]:
        return {
            "forward_groups": len(self.runs),
            "runtime_sessions": 0,
            "state_attempts": sum(r["state_attempts"] for r in self.runs),
            "chemical_potential_attempts": sum(r["chemical_potential_attempts"] for r in self.runs),
            "property_readouts": sum(len(r["readouts"]) for r in self.runs),
            "failed_groups": sum(r["failure"] is not None for r in self.runs),
            "group_unit": "one attempted homogeneous state row, shared across its observables",
        }

    def fit(self, **options: Any) -> dict[str, Any]:
        result = super().fit(**options)
        result.update(
            input_model=self.model.to_dict(),
            condition_columns={
                "temperature": self.temperature_column,
                "molar_density": self.density_column,
                "composition": dict(self.composition_columns),
            },
            observations=dict(self.observations),
            parameter_basis="original source coefficients; only optimizer coordinates normalized",
            applicability=(
                "local homogeneous regression; source T/P domains and ideal references retained; "
                "synthetic recovery is not measured-data or material qualification; "
                "local Jacobian rank is not global parameter uniqueness"
            ),
        )
        if result["status"] == "fitted":
            candidate = self.candidate(result["estimate"])
            changed_fluids = {
                p.component_ids[0] for p in self.parameters if p.scope == "pure_residual"
            }
            changed_pairs = {p.component_ids for p in self.parameters if p.scope != "pure_residual"}
            annotation = (
                f"; fitted: {self.fitted_source}; dataset={self.table.dataset_id}; train only"
            )
            result["fitted_model"] = replace(
                candidate,
                source=self.model.source + annotation,
                fluids=tuple(
                    replace(f, source=f.source + annotation) if f.fluid_id in changed_fluids else f
                    for f in candidate.fluids
                ),
                binary_pairs=tuple(
                    replace(p, source=p.source + annotation)
                    if p.component_ids in changed_pairs
                    else p
                    for p in candidate.binary_pairs
                ),
            ).to_dict()
        json.dumps(result, allow_nan=False)
        return result
