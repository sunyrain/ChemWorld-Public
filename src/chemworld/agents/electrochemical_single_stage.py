"""Single-stage electrochemical recipe contract for S0 static optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_TASK_CONTRACT,
)

ELECTROCHEMICAL_SINGLE_STAGE_RECIPE_VERSION = (
    "chemworld-electrochemical-static-single-stage-recipe-0.2-s0-dev"
)
ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION = 6
ELECTROCHEMICAL_SINGLE_STAGE_EVENT_COUNT = 8
ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES = ((0, 4), (1, 4))
ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS = (
    {
        "slot_id": "diagnostic-01-ph_meter",
        "instrument": "ph_meter",
        "after_operation": "electrolyze",
        "recipe_step_index": 4,
        "selection_policy": "agent_selectable",
        "scientific_role": "electrolyte_and_phase_state_after_electrolysis",
        "stage_id": "post_electrolysis_equilibrium_diagnostic",
        "model_facing_metric_ids": [
            "pH_normalized",
            "acid_dissociation_fraction",
            "precipitation_signal",
            "equilibrium_residual",
            "equilibrium_confidence",
        ],
    },
    {
        "slot_id": "diagnostic-02-uvvis",
        "instrument": "uvvis",
        "after_operation": "measure",
        "recipe_step_index": 5,
        "selection_policy": "agent_selectable",
        "scientific_role": "electrochemical_efficiency_after_electrolysis",
        "stage_id": "post_electrolysis_efficiency_diagnostic",
        "model_facing_metric_ids": [
            "faradaic_efficiency",
            "transport_efficiency",
            "ohmic_efficiency",
            "energy_efficiency",
        ],
    },
)


def electrochemical_single_stage_parameter_schema() -> dict[str, dict[str, Any]]:
    """Return the six model-facing controls for one complete S0 experiment."""

    return {
        "electrolyte_profile": {"type": "integer", "minimum": 0, "maximum": 3},
        "solvent": {"type": "integer", "minimum": 0, "maximum": 3},
        "reagent_amount_mol": {
            "type": "number",
            "minimum": 0.003,
            "maximum": 0.030,
            "unit": "mol",
        },
        "potential_V": {
            "type": "number",
            "minimum": ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V[0],
            "maximum": ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V[1],
            "unit": "V",
        },
        "current_mA": {
            "type": "number",
            "minimum": ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA[0],
            "maximum": ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA[1],
            "unit": "mA",
            "semantics": "nonnegative_magnitude_cap",
        },
        "duration_s": {
            "type": "number",
            "minimum": 300.0,
            "maximum": 3600.0,
            "unit": "s",
        },
    }


def electrochemical_single_stage_parameters_from_unit_vector(
    vector: np.ndarray,
) -> dict[str, int | float]:
    values: NDArray[np.float64] = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION:
        raise ValueError("single-stage electrochemical recipe requires six coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError("single-stage electrochemical coordinates must be finite")
    values = np.clip(values, 0.0, 1.0)
    potential_low, potential_high = ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V
    current_low, current_high = ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA
    return {
        "electrolyte_profile": _choice(values[0], 4),
        "solvent": _choice(values[1], 4),
        "reagent_amount_mol": _bounded_scale(values[2], 0.003, 0.030),
        "potential_V": _bounded_scale(values[3], potential_low, potential_high),
        "current_mA": _bounded_scale(values[4], current_low, current_high),
        "duration_s": _bounded_scale(values[5], 300.0, 3600.0),
    }


def electrochemical_single_stage_unit_vector_from_parameters(
    payload: object,
) -> np.ndarray:
    schema = electrochemical_single_stage_parameter_schema()
    if not isinstance(payload, Mapping) or set(payload) != set(schema):
        raise ValueError(
            "single-stage electrochemical recipe_parameters fields do not match the contract"
        )
    normalized: dict[str, float] = {}
    for field, specification in schema.items():
        value = payload[field]
        if specification["type"] == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{field} must be finite")
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        if numeric < minimum - 1.0e-12 or numeric > maximum + 1.0e-12:
            raise ValueError(f"{field} is outside its physical bounds")
        normalized[field] = min(max(numeric, minimum), maximum)
    potential_low, potential_high = ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V
    current_low, current_high = ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA
    return np.asarray(
        [
            (normalized["electrolyte_profile"] + 0.5) / 4.0,
            (normalized["solvent"] + 0.5) / 4.0,
            _unscale(normalized["reagent_amount_mol"], 0.003, 0.030),
            _unscale(normalized["potential_V"], potential_low, potential_high),
            _unscale(normalized["current_mA"], current_low, current_high),
            _unscale(normalized["duration_s"], 300.0, 3600.0),
        ],
        dtype=float,
    )


def electrochemical_single_stage_recipe_from_unit_vector(
    task_info: Mapping[str, Any],
    vector: np.ndarray,
) -> dict[str, Any]:
    parameters = electrochemical_single_stage_parameters_from_unit_vector(vector)
    steps = [
        {
            "operation": "add_solvent",
            "volume_L": 0.025,
            "solvent": parameters["solvent"],
        },
        {
            "operation": "add_reagent",
            "amount_mol": parameters["reagent_amount_mol"],
        },
        {
            "operation": "set_potential",
            "potential_V": parameters["potential_V"],
            "current_mA": parameters["current_mA"],
            "electrolyte_profile": parameters["electrolyte_profile"],
        },
        {"operation": "electrolyze", "duration_s": parameters["duration_s"]},
        {"operation": "measure", "instrument": "ph_meter"},
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    return {
        "steps": steps,
        "metadata": {
            "search_space_version": ELECTROCHEMICAL_SINGLE_STAGE_RECIPE_VERSION,
            "search_space_kind": "electrochemical",
            "electrochemical_workflow_mode": "static_single_stage",
            "task_id": task_info.get("task_id"),
            "search_vector": [
                float(value)
                for value in electrochemical_single_stage_unit_vector_from_parameters(parameters)
            ],
            "recipe_parameters": parameters,
        },
    }


def _choice(value: float, count: int) -> int:
    return min(int(float(value) * count), count - 1)


def _scale(value: float, low: float, high: float) -> float:
    return float(low + float(value) * (high - low))


def _bounded_scale(value: float, low: float, high: float) -> float:
    return min(max(_scale(value, low, high), low), high)


def _unscale(value: float, low: float, high: float) -> float:
    return float((value - low) / (high - low))


__all__ = [
    "ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES",
    "ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION",
    "ELECTROCHEMICAL_SINGLE_STAGE_EVENT_COUNT",
    "ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS",
    "ELECTROCHEMICAL_SINGLE_STAGE_RECIPE_VERSION",
    "electrochemical_single_stage_parameter_schema",
    "electrochemical_single_stage_parameters_from_unit_vector",
    "electrochemical_single_stage_recipe_from_unit_vector",
    "electrochemical_single_stage_unit_vector_from_parameters",
]
