"""Named complete-experiment contract for reaction-to-crystallization S0."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

CRYSTALLIZATION_SINGLE_STAGE_RECIPE_VERSION = (
    "chemworld-reaction-crystallization-static-single-stage-recipe-0.2-s0-dev"
)
CRYSTALLIZATION_SINGLE_STAGE_DIMENSION = 10
CRYSTALLIZATION_SINGLE_STAGE_EVENT_COUNT = 12
CRYSTALLIZATION_SINGLE_STAGE_CATEGORICAL_COORDINATES = ((4, 4), (6, 4))
CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS = (
    {
        "slot_id": "diagnostic-01-hplc",
        "instrument": "hplc",
        "after_operation": "quench",
        "recipe_step_index": 5,
        "selection_policy": "agent_selectable",
        "scientific_role": "reaction_outcome_before_crystallization",
        "stage_id": "post_reaction_pre_crystallization",
        "model_facing_metric_ids": [
            "conversion",
            "yield",
            "selectivity",
            "byproduct_signal",
        ],
    },
    {
        "slot_id": "diagnostic-02-hplc",
        "instrument": "hplc",
        "after_operation": "cool_crystallize",
        "recipe_step_index": 8,
        "selection_policy": "agent_selectable",
        "scientific_role": "slurry_outcome_before_filtration",
        "stage_id": "post_crystallization_pre_filtration",
        "model_facing_metric_ids": [
            "crystal_purity",
            "yield",
            "byproduct_signal",
        ],
    },
)


def crystallization_single_stage_parameter_schema() -> dict[str, dict[str, Any]]:
    """Return the model-facing controls for one complete reaction/crystallization run."""

    return {
        "reaction_temperature_K": {
            "type": "number",
            "minimum": 333.15,
            "maximum": 423.15,
            "unit": "K",
        },
        "reaction_duration_s": {
            "type": "number",
            "minimum": 900.0,
            "maximum": 7200.0,
            "unit": "s",
        },
        "reagent_amount_mol": {
            "type": "number",
            "minimum": 0.003,
            "maximum": 0.030,
            "unit": "mol",
        },
        "stirring_speed_rpm": {
            "type": "number",
            "minimum": 300.0,
            "maximum": 1050.0,
            "unit": "rpm",
        },
        "catalyst": {"type": "integer", "minimum": 0, "maximum": 3},
        "catalyst_amount_mol": {
            "type": "number",
            "minimum": 0.00008,
            "maximum": 0.00055,
            "unit": "mol",
        },
        "solvent": {"type": "integer", "minimum": 0, "maximum": 3},
        "seed_mass_g": {
            "type": "number",
            "minimum": 0.001,
            "maximum": 0.015,
            "unit": "g",
        },
        "crystallization_temperature_K": {
            "type": "number",
            "minimum": 270.0,
            "maximum": 315.0,
            "unit": "K",
            "coupled_maximum": "min(315.0, reaction_temperature_K - 55.0)",
            "semantics": "post_quench_seeded_cooling_target",
        },
        "crystallization_duration_s": {
            "type": "number",
            "minimum": 600.0,
            "maximum": 14400.0,
            "unit": "s",
        },
    }


def crystallization_single_stage_parameters_from_unit_vector(
    vector: np.ndarray,
) -> dict[str, int | float]:
    values: NDArray[np.float64] = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != CRYSTALLIZATION_SINGLE_STAGE_DIMENSION:
        raise ValueError("reaction-crystallization recipe requires ten coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError("reaction-crystallization coordinates must be finite")
    values = np.clip(values, 0.0, 1.0)
    reaction_temperature = _bounded_scale(values[0], 333.15, 423.15)
    crystallization_temperature = min(
        _bounded_scale(values[8], 270.0, 315.0),
        reaction_temperature - 55.0,
    )
    return {
        "reaction_temperature_K": reaction_temperature,
        "reaction_duration_s": _bounded_scale(values[1], 900.0, 7200.0),
        "reagent_amount_mol": _bounded_scale(values[2], 0.003, 0.030),
        "stirring_speed_rpm": _bounded_scale(values[3], 300.0, 1050.0),
        "catalyst": _choice(values[4], 4),
        "catalyst_amount_mol": _bounded_scale(values[5], 0.00008, 0.00055),
        "solvent": _choice(values[6], 4),
        "seed_mass_g": _bounded_scale(values[7], 0.001, 0.015),
        "crystallization_temperature_K": float(np.clip(crystallization_temperature, 250.0, 315.0)),
        "crystallization_duration_s": _bounded_scale(values[9], 600.0, 14400.0),
    }


def crystallization_single_stage_unit_vector_from_parameters(
    payload: object,
) -> np.ndarray:
    schema = crystallization_single_stage_parameter_schema()
    if not isinstance(payload, Mapping) or set(payload) != set(schema):
        raise ValueError(
            "reaction-crystallization recipe_parameters fields do not match the contract"
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
    coupled_maximum = min(315.0, normalized["reaction_temperature_K"] - 55.0)
    if normalized["crystallization_temperature_K"] > coupled_maximum + 1.0e-12:
        raise ValueError(
            "crystallization_temperature_K must not exceed "
            "min(315 K, reaction_temperature_K - 55 K)"
        )
    return np.asarray(
        [
            _unscale(normalized["reaction_temperature_K"], 333.15, 423.15),
            _unscale(normalized["reaction_duration_s"], 900.0, 7200.0),
            _unscale(normalized["reagent_amount_mol"], 0.003, 0.030),
            _unscale(normalized["stirring_speed_rpm"], 300.0, 1050.0),
            (normalized["catalyst"] + 0.5) / 4.0,
            _unscale(normalized["catalyst_amount_mol"], 0.00008, 0.00055),
            (normalized["solvent"] + 0.5) / 4.0,
            _unscale(normalized["seed_mass_g"], 0.001, 0.015),
            _unscale(normalized["crystallization_temperature_K"], 270.0, 315.0),
            _unscale(normalized["crystallization_duration_s"], 600.0, 14400.0),
        ],
        dtype=float,
    )


def crystallization_single_stage_recipe_from_unit_vector(
    task_info: Mapping[str, Any],
    vector: np.ndarray,
) -> dict[str, Any]:
    parameters = crystallization_single_stage_parameters_from_unit_vector(vector)
    steps = [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": parameters["solvent"]},
        {"operation": "add_reagent", "amount_mol": parameters["reagent_amount_mol"]},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": parameters["catalyst_amount_mol"],
            "catalyst": parameters["catalyst"],
        },
        {
            "operation": "heat",
            "target_temperature_K": parameters["reaction_temperature_K"],
            "duration_s": parameters["reaction_duration_s"],
            "stirring_speed_rpm": parameters["stirring_speed_rpm"],
        },
        {"operation": "quench"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "seed_crystals", "seed_mass_g": parameters["seed_mass_g"]},
        {
            "operation": "cool_crystallize",
            "target_temperature_K": parameters["crystallization_temperature_K"],
            "duration_s": parameters["crystallization_duration_s"],
        },
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "filter_crystals"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    return {
        "steps": steps,
        "metadata": {
            "search_space_version": CRYSTALLIZATION_SINGLE_STAGE_RECIPE_VERSION,
            "search_space_kind": "reaction_crystallization",
            "workflow_mode": "static_single_stage",
            "task_id": task_info.get("task_id"),
            "search_vector": [
                float(value)
                for value in crystallization_single_stage_unit_vector_from_parameters(parameters)
            ],
            "recipe_parameters": parameters,
        },
    }


def _choice(value: float, count: int) -> int:
    return min(int(float(value) * count), count - 1)


def _bounded_scale(value: float, low: float, high: float) -> float:
    return min(max(float(low + float(value) * (high - low)), low), high)


def _unscale(value: float, low: float, high: float) -> float:
    return float((value - low) / (high - low))


__all__ = [
    "CRYSTALLIZATION_SINGLE_STAGE_CATEGORICAL_COORDINATES",
    "CRYSTALLIZATION_SINGLE_STAGE_DIMENSION",
    "CRYSTALLIZATION_SINGLE_STAGE_EVENT_COUNT",
    "CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS",
    "CRYSTALLIZATION_SINGLE_STAGE_RECIPE_VERSION",
    "crystallization_single_stage_parameter_schema",
    "crystallization_single_stage_parameters_from_unit_vector",
    "crystallization_single_stage_recipe_from_unit_vector",
    "crystallization_single_stage_unit_vector_from_parameters",
]
