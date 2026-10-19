"""Task-aware search recipes for official optimization baselines.

The public benchmark contains several different operation families.  A reaction
recipe is not a valid candidate for a partition, purification, flow,
electrochemical, or equilibrium task, so official search baselines use the
adapters in this module instead of sending one universal reaction sequence to
every environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_TASK_CONTRACT,
)
from chemworld.world.continuous_flow import FIXED_FLOW_REACTOR_VOLUME_L

TASK_RECIPE_SPACE_VERSION = "chemworld-task-recipe-space-1.3"
DIAGNOSTIC_RECIPE_DESIGN_V1 = "deterministic_task_aware_relational_diagnostic_design_v1"
DIAGNOSTIC_RECIPE_DESIGN_V2 = "deterministic_task_aware_relational_diagnostic_design_v2"
DIAGNOSTIC_RECIPE_DESIGNS = frozenset(
    {
        DIAGNOSTIC_RECIPE_DESIGN_V1,
        DIAGNOSTIC_RECIPE_DESIGN_V2,
    }
)

# Fixed-volume flow hardware derives residence time from Q. Complete-recipe
# baselines reserve at least one derived residence time before measurement.
FLOW_RECIPE_MAX_RESIDENCE_MULTIPLIER = 1.0

_CONSERVATIVE_BASE_VECTORS = {
    "equilibrium": (0.5, 0.12, 0.12),
    "flow": (0.0, 0.15, 0.0, 0.15, 0.70, 0.15, 0.10, 0.12),
    # Low substrate inventory with a moderate probe and a longer controlled
    # electrolysis step.  The former all-low vector drove most public worlds to
    # a zero-yield plateau and was not a useful diagnostic reference.
    "electrochemical": (0.625, 0.125, 0.01, 0.286, 0.472, 0.211, 0.254, 0.361, 0.387),
    "partition": (0.0, 0.35, 0.50, 0.0, 0.35, 0.15, 0.65, 0.15),
    "reaction_crystallization": (
        0.10,
        0.15,
        0.15,
        0.30,
        0.0,
        0.10,
        0.0,
        0.50,
        0.50,
        0.30,
    ),
    "reaction_distillation": (
        0.10,
        0.15,
        0.15,
        0.30,
        0.0,
        0.10,
        0.0,
        0.20,
        0.20,
        0.35,
        0.35,
        0.50,
        0.60,
    ),
    "reaction_purification": (
        0.10,
        0.15,
        0.15,
        0.30,
        0.0,
        0.10,
        0.0,
        0.50,
        0.0,
        0.50,
        0.30,
        0.40,
        0.30,
        0.30,
        0.30,
        0.80,
    ),
    "reaction": (0.10, 0.15, 0.15, 0.30, 0.0, 0.10, 0.0),
}


def task_recipe_kind(task_info: dict[str, Any]) -> str:
    operations = set(task_info.get("allowed_operations", ()))
    instruments = set(task_info.get("allowed_instruments", ()))
    if "run_flow" in operations:
        return "flow"
    if "electrolyze" in operations:
        return "electrochemical"
    if "ph_meter" in instruments:
        return "equilibrium"
    if "heat" in operations and {"add_phase", "wash", "transfer"} <= operations:
        return "reaction_purification"
    if "add_phase" in operations and "heat" not in operations:
        return "partition"
    if "cool_crystallize" in operations:
        return "reaction_crystallization"
    if "distill" in operations:
        return "reaction_distillation"
    return "reaction"


def task_recipe_dimension(task_info: dict[str, Any]) -> int:
    return {
        "equilibrium": 3,
        "flow": 8,
        "electrochemical": 9,
        "partition": 8,
        "reaction_crystallization": 10,
        "reaction_distillation": 13,
        "reaction_purification": 16,
        "reaction": 7,
    }[task_recipe_kind(task_info)]


def sample_task_recipe(
    task_info: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    vector = rng.random(task_recipe_dimension(task_info))
    return task_recipe_from_unit_vector(task_info, vector)


def sample_conservative_task_recipe(
    task_info: dict[str, Any],
    rng: np.random.Generator,
    *,
    perturbation_scale: float = 0.06,
) -> dict[str, Any]:
    """Sample around a public low-intensity starting design for safe exploration."""

    if not 0.0 <= perturbation_scale <= 0.25:
        raise ValueError("perturbation_scale must be between zero and 0.25")
    base = np.asarray(_CONSERVATIVE_BASE_VECTORS[task_recipe_kind(task_info)], dtype=float)
    perturbation = rng.normal(0.0, perturbation_scale, size=base.shape)
    return task_recipe_from_unit_vector(task_info, np.clip(base + perturbation, 0.0, 1.0))


def task_recipe_from_unit_vector(
    task_info: dict[str, Any],
    vector: np.ndarray,
) -> dict[str, Any]:
    dimension = task_recipe_dimension(task_info)
    raw_values = np.asarray(vector, dtype=float).reshape(-1)
    if raw_values.size != dimension:
        raise ValueError(f"Expected {dimension} task-recipe coordinates, got {raw_values.size}")
    values = np.asarray(np.clip(raw_values, 0.0, 1.0), dtype=float).reshape(-1)
    kind = task_recipe_kind(task_info)
    if kind == "equilibrium":
        steps = _equilibrium_steps(values)
    elif kind == "flow":
        steps = _flow_steps(values)
    elif kind == "electrochemical":
        steps = _electrochemical_steps(values)
    elif kind == "partition":
        steps = _partition_steps(values)
    else:
        steps = _reaction_steps(values, kind=kind)
    return {
        "steps": steps,
        "metadata": {
            "search_space_version": TASK_RECIPE_SPACE_VERSION,
            "search_space_kind": kind,
            "task_id": task_info.get("task_id"),
            "search_vector": [float(value) for value in values],
        },
    }


def task_recipe_to_vector(recipe: dict[str, Any]) -> np.ndarray:
    metadata = recipe.get("metadata")
    if not isinstance(metadata, dict) or "search_vector" not in metadata:
        raise ValueError("Task search recipe is missing metadata.search_vector")
    values = np.asarray(metadata["search_vector"], dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Task search vector must contain finite coordinates")
    return values


def task_recipe_to_model_vector(
    task_info: dict[str, Any],
    recipe: dict[str, Any],
) -> np.ndarray:
    """Encode a recipe without imposing ordinal distance on material choices."""

    values = task_recipe_to_vector(recipe)
    categorical_coordinates = task_recipe_categorical_coordinates(task_info)
    categorical_indices = {coordinate for coordinate, _ in categorical_coordinates}
    continuous = np.asarray(
        [value for index, value in enumerate(values) if index not in categorical_indices],
        dtype=float,
    )
    encoded = [continuous]
    for coordinate, category_count in categorical_coordinates:
        one_hot = np.zeros(category_count, dtype=float)
        one_hot[_choice(float(values[coordinate]), category_count)] = 1.0
        encoded.append(one_hot)
    return np.concatenate(encoded)


def task_recipe_categorical_coordinates(
    task_info: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return ``(coordinate, cardinality)`` pairs for nominal material choices."""

    return {
        "equilibrium": (),
        "flow": ((0, 4), (2, 4)),
        "electrochemical": ((0, 4), (1, 4)),
        "partition": ((0, 4), (3, 4)),
        "reaction_crystallization": ((4, 4), (6, 4)),
        "reaction_distillation": ((4, 4), (6, 4)),
        "reaction_purification": ((4, 4), (6, 4), (8, 4)),
        "reaction": ((4, 4), (6, 4)),
    }[task_recipe_kind(task_info)]


def task_recipe_event_count(task_info: dict[str, Any]) -> int:
    vector = np.full(task_recipe_dimension(task_info), 0.5, dtype=float)
    return len(task_recipe_from_unit_vector(task_info, vector)["steps"])


def task_recipe_coordinate_schema(task_info: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Describe public unit-vector coordinates in physical-control language."""

    kind = task_recipe_kind(task_info)
    if kind == "electrochemical":
        return (
            {
                "coordinate": 0,
                "control_id": "electrolyte_profile",
                "kind": "categorical",
                "category_count": 4,
            },
            {
                "coordinate": 1,
                "control_id": "solvent",
                "kind": "categorical",
                "category_count": 4,
            },
            {
                "coordinate": 2,
                "control_id": "reagent_amount_mol",
                "kind": "linear",
                "physical_bounds": [0.003, 0.030],
                "unit": "mol",
            },
            {
                "coordinate": 3,
                "control_id": "probe_potential_V",
                "kind": "linear",
                "physical_bounds": [0.65, 1.25],
                "unit": "V",
            },
            {
                "coordinate": 4,
                "control_id": "probe_current_mA",
                "kind": "linear_magnitude_cap",
                "physical_bounds": [15.0, 90.0],
                "unit": "mA",
            },
            {
                "coordinate": 5,
                "control_id": "probe_duration_s",
                "kind": "linear",
                "physical_bounds": [180.0, 900.0],
                "unit": "s",
            },
            {
                "coordinate": 6,
                "control_id": "controlled_potential_delta_V",
                "kind": "signed_delta_from_probe",
                "absolute_delta_bounds": [0.02, 0.55],
                "result_bounds": list(ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V),
                "unit": "V",
            },
            {
                "coordinate": 7,
                "control_id": "controlled_current_delta_mA",
                "kind": "signed_delta_from_probe_magnitude_cap",
                "absolute_delta_bounds": [1.0, 100.0],
                "result_bounds": list(ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA),
                "unit": "mA",
            },
            {
                "coordinate": 8,
                "control_id": "controlled_duration_s",
                "kind": "linear",
                "physical_bounds": [300.0, 3600.0],
                "unit": "s",
            },
        )
    common_reaction = (
        _linear_coordinate(0, "reaction_temperature_K", 333.15, 423.15, "K"),
        _linear_coordinate(1, "reaction_duration_s", 900.0, 7200.0, "s"),
        _linear_coordinate(2, "reagent_amount_mol", 0.003, 0.030, "mol"),
        _linear_coordinate(3, "stirring_speed_rpm", 300.0, 1050.0, "rpm"),
        _categorical_coordinate(4, "catalyst", 4),
        _linear_coordinate(5, "catalyst_amount_mol", 0.00008, 0.00055, "mol"),
        _categorical_coordinate(6, "solvent", 4),
    )
    if kind == "reaction":
        return common_reaction
    if kind == "reaction_crystallization":
        return (
            *common_reaction,
            _linear_coordinate(7, "seed_mass_g", 0.001, 0.015, "g"),
            {
                **_linear_coordinate(8, "crystallization_temperature_K", 270.0, 315.0, "K"),
                "coupled_maximum": "min(315.0, reaction_temperature_K - 55.0)",
            },
            _linear_coordinate(9, "crystallization_duration_s", 600.0, 14400.0, "s"),
        )
    if kind == "reaction_distillation":
        return (
            *common_reaction,
            _linear_coordinate(7, "evaporation_temperature_K", 315.0, 350.0, "K"),
            _linear_coordinate(8, "evaporation_duration_s", 300.0, 1500.0, "s"),
            _linear_coordinate(9, "distillation_temperature_K", 345.0, 395.0, "K"),
            _linear_coordinate(10, "distillation_duration_s", 900.0, 3600.0, "s"),
            _linear_coordinate(11, "reflux_ratio", 0.5, 5.0, "dimensionless"),
            _linear_coordinate(12, "transfer_fraction", 0.55, 0.99, "fraction"),
        )
    if kind == "reaction_purification":
        return (
            *common_reaction,
            _linear_coordinate(7, "aqueous_volume_L", 0.006, 0.024, "L"),
            _categorical_coordinate(8, "extractant", 4),
            _linear_coordinate(9, "extractant_volume_L", 0.008, 0.030, "L"),
            _linear_coordinate(10, "mix_duration_s", 60.0, 600.0, "s"),
            _linear_coordinate(11, "mix_stirring_speed_rpm", 300.0, 1100.0, "rpm"),
            _linear_coordinate(12, "settle_duration_s", 120.0, 1200.0, "s"),
            _linear_coordinate(13, "wash_volume_L", 0.003, 0.015, "L"),
            _linear_coordinate(14, "concentrate_duration_s", 300.0, 1500.0, "s"),
            _linear_coordinate(15, "transfer_fraction", 0.55, 0.99, "fraction"),
        )
    if kind == "flow":
        return (
            _categorical_coordinate(0, "solvent", 4),
            _linear_coordinate(1, "reagent_amount_mol", 0.003, 0.030, "mol"),
            _categorical_coordinate(2, "catalyst", 4),
            _linear_coordinate(3, "catalyst_amount_mol", 0.00008, 0.00055, "mol"),
            _linear_coordinate(4, "flow_rate_mL_min", 0.2, 4.0, "mL/min"),
            _linear_coordinate(5, "feed_volume_L", 0.015, 0.035, "L"),
            _linear_coordinate(6, "flow_temperature_K", 330.0, 430.0, "K"),
            {
                **_linear_coordinate(7, "requested_run_duration_s", 600.0, 3600.0, "s"),
                "coupled_minimum": "fixed_hardware_derived_residence_time_s",
            },
        )
    if kind == "partition":
        return (
            _categorical_coordinate(0, "solvent", 4),
            _linear_coordinate(1, "solvent_volume_L", 0.012, 0.028, "L"),
            _linear_coordinate(2, "aqueous_volume_L", 0.006, 0.024, "L"),
            _categorical_coordinate(3, "extractant", 4),
            _linear_coordinate(4, "extractant_volume_L", 0.008, 0.030, "L"),
            _linear_coordinate(5, "mix_duration_s", 60.0, 600.0, "s"),
            _linear_coordinate(6, "settle_duration_s", 120.0, 1200.0, "s"),
            _linear_coordinate(7, "stirring_speed_rpm", 300.0, 1100.0, "rpm"),
        )
    if kind == "equilibrium":
        return (
            _linear_coordinate(0, "aqueous_volume_L", 0.018, 0.040, "L"),
            _linear_coordinate(1, "first_reagent_amount_mol", 0.001, 0.018, "mol"),
            _linear_coordinate(2, "second_reagent_amount_mol", 0.001, 0.018, "mol"),
        )
    raise ValueError(f"unsupported task recipe kind: {kind}")


def _linear_coordinate(
    coordinate: int,
    control_id: str,
    low: float,
    high: float,
    unit: str,
) -> dict[str, Any]:
    return {
        "coordinate": coordinate,
        "control_id": control_id,
        "kind": "linear",
        "physical_bounds": [low, high],
        "unit": unit,
    }


def _categorical_coordinate(
    coordinate: int, control_id: str, category_count: int
) -> dict[str, Any]:
    return {
        "coordinate": coordinate,
        "control_id": control_id,
        "kind": "categorical",
        "category_count": category_count,
        "selection_semantics": "independent_unordered_nominal_choice",
        "numeric_order_has_scientific_meaning": False,
        "numeric_distance_has_scientific_meaning": False,
    }


def task_recipe_public_controls(
    task_info: dict[str, Any], vector: np.ndarray
) -> tuple[dict[str, Any], ...]:
    """Decode a public vector into the non-measurement controls it executes."""

    recipe = task_recipe_from_unit_vector(task_info, vector)
    return tuple(
        {"step_index": step_index, **dict(step)}
        for step_index, step in enumerate(recipe["steps"])
        if step.get("operation") not in {"measure", "terminate"}
    )


def electrochemical_recipe_parameter_schema() -> dict[str, dict[str, Any]]:
    """Model-facing physical parameter contract for one electrochemical experiment."""

    return {
        "electrolyte_profile": {"type": "integer", "minimum": 0, "maximum": 3},
        "solvent": {"type": "integer", "minimum": 0, "maximum": 3},
        "reagent_amount_mol": {"type": "number", "minimum": 0.003, "maximum": 0.030},
        "probe_potential_V": {"type": "number", "minimum": 0.65, "maximum": 1.25},
        "probe_current_mA": {
            "type": "number",
            "minimum": 15.0,
            "maximum": 90.0,
            "semantics": "nonnegative_magnitude_cap",
        },
        "probe_duration_s": {"type": "number", "minimum": 180.0, "maximum": 900.0},
        "controlled_potential_V": {
            "type": "number",
            "minimum": ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V[0],
            "maximum": ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V[1],
            "minimum_absolute_change_from_probe": 0.02,
            "maximum_absolute_change_from_probe": 0.55,
        },
        "controlled_current_mA": {
            "type": "number",
            "minimum": ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA[0],
            "maximum": ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA[1],
            "minimum_absolute_change_from_probe": 1.0,
            "maximum_absolute_change_from_probe": 100.0,
            "semantics": "nonnegative_magnitude_cap",
        },
        "controlled_duration_s": {
            "type": "number",
            "minimum": 300.0,
            "maximum": 3600.0,
        },
    }


def electrochemical_recipe_parameters_from_unit_vector(
    vector: np.ndarray,
) -> dict[str, int | float]:
    """Decode the legacy optimizer coordinates into physical model-facing fields."""

    task_info = {"allowed_operations": ["electrolyze"], "allowed_instruments": []}
    values = np.asarray(np.clip(vector, 0.0, 1.0), dtype=float).reshape(-1)
    if values.size != 9:
        raise ValueError("electrochemical recipe requires nine internal coordinates")
    steps = task_recipe_from_unit_vector(task_info, values)["steps"]
    probe = steps[2]
    controlled = steps[6]
    return {
        "electrolyte_profile": int(probe["electrolyte_profile"]),
        "solvent": int(steps[0]["solvent"]),
        "reagent_amount_mol": float(steps[1]["amount_mol"]),
        "probe_potential_V": float(probe["potential_V"]),
        "probe_current_mA": float(probe["current_mA"]),
        "probe_duration_s": float(steps[3]["duration_s"]),
        "controlled_potential_V": float(controlled["potential_V"]),
        "controlled_current_mA": float(controlled["current_mA"]),
        "controlled_duration_s": float(steps[7]["duration_s"]),
    }


def electrochemical_recipe_unit_vector_from_parameters(
    payload: object,
) -> np.ndarray:
    """Validate physical electrochemical controls and encode them deterministically."""

    schema = electrochemical_recipe_parameter_schema()
    if not isinstance(payload, dict) or set(payload) != set(schema):
        raise ValueError("electrochemical recipe_parameters fields do not match the contract")
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
        if not float(specification["minimum"]) <= numeric <= float(specification["maximum"]):
            raise ValueError(f"{field} is outside its physical bounds")
        normalized[field] = numeric

    potential_delta = normalized["controlled_potential_V"] - normalized["probe_potential_V"]
    current_delta = normalized["controlled_current_mA"] - normalized["probe_current_mA"]
    if not 0.02 <= abs(potential_delta) <= 0.55:
        raise ValueError("controlled_potential_V must differ from the probe by 0.02 to 0.55 V")
    if not 1.0 <= abs(current_delta) <= 100.0:
        raise ValueError("controlled_current_mA must differ from the probe by 1 to 100 mA")

    return np.asarray(
        [
            (normalized["electrolyte_profile"] + 0.5) / 4.0,
            (normalized["solvent"] + 0.5) / 4.0,
            _unscale(normalized["reagent_amount_mol"], 0.003, 0.030),
            _unscale(normalized["probe_potential_V"], 0.65, 1.25),
            _unscale(normalized["probe_current_mA"], 15.0, 90.0),
            _unscale(normalized["probe_duration_s"], 180.0, 900.0),
            _signed_delta_coordinate(potential_delta, 0.02, 0.55),
            _signed_delta_coordinate(current_delta, 1.0, 100.0),
            _unscale(normalized["controlled_duration_s"], 300.0, 3600.0),
        ],
        dtype=float,
    )


def diagnostic_task_recipe_vectors(
    task_info: dict[str, Any],
    *,
    action_count: int,
    rng: np.random.Generator,
    design_id: str = DIAGNOSTIC_RECIPE_DESIGN_V1,
) -> tuple[np.ndarray, ...]:
    """Build a deterministic task-aware diagnostic design in the unit cube.

    Coupled coordinates should not all move together: doing so can confound a
    physical response with an invalid or zero-performance recipe.  The generic
    fallback retains broad anchors. Reaction-crystallization tasks use paired
    catalyst-dose, thermal, and catalyst-identity probes; electrochemical tasks
    use a reference recipe, a same-material potential sweep, and one probe for
    each categorical material coordinate.
    """

    if design_id not in DIAGNOSTIC_RECIPE_DESIGNS:
        raise ValueError(f"unsupported diagnostic recipe design: {design_id}")
    if action_count < 3:
        raise ValueError("a diagnostic recipe design requires at least three actions")
    kind = task_recipe_kind(task_info)
    dimension = task_recipe_dimension(task_info)
    if kind == "reaction_crystallization":
        reaction_reference = np.full(dimension, 0.50, dtype=float)
        # Keep solvent fixed and use catalyst 0 for the kinetic controls.
        reaction_reference[4] = 0.125
        reaction_reference[6] = 0.125
        reaction_vectors: list[np.ndarray] = []

        # A low/mid/high series crosses the formal rate-law activity pivot
        # while every other process coordinate remains fixed.
        for catalyst_dose in (0.05, 0.36, 0.90):
            dose_vector = np.array(reaction_reference, copy=True)
            dose_vector[5] = catalyst_dose
            reaction_vectors.append(dose_vector)

        # A thermal perturbation separates catalyst-order response from a
        # generic temperature/residence response.
        thermal = np.array(reaction_reference, copy=True)
        thermal[0] = 0.78
        thermal[1] = 0.32
        thermal[5] = 0.36
        reaction_vectors.append(thermal)

        # Probe the category opposite the reference catalyst while retaining
        # identical continuous controls.  This creates a true material pair
        # with the mid-dose reference recipe above.
        material = np.array(reaction_reference, copy=True)
        material[4] = 0.625
        material[5] = 0.36
        reaction_vectors.append(material)

        # Pair the high-dose reference with the alternative catalyst too; a
        # material permutation must be able to move the best high-dose policy,
        # not merely perturb a dominated mid-dose recipe.
        high_dose_material = np.array(reaction_reference, copy=True)
        high_dose_material[4] = 0.625
        high_dose_material[5] = 0.90
        reaction_vectors.append(high_dose_material)

        while len(reaction_vectors) < action_count:
            jittered = np.asarray(
                np.clip(
                    reaction_reference + rng.normal(0.0, 0.06, size=dimension),
                    0.0,
                    1.0,
                ),
                dtype=float,
            )
            jittered[4] = ((len(reaction_vectors) % 4) + 0.5) / 4.0
            reaction_vectors.append(jittered)
        return tuple(reaction_vectors[:action_count])

    if kind != "electrochemical":
        anchors = [
            np.full(dimension, 0.15, dtype=float),
            np.full(dimension, 0.50, dtype=float),
            np.full(dimension, 0.85, dtype=float),
        ]
        return tuple(anchors + [rng.random(dimension) for _ in range(action_count - len(anchors))])

    electro_reference = np.asarray(_CONSERVATIVE_BASE_VECTORS[kind], dtype=float)
    # Probe-potential and controlled-potential-delta coordinates.  All other
    # continuous controls remain at the same public reference so this is a
    # relational diagnostic sweep rather than a correlated random recipe.
    potential_profiles = (
        (0.286, 0.254),
        (0.500, 0.750),
        (0.650, 0.800),
        (0.200, 0.500),
    )
    electrochemical_vectors: list[np.ndarray] = []
    potential_profile_limit = (
        min(action_count, 3) if design_id == DIAGNOSTIC_RECIPE_DESIGN_V2 else action_count
    )
    for probe_potential, controlled_delta in potential_profiles[:potential_profile_limit]:
        potential_vector = np.array(electro_reference, copy=True)
        potential_vector[3] = probe_potential
        potential_vector[6] = controlled_delta
        electrochemical_vectors.append(potential_vector)

    categorical_coordinates = task_recipe_categorical_coordinates(task_info)
    if design_id == DIAGNOSTIC_RECIPE_DESIGN_V2 and len(electrochemical_vectors) < action_count:
        # Every categorical intervention needs a shared, same-condition
        # reference.  Without it, two alternative-material probes may differ
        # in multiple categorical fields and cannot isolate either mapping.
        material_reference = np.array(electro_reference, copy=True)
        material_reference[3] = 0.400
        material_reference[6] = 0.720
        electrochemical_vectors.append(material_reference)

    for coordinate, category_count in categorical_coordinates:
        if len(electrochemical_vectors) >= action_count:
            break
        material_vector = (
            np.array(material_reference, copy=True)
            if design_id == DIAGNOSTIC_RECIPE_DESIGN_V2
            else np.array(electro_reference, copy=True)
        )
        if design_id == DIAGNOSTIC_RECIPE_DESIGN_V1:
            material_vector[3] = 0.400
            material_vector[6] = 0.720
        baseline_category = _choice(
            float(electro_reference[coordinate]),
            category_count,
        )
        alternative = (baseline_category + max(category_count // 2, 1)) % category_count
        material_vector[coordinate] = (alternative + 0.5) / category_count
        electrochemical_vectors.append(material_vector)

    while len(electrochemical_vectors) < action_count:
        random_vector = np.asarray(
            np.clip(
                electro_reference + rng.normal(0.0, 0.08, size=dimension),
                0.0,
                1.0,
            ),
            dtype=float,
        )
        for offset, (coordinate, category_count) in enumerate(categorical_coordinates):
            category = (len(electrochemical_vectors) + offset) % category_count
            random_vector[coordinate] = (category + 0.5) / category_count
        electrochemical_vectors.append(random_vector)
    return tuple(electrochemical_vectors)


def _scale(value: float, low: float, high: float) -> float:
    return low + float(value) * (high - low)


def _unscale(value: float, low: float, high: float) -> float:
    return (float(value) - low) / (high - low)


def _signed_delta_coordinate(delta: float, low: float, high: float) -> float:
    magnitude_coordinate = _unscale(abs(delta), low, high)
    return (1.0 + magnitude_coordinate) / 2.0 if delta > 0.0 else (1.0 - magnitude_coordinate) / 2.0


def _choice(value: float, count: int) -> int:
    return min(int(float(value) * count), count - 1)


def _reaction_charge_steps(values: np.ndarray) -> list[dict[str, Any]]:
    temperature_K = _scale(values[0], 333.15, 423.15)
    duration_s = _scale(values[1], 900.0, 7200.0)
    amount_mol = _scale(values[2], 0.003, 0.030)
    stirring_speed_rpm = _scale(values[3], 300.0, 1050.0)
    catalyst = _choice(values[4], 4)
    solvent = _choice(values[6], 4)
    return [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": solvent},
        {"operation": "add_reagent", "amount_mol": amount_mol},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": _scale(values[5], 0.00008, 0.00055),
            "catalyst": catalyst,
        },
        {
            "operation": "heat",
            "target_temperature_K": temperature_K,
            "duration_s": duration_s,
            "stirring_speed_rpm": stirring_speed_rpm,
        },
        {"operation": "quench"},
    ]


def _reaction_steps(values: np.ndarray, *, kind: str) -> list[dict[str, Any]]:
    steps = _reaction_charge_steps(values)
    if kind == "reaction_crystallization":
        reaction_temperature_K = _scale(values[0], 333.15, 423.15)
        cooling_temperature_K = float(
            np.clip(
                min(
                    _scale(values[8], 270.0, 315.0),
                    reaction_temperature_K - 55.0,
                ),
                250.0,
                315.0,
            )
        )
        steps.extend(
            [
                {"operation": "measure", "instrument": "hplc"},
                {
                    "operation": "seed_crystals",
                    "seed_mass_g": _scale(values[7], 0.001, 0.015),
                },
                {
                    "operation": "cool_crystallize",
                    "target_temperature_K": cooling_temperature_K,
                    "duration_s": _scale(values[9], 600.0, 14_400.0),
                },
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "filter_crystals"},
            ]
        )
    elif kind == "reaction_distillation":
        steps.extend(
            [
                {"operation": "measure", "instrument": "hplc"},
                {
                    "operation": "evaporate",
                    "target_temperature_K": _scale(values[7], 315.0, 350.0),
                    "duration_s": _scale(values[8], 300.0, 1500.0),
                },
                {
                    "operation": "distill",
                    "target_temperature_K": _scale(values[9], 345.0, 395.0),
                    "duration_s": _scale(values[10], 900.0, 3600.0),
                    "reflux_ratio": _scale(values[11], 0.5, 5.0),
                },
                {
                    "operation": "collect_fraction",
                    "transfer_fraction": _scale(values[12], 0.55, 0.99),
                },
                {"operation": "measure", "instrument": "gc"},
            ]
        )
    elif kind == "reaction_purification":
        steps.extend(
            [
                {"operation": "measure", "instrument": "hplc"},
                {
                    "operation": "add_phase",
                    "phase": "aqueous",
                    "volume_L": _scale(values[7], 0.006, 0.024),
                },
                {
                    "operation": "add_extractant",
                    "extractant": _choice(values[8], 4),
                    "volume_L": _scale(values[9], 0.008, 0.030),
                },
                {
                    "operation": "mix",
                    "duration_s": _scale(values[10], 60.0, 600.0),
                    "stirring_speed_rpm": _scale(values[11], 300.0, 1100.0),
                },
                {
                    "operation": "settle",
                    "duration_s": _scale(values[12], 120.0, 1200.0),
                },
                {"operation": "separate_phase", "target_phase": "organic"},
                {"operation": "measure", "instrument": "hplc"},
                {
                    "operation": "wash",
                    "wash_volume_L": _scale(values[13], 0.003, 0.015),
                },
                {"operation": "dry"},
                {
                    "operation": "concentrate",
                    "duration_s": _scale(values[14], 300.0, 1500.0),
                },
                {
                    "operation": "transfer",
                    "transfer_fraction": _scale(values[15], 0.55, 0.99),
                },
                {"operation": "measure", "instrument": "hplc"},
            ]
        )
    else:
        steps.append({"operation": "measure", "instrument": "hplc"})
    steps.extend(
        [
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
    )
    return steps


def _partition_steps(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "operation": "add_solvent",
            "volume_L": _scale(values[1], 0.012, 0.028),
            "solvent": _choice(values[0], 4),
        },
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": _scale(values[2], 0.006, 0.024),
        },
        {
            "operation": "add_extractant",
            "extractant": _choice(values[3], 4),
            "volume_L": _scale(values[4], 0.008, 0.030),
        },
        {
            "operation": "mix",
            "duration_s": _scale(values[5], 60.0, 600.0),
            "stirring_speed_rpm": _scale(values[7], 300.0, 1100.0),
        },
        {"operation": "settle", "duration_s": _scale(values[6], 120.0, 1200.0)},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def _flow_steps(values: np.ndarray) -> list[dict[str, Any]]:
    flow_rate_mL_min = _scale(values[4], 0.2, 4.0)
    residence_time_s = FIXED_FLOW_REACTOR_VOLUME_L * 1000.0 * 60.0 / flow_rate_mL_min
    requested_duration_s = _scale(values[7], 600.0, 3600.0)
    run_duration_s = max(
        requested_duration_s,
        residence_time_s * FLOW_RECIPE_MAX_RESIDENCE_MULTIPLIER,
    )
    return [
        {
            "operation": "add_solvent",
            "volume_L": _scale(values[5], 0.015, 0.035),
            "solvent": _choice(values[0], 4),
        },
        {"operation": "add_reagent", "amount_mol": _scale(values[1], 0.003, 0.030)},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": _scale(values[3], 0.00008, 0.00055),
            "catalyst": _choice(values[2], 4),
        },
        {
            "operation": "set_flow_rate",
            "flow_rate_mL_min": flow_rate_mL_min,
            "residence_time_s": residence_time_s,
        },
        {
            "operation": "run_flow",
            "target_temperature_K": _scale(values[6], 330.0, 430.0),
            "duration_s": run_duration_s,
        },
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def _electrochemical_steps(values: np.ndarray) -> list[dict[str, Any]]:
    probe_potential = _scale(values[3], 0.65, 1.25)
    probe_current = _scale(values[4], 15.0, 90.0)
    potential_delta = np.copysign(
        _scale(abs(2.0 * values[6] - 1.0), 0.02, 0.55),
        2.0 * values[6] - 1.0 if values[6] != 0.5 else 1.0,
    )
    controlled_potential = probe_potential + potential_delta
    potential_low, potential_high = ELECTROCHEMICAL_TASK_CONTRACT.s0_potential_bounds_V
    if not potential_low <= controlled_potential <= potential_high:
        controlled_potential = probe_potential - potential_delta
    current_delta = np.copysign(
        _scale(abs(2.0 * values[7] - 1.0), 1.0, 100.0),
        2.0 * values[7] - 1.0 if values[7] != 0.5 else 1.0,
    )
    controlled_current = float(
        np.clip(
            probe_current + current_delta,
            *ELECTROCHEMICAL_TASK_CONTRACT.s0_current_magnitude_bounds_mA,
        )
    )
    return [
        {
            "operation": "add_solvent",
            "volume_L": 0.025,
            "solvent": _choice(values[1], 4),
        },
        {"operation": "add_reagent", "amount_mol": _scale(values[2], 0.003, 0.030)},
        {
            "operation": "set_potential",
            "potential_V": probe_potential,
            "current_mA": probe_current,
            "electrolyte_profile": _choice(values[0], 4),
        },
        {"operation": "electrolyze", "duration_s": _scale(values[5], 180.0, 900.0)},
        {"operation": "measure", "instrument": "ph_meter"},
        {"operation": "measure", "instrument": "uvvis"},
        {
            "operation": "set_potential",
            "potential_V": controlled_potential,
            "current_mA": controlled_current,
            "electrolyte_profile": _choice(values[0], 4),
        },
        {"operation": "electrolyze", "duration_s": _scale(values[8], 300.0, 3600.0)},
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def _equilibrium_steps(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "operation": "add_solvent",
            "volume_L": _scale(values[0], 0.018, 0.040),
            "solvent": 0,
        },
        {"operation": "add_reagent", "amount_mol": _scale(values[1], 0.001, 0.018)},
        {"operation": "measure", "instrument": "ph_meter"},
        {"operation": "add_reagent", "amount_mol": _scale(values[2], 0.001, 0.018)},
        {"operation": "measure", "instrument": "ph_meter"},
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


__all__ = [
    "DIAGNOSTIC_RECIPE_DESIGNS",
    "DIAGNOSTIC_RECIPE_DESIGN_V1",
    "DIAGNOSTIC_RECIPE_DESIGN_V2",
    "FLOW_RECIPE_MAX_RESIDENCE_MULTIPLIER",
    "TASK_RECIPE_SPACE_VERSION",
    "diagnostic_task_recipe_vectors",
    "electrochemical_recipe_parameter_schema",
    "electrochemical_recipe_parameters_from_unit_vector",
    "electrochemical_recipe_unit_vector_from_parameters",
    "sample_conservative_task_recipe",
    "sample_task_recipe",
    "task_recipe_categorical_coordinates",
    "task_recipe_coordinate_schema",
    "task_recipe_dimension",
    "task_recipe_event_count",
    "task_recipe_from_unit_vector",
    "task_recipe_kind",
    "task_recipe_public_controls",
    "task_recipe_to_model_vector",
    "task_recipe_to_vector",
]
