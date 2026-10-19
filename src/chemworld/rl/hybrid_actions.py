"""Conditional hybrid action and policy semantics for RL adapters.

The benchmark action is categorical at the operation level and exposes only
the parameters required by the selected operation.  Stable-Baselines3 does not
provide a native categorical-plus-continuous policy distribution, so PPO/SAC
use an explicitly labelled Box latent adapter.  Irrelevant latent coordinates
never enter the executed action or trajectory digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from chemworld.data.logging import to_builtin
from chemworld.operation_validator import OPERATION_AFFORDANCE_STATE_MACHINE_VERSION
from chemworld.world.operations import OPERATION_TYPES, operation_contracts

ACTION_SCHEMA_VERSION = "chemworld-conditional-hybrid-action-0.8"
LATENT_ADAPTER_VERSION = "chemworld-sb3-box-latent-adapter-0.2"
POLICY_DISTRIBUTION_SCHEMA_VERSION = "chemworld-masked-conditional-ppo-0.1"


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _space_contract(space: gym.Space[Any]) -> dict[str, Any]:
    if isinstance(space, gym.spaces.Discrete):
        return {"kind": "categorical", "cardinality": int(space.n)}
    if isinstance(space, gym.spaces.Box):
        return {
            "kind": "continuous",
            "shape": list(space.shape or ()),
            "low": to_builtin(np.asarray(space.low)),
            "high": to_builtin(np.asarray(space.high)),
            "dtype": str(space.dtype),
        }
    raise TypeError(f"unsupported event action component: {type(space).__name__}")


def conditional_hybrid_action_contract(
    event_action_space: gym.spaces.Dict,
) -> dict[str, Any]:
    """Return the canonical semantic contract and its deterministic digest."""

    parameter_keys = tuple(key for key in event_action_space.spaces if key != "operation")
    contracts = operation_contracts()
    conditional_parameters = {
        operation: [
            {
                "field": field,
                **_space_contract(event_action_space[field]),
            }
            for field in contracts[operation].required_fields
        ]
        for operation in OPERATION_TYPES
    }
    payload: dict[str, Any] = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "semantic_action": {
            "operation": {
                "kind": "categorical",
                "categories": list(OPERATION_TYPES),
                "selection_policy": "public-affordance-masked categorical selection",
            },
            "parameters": {
                "kind": "operation_conditional",
                "by_operation": conditional_parameters,
                "irrelevant_parameter_policy": "excluded_from_execution_and_trajectory",
            },
        },
        "training_adapter": {
            "schema_version": LATENT_ADAPTER_VERSION,
            "framework_scope": ["stable-baselines3:PPO", "stable-baselines3:SAC"],
            "kind": "bounded_box_latent",
            "shape": [len(OPERATION_TYPES) + len(parameter_keys)],
            "bounds": [-1.0, 1.0],
            "operation_coordinate_count": len(OPERATION_TYPES),
            "parameter_coordinate_keys": list(parameter_keys),
            "operation_decode": "masked_argmax",
            "native_hybrid_distribution": False,
        },
        "execution_projection": {
            "source": "public action_schema(operation) and validate_action(action)",
            "affordance_state_machine_version": (OPERATION_AFFORDANCE_STATE_MACHINE_VERSION),
            "state_dependent_numeric_bounds": True,
            "state_dependent_categorical_choices": True,
            "cross_field_constraints": [
                "maximum_cooling_rate_K_s",
                "absolute_value_upper_bound",
                "vessel_pressure_capacity",
            ],
            "invalid_projection_policy": (
                "retain the decoded action and let the public validator record failure"
            ),
            "hidden_state_access": False,
        },
        "crystallization_completion_affordance": {
            "initial_seed_requirements": [
                "reaction_advance_count > 0",
                "non-final process assay at current process time",
            ],
            "post_seed_allowed_operations": [
                "seed_crystals",
                "measure",
                "cool_crystallize",
                "terminate",
            ],
            "post_cooling_filter_requirement": (
                "non-final slurry assay at current process time"
            ),
            "post_cooling_allowed_operations": [
                "measure",
                "filter_crystals",
                "terminate",
            ],
            "post_filter_allowed_operations": ["measure", "terminate"],
            "public_state_sources": [
                "process.reaction_advance_count",
                "instrument:<id>.last_time_s",
                "crystallizer.crystal_seeded",
                "crystallizer.execution_history",
                "crystal_filter.crystals_filtered",
                "vessel.max_pressure_Pa",
                "state.temperature_K",
                "state.volume_L",
                "selected_phase.species_amounts_mol",
            ],
        },
        "empty_operation_mask_policy": (
            "retain global argmax so the environment records the invalid transition"
        ),
        "numeric_policy": (
            "map latent coordinates through current public field bounds before execution"
        ),
    }
    payload["contract_hash"] = _sha256_json(payload)
    return payload


def policy_distribution_contract(parameter_keys: tuple[str, ...]) -> dict[str, Any]:
    """Return the dependency-free PPO probability-law contract.

    Preflight and checkpoint audits can verify this payload without importing
    Torch or Stable-Baselines3.  The executable distribution in
    :mod:`chemworld.rl.hybrid_policy` consumes the same contract.
    """

    if not parameter_keys or len(set(parameter_keys)) != len(parameter_keys):
        raise ValueError("policy parameter keys must be non-empty and unique")
    contracts = operation_contracts()
    payload: dict[str, Any] = {
        "schema_version": POLICY_DISTRIBUTION_SCHEMA_VERSION,
        "operation_distribution": "public-affordance-masked categorical",
        "parameter_distribution": "operation-conditional diagonal Gaussian",
        "parameter_keys": list(parameter_keys),
        "active_parameters": {
            operation: list(contracts[operation].required_fields) for operation in OPERATION_TYPES
        },
        "irrelevant_parameter_log_prob": False,
        "irrelevant_parameter_entropy": False,
        "box_carrier_is_semantic_distribution": False,
    }
    payload["contract_hash"] = _sha256_json(payload)
    return payload


def decode_conditional_hybrid_action(
    action: Any,
    *,
    event_action_space: gym.spaces.Dict,
    operation_mask: list[bool] | np.ndarray,
    operation_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode one SB3 latent vector into a categorical, conditional action."""

    parameter_keys = tuple(key for key in event_action_space.spaces if key != "operation")
    expected_shape = (len(OPERATION_TYPES) + len(parameter_keys),)
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != expected_shape or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"hybrid action latent must be a finite vector with shape {expected_shape}"
        )

    public_mask = np.asarray(operation_mask, dtype=bool)
    if public_mask.shape != (len(OPERATION_TYPES),):
        raise ValueError("public operation mask does not match the operation registry")
    logits = vector[: len(OPERATION_TYPES)]
    operation_index = int(
        np.argmax(np.where(public_mask, logits, -np.inf))
        if np.any(public_mask)
        else np.argmax(logits)
    )
    operation = OPERATION_TYPES[operation_index]
    required = set(operation_contracts()[operation].required_fields)
    field_schemas: dict[str, Mapping[str, Any]] = {}
    if operation_schema is not None:
        if (
            operation_schema.get("operation") != operation
            or operation_schema.get("valid_operation_type") is not True
            or set(operation_schema.get("required_fields", ())) != required
        ):
            raise ValueError("public operation schema does not match the selected operation")
        fields = operation_schema.get("fields")
        if not isinstance(fields, list):
            raise ValueError("public operation schema is missing its field contracts")
        field_schemas = {
            str(field["field"]): field
            for field in fields
            if isinstance(field, Mapping) and isinstance(field.get("field"), str)
        }
    unit = np.clip((vector[len(OPERATION_TYPES) :] + 1.0) / 2.0, 0.0, 1.0)
    payload: dict[str, Any] = {"operation": operation_index}
    for index, key in enumerate(parameter_keys):
        if key not in required:
            continue
        space = event_action_space[key]
        coordinate = float(unit[index])
        field_schema = field_schemas.get(key, {})
        choices = field_schema.get("choices")
        bounds = field_schema.get("bounds")
        if isinstance(choices, list) and choices:
            payload[key] = choices[min(int(coordinate * len(choices)), len(choices) - 1)]
        elif isinstance(space, gym.spaces.Discrete):
            payload[key] = min(int(coordinate * space.n), space.n - 1)
        elif isinstance(space, gym.spaces.Box):
            low = float(bounds["low"]) if isinstance(bounds, Mapping) else float(space.low[0])
            high = float(bounds["high"]) if isinstance(bounds, Mapping) else float(space.high[0])
            payload[key] = _box_coordinate_payload(
                space,
                coordinate=coordinate,
                low=low,
                high=high,
                lower_inclusive=field_schema.get("lower_bound_inclusive") is not False,
                upper_inclusive=field_schema.get("upper_bound_inclusive") is not False,
            )
        else:
            raise TypeError(f"unsupported event action component: {key}={type(space).__name__}")
    if operation_schema is not None:
        _apply_public_cross_field_constraints(
            payload,
            operation_schema=operation_schema,
            event_action_space=event_action_space,
        )
    return payload


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _box_payload(space: gym.spaces.Box, value: float) -> Any:
    return to_builtin(np.full(space.shape, value, dtype=space.dtype))


def _representable_bound(
    value: float,
    *,
    toward: float,
    inclusive: bool,
    lower: bool,
    dtype: np.dtype[Any],
) -> float:
    projected = np.asarray(value, dtype=dtype)
    scalar = float(projected)
    violates = scalar < value if lower and inclusive else scalar <= value if lower else False
    if not lower:
        violates = scalar > value if inclusive else scalar >= value
    if violates:
        projected = np.nextafter(projected, np.asarray(toward, dtype=dtype))
        scalar = float(projected)
    return scalar


def _box_coordinate_payload(
    space: gym.spaces.Box,
    *,
    coordinate: float,
    low: float,
    high: float,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> Any:
    dtype = space.dtype
    if dtype is None or not low <= high or not np.issubdtype(dtype, np.floating):
        raise ValueError("public numeric bounds must define a finite floating interval")
    dtype = np.dtype(dtype)
    representable_low = _representable_bound(
        low,
        toward=high,
        inclusive=lower_inclusive,
        lower=True,
        dtype=dtype,
    )
    representable_high = _representable_bound(
        high,
        toward=low,
        inclusive=upper_inclusive,
        lower=False,
        dtype=dtype,
    )
    if representable_low > representable_high:
        raise ValueError("public numeric interval has no value representable by the action space")
    value = representable_low + coordinate * (representable_high - representable_low)
    value = float(np.asarray(value, dtype=dtype))
    return _box_payload(space, float(np.clip(value, representable_low, representable_high)))


def _box_payload_at_least(space: gym.spaces.Box, value: float) -> Any:
    projected = np.asarray(value, dtype=space.dtype)
    if float(projected) < value:
        projected = np.nextafter(projected, np.asarray(np.inf, dtype=space.dtype))
    return _box_payload(space, float(projected))


def _box_payload_within_abs(space: gym.spaces.Box, value: float, limit: float) -> Any:
    projected = np.asarray(np.clip(value, -limit, limit), dtype=space.dtype)
    if abs(float(projected)) > limit:
        projected = np.nextafter(projected, np.asarray(0.0, dtype=space.dtype))
    return _box_payload(space, float(projected))


def _apply_public_cross_field_constraints(
    payload: dict[str, Any],
    *,
    operation_schema: Mapping[str, Any],
    event_action_space: gym.spaces.Dict,
) -> None:
    constraints = operation_schema.get("constraints")
    if not isinstance(constraints, list):
        return
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            continue
        kind = constraint.get("kind")
        parameters = constraint.get("parameters")
        if kind == "material_routes" and constraint.get("routes"):
            routes = constraint["routes"]
            route = min(routes, key=lambda item: (
                item["source_container"] != payload["source_container"],
                item["destination_container"] != payload["destination_container"],
            ))
            payload["source_container"] = route["source_container"]
            payload["destination_container"] = route["destination_container"]
            payload["mixing"] = max(int(payload["mixing"]), route["minimum_mixing"])
            maximum = float(route["maximum_fraction"])
            space = event_action_space["transfer_fraction"]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError("transfer_fraction must use a Box action component")
            value = float(np.asarray(min(_scalar(payload["transfer_fraction"]), maximum),
                                     dtype=space.dtype))
            if value > maximum:
                value = float(np.nextafter(np.asarray(value, dtype=space.dtype),
                                           np.asarray(0, dtype=space.dtype)))
            payload["transfer_fraction"] = _box_payload(space, value)
        elif kind == "cross_field_upper_bound" and isinstance(parameters, Mapping):
            if constraint.get("id") != "payload_coupling:maximum_cooling_rate_K_s":
                continue
            current = float(parameters["current_temperature_K"])
            maximum_rate = float(parameters["maximum_cooling_rate_K_s"])
            target = _scalar(payload["target_temperature_K"])
            duration = _scalar(payload["duration_s"])
            required_duration = max((current - target) / maximum_rate, 0.0)
            if duration < required_duration:
                space = event_action_space["duration_s"]
                if not isinstance(space, gym.spaces.Box):
                    raise TypeError("duration_s must use a Box action component")
                payload["duration_s"] = _box_payload_at_least(space, required_duration)
        elif kind == "absolute_value_upper_bound" and isinstance(parameters, Mapping):
            field_names = constraint.get("fields")
            if not isinstance(field_names, list) or len(field_names) != 1:
                continue
            field = str(field_names[0])
            limit = float(parameters["default_voltage_window_V"])
            space = event_action_space[field]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"{field} must use a Box action component")
            payload[field] = _box_payload_within_abs(space, _scalar(payload[field]), limit)


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "LATENT_ADAPTER_VERSION",
    "POLICY_DISTRIBUTION_SCHEMA_VERSION",
    "conditional_hybrid_action_contract",
    "decode_conditional_hybrid_action",
    "policy_distribution_contract",
]
