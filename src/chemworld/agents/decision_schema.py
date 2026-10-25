"""Strict generation-time JSON Schema for live G2 operation decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

PUBLIC_ACTION_SCHEMA_VERSION = "chemworld-public-action-affordance-0.2"

_DECISION_REQUIRED_FIELDS = (
    "action",
    "expected_effect",
    "diagnostic_target",
    "expected_information_gain",
    "belief_update_rule",
    "uncertainty",
    "request_historical_spectrum_id",
)


class DecisionSchemaError(ValueError):
    """The public action affordance cannot produce a strict decision schema."""


def build_decision_output_schema(tool_json: Mapping[str, Any]) -> dict[str, Any]:
    """Build a closed response schema from the current public tool view."""

    if not isinstance(tool_json, Mapping):
        raise DecisionSchemaError("tool_json must be an object")
    available_actions = tool_json.get("available_actions")
    return _build_from_available_actions(available_actions)


def _build_from_available_actions(
    available_actions: object,
) -> dict[str, Any]:
    """Build a closed response schema from current public action affordances.

    Only entries explicitly marked ``valid=True`` become action variants. Public
    affordances must completely specify every required parameter with either a
    finite enum or finite numeric bounds; ambiguous fields fail closed.
    """

    if isinstance(available_actions, str | bytes) or not isinstance(available_actions, Sequence):
        raise DecisionSchemaError("tool_json.available_actions must be a sequence")
    if not available_actions:
        raise DecisionSchemaError("tool_json.available_actions must contain at least one entry")

    variants: list[dict[str, Any]] = []
    operations: set[str] = set()
    for index, action in enumerate(available_actions):
        path = f"tool_json.available_actions[{index}]"
        if not isinstance(action, Mapping):
            raise DecisionSchemaError(f"{path} must be an object")
        valid = action.get("valid")
        if not isinstance(valid, bool):
            raise DecisionSchemaError(f"{path}.valid must be boolean")
        if not valid:
            continue
        operation = _required_nonempty_string(action.get("operation"), f"{path}.operation")
        if operation in operations:
            raise DecisionSchemaError(f"duplicate legal operation: {operation}")
        operations.add(operation)
        variants.append(_action_variant(action, operation=operation, path=path))

    if not variants:
        raise DecisionSchemaError("tool_json.available_actions contains no explicitly valid action")

    schema = {
        "type": "object",
        "properties": {
            "action": {"anyOf": variants},
            "expected_effect": {"type": "string", "minLength": 1, "maxLength": 512},
            "diagnostic_target": {"type": "string", "minLength": 1, "maxLength": 512},
            "expected_information_gain": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "belief_update_rule": {
                "type": "object",
                "properties": {
                    "if_supported": {"type": "string", "minLength": 1, "maxLength": 512},
                    "if_not_supported": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                },
                "required": ["if_supported", "if_not_supported"],
                "additionalProperties": False,
            },
            "uncertainty": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "request_historical_spectrum_id": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 256},
                    {"type": "null"},
                ]
            },
        },
        "required": list(_DECISION_REQUIRED_FIELDS),
        "additionalProperties": False,
    }
    # Keep provider handoff deterministic and reject non-JSON numeric sentinels.
    json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return schema


def _action_variant(
    action: Mapping[str, Any],
    *,
    operation: str,
    path: str,
) -> dict[str, Any]:
    raw_schema = action.get("schema")
    if not isinstance(raw_schema, Mapping):
        raise DecisionSchemaError(f"{path}.schema must be an object")
    schema_path = f"{path}.schema"
    if raw_schema.get("schema_version") != PUBLIC_ACTION_SCHEMA_VERSION:
        raise DecisionSchemaError(
            f"{schema_path}.schema_version must be {PUBLIC_ACTION_SCHEMA_VERSION!r}"
        )
    if raw_schema.get("valid_operation_type") is not True:
        raise DecisionSchemaError(f"{schema_path}.valid_operation_type must be true")
    if raw_schema.get("task_allowed") is not True:
        raise DecisionSchemaError(f"{schema_path}.task_allowed must be true")
    if raw_schema.get("operation") != operation:
        raise DecisionSchemaError(f"{schema_path}.operation must match {path}.operation")

    required_fields = _required_field_names(
        raw_schema.get("required_fields"),
        path=f"{schema_path}.required_fields",
    )
    raw_fields = raw_schema.get("fields")
    if not isinstance(raw_fields, list):
        raise DecisionSchemaError(f"{schema_path}.fields must be a list")

    fields_by_name: dict[str, Mapping[str, Any]] = {}
    for index, field in enumerate(raw_fields):
        field_path = f"{schema_path}.fields[{index}]"
        if not isinstance(field, Mapping):
            raise DecisionSchemaError(f"{field_path} must be an object")
        name = _required_nonempty_string(field.get("field"), f"{field_path}.field")
        if name == "operation":
            raise DecisionSchemaError(f"{field_path}.field cannot redefine operation")
        if name in fields_by_name:
            raise DecisionSchemaError(f"duplicate field in {schema_path}.fields: {name}")
        if field.get("required") is not True:
            raise DecisionSchemaError(f"{field_path}.required must be true")
        fields_by_name[name] = field

    if set(fields_by_name) != set(required_fields):
        missing = sorted(set(required_fields) - set(fields_by_name))
        extra = sorted(set(fields_by_name) - set(required_fields))
        raise DecisionSchemaError(
            f"{schema_path} fields must exactly match required_fields; "
            f"missing={missing}, extra={extra}"
        )

    properties: dict[str, Any] = {"operation": {"type": "string", "enum": [operation]}}
    for name in required_fields:
        properties[name] = _parameter_schema(
            fields_by_name[name],
            path=f"{schema_path}.fields[{name!r}]",
        )
    return {
        "type": "object",
        "properties": properties,
        "required": ["operation", *required_fields],
        "additionalProperties": False,
    }


def _required_field_names(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list):
        raise DecisionSchemaError(f"{path} must be a list")
    names: list[str] = []
    for index, item in enumerate(value):
        name = _required_nonempty_string(item, f"{path}[{index}]")
        if name == "operation":
            raise DecisionSchemaError(f"{path}[{index}] cannot be operation")
        if name in names:
            raise DecisionSchemaError(f"{path} contains duplicate field {name!r}")
        names.append(name)
    return names


def _parameter_schema(field: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    choices_present = "choices" in field
    bounds_present = "bounds" in field
    if not choices_present and not bounds_present:
        if field.get("unit") == "text":
            return {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            }
        raise DecisionSchemaError(f"{path} must provide non-empty choices or finite numeric bounds")

    parameter: dict[str, Any] = {}
    if choices_present:
        choices = field.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DecisionSchemaError(f"{path}.choices must be a non-empty list")
        parameter["type"] = _enum_type(choices, path=f"{path}.choices")
        parameter["enum"] = list(choices)

    if bounds_present:
        bounds = field.get("bounds")
        if not isinstance(bounds, Mapping):
            raise DecisionSchemaError(f"{path}.bounds must be an object")
        low = _finite_number(bounds.get("low"), path=f"{path}.bounds.low")
        high = _finite_number(bounds.get("high"), path=f"{path}.bounds.high")
        lower_inclusive = field.get("lower_bound_inclusive")
        upper_inclusive = field.get("upper_bound_inclusive")
        if not isinstance(lower_inclusive, bool):
            raise DecisionSchemaError(f"{path}.lower_bound_inclusive must be boolean")
        if not isinstance(upper_inclusive, bool):
            raise DecisionSchemaError(f"{path}.upper_bound_inclusive must be boolean")
        if low > high or (low == high and (not lower_inclusive or not upper_inclusive)):
            raise DecisionSchemaError(f"{path}.bounds describe an empty interval")
        existing_type = parameter.get("type")
        if existing_type not in {None, "integer", "number"}:
            raise DecisionSchemaError(f"{path} cannot combine non-numeric choices and bounds")
        parameter["type"] = "number" if existing_type is None else existing_type
        parameter["minimum" if lower_inclusive else "exclusiveMinimum"] = low
        parameter["maximum" if upper_inclusive else "exclusiveMaximum"] = high
        if choices_present:
            _validate_enum_within_bounds(
                parameter["enum"],
                low=low,
                high=high,
                lower_inclusive=lower_inclusive,
                upper_inclusive=upper_inclusive,
                path=f"{path}.choices",
            )
    return parameter


def _enum_type(choices: list[Any], *, path: str) -> str:
    scalar_types: set[str] = set()
    semantic_values: set[tuple[str, Any]] = set()
    for index, choice in enumerate(choices):
        value_path = f"{path}[{index}]"
        scalar_type = _json_scalar_type(choice, path=value_path)
        scalar_types.add(scalar_type)
        semantic_type = "number" if scalar_type in {"integer", "number"} else scalar_type
        semantic_key = (semantic_type, choice)
        if semantic_key in semantic_values:
            raise DecisionSchemaError(f"{path} contains duplicate JSON values")
        semantic_values.add(semantic_key)
    if scalar_types <= {"integer"}:
        return "integer"
    if scalar_types <= {"integer", "number"}:
        return "number"
    if len(scalar_types) == 1:
        return next(iter(scalar_types))
    raise DecisionSchemaError(f"{path} must contain one compatible JSON scalar type")


def _json_scalar_type(value: Any, *, path: str) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisionSchemaError(f"{path} must be finite")
        return "number"
    if isinstance(value, str):
        if not value:
            raise DecisionSchemaError(f"{path} must not be an empty string")
        return "string"
    raise DecisionSchemaError(f"{path} must be a non-null JSON scalar")


def _finite_number(value: Any, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DecisionSchemaError(f"{path} must be numeric")
    if not math.isfinite(value):
        raise DecisionSchemaError(f"{path} must be finite")
    return value


def _validate_enum_within_bounds(
    choices: list[Any],
    *,
    low: int | float,
    high: int | float,
    lower_inclusive: bool,
    upper_inclusive: bool,
    path: str,
) -> None:
    for index, choice in enumerate(choices):
        if isinstance(choice, bool) or not isinstance(choice, int | float):
            raise DecisionSchemaError(f"{path}[{index}] must be numeric")
        lower_valid = choice >= low if lower_inclusive else choice > low
        upper_valid = choice <= high if upper_inclusive else choice < high
        if not lower_valid or not upper_valid:
            raise DecisionSchemaError(f"{path}[{index}] is outside declared bounds")


def _required_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionSchemaError(f"{path} must be a non-empty string")
    return value
