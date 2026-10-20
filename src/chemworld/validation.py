"""Public validation helpers for actions and recipes."""

from __future__ import annotations

from typing import Any

from chemworld.action_codec import ActionCodec
from chemworld.schemas import SchemaValidationResult, validate_action_schema
from chemworld.world.recipes import validate_recipe


def validate_action(
    action: dict[str, Any],
    task_info: dict[str, Any],
    state_summary: dict[str, Any] | None = None,
    *,
    modules=None,
) -> SchemaValidationResult:
    """Validate an action against schema and task policy.

    This helper intentionally avoids hidden state access. Use
    ``validate_event_action(action, env)`` when current-state preconditions must
    be checked.
    """

    del state_summary
    codec = ActionCodec(modules=modules)
    schema_result = validate_action_schema(
        action,
        operation_types=codec.operation_types,
        instruments=codec.instruments,
        modules=modules,
    )
    if not schema_result.valid:
        return schema_result
    canonical = codec.canonicalize(action)
    errors: list[str] = []
    allowed_operations = set(task_info.get("allowed_operations", []))
    if allowed_operations and canonical["operation"] not in allowed_operations:
        errors.append(f"operation not allowed by task: {canonical['operation']}")
    if canonical["operation"] == "measure":
        allowed_instruments = set(task_info.get("allowed_instruments", []))
        instrument = str(canonical.get("instrument", "hplc"))
        if allowed_instruments and instrument not in allowed_instruments:
            errors.append(f"instrument not allowed by task: {instrument}")
    return SchemaValidationResult(not errors, tuple(errors))


__all__ = ["validate_action", "validate_recipe"]
