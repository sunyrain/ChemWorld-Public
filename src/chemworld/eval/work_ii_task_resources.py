"""Portable task-specific resource-card contracts for Work II."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from chemworld.campaign_resources import (
    DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS,
    PROTECTED_CLOSEOUT_POLICY,
)
from chemworld.eval.provenance import canonical_json_sha256

RESOURCE_CALIBRATION_ROUNDS = (8, 10, 12)
RESOURCE_CALIBRATION_LOCI = {8: "A_E", 10: "A_P", 12: "A_S"}
RESOURCE_CALIBRATION_CHECKPOINTS = {
    8: (0, 2, 4, 6, 8),
    10: (0, 2, 4, 7, 10),
    12: (0, 3, 6, 9, 12),
}
RESOURCE_CALIBRATION_CAP_FIELDS = (
    "operation_attempt_limit",
    "protected_closeout_operation_reserve",
    "maximum_exact_repeats",
    "process_time_limit_s",
    "protected_closeout_reserve_s",
    "input_token_limit",
    "uncached_input_token_limit",
    "output_token_limit",
    "provider_wall_time_limit_s",
    "currency_ceiling_usd",
)
TASK_RESOURCE_CALIBRATED_CAP_FIELDS = tuple(
    field
    for field in RESOURCE_CALIBRATION_CAP_FIELDS
    if field != "maximum_exact_repeats"
)
OPTIONAL_AGENT_INVALID_RECOVERY_CAP_FIELDS = (
    "max_recovered_mcp_tool_failures",
    "max_consecutive_mcp_tool_failures",
)


def build_task_resource_formula_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind task resource inputs while excluding the caps calibration measures."""

    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    process_policy = campaign.get("process_time_policy")
    process_policy = process_policy if isinstance(process_policy, Mapping) else {}
    closeout_policy = campaign.get("closeout_policy")
    closeout_policy = closeout_policy if isinstance(closeout_policy, Mapping) else {}
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    payload = {
        "task_id": config.get("task_id"),
        "complete_experiments": campaign.get("complete_experiments"),
        "checkpoint_complete_experiments": list(
            campaign.get("checkpoint_complete_experiments", [])
        ),
        "complete_experiment_limit": resources.get("complete_experiment_limit"),
        "method_checkpoint_complete_experiments": list(
            resources.get("checkpoint_complete_experiments", [])
        ),
        "minimum_unique_recipes": qualification.get("minimum_unique_recipes"),
        "maximum_participant_selected_exact_repeats": qualification.get(
            "maximum_exact_repeats"
        ),
        "campaign_design_limits": {
            key: copy.deepcopy(campaign.get(key))
            for key in (
                "vessel_start_limit",
                "final_assay_limit",
                "nonfinal_instrument_use_limit",
                "stock_limits",
                "per_instrument_limits",
                "implicit_operation_time_s",
                "operation_repeat_limits",
            )
        },
        "method_design_limits": {
            key: copy.deepcopy(resources.get(key))
            for key in (
                "model_call_limit",
                "training_environment_step_limit",
            )
        },
        "provider_method_contract": {
            key: copy.deepcopy(provider.get(key))
            for key in (
                "id",
                "wire_api",
                "model",
                "reasoning_effort",
                "request_timeout_s",
                "finalization_timeout_s",
            )
        },
        "process_time_formula": {
            key: copy.deepcopy(value)
            for key, value in process_policy.items()
            if key not in {"protected_reserve_s", "resource_status"}
        },
        "closeout_formula": {
            key: copy.deepcopy(value)
            for key, value in closeout_policy.items()
            if key
            not in {
                "final_assay_path_total_operation_reserve",
                "discard_path_total_operation_reserve",
                "policy",
                "resource_status",
            }
        }
        | {
            "allowed_operation_classes": sorted(
                closeout_policy.get(
                    "allowed_operation_classes",
                    DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS,
                )
            )
        },
    }
    return {
        "schema_version": "chemworld-work-ii-task-resource-formula-binding-0.1",
        "formula": payload,
        "canonical_json_sha256": canonical_json_sha256(payload),
    }


def validate_task_resource_card(
    card: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate one portable task-specific W2-26 card."""

    errors: list[str] = []
    identity = card.get("card_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    if expected_identity is not None:
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                errors.append(f"resource card identity differs at {field}")
    rounds = identity.get("rounds")
    locus = identity.get("locus")
    task_id = identity.get("task_id")
    world_seed = identity.get("world_seed")
    if (
        rounds not in RESOURCE_CALIBRATION_ROUNDS
        or locus != RESOURCE_CALIBRATION_LOCI.get(rounds)
        or not isinstance(task_id, str)
        or not task_id
        or isinstance(world_seed, bool)
        or not isinstance(world_seed, int)
    ):
        errors.append("resource card has an invalid task identity")
    campaign_binding = identity.get("calibration_campaign_binding")
    campaign_binding = (
        campaign_binding if isinstance(campaign_binding, Mapping) else {}
    )
    if (
        not isinstance(campaign_binding.get("path"), str)
        or not isinstance(campaign_binding.get("sha256"), str)
        or len(str(campaign_binding.get("sha256", ""))) != 64
    ):
        errors.append("resource card lacks its calibration campaign binding")
    formula = identity.get("resource_formula_binding")
    formula = formula if isinstance(formula, Mapping) else {}
    formula_payload = formula.get("formula")
    if (
        formula.get("schema_version")
        != "chemworld-work-ii-task-resource-formula-binding-0.1"
        or not isinstance(formula_payload, Mapping)
        or formula.get("canonical_json_sha256")
        != canonical_json_sha256(formula_payload)
        or formula_payload.get("task_id") != task_id
        or formula_payload.get("complete_experiments") != rounds
        or isinstance(
            formula_payload.get("maximum_participant_selected_exact_repeats"),
            bool,
        )
        or not isinstance(
            formula_payload.get("maximum_participant_selected_exact_repeats"), int
        )
        or formula_payload.get("maximum_participant_selected_exact_repeats", -1) < 0
        or tuple(formula_payload.get("checkpoint_complete_experiments", []))
        != RESOURCE_CALIBRATION_CHECKPOINTS.get(rounds)
    ):
        errors.append("resource card formula binding is invalid")
    caps = card.get("proposed_hard_caps")
    caps = caps if isinstance(caps, Mapping) else {}
    currency_accounting = card.get("currency_accounting")
    currency_accounting = (
        currency_accounting if isinstance(currency_accounting, Mapping) else {}
    )
    currency_unavailable = (
        currency_accounting.get("status") == "unavailable_provider_pricing"
        and currency_accounting.get("formal_currency_contract_required") is True
        and caps.get("currency_ceiling_usd") is None
    )
    for field in TASK_RESOURCE_CALIBRATED_CAP_FIELDS:
        value = caps.get(field)
        if field == "currency_ceiling_usd" and currency_unavailable:
            continue
        if not _is_nonnegative_number(value) or value == 0:
            errors.append(f"resource card lacks positive cap {field}")
    if "maximum_exact_repeats" in caps:
        errors.append("resource card must not redefine the exact-repeat design invariant")
    recovery_caps_present = [
        field in caps for field in OPTIONAL_AGENT_INVALID_RECOVERY_CAP_FIELDS
    ]
    if any(recovery_caps_present) and not all(recovery_caps_present):
        errors.append("resource card has an incomplete agent-invalid recovery budget")
    elif all(recovery_caps_present):
        for field in OPTIONAL_AGENT_INVALID_RECOVERY_CAP_FIELDS:
            value = caps.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"resource card lacks positive integer cap {field}")
    input_limit = caps.get("input_token_limit")
    uncached_limit = caps.get("uncached_input_token_limit")
    if (
        _is_nonnegative_number(input_limit)
        and _is_nonnegative_number(uncached_limit)
        and float(uncached_limit) > float(input_limit)
    ):
        errors.append("resource card uncached input cap exceeds total input cap")
    card_sha = card.get("card_sha256")
    if card_sha is not None and card_sha != _self_hash(card, "card_sha256"):
        errors.append("resource card self-hash mismatch")
    return errors


def resolve_task_resource_card(
    summary: Mapping[str, Any],
    *,
    rounds: int,
    locus: str,
    task_id: str,
    formal_source_config: Mapping[str, Any],
    formal_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve exactly one task card and prove formula portability."""

    if summary.get("status") != "passed" or summary.get("calibration_passed") is not True:
        raise ValueError("W2-26 resource calibration did not pass")
    proposals = summary.get("resource_card_proposals")
    proposals = proposals if isinstance(proposals, list) else []
    matches = []
    for raw in proposals:
        if not isinstance(raw, Mapping):
            continue
        identity = raw.get("card_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        if (
            identity.get("rounds"),
            identity.get("locus"),
            identity.get("task_id"),
        ) == (rounds, locus, task_id):
            matches.append(raw)
    if len(matches) != 1:
        raise ValueError(
            "W2-26 must contain exactly one task resource card for "
            f"{locus}/{task_id}/{rounds}"
        )
    card = copy.deepcopy(dict(matches[0]))
    errors = validate_task_resource_card(
        card,
        expected_identity={"rounds": rounds, "locus": locus, "task_id": task_id},
    )
    source_formula = build_task_resource_formula_binding(formal_source_config)
    identity = card.get("card_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    if identity.get("resource_formula_binding") != source_formula:
        errors.append("formal source resource formula differs from its calibrated task")
    if formal_source_binding is not None:
        source_digest = canonical_json_sha256(formal_source_config)
        file_digest = formal_source_binding.get("sha256")
        if (
            not isinstance(formal_source_binding.get("path"), str)
            or not isinstance(file_digest, str)
            or len(file_digest) != 64
            or formal_source_binding.get("config_canonical_json_sha256")
            != source_digest
        ):
            errors.append("formal source binding is malformed or differs from its config")
    if errors:
        raise ValueError("task resource card failed: " + "; ".join(errors))
    return card


def materialize_task_resource_caps(
    config: Mapping[str, Any], card: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a config whose executable caps come from its exact task card."""

    errors = validate_task_resource_card(card)
    identity = card.get("card_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    source_formula = build_task_resource_formula_binding(config)
    if identity.get("resource_formula_binding") != source_formula:
        errors.append("execution config resource formula differs from task card")
    if errors:
        raise ValueError("cannot materialize task resource caps: " + "; ".join(errors))
    result = copy.deepcopy(dict(config))
    caps = card["proposed_hard_caps"]
    campaign = result["campaign"]
    resources = result["method_resources"]
    qualification = result.setdefault("qualification", {})
    provider = result.setdefault("provider", {})
    process_policy = campaign["process_time_policy"]
    closeout_policy = campaign["closeout_policy"]
    campaign["operation_attempt_limit"] = int(caps["operation_attempt_limit"])
    campaign["process_time_limit_s"] = float(caps["process_time_limit_s"])
    process_policy["protected_reserve_s"] = float(
        caps["protected_closeout_reserve_s"]
    )
    process_policy["resource_status"] = "calibrated_w2_26_task_specific"
    closeout_policy["final_assay_path_total_operation_reserve"] = int(
        caps["protected_closeout_operation_reserve"]
    )
    closeout_policy["policy"] = PROTECTED_CLOSEOUT_POLICY
    closeout_policy["allowed_operation_classes"] = sorted(
        DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS
    )
    closeout_policy["resource_status"] = "calibrated_w2_26_task_specific"
    resources.update(
        {
            "operation_limit": int(caps["operation_attempt_limit"]),
            "input_token_limit": int(caps["input_token_limit"]),
            "uncached_input_token_limit": int(caps["uncached_input_token_limit"]),
            "output_token_limit": int(caps["output_token_limit"]),
            "wall_time_limit_s": float(caps["provider_wall_time_limit_s"]),
        }
    )
    resources.pop("resource_status", None)
    qualification["resource_calibration_status"] = "passed_w2_26_task_specific"
    provider["session_wall_time_limit_s"] = float(
        caps["provider_wall_time_limit_s"]
    )
    if all(field in caps for field in OPTIONAL_AGENT_INVALID_RECOVERY_CAP_FIELDS):
        provider["max_recovered_mcp_tool_failures"] = int(
            caps["max_recovered_mcp_tool_failures"]
        )
        provider["max_consecutive_mcp_tool_failures"] = int(
            caps["max_consecutive_mcp_tool_failures"]
        )
    if caps.get("currency_ceiling_usd") is not None:
        result["calibrated_currency_ceiling_usd"] = float(
            caps["currency_ceiling_usd"]
        )
    else:
        result.pop("calibrated_currency_ceiling_usd", None)
    result["resource_calibration_card_binding"] = {
        "card_identity": copy.deepcopy(identity),
        "card_sha256": card.get("card_sha256") or canonical_json_sha256(card),
        "source_resource_formula_binding": source_formula,
    }
    return result


def _is_nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


__all__ = [
    "TASK_RESOURCE_CALIBRATED_CAP_FIELDS",
    "build_task_resource_formula_binding",
    "materialize_task_resource_caps",
    "resolve_task_resource_card",
    "validate_task_resource_card",
]
