"""Fail-closed formal currency contracts and attempt-reservation ledgers for Work II."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_formal import FORMAL_ARMS, validate_formal_bindings

FORMAL_COST_CONTRACT_VERSION = "chemworld-work-ii-formal-cost-contract-0.1"
FORMAL_COST_LEDGER_VERSION = "chemworld-work-ii-formal-cost-ledger-0.1"
QUALIFICATION_COST_CONTRACT_VERSION = (
    "chemworld-work-ii-qualification-cost-contract-0.2"
)
QUALIFICATION_COST_LEDGER_VERSION = "chemworld-work-ii-qualification-cost-ledger-0.2"
METHOD_QUALIFICATION_LOCAL_MANIFEST_VERSION = (
    "chemworld-work-ii-method-qualification-local-manifest-0.1"
)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _finite_float(value: object, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or (positive and parsed <= 0):
        return None
    return parsed


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _cost_usd(
    *,
    input_tokens: int,
    uncached_input_tokens: int,
    output_tokens: int,
    cache_hit_input_usd_per_million: float,
    cache_miss_input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    if min(input_tokens, uncached_input_tokens, output_tokens) < 0:
        raise ValueError("token caps must be non-negative")
    if uncached_input_tokens > input_tokens:
        raise ValueError("uncached input cap cannot exceed cumulative input cap")
    cached_input_tokens = input_tokens - uncached_input_tokens
    cost = (
        cached_input_tokens * cache_hit_input_usd_per_million
        + uncached_input_tokens * cache_miss_input_usd_per_million
        + output_tokens * output_usd_per_million
    ) / 1_000_000.0
    return round(cost, 12)


def formal_cost_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _self_hash(contract, "formal_cost_contract_sha256")


def formal_cost_ledger_sha256(ledger: Mapping[str, Any]) -> str:
    return _self_hash(ledger, "formal_cost_ledger_sha256")


def qualification_cost_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _self_hash(contract, "qualification_cost_contract_sha256")


def qualification_cost_ledger_sha256(ledger: Mapping[str, Any]) -> str:
    return _self_hash(ledger, "qualification_cost_ledger_sha256")


def _base_preflight_sha256(manifest: Mapping[str, Any]) -> object:
    authorization = manifest.get("authorization_bindings")
    if isinstance(authorization, Mapping):
        return authorization.get("base_preflight_sha256")
    return manifest.get("preflight_sha256")


def _qualification_campaign(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str, str]:
    contract = manifest.get("method_qualification_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    task_id = str(contract.get("qualification_task_id", ""))
    task_bindings = manifest.get("task_bindings")
    task_bindings = task_bindings if isinstance(task_bindings, list) else []
    matches = [
        row
        for row in task_bindings
        if isinstance(row, Mapping) and row.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError("formal manifest lacks one qualification task binding")
    campaign = matches[0].get("campaign_config")
    if not isinstance(campaign, Mapping):
        raise ValueError("qualification task lacks its campaign binding")
    relative = campaign.get("path")
    digest = campaign.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError("qualification campaign binding is incomplete")
    path = (root / relative).resolve()
    if file_sha256(path) != digest:
        raise ValueError("qualification campaign binding is stale")
    return task_id, _load_object(path), relative, digest


def _qualification_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the local W2-27 identity, never a formal/C2 preflight identity."""

    if manifest.get("schema_version") != METHOD_QUALIFICATION_LOCAL_MANIFEST_VERSION:
        raise ValueError("qualification cost requires the W2-27 local manifest")
    digest = manifest.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("qualification local manifest lacks its content identity")
    expected = _self_hash(manifest, "manifest_sha256")
    if digest != expected:
        raise ValueError("qualification local manifest self-hash is stale")
    if manifest.get("status") != "passed" or manifest.get("errors") != []:
        raise ValueError("qualification local manifest has not passed")
    if (
        manifest.get("formal_result") is not False
        or manifest.get("formal_execution_authorized") is not False
    ):
        raise ValueError("qualification local manifest crossed the formal boundary")
    return digest


def build_qualification_cost_contract(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    qualification_currency_ceiling_usd: float | None,
    pricing_source: str | None,
    pricing_observed_at: str | None,
    cache_hit_input_usd_per_million: float | None,
    cache_miss_input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    unlimited_spend_authorized: bool = False,
    pricing_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build the pre-call reservation contract for all qualification attempts."""

    root = root.resolve()
    manifest_digest = _qualification_manifest_sha256(manifest)
    if not isinstance(pricing_source, str) or not pricing_source.strip():
        raise ValueError("qualification pricing source is required")
    if not isinstance(pricing_observed_at, str) or not pricing_observed_at.strip():
        raise ValueError("qualification pricing observation timestamp is required")
    if unlimited_spend_authorized:
        if qualification_currency_ceiling_usd is not None or any(
            rate is not None
            for rate in (
                cache_hit_input_usd_per_million,
                cache_miss_input_usd_per_million,
                output_usd_per_million,
            )
        ):
            raise ValueError(
                "unlimited qualification authorization cannot invent pricing or a ceiling"
            )
        if (
            not isinstance(pricing_unavailable_reason, str)
            or not pricing_unavailable_reason.strip()
        ):
            raise ValueError("unlimited qualification authorization requires a pricing reason")
        ceiling = None
        hit_rate = miss_rate = output_rate = None
    else:
        ceiling = _finite_float(qualification_currency_ceiling_usd, positive=True)
        rates = (
            _finite_float(cache_hit_input_usd_per_million),
            _finite_float(cache_miss_input_usd_per_million),
            _finite_float(output_usd_per_million),
        )
        if ceiling is None:
            raise ValueError("qualification currency ceiling must be finite and positive")
        if any(rate is None for rate in rates) or not any(
            float(rate or 0.0) > 0 for rate in rates
        ):
            raise ValueError(
                "qualification pricing rates must be finite, non-negative and non-zero"
            )
        if pricing_unavailable_reason is not None:
            raise ValueError(
                "priced qualification authorization cannot declare pricing unavailable"
            )
        hit_rate, miss_rate, output_rate = (
            float(rate) for rate in rates if rate is not None
        )
    task_id, config, relative, digest = _qualification_campaign(root, manifest)
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    input_cap = int(resources.get("input_token_limit", -1))
    uncached_cap = int(resources.get("uncached_input_token_limit", -1))
    output_cap = int(resources.get("output_token_limit", -1))
    per_attempt_cost = (
        None
        if unlimited_spend_authorized
        else _cost_usd(
            input_tokens=input_cap,
            uncached_input_tokens=uncached_cap,
            output_tokens=output_cap,
            cache_hit_input_usd_per_million=float(hit_rate),
            cache_miss_input_usd_per_million=float(miss_rate),
            output_usd_per_million=float(output_rate),
        )
    )
    method_contract = manifest.get("method_qualification_contract")
    method_contract = method_contract if isinstance(method_contract, Mapping) else {}
    initial_attempts = len(FORMAL_ARMS)
    per_arm_attempt_cap = 1 + int(
        method_contract.get("maximum_infrastructure_resume_attempts_per_cell", -1)
    )
    hard_attempts = initial_attempts * per_arm_attempt_cap
    if (
        initial_attempts != int(method_contract.get("qualification_cell_count", -1))
        or hard_attempts != int(method_contract.get("maximum_total_provider_attempts", -1))
        or per_arm_attempt_cap != 2
    ):
        raise ValueError("qualification provider-attempt contract drifted")

    def envelope(attempts: int) -> dict[str, Any]:
        return {
            "provider_attempt_count": attempts,
            "token_caps": {
                "input_tokens": input_cap * attempts,
                "uncached_input_tokens": uncached_cap * attempts,
                "output_tokens": output_cap * attempts,
            },
            "cost_cap_usd": (
                None
                if per_attempt_cost is None
                else round(per_attempt_cost * attempts, 12)
            ),
        }

    initial = envelope(initial_attempts)
    hard = envelope(hard_attempts)
    if ceiling is not None and ceiling < float(hard["cost_cap_usd"]):
        raise ValueError(
            "qualification currency ceiling is below the frozen all-attempt cost cap "
            f"({ceiling} < {hard['cost_cap_usd']})"
        )
    provider = manifest.get("provider_contract")
    provider = provider if isinstance(provider, Mapping) else {}
    contract: dict[str, Any] = {
        "schema_version": QUALIFICATION_COST_CONTRACT_VERSION,
        "currency": "USD",
        "provider_id": provider.get("id"),
        "model_id": provider.get("model"),
        "provider_contract_sha256": canonical_json_sha256(provider),
        "qualification_manifest_sha256": manifest_digest,
        "method_qualification_contract_sha256": manifest.get(
            "method_qualification_contract_sha256"
        ),
        "qualification_task_id": task_id,
        "campaign_config_path": relative,
        "campaign_config_sha256": digest,
        "prior_arms": list(FORMAL_ARMS),
        "per_arm_provider_attempt_hard_cap": per_arm_attempt_cap,
        "per_attempt_token_caps": {
            "input_tokens": input_cap,
            "uncached_input_tokens": uncached_cap,
            "output_tokens": output_cap,
        },
        "per_attempt_cost_cap_usd": per_attempt_cost,
        "pricing": {
            "source": pricing_source,
            "observed_at": pricing_observed_at,
            "unit": "usd_per_million_tokens",
            "cache_hit_input": hit_rate,
            "cache_miss_input": miss_rate,
            "output": output_rate,
            "pricing_available": not unlimited_spend_authorized,
            "pricing_unavailable_reason": pricing_unavailable_reason,
        },
        "initial_schedule": initial,
        "all_infrastructure_resumes": hard,
        "qualification_currency_ceiling_usd": ceiling,
        "unlimited_spend_authorized": unlimited_spend_authorized,
        "currency_headroom_over_all_attempts_usd": (
            None
            if ceiling is None
            else round(ceiling - float(hard["cost_cap_usd"]), 12)
        ),
        "runtime_enforcement": {
            "reserve_full_token_cost_before_each_provider_process_launch": True,
            "missing_infrastructure_only_resume": True,
            "reject_launch_if_reservations_exceed_ceiling": True,
            "unknown_or_missing_actual_billing_never_reduces_reservation": True,
            "token_and_attempt_caps_enforced_without_attributable_usd_pricing": True,
        },
    }
    contract["qualification_cost_contract_sha256"] = (
        qualification_cost_contract_sha256(contract)
    )
    return contract


def validate_qualification_cost_contract(
    root: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Rebuild and validate the qualification price/token envelope."""

    errors: list[str] = []
    if contract.get("schema_version") != QUALIFICATION_COST_CONTRACT_VERSION:
        errors.append("unexpected Work II qualification cost contract schema")
    if contract.get("qualification_cost_contract_sha256") != (
        qualification_cost_contract_sha256(contract)
    ):
        errors.append("Work II qualification cost contract self-hash mismatch")
    pricing = contract.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    try:
        unlimited = contract.get("unlimited_spend_authorized") is True
        ceiling = (
            None
            if unlimited
            else _finite_float(
                contract.get("qualification_currency_ceiling_usd"), positive=True
            )
        )
        hit_rate = None if unlimited else _finite_float(pricing.get("cache_hit_input"))
        miss_rate = None if unlimited else _finite_float(pricing.get("cache_miss_input"))
        output_rate = None if unlimited else _finite_float(pricing.get("output"))
        if not unlimited and (
            ceiling is None or hit_rate is None or miss_rate is None or output_rate is None
        ):
            raise ValueError("qualification pricing values are missing or invalid")
        rebuilt = build_qualification_cost_contract(
            root,
            manifest,
            qualification_currency_ceiling_usd=ceiling,
            pricing_source=str(pricing.get("source", "")),
            pricing_observed_at=str(pricing.get("observed_at", "")),
            cache_hit_input_usd_per_million=hit_rate,
            cache_miss_input_usd_per_million=miss_rate,
            output_usd_per_million=output_rate,
            unlimited_spend_authorized=unlimited,
            pricing_unavailable_reason=(
                str(pricing.get("pricing_unavailable_reason", ""))
                if unlimited
                else None
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"Work II qualification cost contract cannot be rebuilt: {error}")
    else:
        if dict(contract) != rebuilt:
            errors.append("Work II qualification cost contract differs from deterministic rebuild")
    return errors


def build_qualification_cost_ledger(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    provider_attempt_counts_by_arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Reserve the full token envelope for each qualification process launch."""

    allowed = set(FORMAL_ARMS)
    cap = int(contract.get("per_arm_provider_attempt_hard_cap", -1))
    unlimited = contract.get("unlimited_spend_authorized") is True
    cost = (
        None
        if unlimited
        else _finite_float(contract.get("per_attempt_cost_cap_usd"))
    )
    if cap != 2 or (not unlimited and cost is None):
        raise ValueError("qualification cost ledger lacks its per-attempt contract")
    counts: dict[str, int] = {}
    for arm, raw_count in provider_attempt_counts_by_arm.items():
        if arm not in allowed:
            raise ValueError(f"qualification cost ledger contains an unknown arm: {arm}")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError(f"qualification cost ledger has a non-integer count: {arm}")
        if raw_count < 0 or raw_count > cap:
            raise ValueError(f"qualification provider-attempt cap exceeded: {arm}")
        counts[arm] = raw_count
    reserved = None if cost is None else round(sum(counts.values()) * cost, 12)
    ceiling_value = (
        None
        if unlimited
        else _finite_float(
            contract.get("qualification_currency_ceiling_usd"), positive=True
        )
    )
    if not unlimited and ceiling_value is None:
        raise ValueError("qualification cost ledger lacks a positive ceiling")
    ledger: dict[str, Any] = {
        "schema_version": QUALIFICATION_COST_LEDGER_VERSION,
        "qualification_cost_contract_sha256": contract.get(
            "qualification_cost_contract_sha256"
        ),
        "qualification_manifest_sha256": _qualification_manifest_sha256(manifest),
        "provider_attempt_count": sum(counts.values()),
        "provider_attempt_counts_by_arm": {
            arm: counts.get(arm, 0) for arm in FORMAL_ARMS
        },
        "reserved_cost_usd": reserved,
        "qualification_currency_ceiling_usd": ceiling_value,
        "unlimited_spend_authorized": unlimited,
        "remaining_unreserved_usd": (
            None
            if unlimited
            else round(float(ceiling_value) - float(reserved), 12)
        ),
        "within_ceiling": None if unlimited else reserved <= ceiling_value,
        "within_authorized_spend": unlimited or reserved <= ceiling_value,
        "reservation_semantics": "full_frozen_token_cap_per_launched_provider_process",
    }
    ledger["qualification_cost_ledger_sha256"] = qualification_cost_ledger_sha256(
        ledger
    )
    return ledger


def validate_qualification_cost_ledger(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> list[str]:
    """Validate a persisted qualification reservation ledger by deterministic rebuild."""

    errors: list[str] = []
    if ledger.get("schema_version") != QUALIFICATION_COST_LEDGER_VERSION:
        errors.append("unexpected Work II qualification cost ledger schema")
    if ledger.get("qualification_cost_ledger_sha256") != (
        qualification_cost_ledger_sha256(ledger)
    ):
        errors.append("Work II qualification cost ledger self-hash mismatch")
    counts = ledger.get("provider_attempt_counts_by_arm")
    if not isinstance(counts, Mapping):
        errors.append("Work II qualification cost ledger lacks attempt counts")
        return errors
    try:
        rebuilt = build_qualification_cost_ledger(manifest, contract, counts)
    except (TypeError, ValueError) as error:
        errors.append(f"Work II qualification cost ledger cannot be rebuilt: {error}")
    else:
        if dict(ledger) != rebuilt:
            errors.append("Work II qualification cost ledger differs from deterministic rebuild")
    return errors


def build_formal_cost_contract(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    formal_currency_ceiling_usd: float,
    pricing_source: str,
    pricing_observed_at: str,
    cache_hit_input_usd_per_million: float,
    cache_miss_input_usd_per_million: float,
    output_usd_per_million: float,
) -> dict[str, Any]:
    """Build a budget that covers every frozen provider-attempt token envelope."""

    root = root.resolve()
    binding_errors = validate_formal_bindings(root, manifest)
    if binding_errors:
        raise ValueError("formal bindings are invalid: " + "; ".join(binding_errors))
    ceiling = _finite_float(formal_currency_ceiling_usd, positive=True)
    rates = (
        _finite_float(cache_hit_input_usd_per_million),
        _finite_float(cache_miss_input_usd_per_million),
        _finite_float(output_usd_per_million),
    )
    if ceiling is None:
        raise ValueError("formal currency ceiling must be finite and positive")
    if any(rate is None for rate in rates) or not any(float(rate or 0.0) > 0 for rate in rates):
        raise ValueError("formal pricing rates must be finite, non-negative and non-zero")
    if not isinstance(pricing_source, str) or not pricing_source.strip():
        raise ValueError("formal pricing source is required")
    if not isinstance(pricing_observed_at, str) or not pricing_observed_at.strip():
        raise ValueError("formal pricing observation timestamp is required")
    hit_rate, miss_rate, output_rate = (float(rate) for rate in rates if rate is not None)

    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("formal manifest lacks cells")
    task_bindings = manifest.get("task_bindings")
    if not isinstance(task_bindings, list):
        raise ValueError("formal manifest lacks task bindings")
    binding_by_task = {
        str(row.get("task_binding_key")): row
        for row in task_bindings
        if isinstance(row, Mapping)
    }
    task_ids = sorted(
        {
            str(cell.get("task_binding_key"))
            for cell in cells
            if isinstance(cell, Mapping)
        }
    )
    per_task: list[dict[str, Any]] = []
    accepted_totals = {"input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0}
    hard_totals = {"input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0}
    initial_cost = 0.0
    hard_cost = 0.0
    for task_id in task_ids:
        binding = binding_by_task.get(task_id)
        binding_record: Mapping[str, Any] = binding if isinstance(binding, Mapping) else {}
        campaign = binding_record.get("campaign_config")
        if not isinstance(campaign, Mapping):
            raise ValueError(f"formal task lacks campaign binding: {task_id}")
        relative = campaign.get("path")
        digest = campaign.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"formal task campaign binding is incomplete: {task_id}")
        config_path = (root / relative).resolve()
        if file_sha256(config_path) != digest:
            raise ValueError(f"formal task campaign binding is stale: {task_id}")
        config = _load_object(config_path)
        resources = config.get("method_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        input_cap = int(resources.get("input_token_limit", -1))
        uncached_cap = int(resources.get("uncached_input_token_limit", -1))
        output_cap = int(resources.get("output_token_limit", -1))
        per_attempt_cost = _cost_usd(
            input_tokens=input_cap,
            uncached_input_tokens=uncached_cap,
            output_tokens=output_cap,
            cache_hit_input_usd_per_million=hit_rate,
            cache_miss_input_usd_per_million=miss_rate,
            output_usd_per_million=output_rate,
        )
        task_cells = [
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("task_binding_key") == task_id
        ]
        initial_attempts = len(task_cells)
        hard_attempts = sum(int(cell.get("provider_attempt_limit", -1)) for cell in task_cells)
        if initial_attempts <= 0 or hard_attempts < initial_attempts:
            raise ValueError(f"formal provider-attempt denominator is invalid: {task_id}")
        for totals, multiplier in (
            (accepted_totals, initial_attempts),
            (hard_totals, hard_attempts),
        ):
            totals["input_tokens"] += input_cap * multiplier
            totals["uncached_input_tokens"] += uncached_cap * multiplier
            totals["output_tokens"] += output_cap * multiplier
        initial_cost += per_attempt_cost * initial_attempts
        hard_cost += per_attempt_cost * hard_attempts
        per_task.append(
            {
                "task_binding_key": task_id,
                "c2_locus": binding_record.get("c2_locus"),
                "task_id": binding_record.get("task_id"),
                "campaign_config_path": relative,
                "campaign_config_sha256": digest,
                "participant_cell_count": initial_attempts,
                "provider_attempt_hard_cap": hard_attempts,
                "per_attempt_token_caps": {
                    "input_tokens": input_cap,
                    "uncached_input_tokens": uncached_cap,
                    "output_tokens": output_cap,
                },
                "per_attempt_cost_cap_usd": per_attempt_cost,
            }
        )

    initial_cost = round(initial_cost, 12)
    hard_cost = round(hard_cost, 12)
    if ceiling < hard_cost:
        raise ValueError(
            "formal currency ceiling is below the frozen all-attempt cost cap "
            f"({ceiling} < {hard_cost})"
        )
    provider = manifest.get("provider_contract")
    provider = provider if isinstance(provider, Mapping) else {}
    contract: dict[str, Any] = {
        "schema_version": FORMAL_COST_CONTRACT_VERSION,
        "currency": "USD",
        "provider_id": provider.get("id"),
        "model_id": provider.get("model"),
        "provider_contract_sha256": canonical_json_sha256(provider),
        "formal_preflight_sha256": _base_preflight_sha256(manifest),
        "pricing": {
            "source": pricing_source,
            "observed_at": pricing_observed_at,
            "unit": "usd_per_million_tokens",
            "cache_hit_input": hit_rate,
            "cache_miss_input": miss_rate,
            "output": output_rate,
        },
        "per_task_attempt_caps": per_task,
        "initial_schedule": {
            "provider_attempt_count": len(cells),
            "token_caps": accepted_totals,
            "cost_cap_usd": initial_cost,
        },
        "all_infrastructure_resumes": {
            "provider_attempt_count": sum(
                int(cell.get("provider_attempt_limit", -1))
                for cell in cells
                if isinstance(cell, Mapping)
            ),
            "token_caps": hard_totals,
            "cost_cap_usd": hard_cost,
        },
        "formal_currency_ceiling_usd": ceiling,
        "currency_headroom_over_all_attempts_usd": round(ceiling - hard_cost, 12),
        "runtime_enforcement": {
            "reserve_full_token_cost_before_each_provider_process_launch": True,
            "reject_launch_if_reservations_exceed_ceiling": True,
            "unknown_or_missing_actual_billing_never_reduces_reservation": True,
            "persisted_scientific_trajectory_never_replaced": True,
        },
    }
    contract["formal_cost_contract_sha256"] = formal_cost_contract_sha256(contract)
    return contract


def validate_formal_cost_contract(
    root: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Validate pricing, token denominators and a ceiling covering all attempts."""

    errors: list[str] = []
    if contract.get("schema_version") != FORMAL_COST_CONTRACT_VERSION:
        errors.append("unexpected Work II formal cost contract schema")
    if contract.get("formal_cost_contract_sha256") != formal_cost_contract_sha256(contract):
        errors.append("Work II formal cost contract self-hash mismatch")
    pricing = contract.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    try:
        ceiling = _finite_float(contract.get("formal_currency_ceiling_usd"), positive=True)
        hit_rate = _finite_float(pricing.get("cache_hit_input"))
        miss_rate = _finite_float(pricing.get("cache_miss_input"))
        output_rate = _finite_float(pricing.get("output"))
        if ceiling is None or hit_rate is None or miss_rate is None or output_rate is None:
            raise ValueError("formal pricing values are missing or invalid")
        rebuilt = build_formal_cost_contract(
            root,
            manifest,
            formal_currency_ceiling_usd=ceiling,
            pricing_source=str(pricing.get("source", "")),
            pricing_observed_at=str(pricing.get("observed_at", "")),
            cache_hit_input_usd_per_million=hit_rate,
            cache_miss_input_usd_per_million=miss_rate,
            output_usd_per_million=output_rate,
        )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"Work II formal cost contract cannot be rebuilt: {error}")
    else:
        if dict(contract) != rebuilt:
            errors.append("Work II formal cost contract differs from deterministic rebuild")
    return errors


def build_formal_cost_ledger(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    provider_attempt_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Reserve the full per-attempt token cost for every launched provider process."""

    cells = manifest.get("cells")
    cells = cells if isinstance(cells, list) else []
    cell_by_key = {
        str(cell.get("cell_key_sha256")): cell
        for cell in cells
        if isinstance(cell, Mapping)
    }
    task_rows = contract.get("per_task_attempt_caps")
    task_rows = task_rows if isinstance(task_rows, list) else []
    task_cost: dict[str, float] = {}
    for row in task_rows:
        if not isinstance(row, Mapping):
            continue
        cost = _finite_float(row.get("per_attempt_cost_cap_usd"))
        if cost is None:
            raise ValueError("cost ledger contains an invalid per-attempt cost cap")
        task_cost[str(row.get("task_binding_key"))] = cost
    normalized_counts: dict[str, int] = {}
    reserved = 0.0
    for key, raw_count in provider_attempt_counts.items():
        if key not in cell_by_key:
            raise ValueError(f"cost ledger contains an unknown formal cell: {key}")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError(f"cost ledger contains a non-integer attempt count: {key}")
        count = raw_count
        cell = cell_by_key[key]
        if count < 0 or count > int(cell.get("provider_attempt_limit", -1)):
            raise ValueError(f"cost ledger exceeds the provider-attempt cap: {key}")
        task_id = str(cell.get("task_binding_key"))
        if task_id not in task_cost:
            raise ValueError(f"cost ledger lacks task pricing: {task_id}")
        normalized_counts[key] = count
        reserved += count * task_cost[task_id]
    reserved = round(reserved, 12)
    ceiling_value = _finite_float(contract.get("formal_currency_ceiling_usd"), positive=True)
    if ceiling_value is None:
        raise ValueError("cost ledger lacks a positive formal currency ceiling")
    ceiling = ceiling_value
    ledger: dict[str, Any] = {
        "schema_version": FORMAL_COST_LEDGER_VERSION,
        "formal_cost_contract_sha256": contract.get("formal_cost_contract_sha256"),
        "formal_preflight_sha256": _base_preflight_sha256(manifest),
        "currency": "USD",
        "provider_attempt_count": sum(normalized_counts.values()),
        "provider_attempt_counts_by_cell_key_sha256": dict(sorted(normalized_counts.items())),
        "reserved_cost_usd": reserved,
        "formal_currency_ceiling_usd": ceiling,
        "remaining_unreserved_usd": round(ceiling - reserved, 12),
        "within_ceiling": reserved <= ceiling,
        "reservation_semantics": "full_frozen_token_cap_per_launched_provider_process",
    }
    ledger["formal_cost_ledger_sha256"] = formal_cost_ledger_sha256(ledger)
    return ledger


__all__ = [
    "FORMAL_COST_CONTRACT_VERSION",
    "FORMAL_COST_LEDGER_VERSION",
    "QUALIFICATION_COST_CONTRACT_VERSION",
    "QUALIFICATION_COST_LEDGER_VERSION",
    "build_formal_cost_contract",
    "build_formal_cost_ledger",
    "build_qualification_cost_contract",
    "build_qualification_cost_ledger",
    "formal_cost_contract_sha256",
    "formal_cost_ledger_sha256",
    "qualification_cost_contract_sha256",
    "qualification_cost_ledger_sha256",
    "validate_formal_cost_contract",
    "validate_qualification_cost_contract",
    "validate_qualification_cost_ledger",
]
