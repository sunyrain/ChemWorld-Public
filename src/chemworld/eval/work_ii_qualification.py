"""Fail-closed authorization, accounting, and receipt contracts for Work II qualification."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_cost import (
    build_qualification_cost_contract,
    validate_qualification_cost_contract,
    validate_qualification_cost_ledger,
)
from chemworld.eval.work_ii_formal import (
    EXPECTED_METHOD_QUALIFICATION_CONTRACT,
    FORMAL_ARMS,
    FORMAL_SNAPSHOT_STAGES,
)

METHOD_QUALIFICATION_REPORT_VERSION = "chemworld-work-ii-campaign-pilot-report-0.4"
METHOD_QUALIFICATION_RECEIPT_VERSION = "chemworld-work-ii-method-qualification-receipt-0.4"
METHOD_QUALIFICATION_EXECUTION_AUTHORIZATION_VERSION = (
    "chemworld-work-ii-method-qualification-execution-authorization-0.2"
)
METHOD_QUALIFICATION_ATTEMPT_AUTHORIZATION_VERSION = (
    "chemworld-work-ii-method-qualification-attempt-authorization-0.1"
)
METHOD_QUALIFICATION_EXECUTION_JOURNAL_VERSION = (
    "chemworld-work-ii-method-qualification-execution-journal-0.1"
)
QUALIFICATION_TRIPLET_RUNNER_PATH = (
    "scripts/run_work_ii_method_qualification_triplet.py"
)
QUALIFICATION_TERMINAL_RECEIPT_VERSION = (
    "chemworld-work-ii-qualification-terminal-receipt-0.1"
)
REQUIRED_CELL_QUALIFICATION_CHECKS = (
    "planned_complete_experiments",
    "typed_belief_checkpoints_complete",
    "recipe_diversity_reconciled",
    "one_campaign_session",
    "provider_session_completed",
    "final_recommendation_committed",
    "tool_integrity",
    "no_resource_rejection",
    "campaign_terminal",
    "process_time_reconciled",
    "task_required_operations_reconciled",
    "exact_replay",
    "execution_audit",
    "provider_usage_reconciled",
    "provider_operational_limits_reconciled",
)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _is_finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _finite_float(value: object, *, minimum: float = 0.0) -> float | None:
    if (
        not _is_finite_number(value, minimum=minimum)
        or isinstance(value, bool)
        or not isinstance(value, int | float)
    ):
        return None
    return float(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _task_binding(manifest: Mapping[str, Any], task_id: str) -> Mapping[str, Any] | None:
    rows = manifest.get("task_bindings")
    if not isinstance(rows, list):
        return None
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("task_id") == task_id]
    return matches[0] if len(matches) == 1 else None


def method_qualification_report_sha256(report: Mapping[str, Any]) -> str:
    """Return the embedded content hash for a qualification report."""

    return _self_hash(report, "report_sha256")


def qualification_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return _self_hash(receipt, "receipt_sha256")


def qualification_execution_authorization_sha256(
    authorization: Mapping[str, Any],
) -> str:
    return _self_hash(authorization, "authorization_sha256")


def qualification_attempt_authorization_sha256(
    authorization: Mapping[str, Any],
) -> str:
    return _self_hash(authorization, "attempt_authorization_sha256")


def qualification_execution_journal_sha256(journal: Mapping[str, Any]) -> str:
    return _self_hash(journal, "execution_journal_sha256")


def build_qualification_execution_authorization(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    currency_ceiling_usd: float | None,
    approved_at: str,
    pricing_source: str | None,
    pricing_observed_at: str | None,
    cache_hit_input_usd_per_million: float | None,
    cache_miss_input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    unlimited_spend_authorized: bool = False,
    pricing_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build the credential-free authorization that must exist before provider execution."""

    contract = manifest.get("method_qualification_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    task_id = str(contract.get("qualification_task_id", ""))
    task_binding = _task_binding(manifest, task_id)
    campaign_binding = (
        task_binding.get("campaign_config") if isinstance(task_binding, Mapping) else None
    )
    campaign_binding = campaign_binding if isinstance(campaign_binding, Mapping) else {}
    cost_contract = build_qualification_cost_contract(
        root,
        manifest,
        qualification_currency_ceiling_usd=(
            None
            if currency_ceiling_usd is None
            else float(currency_ceiling_usd)
        ),
        pricing_source=pricing_source,
        pricing_observed_at=pricing_observed_at,
        cache_hit_input_usd_per_million=(
            None
            if cache_hit_input_usd_per_million is None
            else float(cache_hit_input_usd_per_million)
        ),
        cache_miss_input_usd_per_million=(
            None
            if cache_miss_input_usd_per_million is None
            else float(cache_miss_input_usd_per_million)
        ),
        output_usd_per_million=(
            None
            if output_usd_per_million is None
            else float(output_usd_per_million)
        ),
        unlimited_spend_authorized=unlimited_spend_authorized,
        pricing_unavailable_reason=pricing_unavailable_reason,
    )
    authorization: dict[str, Any] = {
        "schema_version": METHOD_QUALIFICATION_EXECUTION_AUTHORIZATION_VERSION,
        "status": "authorized",
        "formal_result": False,
        "provider_execution_allowed": True,
        "formal_execution_authorized": False,
        "formal_participant_outcome_count_before_authorization": 0,
        "qualification_manifest_sha256": manifest.get("manifest_sha256"),
        "provider_contract_sha256": canonical_json_sha256(manifest.get("provider_contract")),
        "participant_execution_contract_sha256": manifest.get(
            "participant_execution_contract_sha256"
        ),
        "method_qualification_contract_sha256": manifest.get(
            "method_qualification_contract_sha256"
        ),
        "qualification_currency_budget": cost_contract,
        "qualification_schedule": {
            "task_id": task_id,
            "world_split": "development_and_qualification",
            "world_seed": contract.get("qualification_world_seed"),
            "prior_arms": list(FORMAL_ARMS),
            "campaign_config_path": campaign_binding.get("path"),
            "campaign_config_sha256": campaign_binding.get("sha256"),
            "provider_process_attempts_initial": 3,
            "provider_process_attempts_hard_cap": 6,
        },
        "user_authorization": {
            "authorized_by": "user",
            "approved_at": approved_at,
            "provider_contract_confirmed": True,
            "credential_rotation_confirmed": True,
            "currency": "USD",
            "currency_ceiling_usd": currency_ceiling_usd,
            "unlimited_spend_authorized": unlimited_spend_authorized,
            "scope_method_qualification_contract_sha256": manifest.get(
                "method_qualification_contract_sha256"
            ),
            "credentials_present": False,
        },
    }
    authorization["authorization_sha256"] = qualification_execution_authorization_sha256(
        authorization
    )
    return authorization


def validate_qualification_execution_authorization(
    root: Path,
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Validate the explicit pre-call user authorization for the exact qualification triplet."""

    errors: list[str] = []
    if (
        authorization.get("schema_version")
        != METHOD_QUALIFICATION_EXECUTION_AUTHORIZATION_VERSION
    ):
        errors.append("unexpected method-qualification execution authorization schema")
    if authorization.get(
        "authorization_sha256"
    ) != qualification_execution_authorization_sha256(authorization):
        errors.append("method-qualification execution authorization self-hash mismatch")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("provider_execution_allowed") is not True
        or authorization.get("formal_execution_authorized") is not False
        or authorization.get("formal_result") is not False
        or authorization.get("formal_participant_outcome_count_before_authorization") != 0
    ):
        errors.append("method-qualification execution is not outcome-blind authorized")
    if authorization.get("qualification_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        errors.append("method-qualification execution binds a different local manifest")
    expected_bindings = {
        "provider_contract_sha256": canonical_json_sha256(manifest.get("provider_contract")),
        "participant_execution_contract_sha256": manifest.get(
            "participant_execution_contract_sha256"
        ),
        "method_qualification_contract_sha256": manifest.get(
            "method_qualification_contract_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if authorization.get(field) != expected:
            errors.append(f"method-qualification execution has a mismatched {field}")
    cost_contract = authorization.get("qualification_currency_budget")
    if not isinstance(cost_contract, Mapping):
        errors.append("method-qualification execution lacks a currency budget contract")
    else:
        errors.extend(validate_qualification_cost_contract(root, manifest, cost_contract))
    contract = manifest.get("method_qualification_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    task_binding = _task_binding(manifest, str(contract.get("qualification_task_id", "")))
    campaign_binding = (
        task_binding.get("campaign_config") if isinstance(task_binding, Mapping) else None
    )
    campaign_binding = campaign_binding if isinstance(campaign_binding, Mapping) else {}
    schedule = authorization.get("qualification_schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    if schedule != {
        "task_id": contract.get("qualification_task_id"),
        "world_split": "development_and_qualification",
        "world_seed": contract.get("qualification_world_seed"),
        "prior_arms": list(FORMAL_ARMS),
        "campaign_config_path": campaign_binding.get("path"),
        "campaign_config_sha256": campaign_binding.get("sha256"),
        "provider_process_attempts_initial": 3,
        "provider_process_attempts_hard_cap": 6,
    }:
        errors.append("method-qualification execution schedule is not the frozen triplet")
    user = authorization.get("user_authorization")
    user = user if isinstance(user, Mapping) else {}
    ceiling = user.get("currency_ceiling_usd")
    unlimited = user.get("unlimited_spend_authorized") is True
    ceiling_value = (
        None
        if unlimited
        else (
        float(cast(int | float, ceiling))
        if _is_finite_number(ceiling)
        else None
        )
    )
    if (
        user.get("authorized_by") != "user"
        or not isinstance(user.get("approved_at"), str)
        or not user.get("approved_at")
        or user.get("provider_contract_confirmed") is not True
        or user.get("credential_rotation_confirmed") is not True
        or user.get("currency") != "USD"
        or (not unlimited and (ceiling_value is None or ceiling_value < 0.000000001))
        or (unlimited and ceiling is not None)
        or user.get("scope_method_qualification_contract_sha256")
        != manifest.get("method_qualification_contract_sha256")
        or user.get("credentials_present") is not False
    ):
        errors.append("method-qualification execution lacks valid user/provider/currency approval")
    if isinstance(cost_contract, Mapping):
        if cost_contract.get("qualification_currency_ceiling_usd") != ceiling_value:
            errors.append(
                "method-qualification execution currency budget differs from user approval"
            )
        if cost_contract.get("unlimited_spend_authorized") is not unlimited:
            errors.append(
                "method-qualification execution spend mode differs from user approval"
            )
    return errors


def build_qualification_attempt_authorization(
    execution_authorization: Mapping[str, Any],
    *,
    arm: str,
    attempt_number: int,
    attempt_id: str,
    qualification_cost_ledger_sha256: str,
) -> dict[str, Any]:
    """Bind one parent-managed provider process launch to the approved triplet."""

    receipt: dict[str, Any] = {
        "schema_version": METHOD_QUALIFICATION_ATTEMPT_AUTHORIZATION_VERSION,
        "status": "authorized_provider_process_launch",
        "formal_result": False,
        "execution_authorization_sha256": execution_authorization.get(
            "authorization_sha256"
        ),
        "arm": arm,
        "attempt_number": attempt_number,
        "attempt_id": attempt_id,
        "qualification_cost_ledger_sha256": qualification_cost_ledger_sha256,
        "provider_process_launch_allowed": True,
    }
    receipt["attempt_authorization_sha256"] = (
        qualification_attempt_authorization_sha256(receipt)
    )
    errors = validate_qualification_attempt_authorization(
        receipt, execution_authorization
    )
    if errors:
        raise ValueError("invalid qualification attempt authorization: " + "; ".join(errors))
    return receipt


def validate_qualification_attempt_authorization(
    receipt: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
) -> list[str]:
    """Validate one immutable, parent-issued qualification launch receipt."""

    errors: list[str] = []
    if receipt.get("schema_version") != METHOD_QUALIFICATION_ATTEMPT_AUTHORIZATION_VERSION:
        errors.append("unexpected qualification attempt authorization schema")
    if receipt.get("attempt_authorization_sha256") != (
        qualification_attempt_authorization_sha256(receipt)
    ):
        errors.append("qualification attempt authorization self-hash mismatch")
    schedule = execution_authorization.get("qualification_schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    attempt_number = receipt.get("attempt_number")
    if (
        receipt.get("status") != "authorized_provider_process_launch"
        or receipt.get("formal_result") is not False
        or receipt.get("provider_process_launch_allowed") is not True
        or receipt.get("execution_authorization_sha256")
        != execution_authorization.get("authorization_sha256")
        or receipt.get("arm") not in schedule.get("prior_arms", [])
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or not 1 <= attempt_number <= 2
        or not isinstance(receipt.get("attempt_id"), str)
        or not receipt.get("attempt_id")
        or not _is_sha256(receipt.get("qualification_cost_ledger_sha256"))
    ):
        errors.append("qualification attempt authorization is not an exact allowed launch")
    return errors


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _journal_artifact_binding(
    root: Path,
    path: Path,
    *,
    embedded_field: str,
) -> dict[str, Any]:
    payload = _load_object(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(path),
        embedded_field: payload.get(embedded_field),
    }


def _journal_bound_path(
    root: Path,
    binding: object,
    *,
    label: str,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    if not isinstance(binding, Mapping):
        return None, [f"{label} binding is missing"]
    relative = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        return None, [f"{label} binding is incomplete"]
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None, [f"{label} binding escapes the repository"]
    if not path.is_file() or file_sha256(path) != digest:
        errors.append(f"{label} binding is missing or stale")
    return path, errors


def build_qualification_execution_journal(
    root: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every provider attempt and terminal row before aggregating the report."""

    root = root.resolve()
    output_root = output_root.resolve()
    try:
        relative_output = output_root.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("qualification output root must be inside the repository") from error
    authorization_path = output_root / "execution_authorization.json"
    cost_contract_path = output_root / "qualification_cost_contract.json"
    final_ledger_path = output_root / "qualification_cost_ledger.json"
    authorization = _load_object(authorization_path)
    authorization_errors = validate_qualification_execution_authorization(
        root, authorization, manifest
    )
    if authorization_errors:
        raise ValueError(
            "qualification journal authorization is invalid: "
            + "; ".join(authorization_errors)
        )
    cost_contract = _load_object(cost_contract_path)
    if cost_contract != authorization.get("qualification_currency_budget"):
        raise ValueError("qualification journal cost contract differs from authorization")
    cost_errors = validate_qualification_cost_contract(root, manifest, cost_contract)
    if cost_errors:
        raise ValueError(
            "qualification journal cost contract is invalid: " + "; ".join(cost_errors)
        )
    final_ledger = _load_object(final_ledger_path)
    ledger_errors = validate_qualification_cost_ledger(
        manifest, cost_contract, final_ledger
    )
    if ledger_errors:
        raise ValueError(
            "qualification journal final ledger is invalid: "
            + "; ".join(ledger_errors)
        )

    attempts: list[dict[str, Any]] = []
    for arm in FORMAL_ARMS:
        directory = output_root / "attempt_authorizations" / arm
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        for authorization_file in paths:
            attempt = _load_object(authorization_file)
            attempt_number = int(attempt.get("attempt_number", -1))
            attempt_id = str(attempt.get("attempt_id", ""))
            ledger_file = (
                output_root
                / "cost_ledgers"
                / arm
                / f"{attempt_number}-{attempt_id}.json"
            )
            attempts.append(
                {
                    "arm": arm,
                    "attempt_number": attempt_number,
                    "attempt_id": attempt_id,
                    "attempt_authorization_binding": _journal_artifact_binding(
                        root,
                        authorization_file,
                        embedded_field="attempt_authorization_sha256",
                    ),
                    "cost_ledger_snapshot_binding": _journal_artifact_binding(
                        root,
                        ledger_file,
                        embedded_field="qualification_cost_ledger_sha256",
                    ),
                }
            )

    terminals: list[dict[str, Any]] = []
    for arm in FORMAL_ARMS:
        receipt_path = output_root / "terminal_receipts" / f"{arm}.json"
        receipt = _load_object(receipt_path)
        terminals.append(
            {
                "arm": arm,
                "state": receipt.get("state"),
                "attempt_id": receipt.get("attempt_id"),
                "terminal_receipt_binding": _journal_artifact_binding(
                    root,
                    receipt_path,
                    embedded_field="terminal_receipt_sha256",
                ),
            }
        )

    runner_path = root / QUALIFICATION_TRIPLET_RUNNER_PATH
    journal: dict[str, Any] = {
        "schema_version": METHOD_QUALIFICATION_EXECUTION_JOURNAL_VERSION,
        "status": "terminalized_triplet",
        "formal_result": False,
        "output_root": relative_output,
        "runner_binding": {
            "path": QUALIFICATION_TRIPLET_RUNNER_PATH,
            "sha256": file_sha256(runner_path),
        },
        "execution_authorization_binding": _journal_artifact_binding(
            root,
            authorization_path,
            embedded_field="authorization_sha256",
        ),
        "cost_contract_binding": _journal_artifact_binding(
            root,
            cost_contract_path,
            embedded_field="qualification_cost_contract_sha256",
        ),
        "final_cost_ledger_binding": _journal_artifact_binding(
            root,
            final_ledger_path,
            embedded_field="qualification_cost_ledger_sha256",
        ),
        "provider_attempt_count": final_ledger.get("provider_attempt_count"),
        "provider_attempt_counts_by_arm": final_ledger.get(
            "provider_attempt_counts_by_arm"
        ),
        "reserved_cost_usd": final_ledger.get("reserved_cost_usd"),
        "terminal_arm_count": len(terminals),
        "pending_arms": [],
        "attempts": attempts,
        "terminal_receipts": terminals,
    }
    journal["execution_journal_sha256"] = qualification_execution_journal_sha256(
        journal
    )
    errors = validate_qualification_execution_journal(root, journal, manifest)
    if errors:
        raise ValueError("built qualification execution journal is invalid: " + "; ".join(errors))
    return journal


def validate_qualification_execution_journal(
    root: Path,
    journal: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the parent-runner attempt journal and its terminal scientific rows."""

    root = root.resolve()
    errors: list[str] = []
    if journal.get("schema_version") != METHOD_QUALIFICATION_EXECUTION_JOURNAL_VERSION:
        errors.append("unexpected qualification execution journal schema")
    if journal.get("execution_journal_sha256") != qualification_execution_journal_sha256(
        journal
    ):
        errors.append("qualification execution journal self-hash mismatch")
    if (
        journal.get("status") != "terminalized_triplet"
        or journal.get("formal_result") is not False
        or journal.get("terminal_arm_count") != len(FORMAL_ARMS)
        or journal.get("pending_arms") != []
    ):
        errors.append("qualification execution journal is not a terminal triplet")
    relative_output = journal.get("output_root")
    if not isinstance(relative_output, str):
        errors.append("qualification execution journal lacks its output root")
        return errors
    output_root = (root / relative_output).resolve()
    try:
        output_root.relative_to(root)
    except ValueError:
        errors.append("qualification execution journal output root escapes the repository")
        return errors
    if not output_root.is_dir():
        errors.append("qualification execution journal output root is missing")
        return errors

    runner = journal.get("runner_binding")
    if (
        not isinstance(runner, Mapping)
        or runner.get("path") != QUALIFICATION_TRIPLET_RUNNER_PATH
        or not _is_sha256(runner.get("sha256"))
    ):
        errors.append("qualification execution journal has an invalid parent runner binding")

    authorization_path, binding_errors = _journal_bound_path(
        root,
        journal.get("execution_authorization_binding"),
        label="qualification journal authorization",
    )
    errors.extend(binding_errors)
    if authorization_path is None or not authorization_path.is_file():
        return errors
    if authorization_path != output_root / "execution_authorization.json":
        errors.append("qualification journal authorization is outside its output root")
    try:
        authorization = _load_object(authorization_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"qualification journal authorization cannot be loaded: {error}")
        return errors
    authorization_binding = journal.get("execution_authorization_binding")
    if (
        not isinstance(authorization_binding, Mapping)
        or authorization_binding.get("authorization_sha256")
        != authorization.get("authorization_sha256")
    ):
        errors.append("qualification journal embedded authorization hash is stale")
    errors.extend(
        validate_qualification_execution_authorization(root, authorization, manifest)
    )

    contract_path, contract_binding_errors = _journal_bound_path(
        root,
        journal.get("cost_contract_binding"),
        label="qualification journal cost contract",
    )
    errors.extend(contract_binding_errors)
    if contract_path is None or not contract_path.is_file():
        return errors
    if contract_path != output_root / "qualification_cost_contract.json":
        errors.append("qualification journal cost contract is outside its output root")
    try:
        cost_contract = _load_object(contract_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"qualification journal cost contract cannot be loaded: {error}")
        return errors
    contract_binding = journal.get("cost_contract_binding")
    if (
        not isinstance(contract_binding, Mapping)
        or contract_binding.get("qualification_cost_contract_sha256")
        != cost_contract.get("qualification_cost_contract_sha256")
        or cost_contract != authorization.get("qualification_currency_budget")
    ):
        errors.append("qualification journal cost contract differs from authorization")
    errors.extend(validate_qualification_cost_contract(root, manifest, cost_contract))

    final_ledger_path, final_binding_errors = _journal_bound_path(
        root,
        journal.get("final_cost_ledger_binding"),
        label="qualification journal final cost ledger",
    )
    errors.extend(final_binding_errors)
    if final_ledger_path is None or not final_ledger_path.is_file():
        return errors
    if final_ledger_path != output_root / "qualification_cost_ledger.json":
        errors.append("qualification journal final ledger is outside its output root")
    try:
        final_ledger = _load_object(final_ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"qualification journal final ledger cannot be loaded: {error}")
        return errors
    final_binding = journal.get("final_cost_ledger_binding")
    if (
        not isinstance(final_binding, Mapping)
        or final_binding.get("qualification_cost_ledger_sha256")
        != final_ledger.get("qualification_cost_ledger_sha256")
    ):
        errors.append("qualification journal embedded final-ledger hash is stale")
    errors.extend(
        validate_qualification_cost_ledger(manifest, cost_contract, final_ledger)
    )

    raw_attempts = journal.get("attempts")
    if not isinstance(raw_attempts, list):
        errors.append("qualification execution journal lacks its attempts")
        return errors
    observed_authorizations: set[Path] = set()
    observed_snapshots: set[Path] = set()
    attempt_records: dict[tuple[str, str], tuple[dict[str, Any], Path, dict[str, Any], Path]] = {}
    attempt_numbers: dict[str, list[int]] = {arm: [] for arm in FORMAL_ARMS}
    ordinals: list[int] = []
    for raw_attempt in raw_attempts:
        if not isinstance(raw_attempt, Mapping):
            errors.append("qualification execution journal contains a malformed attempt")
            continue
        arm = str(raw_attempt.get("arm", ""))
        attempt_id = str(raw_attempt.get("attempt_id", ""))
        attempt_number = raw_attempt.get("attempt_number")
        if arm not in FORMAL_ARMS or not isinstance(attempt_number, int):
            errors.append("qualification execution journal contains an unknown attempt")
            continue
        attempt_path, attempt_binding_errors = _journal_bound_path(
            root,
            raw_attempt.get("attempt_authorization_binding"),
            label=f"qualification attempt {arm}/{attempt_id}",
        )
        snapshot_path, snapshot_binding_errors = _journal_bound_path(
            root,
            raw_attempt.get("cost_ledger_snapshot_binding"),
            label=f"qualification ledger snapshot {arm}/{attempt_id}",
        )
        errors.extend(attempt_binding_errors)
        errors.extend(snapshot_binding_errors)
        if (
            attempt_path is None
            or snapshot_path is None
            or not attempt_path.is_file()
            or not snapshot_path.is_file()
        ):
            continue
        expected_attempt_path = (
            output_root
            / "attempt_authorizations"
            / arm
            / f"{attempt_number}-{attempt_id}.json"
        )
        expected_snapshot_path = (
            output_root
            / "cost_ledgers"
            / arm
            / f"{attempt_number}-{attempt_id}.json"
        )
        if attempt_path != expected_attempt_path or snapshot_path != expected_snapshot_path:
            errors.append(f"qualification attempt journal path is inconsistent: {arm}/{attempt_id}")
        try:
            attempt = _load_object(attempt_path)
            snapshot = _load_object(snapshot_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"qualification attempt journal cannot be loaded: {error}")
            continue
        if (
            attempt.get("arm") != arm
            or attempt.get("attempt_id") != attempt_id
            or attempt.get("attempt_number") != attempt_number
        ):
            errors.append(f"qualification attempt metadata is inconsistent: {arm}/{attempt_id}")
        errors.extend(validate_qualification_attempt_authorization(attempt, authorization))
        errors.extend(validate_qualification_cost_ledger(manifest, cost_contract, snapshot))
        attempt_binding = raw_attempt.get("attempt_authorization_binding")
        snapshot_binding = raw_attempt.get("cost_ledger_snapshot_binding")
        if (
            not isinstance(attempt_binding, Mapping)
            or attempt_binding.get("attempt_authorization_sha256")
            != attempt.get("attempt_authorization_sha256")
            or not isinstance(snapshot_binding, Mapping)
            or snapshot_binding.get("qualification_cost_ledger_sha256")
            != snapshot.get("qualification_cost_ledger_sha256")
            or attempt.get("qualification_cost_ledger_sha256")
            != snapshot.get("qualification_cost_ledger_sha256")
        ):
            errors.append(f"qualification attempt ledger binding is stale: {arm}/{attempt_id}")
        counts = snapshot.get("provider_attempt_counts_by_arm")
        if not isinstance(counts, Mapping) or counts.get(arm) != attempt_number:
            errors.append(f"qualification attempt number differs from ledger: {arm}/{attempt_id}")
        ordinal = snapshot.get("provider_attempt_count")
        if isinstance(ordinal, int):
            ordinals.append(ordinal)
        attempt_numbers[arm].append(attempt_number)
        observed_authorizations.add(attempt_path)
        observed_snapshots.add(snapshot_path)
        key = (arm, attempt_id)
        if key in attempt_records:
            errors.append(f"qualification execution journal repeats attempt: {arm}/{attempt_id}")
        attempt_records[key] = (attempt, attempt_path, snapshot, snapshot_path)

    actual_authorizations = {
        path.resolve()
        for path in (output_root / "attempt_authorizations").glob("*/*.json")
    }
    actual_snapshots = {
        path.resolve() for path in (output_root / "cost_ledgers").glob("*/*.json")
    }
    if observed_authorizations != actual_authorizations:
        errors.append("qualification execution journal omits or adds attempt authorizations")
    if observed_snapshots != actual_snapshots:
        errors.append("qualification execution journal omits or adds cost-ledger snapshots")
    for arm in FORMAL_ARMS:
        numbers = sorted(attempt_numbers[arm])
        if numbers != list(range(1, len(numbers) + 1)):
            errors.append(
                "qualification execution journal attempt numbers are not contiguous: "
                + arm
            )
    attempt_count = len(raw_attempts)
    if sorted(ordinals) != list(range(1, attempt_count + 1)):
        errors.append("qualification execution journal provider-attempt order is not contiguous")
    derived_counts = {arm: len(attempt_numbers[arm]) for arm in FORMAL_ARMS}
    if (
        journal.get("provider_attempt_count") != attempt_count
        or journal.get("provider_attempt_counts_by_arm") != derived_counts
        or final_ledger.get("provider_attempt_count") != attempt_count
        or final_ledger.get("provider_attempt_counts_by_arm") != derived_counts
        or journal.get("reserved_cost_usd") != final_ledger.get("reserved_cost_usd")
    ):
        errors.append("qualification execution journal totals differ from its final ledger")

    raw_terminals = journal.get("terminal_receipts")
    if not isinstance(raw_terminals, list):
        errors.append("qualification execution journal lacks terminal receipts")
        return errors
    terminal_rows: list[dict[str, Any]] = []
    terminal_states: dict[str, str] = {}
    observed_terminal_paths: set[Path] = set()
    observed_terminal_arms: list[str] = []
    for raw_terminal in raw_terminals:
        if not isinstance(raw_terminal, Mapping):
            errors.append("qualification execution journal contains a malformed terminal")
            continue
        arm = str(raw_terminal.get("arm", ""))
        attempt_id = str(raw_terminal.get("attempt_id", ""))
        terminal_path, terminal_binding_errors = _journal_bound_path(
            root,
            raw_terminal.get("terminal_receipt_binding"),
            label=f"qualification terminal {arm}",
        )
        errors.extend(terminal_binding_errors)
        if terminal_path is None or not terminal_path.is_file():
            continue
        if arm not in FORMAL_ARMS or terminal_path != (
            output_root / "terminal_receipts" / f"{arm}.json"
        ):
            errors.append(f"qualification terminal journal path is inconsistent: {arm}")
        try:
            terminal = _load_object(terminal_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"qualification terminal journal cannot be loaded: {error}")
            continue
        terminal_binding = raw_terminal.get("terminal_receipt_binding")
        if (
            terminal.get("schema_version") != QUALIFICATION_TERMINAL_RECEIPT_VERSION
            or terminal.get("terminal_receipt_sha256")
            != _self_hash(terminal, "terminal_receipt_sha256")
            or not isinstance(terminal_binding, Mapping)
            or terminal_binding.get("terminal_receipt_sha256")
            != terminal.get("terminal_receipt_sha256")
            or terminal.get("arm") != arm
            or terminal.get("attempt_id") != attempt_id
            or terminal.get("state") != raw_terminal.get("state")
        ):
            errors.append(f"qualification terminal receipt is invalid: {arm}")
        attempt_record = attempt_records.get((arm, attempt_id))
        if attempt_record is None:
            errors.append(f"qualification terminal lacks an authorized attempt: {arm}")
            continue
        attempt, attempt_path, snapshot, snapshot_path = attempt_record
        if attempt.get("attempt_number") != derived_counts.get(arm):
            errors.append(f"qualification terminal does not bind the latest attempt: {arm}")
        terminal_attempt_binding = terminal.get("attempt_authorization_binding")
        if not isinstance(terminal_attempt_binding, Mapping):
            errors.append(f"qualification terminal lacks its attempt binding: {arm}")
            continue
        bound_attempt = (
            output_root / str(terminal_attempt_binding.get("path", ""))
        ).resolve()
        if (
            bound_attempt != attempt_path
            or terminal_attempt_binding.get("sha256") != file_sha256(attempt_path)
            or terminal_attempt_binding.get("attempt_authorization_sha256")
            != attempt.get("attempt_authorization_sha256")
        ):
            errors.append(f"qualification terminal attempt binding is stale: {arm}")
        row_binding = terminal.get("row_binding")
        if not isinstance(row_binding, Mapping):
            errors.append(f"qualification terminal lacks its row binding: {arm}")
            continue
        row_path = (output_root / str(row_binding.get("path", ""))).resolve()
        try:
            row_path.relative_to(output_root)
        except ValueError:
            errors.append(f"qualification terminal row escapes output root: {arm}")
            continue
        if not row_path.is_file() or row_binding.get("sha256") != file_sha256(row_path):
            errors.append(f"qualification terminal row binding is stale: {arm}")
            continue
        try:
            row = _load_object(row_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"qualification terminal row cannot be loaded: {error}")
            continue
        expected_child_binding = {
            "path": attempt_path.relative_to(root).as_posix(),
            "sha256": file_sha256(attempt_path),
            "attempt_authorization_sha256": attempt.get(
                "attempt_authorization_sha256"
            ),
            "qualification_cost_ledger_path": snapshot_path.relative_to(root).as_posix(),
            "qualification_cost_ledger_sha256": snapshot.get(
                "qualification_cost_ledger_sha256"
            ),
        }
        if (
            row.get("arm") != arm
            or row.get("qualification_attempt_authorization_binding")
            != expected_child_binding
            or (terminal.get("state") == "completed") != (row.get("completed") is True)
        ):
            errors.append(f"qualification terminal row differs from its attempt: {arm}")
        terminal_rows.append(row)
        terminal_states[arm] = str(terminal.get("state"))
        observed_terminal_paths.add(terminal_path)
        observed_terminal_arms.append(arm)

    actual_terminal_paths = {
        path.resolve() for path in (output_root / "terminal_receipts").glob("*.json")
    }
    if observed_terminal_paths != actual_terminal_paths:
        errors.append("qualification execution journal omits or adds terminal receipts")
    if observed_terminal_arms != list(FORMAL_ARMS):
        errors.append("qualification execution journal does not preserve the arm triplet")
    if report is not None:
        if report.get("results") != terminal_rows:
            errors.append("qualification report results differ from execution journal")
        if report.get("terminal_states_by_arm") != terminal_states:
            errors.append("qualification report states differ from execution journal")
        report_authorization = report.get(
            "qualification_execution_authorization_binding"
        )
        if (
            not isinstance(report_authorization, Mapping)
            or report_authorization.get("authorization_sha256")
            != authorization.get("authorization_sha256")
        ):
            errors.append("qualification report authorization differs from execution journal")
    return errors


def validate_qualification_execution_journal_binding(
    root: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Validate the report's immutable binding to its parent-runner journal."""

    binding = report.get("qualification_execution_journal_binding")
    journal_path, errors = _journal_bound_path(
        root,
        binding,
        label="qualification execution journal",
    )
    if journal_path is None or not journal_path.is_file():
        return errors
    try:
        journal = _load_object(journal_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"qualification execution journal cannot be loaded: {error}")
        return errors
    if (
        not isinstance(binding, Mapping)
        or binding.get("execution_journal_sha256")
        != journal.get("execution_journal_sha256")
    ):
        errors.append("qualification execution journal embedded hash is stale")
    output_root = (root.resolve() / str(journal.get("output_root", ""))).resolve()
    if journal_path != output_root / "execution_journal.json":
        errors.append("qualification execution journal is outside its output root")
    errors.extend(
        validate_qualification_execution_journal(
            root,
            journal,
            manifest,
            report=report,
        )
    )
    return errors


def _qualification_usage_accounting(
    report: Mapping[str, Any],
    cost_contract: Mapping[str, Any],
) -> dict[str, Any]:
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_ARMS):
        raise ValueError("qualification usage accounting requires the exact arm triplet")
    totals = {
        "input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
    }
    field_map = {
        "input_tokens": "input_token_count",
        "uncached_input_tokens": "uncached_input_token_count",
        "output_tokens": "output_token_count",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("qualification usage accounting contains a malformed row")
        usage = row.get("method_resources")
        if not isinstance(usage, Mapping):
            raise ValueError("qualification usage accounting lacks method resources")
        for total_field, row_field in field_map.items():
            value = usage.get(row_field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"qualification usage accounting has invalid {row_field}"
                )
            totals[total_field] += value
    if totals["uncached_input_tokens"] > totals["input_tokens"]:
        raise ValueError("qualification uncached input exceeds cumulative input")
    pricing = cost_contract.get("pricing")
    if not isinstance(pricing, Mapping):
        raise ValueError("qualification usage accounting lacks frozen pricing")
    if pricing.get("pricing_available") is False:
        if (
            any(
                pricing.get(field) is not None
                for field in ("cache_hit_input", "cache_miss_input", "output")
            )
            or not isinstance(pricing.get("pricing_unavailable_reason"), str)
            or not pricing.get("pricing_unavailable_reason")
        ):
            raise ValueError("qualification unavailable-pricing contract is invalid")
        return {
            "token_totals": totals,
            "calculated_cost_usd": None,
            "pricing_contract_sha256": cost_contract.get(
                "qualification_cost_contract_sha256"
            ),
            "pricing_unavailable_reason": pricing.get(
                "pricing_unavailable_reason"
            ),
        }
    hit_rate = _finite_float(pricing.get("cache_hit_input"))
    miss_rate = _finite_float(pricing.get("cache_miss_input"))
    output_rate = _finite_float(pricing.get("output"))
    if hit_rate is None or miss_rate is None or output_rate is None:
        raise ValueError("qualification usage accounting has invalid frozen pricing")
    cached = totals["input_tokens"] - totals["uncached_input_tokens"]
    calculated_cost = round(
        (
            cached * hit_rate
            + totals["uncached_input_tokens"] * miss_rate
            + totals["output_tokens"] * output_rate
        )
        / 1_000_000.0,
        12,
    )
    return {
        "token_totals": totals,
        "calculated_cost_usd": calculated_cost,
        "pricing_contract_sha256": cost_contract.get(
            "qualification_cost_contract_sha256"
        ),
    }


def validate_method_qualification_report(
    root: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    """Validate the scientific-process contents of a three-arm qualification report."""

    errors: list[str] = []
    contract = manifest.get("method_qualification_contract")
    if contract != EXPECTED_METHOD_QUALIFICATION_CONTRACT:
        errors.append("qualification report was checked against an invalid gate contract")
        contract = EXPECTED_METHOD_QUALIFICATION_CONTRACT
    if report.get("schema_version") != METHOD_QUALIFICATION_REPORT_VERSION:
        errors.append("unexpected method qualification report schema")
    if report.get("report_sha256") != method_qualification_report_sha256(report):
        errors.append("method qualification report self-hash mismatch")
    if report.get("formal_result") is not False:
        errors.append("method qualification report is mislabeled as a formal result")
    if report.get("qualification_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        errors.append("method qualification report binds a different local manifest")
    if report.get("qualification_execution_authorized") is not True:
        errors.append("method qualification report lacks pre-execution user authorization")
    authorization_binding = report.get("qualification_execution_authorization_binding")
    if not isinstance(authorization_binding, Mapping):
        errors.append("method qualification report lacks its execution-authorization binding")
    else:
        relative = authorization_binding.get("path")
        digest = authorization_binding.get("sha256")
        embedded = authorization_binding.get("authorization_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            errors.append("method qualification execution-authorization binding is incomplete")
        else:
            root = root.resolve()
            authorization_path = (root / relative).resolve()
            try:
                authorization_path.relative_to(root)
            except ValueError:
                errors.append("method qualification execution authorization escapes the repository")
            else:
                if (
                    not authorization_path.is_file()
                    or file_sha256(authorization_path) != digest
                ):
                    errors.append(
                        "method qualification execution authorization is missing or stale"
                    )
                else:
                    loaded = json.loads(authorization_path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        errors.append(
                            "method qualification execution authorization is not an object"
                        )
                    else:
                        if loaded.get("authorization_sha256") != embedded:
                            errors.append(
                                "method qualification embedded execution authorization is stale"
                            )
                        errors.extend(
                            validate_qualification_execution_authorization(
                                root, loaded, manifest
                            )
                        )
    if report.get("qualification_execution_authorized") is True:
        errors.extend(
            validate_qualification_execution_journal_binding(
                root, report, manifest
            )
        )
    if report.get("world_seed") != contract["qualification_world_seed"]:
        errors.append("method qualification report uses the wrong development world")
    if (
        report.get("cell_count") != contract["qualification_cell_count"]
        or report.get("completed_cell_count") != contract["qualification_cell_count"]
    ):
        errors.append("method qualification report does not complete three cells")

    task_id = str(contract["qualification_task_id"])
    task_binding = _task_binding(manifest, task_id)
    campaign_binding = (
        task_binding.get("campaign_config") if isinstance(task_binding, Mapping) else None
    )
    if not isinstance(campaign_binding, Mapping):
        errors.append("method qualification report lacks a frozen task binding")
    elif report.get("config_file_sha256") != campaign_binding.get("sha256"):
        errors.append("method qualification report used a different campaign config")
    if not _is_sha256(report.get("config_sha256")):
        errors.append("method qualification report lacks its canonical config hash")

    rows = report.get("results")
    if not isinstance(rows, list):
        errors.append("method qualification report results are missing")
        return errors
    observed_arms = [row.get("arm") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 3 or tuple(observed_arms) != FORMAL_ARMS:
        errors.append("method qualification report does not retain the exact arm triplet")

    provider = manifest.get("provider_contract")
    provider = provider if isinstance(provider, Mapping) else {}
    for index, raw_row in enumerate(rows):
        arm = FORMAL_ARMS[index] if index < len(FORMAL_ARMS) else f"cell_{index + 1}"
        if not isinstance(raw_row, Mapping):
            errors.append(f"{arm}: method qualification cell is malformed")
            continue
        if raw_row.get("completed") is not True or raw_row.get("failure") is not None:
            errors.append(f"{arm}: method qualification cell did not complete")
        qualification = raw_row.get("qualification")
        qualification = qualification if isinstance(qualification, Mapping) else {}
        checks = qualification.get("checks")
        checks = checks if isinstance(checks, Mapping) else {}
        if (
            qualification.get("passed") is not True
            or qualification.get("failed_checks") != []
            or set(checks) != set(REQUIRED_CELL_QUALIFICATION_CHECKS)
            or any(checks.get(name) is not True for name in REQUIRED_CELL_QUALIFICATION_CHECKS)
        ):
            errors.append(f"{arm}: fail-closed cell qualification did not pass")

        replay = raw_row.get("exact_replay")
        replay = replay if isinstance(replay, Mapping) else {}
        if replay.get("verified") is not True or replay.get("mismatches") != []:
            errors.append(f"{arm}: exact replay did not pass")

        analysis = raw_row.get("analysis")
        analysis = analysis if isinstance(analysis, Mapping) else {}
        experiments = analysis.get("experiments")
        experiments = experiments if isinstance(experiments, list) else []
        experiment_indices = [
            item.get("experiment_index") for item in experiments if isinstance(item, Mapping)
        ]
        snapshots = analysis.get("belief_snapshots")
        snapshots = snapshots if isinstance(snapshots, list) else []
        snapshot_stages = [item.get("stage") for item in snapshots if isinstance(item, Mapping)]
        resources = analysis.get("final_campaign_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        state = resources.get("state")
        state = state if isinstance(state, Mapping) else {}
        audit = analysis.get("execution_audit")
        audit = audit if isinstance(audit, Mapping) else {}
        recommendation = analysis.get("final_recommendation")
        recommendation = recommendation if isinstance(recommendation, Mapping) else {}
        recommendation_hash = canonical_json_sha256(recommendation) if recommendation else None
        expected_experiments = int(contract["complete_experiments_per_cell"])
        if (
            analysis.get("complete_experiment_count") != expected_experiments
            or analysis.get("right_censored_open_experiment") is not False
            or experiment_indices != list(range(1, expected_experiments + 1))
            or snapshot_stages != list(FORMAL_SNAPSHOT_STAGES)
            or analysis.get("resource_rejection_count") != 0
            or resources.get("campaign_terminal") is not True
            or state.get("closed_batches") != expected_experiments
            or state.get("final_assays") != expected_experiments
            or audit.get("passed") is not True
            or recommendation_hash != analysis.get("final_recommendation_sha256")
        ):
            errors.append(f"{arm}: campaign lifecycle or execution audit is incomplete")

        usage = raw_row.get("method_resources")
        usage = usage if isinstance(usage, Mapping) else {}
        provenance = usage.get("model_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        parameters = provenance.get("request_parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        token_fields = (
            "input_token_count",
            "uncached_input_token_count",
            "output_token_count",
        )
        if (
            usage.get("provider_session_count") != 1
            or usage.get("model_call_count") != 1
            or usage.get("provider_usage_pending") is not False
            or usage.get("provider_usage_accounting_complete") is not True
            or usage.get("in_flight_model_call_count") != 0
            or any(not _is_finite_number(usage.get(field)) for field in token_fields)
            or provenance.get("provider_id") != provider.get("id")
            or provenance.get("model_id") != provider.get("model")
            or parameters.get("reasoning_effort") != provider.get("reasoning_effort")
        ):
            errors.append(f"{arm}: provider/session/resource accounting is invalid")

        receipts = raw_row.get("provider_receipts")
        receipts = receipts if isinstance(receipts, list) else []
        receipt = receipts[0] if len(receipts) == 1 and isinstance(receipts[0], Mapping) else {}
        host_commit_required = (
            receipt.get("final_recommendation_source") == "host_mcp_commit"
            or isinstance(receipt.get("final_recommendation_commit"), Mapping)
        )
        provider_terminal_completed = (
            receipt.get("status") == "completed"
            and receipt.get("return_code") == 0
            and (
                (
                    host_commit_required
                    and receipt.get("final_recommendation_source") == "host_mcp_commit"
                )
                or (
                    not host_commit_required
                    and receipt.get("final_payload_valid") is True
                    and receipt.get("final_payload_status") == "campaign_complete"
                )
            )
        )
        if (
            len(receipts) != 1
            or receipt.get("session_scope") != "campaign"
            or not provider_terminal_completed
            or receipt.get("final_recommendation_sha256") != recommendation_hash
            or (
                host_commit_required
                and (
                    receipt.get("final_recommendation_source") != "host_mcp_commit"
                    or not any(
                        isinstance(item, Mapping)
                        and item.get("tool") == "commit_final_recommendation"
                        and item.get("status") == "completed"
                        for item in receipt.get("mcp_tool_calls", [])
                    )
                )
            )
            or receipt.get("experiment_tool_integrity_verified_after_session") is not True
            or receipt.get("lab_tool_integrity_verified_after_session") is not True
            or receipt.get("mcp_tool_integrity_verified_after_session") is not True
            or receipt.get("model_id") != provider.get("model")
            or receipt.get("reasoning_effort") != provider.get("reasoning_effort")
            or receipt.get("usage_complete") is not True
        ):
            errors.append(f"{arm}: provider receipt is invalid")
    return errors




def build_method_qualification_receipt(
    root: Path,
    report_path: Path,
    manifest: Mapping[str, Any],
    *,
    observed_cost_usd: float | None,
    pricing_source: str,
    pricing_observed_at: str,
) -> dict[str, Any]:
    """Build the formal authorization receipt from one validated three-arm report."""

    root = root.resolve()
    report_path = report_path.resolve()
    try:
        relative_report = report_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("qualification report must be inside the repository") from error
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("qualification report must contain an object")
    report_errors = validate_method_qualification_report(root, report, manifest)
    if report_errors:
        raise ValueError("invalid method qualification report: " + "; ".join(report_errors))
    authorization_binding = report["qualification_execution_authorization_binding"]
    authorization_path = root / str(authorization_binding["path"])
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not isinstance(authorization, dict):
        raise ValueError("qualification execution authorization must contain an object")
    authorization_errors = validate_qualification_execution_authorization(
        root, authorization, manifest
    )
    if authorization_errors:
        raise ValueError(
            "invalid qualification execution authorization: "
            + "; ".join(authorization_errors)
        )
    user = authorization["user_authorization"]
    unlimited = user.get("unlimited_spend_authorized") is True
    raw_ceiling = user.get("currency_ceiling_usd")
    ceiling = None if unlimited else float(raw_ceiling)
    cost_contract = authorization.get("qualification_currency_budget")
    if not isinstance(cost_contract, Mapping):
        raise ValueError("qualification authorization lacks its cost contract")
    usage_accounting = _qualification_usage_accounting(report, cost_contract)
    frozen_pricing = cost_contract.get("pricing")
    frozen_pricing = frozen_pricing if isinstance(frozen_pricing, Mapping) else {}
    observed_cost_value = (
        None
        if unlimited
        else (
            round(float(observed_cost_usd), 12)
            if _is_finite_number(observed_cost_usd)
            else None
        )
    )
    if (
        (unlimited and observed_cost_usd is not None)
        or (
            not unlimited
            and (
                observed_cost_value is None
                or ceiling is None
                or observed_cost_value > ceiling
            )
        )
        or observed_cost_value != usage_accounting["calculated_cost_usd"]
        or pricing_source != frozen_pricing.get("source")
        or pricing_observed_at != frozen_pricing.get("observed_at")
    ):
        raise ValueError(
            "qualification cost accounting differs from frozen prices, observed tokens or approval"
        )
    contract = EXPECTED_METHOD_QUALIFICATION_CONTRACT
    task_binding = _task_binding(manifest, str(contract["qualification_task_id"]))
    campaign_binding = (
        task_binding.get("campaign_config") if isinstance(task_binding, Mapping) else None
    )
    if not isinstance(campaign_binding, Mapping):
        raise ValueError("formal manifest lacks the qualification campaign binding")
    receipt: dict[str, Any] = {
        "schema_version": METHOD_QUALIFICATION_RECEIPT_VERSION,
        "status": "passed",
        "formal_execution_authorized": False,
        "qualification_manifest_sha256": manifest.get("manifest_sha256"),
        "provider_contract_sha256": canonical_json_sha256(manifest.get("provider_contract")),
        "provider_attempt_contract_sha256": canonical_json_sha256(
            manifest.get("provider_attempt_contract")
        ),
        "participant_execution_contract_sha256": manifest.get(
            "participant_execution_contract_sha256"
        ),
        "method_qualification_contract_sha256": manifest.get(
            "method_qualification_contract_sha256"
        ),
        "qualification_execution_authorization_sha256": authorization.get(
            "authorization_sha256"
        ),
        "qualification_split": "development_seed_0",
        "qualification_task_id": contract["qualification_task_id"],
        "qualification_world_seed": contract["qualification_world_seed"],
        "qualification_campaign_config_sha256": campaign_binding.get("sha256"),
        "qualified_prior_arms": list(FORMAL_ARMS),
        "qualified_cell_count": 3,
        "formal_participant_outcome_count_before_authorization": 0,
        "approved_provider_attempt_hard_cap": manifest.get("expected_counts", {}).get(
            "provider_attempts_hard_cap"
        ),
        "qualification_cost_accounting": {
            "currency": "USD",
            "accounting_complete": True,
            "observed_cost_usd": observed_cost_value,
            **usage_accounting,
            "approved_ceiling_usd": ceiling,
            "unlimited_spend_authorized": unlimited,
            "approved_by": "user",
            "approved_at": user["approved_at"],
            "pricing_source": pricing_source,
            "pricing_observed_at": pricing_observed_at,
            "pricing_unavailable_reason": usage_accounting.get(
                "pricing_unavailable_reason"
            ),
            "scope_method_qualification_contract_sha256": manifest.get(
                "method_qualification_contract_sha256"
            ),
        },
        "approved_currency_ceiling_usd": ceiling,
        "currency_approval": {
            "approved_by": "user",
            "approved_at": user["approved_at"],
            "approved_currency_ceiling_usd": ceiling,
            "unlimited_spend_authorized": unlimited,
            "scope_qualification_manifest_sha256": manifest.get(
                "manifest_sha256"
            ),
        },
        "qualification_report_binding": {
            "path": relative_report,
            "sha256": file_sha256(report_path),
            "report_sha256": report.get("report_sha256"),
        },
    }
    receipt["receipt_sha256"] = qualification_receipt_sha256(receipt)
    errors = validate_method_qualification_receipt(
        root,
        receipt,
        manifest,
        currency_ceiling_usd=ceiling,
    )
    if errors:
        raise ValueError("built method qualification receipt is invalid: " + "; ".join(errors))
    return receipt


def validate_method_qualification_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    currency_ceiling_usd: float | None,
) -> list[str]:
    """Validate method qualification, user cost approval and artifact binding."""

    errors: list[str] = []
    if receipt.get("schema_version") != METHOD_QUALIFICATION_RECEIPT_VERSION:
        errors.append("unexpected method qualification receipt schema")
    if receipt.get("receipt_sha256") != qualification_receipt_sha256(receipt):
        errors.append("method qualification receipt self-hash mismatch")
    if (
        receipt.get("status") != "passed"
        or receipt.get("formal_execution_authorized") is not False
    ):
        errors.append("method qualification receipt crossed the formal release boundary")
    if receipt.get("qualification_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        errors.append("method qualification receipt binds a different local manifest")
    expected_bindings = {
        "provider_contract_sha256": canonical_json_sha256(manifest.get("provider_contract")),
        "provider_attempt_contract_sha256": canonical_json_sha256(
            manifest.get("provider_attempt_contract")
        ),
        "participant_execution_contract_sha256": manifest.get(
            "participant_execution_contract_sha256"
        ),
        "method_qualification_contract_sha256": manifest.get(
            "method_qualification_contract_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if receipt.get(field) != expected:
            errors.append(f"method qualification receipt has a mismatched {field}")
    contract = EXPECTED_METHOD_QUALIFICATION_CONTRACT
    if receipt.get("qualification_split") != "development_seed_0":
        errors.append("method qualification receipt uses the wrong development split")
    if receipt.get("qualification_task_id") != contract["qualification_task_id"]:
        errors.append("method qualification receipt uses the wrong qualification task")
    if receipt.get("qualification_world_seed") != contract["qualification_world_seed"]:
        errors.append("method qualification receipt uses the wrong qualification world")
    if receipt.get("qualified_prior_arms") != list(FORMAL_ARMS):
        errors.append("method qualification receipt does not cover the exact three arms")
    if receipt.get("qualified_cell_count") != 3:
        errors.append("method qualification receipt does not cover three development cells")
    if receipt.get("formal_participant_outcome_count_before_authorization") != 0:
        errors.append("formal participant outcomes existed before method authorization")
    task_binding = _task_binding(manifest, str(contract["qualification_task_id"]))
    campaign_binding = (
        task_binding.get("campaign_config") if isinstance(task_binding, Mapping) else None
    )
    if not isinstance(campaign_binding, Mapping) or receipt.get(
        "qualification_campaign_config_sha256"
    ) != campaign_binding.get("sha256"):
        errors.append("method qualification receipt binds a different campaign config")
    attempt_cap = manifest.get("expected_counts", {}).get("provider_attempts_hard_cap")
    if receipt.get("approved_provider_attempt_hard_cap") != attempt_cap:
        errors.append("method qualification receipt has a mismatched provider-attempt cap")

    qualification_cost = receipt.get("qualification_cost_accounting")
    qualification_cost = qualification_cost if isinstance(qualification_cost, Mapping) else {}
    observed_cost = qualification_cost.get("observed_cost_usd")
    calculated_cost = qualification_cost.get("calculated_cost_usd")
    qualification_ceiling = qualification_cost.get("approved_ceiling_usd")
    unlimited = qualification_cost.get("unlimited_spend_authorized") is True
    observed_cost_value = _finite_float(observed_cost)
    calculated_cost_value = _finite_float(calculated_cost)
    qualification_ceiling_value = (
        None
        if unlimited
        else _finite_float(qualification_ceiling, minimum=0.000000001)
    )
    if (
        qualification_cost.get("currency") != "USD"
        or qualification_cost.get("accounting_complete") is not True
        or qualification_cost.get("approved_by") != "user"
        or not isinstance(qualification_cost.get("approved_at"), str)
        or not qualification_cost.get("approved_at")
        or not isinstance(qualification_cost.get("pricing_source"), str)
        or not qualification_cost.get("pricing_source")
        or not isinstance(qualification_cost.get("pricing_observed_at"), str)
        or not qualification_cost.get("pricing_observed_at")
        or (
            unlimited
            and (
                observed_cost is not None
                or calculated_cost is not None
                or qualification_ceiling is not None
                or not isinstance(
                    qualification_cost.get("pricing_unavailable_reason"), str
                )
                or not qualification_cost.get("pricing_unavailable_reason")
            )
        )
        or (
            not unlimited
            and (
                observed_cost_value is None
                or calculated_cost_value is None
                or observed_cost_value != calculated_cost_value
            )
        )
        or not isinstance(qualification_cost.get("token_totals"), Mapping)
        or not _is_sha256(qualification_cost.get("pricing_contract_sha256"))
        or (not unlimited and qualification_ceiling_value is None)
        or (
            observed_cost_value is not None
            and qualification_ceiling_value is not None
            and observed_cost_value > qualification_ceiling_value
        )
        or qualification_cost.get("scope_method_qualification_contract_sha256")
        != manifest.get("method_qualification_contract_sha256")
    ):
        errors.append("method qualification receipt has invalid qualification cost accounting")

    approved = receipt.get("approved_currency_ceiling_usd")
    approved_value = (
        None if unlimited else _finite_float(approved, minimum=0.000000001)
    )
    if (
        (unlimited and (approved is not None or currency_ceiling_usd is not None))
        or (
            not unlimited
            and (
                approved_value is None
                or currency_ceiling_usd is None
                or approved_value != float(currency_ceiling_usd)
            )
        )
    ):
        errors.append("method qualification receipt has a mismatched currency ceiling")
    if approved_value is not None and (
        qualification_ceiling_value is None
        or qualification_ceiling_value != approved_value
    ):
        errors.append("qualification cost ceiling differs from the user-approved ceiling")
    approval = receipt.get("currency_approval")
    if not isinstance(approval, Mapping):
        errors.append("method qualification receipt lacks user currency approval")
    elif (
        approval.get("approved_by") != "user"
        or approval.get("scope_qualification_manifest_sha256")
        != manifest.get("manifest_sha256")
        or approval.get("approved_currency_ceiling_usd") != approved
        or approval.get("unlimited_spend_authorized") is not unlimited
        or not isinstance(approval.get("approved_at"), str)
        or not approval.get("approved_at")
    ):
        errors.append("method qualification receipt has invalid user currency approval")

    binding = receipt.get("qualification_report_binding")
    if not isinstance(binding, Mapping):
        errors.append("method qualification receipt lacks its report binding")
    else:
        relative = binding.get("path")
        digest = binding.get("sha256")
        embedded_digest = binding.get("report_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            errors.append("method qualification report binding is incomplete")
        else:
            root = root.resolve()
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append("method qualification report binding escapes the repository")
            else:
                if not path.is_file() or file_sha256(path) != digest:
                    errors.append("method qualification report binding is missing or stale")
                else:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        errors.append("method qualification report is not an object")
                    else:
                        if loaded.get("report_sha256") != embedded_digest:
                            errors.append("method qualification embedded report hash is stale")
                        authorization_binding = loaded.get(
                            "qualification_execution_authorization_binding"
                        )
                        authorization_binding = (
                            authorization_binding
                            if isinstance(authorization_binding, Mapping)
                            else {}
                        )
                        if receipt.get(
                            "qualification_execution_authorization_sha256"
                        ) != authorization_binding.get("authorization_sha256"):
                            errors.append(
                                "method qualification receipt does not bind pre-call authorization"
                            )
                        authorization_relative = authorization_binding.get("path")
                        if isinstance(authorization_relative, str):
                            authorization_path = (root / authorization_relative).resolve()
                            try:
                                authorization_path.relative_to(root)
                                authorization = _load_object(authorization_path)
                                cost_contract = authorization.get(
                                    "qualification_currency_budget"
                                )
                                if not isinstance(cost_contract, Mapping):
                                    raise ValueError(
                                        "authorization lacks qualification cost contract"
                                    )
                                expected_accounting = _qualification_usage_accounting(
                                    loaded, cost_contract
                                )
                                frozen_pricing = cost_contract.get("pricing")
                                frozen_pricing = (
                                    frozen_pricing
                                    if isinstance(frozen_pricing, Mapping)
                                    else {}
                                )
                                if (
                                    qualification_cost.get("token_totals")
                                    != expected_accounting["token_totals"]
                                    or qualification_cost.get("calculated_cost_usd")
                                    != expected_accounting["calculated_cost_usd"]
                                    or qualification_cost.get("observed_cost_usd")
                                    != expected_accounting["calculated_cost_usd"]
                                    or qualification_cost.get(
                                        "pricing_contract_sha256"
                                    )
                                    != expected_accounting["pricing_contract_sha256"]
                                    or qualification_cost.get("pricing_source")
                                    != frozen_pricing.get("source")
                                    or qualification_cost.get("pricing_observed_at")
                                    != frozen_pricing.get("observed_at")
                                ):
                                    errors.append(
                                        "method qualification receipt cost differs from "
                                        "frozen prices or report tokens"
                                    )
                            except (OSError, ValueError, json.JSONDecodeError) as error:
                                errors.append(
                                    "method qualification receipt cost cannot be rebuilt: "
                                    + str(error)
                                )
                        errors.extend(validate_method_qualification_report(root, loaded, manifest))
    return errors


def qualification_receipt_currency_ceiling(
    receipt: Mapping[str, Any],
) -> float | None:
    """Return the approved qualification ceiling, including explicit unlimited use."""

    accounting = receipt.get("qualification_cost_accounting")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    approved = receipt.get("approved_currency_ceiling_usd")
    if accounting.get("unlimited_spend_authorized") is True:
        if approved is not None:
            raise ValueError(
                "unlimited method qualification cannot declare a currency ceiling"
            )
        return None
    approved_value = _finite_float(approved, minimum=0.000000001)
    if approved_value is None:
        raise ValueError(
            "priced method qualification lacks a positive currency ceiling"
        )
    return approved_value


__all__ = [
    "METHOD_QUALIFICATION_ATTEMPT_AUTHORIZATION_VERSION",
    "METHOD_QUALIFICATION_EXECUTION_AUTHORIZATION_VERSION",
    "METHOD_QUALIFICATION_EXECUTION_JOURNAL_VERSION",
    "METHOD_QUALIFICATION_RECEIPT_VERSION",
    "METHOD_QUALIFICATION_REPORT_VERSION",
    "REQUIRED_CELL_QUALIFICATION_CHECKS",
    "build_qualification_attempt_authorization",
    "build_qualification_execution_authorization",
    "build_qualification_execution_journal",
    "method_qualification_report_sha256",
    "qualification_attempt_authorization_sha256",
    "qualification_execution_authorization_sha256",
    "qualification_execution_journal_sha256",
    "qualification_receipt_currency_ceiling",
    "qualification_receipt_sha256",
    "validate_method_qualification_receipt",
    "validate_method_qualification_report",
    "validate_qualification_attempt_authorization",
    "validate_qualification_execution_authorization",
    "validate_qualification_execution_journal",
    "validate_qualification_execution_journal_binding",
]
