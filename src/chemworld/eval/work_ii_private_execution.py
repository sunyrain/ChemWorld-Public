"""Fail-closed authorization and execution for the sealed Work II private matrix."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.work_ii_private import WORK_II_PRIVATE_PREFLIGHT_VERSION
from chemworld.eval.work_ii_qualification import method_qualification_report_sha256
from chemworld.eval.work_ii_release import validate_clean_release_receipt

PRIVATE_AUTHORIZATION_VERSION = "chemworld-work-ii-private-execution-authorization-0.2"
PRIVATE_EXECUTION_MANIFEST_VERSION = "chemworld-work-ii-private-execution-manifest-0.2"
PRIVATE_EXECUTION_PROGRESS_VERSION = "chemworld-work-ii-private-execution-progress-0.1"
PRIVATE_TERMINAL_RECEIPT_VERSION = "chemworld-work-ii-private-terminal-receipt-0.1"
PRIVATE_ATTEMPT_RECEIPT_VERSION = "chemworld-work-ii-private-attempt-receipt-0.1"
PRIVATE_TERMINAL_STATES = frozenset({"completed", "right_censored", "failed"})
PRIVATE_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
PRIVATE_RUNTIME_ENFORCEMENT = {
    "one_shot_scientific_identity": True,
    "missing_infrastructure_only_resume": True,
    "persisted_trajectory_forbids_replacement": True,
    "reserve_full_token_cost_before_provider_launch": True,
    "accepted_terminal_cells_are_immutable": True,
}


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _cell_hash(cell: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in cell.items() if key != "cell_key_sha256"}
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite private evidence: {path}")
    write_json_atomic(path, dict(payload))


class PrivateCellStore:
    """Write-once private terminal store with missing-infrastructure-only resume."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> None:
        self.root = root.resolve()
        self.receipts = self.root / "terminal"
        self.provider_attempts = self.root / "provider_attempts"
        self.infrastructure_attempts = self.root / "infrastructure_attempts"
        cells = manifest.get("cells")
        cells = cells if isinstance(cells, list) else []
        self.cells = {
            str(cell["cell_key_sha256"]): dict(cell)
            for cell in cells
            if isinstance(cell, Mapping)
        }
        if len(self.cells) != 75 or len(self.cells) != len(cells):
            raise ValueError("private store requires 75 unique scheduled cells")
        self.manifest_sha256 = str(manifest["execution_manifest_sha256"])
        self.authorization_sha256 = str(authorization["authorization_sha256"])
        self.attempt_costs = {
            key: _attempt_cost(authorization, cell) for key, cell in self.cells.items()
        }
        self.manifest_path = self.root.parent / "execution_manifest.json"
        if not self.manifest_path.is_file() or _load_object(self.manifest_path) != dict(
            manifest
        ):
            raise ValueError("private immutable execution manifest changed")

    def record_provider_attempt_launch(self, key: str, *, attempt_id: str) -> Path:
        cell = self._cell(key)
        existing = sorted((self.provider_attempts / key).glob("*.json"))
        limit = int(cell["provider_attempt_limit"])
        if len(existing) >= limit:
            raise ValueError(f"private cell exhausted provider attempt cap: {cell['cell_id']}")
        payload: dict[str, Any] = {
            "schema_version": PRIVATE_ATTEMPT_RECEIPT_VERSION,
            "state": "provider_process_launch_authorized",
            "cell_key_sha256": key,
            "attempt_id": attempt_id,
            "attempt_index": len(existing) + 1,
            "attempt_limit": limit,
            "reserved_cost_cap_usd": self.attempt_costs[key],
            "execution_manifest_sha256": self.manifest_sha256,
            "authorization_sha256": self.authorization_sha256,
        }
        payload["attempt_sha256"] = _self_hash(payload, "attempt_sha256")
        target = self.provider_attempts / key / f"{attempt_id}.json"
        _write_once(target, payload)
        return target

    def record_infrastructure_failure(
        self,
        key: str,
        error: BaseException,
        *,
        attempt_id: str,
        log_reference: str,
        log_sha256: str,
    ) -> Path:
        self._cell(key)
        payload: dict[str, Any] = {
            "schema_version": PRIVATE_ATTEMPT_RECEIPT_VERSION,
            "state": "retryable_infrastructure_failure",
            "cell_key_sha256": key,
            "attempt_id": attempt_id,
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
            "log_reference": log_reference,
            "log_sha256": log_sha256,
            "execution_manifest_sha256": self.manifest_sha256,
            "authorization_sha256": self.authorization_sha256,
        }
        payload["attempt_sha256"] = _self_hash(payload, "attempt_sha256")
        target = self.infrastructure_attempts / key / f"{attempt_id}.json"
        _write_once(target, payload)
        return target

    def write_terminal(
        self,
        key: str,
        *,
        attempt_id: str,
        state: str,
        reason_code: str,
        result: Mapping[str, Any],
    ) -> Path:
        self._cell(key)
        if state not in PRIVATE_TERMINAL_STATES:
            raise ValueError("invalid private terminal state")
        result_payload = dict(result)
        payload: dict[str, Any] = {
            "schema_version": PRIVATE_TERMINAL_RECEIPT_VERSION,
            "cell_key_sha256": key,
            "execution_manifest_binding": {
                "path": self.manifest_path.relative_to(self.root.parent).as_posix(),
                "sha256": file_sha256(self.manifest_path),
            },
            "attempt_id": attempt_id,
            "state": state,
            "reason_code": reason_code,
            "result": result_payload,
            "result_sha256": canonical_json_sha256(result_payload),
            "execution_manifest_sha256": self.manifest_sha256,
            "authorization_sha256": self.authorization_sha256,
        }
        payload["receipt_sha256"] = _self_hash(payload, "receipt_sha256")
        target = self.receipts / f"{key}.json"
        _write_once(target, payload)
        return target

    def pending_cells(self, *, resume: bool) -> list[dict[str, Any]]:
        audit = self.audit()
        if audit["invalid_receipts"]:
            raise ValueError("private store contains invalid receipts")
        if audit["terminal_count"] and not resume:
            raise ValueError("private store contains terminal cells; use resume")
        exhausted = [
            key
            for key, cell in self.cells.items()
            if key not in set(audit["terminal_cell_key_sha256"])
            and int(audit["provider_attempt_counts_by_cell_key_sha256"].get(key, 0))
            >= int(cell["provider_attempt_limit"])
        ]
        if exhausted:
            raise ValueError("private missing cells exhausted attempt cap: " + ", ".join(exhausted))
        terminal = set(audit["terminal_cell_key_sha256"])
        return [dict(cell) for key, cell in self.cells.items() if key not in terminal]

    def audit(self) -> dict[str, Any]:
        terminal: dict[str, Mapping[str, Any]] = {}
        invalid: list[str] = []
        for path in sorted(self.receipts.glob("*.json")):
            try:
                receipt = _load_object(path)
                key = str(receipt["cell_key_sha256"])
                result = receipt.get("result")
                if (
                    receipt.get("schema_version") != PRIVATE_TERMINAL_RECEIPT_VERSION
                    or key not in self.cells
                    or path.stem != key
                    or key in terminal
                    or not self._artifact_binding_valid(
                        receipt.get("execution_manifest_binding")
                    )
                    or receipt.get("state") not in PRIVATE_TERMINAL_STATES
                    or receipt.get("execution_manifest_sha256") != self.manifest_sha256
                    or receipt.get("authorization_sha256") != self.authorization_sha256
                    or not isinstance(result, Mapping)
                    or receipt.get("result_sha256") != canonical_json_sha256(result)
                    or receipt.get("receipt_sha256") != _self_hash(receipt, "receipt_sha256")
                ):
                    raise ValueError("invalid private terminal receipt")
                terminal[key] = receipt
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        counts: dict[str, int] = {}
        observed_indices: dict[str, set[int]] = {}
        provider_attempt_ids: set[tuple[str, str]] = set()
        for path in sorted(self.provider_attempts.glob("*/*.json")):
            try:
                receipt = _load_object(path)
                key = str(receipt["cell_key_sha256"])
                index = int(receipt["attempt_index"])
                indices = observed_indices.setdefault(key, set())
                if (
                    receipt.get("schema_version") != PRIVATE_ATTEMPT_RECEIPT_VERSION
                    or receipt.get("state") != "provider_process_launch_authorized"
                    or key not in self.cells
                    or path.parent.name != key
                    or path.stem != receipt.get("attempt_id")
                    or index < 1
                    or index > int(self.cells[key]["provider_attempt_limit"])
                    or index in indices
                    or receipt.get("reserved_cost_cap_usd") != self.attempt_costs[key]
                    or receipt.get("execution_manifest_sha256") != self.manifest_sha256
                    or receipt.get("authorization_sha256") != self.authorization_sha256
                    or receipt.get("attempt_sha256") != _self_hash(receipt, "attempt_sha256")
                ):
                    raise ValueError("invalid private attempt receipt")
                indices.add(index)
                provider_attempt_ids.add((key, str(receipt["attempt_id"])))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        counts = {key: len(indices) for key, indices in observed_indices.items()}
        for path in sorted(self.infrastructure_attempts.glob("*/*.json")):
            try:
                receipt = _load_object(path)
                key = str(receipt["cell_key_sha256"])
                attempt_id = str(receipt["attempt_id"])
                if (
                    receipt.get("schema_version") != PRIVATE_ATTEMPT_RECEIPT_VERSION
                    or receipt.get("state") != "retryable_infrastructure_failure"
                    or key not in self.cells
                    or path.parent.name != key
                    or path.stem != attempt_id
                    or (key, attempt_id) not in provider_attempt_ids
                    or receipt.get("execution_manifest_sha256") != self.manifest_sha256
                    or receipt.get("authorization_sha256") != self.authorization_sha256
                    or receipt.get("attempt_sha256") != _self_hash(receipt, "attempt_sha256")
                    or not self._artifact_binding_valid(
                        {
                            "path": receipt.get("log_reference"),
                            "sha256": receipt.get("log_sha256"),
                        }
                    )
                ):
                    raise ValueError("invalid private infrastructure receipt")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        for key, terminal_receipt in terminal.items():
            if (key, str(terminal_receipt.get("attempt_id", ""))) not in provider_attempt_ids:
                invalid.append((self.receipts / f"{key}.json").as_posix())
                continue
            result = terminal_receipt.get("result")
            result = result if isinstance(result, Mapping) else {}
            state = terminal_receipt.get("state")
            trajectory = result.get("trajectory")
            if state == "right_censored":
                if not self._artifact_binding_valid(trajectory):
                    invalid.append((self.receipts / f"{key}.json").as_posix())
            elif (
                not self._artifact_binding_valid(result.get("summary"))
                or not self._artifact_binding_valid(result.get("child_report"))
                or (
                    state == "completed"
                    and not self._artifact_binding_valid(trajectory)
                )
                or (state == "failed" and trajectory is not None)
            ):
                invalid.append((self.receipts / f"{key}.json").as_posix())
        terminal_keys = set(terminal)
        states = {
            state: sum(receipt.get("state") == state for receipt in terminal.values())
            for state in sorted(PRIVATE_TERMINAL_STATES)
        }
        report: dict[str, Any] = {
            "expected_cell_count": len(self.cells),
            "terminal_count": len(terminal_keys),
            "terminal_cell_key_sha256": sorted(terminal_keys),
            "missing_cell_key_sha256": sorted(set(self.cells) - terminal_keys),
            "state_counts": states,
            "provider_attempt_count": sum(counts.values()),
            "provider_attempt_counts_by_cell_key_sha256": counts,
            "invalid_receipts": invalid,
            "complete": terminal_keys == set(self.cells) and not invalid,
        }
        report["audit_sha256"] = canonical_json_sha256(report)
        return report

    def _cell(self, key: str) -> dict[str, Any]:
        try:
            return self.cells[key]
        except KeyError as error:
            raise ValueError(f"unknown private cell key: {key}") from error

    def _artifact_binding_valid(self, binding: object) -> bool:
        if not isinstance(binding, Mapping):
            return False
        relative = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            return False
        path = (self.root.parent / relative).resolve()
        return (
            path.is_relative_to(self.root.parent)
            and path.is_file()
            and file_sha256(path) == digest
        )


def validate_private_execution_preflight(root: Path, preflight: Mapping[str, Any]) -> list[str]:
    """Validate the outcome-free schedule without requiring the secret seal again."""

    errors: list[str] = []
    if preflight.get("schema_version") != WORK_II_PRIVATE_PREFLIGHT_VERSION:
        errors.append("unexpected private execution preflight schema")
    if preflight.get("preflight_sha256") != _self_hash(preflight, "preflight_sha256"):
        errors.append("private execution preflight self-hash mismatch")
    if (
        preflight.get("status") != "passed_private_execution_blocked"
        or preflight.get("private_execution_allowed") is not False
        or preflight.get("formal_result") is not False
        or preflight.get("private_confirmation_result") is not False
        or preflight.get("provider_calls_executed") != 0
    ):
        errors.append("private execution preflight crossed its blocked boundary")
    cells = preflight.get("cells")
    cells = cells if isinstance(cells, list) else []
    valid_cells = [cell for cell in cells if isinstance(cell, Mapping)]
    keys = [str(cell.get("cell_key_sha256", "")) for cell in valid_cells]
    clusters = {str(cell.get("world_cluster_id", "")) for cell in valid_cells}
    tasks = {str(cell.get("task_id", "")) for cell in valid_cells}
    if (
        len(valid_cells) != 75
        or len(set(keys)) != 75
        or len(clusters) != 25
        or len(tasks) != 5
        or any(_cell_hash(cell) != cell.get("cell_key_sha256") for cell in valid_cells)
    ):
        errors.append("private execution preflight cell roster is invalid")
    if [cell.get("schedule_index") for cell in valid_cells] != list(range(1, 76)):
        errors.append("private execution preflight schedule order is invalid")
    cluster_cells: dict[str, list[Mapping[str, Any]]] = {}
    task_counts: dict[str, int] = {}
    for cell in valid_cells:
        cluster_cells.setdefault(str(cell.get("world_cluster_id", "")), []).append(cell)
        task_id = str(cell.get("task_id", ""))
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    for members in cluster_cells.values():
        signatures = {
            (str(cell.get("task_id", "")), int(cell.get("world_seed", -1)))
            for cell in members
        }
        if (
            len(members) != 3
            or len(signatures) != 1
            or {str(cell.get("prior_arm", "")) for cell in members} != set(PRIVATE_ARMS)
        ):
            errors.append("private execution preflight world triplets are invalid")
            break
    if sorted(task_counts.values()) != [15] * 5:
        errors.append("private execution preflight task allocation is invalid")
    exact_cell_contract = {
        "world_split": "private_confirmation",
        "complete_experiment_count": 8,
        "belief_checkpoint_count": 5,
        "held_out_query_count_per_snapshot": 4,
        "held_out_query_metric_count_per_snapshot": 4,
        "provider_session_limit": 1,
        "provider_attempt_limit": 2,
        "provider_repeat": 1,
        "participant_final_recommendation_count": 1,
        "blind_validation_target_count": 2,
        "blind_replicates_per_target": 3,
        "blind_validation_execution_count": 6,
        "terminal_states": ["completed", "right_censored", "failed"],
    }
    if any(
        any(cell.get(field) != value for field, value in exact_cell_contract.items())
        for cell in valid_cells
    ):
        errors.append("private execution preflight per-cell contract is invalid")
    expected = preflight.get("expected_counts")
    expected = expected if isinstance(expected, Mapping) else {}
    observed = {
        "tasks": len(tasks),
        "independent_task_world_clusters": len(clusters),
        "participant_cells": len(valid_cells),
        "complete_experiments": sum(
            int(cell.get("complete_experiment_count", 0)) for cell in valid_cells
        ),
        "belief_checkpoints": sum(
            int(cell.get("belief_checkpoint_count", 0)) for cell in valid_cells
        ),
        "provider_sessions": sum(
            int(cell.get("provider_session_limit", 0)) for cell in valid_cells
        ),
        "provider_attempts_initial_planned": len(valid_cells),
        "provider_attempts_hard_cap": sum(
            int(cell.get("provider_attempt_limit", 0)) for cell in valid_cells
        ),
        "evaluator_truth_executions": len(clusters) * 4,
        "blind_validation_executions": sum(
            int(cell.get("blind_validation_execution_count", 0)) for cell in valid_cells
        ),
    }
    if dict(expected) != observed or observed["complete_experiments"] != 600:
        errors.append("private execution preflight denominators are not exact")
    for cell in valid_cells:
        relative = cell.get("campaign_config_path")
        digest = cell.get("campaign_config_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            errors.append("private cell campaign binding is incomplete")
            continue
        path = (root.resolve() / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append("private cell campaign binding escapes the repository")
            continue
        if not path.is_file() or file_sha256(path) != digest:
            errors.append(f"private cell campaign binding is stale: {cell.get('cell_id')}")
    return errors


def _provider_and_cost_contracts(
    root: Path,
    preflight: Mapping[str, Any],
    *,
    cache_hit_input_usd_per_million: float,
    cache_miss_input_usd_per_million: float,
    output_usd_per_million: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float]:
    cells = [dict(cell) for cell in preflight["cells"]]
    paths = sorted({str(cell["campaign_config_path"]) for cell in cells})
    providers: list[dict[str, Any]] = []
    task_contracts: list[dict[str, Any]] = []
    initial_total = 0.0
    hard_total = 0.0
    for relative in paths:
        config = _load_object(root / relative)
        provider = config.get("provider")
        provider = provider if isinstance(provider, Mapping) else {}
        providers.append(
            {
                key: provider.get(key)
                for key in ("id", "name", "base_url", "wire_api", "model", "reasoning_effort")
            }
        )
        resources = config.get("method_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        input_tokens = int(resources.get("input_token_limit", -1))
        uncached_tokens = int(resources.get("uncached_input_token_limit", -1))
        output_tokens = int(resources.get("output_token_limit", -1))
        if min(input_tokens, uncached_tokens, output_tokens) < 0 or uncached_tokens > input_tokens:
            raise ValueError(f"invalid private token caps: {relative}")
        per_attempt = (
            (input_tokens - uncached_tokens) * cache_hit_input_usd_per_million
            + uncached_tokens * cache_miss_input_usd_per_million
            + output_tokens * output_usd_per_million
        ) / 1_000_000.0
        task_cells = [cell for cell in cells if cell["campaign_config_path"] == relative]
        initial = len(task_cells)
        hard = sum(int(cell["provider_attempt_limit"]) for cell in task_cells)
        initial_total += per_attempt * initial
        hard_total += per_attempt * hard
        task_contracts.append(
            {
                "task_id": config.get("task_id"),
                "campaign_config_path": relative,
                "campaign_config_sha256": file_sha256(root / relative),
                "participant_cell_count": initial,
                "provider_attempt_hard_cap": hard,
                "per_attempt_token_caps": {
                    "input_tokens": input_tokens,
                    "uncached_input_tokens": uncached_tokens,
                    "output_tokens": output_tokens,
                },
                "per_attempt_cost_cap_usd": round(per_attempt, 12),
            }
        )
    if not providers or any(provider != providers[0] for provider in providers[1:]):
        raise ValueError("private matrix provider contract is absent or inconsistent")
    return providers[0], task_contracts, round(initial_total, 12), round(hard_total, 12)


def build_private_execution_authorization(
    root: Path,
    preflight: Mapping[str, Any],
    clean_release_receipt_path: Path,
    *,
    approved_at: str,
    pricing_source: str,
    pricing_observed_at: str,
    cache_hit_input_usd_per_million: float,
    cache_miss_input_usd_per_million: float,
    output_usd_per_million: float,
    private_currency_ceiling_usd: float,
    provider_contract_confirmed_by_user: bool,
    credential_rotation_confirmed_by_user: bool,
    private_one_shot_execution_confirmed_by_user: bool,
) -> dict[str, Any]:
    """Build one write-once, credential-free authorization record."""

    root = root.resolve()
    errors = validate_private_execution_preflight(root, preflight)
    if errors:
        raise ValueError("private preflight is invalid: " + "; ".join(errors))
    if not all(
        (
            provider_contract_confirmed_by_user,
            credential_rotation_confirmed_by_user,
            private_one_shot_execution_confirmed_by_user,
        )
    ):
        raise ValueError("private authorization lacks explicit user confirmations")
    if not approved_at or not pricing_source or not pricing_observed_at:
        raise ValueError("private authorization metadata is incomplete")
    hit = _nonnegative_finite(cache_hit_input_usd_per_million, "cache-hit price")
    miss = _nonnegative_finite(cache_miss_input_usd_per_million, "cache-miss price")
    output = _nonnegative_finite(output_usd_per_million, "output price")
    if hit == miss == output == 0.0:
        raise ValueError("private authorization pricing cannot be all zero")
    ceiling = _positive_finite(private_currency_ceiling_usd, "private currency ceiling")
    release_path = clean_release_receipt_path.resolve()
    release = _load_object(release_path)
    release_errors = validate_clean_release_receipt(release, root=root)
    if release_errors:
        raise ValueError("private clean-release receipt is invalid: " + "; ".join(release_errors))
    provider, tasks, initial_cost, hard_cost = _provider_and_cost_contracts(
        root,
        preflight,
        cache_hit_input_usd_per_million=hit,
        cache_miss_input_usd_per_million=miss,
        output_usd_per_million=output,
    )
    if ceiling < hard_cost:
        raise ValueError("private currency ceiling is below the all-attempt cost cap")
    authorization: dict[str, Any] = {
        "schema_version": PRIVATE_AUTHORIZATION_VERSION,
        "status": "authorized_private_execution_only",
        "formal_result": False,
        "private_confirmation_result": False,
        "private_execution_allowed": True,
        "base_private_preflight_sha256": preflight["preflight_sha256"],
        "approved_at": approved_at,
        "provider_contract": provider,
        "provider_contract_confirmed_by_user": True,
        "credential_rotation_confirmed_by_user": True,
        "private_one_shot_execution_confirmed_by_user": True,
        "clean_release_receipt_binding": {
            "path": release_path.relative_to(root).as_posix(),
            "file_sha256": file_sha256(release_path),
            "receipt_sha256": release.get("receipt_sha256"),
        },
        "pricing": {
            "source": pricing_source,
            "observed_at": pricing_observed_at,
            "unit": "usd_per_million_tokens",
            "cache_hit_input": hit,
            "cache_miss_input": miss,
            "output": output,
        },
        "task_attempt_contracts": tasks,
        "initial_schedule": {"provider_process_attempts": 75, "cost_cap_usd": initial_cost},
        "all_infrastructure_resumes": {
            "provider_process_attempts": 150,
            "cost_cap_usd": hard_cost,
        },
        "private_currency_ceiling_usd": ceiling,
        "runtime_enforcement": dict(PRIVATE_RUNTIME_ENFORCEMENT),
    }
    authorization["authorization_sha256"] = _self_hash(
        authorization, "authorization_sha256"
    )
    return authorization


def validate_private_execution_authorization(
    root: Path,
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> list[str]:
    errors = validate_private_execution_preflight(root.resolve(), preflight)
    if authorization.get("schema_version") != PRIVATE_AUTHORIZATION_VERSION:
        errors.append("unexpected private authorization schema")
    if authorization.get("authorization_sha256") != _self_hash(
        authorization, "authorization_sha256"
    ):
        errors.append("private authorization self-hash mismatch")
    if (
        authorization.get("status") != "authorized_private_execution_only"
        or authorization.get("formal_result") is not False
        or authorization.get("private_confirmation_result") is not False
        or authorization.get("private_execution_allowed") is not True
        or authorization.get("base_private_preflight_sha256")
        != preflight.get("preflight_sha256")
    ):
        errors.append("private authorization state or preflight binding is invalid")
    if any(
        authorization.get(field) is not True
        for field in (
            "provider_contract_confirmed_by_user",
            "credential_rotation_confirmed_by_user",
            "private_one_shot_execution_confirmed_by_user",
        )
    ):
        errors.append("private authorization lacks explicit user confirmations")
    if not authorization.get("approved_at"):
        errors.append("private authorization approval metadata is incomplete")
    if authorization.get("runtime_enforcement") != PRIVATE_RUNTIME_ENFORCEMENT:
        errors.append("private authorization runtime enforcement drifted")
    pricing = authorization.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    try:
        hit = _nonnegative_finite(pricing.get("cache_hit_input"), "cache-hit price")
        miss = _nonnegative_finite(pricing.get("cache_miss_input"), "cache-miss price")
        output = _nonnegative_finite(pricing.get("output"), "output price")
        if hit == miss == output == 0.0:
            raise ValueError("private prices cannot all be zero")
        provider, tasks, initial_cost, hard_cost = _provider_and_cost_contracts(
            root.resolve(),
            preflight,
            cache_hit_input_usd_per_million=hit,
            cache_miss_input_usd_per_million=miss,
            output_usd_per_million=output,
        )
        if (
            pricing.get("unit") != "usd_per_million_tokens"
            or not pricing.get("source")
            or not pricing.get("observed_at")
            or authorization.get("provider_contract") != provider
            or authorization.get("task_attempt_contracts") != tasks
            or authorization.get("initial_schedule")
            != {"provider_process_attempts": 75, "cost_cap_usd": initial_cost}
            or authorization.get("all_infrastructure_resumes")
            != {"provider_process_attempts": 150, "cost_cap_usd": hard_cost}
            or _positive_finite(
                authorization.get("private_currency_ceiling_usd"),
                "private currency ceiling",
            )
            < hard_cost
        ):
            errors.append("private authorization provider or cost contract is invalid")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("private authorization provider or cost contract is invalid")
    binding = authorization.get("clean_release_receipt_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    path = (root.resolve() / str(binding.get("path", ""))).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or binding.get("file_sha256") != file_sha256(path)
    ):
        errors.append("private clean-release receipt binding is stale")
    else:
        release = _load_object(path)
        if (
            binding.get("receipt_sha256") != release.get("receipt_sha256")
            or validate_clean_release_receipt(release, root=root.resolve())
        ):
            errors.append("private clean-release receipt is no longer valid")
    return errors


def build_private_execution_manifest(
    preflight: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_MANIFEST_VERSION,
        "status": "passed_private_execution_authorized",
        "formal_result": False,
        "private_confirmation_result": False,
        "private_execution_allowed": True,
        "blocking_requirements": [],
        "base_private_preflight_sha256": preflight["preflight_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
        "expected_counts": preflight["expected_counts"],
        "participant_runtime": {
            "preflight_world_split_label": "private_confirmation",
            "engine_world_split": "private-eval",
            "campaign_config_materialization": "copy_bound_source_then_replace_world_split_only",
            "runtime_configs_remain_under_ignored_runs_private": True,
        },
        "private_seal_commitment_sha256": preflight["private_seal_commitment_sha256"],
        "private_identity_schedule_sha256": preflight["private_identity_schedule_sha256"],
        "cells": preflight["cells"],
    }
    manifest["execution_manifest_sha256"] = _self_hash(
        manifest, "execution_manifest_sha256"
    )
    return manifest


def validate_private_execution_manifest(
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != PRIVATE_EXECUTION_MANIFEST_VERSION:
        errors.append("unexpected private execution manifest schema")
    if manifest.get("execution_manifest_sha256") != _self_hash(
        manifest, "execution_manifest_sha256"
    ):
        errors.append("private execution manifest self-hash mismatch")
    if (
        manifest.get("status") != "passed_private_execution_authorized"
        or manifest.get("private_execution_allowed") is not True
        or manifest.get("formal_result") is not False
        or manifest.get("private_confirmation_result") is not False
        or manifest.get("blocking_requirements") != []
        or manifest.get("base_private_preflight_sha256") != preflight.get("preflight_sha256")
        or manifest.get("authorization_sha256") != authorization.get("authorization_sha256")
        or manifest.get("cells") != preflight.get("cells")
        or manifest.get("expected_counts") != preflight.get("expected_counts")
        or manifest.get("participant_runtime")
        != {
            "preflight_world_split_label": "private_confirmation",
            "engine_world_split": "private-eval",
            "campaign_config_materialization": "copy_bound_source_then_replace_world_split_only",
            "runtime_configs_remain_under_ignored_runs_private": True,
        }
    ):
        errors.append("private execution manifest bindings are invalid")
    return errors


def _emit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(rendered, flush=True)


def _attempt_cost(authorization: Mapping[str, Any], cell: Mapping[str, Any]) -> float:
    matches = [
        row
        for row in authorization.get("task_attempt_contracts", [])
        if isinstance(row, Mapping)
        and row.get("campaign_config_path") == cell.get("campaign_config_path")
    ]
    if len(matches) != 1:
        raise ValueError("private cell lacks one cost contract")
    return float(matches[0]["per_attempt_cost_cap_usd"])


def _materialize_private_campaign_config(
    root: Path,
    output_root: Path,
    cell: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_relative = str(cell["campaign_config_path"])
    source_path = (root / source_relative).resolve()
    if not source_path.is_relative_to(root) or not source_path.is_file():
        raise ValueError("private source campaign config is outside the repository")
    if file_sha256(source_path) != cell.get("campaign_config_sha256"):
        raise ValueError("private source campaign config differs from its manifest")
    source = _load_object(source_path)
    if source.get("task_id") != cell.get("task_id") or source.get("world_split") != "public-test":
        raise ValueError("private source campaign config has an unexpected task or split")
    runtime = deepcopy(source)
    runtime["world_split"] = "private-eval"
    runtime_path = output_root / "runtime_configs" / f"{cell['task_id']}.json"
    if runtime_path.is_file():
        if _load_object(runtime_path) != runtime:
            raise ValueError("private runtime campaign config is not immutable")
    else:
        write_json_atomic(runtime_path, runtime)
    return runtime_path, {
        "source_path": source_relative,
        "source_file_sha256": file_sha256(source_path),
        "runtime_path": runtime_path.relative_to(output_root).as_posix(),
        "runtime_file_sha256": file_sha256(runtime_path),
        "runtime_canonical_json_sha256": canonical_json_sha256(runtime),
        "world_split": "private-eval",
    }


def _private_cell_runtime_argument(cell: Mapping[str, Any], field: str) -> str:
    path = Path(str(cell[field]))
    if path.is_absolute():
        raise ValueError(f"private {field} must be relative to runs/private")
    rendered = path.as_posix()
    if not rendered or rendered == "." or rendered.startswith("../"):
        raise ValueError(f"private {field} escapes runs/private")
    return rendered


def execute_private_manifest(
    root: Path,
    *,
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output_root: Path,
    progress_path: Path,
    resume: bool,
    cell_runner: Path,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Execute or infrastructure-resume the exact 75-cell private schedule."""

    root = root.resolve()
    authorization_errors = validate_private_execution_authorization(
        root, authorization, preflight
    )
    if authorization_errors:
        raise ValueError("private authorization failed: " + "; ".join(authorization_errors))
    manifest = build_private_execution_manifest(preflight, authorization)
    manifest_errors = validate_private_execution_manifest(manifest, preflight, authorization)
    if manifest_errors:
        raise ValueError("private execution manifest failed: " + "; ".join(manifest_errors))
    private_root = (root / "runs" / "private").resolve()
    output_root = output_root.resolve()
    progress_path = progress_path.resolve()
    if (
        not output_root.is_relative_to(private_root)
        or not progress_path.is_relative_to(private_root)
    ):
        raise ValueError("private execution artifacts must remain under runs/private")
    expected_runner = (root / "scripts/run_work_ii_campaign_pilot.py").resolve()
    if cell_runner.resolve() != expected_runner or not expected_runner.is_file():
        raise ValueError("private execution child runner differs from the frozen entry point")
    if output_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite private output: {output_root}")
    if not output_root.exists() and resume:
        raise FileNotFoundError("private resume requires an existing output root")
    output_root.mkdir(parents=True, exist_ok=resume)
    manifest_path = output_root / "execution_manifest.json"
    authorization_path = output_root / "execution_authorization.json"
    for path, payload in ((manifest_path, manifest), (authorization_path, authorization)):
        if path.is_file():
            if _load_object(path) != dict(payload):
                raise ValueError(f"private immutable artifact changed: {path.name}")
        else:
            write_json_atomic(path, dict(payload))
    store = PrivateCellStore(output_root / "store", manifest, authorization)
    pending = store.pending_cells(resume=resume)
    pending_keys = {str(cell["cell_key_sha256"]) for cell in pending}
    clusters: list[list[dict[str, Any]]] = []
    for raw in manifest["cells"]:
        cell = dict(raw)
        if cell["cell_key_sha256"] not in pending_keys:
            continue
        if not clusters or clusters[-1][0]["world_cluster_id"] != cell["world_cluster_id"]:
            clusters.append([])
        clusters[-1].append(cell)
    infrastructure_failures = 0
    _emit(
        progress_path,
        {
            "event": "private_matrix_started" if not resume else "private_matrix_resumed",
            "expected_cells": 75,
            "expected_complete_experiments": 600,
            "pending_cells": len(pending),
        },
    )
    for cluster_index, cells in enumerate(clusters, start=1):
        processes: list[dict[str, Any]] = []
        for cell in cells:
            key = str(cell["cell_key_sha256"])
            audit = store.audit()
            counts = dict(audit["provider_attempt_counts_by_cell_key_sha256"])
            proposed_cost = sum(
                int(counts.get(str(candidate["cell_key_sha256"]), 0))
                * _attempt_cost(authorization, candidate)
                for candidate in manifest["cells"]
            ) + _attempt_cost(authorization, cell)
            if proposed_cost > float(authorization["private_currency_ceiling_usd"]):
                raise ValueError("private currency ceiling would be exceeded before launch")
            attempt_id = uuid4().hex
            store.record_provider_attempt_launch(key, attempt_id=attempt_id)
            attempt_root = output_root / "attempts" / key / attempt_id
            log_path = output_root / "logs" / key / f"{attempt_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            child_progress = output_root / "progress" / key / f"{attempt_id}.jsonl"
            runtime_config_path, runtime_config_binding = _materialize_private_campaign_config(
                root, output_root, cell
            )
            command = [
                sys.executable,
                str(cell_runner),
                "--config",
                str(runtime_config_path),
                "--output",
                str(attempt_root),
                "--progress-file",
                str(child_progress),
                "--world-seed",
                str(cell["world_seed"]),
                "--prior-arm",
                str(cell["prior_arm"]),
            ]
            optional_runtime_fields = {
                "private_scenario_path": "--private-scenario",
                "private_composition_path": "--private-composition",
            }
            for field, flag in optional_runtime_fields.items():
                if cell.get(field) is not None:
                    relative_private_artifact = _private_cell_runtime_argument(cell, field)
                    private_artifact_path = (
                        output_root / relative_private_artifact
                    ).resolve()
                    if not private_artifact_path.is_relative_to(output_root):
                        raise ValueError(f"private {field} escapes its output root")
                    command.extend(
                        [
                            flag,
                            str(private_artifact_path),
                        ]
                    )
            log_handle = log_path.open("w", encoding="utf-8")
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                process = popen_factory(
                    command,
                    cwd=root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **kwargs,
                )
            except OSError as error:
                log_handle.write(f"private child launch failed: {type(error).__name__}\n")
                log_handle.close()
                store.record_infrastructure_failure(
                    key,
                    error,
                    attempt_id=attempt_id,
                    log_reference=log_path.relative_to(output_root).as_posix(),
                    log_sha256=file_sha256(log_path),
                )
                infrastructure_failures += 1
                continue
            processes.append(
                {
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                    "attempt_root": attempt_root,
                    "attempt_id": attempt_id,
                    "runtime_config_binding": runtime_config_binding,
                    "cell": cell,
                }
            )
        started = time.monotonic()
        next_heartbeat = started + 30.0
        while any(item["process"].poll() is None for item in processes):
            now = time.monotonic()
            if now >= next_heartbeat:
                _emit(
                    progress_path,
                    {
                        "event": "private_world_triplet_heartbeat",
                        "cluster_index": cluster_index,
                        "cluster_count": 25,
                        "active_processes": sum(
                            item["process"].poll() is None for item in processes
                        ),
                        "elapsed_s": round(now - started, 3),
                    },
                )
                next_heartbeat = now + 30.0
            time.sleep(0.1)
        for item in processes:
            return_code = int(item["process"].wait())
            item["log_handle"].close()
            cell = item["cell"]
            key = str(cell["cell_key_sha256"])
            attempt_id = str(item["attempt_id"])
            attempt_root = item["attempt_root"]
            summary_path = attempt_root / "summary.json"
            trajectory_path = attempt_root / "trajectory.jsonl"
            report_path = attempt_root / "report.json"
            try:
                summary = _load_object(summary_path)
                child_report = _load_object(report_path)
                if (
                    summary.get("arm") != cell["prior_arm"]
                    or int(summary.get("analysis", {}).get("complete_experiment_count", -1))
                    != int(cell["complete_experiment_count"])
                    or child_report.get("config_file_sha256")
                    != item["runtime_config_binding"]["runtime_file_sha256"]
                    or child_report.get("world_seed") != cell["world_seed"]
                    or child_report.get("cell_count") != 1
                    or child_report.get("results") != [summary]
                    or child_report.get("report_sha256")
                    != method_qualification_report_sha256(child_report)
                ):
                    raise ValueError("private child result differs from its scheduled cell")
                completed = (
                    summary.get("completed") is True
                    and summary.get("qualification", {}).get("passed") is True
                    and summary.get("exact_replay", {}).get("verified") is True
                )
                state = (
                    "completed"
                    if completed
                    else "right_censored"
                    if trajectory_path.is_file()
                    else "failed"
                )
                reason = (
                    "private_scientific_completed_qualified_campaign"
                    if state == "completed"
                    else "private_method_right_censored_retained"
                    if state == "right_censored"
                    else "private_method_failed_before_trajectory"
                )
                result = {
                    "return_code": return_code,
                    "runtime_campaign_config": item["runtime_config_binding"],
                    "child_report": {
                        "path": report_path.relative_to(output_root).as_posix(),
                        "sha256": file_sha256(report_path),
                        "report_sha256": child_report["report_sha256"],
                    },
                    "summary": {
                        "path": summary_path.relative_to(output_root).as_posix(),
                        "sha256": file_sha256(summary_path),
                    },
                    "trajectory": (
                        {
                            "path": trajectory_path.relative_to(output_root).as_posix(),
                            "sha256": file_sha256(trajectory_path),
                        }
                        if trajectory_path.is_file()
                        else None
                    ),
                    "analysis": summary.get("analysis"),
                    "method_resources": summary.get("method_resources"),
                    "exact_replay": summary.get("exact_replay"),
                    "qualification": summary.get("qualification"),
                }
                store.write_terminal(
                    key,
                    attempt_id=attempt_id,
                    state=state,
                    reason_code=reason,
                    result=result,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                if trajectory_path.is_file() and trajectory_path.stat().st_size > 0:
                    store.write_terminal(
                        key,
                        attempt_id=attempt_id,
                        state="right_censored",
                        reason_code="private_unfinalized_child_after_trajectory_evidence",
                        result={
                            "summary_validation_error_type": type(error).__name__,
                            "trajectory": {
                                "path": trajectory_path.relative_to(output_root).as_posix(),
                                "sha256": file_sha256(trajectory_path),
                            },
                        },
                    )
                else:
                    store.record_infrastructure_failure(
                        key,
                        error,
                        attempt_id=attempt_id,
                        log_reference=item["log_path"].relative_to(output_root).as_posix(),
                        log_sha256=file_sha256(item["log_path"]),
                    )
                    infrastructure_failures += 1
        if infrastructure_failures:
            break
    audit = store.audit()
    report: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_PROGRESS_VERSION,
        "status": (
            "all_private_cells_terminal"
            if audit["complete"]
            else "private_infrastructure_incomplete_missing_only_resume_required"
        ),
        "formal_result": False,
        "private_confirmation_result": False,
        "expected_cell_count": 75,
        "expected_complete_experiment_count": 600,
        "terminal_count": audit["terminal_count"],
        "state_counts": audit["state_counts"],
        "missing_cell_count": len(audit["missing_cell_key_sha256"]),
        "provider_attempt_count": audit["provider_attempt_count"],
        "provider_attempt_hard_cap": 150,
        "infrastructure_failure_count_this_attempt": infrastructure_failures,
        "store_audit_sha256": audit["audit_sha256"],
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "authorization_sha256": authorization["authorization_sha256"],
    }
    report["progress_sha256"] = _self_hash(report, "progress_sha256")
    write_json_atomic(output_root / "store_audit.json", audit)
    write_json_atomic(output_root / "execution_progress.json", report)
    _emit(progress_path, {"event": "private_matrix_attempt_finished", **report})
    return report


__all__ = [
    "PRIVATE_AUTHORIZATION_VERSION",
    "PRIVATE_EXECUTION_MANIFEST_VERSION",
    "PRIVATE_EXECUTION_PROGRESS_VERSION",
    "build_private_execution_authorization",
    "build_private_execution_manifest",
    "execute_private_manifest",
    "validate_private_execution_authorization",
    "validate_private_execution_manifest",
    "validate_private_execution_preflight",
]
