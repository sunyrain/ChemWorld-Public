"""First-paper cross-world infrastructure qualification.

The qualification is deliberately deterministic.  Registered task/world
configurations are the independent units; recipes, operations, negative
probes, and replay events are repeated checks within those units.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle
from chemworld.agents.task_recipes import (
    TASK_RECIPE_SPACE_VERSION,
    task_recipe_coordinate_schema,
    task_recipe_dimension,
    task_recipe_from_unit_vector,
)
from chemworld.campaign_resources import derive_campaign_resource_delta
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.eval.report_io import git_output as _git
from chemworld.eval.verify import verify_records
from chemworld.foundation.public_leakage import audit_public_payload
from chemworld.tasks import TASK_CONTRACT_VERSION, TaskSpec, list_tasks
from chemworld.world.recipes import compile_recipe

PROTOCOL_SCHEMA_VERSION = "chemworld-first-paper-infrastructure-qualification-protocol-0.1"
REPORT_SCHEMA_VERSION = "chemworld-first-paper-infrastructure-qualification-report-0.1"
MANIFEST_SCHEMA_VERSION = "chemworld-first-paper-infrastructure-qualification-manifest-0.1"
PROPERTY_IDS = (
    "units_and_action_domains",
    "applicable_conservation",
    "transaction_atomicity",
    "resource_reconciliation",
    "lifecycle_closure",
    "public_private_separation",
    "exact_replay",
)


class QualificationProtocolError(ValueError):
    """Raised when a frozen protocol no longer binds the executable surface."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QualificationProtocolError("protocol root must be an object")
    return payload


def validate_protocol_bindings(
    protocol: dict[str, Any],
    *,
    repository_root: str | Path,
    require_clean: bool,
) -> dict[str, Any]:
    """Fail closed if the frozen task, source, owner, or write-set binding drifted."""

    root = Path(repository_root).resolve()
    errors: list[str] = []
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        errors.append("protocol schema version drifted")
    if protocol.get("status") != "frozen_before_formal_execution":
        errors.append("protocol is not frozen before formal execution")
    if protocol.get("owner") != "Yijun":
        errors.append("protocol owner is not Yijun")

    binding = protocol.get("source_binding")
    if not isinstance(binding, dict):
        errors.append("source_binding is missing")
        binding = {}
    if binding.get("task_contract_version") != TASK_CONTRACT_VERSION:
        errors.append("task contract version drifted")
    if binding.get("task_recipe_space_version") != TASK_RECIPE_SPACE_VERSION:
        errors.append("task recipe space version drifted")

    raw_bound_tasks = binding.get("tasks")
    if not isinstance(raw_bound_tasks, list):
        errors.append("source_binding.tasks must be a list")
        raw_bound_tasks = []
    bound_tasks = {
        str(row.get("task_id")): row
        for row in raw_bound_tasks
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    }
    actual_tasks = {task.task_id: task for task in list_tasks()}
    if set(bound_tasks) != set(actual_tasks):
        errors.append("registered task IDs drifted")
    for task_id, task in actual_tasks.items():
        row = bound_tasks.get(task_id, {})
        if row.get("contract_sha256") != task.contract_hash:
            errors.append(f"task contract hash drifted: {task_id}")
        if row.get("world_seeds") != list(task.seeds):
            errors.append(f"registered world seeds drifted: {task_id}")
    unit_count = sum(
        len(row.get("world_seeds", []))
        for row in raw_bound_tasks
        if isinstance(row, dict) and isinstance(row.get("world_seeds"), list)
    )
    if binding.get("registered_task_count") != len(actual_tasks):
        errors.append("registered task count drifted")
    if binding.get("registered_task_world_unit_count") != unit_count:
        errors.append("registered task/world unit count drifted")

    coordination = protocol.get("coordination")
    if not isinstance(coordination, dict):
        errors.append("coordination binding is missing")
        coordination = {}
    raw_write_set = coordination.get("write_set")
    write_set = (
        {str(path) for path in raw_write_set if isinstance(path, str)}
        if isinstance(raw_write_set, list)
        else set()
    )
    if not write_set:
        errors.append("write set is empty")

    source_commit = str(binding.get("source_commit", ""))
    try:
        if not source_commit:
            raise subprocess.CalledProcessError(1, ["git", "merge-base"])
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        errors.append("HEAD does not descend from the frozen source commit")
    else:
        changed_paths = {
            line
            for line in _git(root, "diff", "--name-only", f"{source_commit}..HEAD").splitlines()
            if line
        }
        outside_write_set = sorted(changed_paths - write_set)
        if outside_write_set:
            errors.append(
                "tracked source drift outside the frozen write set: " + ", ".join(outside_write_set)
            )
    if require_clean:
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            errors.append("formal execution requires a clean worktree")

    if errors:
        raise QualificationProtocolError("; ".join(errors))
    return {
        "status": "passed",
        "owner": protocol["owner"],
        "task_count": len(actual_tasks),
        "task_world_unit_count": unit_count,
        "source_commit": source_commit,
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "write_set": sorted(write_set),
    }


def recipe_cases(task: TaskSpec) -> list[dict[str, Any]]:
    """Materialize the frozen midpoint, true-boundary, and category cases."""

    task_info = task.to_dict()
    dimension = task_recipe_dimension(task_info)
    schema = task_recipe_coordinate_schema(task_info)
    if len(schema) != dimension:
        raise QualificationProtocolError(
            f"recipe coordinate schema mismatch for {task.task_id}: {len(schema)} != {dimension}"
        )

    cases: list[dict[str, Any]] = []

    def add_case(case_id: str, kind: str, vector: np.ndarray, **metadata: Any) -> None:
        recipe = task_recipe_from_unit_vector(task_info, vector)
        compiled = compile_recipe(recipe, task_info=task_info)
        cases.append(
            {
                "case_id": case_id,
                "kind": kind,
                "metadata": metadata,
                "vector": [float(value) for value in vector],
                "vector_sha256": _sha256_value([float(value) for value in vector]),
                "compiled_actions": compiled,
                "compiled_actions_sha256": _sha256_value(compiled),
            }
        )

    midpoint = np.full(dimension, 0.5, dtype=float)
    add_case("midpoint", "midpoint", midpoint)
    for coordinate in range(dimension):
        for label, value in (("low", 0.0), ("high", 1.0)):
            vector = midpoint.copy()
            vector[coordinate] = value
            add_case(
                f"coordinate-{coordinate}-{label}",
                "continuous_boundary",
                vector,
                coordinate=coordinate,
                boundary=label,
            )
    for entry in schema:
        if entry.get("kind") != "categorical":
            continue
        coordinate = int(entry["coordinate"])
        category_count = int(entry["category_count"])
        for category in range(category_count):
            vector = midpoint.copy()
            vector[coordinate] = (category + 0.5) / category_count
            add_case(
                f"coordinate-{coordinate}-category-{category}",
                "categorical_coverage",
                vector,
                coordinate=coordinate,
                category=category,
                category_count=category_count,
            )
    return sorted(cases, key=lambda row: str(row["case_id"]))


def _physical_snapshot(env: Any) -> dict[str, Any]:
    state = env.unwrapped._state.to_dict(include_hidden=True)
    state.pop("ledger", None)
    state.pop("process", None)
    return state


def _rng_snapshot(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    return {
        "rng": copy.deepcopy(base._rng.bit_generator.state),
        "observation_occurrences": copy.deepcopy(base._observation_occurrences),
    }


_RESOURCE_COUNT_FIELDS = (
    "operation_attempts",
    "vessel_starts",
    "final_assays",
    "discarded_batches",
    "nonfinal_instrument_uses",
)
_RESOURCE_REPORT_FIELDS = (
    "process_time_s",
    "sample_consumed_L",
    "physical_cost",
    "accumulated_risk",
)


def _world_state_sections(env: Any) -> dict[str, dict[str, Any]]:
    state = env.unwrapped._state.to_dict(include_hidden=True)
    ledger = copy.deepcopy(state.pop("ledger", {}))
    process = copy.deepcopy(state.pop("process", {}))
    return {
        "physical": state,
        "ledger": ledger if isinstance(ledger, dict) else {},
        "process": process if isinstance(process, dict) else {},
    }


def _empty_resource_delta() -> dict[str, Any]:
    return {
        **dict.fromkeys(_RESOURCE_COUNT_FIELDS, 0),
        "instrument_uses": {},
        "stocks": {},
        "report_only": {
            **dict.fromkeys(_RESOURCE_REPORT_FIELDS, 0.0),
            "observed_risk": 0.0,
        },
    }


def _add_resource_delta(total: dict[str, Any], raw_delta: Any) -> None:
    if not isinstance(raw_delta, Mapping):
        return
    for key in _RESOURCE_COUNT_FIELDS:
        total[key] = int(total[key]) + int(raw_delta.get(key, 0))
    for key in ("instrument_uses", "stocks"):
        raw_values = raw_delta.get(key, {})
        if not isinstance(raw_values, Mapping):
            continue
        values = total[key]
        for item, value in raw_values.items():
            item_id = str(item)
            if key == "instrument_uses":
                values[item_id] = int(values.get(item_id, 0)) + int(value)
            else:
                values[item_id] = float(values.get(item_id, 0.0)) + float(value)
    raw_report = raw_delta.get("report_only", {})
    if not isinstance(raw_report, Mapping):
        return
    report = total["report_only"]
    for key in _RESOURCE_REPORT_FIELDS:
        report[key] = float(report[key]) + float(raw_report.get(key, 0.0))
    report["observed_risk"] = max(
        float(report["observed_risk"]),
        float(raw_report.get("observed_risk", 0.0)),
    )


def _resource_state_view(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {}
    state = snapshot.get("state", {})
    return copy.deepcopy(dict(state)) if isinstance(state, Mapping) else {}


def _resource_state_delta(before: Any, after: Any) -> dict[str, Any]:
    before_state = _resource_state_view(before)
    after_state = _resource_state_view(after)
    delta = _empty_resource_delta()
    for key in _RESOURCE_COUNT_FIELDS:
        delta[key] = int(after_state.get(key, 0)) - int(before_state.get(key, 0))
    for source_key, target_key in (
        ("instrument_uses", "instrument_uses"),
        ("stocks_used", "stocks"),
    ):
        before_values = before_state.get(source_key, {})
        after_values = after_state.get(source_key, {})
        if not isinstance(before_values, Mapping) or not isinstance(after_values, Mapping):
            continue
        delta[target_key] = {
            str(item): (
                int(after_values.get(item, 0)) - int(before_values.get(item, 0))
                if target_key == "instrument_uses"
                else float(after_values.get(item, 0.0)) - float(before_values.get(item, 0.0))
            )
            for item in sorted(set(before_values) | set(after_values))
            if (
                int(after_values.get(item, 0)) - int(before_values.get(item, 0))
                if target_key == "instrument_uses"
                else abs(float(after_values.get(item, 0.0)) - float(before_values.get(item, 0.0)))
                > 1.0e-12
            )
        }
    before_report = before_state.get("report_only", {})
    after_report = after_state.get("report_only", {})
    if isinstance(before_report, Mapping) and isinstance(after_report, Mapping):
        delta["report_only"] = {
            key: float(after_report.get(key, 0.0)) - float(before_report.get(key, 0.0))
            for key in _RESOURCE_REPORT_FIELDS
        }
        delta["report_only"]["observed_risk"] = float(after_report.get("peak_risk", 0.0))
    return delta


def _resource_delta_mismatches(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key in _RESOURCE_COUNT_FIELDS:
        if int(expected.get(key, 0)) != int(observed.get(key, 0)):
            mismatches.append(key)
    for key in ("instrument_uses", "stocks"):
        expected_values = expected.get(key, {})
        observed_values = observed.get(key, {})
        if not isinstance(expected_values, Mapping) or not isinstance(observed_values, Mapping):
            mismatches.append(key)
            continue
        all_items = set(expected_values) | set(observed_values)
        for item in sorted(all_items):
            expected_value = float(expected_values.get(item, 0.0))
            observed_value = float(observed_values.get(item, 0.0))
            if abs(expected_value - observed_value) > 1.0e-12:
                mismatches.append(f"{key}.{item}")
    expected_report = expected.get("report_only", {})
    observed_report = observed.get("report_only", {})
    if not isinstance(expected_report, Mapping) or not isinstance(observed_report, Mapping):
        return [*mismatches, "report_only"]
    for key in _RESOURCE_REPORT_FIELDS:
        if (
            abs(float(expected_report.get(key, 0.0)) - float(observed_report.get(key, 0.0)))
            > 1.0e-12
        ):
            mismatches.append(f"report_only.{key}")
    return mismatches


def _recipe_resource_card(
    task: TaskSpec,
    world_seed: int,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    proposed = _empty_resource_delta()
    actions = list(case.get("compiled_actions", []))
    for action in actions:
        _add_resource_delta(
            proposed,
            derive_campaign_resource_delta(action).to_dict(),
        )
    stock_limits = {
        stock_id: float(amount) + 1.0e-12 for stock_id, amount in proposed["stocks"].items()
    }
    return {
        "card_id": (f"first-paper-reference-{task.task_id}-seed-{world_seed}-{case['case_id']}"),
        "operation_attempt_limit": max(int(task.budget), len(actions)),
        "vessel_start_limit": 1,
        "final_assay_limit": max(int(proposed["final_assays"]), 1),
        "nonfinal_instrument_use_limit": int(proposed["nonfinal_instrument_uses"]),
        "stock_limits": stock_limits,
        "per_instrument_limits": copy.deepcopy(proposed["instrument_uses"]),
        "metadata": {
            "task_id": task.task_id,
            "world_seed": world_seed,
            "case_id": str(case["case_id"]),
            "scope": "reference_valid_recipe_receipt",
        },
    }


def _resource_receipt_summary(
    step_receipts: list[dict[str, Any]],
    *,
    before_snapshot: Any,
    after_snapshot: Any,
    public_state: Any,
) -> dict[str, Any]:
    proposed = _empty_resource_delta()
    outcome = _empty_resource_delta()
    allowed_count = 0
    attempt_charged_count = 0
    operations_committed = 0
    rejection_reasons: Counter[str] = Counter()
    for receipt in step_receipts:
        preflight = receipt.get("preflight", {})
        if isinstance(preflight, Mapping):
            allowed_count += int(preflight.get("allowed") is True)
            attempt_charged_count += int(preflight.get("attempt_charged") is True)
            rejection_reasons.update(
                str(reason) for reason in preflight.get("rejection_reasons", [])
            )
            _add_resource_delta(proposed, preflight.get("proposed_delta", {}))
        operations_committed += int(receipt.get("operation_committed") is True)
        _add_resource_delta(outcome, receipt.get("outcome_delta", {}))
    observed_delta = _resource_state_delta(before_snapshot, after_snapshot)
    expected_delta = copy.deepcopy(outcome)
    expected_delta["operation_attempts"] = attempt_charged_count
    mismatches = _resource_delta_mismatches(expected_delta, observed_delta)
    private_hash = (
        after_snapshot.get("ledger_sha256") if isinstance(after_snapshot, Mapping) else None
    )
    public_hash = public_state.get("ledger_sha256") if isinstance(public_state, Mapping) else None
    if not isinstance(private_hash, str) or public_hash != private_hash:
        mismatches.append("public_private_ledger_sha256")
    return {
        "preflight": {
            "receipt_count": len(step_receipts),
            "allowed_count": allowed_count,
            "attempt_charged_count": attempt_charged_count,
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "proposed_delta": proposed,
        },
        "outcome_delta": {
            **outcome,
            "operations_committed": operations_committed,
        },
        "observed_ledger_delta": observed_delta,
        "initial_ledger_state": _resource_state_view(before_snapshot),
        "final_ledger_state": _resource_state_view(after_snapshot),
        "private_ledger_sha256": private_hash,
        "public_ledger_sha256": public_hash,
        "ledger_hash_reconciled": bool(private_hash) and public_hash == private_hash,
        "resource_reconciled": not mismatches,
        "reconciliation_mismatches": mismatches,
        "step_receipts": step_receipts,
    }


def _post_termination_validation_receipt(
    base: Any,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = next(
        (
            copy.deepcopy(action)
            for action in actions
            if action.get("operation") != "terminate"
            and not (
                action.get("operation") == "measure" and action.get("instrument") == "final_assay"
            )
        ),
        {"operation": "add_solvent", "volume_L": 0.001, "solvent": 0},
    )
    state_before = _world_state_sections(base)
    resource_before = base.campaign_resource_snapshot()
    validation = base.validate_action(candidate)
    state_after = _world_state_sections(base)
    resource_after = base.campaign_resource_snapshot()
    observed_valid = bool(validation.get("valid"))
    return {
        "validate_only": True,
        "candidate_action": candidate,
        "expected_valid": False,
        "observed_valid": observed_valid,
        "invalid_reasons": copy.deepcopy(validation.get("invalid_reasons", [])),
        "preconditions": copy.deepcopy(validation.get("preconditions", {})),
        "world_state_preserved": state_after == state_before,
        "resource_state_preserved": resource_after == resource_before,
        "passed": (
            not observed_valid and state_after == state_before and resource_after == resource_before
        ),
    }


def _leakage_findings(env: Any, payload: Any, surface: str) -> list[dict[str, Any]]:
    base = env.unwrapped
    hidden_species = set(base._state.species_amounts)
    return [
        {"surface": surface, **finding.to_dict()}
        for finding in audit_public_payload(payload, hidden_species_ids=hidden_species)
    ]


def _schema_receipt(env: Any, task: TaskSpec) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    field_count = 0
    bounded_field_count = 0
    categorical_field_count = 0
    for operation in sorted(task.allowed_operations):
        schema = env.unwrapped.action_schema(operation)
        required_fields = set(schema.get("required_fields", []))
        fields = {
            str(field.get("field")): field
            for field in schema.get("fields", [])
            if isinstance(field, dict)
        }
        if not bool(schema.get("task_allowed")):
            failures.append({"operation": operation, "reason": "operation_not_task_allowed"})
        if set(fields) != required_fields:
            failures.append(
                {
                    "operation": operation,
                    "reason": "required_field_schema_mismatch",
                    "required_fields": sorted(required_fields),
                    "schema_fields": sorted(fields),
                }
            )
        for field_name, field in sorted(fields.items()):
            field_count += 1
            unit = field.get("unit")
            if not isinstance(unit, str) or not unit:
                failures.append(
                    {"operation": operation, "field": field_name, "reason": "missing_unit"}
                )
            bounds = field.get("bounds")
            choices = field.get("choices")
            if isinstance(bounds, dict):
                bounded_field_count += 1
                low = bounds.get("low")
                high = bounds.get("high")
                if not isinstance(low, int | float) or not isinstance(high, int | float):
                    failures.append(
                        {
                            "operation": operation,
                            "field": field_name,
                            "reason": "non_numeric_bounds",
                        }
                    )
                elif not np.isfinite([float(low), float(high)]).all() or float(low) > float(high):
                    failures.append(
                        {"operation": operation, "field": field_name, "reason": "invalid_bounds"}
                    )
            elif isinstance(choices, list) and choices:
                categorical_field_count += 1
            else:
                failures.append(
                    {"operation": operation, "field": field_name, "reason": "missing_domain"}
                )
    return {
        "passed": not failures,
        "operation_count": len(task.allowed_operations),
        "field_count": field_count,
        "bounded_field_count": bounded_field_count,
        "categorical_field_count": categorical_field_count,
        "failures": failures,
    }


def _run_recipe_case(
    task: TaskSpec,
    world_seed: int,
    case: dict[str, Any],
    *,
    scratch_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    resource_card = _recipe_resource_card(task, world_seed, case)
    env = gym.make(
        task.env_id,
        **task.env_kwargs(seed=world_seed),
        campaign_resource_card=resource_card,
    )
    trajectory_path = scratch_dir / f"{task.task_id}-{world_seed}-{case['case_id']}.jsonl"
    failures: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    resource_step_receipts: list[dict[str, Any]] = []
    constitution_check_names: set[str] = set()
    initial_public_view_sha256 = ""
    evaluation_receipt: dict[str, Any] = {}
    resource_summary: dict[str, Any] = {
        "preflight": {},
        "outcome_delta": {},
        "resource_reconciled": False,
        "reconciliation_mismatches": ["execution_did_not_complete"],
    }
    post_termination_receipt: dict[str, Any] = {
        "validate_only": True,
        "passed": False,
        "failure": "terminate action was not reached",
    }
    all_transactions_committed = True
    all_actions_prevalidated = True
    constitution_failure_count = 0
    terminate_count = 0
    terminate_committed_count = 0
    final_assay_count = 0
    final_assay_committed_count = 0
    final_terminated = False
    final_truncated = False
    right_censored = False
    try:
        observation, reset_info = env.reset(seed=world_seed)
        base: Any = env.unwrapped
        resource_before = base.campaign_resource_snapshot()
        task_info = base.task_info()
        logging_task_info = {**task_info, **base.evaluator_provenance()}
        leakage.extend(_leakage_findings(env, reset_info, "reset_info"))
        initial_public_view = agent_view_bundle(env, observation, {})
        initial_public_view_sha256 = _sha256_value(initial_public_view)
        leakage.extend(_leakage_findings(env, initial_public_view, "initial_agent_view"))
        with TrajectoryLogger(trajectory_path) as logger:
            for step, action in enumerate(case["compiled_actions"], start=1):
                validation = base.validate_action(action)
                if not bool(validation.get("valid")):
                    all_actions_prevalidated = False
                    failures.append(
                        {
                            "step": step,
                            "class": "valid_recipe_prevalidation_failed",
                            "action": action,
                            "invalid_reasons": validation.get("invalid_reasons", []),
                        }
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                committed = info.get("transaction_status") == "committed"
                preflight = info.get("campaign_resource_preflight")
                outcome_delta = info.get("campaign_resource_outcome_delta")
                step_receipt = {
                    "step": step,
                    "action": copy.deepcopy(action),
                    "action_sha256": _sha256_value(action),
                    "operation": action.get("operation"),
                    "instrument": action.get("instrument"),
                    "schema_validation": {
                        "valid": bool(validation.get("valid")),
                        "invalid_reasons": copy.deepcopy(validation.get("invalid_reasons", [])),
                        "canonical_action_sha256": _sha256_value(
                            validation.get("canonical_action")
                        ),
                    },
                    "transaction_status": info.get("transaction_status"),
                    "operation_committed": committed,
                    "preflight": copy.deepcopy(preflight),
                    "outcome_delta": copy.deepcopy(outcome_delta),
                }
                resource_step_receipts.append(step_receipt)
                if not isinstance(preflight, Mapping) or not isinstance(outcome_delta, Mapping):
                    failures.append(
                        {
                            "step": step,
                            "class": "campaign_resource_receipt_missing",
                        }
                    )
                if not committed:
                    all_transactions_committed = False
                    failures.append(
                        {
                            "step": step,
                            "class": "valid_recipe_transaction_not_committed",
                            "action": action,
                            "transaction_status": info.get("transaction_status"),
                            "preconditions": info.get("preconditions", {}),
                        }
                    )
                failed_checks = [
                    check
                    for check in info.get("constitution_checks", [])
                    if isinstance(check, dict) and check.get("passed") is False
                ]
                named_checks = sorted(
                    str(check.get("name"))
                    for check in info.get("constitution_checks", [])
                    if isinstance(check, dict) and isinstance(check.get("name"), str)
                )
                constitution_check_names.update(named_checks)
                constitution_failure_count += len(failed_checks)
                if failed_checks:
                    failures.append(
                        {
                            "step": step,
                            "class": "constitution_check_failed",
                            "checks": failed_checks,
                        }
                    )
                if (
                    action.get("operation") == "measure"
                    and action.get("instrument") == "final_assay"
                ):
                    final_assay_count += 1
                    final_assay_committed_count += int(committed)
                    final_terminated = bool(terminated)
                    final_truncated = bool(truncated)
                    evaluation_receipt = {
                        "reward": float(reward),
                        "environment_reward": info.get("environment_reward"),
                        "observed_reward": info.get("observed_reward"),
                        "leaderboard_score": info.get("leaderboard_score"),
                        "experiment_completed": info.get("experiment_completed"),
                        "experiment_ended": info.get("experiment_ended"),
                        "transaction_status": info.get("transaction_status"),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                if action.get("operation") == "terminate":
                    terminate_count += 1
                    terminate_committed_count += int(committed)
                    if committed:
                        post_termination_receipt = _post_termination_validation_receipt(
                            base,
                            case["compiled_actions"],
                        )
                right_censored = right_censored or bool(
                    info.get("right_censored_open_batch", False)
                )
                public_view = agent_view_bundle(env, observation, info)
                step_leakage = _leakage_findings(
                    env,
                    public_view,
                    f"agent_view.step-{step}",
                )
                leakage.extend(step_leakage)
                world_events = [
                    event for event in info.get("world_events", []) if isinstance(event, dict)
                ]
                step_receipt.update(
                    {
                        "state_transition_sha256": _sha256_value(
                            {
                                "delta": info.get("state_delta_summary"),
                                "patches": info.get("state_patches_summary"),
                            }
                        ),
                        "failed_preconditions": sorted(
                            str(name)
                            for name, passed in info.get("preconditions", {}).items()
                            if passed is False
                        ),
                        "constitution_check_count": len(named_checks),
                        "constitution_failed_check_names": sorted(
                            str(check.get("name")) for check in failed_checks
                        ),
                        "world_event_count": len(world_events),
                        "world_event_types": sorted(
                            str(event.get("event_type")) for event in world_events
                        ),
                        "event_propagation_matches_operation": any(
                            event.get("event_type") == "operation_applied"
                            and event.get("operation_type") == action.get("operation")
                            for event in world_events
                        ),
                        "public_observation_sha256": _sha256_value(
                            observation_to_json(observation)
                        ),
                        "public_private_leakage_count": len(step_leakage),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                logger.log(
                    task_info=logging_task_info,
                    step=step,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    agent_metadata={
                        "agent_id": "frozen_deterministic_infrastructure_probe",
                        "policy_randomness": "none",
                    },
                    agent_view=public_view,
                )
                if (terminated or truncated) and step != len(case["compiled_actions"]):
                    failures.append(
                        {
                            "step": step,
                            "class": "lifecycle_ended_before_frozen_recipe",
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    break
        resource_after = base.campaign_resource_snapshot()
        public_resource_state = base.public_campaign_resource_state()
        resource_summary = _resource_receipt_summary(
            resource_step_receipts,
            before_snapshot=resource_before,
            after_snapshot=resource_after,
            public_state=public_resource_state,
        )
        if not bool(resource_summary["resource_reconciled"]):
            failures.append(
                {
                    "class": "campaign_resource_reconciliation_failed",
                    "mismatches": resource_summary["reconciliation_mismatches"],
                }
            )
        if not bool(post_termination_receipt.get("passed")):
            failures.append(
                {
                    "class": "post_termination_validate_only_rejection_failed",
                    "receipt": post_termination_receipt,
                }
            )
        records = load_jsonl(trajectory_path)
        replay = verify_records(records, tolerance=0.0).to_dict()
        if not replay["verified"]:
            failures.append(
                {
                    "class": "exact_replay_failed",
                    "mismatch_count": len(replay["mismatches"]),
                }
            )
    except Exception as exc:  # qualification must retain every deterministic failure
        replay = {
            "verified": False,
            "checked_steps": 0,
            "max_abs_error": None,
            "mismatches": [],
        }
        failures.append(
            {
                "class": "execution_exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        env.close()

    if leakage:
        failures.append({"class": "public_private_leakage", "finding_count": len(leakage)})
    lifecycle_closed = (
        final_assay_count == 1
        and final_assay_committed_count == 1
        and not right_censored
        and len(case["compiled_actions"]) <= task.budget
    )
    if not lifecycle_closed:
        failures.append(
            {
                "class": "lifecycle_not_closed",
                "final_assay_count": final_assay_count,
                "committed_final_assay_count": final_assay_committed_count,
                "right_censored": right_censored,
            }
        )
    compact_step_receipts = [
        {
            **{
                key: value
                for key, value in receipt.items()
                if key not in {"preflight", "outcome_delta"}
            },
            "preflight_sha256": _sha256_value(receipt.get("preflight")),
            "outcome_delta_sha256": _sha256_value(receipt.get("outcome_delta")),
        }
        for receipt in resource_step_receipts
    ]
    elapsed = time.perf_counter() - started
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "metadata": case["metadata"],
        "vector_sha256": case["vector_sha256"],
        "compiled_actions_sha256": case["compiled_actions_sha256"],
        "compiled_operation_count": len(case["compiled_actions"]),
        "within_official_budget": len(case["compiled_actions"]) <= task.budget,
        "all_actions_prevalidated": all_actions_prevalidated,
        "all_transactions_committed": all_transactions_committed,
        "constitution_failure_count": constitution_failure_count,
        "final_assay_count": final_assay_count,
        "final_assay_committed_count": final_assay_committed_count,
        "right_censored_open_batch": right_censored,
        "lifecycle_closed": lifecycle_closed,
        "execution_receipt": {
            "compiled": bool(case["compiled_actions"]),
            "executed": len(resource_step_receipts) == len(case["compiled_actions"]),
            "closed": lifecycle_closed,
            "resource_reconciled": bool(resource_summary["resource_reconciled"]),
        },
        "lifecycle_receipt": {
            "terminate_count": terminate_count,
            "terminate_committed_count": terminate_committed_count,
            "final_assay_count": final_assay_count,
            "final_assay_committed_count": final_assay_committed_count,
            "final_step_terminated": final_terminated,
            "final_step_truncated": final_truncated,
            "right_censored_open_batch": right_censored,
            "post_termination_nonfinal_validation": post_termination_receipt,
        },
        "resource_card": resource_card,
        "resource_preflight": resource_summary["preflight"],
        "resource_outcome_delta": resource_summary["outcome_delta"],
        "resource_reconciled": resource_summary["resource_reconciled"],
        "resource_reconciliation": {
            key: value
            for key, value in resource_summary.items()
            if key not in {"preflight", "outcome_delta", "step_receipts"}
        },
        "step_receipts": compact_step_receipts,
        "constitution_receipt": {
            "named_check_count": len(constitution_check_names),
            "check_names": sorted(constitution_check_names),
            "failure_count": constitution_failure_count,
            "passed": constitution_failure_count == 0,
        },
        "public_observation_receipt": {
            "initial_public_view_sha256": initial_public_view_sha256,
            "step_surface_count": len(compact_step_receipts),
            "leakage_findings": leakage,
        },
        "evaluation_receipt": evaluation_receipt,
        "public_private_leakage_count": len(leakage),
        "public_private_leakage_findings": leakage,
        "exact_replay": replay,
        "elapsed_s": elapsed,
        "trajectory_bytes": trajectory_path.stat().st_size if trajectory_path.exists() else 0,
        "passed": not failures,
        "failures": failures,
    }


def _observed_rejection_reasons(info: Mapping[str, Any]) -> list[str]:
    reasons: set[str] = {
        str(reason) for reason in info.get("campaign_resource_rejection_reasons", [])
    }
    preconditions = info.get("preconditions", {})
    if isinstance(preconditions, Mapping):
        reasons.update(str(name) for name, passed in preconditions.items() if passed is False)
    world_events = info.get("world_events", [])
    if isinstance(world_events, list):
        for event in world_events:
            if not isinstance(event, Mapping):
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            for key in (
                "invalid_reasons",
                "failed_preconditions",
                "rejection_reasons",
            ):
                values = payload.get(key, [])
                if isinstance(values, list):
                    reasons.update(str(value) for value in values)
    return sorted(reasons)


def _accounting_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, float]:
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in ("time_s", "cost", "risk", "sample_consumed_L")
    }


def _declared_failure_penalty(info: Mapping[str, Any]) -> dict[str, float]:
    declared = {
        "time_s": 0.0,
        "cost": 0.0,
        "risk": 0.0,
        "sample_consumed_L": 0.0,
    }
    raw_patches = info.get("state_patches_summary", [])
    if not isinstance(raw_patches, list):
        return declared
    for patch in raw_patches:
        if not isinstance(patch, Mapping):
            continue
        summary = patch.get("summary", {})
        if not isinstance(summary, Mapping):
            continue
        declared["cost"] += float(summary.get("delta_cost", 0.0))
        declared["risk"] += float(summary.get("delta_risk", 0.0))
        declared["sample_consumed_L"] += float(summary.get("delta_sample_consumed_L", 0.0))
    return declared


def _accounting_delta_matches(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    return all(
        abs(float(expected.get(key, 0.0)) - float(observed.get(key, 0.0))) <= 1.0e-12
        for key in ("time_s", "cost", "risk", "sample_consumed_L")
    )


def _negative_ghost_state_receipt(
    *,
    action: Mapping[str, Any],
    info: Mapping[str, Any],
    state_before: Mapping[str, Mapping[str, Any]],
    state_after: Mapping[str, Mapping[str, Any]],
    rng_preserved: bool,
    resource_before: Any,
    resource_after: Any,
    public_resource_state: Any,
) -> dict[str, Any]:
    ledger_before = state_before["ledger"]
    ledger_after = state_after["ledger"]
    process_before = state_before["process"]
    process_after = state_after["process"]
    ledger_delta = _accounting_delta(ledger_before, ledger_after)
    process_delta = _accounting_delta(process_before, process_after)
    declared_penalty = _declared_failure_penalty(info)
    ledger_nonaccounting_before = {
        key: value
        for key, value in ledger_before.items()
        if key not in {"time_s", "cost", "risk", "sample_consumed_L"}
    }
    ledger_nonaccounting_after = {
        key: value
        for key, value in ledger_after.items()
        if key not in {"time_s", "cost", "risk", "sample_consumed_L"}
    }
    process_nonaccounting_before = {
        key: value
        for key, value in process_before.items()
        if key not in {"time_s", "cost", "risk", "sample_consumed_L"}
    }
    process_nonaccounting_after = {
        key: value
        for key, value in process_after.items()
        if key not in {"time_s", "cost", "risk", "sample_consumed_L"}
    }
    resource_step = {
        "step": 1,
        "operation": action.get("operation"),
        "instrument": action.get("instrument"),
        "transaction_status": info.get("transaction_status"),
        "operation_committed": info.get("transaction_status") == "committed",
        "preflight": copy.deepcopy(info.get("campaign_resource_preflight")),
        "outcome_delta": copy.deepcopy(info.get("campaign_resource_outcome_delta")),
    }
    resource = _resource_receipt_summary(
        [resource_step],
        before_snapshot=resource_before,
        after_snapshot=resource_after,
        public_state=public_resource_state,
    )
    physical_preserved = state_after["physical"] == state_before["physical"]
    ledger_reconciled = (
        ledger_nonaccounting_after == ledger_nonaccounting_before
        and _accounting_delta_matches(declared_penalty, ledger_delta)
    )
    process_reconciled = (
        process_nonaccounting_after == process_nonaccounting_before
        and _accounting_delta_matches(declared_penalty, process_delta)
    )
    raw_events = info.get("world_events", [])
    world_events = copy.deepcopy(raw_events) if isinstance(raw_events, list) else []
    event_reconciled = bool(world_events) and all(
        isinstance(event, Mapping) and event.get("operation_type") == action.get("operation")
        for event in world_events
    )
    ghost_state_preserved = bool(
        physical_preserved
        and rng_preserved
        and ledger_reconciled
        and process_reconciled
        and resource["resource_reconciled"]
        and event_reconciled
    )
    return {
        "ghost_state_preserved": ghost_state_preserved,
        "physical": {
            "before_sha256": _sha256_value(state_before["physical"]),
            "after_sha256": _sha256_value(state_after["physical"]),
            "preserved": physical_preserved,
        },
        "observation_rng": {"preserved": rng_preserved},
        "ledger": {
            "before": copy.deepcopy(ledger_before),
            "after": copy.deepcopy(ledger_after),
            "actual_delta": ledger_delta,
            "declared_failure_penalty": declared_penalty,
            "nonaccounting_state_preserved": (
                ledger_nonaccounting_after == ledger_nonaccounting_before
            ),
            "declared_penalty_reconciled": _accounting_delta_matches(
                declared_penalty, ledger_delta
            ),
            "ghost_state_preserved": ledger_reconciled,
        },
        "process": {
            "before": copy.deepcopy(process_before),
            "after": copy.deepcopy(process_after),
            "actual_delta": process_delta,
            "nonaccounting_state_preserved": (
                process_nonaccounting_after == process_nonaccounting_before
            ),
            "declared_penalty_reconciled": _accounting_delta_matches(
                declared_penalty, process_delta
            ),
            "ghost_state_preserved": process_reconciled,
        },
        "events": {
            "world_events": world_events,
            "state_patches_summary": copy.deepcopy(info.get("state_patches_summary", [])),
            "reconciled": event_reconciled,
        },
        "resource": resource,
    }


def _negative_probe(
    task: TaskSpec,
    world_seed: int,
    *,
    probe_id: str,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = gym.make(
        task.env_id,
        **task.env_kwargs(seed=world_seed),
        **(env_kwargs or {}),
    )
    failures: list[str] = []
    info: dict[str, Any] = {}
    leakage: list[dict[str, Any]] = []
    expected_status = "unknown"
    expected_reason = "unknown"
    observed_reasons: list[str] = []
    setup_receipt: dict[str, Any] | None = None
    ghost_state: dict[str, Any] = {
        "ghost_state_preserved": False,
        "failure": "probe did not complete",
    }
    try:
        env.reset(seed=world_seed)
        base: Any = env.unwrapped
        action: dict[str, Any]
        if probe_id == "invalid_operation":
            action = {"operation": "not-a-real-operation"}
            expected_status = "validation_failed"
            expected_reason = "unknown operation: not-a-real-operation"
        elif probe_id == "precondition_failure":
            instruments = sorted(
                instrument for instrument in task.allowed_instruments if instrument != "final_assay"
            )
            instrument = instruments[0] if instruments else "final_assay"
            action = {"operation": "measure", "instrument": instrument}
            # The payload is structurally valid.  The runtime therefore emits
            # a replayable rolled-back transaction for the failed has_volume
            # precondition instead of classifying it as a schema failure.
            expected_status = "rolled_back"
            expected_reason = "has_volume"
        elif probe_id == "resource_exhaustion":
            first_action = {"operation": "add_solvent", "volume_L": 0.026, "solvent": 1}
            _observation, _reward, _terminated, _truncated, first_info = env.step(first_action)
            if first_info.get("transaction_status") != "committed":
                failures.append("resource setup action did not commit")
            setup_receipt = {
                "action": first_action,
                "transaction_status": first_info.get("transaction_status"),
                "operation_committed": (first_info.get("transaction_status") == "committed"),
                "preflight": copy.deepcopy(first_info.get("campaign_resource_preflight")),
                "outcome_delta": copy.deepcopy(first_info.get("campaign_resource_outcome_delta")),
            }
            action = {"operation": "add_solvent", "volume_L": 0.001, "solvent": 1}
            expected_status = "campaign_resource_rejected"
            expected_reason = "stock_limit:solvent_L"
        else:
            raise ValueError(f"unknown negative probe: {probe_id}")

        state_before = _world_state_sections(env)
        rng_before = _rng_snapshot(env)
        resource_before = base.campaign_resource_snapshot()
        _observation, _reward, _terminated, _truncated, info = env.step(action)
        state_after = _world_state_sections(env)
        rng_preserved = _rng_snapshot(env) == rng_before
        resource_after = base.campaign_resource_snapshot()
        public_resource_state = base.public_campaign_resource_state()
        observed_reasons = _observed_rejection_reasons(info)
        if info.get("transaction_status") != expected_status:
            failures.append(
                "expected transaction_status="
                f"{expected_status}, got {info.get('transaction_status')}"
            )
        if expected_reason not in observed_reasons:
            failures.append(f"expected rejection reason={expected_reason}, got {observed_reasons}")
        if expected_reason == "stock_limit:solvent_L":
            if info.get("campaign_resource_rejection_reasons") != [expected_reason]:
                failures.append("resource rejection reason drifted")
            if info.get("campaign_resource_outcome_delta", {}).get("stocks") != {}:
                failures.append("rejected resource probe consumed physical stocks")
            snapshot = base.campaign_resource_snapshot()
            public = base.public_campaign_resource_state()
            if not isinstance(snapshot, dict) or not isinstance(public, dict):
                failures.append("resource ledger snapshot is missing")
            else:
                state = snapshot.get("state", {})
                if state.get("operation_attempts") != 2:
                    failures.append("resource operation attempts do not reconcile")
                if abs(float(state.get("stocks_used", {}).get("solvent_L", -1.0)) - 0.026) > 1e-12:
                    failures.append("resource solvent stock does not reconcile")
                if public.get("ledger_sha256") != snapshot.get("ledger_sha256"):
                    failures.append("public/private resource ledger hashes do not reconcile")
        ghost_state = _negative_ghost_state_receipt(
            action=action,
            info=info,
            state_before=state_before,
            state_after=state_after,
            rng_preserved=rng_preserved,
            resource_before=resource_before,
            resource_after=resource_after,
            public_resource_state=public_resource_state,
        )
        if not bool(ghost_state["physical"]["preserved"]):
            failures.append("negative probe mutated physical state")
        if not rng_preserved:
            failures.append("negative probe mutated observation RNG state")
        if not bool(ghost_state["ledger"]["ghost_state_preserved"]):
            failures.append("negative probe ledger penalty did not reconcile")
        if not bool(ghost_state["process"]["ghost_state_preserved"]):
            failures.append("negative probe process penalty did not reconcile")
        if not bool(ghost_state["events"]["reconciled"]):
            failures.append("negative probe failure events did not reconcile")
        if not bool(ghost_state["resource"]["resource_reconciled"]):
            failures.append("negative probe resource ledger did not reconcile")
        leakage = _leakage_findings(
            env,
            agent_view_bundle(env, base._last_observation, info),
            f"negative_probe.{probe_id}",
        )
        if leakage:
            failures.append("negative probe public view leaked private state")
    except Exception as exc:
        info = {}
        leakage = []
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        env.close()
    return {
        "probe_id": probe_id,
        "passed": not failures,
        "action": action if "action" in locals() else None,
        "expected_rejection": {
            "transaction_status": expected_status,
            "reason": expected_reason,
        },
        "observed_rejection": {
            "transaction_status": info.get("transaction_status"),
            "rollback_reason": info.get("rollback_reason"),
            "reasons": observed_reasons,
            "preconditions": copy.deepcopy(info.get("preconditions", {})),
            "campaign_resource_rejection_reasons": copy.deepcopy(
                info.get("campaign_resource_rejection_reasons", [])
            ),
        },
        "transaction_status": info.get("transaction_status"),
        "physical_state_preserved": bool(ghost_state.get("physical", {}).get("preserved", False)),
        "observation_rng_preserved": bool(
            ghost_state.get("observation_rng", {}).get("preserved", False)
        ),
        "ghost_state": ghost_state,
        "resource_preflight": copy.deepcopy(info.get("campaign_resource_preflight")),
        "resource_outcome_delta": copy.deepcopy(info.get("campaign_resource_outcome_delta")),
        "resource_reconciliation": copy.deepcopy(ghost_state.get("resource", {})),
        "setup_receipt": setup_receipt,
        "public_private_leakage_count": len(leakage),
        "public_private_leakage_findings": leakage,
        "failures": failures,
    }


def run_negative_probes(
    task: TaskSpec,
    world_seed: int,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    resource_spec = copy.deepcopy(
        protocol["intervention"]["negative_probes_per_unit"]["resource_exhaustion"]["resource_card"]
    )
    resource_spec.update(
        {
            "card_id": f"first-paper-infrastructure-{task.task_id}-seed-{world_seed}",
            "metadata": {
                "task_id": task.task_id,
                "world_seed": world_seed,
                "protocol_id": protocol["protocol_id"],
            },
        }
    )
    return [
        _negative_probe(
            task,
            world_seed,
            probe_id="invalid_operation",
            env_kwargs={"campaign_resource_card": copy.deepcopy(resource_spec)},
        ),
        _negative_probe(
            task,
            world_seed,
            probe_id="precondition_failure",
            env_kwargs={"campaign_resource_card": copy.deepcopy(resource_spec)},
        ),
        _negative_probe(
            task,
            world_seed,
            probe_id="resource_exhaustion",
            env_kwargs={"campaign_resource_card": copy.deepcopy(resource_spec)},
        ),
    ]


def _unit_property_status(
    schema: dict[str, Any],
    cases: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "units_and_action_domains": bool(schema["passed"]),
        "applicable_conservation": all(case["constitution_failure_count"] == 0 for case in cases),
        "transaction_atomicity": all(
            probe["physical_state_preserved"] and probe["observation_rng_preserved"]
            for probe in probes
        ),
        "resource_reconciliation": next(
            probe["passed"] for probe in probes if probe["probe_id"] == "resource_exhaustion"
        ),
        "lifecycle_closure": all(case["lifecycle_closed"] for case in cases),
        "public_private_separation": (
            all(case["public_private_leakage_count"] == 0 for case in cases)
            and all(probe["public_private_leakage_count"] == 0 for probe in probes)
        ),
        "exact_replay": all(case["exact_replay"]["verified"] for case in cases),
    }


def run_task_world_unit(
    task: TaskSpec,
    world_seed: int,
    protocol: dict[str, Any],
    *,
    scratch_dir: Path,
) -> dict[str, Any]:
    env = gym.make(task.env_id, **task.env_kwargs(seed=world_seed))
    try:
        env.reset(seed=world_seed)
        schema = _schema_receipt(env, task)
    finally:
        env.close()
    cases = [
        _run_recipe_case(task, world_seed, case, scratch_dir=scratch_dir)
        for case in recipe_cases(task)
    ]
    probes = run_negative_probes(task, world_seed, protocol)
    properties = _unit_property_status(schema, cases, probes)
    failures = [{"surface": "schema", **failure} for failure in schema["failures"]]
    failures.extend(
        {
            "surface": "valid_recipe",
            "case_id": case["case_id"],
            "failures": case["failures"],
        }
        for case in cases
        if not case["passed"]
    )
    failures.extend(
        {
            "surface": "negative_probe",
            "probe_id": probe["probe_id"],
            "failures": probe["failures"],
        }
        for probe in probes
        if not probe["passed"]
    )
    return {
        "unit_id": f"{task.task_id}:seed-{world_seed}",
        "task_id": task.task_id,
        "world_seed": world_seed,
        "task_contract_sha256": task.contract_hash,
        "schema_receipt": schema,
        "valid_recipe_case_count": len(cases),
        "valid_recipe_cases": cases,
        "negative_probe_count": len(probes),
        "negative_probes": probes,
        "properties": properties,
        "passed": all(properties.values()) and not failures,
        "failures": failures,
    }


def build_report(
    protocol: dict[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    binding = validate_protocol_bindings(
        protocol,
        repository_root=root,
        require_clean=True,
    )
    units: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chemworld-infrastructure-qualification-") as tmp:
        scratch = Path(tmp)
        for task in list_tasks():
            for world_seed in task.seeds:
                units.append(
                    run_task_world_unit(
                        task,
                        int(world_seed),
                        protocol,
                        scratch_dir=scratch,
                    )
                )

    property_matrix: list[dict[str, Any]] = []
    for task in list_tasks():
        task_units = [unit for unit in units if unit["task_id"] == task.task_id]
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "independent_unit_count": len(task_units),
        }
        for property_id in PROPERTY_IDS:
            passed = sum(bool(unit["properties"][property_id]) for unit in task_units)
            row[property_id] = {
                "passed": passed,
                "denominator": len(task_units),
                "status": "passed" if passed == len(task_units) else "failed",
            }
        property_matrix.append(row)

    valid_cases = [case for unit in units for case in unit["valid_recipe_cases"]]
    probes = [probe for unit in units for probe in unit["negative_probes"]]
    failure_classes: Counter[str] = Counter()
    for unit in units:
        for failure in unit["failures"]:
            nested = failure.get("failures")
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        failure_classes[
                            str(item.get("class", item.get("reason", "unclassified")))
                        ] += 1
                    else:
                        failure_classes[str(item)] += 1
            else:
                failure_classes[
                    str(failure.get("class", failure.get("reason", "unclassified")))
                ] += 1
    compiled_hash_counts = Counter(case["compiled_actions_sha256"] for case in valid_cases)
    collision_case_count = sum(count for count in compiled_hash_counts.values() if count > 1)
    summaries = {
        "registered_task_count": len({unit["task_id"] for unit in units}),
        "independent_unit_count": len(units),
        "independent_unit_pass_count": sum(bool(unit["passed"]) for unit in units),
        "valid_recipe_case_count": len(valid_cases),
        "valid_recipe_case_pass_count": sum(bool(case["passed"]) for case in valid_cases),
        "negative_probe_count": len(probes),
        "negative_probe_pass_count": sum(bool(probe["passed"]) for probe in probes),
        "exact_replay_case_count": len(valid_cases),
        "exact_replay_pass_count": sum(
            bool(case["exact_replay"]["verified"]) for case in valid_cases
        ),
        "public_private_leakage_count": sum(
            int(case["public_private_leakage_count"]) for case in valid_cases
        )
        + sum(int(probe["public_private_leakage_count"]) for probe in probes),
        "compiled_action_hash_collision_case_count": collision_case_count,
        "failure_class_counts": dict(sorted(failure_classes.items())),
    }
    overall_pass = (
        summaries["independent_unit_count"] == 64
        and summaries["independent_unit_pass_count"] == 64
        and summaries["valid_recipe_case_pass_count"] == summaries["valid_recipe_case_count"]
        and summaries["negative_probe_pass_count"] == 192
        and summaries["exact_replay_pass_count"] == summaries["exact_replay_case_count"]
        and summaries["public_private_leakage_count"] == 0
        and not failure_classes
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256_value(protocol),
        "owner": protocol["owner"],
        "status": "passed" if overall_pass else "failed",
        "claim_boundary": protocol["claim_boundary"],
        "source_binding": binding,
        "counting_rule": {
            "independent_unit": "registered task/world-seed configuration",
            "repeated_observations": (
                "recipes, operations, probes, and replay events within each unit"
            ),
            "statistics": "deterministic descriptive counts only",
        },
        "summary": summaries,
        "task_by_property_matrix": property_matrix,
        "units": units,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# First-paper cross-world infrastructure qualification",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        f"Owner: `{report['owner']}`",
        "",
        "## Deterministic counts",
        "",
        "| Quantity | Passed | Denominator |",
        "| --- | ---: | ---: |",
        (
            f"| Registered task/world units | {summary['independent_unit_pass_count']} "
            f"| {summary['independent_unit_count']} |"
        ),
        (
            "| Valid midpoint/boundary/category recipes | "
            f"{summary['valid_recipe_case_pass_count']} "
            f"| {summary['valid_recipe_case_count']} |"
        ),
        (
            f"| Invalid/precondition/resource probes | {summary['negative_probe_pass_count']} "
            f"| {summary['negative_probe_count']} |"
        ),
        (
            f"| Exact replays | {summary['exact_replay_pass_count']} "
            f"| {summary['exact_replay_case_count']} |"
        ),
        "",
        f"Public/private leakage findings: `{summary['public_private_leakage_count']}`.",
        "",
        "## Task-by-property matrix",
        "",
        (
            "| Task | Units | Units/domains | Conservation | Atomicity | Resources | "
            "Lifecycle | Separation | Replay |"
        ),
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "passed": "PASS",
        "failed": "FAIL",
    }
    for row in report["task_by_property_matrix"]:
        statuses = [labels[row[property_id]["status"]] for property_id in PROPERTY_IDS]
        lines.append(
            f"| {row['task_id']} | {row['independent_unit_count']} | " + " | ".join(statuses) + " |"
        )
    lines.extend(
        [
            "",
            "## Failure classes",
            "",
        ]
    )
    failures = summary["failure_class_counts"]
    if failures:
        lines.extend(f"- `{name}`: {count}" for name, count in failures.items())
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(
    *,
    protocol_path: Path,
    report_path: Path,
    markdown_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    write_set = protocol["coordination"]["write_set"]
    bound_files: dict[str, str] = {}
    for relative in write_set:
        path = repository_root / relative
        if path.exists() and path.is_file() and not relative.endswith(".manifest.json"):
            bound_files[relative] = _sha256_path(path)
    tracked_source_paths = sorted(
        path for path in _git(repository_root, "ls-files", "src", "scripts").splitlines() if path
    )
    tracked_source_digest = hashlib.sha256()
    for relative in tracked_source_paths:
        tracked_source_digest.update(relative.encode("utf-8"))
        tracked_source_digest.update(b"\0")
        tracked_source_digest.update((repository_root / relative).read_bytes())
        tracked_source_digest.update(b"\0")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "owner": protocol["owner"],
        "execution_commit": _git(repository_root, "rev-parse", "HEAD"),
        "execution_command": protocol["formal_execution"]["command"],
        "tracked_source_sha256": tracked_source_digest.hexdigest(),
        "tracked_source_file_count": len(tracked_source_paths),
        "artifacts": dict(sorted(bound_files.items())),
        "report_path": str(report_path.relative_to(repository_root)),
        "markdown_path": str(markdown_path.relative_to(repository_root)),
    }


def write_outputs(
    report: dict[str, Any],
    *,
    protocol_path: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
) -> tuple[Path, Path, Path]:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    report_file = Path(output_path).resolve()
    markdown_file = report_file.with_suffix(".md")
    manifest_file = report_file.with_name(f"{report_file.stem}.manifest.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_file.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    manifest = build_manifest(
        protocol_path=protocol_file,
        report_path=report_file,
        markdown_path=markdown_file,
        repository_root=root,
    )
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_file, markdown_file, manifest_file


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PROPERTY_IDS",
    "PROTOCOL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "QualificationProtocolError",
    "build_manifest",
    "build_report",
    "load_protocol",
    "recipe_cases",
    "render_markdown",
    "run_negative_probes",
    "run_task_world_unit",
    "validate_protocol_bindings",
    "write_outputs",
]
