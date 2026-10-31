"""Deterministic audit of public action affordances against runtime domains.

The audit deliberately stays outside benchmark scoring.  It builds reachable
states with the public task-recipe interface, samples only values advertised by
``action_schema``, and then checks the validator, transactional runtime, and
observation kernel in that order.  This separates agent errors from contract
holes. Converged trajectories outside explicit physical limits are retained as
rejections when the runtime reports the violation and rolls back atomically;
legal input syntax does not guarantee a successful physical experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

import chemworld  # noqa: F401  # registers the Gym environment
from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_from_unit_vector,
)
from chemworld.data.logging import to_builtin
from chemworld.foundation import WorldState
from chemworld.tasks import SERIOUS_TASK_IDS

AUDIT_SCHEMA_VERSION = "chemworld-runtime-domain-affordance-audit-0.5"
GUARDED_SOURCE_PATHS = (
    "src/chemworld/action_codec.py",
    "src/chemworld/agent_interface.py",
    "src/chemworld/envs",
    "src/chemworld/foundation",
    "src/chemworld/operation_validator.py",
    "src/chemworld/physchem",
    "src/chemworld/runtime",
    "src/chemworld/tasks.py",
    "src/chemworld/world",
)


def guarded_source_manifest(repository_root: str | Path) -> dict[str, str]:
    """Hash the executable Python sources covered by the affordance audit."""

    root = Path(repository_root).resolve()
    files: set[Path] = set()
    for relative in GUARDED_SOURCE_PATHS:
        target = root / relative
        if target.is_file():
            files.add(target)
        elif target.is_dir():
            files.update(
                path
                for path in target.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def guarded_source_sha256(repository_root: str | Path) -> str:
    manifest = guarded_source_manifest(repository_root)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReachableState:
    """One immutable runtime state reached through public actions."""

    step_index: int
    state: Any


def _schema_digest(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        to_builtin(schema),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_actions(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return a bounded one-factor sample of values advertised by a schema."""

    operation = str(schema["operation"])
    fields = schema.get("fields", [])
    base: dict[str, Any] = {"operation": operation}
    for field in fields:
        name = str(field["field"])
        bounds = field.get("bounds")
        choices = field.get("choices")
        if isinstance(bounds, dict):
            low = float(bounds["low"])
            high = float(bounds["high"])
            if field.get("type") == "integer":
                integer_low = (math.floor(low) + 1 if field.get("lower_bound_inclusive") is False
                               else math.ceil(low))
                integer_high = math.floor(high)
                if integer_low > integer_high:
                    raise ValueError(f"No integer value is available for {name}")
                base[name] = (integer_low + integer_high) // 2
            else:
                base[name] = low + 0.5 * (high - low)
        elif isinstance(choices, list) and choices:
            base[name] = choices[0]

    candidates: list[tuple[str, dict[str, Any]]] = [("midpoint", dict(base))]
    for field in fields:
        name = str(field["field"])
        bounds = field.get("bounds")
        choices = field.get("choices")
        if isinstance(bounds, dict):
            for boundary in ("low", "high"):
                candidate = dict(base)
                value = float(bounds[boundary])
                origin = f"{name}:{boundary}"
                if boundary == "low" and field.get("lower_bound_inclusive") is False:
                    high = float(bounds["high"])
                    value = (math.floor(value) + 1 if field.get("type") == "integer"
                             else value + 0.01 * (high - value))
                    origin = f"{name}:near_low_exclusive"
                if field.get("type") == "integer":
                    value = math.ceil(value) if boundary == "low" else math.floor(value)
                candidate[name] = value
                candidates.append((origin, candidate))
        elif isinstance(choices, list) and choices:
            for index in sorted({0, len(choices) - 1}):
                candidate = dict(base)
                candidate[name] = choices[index]
                candidates.append((f"{name}:choice:{index}", candidate))

    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for origin, candidate in candidates:
        key = json.dumps(to_builtin(candidate), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique.append((origin, candidate))
    return unique


def _physical_domain_rejection(
    before: WorldState, after: WorldState, info: dict[str, Any],
) -> bool:
    """Recognize a proven physical limit, without accepting arbitrary rollbacks."""
    diagnostic = info.get("outcome_diagnostics", {})
    if not (
        info.get("transaction_status") == "rolled_back"
        and info.get("operation_outcome") == "rejected"
        and info.get("rollback_reason") == "rejected"
        and info.get("actual_duration_s") == 0.0
        and diagnostic.get("failure_reason") == "reaction_validity_domain_exceeded"
    ):
        return False
    observed = diagnostic.get("reactor_diagnostic", {})
    limits = diagnostic.get("validity_domain", {})
    violations = diagnostic.get("validity_violations", [])
    if not (
        violations and observed.get("within_validity_domain") is False
        and observed.get("material_balance_closed") is True
        and observed.get("nonnegative_species") is True
    ):
        return False
    try:
        values = [float(observed[key]) for key in (
            "minimum_temperature_K", "maximum_temperature_K", "maximum_temperature_rate_K_s",
        )] + [float(limits[key]) for key in (
            "minimum_temperature_K", "maximum_temperature_K", "maximum_temperature_rate_K_s",
            "minimum_pressure_Pa", "maximum_pressure_Pa",
        )]
        if not all(math.isfinite(value) for value in values):
            return False
        low, high, rate, low_limit, high_limit, rate_limit, p_low, p_high = values
        breached = {
            "thermal_runaway_rate_exceeds_limit": rate > rate_limit,
            "temperature_outside_validity_domain": low < low_limit or high > high_limit,
            "pressure_outside_validity_domain": not p_low <= before.pressure_Pa <= p_high,
        }
        if not all(breached.get(reason, False) for reason in violations):
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    # Only the existing failure cost/risk may change; material, energy, time,
    # equipment, phase state and all other fields must remain unchanged.
    left, right = before.to_dict(), after.to_dict()
    for payload in (left, right):
        for key in ("ledger", "process"):
            if isinstance(payload.get(key), dict):
                payload[key].pop("cost", None)
                payload[key].pop("risk", None)
    return (left == right and after.ledger.cost >= before.ledger.cost
            and after.ledger.risk >= before.ledger.risk)


def _reachable_states(
    task_id: str,
    *,
    seed: int,
) -> tuple[Any, list[ReachableState], list[dict[str, Any]], list[dict[str, Any]]]:
    env = gym.make(
        "ChemWorld",
        task_id=task_id,
        seed=seed,
        budget_override=128,
        episode_mode_override="single_experiment",
    )
    env.reset(seed=seed)
    base = cast(Any, env.unwrapped)
    vector = np.full(
        (task_recipe_dimension(base.task_info()),),
        np.float64(0.5),
        dtype=np.float64,
    )
    recipe = task_recipe_from_unit_vector(base.task_info(), vector)
    states: list[ReachableState] = [ReachableState(0, base._state)]
    failures: list[dict[str, Any]] = []
    physical_rejections: list[dict[str, Any]] = []
    for step_index, action in enumerate(recipe["steps"], start=1):
        before = base._state
        _observation, _reward, terminated, truncated, info = env.step(action)
        if info.get("transaction_status") != "committed":
            if _physical_domain_rejection(before, base._state, info):
                physical_rejections.append({
                    "step_index": step_index, "action": action,
                    "violations": info["outcome_diagnostics"]["validity_violations"],
                    "outcome_diagnostics": info["outcome_diagnostics"],
                    "rollback_atomic": True,
                })
                states.append(ReachableState(step_index, base._state))
                # Keep the original stop-at-first-rejection preparation rule.
                # Later recipe steps do not have the material this step failed
                # to produce; do not manufacture a successful prefix for them.
                break
            failures.append(
                {
                    "step_index": step_index,
                    "operation": action.get("operation"),
                    "transaction_status": info.get("transaction_status"),
                    "rollback_reason": info.get("rollback_reason"),
                    "invalid_reasons": list(
                        info.get("world_events", [{}])[-1].get("payload", {}).get(
                            "invalid_reasons", []
                        )
                    ),
                }
            )
            break
        states.append(ReachableState(step_index, base._state))
        if info.get("leaderboard_score") is not None or terminated or truncated:
            break
    return env, states, failures, physical_rejections


def _finding(
    *,
    kind: str,
    task_id: str,
    operation: str,
    state_step_index: int,
    schema_digest: str,
    candidate_origin: str,
    action: dict[str, Any],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "task_id": task_id,
        "operation": operation,
        "state_step_index": state_step_index,
        "schema_sha256": schema_digest,
        "candidate_origin": candidate_origin,
        "action": to_builtin(action),
        **to_builtin(details),
    }


def audit_runtime_domain_affordances(
    *,
    source_commit: str,
    task_ids: Iterable[str] = SERIOUS_TASK_IDS,
    seed: int = 0,
    maximum_state_schemas_per_operation: int = 2,
) -> dict[str, Any]:
    """Audit public schemas at deterministic reachable states.

    Only actions accepted by the central validator are dispatched to the
    runtime.  Rejected exact schema boundaries are reported separately from
    runtime and observation failures.
    """

    selected_tasks = tuple(str(task_id) for task_id in task_ids)
    findings: list[dict[str, Any]] = []
    preparation_failures: dict[str, list[dict[str, Any]]] = {}
    preparation_rejections: dict[str, list[dict[str, Any]]] = {}
    physical_rejections: list[dict[str, Any]] = []
    task_summaries: dict[str, dict[str, Any]] = {}
    total_candidates = 0
    total_validator_valid = 0
    total_runtime_committed = 0

    for task_id in selected_tasks:
        env, states, task_preparation_failures, task_preparation_rejections = _reachable_states(
            task_id, seed=seed
        )
        base = env.unwrapped
        per_operation_schema_counts: Counter[str] = Counter()
        task_candidates = 0
        task_valid = 0
        task_committed = 0
        task_findings_start = len(findings)
        task_rejections_start = len(physical_rejections)
        try:
            for reachable in states:
                base._state = reachable.state
                for affordance in base.available_actions():
                    operation = str(affordance["operation"])
                    schema = affordance["schema"]
                    digest = _schema_digest(schema)
                    declared_constraint_ids = {
                        str(item.get("id"))
                        for item in schema.get("constraints", [])
                        if isinstance(item, dict) and item.get("id")
                    }
                    count_key = f"{operation}:{digest}"
                    if per_operation_schema_counts[count_key] > 0:
                        continue
                    if (
                        sum(
                            count
                            for key, count in per_operation_schema_counts.items()
                            if key.startswith(f"{operation}:")
                        )
                        >= maximum_state_schemas_per_operation
                    ):
                        continue
                    per_operation_schema_counts[count_key] += 1
                    for candidate_index, (origin, action) in enumerate(
                        _candidate_actions(schema)
                    ):
                        task_candidates += 1
                        total_candidates += 1
                        validation = base.operation_validator.validate(
                            action,
                            reachable.state,
                        )
                        if not validation.is_valid:
                            payload_reasons = [
                                reason
                                for reason in validation.invalid_reasons
                                if reason.startswith("payload_")
                            ]
                            if payload_reasons:
                                undeclared_reasons = [
                                    reason
                                    for reason in payload_reasons
                                    if reason not in declared_constraint_ids
                                ]
                                if undeclared_reasons:
                                    findings.append(
                                        _finding(
                                            kind="advertised_payload_rejected",
                                            task_id=task_id,
                                            operation=operation,
                                            state_step_index=reachable.step_index,
                                            schema_digest=digest,
                                            candidate_origin=origin,
                                            action=action,
                                            details={
                                                "invalid_reasons": undeclared_reasons
                                            },
                                        )
                                    )
                            continue

                        task_valid += 1
                        total_validator_valid += 1
                        try:
                            result = base.runtime.apply_transaction(
                                reachable.state,
                                action,
                            )
                        except (ArithmeticError, ValueError) as error:
                            findings.append(
                                _finding(
                                    kind="validator_valid_runtime_exception",
                                    task_id=task_id,
                                    operation=operation,
                                    state_step_index=reachable.step_index,
                                    schema_digest=digest,
                                    candidate_origin=origin,
                                    action=action,
                                    details={"error_type": type(error).__name__},
                                )
                            )
                            continue
                        if result.kernel_result.transaction_status != "committed":
                            if _physical_domain_rejection(
                                reachable.state, result.state, result.info_payload(),
                            ):
                                physical_rejections.append(_finding(
                                    kind="physical_domain_rejection", task_id=task_id,
                                    operation=operation, state_step_index=reachable.step_index,
                                    schema_digest=digest, candidate_origin=origin, action=action,
                                    details={
                                        "violations": result.kernel_result.diagnostics[
                                            "validity_violations"
                                        ],
                                        "outcome_diagnostics": result.kernel_result.diagnostics,
                                        "rollback_atomic": True,
                                    },
                                ))
                                continue
                            findings.append(
                                _finding(
                                    kind="validator_valid_runtime_rollback",
                                    task_id=task_id,
                                    operation=operation,
                                    state_step_index=reachable.step_index,
                                    schema_digest=digest,
                                    candidate_origin=origin,
                                    action=action,
                                    details={
                                        "transaction_status": (
                                            result.kernel_result.transaction_status
                                        ),
                                        "rollback_reason": result.kernel_result.rollback_reason,
                                    },
                                )
                            )
                            continue

                        task_committed += 1
                        total_runtime_committed += 1
                        try:
                            base.observation_kernel.observe(
                                result.state,
                                action,
                                np.random.default_rng(
                                    seed + 10_000 * reachable.step_index + candidate_index
                                ),
                            )
                        except (ArithmeticError, ValueError) as error:
                            findings.append(
                                _finding(
                                    kind="validator_valid_observation_exception",
                                    task_id=task_id,
                                    operation=operation,
                                    state_step_index=reachable.step_index,
                                    schema_digest=digest,
                                    candidate_origin=origin,
                                    action=action,
                                    details={"error_type": type(error).__name__},
                                )
                            )
        finally:
            env.close()

        if task_preparation_failures:
            preparation_failures[task_id] = task_preparation_failures
        if task_preparation_rejections:
            preparation_rejections[task_id] = task_preparation_rejections
        task_findings = findings[task_findings_start:]
        task_summaries[task_id] = {
            "preparation_completed": not task_preparation_failures
            and not task_preparation_rejections,
            "reachable_state_count": len(states),
            "schema_count": sum(per_operation_schema_counts.values()),
            "operation_count": len(
                {key.split(":", maxsplit=1)[0] for key in per_operation_schema_counts}
            ),
            "candidate_count": task_candidates,
            "validator_valid_count": task_valid,
            "runtime_committed_count": task_committed,
            "physical_rejection_count": len(physical_rejections) - task_rejections_start,
            "finding_count": len(task_findings),
            "finding_counts": dict(
                sorted(Counter(item["kind"] for item in task_findings).items())
            ),
        }

    finding_counts = dict(sorted(Counter(item["kind"] for item in findings).items()))
    findings_by_operation: dict[str, int] = defaultdict(int)
    for item in findings:
        findings_by_operation[str(item["operation"])] += 1
    checks = {
        "all_serious_tasks_covered": selected_tasks == tuple(SERIOUS_TASK_IDS),
        "preparation_without_unexpected_failure": not preparation_failures,
        "public_schema_boundaries_match_validator": finding_counts.get(
            "advertised_payload_rejected", 0
        )
        == 0,
        "validator_valid_runtime_domain_clean": (
            finding_counts.get("validator_valid_runtime_exception", 0) == 0
            and finding_counts.get("validator_valid_runtime_rollback", 0) == 0
        ),
        "validator_valid_observation_domain_clean": finding_counts.get(
            "validator_valid_observation_exception", 0
        )
        == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if passed else "contract_holes_detected",
        "passed": passed,
        "source_commit": source_commit,
        "seed": seed,
        "task_ids": list(selected_tasks),
        "sampling_contract": {
            "reachable_state_source": "public_midpoint_task_recipe_prefixes",
            "candidate_source": "public_action_schema_midpoint_and_one_field_boundaries",
            "maximum_state_schemas_per_operation": maximum_state_schemas_per_operation,
            "validator_required_before_runtime_dispatch": True,
            "hidden_species_or_rate_parameters_recorded": False,
        },
        "checks": checks,
        "summary": {
            "task_count": len(selected_tasks),
            "preparation_completed_count": sum(
                bool(task["preparation_completed"]) for task in task_summaries.values()
            ),
            "candidate_count": total_candidates,
            "validator_valid_count": total_validator_valid,
            "runtime_committed_count": total_runtime_committed,
            "physical_rejection_count": len(physical_rejections),
            "preparation_physical_rejection_count": sum(map(len, preparation_rejections.values())),
            "finding_count": len(findings),
            "finding_counts": finding_counts,
            "findings_by_operation": dict(sorted(findings_by_operation.items())),
        },
        "tasks": task_summaries,
        "preparation_failures": preparation_failures,
        "preparation_physical_rejections": preparation_rejections,
        "physical_rejections": physical_rejections,
        "findings": findings,
        "benchmark_claim_allowed": False,
        "publication_ready": False,
        "limitations": [
            "This is a deterministic boundary/conformance audit, not an exhaustive proof over "
            "the continuous action-state space.",
            "The audit does not evaluate algorithm quality or benchmark performance.",
            "Physical-limit rejections are counted separately; "
            "they are not successful experiments.",
            "Preparation stops at its first rejection. Later recipe states are not covered; "
            "preparation_completed_count reports this sampling limitation.",
            "Optional maintainer-only payload extensions outside the public action schema are "
            "not sampled.",
        ],
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "GUARDED_SOURCE_PATHS",
    "audit_runtime_domain_affordances",
    "guarded_source_manifest",
    "guarded_source_sha256",
]
