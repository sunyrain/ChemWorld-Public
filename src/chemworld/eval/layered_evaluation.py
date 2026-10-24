"""Layered vNext evaluation without changing frozen v0.1 result semantics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from chemworld.data.records import trajectory_value
from chemworld.task_design import SERIOUS_TASK_DESIGNS
from chemworld.tasks import get_task

LAYERED_EVALUATION_VERSION = "chemworld-layered-evaluation-0.3"


@dataclass(frozen=True)
class TaskEvaluationContract:
    task_id: str
    objective_metric: str
    primary_task_metric: str
    direction: Literal["maximize", "minimize"]
    terminal_selector: Literal["successful_final_assay"]
    online_reward_role: Literal["diagnostic_shaping_only"]
    safety_limit: float
    risk_aggregation: Literal["max_operation_risk_per_experiment"]
    risk_limit_semantics: Literal["benchmark_operational_risk_budget"]
    cost_aggregation: Literal["terminal_total_minus_measurement_cost"]
    missing_primary_policy: Literal["fail"] = "fail"

    def __post_init__(self) -> None:
        if not 0.0 < self.safety_limit <= 1.0:
            raise ValueError("safety_limit must be in (0, 1]")
        if not self.primary_task_metric:
            raise ValueError("primary_task_metric cannot be empty")

    @classmethod
    def for_task(
        cls,
        task_id: str,
        *,
        risk_limit: float | None = None,
    ) -> TaskEvaluationContract:
        task = get_task(task_id)
        design = SERIOUS_TASK_DESIGNS[task_id]
        return cls(
            task_id=task_id,
            objective_metric="leaderboard_score",
            primary_task_metric=design.primary_metric,
            direction="maximize",
            terminal_selector="successful_final_assay",
            online_reward_role="diagnostic_shaping_only",
            safety_limit=task.safety_limit if risk_limit is None else risk_limit,
            risk_aggregation="max_operation_risk_per_experiment",
            risk_limit_semantics="benchmark_operational_risk_budget",
            cost_aggregation="terminal_total_minus_measurement_cost",
        )

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAYERED_EVALUATION_VERSION,
            **asdict(self),
            "contract_hash": self.contract_hash,
            "layers": {
                "objective": self.objective_metric,
                "task_primary": self.primary_task_metric,
                "online_shaping": self.online_reward_role,
                "constraint": self.risk_aggregation,
                "resource": self.cost_aggregation,
                "validity": "precondition and constitution flags",
            },
            "diagnostics": {
                "interaction": "declared capability stratum plus observed decision evidence",
                "method_resources": "external model/training ledger; never scalarized into score",
                "harness_assistance": "official runner assistance policy",
            },
        }


def evaluate_layered_records(
    records: list[dict[str, Any]],
    *,
    contract: TaskEvaluationContract,
) -> dict[str, Any]:
    """Recompute six disjoint evaluation layers from a public trajectory."""

    if not records:
        raise ValueError("cannot evaluate an empty trajectory")
    attempts = _attempt_summaries(records)
    terminal_records = [attempt["terminal_record"] for attempt in attempts if attempt["complete"]]
    primary_values = [
        _required_observation(record, contract.primary_task_metric) for record in terminal_records
    ]
    objective_values = [
        _finite_float(trajectory_value(record, 'leaderboard_score'), "leaderboard_score")
        for record in terminal_records
    ]
    terminal_costs = [float(attempt["total_cost"]) for attempt in attempts if attempt["complete"]]
    observed_risks = [
        _finite_float(trajectory_value(record, 'observation')["safety_risk"], "safety_risk")
        for record in records
        if isinstance(trajectory_value(record, 'observation', None), dict)
        and trajectory_value(record, 'observation').get("safety_risk") is not None
    ]
    flags = [
        record.get("constraint_flags", {})
        if isinstance(record.get("constraint_flags"), dict)
        else {}
        for record in records
    ]
    unsafe_count = sum(bool(item.get("unsafe")) for item in flags)
    high_cost_count = sum(bool(item.get("high_cost")) for item in flags)
    invalid_count = sum(
        bool(item.get("precondition_failed")) or bool(item.get("constitution_failed"))
        for item in flags
    )
    rewards = [
        _finite_float(trajectory_value(record, "reward", 0.0), "reward") for record in records
    ]
    measurement_cost = sum(float(attempt["measurement_cost"]) for attempt in attempts)
    process_costs = [float(attempt["process_cost"]) for attempt in attempts]
    max_risks = [
        float(attempt["max_risk"]) for attempt in attempts if attempt["max_risk"] is not None
    ]
    risk_exceedance_count = sum(risk >= contract.safety_limit for risk in max_risks)
    maximize = contract.direction == "maximize"
    select = max if maximize else min
    completed = bool(terminal_records)
    interaction = _interaction_diagnostics(records)
    final_method_resources = records[-1].get("method_resources", {})
    if not isinstance(final_method_resources, dict):
        final_method_resources = {}
    return {
        "schema_version": LAYERED_EVALUATION_VERSION,
        "task_id": contract.task_id,
        "evaluation_contract_hash": contract.contract_hash,
        "status": "evaluated" if completed else "failed_no_successful_final_assay",
        "endpoint_available": completed,
        "failure_reason": None if completed else "no_successful_final_assay",
        "objective": {
            "metric": contract.objective_metric,
            "best": select(objective_values) if completed else 0.0,
            "last": objective_values[-1] if completed else 0.0,
            "terminal_values": objective_values,
            "missing_endpoint_policy": "fail_closed_zero",
        },
        "task_primary": {
            "metric": contract.primary_task_metric,
            "best": select(primary_values) if completed else 0.0,
            "last": primary_values[-1] if completed else 0.0,
            "terminal_values": primary_values,
            "missing_endpoint_policy": "fail_closed_zero",
        },
        "online_shaping": {
            "role": contract.online_reward_role,
            "mean_reward": sum(rewards) / len(rewards),
            "reward_sum": sum(rewards),
            "eligible_as_primary_endpoint": False,
        },
        "constraints": {
            "safety_limit": contract.safety_limit,
            "risk_limit_semantics": contract.risk_limit_semantics,
            "risk_aggregation": contract.risk_aggregation,
            "max_observed_safety_risk": max(observed_risks) if observed_risks else None,
            "experiment_max_risks": max_risks,
            "risk_budget_exceedance_count": risk_exceedance_count,
            "risk_budget_exceedance_rate": risk_exceedance_count / len(max_risks)
            if max_risks
            else 0.0,
            "legacy_unsafe_operation_count": unsafe_count,
            "high_cost_operation_count": high_cost_count,
            "constraint_activated": risk_exceedance_count > 0,
        },
        "resources": {
            "campaign_total_cost": sum(float(attempt["total_cost"]) for attempt in attempts),
            "campaign_process_cost": sum(process_costs),
            "terminal_experiment_costs": terminal_costs,
            "measurement_cost": measurement_cost,
            "attempt_process_costs": process_costs,
            "operation_count": len(records),
            "complete_experiment_count": len(terminal_records),
            "incomplete_experiment_count": sum(not attempt["complete"] for attempt in attempts),
            "method_resource_ledger": final_method_resources,
            "method_resource_accounting_complete": bool(
                final_method_resources.get("accounting_complete")
                or final_method_resources.get("agent_usage", {}).get("accounting_complete")
            ),
            "resource_axes_scalarized_into_endpoint": False,
        },
        "validity": {
            "invalid_operation_count": invalid_count,
            "invalid_operation_rate": invalid_count / len(records),
        },
        "interaction": interaction,
    }


def _interaction_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    first_metadata = records[0].get("agent_metadata", {})
    if not isinstance(first_metadata, dict):
        first_metadata = {}
    capabilities = first_metadata.get("interaction_capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    decision_scope = str(capabilities.get("decision_scope", "unknown"))
    if decision_scope == "experiment_recipe":
        stratum = "recipe_search"
    elif decision_scope == "operation" and bool(capabilities.get("adapts_within_experiment")):
        stratum = "operation_closed_loop"
    elif decision_scope == "operation":
        stratum = "operation_open_loop"
    else:
        stratum = "undeclared"

    audits: list[dict[str, Any]] = []
    spectral_packet_decisions = 0
    for record in records:
        explanation = record.get("explanation", {})
        if not isinstance(explanation, dict):
            continue
        audit = explanation.get("decision_audit", {})
        if isinstance(audit, dict):
            audits.append(audit)
        outcome = explanation.get("outcome", {})
        if isinstance(outcome, dict) and bool(outcome.get("has_spectral_packet")):
            spectral_packet_decisions += 1
    provided = [audit for audit in audits if audit.get("status") == "provided"]
    adaptation_counts: dict[str, int] = {}
    for audit in provided:
        source = str(audit.get("adaptation_source", "none"))
        adaptation_counts[source] = adaptation_counts.get(source, 0) + 1

    runner_policy = first_metadata.get("official_runner_policy", {})
    if not isinstance(runner_policy, dict):
        runner_policy = {}
    unassisted = bool(runner_policy) and all(
        runner_policy.get(field) is False
        for field in (
            "automatic_action_repair",
            "automatic_terminate",
            "automatic_final_assay",
        )
    )
    return {
        "capability_stratum": stratum,
        "declared_capabilities": capabilities,
        "decision_audit_count": len(audits),
        "provided_decision_audit_count": len(provided),
        "adaptation_source_counts": adaptation_counts,
        "spectral_packet_decision_count": spectral_packet_decisions,
        "observed_within_experiment_adaptation": any(
            adaptation_counts.get(source, 0) > 0
            for source in ("measurement", "spectrum", "validator")
        ),
        "observed_across_experiment_adaptation": adaptation_counts.get(
            "experiment_memory", 0
        )
        > 0,
        "official_runner_policy": runner_policy,
        "harness_assistance_absent": unassisted,
        "cross_stratum_interpretation": (
            "system-level comparison only; algorithm-only effects require the same "
            "interaction capability stratum"
        ),
        "interaction_diagnostics_scalarized_into_endpoint": False,
    }


def _attempt_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        current.append(record)
        if trajectory_value(record, 'leaderboard_score', None) is not None:
            attempts.append(_summarize_attempt(current, complete=True))
            current = []
    if current:
        attempts.append(_summarize_attempt(current, complete=False))
    return attempts


def _summarize_attempt(
    records: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    risks = [
        _finite_float(trajectory_value(record, 'observation')["safety_risk"], "safety_risk")
        for record in records
        if isinstance(trajectory_value(record, 'observation', None), dict)
        and trajectory_value(record, 'observation').get("safety_risk") is not None
    ]
    measurement_cost = sum(
        _finite_float(record.get("measurement_cost", 0.0), "measurement_cost") for record in records
    )
    try:
        total_cost = attempt_total_cost(records)
    except ValueError:
        if complete:
            raise
        # A rejected or truncated tail can contain no public cost observation.
        # It adds no process cost, but any explicitly charged measurement cost
        # remains part of the resource ledger.
        total_cost = measurement_cost
    process_cost = total_cost - measurement_cost
    if process_cost < -1.0e-9:
        raise ValueError("measurement cost exceeds the experiment total cost")
    return {
        "complete": complete,
        "terminal_record": records[-1],
        "total_cost": total_cost,
        "measurement_cost": measurement_cost,
        "process_cost": max(process_cost, 0.0),
        "max_risk": max(risks) if risks else None,
    }


def attempt_total_cost(records: list[dict[str, Any]]) -> float:
    """Return unsaturated ledger cost for one experiment when it is available.

    The public observation deliberately caps ``cost`` at 1.0.  Summing the
    public ``state_delta_summary.delta_cost`` values preserves the exact ledger
    units needed to separate process and measurement cost, including campaigns
    with many instrument calls.  Older/minimal records without state deltas
    retain the terminal-observation fallback.
    """

    deltas: list[float] = []
    for record in records:
        summary = record.get("state_delta_summary")
        if not isinstance(summary, dict) or summary.get("delta_cost") is None:
            return _last_observed_value(records, "cost")
        delta = _finite_float(summary["delta_cost"], "delta_cost")
        if delta < -1.0e-9:
            raise ValueError("experiment cost delta cannot be negative")
        deltas.append(max(delta, 0.0))
    return sum(deltas)


def _last_observed_value(records: list[dict[str, Any]], key: str) -> float:
    for record in reversed(records):
        observation = trajectory_value(record, 'observation', None)
        if isinstance(observation, dict) and observation.get(key) is not None:
            return _finite_float(observation[key], key)
    raise ValueError(f"experiment has no observed {key!r}")


def _required_observation(record: dict[str, Any], key: str) -> float:
    observation = trajectory_value(record, 'observation', None)
    if not isinstance(observation, dict) or observation.get(key) is None:
        raise ValueError(f"successful final assay is missing required {key!r}")
    return _finite_float(observation[key], key)


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


__all__ = [
    "LAYERED_EVALUATION_VERSION",
    "TaskEvaluationContract",
    "attempt_total_cost",
    "evaluate_layered_records",
]
