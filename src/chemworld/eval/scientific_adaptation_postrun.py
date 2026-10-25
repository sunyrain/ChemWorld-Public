"""Development-only postrun audit for scientific adaptation receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from chemworld.agents.scientific_adaptation import (
    ScientificExperimentPlan,
    canonical_sha256,
)
from chemworld.eval.mechanism_adaptation_execution import selected_campaign_rows
from chemworld.eval.scientific_adaptation_execution import (
    ScientificAdaptationExperimentSession,
)

SCIENTIFIC_ADAPTATION_POSTRUN_VERSION = (
    "chemworld-scientific-adaptation-postrun-audit-0.1-dev"
)


def scientific_plan_from_receipt(payload: Mapping[str, Any]) -> ScientificExperimentPlan:
    """Reconstruct a previously validated plan without consulting a provider."""

    return ScientificExperimentPlan(
        experiment_intent=str(payload["experiment_intent"]),
        search_vector=tuple(float(item) for item in payload["search_vector"]),
        requested_measurement_slots=tuple(
            str(item) for item in payload["requested_measurement_slots"]
        ),
        diagnostic_target=str(payload["diagnostic_target"]),
        mechanism_distribution={
            str(key): float(value)
            for key, value in dict(payload["mechanism_distribution"]).items()
        },
        expected_effect=str(payload["expected_effect"]),
        belief_update_rule=str(payload["belief_update_rule"]),
        uncertainty=float(payload["uncertainty"]),
        scientific_state=copy.deepcopy(payload.get("scientific_state")),
    )


def stable_scientific_observation_seed(pair_id: str) -> int:
    """Return the observation seed frozen by the development shakedown runner."""

    digest = hashlib.sha256(f"scientific-adaptation|{pair_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def replay_scientific_adaptation_receipt(
    receipt: Mapping[str, Any],
    campaign_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the completed prefix of one terminal receipt exactly."""

    pair = receipt["pair"]
    pair_id = str(pair["pair_id"])
    pre_experiments = int(receipt["development_horizon"]["pre_change_experiments"])
    experiment_audits: list[dict[str, Any]] = []
    for recorded_experiment in receipt["experiments"]:
        recorded_result = recorded_experiment["result"]
        experiment_index = int(recorded_result["experiment_index"])
        interventions = (
            ()
            if experiment_index < pre_experiments
            else tuple(campaign_row["world_interventions"])
        )
        plan = scientific_plan_from_receipt(recorded_result["plan"])
        with ScientificAdaptationExperimentSession(
            task_id=str(pair["task_id"]),
            seed=int(pair["world_seed"]),
            experiment_horizon=1,
            experiment_index_offset=experiment_index,
            interventions=interventions,
            observation_seed=stable_scientific_observation_seed(pair_id),
            observation_noise_namespace=(
                f"scientific-adaptation-shakedown-{pair_id}-"
                f"experiment-{experiment_index:03d}"
            ),
        ) as session:
            replayed_result = session.execute(plan).to_dict()
        compared_fields = (
            "schema_version",
            "interface_version",
            "task_id",
            "experiment_index",
            "plan",
            "executed_steps",
            "measurement_evidence",
            "terminal_summary",
            "completed",
            "operation_count",
            "peak_safety_risk",
        )
        mismatch_fields = [
            field
            for field in compared_fields
            if replayed_result.get(field) != recorded_result.get(field)
        ]
        experiment_audits.append(
            {
                "experiment_index": experiment_index,
                "phase": str(recorded_experiment["phase"]),
                "verified": not mismatch_fields,
                "mismatch_fields": mismatch_fields,
                "recorded_result_sha256": canonical_sha256(recorded_result),
                "replayed_result_sha256": canonical_sha256(replayed_result),
            }
        )
    return {
        "method_id": str(receipt["method_id"]),
        "pair_id": pair_id,
        "task_id": str(pair["task_id"]),
        "arm": str(pair["arm"]),
        "cell_status": str(receipt["cell_status"]),
        "completed_prefix_only": receipt["cell_status"] != "completed",
        "replayed_experiment_count": len(experiment_audits),
        "verified": all(item["verified"] for item in experiment_audits),
        "experiments": experiment_audits,
    }


def _load_run_artifacts(run_root: Path) -> dict[str, Any]:
    report_path = run_root / "report.json"
    receipt_paths = sorted((run_root / "receipts").glob("*.json"))
    if not report_path.is_file() or not receipt_paths:
        raise ValueError(f"run root lacks report.json or terminal receipts: {run_root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
    return {
        "run_root": str(run_root),
        "report_path": str(report_path),
        "report": report,
        "receipt_paths": receipt_paths,
        "receipts": receipts,
    }


def _cell_identity(receipt: Mapping[str, Any]) -> str:
    pair = receipt["pair"]
    return ":".join(
        (
            str(receipt["provider_mode"]),
            str(receipt["method_id"]),
            str(pair["pair_id"]),
            str(pair["arm"]),
        )
    )


def _aggregate_cells(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(item["cell_status"]) for item in receipts)
    known_cost = sum(
        float(item["resources"]["monetary_cost_usd"])
        for item in receipts
        if item["resources"]["accounting_complete"]
    )
    return {
        "cell_count": len(receipts),
        "cell_status_counts": dict(sorted(statuses.items())),
        "completed_cell_count": statuses.get("completed", 0),
        "planned_experiment_count": sum(
            int(item["planned_experiment_count"]) for item in receipts
        ),
        "completed_experiment_count": sum(
            int(item["completed_experiment_count"]) for item in receipts
        ),
        "experiment_completion_rate": (
            sum(int(item["completed_experiment_count"]) for item in receipts)
            / sum(int(item["planned_experiment_count"]) for item in receipts)
            if receipts
            else None
        ),
        "provider_call_count": sum(
            int(item["resources"]["model_call_count"]) for item in receipts
        ),
        "provider_attempt_count": sum(
            int(item["resources"]["provider_attempt_count"]) for item in receipts
        ),
        "provider_reported_prompt_tokens": sum(
            int(item["resources"]["provider_usage"].get("prompt_tokens", 0))
            for item in receipts
        ),
        "provider_reported_completion_tokens": sum(
            int(item["resources"]["provider_usage"].get("completion_tokens", 0))
            for item in receipts
        ),
        "provider_reported_total_tokens": sum(
            int(item["resources"]["provider_usage"].get("total_tokens", 0))
            for item in receipts
        ),
        "accounting_complete": all(
            bool(item["resources"]["accounting_complete"]) for item in receipts
        ),
        "known_billed_cost_usd": known_cost,
        "unknown_cost_cell_count": sum(
            int(not item["resources"]["accounting_complete"]) for item in receipts
        ),
    }


def _grouped_aggregates(
    receipts: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        values: list[str] = []
        for field in key_fields:
            if field in receipt:
                values.append(str(receipt[field]))
            elif field in receipt["pair"]:
                values.append(str(receipt["pair"][field]))
            else:
                values.append(str(receipt["method"][field]))
        groups["|".join(values)].append(receipt)
    return {key: _aggregate_cells(value) for key, value in sorted(groups.items())}


def _failure_classification(receipt: Mapping[str, Any]) -> str | None:
    failure = receipt.get("failure")
    if not isinstance(failure, Mapping):
        return None
    diagnostics = failure.get("validation_diagnostics")
    if isinstance(diagnostics, Mapping):
        return f"{diagnostics.get('field_path')}:{diagnostics.get('constraint')}"
    message = str(failure.get("message", ""))
    known_messages = {
        "scientific_state exceeds its JSON character limit": (
            "scientific_state:max_json_characters"
        ),
        "belief_update_rule exceeds its character limit": (
            "belief_update_rule:max_characters"
        ),
        "scientific_state.next_experiment_plan.varied_variable must be a non-empty string": (
            "scientific_state.next_experiment_plan.varied_variable:non_empty_string"
        ),
    }
    return known_messages.get(message, str(failure.get("reason_code", "unclassified")))


def _score_description(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    scores = [float(item) for item in receipt["scores"]]
    pre_count = int(receipt["development_horizon"]["pre_change_experiments"])
    if len(scores) < pre_count + 1:
        return None
    pre_mean = statistics.fmean(scores[:pre_count])
    post_mean = statistics.fmean(scores[pre_count:])
    return {
        "pre_mean_score": pre_mean,
        "post_mean_score": post_mean,
        "post_minus_pre_mean_score": post_mean - pre_mean,
    }


def _last_belief(receipt: Mapping[str, Any]) -> dict[str, float] | None:
    experiments = receipt["experiments"]
    if not experiments:
        return None
    return {
        str(key): float(value)
        for key, value in experiments[-1]["result"]["plan"][
            "mechanism_distribution"
        ].items()
    }


def _top_candidate(distribution: Mapping[str, float]) -> str:
    maximum = max(distribution.values())
    return sorted(key for key, value in distribution.items() if value == maximum)[0]


def _cell_descriptive(receipt: Mapping[str, Any]) -> dict[str, Any]:
    pair = receipt["pair"]
    belief = _last_belief(receipt)
    score = _score_description(receipt)
    payload: dict[str, Any] = {
        "cell_id": _cell_identity(receipt),
        "provider": str(receipt["provider_mode"]),
        "method_id": str(receipt["method_id"]),
        "model_id": str(receipt["method"]["model_id"]),
        "scaffold_id": str(receipt["method"]["scaffold_id"]),
        "task_id": str(pair["task_id"]),
        "arm": str(pair["arm"]),
        "truth_id": str(pair["truth_id"]),
        "cell_status": str(receipt["cell_status"]),
        "completed_experiment_count": int(receipt["completed_experiment_count"]),
        "planned_experiment_count": int(receipt["planned_experiment_count"]),
        "failure_classification": _failure_classification(receipt),
        "score_description": score,
    }
    if belief is not None:
        payload["last_available_belief"] = belief
        payload["last_top_candidate"] = _top_candidate(belief)
        payload["last_truth_probability"] = belief.get(str(pair["truth_id"]))
        payload["last_no_change_probability"] = belief.get("no_change")
    return payload


def _complete_arm_summary(
    cells: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, Any]:
    selected = [
        item
        for item in cells
        if item["arm"] == arm
        and item["cell_status"] == "completed"
        and item["score_description"] is not None
        and "last_available_belief" in item
    ]
    deltas = [item["score_description"]["post_minus_pre_mean_score"] for item in selected]
    true_probabilities = [
        float(item["last_truth_probability"])
        for item in selected
        if item["last_truth_probability"] is not None
    ]
    summary: dict[str, Any] = {
        "complete_cell_count": len(selected),
        "mean_post_minus_pre_score": statistics.fmean(deltas) if deltas else None,
        "score_improved_cell_count": sum(delta > 0.0 for delta in deltas),
        "last_truth_top1_cell_count": sum(
            item["last_top_candidate"] == item["truth_id"] for item in selected
        ),
        "mean_last_truth_probability": (
            statistics.fmean(true_probabilities) if true_probabilities else None
        ),
    }
    if arm == "changed":
        summary["last_no_change_below_half_cell_count"] = sum(
            float(item["last_no_change_probability"]) < 0.5 for item in selected
        )
    else:
        summary["descriptive_false_positive_cell_count"] = sum(
            float(item["last_no_change_probability"]) < 0.5 for item in selected
        )
    return summary


def _last_available_twin_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [
        item
        for item in cells
        if item["arm"] == "no_change_twin" and "last_available_belief" in item
    ]
    return {
        "cell_count": len(selected),
        "descriptive_false_positive_cell_count": sum(
            float(item["last_no_change_probability"]) < 0.5 for item in selected
        ),
        "definition": "last available no_change probability below 0.5",
    }


def _run_integrity(run: Mapping[str, Any]) -> dict[str, Any]:
    report = run["report"]
    receipts = run["receipts"]
    report_hashes = report.get("receipt_sha256", {})
    hash_mismatches: list[str] = []
    for receipt in receipts:
        key = (
            f"{receipt['method_id']}:{receipt['pair']['pair_id']}:"
            f"{receipt['pair']['arm']}"
        )
        if report_hashes.get(key) != canonical_sha256(receipt):
            hash_mismatches.append(key)
    return {
        "run_root": run["run_root"],
        "provider_mode": str(report["provider_mode"]),
        "report_receipt_count": int(report["terminal_receipt_count"]),
        "discovered_receipt_count": len(receipts),
        "canonical_receipt_hashes_match_report": not hash_mismatches,
        "receipt_hash_mismatches": hash_mismatches,
        "all_formal_result_false": all(not item["formal_result"] for item in receipts),
        "all_benchmark_claim_disallowed": all(
            not item["benchmark_claim_allowed"] for item in receipts
        ),
        "all_terminal": all(
            item["cell_status"] != "infrastructure_failure" for item in receipts
        ),
        "method_config_sha256": str(report["method_config_sha256"]),
        "runner_source_sha256": str(report["runner_source_sha256"]),
    }


def audit_scientific_adaptation_postrun(
    *,
    protocol: Mapping[str, Any],
    run_roots: Sequence[str | Path],
    replay: bool = True,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Build one cross-provider development audit without provider calls."""

    if not run_roots:
        raise ValueError("at least one run root is required")
    if replay_workers <= 0:
        raise ValueError("replay_workers must be positive")
    runs = [_load_run_artifacts(Path(item)) for item in run_roots]
    receipts = [receipt for run in runs for receipt in run["receipts"]]
    identities = [_cell_identity(item) for item in receipts]
    if len(identities) != len(set(identities)):
        raise ValueError("terminal receipt identities must be unique across run roots")
    protocol_sha256 = canonical_sha256(protocol)
    if any(str(item["protocol_sha256"]) != protocol_sha256 for item in receipts):
        raise ValueError("terminal receipt protocol hash does not match the supplied protocol")

    campaign_rows = {
        (str(row["pair_id"]), str(row["arm"])): row
        for row in selected_campaign_rows(protocol)
    }
    replay_audits: list[dict[str, Any]] = []
    if replay:
        replay_inputs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for receipt in receipts:
            pair = receipt["pair"]
            key = (str(pair["pair_id"]), str(pair["arm"]))
            try:
                row = campaign_rows[key]
            except KeyError as error:
                raise ValueError(f"receipt campaign row is absent from protocol: {key}") from error
            replay_inputs.append((receipt, row))
        if replay_workers == 1:
            replay_audits = [
                replay_scientific_adaptation_receipt(receipt, row)
                for receipt, row in replay_inputs
            ]
        else:
            with ProcessPoolExecutor(max_workers=replay_workers) as executor:
                futures = [
                    executor.submit(replay_scientific_adaptation_receipt, receipt, row)
                    for receipt, row in replay_inputs
                ]
                replay_audits = [future.result() for future in futures]

    cells = [_cell_descriptive(item) for item in receipts]
    failure_counts = Counter(
        classification
        for classification in (_failure_classification(item) for item in receipts)
        if classification is not None
    )
    all_decision_audits = [
        experiment["decision_audit"]
        for receipt in receipts
        for experiment in receipt["experiments"]
    ]
    prompt_caps = {
        str(item["method_id"]): int(item["agent_manifest"]["prompt_token_estimate_cap"])
        for item in receipts
    }
    prompt_maxima: dict[str, int] = defaultdict(int)
    for receipt in receipts:
        for experiment in receipt["experiments"]:
            prompt_maxima[str(receipt["method_id"])] = max(
                prompt_maxima[str(receipt["method_id"])] ,
                int(experiment["decision_audit"]["prompt_estimated_tokens"]),
            )

    return {
        "schema_version": SCIENTIFIC_ADAPTATION_POSTRUN_VERSION,
        "formal_result": False,
        "benchmark_claim_allowed": False,
        "development_only": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "source_run_integrity": [_run_integrity(run) for run in runs],
        "overall": _aggregate_cells(receipts),
        "groups": {
            "by_provider": _grouped_aggregates(receipts, ("provider_mode",)),
            "by_scaffold": _grouped_aggregates(receipts, ("scaffold_id",)),
            "by_task": _grouped_aggregates(receipts, ("task_id",)),
            "by_arm": _grouped_aggregates(receipts, ("arm",)),
            "by_provider_scaffold": _grouped_aggregates(
                receipts, ("provider_mode", "scaffold_id")
            ),
            "by_method": _grouped_aggregates(receipts, ("method_id",)),
        },
        "method_failures": {
            "cell_count": sum(failure_counts.values()),
            "classification_counts": dict(sorted(failure_counts.items())),
            "cells": [
                item for item in cells if item["failure_classification"] is not None
            ],
        },
        "prompt_contract": {
            "pre_call_cap_enforced_by_runner": True,
            "successful_decision_audit_count": len(all_decision_audits),
            "successful_decisions_within_cap": all(
                int(item["prompt_estimated_tokens"])
                <= int(item["prompt_token_estimate_cap"])
                for item in all_decision_audits
            ),
            "max_successful_prompt_estimate_by_method": dict(sorted(prompt_maxima.items())),
            "prompt_estimate_cap_by_method": dict(sorted(prompt_caps.items())),
            "failed_response_prompt_estimates_retained": False,
        },
        "physical_replay": {
            "enabled": replay,
            "worker_count": replay_workers if replay else 0,
            "receipt_count": len(replay_audits),
            "replayed_experiment_count": sum(
                int(item["replayed_experiment_count"]) for item in replay_audits
            ),
            "verified_receipt_count": sum(item["verified"] for item in replay_audits),
            "all_verified": replay and all(item["verified"] for item in replay_audits),
            "receipts": replay_audits,
        },
        "descriptive_science": {
            "formal_estimand": False,
            "preregistered_outcome": False,
            "complete_changed_cells": _complete_arm_summary(cells, arm="changed"),
            "complete_no_change_twins": _complete_arm_summary(
                cells, arm="no_change_twin"
            ),
            "all_twins_last_available_belief": _last_available_twin_summary(cells),
            "cell_descriptives": cells,
            "interpretation_guardrail": (
                "Descriptive development diagnostics only; thresholds are not O1-O5 "
                "estimands and do not identify provider or scaffold effects."
            ),
        },
        "claims": {
            "execution_and_accounting_auditable": True,
            "physical_results_replay_verified": (
                replay and all(item["verified"] for item in replay_audits)
            ),
            "provider_effect_claim_allowed": False,
            "scaffold_effect_claim_allowed": False,
            "participant_outcome_claim_allowed": False,
        },
    }


__all__ = [
    "SCIENTIFIC_ADAPTATION_POSTRUN_VERSION",
    "audit_scientific_adaptation_postrun",
    "replay_scientific_adaptation_receipt",
    "scientific_plan_from_receipt",
    "stable_scientific_observation_seed",
]
