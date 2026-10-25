"""Current W2-48 open-action candidate packet and binding helpers.

This module deliberately keeps participant exploration open.  It only freezes the terminal
candidate packet and verifies that the public ActionPlan is the exact plan evaluated by truth and
replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_reviewer_followup import B3_METRIC_IDS, build_b3_candidate_queries
from chemworld.eval.work_ii_truth import compile_evaluator_truth_query

ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
OPEN_TERMINAL_SCHEMA = "chemworld-work-ii-terminal-action-readout-contract-0.1"
OPEN_PROTOCOL_VERSION = "chemworld-work-ii-as-open-action-decision-pilot-0.1"


def _packet_digest(packet_seed: int, query_id: str) -> str:
    return sha256(f"{int(packet_seed)}:{query_id}".encode()).hexdigest()


def _workflow_family(action_plan: Sequence[Mapping[str, Any]]) -> str:
    operations = tuple(str(item.get("operation")) for item in action_plan)
    measurements = tuple(
        str(item.get("instrument"))
        for item in action_plan
        if item.get("operation") == "measure"
    )
    if "separate_phase" in operations and measurements.count("hplc") >= 2:
        return "partition_measure_before_and_after_separation"
    return "partition_complete_extraction"


def _public_action_plan(
    query: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    actions = [deepcopy(dict(item)) for item in compiled["action_plan"]]
    if not actions or actions[-1].get("operation") != "measure" or actions[-1].get(
        "instrument"
    ) != "final_assay":
        raise ValueError(f"{query['query_id']}: action plan lacks terminal final_assay")
    operation_names = [str(item.get("operation")) for item in actions]
    measurement_positions = [
        index + 1
        for index, item in enumerate(actions)
        if item.get("operation") == "measure"
    ]
    public = {
        "query_id": str(query["query_id"]),
        "pair_id": str(query["pair_id"]),
        "initial_state_contract": "fresh_partition_batch",
        "ordered_operations": operation_names,
        "all_operation_parameters": actions,
        "action_plan": actions,
        "action_plan_sha256": str(compiled["action_plan_sha256"]),
        "measurement_positions": measurement_positions,
        "terminal_assay": {"operation": "measure", "instrument": "final_assay"},
        "workflow_family": _workflow_family(actions),
        "omitted_optional_operations": [],
        "objective": "maximize partition-discovery leaderboard score",
        "metric_ids": list(B3_METRIC_IDS),
    }
    if canonical_json_sha256(actions) != public["action_plan_sha256"]:
        raise ValueError(f"{query['query_id']}: action-plan hash is not self-consistent")
    return public


def select_open_candidate_queries(
    runtime: Mapping[str, Any],
    *,
    candidate_grid_protocol: Mapping[str, Any],
    packet_seed: int,
    candidate_count: int = 8,
) -> list[dict[str, Any]]:
    """Select a truth-blind, coverage-constrained packet from the frozen 128-query pool."""

    if candidate_count != 8:
        raise ValueError("W2-48 pilot requires exactly eight candidates")
    pool = build_b3_candidate_queries(candidate_grid_protocol)
    ordered = sorted(
        pool,
        key=lambda row: (_packet_digest(packet_seed, str(row["query_id"])), str(row["query_id"])),
    )
    selected: list[dict[str, Any]] = []
    pairs: set[str] = set()
    volume_counts: dict[float, int] = {}
    mixing_counts: dict[float, int] = {}
    for row in ordered:
        features = row["feature_values"]
        pair_id = str(row["pair_id"])
        volume = float(features["aqueous_phase_volume_L"])
        mixing = float(features["mix_duration_s"])
        if (
            pair_id in pairs
            or volume_counts.get(volume, 0) >= 2
            or mixing_counts.get(mixing, 0) >= 4
        ):
            continue
        selected.append(deepcopy(dict(row)))
        pairs.add(pair_id)
        volume_counts[volume] = volume_counts.get(volume, 0) + 1
        mixing_counts[mixing] = mixing_counts.get(mixing, 0) + 1
        if len(selected) == candidate_count:
            break
    if len(selected) != candidate_count or len(pairs) != 8:
        raise ValueError(
            "packet-hash permutation could not satisfy the frozen pair/volume/mixing coverage"
        )
    if sorted(volume_counts.values()) != [2, 2, 2, 2]:
        raise ValueError("candidate packet volume coverage is not two per regime")
    if sorted(mixing_counts.values()) != [4, 4]:
        raise ValueError("candidate packet mixing coverage is not four per regime")
    return selected


def compile_public_candidate_packet(
    runtime: Mapping[str, Any],
    *,
    candidate_grid_protocol: Mapping[str, Any],
    packet_seed: int,
    candidate_count: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    queries = select_open_candidate_queries(
        runtime,
        candidate_grid_protocol=candidate_grid_protocol,
        packet_seed=packet_seed,
        candidate_count=candidate_count,
    )
    public: list[dict[str, Any]] = []
    compiled_by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        compiled = compile_evaluator_truth_query(runtime, query)
        row = _public_action_plan(query, compiled)
        compiled_by_id[str(query["query_id"])] = compiled
        public.append(row)
    return public, compiled_by_id


def build_open_terminal_contract(
    *,
    study_id: str,
    world_seed: int,
    public_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(public_candidates) != 8:
        raise ValueError("open terminal contract requires eight candidates")
    rows = [deepcopy(dict(row)) for row in public_candidates]
    ids = [str(row.get("query_id")) for row in rows]
    if len(set(ids)) != 8 or any(not row.get("action_plan") for row in rows):
        raise ValueError("open terminal contract candidate identity/action coverage is invalid")
    contract: dict[str, Any] = {
        "schema_version": OPEN_TERMINAL_SCHEMA,
        "protocol_version": OPEN_PROTOCOL_VERSION,
        "readout_id": f"{study_id}--seed{world_seed}",
        "task_id": "partition-discovery",
        "selection_mode": "rank_all_select_one",
        "prediction_mode": "ranking_only",
        "reveal_gate": "campaign_terminal_and_all_belief_checkpoints_committed",
        "metric_ids": list(B3_METRIC_IDS),
        "candidate_queries": rows,
        "candidate_outcomes_included": False,
        "hidden_ranks_included": False,
        "additional_evidence_included": False,
        "public_plan_contract": "complete_action_plan_outcome_blind_v0.1",
        "no_implicit_defaults": True,
        "public_plan_hash_equals_truth_plan_hash": True,
        "truth_plan_hash_equals_executed_plan_hash": True,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def validate_public_truth_binding(
    public_candidates: Sequence[Mapping[str, Any]],
    compiled_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for row in public_candidates:
        query_id = str(row.get("query_id"))
        compiled = compiled_by_id.get(query_id)
        if compiled is None:
            errors.append(f"missing compiled truth plan for {query_id}")
            continue
        public_hash = str(row.get("action_plan_sha256"))
        truth_hash = str(compiled.get("action_plan_sha256"))
        if public_hash != truth_hash:
            errors.append(f"public/truth action-plan hash mismatch for {query_id}")
        if row.get("action_plan") != compiled.get("action_plan"):
            errors.append(f"public/truth action-plan content mismatch for {query_id}")
        if row.get("metric_ids") != list(B3_METRIC_IDS):
            errors.append(f"metric coverage drift for {query_id}")
    return errors


__all__ = [
    "ARMS",
    "OPEN_PROTOCOL_VERSION",
    "OPEN_TERMINAL_SCHEMA",
    "build_open_terminal_contract",
    "compile_public_candidate_packet",
    "select_open_candidate_queries",
    "validate_public_truth_binding",
]
