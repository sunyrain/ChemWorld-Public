"""A-S Study B4 law-guided decision assay."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, write_json_atomic
from chemworld.eval.work_ii_reviewer_followup import (
    B3_FAMILIES,
    B3_METRIC_IDS,
    _report_truth,
    _truth_report,
    build_b3_candidate_queries,
)

B4_PROTOCOL_VERSION = "chemworld-work-ii-as-study-b4-law-guided-decision-protocol-0.1"
B4_MANIFEST_VERSION = "chemworld-work-ii-as-study-b4-law-guided-decision-manifest-0.1"
B4_CELL_VERSION = "chemworld-work-ii-as-study-b4-law-guided-decision-cell-result-0.1"
B4_SUMMARY_VERSION = "chemworld-work-ii-as-study-b4-law-guided-decision-summary-0.1"
B4_TRUTH_VERSION = "chemworld-work-ii-as-study-b4-law-guided-decision-truth-0.1"
B4_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
RETAIN_INCUMBENT = "RETAIN_INCUMBENT"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _protocol(protocol_path: str | Path, root: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(protocol_path)
    path = path if path.is_absolute() else root / path
    protocol = _load_object(path)
    if protocol.get("schema_version") != B4_PROTOCOL_VERSION:
        raise ValueError("unsupported A-S Study B4 protocol version")
    if protocol.get("arms") != list(B4_ARMS):
        raise ValueError("A-S Study B4 arm order drifted")
    if protocol.get("metric_ids") != list(B3_METRIC_IDS):
        raise ValueError("A-S Study B4 metric roster drifted")
    positions = protocol.get("action_rank_positions")
    if positions != [1, 18, 35, 52, 69, 86, 103, 120]:
        raise ValueError("A-S Study B4 action-rank positions drifted")
    return path, protocol


def _initial_model(arm: str) -> dict[str, Any]:
    if arm == "opaque":
        return {
            "availability": "opaque_for_target_locus",
            "mechanism_family": None,
            "reference_exponent": None,
            "confidence": 0.7,
            "scope_limit": "Public evidence is authoritative.",
        }
    if arm == "aligned_nominal":
        return {
            "availability": "supplied_incomplete_executable_law",
            "mechanism_family": "FAMILY_B_POWER",
            "reference_exponent": 1.75,
            "confidence": 0.7,
            "scope_limit": "This is an incomplete local law. Public evidence is authoritative.",
        }
    if arm == "misindexed_nominal":
        return {
            "availability": "supplied_incomplete_executable_law",
            "mechanism_family": "FAMILY_A_LINEAR",
            "reference_exponent": 1.0,
            "confidence": 0.7,
            "scope_limit": "This is an incomplete local law. Public evidence is authoritative.",
        }
    raise ValueError(f"unknown A-S Study B4 arm: {arm}")


def _public_query(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(query["query_id"]),
        "nominal_pair_id": str(query["pair_id"]),
        "reference_partition_coefficient": float(query["reference_partition_coefficient"]),
        "feature_values": deepcopy(dict(query["feature_values"])),
        "metric_ids": list(B3_METRIC_IDS),
    }


def _select_ranked_actions(
    candidates: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, float]],
    positions: Sequence[int],
    *,
    minimum_pair_count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (-float(truth[str(item["query_id"])]["score"]), str(item["query_id"])),
    )
    rank_by_id = {str(item["query_id"]): index for index, item in enumerate(ordered, start=1)}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_pairs: set[str] = set()
    for target_position in positions:
        require_new_pair = len(selected_pairs) < minimum_pair_count
        eligible = [
            item
            for item in ordered
            if str(item["query_id"]) not in selected_ids
            and (not require_new_pair or str(item["pair_id"]) not in selected_pairs)
        ]
        if not eligible:
            eligible = [item for item in ordered if str(item["query_id"]) not in selected_ids]
        chosen = min(
            eligible,
            key=lambda item: (
                abs(rank_by_id[str(item["query_id"])] - int(target_position)),
                rank_by_id[str(item["query_id"])],
                str(item["query_id"]),
            ),
        )
        query_id = str(chosen["query_id"])
        selected_ids.add(query_id)
        selected_pairs.add(str(chosen["pair_id"]))
        selected.append(
            {
                "query": deepcopy(dict(chosen)),
                "pool_rank": rank_by_id[query_id],
                "target_rank_position": int(target_position),
                "truth": deepcopy(dict(truth[query_id])),
            }
        )
    if len(selected) != 8 or len(selected_pairs) < minimum_pair_count:
        raise ValueError("A-S Study B4 action generator failed its denominator or pair gate")
    return selected


def _truth_liveness(
    progress: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase: str,
    world_seed: int,
    law_id: str,
    current_unit: int,
    total_units: int,
    query_count: int,
) -> Callable[[float], None] | None:
    if progress is None:
        return None
    return lambda elapsed_s: progress(
        {
            "stage": f"b4_{phase}_liveness",
            "world_seed": world_seed,
            "law_id": law_id,
            "current_unit": current_unit,
            "total_units": total_units,
            "query_count_in_current_unit": query_count,
            "elapsed_s": elapsed_s,
        }
    )


def prepare_b4(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    target = Path(output_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    runtime = _load_object(_resolve(root, protocol["runtime_config"], field="runtime_config"))
    grid_protocol = _load_object(
        _resolve(root, protocol["candidate_grid_source"], field="candidate_grid_source")
    )
    candidates = build_b3_candidate_queries(grid_protocol)
    by_id = {str(item["query_id"]): item for item in candidates}
    evidence_ids = [str(item) for item in protocol["evidence_query_ids"]]
    if len(evidence_ids) != 8 or len(set(evidence_ids)) != 8:
        raise ValueError("A-S Study B4 evidence roster must contain eight unique queries")
    if any(query_id not in by_id for query_id in evidence_ids):
        raise ValueError("A-S Study B4 evidence roster is outside the candidate grid")
    evidence_queries = [by_id[query_id] for query_id in evidence_ids]
    action_pool = [item for item in candidates if str(item["query_id"]) not in evidence_ids]
    if len(action_pool) != 120:
        raise ValueError("A-S Study B4 action-pool denominator differs from 120")
    seeds = [int(item) for item in protocol["fresh_public_world_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("A-S Study B4 requires five unique fresh worlds")

    world_records: list[dict[str, Any]] = []
    total_units = len(seeds) * 2
    completed_units = 0
    completed_truth_queries = 0
    for seed in seeds:
        cluster_id = f"A_S_B4--partition-discovery--seed{seed}"
        linear_report = _truth_report(
            runtime=runtime,
            queries=evidence_queries,
            exponent=1.0,
            world_seed=seed,
            cluster_id=cluster_id,
            output_root=target / "truth" / "linear-evidence" / cluster_id,
            liveness=_truth_liveness(
                progress,
                phase="fresh_truth",
                world_seed=seed,
                law_id="linear_evidence",
                current_unit=completed_units + 1,
                total_units=total_units,
                query_count=len(evidence_queries),
            ),
        )
        completed_units += 1
        completed_truth_queries += len(evidence_queries)
        if progress is not None:
            progress(
                {
                    "stage": "b4_fresh_truth_progress",
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "completed_truth_queries": completed_truth_queries,
                    "total_truth_queries": 680,
                }
            )
        power_report = _truth_report(
            runtime=runtime,
            queries=candidates,
            exponent=1.75,
            world_seed=seed,
            cluster_id=cluster_id,
            output_root=target / "truth" / "power-full-grid" / cluster_id,
            liveness=_truth_liveness(
                progress,
                phase="fresh_truth",
                world_seed=seed,
                law_id="power_full_grid",
                current_unit=completed_units + 1,
                total_units=total_units,
                query_count=len(candidates),
            ),
        )
        completed_units += 1
        completed_truth_queries += len(candidates)
        linear_truth = _report_truth(linear_report)
        power_truth = _report_truth(power_report)
        selected = _select_ranked_actions(
            action_pool,
            power_truth,
            protocol["action_rank_positions"],
            minimum_pair_count=int(protocol["minimum_action_pair_count"]),
        )
        incumbent = max(power_truth[query_id]["score"] for query_id in evidence_ids)
        best_candidate = max(float(item["truth"]["score"]) for item in selected)
        opportunity = best_candidate >= incumbent + float(protocol["improvement_margin"])
        world_records.append(
            {
                "world_seed": seed,
                "cluster_id": cluster_id,
                "evidence_incumbent_score": incumbent,
                "best_candidate_score": best_candidate,
                "improvement_opportunity": opportunity,
                "oracle_policy": "execute_candidate" if opportunity else "retain_incumbent",
                "linear_evidence_truth": {
                    query_id: linear_truth[query_id] for query_id in evidence_ids
                },
                "power_evidence_truth": {
                    query_id: power_truth[query_id] for query_id in evidence_ids
                },
                "selected_actions": selected,
                "power_report_sha256": power_report["report_sha256"],
                "linear_report_sha256": linear_report["report_sha256"],
            }
        )
        if progress is not None:
            progress(
                {
                    "stage": "b4_fresh_truth_progress",
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "completed_truth_queries": completed_truth_queries,
                    "total_truth_queries": 680,
                }
            )

    truth_manifest: dict[str, Any] = {
        "schema_version": B4_TRUTH_VERSION,
        "study_id": protocol["study_id"],
        "status": "preflight_passed",
        "fresh_world_count": 5,
        "truth_execution_count": 680,
        "exact_replay_query_count": 680,
        "provider_call_count": 0,
        "participant_session_count": 0,
        "evidence_query_count_per_world": 8,
        "action_pool_query_count_per_world": 120,
        "presented_action_count_per_world": 8,
        "improvement_opportunity_world_count": sum(
            item["improvement_opportunity"] for item in world_records
        ),
        "worlds": world_records,
    }
    truth_manifest["truth_manifest_sha256"] = canonical_json_sha256(truth_manifest)
    write_json_atomic(target / "truth_manifest.json", truth_manifest)
    manifest = build_b4_manifest(
        protocol_file,
        repository_root=root,
        output_root=target,
    )
    write_json_atomic(target / "input_manifest.json", manifest)
    return manifest


def build_b4_manifest(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    grid_protocol = _load_object(
        _resolve(root, protocol["candidate_grid_source"], field="candidate_grid_source")
    )
    by_id = {
        str(item["query_id"]): item for item in build_b3_candidate_queries(grid_protocol)
    }
    truth_manifest = _load_object(Path(output_root).resolve() / "truth_manifest.json")
    if truth_manifest.get("status") != "preflight_passed":
        raise ValueError("A-S Study B4 truth preflight is unavailable")
    cells: list[dict[str, Any]] = []
    cluster_packets: list[dict[str, Any]] = []
    for world_index, world in enumerate(truth_manifest["worlds"]):
        evidence = []
        for query_id in protocol["evidence_query_ids"]:
            query = by_id[str(query_id)]
            evidence.append(
                {
                    **_public_query(query),
                    "reference_linear_observations": deepcopy(
                        world["linear_evidence_truth"][str(query_id)]
                    ),
                    "target_observations": deepcopy(
                        world["power_evidence_truth"][str(query_id)]
                    ),
                }
            )
        action_queries = [_public_query(item["query"]) for item in world["selected_actions"]]
        action_truth = {
            str(item["query"]["query_id"]): deepcopy(item["truth"])
            for item in world["selected_actions"]
        }
        hidden_ranks = {
            str(item["query"]["query_id"]): int(index)
            for index, item in enumerate(
                sorted(
                    world["selected_actions"],
                    key=lambda row: (
                        -float(row["truth"]["score"]),
                        str(row["query"]["query_id"]),
                    ),
                ),
                start=1,
            )
        }
        public_packet = {
            "schema_version": "chemworld-work-ii-as-study-b4-public-packet-0.1",
            "cluster_id": world["cluster_id"],
            "task_id": "partition-discovery",
            "metric_range": [0.0, 1.0],
            "candidate_mechanism_families": [
                {
                    "mechanism_family": "FAMILY_A_LINEAR",
                    "anonymous_formula": "D_effective = D_reference * process_factor",
                },
                {
                    "mechanism_family": "FAMILY_B_POWER",
                    "anonymous_formula": (
                        "D_effective = D_reference ^ alpha * process_factor; alpha is estimated"
                    ),
                },
                {
                    "mechanism_family": "FAMILY_C_SATURATING",
                    "anonymous_formula": (
                        "D_effective = (a + b*D_reference/(c + D_reference)) * process_factor"
                    ),
                },
                {
                    "mechanism_family": "FAMILY_D_CONSTANT",
                    "anonymous_formula": "observable endpoint is constant across D_reference",
                },
            ],
            "evidence": evidence,
            "unseen_action_candidates": action_queries,
            "decision_options": ["execute_candidate", "retain_incumbent"],
            "improvement_margin": float(protocol["improvement_margin"]),
        }
        packet_hash = canonical_json_sha256(public_packet)
        cluster_packets.append(
            {
                "cluster_id": world["cluster_id"],
                "world_seed": world["world_seed"],
                "public_packet_sha256": packet_hash,
                "evidence_incumbent_score": world["evidence_incumbent_score"],
                "best_candidate_score": world["best_candidate_score"],
                "improvement_opportunity": world["improvement_opportunity"],
                "oracle_policy": world["oracle_policy"],
            }
        )
        rotated = [B4_ARMS[(world_index + offset) % 3] for offset in range(3)]
        for arm in rotated:
            cells.append(
                {
                    "cell_index": len(cells) + 1,
                    "study_id": protocol["study_id"],
                    "cell_id": f"{world['cluster_id']}--{arm}",
                    "cluster_id": world["cluster_id"],
                    "locus": "A_S_B4",
                    "task_id": "partition-discovery",
                    "world_seed": world["world_seed"],
                    "arm": arm,
                    "initial_world_model": _initial_model(arm),
                    "public_packet": deepcopy(public_packet),
                    "public_packet_sha256": packet_hash,
                    "scoring_truth": deepcopy(action_truth),
                    "hidden_action_ranks": hidden_ranks,
                    "evidence_incumbent_score": float(world["evidence_incumbent_score"]),
                    "best_candidate_score": float(world["best_candidate_score"]),
                    "improvement_opportunity": bool(world["improvement_opportunity"]),
                    "oracle_policy": str(world["oracle_policy"]),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": B4_MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "provider": deepcopy(protocol["provider"]),
        "execution": deepcopy(protocol["execution"]),
        "arms": list(B4_ARMS),
        "cell_count": 15,
        "cluster_count": 5,
        "scoring_term_count": 32,
        "participant_physical_experiment_count": 0,
        "truth_manifest_sha256": truth_manifest["truth_manifest_sha256"],
        "cluster_packets": cluster_packets,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def b4_output_schema(
    action_queries: Sequence[Mapping[str, Any]], *, stage: str
) -> dict[str, Any]:
    if stage not in {"pre", "post"}:
        raise ValueError("A-S Study B4 stage must be pre or post")
    query_ids = [str(item["query_id"]) for item in action_queries]
    properties: dict[str, Any] = {
        "status": {"type": "string", "const": f"{stage}_submission_complete"},
        "mechanism_family": {"type": "string", "enum": list(B3_FAMILIES)},
        "estimated_reference_exponent": {
            "type": "number",
            "minimum": 0.25,
            "maximum": 3.0,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "typed_law": {
            "type": "object",
            "additionalProperties": False,
            "required": ["law_type", "mechanism_family", "reference_exponent"],
            "properties": {
                "law_type": {"type": "string", "const": "reference_coefficient_power"},
                "mechanism_family": {"type": "string", "enum": list(B3_FAMILIES)},
                "reference_exponent": {
                    "type": "number",
                    "minimum": 0.25,
                    "maximum": 3.0,
                },
            },
        },
        "predictions": {
            "type": "array",
            "minItems": len(query_ids),
            "maxItems": len(query_ids),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query_id", "metrics"],
                "properties": {
                    "query_id": {"type": "string", "enum": query_ids},
                    "metrics": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(B3_METRIC_IDS),
                        "properties": {
                            metric_id: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            for metric_id in B3_METRIC_IDS
                        },
                    },
                },
            },
        },
        "model_summary": {"type": "string", "maxLength": 1200},
    }
    required = [
        "status",
        "mechanism_family",
        "estimated_reference_exponent",
        "confidence",
        "typed_law",
        "predictions",
        "model_summary",
    ]
    if stage == "post":
        properties.update(
            {
                "decision_type": {
                    "type": "string",
                    "enum": ["execute_candidate", "retain_incumbent"],
                },
                "selected_action_query_id": {
                    "type": "string",
                    "enum": [*query_ids, RETAIN_INCUMBENT],
                },
                "evidence_assessment": {"type": "string", "maxLength": 1200},
            }
        )
        required.extend(
            ["decision_type", "selected_action_query_id", "evidence_assessment"]
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def validate_b4_payload(
    payload: Mapping[str, Any],
    action_queries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> list[str]:
    from chemworld.eval.work_ii_reviewer_followup import validate_b3_payload

    errors = validate_b3_payload(payload, action_queries, stage="pre")
    expected_status = f"{stage}_submission_complete"
    errors = [error for error in errors if "status is invalid" not in error]
    if payload.get("status") != expected_status:
        errors.append(f"{stage} status is invalid")
    if stage == "post":
        query_ids = {str(item["query_id"]) for item in action_queries}
        decision_type = payload.get("decision_type")
        selected = payload.get("selected_action_query_id")
        if decision_type == "execute_candidate" and selected not in query_ids:
            errors.append("post execute decision lacks a candidate query ID")
        if decision_type == "retain_incumbent" and selected != RETAIN_INCUMBENT:
            errors.append("post retain decision must use RETAIN_INCUMBENT")
        if decision_type not in {"execute_candidate", "retain_incumbent"}:
            errors.append("post decision type is invalid")
    return errors


def summarize_b4_results(
    manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = {str(item["cell_id"]) for item in manifest["cells"]}
    observed_ids = {str(item.get("cell_id")) for item in results}
    completed = [item for item in results if item.get("status") == "completed"]
    failures = [item for item in results if item.get("status") != "completed"]
    rows: list[dict[str, Any]] = []
    by_cluster: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in completed:
        post = result["post_submission"]
        decision = result["decision_evaluation"]
        row = {
            "cell_id": result["cell_id"],
            "cluster_id": result["cluster_id"],
            "world_seed": result["world_seed"],
            "arm": result["arm"],
            "pre_error": float(result["scores"]["pre"]["mean_normalized_absolute_error"]),
            "post_error": float(result["scores"]["post"]["mean_normalized_absolute_error"]),
            "update_gain": float(result["scores"]["pre"]["mean_normalized_absolute_error"])
            - float(result["scores"]["post"]["mean_normalized_absolute_error"]),
            "post_family": post["mechanism_family"],
            "post_exponent": float(post["estimated_reference_exponent"]),
            "post_exponent_absolute_error": abs(
                float(post["estimated_reference_exponent"]) - 1.75
            ),
            **deepcopy(dict(decision)),
        }
        rows.append(row)
        by_cluster[str(row["cluster_id"])][str(row["arm"])] = row
    cluster_rows = [
        {
            "cluster_id": cluster_id,
            "world_seed": arms["opaque"]["world_seed"],
            "normalized_policy_regret_by_arm": {
                arm: arms[arm]["normalized_policy_regret"] for arm in B4_ARMS
            },
            "selected_rank_by_arm": {arm: arms[arm]["selected_rank"] for arm in B4_ARMS},
            "oracle_policy": arms["opaque"]["oracle_policy"],
        }
        for cluster_id, arms in sorted(by_cluster.items())
        if set(arms) == set(B4_ARMS)
    ]
    by_arm: dict[str, Any] = {}
    for arm in B4_ARMS:
        members = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "completed_cell_count": len(members),
            "mean_post_error": mean(row["post_error"] for row in members) if members else None,
            "exact_family_recovery_count": sum(
                row["post_family"] == "FAMILY_B_POWER" for row in members
            ),
            "exponent_within_0_10_count": sum(
                row["post_exponent_absolute_error"] <= 0.10 for row in members
            ),
            "top1_candidate_count": sum(row["top1_candidate_selected"] for row in members),
            "mean_selected_rank": mean(
                row["selected_rank"] for row in members if row["selected_rank"] is not None
            )
            if any(row["selected_rank"] is not None for row in members)
            else None,
            "mean_normalized_policy_regret": mean(
                row["normalized_policy_regret"] for row in members
            )
            if members
            else None,
            "correct_abstention_count": sum(row["correct_abstention"] for row in members),
            "false_abstention_count": sum(row["false_abstention"] for row in members),
            "false_execution_count": sum(row["false_execution"] for row in members),
            "positive_gain_over_incumbent_count": sum(
                row["gain_over_incumbent"] is not None
                and row["gain_over_incumbent"] > 0.0
                for row in members
            ),
        }
    return {
        "schema_version": B4_SUMMARY_VERSION,
        "study_id": manifest["study_id"],
        "status": (
            "completed"
            if observed_ids == expected_ids and len(completed) == len(expected_ids)
            else "incomplete"
        ),
        "scheduled_cell_count": len(expected_ids),
        "observed_cell_count": len(observed_ids),
        "completed_cell_count": len(completed),
        "failed_cell_count": len(failures),
        "complete_cluster_count": len(cluster_rows),
        "participant_physical_experiment_count": 0,
        "by_arm": by_arm,
        "cell_rows": sorted(rows, key=lambda row: row["cell_id"]),
        "cluster_rows": cluster_rows,
        "failures": [
            {"cell_id": item.get("cell_id"), "failure": item.get("failure")}
            for item in failures
        ],
    }


def evaluate_b4_decision(
    cell: Mapping[str, Any], post_payload: Mapping[str, Any]
) -> dict[str, Any]:
    truth = cell["scoring_truth"]
    ranks = cell["hidden_action_ranks"]
    candidate_scores = {query_id: float(metrics["score"]) for query_id, metrics in truth.items()}
    incumbent = float(cell["evidence_incumbent_score"])
    best_query_id = min(
        candidate_scores,
        key=lambda query_id: (-candidate_scores[query_id], query_id),
    )
    best_candidate = candidate_scores[best_query_id]
    worst_policy_score = min(incumbent, min(candidate_scores.values()))
    oracle_policy_score = max(incumbent, best_candidate)
    decision_type = str(post_payload["decision_type"])
    selected_query_id = str(post_payload["selected_action_query_id"])
    if decision_type == "retain_incumbent":
        selected_policy_score = incumbent
        selected_rank = None
        gain = None
    else:
        selected_policy_score = candidate_scores[selected_query_id]
        selected_rank = int(ranks[selected_query_id])
        gain = selected_policy_score - incumbent
    regret = oracle_policy_score - selected_policy_score
    scale = oracle_policy_score - worst_policy_score
    normalized_regret = 0.0 if scale <= 1.0e-12 else regret / scale
    oracle_policy = str(cell["oracle_policy"])
    return {
        "decision_type": decision_type,
        "selected_action_query_id": selected_query_id,
        "selected_rank": selected_rank,
        "selected_rank_percentile": (
            None if selected_rank is None else 1.0 - (selected_rank - 1) / 7.0
        ),
        "top1_candidate_selected": selected_query_id == best_query_id,
        "best_candidate_query_id": best_query_id,
        "best_candidate_score": best_candidate,
        "evidence_incumbent_score": incumbent,
        "selected_policy_score": selected_policy_score,
        "gain_over_incumbent": gain,
        "oracle_policy": oracle_policy,
        "oracle_policy_score": oracle_policy_score,
        "raw_policy_regret": regret,
        "normalized_policy_regret": normalized_regret,
        "random_candidate_mean_score": mean(candidate_scores.values()),
        "selected_minus_random_candidate_mean": selected_policy_score
        - mean(candidate_scores.values()),
        "correct_abstention": (
            oracle_policy == "retain_incumbent" and decision_type == "retain_incumbent"
        ),
        "false_abstention": (
            oracle_policy == "execute_candidate" and decision_type == "retain_incumbent"
        ),
        "false_execution": (
            oracle_policy == "retain_incumbent" and decision_type == "execute_candidate"
        ),
    }


__all__ = [
    "B4_ARMS",
    "B4_CELL_VERSION",
    "B4_MANIFEST_VERSION",
    "B4_PROTOCOL_VERSION",
    "B4_SUMMARY_VERSION",
    "B4_TRUTH_VERSION",
    "RETAIN_INCUMBENT",
    "b4_output_schema",
    "build_b4_manifest",
    "evaluate_b4_decision",
    "prepare_b4",
    "summarize_b4_results",
    "validate_b4_payload",
]
