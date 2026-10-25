"""Shared terminal decision contracts and retained W2-43 evaluation compatibility.

The retired full-prediction W2-40 launcher and the superseded W2-47 feature-only packet are no
longer current authorities. New multi-world candidate selection belongs to the open-action design
once its full public-plan materializer is implemented.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, write_json_atomic
from chemworld.eval.work_ii_reviewer_followup import (
    B3_METRIC_IDS,
    _report_truth,
    _truth_report,
    build_b3_candidate_queries,
)
from chemworld.eval.work_ii_truth import (
    build_evaluator_truth_plan,
    compile_evaluator_truth_query,
    execute_evaluator_truth_plan,
    validate_evaluator_truth_plan,
    validate_evaluator_truth_report,
)

PROTOCOL_VERSION = "chemworld-work-ii-as-longitudinal-action-readout-protocol-0.1"
TRUTH_VERSION = "chemworld-work-ii-as-longitudinal-action-readout-truth-0.1"
MANIFEST_VERSION = "chemworld-work-ii-as-longitudinal-action-readout-manifest-0.1"
CELL_VERSION = "chemworld-work-ii-as-longitudinal-action-readout-cell-0.1"
SUMMARY_VERSION = "chemworld-work-ii-as-longitudinal-action-readout-summary-0.1"
TERMINAL_CONTRACT_VERSION = "chemworld-work-ii-terminal-action-readout-contract-0.1"

ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
FRESH_WORLD_SEEDS = (153150025, 395988875, 302275745, 481313948, 491681886)
ACTION_RANK_POSITIONS = (1, 18, 35, 52, 69, 86, 103, 120)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _resolve(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_protocol(path: str | Path, *, repository_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(repository_root).resolve()
    protocol_path = Path(path)
    protocol_path = protocol_path if protocol_path.is_absolute() else root / protocol_path
    protocol = _load(protocol_path)
    if protocol.get("status") == "historical_candidate_grid_input_only_retired_unexecuted_design":
        raise ValueError(
            "the retired W2-40 protocol is a historical W2-43 grid input, not a current launcher"
        )
    if protocol.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported longitudinal action-readout protocol")
    if tuple(protocol.get("fresh_world_seeds", [])) != FRESH_WORLD_SEEDS:
        raise ValueError("longitudinal action-readout world seeds drifted")
    if tuple(protocol.get("arms", [])) != ARMS:
        raise ValueError("longitudinal action-readout arms drifted")
    if int(protocol.get("campaign_complete_experiments", -1)) != 12:
        raise ValueError("longitudinal action readout requires twelve experiments")
    if protocol.get("checkpoint_complete_experiments") != [0, 3, 6, 9, 12]:
        raise ValueError("longitudinal action-readout checkpoint schedule drifted")
    if tuple(protocol.get("action_rank_positions", [])) != ACTION_RANK_POSITIONS:
        raise ValueError("longitudinal action-readout rank positions drifted")
    if protocol.get("metric_ids") != list(B3_METRIC_IDS):
        raise ValueError("longitudinal action-readout metrics drifted")
    if not math.isclose(
        float(protocol.get("maximum_adequate_law_normalized_mae", math.nan)),
        0.05,
        abs_tol=1.0e-12,
    ):
        raise ValueError("longitudinal action-readout law adequacy threshold drifted")
    if protocol.get("selection_mode") != "rank_all_select_one":
        raise ValueError("longitudinal action-readout selection mode drifted")
    if protocol.get("terminal_reveal_gate") != (
        "campaign_terminal_and_all_belief_checkpoints_committed"
    ):
        raise ValueError("longitudinal action-readout reveal gate drifted")
    return protocol_path, protocol


def build_candidate_queries(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the reserved 128-operation terminal pool without B3 evidence semantics."""

    queries = build_b3_candidate_queries(protocol)
    result: list[dict[str, Any]] = []
    for query in queries:
        row = deepcopy(dict(query))
        query_id = str(row["query_id"])
        if not query_id.startswith("b3-"):
            raise ValueError("candidate query prefix drifted")
        row["query_id"] = "lar-" + query_id[3:]
        result.append(row)
    if len(result) != 128 or len({row["query_id"] for row in result}) != 128:
        raise ValueError("longitudinal action-readout candidate denominator differs from 128")
    return result


def _public_query(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(query["query_id"]),
        "nominal_pair_id": str(query["pair_id"]),
        "reference_partition_coefficient": float(query["reference_partition_coefficient"]),
        "feature_values": deepcopy(dict(query["feature_values"])),
        "metric_ids": list(B3_METRIC_IDS),
    }


def select_ranked_actions(
    candidates: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, float]],
    *,
    positions: Sequence[int] = ACTION_RANK_POSITIONS,
    minimum_pair_count: int = 4,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (-float(truth[str(item["query_id"])]["score"]), str(item["query_id"])),
    )
    rank_by_id = {str(item["query_id"]): index for index, item in enumerate(ordered, start=1)}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_pairs: set[str] = set()
    for position in positions:
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
                abs(rank_by_id[str(item["query_id"])] - int(position)),
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
                "target_rank_position": int(position),
                "truth": deepcopy(dict(truth[query_id])),
            }
        )
    if len(selected) != 8 or len(selected_pairs) < minimum_pair_count:
        raise ValueError("terminal candidate selection failed its denominator or pair gate")
    return selected


def build_terminal_contract(
    *,
    study_id: str,
    world_seed: int,
    candidates: Sequence[Mapping[str, Any]],
    prediction_mode: str = "full_metrics",
) -> dict[str, Any]:
    if prediction_mode not in {"full_metrics", "ranking_only"}:
        raise ValueError("unsupported terminal action prediction mode")
    contract: dict[str, Any] = {
        "schema_version": TERMINAL_CONTRACT_VERSION,
        "readout_id": f"{study_id}--seed{world_seed}",
        "task_id": "partition-discovery",
        "selection_mode": "rank_all_select_one",
        "prediction_mode": prediction_mode,
        "reveal_gate": "campaign_terminal_and_all_belief_checkpoints_committed",
        "metric_ids": list(B3_METRIC_IDS),
        "candidate_queries": [_public_query(item) for item in candidates],
        "candidate_outcomes_included": False,
        "hidden_ranks_included": False,
        "additional_evidence_included": False,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    validate_terminal_contract(contract)
    return contract


def validate_terminal_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != TERMINAL_CONTRACT_VERSION:
        raise ValueError("terminal action-readout contract version is invalid")
    expected_hash = canonical_json_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    if contract.get("contract_sha256") != expected_hash:
        raise ValueError("terminal action-readout contract hash mismatch")
    candidates = contract.get("candidate_queries")
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError("terminal action-readout contract must contain eight candidates")
    query_ids = [str(item.get("query_id")) for item in candidates if isinstance(item, Mapping)]
    pairs = [str(item.get("nominal_pair_id")) for item in candidates if isinstance(item, Mapping)]
    if len(query_ids) != 8 or len(set(query_ids)) != 8 or len(set(pairs)) < 4:
        raise ValueError("terminal action-readout candidate identity coverage is invalid")
    forbidden = {"truth", "rank", "pool_rank", "target_rank_position", "observations"}
    if any(forbidden.intersection(item) for item in candidates if isinstance(item, Mapping)):
        raise ValueError("terminal action-readout contract leaks hidden candidate information")
    if contract.get("metric_ids") != list(B3_METRIC_IDS):
        raise ValueError("terminal action-readout metric coverage is invalid")
    if contract.get("prediction_mode", "full_metrics") not in {
        "full_metrics",
        "ranking_only",
    }:
        raise ValueError("terminal action-readout prediction mode is invalid")
    if any(
        not isinstance(item, Mapping) or item.get("metric_ids") != list(B3_METRIC_IDS)
        for item in candidates
    ):
        raise ValueError("terminal action candidate metric coverage is invalid")


def _checkpoint_action_hashes(runtime: Mapping[str, Any]) -> set[str]:
    checkpoint = runtime.get("belief_checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    queries = checkpoint.get("held_out_queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("runtime checkpoint queries are unavailable")
    return {
        str(compile_evaluator_truth_query(runtime, query)["action_plan_sha256"])
        for query in queries
        if isinstance(query, Mapping)
    }


def _world_campaign_config(
    runtime: Mapping[str, Any],
    *,
    study_id: str,
    world_seed: int,
    terminal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    config = deepcopy(dict(runtime))
    config["pilot_id"] = f"{study_id}--seed{world_seed}"
    config["world_seed"] = int(world_seed)
    config["observation_noise_namespace"] = study_id
    config["formal_result"] = False
    config["terminal_action_readout"] = deepcopy(dict(terminal_contract))
    prediction_mode = str(terminal_contract.get("prediction_mode", "full_metrics"))
    analysis = dict(config.get("analysis", {}))
    analysis.update(
        {
            "terminal_action_readout_required": True,
            "terminal_action_selection_mode": "rank_all_select_one",
            "terminal_action_candidate_count": 8,
            "terminal_action_metric_count": 32 if prediction_mode == "full_metrics" else 0,
            "terminal_action_prediction_mode": prediction_mode,
        }
    )
    config["analysis"] = analysis
    execution_context = dict(config.get("execution_context", {}))
    execution_context.update(
        {
            "execution_mode": "development",
            "evidence_status": "development_only",
            "release_eligible": False,
            "tested_commit": None,
            "freeze_id": None,
        }
    )
    config["execution_context"] = execution_context
    qualification = dict(config.get("qualification", {}))
    qualification.update(
        {
            "execution_authorized": False,
            "formal_r5_authorized": False,
            "terminal_action_readout_provider_execution_authorized": False,
        }
    )
    config["qualification"] = qualification
    return config


def prepare_longitudinal_action_readout(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute provider-free truth and build frozen per-world campaign configs."""

    root = Path(repository_root).resolve()
    protocol_file, protocol = load_protocol(protocol_path, repository_root=root)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = _load(_resolve(root, protocol["runtime_config"], field="runtime_config"))
    if int(runtime.get("campaign", {}).get("complete_experiments", -1)) != 12:
        raise ValueError("runtime campaign does not contain twelve experiments")
    if runtime.get("campaign", {}).get("checkpoint_complete_experiments") != [0, 3, 6, 9, 12]:
        raise ValueError("runtime checkpoint schedule differs from the protocol")
    candidates = build_candidate_queries(protocol)
    checkpoint_hashes = _checkpoint_action_hashes(runtime)
    candidate_plans = {
        str(query["query_id"]): compile_evaluator_truth_query(runtime, query)
        for query in candidates
    }
    if checkpoint_hashes.intersection(
        str(plan["action_plan_sha256"]) for plan in candidate_plans.values()
    ):
        raise ValueError("terminal candidates collide with campaign checkpoint queries")

    worlds: list[dict[str, Any]] = []
    config_root = output / "campaign-configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(FRESH_WORLD_SEEDS, start=1):
        cluster_id = f"A_S_LAR--partition-discovery--seed{seed}"
        checkpoint_plan = build_evaluator_truth_plan(
            {
                "world_cluster_id": cluster_id,
                "task_id": "partition-discovery",
                "world_seed": seed,
            },
            runtime,
            formal_result=False,
            formal_preflight_sha256=None,
        )
        checkpoint_plan_errors = validate_evaluator_truth_plan(checkpoint_plan)
        if checkpoint_plan_errors:
            raise ValueError(
                f"{cluster_id}: invalid checkpoint truth plan: "
                + "; ".join(checkpoint_plan_errors)
            )
        checkpoint_root = output / "checkpoint-truth" / cluster_id
        if checkpoint_root.exists():
            checkpoint_report = _load(checkpoint_root / "report.json")
        else:
            checkpoint_report = execute_evaluator_truth_plan(
                checkpoint_plan,
                runtime,
                checkpoint_root,
            )
        checkpoint_report_errors = validate_evaluator_truth_report(
            checkpoint_report,
            checkpoint_plan,
        )
        if checkpoint_report_errors or checkpoint_report.get("status") != "completed":
            raise ValueError(
                f"{cluster_id}: invalid checkpoint truth report: "
                + ("; ".join(checkpoint_report_errors) or str(checkpoint_report.get("status")))
            )
        report = _truth_report(
            runtime=runtime,
            queries=candidates,
            exponent=1.75,
            world_seed=seed,
            cluster_id=cluster_id,
            output_root=output / "truth" / cluster_id,
            liveness=(
                (
                    lambda elapsed_s, seed=seed, index=index: progress(
                        {
                            "stage": "lar_truth_liveness",
                            "world_seed": seed,
                            "completed_worlds": index - 1,
                            "total_worlds": 5,
                            "query_count_in_current_world": 128,
                            "elapsed_s": elapsed_s,
                        }
                    )
                )
                if progress is not None
                else None
            ),
        )
        truth = _report_truth(report)
        selected = select_ranked_actions(
            candidates,
            truth,
            positions=protocol["action_rank_positions"],
            minimum_pair_count=int(protocol["minimum_action_pair_count"]),
        )
        selected_queries = [item["query"] for item in selected]
        contract = build_terminal_contract(
            study_id=str(protocol["study_id"]),
            world_seed=seed,
            candidates=selected_queries,
        )
        config = _world_campaign_config(
            runtime,
            study_id=str(protocol["study_id"]),
            world_seed=seed,
            terminal_contract=contract,
        )
        config_path = config_root / f"seed-{seed}.json"
        write_json_atomic(config_path, config)
        selected_truth = {
            str(item["query"]["query_id"]): deepcopy(dict(item["truth"]))
            for item in selected
        }
        selected_plan_hashes = {
            str(item["query"]["query_id"]): str(
                candidate_plans[str(item["query"]["query_id"])]["action_plan_sha256"]
            )
            for item in selected
        }
        presented_ranks = {
            query_id: rank
            for rank, query_id in enumerate(
                sorted(
                    selected_truth,
                    key=lambda query_id: (-float(selected_truth[query_id]["score"]), query_id),
                ),
                start=1,
            )
        }
        worlds.append(
            {
                "world_seed": seed,
                "cluster_id": cluster_id,
                "terminal_action_readout": contract,
                "campaign_config_path": config_path.relative_to(output).as_posix(),
                "candidate_truth": selected_truth,
                "presented_candidate_ranks": presented_ranks,
                "candidate_pool_ranks": {
                    str(item["query"]["query_id"]): int(item["pool_rank"])
                    for item in selected
                },
                "candidate_action_plan_sha256": selected_plan_hashes,
                "checkpoint_truth_plan": checkpoint_plan,
                "checkpoint_truth": deepcopy(dict(checkpoint_report["truth"])),
                "checkpoint_truth_report_sha256": checkpoint_report["report_sha256"],
                "truth_report_sha256": report["report_sha256"],
            }
        )
        if progress is not None:
            progress(
                {
                    "stage": "lar_truth_progress",
                    "completed_worlds": index,
                    "total_worlds": 5,
                    "completed_truth_queries": index * 144,
                    "total_truth_queries": 720,
                }
            )

    truth_manifest: dict[str, Any] = {
        "schema_version": TRUTH_VERSION,
        "study_id": protocol["study_id"],
        "status": "provider_free_preflight_passed",
        "world_count": 5,
        "candidate_pool_query_count_per_world": 128,
        "presented_candidate_count_per_world": 8,
        "terminal_candidate_truth_execution_count": 640,
        "checkpoint_truth_execution_count": 80,
        "truth_execution_count": 720,
        "exact_replay_query_count": 720,
        "provider_call_count": 0,
        "participant_physical_experiment_count": 0,
        "worlds": worlds,
    }
    truth_manifest["truth_manifest_sha256"] = canonical_json_sha256(truth_manifest)
    write_json_atomic(output / "truth_manifest.json", truth_manifest)

    cells = [
        {
            "cell_id": f"{world['cluster_id']}--{arm}",
            "cluster_id": world["cluster_id"],
            "world_seed": world["world_seed"],
            "arm": arm,
            "campaign_config_path": world["campaign_config_path"],
            "terminal_action_readout": deepcopy(world["terminal_action_readout"]),
            "candidate_truth": deepcopy(world["candidate_truth"]),
            "presented_candidate_ranks": deepcopy(world["presented_candidate_ranks"]),
            "candidate_pool_ranks": deepcopy(world["candidate_pool_ranks"]),
            "candidate_action_plan_sha256": deepcopy(
                world["candidate_action_plan_sha256"]
            ),
            "checkpoint_truth_plan": deepcopy(world["checkpoint_truth_plan"]),
            "checkpoint_truth": deepcopy(world["checkpoint_truth"]),
        }
        for world in worlds
        for arm in ARMS
    ]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "status": "prepared_provider_execution_not_authorized",
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "truth_manifest_sha256": truth_manifest["truth_manifest_sha256"],
        "cluster_count": 5,
        "cell_count": 15,
        "campaign_experiment_count_per_cell": 12,
        "participant_physical_experiment_count": 180,
        "terminal_prediction_term_count_per_cell": 32,
        "provider_execution_authorized": False,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json_atomic(output / "input_manifest.json", manifest)
    return manifest


def _prediction_map(recommendation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    predictions = recommendation.get("candidate_predictions")
    if not isinstance(predictions, list):
        raise ValueError("terminal recommendation lacks candidate predictions")
    result: dict[str, dict[str, float]] = {}
    for row in predictions:
        if not isinstance(row, Mapping):
            raise ValueError("terminal prediction row is malformed")
        query_id = row.get("query_id")
        metrics = row.get("metrics")
        if not isinstance(query_id, str) or query_id in result or not isinstance(metrics, Mapping):
            raise ValueError("terminal prediction identity is malformed")
        result[query_id] = {str(key): float(value) for key, value in metrics.items()}
    return result


def evaluate_terminal_readout(
    cell: Mapping[str, Any],
    campaign_summary: Mapping[str, Any],
    *,
    maximum_adequate_law_normalized_mae: float = 0.05,
) -> dict[str, Any]:
    analysis = campaign_summary.get("analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    recommendation = analysis.get("final_recommendation")
    if not isinstance(recommendation, Mapping):
        return {
            "schema_version": CELL_VERSION,
            "cell_id": cell.get("cell_id"),
            "status": "failed_missing_terminal_action_readout",
        }
    truth = cell.get("candidate_truth")
    ranks = cell.get("presented_candidate_ranks")
    action_hashes = cell.get("candidate_action_plan_sha256")
    if not all(isinstance(value, Mapping) for value in (truth, ranks, action_hashes)):
        raise ValueError("cell lacks hidden terminal action truth")
    query_ids = set(map(str, truth))
    terminal_contract = cell.get("terminal_action_readout")
    terminal_contract = terminal_contract if isinstance(terminal_contract, Mapping) else {}
    prediction_mode = str(terminal_contract.get("prediction_mode", "full_metrics"))
    if prediction_mode == "full_metrics":
        predictions = _prediction_map(recommendation)
        if set(predictions) != query_ids:
            raise ValueError("terminal prediction denominator differs from hidden truth")
        terms = [
            abs(
                float(predictions[query_id][metric_id])
                - float(truth[query_id][metric_id])
            )
            for query_id in sorted(query_ids)
            for metric_id in B3_METRIC_IDS
        ]
    elif prediction_mode == "ranking_only":
        if recommendation.get("candidate_predictions") is not None:
            raise ValueError("ranking-only recommendation contains candidate predictions")
        terms = []
    else:
        raise ValueError("terminal action prediction mode is invalid")
    ranking = recommendation.get("ranking")
    selected = recommendation.get("selected_action_query_id")
    if (
        not isinstance(ranking, list)
        or set(map(str, ranking)) != query_ids
        or len(ranking) != len(query_ids)
        or not isinstance(selected, str)
        or selected != ranking[0]
    ):
        raise ValueError("terminal ranking or selection is invalid")
    scores = {query_id: float(truth[query_id]["score"]) for query_id in query_ids}
    best_id = min(scores, key=lambda query_id: (-scores[query_id], query_id))
    best = scores[best_id]
    worst = min(scores.values())
    selected_score = scores[selected]
    regret = best - selected_score
    scale = best - worst
    normalized_regret = 0.0 if scale <= 1.0e-12 else regret / scale
    executed_hashes = {
        str(item.get("recipe_sha256"))
        for item in analysis.get("experiments", [])
        if isinstance(item, Mapping) and isinstance(item.get("recipe_sha256"), str)
    }
    overlapping_query_ids = sorted(
        query_id
        for query_id, digest in action_hashes.items()
        if str(digest) in executed_hashes
    )
    snapshots = analysis.get("belief_snapshots")
    snapshots = snapshots if isinstance(snapshots, list) else []
    final_snapshot = snapshots[-1] if snapshots and isinstance(snapshots[-1], Mapping) else None
    law = campaign_summary.get("law_summary_evaluation")
    law = law if isinstance(law, Mapping) else {}
    law_mae = law.get("normalized_mae")
    law_adequate = (
        isinstance(law_mae, int | float)
        and not isinstance(law_mae, bool)
        and math.isfinite(float(law_mae))
        and float(law_mae) <= maximum_adequate_law_normalized_mae
    )
    action_correct = selected == best_id
    mechanism_action_category = (
        ("adequate_law" if law_adequate else "inadequate_law")
        + "__"
        + ("correct_action" if action_correct else "wrong_action")
        if law.get("status") == "evaluated"
        else "law_not_evaluated"
    )
    provider_receipts = campaign_summary.get("provider_receipts")
    provider_receipts = provider_receipts if isinstance(provider_receipts, list) else []
    same_thread = (
        len(
            {
                str(receipt.get("thread_id"))
                for receipt in provider_receipts
                if isinstance(receipt, Mapping) and isinstance(receipt.get("thread_id"), str)
            }
        )
        == 1
    )
    return {
        "schema_version": CELL_VERSION,
        "cell_id": cell.get("cell_id"),
        "cluster_id": cell.get("cluster_id"),
        "world_seed": cell.get("world_seed"),
        "arm": cell.get("arm"),
        "status": (
            "completed_uncontaminated"
            if not overlapping_query_ids
            and campaign_summary.get("completed") is True
            and final_snapshot is not None
            and final_snapshot.get("stage") == "final"
            else "completed_contaminated"
            if overlapping_query_ids
            else "incomplete_campaign_or_checkpoint"
        ),
        "same_thread": same_thread,
        "campaign_complete_experiment_count": analysis.get("complete_experiment_count"),
        "final_checkpoint_present": (
            final_snapshot is not None and final_snapshot.get("stage") == "final"
        ),
        "terminal_prediction_term_count": len(terms),
        "terminal_prediction_mae": mean(terms) if terms else None,
        "terminal_prediction_mode": prediction_mode,
        "participant_ranking": [str(query_id) for query_id in ranking],
        "participant_ranking_true_ranks": [
            int(ranks[str(query_id)]) for query_id in ranking
        ],
        "candidate_true_order": sorted(query_ids, key=lambda query_id: int(ranks[query_id])),
        "selected_action_query_id": selected,
        "selected_rank": int(ranks[selected]),
        "top1_selected": action_correct,
        "best_action_query_id": best_id,
        "selected_score": selected_score,
        "best_score": best,
        "raw_regret": regret,
        "normalized_regret": normalized_regret,
        "random_candidate_mean_score": mean(scores.values()),
        "selected_minus_random_candidate_mean": selected_score - mean(scores.values()),
        "candidate_overlap_count": len(overlapping_query_ids),
        "candidate_overlap_query_ids": overlapping_query_ids,
        "law_summary_status": law.get("status"),
        "law_normalized_mae": law_mae,
        "law_adequate": law_adequate if law.get("status") == "evaluated" else None,
        "mechanism_action_category": mechanism_action_category,
    }


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        members = [row for row in results if row.get("arm") == arm]
        eligible = [row for row in members if row.get("status") == "completed_uncontaminated"]
        by_arm[arm] = {
            "scheduled_cell_count": len(members),
            "eligible_cell_count": len(eligible),
            "top1_count": sum(row.get("top1_selected") is True for row in eligible),
            "mean_selected_rank": (
                mean(float(row["selected_rank"]) for row in eligible) if eligible else None
            ),
            "mean_normalized_regret": (
                mean(float(row["normalized_regret"]) for row in eligible) if eligible else None
            ),
            "mean_terminal_prediction_mae": (
                mean(
                    float(row["terminal_prediction_mae"])
                    for row in eligible
                    if isinstance(row.get("terminal_prediction_mae"), int | float)
                    and not isinstance(row.get("terminal_prediction_mae"), bool)
                )
                if any(
                    isinstance(row.get("terminal_prediction_mae"), int | float)
                    and not isinstance(row.get("terminal_prediction_mae"), bool)
                    for row in eligible
                )
                else None
            ),
        }
    categories: dict[str, int] = defaultdict(int)
    for row in results:
        categories[str(row.get("mechanism_action_category"))] += 1
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "scheduled_cell_count": len(results),
        "eligible_cell_count": sum(
            row.get("status") == "completed_uncontaminated" for row in results
        ),
        "failed_or_ineligible_cell_count": sum(
            row.get("status") != "completed_uncontaminated" for row in results
        ),
        "participant_physical_experiment_count": sum(
            int(row.get("campaign_complete_experiment_count") or 0) for row in results
        ),
        "by_arm": by_arm,
        "mechanism_action_categories": dict(sorted(categories.items())),
        "cell_rows": [deepcopy(dict(row)) for row in results],
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return summary


__all__ = [
    "ARMS",
    "SUMMARY_VERSION",
    "TERMINAL_CONTRACT_VERSION",
    "build_terminal_contract",
    "evaluate_terminal_readout",
    "summarize_results",
    "validate_terminal_contract",
]
