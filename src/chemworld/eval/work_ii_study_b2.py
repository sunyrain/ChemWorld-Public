"""A-S Study B2 with diagnostic phase-process evidence.

The source paired-law qualification reports are used only to freeze and verify
an outcome-independent query roster. Participant-visible evidence is generated
again under the registered public C2 seeds and power-law intervention.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256, write_json_atomic
from chemworld.eval.work_ii_constitutive_structural_qualification import report_sha256
from chemworld.eval.work_ii_study_b import summarize_study_b_results
from chemworld.eval.work_ii_truth import (
    build_evaluator_truth_plan,
    execute_evaluator_truth_plan,
    validate_evaluator_truth_plan,
    validate_evaluator_truth_report,
)

STUDY_B2_PROTOCOL_VERSION = "chemworld-work-ii-as-study-b2-phase-process-protocol-0.1"
STUDY_B2_TRUTH_MANIFEST_VERSION = "chemworld-work-ii-as-study-b2-truth-manifest-0.1"
STUDY_B2_MANIFEST_VERSION = "chemworld-work-ii-as-study-b2-input-manifest-0.1"
STUDY_B2_SUMMARY_VERSION = "chemworld-work-ii-as-study-b2-summary-0.1"
ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")


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
    if protocol.get("schema_version") != STUDY_B2_PROTOCOL_VERSION:
        raise ValueError("unsupported A-S Study B2 protocol version")
    if protocol.get("arms") != list(ARMS):
        raise ValueError("A-S Study B2 arm order is not frozen")
    return path, protocol


def _exact_int_list(value: Any, *, field: str, count: int) -> list[int]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field} contains a non-integer")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicates")
    return result


def _finite_metric_map(value: Any, metric_ids: Sequence[str], *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result: dict[str, float] = {}
    for metric_id in metric_ids:
        raw = value.get(metric_id)
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"{field}.{metric_id} is unavailable")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{field}.{metric_id} is not finite")
        result[metric_id] = number
    return result


def _source_bundle(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    selection = protocol.get("query_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("A-S Study B2 query selection is unavailable")
    evidence_positions = _exact_int_list(
        selection.get("evidence_positions"), field="evidence_positions", count=8
    )
    scoring_positions = _exact_int_list(
        selection.get("scoring_positions"), field="scoring_positions", count=8
    )
    if set(evidence_positions) & set(scoring_positions):
        raise ValueError("A-S Study B2 evidence and scoring positions overlap")
    if any(position < 0 or position >= 64 for position in evidence_positions + scoring_positions):
        raise ValueError("A-S Study B2 query position is outside the 64-coordinate pool")
    expected_evidence = selection.get("expected_evidence_query_ids")
    expected_scoring = selection.get("expected_scoring_query_ids")
    if not isinstance(expected_evidence, list) or not isinstance(expected_scoring, list):
        raise ValueError("A-S Study B2 expected query IDs are unavailable")
    metric_ids = selection.get("metric_ids")
    if not isinstance(metric_ids, list) or metric_ids != [
        "product_in_organic",
        "product_in_aqueous",
        "phase_ratio",
    ]:
        raise ValueError("A-S Study B2 metric roster drifted")
    gates = selection.get("paired_law_effect_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("A-S Study B2 paired-law gates are unavailable")
    metric_gates = {
        metric_id: float(gates[metric_id])
        for metric_id in metric_ids
    }
    required_metrics = int(selection.get("required_passing_metric_count_per_query", 0))
    if required_metrics != 2:
        raise ValueError("A-S Study B2 requires exactly two passing metrics per query")
    qualification_indices = _exact_int_list(
        protocol.get("qualification_world_indices"), field="qualification_world_indices", count=5
    )
    source_root = _resolve(
        root, protocol.get("source_qualification_root"), field="source_qualification_root"
    )
    evidence_queries: list[dict[str, Any]] | None = None
    scoring_queries: list[dict[str, Any]] | None = None
    source_bindings: list[dict[str, Any]] = []
    diagnostic_worlds: list[dict[str, Any]] = []
    for world_index in qualification_indices:
        report_path = source_root / f"world-{world_index}" / "world-report.json"
        report = _load_object(report_path)
        if (
            report.get("candidate_id") != "partition_power_response"
            or report.get("task_id") != "partition-discovery"
            or report.get("world_seed") != world_index
            or report.get("provider_call_count") != 0
            or report.get("participant_session_count") != 0
            or report.get("report_sha256") != report_sha256(report)
        ):
            raise ValueError(f"qualification report is invalid for world {world_index}")
        rows = report.get("rows")
        if not isinstance(rows, list) or len(rows) != 1024:
            raise ValueError(f"qualification denominator drifted for world {world_index}")
        paired: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"qualification row is malformed for world {world_index}")
            if (
                row.get("phase") != selection.get("phase")
                or row.get("intervention_family") != selection.get("intervention_family")
            ):
                continue
            if (
                row.get("status") != "completed"
                or row.get("exact_replay") is not True
                or row.get("safe") is not True
                or row.get("physical_failure") is not None
                or row.get("platform_failure") is not None
                or row.get("participant_visible_leakage_matches") != []
            ):
                raise ValueError(f"qualification source row failed for world {world_index}")
            coordinate_id = str(row["coordinate_id"])
            paired.setdefault(coordinate_id, {})[str(row["law_id"])] = row
        law_ids = selection.get("candidate_laws")
        if law_ids != ["linear_response", "power_response"]:
            raise ValueError("A-S Study B2 candidate-law order drifted")
        complete = {
            coordinate_id: laws
            for coordinate_id, laws in paired.items()
            if set(laws) == set(law_ids)
        }
        if len(complete) != 64:
            raise ValueError(
                f"qualification phase-process pool is not 64 pairs for world {world_index}"
            )
        ordered_ids = sorted(
            complete,
            key=lambda coordinate_id: int(
                complete[coordinate_id]["linear_response"]["coordinate_index"]
            ),
        )
        selected_ids = {
            "evidence": [ordered_ids[position] for position in evidence_positions],
            "scoring": [ordered_ids[position] for position in scoring_positions],
        }
        if (
            selected_ids["evidence"] != expected_evidence
            or selected_ids["scoring"] != expected_scoring
        ):
            raise ValueError("A-S Study B2 coordinate-only query roster drifted")
        world_audit: dict[str, Any] = {
            "qualification_world_index": world_index,
            "source_report_sha256": report["report_sha256"],
            "source_file_sha256": file_sha256(report_path),
            "evidence": [],
            "scoring": [],
        }
        world_query_sets: dict[str, list[dict[str, Any]]] = {"evidence": [], "scoring": []}
        for role in ("evidence", "scoring"):
            for coordinate_id in selected_ids[role]:
                laws = complete[coordinate_id]
                linear = laws["linear_response"]
                power = laws["power_response"]
                linear_metrics = _finite_metric_map(
                    linear.get("metrics"), metric_ids, field=f"{coordinate_id}.linear"
                )
                power_metrics = _finite_metric_map(
                    power.get("metrics"), metric_ids, field=f"{coordinate_id}.power"
                )
                gaps = {
                    metric_id: abs(power_metrics[metric_id] - linear_metrics[metric_id])
                    for metric_id in metric_ids
                }
                passing = [
                    metric_id
                    for metric_id in metric_ids
                    if gaps[metric_id] >= metric_gates[metric_id]
                ]
                if len(passing) < required_metrics:
                    raise ValueError(
                        f"{coordinate_id} is not diagnostic in qualification world {world_index}"
                    )
                query = {
                    "query_id": coordinate_id,
                    "feature_values": deepcopy(dict(power["feature_values"])),
                    "intervention_family": "phase_process",
                    "metric_ids": list(metric_ids),
                    "q2_coordinate_sha256": power["coordinate_sha256"],
                }
                world_query_sets[role].append(query)
                world_audit[role].append(
                    {
                        "query_id": coordinate_id,
                        "metric_gaps": gaps,
                        "passing_metric_ids": passing,
                    }
                )
        if evidence_queries is None:
            evidence_queries = world_query_sets["evidence"]
            scoring_queries = world_query_sets["scoring"]
        elif (
            canonical_json_sha256(evidence_queries)
            != canonical_json_sha256(world_query_sets["evidence"])
            or canonical_json_sha256(scoring_queries or [])
            != canonical_json_sha256(world_query_sets["scoring"])
        ):
            raise ValueError("A-S Study B2 query coordinates differ across qualification worlds")
        source_bindings.append(
            {
                "qualification_world_index": world_index,
                "path": report_path.relative_to(root).as_posix(),
                "file_sha256": file_sha256(report_path),
                "report_sha256": report["report_sha256"],
            }
        )
        diagnostic_worlds.append(world_audit)
    assert evidence_queries is not None and scoring_queries is not None
    return {
        "evidence_queries": evidence_queries,
        "scoring_queries": scoring_queries,
        "query_roster_sha256": canonical_json_sha256(
            {"evidence": evidence_queries, "scoring": scoring_queries}
        ),
        "source_bindings": source_bindings,
        "diagnostic_worlds": diagnostic_worlds,
    }


def _truth_config(
    protocol: Mapping[str, Any], root: Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_path = _resolve(root, protocol.get("runtime_config"), field="runtime_config")
    config = _load_object(runtime_path)
    if config.get("task_id") != "partition-discovery" or not config.get("world_interventions"):
        raise ValueError("A-S Study B2 runtime config lacks the registered structural intervention")
    checkpoint = config.get("belief_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("A-S Study B2 runtime checkpoint is unavailable")
    checkpoint["held_out_queries"] = deepcopy(
        list(bundle["evidence_queries"]) + list(bundle["scoring_queries"])
    )
    return config


def prepare_study_b2_truth(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Prepare all 80 provider-free power-law truth executions before participant calls."""

    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    bundle = _source_bundle(protocol, root)
    config = _truth_config(protocol, root, bundle)
    public_seeds = _exact_int_list(
        protocol.get("public_world_seeds"), field="public_world_seeds", count=5
    )
    target_root = Path(output_root).resolve()
    truth_root = target_root / "truth"
    truth_root.mkdir(parents=True, exist_ok=True)
    bindings: list[dict[str, Any]] = []
    for index, world_seed in enumerate(public_seeds, start=1):
        cluster_id = f"A_S_B2--partition-discovery--seed{world_seed}"
        plan = build_evaluator_truth_plan(
            {
                "world_cluster_id": cluster_id,
                "task_id": "partition-discovery",
                "world_seed": world_seed,
            },
            config,
            formal_result=False,
            formal_preflight_sha256=None,
        )
        plan_errors = validate_evaluator_truth_plan(plan)
        if plan_errors:
            raise ValueError(f"{cluster_id}: invalid truth plan: {'; '.join(plan_errors)}")
        if plan["truth_query_count"] != 16 or plan["truth_query_metric_count"] != 48:
            raise ValueError(f"{cluster_id}: A-S Study B2 truth denominator drifted")
        unit_root = truth_root / cluster_id
        if unit_root.exists():
            stored_plan = _load_object(unit_root / "plan.json")
            report = _load_object(unit_root / "report.json")
            if stored_plan != plan:
                raise ValueError(f"{cluster_id}: stored B2 truth plan drifted")
        else:
            report = execute_evaluator_truth_plan(plan, config, unit_root)
        report_errors = validate_evaluator_truth_report(report, plan)
        if report_errors or report.get("status") != "completed":
            raise ValueError(
                f"{cluster_id}: invalid B2 truth report: {'; '.join(report_errors) or 'incomplete'}"
            )
        bindings.append(
            {
                "cluster_id": cluster_id,
                "world_seed": world_seed,
                "plan_sha256": plan["plan_sha256"],
                "report_sha256": report["report_sha256"],
                "completed_query_count": report["completed_truth_query_count"],
            }
        )
        if progress is not None:
            progress(
                {
                    "stage": "study_b2_truth_progress",
                    "completed_worlds": index,
                    "total_worlds": len(public_seeds),
                    "completed_queries": index * 16,
                    "total_queries": 80,
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": STUDY_B2_TRUTH_MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "runtime_config_sha256": canonical_json_sha256(config),
        "query_roster_sha256": bundle["query_roster_sha256"],
        "provider_call_count": 0,
        "participant_physical_experiment_count": 0,
        "truth_execution_count": 80,
        "source_bindings": bundle["source_bindings"],
        "diagnostic_worlds": bundle["diagnostic_worlds"],
        "truth_bindings": bindings,
    }
    manifest["truth_manifest_sha256"] = canonical_json_sha256(manifest)
    write_json_atomic(target_root / "truth_manifest.json", manifest)
    return manifest


def build_study_b2_manifest(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build the exact 15-cell B2 schedule after provider-free truth preparation."""

    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    target_root = Path(output_root).resolve()
    truth_manifest = prepare_study_b2_truth(
        protocol_file,
        repository_root=root,
        output_root=target_root,
    )
    bundle = _source_bundle(protocol, root)
    config = _truth_config(protocol, root, bundle)
    prior_arms = config.get("prior_arms")
    if not isinstance(prior_arms, Mapping):
        raise ValueError("A-S Study B2 prior arms are unavailable")
    public_seeds = _exact_int_list(
        protocol.get("public_world_seeds"), field="public_world_seeds", count=5
    )
    cells: list[dict[str, Any]] = []
    cluster_packets: list[dict[str, Any]] = []
    for world_index, world_seed in enumerate(public_seeds):
        cluster_id = f"A_S_B2--partition-discovery--seed{world_seed}"
        report = _load_object(target_root / "truth" / cluster_id / "report.json")
        truth = report.get("truth")
        if not isinstance(truth, Mapping):
            raise ValueError(f"{cluster_id}: B2 truth values are unavailable")
        evidence: list[dict[str, Any]] = []
        for query in bundle["evidence_queries"]:
            query_id = str(query["query_id"])
            evidence.append(
                {
                    "observation_id": f"evidence-{query_id}",
                    "query_id": query_id,
                    "feature_values": deepcopy(query["feature_values"]),
                    "intervention_family": "phase_process",
                    "observations": _finite_metric_map(
                        truth.get(query_id), query["metric_ids"], field=f"{cluster_id}.{query_id}"
                    ),
                }
            )
        scoring_queries: list[dict[str, Any]] = []
        scoring_truth: dict[str, dict[str, float]] = {}
        for query in bundle["scoring_queries"]:
            query_id = str(query["query_id"])
            scoring_queries.append(
                {
                    "query_id": query_id,
                    "feature_values": deepcopy(query["feature_values"]),
                    "intervention_family": "phase_process",
                    "metric_ids": list(query["metric_ids"]),
                }
            )
            scoring_truth[query_id] = _finite_metric_map(
                truth.get(query_id), query["metric_ids"], field=f"{cluster_id}.{query_id}"
            )
        public_packet = {
            "schema_version": "chemworld-work-ii-as-study-b2-public-packet-0.1",
            "cluster_id": cluster_id,
            "task_id": "partition-discovery",
            "metric_range": [0.0, 1.0],
            "evidence": evidence,
            "scoring_queries": scoring_queries,
        }
        packet_hash = canonical_json_sha256(public_packet)
        cluster_packets.append(
            {
                "cluster_id": cluster_id,
                "public_packet_sha256": packet_hash,
                "evidence_query_count": 8,
                "scoring_query_count": 8,
                "scoring_term_count": 24,
                "truth_report_sha256": report["report_sha256"],
            }
        )
        rotated_arms = [ARMS[(world_index + offset) % len(ARMS)] for offset in range(len(ARMS))]
        for arm in rotated_arms:
            arm_contract = prior_arms.get(arm)
            if not isinstance(arm_contract, Mapping):
                raise ValueError(f"A-S Study B2 prior arm {arm} is unavailable")
            cells.append(
                {
                    "cell_index": len(cells) + 1,
                    "study_id": protocol["study_id"],
                    "cell_id": f"{cluster_id}--{arm}",
                    "cluster_id": cluster_id,
                    "locus": "A_S_B2",
                    "task_id": "partition-discovery",
                    "world_seed": world_seed,
                    "arm": arm,
                    "initial_world_model": deepcopy(arm_contract.get("initial_world_model")),
                    "public_packet": deepcopy(public_packet),
                    "public_packet_sha256": packet_hash,
                    "scoring_truth": deepcopy(scoring_truth),
                }
            )
    if len(cells) != 15 or len(cluster_packets) != 5:
        raise ValueError("A-S Study B2 denominator differs from 15 cells / 5 clusters")
    manifest: dict[str, Any] = {
        "schema_version": STUDY_B2_MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "provider": deepcopy(protocol["provider"]),
        "execution": deepcopy(protocol["execution"]),
        "arms": list(ARMS),
        "cell_count": 15,
        "cluster_count": 5,
        "scoring_term_count": 24,
        "participant_physical_experiment_count": 0,
        "query_roster_sha256": bundle["query_roster_sha256"],
        "truth_manifest_sha256": truth_manifest["truth_manifest_sha256"],
        "cluster_packets": cluster_packets,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def summarize_study_b2_results(
    manifest: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = summarize_study_b_results(manifest, cell_results)
    summary["schema_version"] = STUDY_B2_SUMMARY_VERSION
    summary["query_roster_sha256"] = manifest["query_roster_sha256"]
    summary["truth_manifest_sha256"] = manifest["truth_manifest_sha256"]
    return summary


__all__ = [
    "STUDY_B2_MANIFEST_VERSION",
    "STUDY_B2_PROTOCOL_VERSION",
    "STUDY_B2_SUMMARY_VERSION",
    "STUDY_B2_TRUTH_MANIFEST_VERSION",
    "build_study_b2_manifest",
    "prepare_study_b2_truth",
    "summarize_study_b2_results",
]
