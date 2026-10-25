"""Matched-evidence belief-updating study for Work II Study B.

The participant-facing packet never contains evaluator truth for the scoring
queries.  Truth is retained only in the host-owned manifest and scoring path.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256

STUDY_B_PROTOCOL_VERSION = "chemworld-work-ii-study-b-matched-evidence-protocol-0.1"
STUDY_B_REPLICATION_PROTOCOL_VERSION = (
    "chemworld-work-ii-study-b-matched-evidence-replication-protocol-0.1"
)
STUDY_B_MANIFEST_VERSION = "chemworld-work-ii-study-b-input-manifest-0.1"
STUDY_B_CELL_VERSION = "chemworld-work-ii-study-b-cell-result-0.1"
STUDY_B_SUMMARY_VERSION = "chemworld-work-ii-study-b-summary-0.1"


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


def _exact_indices(value: Any, *, field: str, query_count: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError(f"{field} must contain exactly eight indices")
    indices: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < query_count:
            raise ValueError(f"{field} contains an invalid query index")
        indices.append(raw)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{field} contains duplicate query indices")
    return indices


def _metric_truth(report: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, float]:
    query_id = str(query["query_id"])
    truth = report.get("truth")
    values = truth.get(query_id) if isinstance(truth, Mapping) else None
    if not isinstance(values, Mapping):
        raise ValueError(f"truth report is missing query {query_id}")
    result: dict[str, float] = {}
    for metric_id in query["metric_ids"]:
        value = values.get(metric_id)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"truth report is missing {query_id}.{metric_id}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"truth report has non-finite {query_id}.{metric_id}")
        result[str(metric_id)] = number
    return result


def build_study_b_manifest(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build and validate a frozen matched-evidence schedule."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path)
    if not protocol_file.is_absolute():
        protocol_file = root / protocol_file
    protocol = _load_object(protocol_file)
    protocol_version = protocol.get("schema_version")
    if protocol_version not in {
        STUDY_B_PROTOCOL_VERSION,
        STUDY_B_REPLICATION_PROTOCOL_VERSION,
    }:
        raise ValueError("unsupported Study B protocol version")
    arms = protocol.get("arms")
    if arms != ["opaque", "aligned_nominal", "misindexed_nominal"]:
        raise ValueError("Study B arm order is not frozen")
    loci = protocol.get("loci")
    expected_locus_count = 2 if protocol_version == STUDY_B_PROTOCOL_VERSION else 1
    if not isinstance(loci, list) or len(loci) != expected_locus_count:
        raise ValueError(
            f"Study B protocol requires exactly {expected_locus_count} registered loci"
        )
    if (
        protocol_version == STUDY_B_REPLICATION_PROTOCOL_VERSION
        and loci[0].get("locus") != "A_P"
    ):
        raise ValueError("Study B single-locus replication must retain the registered A-P locus")
    truth_root = _resolve(root, protocol.get("source_truth_root"), field="source_truth_root")
    cells: list[dict[str, Any]] = []
    cluster_packets: list[dict[str, Any]] = []
    for locus_entry in loci:
        if not isinstance(locus_entry, Mapping):
            raise ValueError("Study B locus entry must be an object")
        locus = str(locus_entry["locus"])
        task_id = str(locus_entry["task_id"])
        runtime_path = _resolve(root, locus_entry.get("runtime_config"), field="runtime_config")
        runtime = _load_object(runtime_path)
        if runtime.get("task_id") != task_id:
            raise ValueError(f"runtime task mismatch for {locus}")
        checkpoint = runtime.get("belief_checkpoint")
        queries = checkpoint.get("held_out_queries") if isinstance(checkpoint, Mapping) else None
        if not isinstance(queries, list) or len(queries) != 16:
            raise ValueError(f"{locus} must expose exactly 16 registered queries")
        evidence_indices = _exact_indices(
            locus_entry.get("evidence_query_indices"),
            field=f"{locus}.evidence_query_indices",
            query_count=len(queries),
        )
        scoring_indices = _exact_indices(
            locus_entry.get("scoring_query_indices"),
            field=f"{locus}.scoring_query_indices",
            query_count=len(queries),
        )
        if set(evidence_indices) & set(scoring_indices):
            raise ValueError(f"{locus} evidence and scoring queries overlap")
        if set(evidence_indices) | set(scoring_indices) != set(range(16)):
            raise ValueError(f"{locus} query split does not cover all registered queries")
        world_seeds = locus_entry.get("world_seeds")
        if not isinstance(world_seeds, list) or len(world_seeds) != 5:
            raise ValueError(f"{locus} requires exactly five world seeds")
        prior_arms = runtime.get("prior_arms")
        if not isinstance(prior_arms, Mapping):
            raise ValueError(f"{locus} runtime prior arms are unavailable")
        for world_index, raw_seed in enumerate(world_seeds):
            if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                raise ValueError(f"{locus} contains an invalid world seed")
            world_seed = int(raw_seed)
            cluster_id = f"{locus}--{task_id}--seed{world_seed}"
            report_path = truth_root / cluster_id / "report.json"
            report = _load_object(report_path)
            if (
                report.get("status") != "completed"
                or report.get("task_id") != task_id
                or report.get("world_seed") != world_seed
                or report.get("failed_truth_query_count") != 0
            ):
                raise ValueError(f"truth report is not complete for {cluster_id}")
            evidence: list[dict[str, Any]] = []
            for index in evidence_indices:
                query = queries[index]
                evidence.append(
                    {
                        "observation_id": f"evidence-{query['query_id']}",
                        "query_id": query["query_id"],
                        "feature_values": deepcopy(query["feature_values"]),
                        "intervention_family": query.get("intervention_family"),
                        "observations": _metric_truth(report, query),
                    }
                )
            scoring_queries: list[dict[str, Any]] = []
            scoring_truth: dict[str, dict[str, float]] = {}
            for index in scoring_indices:
                query = queries[index]
                query_id = str(query["query_id"])
                scoring_queries.append(
                    {
                        "query_id": query_id,
                        "feature_values": deepcopy(query["feature_values"]),
                        "intervention_family": query.get("intervention_family"),
                        "metric_ids": list(query["metric_ids"]),
                    }
                )
                scoring_truth[query_id] = _metric_truth(report, query)
            public_packet = {
                "schema_version": "chemworld-work-ii-study-b-public-packet-0.1",
                "cluster_id": cluster_id,
                "task_id": task_id,
                "metric_range": [0.0, 1.0],
                "evidence": evidence,
                "scoring_queries": scoring_queries,
            }
            packet_hash = canonical_json_sha256(public_packet)
            cluster_packets.append(
                {
                    "cluster_id": cluster_id,
                    "public_packet_sha256": packet_hash,
                    "evidence_query_count": len(evidence),
                    "scoring_query_count": len(scoring_queries),
                    "scoring_term_count": sum(len(q["metric_ids"]) for q in scoring_queries),
                }
            )
            rotated_arms = [arms[(world_index + offset) % len(arms)] for offset in range(len(arms))]
            for arm in rotated_arms:
                arm_contract = prior_arms.get(arm)
                if not isinstance(arm_contract, Mapping):
                    raise ValueError(f"{locus} is missing arm {arm}")
                cells.append(
                    {
                        "cell_index": len(cells) + 1,
                        "cell_id": f"{cluster_id}--{arm}",
                        "cluster_id": cluster_id,
                        "locus": locus,
                        "task_id": task_id,
                        "world_seed": world_seed,
                        "arm": arm,
                        "initial_world_model": deepcopy(arm_contract.get("initial_world_model")),
                        "public_packet": deepcopy(public_packet),
                        "public_packet_sha256": packet_hash,
                        "scoring_truth": deepcopy(scoring_truth),
                    }
                )
    expected_clusters = expected_locus_count * 5
    expected_cells = expected_clusters * 3
    if len(cells) != expected_cells or len(cluster_packets) != expected_clusters:
        raise ValueError(
            "Study B manifest denominator differs from the registered locus coverage"
        )
    execution = protocol.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    if execution.get("formal_sessions") != expected_cells:
        raise ValueError("Study B formal session denominator differs from the schedule")
    manifest = {
        "schema_version": STUDY_B_MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "provider": deepcopy(protocol["provider"]),
        "execution": deepcopy(protocol["execution"]),
        "arms": list(arms),
        "cell_count": len(cells),
        "cluster_count": len(cluster_packets),
        "participant_physical_experiment_count": 0,
        "cluster_packets": cluster_packets,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def prediction_output_schema(
    scoring_queries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"pre", "post"}:
        raise ValueError("stage must be pre or post")
    metric_ids = sorted(
        {str(metric) for query in scoring_queries for metric in query["metric_ids"]}
    )
    metric_properties = {
        metric_id: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for metric_id in metric_ids
    }
    properties: dict[str, Any] = {
        "status": {"type": "string", "const": f"{stage}_evidence_complete"},
        "predictions": {
            "type": "array",
            "minItems": len(scoring_queries),
            "maxItems": len(scoring_queries),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query_id", "metrics"],
                "properties": {
                    "query_id": {
                        "type": "string",
                        "enum": [str(query["query_id"]) for query in scoring_queries],
                    },
                    "metrics": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": metric_properties,
                        "required": metric_ids,
                    },
                },
            },
        },
        "model_summary": {"type": "string", "maxLength": 1200},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    required = ["status", "predictions", "model_summary", "confidence"]
    if stage == "post":
        properties["evidence_assessment"] = {"type": "string", "maxLength": 1200}
        required.append("evidence_assessment")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def validate_prediction_payload(
    payload: Mapping[str, Any],
    scoring_queries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != f"{stage}_evidence_complete":
        errors.append(f"{stage} status is invalid")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return [*errors, f"{stage} predictions are unavailable"]
    expected = {
        str(query["query_id"]): set(map(str, query["metric_ids"]))
        for query in scoring_queries
    }
    observed: dict[str, set[str]] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            errors.append(f"{stage} prediction {index} is not an object")
            continue
        query_id = prediction.get("query_id")
        metrics = prediction.get("metrics")
        if not isinstance(query_id, str) or query_id in observed:
            errors.append(f"{stage} prediction {index} has an invalid or duplicate query_id")
            continue
        if not isinstance(metrics, Mapping):
            errors.append(f"{stage} prediction {query_id} metrics are unavailable")
            continue
        observed[query_id] = set(map(str, metrics))
        for metric_id, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"{stage} prediction {query_id}.{metric_id} is invalid")
    if set(observed) != set(expected):
        errors.append(f"{stage} prediction query denominator differs from the contract")
    for query_id in set(observed) & set(expected):
        if observed[query_id] != expected[query_id]:
            errors.append(f"{stage} prediction metric denominator differs for {query_id}")
    return errors


def score_prediction_payload(
    payload: Mapping[str, Any],
    scoring_truth: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    predictions = {
        str(item["query_id"]): item["metrics"]
        for item in payload["predictions"]
        if isinstance(item, Mapping)
    }
    terms: list[dict[str, Any]] = []
    for query_id, truth_metrics in scoring_truth.items():
        for metric_id, truth in truth_metrics.items():
            predicted = float(predictions[query_id][metric_id])
            terms.append(
                {
                    "query_id": query_id,
                    "metric_id": metric_id,
                    "predicted": predicted,
                    "truth": float(truth),
                    "normalized_absolute_error": abs(predicted - float(truth)),
                }
            )
    return {
        "mean_normalized_absolute_error": mean(
            term["normalized_absolute_error"] for term in terms
        ),
        "term_count": len(terms),
        "terms": terms,
    }


def summarize_study_b_results(
    manifest: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = {str(cell["cell_id"]) for cell in manifest["cells"]}
    observed_ids = {str(result.get("cell_id")) for result in cell_results}
    terminal = [result for result in cell_results if result.get("status") == "completed"]
    failures = [result for result in cell_results if result.get("status") != "completed"]
    cell_rows: list[dict[str, Any]] = []
    by_cluster: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in terminal:
        pre_error = float(result["scores"]["pre"]["mean_normalized_absolute_error"])
        post_error = float(result["scores"]["post"]["mean_normalized_absolute_error"])
        row = {
            "cell_id": result["cell_id"],
            "cluster_id": result["cluster_id"],
            "locus": result["locus"],
            "task_id": result["task_id"],
            "world_seed": result["world_seed"],
            "arm": result["arm"],
            "pre_error": pre_error,
            "post_error": post_error,
            "update_gain": pre_error - post_error,
            "scoring_term_count": result["scores"]["post"]["term_count"],
        }
        cell_rows.append(row)
        by_cluster[str(row["cluster_id"])][str(row["arm"])] = row
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, arm_rows in sorted(by_cluster.items()):
        if set(arm_rows) != {"opaque", "aligned_nominal", "misindexed_nominal"}:
            continue
        aligned = arm_rows["aligned_nominal"]
        misindexed = arm_rows["misindexed_nominal"]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "locus": aligned["locus"],
                "task_id": aligned["task_id"],
                "world_seed": aligned["world_seed"],
                "update_gains": {
                    arm: arm_rows[arm]["update_gain"] for arm in sorted(arm_rows)
                },
                "primary_contrast": misindexed["update_gain"] - aligned["update_gain"],
            }
        )
    locus_rows: list[dict[str, Any]] = []
    for locus in sorted({str(row["locus"]) for row in cluster_rows}):
        members = [row for row in cluster_rows if row["locus"] == locus]
        locus_rows.append(
            {
                "locus": locus,
                "cluster_count": len(members),
                "mean_primary_contrast": mean(row["primary_contrast"] for row in members),
                "mean_update_gain_by_arm": {
                    arm: mean(row["update_gains"][arm] for row in members)
                    for arm in ("opaque", "aligned_nominal", "misindexed_nominal")
                },
                "positive_primary_contrast_world_count": sum(
                    row["primary_contrast"] > 0 for row in members
                ),
            }
        )
    return {
        "schema_version": STUDY_B_SUMMARY_VERSION,
        "study_id": manifest["study_id"],
        "scheduled_cell_count": len(expected_ids),
        "observed_cell_count": len(observed_ids),
        "missing_cell_ids": sorted(expected_ids - observed_ids),
        "unexpected_cell_ids": sorted(observed_ids - expected_ids),
        "completed_cell_count": len(terminal),
        "failed_cell_count": len(failures),
        "complete_cluster_count": len(cluster_rows),
        "participant_physical_experiment_count": 0,
        "status": (
            "completed"
            if observed_ids == expected_ids and len(terminal) == len(expected_ids)
            else "incomplete"
        ),
        "cell_rows": sorted(cell_rows, key=lambda row: row["cell_id"]),
        "cluster_rows": cluster_rows,
        "locus_rows": locus_rows,
        "failures": [
            {
                "cell_id": result.get("cell_id"),
                "failure": result.get("failure"),
            }
            for result in failures
        ],
    }


__all__ = [
    "STUDY_B_CELL_VERSION",
    "STUDY_B_MANIFEST_VERSION",
    "STUDY_B_PROTOCOL_VERSION",
    "STUDY_B_REPLICATION_PROTOCOL_VERSION",
    "STUDY_B_SUMMARY_VERSION",
    "build_study_b_manifest",
    "prediction_output_schema",
    "score_prediction_payload",
    "summarize_study_b_results",
    "validate_prediction_payload",
]
