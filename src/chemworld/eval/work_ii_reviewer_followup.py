"""Reviewer-requested Work II mechanism follow-up experiments.

The first block implemented here is A-S Study B3.  It separates a public,
participant-visible constitutive-law identification problem from free-text
description, and it binds the participant's action choice to an evaluator-owned
recipe that has not appeared in the evidence packet.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from chemworld.eval.provenance import canonical_json_sha256, write_json_atomic
from chemworld.eval.work_ii_truth import (
    build_evaluator_truth_plan,
    execute_evaluator_truth_plan,
    validate_evaluator_truth_plan,
    validate_evaluator_truth_report,
)
from chemworld.world.phase_kernel import (
    PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION,
    nominal_partition_pair_tables,
)

B3_PROTOCOL_VERSION = "chemworld-work-ii-as-study-b3-protocol-0.1"
B3_MANIFEST_VERSION = "chemworld-work-ii-as-study-b3-input-manifest-0.2"
B3_CELL_VERSION = "chemworld-work-ii-as-study-b3-cell-result-0.2"
B3_SUMMARY_VERSION = "chemworld-work-ii-as-study-b3-summary-0.2"
B3_QUALIFICATION_VERSION = "chemworld-work-ii-as-study-b3-qualification-0.1"
B3_PUBLIC_TRUTH_VERSION = "chemworld-work-ii-as-study-b3-public-truth-0.1"

B3_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
B3_ACTION_SELECTION_ENCODINGS = ("query_id", "zero_based_index")
B3_STAGE_STATUS_ENCODINGS = ("explicit_const", "runner_derived")
B3_METRIC_IDS = (
    "product_in_organic",
    "product_in_aqueous",
    "phase_ratio",
    "score",
)
B3_FAMILIES = (
    "FAMILY_A_LINEAR",
    "FAMILY_B_POWER",
    "FAMILY_C_SATURATING",
    "FAMILY_D_CONSTANT",
)

AP_PROTOCOL_VERSION = "chemworld-work-ii-ap-evidence-acquisition-protocol-0.1"
AP_MANIFEST_VERSION = "chemworld-work-ii-ap-evidence-acquisition-manifest-0.1"
AP_CELL_VERSION = "chemworld-work-ii-ap-evidence-acquisition-cell-result-0.1"
AP_SUMMARY_VERSION = "chemworld-work-ii-ap-evidence-acquisition-summary-0.1"
AP_QUALIFICATION_VERSION = "chemworld-work-ii-ap-evidence-acquisition-qualification-0.1"
AP_PUBLIC_TRUTH_VERSION = "chemworld-work-ii-ap-evidence-acquisition-public-truth-0.1"
AP_ARMS = ("aligned_nominal", "misindexed_nominal")
AP_CONDITIONS = ("active_choice", "forced_diagnostic", "forced_low_information")
AP_METRIC_IDS = (
    "selective_product_yield",
    "electrochemical_selectivity",
    "faradaic_efficiency",
    "energy_efficiency",
    "safety_risk",
    "score",
)
AP_DIRECTION_CHOICES = (
    "lower_controlled_potential",
    "higher_controlled_potential",
    "undetermined",
)
POWER_INTERVENTION = {
    "kind": "mechanism_family",
    "mode": "constitutive_law_family",
    "severity": 1.0,
    "constitutive_law_change": {
        "transform_id": "partition_power_response_stress_v1",
        "partition_coefficient_exponent_at_full_severity": 1.75,
    },
}


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
    if protocol.get("schema_version") != B3_PROTOCOL_VERSION:
        raise ValueError("unsupported A-S Study B3 protocol version")
    if protocol.get("arms") != list(B3_ARMS):
        raise ValueError("A-S Study B3 arm order drifted")
    if protocol.get("metric_ids") != list(B3_METRIC_IDS):
        raise ValueError("A-S Study B3 metric roster drifted")
    return path, protocol


def _finite_metric_map(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is unavailable")
    result: dict[str, float] = {}
    for metric_id in B3_METRIC_IDS:
        raw = value.get(metric_id)
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"{field}.{metric_id} is unavailable")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{field}.{metric_id} is not finite")
        result[metric_id] = number
    return result


def _reference_coefficients() -> dict[tuple[int, int], float]:
    table = nominal_partition_pair_tables()["product_distribution_coefficients"]
    return {
        (solvent, extractant): float(
            PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION * table[solvent][extractant]
        )
        for solvent in range(4)
        for extractant in range(4)
    }


def build_b3_candidate_queries(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the frozen 16-pair x 4-volume x 2-mixing candidate grid."""

    grid = protocol.get("candidate_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("A-S Study B3 candidate grid is unavailable")
    volume_configs = grid.get("volume_configs")
    mixing_configs = grid.get("mixing_configs")
    if not isinstance(volume_configs, list) or len(volume_configs) != 4:
        raise ValueError("A-S Study B3 requires exactly four volume configurations")
    if not isinstance(mixing_configs, list) or len(mixing_configs) != 2:
        raise ValueError("A-S Study B3 requires exactly two mixing configurations")
    references = _reference_coefficients()
    queries: list[dict[str, Any]] = []
    for solvent in range(4):
        for extractant in range(4):
            for volume_index, volume in enumerate(volume_configs):
                if not isinstance(volume, Mapping):
                    raise ValueError("A-S Study B3 volume configuration is malformed")
                for mixing_index, mixing in enumerate(mixing_configs):
                    if not isinstance(mixing, Mapping):
                        raise ValueError("A-S Study B3 mixing configuration is malformed")
                    query_id = (
                        f"b3-s{solvent}-e{extractant}-v{volume_index}-m{mixing_index}"
                    )
                    queries.append(
                        {
                            "query_id": query_id,
                            "pair_id": f"pair-s{solvent}-e{extractant}",
                            "reference_partition_coefficient": references[(solvent, extractant)],
                            "feature_values": {
                                "solvent": solvent,
                                "aqueous_phase_volume_L": float(
                                    volume["aqueous_phase_volume_L"]
                                ),
                                "extractant": extractant,
                                "extractant_volume_L": float(
                                    volume["extractant_volume_L"]
                                ),
                                "mix_duration_s": float(mixing["mix_duration_s"]),
                                "settle_duration_s": float(mixing["settle_duration_s"]),
                                "stirring_speed_rpm": float(
                                    mixing["stirring_speed_rpm"]
                                ),
                            },
                            "metric_ids": list(B3_METRIC_IDS),
                        }
                    )
    if len(queries) != 128 or len({item["query_id"] for item in queries}) != 128:
        raise ValueError("A-S Study B3 candidate denominator differs from 128")
    return queries


def _truth_config(
    runtime: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    *,
    exponent: float,
) -> dict[str, Any]:
    config = deepcopy(dict(runtime))
    if config.get("task_id") != "partition-discovery":
        raise ValueError("A-S Study B3 requires the partition-discovery runtime")
    checkpoint = config.get("belief_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("A-S Study B3 runtime lacks its checkpoint contract")
    checkpoint["allowed_metric_ids"] = list(B3_METRIC_IDS)
    checkpoint["held_out_queries"] = [
        {
            "query_id": str(item["query_id"]),
            "feature_values": deepcopy(dict(item["feature_values"])),
            "metric_ids": list(B3_METRIC_IDS),
        }
        for item in queries
    ]
    if math.isclose(exponent, 1.0, abs_tol=1e-12):
        config["world_interventions"] = []
    else:
        intervention = deepcopy(POWER_INTERVENTION)
        intervention["constitutive_law_change"][
            "partition_coefficient_exponent_at_full_severity"
        ] = float(exponent)
        config["world_interventions"] = [intervention]
    return config


def _apply_shared_truth_noise(
    plan: Mapping[str, Any],
    *,
    shared_observation_seed: int,
    world_seed: int,
) -> dict[str, Any]:
    if isinstance(shared_observation_seed, bool) or shared_observation_seed < 0:
        raise ValueError("shared observation seed must be a non-negative integer")
    paired = deepcopy(dict(plan))
    paired["queries"] = [deepcopy(dict(query)) for query in plan["queries"]]
    pairing_contract = {
        "mode": "common_seed_across_candidates_and_laws_within_world",
        "shared_observation_seed": int(shared_observation_seed),
        "world_seed": int(world_seed),
        "query_count": len(paired["queries"]),
    }
    coordinate_sha256 = canonical_json_sha256(pairing_contract)
    for query in paired["queries"]:
        query["observation_seed"] = int(shared_observation_seed)
        query["observation_coordinate_sha256"] = coordinate_sha256
    paired["truth_noise_pairing"] = pairing_contract
    paired["plan_sha256"] = canonical_json_sha256(
        {key: value for key, value in paired.items() if key != "plan_sha256"}
    )
    return paired


def _truth_report(
    *,
    runtime: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    exponent: float,
    world_seed: int,
    cluster_id: str,
    output_root: Path,
    liveness: Callable[[float], None] | None = None,
    progress_interval_s: float = 30.0,
    shared_observation_seed: int | None = None,
) -> dict[str, Any]:
    config = _truth_config(runtime, queries, exponent=exponent)
    plan = build_evaluator_truth_plan(
        {
            "world_cluster_id": cluster_id,
            "task_id": "partition-discovery",
            "world_seed": int(world_seed),
        },
        config,
        formal_result=False,
        formal_preflight_sha256=None,
    )
    if shared_observation_seed is not None:
        plan = _apply_shared_truth_noise(
            plan,
            shared_observation_seed=shared_observation_seed,
            world_seed=world_seed,
        )
    plan_errors = validate_evaluator_truth_plan(plan)
    if plan_errors:
        raise ValueError(f"{cluster_id}: invalid truth plan: {'; '.join(plan_errors)}")
    if output_root.exists():
        stored_plan = _load_object(output_root / "plan.json")
        report = _load_object(output_root / "report.json")
        if stored_plan != plan:
            raise ValueError(f"{cluster_id}: stored truth plan drifted")
    else:
        stop = threading.Event()
        started = time.perf_counter()

        def emit_liveness() -> None:
            while not stop.wait(progress_interval_s):
                if liveness is not None:
                    liveness(round(time.perf_counter() - started, 1))

        heartbeat = threading.Thread(target=emit_liveness, daemon=True)
        heartbeat.start()
        try:
            report = execute_evaluator_truth_plan(plan, config, output_root)
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
    report_errors = validate_evaluator_truth_report(report, plan)
    if report_errors or report.get("status") != "completed":
        raise ValueError(
            f"{cluster_id}: incomplete truth block: "
            f"{'; '.join(report_errors) or report.get('status')}"
        )
    return report


def _report_truth(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = report.get("truth")
    if not isinstance(raw, Mapping):
        raise ValueError("truth report has no truth map")
    return {
        str(query_id): _finite_metric_map(metrics, field=f"truth.{query_id}")
        for query_id, metrics in raw.items()
    }


def _paired_truth(
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]]
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    return {
        int(seed): {
            law_id: _report_truth(report)
            for law_id, report in law_reports.items()
        }
        for seed, law_reports in reports.items()
    }


def _mean_metric_error(
    predicted: Mapping[str, Mapping[str, float]],
    truth: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
) -> float:
    terms = [
        abs(float(predicted[query_id][metric_id]) - float(truth[query_id][metric_id]))
        for query_id in query_ids
        for metric_id in B3_METRIC_IDS
    ]
    return mean(terms)


def _saturation_predictions(
    evidence_queries: Sequence[Mapping[str, Any]],
    scoring_queries: Sequence[Mapping[str, Any]],
    evidence_truth: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Fit the frozen coefficient-only saturation alternative on evidence rows."""

    evidence_x = np.asarray(
        [float(item["reference_partition_coefficient"]) for item in evidence_queries]
    )
    scoring_x = np.asarray(
        [float(item["reference_partition_coefficient"]) for item in scoring_queries]
    )
    best: dict[str, tuple[float, np.ndarray]] = {}
    for metric_id in B3_METRIC_IDS:
        y = np.asarray(
            [float(evidence_truth[str(item["query_id"])][metric_id]) for item in evidence_queries]
        )
        for half_saturation in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            basis = evidence_x / (half_saturation + evidence_x)
            design = np.column_stack((np.ones(len(basis)), basis))
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = np.clip(design @ coefficients, 0.0, 1.0)
            error = float(np.mean(np.abs(fitted - y)))
            if metric_id not in best or error < best[metric_id][0]:
                score_basis = scoring_x / (half_saturation + scoring_x)
                score_design = np.column_stack((np.ones(len(score_basis)), score_basis))
                best[metric_id] = (
                    error,
                    np.clip(score_design @ coefficients, 0.0, 1.0),
                )
    return {
        str(query["query_id"]): {
            metric_id: float(best[metric_id][1][index])
            for metric_id in B3_METRIC_IDS
        }
        for index, query in enumerate(scoring_queries)
    }


def _constant_predictions(
    evidence_queries: Sequence[Mapping[str, Any]],
    scoring_queries: Sequence[Mapping[str, Any]],
    evidence_truth: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    values = {
        metric_id: mean(
            float(evidence_truth[str(query["query_id"])][metric_id])
            for query in evidence_queries
        )
        for metric_id in B3_METRIC_IDS
    }
    return {str(query["query_id"]): dict(values) for query in scoring_queries}


def select_b3_rosters(
    candidates: Sequence[Mapping[str, Any]],
    paired: Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]],
    *,
    action_gain_threshold: float,
) -> dict[str, Any]:
    """Select one common outcome-independent roster using development worlds only."""

    seeds = sorted(paired)
    by_id = {str(item["query_id"]): deepcopy(dict(item)) for item in candidates}
    statistics: dict[str, dict[str, float]] = {}
    for query_id in sorted(by_id):
        gaps = [
            mean(
                abs(
                    paired[seed]["power"][query_id][metric_id]
                    - paired[seed]["linear"][query_id][metric_id]
                )
                for metric_id in B3_METRIC_IDS
            )
            for seed in seeds
        ]
        scores = [paired[seed]["power"][query_id]["score"] for seed in seeds]
        statistics[query_id] = {
            "minimum_law_gap": min(gaps),
            "mean_law_gap": mean(gaps),
            "maximum_power_score": max(scores),
            "mean_power_score": mean(scores),
        }

    pairs = defaultdict(list)
    for query in candidates:
        pairs[str(query["pair_id"])].append(str(query["query_id"]))
    ordered_pairs = sorted(
        pairs,
        key=lambda pair_id: (
            float(by_id[pairs[pair_id][0]]["reference_partition_coefficient"]),
            pair_id,
        ),
    )
    pair_evidence_candidates = {
        pair_id: sorted(
            pairs[pair_id],
            key=lambda query_id: (
                statistics[query_id]["maximum_power_score"],
                -statistics[query_id]["minimum_law_gap"],
                query_id,
            ),
        )[:2]
        for pair_id in ordered_pairs
    }

    def scoring_roster(
        evidence_query_ids: Sequence[str],
    ) -> tuple[list[str], dict[int, float]] | None:
        incumbents = {
            seed: max(
                paired[seed]["power"][query_id]["score"]
                for query_id in evidence_query_ids
            )
            for seed in seeds
        }
        remaining_ids = [
            query_id for query_id in sorted(by_id) if query_id not in evidence_query_ids
        ]
        scoring_ids: list[str] = []
        uncovered = set(seeds)
        while uncovered:
            ranked = []
            for query_id in remaining_ids:
                if query_id in scoring_ids:
                    continue
                covered = {
                    seed
                    for seed in uncovered
                    if paired[seed]["power"][query_id]["score"]
                    >= incumbents[seed] + action_gain_threshold
                }
                ranked.append(
                    (
                        -len(covered),
                        -min(
                            paired[seed]["power"][query_id]["score"] - incumbents[seed]
                            for seed in seeds
                        ),
                        -statistics[query_id]["minimum_law_gap"],
                        query_id,
                        covered,
                    )
                )
            ranked.sort(key=lambda row: row[:4])
            if not ranked or -ranked[0][0] == 0:
                return None
            query_id = ranked[0][3]
            scoring_ids.append(query_id)
            uncovered -= ranked[0][4]

        globally_nonimproving = sorted(
            (
                query_id
                for query_id in remaining_ids
                if query_id not in scoring_ids
                and all(
                    paired[seed]["power"][query_id]["score"] <= incumbents[seed]
                    for seed in seeds
                )
            ),
            key=lambda query_id: (
                statistics[query_id]["mean_power_score"],
                -statistics[query_id]["minimum_law_gap"],
                query_id,
            ),
        )
        if len(globally_nonimproving) < 2:
            return None
        scoring_ids.extend(globally_nonimproving[:2])
        scoring_pair_ids = {str(by_id[query_id]["pair_id"]) for query_id in scoring_ids}
        fillers = sorted(
            (
                query_id
                for query_id in remaining_ids
                if query_id not in scoring_ids
            ),
            key=lambda query_id: (
                str(by_id[query_id]["pair_id"]) in scoring_pair_ids,
                -statistics[query_id]["minimum_law_gap"],
                query_id,
            ),
        )
        for query_id in fillers:
            if len(scoring_ids) >= 8:
                break
            scoring_ids.append(query_id)
            scoring_pair_ids.add(str(by_id[query_id]["pair_id"]))
        if len(scoring_ids) != 8 or len(scoring_pair_ids) < 4:
            return None
        for seed in seeds:
            gains = [
                paired[seed]["power"][query_id]["score"] - incumbents[seed]
                for query_id in scoring_ids
            ]
            if max(gains) < action_gain_threshold or sum(gain <= 0.0 for gain in gains) < 2:
                return None
        return scoring_ids, incumbents

    evidence_ids: list[str] | None = None
    selected: list[str] | None = None
    evidence_incumbents: dict[int, float] | None = None
    selected_pair_ids: tuple[str, ...] | None = None
    selected_combination_index: int | None = None
    for combination_index, pair_ids in enumerate(combinations(ordered_pairs, 4), start=1):
        candidate_evidence = [
            query_id for pair_id in pair_ids for query_id in pair_evidence_candidates[pair_id]
        ]
        candidate_scoring = scoring_roster(candidate_evidence)
        if candidate_scoring is None:
            continue
        evidence_ids = candidate_evidence
        selected, evidence_incumbents = candidate_scoring
        selected_pair_ids = pair_ids
        selected_combination_index = combination_index
        break
    if (
        evidence_ids is None
        or selected is None
        or evidence_incumbents is None
        or selected_pair_ids is None
        or selected_combination_index is None
    ):
        raise ValueError("A-S Study B3 candidate pool lacks a qualified lexicographic roster")

    evidence = [by_id[query_id] for query_id in evidence_ids]
    scoring = [by_id[query_id] for query_id in selected]
    world_checks: list[dict[str, Any]] = []
    for seed in seeds:
        gains = {
            query_id: paired[seed]["power"][query_id]["score"] - evidence_incumbents[seed]
            for query_id in selected
        }
        improving = [
            query_id for query_id, gain in gains.items() if gain >= action_gain_threshold
        ]
        nonimproving = [query_id for query_id, gain in gains.items() if gain <= 0.0]
        if not improving or len(nonimproving) < 2:
            raise ValueError(f"A-S Study B3 action gate failed in development world {seed}")
        world_checks.append(
            {
                "world_seed": seed,
                "evidence_incumbent_score": evidence_incumbents[seed],
                "maximum_scoring_gain": max(gains.values()),
                "improving_query_ids": improving,
                "nonimproving_query_ids": nonimproving,
            }
        )
    return {
        "evidence_queries": evidence,
        "scoring_queries": scoring,
        "evidence_query_ids": evidence_ids,
        "scoring_query_ids": selected,
        "evidence_pair_count": len({item["pair_id"] for item in evidence}),
        "scoring_pair_count": len({item["pair_id"] for item in scoring}),
        "selection_rule": (
            "lexicographically_first_passing_four_pairs_two_low-incumbent_rows_each"
        ),
        "selected_evidence_pair_ids": list(selected_pair_ids),
        "selected_evidence_pair_combination_index": selected_combination_index,
        "query_statistics": {
            query_id: statistics[query_id] for query_id in evidence_ids + selected
        },
        "development_action_checks": world_checks,
        "roster_sha256": canonical_json_sha256(
            {"evidence": evidence_ids, "scoring": selected}
        ),
    }


def _fit_exponent(
    target_truth: Mapping[str, Mapping[str, float]],
    exponent_truth: Mapping[float, Mapping[str, Mapping[str, float]]],
    query_ids: Sequence[str],
) -> dict[str, Any]:
    rows = [
        {
            "exponent": exponent,
            "mean_absolute_error": _mean_metric_error(
                predictions, target_truth, query_ids
            ),
        }
        for exponent, predictions in sorted(exponent_truth.items())
    ]
    winner = min(rows, key=lambda row: (row["mean_absolute_error"], row["exponent"]))
    return {
        "estimated_exponent": winner["exponent"],
        "absolute_error": abs(float(winner["exponent"]) - 1.75),
        "grid_rows": rows,
    }


def _alternative_checks(
    paired: Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]],
    evidence_queries: Sequence[Mapping[str, Any]],
    scoring_queries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence_ids = [str(item["query_id"]) for item in evidence_queries]
    scoring_ids = [str(item["query_id"]) for item in scoring_queries]
    rows: list[dict[str, Any]] = []
    for seed in sorted(paired):
        power = paired[seed]["power"]
        linear = paired[seed]["linear"]
        constant = _constant_predictions(evidence_queries, scoring_queries, power)
        saturation = _saturation_predictions(evidence_queries, scoring_queries, power)
        errors = {
            "power_1_75": 0.0,
            "linear_1_0": _mean_metric_error(linear, power, scoring_ids),
            "constant_endpoint": _mean_metric_error(constant, power, scoring_ids),
            "coefficient_only_saturation": _mean_metric_error(saturation, power, scoring_ids),
        }
        if any(value <= 1.0e-12 for key, value in errors.items() if key != "power_1_75"):
            raise ValueError(f"A-S Study B3 alternatives are not separated in world {seed}")
        rows.append(
            {
                "world_seed": seed,
                "evidence_query_count": len(evidence_ids),
                "scoring_query_count": len(scoring_ids),
                "scoring_mean_absolute_error_by_model": errors,
                "power_uniquely_best": min(errors, key=errors.get) == "power_1_75",
            }
        )
    return rows


def _execute_law_pair(
    *,
    runtime: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    output_root: Path,
    phase: str,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> dict[int, dict[str, dict[str, Any]]]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    total = len(seeds) * 2
    completed = 0
    for seed in seeds:
        reports[int(seed)] = {}
        cluster_id = f"A_S_B3--partition-discovery--seed{seed}"
        for law_id, exponent in (("linear", 1.0), ("power", 1.75)):
            report = _truth_report(
                runtime=runtime,
                queries=queries,
                exponent=exponent,
                world_seed=int(seed),
                cluster_id=cluster_id,
                output_root=output_root / phase / law_id / cluster_id,
                liveness=(
                    (
                        lambda elapsed_s,
                        seed=seed,
                        law_id=law_id,
                        current_unit=completed + 1: progress(
                            {
                                "stage": f"b3_{phase}_law_liveness",
                                "world_seed": int(seed),
                                "law_id": law_id,
                                "current_unit": current_unit,
                                "total_units": total,
                                "query_count_in_current_unit": len(queries),
                                "elapsed_s": elapsed_s,
                            }
                        )
                    )
                    if progress is not None
                    else None
                ),
            )
            reports[int(seed)][law_id] = report
            completed += 1
            if progress is not None:
                progress(
                    {
                        "stage": f"b3_{phase}_law_progress",
                        "completed_units": completed,
                        "total_units": total,
                        "completed_truth_queries": completed * len(queries),
                        "total_truth_queries": total * len(queries),
                    }
                )
    return reports


def _execute_exponent_grid(
    *,
    runtime: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    exponents: Sequence[float],
    output_root: Path,
    phase: str,
    target_by_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    total = len(seeds) * len(exponents)
    completed = 0
    query_ids = [str(item["query_id"]) for item in queries]
    for seed in seeds:
        truth_by_exponent: dict[float, dict[str, dict[str, float]]] = {}
        cluster_id = f"A_S_B3--partition-discovery--seed{seed}"
        for exponent in exponents:
            if math.isclose(float(exponent), 1.75, abs_tol=1.0e-12):
                truth_by_exponent[float(exponent)] = {
                    query_id: deepcopy(dict(target_by_seed[int(seed)][query_id]))
                    for query_id in query_ids
                }
            else:
                label = f"e{exponent:.2f}".replace(".", "p")
                report = _truth_report(
                    runtime=runtime,
                    queries=queries,
                    exponent=float(exponent),
                    world_seed=int(seed),
                    cluster_id=cluster_id,
                    output_root=output_root / phase / label / cluster_id,
                    liveness=(
                        (
                            lambda elapsed_s,
                            seed=seed,
                            exponent=exponent,
                            current_unit=completed + 1: progress(
                                {
                                    "stage": f"b3_{phase}_exponent_liveness",
                                    "world_seed": int(seed),
                                    "exponent": float(exponent),
                                    "current_unit": current_unit,
                                    "total_units": total,
                                    "query_count_in_current_unit": len(queries),
                                    "elapsed_s": elapsed_s,
                                }
                            )
                        )
                        if progress is not None
                        else None
                    ),
                )
                truth_by_exponent[float(exponent)] = _report_truth(report)
            completed += 1
            if progress is not None:
                progress(
                    {
                        "stage": f"b3_{phase}_exponent_progress",
                        "completed_units": completed,
                        "total_units": total,
                        "completed_truth_queries": completed * len(queries),
                        "total_truth_queries": total * len(queries),
                    }
                )
        fit = _fit_exponent(target_by_seed[int(seed)], truth_by_exponent, query_ids)
        fit["world_seed"] = int(seed)
        fits.append(fit)
    return fits


def prepare_b3(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run development qualification, freeze the roster, and execute public truth."""

    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    target = Path(output_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    runtime = _load_object(_resolve(root, protocol["runtime_config"], field="runtime_config"))
    candidates = build_b3_candidate_queries(protocol)
    development_seeds = [int(item) for item in protocol["development_world_seeds"]]
    public_seeds = [int(item) for item in protocol["public_world_seeds"]]
    if development_seeds != [0, 1, 2, 3, 4] or len(public_seeds) != 5:
        raise ValueError("A-S Study B3 world coverage drifted")

    development_reports = _execute_law_pair(
        runtime=runtime,
        queries=candidates,
        seeds=development_seeds,
        output_root=target / "truth",
        phase="development-candidate-grid",
        progress=progress,
    )
    development_paired = _paired_truth(development_reports)
    roster = select_b3_rosters(
        candidates,
        development_paired,
        action_gain_threshold=float(protocol["qualification"]["minimum_action_gain"]),
    )
    write_json_atomic(target / "frozen_roster.json", roster)

    exponents = [float(item) for item in protocol["qualification"]["exponent_grid"]]
    development_fits = _execute_exponent_grid(
        runtime=runtime,
        queries=roster["evidence_queries"],
        seeds=development_seeds,
        exponents=exponents,
        output_root=target / "truth",
        phase="development-exponent-grid",
        target_by_seed={seed: development_paired[seed]["power"] for seed in development_seeds},
        progress=progress,
    )
    tolerance = float(protocol["qualification"]["maximum_exponent_error"])
    if any(row["absolute_error"] > tolerance for row in development_fits):
        raise ValueError("A-S Study B3 development exponent recovery gate failed")
    alternative_rows = _alternative_checks(
        development_paired,
        roster["evidence_queries"],
        roster["scoring_queries"],
    )
    qualification: dict[str, Any] = {
        "schema_version": B3_QUALIFICATION_VERSION,
        "study_id": protocol["study_id"],
        "status": "qualified",
        "candidate_query_count": 128,
        "development_world_count": 5,
        "linear_and_power_truth_execution_count": 1280,
        "exponent_grid_truth_execution_count": len(development_seeds)
        * (len(exponents) - 1)
        * len(roster["evidence_queries"]),
        "exponent_grid_reused_target_truth_count": len(development_seeds)
        * len(roster["evidence_queries"]),
        "provider_call_count": 0,
        "participant_session_count": 0,
        "roster_sha256": roster["roster_sha256"],
        "evidence_pair_count": roster["evidence_pair_count"],
        "scoring_pair_count": roster["scoring_pair_count"],
        "development_action_checks": roster["development_action_checks"],
        "development_exponent_fits": development_fits,
        "alternative_model_checks": alternative_rows,
    }
    qualification["qualification_sha256"] = canonical_json_sha256(qualification)
    write_json_atomic(target / "qualification_summary.json", qualification)

    selected_queries = roster["evidence_queries"] + roster["scoring_queries"]
    public_reports = _execute_law_pair(
        runtime=runtime,
        queries=selected_queries,
        seeds=public_seeds,
        output_root=target / "truth",
        phase="public-selected-roster",
        progress=progress,
    )
    public_paired = _paired_truth(public_reports)
    public_alternatives = _alternative_checks(
        public_paired,
        roster["evidence_queries"],
        roster["scoring_queries"],
    )

    public_worlds: list[dict[str, Any]] = []
    failed_public_worlds: list[dict[str, Any]] = []
    minimum_gain = float(protocol["qualification"]["minimum_action_gain"])
    for seed in public_seeds:
        power = public_paired[seed]["power"]
        evidence_ids = roster["evidence_query_ids"]
        scoring_ids = roster["scoring_query_ids"]
        incumbent = max(power[query_id]["score"] for query_id in evidence_ids)
        gains = {query_id: power[query_id]["score"] - incumbent for query_id in scoring_ids}
        improving = [query_id for query_id, gain in gains.items() if gain >= minimum_gain]
        nonimproving = [query_id for query_id, gain in gains.items() if gain <= 0.0]
        world_passed = bool(improving) and len(nonimproving) >= 2
        if not world_passed:
            failed_public_worlds.append(
                {
                    "world_seed": seed,
                    "maximum_scoring_gain": max(gains.values()),
                    "improving_action_count": len(improving),
                    "nonimproving_action_count": len(nonimproving),
                }
            )
        public_worlds.append(
            {
                "world_seed": seed,
                "preflight_passed": world_passed,
                "evidence_incumbent_score": incumbent,
                "scoring_action_gains": gains,
                "improving_query_ids": improving,
                "nonimproving_query_ids": nonimproving,
                "power_truth": {
                    query_id: power[query_id] for query_id in evidence_ids + scoring_ids
                },
                "linear_truth": {
                    query_id: public_paired[seed]["linear"][query_id]
                    for query_id in evidence_ids + scoring_ids
                },
            }
        )
    require_action_opportunity = bool(
        protocol["qualification"].get("require_action_opportunity_preflight", True)
    )
    public_truth: dict[str, Any] = {
        "schema_version": B3_PUBLIC_TRUTH_VERSION,
        "study_id": protocol["study_id"],
        "status": (
            "preflight_passed"
            if not require_action_opportunity or not failed_public_worlds
            else "preflight_rejected"
        ),
        "public_world_count": 5,
        "selected_query_count_per_law_per_world": 16,
        "linear_and_power_truth_execution_count": 160,
        "exponent_grid_truth_execution_count": 0,
        "provider_call_count": 0,
        "participant_physical_experiment_count": 0,
        "roster_sha256": roster["roster_sha256"],
        "failed_public_world_count": len(failed_public_worlds),
        "failed_public_worlds": failed_public_worlds,
        "action_opportunity_preflight_required": require_action_opportunity,
        "action_opportunity_eligible_world_count": len(public_seeds)
        - len(failed_public_worlds),
        "public_law_decodability": {
            "distinct_reference_coefficient_count": roster["evidence_pair_count"],
            "exact_fixed_pair_exponent_coefficient_alias_present": False,
            "power_uniquely_best_world_count": sum(
                row["power_uniquely_best"] for row in public_alternatives
            ),
            "world_count": len(public_seeds),
        },
        "alternative_model_checks": public_alternatives,
        "worlds": public_worlds,
    }
    public_truth["public_truth_sha256"] = canonical_json_sha256(public_truth)
    write_json_atomic(target / "public_truth_manifest.json", public_truth)
    if failed_public_worlds and require_action_opportunity:
        raise ValueError(
            "A-S Study B3 public preflight was retained as rejected before participant calls"
        )
    manifest = build_b3_manifest(
        protocol_file,
        repository_root=root,
        output_root=target,
    )
    write_json_atomic(target / "input_manifest.json", manifest)
    return manifest


def _candidate_families() -> list[dict[str, Any]]:
    return [
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
    ]


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
    raise ValueError(f"unknown A-S Study B3 arm: {arm}")


def _public_query(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(query["query_id"]),
        "nominal_pair_id": str(query["pair_id"]),
        "reference_partition_coefficient": float(query["reference_partition_coefficient"]),
        "feature_values": deepcopy(dict(query["feature_values"])),
        "metric_ids": list(B3_METRIC_IDS),
    }


def build_b3_manifest(
    protocol_path: str | Path,
    *,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file, protocol = _protocol(protocol_path, root)
    target = Path(output_root).resolve()
    roster = _load_object(target / "frozen_roster.json")
    qualification = _load_object(target / "qualification_summary.json")
    public_truth = _load_object(target / "public_truth_manifest.json")
    provider_free_source_study_id = qualification.get("study_id")
    if (
        not isinstance(provider_free_source_study_id, str)
        or public_truth.get("study_id") != provider_free_source_study_id
    ):
        raise ValueError("A-S Study B3 provider-free source study identity drifted")
    if qualification.get("status") != "qualified":
        raise ValueError("A-S Study B3 provider-free qualification is not complete")
    if public_truth.get("status") != "preflight_passed":
        raise ValueError("A-S Study B3 public truth preflight is not complete")
    if len(roster.get("evidence_queries", [])) != 8 or len(roster.get("scoring_queries", [])) != 8:
        raise ValueError("A-S Study B3 frozen roster denominator drifted")
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("A-S Study B3 execution contract is unavailable")
    replicates_per_arm = int(execution.get("replicates_per_arm", 1))
    if not 1 <= replicates_per_arm <= 10:
        raise ValueError("A-S Study B3 replicates_per_arm must be between 1 and 10")
    expected_formal_sessions = (
        len(protocol["public_world_seeds"]) * len(B3_ARMS) * replicates_per_arm
    )
    if int(execution.get("formal_sessions", -1)) != expected_formal_sessions:
        raise ValueError("A-S Study B3 formal session denominator drifted")
    if int(execution.get("canary_sessions", -1)) != len(B3_ARMS):
        raise ValueError("A-S Study B3 canary session denominator drifted")
    action_selection_encoding = str(
        protocol.get("action_selection_encoding", "query_id")
    )
    if action_selection_encoding not in B3_ACTION_SELECTION_ENCODINGS:
        raise ValueError("A-S Study B3 action selection encoding is unsupported")
    stage_status_encoding = str(
        protocol.get("stage_status_encoding", "explicit_const")
    )
    if stage_status_encoding not in B3_STAGE_STATUS_ENCODINGS:
        raise ValueError("A-S Study B3 stage status encoding is unsupported")
    worlds = {int(item["world_seed"]): item for item in public_truth["worlds"]}
    cells: list[dict[str, Any]] = []
    cluster_packets: list[dict[str, Any]] = []
    for world_index, seed in enumerate(protocol["public_world_seeds"]):
        seed = int(seed)
        world = worlds[seed]
        evidence = []
        for query in roster["evidence_queries"]:
            query_id = str(query["query_id"])
            evidence.append(
                {
                    **_public_query(query),
                    "reference_linear_observations": deepcopy(
                        world["linear_truth"][query_id]
                    ),
                    "target_observations": deepcopy(world["power_truth"][query_id]),
                }
            )
        scoring_queries = []
        for action_index, item in enumerate(roster["scoring_queries"]):
            public_query = _public_query(item)
            if action_selection_encoding == "zero_based_index":
                public_query["action_index"] = action_index
            scoring_queries.append(public_query)
        scoring_truth = {
            str(query["query_id"]): deepcopy(world["power_truth"][str(query["query_id"])])
            for query in roster["scoring_queries"]
        }
        ordered_action_ids = sorted(
            scoring_truth,
            key=lambda query_id: (-float(scoring_truth[query_id]["score"]), query_id),
        )
        hidden_action_ranks = {
            query_id: rank for rank, query_id in enumerate(ordered_action_ids, start=1)
        }
        best_action_query_id = ordered_action_ids[0]
        best_action_score = float(scoring_truth[best_action_query_id]["score"])
        worst_action_score = min(float(row["score"]) for row in scoring_truth.values())
        maximum_available_action_gain = best_action_score - float(
            world["evidence_incumbent_score"]
        )
        action_opportunity_eligible = bool(world["preflight_passed"])
        action_opportunity_threshold = float(protocol["qualification"]["minimum_action_gain"])
        public_packet = {
            "schema_version": (
                "chemworld-work-ii-as-study-b3-public-packet-0.2"
                if action_selection_encoding == "zero_based_index"
                else "chemworld-work-ii-as-study-b3-public-packet-0.1"
            ),
            "cluster_id": f"A_S_B3--partition-discovery--seed{seed}",
            "task_id": "partition-discovery",
            "metric_range": [0.0, 1.0],
            "candidate_mechanism_families": _candidate_families(),
            "evidence": evidence,
            "scoring_action_queries": scoring_queries,
        }
        packet_hash = canonical_json_sha256(public_packet)
        cluster_id = str(public_packet["cluster_id"])
        cluster_packets.append(
            {
                "cluster_id": cluster_id,
                "world_seed": seed,
                "public_packet_sha256": packet_hash,
                "evidence_query_count": 8,
                "scoring_action_query_count": 8,
                "scoring_term_count": 32,
                "evidence_incumbent_score": world["evidence_incumbent_score"],
                "maximum_available_action_gain": maximum_available_action_gain,
                "action_opportunity_eligible": action_opportunity_eligible,
                "action_opportunity_threshold": action_opportunity_threshold,
            }
        )
        rotated = [B3_ARMS[(world_index + offset) % 3] for offset in range(3)]
        for replicate_index in range(1, replicates_per_arm + 1):
            replicate_block_id = f"{cluster_id}--replicate-{replicate_index:02d}"
            for arm in rotated:
                cell_id = (
                    f"{cluster_id}--{arm}"
                    if replicates_per_arm == 1
                    else f"{replicate_block_id}--{arm}"
                )
                cells.append(
                    {
                        "cell_index": len(cells) + 1,
                        "study_id": protocol["study_id"],
                        "cell_id": cell_id,
                        "cluster_id": cluster_id,
                        "replicate_block_id": replicate_block_id,
                        "replicate_index": replicate_index,
                        "locus": "A_S_B3",
                        "task_id": "partition-discovery",
                        "world_seed": seed,
                        "arm": arm,
                        "action_selection_encoding": action_selection_encoding,
                        "stage_status_encoding": stage_status_encoding,
                        "initial_world_model": _initial_model(arm),
                        "public_packet": deepcopy(public_packet),
                        "public_packet_sha256": packet_hash,
                        "scoring_truth": scoring_truth,
                        "hidden_action_ranks": hidden_action_ranks,
                        "best_action_query_id": best_action_query_id,
                        "best_action_score": best_action_score,
                        "worst_action_score": worst_action_score,
                        "evidence_incumbent_score": float(world["evidence_incumbent_score"]),
                        "maximum_available_action_gain": maximum_available_action_gain,
                        "action_opportunity_eligible": action_opportunity_eligible,
                        "action_opportunity_threshold": action_opportunity_threshold,
                    }
                )
    manifest: dict[str, Any] = {
        "schema_version": B3_MANIFEST_VERSION,
        "study_id": protocol["study_id"],
        "protocol_path": protocol_file.relative_to(root).as_posix(),
        "protocol_status": protocol.get("status"),
        "provider_free_artifact_root": (
            target.relative_to(root).as_posix()
            if target.is_relative_to(root)
            else str(target)
        ),
        "provider_free_source_study_id": provider_free_source_study_id,
        "provider_free_truth_reused": provider_free_source_study_id
        != protocol["study_id"],
        "provider": deepcopy(protocol["provider"]),
        "execution": deepcopy(execution),
        "action_selection_encoding": action_selection_encoding,
        "stage_status_encoding": stage_status_encoding,
        "arms": list(B3_ARMS),
        "cell_count": expected_formal_sessions,
        "cluster_count": 5,
        "replicate_block_count": len(protocol["public_world_seeds"]) * replicates_per_arm,
        "replicates_per_arm": replicates_per_arm,
        "scoring_term_count": 32,
        "participant_physical_experiment_count": 0,
        "qualification_sha256": qualification["qualification_sha256"],
        "public_truth_sha256": public_truth["public_truth_sha256"],
        "roster_sha256": roster["roster_sha256"],
        "cluster_packets": cluster_packets,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def b3_output_schema(
    scoring_queries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    action_selection_encoding: str = "query_id",
    stage_status_encoding: str = "explicit_const",
) -> dict[str, Any]:
    if stage not in {"pre", "post"}:
        raise ValueError("A-S Study B3 stage must be pre or post")
    if action_selection_encoding not in B3_ACTION_SELECTION_ENCODINGS:
        raise ValueError("A-S Study B3 action selection encoding is unsupported")
    if stage_status_encoding not in B3_STAGE_STATUS_ENCODINGS:
        raise ValueError("A-S Study B3 stage status encoding is unsupported")
    query_ids = [str(item["query_id"]) for item in scoring_queries]
    metric_properties = {
        metric_id: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for metric_id in B3_METRIC_IDS
    }
    properties: dict[str, Any] = {
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
                "law_type": {
                    "type": "string",
                    "const": "reference_coefficient_power",
                },
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
                        "properties": metric_properties,
                    },
                },
            },
        },
        "model_summary": {"type": "string", "maxLength": 1200},
    }
    required = [
        "mechanism_family",
        "estimated_reference_exponent",
        "confidence",
        "typed_law",
        "predictions",
        "model_summary",
    ]
    if stage_status_encoding == "explicit_const":
        properties["status"] = {
            "type": "string",
            "const": f"{stage}_submission_complete",
        }
        required.insert(0, "status")
    if stage == "post":
        if action_selection_encoding == "query_id":
            properties["selected_action_query_id"] = {
                "type": "string",
                "enum": query_ids,
            }
            required.append("selected_action_query_id")
        else:
            properties["selected_action_index"] = {
                "type": "integer",
                "enum": list(range(len(query_ids))),
            }
            required.append("selected_action_index")
        properties["evidence_assessment"] = {"type": "string", "maxLength": 1200}
        required.append("evidence_assessment")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def validate_b3_payload(
    payload: Mapping[str, Any],
    scoring_queries: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    action_selection_encoding: str = "query_id",
    stage_status_encoding: str = "explicit_const",
) -> list[str]:
    errors: list[str] = []
    if stage_status_encoding not in B3_STAGE_STATUS_ENCODINGS:
        raise ValueError("A-S Study B3 stage status encoding is unsupported")
    if (
        stage_status_encoding == "explicit_const"
        and payload.get("status") != f"{stage}_submission_complete"
    ):
        errors.append(f"{stage} status is invalid")
    family = payload.get("mechanism_family")
    if family not in B3_FAMILIES:
        errors.append(f"{stage} mechanism family is invalid")
    exponent = payload.get("estimated_reference_exponent")
    if (
        isinstance(exponent, bool)
        or not isinstance(exponent, int | float)
        or not 0.25 <= float(exponent) <= 3.0
    ):
        errors.append(f"{stage} exponent is invalid")
    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append(f"{stage} confidence is invalid")
    model_summary = payload.get("model_summary")
    if not isinstance(model_summary, str) or len(model_summary) > 1200:
        errors.append(f"{stage} model summary is invalid")
    typed = payload.get("typed_law")
    if not isinstance(typed, Mapping):
        errors.append(f"{stage} typed law is unavailable")
    else:
        if typed.get("law_type") != "reference_coefficient_power":
            errors.append(f"{stage} typed law type is invalid")
        if typed.get("mechanism_family") != family:
            errors.append(f"{stage} typed-law family differs from the family choice")
        typed_exponent = typed.get("reference_exponent")
        if (
            isinstance(typed_exponent, bool)
            or not isinstance(typed_exponent, int | float)
            or not math.isclose(float(typed_exponent), float(exponent), abs_tol=1.0e-12)
        ):
            errors.append(f"{stage} typed-law exponent differs from the exponent estimate")
    expected = {
        str(query["query_id"]): set(map(str, query["metric_ids"]))
        for query in scoring_queries
    }
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return [*errors, f"{stage} predictions are unavailable"]
    observed: dict[str, set[str]] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            errors.append(f"{stage} prediction is malformed")
            continue
        query_id = prediction.get("query_id")
        metrics = prediction.get("metrics")
        if not isinstance(query_id, str) or query_id in observed:
            errors.append(f"{stage} prediction has an invalid or duplicate query ID")
            continue
        if not isinstance(metrics, Mapping):
            errors.append(f"{stage} prediction {query_id} lacks metrics")
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
    if observed != expected:
        errors.append(f"{stage} prediction denominator differs from the contract")
    if stage == "post":
        evidence_assessment = payload.get("evidence_assessment")
        if not isinstance(evidence_assessment, str) or len(evidence_assessment) > 1200:
            errors.append("post evidence assessment is invalid")
        try:
            resolve_b3_selected_action_query_id(
                payload,
                scoring_queries,
                action_selection_encoding=action_selection_encoding,
            )
        except ValueError as error:
            errors.append(str(error))
    return errors


def resolve_b3_selected_action_query_id(
    payload: Mapping[str, Any],
    scoring_queries: Sequence[Mapping[str, Any]],
    *,
    action_selection_encoding: str = "query_id",
) -> str:
    """Resolve a participant action choice to the canonical scoring-query ID."""

    query_ids = [str(item["query_id"]) for item in scoring_queries]
    if action_selection_encoding == "query_id":
        selected = payload.get("selected_action_query_id")
        if selected not in query_ids:
            raise ValueError("post selected action query ID is invalid")
        return str(selected)
    if action_selection_encoding == "zero_based_index":
        selected = payload.get("selected_action_index")
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or not 0 <= selected < len(query_ids)
        ):
            raise ValueError("post selected action index is invalid")
        return query_ids[selected]
    raise ValueError("A-S Study B3 action selection encoding is unsupported")


def evaluate_b3_selected_action(
    cell: Mapping[str, Any], selected_query_id: str
) -> dict[str, Any]:
    """Score one B3 candidate choice against the complete hidden scoring roster."""

    truth = cell["scoring_truth"]
    if selected_query_id not in truth:
        raise ValueError("B3 selected action query ID is absent from hidden truth")
    candidate_scores = {
        str(query_id): float(metrics["score"]) for query_id, metrics in truth.items()
    }
    if not candidate_scores or not all(math.isfinite(score) for score in candidate_scores.values()):
        raise ValueError("B3 action truth scores are empty or non-finite")
    ordered_ids = sorted(
        candidate_scores,
        key=lambda query_id: (-candidate_scores[query_id], query_id),
    )
    ranks = {query_id: rank for rank, query_id in enumerate(ordered_ids, start=1)}
    best_query_id = ordered_ids[0]
    best_score = candidate_scores[best_query_id]
    worst_score = min(candidate_scores.values())
    selected_score = candidate_scores[selected_query_id]
    raw_regret = best_score - selected_score
    score_range = best_score - worst_score
    normalized_regret = 0.0 if score_range <= 1.0e-12 else raw_regret / score_range
    incumbent = float(cell["evidence_incumbent_score"])
    maximum_available_gain = best_score - incumbent
    threshold = float(cell.get("action_opportunity_threshold", 0.02))
    eligible = bool(
        cell.get("action_opportunity_eligible", maximum_available_gain >= threshold)
    )
    return {
        "query_id": selected_query_id,
        "true_score": selected_score,
        "selected_true_rank": int(ranks[selected_query_id]),
        "top1_selected": selected_query_id == best_query_id,
        "best_action_query_id": best_query_id,
        "best_action_score": best_score,
        "worst_action_score": worst_score,
        "raw_regret": raw_regret,
        "normalized_regret": normalized_regret,
        "evidence_incumbent_score": incumbent,
        "gain_over_evidence_incumbent": selected_score - incumbent,
        "maximum_available_action_gain": maximum_available_gain,
        "action_opportunity_threshold": threshold,
        "action_opportunity_eligible": eligible,
        "participant_executed_before_selection": False,
    }


def summarize_b3_canary_closeout(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    canary_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a failure-aware terminal summary for the three-session B3 canary."""

    first_cluster = str(manifest["cells"][0]["cluster_id"])
    expected_cells = [
        cell
        for cell in manifest["cells"]
        if str(cell["cluster_id"]) == first_cluster
        and int(cell.get("replicate_index", 1)) == 1
    ]
    expected_ids = {str(cell["cell_id"]) for cell in expected_cells}
    observed_ids = {str(result.get("cell_id")) for result in results}
    if expected_ids != observed_ids:
        raise ValueError("B3 canary result denominator differs from the manifest")
    rows: list[dict[str, Any]] = []
    for result in results:
        receipts = result.get("provider_receipts")
        receipts = receipts if isinstance(receipts, list) else []
        failure = result.get("failure")
        failure = failure if isinstance(failure, Mapping) else {}
        selected = result.get("selected_action")
        selected = selected if isinstance(selected, Mapping) else {}
        post = result.get("post_submission")
        post = post if isinstance(post, Mapping) else {}
        post_scores = result.get("scores", {}).get("post", {})
        post_scores = post_scores if isinstance(post_scores, Mapping) else {}
        rows.append(
            {
                "cell_id": result["cell_id"],
                "arm": result["arm"],
                "status": result["status"],
                "elapsed_s": result.get("elapsed_s"),
                "provider_attempt_count": int(result.get("provider_attempt_count", 0)),
                "provider_turn_count": len(receipts),
                "completed_provider_turn_count": sum(
                    receipt.get("status") == "completed"
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                ),
                "tool_event_count": sum(
                    int(receipt.get("tool_event_count", 0) or 0)
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                ),
                "failure_classification": failure.get("classification"),
                "failure_type": failure.get("type"),
                "failure_message": failure.get("message"),
                "same_thread": result.get("same_thread"),
                "post_family": post.get("mechanism_family"),
                "post_exponent": post.get("estimated_reference_exponent"),
                "post_prediction_mae": post_scores.get(
                    "mean_normalized_absolute_error"
                ),
                "selected_true_rank": selected.get("selected_true_rank"),
                "top1_selected": selected.get("top1_selected"),
                "normalized_regret": selected.get("normalized_regret"),
                "action_opportunity_eligible": selected.get(
                    "action_opportunity_eligible"
                ),
            }
        )
    completed = [row for row in rows if row["status"] == "completed"]
    failures = [row for row in rows if row["status"] != "completed"]
    result: dict[str, Any] = {
        "schema_version": "chemworld-work-ii-as-study-b3-canary-closeout-0.1",
        "study_id": manifest["study_id"],
        "status": (
            "terminal_canary_rejected_before_formal"
            if canary_summary.get("qualified") is False
            else "canary_qualified"
        ),
        "canary_qualified": canary_summary.get("qualified") is True,
        "scheduled_canary_session_count": len(expected_ids),
        "observed_canary_session_count": len(rows),
        "completed_canary_session_count": len(completed),
        "failed_canary_session_count": len(failures),
        "participant_schema_failure_count": sum(
            row["failure_classification"] == "participant_schema" for row in failures
        ),
        "infrastructure_failure_count": sum(
            row["failure_classification"]
            in {"provider_infrastructure", "runner_infrastructure"}
            for row in failures
        ),
        "provider_session_attempt_count": sum(
            row["provider_attempt_count"] for row in rows
        ),
        "provider_turn_count": sum(row["provider_turn_count"] for row in rows),
        "completed_provider_turn_count": sum(
            row["completed_provider_turn_count"] for row in rows
        ),
        "tool_event_count": sum(row["tool_event_count"] for row in rows),
        "participant_physical_experiment_count": 0,
        "scheduled_formal_session_count": int(manifest["cell_count"]),
        "launched_formal_session_count": 0,
        "unstarted_formal_session_count": int(manifest["cell_count"]),
        "scientific_outcomes_used_for_design": False,
        "outcome_replacement_count": 0,
        "cell_rows": sorted(rows, key=lambda row: row["cell_id"]),
        "canary_errors": list(canary_summary.get("errors", [])),
    }
    result["summary_sha256"] = canonical_json_sha256(result)
    return result


def summarize_b3_results(
    manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = {str(item["cell_id"]) for item in manifest["cells"]}
    observed_ids = {str(item.get("cell_id")) for item in results}
    completed = [item for item in results if item.get("status") == "completed"]
    failures = [item for item in results if item.get("status") != "completed"]
    manifest_cells = {str(item["cell_id"]): item for item in manifest["cells"]}
    cell_rows: list[dict[str, Any]] = []
    replicate_arms: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in completed:
        pre = result["pre_submission"]
        post = result["post_submission"]
        pre_error = float(result["scores"]["pre"]["mean_normalized_absolute_error"])
        post_error = float(result["scores"]["post"]["mean_normalized_absolute_error"])
        cell = manifest_cells[str(result["cell_id"])]
        public_packet = cell.get("public_packet")
        scoring_queries = (
            public_packet.get("scoring_action_queries")
            if isinstance(public_packet, Mapping)
            else None
        )
        if not isinstance(scoring_queries, list):
            scoring_queries = [
                {"query_id": query_id} for query_id in cell["scoring_truth"]
            ]
        selected = resolve_b3_selected_action_query_id(
            post,
            scoring_queries,
            action_selection_encoding=str(
                cell.get(
                    "action_selection_encoding",
                    manifest.get("action_selection_encoding", "query_id"),
                )
            ),
        )
        action = evaluate_b3_selected_action(cell, selected)
        selected_score = float(action["true_score"])
        incumbent = float(action["evidence_incumbent_score"])
        row = {
            "cell_id": result["cell_id"],
            "cluster_id": result["cluster_id"],
            "replicate_block_id": result.get(
                "replicate_block_id", f"{result['cluster_id']}--replicate-01"
            ),
            "replicate_index": int(result.get("replicate_index", 1)),
            "world_seed": result["world_seed"],
            "arm": result["arm"],
            "pre_error": pre_error,
            "post_error": post_error,
            "update_gain": pre_error - post_error,
            "pre_family": pre["mechanism_family"],
            "post_family": post["mechanism_family"],
            "pre_exponent": float(pre["estimated_reference_exponent"]),
            "post_exponent": float(post["estimated_reference_exponent"]),
            "post_exponent_absolute_error": abs(
                float(post["estimated_reference_exponent"]) - 1.75
            ),
            "selected_action_query_id": selected,
            "selected_action_true_score": selected_score,
            "selected_true_rank": int(action["selected_true_rank"]),
            "top1_selected": bool(action["top1_selected"]),
            "best_action_query_id": action["best_action_query_id"],
            "best_action_score": float(action["best_action_score"]),
            "raw_regret": float(action["raw_regret"]),
            "normalized_regret": float(action["normalized_regret"]),
            "evidence_incumbent_score": incumbent,
            "selected_action_gain": selected_score - incumbent,
            "maximum_available_action_gain": float(action["maximum_available_action_gain"]),
            "action_opportunity_eligible": bool(action["action_opportunity_eligible"]),
            "action_opportunity_threshold": float(action["action_opportunity_threshold"]),
        }
        cell_rows.append(row)
        replicate_arms[str(row["replicate_block_id"])][str(row["arm"])] = row
    replicate_rows: list[dict[str, Any]] = []
    for replicate_block_id, arms in sorted(replicate_arms.items()):
        if set(arms) != set(B3_ARMS):
            continue
        replicate_rows.append(
            {
                "replicate_block_id": replicate_block_id,
                "replicate_index": arms["opaque"]["replicate_index"],
                "cluster_id": arms["opaque"]["cluster_id"],
                "world_seed": arms["opaque"]["world_seed"],
                "post_error_by_arm": {arm: arms[arm]["post_error"] for arm in B3_ARMS},
                "post_family_by_arm": {arm: arms[arm]["post_family"] for arm in B3_ARMS},
                "post_exponent_error_by_arm": {
                    arm: arms[arm]["post_exponent_absolute_error"] for arm in B3_ARMS
                },
                "action_gain_by_arm": {
                    arm: arms[arm]["selected_action_gain"] for arm in B3_ARMS
                },
                "selected_true_rank_by_arm": {
                    arm: arms[arm]["selected_true_rank"] for arm in B3_ARMS
                },
                "top1_selected_by_arm": {
                    arm: arms[arm]["top1_selected"] for arm in B3_ARMS
                },
                "normalized_regret_by_arm": {
                    arm: arms[arm]["normalized_regret"] for arm in B3_ARMS
                },
                "action_opportunity_eligible": arms["opaque"][
                    "action_opportunity_eligible"
                ],
                "maximum_available_action_gain": arms["opaque"][
                    "maximum_available_action_gain"
                ],
            }
        )
    expected_replicates = int(manifest.get("replicates_per_arm", 1))
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replicate_rows:
        by_cluster[str(row["cluster_id"])].append(row)
    cluster_rows = [
        {
            "cluster_id": cluster_id,
            "world_seed": rows[0]["world_seed"],
            "complete_replicate_count": len(rows),
            "expected_replicate_count": expected_replicates,
            "mean_post_error_by_arm": {
                arm: mean(row["post_error_by_arm"][arm] for row in rows) for arm in B3_ARMS
            },
            "mean_exponent_error_by_arm": {
                arm: mean(row["post_exponent_error_by_arm"][arm] for row in rows)
                for arm in B3_ARMS
            },
            "mean_action_gain_by_arm": {
                arm: mean(row["action_gain_by_arm"][arm] for row in rows) for arm in B3_ARMS
            }
            if rows[0]["action_opportunity_eligible"]
            else None,
            "mean_selected_true_rank_by_arm": {
                arm: mean(row["selected_true_rank_by_arm"][arm] for row in rows)
                for arm in B3_ARMS
            },
            "top1_rate_by_arm": {
                arm: mean(float(row["top1_selected_by_arm"][arm]) for row in rows)
                for arm in B3_ARMS
            },
            "mean_normalized_regret_by_arm": {
                arm: mean(row["normalized_regret_by_arm"][arm] for row in rows)
                for arm in B3_ARMS
            },
            "action_opportunity_eligible": rows[0]["action_opportunity_eligible"],
            "maximum_available_action_gain": rows[0]["maximum_available_action_gain"],
        }
        for cluster_id, rows in sorted(by_cluster.items())
        if len(rows) == expected_replicates
    ]
    by_arm: dict[str, Any] = {}
    for arm in B3_ARMS:
        rows = [row for row in cell_rows if row["arm"] == arm]
        gain_rows = [row for row in rows if row["action_opportunity_eligible"]]
        by_arm[arm] = {
            "completed_cell_count": len(rows),
            "all_world_rank_regret_denominator": len(rows),
            "action_opportunity_eligible_gain_denominator": len(gain_rows),
            "mean_pre_error": mean(row["pre_error"] for row in rows) if rows else None,
            "mean_post_error": mean(row["post_error"] for row in rows) if rows else None,
            "mean_update_gain": mean(row["update_gain"] for row in rows) if rows else None,
            "exact_family_recovery_count": sum(
                row["post_family"] == "FAMILY_B_POWER" for row in rows
            ),
            "exponent_within_0_10_count": sum(
                row["post_exponent_absolute_error"] <= 0.10 for row in rows
            ),
            "top1_selected_count": sum(row["top1_selected"] for row in rows),
            "mean_selected_true_rank": (
                mean(row["selected_true_rank"] for row in rows) if rows else None
            ),
            "mean_normalized_regret": (
                mean(row["normalized_regret"] for row in rows) if rows else None
            ),
            "positive_action_gain_count": sum(
                row["selected_action_gain"] > 0.0 for row in gain_rows
            ),
            "action_gain_at_least_0_02_count": sum(
                row["selected_action_gain"] >= row["action_opportunity_threshold"]
                for row in gain_rows
            ),
            "mean_action_gain": (
                mean(row["selected_action_gain"] for row in gain_rows) if gain_rows else None
            ),
        }
    eligible_cluster_ids = {
        str(item["cluster_id"])
        for item in manifest.get("cluster_packets", [])
        if item.get("action_opportunity_eligible") is True
    }
    return {
        "schema_version": B3_SUMMARY_VERSION,
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
        "complete_replicate_block_count": len(replicate_rows),
        "replicates_per_arm": expected_replicates,
        "all_world_rank_regret_cell_denominator": len(cell_rows),
        "action_opportunity_eligible_gain_cell_denominator": sum(
            row["action_opportunity_eligible"] for row in cell_rows
        ),
        "scheduled_action_opportunity_eligible_world_count": len(eligible_cluster_ids),
        "complete_action_opportunity_eligible_world_count": sum(
            row["cluster_id"] in eligible_cluster_ids for row in cluster_rows
        ),
        "missing_cell_ids": sorted(expected_ids - observed_ids),
        "unexpected_cell_ids": sorted(observed_ids - expected_ids),
        "participant_physical_experiment_count": 0,
        "scoring_term_count_per_submission": manifest["scoring_term_count"],
        "by_arm": by_arm,
        "cell_rows": sorted(cell_rows, key=lambda row: row["cell_id"]),
        "cluster_rows": cluster_rows,
        "replicate_rows": replicate_rows,
        "failures": [
            {"cell_id": item.get("cell_id"), "failure": item.get("failure")}
            for item in failures
        ],
    }


__all__ = [
    "B3_ACTION_SELECTION_ENCODINGS",
    "B3_ARMS",
    "B3_CELL_VERSION",
    "B3_FAMILIES",
    "B3_MANIFEST_VERSION",
    "B3_METRIC_IDS",
    "B3_PROTOCOL_VERSION",
    "B3_PUBLIC_TRUTH_VERSION",
    "B3_QUALIFICATION_VERSION",
    "B3_STAGE_STATUS_ENCODINGS",
    "B3_SUMMARY_VERSION",
    "b3_output_schema",
    "build_b3_candidate_queries",
    "build_b3_manifest",
    "evaluate_b3_selected_action",
    "prepare_b3",
    "resolve_b3_selected_action_query_id",
    "select_b3_rosters",
    "summarize_b3_canary_closeout",
    "summarize_b3_results",
    "validate_b3_payload",
]
