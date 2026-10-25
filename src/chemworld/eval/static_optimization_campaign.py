"""Aggregate the frozen S0 two-flagship campaign at the world level."""

# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired

import numpy as np

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.static_campaign_statistics import ScoreSummary, score_summary
from chemworld.eval.static_campaign_statistics import mean_mapping as _mean_mapping

STATIC_S0_CAMPAIGN_SUMMARY_VERSION = "chemworld-static-s0-campaign-summary-1.0"
_BASELINE_CELLS_MARKER = b'\n  "cells":'


class _ScoreSummary(ScoreSummary):
    world_bootstrap_95_interval: NotRequired[list[float]]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: Sequence[float]) -> _ScoreSummary:
    return {**score_summary(values)}


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    draws: int,
    label: str,
) -> list[float]:
    normalized = np.asarray([float(value) for value in values], dtype=float)
    if normalized.size == 0 or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    label_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(int(seed) ^ label_seed)
    indices = rng.integers(
        0,
        normalized.size,
        size=(int(draws), normalized.size),
    )
    means = normalized[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _read_baseline_report_header(path: Path) -> dict[str, Any]:
    """Read the small aggregate prefix without loading multi-hundred-MB cells."""

    payload = bytearray()
    with path.open("rb") as handle:
        while _BASELINE_CELLS_MARKER not in payload:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"baseline report has no cells marker: {path}")
            payload.extend(chunk)
    prefix = bytes(payload).split(_BASELINE_CELLS_MARKER, 1)[0].rstrip()
    if prefix.endswith(b","):
        prefix = prefix[:-1]
    parsed = json.loads(prefix + b"\n}")
    if not isinstance(parsed, dict):
        raise ValueError(f"baseline report prefix is not an object: {path}")
    return parsed


def _audit_verified(audit: Mapping[str, Any]) -> bool:
    return all(
        (
            audit.get("static_world_verified") is True,
            audit.get("no_mechanism_fields_in_plans") is True,
            audit.get("report_receipt_hashes_match") is True,
            audit.get("atomic_executor_verified") is True,
            audit.get("final_recommendation_validation_matches") is True,
            audit.get("replay", {}).get("all_verified") is True,
        )
    )


def _algorithm_role(information_condition: str) -> str:
    if information_condition == "opaque_score_and_public_measurements":
        return "information_matched"
    if information_condition.startswith("negative_control_"):
        return "negative_control"
    if information_condition.startswith("privileged_"):
        return "privileged_calibration"
    raise ValueError(f"unknown baseline information condition: {information_condition}")


def _validate_campaign_index(
    root: Path,
    *,
    expected_cells: int,
    freeze_sha256: str,
) -> dict[str, Any]:
    path = root / "campaign_execution_index.json"
    index = _load_json(path)
    if (
        index.get("all_requested_cells_completed") is not True
        or index.get("all_exact_replay_verified") is not True
        or len(index.get("results", [])) != expected_cells
        or index.get("freeze_sha256") != freeze_sha256
    ):
        raise ValueError(f"campaign execution index is incomplete or stale: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "source_commit": str(index["source_commit"]),
        "result_count": len(index["results"]),
        "all_exact_replay_verified": True,
    }


def _participant_task(
    root: Path,
    *,
    track: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> tuple[dict[str, Any], dict[int, float]]:
    task_root = root / "participants" / str(track["track_id"])
    worlds: list[dict[str, Any]] = []
    scores_by_world: dict[int, float] = {}
    predictive_rows: list[dict[str, float]] = []
    declared_rows: list[dict[str, float]] = []
    audit_paths: list[dict[str, Any]] = []
    accounting = {
        "exploration_experiments": 0,
        "predictive_physical_experiments": 0,
        "blind_validation_experiments": 0,
        "total_physical_experiments": 0,
        "provider_calls": 0,
        "method_failures": 0,
    }
    for seed in range(int(track["world_count"])):
        world_root = task_root / f"world-{seed:02d}"
        report_path = world_root / "report.json"
        audit_path = world_root / "postrun_audit.json"
        report = _load_json(report_path)
        audit = _load_json(audit_path)
        if not _audit_verified(audit):
            raise ValueError(f"participant audit is not verified: {audit_path}")
        cell = report["cells"][0]
        if int(cell["cell"]["world_seed"]) != seed:
            raise ValueError(f"participant world seed mismatch: {report_path}")
        score = float(cell["primary_score"])
        scores_by_world[seed] = score
        predictive = {
            key: float(cell["predictive_validation"]["score"][key])
            for key in (
                "directional_accuracy",
                "confidence_brier_score",
                "nontrivial_actual_effect_rate",
            )
        }
        predictive_rows.append(predictive)
        declared_rows.append(
            {
                key: float(value)
                for key, value in audit["world_understanding"]["mean_scores"].items()
            }
        )
        validation = cell["validation"]
        worlds.append(
            {
                "world_seed": seed,
                "primary_score": score,
                "recommendation_type": cell["final_synthesis"]["recommendation"][
                    "recommendation_type"
                ],
                "recommendation_gain_over_incumbent": float(
                    validation["recommendation_gain_over_incumbent_mean"]
                ),
                "predictive": predictive,
                "report_sha256": _sha256(report_path),
                "audit_sha256": _sha256(audit_path),
            }
        )
        accounting["exploration_experiments"] += int(
            cell["completed_experiment_count"]
        )
        accounting["predictive_physical_experiments"] += int(
            cell["completed_predictive_validation_experiment_count"]
        )
        accounting["blind_validation_experiments"] += int(
            cell["completed_validation_experiment_count"]
        )
        accounting["total_physical_experiments"] += int(
            cell["total_physical_experiment_count"]
        )
        accounting["provider_calls"] += int(report["provider_call_count"])
        accounting["method_failures"] += int(report["method_failure_cell_count"])
        audit_paths.append(
            {
                "world_seed": seed,
                "path": str(audit_path),
                "verified": True,
            }
        )
    ordered_scores = [scores_by_world[seed] for seed in sorted(scores_by_world)]
    primary = _summary(ordered_scores)
    primary["world_bootstrap_95_interval"] = _bootstrap_interval(
        ordered_scores,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
        label=f"{track['track_id']}:participant",
    )
    return (
        {
            "method_id": track["method_id"],
            "provider": track["provider"],
            "model_id": track["model_id"],
            "reasoning_effort": track["reasoning_effort"],
            "primary_score": primary,
            "worlds": worlds,
            "predictive_secondary_diagnostic_mean": _mean_mapping(predictive_rows),
            "declared_world_understanding_secondary_diagnostic_mean": _mean_mapping(
                declared_rows
            ),
            "all_recommendations_tested": all(
                row["recommendation_type"] == "tested" for row in worlds
            ),
            "all_recommendation_gains_over_incumbent_zero": all(
                abs(float(row["recommendation_gain_over_incumbent"])) <= 1e-12
                for row in worlds
            ),
            "accounting": accounting,
            "audits": audit_paths,
        },
        scores_by_world,
    )


def _baseline_task(
    root: Path,
    *,
    track: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, float]], dict[str, Any]]:
    protocol = _load_json(Path(str(track["protocol_path"])))
    algorithm_contracts = protocol["algorithms"]
    task_root = root / "baselines" / str(track["track_id"])
    by_algorithm: dict[str, dict[int, float]] = {}
    audit_count = 0
    exploration_count = 0
    validation_count = 0
    for seed in range(int(track["world_count"])):
        world_root = task_root / f"world-{seed:02d}"
        report_path = world_root / "report.json"
        audit_path = world_root / "postrun_audit.json"
        header = _read_baseline_report_header(report_path)
        audit = _load_json(audit_path)
        if not _audit_verified(audit):
            raise ValueError(f"baseline audit is not verified: {audit_path}")
        audit_count += 1
        exploration_count += int(audit["replay"]["replayed_experiment_count"])
        validation_count += int(
            audit["replay"]["replayed_validation_experiment_count"]
        )
        for algorithm in header["aggregate"]["algorithms"]:
            algorithm_id = str(algorithm["algorithm_id"])
            by_algorithm.setdefault(algorithm_id, {})[seed] = float(
                algorithm["validated_final_score"]["mean"]
            )
            if int(algorithm["run_count"]) != int(track["algorithm_seed_count"]):
                raise ValueError(
                    f"baseline algorithm seed count mismatch: {report_path}"
                )
    rows: list[dict[str, Any]] = []
    for algorithm_id, world_scores in sorted(by_algorithm.items()):
        if set(world_scores) != set(range(int(track["world_count"]))):
            raise ValueError(f"baseline world coverage is incomplete: {algorithm_id}")
        values = [world_scores[seed] for seed in sorted(world_scores)]
        summary = _summary(values)
        summary["world_bootstrap_95_interval"] = _bootstrap_interval(
            values,
            seed=bootstrap_seed,
            draws=bootstrap_draws,
            label=f"{track['track_id']}:{algorithm_id}:baseline",
        )
        information_condition = str(
            algorithm_contracts[algorithm_id]["information_condition"]
        )
        rows.append(
            {
                "algorithm_id": algorithm_id,
                "role": _algorithm_role(information_condition),
                "information_condition": information_condition,
                "validated_final_score": summary,
                "world_scores_after_algorithm_seed_mean": [
                    {"world_seed": seed, "score": world_scores[seed]}
                    for seed in sorted(world_scores)
                ],
            }
        )
    return (
        rows,
        by_algorithm,
        {
            "world_audit_count": audit_count,
            "all_world_audits_verified": audit_count == int(track["world_count"]),
            "cell_count": int(track["cell_count"]),
            "algorithm_count": len(rows),
            "algorithm_seed_count": int(track["algorithm_seed_count"]),
            "exploration_experiments": exploration_count,
            "blind_validation_experiments": validation_count,
            "total_physical_experiments": exploration_count + validation_count,
        },
    )


def _paired_comparisons(
    *,
    task_id: str,
    participant: Mapping[int, float],
    baselines: Mapping[str, Mapping[int, float]],
    baseline_rows: Sequence[Mapping[str, Any]],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> list[dict[str, Any]]:
    roles = {str(row["algorithm_id"]): str(row["role"]) for row in baseline_rows}
    comparisons: list[dict[str, Any]] = []
    for algorithm_id, world_scores in sorted(baselines.items()):
        seeds = sorted(participant)
        differences = [
            float(participant[seed]) - float(world_scores[seed]) for seed in seeds
        ]
        comparisons.append(
            {
                "algorithm_id": algorithm_id,
                "role": roles[algorithm_id],
                "participant_minus_baseline": {
                    **_summary(differences),
                    "world_bootstrap_95_interval": _bootstrap_interval(
                        differences,
                        seed=bootstrap_seed,
                        draws=bootstrap_draws,
                        label=f"{task_id}:{algorithm_id}:paired",
                    ),
                },
                "participant_win_tie_loss": {
                    "wins": sum(value > 1e-12 for value in differences),
                    "ties": sum(abs(value) <= 1e-12 for value in differences),
                    "losses": sum(value < -1e-12 for value in differences),
                },
                "world_differences": [
                    {"world_seed": seed, "difference": difference}
                    for seed, difference in zip(seeds, differences, strict=True)
                ],
                "interpretation": "descriptive_only_no_preregistered_superiority_test",
            }
        )
    return sorted(
        comparisons,
        key=lambda row: float(row["participant_minus_baseline"]["mean"]),
        reverse=True,
    )


def build_static_s0_campaign_summary(
    *,
    manifest_path: str | Path,
    participant_root: str | Path,
    baseline_root: str | Path,
    bootstrap_seed: int = 20260729,
    bootstrap_draws: int = 200_000,
) -> dict[str, Any]:
    """Build and validate the formal two-task campaign summary."""

    manifest_file = Path(manifest_path)
    participant_path = Path(participant_root)
    baseline_path = Path(baseline_root)
    manifest = _load_json(manifest_file)
    freeze_sha256 = canonical_json_sha256(manifest)
    participant_index = _validate_campaign_index(
        participant_path,
        expected_cells=len(manifest["participant_tracks"])
        * len(manifest["world_seeds"]),
        freeze_sha256=freeze_sha256,
    )
    baseline_index = _validate_campaign_index(
        baseline_path,
        expected_cells=len(manifest["baseline_tracks"]) * len(manifest["world_seeds"]),
        freeze_sha256=freeze_sha256,
    )
    participant_tracks = {
        str(track["track_id"]): track for track in manifest["participant_tracks"]
    }
    baseline_tracks = {
        str(track["track_id"]): track for track in manifest["baseline_tracks"]
    }
    if set(participant_tracks) != set(baseline_tracks):
        raise ValueError("participant and baseline tracks are not aligned")

    tasks: dict[str, Any] = {}
    actual_accounting = {
        "participant_total_physical_experiments": 0,
        "participant_provider_calls": 0,
        "baseline_total_physical_experiments": 0,
    }
    for track_id in sorted(participant_tracks):
        participant_summary, participant_scores = _participant_task(
            participant_path,
            track=participant_tracks[track_id],
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
        )
        baseline_rows, baseline_scores, baseline_accounting = _baseline_task(
            baseline_path,
            track=baseline_tracks[track_id],
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
        )
        comparisons = _paired_comparisons(
            task_id=track_id,
            participant=participant_scores,
            baselines=baseline_scores,
            baseline_rows=baseline_rows,
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
        )
        tasks[track_id] = {
            "task_id": participant_tracks[track_id]["task_id"],
            "participant": participant_summary,
            "baselines": baseline_rows,
            "baseline_accounting": baseline_accounting,
            "paired_comparisons": comparisons,
        }
        actual_accounting["participant_total_physical_experiments"] += int(
            participant_summary["accounting"]["total_physical_experiments"]
        )
        actual_accounting["participant_provider_calls"] += int(
            participant_summary["accounting"]["provider_calls"]
        )
        actual_accounting["baseline_total_physical_experiments"] += int(
            baseline_accounting["total_physical_experiments"]
        )
    actual_accounting["campaign_total_physical_experiments"] = (
        actual_accounting["participant_total_physical_experiments"]
        + actual_accounting["baseline_total_physical_experiments"]
    )
    planned = manifest["planned_accounting"]
    expected = {
        key: int(planned[key])
        for key in (
            "participant_total_physical_experiments",
            "participant_provider_calls",
            "baseline_total_physical_experiments",
            "campaign_total_physical_experiments",
        )
    }
    if actual_accounting != expected:
        raise ValueError(
            f"formal campaign accounting mismatch: {actual_accounting} != {expected}"
        )
    return {
        "schema_version": STATIC_S0_CAMPAIGN_SUMMARY_VERSION,
        "status": "completed_audited_formal_descriptive_result",
        "formal_result": True,
        "benchmark_claim_allowed": bool(manifest["benchmark_claim_allowed"]),
        "freeze": {
            "freeze_id": manifest["freeze_id"],
            "manifest_path": str(manifest_file),
            "manifest_sha256": freeze_sha256,
        },
        "execution": {
            "participant": participant_index,
            "baselines": baseline_index,
            "formal_experiment_source_commit": baseline_index["source_commit"],
            "postrun_audit_source_commit": participant_index["source_commit"],
        },
        "bootstrap": {
            "unit": "independent_world",
            "algorithm_seeds_are_nested_technical_repeats": True,
            "seed": int(bootstrap_seed),
            "draws": int(bootstrap_draws),
            "interval": "percentile_95",
        },
        "accounting": actual_accounting,
        "tasks": tasks,
        "reporting_boundaries": {
            **manifest["reporting_boundaries"],
            "all_algorithm_comparisons": (
                "descriptive_only_no_preregistered_superiority_threshold_or_"
                "multiplicity_plan"
            ),
        },
    }


def render_static_s0_campaign_summary_zh(summary: Mapping[str, Any]) -> str:
    """Render a concise Chinese handoff from the machine-readable summary."""

    lines = [
        "# Scientific Optimization S0 v1.0 正式结果",
        "",
        "状态：两个旗舰任务均完成 10 个独立世界、每世界 20 轮探索，"
        "参与者与全部经典基线均已精确重放审计。",
        "",
        "所有算法比较均为描述性结果：本轮没有预注册 superiority 阈值或"
        "多重比较方案；特权基线只作校准。",
        "",
    ]
    for track_id in ("electrochemical", "crystallization"):
        task = summary["tasks"][track_id]
        participant = task["participant"]["primary_score"]
        interval = participant["world_bootstrap_95_interval"]
        lines.extend(
            [
                f"## {track_id}",
                "",
                (
                    f"Codex 盲测主分均值 {participant['mean']:.4f}，"
                    f"世界 bootstrap 95% 区间 "
                    f"[{interval[0]:.4f}, {interval[1]:.4f}]。"
                ),
                "",
                "| 基线 | 角色 | 基线均值 | Codex−基线 | 配对95%区间 | 胜/平/负 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        baselines = {
            row["algorithm_id"]: row for row in task["baselines"]
        }
        comparisons = sorted(
            task["paired_comparisons"],
            key=lambda row: float(baselines[row["algorithm_id"]][
                "validated_final_score"
            ]["mean"]),
            reverse=True,
        )
        for row in comparisons:
            baseline = baselines[row["algorithm_id"]]
            diff = row["participant_minus_baseline"]
            diff_interval = diff["world_bootstrap_95_interval"]
            wtl = row["participant_win_tie_loss"]
            lines.append(
                f"| {row['algorithm_id']} | {row['role']} | "
                f"{baseline['validated_final_score']['mean']:.4f} | "
                f"{diff['mean']:+.4f} | "
                f"[{diff_interval[0]:+.4f}, {diff_interval[1]:+.4f}] | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
            )
        predictive = task["participant"][
            "predictive_secondary_diagnostic_mean"
        ]
        lines.extend(
            [
                "",
                (
                    "二级预测诊断：方向准确率 "
                    f"{predictive['directional_accuracy']:.3f}，Brier "
                    f"{predictive['confidence_brier_score']:.3f}。"
                ),
                "",
            ]
        )
    accounting = summary["accounting"]
    lines.extend(
        [
            "## 证据边界",
            "",
            f"- 实际物理实验总数：{accounting['campaign_total_physical_experiments']}。",
            "- 参与者最终推荐全部为已测试条件，相对已验证 incumbent 增益均为 0。",
            "- 不允许从本轮推断 provider 因果效应或超出采样世界的广泛泛化。",
            "- 旧 0.3902/0.4829 结果已撤回，不属于本正式证据。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "STATIC_S0_CAMPAIGN_SUMMARY_VERSION",
    "build_static_s0_campaign_summary",
    "render_static_s0_campaign_summary_zh",
]
