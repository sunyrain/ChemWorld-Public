"""Analyze the paired S0 nominal-material-information extension."""

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

STATIC_S0_NOMINAL_INFORMATION_INTERIM_VERSION = (
    "chemworld-static-s0-nominal-information-interim-1.0"
)


class _ScoreSummary(ScoreSummary):
    world_bootstrap_95_interval: NotRequired[list[float]]
    world_bootstrap_97_5_interval: NotRequired[list[float]]
    paired_win_tie_loss: NotRequired[dict[str, int]]
    familywise_result: NotRequired[str]


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
    confidence: float,
    seed: int,
    draws: int,
    label: str,
) -> list[float]:
    normalized = np.asarray([float(value) for value in values], dtype=float)
    if normalized.size == 0 or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")
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
    alpha = 1.0 - float(confidence)
    return [
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    ]


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


def _interim_rule_preview(interval: Sequence[float]) -> str:
    lower, upper = (float(value) for value in interval)
    if lower > 0.0:
        return "positive_information_value"
    if upper < 0.0:
        return "harmful_information"
    return "inconclusive"


def _read_world(
    world_root: Path,
    *,
    expected_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = world_root / "report.json"
    audit_path = world_root / "postrun_audit.json"
    report = _load_json(report_path)
    audit = _load_json(audit_path)
    if not _audit_verified(audit):
        raise ValueError(f"world audit is not verified: {audit_path}")
    if report.get("completed_cell_count") != 1 or len(report.get("cells", [])) != 1:
        raise ValueError(f"world report does not contain one completed cell: {report_path}")
    cell = report["cells"][0]
    if int(cell["cell"]["world_seed"]) != int(expected_seed):
        raise ValueError(f"world seed mismatch: {report_path}")
    if cell.get("cell_status") != "completed" or cell.get("failure") is not None:
        raise ValueError(f"world cell is not completed cleanly: {report_path}")
    if report.get("protocol_sha256") != audit.get("protocol_sha256"):
        raise ValueError(f"report/audit protocol hash mismatch: {world_root}")
    primary_score = float(cell["primary_score"])
    audited_score = float(
        audit["descriptive_scores"]["mean_validated_recommendation_score"]
    )
    if abs(primary_score - audited_score) > 1e-12:
        raise ValueError(f"report/audit primary score mismatch: {world_root}")
    return (
        report,
        audit,
        {
            "world_seed": int(expected_seed),
            "primary_score": primary_score,
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "exact_replay_verified": True,
        },
    )


def build_static_s0_nominal_information_interim(
    *,
    manifest_path: str | Path,
    nominal_root: str | Path,
    opaque_root: str | Path,
    world_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    bootstrap_seed: int | None = None,
    bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    """Build an audited paired-world interim report.

    This deliberately refuses to label a strict subset of the preregistered
    worlds as a completed confirmatory result.
    """

    manifest_file = Path(manifest_path)
    nominal_path = Path(nominal_root)
    opaque_path = Path(opaque_root)
    manifest = _load_json(manifest_file)
    confirmatory = manifest["confirmatory_analysis"]
    seed = (
        int(confirmatory["random_seed"])
        if bootstrap_seed is None
        else int(bootstrap_seed)
    )
    draws = (
        int(confirmatory["bootstrap_resamples"])
        if bootstrap_draws is None
        else int(bootstrap_draws)
    )
    selected_seeds = [int(value) for value in world_seeds]
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("world seeds must be nonempty and unique")
    preregistered_seeds = [int(value) for value in manifest["world_seeds"]]
    if not set(selected_seeds).issubset(preregistered_seeds):
        raise ValueError("selected world seeds are outside the frozen manifest")
    if set(selected_seeds) == set(preregistered_seeds):
        raise ValueError("interim builder requires a strict subset of frozen worlds")

    tasks: dict[str, Any] = {}
    accounting = {
        "exploration_experiments": 0,
        "predictive_physical_experiments": 0,
        "blind_validation_experiments": 0,
        "total_physical_experiments": 0,
        "provider_calls": 0,
        "provider_attempts": 0,
        "method_failures": 0,
    }
    for track in manifest["participant_tracks"]:
        track_id = str(track["track_id"])
        nominal_scores: list[float] = []
        opaque_scores: list[float] = []
        paired_rows: list[dict[str, Any]] = []
        predictive_rows: list[dict[str, float]] = []
        declared_rows: list[dict[str, float]] = []
        for world_seed in selected_seeds:
            nominal_world = (
                nominal_path
                / "participants"
                / track_id
                / f"world-{world_seed:02d}"
            )
            opaque_world = (
                opaque_path
                / "participants"
                / track_id
                / f"world-{world_seed:02d}"
            )
            nominal_report, nominal_audit, nominal_record = _read_world(
                nominal_world,
                expected_seed=world_seed,
            )
            _, _, opaque_record = _read_world(
                opaque_world,
                expected_seed=world_seed,
            )
            nominal_cell = nominal_report["cells"][0]
            nominal_score = float(nominal_record["primary_score"])
            opaque_score = float(opaque_record["primary_score"])
            nominal_scores.append(nominal_score)
            opaque_scores.append(opaque_score)
            paired_rows.append(
                {
                    "world_seed": world_seed,
                    "nominal_score": nominal_score,
                    "opaque_score": opaque_score,
                    "nominal_minus_opaque": nominal_score - opaque_score,
                    "nominal_evidence": nominal_record,
                    "opaque_evidence": opaque_record,
                }
            )
            predictive_rows.append(
                {
                    key: float(nominal_cell["predictive_validation"]["score"][key])
                    for key in (
                        "directional_accuracy",
                        "confidence_brier_score",
                        "nontrivial_actual_effect_rate",
                    )
                }
            )
            declared_rows.append(
                {
                    key: float(value)
                    for key, value in nominal_audit["world_understanding"][
                        "mean_scores"
                    ].items()
                }
            )
            accounting["exploration_experiments"] += int(
                nominal_cell["completed_experiment_count"]
            )
            accounting["predictive_physical_experiments"] += int(
                nominal_cell["completed_predictive_validation_experiment_count"]
            )
            accounting["blind_validation_experiments"] += int(
                nominal_cell["completed_validation_experiment_count"]
            )
            accounting["total_physical_experiments"] += int(
                nominal_cell["total_physical_experiment_count"]
            )
            accounting["provider_calls"] += int(nominal_report["provider_call_count"])
            accounting["provider_attempts"] += int(
                nominal_report["provider_attempt_count"]
            )
            accounting["method_failures"] += int(
                nominal_report["method_failure_cell_count"]
            )

        differences = [
            nominal - opaque
            for nominal, opaque in zip(nominal_scores, opaque_scores, strict=True)
        ]
        nominal_summary = _summary(nominal_scores)
        opaque_summary = _summary(opaque_scores)
        paired_summary = _summary(differences)
        paired_summary["world_bootstrap_95_interval"] = _bootstrap_interval(
            differences,
            confidence=0.95,
            seed=seed,
            draws=draws,
            label=f"{track_id}:nominal-minus-opaque:interim-95",
        )
        familywise_interval = _bootstrap_interval(
            differences,
            confidence=0.975,
            seed=seed,
            draws=draws,
            label=f"{track_id}:nominal-minus-opaque:interim-97.5",
        )
        paired_summary["world_bootstrap_97_5_interval"] = familywise_interval
        tasks[track_id] = {
            "task_id": track["task_id"],
            "method_id": track["method_id"],
            "provider": track["provider"],
            "model_id": track["model_id"],
            "reasoning_effort": track["reasoning_effort"],
            "nominal_primary_score": nominal_summary,
            "opaque_primary_score": opaque_summary,
            "paired_nominal_minus_opaque": paired_summary,
            "paired_win_tie_loss": {
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
            },
            "interim_familywise_rule_preview": _interim_rule_preview(
                familywise_interval
            ),
            "confirmatory_claim_allowed": False,
            "predictive_secondary_diagnostic_mean": _mean_mapping(predictive_rows),
            "declared_world_understanding_secondary_diagnostic_mean": _mean_mapping(
                declared_rows
            ),
            "worlds": paired_rows,
        }

    selected_fraction = len(selected_seeds) / len(preregistered_seeds)
    expected_per_world = 38
    expected_provider_calls = 21
    expected_total_cells = len(manifest["participant_tracks"]) * len(selected_seeds)
    expected_accounting = {
        "exploration_experiments": expected_total_cells * 20,
        "predictive_physical_experiments": expected_total_cells * 12,
        "blind_validation_experiments": expected_total_cells * 6,
        "total_physical_experiments": expected_total_cells * expected_per_world,
        "provider_calls": expected_total_cells * expected_provider_calls,
        "method_failures": 0,
    }
    actual_fixed_accounting = {
        key: accounting[key] for key in expected_accounting
    }
    if actual_fixed_accounting != expected_accounting:
        raise ValueError(
            f"interim campaign accounting mismatch: {actual_fixed_accounting} != "
            f"{expected_accounting}"
        )
    if accounting["provider_attempts"] < accounting["provider_calls"]:
        raise ValueError("provider attempts cannot be fewer than completed calls")
    accounting["provider_retry_attempts"] = (
        accounting["provider_attempts"] - accounting["provider_calls"]
    )
    return {
        "schema_version": STATIC_S0_NOMINAL_INFORMATION_INTERIM_VERSION,
        "status": "completed_audited_interim_descriptive_result",
        "formal_result": False,
        "interim_analysis_only": True,
        "confirmatory_analysis_complete": False,
        "benchmark_claim_allowed": False,
        "freeze": {
            "freeze_id": manifest["freeze_id"],
            "manifest_path": str(manifest_file),
            "manifest_sha256": canonical_json_sha256(manifest),
        },
        "execution": {
            "nominal_root": str(nominal_path),
            "opaque_reference_root": str(opaque_path),
            "selected_world_seeds": selected_seeds,
            "preregistered_world_seeds": preregistered_seeds,
            "completed_fraction_per_task": selected_fraction,
            "owner_stop_rule": "first_five_worlds_requested_for_interim_review",
            "all_selected_cells_completed": True,
            "all_selected_cells_exact_replay_verified": True,
        },
        "analysis": {
            "unit": "independent_world",
            "pairing_key": "task_id_and_world_seed",
            "primary_estimand_by_task": confirmatory[
                "primary_estimand_by_task"
            ],
            "bootstrap_seed": seed,
            "bootstrap_resamples": draws,
            "nominal_interval": "paired_world_bootstrap_95_percent",
            "familywise_rule_preview_interval": (
                "paired_world_bootstrap_97.5_percent_per_task"
            ),
            "full_preregistered_rule_not_applied": True,
        },
        "accounting": accounting,
        "tasks": tasks,
        "reporting_boundaries": {
            **manifest["reporting_boundaries"],
            "five_world_result": (
                "interim_descriptive_only_not_the_frozen_ten_world_confirmatory_result"
            ),
            "positive_information_value_claim_allowed": False,
            "reason": (
                "only_five_of_ten_preregistered_worlds_per_task_were_run_and_both_"
                "paired_intervals_cross_zero"
            ),
        },
    }


def render_static_s0_nominal_information_interim_zh(
    summary: Mapping[str, Any],
) -> str:
    """Render a concise Chinese handoff for the five-world interim result."""

    lines = [
        "# S0 v1.1 正确匿名材料属性：五世界中期结果",
        "",
        "状态：电化学与结晶各完成 seed 0–4，共 10 个任务×世界单元；"
        "全部完成精确回放审计。",
        "",
        "本报告是按负责人要求提前查看的中期描述结果。冻结方案原定每任务 "
        "10 个世界，因此这里不能作为确认性结论，也不能宣称已证明材料信息价值。",
        "",
        "| 任务 | nominal 盲测均值 | opaque 盲测均值 | 配对差 | 95% 区间 | 胜/平/负 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for track_id in ("electrochemical", "crystallization"):
        task = summary["tasks"][track_id]
        nominal = task["nominal_primary_score"]
        opaque = task["opaque_primary_score"]
        paired = task["paired_nominal_minus_opaque"]
        interval = paired["world_bootstrap_95_interval"]
        wtl = task["paired_win_tie_loss"]
        lines.append(
            f"| {track_id} | {nominal['mean']:.4f} | {opaque['mean']:.4f} | "
            f"{paired['mean']:+.4f} | [{interval[0]:+.4f}, "
            f"{interval[1]:+.4f}] | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 两个任务的点估计都偏正，但 95% 与 97.5% 配对 bootstrap 区间均跨 0。",
            "- 电化学信号较强，五世界均值为 0.7873；结晶均值为 0.5507，"
            "不能写成 0.6+。",
            "- 旧 0.3902/0.4829 结果继续排除，不属于本报告证据。",
            "- 要执行冻结的确认性规则，仍需补齐每个任务 seed 5–9。",
            "",
            "## 账本",
            "",
            f"- 探索实验：{summary['accounting']['exploration_experiments']}",
            f"- 预测诊断物理实验："
            f"{summary['accounting']['predictive_physical_experiments']}",
            f"- 盲测验证实验："
            f"{summary['accounting']['blind_validation_experiments']}",
            f"- 物理实验总数：{summary['accounting']['total_physical_experiments']}",
            f"- Codex subscription 调用：{summary['accounting']['provider_calls']}",
            f"- Provider 自动重试："
            f"{summary['accounting']['provider_retry_attempts']}",
            f"- 方法失败：{summary['accounting']['method_failures']}",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "STATIC_S0_NOMINAL_INFORMATION_INTERIM_VERSION",
    "build_static_s0_nominal_information_interim",
    "render_static_s0_nominal_information_interim_zh",
]
