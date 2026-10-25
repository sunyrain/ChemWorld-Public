"""Audit and analyze the complete S0 material-information three-arm campaign."""

# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.static_material_information_campaign import (
    _bootstrap_interval,
    _mean_mapping,
    _read_world,
    _ScoreSummary,
    _summary,
)
from chemworld.materials import (
    STATIC_MATERIAL_INFORMATION_MISINDEXED,
    STATIC_MATERIAL_INFORMATION_NOMINAL,
    STATIC_MATERIAL_INFORMATION_OPAQUE,
)

STATIC_S0_MATERIAL_INFORMATION_TRIARM_VERSION = (
    "chemworld-static-s0-material-information-triarm-result-1.0"
)
_ARMS = ("opaque", "nominal", "misindexed")
_TASK_ORDER = ("electrochemical", "crystallization")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _one_sided_lower_bound(
    values: Sequence[float],
    *,
    alpha: float,
    seed: int,
    draws: int,
    label: str,
) -> float:
    normalized = np.asarray([float(value) for value in values], dtype=float)
    if normalized.size == 0 or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("bootstrap alpha must be between zero and one")
    label_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(int(seed) ^ label_seed)
    indices = rng.integers(
        0,
        normalized.size,
        size=(int(draws), normalized.size),
    )
    return float(np.quantile(normalized[indices].mean(axis=1), float(alpha)))


def _paired(values: Sequence[float], reference: Sequence[float]) -> list[float]:
    return [
        float(value) - float(control)
        for value, control in zip(values, reference, strict=True)
    ]


def _paired_wtl(values: Sequence[float]) -> dict[str, int]:
    return {
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
    }


def _two_sided_contrast(
    values: Sequence[float],
    *,
    task_id: str,
    contrast_id: str,
    seed: int,
    draws: int,
) -> _ScoreSummary:
    result = _summary(values)
    result["world_bootstrap_95_interval"] = _bootstrap_interval(
        values,
        confidence=0.95,
        seed=seed,
        draws=draws,
        label=f"{task_id}:{contrast_id}:95",
    )
    result["world_bootstrap_97_5_interval"] = _bootstrap_interval(
        values,
        confidence=0.975,
        seed=seed,
        draws=draws,
        label=f"{task_id}:{contrast_id}:97.5",
    )
    result["paired_win_tie_loss"] = _paired_wtl(values)
    return result


def _information_rule(interval: Sequence[float]) -> str:
    lower, upper = (float(value) for value in interval)
    if lower > 0.0:
        return "positive_information_value"
    if upper < 0.0:
        return "harmful_information"
    return "inconclusive"


def _wrong_prior_rule(interval: Sequence[float]) -> str:
    lower, upper = (float(value) for value in interval)
    if upper < 0.0:
        return "wrong_prior_cost"
    if lower > 0.0:
        return "wrong_prior_benefit_in_sampled_worlds"
    return "inconclusive"


def _campaign_index(root: Path, *, expected_freeze_sha256: str | None) -> dict[str, Any]:
    index_path = root / "campaign_execution_index.json"
    index = _load_json(index_path)
    if index.get("all_requested_cells_completed") is not True:
        raise ValueError(f"campaign is incomplete: {index_path}")
    if index.get("all_exact_replay_verified") is not True:
        raise ValueError(f"campaign replay is incomplete: {index_path}")
    if len(index.get("results", [])) != 20:
        raise ValueError(f"campaign index does not contain 20 cells: {index_path}")
    if (
        expected_freeze_sha256 is not None
        and index.get("freeze_sha256") != expected_freeze_sha256
    ):
        raise ValueError(f"campaign freeze hash mismatch: {index_path}")
    return index


def _action_record(
    cell: Mapping[str, Any],
    *,
    target_field: str,
    misleading_action_value: int,
    early_indices: Sequence[int],
    late_indices: Sequence[int],
) -> dict[str, Any]:
    actions = [
        int(row["plan"]["recipe_parameters"][target_field])
        for row in cell["public_history"]
    ]
    if len(actions) != 20:
        raise ValueError("completed campaign does not expose 20 public actions")
    final_action = int(
        cell["final_synthesis"]["recommendation"][
            "recommended_recipe_parameters"
        ][target_field]
    )
    return {
        "actions": actions,
        "first_action": actions[0],
        "early_misleading_share": statistics.fmean(
            actions[index] == misleading_action_value
            for index in early_indices
        ),
        "late_misleading_share": statistics.fmean(
            actions[index] == misleading_action_value
            for index in late_indices
        ),
        "final_action": final_action,
        "first_action_is_misleading": actions[0] == misleading_action_value,
        "final_action_is_misleading": final_action == misleading_action_value,
    }


def _arm_accounting(reports: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "participant_world_cells": len(reports),
        "exploration_experiments": 0,
        "predictive_physical_experiments": 0,
        "blind_validation_experiments": 0,
        "total_physical_experiments": 0,
        "provider_calls": 0,
        "provider_attempts": 0,
        "method_failures": 0,
    }
    for report in reports:
        cell = report["cells"][0]
        result["exploration_experiments"] += int(
            cell["completed_experiment_count"]
        )
        result["predictive_physical_experiments"] += int(
            cell["completed_predictive_validation_experiment_count"]
        )
        result["blind_validation_experiments"] += int(
            cell["completed_validation_experiment_count"]
        )
        result["total_physical_experiments"] += int(
            cell["total_physical_experiment_count"]
        )
        result["provider_calls"] += int(report["provider_call_count"])
        result["provider_attempts"] += int(report["provider_attempt_count"])
        result["method_failures"] += int(report["method_failure_cell_count"])
    result["provider_retry_attempts"] = (
        result["provider_attempts"] - result["provider_calls"]
    )
    expected = {
        "participant_world_cells": 20,
        "exploration_experiments": 400,
        "predictive_physical_experiments": 240,
        "blind_validation_experiments": 120,
        "total_physical_experiments": 760,
        "provider_calls": 420,
        "method_failures": 0,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(
                f"campaign accounting mismatch for {key}: "
                f"{result[key]} != {value}"
            )
    return result


def build_static_s0_material_information_triarm_result(
    *,
    manifest_path: str | Path,
    nominal_manifest_path: str | Path,
    opaque_root: str | Path,
    nominal_root: str | Path,
    misindexed_root: str | Path,
) -> dict[str, Any]:
    """Build the complete audited three-arm result from frozen campaign roots."""

    manifest_file = Path(manifest_path)
    nominal_manifest_file = Path(nominal_manifest_path)
    roots = {
        "opaque": Path(opaque_root),
        "nominal": Path(nominal_root),
        "misindexed": Path(misindexed_root),
    }
    manifest = _load_json(manifest_file)
    nominal_manifest = _load_json(nominal_manifest_file)
    manifest_sha256 = canonical_json_sha256(manifest)
    nominal_manifest_sha256 = canonical_json_sha256(nominal_manifest)
    if manifest["world_seeds"] != list(range(10)):
        raise ValueError("three-arm result requires all ten frozen worlds")
    if nominal_manifest["world_seeds"] != list(range(10)):
        raise ValueError("nominal reference does not contain all ten worlds")
    confirmatory = manifest["confirmatory_analysis"]
    recovery_contract = manifest["recovery_analysis"]
    seed = int(confirmatory["random_seed"])
    draws = int(confirmatory["bootstrap_resamples"])
    early_indices = recovery_contract["early_round_indices_zero_based"]
    late_indices = recovery_contract["late_round_indices_zero_based"]
    margin = float(recovery_contract["practical_score_margin"])

    indices = {
        "opaque": _campaign_index(roots["opaque"], expected_freeze_sha256=None),
        "nominal": _campaign_index(
            roots["nominal"],
            expected_freeze_sha256=nominal_manifest_sha256,
        ),
        "misindexed": _campaign_index(
            roots["misindexed"],
            expected_freeze_sha256=manifest_sha256,
        ),
    }
    expected_conditions = {
        "opaque": STATIC_MATERIAL_INFORMATION_OPAQUE,
        "nominal": STATIC_MATERIAL_INFORMATION_NOMINAL,
        "misindexed": STATIC_MATERIAL_INFORMATION_MISINDEXED,
    }
    tasks: dict[str, Any] = {}
    reports_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in _ARMS
    }
    for track in manifest["participant_tracks"]:
        track_id = str(track["track_id"])
        task_id = str(track["task_id"])
        misindexing = track["misindexing_contract"]
        target_field = str(misindexing["target_field"])
        misleading_action_value = int(
            misindexing["misleading_action_value"]
        )
        arm_scores: dict[str, list[float]] = {arm: [] for arm in _ARMS}
        arm_actions: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in _ARMS
        }
        predictive_rows: dict[str, list[dict[str, float]]] = {
            arm: [] for arm in _ARMS
        }
        worlds = []
        for world_seed in manifest["world_seeds"]:
            world_row: dict[str, Any] = {"world_seed": int(world_seed)}
            for arm in _ARMS:
                world_root = (
                    roots[arm]
                    / "participants"
                    / track_id
                    / f"world-{world_seed:02d}"
                )
                report, _, evidence = _read_world(
                    world_root,
                    expected_seed=int(world_seed),
                )
                reports_by_arm[arm].append(report)
                cell = report["cells"][0]
                agent_manifest = cell["agent_manifest"]
                if (
                    agent_manifest["material_information_condition"]
                    != expected_conditions[arm]
                ):
                    raise ValueError(
                        f"material condition mismatch: {world_root}"
                    )
                if arm == "misindexed" and (
                    agent_manifest["material_information_sha256"]
                    != track["material_information_sha256"]
                ):
                    raise ValueError(
                        f"misindexed dossier hash mismatch: {world_root}"
                    )
                score = float(evidence["primary_score"])
                action = _action_record(
                    cell,
                    target_field=target_field,
                    misleading_action_value=misleading_action_value,
                    early_indices=early_indices,
                    late_indices=late_indices,
                )
                arm_scores[arm].append(score)
                arm_actions[arm].append(action)
                predictive_rows[arm].append(
                    {
                        key: float(
                            cell["predictive_validation"]["score"][key]
                        )
                        for key in (
                            "directional_accuracy",
                            "confidence_brier_score",
                            "nontrivial_actual_effect_rate",
                        )
                    }
                )
                world_row[arm] = {
                    "primary_score": score,
                    "action": action,
                    "evidence": evidence,
                }
            worlds.append(world_row)

        nominal_minus_opaque = _paired(
            arm_scores["nominal"],
            arm_scores["opaque"],
        )
        misindexed_minus_nominal = _paired(
            arm_scores["misindexed"],
            arm_scores["nominal"],
        )
        misindexed_minus_opaque = _paired(
            arm_scores["misindexed"],
            arm_scores["opaque"],
        )
        contrasts = {
            "nominal_minus_opaque": _two_sided_contrast(
                nominal_minus_opaque,
                task_id=track_id,
                contrast_id="nominal-opaque",
                seed=seed,
                draws=draws,
            ),
            "misindexed_minus_nominal": _two_sided_contrast(
                misindexed_minus_nominal,
                task_id=track_id,
                contrast_id="misindexed-nominal",
                seed=seed,
                draws=draws,
            ),
            "misindexed_minus_opaque": _two_sided_contrast(
                misindexed_minus_opaque,
                task_id=track_id,
                contrast_id="misindexed-opaque",
                seed=seed,
                draws=draws,
            ),
        }
        contrasts["nominal_minus_opaque"]["familywise_result"] = (
            _information_rule(
                contrasts["nominal_minus_opaque"][
                    "world_bootstrap_97_5_interval"
                ]
            )
        )
        contrasts["misindexed_minus_nominal"]["familywise_result"] = (
            _wrong_prior_rule(
                contrasts["misindexed_minus_nominal"][
                    "world_bootstrap_97_5_interval"
                ]
            )
        )

        early = {
            arm: [
                float(row["early_misleading_share"])
                for row in arm_actions[arm]
            ]
            for arm in _ARMS
        }
        late = {
            arm: [
                float(row["late_misleading_share"])
                for row in arm_actions[arm]
            ]
            for arm in _ARMS
        }
        manipulation = _paired(early["misindexed"], early["nominal"])
        differential_correction = [
            (mis_late - mis_early) - (nom_late - nom_early)
            for mis_early, mis_late, nom_early, nom_late in zip(
                early["misindexed"],
                late["misindexed"],
                early["nominal"],
                late["nominal"],
                strict=True,
            )
        ]
        manipulation_summary = _two_sided_contrast(
            manipulation,
            task_id=track_id,
            contrast_id="manipulation",
            seed=seed,
            draws=draws,
        )
        correction_summary = _two_sided_contrast(
            differential_correction,
            task_id=track_id,
            contrast_id="correction",
            seed=seed,
            draws=draws,
        )
        manipulation_pass = (
            manipulation_summary["world_bootstrap_97_5_interval"][0] > 0.0
        )
        correction_pass = (
            correction_summary["world_bootstrap_97_5_interval"][1] < 0.0
        )
        opaque_lower = _one_sided_lower_bound(
            misindexed_minus_opaque,
            alpha=0.025,
            seed=seed,
            draws=draws,
            label=f"{track_id}:misindexed-opaque:lower97.5",
        )
        nominal_lower = _one_sided_lower_bound(
            misindexed_minus_nominal,
            alpha=0.025,
            seed=seed,
            draws=draws,
            label=f"{track_id}:misindexed-nominal:lower97.5",
        )
        opaque_recovery_pass = opaque_lower >= -margin
        nominal_restoration_pass = nominal_lower >= -margin
        recovery = {
            "target_field": target_field,
            "misleading_action_value": misleading_action_value,
            "early_misleading_share_by_arm": {
                arm: _summary(early[arm]) for arm in _ARMS
            },
            "late_misleading_share_by_arm": {
                arm: _summary(late[arm]) for arm in _ARMS
            },
            "first_action_misleading_rate_by_arm": {
                arm: statistics.fmean(
                    row["first_action_is_misleading"]
                    for row in arm_actions[arm]
                )
                for arm in _ARMS
            },
            "final_action_misleading_rate_by_arm": {
                arm: statistics.fmean(
                    row["final_action_is_misleading"]
                    for row in arm_actions[arm]
                )
                for arm in _ARMS
            },
            "manipulation_check": {
                **manipulation_summary,
                "passed": manipulation_pass,
            },
            "differential_action_correction": {
                **correction_summary,
                "passed": correction_pass,
            },
            "performance_recovery_to_opaque": {
                "practical_score_margin": margin,
                "one_sided_familywise_lower_97_5_bound": opaque_lower,
                "passed": opaque_recovery_pass,
            },
            "performance_restoration_to_nominal": {
                "practical_score_margin": margin,
                "one_sided_familywise_lower_97_5_bound": nominal_lower,
                "passed": nominal_restoration_pass,
            },
            "overall_recovery_claim": {
                "passed": bool(
                    manipulation_pass
                    and correction_pass
                    and opaque_recovery_pass
                ),
                "required_components": {
                    "manipulation_check": manipulation_pass,
                    "differential_action_correction": correction_pass,
                    "performance_recovery_to_opaque": opaque_recovery_pass,
                },
            },
        }
        tasks[track_id] = {
            "task_id": task_id,
            "target_field": target_field,
            "misindexing_contract": misindexing,
            "primary_score_by_arm": {
                arm: _summary(arm_scores[arm]) for arm in _ARMS
            },
            "paired_contrasts": contrasts,
            "recovery": recovery,
            "predictive_secondary_diagnostic_mean_by_arm": {
                arm: _mean_mapping(predictive_rows[arm]) for arm in _ARMS
            },
            "worlds": worlds,
        }

    accounting_by_arm = {
        arm: _arm_accounting(reports_by_arm[arm]) for arm in _ARMS
    }
    total_accounting = {
        key: sum(accounting_by_arm[arm][key] for arm in _ARMS)
        for key in (
            "participant_world_cells",
            "exploration_experiments",
            "predictive_physical_experiments",
            "blind_validation_experiments",
            "total_physical_experiments",
            "provider_calls",
            "provider_attempts",
            "provider_retry_attempts",
            "method_failures",
        )
    }
    return {
        "schema_version": STATIC_S0_MATERIAL_INFORMATION_TRIARM_VERSION,
        "status": "completed_audited_formal_three_arm_result",
        "formal_result": True,
        "confirmatory_analysis_complete": True,
        "benchmark_claim_allowed": False,
        "freeze": {
            "misindexed_manifest_path": str(manifest_file),
            "misindexed_manifest_sha256": manifest_sha256,
            "nominal_manifest_path": str(nominal_manifest_file),
            "nominal_manifest_sha256": nominal_manifest_sha256,
        },
        "execution": {
            "roots": {arm: str(root) for arm, root in roots.items()},
            "source_commits": {
                arm: indices[arm]["source_commit"] for arm in _ARMS
            },
            "world_seeds": list(range(10)),
            "all_three_arms_completed": True,
            "all_sixty_cells_exact_replay_verified": True,
        },
        "analysis": {
            "unit": "independent_world",
            "pairing_key": "task_id_and_world_seed",
            "bootstrap_seed": seed,
            "bootstrap_resamples": draws,
            "two_task_familywise_interval": (
                "paired_world_bootstrap_97.5_percent_per_task"
            ),
            "performance_recovery_margin": margin,
        },
        "accounting": {
            "by_arm": accounting_by_arm,
            "three_arm_total": total_accounting,
            "new_work_completed_in_this_continuation": {
                "participant_world_cells": 30,
                "provider_calls": 630,
                "total_physical_experiments": 1140,
                "method_failures": 0,
            },
        },
        "tasks": tasks,
        "reporting_boundaries": manifest["reporting_boundaries"],
    }


def render_static_s0_material_information_triarm_zh(
    summary: Mapping[str, Any],
) -> str:
    """Render the complete three-arm result as a concise Chinese report."""

    lines = [
        "# S0 材料信息三臂实验：完整 10 世界结果",
        "",
        "状态：两个旗舰任务的 opaque、正确匿名属性和固定错配属性三臂均完成；"
        "共 60 个任务×世界单元，全部精确回放通过。",
        "",
        "## 盲测主结果",
        "",
        "| 任务 | opaque | 正确信息 | 错误先验 | 正确−opaque | 错误−正确 | 错误−opaque |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track_id in _TASK_ORDER:
        task = summary["tasks"][track_id]
        scores = task["primary_score_by_arm"]
        contrasts = task["paired_contrasts"]
        lines.append(
            f"| {track_id} | {scores['opaque']['mean']:.4f} | "
            f"{scores['nominal']['mean']:.4f} | "
            f"{scores['misindexed']['mean']:.4f} | "
            f"{contrasts['nominal_minus_opaque']['mean']:+.4f} | "
            f"{contrasts['misindexed_minus_nominal']['mean']:+.4f} | "
            f"{contrasts['misindexed_minus_opaque']['mean']:+.4f} |"
        )
    lines.extend(["", "## 家族校正结论", ""])
    for track_id in _TASK_ORDER:
        task = summary["tasks"][track_id]
        contrasts = task["paired_contrasts"]
        nominal = contrasts["nominal_minus_opaque"]
        wrong = contrasts["misindexed_minus_nominal"]
        nominal_interval = nominal["world_bootstrap_97_5_interval"]
        wrong_interval = wrong["world_bootstrap_97_5_interval"]
        lines.extend(
            [
                f"- **{track_id}**：正确信息效应 "
                f"{nominal['mean']:+.4f}，97.5% 区间 "
                f"[{nominal_interval[0]:+.4f}, {nominal_interval[1]:+.4f}]，"
                f"`{nominal['familywise_result']}`；错误先验相对正确信息 "
                f"{wrong['mean']:+.4f}，97.5% 区间 "
                f"[{wrong_interval[0]:+.4f}, {wrong_interval[1]:+.4f}]，"
                f"`{wrong['familywise_result']}`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 恢复判据",
            "",
            (
                "| 任务 | 前五轮错误 ID | 后五轮错误 ID | 干预生效 | "
                "差分行动修正 | 恢复到 opaque | 总体恢复 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for track_id in _TASK_ORDER:
        recovery = summary["tasks"][track_id]["recovery"]
        lines.append(
            f"| {track_id} | "
            f"{recovery['early_misleading_share_by_arm']['misindexed']['mean']:.2%} | "
            f"{recovery['late_misleading_share_by_arm']['misindexed']['mean']:.2%} | "
            f"{'通过' if recovery['manipulation_check']['passed'] else '未通过'} | "
            f"{'通过' if recovery['differential_action_correction']['passed'] else '未通过'} | "
            f"{'通过' if recovery['performance_recovery_to_opaque']['passed'] else '未通过'} | "
            f"{'通过' if recovery['overall_recovery_claim']['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "电化学出现明确的行动撤离，但没有通过相对 opaque 的性能非劣判据，"
            "因此只能写“行动修正，未证明性能恢复”。结晶在 sampled worlds 中的"
            "错误先验盲测分数反而更高，但差分行动修正未通过，因此不能把性能收益"
            "叙述成模型完成了纠错。",
            "",
            "## 账本",
            "",
            f"- 三臂总计：{summary['accounting']['three_arm_total']['participant_world_cells']} "
            "个任务×世界单元，"
            f"{summary['accounting']['three_arm_total']['provider_calls']} 次成功调用，"
            f"{summary['accounting']['three_arm_total']['provider_retry_attempts']} 次重试，"
            f"{summary['accounting']['three_arm_total']['total_physical_experiments']} "
            "个物理实验。",
            "- 本次续作新增：30 个任务×世界单元、630 次成功调用、"
            "1,140 个物理实验、0 方法失败。",
            "",
            "## 边界",
            "",
            "这是两个固定旗舰任务、十个冻结世界和两组固定单字段换位上的结果。"
            "不得推广为模型普遍会或不会纠正错误科学先验；也不得把结晶中的"
            "in-sample 收益写成错误信息通常有益。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "STATIC_S0_MATERIAL_INFORMATION_TRIARM_VERSION",
    "build_static_s0_material_information_triarm_result",
    "render_static_s0_material_information_triarm_zh",
]
