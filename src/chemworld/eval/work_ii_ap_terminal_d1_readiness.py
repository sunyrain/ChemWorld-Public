"""Outcome-blind, provider-free readiness for independent Work II A-P D1 blocks."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AP_D1_READINESS_VERSION = "chemworld-work-ii-ap-independent-d1-readiness-0.1"
AP_D1_PLAN_VERSION = "chemworld-work-ii-ap-independent-d1-plan-0.1"
AP_D1_ARMS = {"opaque", "aligned_nominal", "misindexed_nominal"}
AP_D1_CHECKPOINTS = [0, 2, 4, 7, 10]
AP_D1_EXPERIMENTS = 10
AP_D1_SELECTION_BASIS = "smallest_q2_passed_world_seed_without_participant_exposure"
_PARTICIPANT_SESSION_FIELD = b'"participant_provider_session_count"'


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _contains_bytes(path: Path, needle: bytes, *, chunk_size: int = 1024 * 1024) -> bool:
    """Search a file with bounded memory before deciding whether JSON parsing is needed."""

    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            window = overlap + chunk
            if needle in window:
                return True
            overlap = window[-(len(needle) - 1) :]
    return False


def _inside(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = (root.resolve() / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path escapes the repository")
    return path


def discover_historical_ap_participant_exposure(
    root: Path, task_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Discover every readable tracked A-P evaluation that records provider sessions."""

    root = root.resolve()
    requested = set(task_ids)
    result = {task_id: [] for task_id in requested}
    report_root = root / "workstreams/flagship_tasks/reports"
    for path in sorted(report_root.rglob("*.json")):
        try:
            if not _contains_bytes(path, _PARTICIPANT_SESSION_FIELD):
                continue
            report = _load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        task_id = report.get("task_id")
        denominators = report.get("denominators")
        denominators = denominators if isinstance(denominators, Mapping) else {}
        provider_sessions = denominators.get("participant_provider_session_count")
        if (
            task_id not in requested
            or isinstance(report.get("world_seed"), bool)
            or not isinstance(report.get("world_seed"), int)
            or isinstance(provider_sessions, bool)
            or not isinstance(provider_sessions, int)
            or provider_sessions <= 0
        ):
            continue
        result[str(task_id)].append(
            {
                "path": path.relative_to(root).as_posix(),
                "task_id": task_id,
                "world_seed": report.get("world_seed"),
                "participant_provider_session_count": provider_sessions,
                "status": report.get("status"),
            }
        )
    return result


def _q2_world(
    root: Path,
    candidate: Mapping[str, Any],
    exposed_seeds: Sequence[int],
) -> tuple[dict[str, Any] | None, list[int], list[int], list[str]]:
    errors: list[str] = []
    try:
        package_path = _inside(root, candidate.get("q2_package"), label="Q2 package")
        package = _load(package_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, [], [], [f"Q2 package cannot be read: {error}"]
    task_id = candidate.get("task_id")
    if (
        package.get("task_id") != task_id
        or package.get("formal_result") is not False
        or package.get("qualification_passed") is not True
    ):
        errors.append("Q2 package is not a valid provider-free five-world pass")
    try:
        summary_path = _inside(root, candidate.get("q2_summary"), label="Q2 summary")
        summary = _load(summary_path)
        generated_package = summary.get("generated_package")
        denominators = summary.get("denominators")
        denominators = denominators if isinstance(denominators, Mapping) else {}
        if (
            summary.get("task_id") != task_id
            or summary.get("formal_result") is not False
            or summary.get("qualification_passed") is not True
            or summary.get("provider_call_count") != 0
            or summary.get("world_seeds") != [0, 1, 2, 3, 4]
            or denominators.get("world_count") != 5
            or denominators.get("passed_world_count") != 5
            or not isinstance(generated_package, Mapping)
            or generated_package.get("path") != package_path.relative_to(root.resolve()).as_posix()
        ):
            errors.append("Q2 summary is not a valid provider-free binding of the package")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Q2 summary cannot be read: {error}")
    worlds = package.get("worlds")
    worlds = worlds if isinstance(worlds, list) else []
    world_by_seed: dict[int, dict[str, Any]] = {}
    for world in worlds:
        if (
            not isinstance(world, dict)
            or isinstance(world.get("world_seed"), bool)
            or not isinstance(world.get("world_seed"), int)
        ):
            errors.append("Q2 package has a malformed world row")
            continue
        seed = int(world["world_seed"])
        if seed in world_by_seed:
            errors.append("Q2 package has duplicate world seeds")
            continue
        world_by_seed[seed] = world
        if (
            world.get("task_id") != task_id
            or not isinstance(world.get("qualification_passed"), bool)
            or set(world.get("prior_arms", {})) != AP_D1_ARMS
            or not isinstance(world.get("held_out_queries"), list)
            or not isinstance(world.get("reference_context"), Mapping)
            or not isinstance(world.get("world_package_sha256"), str)
        ):
            errors.append(f"Q2 world {seed} has an invalid qualification contract")
    if sorted(world_by_seed) != [0, 1, 2, 3, 4]:
        errors.append("Q2 package does not contain exactly the five frozen worlds")
    q2_passed_seeds = sorted(
        seed for seed, world in world_by_seed.items() if world.get("qualification_passed") is True
    )
    eligible_seeds = sorted(set(q2_passed_seeds) - set(exposed_seeds))
    if not eligible_seeds:
        errors.append("no Q2-passed world remains after historical participant exposure exclusion")
        return None, q2_passed_seeds, eligible_seeds, errors
    selected_seed = eligible_seeds[0]
    return world_by_seed[selected_seed], q2_passed_seeds, eligible_seeds, errors


def _config_errors(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    world: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    independent = config.get("independent_terminal_d1")
    independent = independent if isinstance(independent, Mapping) else {}
    if config.get("task_id") != candidate.get("task_id"):
        errors.append("independent D1 config task mismatch")
    selected_seed = int(world["world_seed"])
    expected_pilot_id = f"{candidate.get('pilot_id_prefix')}-seed{selected_seed}"
    expected_noise_namespace = f"{candidate.get('noise_namespace_prefix')}-seed{selected_seed}"
    if config.get("world_seed") != selected_seed:
        errors.append("independent D1 config seed mismatch")
    if config.get("pilot_id") != expected_pilot_id:
        errors.append("independent D1 config pilot_id mismatch")
    if config.get("observation_noise_namespace") != expected_noise_namespace:
        errors.append("independent D1 config noise namespace mismatch")
    if config.get("formal_result") is not False:
        errors.append("independent D1 config crossed the formal boundary")
    if set(config.get("prior_arms", {})) != AP_D1_ARMS:
        errors.append("independent D1 config does not contain the frozen three arms")
    if config.get("prior_arms") != world.get("prior_arms"):
        errors.append("independent D1 config priors differ from the selected Q2 world")
    if (
        campaign.get("complete_experiments") != AP_D1_EXPERIMENTS
        or campaign.get("checkpoint_complete_experiments") != AP_D1_CHECKPOINTS
        or config.get("snapshot_stages")
        != [
            "pre_evidence",
            "after_experiment_2",
            "after_experiment_4",
            "after_experiment_7",
            "final",
        ]
    ):
        errors.append("independent D1 rounds/checkpoints differ from the frozen A-P contract")
    if (
        qualification.get("q2_passed") is not True
        or qualification.get("execution_authorized") is not False
        or qualification.get("formal_r5_authorized") is not False
    ):
        errors.append("independent D1 config does not remain provider/R5 blocked")
    if (
        independent.get("historical_participant_results_replaced") is not False
        or independent.get("scientific_outcome_is_admission_rule") is not False
        or independent.get("selection_basis") != AP_D1_SELECTION_BASIS
    ):
        errors.append("independent D1 nonreplacement/outcome-blind contract is missing")
    return errors


def _build_config(
    root: Path,
    candidate: Mapping[str, Any],
    world: Mapping[str, Any],
    exposed_seeds: Sequence[int],
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        template_path = _inside(root, candidate.get("template_config"), label="D1 template")
        config = copy.deepcopy(_load(template_path))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, [f"D1 template cannot be read: {error}"]
    selected_seed = int(world["world_seed"])
    config["pilot_id"] = f"{candidate.get('pilot_id_prefix')}-seed{selected_seed}"
    config["observation_noise_namespace"] = (
        f"{candidate.get('noise_namespace_prefix')}-seed{selected_seed}"
    )
    config["world_seed"] = selected_seed
    config["prior_arms"] = copy.deepcopy(world["prior_arms"])
    config["belief_checkpoint"]["held_out_queries"] = copy.deepcopy(world["held_out_queries"])
    config["intervention"]["fixed_reference_context"] = copy.deepcopy(world["reference_context"])
    config["intervention"]["q2_binding_sha256"] = world["world_package_sha256"]
    config["qualification"].update(
        {
            "q2_passed": True,
            "execution_authorized": False,
            "formal_r5_authorized": False,
        }
    )
    config["independent_terminal_d1"] = {
        "readiness_only": True,
        "selection_basis": AP_D1_SELECTION_BASIS,
        "historical_participant_exposed_world_seeds": sorted(set(exposed_seeds)),
        "historical_participant_results_replaced": False,
        "scientific_outcome_is_admission_rule": False,
        "provider_execution_authorized": False,
    }
    errors = _config_errors(config, candidate, world)
    return (None, errors) if errors else (config, [])


def build_independent_ap_d1_readiness(
    root: Path, plan_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build deterministic readiness and any eligible static configs without provider calls."""

    root = root.resolve()
    plan = _load(plan_path.resolve())
    candidates = plan.get("candidates")
    if plan.get("schema_version") != AP_D1_PLAN_VERSION or not isinstance(candidates, list):
        raise ValueError("independent A-P D1 plan is malformed")
    if plan.get("selection_basis") != AP_D1_SELECTION_BASIS:
        raise ValueError("independent A-P D1 plan has an unsupported selection rule")
    if {row.get("task_id") for row in candidates if isinstance(row, Mapping)} != {
        "reaction-safety-constrained",
        "electrochemical-conversion",
    }:
        raise ValueError("independent A-P D1 plan must contain exactly the two frozen tasks")
    task_ids = [str(row["task_id"]) for row in candidates]
    discovered = discover_historical_ap_participant_exposure(root, task_ids)
    rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        task_id = str(candidate["task_id"])
        exposed_seeds = sorted(
            {
                int(item["world_seed"])
                for item in discovered[task_id]
                if isinstance(item.get("world_seed"), int)
            }
        )
        world, q2_passed_seeds, eligible_seeds, q2_errors = _q2_world(
            root, candidate, exposed_seeds
        )
        blockers = list(q2_errors)
        selected_seed = int(world["world_seed"]) if world is not None else None
        config: dict[str, Any] | None = None
        if not blockers and world is not None:
            config, config_errors = _build_config(root, candidate, world, exposed_seeds)
            blockers.extend(config_errors)
        if config is not None and not blockers:
            configs[task_id] = config
        rows.append(
            {
                "task_id": task_id,
                "q2_passed_world_seeds": q2_passed_seeds,
                "historical_participant_exposure": discovered[task_id],
                "historical_participant_exposed_world_seeds": exposed_seeds,
                "eligible_unexposed_q2_passed_world_seeds": eligible_seeds,
                "selected_world_seed": selected_seed,
                "selection_rule_satisfied": (
                    selected_seed is not None
                    and eligible_seeds
                    and selected_seed == min(eligible_seeds)
                ),
                "status": (
                    "ready_static_config_provider_execution_blocked"
                    if config is not None and not blockers
                    else "blocked_fail_closed"
                ),
                "output_config": (
                    str(candidate.get("output_config_pattern")).format(world_seed=selected_seed)
                    if selected_seed is not None
                    else None
                ),
                "provider_execution_authorized": False,
                "historical_participant_results_replaced": False,
                "blockers": blockers,
            }
        )
    readiness = {
        "schema_version": AP_D1_READINESS_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "provider_execution_authorized": False,
        "w2_26_prerequisite": False,
        "c2_terminal_admission_followup": True,
        "selection_basis": AP_D1_SELECTION_BASIS,
        "tasks": rows,
        "ready_task_count": len(configs),
        "blocked_task_count": len(rows) - len(configs),
        "status": "ready" if len(configs) == len(rows) else "blocked_fail_closed",
    }
    return readiness, configs


def validate_independent_ap_d1_readiness(
    root: Path,
    plan_path: Path,
    readiness: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    expected_readiness, expected_configs = build_independent_ap_d1_readiness(root, plan_path)
    errors: list[str] = []
    if dict(readiness) != expected_readiness:
        errors.append("independent A-P D1 readiness differs from deterministic rebuild")
    if {key: dict(value) for key, value in configs.items()} != expected_configs:
        errors.append("independent A-P D1 configs differ from deterministic rebuild")
    return errors


__all__ = [
    "AP_D1_READINESS_VERSION",
    "build_independent_ap_d1_readiness",
    "discover_historical_ap_participant_exposure",
    "validate_independent_ap_d1_readiness",
]
