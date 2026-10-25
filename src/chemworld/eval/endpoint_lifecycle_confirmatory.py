"""Prospective audit and REML analysis for G2 endpoint-lifecycle dissociation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2

from chemworld.eval.autonomous_material_campaign_audit import (
    NOMINAL_ARM,
    OPAQUE_ARM,
    _paired_delta,
)
from chemworld.eval.autonomous_material_replication_audit import (
    _PAIR_IDENTITY_FIELDS,
    _completed_cell_audit,
    _pair_physical_audit,
    _right_censored_cell_audit,
    _validate_state_attempt_policy,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)

ENDPOINT_LIFECYCLE_CONFIRMATORY_VERSION = "chemworld-g2-endpoint-lifecycle-confirmatory-audit-0.1"
EXPECTED_MANIFEST_VERSION = "chemworld-g2-endpoint-lifecycle-confirmatory-run-0.2"
EXPECTED_PROTOCOL_VERSION = "chemworld-g2-endpoint-lifecycle-confirmatory-0.6"
EXPECTED_ANALYSIS_PLAN_VERSION = "chemworld-g2-endpoint-lifecycle-analysis-plan-0.1"
CONDITIONS = {
    "anonymous_nominal_properties": NOMINAL_ARM,
    "opaque_codes": OPAQUE_ARM,
}
ROOT = Path(__file__).resolve().parents[3]


class EndpointLifecycleConfirmatoryError(ValueError):
    """Raised when a confirmatory artifact fails closed."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EndpointLifecycleConfirmatoryError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise EndpointLifecycleConfirmatoryError(f"{label} must be a JSON object")
    return payload


def _resolve_repo_path(relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EndpointLifecycleConfirmatoryError(f"missing {label} path")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise EndpointLifecycleConfirmatoryError(f"invalid {label} path: {relative}")
    return path


def _validate_content_hash(payload: Mapping[str, Any], field: str, *, label: str) -> None:
    unhashed = dict(payload)
    declared = unhashed.pop(field, None)
    if declared != canonical_json_sha256(unhashed):
        raise EndpointLifecycleConfirmatoryError(f"{label} content hash mismatch")


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    lifecycle_metric: str,
    endpoint_metric: str | None,
    time_levels: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if not rows:
        raise EndpointLifecycleConfirmatoryError("REML requires completed pairs")
    levels = (
        sorted({int(row["schedule_time_block"]) for row in rows})
        if time_levels is None
        else [int(value) for value in time_levels]
    )
    columns: list[list[float]] = [[1.0] for _ in rows]
    if endpoint_metric is not None:
        for column, row in zip(columns, rows, strict=True):
            column.append(float(row["nominal_minus_opaque"][endpoint_metric]))
    for level in levels[1:]:
        for column, row in zip(columns, rows, strict=True):
            column.append(float(int(row["schedule_time_block"]) == level))
    y = np.asarray(
        [float(row["nominal_minus_opaque"][lifecycle_metric]) for row in rows],
        dtype=float,
    )
    x = np.asarray(columns, dtype=float)
    groups = np.asarray([int(row["world_seed"]) for row in rows], dtype=int)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise EndpointLifecycleConfirmatoryError("REML inputs must be finite")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise EndpointLifecycleConfirmatoryError("REML fixed-effect design is rank deficient")
    return y, x, groups, levels


def _inverse_and_logdet(
    groups: np.ndarray,
    *,
    residual_variance: float,
    world_variance: float,
) -> tuple[np.ndarray, float]:
    n = len(groups)
    inverse = np.zeros((n, n), dtype=float)
    logdet = 0.0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        count = len(indices)
        block = np.eye(count) / residual_variance
        block -= (
            world_variance / (residual_variance * (residual_variance + count * world_variance))
        ) * np.ones((count, count))
        inverse[np.ix_(indices, indices)] = block
        logdet += (count - 1) * math.log(residual_variance)
        logdet += math.log(residual_variance + count * world_variance)
    return inverse, logdet


def _reml_objective(
    log_variances: Sequence[float],
    *,
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    residual_variance, world_variance = np.exp(np.asarray(log_variances, dtype=float))
    inverse, logdet_v = _inverse_and_logdet(
        groups,
        residual_variance=float(residual_variance),
        world_variance=float(world_variance),
    )
    xtvi = x.T @ inverse
    information = xtvi @ x
    sign, logdet_information = np.linalg.slogdet(information)
    if sign <= 0:
        return math.inf, np.full(x.shape[1], math.nan), information
    beta = np.linalg.solve(information, xtvi @ y)
    residual = y - x @ beta
    quad = float(residual.T @ inverse @ residual)
    degrees = len(y) - x.shape[1]
    if degrees <= 0:
        return math.inf, beta, information
    objective = 0.5 * (
        degrees * math.log(2.0 * math.pi) + logdet_v + float(logdet_information) + quad
    )
    return float(objective), beta, information


def fit_random_intercept_reml(
    rows: Sequence[Mapping[str, Any]],
    *,
    lifecycle_metric: str = "terminal_to_global_best_ratio",
    endpoint_metric: str | None = "best_final_score",
    time_levels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit the frozen Gaussian random-intercept model by REML."""

    y, x, groups, levels = _design_matrix(
        rows,
        lifecycle_metric=lifecycle_metric,
        endpoint_metric=endpoint_metric,
        time_levels=time_levels,
    )
    empirical = max(float(np.var(y, ddof=1)), 1e-4)
    starts = (
        (math.log(empirical * 0.75), math.log(empirical * 0.25)),
        (math.log(empirical * 0.5), math.log(empirical * 0.5)),
        (math.log(empirical * 0.95), math.log(max(empirical * 0.05, 1e-6))),
    )
    best: Any = None
    bounds = ((math.log(1e-10), math.log(4.0)),) * 2
    for start in starts:
        candidate = minimize(
            lambda values: _reml_objective(values, y=y, x=x, groups=groups)[0],
            np.asarray(start),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or float(candidate.fun) < float(best.fun):
            best = candidate
    if best is None or not math.isfinite(float(best.fun)):
        raise EndpointLifecycleConfirmatoryError("REML optimization failed")
    objective, beta, information = _reml_objective(
        best.x,
        y=y,
        x=x,
        groups=groups,
    )
    residual_variance, world_variance = np.exp(best.x)
    covariance = np.linalg.inv(information)
    coefficient_names = ["intercept"]
    if endpoint_metric is not None:
        coefficient_names.append(f"delta_{endpoint_metric}")
    coefficient_names.extend(f"time_block_{level}" for level in levels[1:])
    return {
        "converged": bool(best.success),
        "n_pairs": len(rows),
        "world_count": len(np.unique(groups)),
        "fixed_effect_count": x.shape[1],
        "time_levels": levels,
        "lifecycle_metric": lifecycle_metric,
        "endpoint_metric": endpoint_metric,
        "negative_restricted_log_likelihood": float(objective),
        "sigma_unexplained": math.sqrt(float(residual_variance)),
        "tau_world": math.sqrt(float(world_variance)),
        "coefficients": {
            name: {
                "estimate": float(value),
                "standard_error": math.sqrt(max(float(covariance[index, index]), 0.0)),
            }
            for index, (name, value) in enumerate(zip(coefficient_names, beta, strict=True))
        },
        "_fit": {
            "y": y,
            "x": x,
            "groups": groups,
            "log_variances": np.asarray(best.x, dtype=float),
        },
    }


def profile_lower_bound(fit: Mapping[str, Any], *, confidence: float = 0.95) -> float:
    """Return the one-sided profile-likelihood lower bound for residual SD."""

    internals = fit["_fit"]
    y = internals["y"]
    x = internals["x"]
    groups = internals["groups"]
    optimum = float(fit["negative_restricted_log_likelihood"])
    estimate = float(fit["sigma_unexplained"])
    target = optimum + 0.5 * float(chi2.ppf(2.0 * confidence - 1.0, 1))

    def profiled(log_sigma: float) -> float:
        result = minimize_scalar(
            lambda log_tau: _reml_objective(
                (2.0 * log_sigma, float(log_tau)),
                y=y,
                x=x,
                groups=groups,
            )[0],
            bounds=(math.log(1e-10), math.log(4.0)),
            method="bounded",
            options={"xatol": 1e-9},
        )
        return float(result.fun) - target

    lower_log = math.log(1e-5)
    upper_log = math.log(max(estimate, 1e-5))
    if profiled(lower_log) <= 0.0:
        return 0.0
    return float(math.exp(brentq(profiled, lower_log, upper_log, xtol=1e-9)))


def leave_one_world_out_r_squared(
    rows: Sequence[Mapping[str, Any]],
    *,
    lifecycle_metric: str = "terminal_to_global_best_ratio",
    endpoint_metric: str = "best_final_score",
) -> float:
    levels = sorted({int(row["schedule_time_block"]) for row in rows})
    predictions: list[float] = []
    observed: list[float] = []
    for world in sorted({int(row["world_seed"]) for row in rows}):
        training = [row for row in rows if int(row["world_seed"]) != world]
        held_out = [row for row in rows if int(row["world_seed"]) == world]
        fit = fit_random_intercept_reml(
            training,
            lifecycle_metric=lifecycle_metric,
            endpoint_metric=endpoint_metric,
            time_levels=levels,
        )
        coefficients = fit["coefficients"]
        for row in held_out:
            prediction = float(coefficients["intercept"]["estimate"])
            prediction += float(coefficients[f"delta_{endpoint_metric}"]["estimate"]) * float(
                row["nominal_minus_opaque"][endpoint_metric]
            )
            level = int(row["schedule_time_block"])
            if level != levels[0]:
                prediction += float(coefficients[f"time_block_{level}"]["estimate"])
            predictions.append(prediction)
            observed.append(float(row["nominal_minus_opaque"][lifecycle_metric]))
    observed_array = np.asarray(observed)
    prediction_array = np.asarray(predictions)
    total = float(np.sum((observed_array - np.mean(observed_array)) ** 2))
    if total <= 0.0:
        return 0.0
    return 1.0 - float(np.sum((observed_array - prediction_array) ** 2)) / total


def _arm(cell: Mapping[str, Any]) -> str:
    try:
        return CONDITIONS[str(cell["condition_id"])]
    except KeyError as error:
        raise EndpointLifecycleConfirmatoryError("unknown confirmatory arm") from error


def audit_endpoint_lifecycle_confirmatory(
    manifest_path: str | Path,
    *,
    expected_vessels_per_cell: int = 6,
) -> dict[str, Any]:
    """Audit all immutable trajectories and execute the frozen primary analysis."""

    manifest_file = Path(manifest_path).resolve()
    manifest_root = manifest_file.parent
    manifest = _load_json(manifest_file, label="confirmatory manifest")
    if manifest.get("schema_version") != EXPECTED_MANIFEST_VERSION:
        raise EndpointLifecycleConfirmatoryError("unsupported confirmatory manifest schema")
    _validate_content_hash(manifest, "manifest_sha256", label="manifest")
    if manifest.get("run_status") not in {
        "completed",
        "completed_with_right_censoring",
    }:
        raise EndpointLifecycleConfirmatoryError("confirmatory execution is not finalized")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise EndpointLifecycleConfirmatoryError("manifest source binding is missing")
    protocol_path = _resolve_repo_path(source.get("protocol_file"), label="protocol")
    if source.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise EndpointLifecycleConfirmatoryError("protocol byte hash mismatch")
    protocol = _load_json(protocol_path, label="confirmatory protocol")
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_VERSION:
        raise EndpointLifecycleConfirmatoryError("unsupported confirmatory protocol")
    freeze = protocol.get("confirmatory_freeze")
    if not isinstance(freeze, Mapping):
        raise EndpointLifecycleConfirmatoryError("protocol freeze block is missing")
    plan_binding = freeze.get("analysis_plan")
    if not isinstance(plan_binding, Mapping):
        raise EndpointLifecycleConfirmatoryError("analysis plan binding is missing")
    plan_path = _resolve_repo_path(plan_binding.get("path"), label="analysis plan")
    if plan_binding.get("sha256") != file_sha256(plan_path):
        raise EndpointLifecycleConfirmatoryError("analysis plan byte hash mismatch")
    plan = _load_json(plan_path, label="analysis plan")
    if plan.get("schema_version") != EXPECTED_ANALYSIS_PLAN_VERSION:
        raise EndpointLifecycleConfirmatoryError("unsupported analysis plan")
    worlds = tuple(int(seed) for seed in protocol["task"]["world_seeds"])
    replicates = tuple(str(item) for item in protocol["trajectory_replication"]["replicate_ids"])
    if manifest.get("world_seeds") != list(worlds):
        raise EndpointLifecycleConfirmatoryError("manifest world coverage mismatch")
    if manifest.get("trajectory_replicate_ids") != list(replicates):
        raise EndpointLifecycleConfirmatoryError("manifest replicate coverage mismatch")
    raw_states = manifest.get("cells")
    expected_cell_count = len(worlds) * len(replicates) * 2
    if not isinstance(raw_states, list) or len(raw_states) != expected_cell_count:
        raise EndpointLifecycleConfirmatoryError("manifest cell coverage mismatch")
    states = [
        _validate_state_attempt_policy(state, manifest_root=manifest_root)
        for state in raw_states
        if isinstance(state, Mapping)
    ]
    if len(states) != len(raw_states):
        raise EndpointLifecycleConfirmatoryError("manifest cell states are malformed")
    keyed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for state in states:
        cell = state["cell"]
        key = (
            int(cell["world_seed"]),
            str(cell["trajectory_replicate_id"]),
            _arm(cell),
        )
        if key in keyed:
            raise EndpointLifecycleConfirmatoryError(f"duplicate cell identity: {key}")
        keyed[key] = state
    expected_keys = {
        (world, replicate, arm)
        for world in worlds
        for replicate in replicates
        for arm in (OPAQUE_ARM, NOMINAL_ARM)
    }
    if set(keyed) != expected_keys:
        raise EndpointLifecycleConfirmatoryError("confirmatory grid is incomplete")
    completed: dict[tuple[int, str, str], dict[str, Any]] = {}
    censored: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key, state in keyed.items():
        if state["state"] == "completed":
            completed[key] = _completed_cell_audit(
                state_audit=state,
                manifest=manifest,
                manifest_root=manifest_root,
                expected_vessels=expected_vessels_per_cell,
            )
        else:
            censored[key] = _right_censored_cell_audit(
                state_audit=state,
                manifest_root=manifest_root,
                expected_vessels=expected_vessels_per_cell,
            )
    pair_rows: list[dict[str, Any]] = []
    for world in worlds:
        for replicate in replicates:
            opaque_state = keyed[(world, replicate, OPAQUE_ARM)]
            nominal_state = keyed[(world, replicate, NOMINAL_ARM)]
            physical = _pair_physical_audit(opaque_state, nominal_state)
            if not physical["passed"]:
                raise EndpointLifecycleConfirmatoryError(
                    f"physical pair audit failed: world={world}, replicate={replicate}"
                )
            pair_complete = opaque_state["state"] == nominal_state["state"] == "completed"
            row: dict[str, Any] = {
                "world_seed": world,
                "trajectory_replicate_id": replicate,
                "schedule_time_block": int(opaque_state["cell"]["schedule_time_block"]),
                "agent_seed": physical["agent_seed"],
                "opaque_state": opaque_state["state"],
                "nominal_state": nominal_state["state"],
                "pair_complete": pair_complete,
                "physical_pairing": physical,
            }
            if pair_complete:
                opaque = completed[(world, replicate, OPAQUE_ARM)]
                nominal = completed[(world, replicate, NOMINAL_ARM)]
                mismatches = [
                    field
                    for field in _PAIR_IDENTITY_FIELDS
                    if opaque["identity"][field] != nominal["identity"][field]
                ]
                if mismatches:
                    raise EndpointLifecycleConfirmatoryError(
                        "paired audited identity mismatch: " + ", ".join(mismatches)
                    )
                row.update(_paired_delta(nominal, opaque))
            else:
                row["nominal_minus_opaque"] = None
            pair_rows.append(row)
    complete_rows = [row for row in pair_rows if row["pair_complete"]]
    completed_by_world = Counter(int(row["world_seed"]) for row in complete_rows)
    minimum_fraction = float(plan["coverage_gate"]["minimum_completed_pair_fraction"])
    minimum_worlds = int(plan["coverage_gate"]["minimum_worlds_with_three_completed_pairs"])
    coverage = {
        "completed_pair_count": len(complete_rows),
        "planned_pair_count": len(pair_rows),
        "completed_pair_fraction": len(complete_rows) / len(pair_rows),
        "worlds_with_at_least_three_completed_pairs": sum(
            count >= 3 for count in completed_by_world.values()
        ),
        "minimum_completed_pair_fraction": minimum_fraction,
        "minimum_worlds_with_three_completed_pairs": minimum_worlds,
    }
    coverage["passed"] = (
        coverage["completed_pair_fraction"] >= minimum_fraction
        and coverage["worlds_with_at_least_three_completed_pairs"] >= minimum_worlds
    )
    primary: dict[str, Any] = {
        "status": "coverage_inconclusive",
        "success": False,
    }
    if coverage["passed"]:
        fit = fit_random_intercept_reml(complete_rows)
        lower = profile_lower_bound(fit)
        margin = float(plan["primary_estimand"]["substantive_margin_tbr_units"])
        fit_public = {key: value for key, value in fit.items() if key != "_fit"}
        primary = {
            "status": "primary_success" if lower > margin else "margin_not_crossed",
            "success": lower > margin,
            "substantive_margin": margin,
            "one_sided_confidence_level": 0.95,
            "sigma_unexplained_lower_bound": lower,
            "model": fit_public,
            "leave_one_world_out_r_squared": leave_one_world_out_r_squared(complete_rows),
        }
    censoring_by_arm = Counter(
        state["cell"]["condition_id"] for state in states if state["state"] == "right_censored"
    )
    censoring_by_world = Counter(
        int(state["cell"]["world_seed"]) for state in states if state["state"] == "right_censored"
    )
    censoring_by_block = Counter(
        int(state["cell"]["schedule_time_block"])
        for state in states
        if state["state"] == "right_censored"
    )
    report: dict[str, Any] = {
        "schema_version": ENDPOINT_LIFECYCLE_CONFIRMATORY_VERSION,
        "status": "completed_audited_confirmatory_analysis",
        "formal_result": True,
        "confirmatory_claim_allowed": bool(primary["success"]),
        "manifest": {
            "path": manifest_file.as_posix(),
            "sha256": file_sha256(manifest_file),
            "declared_manifest_sha256": manifest["manifest_sha256"],
        },
        "frozen_bindings": {
            "protocol_path": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": file_sha256(protocol_path),
            "analysis_plan_path": plan_path.relative_to(ROOT).as_posix(),
            "analysis_plan_sha256": file_sha256(plan_path),
            "source_tree_sha256": source.get("material_source_tree_sha256"),
            "source_git_commit": source.get("git_commit"),
            "source_worktree_dirty": source.get("worktree_dirty"),
        },
        "matrix": {
            "world_seeds": list(worlds),
            "trajectory_replicate_ids": list(replicates),
            "planned_cell_count": expected_cell_count,
            "completed_cell_count": len(completed),
            "right_censored_cell_count": len(censored),
            "all_physical_pairs_verified": all(
                row["physical_pairing"]["passed"] for row in pair_rows
            ),
            "all_terminal_cells_resource_replay_verified": all(
                cell["resource_ledger"]["verified"] and cell["exact_replay"]["verified"]
                for cell in (*completed.values(), *censored.values())
            ),
        },
        "coverage_gate": coverage,
        "primary_analysis": primary,
        "censoring": {
            "by_arm": dict(sorted(censoring_by_arm.items())),
            "by_world": {str(key): value for key, value in sorted(censoring_by_world.items())},
            "by_time_block": {str(key): value for key, value in sorted(censoring_by_block.items())},
            "right_censored_cells": [censored[key] for key in sorted(censored)],
        },
        "paired_trajectories": pair_rows,
        "completed_cells": [completed[key] for key in sorted(completed)],
    }
    report["audit_sha256"] = canonical_json_sha256(report)
    return report


def write_endpoint_lifecycle_confirmatory_audit(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = audit_endpoint_lifecycle_confirmatory(manifest_path)
    write_json_atomic(Path(output_path), report)
    return report


__all__ = [
    "ENDPOINT_LIFECYCLE_CONFIRMATORY_VERSION",
    "EndpointLifecycleConfirmatoryError",
    "audit_endpoint_lifecycle_confirmatory",
    "fit_random_intercept_reml",
    "leave_one_world_out_r_squared",
    "profile_lower_bound",
    "write_endpoint_lifecycle_confirmatory_audit",
]
