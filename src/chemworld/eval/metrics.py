"""Benchmark metrics for ChemWorld trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np

from chemworld.data.records import trajectory_value

EVALUATION_METRICS_VERSION = "chemworld-evaluation-metrics-0.3"


@dataclass(frozen=True)
class EvaluationResult:
    agent_name: str
    env_id: str
    benchmark_task_id: str | None
    world_split: str
    seed: int
    episode_mode: str
    steps: int
    final_assay_count: int
    final_best_score: float
    best_valid_score: float
    best_valid_yield: float
    area_under_best_score: float
    sample_efficiency_step: int | None
    safety_violations: int
    high_cost_violations: int
    invalid_action_count: int
    invalid_action_rate: float
    precondition_recovery_count: int
    mean_cost: float
    mean_safety_risk: float
    mean_purity: float
    mean_recovery: float
    mean_phase_ratio: float
    mean_product_in_organic: float
    mean_product_in_aqueous: float
    purification_score: float
    mean_crystal_yield: float
    mean_crystal_purity: float
    mean_distillate_purity: float
    mean_distillate_recovery: float
    mean_flow_conversion: float
    mean_selective_product_yield: float
    mean_electrochemical_selectivity: float
    mean_energy_efficiency: float
    mean_pH_normalized: float
    mean_acid_dissociation_fraction: float
    mean_precipitation_signal: float
    mean_equilibrium_residual: float
    mean_equilibrium_confidence: float
    phase_mass_balance_violations: int
    process_mass_balance_violation_count: int
    instrument_policy_violation_count: int
    precondition_failure_count: int
    campaign_area_under_best_score: float
    safety_cost: float
    safety_aware_score: float
    cost_aware_score: float
    bo_initial_recipe_count: int
    bo_acquisition_recipe_count: int
    bo_entered_acquisition: bool
    observation_use_summary: dict[str, Any]
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _agent_name(records: list[dict[str, Any]]) -> str:
    metadata = records[0].get("agent_metadata", {})
    return str(metadata.get("agent_name") or metadata.get("agent_family") or "unknown")


def _obs_float(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = observation.get(key)
    return default if value is None else _finite_float(value, f"observation.{key}")


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _latest_agent_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(records):
        trace = record.get("agent_trace")
        if isinstance(trace, list) and trace:
            return [item for item in trace if isinstance(item, dict)]
    return []


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.75,
) -> EvaluationResult:
    """Evaluate one trajectory JSONL payload."""

    if not records:
        raise ValueError("Cannot evaluate an empty trajectory")
    threshold = _finite_float(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    leaderboard_values = [trajectory_value(record, 'leaderboard_score', None) for record in records]
    finite_leaderboard_values = [
        _finite_float(value, "leaderboard_score")
        for value in leaderboard_values
        if value is not None
    ]
    if any(not 0.0 <= value <= 1.0 for value in finite_leaderboard_values):
        raise ValueError("leaderboard_score must be in [0, 1]")
    official_scores = np.asarray(
        [
            0.0 if value is None else _finite_float(value, "leaderboard_score")
            for value in leaderboard_values
        ],
        dtype=float,
    )
    scored_mask = np.asarray([value is not None for value in leaderboard_values])
    best_curve = np.maximum.accumulate(official_scores)
    observations = [trajectory_value(record, 'observation') for record in records]
    flags = [record.get("constraint_flags", {}) for record in records]

    unsafe = np.asarray([bool(flag.get("unsafe", False)) for flag in flags])
    high_cost = np.asarray([bool(flag.get("high_cost", False)) for flag in flags])
    invalid_actions = np.asarray(
        [
            bool(flag.get("precondition_failed", False))
            or bool(flag.get("constitution_failed", False))
            for flag in flags
        ]
    )
    precondition_failures = np.asarray(
        [bool(flag.get("precondition_failed", False)) for flag in flags]
    )
    phase_mass_balance_failures = np.asarray(
        [bool(flag.get("phase_mass_balance_failed", False)) for flag in flags]
    )
    instrument_policy_violations = np.asarray(
        [
            record.get("operation_type") == "measure"
            and not record.get("preconditions", {}).get("instrument_allowed_by_task", True)
            for record in records
        ]
    )
    valid_mask = scored_mask & ~(unsafe | high_cost)
    valid_scores = official_scores[valid_mask]

    yields = np.asarray([_obs_float(obs, "yield") for obs in observations], dtype=float)
    valid_yields = yields[valid_mask]
    costs = [_obs_float(obs, "cost") for obs in observations]
    risks = [_obs_float(obs, "safety_risk") for obs in observations]
    purities = [_obs_float(obs, "purity") for obs in observations if obs.get("purity") is not None]
    recoveries = [
        _obs_float(obs, "recovery") for obs in observations if obs.get("recovery") is not None
    ]
    phase_ratios = [
        _obs_float(obs, "phase_ratio") for obs in observations if obs.get("phase_ratio") is not None
    ]
    products_in_organic = [
        _obs_float(obs, "product_in_organic")
        for obs in observations
        if obs.get("product_in_organic") is not None
    ]
    products_in_aqueous = [
        _obs_float(obs, "product_in_aqueous")
        for obs in observations
        if obs.get("product_in_aqueous") is not None
    ]
    crystal_yields = [
        _obs_float(obs, "crystal_yield")
        for obs in observations
        if obs.get("crystal_yield") is not None
    ]
    crystal_purities = [
        _obs_float(obs, "crystal_purity")
        for obs in observations
        if obs.get("crystal_purity") is not None
    ]
    distillate_purities = [
        _obs_float(obs, "distillate_purity")
        for obs in observations
        if obs.get("distillate_purity") is not None
    ]
    distillate_recoveries = [
        _obs_float(obs, "distillate_recovery")
        for obs in observations
        if obs.get("distillate_recovery") is not None
    ]
    flow_conversions = [
        _obs_float(obs, "flow_conversion")
        for obs in observations
        if obs.get("flow_conversion") is not None
    ]
    electrochemical_selectivities = [
        _obs_float(obs, "electrochemical_selectivity")
        for obs in observations
        if obs.get("electrochemical_selectivity") is not None
    ]
    selective_product_yields = [
        _obs_float(obs, "selective_product_yield")
        for obs in observations
        if obs.get("selective_product_yield") is not None
    ]
    energy_efficiencies = [
        _obs_float(obs, "energy_efficiency")
        for obs in observations
        if obs.get("energy_efficiency") is not None
    ]
    ph_normalized_values = [
        _obs_float(obs, "pH_normalized")
        for obs in observations
        if obs.get("pH_normalized") is not None
    ]
    acid_dissociation_values = [
        _obs_float(obs, "acid_dissociation_fraction")
        for obs in observations
        if obs.get("acid_dissociation_fraction") is not None
    ]
    precipitation_values = [
        _obs_float(obs, "precipitation_signal")
        for obs in observations
        if obs.get("precipitation_signal") is not None
    ]
    equilibrium_residuals = [
        _obs_float(obs, "equilibrium_residual")
        for obs in observations
        if obs.get("equilibrium_residual") is not None
    ]
    equilibrium_confidences = [
        _obs_float(obs, "equilibrium_confidence")
        for obs in observations
        if obs.get("equilibrium_confidence") is not None
    ]
    mass_balance_errors = [
        _obs_float(obs, "process_mass_balance_error")
        for obs in observations
        if obs.get("process_mass_balance_error") is not None
    ]

    hit_indices = np.where(best_curve >= threshold)[0]
    sample_efficiency_step = int(hit_indices[0] + 1) if hit_indices.size else None
    final_best_score = float(best_curve[-1])
    best_valid_score = float(np.max(valid_scores)) if valid_scores.size else 0.0
    best_valid_yield = float(np.max(valid_yields)) if valid_yields.size else 0.0
    area = float(np.mean(best_curve))
    safety_violations = int(np.sum(unsafe))
    high_cost_violations = int(np.sum(high_cost))
    invalid_action_count = int(np.sum(invalid_actions))
    invalid_action_rate = invalid_action_count / max(len(records), 1)
    mean_cost = float(fmean(costs))
    mean_risk = float(fmean(risks))
    mean_purity = float(fmean(purities)) if purities else 0.0
    mean_recovery = float(fmean(recoveries)) if recoveries else 0.0
    mean_phase_ratio = float(fmean(phase_ratios)) if phase_ratios else 0.0
    mean_product_in_organic = float(fmean(products_in_organic)) if products_in_organic else 0.0
    mean_product_in_aqueous = float(fmean(products_in_aqueous)) if products_in_aqueous else 0.0
    mean_crystal_yield = float(fmean(crystal_yields)) if crystal_yields else 0.0
    mean_crystal_purity = float(fmean(crystal_purities)) if crystal_purities else 0.0
    mean_distillate_purity = float(fmean(distillate_purities)) if distillate_purities else 0.0
    mean_distillate_recovery = float(fmean(distillate_recoveries)) if distillate_recoveries else 0.0
    mean_flow_conversion = float(fmean(flow_conversions)) if flow_conversions else 0.0
    mean_electrochemical_selectivity = (
        float(fmean(electrochemical_selectivities)) if electrochemical_selectivities else 0.0
    )
    mean_selective_product_yield = (
        float(fmean(selective_product_yields)) if selective_product_yields else 0.0
    )
    mean_energy_efficiency = float(fmean(energy_efficiencies)) if energy_efficiencies else 0.0
    mean_pH_normalized = float(fmean(ph_normalized_values)) if ph_normalized_values else 0.0
    mean_acid_dissociation_fraction = (
        float(fmean(acid_dissociation_values)) if acid_dissociation_values else 0.0
    )
    mean_precipitation_signal = float(fmean(precipitation_values)) if precipitation_values else 0.0
    mean_equilibrium_residual = (
        float(fmean(equilibrium_residuals)) if equilibrium_residuals else 0.0
    )
    mean_equilibrium_confidence = (
        float(fmean(equilibrium_confidences)) if equilibrium_confidences else 0.0
    )
    mean_mass_balance_error = float(fmean(mass_balance_errors)) if mass_balance_errors else 0.0
    purification_score = float(
        np.clip(
            0.55 * mean_purity + 0.35 * mean_recovery - 0.10 * mean_mass_balance_error,
            0.0,
            1.0,
        )
    )
    process_mass_balance_violation_count = int(sum(error > 0.05 for error in mass_balance_errors))
    precondition_recovery_count = int(
        sum(
            bool(precondition_failures[index]) and not bool(precondition_failures[index + 1])
            for index in range(max(len(precondition_failures) - 1, 0))
        )
    )
    instrument_counts: dict[str, int] = {}
    observed_key_counts: dict[str, int] = {}
    for record in records:
        instrument = record.get("instrument")
        if instrument:
            instrument_counts[str(instrument)] = instrument_counts.get(str(instrument), 0) + 1
        for key in record.get("observed_keys", []):
            observed_key_counts[str(key)] = observed_key_counts.get(str(key), 0) + 1
    observation_use_summary = {
        "instrument_counts": instrument_counts,
        "observed_key_counts": observed_key_counts,
        "final_assay_count": int(np.sum(scored_mask)),
    }
    agent_trace = _latest_agent_trace(records)
    surrogate_decisions = [
        entry for entry in agent_trace if entry.get("trace_type") == "surrogate_recipe_decision"
    ]
    bo_initial_recipe_count = sum(
        1 for entry in surrogate_decisions if entry.get("phase") == "initial"
    )
    bo_acquisition_recipe_count = sum(
        1 for entry in surrogate_decisions if entry.get("phase") == "acquisition"
    )

    safety_penalty = min(
        1.0,
        0.08 * safety_violations
        + 0.04 * high_cost_violations
        + 0.05 * invalid_action_count
        + 0.20 * mean_risk,
    )
    safety_aware_score = float(np.clip(best_valid_score - safety_penalty, 0.0, 1.0))
    cost_aware_score = float(
        np.clip(best_valid_score - 0.10 * mean_cost - 0.10 * mean_risk, 0.0, 1.0)
    )
    efficiency_score = area
    if sample_efficiency_step is not None:
        efficiency_score = max(
            efficiency_score,
            1.0 - (sample_efficiency_step - 1) / max(len(records), 1),
        )

    total_score = float(
        np.clip(
            0.40 * final_best_score
            + 0.25 * efficiency_score
            + 0.20 * safety_aware_score
            + 0.15 * best_valid_score,
            0.0,
            1.0,
        )
    )

    first = records[0]
    return EvaluationResult(
        agent_name=_agent_name(records),
        env_id=str(first["env_id"]),
        benchmark_task_id=trajectory_value(first, 'benchmark_task_id', None),
        world_split=str(first["world_split"]),
        seed=int(first["seed"]),
        episode_mode=str(first.get("episode_mode", "single_experiment")),
        steps=len(records),
        final_assay_count=int(np.sum(scored_mask)),
        final_best_score=final_best_score,
        best_valid_score=best_valid_score,
        best_valid_yield=best_valid_yield,
        area_under_best_score=area,
        sample_efficiency_step=sample_efficiency_step,
        safety_violations=safety_violations,
        high_cost_violations=high_cost_violations,
        invalid_action_count=invalid_action_count,
        invalid_action_rate=invalid_action_rate,
        precondition_recovery_count=precondition_recovery_count,
        mean_cost=mean_cost,
        mean_safety_risk=mean_risk,
        mean_purity=mean_purity,
        mean_recovery=mean_recovery,
        mean_phase_ratio=mean_phase_ratio,
        mean_product_in_organic=mean_product_in_organic,
        mean_product_in_aqueous=mean_product_in_aqueous,
        purification_score=purification_score,
        mean_crystal_yield=mean_crystal_yield,
        mean_crystal_purity=mean_crystal_purity,
        mean_distillate_purity=mean_distillate_purity,
        mean_distillate_recovery=mean_distillate_recovery,
        mean_flow_conversion=mean_flow_conversion,
        mean_selective_product_yield=mean_selective_product_yield,
        mean_electrochemical_selectivity=mean_electrochemical_selectivity,
        mean_energy_efficiency=mean_energy_efficiency,
        mean_pH_normalized=mean_pH_normalized,
        mean_acid_dissociation_fraction=mean_acid_dissociation_fraction,
        mean_precipitation_signal=mean_precipitation_signal,
        mean_equilibrium_residual=mean_equilibrium_residual,
        mean_equilibrium_confidence=mean_equilibrium_confidence,
        phase_mass_balance_violations=int(np.sum(phase_mass_balance_failures)),
        process_mass_balance_violation_count=process_mass_balance_violation_count,
        instrument_policy_violation_count=int(np.sum(instrument_policy_violations)),
        precondition_failure_count=int(np.sum(precondition_failures)),
        campaign_area_under_best_score=area,
        safety_cost=safety_penalty,
        safety_aware_score=safety_aware_score,
        cost_aware_score=cost_aware_score,
        bo_initial_recipe_count=bo_initial_recipe_count,
        bo_acquisition_recipe_count=bo_acquisition_recipe_count,
        bo_entered_acquisition=bo_acquisition_recipe_count > 0,
        observation_use_summary=observation_use_summary,
        total_score=total_score,
    )


__all__ = [
    "EVALUATION_METRICS_VERSION",
    "EvaluationResult",
    "evaluate_records",
]
