"""Task-aware scoring helpers for ChemWorld."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TASK_DERIVED_SCORING_CONTRACT = "task-derived-scoring-v1"
ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2 = "electrochemical-s0-balanced-efficiency-v2"
CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1 = "reaction-crystallization-s0-balanced-product-v1"
DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2 = "reaction-distillation-s0-balanced-audit-safety-v2"
PARTITION_S0_EXTRACTION_EFFICIENCY_V2 = "partition-s0-extraction-efficiency-v2"
PARTITION_S0_EXTRACTION_EFFICIENCY_V3 = "partition-s0-extraction-efficiency-v3"
FLOW_S0_BALANCED_PROCESS_V1 = "continuous-flow-s0-balanced-process-v1"


def risk_signal_contract() -> dict[str, Any]:
    """Explain the existing mixed penalty signal without changing score arithmetic."""
    return {
        "contract_id": "mixed-process-and-procedure-penalty-v1",
        "field": "safety_risk",
        "kind": "dimensionless_benchmark_penalty",
        "includes": ["process_model_proxies", "invalid_action_and_rollback_penalties"],
        "physical_hazard_probability": False,
        "unsafe_flag": "observed_penalty_at_or_above_task_limit",
        "missing_penalty": (
            "Rejected observations remain masked; unsafe=false then means no observed exceedance, "
            "not a safety assessment."
        ),
        "cost_components": {
            "safety_risk": "task_penalty_limit_exceeded",
            "high_cost": "normalized_process_cost_limit_exceeded",
            "precondition_failure": "rejected_operation",
            "constitution_failure": "transaction_invariant_failure",
        },
        "interpretation": (
            "Procedure penalties can increase on a rejected action with no physical state change. "
            "Process proxies are synthetic; this signal is not a calibrated laboratory hazard."
        ),
    }


@dataclass(frozen=True)
class ObjectiveWeights:
    yield_weight: float
    selectivity_weight: float
    conversion_weight: float
    cost_penalty: float
    risk_penalty: float


@dataclass(frozen=True)
class TaskScoringContract:
    """Serializable score contract compiled from task success metrics."""

    objective: str
    success_metrics: tuple[str, ...]
    score_family: str
    component_weights: dict[str, float]
    multiplicative_gates: dict[str, float] = field(default_factory=dict)
    contract_id: str = TASK_DERIVED_SCORING_CONTRACT

    @classmethod
    def from_success_metrics(
        cls,
        *,
        objective: str,
        success_metrics: tuple[str, ...] = (),
        contract_id: str = TASK_DERIVED_SCORING_CONTRACT,
    ) -> TaskScoringContract:
        metrics = frozenset(success_metrics)
        if contract_id == DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2:
            required = {
                "distillate_purity",
                "distillate_recovery",
                "solvent_loss",
            }
            if not required.issubset(metrics):
                raise ValueError(
                    "reaction-distillation S0 v2 scoring requires the distillation task metrics"
                )
            return cls(
                objective=objective,
                success_metrics=success_metrics,
                score_family="distillation",
                component_weights={
                    "reaction_score": 0.40,
                    "distillate_purity": 0.34,
                    "distillate_recovery": 0.22,
                    "solvent_loss": -0.10,
                },
                contract_id=contract_id,
            )
        if contract_id == CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1:
            required = {
                "crystal_yield",
                "crystal_purity",
                "crystal_size",
                "crystal_csd_quality",
                "crystal_fines_fraction",
            }
            if not required.issubset(metrics):
                raise ValueError(
                    "reaction-crystallization S0 v1 scoring requires the "
                    "crystallization task metrics"
                )
            return cls(
                objective=objective,
                success_metrics=success_metrics,
                score_family="crystallization",
                component_weights={
                    "reaction_score": 0.20,
                    "crystal_yield": 0.25,
                    "crystal_purity": 0.20,
                    "crystal_size": 0.10,
                    "crystal_csd_quality": 0.25,
                    "crystal_fines_fraction": -0.10,
                },
                contract_id=contract_id,
            )
        if contract_id in {
            PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
            PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
        }:
            required = {
                "phase_ratio",
                "product_in_organic",
                "product_in_aqueous",
            }
            if not required.issubset(metrics):
                raise ValueError("partition S0 v2 scoring requires the partition task metrics")
            return cls(
                objective=objective,
                success_metrics=success_metrics,
                score_family="partition",
                component_weights={
                    "product_in_organic": 0.85,
                    "product_in_aqueous": -0.10,
                    "phase_ratio": -0.10,
                },
                contract_id=contract_id,
            )
        if contract_id == FLOW_S0_BALANCED_PROCESS_V1:
            required = {"flow_conversion", "yield", "safety_risk"}
            if not required.issubset(metrics):
                raise ValueError("continuous-flow S0 v1 scoring requires the flow task metrics")
            return cls(
                objective=objective,
                success_metrics=success_metrics,
                score_family="continuous_flow",
                component_weights={
                    "flow_conversion": 0.40,
                    "yield": 0.30,
                    "selectivity": 0.20,
                    "cost": -0.04,
                    "safety_risk": -0.06,
                },
                contract_id=contract_id,
            )
        if contract_id == ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2:
            required = {
                "selective_product_yield",
                "electrochemical_selectivity",
                "faradaic_efficiency",
                "transport_efficiency",
                "ohmic_efficiency",
                "energy_efficiency",
            }
            if not required.issubset(metrics):
                raise ValueError(
                    "electrochemical S0 v2 scoring requires the electrochemical task metrics"
                )
            return cls(
                objective=objective,
                success_metrics=success_metrics,
                score_family="electrochemistry",
                component_weights={
                    "selective_product_yield": 0.30,
                    "electrochemical_selectivity": 0.15,
                    "electrochemical_conversion": 0.10,
                    "faradaic_efficiency": 0.12,
                    "transport_efficiency": 0.10,
                    "ohmic_efficiency": 0.08,
                    "energy_efficiency": 0.15,
                },
                multiplicative_gates={"selective_product_yield": 0.02},
                contract_id=contract_id,
            )
        if contract_id != TASK_DERIVED_SCORING_CONTRACT:
            raise ValueError(f"unsupported scoring contract ID: {contract_id}")
        if metrics.intersection(
            {
                "crystal_yield",
                "crystal_purity",
                "crystal_size",
                "crystal_csd_quality",
                "crystal_fines_fraction",
            }
        ):
            return cls(
                objective,
                success_metrics,
                "crystallization",
                {
                    "reaction_score": 0.25,
                    "crystal_yield": 0.25,
                    "crystal_purity": 0.20,
                    "crystal_size": 0.10,
                    "crystal_csd_quality": 0.20,
                    "crystal_fines_fraction": -0.10,
                },
            )
        if metrics.intersection({"distillate_purity", "distillate_recovery"}):
            return cls(
                objective,
                success_metrics,
                "distillation",
                {
                    "reaction_score": 0.40,
                    "distillate_purity": 0.34,
                    "distillate_recovery": 0.22,
                    "solvent_loss": -0.10,
                },
            )
        if metrics.intersection(
            {
                "electrochemical_selectivity",
                "selective_product_yield",
                "faradaic_efficiency",
                "transport_efficiency",
                "ohmic_efficiency",
                "energy_efficiency",
            }
        ):
            return cls(
                objective,
                success_metrics,
                "electrochemistry",
                {
                    "reaction_score": 0.10,
                    "selective_product_yield": 0.25,
                    "electrochemical_selectivity": 0.15,
                    "faradaic_efficiency": 0.12,
                    "transport_efficiency": 0.10,
                    "ohmic_efficiency": 0.08,
                    "energy_efficiency": 0.15,
                    "conversion": 0.05,
                },
                {"selective_product_yield": 0.02},
            )
        if metrics.intersection(
            {
                "pH_normalized",
                "acid_dissociation_fraction",
                "precipitation_signal",
                "equilibrium_residual",
                "equilibrium_confidence",
            }
        ):
            return cls(
                objective,
                success_metrics,
                "equilibrium_characterization",
                {
                    "equilibrium_confidence": 0.45,
                    "acid_dissociation_fraction": 0.20,
                    "precipitation_signal": 0.15,
                    "pH_normalized": 0.10,
                    "equilibrium_residual": -0.25,
                },
            )
        if "flow_conversion" in metrics:
            return cls(
                objective,
                success_metrics,
                "continuous_flow",
                {
                    "flow_conversion": 0.40,
                    "yield": 0.30,
                    "selectivity": 0.20,
                    "cost": -0.04,
                    "safety_risk": -0.06,
                },
            )
        if metrics.intersection({"purity", "recovery", "process_mass_balance_error"}):
            return cls(
                objective,
                success_metrics,
                "purification",
                {
                    "reaction_score": 0.35,
                    "purity": 0.35,
                    "recovery": 0.25,
                    "process_mass_balance_error": -0.10,
                },
            )
        if metrics.intersection({"phase_ratio", "product_in_organic", "product_in_aqueous"}):
            return cls(
                objective,
                success_metrics,
                "partition",
                {
                    "product_in_organic": 0.85,
                    "product_in_aqueous": -0.10,
                    "phase_ratio": -0.10,
                },
            )
        return cls(
            objective,
            success_metrics,
            "reaction",
            {"reaction_score": 1.0},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "objective": self.objective,
            "success_metrics": list(self.success_metrics),
            "score_family": self.score_family,
            "component_weights": dict(self.component_weights),
            "multiplicative_gates": dict(self.multiplicative_gates),
            "contract_hash": self.contract_hash,
        }

    @property
    def contract_hash(self) -> str:
        payload = {
            "contract_id": self.contract_id,
            "objective": self.objective,
            "success_metrics": list(self.success_metrics),
            "score_family": self.score_family,
            "component_weights": dict(sorted(self.component_weights.items())),
            "multiplicative_gates": dict(sorted(self.multiplicative_gates.items())),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()


OBJECTIVES: dict[str, ObjectiveWeights] = {
    "balanced": ObjectiveWeights(0.50, 0.25, 0.10, 0.15, 0.25),
    "yield": ObjectiveWeights(0.75, 0.10, 0.05, 0.05, 0.10),
    "safe": ObjectiveWeights(0.40, 0.25, 0.10, 0.10, 0.45),
}


def scalar_observation(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = observation.get(key)
    if value is None:
        return default
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    return float(value)


def reaction_score(observation: dict[str, Any]) -> float:
    return scalar_observation(observation, "score")


def purification_score(observation: dict[str, Any]) -> float:
    purity = scalar_observation(observation, "purity")
    recovery = scalar_observation(observation, "recovery")
    mass_balance_error = scalar_observation(observation, "process_mass_balance_error")
    return float(np.clip(0.55 * purity + 0.35 * recovery - 0.10 * mass_balance_error, 0.0, 1.0))


def safety_cost_from_flags(flags: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components = {
        "safety_risk": float(bool(flags.get("unsafe", False))),
        "high_cost": float(bool(flags.get("high_cost", False))),
        "precondition_failure": float(bool(flags.get("precondition_failed", False))),
        "constitution_failure": float(bool(flags.get("constitution_failed", False))),
    }
    return min(1.0, sum(components.values())), components


def score_observation(
    *,
    objective: str,
    product_yield: float,
    selectivity: float,
    conversion: float,
    cost: float,
    safety_risk: float,
) -> float:
    """Compute the benchmark scalar score in [0, 1]."""

    if objective not in OBJECTIVES:
        allowed = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"Unknown objective {objective!r}. Allowed: {allowed}")

    weights = OBJECTIVES[objective]
    raw = (
        weights.yield_weight * product_yield
        + weights.selectivity_weight * selectivity
        + weights.conversion_weight * conversion
        - weights.cost_penalty * cost
        - weights.risk_penalty * safety_risk
    )
    return float(np.clip(raw, 0.0, 1.0))


def task_score_observation(
    *,
    contract: TaskScoringContract,
    values: dict[str, float | None],
) -> float:
    """Compute the task-specific scalar score in [0, 1]."""

    reaction_component = score_observation(
        objective=contract.objective,
        product_yield=scalar_observation(values, "yield"),
        selectivity=scalar_observation(values, "selectivity"),
        conversion=scalar_observation(values, "conversion"),
        cost=scalar_observation(values, "cost"),
        safety_risk=(
            0.0
            if contract.contract_id == DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2
            else scalar_observation(values, "safety_risk")
        ),
    )
    components = {"reaction_score": reaction_component}
    components.update(
        {
            key: scalar_observation(values, key)
            for key in contract.component_weights
            if key != "reaction_score"
        }
    )
    raw = sum(
        weight * components.get(key, 0.0) for key, weight in contract.component_weights.items()
    )
    for metric, full_credit_threshold in contract.multiplicative_gates.items():
        if full_credit_threshold <= 0.0:
            raise ValueError("score gate thresholds must be positive")
        raw *= float(np.clip(scalar_observation(values, metric) / full_credit_threshold, 0.0, 1.0))
    return float(np.clip(raw, 0.0, 1.0))


__all__ = [
    "CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1",
    "DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2",
    "ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2",
    "FLOW_S0_BALANCED_PROCESS_V1",
    "OBJECTIVES",
    "PARTITION_S0_EXTRACTION_EFFICIENCY_V2",
    "PARTITION_S0_EXTRACTION_EFFICIENCY_V3",
    "TASK_DERIVED_SCORING_CONTRACT",
    "ObjectiveWeights",
    "TaskScoringContract",
    "purification_score",
    "reaction_score",
    "safety_cost_from_flags",
    "scalar_observation",
    "score_observation",
    "task_score_observation",
]
