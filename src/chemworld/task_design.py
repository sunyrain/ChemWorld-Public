"""Machine-readable readiness contracts for serious benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chemworld.physchem.maturity import MaturityLevel
from chemworld.tasks import SERIOUS_TASK_IDS, TaskSpec, get_task
from chemworld.world.operations import PUBLIC_OBSERVATION_KEYS
from chemworld.world.parameters import WORLD_FAMILY_VERSION
from chemworld.world.world_family import WORLD_AXIS_REGISTRY, axes_for_task

TASK_DESIGN_VERSION = "chemworld-serious-task-design-0.4"


def _generalization_contract(
    axis_id: str,
    *,
    hidden_drivers: tuple[str, ...],
    evaluation: str,
    support_scope: str = "candidate_task_axis",
    required_context: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Bind scientific annotations to the executable world-axis registry."""

    axis = WORLD_AXIS_REGISTRY[axis_id]
    return {
        "axis_id": axis.axis_id,
        "label": axis.label,
        "target_kind": axis.target_kind,
        "runtime_target_keys": axis.target_keys,
        "hidden_drivers": hidden_drivers,
        "evaluation": evaluation,
        "support_scope": support_scope,
        "required_context": required_context,
    }


def _generalization_axis_labels(task_id: str) -> tuple[str, ...]:
    return tuple(axis.label for axis in axes_for_task(task_id))


SERIOUS_GENERALIZATION_CONTRACTS: dict[str, tuple[dict[str, Any], ...]] = {
    "partition-discovery": (
        _generalization_contract(
            "partition.distribution-coefficient",
            hidden_drivers=("solvent family", "partition-law seed"),
            evaluation="frozen-seed stratification and private-salt shift",
        ),
        _generalization_contract(
            "partition.phase-volume-ratio",
            hidden_drivers=("effective organic-to-aqueous contact-volume ratio",),
            evaluation="task-recipe response-surface audit",
        ),
    ),
    "reaction-to-crystallization": (
        _generalization_contract(
            "crystallization.kinetic-profile",
            hidden_drivers=(
                "primary nucleation coefficient",
                "supersaturation and seed/cooling interaction",
            ),
            evaluation="frozen-axis response-surface audit",
        ),
        _generalization_contract(
            "crystallization.solubility-cooling-profile",
            hidden_drivers=(
                "reference solubility scale",
                "temperature-dependent solubility response",
            ),
            evaluation="task-recipe response-surface audit",
        ),
        _generalization_contract(
            "crystallization.population-regime",
            hidden_drivers=("opposed primary nucleation and growth multipliers",),
            evaluation="development population-response and material-closure checks",
            support_scope="development_authoring",
        ),
        _generalization_contract(
            "crystallization.impurity-occlusion-law",
            hidden_drivers=("supersaturation-transfer versus surface-saturation law",),
            evaluation="development structural-law comparison",
            support_scope="development_authoring",
        ),
        _generalization_contract(
            "crystallization.impurity-occlusion-capacity",
            hidden_drivers=("constant occlusion-capacity multiplier",),
            evaluation="development scalar-null comparison; response can be zero",
            support_scope="development_authoring",
        ),
    ),
    "reaction-to-distillation": (
        _generalization_contract(
            "distillation.relative-volatility",
            hidden_drivers=("relative-volatility scale", "temperature and reflux response"),
            evaluation="frozen-seed stratification and response-surface audit",
        ),
        _generalization_contract(
            "distillation.reaction-selectivity",
            hidden_drivers=("catalyst effects", "solvent effects"),
            evaluation="frozen-seed stratification and private-salt shift",
        ),
    ),
    "flow-reaction-optimization": (
        _generalization_contract(
            "flow.reaction-kinetics",
            hidden_drivers=("compiled PFR reaction-rate scale",),
            evaluation="frozen-seed stratification and private-salt shift",
        ),
        _generalization_contract(
            "flow.residence-thermal-boundary",
            hidden_drivers=("residence-time calibration", "wall heat-transfer strength"),
            evaluation="task-recipe response-surface audit",
        ),
    ),
    "electrochemical-conversion": (
        _generalization_contract(
            "electrochem.selectivity-kinetics",
            hidden_drivers=("overpotential-selectivity sensitivity",),
            evaluation="frozen-seed stratification and private-salt shift",
        ),
        _generalization_contract(
            "electrochem.mass-transfer-resistance",
            hidden_drivers=("electrolyte resistance", "contact resistance"),
            evaluation="task-recipe response-surface audit",
        ),
    ),
    "equilibrium-characterization": (
        _generalization_contract(
            "equilibrium.acid-base-constants",
            hidden_drivers=("hidden pKa", "solution composition"),
            evaluation="frozen-seed stratification and private-salt shift",
        ),
        _generalization_contract(
            "equilibrium.activity-coefficient-regime",
            hidden_drivers=("hidden activity-coefficient ratio", "solution concentration"),
            evaluation="task-recipe response-surface audit",
        ),
        _generalization_contract(
            "equilibrium.mechanism-benchmark",
            hidden_drivers=("coupled weak-acid precipitation substrate",),
            evaluation="development coupled-equilibrium closure checks",
            support_scope="development_authoring",
        ),
        _generalization_contract(
            "equilibrium.aqueous-ion-pair-network",
            hidden_drivers=("aqueous association intermediate and mass-action constant",),
            evaluation="development paired-network response checks",
            support_scope="development_authoring",
            required_context=("equilibrium.mechanism-benchmark substrate",),
        ),
    ),
}
EXECUTABLE_EVALUATION_METRICS = frozenset(
    {
        *PUBLIC_OBSERVATION_KEYS,
        "total_score",
        "safety_aware_score",
        "cost_aware_score",
        "sample_efficiency_step",
        "invalid_action_rate",
        "area_under_best_score",
        "campaign_area_under_best_score",
    }
)


@dataclass(frozen=True)
class SeriousTaskDesign:
    task_id: str
    research_question: str
    capability_claim: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    generalization_axes: tuple[str, ...]
    required_baselines: tuple[str, ...]
    required_evidence: tuple[str, ...]
    anti_gaming_checks: tuple[str, ...]
    minimum_maturity: MaturityLevel = MaturityLevel.LITE
    allow_proxy: bool = False
    status: str = "candidate"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_maturity",
            MaturityLevel.normalize(self.minimum_maturity),
        )
        for name in ("task_id", "research_question", "capability_claim", "primary_metric"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.status not in {"candidate", "validated"}:
            raise ValueError("status must be 'candidate' or 'validated'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_version": TASK_DESIGN_VERSION,
            "task_id": self.task_id,
            "research_question": self.research_question,
            "capability_claim": self.capability_claim,
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "generalization_axes": list(self.generalization_axes),
            "required_baselines": list(self.required_baselines),
            "required_evidence": list(self.required_evidence),
            "anti_gaming_checks": list(self.anti_gaming_checks),
            "minimum_maturity": self.minimum_maturity.value,
            "allow_proxy": self.allow_proxy,
            "status": self.status,
        }


@dataclass(frozen=True)
class TaskDesignCheck:
    check_id: str
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass(frozen=True)
class TaskDesignReview:
    task_id: str
    checks: tuple[TaskDesignCheck, ...]
    empirical_status: str

    @property
    def contract_ready(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def benchmark_ready(self) -> bool:
        return self.contract_ready and self.empirical_status == "validated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "contract_ready": self.contract_ready,
            "benchmark_ready": self.benchmark_ready,
            "empirical_status": self.empirical_status,
            "checks": [check.to_dict() for check in self.checks],
        }


def review_task_design(
    task: TaskSpec,
    design: SeriousTaskDesign,
    *,
    empirical_status: str | None = None,
) -> TaskDesignReview:
    """Evaluate whether a task has an executable, non-proxy research contract."""

    success_metrics = set(task.success_metrics)
    declared_metrics = {design.primary_metric, *design.secondary_metrics}
    unsupported_metrics = sorted(success_metrics - EXECUTABLE_EVALUATION_METRICS)
    missing_declared_metrics = sorted(declared_metrics - success_metrics)
    maturity = task.kernel_maturity.lowest_level
    generalization_contract = SERIOUS_GENERALIZATION_CONTRACTS.get(task.task_id, ())
    executable_axes = axes_for_task(task.task_id)
    executable_axes_by_id = {axis.axis_id: axis for axis in executable_axes}
    declared_axis_ids = tuple(str(item.get("axis_id", "")) for item in generalization_contract)
    runtime_axis_binding_valid = (
        len(declared_axis_ids) == len(set(declared_axis_ids))
        and set(declared_axis_ids) == set(executable_axes_by_id)
        and all(
            item.get("label") == executable_axes_by_id[str(item["axis_id"])].label
            and item.get("target_kind") == executable_axes_by_id[str(item["axis_id"])].target_kind
            and tuple(item.get("runtime_target_keys", ()))
            == executable_axes_by_id[str(item["axis_id"])].target_keys
            for item in generalization_contract
            if str(item.get("axis_id", "")) in executable_axes_by_id
        )
    )
    checks = (
        TaskDesignCheck(
            "identity",
            task.task_id == design.task_id,
            "design and task ids must match",
        ),
        TaskDesignCheck(
            "world_law",
            task.world_law_id == WORLD_FAMILY_VERSION,
            "task must use the current frozen world law",
        ),
        TaskDesignCheck(
            "public_evaluation_split",
            task.world_split == "public-test",
            "candidate tasks require a reproducible public-test split",
        ),
        TaskDesignCheck(
            "seed_depth",
            len(task.seeds) >= 5,
            "at least five frozen seeds are required for the serious suite",
        ),
        TaskDesignCheck(
            "decision_horizon",
            task.budget >= 24,
            "budget must support a non-trivial decision horizon",
        ),
        TaskDesignCheck(
            "campaign_learning",
            task.episode_mode == "campaign",
            "serious tasks must permit observation, learning, and a later experiment",
        ),
        TaskDesignCheck(
            "metric_implementation",
            not unsupported_metrics,
            f"unsupported success metrics: {unsupported_metrics}",
        ),
        TaskDesignCheck(
            "metric_alignment",
            not missing_declared_metrics,
            f"design metrics absent from task contract: {missing_declared_metrics}",
        ),
        TaskDesignCheck(
            "final_assay",
            "final_assay" in task.allowed_instruments,
            "serious tasks require a final hidden-state assay boundary",
        ),
        TaskDesignCheck(
            "maturity_floor",
            maturity.rank >= design.minimum_maturity.rank,
            f"task maturity {maturity.value} must meet {design.minimum_maturity.value}",
        ),
        TaskDesignCheck(
            "proxy_policy",
            design.allow_proxy or not task.kernel_maturity.proxy_allowed,
            "proxy modules are excluded from the serious candidate suite",
        ),
        TaskDesignCheck(
            "baseline_diversity",
            len(design.required_baselines) >= 3
            and "random" in design.required_baselines
            and any(
                name in design.required_baselines
                for name in ("gp_bo", "safe_gp_bo", "scripted_chemistry")
            ),
            "require random plus at least two informative baselines",
        ),
        TaskDesignCheck(
            "generalization_axes",
            len(design.generalization_axes) >= 2,
            "at least two explicit generalization axes are required",
        ),
        TaskDesignCheck(
            "generalization_contract",
            tuple(item.get("label") for item in generalization_contract)
            == design.generalization_axes
            and all(item.get("hidden_drivers") for item in generalization_contract)
            and all(item.get("evaluation") for item in generalization_contract),
            "generalization axes require hidden drivers and an executable audit mode",
        ),
        TaskDesignCheck(
            "generalization_runtime_binding",
            runtime_axis_binding_valid,
            "generalization contracts must bind every executable world axis exactly once "
            "with matching labels, target kinds, and runtime target keys; "
            f"declared={list(declared_axis_ids)}, "
            f"executable={list(executable_axes_by_id)}",
        ),
        TaskDesignCheck(
            "evidence_plan",
            len(design.required_evidence) >= 3,
            "task design requires baseline, replay, and failure-analysis evidence",
        ),
        TaskDesignCheck(
            "anti_gaming",
            len(design.anti_gaming_checks) >= 2,
            "at least two anti-gaming checks are required",
        ),
    )
    return TaskDesignReview(
        task.task_id,
        checks,
        design.status if empirical_status is None else empirical_status,
    )


_COMMON_BASELINES = (
    "random",
    "lhs",
    "gp_bo",
    "structured_gp_bo",
    "structured_safe_gp_bo",
)
_COMMON_EVIDENCE = (
    "multi-seed baseline confidence intervals",
    "deterministic replay verification",
    "per-task failure and constraint analysis",
)
_COMMON_ANTI_GAMING = (
    "reject hidden-state access and contract-hash drift",
    "score final-assay results separately from intermediate observations",
)

SERIOUS_TASK_DESIGNS: dict[str, SeriousTaskDesign] = {
    "partition-discovery": SeriousTaskDesign(
        task_id="partition-discovery",
        research_question="Can an agent identify a hidden partition law with few contacts?",
        capability_claim="active phase-equilibrium characterization under a budget",
        primary_metric="product_in_organic",
        secondary_metrics=("phase_ratio", "product_in_aqueous"),
        generalization_axes=_generalization_axis_labels("partition-discovery"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=_COMMON_ANTI_GAMING,
    ),
    "reaction-to-crystallization": SeriousTaskDesign(
        task_id="reaction-to-crystallization",
        research_question="Can an agent balance reaction quality, recovery, purity, and CSD?",
        capability_claim="closed-loop reaction and cooling-crystallization planning",
        primary_metric="crystal_yield",
        secondary_metrics=(
            "score",
            "crystal_purity",
            "crystal_size",
            "crystal_csd_quality",
            "crystal_fines_fraction",
        ),
        generalization_axes=_generalization_axis_labels("reaction-to-crystallization"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=(
            *_COMMON_ANTI_GAMING,
            "ordered reaction-assay, seeded-cooling, slurry-assay, and isolation milestones",
            "cached observations cannot reward non-measurement operations",
        ),
    ),
    "reaction-to-distillation": SeriousTaskDesign(
        task_id="reaction-to-distillation",
        research_question="Can an agent choose reaction and cut conditions jointly?",
        capability_claim="reaction-distillation trade-off reasoning",
        primary_metric="distillate_purity",
        secondary_metrics=("score", "distillate_recovery", "solvent_loss"),
        generalization_axes=_generalization_axis_labels("reaction-to-distillation"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=_COMMON_ANTI_GAMING,
    ),
    "flow-reaction-optimization": SeriousTaskDesign(
        task_id="flow-reaction-optimization",
        research_question="Can an agent optimize conversion without ignoring flow risk?",
        capability_claim="geometry-aware continuous-flow optimization",
        primary_metric="flow_conversion",
        secondary_metrics=("score", "yield", "safety_risk"),
        generalization_axes=_generalization_axis_labels("flow-reaction-optimization"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=_COMMON_ANTI_GAMING,
    ),
    "electrochemical-conversion": SeriousTaskDesign(
        task_id="electrochemical-conversion",
        research_question=(
            "Can an agent use electrolyte and performance diagnostics to adapt a second "
            "electrochemical regime under transport, resistance, and energy constraints?"
        ),
        capability_claim="diagnosis-conditioned electrochemical identification and control",
        primary_metric="selective_product_yield",
        secondary_metrics=(
            "score",
            "electrochemical_selectivity",
            "faradaic_efficiency",
            "transport_efficiency",
            "ohmic_efficiency",
            "energy_efficiency",
            "pH_normalized",
            "precipitation_signal",
            "safety_risk",
        ),
        generalization_axes=_generalization_axis_labels("electrochemical-conversion"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=(
            *_COMMON_ANTI_GAMING,
            (
                "second setpoint must change by at least 0.02 V or 1.0 mA after "
                "pH and performance diagnostics"
            ),
            "cached observations cannot reward non-measurement operations",
        ),
    ),
    "equilibrium-characterization": SeriousTaskDesign(
        task_id="equilibrium-characterization",
        research_question="Can an agent characterize hidden aqueous equilibrium efficiently?",
        capability_claim="instrument-budgeted equilibrium identification",
        primary_metric="equilibrium_confidence",
        secondary_metrics=(
            "pH_normalized",
            "acid_dissociation_fraction",
            "precipitation_signal",
            "equilibrium_residual",
        ),
        generalization_axes=_generalization_axis_labels("equilibrium-characterization"),
        required_baselines=_COMMON_BASELINES,
        required_evidence=_COMMON_EVIDENCE,
        anti_gaming_checks=_COMMON_ANTI_GAMING,
    ),
}


def serious_task_readiness_manifest() -> dict[str, Any]:
    from chemworld.eval.benchmark_validation import official_empirical_statuses

    empirical_statuses = official_empirical_statuses()
    reviews = {
        task_id: review_task_design(
            get_task(task_id),
            SERIOUS_TASK_DESIGNS[task_id],
            empirical_status=empirical_statuses[task_id],
        )
        for task_id in SERIOUS_TASK_IDS
    }
    benchmark_ready_count = sum(review.benchmark_ready for review in reviews.values())
    return {
        "schema_version": TASK_DESIGN_VERSION,
        "suite_status": (
            "validated" if benchmark_ready_count == len(SERIOUS_TASK_IDS) else "candidate"
        ),
        "task_ids": list(SERIOUS_TASK_IDS),
        "contract_ready_count": sum(review.contract_ready for review in reviews.values()),
        "benchmark_ready_count": benchmark_ready_count,
        "designs": {
            task_id: {
                **SERIOUS_TASK_DESIGNS[task_id].to_dict(),
                "status": reviews[task_id].empirical_status,
                "generalization_contract": [
                    {
                        **axis,
                        "hidden_drivers": list(axis["hidden_drivers"]),
                        "runtime_target_keys": list(axis["runtime_target_keys"]),
                    }
                    for axis in SERIOUS_GENERALIZATION_CONTRACTS[task_id]
                ],
            }
            for task_id in SERIOUS_TASK_IDS
        },
        "reviews": {task_id: reviews[task_id].to_dict() for task_id in SERIOUS_TASK_IDS},
    }


__all__ = [
    "EXECUTABLE_EVALUATION_METRICS",
    "SERIOUS_GENERALIZATION_CONTRACTS",
    "SERIOUS_TASK_DESIGNS",
    "TASK_DESIGN_VERSION",
    "SeriousTaskDesign",
    "TaskDesignCheck",
    "TaskDesignReview",
    "review_task_design",
    "serious_task_readiness_manifest",
]
