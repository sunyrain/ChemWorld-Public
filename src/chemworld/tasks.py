"""Formal benchmark task registry for the unified ChemWorld."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModuleMaturity,
    TaskMaturitySpec,
    validate_task_maturity_policy,
)
from chemworld.registration import ENV_ID
from chemworld.world.operations import (
    CRYSTALLIZATION_OPERATIONS,
    DISTILLATION_OPERATIONS,
    ELECTROCHEMISTRY_OPERATIONS,
    FLOW_OPERATIONS,
    INSTRUMENTS,
    REACTION_OPERATIONS,
    SEPARATION_OPERATIONS,
)
from chemworld.world.parameters import WORLD_FAMILY_VERSION
from chemworld.world.scenario import get_scenario_card

WORLD_LAW_ID = WORLD_FAMILY_VERSION
TASK_CONTRACT_VERSION = "chemworld-task-contract-1.1"
CONFIRMATORY_BENCHMARK_TASK_IDS = (
    "reaction-to-crystallization",
    "electrochemical-conversion",
)
# Compatibility alias for pre-RC24 callers. Public documentation and new code use
# ``CONFIRMATORY_BENCHMARK_TASK_IDS`` so these tasks are not confused with the
# platform-breadth Showcase Worlds.
FLAGSHIP_TASK_IDS = CONFIRMATORY_BENCHMARK_TASK_IDS
CORE_TASK_IDS = (
    "reaction-to-assay",
    "reaction-to-purification",
    "partition-discovery",
)
SERIOUS_TASK_IDS = (
    "partition-discovery",
    "reaction-to-crystallization",
    "reaction-to-distillation",
    "flow-reaction-optimization",
    "electrochemical-conversion",
    "equilibrium-characterization",
)
STANDARD_INSTRUMENTS = tuple(instrument for instrument in INSTRUMENTS if instrument != "ph_meter")
REACTION_ALLOWED = REACTION_OPERATIONS
REACTION_SEPARATION_ALLOWED = (*REACTION_OPERATIONS, *SEPARATION_OPERATIONS)
REACTION_CRYSTALLIZATION_ALLOWED = (*REACTION_OPERATIONS, *CRYSTALLIZATION_OPERATIONS)
REACTION_DISTILLATION_ALLOWED = (*REACTION_OPERATIONS, *DISTILLATION_OPERATIONS)
FLOW_REACTION_ALLOWED = (
    "add_solvent",
    "add_reagent",
    "add_catalyst",
    *FLOW_OPERATIONS,
    "measure",
    "terminate",
)
ELECTROCHEMISTRY_ALLOWED = (
    "add_solvent",
    "add_reagent",
    *ELECTROCHEMISTRY_OPERATIONS,
    "measure",
    "terminate",
)
PARTITION_ALLOWED = (
    "add_solvent",
    "add_reagent",
    "add_phase",
    "add_extractant",
    "mix",
    "settle",
    "separate_phase",
    "measure",
    "terminate",
)
REFERENCE_BASELINES = (
    "random",
    "lhs",
    "scripted_chemistry",
    "gp_bo",
    "safe_gp_bo",
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_id: str
    world_law_id: str
    scenario_id: str
    initial_state_id: str
    world_split: str
    objective: str
    budget: int
    seeds: tuple[int, ...]
    threshold: float
    episode_mode: str
    allowed_operations: tuple[str, ...]
    allowed_instruments: tuple[str, ...]
    observation_policy: str
    termination_policy: str
    success_metrics: tuple[str, ...]
    safety_limit: float
    difficulty: str
    description: str
    tags: tuple[str, ...]
    kernel_maturity: TaskMaturitySpec
    domain_modules: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_contract_version": TASK_CONTRACT_VERSION,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "world_law_id": self.world_law_id,
            "scenario_id": self.scenario_id,
            "initial_state_id": self.initial_state_id,
            "world_split": self.world_split,
            "objective": self.objective,
            "budget": self.budget,
            "seeds": list(self.seeds),
            "threshold": self.threshold,
            "objective_score_threshold": self.threshold,
            "threshold_semantics": "leaderboard_objective_score_not_primary_metric",
            "episode_mode": self.episode_mode,
            "allowed_operations": list(self.allowed_operations),
            "allowed_instruments": list(self.allowed_instruments),
            "observation_policy": self.observation_policy,
            "termination_policy": self.termination_policy,
            "success_metrics": list(self.success_metrics),
            "safety_limit": self.safety_limit,
            "difficulty": self.difficulty,
            "description": self.description,
            "tags": list(self.tags),
            "kernel_maturity": self.kernel_maturity.to_dict(),
            "physics_maturity": self.kernel_maturity.lowest_level.value,
            "proxy_allowed": self.kernel_maturity.proxy_allowed,
        }
        payload["contract_hash"] = self.contract_hash
        return payload

    @property
    def contract_hash(self) -> str:
        payload = {
            "task_contract_version": TASK_CONTRACT_VERSION,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "world_law_id": self.world_law_id,
            "scenario_id": self.scenario_id,
            "initial_state_id": self.initial_state_id,
            "world_split": self.world_split,
            "objective": self.objective,
            "budget": self.budget,
            "seeds": list(self.seeds),
            "threshold": self.threshold,
            "objective_score_threshold": self.threshold,
            "threshold_semantics": "leaderboard_objective_score_not_primary_metric",
            "episode_mode": self.episode_mode,
            "allowed_operations": list(self.allowed_operations),
            "allowed_instruments": list(self.allowed_instruments),
            "observation_policy": self.observation_policy,
            "termination_policy": self.termination_policy,
            "success_metrics": list(self.success_metrics),
            "safety_limit": self.safety_limit,
            "difficulty": self.difficulty,
            "description": self.description,
            "tags": list(self.tags),
            "kernel_maturity": self.kernel_maturity.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        validate_task_maturity_policy(
            task_id=self.task_id,
            tags=self.tags,
            maturity=self.kernel_maturity,
        )
        from chemworld.schemas.validation import validate_task_schema

        schema_result = validate_task_schema(self.to_dict(), modules=self.domain_modules)
        if not schema_result.valid:
            raise ValueError(
                f"invalid task contract for {self.task_id!r}: " + "; ".join(schema_result.errors)
            )

    def env_kwargs(self, *, seed: int | None = None) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "world_split": self.world_split,
            "budget": self.budget,
            "objective": self.objective,
            "seed": self.seeds[0] if seed is None else seed,
        }

    def to_card(self) -> dict[str, Any]:
        scenario_card = get_scenario_card(self.scenario_id, split=self.world_split)
        suite_memberships = [
            suite
            for suite, task_ids in (
                ("core", CORE_TASK_IDS),
                ("serious", SERIOUS_TASK_IDS),
            )
            if self.task_id in task_ids
        ]
        return {
            "task_contract_version": TASK_CONTRACT_VERSION,
            "task_id": self.task_id,
            "task_contract_hash": self.contract_hash,
            "release_status": (
                "serious-benchmark-v1"
                if self.task_id in SERIOUS_TASK_IDS
                else ("core" if self.task_id in CORE_TASK_IDS else "registered-task")
            ),
            "suite_memberships": suite_memberships,
            "scientific_motivation": self.description,
            "world_law_id": self.world_law_id,
            "scenario_id": self.scenario_id,
            "allowed_operations": list(self.allowed_operations),
            "allowed_instruments": list(self.allowed_instruments),
            "budget": self.budget,
            "episode_mode": self.episode_mode,
            "reward_leaderboard_metric": {
                "online_reward": (
                    "fresh instrument-measurement score delta; cached observations receive zero"
                ),
                "leaderboard_score": "final-assay score only",
                "success_metrics": list(self.success_metrics),
                "objective_score_threshold": self.threshold,
                "threshold_semantics": "leaderboard_objective_score_not_primary_metric",
            },
            "benchmark_contract": {
                "objective": self.objective,
                "budget": self.budget,
                "episode_mode": self.episode_mode,
                "world_split": self.world_split,
                "public_seeds": list(self.seeds)
                if self.world_split != "private-eval"
                else "maintainer-controlled",
                "safety_limit": self.safety_limit,
                "success_metrics": list(self.success_metrics),
                "allowed_operations": list(self.allowed_operations),
                "allowed_instruments": list(self.allowed_instruments),
            },
            "observation_policy": self.observation_policy,
            "termination_policy": self.termination_policy,
            "public_seeds": list(self.seeds)
            if self.world_split != "private-eval"
            else "maintainer-controlled",
            "private_seeds": "maintainer-controlled",
            "baseline_reference_scores": dict.fromkeys(REFERENCE_BASELINES),
            "reference_baselines": list(REFERENCE_BASELINES),
            "recommended_agent_families": self._recommended_agent_families(),
            "scenario_card": scenario_card,
            "expected_qualitative_behavior": scenario_card["expected_qualitative_behavior"],
            "kernel_maturity": self.kernel_maturity.to_dict(),
            "physics_maturity": self.kernel_maturity.lowest_level.value,
            "proxy_allowed": self.kernel_maturity.proxy_allowed,
            "failure_modes": self._failure_modes(),
            "recommended_use": self._recommended_use(),
            "safety_limit": self.safety_limit,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
        }

    def _failure_modes(self) -> list[str]:
        modes = ["invalid operation preconditions", "unsafe or high-cost trajectory"]
        if "separation" in self.tags or "purification" in self.tags:
            modes.extend(["poor phase split", "low recovery", "mass-balance drift"])
        if "explanation" in self.tags or "mechanism" in self.tags:
            modes.append("weak or unsupported mechanism explanation")
        if "private-eval" in self.tags:
            modes.append("public-world overfitting")
        return modes

    def _recommended_use(self) -> list[str]:
        use = ["benchmark"]
        if "smoke" in self.tags:
            use.append("teaching")
        if "llm-agent" in self.tags or "explanation" in self.tags:
            use.append("LLM-agent")
        if "optimization" in self.tags or "sample-efficiency" in self.tags:
            use.extend(["BO", "RL"])
        return sorted(set(use))

    def _recommended_agent_families(self) -> list[str]:
        families = ["random", "scripted"]
        if "optimization" in self.tags or "sample-efficiency" in self.tags:
            families.extend(["lhs", "gp_bo", "safe_gp_bo"])
        if "separation" in self.tags or "purification" in self.tags:
            families.append("scripted_reaction_to_purification")
        if "llm-agent" in self.tags or "explanation" in self.tags:
            families.extend(["llm_replay", "tool_using_llm_stub"])
        return sorted(set(families))


def _task(
    task_id: str,
    *,
    scenario_id: str,
    world_split: str,
    objective: str = "balanced",
    budget: int,
    seeds: tuple[int, ...],
    threshold: float,
    episode_mode: str = "single_experiment",
    allowed_operations: tuple[str, ...],
    success_metrics: tuple[str, ...],
    safety_limit: float = 0.65,
    difficulty: str = "standard",
    description: str,
    tags: tuple[str, ...],
    instruments: tuple[str, ...] = STANDARD_INSTRUMENTS,
    observation_policy: str = "partial-instrument-observation",
    termination_policy: str | None = None,
    kernel_maturity: TaskMaturitySpec | None = None,
) -> TaskSpec:
    resolved_termination_policy = (
        termination_policy
        if termination_policy is not None
        else ("budget" if episode_mode == "campaign" else "final-assay-or-budget")
    )
    resolved_kernel_maturity = kernel_maturity or default_kernel_maturity(
        allowed_operations,
        allowed_instruments=instruments,
    )
    resolved_tags = ("chemworld", *tags)
    if resolved_kernel_maturity.contains_proxy and not set(resolved_tags).intersection(
        {"teaching", "smoke", "exploratory"}
    ):
        resolved_tags = (*resolved_tags, "exploratory")
    return TaskSpec(
        task_id=task_id,
        env_id=ENV_ID,
        world_law_id=WORLD_LAW_ID,
        scenario_id=scenario_id,
        initial_state_id=f"{scenario_id}:default",
        world_split=world_split,
        objective=objective,
        budget=budget,
        seeds=seeds,
        threshold=threshold,
        episode_mode=episode_mode,
        allowed_operations=allowed_operations,
        allowed_instruments=instruments,
        observation_policy=observation_policy,
        termination_policy=resolved_termination_policy,
        success_metrics=success_metrics,
        safety_limit=safety_limit,
        difficulty=difficulty,
        description=description,
        tags=resolved_tags,
        kernel_maturity=resolved_kernel_maturity,
    )


def default_kernel_maturity(
    allowed_operations: tuple[str, ...],
    *,
    allowed_instruments: tuple[str, ...] = STANDARD_INSTRUMENTS,
) -> TaskMaturitySpec:
    operations = set(allowed_operations)
    instruments = set(allowed_instruments)
    modules: list[ModuleMaturity] = []
    # These operation sets mirror the physical-model routes in
    # runtime.model_reachability. Ledger-only operations do not acquire a
    # physical maturity declaration merely because they share a task.
    if operations.intersection({"heat", "wait", "run_flow"}):
        modules.append(
            ModuleMaturity(
                "reaction_kinetics",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("reaction_ode_mass_action_arrhenius_reference_slice",),
                notes=(
                    "Reference-validated stoichiometric reaction-network and "
                    "Arrhenius runtime slice.",
                ),
            )
        )
    if operations.intersection({"heat", "wait"}):
        modules.append(
            ModuleMaturity(
                "reactors",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("dynamic_batch_heat_release_jacket_sampling",),
                notes=("Validated dynamic batch runtime for heat and wait.",),
            )
        )
    if "measure" in operations:
        instrument_model_ids = ["chemworld_validated_synthetic_instruments_v1"]
        if "uvvis" in instruments:
            instrument_model_ids.append("beer_lambert_uvvis")
        if instruments.intersection({"hplc", "gc"}):
            instrument_model_ids.append("chromatography_retention_plate")
        if "ph_meter" in instruments:
            if "electrolyze" not in operations:
                instrument_model_ids.append("aqueous_acid_base_ph_observation")
            instrument_model_ids.append("potentiometric_ph_public_reference")
        modules.append(
            ModuleMaturity(
                "spectroscopy_instruments",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=tuple(instrument_model_ids),
                notes=(
                    "State-coupled synthetic observations declare only the physical "
                    "instrument providers reachable from this task's allowed instruments; "
                    "they are not empirical spectral predictions.",
                ),
            )
        )
    if "mix" in operations:
        modules.append(
            ModuleMaturity(
                "phase_equilibrium",
                MaturityLevel.PROFESSIONAL_CANDIDATE,
                model_ids=("chemworld_stability_aware_lle_vnext",),
                notes=(
                    "Runtime phase contact uses the stability-gated, activity-corrected "
                    "extraction train with explicit entrainment and TPD-style diagnostics; "
                    "intrinsic distribution coefficients remain benchmark-calibrated.",
                ),
            )
        )
    if "cool_crystallize" in operations:
        modules.append(
            ModuleMaturity(
                "crystallization",
                MaturityLevel.PROFESSIONAL_CANDIDATE,
                model_ids=("cooling_crystallization_population_balance_v1",),
                notes=(
                    "Runtime cooling crystallization uses van't Hoff solubility, "
                    "explicit seed mass, nucleation/growth cohorts, impurity "
                    "occlusion, material closure, and CSD diagnostics.",
                ),
            )
        )
    if "distill" in operations:
        modules.append(
            ModuleMaturity(
                "distillation",
                MaturityLevel.PROFESSIONAL_CANDIDATE,
                model_ids=("chemworld_duty_limited_distillation_vnext",),
                notes=(
                    "Distillation uses a bubble-gated, equipment- and duty-limited "
                    "VLE/Fenske engine with explicit material and energy ledgers.",
                ),
            )
        )
    if "run_flow" in operations:
        modules.append(
            ModuleMaturity(
                "continuous_flow",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("chemworld_geometry_resolved_pfr_v2",),
                notes=(
                    "Runtime flow uses the shared compiled reaction network in a "
                    "geometry-resolved PFR with residence time, distributed thermal "
                    "boundary, Darcy pressure drop, solver diagnostics, and ledgers.",
                ),
            )
        )
    if "electrolyze" in operations:
        modules.append(
            ModuleMaturity(
                "electrochemistry",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=(
                    "nernst_butler_volmer_faradaic_v1",
                    "diffusion_layer_limiting_current_v1",
                    "randles_double_layer_transient_v1",
                ),
                notes=(
                    "Electrochemistry uses Nernst potential, Butler-Volmer current, "
                    "mass-transfer limits, Randles double-layer response, Faraday "
                    "charge accounting, and signed electrical-work ledgers.",
                ),
            )
        )
        modules.append(
            ModuleMaturity(
                "equilibrium_chemistry",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("aqueous_acid_base_ph_observation",),
                notes=(
                    "Electrolyte state uses weak-acid charge balance, bounded Davies "
                    "activities, and sequential Ksp hooks with fail-closed convergence.",
                ),
            )
        )
    if "wash" in operations:
        modules.append(
            ModuleMaturity(
                "extraction_wash",
                MaturityLevel.PROFESSIONAL_CANDIDATE,
                model_ids=("chemworld_stability_aware_lle_vnext",),
                notes=(
                    "Aqueous wash contacts use the same distribution, convergence, "
                    "entrainment, and material-balance contract as extraction stages.",
                ),
            )
        )
    downstream_models = {
        "dry": "chemworld_sorbent_drying_vnext",
        "concentrate": "chemworld_vacuum_concentration_vnext",
        "transfer": "chemworld_transfer_holdup_vnext",
    }
    reached_downstream = tuple(
        model_id for operation, model_id in downstream_models.items() if operation in operations
    )
    if reached_downstream:
        modules.append(
            ModuleMaturity(
                "separations",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=reached_downstream,
                notes=(
                    "Drying, vacuum concentration, and transfer use explicit finite-capacity "
                    "equipment, material, volume, and energy ledgers. Their bounded runtime "
                    "parameterization is not an industrial process-design claim.",
                ),
            )
        )
    if not modules:
        modules.append(
            ModuleMaturity(
                "ledger_operations",
                MaturityLevel.REFERENCE_VALIDATED,
                notes=(
                    "This operation set reaches only typed ledger/equipment "
                    "transitions and has no declared physical model provider.",
                ),
            )
        )
    proxy_allowed = any(module.level is MaturityLevel.PROXY for module in modules)
    notes = (
        "Maturity is computed from the exact runtime modules used by this task; "
        "professional-candidate does not imply industrial validation.",
    )
    return TaskMaturitySpec(
        modules=tuple(modules),
        proxy_allowed=proxy_allowed,
        notes=notes,
    )


def equilibrium_kernel_maturity() -> TaskMaturitySpec:
    """Return the D4 maturity contract for the equilibrium characterization task."""

    return TaskMaturitySpec(
        modules=(
            ModuleMaturity(
                "equilibrium_chemistry",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("aqueous_acid_base_ph_observation",),
                notes=(
                    "D4 equilibrium slice: weak-acid charge balance, public pH-meter "
                    "observation and sequential Ksp hooks. The fixed-T,P Gibbs solver "
                    "remains reference-only and is not a runtime task dependency.",
                ),
            ),
            ModuleMaturity(
                "reaction_kinetics",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("reaction_ode_mass_action_arrhenius_reference_slice",),
                notes=("Heat and wait use the validated reaction-network runtime slice.",),
            ),
            ModuleMaturity(
                "reactors",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=("dynamic_batch_heat_release_jacket_sampling",),
                notes=("Heat and wait use the validated dynamic batch runtime slice.",),
            ),
            ModuleMaturity(
                "spectroscopy_instruments",
                MaturityLevel.REFERENCE_VALIDATED,
                model_ids=(
                    "beer_lambert_uvvis",
                    "chemworld_validated_synthetic_instruments_v1",
                    "potentiometric_ph_public_reference",
                ),
                notes=(
                    "pH-meter and UV/Vis signals are instrument-facing and benchmark-calibrated.",
                ),
            ),
        ),
        proxy_allowed=False,
        notes=(
            "Equilibrium characterization is a bounded D4 task, not a general "
            "aqueous speciation engine.",
        ),
    )


TASK_REGISTRY: dict[str, TaskSpec] = {
    "reaction-optimization-standard": _task(
        "reaction-optimization-standard",
        scenario_id="reaction-optimization",
        world_split="public-test",
        budget=72,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.75,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("score", "yield", "selectivity", "sample_efficiency"),
        description="Primary reaction optimization task in the shared ChemWorld law.",
        tags=("reaction", "optimization", "public-test"),
    ),
    "reaction-safety-constrained": _task(
        "reaction-safety-constrained",
        scenario_id="reaction-safety",
        world_split="public-test",
        objective="safe",
        budget=72,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.70,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("score", "safety_risk", "constraint_violations"),
        safety_limit=0.35,
        description="Reaction optimization under a stricter safety-risk target.",
        tags=("reaction", "safety", "public-test"),
    ),
    "reaction-mechanism-explanation": _task(
        "reaction-mechanism-explanation",
        scenario_id="reaction-mechanism",
        world_split="public-test",
        budget=36,
        seeds=(0, 1, 2),
        threshold=0.68,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("score", "mechanism_explanation", "failure_analysis"),
        description="Optimize while producing structured hypotheses about hidden kinetics.",
        tags=("reaction", "explanation", "mechanism"),
    ),
    "reaction-to-assay": _task(
        "reaction-to-assay",
        scenario_id="reaction-to-assay",
        world_split="public-dev",
        budget=18,
        seeds=(0,),
        threshold=0.55,
        episode_mode="single_experiment",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("final_assay_score", "trajectory_validity"),
        difficulty="smoke",
        description="Short event-sequence task from charging the reactor to final assay.",
        tags=("reaction", "assay", "smoke"),
    ),
    "reaction-to-purification": _task(
        "reaction-to-purification",
        scenario_id="reaction-to-purification",
        world_split="public-test",
        budget=90,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.70,
        episode_mode="single_experiment",
        allowed_operations=REACTION_SEPARATION_ALLOWED,
        success_metrics=("score", "purity", "recovery", "process_mass_balance_error"),
        description="Closed-loop reaction, extraction, phase separation, purification, and assay.",
        tags=("reaction", "separation", "purification"),
    ),
    "reaction-to-crystallization": _task(
        "reaction-to-crystallization",
        scenario_id="reaction-to-crystallization",
        world_split="public-test",
        budget=72,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.55,
        episode_mode="campaign",
        allowed_operations=REACTION_CRYSTALLIZATION_ALLOWED,
        success_metrics=(
            "crystal_yield",
            "score",
            "crystal_purity",
            "crystal_size",
            "crystal_csd_quality",
            "crystal_fines_fraction",
        ),
        description=(
            "Use reaction assays to adapt a seeded cooling trajectory, then "
            "isolate product while controlling yield, purity, and particle-size distribution."
        ),
        instruments=("hplc", "final_assay"),
        termination_policy="budget-with-workflow-gated-final-assay",
        tags=("reaction", "crystallization", "purification", "closed-loop", "flagship"),
    ),
    "reaction-to-distillation": _task(
        "reaction-to-distillation",
        scenario_id="reaction-to-distillation",
        world_split="public-test",
        budget=72,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.29,
        episode_mode="campaign",
        allowed_operations=REACTION_DISTILLATION_ALLOWED,
        success_metrics=("score", "distillate_purity", "distillate_recovery", "solvent_loss"),
        description="Run a reaction and evaluate volatile-product recovery through distillation.",
        tags=("reaction", "distillation", "purification", "year2"),
    ),
    "flow-reaction-optimization": _task(
        "flow-reaction-optimization",
        scenario_id="flow-reaction-optimization",
        world_split="public-test",
        budget=60,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.18,
        episode_mode="campaign",
        allowed_operations=FLOW_REACTION_ALLOWED,
        success_metrics=("score", "flow_conversion", "yield", "safety_risk"),
        description="Optimize a geometry-resolved PFR using the shared reaction network.",
        tags=("reaction", "continuous-flow", "optimization", "year2"),
    ),
    "electrochemical-conversion": _task(
        "electrochemical-conversion",
        scenario_id="electrochemical-conversion",
        world_split="public-test",
        budget=48,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.58,
        episode_mode="campaign",
        allowed_operations=ELECTROCHEMISTRY_ALLOWED,
        success_metrics=(
            "selective_product_yield",
            "electrochemical_selectivity",
            "score",
            "faradaic_efficiency",
            "transport_efficiency",
            "ohmic_efficiency",
            "energy_efficiency",
            "pH_normalized",
            "precipitation_signal",
            "safety_risk",
        ),
        description=(
            "Select a bounded solvent medium and electrolyte profile, identify coupled "
            "effective equilibrium, transport, double-layer, kinetic, and ohmic behavior "
            "(the pH diagnostic is a normalized effective proton-activity index, not a "
            "literal non-aqueous pH claim), and optimize potential, current, and duration "
            "for selective, charge- and energy-efficient conversion. The "
            "public current_mA control is a nonnegative power-supply magnitude cap; the "
            "signed electrochemical current and reaction direction follow Butler-Volmer."
        ),
        instruments=("ph_meter", "uvvis", "final_assay"),
        termination_policy="budget-with-workflow-gated-final-assay",
        tags=("reaction", "electrochemistry", "optimization", "closed-loop", "flagship"),
    ),
    "equilibrium-characterization": _task(
        "equilibrium-characterization",
        scenario_id="equilibrium-characterization",
        world_split="public-test",
        budget=24,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.28,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        instruments=("ph_meter", "uvvis", "final_assay"),
        success_metrics=(
            "pH_normalized",
            "acid_dissociation_fraction",
            "precipitation_signal",
            "equilibrium_residual",
            "equilibrium_confidence",
        ),
        description=(
            "Characterize a bounded aqueous-equilibrium slice using public pH-meter "
            "and final-assay observations."
        ),
        tags=(
            "equilibrium",
            "characterization",
            "world-model-learning",
            "serious-benchmark-v1",
        ),
        kernel_maturity=equilibrium_kernel_maturity(),
    ),
    "partition-discovery": _task(
        "partition-discovery",
        scenario_id="partition-discovery",
        world_split="public-test",
        budget=48,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.58,
        episode_mode="campaign",
        allowed_operations=PARTITION_ALLOWED,
        success_metrics=("phase_ratio", "product_in_organic", "product_in_aqueous"),
        description="Learn unknown solvent/product partition behavior through instruments.",
        tags=("phase", "partition", "world-model-learning"),
    ),
    "purity-yield-tradeoff": _task(
        "purity-yield-tradeoff",
        scenario_id="purity-yield-tradeoff",
        world_split="public-test",
        budget=90,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.70,
        episode_mode="campaign",
        allowed_operations=REACTION_SEPARATION_ALLOWED,
        success_metrics=("yield", "purity", "recovery", "cost"),
        description="Optimize the downstream tradeoff between product purity, recovery, and cost.",
        tags=("separation", "multi-objective", "purification"),
    ),
    "public-private-generalization": _task(
        "public-private-generalization",
        scenario_id="generalization",
        world_split="private-eval",
        budget=72,
        seeds=(0, 1, 2, 3, 4),
        threshold=0.72,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("score", "public_private_gap", "rank_confidence"),
        description="Private-split task for evaluating overfitting to public worlds.",
        tags=("reaction", "private-eval", "generalization"),
    ),
    "low-budget-characterization": _task(
        "low-budget-characterization",
        scenario_id="low-budget-characterization",
        world_split="public-test",
        budget=18,
        seeds=(0, 1, 2),
        threshold=0.55,
        episode_mode="campaign",
        allowed_operations=REACTION_ALLOWED,
        success_metrics=("sample_efficiency", "uncertainty", "local_model_quality"),
        difficulty="hard",
        description="Build a useful local world model with very few measurements.",
        tags=("reaction", "sample-efficiency", "low-budget"),
    ),
    "tool-agent-planning": _task(
        "tool-agent-planning",
        scenario_id="tool-agent-planning",
        world_split="public-dev",
        budget=48,
        seeds=(0, 1),
        threshold=0.62,
        episode_mode="single_experiment",
        allowed_operations=REACTION_SEPARATION_ALLOWED,
        success_metrics=("trajectory_validity", "validator_use", "score", "explanation"),
        description=(
            "Task slice for LLM/tool agents that use validators, instruments, and surrogates."
        ),
        tags=("llm-agent", "tool-use", "planning"),
    ),
}


def list_tasks() -> list[TaskSpec]:
    return [TASK_REGISTRY[key] for key in sorted(TASK_REGISTRY)]


def list_task_cards() -> list[dict[str, Any]]:
    return [task.to_card() for task in list_tasks()]


def list_core_tasks() -> list[TaskSpec]:
    return [get_task(task_id) for task_id in CORE_TASK_IDS]


def list_core_task_cards() -> list[dict[str, Any]]:
    return [task.to_card() for task in list_core_tasks()]


def list_serious_tasks() -> list[TaskSpec]:
    return [get_task(task_id) for task_id in SERIOUS_TASK_IDS]


def list_serious_task_cards() -> list[dict[str, Any]]:
    return [task.to_card() for task in list_serious_tasks()]


def list_confirmatory_benchmark_tasks() -> list[TaskSpec]:
    return [get_task(task_id) for task_id in CONFIRMATORY_BENCHMARK_TASK_IDS]


def list_confirmatory_benchmark_task_cards() -> list[dict[str, Any]]:
    return [task.to_card() for task in list_confirmatory_benchmark_tasks()]


def list_flagship_tasks() -> list[TaskSpec]:
    """Compatibility alias for :func:`list_confirmatory_benchmark_tasks`."""

    return list_confirmatory_benchmark_tasks()


def list_flagship_task_cards() -> list[dict[str, Any]]:
    """Compatibility alias for :func:`list_confirmatory_benchmark_task_cards`."""

    return list_confirmatory_benchmark_task_cards()


def task_maturity_manifest(task_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Return a JSON-friendly maturity manifest for benchmark tasks."""

    tasks = list_tasks() if task_ids is None else [get_task(task_id) for task_id in task_ids]
    by_task: dict[str, dict[str, Any]] = {}
    by_level: dict[str, dict[str, Any]] = {}
    proxy_allowed_task_ids: list[str] = []
    for task in tasks:
        payload = task.to_dict()
        maturity_payload = {
            "kernel_maturity": payload["kernel_maturity"],
            "physics_maturity": payload["physics_maturity"],
            "proxy_allowed": payload["proxy_allowed"],
            "world_split": payload["world_split"],
            "episode_mode": payload["episode_mode"],
            "tags": payload["tags"],
        }
        by_task[task.task_id] = maturity_payload
        level = str(payload["physics_maturity"])
        level_entry = by_level.setdefault(
            level,
            {"task_ids": [], "proxy_allowed_task_ids": []},
        )
        level_entry["task_ids"].append(task.task_id)
        if payload["proxy_allowed"]:
            proxy_allowed_task_ids.append(task.task_id)
            level_entry["proxy_allowed_task_ids"].append(task.task_id)

    for entry in by_level.values():
        entry["task_ids"] = sorted(entry["task_ids"])
        entry["proxy_allowed_task_ids"] = sorted(entry["proxy_allowed_task_ids"])
        entry["task_count"] = len(entry["task_ids"])

    return {
        "schema_version": "chemworld-task-maturity-manifest-0.1",
        "task_count": len(tasks),
        "by_task": by_task,
        "by_physics_maturity": dict(sorted(by_level.items())),
        "proxy_allowed_task_ids": sorted(proxy_allowed_task_ids),
    }


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[task_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task_id={task_id!r}. Allowed: {allowed}") from exc


def get_task_card(task_id: str) -> dict[str, Any]:
    return get_task(task_id).to_card()
