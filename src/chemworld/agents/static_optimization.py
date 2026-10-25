"""Static scientific optimization agent for explicit fixed-world protocols."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, ClassVar

import numpy as np

from chemworld.agents.crystallization_single_stage import (
    CRYSTALLIZATION_SINGLE_STAGE_CATEGORICAL_COORDINATES,
    CRYSTALLIZATION_SINGLE_STAGE_DIMENSION,
    CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS,
    CRYSTALLIZATION_SINGLE_STAGE_RECIPE_VERSION,
    crystallization_single_stage_parameter_schema,
    crystallization_single_stage_parameters_from_unit_vector,
    crystallization_single_stage_recipe_from_unit_vector,
    crystallization_single_stage_unit_vector_from_parameters,
)
from chemworld.agents.electrochemical_single_stage import (
    ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES,
    ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION,
    ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS,
    ELECTROCHEMICAL_SINGLE_STAGE_RECIPE_VERSION,
    electrochemical_single_stage_parameter_schema,
    electrochemical_single_stage_parameters_from_unit_vector,
    electrochemical_single_stage_recipe_from_unit_vector,
    electrochemical_single_stage_unit_vector_from_parameters,
)
from chemworld.agents.prompt_context import PromptBudgetExceededError, estimate_prompt_tokens
from chemworld.agents.scientific_adaptation import (
    ResourceLedger,
    ScientificPlanValidationError,
    canonical_sha256,
    required_scientific_measurement_slot_ids,
    scientific_measurement_slots,
)
from chemworld.agents.static_candidate_portfolio import (
    public_surrogate_candidate_portfolio,
)
from chemworld.agents.task_recipes import (
    TASK_RECIPE_SPACE_VERSION,
    electrochemical_recipe_parameter_schema,
    electrochemical_recipe_parameters_from_unit_vector,
    electrochemical_recipe_unit_vector_from_parameters,
    task_recipe_categorical_coordinates,
    task_recipe_coordinate_schema,
    task_recipe_dimension,
    task_recipe_from_unit_vector,
    task_recipe_kind,
    task_recipe_public_controls,
)
from chemworld.data.logging import to_builtin
from chemworld.eval.crystallization_predictive import (
    CrystallizationPredictionQuery,
    build_crystallization_prediction_queries,
)
from chemworld.eval.electrochemical_predictive import (
    ElectrochemicalPredictionQuery,
    build_electrochemical_prediction_queries,
    parse_counterfactual_predictions,
)
from chemworld.eval.world_understanding import (
    parse_world_understanding_claims,
    parse_world_understanding_claims_tolerant,
)
from chemworld.legacy.agents.live_llm import JsonPlannerClientLike
from chemworld.materials import (
    normalize_static_material_information_config,
    static_material_information_dossier,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
    normalize_electrochemical_workflow_mode,
)
from chemworld.world.scoring import DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2

STATIC_OPTIMIZATION_INTERFACE_VERSION = "chemworld-static-optimization-interface-0.3-s0-dev"
STATIC_OPTIMIZATION_PROMPT_VERSION = "chemworld-static-optimization-prompt-0.3-s0-dev"
STATIC_OPTIMIZATION_COVERAGE_PROMPT_VERSION = (
    "chemworld-static-optimization-prompt-0.5-s0-dev"
)
STATIC_OPTIMIZATION_PORTFOLIO_PROMPT_VERSION = (
    "chemworld-static-optimization-prompt-0.6-s0-dev"
)
STATIC_OPTIMIZATION_NONDUPLICATE_PORTFOLIO_PROMPT_VERSION = (
    "chemworld-static-optimization-prompt-0.7-s0-dev"
)
STATIC_OPTIMIZATION_SCHEDULED_PORTFOLIO_PROMPT_VERSION = (
    "chemworld-static-optimization-prompt-0.8-s0-dev"
)
STATIC_FINAL_SYNTHESIS_VERSION = "chemworld-static-final-synthesis-0.3-s0-dev"
STATIC_FINAL_SYNTHESIS_SEPARATE_VERSION = "chemworld-static-final-synthesis-0.4-s0-dev"
STATIC_FINAL_SYNTHESIS_TOLERANT_DECLARED_VERSION = "chemworld-static-final-synthesis-0.5-s0-dev"
STATIC_PREDICTIVE_SYNTHESIS_VERSION = "chemworld-static-predictive-synthesis-0.1-s0-dev"


DECLARED_CLAIM_VALIDATION_STRICT = "strict"
DECLARED_CLAIM_VALIDATION_UNSCORED_UNKNOWN = "unscored_unknown_terms"
DECLARED_CLAIM_VALIDATION_POLICIES = frozenset(
    {
        DECLARED_CLAIM_VALIDATION_STRICT,
        DECLARED_CLAIM_VALIDATION_UNSCORED_UNKNOWN,
    }
)
DIRECT_OPTIMIZATION_SCAFFOLD_ID = (
    "direct_known_horizon_five_task_public_contract_full_history_v15"
)
COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V16_ID = (
    "coverage_then_adaptive_five_task_public_contract_full_history_v16"
)
COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V17_ID = (
    "coverage_surrogate_portfolio_five_task_public_contract_full_history_v17"
)
COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID = (
    "coverage_surrogate_portfolio_nonduplicate_five_task_public_contract_full_history_v18"
)
COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID = (
    "coverage_scheduled_public_portfolio_five_task_public_contract_full_history_v19"
)
COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_IDS = frozenset(
    {
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V16_ID,
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V17_ID,
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID,
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID,
    }
)
SURROGATE_PORTFOLIO_OPTIMIZATION_SCAFFOLD_IDS = frozenset(
    {
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V17_ID,
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID,
        COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID,
    }
)
COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS = 8
SURROGATE_PORTFOLIO_V17_FINAL_EXPERIMENT_INDEX = 16
SURROGATE_PORTFOLIO_NONDUPLICATE_FINAL_EXPERIMENT_INDEX = 19

_ELECTROCHEMICAL_CLAIM_CAUSES = (
    "controlled_potential_V",
    "controlled_current_mA",
    "controlled_duration_s",
    "electrolyte_profile",
    "solvent",
    "reagent_amount_mol",
)
_ELECTROCHEMICAL_SINGLE_STAGE_CLAIM_CAUSES = (
    "potential_V",
    "current_mA",
    "duration_s",
    "electrolyte_profile",
    "solvent",
    "reagent_amount_mol",
)
_ELECTROCHEMICAL_CLAIM_EFFECTS = (
    "selective_product_yield",
    "electrochemical_conversion",
    "electrochemical_selectivity",
    "energy_efficiency",
    "faradaic_efficiency",
    "transport_efficiency",
    "ohmic_efficiency",
    "pH_normalized",
    "precipitation_signal",
    "leaderboard_score",
)
_ELECTROCHEMICAL_MECHANISM_TAGS = (
    "nernst_equilibrium",
    "butler_volmer_kinetics",
    "current_magnitude_cap",
    "faraday_charge",
    "double_layer_charging",
    "mass_transfer_limit",
    "ohmic_loss",
    "overpotential_stress",
    "acid_base_activity",
    "precipitation_equilibrium",
    "categorical_medium_effect",
    "inventory_limit",
    "direction_reversal",
)
_CRYSTALLIZATION_CLAIM_CAUSES = (
    "reaction_temperature_K",
    "reaction_duration_s",
    "reagent_amount_mol",
    "stirring_speed_rpm",
    "catalyst",
    "catalyst_amount_mol",
    "solvent",
    "seed_mass_g",
    "crystallization_temperature_K",
    "crystallization_duration_s",
)
_CRYSTALLIZATION_CLAIM_EFFECTS = (
    "reaction_score",
    "yield",
    "selectivity",
    "crystal_yield",
    "crystal_purity",
    "crystal_size",
    "crystal_csd_quality",
    "crystal_fines_fraction",
    "leaderboard_score",
)
_CRYSTALLIZATION_MECHANISM_TAGS = (
    "arrhenius_kinetics",
    "catalyst_activity",
    "categorical_solvent_effect",
    "inventory_limit",
    "vanthoff_solubility",
    "supersaturation",
    "primary_nucleation",
    "seeded_growth",
    "impurity_occlusion",
    "cooling_rate",
    "filtration_recovery",
    "yield_purity_size_tradeoff",
)

SYSTEM_PROMPT = """You are a static scientific optimization agent in ChemWorld.
The world is fixed for the entire campaign. Optimize the task objective using only the
public task contract and public experiment history. Choose one complete experiment and
return exactly one JSON object without exposing private chain-of-thought. When named
physical recipe parameters are supplied, reason and report in those parameters and units;
do not invent or request hidden normalized coordinates.
"""

COVERAGE_ADAPTIVE_SYSTEM_PROMPT = """You are a static scientific optimization
agent in ChemWorld. The world is fixed for the entire campaign. Optimize the task
objective using only the public task contract, public experiment history, and public
campaign_scaffold. During the first eight experiments, the protocol executor commits the
balanced-design recipe shown in campaign_scaffold; use that condition to choose diagnostics,
state the experiment's purpose, and predict its effect, but do not return recipe fields.
After those eight experiments, choose the complete recipe yourself. Never treat nominal
category codes as ordered or transferable across controls; compare completed experiments
by terminal leaderboard_score; and distinguish deliberate replication from accidental
duplication. Return exactly one JSON object without exposing private chain-of-thought.
When named physical recipe parameters are supplied, reason and report in those parameters
and units; do not invent or request hidden normalized coordinates.
"""

SURROGATE_PORTFOLIO_SYSTEM_PROMPT = """You are a static scientific optimization
agent in ChemWorld. The world is fixed for the entire campaign. Optimize the task
objective using only the public task contract, public experiment history, and public
campaign_scaffold. During the first eight experiments, the protocol executor commits the
balanced-design recipe shown in campaign_scaffold; use that condition to choose diagnostics,
state the experiment's purpose, and predict its effect, but do not return recipe fields.
During experiments 9 through 17, select exactly one candidate_id from the displayed
public-history-only candidate portfolio and return that candidate's exact complete recipe.
The portfolio supplies task-neutral numerical search discipline; you remain responsible for
choosing among acquisition, global-coverage, uncertainty, and boundary candidates using the
public scientific evidence. Follow campaign_scaffold.task_neutral_default_candidate_id
unless you can identify specific completed public experiments or evidence IDs that justify
another displayed candidate; explain any deviation in experiment_intent. During experiments
18 through 20, choose the complete recipe freely for convergence or deliberate replication.
Never treat nominal category codes as
ordered or transferable across controls. The initial space-filling design is confounded, so
do not infer a one-control direction merely by comparing two unrelated initial recipes.
Do not spend the discrimination phase on tiny sequential changes or exact duplicates; three
blind validation replicates occur after the final recommendation. Compare completed
experiments by terminal leaderboard_score and distinguish deliberate replication from
accidental duplication. Return exactly one JSON object without exposing private
chain-of-thought. When named physical recipe parameters are supplied, reason and report in
those parameters and units; do not invent or request hidden normalized coordinates.
"""

NONDUPLICATE_SURROGATE_PORTFOLIO_SYSTEM_PROMPT = """You are a static scientific
optimization agent in ChemWorld. The world is fixed for the entire campaign. Optimize the
task objective using only the public task contract, public experiment history, and public
campaign_scaffold. During the first eight experiments, the protocol executor commits the
balanced-design recipe shown in campaign_scaffold; use that condition to choose diagnostics,
state the experiment's purpose, and predict its effect, but do not return recipe fields.
During experiments 9 through 20, select exactly one candidate_id from the displayed
public-history-only candidate portfolio and return that candidate's exact complete recipe.
The portfolio supplies task-neutral numerical search discipline and excludes completed
recipes; you remain responsible for choosing among acquisition, global-coverage,
uncertainty, and boundary candidates using the public scientific evidence. Follow
campaign_scaffold.task_neutral_default_candidate_id unless specific completed public
experiments or evidence IDs justify another displayed candidate, and explain any deviation
in experiment_intent. Never treat nominal category codes as ordered or transferable across
controls. The initial space-filling design is confounded, so do not infer a one-control
direction merely by comparing two unrelated initial recipes. Inspect the signed scoring
weights and stage-resolved measurements, not only terminal leaderboard_score. A strong
aggregate score with a positively weighted component near its floor, or a negatively
weighted component near its ceiling, is an unresolved scientific bottleneck. Use remaining
experiments to challenge that bottleneck or a credible competing basin. Do not repeat a
completed recipe: three blind validation replicates occur after the final recommendation
and provide the replication evidence. Return exactly one JSON object without exposing
private chain-of-thought. When named physical recipe parameters are supplied, reason and
report in those parameters and units; do not invent or request hidden normalized
coordinates.
"""

SCHEDULED_SURROGATE_PORTFOLIO_SYSTEM_PROMPT = """You are the scientific-analysis
component of a static hybrid optimization agent in ChemWorld. The world is fixed for the
entire campaign. Use only the public task contract, public experiment history, and public
campaign_scaffold. For every experiment, the protocol executor commits the displayed
task-neutral recipe. During experiments 1 through 8 it uses balanced coverage; during
experiments 9 through 14 it commits the public-history maximin candidate; and during
experiments 15 through 20 it commits the public-history boundary-challenge candidate. Use
the model call to choose diagnostics, state the scientific purpose of the committed recipe,
and predict its effect, but do not return recipe fields or a candidate_id. Never treat
nominal category codes as ordered or transferable across controls. The initial space-filling
design is confounded, so do not infer a one-control direction merely by comparing unrelated
recipes. Compare completed experiments using the frozen terminal leaderboard_score as the
primary objective; signed component weights and stage measurements diagnose why scores
change but do not replace that registered objective. Return exactly one JSON object without
exposing private chain-of-thought.
"""

FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are completing a fixed-world scientific
optimization campaign in ChemWorld. Exploration is finished. Submit one final experimental
method that represents your overall conclusion from the public evidence. The method may
reuse a tested condition, interpolate between tested conditions, or extrapolate within the
declared public search bounds. Return exactly one JSON object without exposing private
chain-of-thought. Ground the recommendation and working scientific explanation in public
experiment indices and evidence IDs. Separate empirical relations from mechanism claims,
and express structured claims only with the declared public vocabulary.
"""

COVERAGE_ADAPTIVE_FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are completing a
fixed-world scientific optimization campaign in ChemWorld. Exploration is finished.
Submit one final experimental method using only the public campaign evidence and public
campaign_scaffold. Prefer a tested condition supported by repeated or locally consistent
evidence over a single noisy maximum. Use blind validation only after the recommendation
is committed. Do not infer order or distance between nominal codes, and do not optimize
an audit-only metric. Return exactly one JSON object without exposing private
chain-of-thought. Ground the recommendation and working scientific explanation in public
experiment indices and evidence IDs, separate empirical relationships from mechanistic
hypotheses, and use only the declared public claim vocabulary.
"""

SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are completing a
fixed-world scientific optimization campaign in ChemWorld. Exploration is finished.
Submit one final experimental method using only the public campaign evidence and public
campaign_scaffold. Candidate portfolios were generated only from prior public outcomes;
their surrogate predictions are aids, not ground truth. Prefer a tested condition supported
by repeated or locally consistent evidence over a single noisy maximum, but do not discard
a clearly superior, scientifically coherent tested region merely because it has fewer
exploration duplicates: three blind validation replicates follow the committed
recommendation. Do not infer order or distance between nominal codes, and do not optimize
an audit-only metric. Return exactly one JSON object without exposing private
chain-of-thought. Ground the recommendation and working scientific explanation in public
experiment indices and evidence IDs, separate empirical relationships from mechanistic
hypotheses, and use only the declared public claim vocabulary.
"""

NONDUPLICATE_SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are
completing a fixed-world scientific optimization campaign in ChemWorld. Exploration is
finished. Submit one final experimental method using only the public campaign evidence and
public campaign_scaffold. Candidate portfolios were generated only from prior public
outcomes; their surrogate predictions are aids, not ground truth. Compare terminal scores,
signed scoring weights, and stage-resolved component measurements. Do not treat a high
aggregate score as robust when a positively weighted component remains near its floor or a
negatively weighted component remains near its ceiling. Prefer a tested condition supported
by local consistency and balanced component performance; a clearly superior coherent tested
region need not have campaign duplicates because three blind validation replicates follow
the committed recommendation. Do not infer order or distance between nominal codes, and do
not optimize an audit-only metric. Return exactly one JSON object without exposing private
chain-of-thought. Ground the recommendation and working scientific explanation in public
experiment indices and evidence IDs, separate empirical relationships from mechanistic
hypotheses, and use only the declared public claim vocabulary.
"""

SCHEDULED_SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are completing
a fixed-world hybrid scientific optimization campaign in ChemWorld. Exploration is
finished. Submit one final experimental method using only the public campaign evidence and
public campaign_scaffold. The frozen terminal leaderboard_score is the registered primary
objective and already encodes the signed component tradeoffs. Choose the tested condition
with the strongest expected blind-validation score, using local consistency and measurement
uncertainty only to distinguish plausibly similar scores. Do not replace the registered
objective with an unregistered requirement that every component be individually balanced.
Three blind validation replicates follow the committed recommendation, so campaign
duplicates are not required. Do not infer order or distance between nominal codes, and do
not optimize an audit-only metric. Return exactly one JSON object without exposing private
chain-of-thought. Ground the recommendation and working scientific explanation in public
experiment indices and evidence IDs, separate empirical relationships from mechanistic
hypotheses, and use only the declared public claim vocabulary.
"""

FINAL_SYNTHESIS_SEPARATE_PREDICTIVE_SYSTEM_PROMPT = """You are completing a
fixed-world scientific optimization campaign in ChemWorld. Exploration is finished.
Submit one final experimental method and a working scientific explanation using only the
public campaign evidence. The final method will be committed before a later, separate
predictive diagnostic. Do not include counterfactual_predictions, predictive queries, or
answers to unseen interventions in this response. Return exactly the declared JSON fields
without exposing private chain-of-thought. Ground the recommendation in public experiment
indices and evidence IDs, and use only the declared public mechanism vocabulary.
"""

PREDICTIVE_SYNTHESIS_SYSTEM_PROMPT = """You are completing a held-out predictive
diagnostic for a fixed-world scientific campaign in ChemWorld. A final experimental method
has already been committed and cannot be changed. Use only the supplied public experiment
history to predict the direction of each pre-registered intervention. Return exactly one
JSON object without exposing private chain-of-thought. Do not recommend, revise, or rank
experimental methods; this call is prediction-only.
"""

_PUBLIC_TASK_KEYS = (
    "task_contract_version",
    "task_id",
    "objective",
    "budget",
    "allowed_operations",
    "allowed_instruments",
    "observation_policy",
    "termination_policy",
    "success_metrics",
    "safety_limit",
    "description",
    "contract_hash",
)
_PUBLIC_TERMINAL_KEYS = ("leaderboard_score", "cost", "safety_risk")


def _uses_single_stage_electrochemistry(task_info: Mapping[str, Any], workflow_mode: str) -> bool:
    return (
        task_recipe_kind(dict(task_info)) == "electrochemical"
        and workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    )


def _uses_named_physical_controls(task_info: Mapping[str, Any]) -> bool:
    return task_recipe_kind(dict(task_info)) in {
        "electrochemical",
        "reaction_crystallization",
    }


def _recipe_parameter_schema(
    task_info: Mapping[str, Any], workflow_mode: str
) -> dict[str, dict[str, Any]]:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return electrochemical_single_stage_parameter_schema()
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return crystallization_single_stage_parameter_schema()
    return electrochemical_recipe_parameter_schema()


def _recipe_dimension(task_info: Mapping[str, Any], workflow_mode: str) -> int:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return CRYSTALLIZATION_SINGLE_STAGE_DIMENSION
    return task_recipe_dimension(dict(task_info))


def _recipe_categorical_coordinates(
    task_info: Mapping[str, Any], workflow_mode: str
) -> tuple[tuple[int, int], ...]:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return CRYSTALLIZATION_SINGLE_STAGE_CATEGORICAL_COORDINATES
    return task_recipe_categorical_coordinates(dict(task_info))


def _measurement_slots(
    task_info: Mapping[str, Any], workflow_mode: str
) -> tuple[dict[str, Any], ...]:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return tuple(copy.deepcopy(item) for item in ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS)
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return tuple(copy.deepcopy(item) for item in CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS)
    return scientific_measurement_slots(task_info)


def _parameters_from_vector(
    task_info: Mapping[str, Any], workflow_mode: str, vector: np.ndarray
) -> dict[str, int | float]:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return electrochemical_single_stage_parameters_from_unit_vector(vector)
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return crystallization_single_stage_parameters_from_unit_vector(vector)
    return electrochemical_recipe_parameters_from_unit_vector(vector)


def _vector_from_parameters(
    task_info: Mapping[str, Any], workflow_mode: str, payload: object
) -> np.ndarray:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return electrochemical_single_stage_unit_vector_from_parameters(payload)
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return crystallization_single_stage_unit_vector_from_parameters(payload)
    return electrochemical_recipe_unit_vector_from_parameters(payload)


def _recipe_from_vector(
    task_info: Mapping[str, Any], workflow_mode: str, vector: np.ndarray
) -> dict[str, Any]:
    if _uses_single_stage_electrochemistry(task_info, workflow_mode):
        return electrochemical_single_stage_recipe_from_unit_vector(task_info, vector)
    if task_recipe_kind(dict(task_info)) == "reaction_crystallization":
        return crystallization_single_stage_recipe_from_unit_vector(task_info, vector)
    return task_recipe_from_unit_vector(dict(task_info), vector)


def _compact(value: Any) -> Any:
    normalized = to_builtin(value)
    if isinstance(normalized, float):
        return round(normalized, 6)
    if isinstance(normalized, dict):
        return {str(key): _compact(item) for key, item in normalized.items()}
    if isinstance(normalized, list):
        return [_compact(item) for item in normalized]
    return copy.deepcopy(normalized)


@dataclass(frozen=True)
class StaticOptimizationPlan:
    """Validated Agent-authored selection for one experiment in a fixed world."""

    experiment_intent: str
    search_vector: tuple[float, ...]
    requested_measurement_slots: tuple[str, ...]
    measurement_objective: str
    expected_effect: str
    uncertainty: float
    recipe_parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "experiment_intent": self.experiment_intent,
            "search_vector": list(self.search_vector),
            "requested_measurement_slots": list(self.requested_measurement_slots),
            "measurement_objective": self.measurement_objective,
            "expected_effect": self.expected_effect,
            "uncertainty": self.uncertainty,
        }
        if self.recipe_parameters is not None:
            payload["recipe_parameters"] = copy.deepcopy(self.recipe_parameters)
        return payload


@dataclass(frozen=True)
class StaticFinalRecommendation:
    """One terminal scientific recommendation after the exploration budget is exhausted."""

    recommended_search_vector: tuple[float, ...]
    recommended_measurement_slots: tuple[str, ...]
    recommendation_type: str
    source_experiment_indices: tuple[int, ...]
    predicted_score: float
    confidence: float
    method_summary: str
    evidence_refs: tuple[str, ...]
    working_explanation: dict[str, Any]
    remaining_risks: tuple[str, ...]
    recommended_followup: str
    recommended_recipe_parameters: dict[str, Any] | None = None
    counterfactual_predictions: tuple[dict[str, Any], ...] = ()
    schema_version: str = STATIC_FINAL_SYNTHESIS_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "recommended_search_vector": list(self.recommended_search_vector),
            "recommended_measurement_slots": list(self.recommended_measurement_slots),
            "recommendation_type": self.recommendation_type,
            "source_experiment_indices": list(self.source_experiment_indices),
            "predicted_score": self.predicted_score,
            "confidence": self.confidence,
            "method_summary": self.method_summary,
            "evidence_refs": list(self.evidence_refs),
            "working_explanation": copy.deepcopy(self.working_explanation),
            "remaining_risks": list(self.remaining_risks),
            "recommended_followup": self.recommended_followup,
        }
        if self.recommended_recipe_parameters is not None:
            payload["recommended_recipe_parameters"] = copy.deepcopy(
                self.recommended_recipe_parameters
            )
        if self.counterfactual_predictions:
            payload["counterfactual_predictions"] = copy.deepcopy(
                list(self.counterfactual_predictions)
            )
        return payload

    def execution_plan(self) -> StaticOptimizationPlan:
        return StaticOptimizationPlan(
            experiment_intent="blind validation of the final campaign recommendation",
            search_vector=self.recommended_search_vector,
            requested_measurement_slots=self.recommended_measurement_slots,
            measurement_objective="validate the submitted method under independent noise",
            expected_effect=self.method_summary,
            uncertainty=1.0 - self.confidence,
            recipe_parameters=(
                None
                if self.recommended_recipe_parameters is None
                else copy.deepcopy(self.recommended_recipe_parameters)
            ),
        )


class StaticOptimizationContextBuilder:
    """Public context with no mechanism candidates or hidden-world fields."""

    def __init__(
        self,
        task_info: Mapping[str, Any],
        *,
        history_limit: int = 8,
        total_experiments: int | None = None,
        final_synthesis_after_exploration: bool = False,
        include_task_operation_budget: bool = True,
        predictive_world_understanding_enabled: bool = False,
        material_information: Mapping[str, Any] | None = None,
        electrochemical_material_family_id: str | None = None,
        crystallization_material_family_id: str | None = None,
        scoring_contract: Mapping[str, Any] | None = None,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
        optimization_scaffold_id: str | None = None,
        algorithm_seed: int = 0,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self.task_info = dict(task_info)
        self.history_limit = int(history_limit)
        self.total_experiments = int(total_experiments) if total_experiments is not None else None
        if self.total_experiments is not None and self.total_experiments <= 0:
            raise ValueError("total_experiments must be positive")
        self.final_synthesis_after_exploration = bool(final_synthesis_after_exploration)
        self.include_task_operation_budget = bool(include_task_operation_budget)
        self.predictive_world_understanding_enabled = bool(predictive_world_understanding_enabled)
        active_material_family_id = (
            crystallization_material_family_id
            if str(self.task_info.get("task_id", "")) == "reaction-to-crystallization"
            else electrochemical_material_family_id
        )
        self.material_information_config = normalize_static_material_information_config(
            material_information,
            task_ids=(str(self.task_info.get("task_id", "")),),
            material_family_id=active_material_family_id,
        )
        self.material_information_condition = str(self.material_information_config["mode"])
        self.material_information = static_material_information_dossier(
            self.material_information_config,
            task_id=str(self.task_info.get("task_id", "")),
            material_family_id=active_material_family_id,
        )
        self.material_information_sha256 = (
            canonical_sha256(self.material_information)
            if self.material_information is not None
            else None
        )
        self.scoring_contract = (
            None if scoring_contract is None else copy.deepcopy(dict(scoring_contract))
        )
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.optimization_scaffold_id = str(
            optimization_scaffold_id or DIRECT_OPTIMIZATION_SCAFFOLD_ID
        )
        self.algorithm_seed = int(algorithm_seed)
        self.measurement_slots = _measurement_slots(
            self.task_info, self.electrochemical_workflow_mode
        )

    def _coverage_design(self) -> np.ndarray:
        """Return a deterministic, task-neutral balanced design for the first rounds."""

        count = COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
        dimension = _recipe_dimension(self.task_info, self.electrochemical_workflow_mode)
        task_id = str(self.task_info.get("task_id", ""))
        task_offset = sum(
            (index + 1) * byte for index, byte in enumerate(task_id.encode("utf-8"))
        )
        rng = np.random.default_rng(self.algorithm_seed + task_offset)
        design = np.empty((count, dimension), dtype=float)
        categorical = dict(
            _recipe_categorical_coordinates(
                self.task_info, self.electrochemical_workflow_mode
            )
        )
        categorical_rank = 0
        for coordinate in range(dimension):
            category_count = categorical.get(coordinate)
            if category_count is not None:
                offset = int(rng.integers(0, category_count))
                rows = np.arange(count)
                base_categories = rows % category_count
                blocks = rows // category_count
                categories = (
                    base_categories * (2 * categorical_rank + 1)
                    + blocks * categorical_rank
                    + offset
                ) % category_count
                design[:, coordinate] = (categories + 0.5) / category_count
                categorical_rank += 1
                continue
            permutation = rng.permutation(count)
            design[:, coordinate] = (permutation + 0.5) / count
        return design

    def _history_vector(self, record: Mapping[str, Any]) -> np.ndarray:
        plan = record.get("plan")
        if not isinstance(plan, Mapping):
            raise ValueError("static history record is missing plan")
        if _uses_named_physical_controls(self.task_info):
            return _vector_from_parameters(
                self.task_info,
                self.electrochemical_workflow_mode,
                plan.get("recipe_parameters"),
            )
        return np.asarray(plan.get("search_vector"), dtype=float).reshape(-1)

    def _coordinate_labels(self) -> tuple[str, ...]:
        if _uses_named_physical_controls(self.task_info):
            return tuple(
                str(control_id)
                for control_id in _recipe_parameter_schema(
                    self.task_info, self.electrochemical_workflow_mode
                )
            )
        return tuple(
            str(item["control_id"]) for item in task_recipe_coordinate_schema(self.task_info)
        )

    def _coverage_audit(
        self, experiment_history: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        vectors = [self._history_vector(record) for record in experiment_history]
        labels = self._coordinate_labels()
        categorical = dict(
            _recipe_categorical_coordinates(
                self.task_info, self.electrochemical_workflow_mode
            )
        )
        continuous_coverage: list[dict[str, Any]] = []
        categorical_coverage: list[dict[str, Any]] = []
        categorical_assignments: dict[str, list[int]] = {}
        for coordinate, label in enumerate(labels):
            values = [float(vector[coordinate]) for vector in vectors]
            category_count = categorical.get(coordinate)
            if category_count is not None:
                counts = {
                    str(category): sum(
                        min(int(value * category_count), category_count - 1) == category
                        for value in values
                    )
                    for category in range(category_count)
                }
                categorical_assignments[label] = [
                    min(int(value * category_count), category_count - 1)
                    for value in values
                ]
                categorical_coverage.append(
                    {
                        "control_id": label,
                        "category_counts": counts,
                        "all_categories_seen": bool(values)
                        and all(count > 0 for count in counts.values()),
                    }
                )
                continue
            continuous_coverage.append(
                {
                    "control_id": label,
                    "distinct_value_count": len({round(value, 8) for value in values}),
                    "lower_region_seen": any(value <= 0.20 for value in values),
                    "upper_region_seen": any(value >= 0.80 for value in values),
                }
            )
        nominal_pair_coverage = []
        cardinalities = {
            str(item["control_id"]): len(item["category_counts"])
            for item in categorical_coverage
        }
        for first, second in combinations(categorical_assignments, 2):
            observed_pairs = set(
                zip(
                    categorical_assignments[first],
                    categorical_assignments[second],
                    strict=True,
                )
            )
            maximum_distinct_pairs = min(
                len(vectors),
                cardinalities[first] * cardinalities[second],
            )
            nominal_pair_coverage.append(
                {
                    "control_ids": [first, second],
                    "distinct_pair_count": len(observed_pairs),
                    "maximum_distinct_pair_count_at_current_budget": (
                        maximum_distinct_pairs
                    ),
                    "maximally_distinct_at_current_budget": bool(vectors)
                    and len(observed_pairs) == maximum_distinct_pairs,
                }
            )
        return {
            "completed_experiment_count": len(experiment_history),
            "continuous_controls": continuous_coverage,
            "categorical_controls": categorical_coverage,
            "nominal_pair_coverage": nominal_pair_coverage,
            "all_continuous_extremes_seen": bool(continuous_coverage)
            and all(
                item["lower_region_seen"] and item["upper_region_seen"]
                for item in continuous_coverage
            ),
            "all_nominal_categories_seen": bool(categorical_coverage)
            and all(item["all_categories_seen"] for item in categorical_coverage),
            "all_nominal_pairs_maximally_distinct": bool(nominal_pair_coverage)
            and all(
                item["maximally_distinct_at_current_budget"]
                for item in nominal_pair_coverage
            ),
        }

    def _candidate_portfolio(
        self,
        experiment_history: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        history_vectors = [
            self._history_vector(record).tolist() for record in experiment_history
        ]
        scores = [
            float(record["terminal_summary"].get("leaderboard_score", 0.0))
            for record in experiment_history
        ]
        task_id = str(self.task_info.get("task_id", ""))
        task_offset = sum(
            (index + 1) * byte for index, byte in enumerate(task_id.encode("utf-8"))
        )
        portfolio = public_surrogate_candidate_portfolio(
            history_vectors,
            scores,
            categorical=_recipe_categorical_coordinates(
                self.task_info,
                self.electrochemical_workflow_mode,
            ),
            seed=(
                self.algorithm_seed
                + task_offset
                + 1009 * len(experiment_history)
            ),
        )
        public_portfolio: list[dict[str, Any]] = []
        for item in portfolio:
            vector = np.asarray(item["search_vector"], dtype=float)
            public_item = {
                key: copy.deepcopy(value)
                for key, value in item.items()
                if key != "search_vector"
            }
            if _uses_named_physical_controls(self.task_info):
                public_item["recipe_parameters"] = _compact(
                    _parameters_from_vector(
                        self.task_info,
                        self.electrochemical_workflow_mode,
                        vector,
                    )
                )
            else:
                public_item["search_vector"] = _compact(vector.tolist())
                public_item["public_physical_controls"] = _compact(
                    task_recipe_public_controls(self.task_info, vector)
                )
            public_portfolio.append(public_item)
        return public_portfolio

    def _campaign_scaffold(
        self, experiment_history: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        completed = len(experiment_history)
        total = self.total_experiments or max(
            COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS, completed + 1
        )
        portfolio_scaffold = (
            self.optimization_scaffold_id
            in SURROGATE_PORTFOLIO_OPTIMIZATION_SCAFFOLD_IDS
        )
        scheduled_portfolio_scaffold = (
            self.optimization_scaffold_id
            == COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID
        )
        nonduplicate_portfolio_scaffold = (
            self.optimization_scaffold_id
            in {
                COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID,
                COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID,
            }
        )
        portfolio_final_experiment_index = (
            SURROGATE_PORTFOLIO_NONDUPLICATE_FINAL_EXPERIMENT_INDEX
            if nonduplicate_portfolio_scaffold
            else SURROGATE_PORTFOLIO_V17_FINAL_EXPERIMENT_INDEX
        )
        portfolio_selection_active = bool(
            portfolio_scaffold
            and COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
            <= completed
            <= portfolio_final_experiment_index
        )
        closeout_start = (
            portfolio_final_experiment_index + 1
            if portfolio_scaffold
            else max(total - 4, COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS)
        )
        if completed < COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS:
            phase = "balanced_coverage"
            requirement = (
                "The protocol executor commits the displayed balanced-design condition. "
                "Use the model call to select diagnostics and explain what this condition "
                "tests; the executor continues with the frozen task-neutral public "
                "portfolio schedule after experiment eight."
                if scheduled_portfolio_scaffold
                else "The protocol executor commits the displayed balanced-design "
                "condition. Use the model call to select diagnostics and explain what "
                "this condition tests; recipe selection becomes model-controlled after "
                "experiment eight."
            )
        elif scheduled_portfolio_scaffold and completed <= 13:
            phase = "scheduled_global_portfolio_discrimination"
            requirement = (
                "The protocol executor commits the displayed public-history maximin "
                "candidate under the frozen task-neutral schedule. Use the model call "
                "to select diagnostics and explain how this condition discriminates "
                "basins; do not return recipe fields or a candidate_id."
            )
        elif portfolio_scaffold and completed <= 13:
            phase = "global_surrogate_discrimination"
            requirement = (
                "Select exactly one displayed candidate_id and return its exact recipe. "
                "Treat the first eight space-filling runs as confounded. Compare GP-EI, "
                "RF-EI, surrogate consensus, maximum-distance, uncertainty, and boundary "
                "candidates; preserve basin discovery rather than making tiny sequential "
                "changes or repeating a recipe."
            )
        elif scheduled_portfolio_scaffold and completed <= (
            SURROGATE_PORTFOLIO_NONDUPLICATE_FINAL_EXPERIMENT_INDEX
        ):
            phase = "scheduled_boundary_portfolio_discrimination"
            requirement = (
                "The protocol executor commits the displayed public-history "
                "boundary-challenge candidate under the frozen task-neutral schedule. "
                "Use the model call to select diagnostics and explain which boundary, "
                "nominal alternative, or competing basin this condition challenges; "
                "do not return recipe fields or a candidate_id."
            )
        elif (
            portfolio_scaffold
            and completed <= SURROGATE_PORTFOLIO_V17_FINAL_EXPERIMENT_INDEX
        ):
            phase = "surrogate_guided_convergence"
            requirement = (
                "Select exactly one displayed candidate_id and return its exact recipe. "
                "Use accumulated public evidence to balance predicted score, uncertainty, "
                "distance from prior recipes, stage diagnostics, and unresolved competing "
                "basins. Exact exploration duplicates remain wasteful because blind "
                "validation follows final recommendation."
            )
        elif nonduplicate_portfolio_scaffold and completed <= (
            SURROGATE_PORTFOLIO_NONDUPLICATE_FINAL_EXPERIMENT_INDEX
        ):
            phase = "nonduplicate_bottleneck_closeout"
            requirement = (
                "Select exactly one displayed candidate_id and return its exact novel "
                "recipe. Do not replicate the incumbent: blind validation follows the "
                "committed recommendation. Inspect signed score-component weights and "
                "stage measurements, identify any near-floor positive component or "
                "near-ceiling penalty in the strongest region, and use the remaining "
                "budget to challenge that bottleneck or a credible competing basin."
            )
        elif completed < closeout_start:
            phase = "adaptive_discrimination"
            requirement = (
                "Use the coverage audit and public scores to compare multiple promising "
                "regions. Prefer controlled one-factor contrasts or local refinements, "
                "while retaining at least one credible alternative nominal category."
            )
        else:
            phase = "robust_closeout"
            requirement = (
                "Resolve uncertainty around the strongest tested region. Deliberately "
                "replicate or challenge the leading condition; avoid unsupported "
                "high-dimensional extrapolation before final synthesis."
            )
        ranked = sorted(
            (
                {
                    "experiment_index": int(record["experiment_index"]),
                    "leaderboard_score": float(
                        record["terminal_summary"].get("leaderboard_score", 0.0)
                    ),
                }
                for record in experiment_history
            ),
            key=lambda item: item["leaderboard_score"],
            reverse=True,
        )[:3]
        scaffold: dict[str, Any] = {
            "scaffold_id": self.optimization_scaffold_id,
            "policy_scope": "task_neutral_public_history_only",
            "phase": phase,
            "initial_balanced_coverage_experiments": (
                COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
            ),
            "initial_design_authority": (
                "protocol_executor"
                if completed < COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
                else "completed"
            ),
            "model_recipe_selection_begins_at_experiment_index": (
                None
                if scheduled_portfolio_scaffold
                else COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
            ),
            "portfolio_candidate_selection_experiment_indices": (
                list(
                    range(
                        COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS,
                        portfolio_final_experiment_index + 1,
                    )
                )
                if portfolio_scaffold
                else []
            ),
            "free_model_closeout_experiment_indices": (
                list(
                    range(
                        portfolio_final_experiment_index + 1,
                        total,
                    )
                )
                if portfolio_scaffold
                else []
            ),
            "decision_requirement": requirement,
            "coverage_audit": self._coverage_audit(experiment_history),
            "top_public_experiments": ranked,
            "invariants": {
                "nominal_codes_are_independent_and_unordered": True,
                "continuous_range_coverage_precedes_local_exploitation": True,
                "terminal_leaderboard_score_is_primary_feedback": True,
                "audit_only_metrics_do_not_drive_candidate_selection": True,
                "final_recommendation_should_be_supported_by_robust_public_evidence": True,
                "candidate_generation_uses_public_history_only": portfolio_scaffold,
                "candidate_generation_does_not_commit_the_recipe": (
                    portfolio_scaffold and not scheduled_portfolio_scaffold
                ),
                "candidate_selection_authority_remains_with_model": (
                    portfolio_scaffold and not scheduled_portfolio_scaffold
                ),
            },
        }
        if (
            nonduplicate_portfolio_scaffold
            and not scheduled_portfolio_scaffold
        ):
            scaffold["invariants"].update(
                {
                    "campaign_exploration_recipes_must_be_distinct": True,
                    "blind_validation_supplies_replication": True,
                    "score_component_bottlenecks_must_be_considered": True,
                }
            )
        elif scheduled_portfolio_scaffold:
            scaffold["invariants"].update(
                {
                    "campaign_exploration_recipes_must_be_distinct": True,
                    "blind_validation_supplies_replication": True,
                    "registered_terminal_score_remains_primary_objective": True,
                    "candidate_selection_authority": (
                        "protocol_executor_task_neutral_schedule"
                    ),
                }
            )
        if completed < COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS:
            suggestion = self._coverage_design()[completed]
            if _uses_named_physical_controls(self.task_info):
                scaffold["executor_committed_recipe_parameters"] = _compact(
                    _parameters_from_vector(
                        self.task_info,
                        self.electrochemical_workflow_mode,
                        suggestion,
                    )
                )
            else:
                scaffold["executor_committed_search_vector"] = _compact(
                    suggestion.tolist()
                )
            scaffold["committed_design_semantics"] = (
                "deterministic balanced Latin-hypercube strata for continuous controls "
                "with balanced independent nominal-category coverage"
            )
        elif portfolio_selection_active:
            default_candidate_id = (
                "maximin_global"
                if completed <= 13
                else "boundary_challenge"
            )
            portfolio = self._candidate_portfolio(experiment_history)
            portfolio.sort(
                key=lambda item: item["candidate_id"] != default_candidate_id
            )
            scaffold["task_neutral_default_candidate_id"] = default_candidate_id
            if scheduled_portfolio_scaffold:
                committed = copy.deepcopy(portfolio[0])
                scaffold["public_history_candidate_portfolio"] = portfolio
                scaffold["executor_committed_candidate_id"] = str(
                    committed["candidate_id"]
                )
                if "recipe_parameters" in committed:
                    scaffold["executor_committed_recipe_parameters"] = copy.deepcopy(
                        committed["recipe_parameters"]
                    )
                else:
                    scaffold["executor_committed_search_vector"] = copy.deepcopy(
                        committed["search_vector"]
                    )
                scaffold["committed_design_semantics"] = (
                    "frozen task-neutral schedule: maximum-distance public candidate "
                    "for experiments 9 through 14, then public boundary-challenge "
                    "candidate for experiments 15 through 20"
                )
                scaffold["candidate_portfolio_contract"] = {
                    "generation_authority": (
                        "protocol_executor_using_public_history_only"
                    ),
                    "selection_authority": (
                        "protocol_executor_task_neutral_schedule"
                    ),
                    "model_selects_diagnostics_and_scientific_interpretation": True,
                    "model_must_not_return_recipe_or_candidate_id": True,
                    "hidden_world_fields_used": False,
                    "completed_recipe_exclusion_minimum_encoded_distance": 0.02,
                }
            else:
                scaffold["model_candidate_portfolio"] = portfolio
                scaffold["default_candidate_deviation_contract"] = (
                    "Use the default candidate unless specific completed public "
                    "experiments or evidence IDs justify another displayed candidate. "
                    "State that evidence in experiment_intent; no hidden-world rationale "
                    "is permitted."
                )
                scaffold["candidate_portfolio_contract"] = {
                    "generation_authority": (
                        "protocol_executor_using_public_history_only"
                    ),
                    "selection_authority": "model",
                    "model_must_return_exact_listed_recipe": True,
                    "hidden_world_fields_used": False,
                    "surrogate_predictions_are_ground_truth": False,
                    "candidate_id_required": True,
                }
            if (
                nonduplicate_portfolio_scaffold
                and not scheduled_portfolio_scaffold
            ):
                scaffold["candidate_portfolio_contract"][
                    "completed_recipe_exclusion_minimum_encoded_distance"
                ] = 0.02
        return scaffold

    def build(
        self,
        experiment_history: Sequence[Mapping[str, Any]],
        *,
        decision_stage: str = "experiment_design",
        include_prediction_queries: bool | None = None,
    ) -> dict[str, Any]:
        if decision_stage not in {"experiment_design", "final_synthesis"}:
            raise ValueError("unsupported static optimization decision stage")
        if include_prediction_queries is None:
            include_prediction_queries = (
                self.predictive_world_understanding_enabled and decision_stage == "final_synthesis"
            )
        if include_prediction_queries and decision_stage != "final_synthesis":
            raise ValueError("prediction queries are only valid after exploration")
        selected = self._select_history(experiment_history)
        history = [self._compact_history_record(item) for item in selected]
        evidence_catalog = [
            str(entry["evidence_id"])
            for record in history
            for entry in record["measurement_evidence"]
        ]
        optimization_contract: dict[str, Any] = {
            "world_policy": "static_for_entire_campaign",
            "objective": "maximize_fixed_task_success_metrics",
            "feedback_policy": "public_processed_estimate_uncertainty_reward",
            "decision_stage": decision_stage,
        }
        if self.scoring_contract is not None:
            optimization_contract["scoring_contract"] = copy.deepcopy(self.scoring_contract)
            if self.scoring_contract.get("contract_id") == DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2:
                optimization_contract["metric_roles"] = {
                    "safety_risk": {
                        "role": "audit_only",
                        "enters_primary_score": False,
                        "constrains_candidate_selection": False,
                        "reference_threshold": self.task_info.get("safety_limit"),
                        "instruction": (
                            "Record and report safety_risk, but do not optimize it or "
                            "treat its audit reference threshold as a candidate-selection "
                            "constraint in this development pilot."
                        ),
                    }
                }
        if self.total_experiments is not None:
            completed = len(experiment_history)
            current = min(completed + 1, self.total_experiments)
            remaining_after_current = max(self.total_experiments - current, 0)
            if decision_stage == "final_synthesis":
                current = self.total_experiments
                remaining_after_current = 0
            optimization_contract["scientific_campaign_budget"] = {
                "total_exploration_experiments": self.total_experiments,
                "completed_experiments": completed,
                "current_experiment_number": current,
                "remaining_experiments_after_current": remaining_after_current,
                "final_synthesis_after_exploration": (self.final_synthesis_after_exploration),
                "validation_feedback_returned_to_agent": False,
            }
        experiment_interface: dict[str, Any] = {
            "decision_scope": "complete_experiment",
            "recipe_space_version": (
                ELECTROCHEMICAL_SINGLE_STAGE_RECIPE_VERSION
                if _uses_single_stage_electrochemistry(
                    self.task_info, self.electrochemical_workflow_mode
                )
                else CRYSTALLIZATION_SINGLE_STAGE_RECIPE_VERSION
                if task_recipe_kind(self.task_info) == "reaction_crystallization"
                else TASK_RECIPE_SPACE_VERSION
            ),
            "recipe_space_kind": task_recipe_kind(self.task_info),
            "diagnostic_measurement_slots": [
                copy.deepcopy(item) for item in self.measurement_slots
            ],
            "required_measurement_slots": list(
                required_scientific_measurement_slot_ids(self.task_info)
            ),
            "closeout": [
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ],
        }
        recipe_kind = task_recipe_kind(self.task_info)
        if _uses_named_physical_controls(self.task_info):
            recipe_parameter_schema = _recipe_parameter_schema(
                self.task_info, self.electrochemical_workflow_mode
            )
            categorical_controls = {
                control_id: (int(specification["maximum"]) - int(specification["minimum"]) + 1)
                for control_id, specification in recipe_parameter_schema.items()
                if specification.get("type") == "integer"
            }
            experiment_interface.update(
                {
                    "parameterization": "named_physical_controls",
                    "recipe_parameter_schema": recipe_parameter_schema,
                    "internal_unit_vector_visible_to_agent": False,
                    "categorical_controls": categorical_controls,
                    "categorical_semantics": {
                        "independent": True,
                        "unordered_nominal": True,
                        "numeric_order_or_distance_meaning": False,
                        "cross_control_code_equality_meaning": False,
                        "instruction": (
                            "Categorical controls are independent nominal choices; code "
                            "order, distance, and equality across controls have no meaning."
                        ),
                    },
                    "world_understanding_claim_contract": {
                        "cause_variables": list(
                            _ELECTROCHEMICAL_SINGLE_STAGE_CLAIM_CAUSES
                            if _uses_single_stage_electrochemistry(
                                self.task_info, self.electrochemical_workflow_mode
                            )
                            else _ELECTROCHEMICAL_CLAIM_CAUSES
                            if recipe_kind == "electrochemical"
                            else _CRYSTALLIZATION_CLAIM_CAUSES
                        ),
                        "effect_variables": list(
                            _ELECTROCHEMICAL_CLAIM_EFFECTS
                            if recipe_kind == "electrochemical"
                            else _CRYSTALLIZATION_CLAIM_EFFECTS
                        ),
                        "relations": [
                            "positive",
                            "negative",
                            "nonmonotonic",
                            "conditional",
                            "no_direct_effect",
                        ],
                        "mechanism_tags": list(
                            _ELECTROCHEMICAL_MECHANISM_TAGS
                            if recipe_kind == "electrochemical"
                            else _CRYSTALLIZATION_MECHANISM_TAGS
                        ),
                        "claim_target": (
                            "observable causal equivalence classes; do not guess hidden species IDs"
                        ),
                    },
                }
            )
            if recipe_kind == "electrochemical":
                experiment_interface["electrochemical_workflow_mode"] = (
                    self.electrochemical_workflow_mode
                )
            if self.material_information is not None:
                experiment_interface["material_information"] = copy.deepcopy(
                    self.material_information
                )
        else:
            experiment_interface.update(
                {
                    "parameterization": "unit_vector_with_public_physical_coordinate_schema",
                    "search_vector_dimension": _recipe_dimension(
                        self.task_info, self.electrochemical_workflow_mode
                    ),
                    "search_vector_bounds": [0.0, 1.0],
                    "search_vector_coordinate_schema": [
                        copy.deepcopy(item)
                        for item in task_recipe_coordinate_schema(self.task_info)
                    ],
                    "physical_controls_are_deterministically_decoded": True,
                    "categorical_coordinates": [
                        {
                            "coordinate": coordinate,
                            "category_count": count,
                            "selection_semantics": "independent_unordered_nominal_choice",
                        }
                        for coordinate, count in _recipe_categorical_coordinates(
                            self.task_info, self.electrochemical_workflow_mode
                        )
                    ],
                    "categorical_semantics": {
                        "coordinates_are_independently_selectable": True,
                        "categories_are_unordered_nominal_choices": True,
                        "numeric_order_has_scientific_meaning": False,
                        "numeric_distance_has_scientific_meaning": False,
                        "matching_codes_across_coordinates_has_scientific_meaning": False,
                        "instruction": (
                            "Treat every categorical coordinate as an independent unordered "
                            "nominal choice. Numeric proximity and equal numeric codes across "
                            "different coordinates carry no scientific meaning."
                        ),
                    },
                }
            )
            if self.material_information is not None:
                experiment_interface["material_information"] = copy.deepcopy(
                    self.material_information
                )
        payload = {
            "schema_version": STATIC_OPTIMIZATION_INTERFACE_VERSION,
            "optimization_contract": optimization_contract,
            "task": {
                key: copy.deepcopy(self.task_info[key])
                for key in _PUBLIC_TASK_KEYS
                if key in self.task_info and (key != "budget" or self.include_task_operation_budget)
            },
            "experiment_interface": experiment_interface,
            "history_window": {
                "selection_policy": "oldest_reference_half_plus_most_recent_half",
                "total_experiment_count": len(experiment_history),
                "included_experiment_indices": [record["experiment_index"] for record in history],
            },
            "experiment_history": history,
            "evidence_catalog": evidence_catalog,
        }
        if self.optimization_scaffold_id in COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_IDS:
            payload["campaign_scaffold"] = self._campaign_scaffold(experiment_history)
        if decision_stage == "final_synthesis" and include_prediction_queries:
            queries = (
                build_electrochemical_prediction_queries(
                    experiment_history,
                    electrochemical_workflow_mode=self.electrochemical_workflow_mode,
                )
                if recipe_kind == "electrochemical"
                else build_crystallization_prediction_queries(experiment_history)
            )
            payload["held_out_prediction_queries"] = [query.to_public_dict() for query in queries]
            optimization_contract["predictive_validation"] = {
                "query_count": len(queries),
                "executed_before_prediction": False,
                "feedback_returned_to_agent": False,
                "one_factor_interventions": True,
            }
        return payload

    def _select_history(self, history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        records = list(history)
        if len(records) <= self.history_limit:
            return records
        reference_count = max(self.history_limit // 2, 1)
        return records[:reference_count] + records[-(self.history_limit - reference_count) :]

    def _compact_history_record(self, item: Mapping[str, Any]) -> dict[str, Any]:
        plan = item.get("plan")
        evidence = item.get("measurement_evidence")
        terminal = item.get("terminal_summary")
        if not isinstance(plan, Mapping):
            raise ValueError("static history record is missing plan")
        if not isinstance(evidence, list) or not all(
            isinstance(entry, Mapping) for entry in evidence
        ):
            raise ValueError("static history measurement_evidence must be a list")
        if not isinstance(terminal, Mapping):
            raise ValueError("static history record is missing terminal_summary")
        compact_plan = {
            key: _compact(plan[key])
            for key in (
                "requested_measurement_slots",
                "measurement_objective",
                "expected_effect",
                "uncertainty",
            )
            if key in plan
        }
        if _uses_named_physical_controls(self.task_info):
            vector = np.asarray(plan["search_vector"], dtype=float)
            compact_plan["recipe_parameters"] = _compact(
                plan.get(
                    "recipe_parameters",
                    _parameters_from_vector(
                        self.task_info,
                        self.electrochemical_workflow_mode,
                        vector,
                    ),
                )
            )
        else:
            vector = np.asarray(plan["search_vector"], dtype=float)
            compact_plan["search_vector"] = _compact(plan["search_vector"])
            compact_plan["public_physical_controls"] = _compact(
                task_recipe_public_controls(self.task_info, vector)
            )
        metrics_by_slot = {
            str(slot["slot_id"]): {
                str(metric) for metric in slot.get("model_facing_metric_ids", ())
            }
            for slot in self.measurement_slots
            if slot.get("model_facing_metric_ids")
        }
        compact_evidence: list[dict[str, Any]] = []
        for entry in evidence:
            slot_id = str(entry.get("measurement_slot_id", ""))
            allowed_metrics = metrics_by_slot.get(slot_id)
            compact_entry = {
                key: _compact(entry[key])
                for key in (
                    "evidence_id",
                    "reward",
                )
                if key in entry
            }
            processed_estimate = entry.get("processed_estimate")
            if isinstance(processed_estimate, Mapping):
                compact_entry["processed_estimate"] = _compact(
                    {
                        key: value
                        for key, value in processed_estimate.items()
                        if allowed_metrics is None or key in allowed_metrics
                    }
                )
            uncertainty = entry.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                compact_entry["uncertainty"] = _compact(
                    {
                        key: value
                        for key, value in uncertainty.items()
                        if allowed_metrics is None or key.removesuffix("_std") in allowed_metrics
                    }
                )
            compact_evidence.append(compact_entry)
        return {
            "experiment_index": int(item["experiment_index"]),
            "plan": compact_plan,
            "measurement_evidence": compact_evidence,
            "terminal_summary": {
                key: _compact(terminal[key]) for key in _PUBLIC_TERMINAL_KEYS if key in terminal
            },
        }


class StaticOptimizationValidator:
    def __init__(
        self,
        task_info: Mapping[str, Any],
        *,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
    ) -> None:
        self.task_info = dict(task_info)
        self.recipe_kind = task_recipe_kind(self.task_info)
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.dimension = _recipe_dimension(self.task_info, self.electrochemical_workflow_mode)
        self.measurement_slot_ids = tuple(
            str(item["slot_id"])
            for item in _measurement_slots(self.task_info, self.electrochemical_workflow_mode)
        )
        self.required_measurement_slot_ids = tuple(
            str(item["slot_id"])
            for item in _measurement_slots(self.task_info, self.electrochemical_workflow_mode)
            if item.get("selection_policy") == "required_by_workflow"
        )

    @staticmethod
    def _text(value: Any, *, field: str, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ScientificPlanValidationError(
                f"{field} must be a non-empty string",
                field_path=field,
                constraint="non_empty_string",
                observed=0 if isinstance(value, str) else type(value).__name__,
                limit=1,
            )
        normalized = value.strip()
        if len(normalized) > maximum:
            raise ScientificPlanValidationError(
                f"{field} exceeds its character limit",
                field_path=field,
                constraint="max_characters",
                observed=len(normalized),
                limit=maximum,
            )
        return normalized

    def validate(self, payload: Any) -> StaticOptimizationPlan:
        if not isinstance(payload, Mapping):
            raise ScientificPlanValidationError(
                "static optimization response must be an object",
                field_path="static_optimization_response",
                constraint="object",
                observed=type(payload).__name__,
            )
        required = {
            "experiment_intent",
            "requested_measurement_slots",
            "measurement_objective",
            "expected_effect",
            "uncertainty",
        }
        required.add(
            "recipe_parameters"
            if _uses_named_physical_controls(self.task_info)
            else "search_vector"
        )
        if set(payload) != required:
            raise ScientificPlanValidationError(
                "static optimization response fields do not match its contract",
                field_path="static_optimization_response",
                constraint="exact_declared_fields",
                observed=len(set(payload) - required),
                limit=0,
            )
        normalized_recipe_parameters: dict[str, Any] | None = None
        if _uses_named_physical_controls(self.task_info):
            try:
                encoded = _vector_from_parameters(
                    self.task_info,
                    self.electrochemical_workflow_mode,
                    payload["recipe_parameters"],
                )
            except ValueError as error:
                raise ScientificPlanValidationError(
                    str(error),
                    field_path="recipe_parameters",
                    constraint="physical_recipe_contract",
                ) from error
            normalized_vector = [float(value) for value in encoded]
            normalized_recipe_parameters = _parameters_from_vector(
                self.task_info,
                self.electrochemical_workflow_mode,
                encoded,
            )
        else:
            vector = payload["search_vector"]
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise ScientificPlanValidationError(
                    "search_vector has the wrong dimension",
                    field_path="search_vector",
                    constraint="exact_items",
                    observed=(len(vector) if isinstance(vector, list) else type(vector).__name__),
                    limit=self.dimension,
                )
            normalized_vector = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ScientificPlanValidationError(
                        "search_vector values must be numeric",
                        field_path="search_vector",
                        constraint="numeric_items",
                        observed=type(value).__name__,
                    )
                coordinate = float(value)
                if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                    raise ScientificPlanValidationError(
                        "search_vector values must be finite and in [0, 1]",
                        field_path="search_vector",
                        constraint="finite_unit_interval_items",
                    )
                normalized_vector.append(coordinate)
        requested = payload["requested_measurement_slots"]
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ScientificPlanValidationError(
                "requested_measurement_slots must be a string list",
                field_path="requested_measurement_slots",
                constraint="string_list",
                observed=type(requested).__name__,
            )
        normalized_requested = [str(item) for item in requested]
        if len(set(normalized_requested)) != len(normalized_requested):
            raise ScientificPlanValidationError(
                "requested_measurement_slots must be unique",
                field_path="requested_measurement_slots",
                constraint="unique_items",
                observed=len(normalized_requested),
                limit=len(set(normalized_requested)),
            )
        if not set(normalized_requested).issubset(self.measurement_slot_ids):
            raise ScientificPlanValidationError(
                "requested_measurement_slots contains an unknown slot",
                field_path="requested_measurement_slots",
                constraint="known_measurement_slot_ids",
            )
        if not set(self.required_measurement_slot_ids).issubset(normalized_requested):
            raise ScientificPlanValidationError(
                "requested_measurement_slots omits a workflow-required slot",
                field_path="requested_measurement_slots",
                constraint="required_measurement_slot_ids",
                observed=len(set(self.required_measurement_slot_ids) - set(normalized_requested)),
                limit=0,
            )
        normalized_requested.sort(key=self.measurement_slot_ids.index)
        uncertainty = payload["uncertainty"]
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, int | float):
            raise ScientificPlanValidationError(
                "uncertainty must be numeric",
                field_path="uncertainty",
                constraint="numeric",
                observed=type(uncertainty).__name__,
            )
        uncertainty_float = float(uncertainty)
        if not math.isfinite(uncertainty_float) or not 0.0 <= uncertainty_float <= 1.0:
            raise ScientificPlanValidationError(
                "uncertainty must be finite and in [0, 1]",
                field_path="uncertainty",
                constraint="finite_unit_interval",
            )
        return StaticOptimizationPlan(
            experiment_intent=self._text(
                payload["experiment_intent"], field="experiment_intent", maximum=1000
            ),
            search_vector=tuple(normalized_vector),
            requested_measurement_slots=tuple(normalized_requested),
            measurement_objective=self._text(
                payload["measurement_objective"],
                field="measurement_objective",
                maximum=1000,
            ),
            expected_effect=self._text(
                payload["expected_effect"], field="expected_effect", maximum=1200
            ),
            uncertainty=uncertainty_float,
            recipe_parameters=normalized_recipe_parameters,
        )


class StaticFinalRecommendationValidator:
    _RECOMMENDATION_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"tested", "interpolated", "extrapolated"}
    )

    def __init__(
        self,
        task_info: Mapping[str, Any],
        *,
        predictive_world_understanding_enabled: bool = False,
        final_synthesis_version: str = STATIC_FINAL_SYNTHESIS_VERSION,
        declared_claim_validation_policy: str = DECLARED_CLAIM_VALIDATION_STRICT,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
    ) -> None:
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.plan_validator = StaticOptimizationValidator(
            task_info,
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
        )
        self.recipe_kind = self.plan_validator.recipe_kind
        self.predictive_world_understanding_enabled = bool(predictive_world_understanding_enabled)
        self.final_synthesis_version = str(final_synthesis_version)
        if declared_claim_validation_policy not in DECLARED_CLAIM_VALIDATION_POLICIES:
            raise ValueError("unknown Declared claim validation policy")
        self.declared_claim_validation_policy = declared_claim_validation_policy

    @staticmethod
    def _string_list(value: Any, *, field: str, maximum_items: int = 16) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ScientificPlanValidationError(
                f"{field} must be a list of non-empty strings",
                field_path=field,
                constraint="non_empty_string_list",
            )
        if len(value) > maximum_items:
            raise ScientificPlanValidationError(
                f"{field} exceeds its item limit",
                field_path=field,
                constraint="max_items",
                observed=len(value),
                limit=maximum_items,
            )
        return [str(item).strip() for item in value]

    @staticmethod
    def _probability(value: Any, *, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ScientificPlanValidationError(
                f"{field} must be numeric",
                field_path=field,
                constraint="numeric",
            )
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ScientificPlanValidationError(
                f"{field} must be in [0, 1]",
                field_path=field,
                constraint="finite_unit_interval",
            )
        return normalized

    def validate(
        self,
        payload: Any,
        *,
        history: Sequence[Mapping[str, Any]],
        evidence_catalog: Sequence[str],
        prediction_queries: Sequence[
            ElectrochemicalPredictionQuery | CrystallizationPredictionQuery
        ] = (),
    ) -> StaticFinalRecommendation:
        if not isinstance(payload, Mapping):
            raise ScientificPlanValidationError(
                "final synthesis response must be an object",
                field_path="final_synthesis_response",
                constraint="object",
            )
        required = {
            "schema_version",
            "recommended_measurement_slots",
            "recommendation_type",
            "source_experiment_indices",
            "predicted_score",
            "confidence",
            "method_summary",
            "evidence_refs",
            "working_explanation",
            "remaining_risks",
            "recommended_followup",
        }
        required.add(
            "recommended_recipe_parameters"
            if _uses_named_physical_controls(self.plan_validator.task_info)
            else "recommended_search_vector"
        )
        if prediction_queries:
            required.add("counterfactual_predictions")
        missing = required - set(payload)
        if missing:
            raise ScientificPlanValidationError(
                "final synthesis response is missing required fields",
                field_path="final_synthesis_response",
                constraint="required_fields_present",
                observed=",".join(sorted(str(item) for item in missing)),
            )
        if set(payload) != required:
            raise ScientificPlanValidationError(
                "final synthesis response fields do not match its contract",
                field_path="final_synthesis_response",
                constraint="exact_declared_fields",
                observed=",".join(sorted(str(item) for item in set(payload) - required)),
            )
        if payload["schema_version"] != self.final_synthesis_version:
            raise ScientificPlanValidationError(
                "final synthesis schema_version does not match the frozen contract",
                field_path="schema_version",
                constraint="exact_schema_version",
                observed=str(payload["schema_version"]),
                limit=self.final_synthesis_version,
            )
        confidence = self._probability(payload["confidence"], field="confidence")
        plan_payload = {
            "experiment_intent": "validate the final recommendation",
            "requested_measurement_slots": payload["recommended_measurement_slots"],
            "measurement_objective": "blind independent validation",
            "expected_effect": "evaluate the submitted fixed-world method",
            "uncertainty": 1.0 - confidence,
        }
        if _uses_named_physical_controls(self.plan_validator.task_info):
            plan_payload["recipe_parameters"] = payload["recommended_recipe_parameters"]
        else:
            plan_payload["search_vector"] = payload["recommended_search_vector"]
        plan = self.plan_validator.validate(plan_payload)
        recommendation_type = str(payload["recommendation_type"])
        if recommendation_type not in self._RECOMMENDATION_TYPES:
            raise ScientificPlanValidationError(
                "recommendation_type is unsupported",
                field_path="recommendation_type",
                constraint="declared_enum",
            )
        raw_indices = payload["source_experiment_indices"]
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw_indices)
        ):
            raise ScientificPlanValidationError(
                "source_experiment_indices must be a non-empty integer list",
                field_path="source_experiment_indices",
                constraint="non_empty_integer_list",
            )
        source_indices = tuple(int(item) for item in raw_indices)
        available_indices = {int(item["experiment_index"]) for item in history}
        if not set(source_indices).issubset(available_indices):
            raise ScientificPlanValidationError(
                "source_experiment_indices contains an unknown experiment",
                field_path="source_experiment_indices",
                constraint="known_experiment_indices",
            )
        if recommendation_type == "tested":
            matching_source_exists = any(
                list(plan.search_vector)
                == [float(value) for value in item["plan"]["search_vector"]]
                and list(plan.requested_measurement_slots)
                == [str(value) for value in item["plan"]["requested_measurement_slots"]]
                for item in history
                if int(item["experiment_index"]) in source_indices
            )
            if not matching_source_exists:
                raise ScientificPlanValidationError(
                    "tested recommendation must reproduce at least one cited source method",
                    field_path=(
                        "recommended_recipe_parameters"
                        if _uses_named_physical_controls(self.plan_validator.task_info)
                        else "recommended_search_vector"
                    ),
                    constraint="matches_tested_source",
                )
        evidence_refs = self._string_list(payload["evidence_refs"], field="evidence_refs")
        if not set(evidence_refs).issubset(set(evidence_catalog)):
            raise ScientificPlanValidationError(
                "evidence_refs contains an unknown evidence ID",
                field_path="evidence_refs",
                constraint="known_evidence_ids",
            )
        explanation = payload["working_explanation"]
        explanation_fields = {
            "empirical_relationships",
            "mechanistic_hypothesis",
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "uncertainty",
        }
        if _uses_named_physical_controls(self.plan_validator.task_info):
            explanation_fields.add("structured_claims")
        missing_explanation_fields = (
            explanation_fields - set(explanation)
            if isinstance(explanation, Mapping)
            else explanation_fields
        )
        if not isinstance(explanation, Mapping) or missing_explanation_fields:
            raise ScientificPlanValidationError(
                "working_explanation is missing required fields",
                field_path="working_explanation",
                constraint="required_fields_present",
                observed=",".join(sorted(missing_explanation_fields)),
            )
        if set(explanation) != explanation_fields:
            raise ScientificPlanValidationError(
                "working_explanation fields do not match its contract",
                field_path="working_explanation",
                constraint="exact_declared_fields",
                observed=",".join(
                    sorted(str(item) for item in set(explanation) - explanation_fields)
                ),
            )
        supporting = self._string_list(
            explanation["supporting_evidence_ids"],
            field="working_explanation.supporting_evidence_ids",
        )
        contradicting = self._string_list(
            explanation["contradicting_evidence_ids"],
            field="working_explanation.contradicting_evidence_ids",
        )
        if not set(supporting + contradicting).issubset(set(evidence_catalog)):
            raise ScientificPlanValidationError(
                "working explanation cites an unknown evidence ID",
                field_path="working_explanation",
                constraint="known_evidence_ids",
            )
        normalized_explanation = {
            "empirical_relationships": self._string_list(
                explanation["empirical_relationships"],
                field="working_explanation.empirical_relationships",
            ),
            "mechanistic_hypothesis": StaticOptimizationValidator._text(
                explanation["mechanistic_hypothesis"],
                field="working_explanation.mechanistic_hypothesis",
                maximum=1200,
            ),
            "supporting_evidence_ids": supporting,
            "contradicting_evidence_ids": contradicting,
            "uncertainty": self._probability(
                explanation["uncertainty"],
                field="working_explanation.uncertainty",
            ),
        }
        if _uses_named_physical_controls(self.plan_validator.task_info):
            claim_arguments = {
                "evidence_catalog": evidence_catalog,
                "allowed_cause_variables": (
                    _ELECTROCHEMICAL_SINGLE_STAGE_CLAIM_CAUSES
                    if self.electrochemical_workflow_mode
                    == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
                    and self.recipe_kind == "electrochemical"
                    else _ELECTROCHEMICAL_CLAIM_CAUSES
                    if self.recipe_kind == "electrochemical"
                    else _CRYSTALLIZATION_CLAIM_CAUSES
                ),
                "allowed_effect_variables": (
                    _ELECTROCHEMICAL_CLAIM_EFFECTS
                    if self.recipe_kind == "electrochemical"
                    else _CRYSTALLIZATION_CLAIM_EFFECTS
                ),
                "allowed_mechanism_tags": (
                    _ELECTROCHEMICAL_MECHANISM_TAGS
                    if self.recipe_kind == "electrochemical"
                    else _CRYSTALLIZATION_MECHANISM_TAGS
                ),
            }
            if self.declared_claim_validation_policy == DECLARED_CLAIM_VALIDATION_UNSCORED_UNKNOWN:
                claims, claim_diagnostics = parse_world_understanding_claims_tolerant(
                    explanation["structured_claims"],
                    **claim_arguments,
                )
                normalized_explanation["structured_claim_diagnostics"] = claim_diagnostics
            else:
                try:
                    claims = parse_world_understanding_claims(
                        explanation["structured_claims"],
                        **claim_arguments,
                    )
                except ValueError as error:
                    raise ScientificPlanValidationError(
                        str(error),
                        field_path="working_explanation.structured_claims",
                        constraint="world_understanding_claim_contract",
                    ) from error
            normalized_explanation["structured_claims"] = [claim.to_dict() for claim in claims]
        counterfactual_predictions: tuple[dict[str, Any], ...] = ()
        if prediction_queries:
            try:
                parsed_predictions = parse_counterfactual_predictions(
                    payload["counterfactual_predictions"],
                    queries=prediction_queries,
                )
            except ValueError as error:
                raise ScientificPlanValidationError(
                    str(error),
                    field_path="counterfactual_predictions",
                    constraint="frozen_predictive_contract",
                ) from error
            counterfactual_predictions = tuple(
                prediction.to_dict() for prediction in parsed_predictions
            )
        return StaticFinalRecommendation(
            recommended_search_vector=plan.search_vector,
            recommended_measurement_slots=plan.requested_measurement_slots,
            recommendation_type=recommendation_type,
            source_experiment_indices=source_indices,
            predicted_score=self._probability(payload["predicted_score"], field="predicted_score"),
            confidence=confidence,
            method_summary=StaticOptimizationValidator._text(
                payload["method_summary"], field="method_summary", maximum=1000
            ),
            evidence_refs=tuple(evidence_refs),
            working_explanation=normalized_explanation,
            remaining_risks=tuple(
                self._string_list(payload["remaining_risks"], field="remaining_risks")
            ),
            recommended_followup=StaticOptimizationValidator._text(
                payload["recommended_followup"],
                field="recommended_followup",
                maximum=1000,
            ),
            recommended_recipe_parameters=plan.recipe_parameters,
            counterfactual_predictions=counterfactual_predictions,
            schema_version=self.final_synthesis_version,
        )


class StaticOptimizationAgent:
    name = "static_optimization_agent"

    def __init__(
        self,
        client: JsonPlannerClientLike,
        *,
        role_id: str,
        response_max_tokens: int,
        history_limit: int,
        prompt_token_estimate_cap: int,
        experiment_horizon: int | None = None,
        horizon_visible: bool = False,
        final_synthesis_enabled: bool = False,
        final_synthesis_prompt_token_estimate_cap: int | None = None,
        predictive_synthesis_prompt_token_estimate_cap: int | None = None,
        include_task_operation_budget: bool = True,
        predictive_world_understanding_enabled: bool = False,
        predictive_queries_in_final_synthesis: bool = True,
        declared_claim_validation_policy: str = DECLARED_CLAIM_VALIDATION_STRICT,
        material_information: Mapping[str, Any] | None = None,
        electrochemical_material_family_id: str | None = None,
        crystallization_material_family_id: str | None = None,
        scoring_contract: Mapping[str, Any] | None = None,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
        optimization_scaffold_id: str | None = None,
    ) -> None:
        self.client = client
        self.role_id = role_id
        self.response_max_tokens = int(response_max_tokens)
        self.history_limit = int(history_limit)
        self.prompt_token_estimate_cap = int(prompt_token_estimate_cap)
        self.experiment_horizon = (
            int(experiment_horizon) if experiment_horizon is not None else None
        )
        self.horizon_visible = bool(horizon_visible)
        self.final_synthesis_enabled = bool(final_synthesis_enabled)
        self.final_synthesis_prompt_token_estimate_cap = int(
            final_synthesis_prompt_token_estimate_cap
            if final_synthesis_prompt_token_estimate_cap is not None
            else prompt_token_estimate_cap
        )
        self.predictive_synthesis_prompt_token_estimate_cap = int(
            predictive_synthesis_prompt_token_estimate_cap
            if predictive_synthesis_prompt_token_estimate_cap is not None
            else self.final_synthesis_prompt_token_estimate_cap
        )
        self.include_task_operation_budget = bool(include_task_operation_budget)
        self.predictive_world_understanding_enabled = bool(predictive_world_understanding_enabled)
        self.predictive_queries_in_final_synthesis = bool(predictive_queries_in_final_synthesis)
        if declared_claim_validation_policy not in DECLARED_CLAIM_VALIDATION_POLICIES:
            raise ValueError("unknown Declared claim validation policy")
        self.declared_claim_validation_policy = declared_claim_validation_policy
        if self.declared_claim_validation_policy == DECLARED_CLAIM_VALIDATION_UNSCORED_UNKNOWN:
            self.final_synthesis_version = STATIC_FINAL_SYNTHESIS_TOLERANT_DECLARED_VERSION
        else:
            self.final_synthesis_version = (
                STATIC_FINAL_SYNTHESIS_SEPARATE_VERSION
                if self.predictive_world_understanding_enabled
                and not self.predictive_queries_in_final_synthesis
                else STATIC_FINAL_SYNTHESIS_VERSION
            )
        self.material_information_config = (
            None if material_information is None else copy.deepcopy(dict(material_information))
        )
        self.electrochemical_material_family_id = electrochemical_material_family_id
        self.crystallization_material_family_id = crystallization_material_family_id
        self.scoring_contract = (
            None if scoring_contract is None else copy.deepcopy(dict(scoring_contract))
        )
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.optimization_scaffold_id = str(
            optimization_scaffold_id or DIRECT_OPTIMIZATION_SCAFFOLD_ID
        )
        coverage_adaptive = (
            self.optimization_scaffold_id
            in COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_IDS
        )
        surrogate_portfolio_v17 = (
            self.optimization_scaffold_id
            == COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V17_ID
        )
        nonduplicate_surrogate_portfolio = (
            self.optimization_scaffold_id
            == COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID
        )
        scheduled_surrogate_portfolio = (
            self.optimization_scaffold_id
            == COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID
        )
        self.prompt_version = (
            STATIC_OPTIMIZATION_SCHEDULED_PORTFOLIO_PROMPT_VERSION
            if scheduled_surrogate_portfolio
            else STATIC_OPTIMIZATION_NONDUPLICATE_PORTFOLIO_PROMPT_VERSION
            if nonduplicate_surrogate_portfolio
            else STATIC_OPTIMIZATION_PORTFOLIO_PROMPT_VERSION
            if surrogate_portfolio_v17
            else STATIC_OPTIMIZATION_COVERAGE_PROMPT_VERSION
            if coverage_adaptive
            else STATIC_OPTIMIZATION_PROMPT_VERSION
        )
        self.experiment_system_prompt = (
            SCHEDULED_SURROGATE_PORTFOLIO_SYSTEM_PROMPT
            if scheduled_surrogate_portfolio
            else NONDUPLICATE_SURROGATE_PORTFOLIO_SYSTEM_PROMPT
            if nonduplicate_surrogate_portfolio
            else SURROGATE_PORTFOLIO_SYSTEM_PROMPT
            if surrogate_portfolio_v17
            else COVERAGE_ADAPTIVE_SYSTEM_PROMPT
            if coverage_adaptive
            else SYSTEM_PROMPT
        )
        self.final_synthesis_system_prompt = (
            SCHEDULED_SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT
            if scheduled_surrogate_portfolio
            else NONDUPLICATE_SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT
            if nonduplicate_surrogate_portfolio
            else SURROGATE_PORTFOLIO_FINAL_SYNTHESIS_SYSTEM_PROMPT
            if surrogate_portfolio_v17
            else COVERAGE_ADAPTIVE_FINAL_SYNTHESIS_SYSTEM_PROMPT
            if coverage_adaptive
            else FINAL_SYNTHESIS_SYSTEM_PROMPT
        )
        self.resource_ledger = ResourceLedger()
        self._last_audit: dict[str, Any] | None = None
        self._last_synthesis_audit: dict[str, Any] | None = None
        self._last_predictive_audit: dict[str, Any] | None = None

    def reset(self, task_info: Mapping[str, Any], seed: int) -> None:
        self.task_info = dict(task_info)
        self.seed = int(seed)
        if self.predictive_world_understanding_enabled and task_recipe_kind(self.task_info) not in {
            "electrochemical",
            "reaction_crystallization",
        }:
            raise ValueError(
                "predictive world understanding is frozen only for the two confirmatory tasks"
            )
        self.context_builder = StaticOptimizationContextBuilder(
            self.task_info,
            history_limit=self.history_limit,
            total_experiments=(self.experiment_horizon if self.horizon_visible else None),
            final_synthesis_after_exploration=self.final_synthesis_enabled,
            include_task_operation_budget=self.include_task_operation_budget,
            predictive_world_understanding_enabled=(self.predictive_world_understanding_enabled),
            material_information=self.material_information_config,
            electrochemical_material_family_id=(self.electrochemical_material_family_id),
            crystallization_material_family_id=(self.crystallization_material_family_id),
            scoring_contract=self.scoring_contract,
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
            optimization_scaffold_id=self.optimization_scaffold_id,
            algorithm_seed=self.seed,
        )
        self.validator = StaticOptimizationValidator(
            self.task_info,
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
        )
        self.final_validator = StaticFinalRecommendationValidator(
            self.task_info,
            predictive_world_understanding_enabled=(self.predictive_world_understanding_enabled),
            final_synthesis_version=self.final_synthesis_version,
            declared_claim_validation_policy=(self.declared_claim_validation_policy),
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
        )
        self.resource_ledger.reset()
        self._last_audit = None
        self._last_synthesis_audit = None
        self._last_predictive_audit = None

    def public_context(self, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not hasattr(self, "context_builder"):
            raise RuntimeError("StaticOptimizationAgent must be reset before use")
        return self.context_builder.build(history)

    def final_synthesis_context(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        include_prediction_queries: bool | None = None,
    ) -> dict[str, Any]:
        if not self.final_synthesis_enabled:
            raise RuntimeError("final synthesis is not enabled for this S0 method")
        if self.experiment_horizon is not None and len(history) != self.experiment_horizon:
            raise RuntimeError("final synthesis requires the complete exploration history")
        if include_prediction_queries is None:
            include_prediction_queries = (
                self.predictive_world_understanding_enabled
                and self.predictive_queries_in_final_synthesis
            )
        return self.context_builder.build(
            history,
            decision_stage="final_synthesis",
            include_prediction_queries=include_prediction_queries,
        )

    def plan_next(self, history: Sequence[Mapping[str, Any]]) -> StaticOptimizationPlan:
        context = self.public_context(history)
        context_sha256 = canonical_sha256(context)
        coverage_design_enforced = bool(
            self.optimization_scaffold_id
            in COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_IDS
            and len(history) < COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS
        )
        scaffold = context.get("campaign_scaffold")
        raw_candidate_portfolio = (
            scaffold.get("model_candidate_portfolio")
            if isinstance(scaffold, Mapping)
            else None
        )
        candidate_portfolio = (
            list(raw_candidate_portfolio)
            if isinstance(raw_candidate_portfolio, list)
            and all(isinstance(item, Mapping) for item in raw_candidate_portfolio)
            else []
        )
        portfolio_selection_enforced = bool(candidate_portfolio)
        scheduled_portfolio_enforced = bool(
            isinstance(scaffold, Mapping)
            and scaffold.get("scaffold_id")
            == COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID
            and isinstance(scaffold.get("executor_committed_candidate_id"), str)
        )
        executor_recipe_enforced = bool(
            coverage_design_enforced or scheduled_portfolio_enforced
        )
        recipe_field = (
            {
                "recipe_parameters": {
                    key: copy.deepcopy(value)
                    for key, value in _recipe_parameter_schema(
                        self.task_info, self.electrochemical_workflow_mode
                    ).items()
                }
            }
            if _uses_named_physical_controls(self.task_info)
            else {
                "search_vector": [
                    "number in [0,1]"
                    for _ in range(
                        _recipe_dimension(self.task_info, self.electrochemical_workflow_mode)
                    )
                ]
            }
        )
        required_recipe_field = {} if executor_recipe_enforced else recipe_field
        required_candidate_field = (
            {"candidate_id": "one exact candidate_id listed in model_candidate_portfolio"}
            if portfolio_selection_enforced
            else {}
        )
        prompt_payload = {
            "schema_version": self.prompt_version,
            "public_experiment_context": context,
            "public_context_sha256": context_sha256,
            "required_json_shape": {
                "experiment_intent": "string",
                **required_recipe_field,
                **required_candidate_field,
                "requested_measurement_slots": ["public diagnostic slot ID"],
                "measurement_objective": "string",
                "expected_effect": "string",
                "uncertainty": "number in [0,1]",
            },
        }
        if executor_recipe_enforced:
            forbidden_json_fields = [
                "candidate_id",
                "recipe_parameters",
                "search_vector",
            ]
            if coverage_design_enforced:
                forbidden_json_fields.remove("candidate_id")
            prompt_payload["forbidden_json_fields"] = forbidden_json_fields
        prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_estimated_tokens = estimate_prompt_tokens(prompt)
        if prompt_estimated_tokens > self.prompt_token_estimate_cap:
            raise PromptBudgetExceededError(
                f"static optimization prompt estimate {prompt_estimated_tokens} exceeds cap "
                f"{self.prompt_token_estimate_cap}"
            )
        try:
            completion = self.client.complete_json(
                system_prompt=self.experiment_system_prompt,
                user_prompt=prompt,
                max_tokens=self.response_max_tokens,
            )
        except Exception as error:
            self.resource_ledger.record_failure(error)
            raise
        self.resource_ledger.record_completion(completion)
        validated_payload = completion.payload
        model_supplied_recipe_ignored = False
        model_supplied_candidate_id_ignored = False
        selected_candidate_id: str | None = None
        if executor_recipe_enforced and isinstance(completion.payload, Mapping):
            validated_payload = copy.deepcopy(dict(completion.payload))
            model_supplied_recipe_ignored = any(
                field in validated_payload
                for field in ("recipe_parameters", "search_vector")
            )
            model_supplied_candidate_id_ignored = bool(
                scheduled_portfolio_enforced
                and "candidate_id" in validated_payload
            )
            if scheduled_portfolio_enforced:
                validated_payload.pop("candidate_id", None)
            validated_payload.pop("recipe_parameters", None)
            validated_payload.pop("search_vector", None)
            scaffold = context["campaign_scaffold"]
            if scheduled_portfolio_enforced:
                selected_candidate_id = str(
                    scaffold["executor_committed_candidate_id"]
                )
            if _uses_named_physical_controls(self.task_info):
                validated_payload["recipe_parameters"] = copy.deepcopy(
                    scaffold["executor_committed_recipe_parameters"]
                )
            else:
                validated_payload["search_vector"] = copy.deepcopy(
                    scaffold["executor_committed_search_vector"]
                )
        elif portfolio_selection_enforced:
            if not isinstance(completion.payload, Mapping):
                raise ScientificPlanValidationError(
                    "portfolio response must be an object",
                    field_path="static_optimization_response",
                    constraint="object",
                )
            validated_payload = copy.deepcopy(dict(completion.payload))
            raw_candidate_id = validated_payload.pop("candidate_id", None)
            if not isinstance(raw_candidate_id, str) or not raw_candidate_id:
                raise ScientificPlanValidationError(
                    "candidate_id must identify one displayed portfolio candidate",
                    field_path="candidate_id",
                    constraint="known_candidate_id",
                )
            selected_candidate_id = raw_candidate_id
        plan = self.validator.validate(validated_payload)
        if portfolio_selection_enforced:
            selected_candidate = next(
                (
                    item
                    for item in candidate_portfolio
                    if item.get("candidate_id") == selected_candidate_id
                ),
                None,
            )
            if selected_candidate is None:
                raise ScientificPlanValidationError(
                    "candidate_id is not in the displayed portfolio",
                    field_path="candidate_id",
                    constraint="known_candidate_id",
                    observed=selected_candidate_id,
                )
            expected_vector = (
                _vector_from_parameters(
                    self.task_info,
                    self.electrochemical_workflow_mode,
                    selected_candidate.get("recipe_parameters"),
                )
                if _uses_named_physical_controls(self.task_info)
                else np.asarray(selected_candidate.get("search_vector"), dtype=float)
            )
            if not np.allclose(
                np.asarray(plan.search_vector, dtype=float),
                expected_vector,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ScientificPlanValidationError(
                    "returned recipe does not match the selected portfolio candidate",
                    field_path=(
                        "recipe_parameters"
                        if _uses_named_physical_controls(self.task_info)
                        else "search_vector"
                    ),
                    constraint="matches_selected_candidate_id",
                    observed=selected_candidate_id,
                )
        self._last_audit = {
            "schema_version": STATIC_OPTIMIZATION_INTERFACE_VERSION,
            "role_id": self.role_id,
            "public_context_sha256": context_sha256,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "model_completion_payload_sha256": canonical_sha256(completion.payload),
            "provider_model": str(completion.model),
            "provider_attempts": int(completion.attempts),
            "provider_usage": copy.deepcopy(to_builtin(completion.usage)),
            "model_call_consumed": True,
            "prompt_estimated_tokens": prompt_estimated_tokens,
            "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
            "static_world_assumed": True,
            "hidden_world_fields_supplied": False,
            "coverage_design_enforced": coverage_design_enforced,
            "coverage_design_experiment_index": (
                len(history) if coverage_design_enforced else None
            ),
            "recipe_selection_authority": (
                "protocol_executor"
                if executor_recipe_enforced
                else "model"
            ),
            "portfolio_selection_enforced": portfolio_selection_enforced,
            "portfolio_candidate_id": selected_candidate_id,
            "portfolio_candidate_generation_authority": (
                "protocol_executor_using_public_history_only"
                if portfolio_selection_enforced or scheduled_portfolio_enforced
                else None
            ),
            "portfolio_candidate_selection_authority": (
                "protocol_executor_task_neutral_schedule"
                if scheduled_portfolio_enforced
                else "model"
                if portfolio_selection_enforced
                else None
            ),
            "portfolio_hidden_world_fields_used": (
                False
                if portfolio_selection_enforced or scheduled_portfolio_enforced
                else None
            ),
            "model_supplied_recipe_ignored": model_supplied_recipe_ignored,
            "material_information_condition": (self.context_builder.material_information_condition),
            "material_information_sha256": (self.context_builder.material_information_sha256),
            "scoring_contract": copy.deepcopy(self.scoring_contract),
        }
        if scheduled_portfolio_enforced:
            self._last_audit.update(
                {
                    "scheduled_portfolio_candidate_enforced": True,
                    "model_supplied_candidate_id_ignored": (
                        model_supplied_candidate_id_ignored
                    ),
                }
            )
        return plan

    def synthesize_final(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        include_prediction_queries: bool | None = None,
    ) -> StaticFinalRecommendation:
        if include_prediction_queries is None:
            include_prediction_queries = (
                self.predictive_world_understanding_enabled
                and self.predictive_queries_in_final_synthesis
            )
        context = self.final_synthesis_context(
            history,
            include_prediction_queries=include_prediction_queries,
        )
        context_sha256 = canonical_sha256(context)
        dimension = _recipe_dimension(self.task_info, self.electrochemical_workflow_mode)
        recipe_kind = task_recipe_kind(self.task_info)
        electrochemical = recipe_kind == "electrochemical"
        named_controls = _uses_named_physical_controls(self.task_info)
        prediction_queries = (
            (
                build_electrochemical_prediction_queries(
                    history,
                    electrochemical_workflow_mode=self.electrochemical_workflow_mode,
                )
                if electrochemical
                else build_crystallization_prediction_queries(history)
            )
            if include_prediction_queries
            else ()
        )
        if prediction_queries:
            public_queries = [query.to_public_dict() for query in prediction_queries]
            if context.get("held_out_prediction_queries") != public_queries:
                raise RuntimeError("predictive query regeneration does not match final context")
        recommendation_field: dict[str, object] = (
            {
                "recommended_recipe_parameters": {
                    key: copy.deepcopy(value)
                    for key, value in _recipe_parameter_schema(
                        self.task_info, self.electrochemical_workflow_mode
                    ).items()
                }
            }
            if named_controls
            else {"recommended_search_vector": ["number in [0,1]" for _ in range(dimension)]}
        )
        structured_claim_shape: dict[str, object] = {
            "structured_claims": [
                {
                    "claim_id": "string",
                    "cause_variables": ["declared public cause variable"],
                    "effect_variable": "declared public effect variable",
                    "relation": ("positive|negative|nonmonotonic|conditional|no_direct_effect"),
                    "mechanism_tags": ["declared public mechanism tag"],
                    "scope": "string",
                    "evidence_ids": ["public evidence ID"],
                    "confidence": "number in [0,1]",
                }
            ]
        }
        counterfactual_prediction_shape: dict[str, object] = (
            {
                "counterfactual_predictions": [
                    {
                        "query_id": query.query_id,
                        "metric_predictions": [
                            {
                                "metric_id": metric_id,
                                "direction": "increase|decrease|no_material_change",
                                "confidence": "number in [0,1]",
                            }
                            for metric_id in query.metric_ids
                        ],
                    }
                    for query in prediction_queries
                ]
            }
            if prediction_queries
            else {}
        )
        working_explanation_shape: dict[str, object] = {
            "empirical_relationships": ["at most 16 strings"],
            "mechanistic_hypothesis": "string",
            "supporting_evidence_ids": ["at most 16 public evidence IDs"],
            "contradicting_evidence_ids": ["at most 16 public evidence IDs"],
            "uncertainty": "number in [0,1]",
        }
        if named_controls:
            working_explanation_shape.update(structured_claim_shape)
        required_json_shape: dict[str, object] = {
            "schema_version": self.final_synthesis_version,
        }
        required_json_shape.update(recommendation_field)
        required_json_shape.update(
            {
                "recommended_measurement_slots": ["public diagnostic slot ID"],
                "recommendation_type": "tested|interpolated|extrapolated",
                "source_experiment_indices": ["integer experiment index"],
                "predicted_score": "number in [0,1]",
                "confidence": "number in [0,1]",
                "method_summary": "string",
                "evidence_refs": ["at most 16 public evidence IDs"],
                "working_explanation": working_explanation_shape,
                "remaining_risks": ["at most 16 strings"],
                "recommended_followup": "string",
            }
        )
        required_json_shape.update(counterfactual_prediction_shape)
        prompt_payload: dict[str, object] = {
            "schema_version": self.final_synthesis_version,
            "public_final_synthesis_context": context,
            "public_context_sha256": context_sha256,
        }
        if self.predictive_world_understanding_enabled and not include_prediction_queries:
            prompt_payload["forbidden_json_fields"] = ["counterfactual_predictions"]
        prompt_payload["required_json_shape"] = required_json_shape
        prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_estimated_tokens = estimate_prompt_tokens(prompt)
        if prompt_estimated_tokens > self.final_synthesis_prompt_token_estimate_cap:
            raise PromptBudgetExceededError(
                "static final synthesis prompt estimate "
                f"{prompt_estimated_tokens} exceeds cap "
                f"{self.final_synthesis_prompt_token_estimate_cap}"
            )
        try:
            completion = self.client.complete_json(
                system_prompt=(
                    FINAL_SYNTHESIS_SEPARATE_PREDICTIVE_SYSTEM_PROMPT
                    if self.predictive_world_understanding_enabled
                    and not include_prediction_queries
                    else self.final_synthesis_system_prompt
                ),
                user_prompt=prompt,
                max_tokens=self.response_max_tokens,
            )
        except Exception as error:
            self.resource_ledger.record_failure(error)
            raise
        self.resource_ledger.record_completion(completion)
        evidence_catalog = [
            str(entry["evidence_id"])
            for record in history
            for entry in record["measurement_evidence"]
        ]
        recommendation = self.final_validator.validate(
            completion.payload,
            history=history,
            evidence_catalog=evidence_catalog,
            prediction_queries=prediction_queries,
        )
        self._last_synthesis_audit = {
            "schema_version": self.final_synthesis_version,
            "role_id": self.role_id,
            "public_context_sha256": context_sha256,
            "recommendation_sha256": canonical_sha256(recommendation.to_dict()),
            "provider_model": str(completion.model),
            "provider_attempts": int(completion.attempts),
            "provider_usage": copy.deepcopy(to_builtin(completion.usage)),
            "prompt_estimated_tokens": prompt_estimated_tokens,
            "prompt_token_estimate_cap": (self.final_synthesis_prompt_token_estimate_cap),
            "static_world_assumed": True,
            "validation_feedback_returned_to_agent": False,
            "predictive_world_understanding_enabled": bool(prediction_queries),
            "predictive_queries_visible": bool(prediction_queries),
            "forbidden_json_fields": (
                ["counterfactual_predictions"]
                if self.predictive_world_understanding_enabled and not bool(prediction_queries)
                else []
            ),
            "recommendation_committed_before_predictive_query_visibility": (
                self.predictive_world_understanding_enabled and not bool(prediction_queries)
            ),
            "predictive_query_sha256": [query.query_sha256 for query in prediction_queries],
            "predictive_query_set_sha256": (
                canonical_sha256([query.to_public_dict() for query in prediction_queries])
                if prediction_queries
                else None
            ),
            "material_information_condition": (self.context_builder.material_information_condition),
            "material_information_sha256": (self.context_builder.material_information_sha256),
        }
        return recommendation

    def predict_counterfactuals(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        prediction_queries: Sequence[
            ElectrochemicalPredictionQuery | CrystallizationPredictionQuery
        ],
        committed_recommendation_sha256: str,
    ) -> tuple[dict[str, Any], ...]:
        if not self.predictive_world_understanding_enabled:
            raise RuntimeError("predictive world understanding is not enabled")
        if self.predictive_queries_in_final_synthesis:
            raise RuntimeError("predictive-only call is disabled for the integrated policy")
        if not prediction_queries:
            raise ValueError("predictive-only call requires a frozen query set")
        if (
            not isinstance(committed_recommendation_sha256, str)
            or len(committed_recommendation_sha256) != 64
        ):
            raise ValueError("committed recommendation requires a SHA256 digest")
        context = self.final_synthesis_context(
            history,
            include_prediction_queries=False,
        )
        if "held_out_prediction_queries" in context:
            raise RuntimeError("final recommendation context leaked predictive queries")
        public_queries = [query.to_public_dict() for query in prediction_queries]
        predictive_context = {
            "public_campaign_context": context,
            "held_out_prediction_queries": public_queries,
            "prediction_contract": {
                "recommendation_committed_before_query_visibility": True,
                "prediction_call_can_modify_recommendation": False,
                "feedback_returned_to_agent": False,
            },
        }
        prompt_payload = {
            "schema_version": STATIC_PREDICTIVE_SYNTHESIS_VERSION,
            "public_predictive_context": predictive_context,
            "public_context_sha256": canonical_sha256(predictive_context),
            "required_json_shape": {
                "schema_version": STATIC_PREDICTIVE_SYNTHESIS_VERSION,
                "counterfactual_predictions": [
                    {
                        "query_id": query.query_id,
                        "metric_predictions": [
                            {
                                "metric_id": metric_id,
                                "direction": "increase|decrease|no_material_change",
                                "confidence": "number in [0,1]",
                            }
                            for metric_id in query.metric_ids
                        ],
                    }
                    for query in prediction_queries
                ],
            },
        }
        prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_estimated_tokens = estimate_prompt_tokens(prompt)
        if prompt_estimated_tokens > self.predictive_synthesis_prompt_token_estimate_cap:
            raise PromptBudgetExceededError(
                "static predictive synthesis prompt estimate "
                f"{prompt_estimated_tokens} exceeds cap "
                f"{self.predictive_synthesis_prompt_token_estimate_cap}"
            )
        try:
            completion = self.client.complete_json(
                system_prompt=PREDICTIVE_SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=self.response_max_tokens,
            )
        except Exception as error:
            self.resource_ledger.record_failure(error)
            raise
        self.resource_ledger.record_completion(completion)
        payload = completion.payload
        required_fields = {"schema_version", "counterfactual_predictions"}
        if not isinstance(payload, Mapping) or set(payload) != required_fields:
            raise ScientificPlanValidationError(
                "predictive-only response fields do not match its contract",
                field_path="predictive_synthesis_response",
                constraint="exact_declared_fields",
            )
        if payload["schema_version"] != STATIC_PREDICTIVE_SYNTHESIS_VERSION:
            raise ScientificPlanValidationError(
                "predictive-only schema_version does not match the frozen contract",
                field_path="schema_version",
                constraint="exact_schema_version",
                observed=str(payload["schema_version"]),
                limit=STATIC_PREDICTIVE_SYNTHESIS_VERSION,
            )
        try:
            parsed = parse_counterfactual_predictions(
                payload["counterfactual_predictions"],
                queries=prediction_queries,
            )
        except ValueError as error:
            raise ScientificPlanValidationError(
                str(error),
                field_path="counterfactual_predictions",
                constraint="frozen_predictive_contract",
            ) from error
        normalized = tuple(prediction.to_dict() for prediction in parsed)
        self._last_predictive_audit = {
            "schema_version": STATIC_PREDICTIVE_SYNTHESIS_VERSION,
            "role_id": self.role_id,
            "public_context_sha256": canonical_sha256(predictive_context),
            "query_sha256": [query.query_sha256 for query in prediction_queries],
            "query_set_sha256": canonical_sha256(public_queries),
            "predictions_sha256": canonical_sha256(list(normalized)),
            "committed_recommendation_sha256": committed_recommendation_sha256,
            "recommendation_visible_to_prediction_call": False,
            "recommendation_committed_before_query_visibility": True,
            "prediction_call_can_modify_recommendation": False,
            "provider_model": str(completion.model),
            "provider_attempts": int(completion.attempts),
            "provider_usage": copy.deepcopy(to_builtin(completion.usage)),
            "prompt_estimated_tokens": prompt_estimated_tokens,
            "prompt_token_estimate_cap": (self.predictive_synthesis_prompt_token_estimate_cap),
        }
        return normalized

    def decision_audit(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_audit)

    def synthesis_audit(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_synthesis_audit)

    def predictive_audit(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_predictive_audit)

    def manifest(self) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "agent_family": type(self).__name__,
            "role_id": self.role_id,
            "seed": self.seed,
            "provider_model": self.client.model,
            "decision_scope": "complete_experiment",
            "interface_version": STATIC_OPTIMIZATION_INTERFACE_VERSION,
            "prompt_version": self.prompt_version,
            "optimization_scaffold_id": self.optimization_scaffold_id,
            "history_limit": self.history_limit,
            "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
            "experiment_horizon": self.experiment_horizon,
            "horizon_visible": self.horizon_visible,
            "final_synthesis_enabled": self.final_synthesis_enabled,
            "final_synthesis_version": self.final_synthesis_version,
            "declared_claim_validation_policy": (self.declared_claim_validation_policy),
            "final_synthesis_prompt_token_estimate_cap": (
                self.final_synthesis_prompt_token_estimate_cap
            ),
            "predictive_synthesis_prompt_token_estimate_cap": (
                self.predictive_synthesis_prompt_token_estimate_cap
            ),
            "predictive_world_understanding_enabled": (self.predictive_world_understanding_enabled),
            "predictive_queries_in_final_synthesis": (self.predictive_queries_in_final_synthesis),
            "electrochemical_workflow_mode": self.electrochemical_workflow_mode,
            "material_information_condition": (self.context_builder.material_information_condition),
            "material_information_sha256": (self.context_builder.material_information_sha256),
            "scoring_contract": copy.deepcopy(self.scoring_contract),
            "mechanical_closeout": True,
            "static_world": True,
            "hidden_world_fields_supplied": False,
        }

    def method_resource_usage(self) -> dict[str, Any]:
        return self.resource_ledger.snapshot(self.client)


def compile_static_optimization_plan(
    task_info: Mapping[str, Any],
    plan: StaticOptimizationPlan,
    *,
    electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
) -> dict[str, Any]:
    workflow_mode = normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
    recipe = _recipe_from_vector(
        task_info,
        workflow_mode,
        np.asarray(plan.search_vector, dtype=float),
    )
    available = {str(item["slot_id"]) for item in _measurement_slots(task_info, workflow_mode)}
    requested = set(plan.requested_measurement_slots)
    if not requested.issubset(available):
        raise ValueError("static plan requests an unknown diagnostic slot")
    required = {
        str(item["slot_id"])
        for item in _measurement_slots(task_info, workflow_mode)
        if item.get("selection_policy") == "required_by_workflow"
    }
    if not required.issubset(requested):
        raise ValueError("static plan omits a workflow-required diagnostic slot")
    steps: list[dict[str, Any]] = []
    slots_by_step: dict[str, str] = {}
    diagnostic_index = 0
    for action in recipe["steps"]:
        operation = action.get("operation")
        instrument = action.get("instrument")
        if operation == "measure" and instrument != "final_assay":
            diagnostic_index += 1
            slot_id = f"diagnostic-{diagnostic_index:02d}-{instrument}"
            if slot_id not in requested:
                continue
            slots_by_step[str(len(steps))] = slot_id
        elif operation == "measure" and instrument == "final_assay":
            slots_by_step[str(len(steps))] = "closeout-final-assay"
        steps.append(copy.deepcopy(action))
    metadata = copy.deepcopy(recipe["metadata"])
    metadata.update(
        {
            "static_optimization_interface_version": STATIC_OPTIMIZATION_INTERFACE_VERSION,
            "requested_measurement_slots": list(plan.requested_measurement_slots),
            "measurement_slots_by_step": slots_by_step,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "closeout_policy": "recipe_compiler_terminate_then_final_assay",
            "static_world": True,
            "electrochemical_workflow_mode": workflow_mode,
        }
    )
    return {"steps": steps, "metadata": metadata}


__all__ = [
    "COVERAGE_ADAPTIVE_INITIAL_EXPERIMENTS",
    "COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_ID",
    "COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_IDS",
    "COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V16_ID",
    "COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V17_ID",
    "COVERAGE_ADAPTIVE_OPTIMIZATION_SCAFFOLD_V18_ID",
    "COVERAGE_ADAPTIVE_SYSTEM_PROMPT",
    "DECLARED_CLAIM_VALIDATION_STRICT",
    "DECLARED_CLAIM_VALIDATION_UNSCORED_UNKNOWN",
    "DIRECT_OPTIMIZATION_SCAFFOLD_ID",
    "FINAL_SYNTHESIS_SEPARATE_PREDICTIVE_SYSTEM_PROMPT",
    "FINAL_SYNTHESIS_SYSTEM_PROMPT",
    "PREDICTIVE_SYNTHESIS_SYSTEM_PROMPT",
    "STATIC_FINAL_SYNTHESIS_SEPARATE_VERSION",
    "STATIC_FINAL_SYNTHESIS_TOLERANT_DECLARED_VERSION",
    "STATIC_FINAL_SYNTHESIS_VERSION",
    "STATIC_OPTIMIZATION_COVERAGE_PROMPT_VERSION",
    "STATIC_OPTIMIZATION_INTERFACE_VERSION",
    "STATIC_OPTIMIZATION_NONDUPLICATE_PORTFOLIO_PROMPT_VERSION",
    "STATIC_OPTIMIZATION_PORTFOLIO_PROMPT_VERSION",
    "STATIC_OPTIMIZATION_PROMPT_VERSION",
    "STATIC_OPTIMIZATION_SCHEDULED_PORTFOLIO_PROMPT_VERSION",
    "STATIC_PREDICTIVE_SYNTHESIS_VERSION",
    "StaticFinalRecommendation",
    "StaticFinalRecommendationValidator",
    "StaticOptimizationAgent",
    "StaticOptimizationContextBuilder",
    "StaticOptimizationPlan",
    "StaticOptimizationValidator",
    "compile_static_optimization_plan",
]
