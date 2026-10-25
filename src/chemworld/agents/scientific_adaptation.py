"""Experiment-level scientific adaptation components.

This development interface separates scientific experiment selection from the
operation-level procedure controller. Direct and stateful methods share one
public context builder and one response validator; the scaffold difference is
confined to agent-authored persistent memory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from chemworld.agents.prompt_context import (
    PromptBudgetExceededError,
    estimate_prompt_tokens,
)
from chemworld.agents.task_recipes import (
    TASK_RECIPE_SPACE_VERSION,
    task_recipe_categorical_coordinates,
    task_recipe_coordinate_schema,
    task_recipe_dimension,
    task_recipe_from_unit_vector,
    task_recipe_kind,
)
from chemworld.data.logging import to_builtin
from chemworld.legacy.agents.live_llm import JsonCompletionLike, JsonPlannerClientLike

SCIENTIFIC_ADAPTATION_INTERFACE_VERSION = "chemworld-scientific-adaptation-interface-0.2-dev"
SCIENTIFIC_ADAPTATION_PROMPT_VERSION = "chemworld-scientific-adaptation-prompt-0.2-dev"
SCIENTIFIC_MEMORY_VERSION = "chemworld-bounded-scientific-memory-0.1-dev"

SYSTEM_PROMPT = """You are an experiment-level scientific agent in ChemWorld.
Choose one complete experiment in the public unit-vector recipe space. The deterministic
executor will run the selected conditions and measurement slots, then mechanically perform
terminate and final_assay. Use only the supplied public history and candidate definitions.
Do not claim hidden truth or expose private chain-of-thought. Return exactly one JSON object.
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
_PUBLIC_PLAN_KEYS = (
    "search_vector",
    "requested_measurement_slots",
    "mechanism_distribution",
    "uncertainty",
)
_PUBLIC_TERMINAL_SUMMARY_KEYS = (
    "leaderboard_score",
    "cost",
    "safety_risk",
)
_RELIABILITY_LEVELS = frozenset({"low", "medium", "high"})


class ScientificPlanValidationError(ValueError):
    """Model-response validation failure with content-free diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str,
        constraint: str,
        observed: int | float | str | None = None,
        limit: int | float | str | None = None,
    ) -> None:
        super().__init__(message)
        diagnostics: dict[str, int | float | str] = {
            "field_path": str(field_path),
            "constraint": str(constraint),
        }
        if observed is not None:
            diagnostics["observed"] = observed
        if limit is not None:
            diagnostics["limit"] = limit
        self.validation_diagnostics = diagnostics


def canonical_sha256(value: Any) -> str:
    """Return a stable digest for JSON-compatible public contracts."""

    return hashlib.sha256(
        json.dumps(
            to_builtin(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _required_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ScientificPlanValidationError(
            f"{field} must be a non-empty string",
            field_path=field,
            constraint="non_empty_string",
            observed=type(value).__name__,
        )
    if not value.strip():
        raise ScientificPlanValidationError(
            f"{field} must be a non-empty string",
            field_path=field,
            constraint="non_empty_string",
            observed=0,
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ScientificPlanValidationError(
            f"{field} keys do not match its declared schema",
            field_path=field,
            constraint="exact_declared_keys",
            observed=len(observed),
            limit=len(expected),
        )


def _probability_distribution(
    value: Any,
    *,
    candidate_ids: Sequence[str],
    field: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ScientificPlanValidationError(
            f"{field} must be an object",
            field_path=field,
            constraint="object",
            observed=type(value).__name__,
        )
    expected = set(candidate_ids)
    if set(value) != expected:
        raise ScientificPlanValidationError(
            f"{field} must contain exactly the public candidate IDs",
            field_path=field,
            constraint="exact_candidate_keys",
            observed=len(value),
            limit=len(expected),
        )
    result: dict[str, float] = {}
    for candidate_id in candidate_ids:
        probability = value[candidate_id]
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ScientificPlanValidationError(
                f"{field}.{candidate_id} must be numeric",
                field_path=f"{field}.{candidate_id}",
                constraint="numeric",
                observed=type(probability).__name__,
            )
        probability_float = float(probability)
        if not math.isfinite(probability_float) or not 0.0 <= probability_float <= 1.0:
            raise ScientificPlanValidationError(
                f"{field}.{candidate_id} must be finite and in [0, 1]",
                field_path=f"{field}.{candidate_id}",
                constraint="finite_probability",
            )
        result[candidate_id] = probability_float
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ScientificPlanValidationError(
            f"{field} must sum to one",
            field_path=field,
            constraint="probability_sum_one",
        )
    return result


def scientific_measurement_slots(task_info: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Describe the fixed diagnostic positions in the public recipe adapter."""

    task = dict(task_info)
    recipe_kind = task_recipe_kind(task)
    vector = np.full(task_recipe_dimension(task), 0.5, dtype=float)
    recipe = task_recipe_from_unit_vector(task, vector)
    slots: list[dict[str, Any]] = []
    previous_operation = "experiment_start"
    diagnostic_index = 0
    for step_index, action in enumerate(recipe["steps"]):
        operation = str(action.get("operation"))
        instrument = action.get("instrument")
        if operation == "measure" and instrument != "final_assay":
            diagnostic_index += 1
            slot = {
                "slot_id": f"diagnostic-{diagnostic_index:02d}-{instrument}",
                "instrument": str(instrument),
                "after_operation": previous_operation,
                "recipe_step_index": step_index,
                "selection_policy": (
                    "required_by_workflow"
                    if recipe_kind == "reaction_distillation"
                    else "agent_selectable"
                ),
            }
            if recipe_kind == "reaction_distillation" and diagnostic_index == 1:
                slot.update(
                    {
                        "stage_id": "post_reaction_pre_separation",
                        "model_facing_metric_ids": [
                            "conversion",
                            "yield",
                            "selectivity",
                            "byproduct_signal",
                        ],
                    }
                )
            elif recipe_kind == "reaction_distillation" and diagnostic_index == 2:
                slot.update(
                    {
                        "stage_id": "post_fraction_collection",
                        "model_facing_metric_ids": [
                            "distillate_purity",
                            "degradation_warning",
                            "byproduct_signal",
                        ],
                    }
                )
            elif recipe_kind == "reaction_crystallization" and diagnostic_index == 1:
                slot.update(
                    {
                        "stage_id": "post_reaction_pre_crystallization",
                        "model_facing_metric_ids": [
                            "conversion",
                            "yield",
                            "selectivity",
                            "byproduct_signal",
                        ],
                    }
                )
            elif recipe_kind == "reaction_crystallization" and diagnostic_index == 2:
                slot.update(
                    {
                        "stage_id": "post_crystallization_pre_filtration",
                        "model_facing_metric_ids": [
                            "crystal_purity",
                            "yield",
                            "byproduct_signal",
                        ],
                    }
                )
            elif recipe_kind == "electrochemical" and diagnostic_index == 1:
                slot.update(
                    {
                        "stage_id": "post_probe_electrolysis_equilibrium_diagnostic",
                        "model_facing_metric_ids": [
                            "pH_normalized",
                            "acid_dissociation_fraction",
                            "precipitation_signal",
                            "equilibrium_residual",
                            "equilibrium_confidence",
                        ],
                    }
                )
            elif recipe_kind == "electrochemical" and diagnostic_index == 2:
                slot.update(
                    {
                        "stage_id": "post_controlled_electrolysis_efficiency_diagnostic",
                        "model_facing_metric_ids": [
                            "faradaic_efficiency",
                            "transport_efficiency",
                            "ohmic_efficiency",
                            "energy_efficiency",
                        ],
                    }
                )
            elif recipe_kind == "partition" and diagnostic_index == 1:
                slot.update(
                    {
                        "stage_id": "post_settle_pre_separation",
                        "model_facing_metric_ids": [
                            "phase_ratio",
                            "product_in_organic",
                            "product_in_aqueous",
                            "impurity_signal",
                        ],
                    }
                )
            elif recipe_kind == "partition" and diagnostic_index == 2:
                slot.update(
                    {
                        "stage_id": "post_selected_phase_separation",
                        "model_facing_metric_ids": [
                            "purity",
                            "recovery",
                            "product_in_organic",
                            "product_in_aqueous",
                            "impurity_signal",
                        ],
                    }
                )
            elif recipe_kind == "flow" and diagnostic_index == 1:
                slot.update(
                    {
                        "stage_id": "post_flow_run",
                        "model_facing_metric_ids": [
                            "flow_conversion",
                            "yield",
                            "selectivity",
                        ],
                    }
                )
            slots.append(slot)
        previous_operation = operation
    return tuple(slots)


def required_scientific_measurement_slot_ids(
    task_info: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return diagnostics that every complete workflow execution must retain."""

    return tuple(
        str(slot["slot_id"])
        for slot in scientific_measurement_slots(task_info)
        if slot.get("selection_policy") == "required_by_workflow"
    )


@dataclass(frozen=True)
class ScientificExperimentPlan:
    """Validated, Agent-authored selection for one complete experiment."""

    experiment_intent: str
    search_vector: tuple[float, ...]
    requested_measurement_slots: tuple[str, ...]
    diagnostic_target: str
    mechanism_distribution: dict[str, float]
    expected_effect: str
    belief_update_rule: str
    uncertainty: float
    scientific_state: dict[str, Any] | None = None

    def to_dict(self, *, include_scientific_state: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_intent": self.experiment_intent,
            "search_vector": list(self.search_vector),
            "requested_measurement_slots": list(self.requested_measurement_slots),
            "diagnostic_target": self.diagnostic_target,
            "mechanism_distribution": dict(self.mechanism_distribution),
            "expected_effect": self.expected_effect,
            "belief_update_rule": self.belief_update_rule,
            "uncertainty": self.uncertainty,
        }
        if include_scientific_state:
            payload["scientific_state"] = copy.deepcopy(self.scientific_state)
        return payload


def compile_scientific_experiment_plan(
    task_info: Mapping[str, Any],
    plan: ScientificExperimentPlan,
) -> dict[str, Any]:
    """Compile conditions exactly and retain only Agent-selected diagnostics.

    The existing recipe compiler remains the sole owner of legal operation order
    and mechanical closeout. This adapter only removes unrequested diagnostic
    measurement slots; it never changes a physical condition or adds a diagnostic.
    """

    task = dict(task_info)
    recipe = task_recipe_from_unit_vector(task, np.asarray(plan.search_vector, dtype=float))
    available = {slot["slot_id"] for slot in scientific_measurement_slots(task)}
    requested = set(plan.requested_measurement_slots)
    if not requested.issubset(available):
        raise ValueError("plan requests an unknown diagnostic measurement slot")
    required = set(required_scientific_measurement_slot_ids(task))
    if not required.issubset(requested):
        raise ValueError("plan omits a workflow-required diagnostic measurement slot")

    steps: list[dict[str, Any]] = []
    measurement_slots_by_step: dict[str, str] = {}
    diagnostic_index = 0
    for action in recipe["steps"]:
        operation = action.get("operation")
        instrument = action.get("instrument")
        if operation == "measure" and instrument != "final_assay":
            diagnostic_index += 1
            slot_id = f"diagnostic-{diagnostic_index:02d}-{instrument}"
            if slot_id not in requested:
                continue
            measurement_slots_by_step[str(len(steps))] = slot_id
        elif operation == "measure" and instrument == "final_assay":
            measurement_slots_by_step[str(len(steps))] = "closeout-final-assay"
        steps.append(copy.deepcopy(action))

    metadata = copy.deepcopy(recipe["metadata"])
    metadata.update(
        {
            "scientific_adaptation_interface_version": (SCIENTIFIC_ADAPTATION_INTERFACE_VERSION),
            "requested_measurement_slots": list(plan.requested_measurement_slots),
            "measurement_slots_by_step": measurement_slots_by_step,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "closeout_policy": "recipe_compiler_terminate_then_final_assay",
        }
    )
    return {"steps": steps, "metadata": metadata}


class ScientificMemoryStore(Protocol):
    """Persistent Agent-authored scientific state, isolated from public context."""

    def reset(self) -> None: ...

    def read(self) -> dict[str, Any] | None: ...

    def write(
        self,
        state: Any,
        *,
        available_evidence_ids: set[str],
    ) -> dict[str, Any] | None: ...

    def manifest(self) -> dict[str, Any]: ...


class NullScientificMemory:
    """Direct-method memory implementation that cannot persist scaffold state."""

    def reset(self) -> None:
        return None

    def read(self) -> None:
        return None

    def write(
        self,
        state: Any,
        *,
        available_evidence_ids: set[str],
    ) -> None:
        del available_evidence_ids
        if state is not None:
            raise ScientificPlanValidationError(
                "direct scaffold cannot persist scientific state",
                field_path="scientific_state",
                constraint="must_be_null_for_direct_scaffold",
                observed=1,
                limit=0,
            )
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "memory_store": type(self).__name__,
            "persistent_scientific_state": False,
        }


class BoundedScientificMemory:
    """Validate and retain a compact relational evidence summary."""

    def __init__(
        self,
        candidate_ids: Sequence[str],
        *,
        max_evidence_items: int = 6,
        max_controlled_variables: int = 10,
        max_json_characters: int = 2_800,
    ) -> None:
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be unique and non-empty")
        if max_evidence_items <= 0 or max_controlled_variables <= 0 or max_json_characters < 500:
            raise ValueError("scientific memory limits are invalid")
        self.candidate_ids = tuple(str(item) for item in candidate_ids)
        self.max_evidence_items = int(max_evidence_items)
        self.max_controlled_variables = int(max_controlled_variables)
        self.max_json_characters = int(max_json_characters)
        self._state: dict[str, Any] | None = None

    def reset(self) -> None:
        self._state = None

    def read(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._state)

    def write(
        self,
        state: Any,
        *,
        available_evidence_ids: set[str],
    ) -> dict[str, Any]:
        validated = self.validate(
            state,
            available_evidence_ids=available_evidence_ids,
        )
        self._state = copy.deepcopy(validated)
        return copy.deepcopy(validated)

    def validate(
        self,
        state: Any,
        *,
        available_evidence_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise ScientificPlanValidationError(
                "scientific_state must be an object",
                field_path="scientific_state",
                constraint="object",
                observed=type(state).__name__,
            )
        _exact_keys(
            state,
            {
                "belief",
                "unresolved_question",
                "next_experiment_plan",
                "evidence_summary",
            },
            field="scientific_state",
        )
        belief = _probability_distribution(
            state["belief"],
            candidate_ids=self.candidate_ids,
            field="scientific_state.belief",
        )
        unresolved_question = _required_text(
            state["unresolved_question"],
            field="scientific_state.unresolved_question",
            maximum=240,
        )
        raw_plan = state["next_experiment_plan"]
        if not isinstance(raw_plan, Mapping):
            raise ScientificPlanValidationError(
                "scientific_state.next_experiment_plan must be an object",
                field_path="scientific_state.next_experiment_plan",
                constraint="object",
                observed=type(raw_plan).__name__,
            )
        _exact_keys(
            raw_plan,
            {"intent", "controlled_variables", "varied_variable"},
            field="scientific_state.next_experiment_plan",
        )
        controlled = raw_plan["controlled_variables"]
        if (
            not isinstance(controlled, list)
            or not all(isinstance(item, str) and item.strip() for item in controlled)
            or len(controlled) > self.max_controlled_variables
        ):
            observed = (
                len(controlled) if isinstance(controlled, list) else type(controlled).__name__
            )
            raise ScientificPlanValidationError(
                "controlled_variables violates its bounded item contract",
                field_path="scientific_state.next_experiment_plan.controlled_variables",
                constraint="non_empty_string_list_with_max_items",
                observed=observed,
                limit=self.max_controlled_variables,
            )
        normalized_controlled = [str(item).strip() for item in controlled]
        if len(set(normalized_controlled)) != len(normalized_controlled):
            raise ScientificPlanValidationError(
                "controlled_variables must not contain duplicates",
                field_path="scientific_state.next_experiment_plan.controlled_variables",
                constraint="unique_items",
                observed=len(normalized_controlled),
                limit=len(set(normalized_controlled)),
            )
        next_plan = {
            "intent": _required_text(
                raw_plan["intent"],
                field="scientific_state.next_experiment_plan.intent",
                maximum=240,
            ),
            "controlled_variables": normalized_controlled,
            "varied_variable": _required_text(
                raw_plan["varied_variable"],
                field="scientific_state.next_experiment_plan.varied_variable",
                maximum=120,
            ),
        }
        raw_evidence = state["evidence_summary"]
        if not isinstance(raw_evidence, list):
            raise ScientificPlanValidationError(
                "scientific_state.evidence_summary must be a list",
                field_path="scientific_state.evidence_summary",
                constraint="list",
                observed=type(raw_evidence).__name__,
            )
        if len(raw_evidence) > self.max_evidence_items:
            raise ScientificPlanValidationError(
                "scientific_state.evidence_summary exceeds its item limit",
                field_path="scientific_state.evidence_summary",
                constraint="max_items",
                observed=len(raw_evidence),
                limit=self.max_evidence_items,
            )
        evidence_summary: list[dict[str, str]] = []
        evidence_ids: list[str] = []
        for index, item in enumerate(raw_evidence):
            if not isinstance(item, Mapping):
                raise ScientificPlanValidationError(
                    f"evidence_summary[{index}] must be an object",
                    field_path=f"scientific_state.evidence_summary[{index}]",
                    constraint="object",
                    observed=type(item).__name__,
                )
            _exact_keys(
                item,
                {"evidence_id", "observation", "interpretation", "reliability"},
                field=f"scientific_state.evidence_summary[{index}]",
            )
            evidence_id = _required_text(
                item["evidence_id"],
                field=f"scientific_state.evidence_summary[{index}].evidence_id",
                maximum=120,
            )
            if evidence_id not in available_evidence_ids:
                raise ScientificPlanValidationError(
                    "scientific_state references an unknown public evidence ID",
                    field_path=f"scientific_state.evidence_summary[{index}].evidence_id",
                    constraint="known_public_evidence_id",
                )
            reliability = item["reliability"]
            if reliability not in _RELIABILITY_LEVELS:
                raise ScientificPlanValidationError(
                    "evidence reliability must be low, medium, or high",
                    field_path=f"scientific_state.evidence_summary[{index}].reliability",
                    constraint="declared_enum",
                )
            evidence_ids.append(evidence_id)
            evidence_summary.append(
                {
                    "evidence_id": evidence_id,
                    "observation": _required_text(
                        item["observation"],
                        field=f"scientific_state.evidence_summary[{index}].observation",
                        maximum=260,
                    ),
                    "interpretation": _required_text(
                        item["interpretation"],
                        field=f"scientific_state.evidence_summary[{index}].interpretation",
                        maximum=300,
                    ),
                    "reliability": str(reliability),
                }
            )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ScientificPlanValidationError(
                "scientific_state evidence IDs must not be duplicated",
                field_path="scientific_state.evidence_summary",
                constraint="unique_evidence_ids",
                observed=len(evidence_ids),
                limit=len(set(evidence_ids)),
            )
        validated = {
            "belief": belief,
            "unresolved_question": unresolved_question,
            "next_experiment_plan": next_plan,
            "evidence_summary": evidence_summary,
        }
        encoded = json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded) > self.max_json_characters:
            raise ScientificPlanValidationError(
                "scientific_state exceeds its JSON character limit",
                field_path="scientific_state",
                constraint="max_json_characters",
                observed=len(encoded),
                limit=self.max_json_characters,
            )
        return validated

    def manifest(self) -> dict[str, Any]:
        return {
            "memory_store": type(self).__name__,
            "schema_version": SCIENTIFIC_MEMORY_VERSION,
            "persistent_scientific_state": True,
            "candidate_ids": list(self.candidate_ids),
            "max_evidence_items": self.max_evidence_items,
            "max_controlled_variables": self.max_controlled_variables,
            "max_json_characters": self.max_json_characters,
            "evidence_policy": "public_evidence_id_required_unique_bounded",
        }


class ScaffoldPolicy(Protocol):
    scaffold_id: str
    requires_scientific_state: bool

    def prompt_context(self, memory: ScientificMemoryStore) -> dict[str, Any]: ...


class DirectScaffoldPolicy:
    scaffold_id = "direct"
    requires_scientific_state = False

    def prompt_context(self, memory: ScientificMemoryStore) -> dict[str, Any]:
        if memory.read() is not None:
            raise ValueError("direct scaffold memory must be empty")
        return {
            "scaffold_id": self.scaffold_id,
            "instruction": (
                "Use the shared public experiment history directly. Do not return or "
                "maintain a separate persistent scientific_state."
            ),
            "prior_scientific_state": None,
        }


class StatefulScientificScaffoldPolicy:
    scaffold_id = "stateful_scientific"
    requires_scientific_state = True

    def prompt_context(self, memory: ScientificMemoryStore) -> dict[str, Any]:
        return {
            "scaffold_id": self.scaffold_id,
            "instruction": (
                "Use the shared public experiment history and update one compact persistent "
                "scientific_state. Every evidence_summary item must cite an evidence_id from "
                "the public evidence catalog."
            ),
            "prior_scientific_state": memory.read(),
            "scientific_state_contract": {
                "belief": "probability distribution over public candidate IDs",
                "unresolved_question": "one concise open scientific question",
                "next_experiment_plan": {
                    "intent": "concise plan",
                    "controlled_variables": ["public variable name"],
                    "varied_variable": "one public variable name",
                },
                "evidence_summary": [
                    {
                        "evidence_id": "exact public evidence ID",
                        "observation": "concise observation",
                        "interpretation": "concise interpretation",
                        "reliability": "low|medium|high",
                    }
                ],
            },
        }


class PublicExperimentContextBuilder:
    """Build the common Agent-visible context for both scaffold conditions."""

    def __init__(
        self,
        task_info: Mapping[str, Any],
        candidate_definitions: Mapping[str, str],
        *,
        history_limit: int = 8,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if not candidate_definitions:
            raise ValueError("candidate_definitions must be non-empty")
        self.task_info = dict(task_info)
        self.candidate_definitions = {
            str(key): _required_text(
                value,
                field=f"candidate_definitions.{key}",
                maximum=600,
            )
            for key, value in candidate_definitions.items()
        }
        self.history_limit = int(history_limit)
        self.measurement_slots = scientific_measurement_slots(self.task_info)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(self.candidate_definitions)

    def build(self, experiment_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        selected_history = self._select_history(experiment_history)
        history = [self._compact_history_record(item) for item in selected_history]
        evidence_catalog = [
            str(evidence["evidence_id"])
            for record in history
            for evidence in record["measurement_evidence"]
        ]
        return {
            "schema_version": SCIENTIFIC_ADAPTATION_INTERFACE_VERSION,
            "task": {
                key: copy.deepcopy(self.task_info[key])
                for key in _PUBLIC_TASK_KEYS
                if key in self.task_info
            },
            "experiment_interface": {
                "decision_scope": "complete_experiment",
                "recipe_space_version": TASK_RECIPE_SPACE_VERSION,
                "recipe_space_kind": task_recipe_kind(self.task_info),
                "search_vector_dimension": task_recipe_dimension(self.task_info),
                "search_vector_bounds": [0.0, 1.0],
                "search_vector_coordinate_schema": [
                    copy.deepcopy(item) for item in task_recipe_coordinate_schema(self.task_info)
                ],
                "physical_controls_are_deterministically_decoded": True,
                "categorical_coordinates": [
                    {
                        "coordinate": coordinate,
                        "category_count": count,
                        "selection_semantics": "independent_unordered_nominal_choice",
                    }
                    for coordinate, count in task_recipe_categorical_coordinates(self.task_info)
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
            },
            "mechanism_candidates": copy.deepcopy(self.candidate_definitions),
            "history_window": {
                "selection_policy": "oldest_reference_half_plus_most_recent_half",
                "total_experiment_count": len(experiment_history),
                "included_experiment_indices": [record["experiment_index"] for record in history],
            },
            "experiment_history": history,
            "evidence_catalog": evidence_catalog,
        }

    def _select_history(
        self,
        experiment_history: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        history = list(experiment_history)
        if len(history) <= self.history_limit:
            return history
        reference_count = max(self.history_limit // 2, 1)
        recent_count = self.history_limit - reference_count
        if recent_count == 0:
            return history[:reference_count]
        return history[:reference_count] + history[-recent_count:]

    def _compact_history_record(self, item: Mapping[str, Any]) -> dict[str, Any]:
        plan = item.get("plan")
        if not isinstance(plan, Mapping):
            raise ValueError("experiment history record is missing its public plan")
        evidence = item.get("measurement_evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(entry, Mapping) for entry in evidence
        ):
            raise ValueError("experiment history measurement_evidence must be a list")
        experiment_index = item.get("experiment_index")
        if isinstance(experiment_index, bool) or not isinstance(experiment_index, int):
            raise ValueError("experiment history index must be an integer")
        compact_plan = {
            key: _compact_public_value(plan[key]) for key in _PUBLIC_PLAN_KEYS if key in plan
        }
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
                key: _compact_public_value(entry[key])
                for key in (
                    "evidence_id",
                    "reward",
                )
                if key in entry
            }
            processed_estimate = entry.get("processed_estimate")
            if isinstance(processed_estimate, Mapping):
                compact_entry["processed_estimate"] = _compact_public_value(
                    {
                        key: value
                        for key, value in processed_estimate.items()
                        if allowed_metrics is None or key in allowed_metrics
                    }
                )
            uncertainty = entry.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                compact_entry["uncertainty"] = _compact_public_value(
                    {
                        key: value
                        for key, value in uncertainty.items()
                        if allowed_metrics is None or key.removesuffix("_std") in allowed_metrics
                    }
                )
            compact_evidence.append(compact_entry)
        return {
            "experiment_index": experiment_index,
            "plan": compact_plan,
            "measurement_evidence": compact_evidence,
            "terminal_summary": {
                key: _compact_public_value(item["terminal_summary"][key])
                for key in _PUBLIC_TERMINAL_SUMMARY_KEYS
                if isinstance(item.get("terminal_summary"), Mapping)
                and key in item["terminal_summary"]
            },
        }


def _compact_public_value(value: Any) -> Any:
    """Keep one stable, decision-grade representation of public numeric evidence."""

    normalized = to_builtin(value)
    if isinstance(normalized, float):
        return round(normalized, 6)
    if isinstance(normalized, dict):
        return {str(key): _compact_public_value(item) for key, item in normalized.items()}
    if isinstance(normalized, list):
        return [_compact_public_value(item) for item in normalized]
    return copy.deepcopy(normalized)


class ExperimentPlanResponseValidator:
    """Validate model output without repairing scientific or physical choices."""

    def __init__(
        self,
        task_info: Mapping[str, Any],
        candidate_ids: Sequence[str],
    ) -> None:
        self.task_info = dict(task_info)
        self.candidate_ids = tuple(candidate_ids)
        self.dimension = task_recipe_dimension(self.task_info)
        self.measurement_slot_ids = tuple(
            str(item["slot_id"]) for item in scientific_measurement_slots(self.task_info)
        )
        self.required_measurement_slot_ids = required_scientific_measurement_slot_ids(
            self.task_info
        )

    def validate(
        self,
        payload: Any,
        *,
        requires_scientific_state: bool,
        memory: ScientificMemoryStore,
        available_evidence_ids: set[str],
    ) -> ScientificExperimentPlan:
        if not isinstance(payload, Mapping):
            raise ScientificPlanValidationError(
                "experiment plan response must be an object",
                field_path="experiment_plan_response",
                constraint="object",
                observed=type(payload).__name__,
            )
        required = {
            "experiment_intent",
            "search_vector",
            "requested_measurement_slots",
            "diagnostic_target",
            "mechanism_distribution",
            "expected_effect",
            "belief_update_rule",
            "uncertainty",
        }
        if requires_scientific_state:
            required.add("scientific_state")
        allowed = required | ({"scientific_state"} if not requires_scientific_state else set())
        if set(payload) - allowed:
            raise ScientificPlanValidationError(
                "experiment plan response contains undeclared fields",
                field_path="experiment_plan_response",
                constraint="declared_fields_only",
                observed=len(set(payload) - allowed),
                limit=0,
            )
        if not required.issubset(payload):
            raise ScientificPlanValidationError(
                "experiment plan response is missing required fields",
                field_path="experiment_plan_response",
                constraint="required_fields",
                observed=len(required - set(payload)),
                limit=0,
            )

        vector = payload["search_vector"]
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise ScientificPlanValidationError(
                f"search_vector must contain exactly {self.dimension} values",
                field_path="search_vector",
                constraint="exact_items",
                observed=len(vector) if isinstance(vector, list) else type(vector).__name__,
                limit=self.dimension,
            )
        normalized_vector: list[float] = []
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
                "requested_measurement_slots must be a list of slot IDs",
                field_path="requested_measurement_slots",
                constraint="string_list",
                observed=(
                    len(requested) if isinstance(requested, list) else type(requested).__name__
                ),
            )
        normalized_requested = [str(item) for item in requested]
        if len(set(normalized_requested)) != len(normalized_requested):
            raise ScientificPlanValidationError(
                "requested_measurement_slots must not contain duplicates",
                field_path="requested_measurement_slots",
                constraint="unique_items",
                observed=len(normalized_requested),
                limit=len(set(normalized_requested)),
            )
        if not set(normalized_requested).issubset(self.measurement_slot_ids):
            raise ScientificPlanValidationError(
                "requested_measurement_slots contains an unknown slot ID",
                field_path="requested_measurement_slots",
                constraint="known_measurement_slot_ids",
            )
        if not set(self.required_measurement_slot_ids).issubset(normalized_requested):
            raise ScientificPlanValidationError(
                "requested_measurement_slots omits a workflow-required slot ID",
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

        experiment_intent = _required_text(
            payload["experiment_intent"],
            field="experiment_intent",
            maximum=500,
        )
        diagnostic_target = _required_text(
            payload["diagnostic_target"],
            field="diagnostic_target",
            maximum=500,
        )
        mechanism_distribution = _probability_distribution(
            payload["mechanism_distribution"],
            candidate_ids=self.candidate_ids,
            field="mechanism_distribution",
        )
        expected_effect = _required_text(
            payload["expected_effect"],
            field="expected_effect",
            maximum=700,
        )
        belief_update_rule = _required_text(
            payload["belief_update_rule"],
            field="belief_update_rule",
            maximum=700,
        )

        # Commit persistent memory only after every other response field passes.
        scientific_state = memory.write(
            payload.get("scientific_state"),
            available_evidence_ids=available_evidence_ids,
        )
        return ScientificExperimentPlan(
            experiment_intent=experiment_intent,
            search_vector=tuple(normalized_vector),
            requested_measurement_slots=tuple(normalized_requested),
            diagnostic_target=diagnostic_target,
            mechanism_distribution=mechanism_distribution,
            expected_effect=expected_effect,
            belief_update_rule=belief_update_rule,
            uncertainty=uncertainty_float,
            scientific_state=scientific_state,
        )


class ResourceLedger:
    """Minimal cumulative provider accounting for experiment-level decisions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.model_call_count = 0
        self.provider_attempt_count = 0
        self.provider_failure_count = 0
        self.provider_attempt_records: list[dict[str, Any]] = []
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }

    def record_completion(self, completion: JsonCompletionLike) -> None:
        self.model_call_count += 1
        self.provider_attempt_count += max(int(completion.attempts), 1)
        self._add_usage(completion.usage)
        self._record_attempts(getattr(completion, "attempt_records", ()))

    def record_failure(self, error: Exception) -> None:
        self.model_call_count += 1
        self.provider_failure_count += 1
        self.provider_attempt_count += max(int(getattr(error, "attempts", 1)), 1)
        usage = getattr(error, "usage", {})
        if isinstance(usage, Mapping):
            self._add_usage(usage)
        self._record_attempts(getattr(error, "attempt_records", ()))

    def _record_attempts(self, records: Any) -> None:
        if not isinstance(records, Sequence) or isinstance(records, str | bytes):
            return
        logical_decision_index = self.model_call_count
        for record in records:
            if not isinstance(record, Mapping):
                continue
            self.provider_attempt_records.append(
                {
                    "logical_decision_index": logical_decision_index,
                    **copy.deepcopy(to_builtin(dict(record))),
                }
            )

    def _add_usage(self, usage: Mapping[str, Any]) -> None:
        for key in self.usage:
            value = usage.get(key, 0)
            if isinstance(value, int | float) and not isinstance(value, bool):
                self.usage[key] += int(value)

    def snapshot(self, client: JsonPlannerClientLike) -> dict[str, Any]:
        pricing_factory = getattr(client, "pricing_snapshot", None)
        cost_factory = getattr(client, "estimate_cost_usd", None)
        pricing = pricing_factory() if callable(pricing_factory) else None
        accounting_complete = False
        cost = 0.0
        if (
            isinstance(pricing, dict)
            and bool(pricing.get("accounting_complete", True))
            and callable(cost_factory)
        ):
            accounting_complete = True
            cost = float(cost_factory(dict(self.usage)))
        return {
            "schema_version": "chemworld-method-resource-usage-0.1",
            "accounting_complete": accounting_complete,
            "usage_source": (
                "provider_usage_and_frozen_price_snapshot"
                if accounting_complete
                else "provider_usage_pricing_unavailable"
            ),
            "model_call_count": self.model_call_count,
            "provider_attempt_count": self.provider_attempt_count,
            "provider_failure_count": self.provider_failure_count,
            "provider_attempt_records": copy.deepcopy(self.provider_attempt_records),
            "provider_usage": dict(self.usage),
            "input_token_count": self.usage["prompt_tokens"],
            "output_token_count": self.usage["completion_tokens"],
            "monetary_cost_usd": cost,
            "training_environment_step_count": 0,
            "cpu_time_s": 0.0,
            "gpu_time_s": 0.0,
            "model_provenance": {
                "model_id": client.model,
                "pricing": pricing,
                "private_reasoning_retained": False,
            },
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("schema_version") != "chemworld-method-resource-usage-0.1":
            raise ValueError("unsupported resource checkpoint schema")
        usage = snapshot.get("provider_usage")
        records = snapshot.get("provider_attempt_records")
        if not isinstance(usage, Mapping) or not isinstance(records, list):
            raise ValueError("resource checkpoint lacks provider usage or attempts")
        self.model_call_count = int(snapshot["model_call_count"])
        self.provider_attempt_count = int(snapshot["provider_attempt_count"])
        self.provider_failure_count = int(snapshot["provider_failure_count"])
        self.usage = {
            key: int(usage.get(key, 0))
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            )
        }
        self.provider_attempt_records = [
            copy.deepcopy(to_builtin(dict(record)))
            for record in records
            if isinstance(record, Mapping)
        ]


class ScientificAdaptationAgent:
    """Compose context, scaffold, memory, validation, and resource accounting."""

    name = "scientific_adaptation_agent"

    def __init__(
        self,
        client: JsonPlannerClientLike,
        *,
        role_id: str,
        candidate_definitions: Mapping[str, str],
        scaffold_policy: ScaffoldPolicy,
        memory_store: ScientificMemoryStore,
        response_max_tokens: int = 2_000,
        history_limit: int = 8,
        prompt_token_estimate_cap: int = 12_000,
    ) -> None:
        if response_max_tokens <= 0:
            raise ValueError("response_max_tokens must be positive")
        if prompt_token_estimate_cap < 500:
            raise ValueError("prompt_token_estimate_cap must be at least 500")
        self.client = client
        self.role_id = role_id
        self.candidate_definitions = dict(candidate_definitions)
        self.scaffold_policy = scaffold_policy
        self.memory_store = memory_store
        self.response_max_tokens = int(response_max_tokens)
        self.history_limit = int(history_limit)
        self.prompt_token_estimate_cap = int(prompt_token_estimate_cap)
        self.resource_ledger = ResourceLedger()
        self._last_audit: dict[str, Any] | None = None

    def reset(self, task_info: Mapping[str, Any], seed: int) -> None:
        self.task_info = dict(task_info)
        self.seed = int(seed)
        self.context_builder = PublicExperimentContextBuilder(
            self.task_info,
            self.candidate_definitions,
            history_limit=self.history_limit,
        )
        self.response_validator = ExperimentPlanResponseValidator(
            self.task_info,
            self.context_builder.candidate_ids,
        )
        self.memory_store.reset()
        self.resource_ledger.reset()
        self._last_audit = None

    def public_context(
        self,
        experiment_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self._require_reset()
        return self.context_builder.build(experiment_history)

    def plan_next(
        self,
        experiment_history: Sequence[Mapping[str, Any]],
    ) -> ScientificExperimentPlan:
        public_context = self.public_context(experiment_history)
        public_context_sha256 = canonical_sha256(public_context)
        evidence_ids = {str(item) for item in public_context["evidence_catalog"]}
        prompt_payload = {
            "schema_version": SCIENTIFIC_ADAPTATION_PROMPT_VERSION,
            "public_experiment_context": public_context,
            "public_context_sha256": public_context_sha256,
            "scaffold_context": self._scaffold_prompt_context(),
            "required_json_shape": self._response_shape(),
        }
        prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_estimated_tokens = estimate_prompt_tokens(prompt)
        if prompt_estimated_tokens > self.prompt_token_estimate_cap:
            raise PromptBudgetExceededError(
                f"scientific adaptation prompt estimate {prompt_estimated_tokens} exceeds cap "
                f"{self.prompt_token_estimate_cap}; revise the public representation explicitly"
            )
        try:
            completion = self.client.complete_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=self.response_max_tokens,
            )
        except Exception as error:
            self.resource_ledger.record_failure(error)
            raise
        self.resource_ledger.record_completion(completion)
        plan = self.response_validator.validate(
            completion.payload,
            requires_scientific_state=(self.scaffold_policy.requires_scientific_state),
            memory=self.memory_store,
            available_evidence_ids=evidence_ids,
        )
        self._last_audit = {
            "schema_version": SCIENTIFIC_ADAPTATION_INTERFACE_VERSION,
            "role_id": self.role_id,
            "scaffold_id": self.scaffold_policy.scaffold_id,
            "public_context_sha256": public_context_sha256,
            "scientific_memory_sha256": (
                canonical_sha256(self.memory_store.read())
                if self.memory_store.read() is not None
                else None
            ),
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "provider_model": str(completion.model),
            "provider_attempts": int(completion.attempts),
            "provider_usage": copy.deepcopy(to_builtin(completion.usage)),
            "prompt_estimated_tokens": prompt_estimated_tokens,
            "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
            "gate_a_or_private_truth_supplied": False,
        }
        return plan

    def decision_audit(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_audit)

    def manifest(self) -> dict[str, Any]:
        self._require_reset()
        return {
            "agent_name": self.name,
            "agent_family": type(self).__name__,
            "role_id": self.role_id,
            "seed": self.seed,
            "provider_model": self.client.model,
            "decision_scope": "complete_experiment",
            "scaffold_id": self.scaffold_policy.scaffold_id,
            "public_context_builder": type(self.context_builder).__name__,
            "response_validator": type(self.response_validator).__name__,
            "memory": self.memory_store.manifest(),
            "interface_version": SCIENTIFIC_ADAPTATION_INTERFACE_VERSION,
            "prompt_version": SCIENTIFIC_ADAPTATION_PROMPT_VERSION,
            "provider_calls_per_logical_decision": 1,
            "history_limit": self.history_limit,
            "history_selection_policy": "oldest_reference_half_plus_most_recent_half",
            "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
            "mechanical_closeout": True,
            "harness_generated_scientific_content": False,
            "gate_a_or_private_truth_supplied": False,
        }

    def method_resource_usage(self) -> dict[str, Any]:
        return self.resource_ledger.snapshot(self.client)

    def restore_development_checkpoint(
        self,
        *,
        experiment_history: Sequence[Mapping[str, Any]],
        scientific_state: Any,
        resources: Mapping[str, Any],
    ) -> None:
        """Restore public Agent state after a retryable development interruption."""

        public_context = self.public_context(experiment_history)
        evidence_ids = {str(item) for item in public_context["evidence_catalog"]}
        self.memory_store.write(
            scientific_state,
            available_evidence_ids=evidence_ids,
        )
        self.resource_ledger.restore(resources)
        self._last_audit = None

    def _response_shape(self) -> dict[str, Any]:
        shape: dict[str, Any] = {
            "experiment_intent": "string",
            "search_vector": [
                "number in [0,1]" for _ in range(task_recipe_dimension(self.task_info))
            ],
            "requested_measurement_slots": ["public diagnostic slot ID"],
            "diagnostic_target": "string",
            "mechanism_distribution": dict.fromkeys(
                self.context_builder.candidate_ids,
                "probability",
            ),
            "expected_effect": "string",
            "belief_update_rule": "string",
            "uncertainty": "number in [0,1]",
        }
        if self.scaffold_policy.requires_scientific_state:
            shape["scientific_state"] = self._scaffold_prompt_context()["scientific_state_contract"]
        return shape

    def _scaffold_prompt_context(self) -> dict[str, Any]:
        context = self.scaffold_policy.prompt_context(self.memory_store)
        if not self.scaffold_policy.requires_scientific_state:
            return context
        contract = context["scientific_state_contract"]
        contract["belief"] = dict.fromkeys(
            self.context_builder.candidate_ids,
            "probability; every listed key is required and no other key is allowed",
        )
        context["scientific_state_constraints"] = {
            "belief_keys": "exactly the public candidate IDs shown in the contract",
            "belief_sum": 1.0,
            "controlled_variables_max_items": 10,
            "evidence_summary_max_items": 6,
            "evidence_ids": "unique exact IDs from public evidence_catalog only",
            "undeclared_fields_allowed": False,
        }
        return context

    def _require_reset(self) -> None:
        if not hasattr(self, "context_builder"):
            raise RuntimeError("ScientificAdaptationAgent must be reset before use")


__all__ = [
    "SCIENTIFIC_ADAPTATION_INTERFACE_VERSION",
    "SCIENTIFIC_ADAPTATION_PROMPT_VERSION",
    "SCIENTIFIC_MEMORY_VERSION",
    "BoundedScientificMemory",
    "DirectScaffoldPolicy",
    "ExperimentPlanResponseValidator",
    "NullScientificMemory",
    "PublicExperimentContextBuilder",
    "ResourceLedger",
    "ScaffoldPolicy",
    "ScientificAdaptationAgent",
    "ScientificExperimentPlan",
    "ScientificMemoryStore",
    "ScientificPlanValidationError",
    "StatefulScientificScaffoldPolicy",
    "canonical_sha256",
    "compile_scientific_experiment_plan",
    "required_scientific_measurement_slot_ids",
    "scientific_measurement_slots",
]
