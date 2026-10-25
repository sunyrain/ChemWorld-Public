"""Fail-closed action/intervention audit for mechanism-adaptation protocols.

The protocol is scientifically meaningful only when every hidden intervention can be
probed through public actions at the frozen budget.  This module checks that invariant
before an identifiability certificate or provider campaign is allowed to run.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from chemworld.agents.interaction import AgentDecisionContext
from chemworld.agents.mechanism_adaptation_live_llm import (
    MechanismAdaptationLiveLLMAgent,
    MechanismCandidateSpec,
)
from chemworld.agents.task_recipes import (
    DIAGNOSTIC_RECIPE_DESIGN_V2,
    task_recipe_from_unit_vector,
)
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.task_design import SERIOUS_TASK_DESIGNS
from chemworld.tasks import get_task
from chemworld.world.mechanism_family import (
    CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS,
    CONSTITUTIVE_TASK_TRANSFORMS,
    CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS,
    ELECTROCHEMICAL_RESPONSE_STRESS,
    EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS,
    PARTITION_POWER_RESPONSE_STRESS,
    RATE_LAW_TRANSFORM_CALIBRATION_FIELDS,
    REVERSIBLE_TARGET_PATHWAY_STRESS,
)
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

DESIGN_AUDIT_SCHEMA_VERSION = "chemworld-mechanism-design-audit-0.2.5"

MATERIAL_FIELD_ACTIONS: dict[str, tuple[str, str]] = {
    "catalyst": ("add_catalyst", "catalyst"),
    "solvent": ("add_solvent", "solvent"),
    "electrolyte_profile": ("set_potential", "electrolyte_profile"),
}

_MECHANISM_CONTROL_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "rate_law_family": (
        ("add_catalyst", "catalyst"),
        ("add_catalyst", "catalyst_amount_mol"),
        ("heat", "target_temperature_K"),
        ("heat", "duration_s"),
    ),
    "topology_family": (
        ("add_catalyst", "catalyst"),
        ("heat", "target_temperature_K"),
    ),
    "constitutive_law_family": (
        ("set_potential", "potential_V"),
        ("set_potential", "current_mA"),
        ("set_potential", "electrolyte_profile"),
    ),
}

_FORBIDDEN_PUBLIC_KEYS = {
    "public_to_baseline",
    "world_interventions",
    "mechanism_family_intervention_hash",
    "material_law_counterfactual_hash",
    "_hidden_material_law_counterfactual_field",
    "_hidden_material_law_public_to_baseline",
}


class _PromptOnlyClient:
    model = "prompt-only-audit"
    thinking = False


def audit_mechanism_design(
    protocol: Mapping[str, Any],
    gate_a_plan: Mapping[str, Any],
    *,
    action_libraries: Mapping[str, Mapping[str, np.ndarray]],
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Audit reachability, recipe coverage, observability, and prompt secrecy."""

    findings: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    definitions = protocol.get("diagnosis_contract", {}).get("candidate_definitions", {})
    contracts = protocol.get("task_mechanism_contracts", {})

    for task_id in protocol.get("design", {}).get("tasks", []):
        task_id = str(task_id)
        _emit_progress(
            progress_callback,
            event="task_started",
            task_id=task_id,
        )
        task_info = get_task(task_id).to_dict()
        contract = contracts.get(task_id, {})
        interventions = contract.get("interventions", {})
        diagnostic_relations = contract.get("diagnostic_relations", {})
        action_library = action_libraries.get(task_id, {})
        recipes = {
            action_id: task_recipe_from_unit_vector(task_info, vector)
            for action_id, vector in action_library.items()
        }
        recipe_values = _recipe_field_values(recipes)
        task_findings: list[dict[str, Any]] = []
        decision_baseline_cache: dict[tuple[int, str], dict[str, float]] = {}

        if not action_library:
            _add(
                task_findings,
                "action_library_present",
                False,
                "The frozen Gate A action library is empty.",
            )
        _add(
            task_findings,
            "diagnostic_measurement_present",
            bool(recipes)
            and all(_has_preterminal_measurement(recipe) for recipe in recipes.values()),
            "Every frozen recipe must contain a public measurement before termination.",
        )

        for candidate_id in contract.get("candidate_ids", []):
            candidate_id = str(candidate_id)
            candidate_interventions = interventions.get(candidate_id)
            if candidate_id == "no_change":
                _add(
                    task_findings,
                    f"{candidate_id}:empty_intervention",
                    candidate_interventions == [],
                    "The no-change twin must have no hidden intervention.",
                )
                continue
            _add(
                task_findings,
                f"{candidate_id}:single_intervention",
                isinstance(candidate_interventions, list) and len(candidate_interventions) == 1,
                "Each certification cell must change exactly one hidden law family.",
            )
            if not isinstance(candidate_interventions, list) or len(candidate_interventions) != 1:
                continue
            _emit_progress(
                progress_callback,
                event="candidate_started",
                task_id=task_id,
                candidate_id=candidate_id,
            )
            intervention = candidate_interventions[0]
            _audit_environment_reset(
                task_findings,
                task_id=task_id,
                candidate_id=candidate_id,
                intervention=intervention,
            )
            if intervention.get("kind") == "material_law_counterfactual":
                _audit_material_alignment(
                    task_findings,
                    task_id=task_id,
                    candidate_id=candidate_id,
                    intervention=intervention,
                    recipes=recipes,
                    recipe_values=recipe_values,
                    require_relational_pair=(
                        gate_a_plan.get("action_library", {}).get("design")
                        == DIAGNOSTIC_RECIPE_DESIGN_V2
                    ),
                )
            else:
                mode = str(intervention.get("mode", ""))
                if mode == "rate_law_family":
                    _audit_rate_law_alignment(
                        task_findings,
                        task_id=task_id,
                        candidate_id=candidate_id,
                        intervention=intervention,
                        recipes=recipes,
                        response_certificate=gate_a_plan.get(
                            "rate_law_response_certificate",
                            {},
                        ),
                    )
                elif mode == "topology_family":
                    _audit_topology_alignment(
                        task_findings,
                        task_id=task_id,
                        candidate_id=candidate_id,
                        intervention=intervention,
                    )
                elif mode == "constitutive_law_family":
                    _audit_constitutive_alignment(
                        task_findings,
                        task_id=task_id,
                        candidate_id=candidate_id,
                        intervention=intervention,
                    )
                controls = _MECHANISM_CONTROL_GROUPS.get(mode, ())
                varied = [
                    {"operation": operation, "field": field}
                    for operation, field in controls
                    if len(recipe_values.get((operation, field), set())) >= 2
                ]
                _add(
                    task_findings,
                    f"{candidate_id}:diagnostic_control_variation",
                    bool(varied),
                    "At least one public control implicated by the hidden law must vary "
                    f"across frozen recipes; varied={varied}.",
                )
                raw_relation = (
                    diagnostic_relations.get(candidate_id)
                    if isinstance(diagnostic_relations, Mapping)
                    else None
                )
                require_relation = (
                    gate_a_plan.get("online_attainability_certificate", {}).get(
                        "all_declared_relations_required_before_change"
                    )
                    is True
                )
                if require_relation:
                    relation_groups = (
                        mechanism_relational_action_groups(
                            recipes,
                            relation=raw_relation,
                        )
                        if isinstance(raw_relation, Mapping)
                        else ()
                    )
                    _add(
                        task_findings,
                        f"{candidate_id}:declared_diagnostic_relation_covered",
                        bool(relation_groups),
                        "The calibrated online track requires at least one "
                        "same-background public action group for every declared "
                        f"mechanism relation; groups={relation_groups}.",
                    )
            _audit_intervention_decision_relevance(
                task_findings,
                task_id=task_id,
                candidate_id=candidate_id,
                intervention=intervention,
                action_library=action_library,
                certificate=gate_a_plan.get("decision_relevance_certificate", {}),
                baseline_cache=decision_baseline_cache,
            )
            _emit_progress(
                progress_callback,
                event="candidate_completed",
                task_id=task_id,
                candidate_id=candidate_id,
            )

        controlled_coverage_required = (
            gate_a_plan.get("paired_phase_design", {}).get(
                "controlled_primary_declared_relational_action_coverage"
            )
            is True
        )
        controlled_coverage: dict[str, Any] = {
            "required": controlled_coverage_required,
        }
        if controlled_coverage_required:
            primary_budget = int(
                gate_a_plan.get("held_out_certificate", {}).get(
                    "primary_gate_budget",
                    0,
                )
            )
            try:
                declarations = declared_relational_action_groups(
                    recipes=recipes,
                    contract=contract,
                )
                controlled_coverage = relational_coverage_witness(
                    declaration_groups=declarations,
                    action_ids=list(action_library),
                    budget=primary_budget,
                )
                controlled_coverage["required"] = True
                coverage_error = None
            except (TypeError, ValueError) as exc:
                coverage_error = f"{type(exc).__name__}: {exc}"
                controlled_coverage = {
                    "required": True,
                    "budget": primary_budget,
                    "feasible": False,
                    "error": coverage_error,
                }
            _add(
                task_findings,
                "controlled_primary_relational_coverage_feasible",
                controlled_coverage.get("feasible") is True,
                "The controlled primary budget must contain one complete public "
                "action group for every declared relation before any formal trial "
                f"is scheduled; coverage={controlled_coverage}, error={coverage_error}.",
            )

        prompt_check = _audit_prompt_boundary(
            task_id=task_id,
            task_info=task_info,
            definitions=definitions,
            candidate_ids=[str(item) for item in contract.get("candidate_ids", [])],
        )
        task_findings.append(prompt_check)
        findings.extend({"task_id": task_id, **item} for item in task_findings)
        task_reports[task_id] = {
            "action_count": len(action_library),
            "controlled_primary_relational_coverage": controlled_coverage,
            "recipe_field_values": {
                f"{operation}.{field}": sorted(values, key=str)
                for (operation, field), values in sorted(recipe_values.items())
            },
            "checks": task_findings,
            "pass": all(item["pass"] for item in task_findings),
        }
        _emit_progress(
            progress_callback,
            event="task_completed",
            task_id=task_id,
            failure_count=sum(
                item["pass"] is not True for item in task_findings
            ),
        )

    failures = [item for item in findings if not item["pass"]]
    return {
        "schema_version": DESIGN_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "protocol_id": protocol.get("protocol_id"),
        "gate_a_plan_id": gate_a_plan.get("plan_id"),
        "material_field_action_contract": {
            key: {"operation": value[0], "public_field": value[1]}
            for key, value in MATERIAL_FIELD_ACTIONS.items()
        },
        "task_reports": task_reports,
        "check_count": len(findings),
        "failure_count": len(failures),
        "failures": failures,
        "pass": not failures,
        "interpretation": (
            "A pass means each frozen hidden target is publicly manipulable, covered by "
            "the Gate A recipes, observable before termination, instantiable, and absent "
            "from the mechanism-Agent prompt. It does not establish identifiability."
        ),
    }


def declared_relational_action_groups(
    *,
    recipes: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Resolve every declared intervention relation to public action groups."""

    declarations: dict[str, tuple[tuple[str, ...], ...]] = {}
    interventions = contract.get("interventions", {})
    diagnostic_relations = contract.get("diagnostic_relations", {})
    if not isinstance(interventions, Mapping):
        raise ValueError("task intervention contract must be an object")
    if not isinstance(diagnostic_relations, Mapping):
        raise ValueError("task diagnostic_relations must be an object")
    for candidate_id, raw_items in interventions.items():
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, str | bytes):
            raise ValueError("candidate interventions must be a sequence")
        for index, raw_intervention in enumerate(raw_items):
            if not isinstance(raw_intervention, Mapping):
                raise ValueError("candidate intervention must be an object")
            if raw_intervention.get("kind") == "material_law_counterfactual":
                groups = material_relational_action_groups(
                    recipes,
                    intervention=raw_intervention,
                )
            else:
                raw_relation = diagnostic_relations.get(candidate_id)
                if raw_relation is None:
                    continue
                if not isinstance(raw_relation, Mapping):
                    raise ValueError("diagnostic_relation must be an object")
                groups = mechanism_relational_action_groups(
                    recipes,
                    relation=raw_relation,
                )
            declaration_id = f"{candidate_id}:{index}"
            if not groups:
                raise ValueError(
                    "relational evidence requires a declared public action group "
                    f"for {declaration_id}"
                )
            declarations[declaration_id] = groups
    return declarations


def relational_coverage_witness(
    *,
    declaration_groups: Mapping[str, Sequence[Sequence[str]]],
    action_ids: Sequence[str],
    budget: int,
) -> dict[str, Any]:
    """Find the smallest action union closing one group per declaration."""

    if budget < 1:
        raise ValueError("relational coverage budget must be positive")
    canonical = [str(item) for item in action_ids]
    canonical_set = set(canonical)
    declaration_ids = sorted(str(item) for item in declaration_groups)
    if not declaration_ids:
        return {
            "budget": budget,
            "declaration_count": 0,
            "minimum_distinct_actions": 0,
            "feasible": True,
            "selected_groups": {},
            "selected_action_ids": [],
        }
    choices: list[tuple[tuple[str, ...], ...]] = []
    for declaration_id in declaration_ids:
        raw_groups = declaration_groups[declaration_id]
        groups = tuple(
            tuple(dict.fromkeys(str(action_id) for action_id in raw_group))
            for raw_group in raw_groups
        )
        if not groups or any(len(group) < 2 for group in groups):
            raise ValueError(
                f"declared relation {declaration_id} has no valid action group"
            )
        unknown = sorted(
            {
                action_id
                for group in groups
                for action_id in group
                if action_id not in canonical_set
            }
        )
        if unknown:
            raise ValueError(
                f"declared relation {declaration_id} references unknown actions: "
                + ", ".join(unknown)
            )
        choices.append(groups)

    candidates: list[
        tuple[int, tuple[str, ...], tuple[tuple[str, ...], ...]]
    ] = []
    for selected_groups in itertools.product(*choices):
        selected_set = {
            action_id
            for group in selected_groups
            for action_id in group
        }
        ordered_actions = tuple(
            action_id for action_id in canonical if action_id in selected_set
        )
        candidates.append(
            (len(ordered_actions), ordered_actions, selected_groups)
        )
    minimum_size, selected_actions, selected_groups = min(candidates)
    return {
        "budget": budget,
        "declaration_count": len(declaration_ids),
        "minimum_distinct_actions": minimum_size,
        "feasible": minimum_size <= budget,
        "selected_groups": {
            declaration_id: list(group)
            for declaration_id, group in zip(
                declaration_ids,
                selected_groups,
                strict=True,
            )
        },
        "selected_action_ids": list(selected_actions),
    }


def _audit_rate_law_alignment(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
    recipes: Mapping[str, Mapping[str, Any]],
    response_certificate: Mapping[str, Any],
) -> None:
    """Prove that a declared semantic pathway, and only that pathway, is changed."""

    change = intervention.get("rate_law_change")
    explicit = isinstance(change, Mapping)
    _add(
        findings,
        f"{candidate_id}:explicit_rate_law_change_contract",
        explicit,
        "A formal rate-law cell must declare its reaction role, transform, and "
        "calibration instead of selecting a reaction by declaration order.",
    )
    if not isinstance(change, Mapping):
        return
    declared_transform_id = str(change.get("transform_id", ""))
    required_fields = {
        "reaction_role",
        "transform_id",
        *RATE_LAW_TRANSFORM_CALIBRATION_FIELDS.get(
            declared_transform_id,
            frozenset(),
        ),
    }
    missing = sorted(required_fields - set(change))
    unsupported_transform = declared_transform_id not in RATE_LAW_TRANSFORM_CALIBRATION_FIELDS
    _add(
        findings,
        f"{candidate_id}:complete_rate_law_change_contract",
        not missing and not unsupported_transform,
        f"Missing rate-law contract fields: {missing}; "
        f"unsupported_transform={unsupported_transform}.",
    )
    if missing or unsupported_transform:
        return

    generator = DefaultScenarioGenerator()
    error: str | None = None
    changed_reactions: list[str] = []
    target_reaction_id: str | None = None
    target_role: str | None = None
    executed_transform_id: str | None = None
    domain_parameters_unchanged = False
    baseline_rate_law: Any | None = None
    shifted_rate_law: Any | None = None
    try:
        scenario = get_scenario(task_id)
        baseline = generator.generate(scenario, 0)
        shifted = generator.generate(scenario, 0, (dict(intervention),))
        for before, after in zip(
            baseline.compiled_mechanism.network.reactions,
            shifted.compiled_mechanism.network.reactions,
            strict=True,
        ):
            if before.rate_law.to_dict() != after.rate_law.to_dict():
                changed_reactions.append(before.reaction_id)
                baseline_rate_law = before.rate_law
                shifted_rate_law = after.rate_law
        metadata = shifted.compiled_mechanism.network.metadata
        target_reaction_id = str(metadata.get("derived_family_target_reaction_id", ""))
        target_role = str(metadata.get("derived_family_target_reaction_role", ""))
        executed_transform_id = str(metadata.get("derived_family_transform_id", ""))
        domain_parameters_unchanged = (
            baseline.parameters.domain_parameters == shifted.parameters.domain_parameters
        )
    except Exception as exc:  # pragma: no cover - details are returned to the audit
        error = f"{type(exc).__name__}: {exc}"

    _add(
        findings,
        f"{candidate_id}:single_declared_reaction_changed",
        error is None and len(changed_reactions) == 1 and changed_reactions == [target_reaction_id],
        "Exactly one rate law must change and it must equal the resolved semantic "
        f"target; changed={changed_reactions}, target={target_reaction_id}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:semantic_reaction_role_bound",
        error is None and target_role == str(change["reaction_role"]),
        "The executed mechanism must bind the declared reaction role; "
        f"declared={change['reaction_role']}, executed={target_role}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:declared_rate_law_transform_bound",
        error is None and executed_transform_id == str(change["transform_id"]),
        "The executed mechanism must bind the declared transform; "
        f"declared={change['transform_id']}, "
        f"executed={executed_transform_id}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:constitutive_domain_parameters_unchanged",
        error is None and domain_parameters_unchanged,
        "A reaction rate-law intervention must not also change crystallization or "
        f"other constitutive domain parameters; unchanged={domain_parameters_unchanged}, "
        f"error={error}.",
    )
    if str(change["transform_id"]) == CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS:
        _audit_pivot_rate_law_response(
            findings,
            candidate_id=candidate_id,
            recipes=recipes,
            response_certificate=response_certificate,
            baseline_rate_law=baseline_rate_law,
            shifted_rate_law=shifted_rate_law,
            error=error,
        )


def _audit_pivot_rate_law_response(
    findings: list[dict[str, Any]],
    *,
    candidate_id: str,
    recipes: Mapping[str, Mapping[str, Any]],
    response_certificate: Mapping[str, Any],
    baseline_rate_law: Any | None,
    shifted_rate_law: Any | None,
    error: str | None,
) -> None:
    """Require a bounded, sign-crossing response over frozen public doses."""

    declared = (
        response_certificate.get("required") is True
        and response_certificate.get("require_crosses_unity") is True
    )
    _add(
        findings,
        f"{candidate_id}:rate_law_response_certificate_declared",
        declared,
        "Pivot-normalized rate-law cells require a preregistered response envelope.",
    )
    if not declared:
        return
    response_error = error
    activities: list[float] = []
    multipliers: list[float] = []
    try:
        if baseline_rate_law is None or shifted_rate_law is None:
            raise ValueError("resolved baseline and shifted rate laws are required")
        baseline_parameters = baseline_rate_law.parameters
        shifted_parameters = shifted_rate_law.parameters
        reference_concentration = float(baseline_parameters["reference_concentration_mol_L"])
        baseline_order = float(baseline_parameters["activity_order"])
        shifted_order = float(shifted_parameters["activity_order"])
        scale_ratio = float(shifted_parameters["A"]) / float(baseline_parameters["A"])
        for recipe in recipes.values():
            steps = recipe.get("steps", ())
            solvent_volume = sum(
                float(step.get("volume_L", 0.0))
                for step in steps
                if step.get("operation") == "add_solvent"
            )
            catalyst_amount = sum(
                float(step.get("catalyst_amount_mol", 0.0))
                for step in steps
                if step.get("operation") == "add_catalyst"
            )
            if solvent_volume <= 0.0 or catalyst_amount <= 0.0:
                raise ValueError("each frozen recipe requires positive solvent and catalyst")
            activity = catalyst_amount / solvent_volume / reference_concentration
            multiplier = scale_ratio * activity ** (shifted_order - baseline_order)
            activities.append(float(activity))
            multipliers.append(float(multiplier))
        if len({round(value, 12) for value in activities}) < 2 or not all(
            np.isfinite(value) and value > 0.0 for value in multipliers
        ):
            raise ValueError("frozen recipes do not span finite positive rate responses")
    except Exception as exc:  # pragma: no cover - details are returned to the audit
        response_error = f"{type(exc).__name__}: {exc}"

    minimum = float(response_certificate["minimum_response_multiplier"])
    maximum = float(response_certificate["maximum_response_multiplier"])
    valid_envelope = 0.0 < minimum < 1.0 < maximum
    bounded = (
        response_error is None
        and valid_envelope
        and bool(multipliers)
        and min(multipliers) >= minimum
        and max(multipliers) <= maximum
    )
    crosses_unity = (
        response_error is None
        and bool(multipliers)
        and min(multipliers) < 1.0
        and max(multipliers) > 1.0
    )
    _add(
        findings,
        f"{candidate_id}:bounded_frozen_rate_response",
        bounded,
        "Frozen public catalyst doses must remain inside the preregistered "
        f"response envelope; activities={activities}, multipliers={multipliers}, "
        f"bounds=({minimum}, {maximum}), valid_envelope={valid_envelope}, "
        f"error={response_error}.",
    )
    _add(
        findings,
        f"{candidate_id}:frozen_rate_response_crosses_unity",
        crosses_unity,
        "A pivot-normalized response must make low/high public probes fall on "
        f"opposite sides of unity; multipliers={multipliers}, error={response_error}.",
    )


def _audit_environment_reset(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
) -> None:
    environment: ChemWorldEnv | None = None
    error: str | None = None
    try:
        environment = ChemWorldEnv(
            task_id=task_id,
            seed=0,
            world_interventions=(dict(intervention),),
        )
        environment.reset(seed=0)
    except Exception as exc:  # pragma: no cover - error text is reported to the caller
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if environment is not None:
            environment.close()
    _add(
        findings,
        f"{candidate_id}:environment_instantiation",
        error is None,
        "The task must reset under the declared intervention."
        + (f" error={error}" if error else ""),
    )


def _audit_topology_alignment(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
) -> None:
    """Prove that one declared reverse channel, and no other law, is added."""

    change = intervention.get("topology_change")
    required_fields = {
        "reaction_role",
        "transform_id",
        "reverse_rate_constant_s_inv_at_full_severity",
    }
    explicit = isinstance(change, Mapping)
    _add(
        findings,
        f"{candidate_id}:explicit_topology_change_contract",
        explicit,
        "A formal topology cell must declare its target role, transform, and "
        "reverse-channel calibration.",
    )
    if not isinstance(change, Mapping):
        _add(
            findings,
            f"{candidate_id}:complete_topology_change_contract",
            False,
            "Topology contract must contain exactly the registered fields; declared=None.",
        )
        return
    complete = (
        set(change) == required_fields
        and change.get("transform_id") == REVERSIBLE_TARGET_PATHWAY_STRESS
    )
    _add(
        findings,
        f"{candidate_id}:complete_topology_change_contract",
        complete,
        f"Topology contract must contain exactly the registered fields; declared={sorted(change)}.",
    )
    if not complete:
        return

    error: str | None = None
    existing_unchanged = False
    added_reaction_ids: list[str] = []
    target_reaction_id = ""
    target_role = ""
    transform_id = ""
    reverse_rate_constant: float | None = None
    domain_parameters_unchanged = False
    try:
        generator = DefaultScenarioGenerator()
        scenario = get_scenario(task_id)
        baseline = generator.generate(scenario, 0)
        shifted = generator.generate(scenario, 0, (dict(intervention),))
        baseline_reactions = baseline.compiled_mechanism.network.reactions
        shifted_reactions = shifted.compiled_mechanism.network.reactions
        existing_unchanged = all(
            before.to_dict() == after.to_dict()
            for before, after in zip(
                baseline_reactions,
                shifted_reactions[: len(baseline_reactions)],
                strict=True,
            )
        )
        added = shifted_reactions[len(baseline_reactions) :]
        added_reaction_ids = [reaction.reaction_id for reaction in added]
        metadata = shifted.compiled_mechanism.network.metadata
        target_reaction_id = str(metadata.get("derived_family_target_reaction_id", ""))
        target_role = str(metadata.get("derived_family_target_reaction_role", ""))
        transform_id = str(metadata.get("derived_family_transform_id", ""))
        reverse_rate_constant = float(
            cast(
                Any,
                metadata["derived_family_reverse_rate_constant_s_inv"],
            )
        )
        domain_parameters_unchanged = (
            baseline.parameters.domain_parameters == shifted.parameters.domain_parameters
        )
    except Exception as exc:  # pragma: no cover - details are returned to the audit
        error = f"{type(exc).__name__}: {exc}"

    expected_rate_constant = float(change["reverse_rate_constant_s_inv_at_full_severity"]) * float(
        intervention["severity"]
    )
    _add(
        findings,
        f"{candidate_id}:single_declared_topology_channel_added",
        error is None and existing_unchanged and added_reaction_ids == ["family_reverse_channel"],
        "Existing reactions must remain identical and exactly one declared channel "
        f"must be added; existing_unchanged={existing_unchanged}, "
        f"added={added_reaction_ids}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:semantic_topology_role_bound",
        error is None and target_role == str(change["reaction_role"]) and bool(target_reaction_id),
        "Executed topology must bind the declared semantic target; "
        f"target={target_reaction_id}, role={target_role}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:declared_topology_transform_bound",
        error is None and transform_id == str(change["transform_id"]),
        "Executed topology must bind the declared transform; "
        f"declared={change['transform_id']}, executed={transform_id}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:topology_rate_calibration_bound",
        error is None
        and reverse_rate_constant is not None
        and bool(np.isclose(reverse_rate_constant, expected_rate_constant)),
        "Executed reverse-channel rate must equal severity times the declared "
        f"full-severity value; expected={expected_rate_constant}, "
        f"executed={reverse_rate_constant}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:topology_domain_parameters_unchanged",
        error is None and domain_parameters_unchanged,
        "A topology intervention must not also change constitutive domain "
        f"parameters; unchanged={domain_parameters_unchanged}, error={error}.",
    )


def _audit_material_alignment(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
    recipes: Mapping[str, Mapping[str, Any]],
    recipe_values: Mapping[tuple[str, str], set[Any]],
    require_relational_pair: bool,
) -> None:
    field = str(intervention.get("material_field", ""))
    action_contract = MATERIAL_FIELD_ACTIONS.get(field)
    _add(
        findings,
        f"{candidate_id}:supported_material_field",
        action_contract is not None,
        f"Supported material targets are {sorted(MATERIAL_FIELD_ACTIONS)}.",
    )
    if action_contract is None:
        return
    operation, public_field = action_contract
    environment = ChemWorldEnv(task_id=task_id, seed=0)
    try:
        environment.reset(seed=0)
        schema = environment.action_schema(operation)
    finally:
        environment.close()
    field_schema: Mapping[str, Any] = next(
        (item for item in schema.get("fields", []) if item.get("field") == public_field),
        {},
    )
    choices = set(field_schema.get("choices", []))
    permutation = intervention.get("public_to_baseline")
    moved = (
        {index for index, value in enumerate(permutation) if index != value}
        if isinstance(permutation, list)
        else set()
    )
    _add(
        findings,
        f"{candidate_id}:public_operation_allowed",
        bool(schema.get("valid_operation_type")) and bool(schema.get("task_allowed")),
        f"{operation}.{public_field} must be available to the Agent in {task_id}.",
    )
    _add(
        findings,
        f"{candidate_id}:public_choice_cardinality",
        len(choices) >= 2,
        "A relational counterfactual needs at least two public choices; "
        f"choices={sorted(choices, key=str)}.",
    )
    _add(
        findings,
        f"{candidate_id}:moved_indices_publicly_reachable",
        bool(moved) and moved <= choices,
        "All moved public indices must be legal; "
        f"moved={sorted(moved)}, choices={sorted(choices, key=str)}.",
    )
    covered = recipe_values.get((operation, public_field), set())
    _add(
        findings,
        f"{candidate_id}:moved_indices_recipe_covered",
        bool(moved) and moved <= covered,
        "Frozen Gate A recipes must exercise every moved index; "
        f"moved={sorted(moved)}, covered={sorted(covered, key=str)}.",
    )
    if require_relational_pair:
        relational_groups = material_relational_action_groups(
            recipes,
            intervention=intervention,
        )
        _add(
            findings,
            f"{candidate_id}:moved_indices_relational_pair_covered",
            bool(relational_groups),
            "A relational counterfactual requires at least one public recipe "
            "group that differs only in the target material field and covers "
            f"all moved indices; groups={relational_groups}.",
        )


def material_relational_action_groups(
    recipes: Mapping[str, Mapping[str, Any]],
    *,
    intervention: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Return exact same-condition recipe groups covering one material permutation."""

    field = str(intervention.get("material_field", ""))
    action_contract = MATERIAL_FIELD_ACTIONS.get(field)
    permutation = intervention.get("public_to_baseline")
    if action_contract is None or not isinstance(permutation, list):
        return ()
    operation, public_field = action_contract
    moved = {
        index
        for index, baseline_index in enumerate(permutation)
        if index != baseline_index
    }
    if not moved:
        return ()

    by_signature: dict[str, list[tuple[str, Any]]] = {}
    for action_id, recipe in recipes.items():
        raw_steps = recipe.get("steps")
        if not isinstance(raw_steps, list):
            continue
        target_steps = [
            step
            for step in raw_steps
            if isinstance(step, Mapping)
            and step.get("operation") == operation
            and public_field in step
        ]
        if not target_steps:
            continue
        target_values = {step[public_field] for step in target_steps}
        if len(target_values) != 1:
            continue
        value = next(iter(target_values))
        normalized_steps: list[dict[str, Any]] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                normalized_steps = []
                break
            step = dict(raw_step)
            if step.get("operation") == operation:
                step.pop(public_field, None)
            normalized_steps.append(step)
        if not normalized_steps:
            continue
        signature = json.dumps(
            normalized_steps,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        by_signature.setdefault(signature, []).append((str(action_id), value))

    groups: list[tuple[str, ...]] = []
    for entries in by_signature.values():
        covered = {value for _, value in entries}
        if moved <= covered:
            groups.append(
                tuple(
                    sorted(
                        action_id
                        for action_id, value in entries
                        if value in moved
                    )
                )
            )
    return tuple(sorted(set(groups)))


def mechanism_relational_action_groups(
    recipes: Mapping[str, Mapping[str, Any]],
    *,
    relation: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Derive same-background mechanism probes from declared public controls."""

    raw_controls = relation.get("varied_fields", relation.get("control_fields"))
    if not isinstance(raw_controls, list):
        raise ValueError("diagnostic relation varied_fields must be a list")
    controls: set[tuple[str, str]] = set()
    for raw_control in raw_controls:
        if not isinstance(raw_control, Mapping):
            raise ValueError("diagnostic relation varied fields must be objects")
        operation = str(raw_control.get("operation", ""))
        field = str(raw_control.get("field", ""))
        if not operation or not field:
            raise ValueError(
                "diagnostic relation varied fields require operation and field"
            )
        controls.add((operation, field))
    minimum_actions = int(relation.get("minimum_distinct_actions", 0) or 0)
    if not controls or minimum_actions < 2:
        raise ValueError(
            "diagnostic relations require controls and at least two distinct actions"
        )

    by_background: dict[str, list[tuple[str, tuple[Any, ...]]]] = {}
    ordered_controls = sorted(controls)
    for action_id, recipe in recipes.items():
        raw_steps = recipe.get("steps")
        if not isinstance(raw_steps, list):
            continue
        normalized_steps: list[dict[str, Any]] = []
        values: dict[tuple[str, str], Any] = {}
        valid = True
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                valid = False
                break
            step = dict(raw_step)
            operation = str(step.get("operation", ""))
            for control_operation, field in controls:
                if operation == control_operation and field in step:
                    values[(control_operation, field)] = step.pop(field)
            normalized_steps.append(step)
        if not valid or set(values) != controls:
            continue
        signature = json.dumps(
            normalized_steps,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        by_background.setdefault(signature, []).append(
            (
                str(action_id),
                tuple(values[control] for control in ordered_controls),
            )
        )

    groups: set[tuple[str, ...]] = set()
    for members in by_background.values():
        members = sorted(members)
        for subset in itertools.combinations(members, minimum_actions):
            if len({values for _action_id, values in subset}) < minimum_actions:
                continue
            groups.add(tuple(action_id for action_id, _values in subset))
    return tuple(sorted(groups))


def _audit_constitutive_alignment(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
) -> None:
    """Bind a formal constitutive cell to one registered parameter contract."""

    change = intervention.get("constitutive_law_change")
    explicit = isinstance(change, Mapping)
    _add(
        findings,
        f"{candidate_id}:explicit_constitutive_change_contract",
        explicit,
        "A formal constitutive cell must declare its transform and all calibration parameters.",
    )
    if not isinstance(change, Mapping):
        _add(
            findings,
            f"{candidate_id}:complete_constitutive_change_contract",
            False,
            "Constitutive contract must exactly match the task transform registry; "
            "declared_fields=None.",
        )
        return
    transform_id = str(change.get("transform_id", ""))
    calibration_fields = CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS.get(
        transform_id,
        frozenset(),
    )
    required_fields = {"transform_id", *calibration_fields}
    expected_transform = CONSTITUTIVE_TASK_TRANSFORMS.get(task_id)
    complete = (
        bool(calibration_fields)
        and set(change) == required_fields
        and transform_id == expected_transform
    )
    _add(
        findings,
        f"{candidate_id}:complete_constitutive_change_contract",
        complete,
        "Constitutive contract must exactly match the task transform registry; "
        f"expected_transform={expected_transform}, declared_transform={transform_id}, "
        f"declared_fields={sorted(change) if isinstance(change, Mapping) else None}.",
    )
    if not complete:
        return

    error: str | None = None
    network_unchanged = False
    executed_transform = ""
    changed_domain_parameters: dict[str, float] = {}
    expected_domain_parameters: dict[str, float] = {}
    equilibrium_ratio: float | None = None
    expected_equilibrium_ratio: float | None = None
    try:
        generator = DefaultScenarioGenerator()
        scenario = get_scenario(task_id)
        baseline = generator.generate(scenario, 0)
        shifted = generator.generate(scenario, 0, (dict(intervention),))
        network_unchanged = (
            baseline.compiled_mechanism.mechanism_hash == shifted.compiled_mechanism.mechanism_hash
        )
        executed_transform = str(
            shifted.initial_state.metadata.get(
                "derived_constitutive_transform_id",
                "",
            )
        )
        severity = float(intervention["severity"])
        if transform_id == ELECTROCHEMICAL_RESPONSE_STRESS:
            expected_domain_parameters = {
                "electro_transfer_asymmetry_multiplier": (
                    1.0
                    + (float(change["transfer_asymmetry_multiplier_at_full_severity"]) - 1.0)
                    * severity
                ),
                "electro_selectivity_decay_multiplier": (
                    1.0
                    + (float(change["selectivity_decay_multiplier_at_full_severity"]) - 1.0)
                    * severity
                ),
                "electro_standard_potential_multiplier": (
                    1.0
                    + (float(change["standard_potential_multiplier_at_full_severity"]) - 1.0)
                    * severity
                ),
            }
        elif transform_id == PARTITION_POWER_RESPONSE_STRESS:
            expected_domain_parameters = {
                "partition_coefficient_exponent": (
                    1.0
                    + (float(change["partition_coefficient_exponent_at_full_severity"]) - 1.0)
                    * severity
                )
            }
        elif transform_id == EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS:
            expected_equilibrium_ratio = (
                float(change["activity_coefficient_ratio_at_full_severity"]) ** severity
            )
            equilibrium_ratio = float(
                shifted.initial_state.metadata["equilibrium_activity_coefficient_ratio"]
            )
        for key in sorted(
            set(baseline.parameters.domain_parameters)
            | set(shifted.parameters.domain_parameters)
        ):
            before = float(baseline.parameters.domain_parameters.get(key, 1.0))
            after = float(shifted.parameters.domain_parameters.get(key, 1.0))
            if not np.isclose(before, after):
                changed_domain_parameters[key] = after
    except Exception as exc:  # pragma: no cover - details are returned to the audit
        error = f"{type(exc).__name__}: {exc}"

    domain_calibration_bound = (
        error is None
        and set(changed_domain_parameters) == set(expected_domain_parameters)
        and all(
            np.isclose(changed_domain_parameters[key], expected)
            for key, expected in expected_domain_parameters.items()
        )
    )
    equilibrium_calibration_bound = expected_equilibrium_ratio is None or (
        equilibrium_ratio is not None and np.isclose(equilibrium_ratio, expected_equilibrium_ratio)
    )
    _add(
        findings,
        f"{candidate_id}:constitutive_network_unchanged",
        error is None and network_unchanged,
        "A constitutive intervention must not rewrite the reaction network; "
        f"unchanged={network_unchanged}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:declared_constitutive_transform_bound",
        error is None and executed_transform == transform_id,
        "Executed constitutive transform must equal the declared transform; "
        f"declared={transform_id}, executed={executed_transform}, error={error}.",
    )
    _add(
        findings,
        f"{candidate_id}:constitutive_calibration_bound",
        bool(domain_calibration_bound and equilibrium_calibration_bound),
        "Exactly the declared constitutive parameters must change to their "
        "severity-interpolated values; "
        f"expected_domain={json.dumps(expected_domain_parameters, sort_keys=True)}, "
        f"executed_domain={json.dumps(changed_domain_parameters, sort_keys=True)}, "
        f"expected_equilibrium_ratio={expected_equilibrium_ratio}, "
        f"executed_equilibrium_ratio={equilibrium_ratio}, error={error}.",
    )


def _audit_intervention_decision_relevance(
    findings: list[dict[str, Any]],
    *,
    task_id: str,
    candidate_id: str,
    intervention: Mapping[str, Any],
    action_library: Mapping[str, np.ndarray],
    certificate: Mapping[str, Any],
    baseline_cache: dict[tuple[int, str], dict[str, float]],
) -> None:
    """Check that every hidden-law change makes adaptation consequential."""

    if certificate.get("required") is not True:
        _add(
            findings,
            f"{candidate_id}:decision_relevance_certificate_declared",
            False,
            "Every changed mechanism requires a preregistered decision-relevance certificate.",
        )
        return
    seeds = [int(seed) for seed in certificate.get("world_seeds", ())]
    if not seeds or not action_library:
        _add(
            findings,
            f"{candidate_id}:decision_relevance_inputs",
            False,
            "Decision relevance requires non-empty world seeds and action library.",
        )
        return
    primary_metric = SERIOUS_TASK_DESIGNS[task_id].primary_metric
    required_metric_ids = tuple(
        str(item) for item in certificate.get("required_metric_ids", ("task_primary",))
    )
    supported_metric_ids = {"task_primary", "leaderboard_score"}
    unsupported = sorted(set(required_metric_ids) - supported_metric_ids)
    if not required_metric_ids or unsupported:
        _add(
            findings,
            f"{candidate_id}:decision_relevance_metrics",
            False,
            "Decision relevance requires supported metric IDs; "
            f"required={list(required_metric_ids)}, unsupported={unsupported}.",
        )
        return
    baseline_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    shifted_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        baseline_by_seed[seed] = {}
        shifted_by_seed[seed] = {}
        for action_id, vector in action_library.items():
            cache_key = (seed, action_id)
            if cache_key not in baseline_cache:
                baseline_cache[cache_key] = _execute_recipe_metrics(
                    task_id,
                    vector,
                    seed=seed,
                    observation_seed=1_700_000_000 + seed,
                    interventions=(),
                    primary_metric=primary_metric,
                )
            baseline_by_seed[seed][action_id] = baseline_cache[cache_key]
            shifted_by_seed[seed][action_id] = _execute_recipe_metrics(
                task_id,
                vector,
                seed=seed,
                observation_seed=1_700_000_000 + seed,
                interventions=(intervention,),
                primary_metric=primary_metric,
            )
    effect_threshold = float(certificate["minimum_median_max_metric_effect"])
    regret_threshold = float(certificate["minimum_median_old_policy_regret"])
    change_threshold = float(certificate["minimum_optimal_action_change_rate"])
    metric_results: dict[str, dict[str, float | bool | str]] = {}
    for metric_id in required_metric_ids:
        max_effects: list[float] = []
        old_policy_regrets: list[float] = []
        optimal_changes: list[float] = []
        for seed in seeds:
            baseline = {
                action_id: values[metric_id] for action_id, values in baseline_by_seed[seed].items()
            }
            shifted = {
                action_id: values[metric_id] for action_id, values in shifted_by_seed[seed].items()
            }
            baseline_best = max(baseline, key=baseline.__getitem__)
            shifted_best = max(shifted, key=shifted.__getitem__)
            max_effects.append(max(abs(shifted[key] - baseline[key]) for key in baseline))
            old_policy_regrets.append(max(shifted.values()) - shifted[baseline_best])
            optimal_changes.append(float(baseline_best != shifted_best))
        median_effect = float(np.median(max_effects))
        median_regret = float(np.median(old_policy_regrets))
        optimal_change_rate = float(np.mean(optimal_changes))
        passed = median_effect >= effect_threshold and (
            median_regret >= regret_threshold or optimal_change_rate >= change_threshold
        )
        metric_name = primary_metric if metric_id == "task_primary" else metric_id
        metric_results[metric_id] = {
            "metric": metric_name,
            "median_max_effect": median_effect,
            "median_old_policy_regret": median_regret,
            "optimal_action_change_rate": optimal_change_rate,
            "pass": passed,
        }
        _add(
            findings,
            f"{candidate_id}:decision_relevance:{metric_id}",
            passed,
            (
                f"metric={metric_name}; median_max_effect={median_effect:.6f}; "
                f"median_old_policy_regret={median_regret:.6f}; "
                f"optimal_action_change_rate={optimal_change_rate:.3f}; "
                f"thresholds=({effect_threshold:.6f}, {regret_threshold:.6f}, "
                f"{change_threshold:.3f})."
            ),
        )
    _add(
        findings,
        f"{candidate_id}:decision_relevance",
        all(bool(result["pass"]) for result in metric_results.values()),
        (
            "Every required scientific/utility metric must make the old policy "
            f"consequentially suboptimal; results={metric_results}."
        ),
    )


def _execute_recipe_metrics(
    task_id: str,
    vector: np.ndarray,
    *,
    seed: int,
    observation_seed: int,
    interventions: tuple[Mapping[str, Any], ...],
    primary_metric: str,
) -> dict[str, float]:
    task_info = get_task(task_id).to_dict()
    recipe = task_recipe_from_unit_vector(task_info, vector)
    environment = ChemWorldEnv(
        task_id=task_id,
        seed=seed,
        budget_override=len(recipe["steps"]) + 1,
        observation_seed_override=observation_seed,
        world_interventions=tuple(dict(item) for item in interventions),
    )
    try:
        environment.reset(seed=seed)
        info: dict[str, Any] = {}
        for action in recipe["steps"]:
            _observation, _reward, _terminated, _truncated, info = environment.step(action)
            if info.get("transaction_status") != "committed":
                raise RuntimeError(
                    f"decision-relevance recipe failed: {task_id}/{action['operation']}"
                )
        value = info.get("processed_estimate", {}).get(primary_metric)
        if value is None or not np.isfinite(float(value)):
            raise RuntimeError(f"decision-relevance recipe lacks primary metric {primary_metric}")
        leaderboard_score = info.get("leaderboard_score")
        if leaderboard_score is None or not np.isfinite(float(leaderboard_score)):
            raise RuntimeError("decision-relevance recipe lacks finite leaderboard_score")
        return {
            "task_primary": float(value),
            "leaderboard_score": float(leaderboard_score),
        }
    finally:
        environment.close()


def _audit_prompt_boundary(
    *,
    task_id: str,
    task_info: dict[str, Any],
    definitions: Mapping[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    specs = tuple(
        MechanismCandidateSpec(candidate_id=item, public_definition=str(definitions[item]))
        for item in candidate_ids
        if item in definitions
    )
    if len(specs) < 2 or "no_change" not in {item.candidate_id for item in specs}:
        return _finding(
            "private_truth_absent_from_agent_prompt",
            False,
            "Prompt audit requires complete candidate definitions including no_change.",
        )
    agent = MechanismAdaptationLiveLLMAgent(
        _PromptOnlyClient(),
        role_id="mechanism-design-audit",
        candidate_specs=specs,
        candidate_order_seed=0,
        randomize_candidate_order=True,
    )
    agent.reset(task_info, 0)
    context = AgentDecisionContext(
        step=0,
        task_id=task_id,
        decision_stage="experiment_setup",
        campaign_state={"remaining_budget": 1},
        visible_metrics={},
        latest_spectra={},
        uncertainty={},
        constraint_flags={},
        available_operations=tuple(task_info.get("allowed_operations", ())),
        previous_event_type=None,
    )
    prompt = json.loads(agent._build_prompt(context, {"tool_json": {}}))
    exposed = sorted(_collect_keys(prompt) & _FORBIDDEN_PUBLIC_KEYS)
    return _finding(
        "private_truth_absent_from_agent_prompt",
        not exposed,
        f"Forbidden intervention keys in prompt={exposed}.",
    )


def _recipe_field_values(
    recipes: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], set[Any]]:
    values: dict[tuple[str, str], set[Any]] = {}
    for recipe in recipes.values():
        for step in recipe.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            operation = str(step.get("operation", ""))
            for field, value in step.items():
                if field != "operation" and isinstance(value, str | int | float):
                    values.setdefault((operation, str(field)), set()).add(value)
    return values


def _has_preterminal_measurement(recipe: Mapping[str, Any]) -> bool:
    operations = [
        str(step.get("operation", ""))
        for step in recipe.get("steps", [])
        if isinstance(step, Mapping)
    ]
    try:
        terminate_index = operations.index("terminate")
    except ValueError:
        return False
    return "measure" in operations[:terminate_index]


def _collect_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested for item in value.values() for nested in _collect_keys(item)
        }
    if isinstance(value, list | tuple):
        return {nested for item in value for nested in _collect_keys(item)}
    return set()


def _finding(check: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": check, "pass": bool(passed), "detail": detail}


def _add(findings: list[dict[str, Any]], check: str, passed: bool, detail: str) -> None:
    findings.append(_finding(check, passed, detail))


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    event: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback({"stage": "mechanism_design_audit", "event": event, **details})


__all__ = [
    "DESIGN_AUDIT_SCHEMA_VERSION",
    "MATERIAL_FIELD_ACTIONS",
    "audit_mechanism_design",
    "declared_relational_action_groups",
    "material_relational_action_groups",
    "mechanism_relational_action_groups",
    "relational_coverage_witness",
]
