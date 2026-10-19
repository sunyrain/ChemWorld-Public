"""Task, render, and step-info report builders for ChemWorldEnv."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from chemworld import __version__
from chemworld.backends import semi_mechanistic_backend_spec
from chemworld.envs.spaces import OBSERVATION_KEYS, value_or_default
from chemworld.foundation.solvents import MIXTURE_MODEL_ID
from chemworld.foundation.state import OperationRecord
from chemworld.materials import public_material_catalog
from chemworld.physchem.coupled_flow import FLOW_MODEL
from chemworld.runtime.component_network import public_network
from chemworld.runtime.continuous_streams import public_streams
from chemworld.runtime.control_program import control_settings, public_control
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.spectral_settings import public_spectral_settings
from chemworld.world.control_contract import CONTROL_MODEL
from chemworld.world.instruments import instrument_contracts
from chemworld.world.operations import (
    OPERATION_TYPES,
    chemworld_operations,
    chemworld_state_variable_contracts,
    operation_contracts,
)
from chemworld.world.scoring import risk_signal_contract, safety_cost_from_flags
from chemworld.world.thermal_control import REACTION_THERMAL_CONTROL_ID
from chemworld.world.world_law import world_law_spec


def build_task_info(env: Any) -> dict[str, Any]:
    compiled_mechanism = env.scenario_instance.compiled_mechanism
    task_spec = env.task_spec
    payload = {
        "env_id": "ChemWorld",
        "task_id": env.task_id,
        "world_law_id": env.world.family_version,
        "scenario_id": env.scenario_spec.scenario_id,
        "scenario": build_public_scenario_summary(env),
        "initial_state_id": env.scenario_spec.initial_state_id,
        "world_split": env.world_split,
        "objective": env.objective,
        "budget": env.budget,
        "official_budget": env.official_budget,
        "episode_mode": env.episode_mode,
        "contract_profile": env.contract_profile,
        "safety_limit": env.safety_limit,
        "task_contract_hash": (None if task_spec is None else task_spec.contract_hash),
        "description": None if task_spec is None else task_spec.description,
        "termination_policy": (None if task_spec is None else task_spec.termination_policy),
        "observation_policy": (None if task_spec is None else task_spec.observation_policy),
        "success_metrics": ([] if task_spec is None else list(task_spec.success_metrics)),
        "runtime_profile_hash": env.runtime.profile.profile_hash,
        "reaction_thermal_control_id": REACTION_THERMAL_CONTROL_ID,
        "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
        "risk_signal_contract": risk_signal_contract(),
        "mechanism_summary": build_public_mechanism_summary(compiled_mechanism),
        "scoring_contract": env.scoring_contract.to_dict(),
        "scoring_contract_hash": env.scoring_contract.contract_hash,
        "observation_contract": build_public_observation_contract_summary(env),
        "observation_contract_hash": env.observation_contract.contract_hash,
        "env_version": __version__,
        "world_family_version": env.world.family_version,
        "operation_types": list(OPERATION_TYPES),
        "allowed_operations": sorted(env.allowed_operations),
        "allowed_instruments": sorted(env.allowed_instruments),
        "material_catalog": public_material_catalog(task_id=env.task_id),
        "control_model": deepcopy(CONTROL_MODEL),
        "continuous_stream_model": deepcopy(FLOW_MODEL),
        "mixture_model": {
            "id": MIXTURE_MODEL_ID,
            "catalog_size": 4,
            "volume_rule": "additive_carrier_volume_L",
            "heat_capacity_rule": "volume_weighted_endpoint_ratios",
            "positive_property_rule": "log_volume_fraction_interpolation",
            "domain": "finite_surrogate_media_not_calibrated_real_solvent_mixtures",
            "solute_volume_rule": "dilute_tracer_solutes_do_not_add_carrier_volume",
        },
        "feed_catalog": [
            {
                "id": f"feed-{i}",
                "component": i,
                "reference_recipe_mol_per_amount_mol": MechanismSpeciesView(compiled_mechanism)
                .reagent_charge_amounts(
                    env.scenario_instance.initial_state, limiting_amount_mol=1.0
                )
                .get(species, 0.0),
            }
            for i, species in enumerate(MechanismSpeciesView(compiled_mechanism).feed_species)
        ],
        "kernel_maturity": env.kernel_maturity.to_dict(),
        "physics_maturity": env.kernel_maturity.lowest_level.value,
        "proxy_allowed": env.kernel_maturity.proxy_allowed,
        "instruments": {
            key: contract.to_dict()
            for key, contract in instrument_contracts(
                include_particle_size=getattr(env, "full_process_contract_id", None) is not None,
            ).items()
        },
        "operations": [operation.to_dict() for operation in chemworld_operations()],
        "operation_contracts": {
            key: contract.to_dict() for key, contract in operation_contracts().items()
        },
        "state_variables": [
            variable.to_dict() for variable in chemworld_state_variable_contracts()
        ],
        "constitution": build_constitution_summary(env),
        "world_law": world_law_spec().to_dict(),
        "backend": semi_mechanistic_backend_spec().to_dict(),
        "observation_keys": list(OBSERVATION_KEYS),
    }
    if getattr(env, "full_process_contract_id", None) is not None:
        payload["full_process_contract_id"] = env.full_process_contract_id
    if env.task_id == "electrochemical-conversion":
        payload["electrochemical_workflow_mode"] = env.electrochemical_workflow_mode
    if env.compiled_composition is not None:
        payload["composition"] = env.compiled_composition.to_public_dict()
        payload["composition_request"] = env.compiled_composition.spec.to_dict()
    if env.task_id in {
        "electrochemical-conversion",
        "partition-discovery",
        "reaction-to-crystallization",
        "reaction-to-distillation",
    }:
        material_information = {
            "mode": env.material_information_condition,
        }
        dossier = env.material_information_dossier()
        if dossier is not None:
            material_information["dossier"] = dossier
            material_information["dossier_sha256"] = env.material_information_sha256
        payload["material_information"] = material_information
    campaign_resources = env.public_campaign_resource_state(include_card=True)
    if getattr(env, "research_brief", None) is not None:
        from chemworld.research_brief import public_research_brief

        payload["research_brief"] = public_research_brief(env.research_brief)
        payload["description"] = payload["research_brief"]["commission"]
    if campaign_resources is not None:
        payload["campaign_resources"] = campaign_resources
    if env.debug_truth:
        payload["debug_mechanism"] = {
            "mechanism_manifest": compiled_mechanism.manifest.to_dict(),
            "reactions": [reaction.to_dict() for reaction in compiled_mechanism.network.reactions],
            "scenario_card": env.scenario_instance.to_card(),
        }
    return payload


def build_public_scenario_summary(env: Any) -> dict[str, Any]:
    spec = env.scenario_spec
    return {
        "scenario_id": spec.scenario_id,
        "world_law_id": spec.world_law_id,
        "family": spec.family,
        "split": spec.split,
        "difficulty": spec.difficulty,
        "initial_state_id": spec.initial_state_id,
        "allowed_module_tags": list(spec.allowed_module_tags),
        "expected_qualitative_behavior": list(spec.expected_qualitative_behavior),
        "world_family_version": env.world.family_version,
        "hidden_parameter_policy": (
            "hidden parameter seeds, parameter profiles, and generated mechanism "
            "internals are not exposed in public task_info"
        ),
    }


def build_public_mechanism_summary(compiled_mechanism: Any) -> dict[str, Any]:
    return {
        "instance_identity_visible": False,
        "topology_summary_visible": False,
        "parameter_summary_visible": False,
        "public_boundary": (
            "instance identity, hashes, topology counts, species ids, reactions, "
            "stoichiometry, rate laws, and parameter policies are evaluator-private"
        ),
    }


def build_evaluator_provenance(env: Any) -> dict[str, Any]:
    """Return replay identity that must never be passed to an Agent."""

    mechanism = env.scenario_instance.compiled_mechanism
    metadata = env.scenario_instance.initial_state.metadata
    return {
        "seed": env.seed,
        "world_seed": env.seed,
        "world_id": env.world.world_id,
        "world_provider": env.world.provider,
        "mechanism_id": mechanism.mechanism_id,
        "mechanism_hash": mechanism.mechanism_hash,
        "mechanism_version": mechanism.mechanism_version,
        "observation_seed": env._observation_seed(env.seed),
        "observation_noise_mode": env.observation_noise_mode,
        "observation_noise_namespace": env.observation_noise_namespace,
        "last_observation_noise": env.observation_noise_provenance(),
        "campaign_resource_card_sha256": (
            None if env.campaign_resource_card is None else env.campaign_resource_card.card_sha256
        ),
        "world_family_intervention_version": metadata.get("world_family_intervention_version"),
        "world_family_intervention_hash": metadata.get("world_family_intervention_hash"),
        "mechanism_family_intervention_version": metadata.get(
            "mechanism_family_intervention_version"
        ),
        "mechanism_family_intervention_hash": metadata.get("mechanism_family_intervention_hash"),
        "electrochemical_material_family_contract_version": metadata.get(
            "electrochemical_material_family_contract_version"
        ),
        "electrochemical_material_family_id": metadata.get("electrochemical_material_family_id"),
        "electrochemical_material_family_sha256": metadata.get(
            "electrochemical_material_family_sha256"
        ),
        "electrochemical_material_instance_sha256": metadata.get(
            "electrochemical_material_instance_sha256"
        ),
        "crystallization_material_family_contract_version": metadata.get(
            "crystallization_material_family_contract_version"
        ),
        "crystallization_material_family_id": metadata.get("crystallization_material_family_id"),
        "crystallization_material_family_sha256": metadata.get(
            "crystallization_material_family_sha256"
        ),
        "crystallization_material_instance_sha256": metadata.get(
            "crystallization_material_instance_sha256"
        ),
        "material_law_counterfactual_version": metadata.get("material_law_counterfactual_version"),
        "material_law_counterfactual_hash": metadata.get("material_law_counterfactual_hash"),
        # The public task payload intentionally exposes only the material
        # dossier (or opaque mode).  The evaluator still needs the exact
        # configuration to reconstruct a misindexed arm during trajectory
        # replay, so keep it in evaluator-only provenance.
        "material_information_config": deepcopy(getattr(env, "material_information_config", None)),
    }


_PRIVATE_AGENT_IDENTITY_KEYS = frozenset(
    {
        "base_mechanism_hash",
        "electrochemical_material_family_contract_version",
        "electrochemical_material_family_id",
        "electrochemical_material_family_sha256",
        "electrochemical_material_instance_sha256",
        "crystallization_material_family_contract_version",
        "crystallization_material_family_id",
        "crystallization_material_family_sha256",
        "crystallization_material_instance_sha256",
        "campaign_resource_card_sha256",
        "seed",
        "world_seed",
        "observation_seed",
        "material_law_counterfactual_hash",
        "material_law_counterfactual_version",
        "mechanism_family_intervention_hash",
        "mechanism_family_intervention_version",
        "mechanism_hash",
        "mechanism_id",
        "mechanism_version",
        "last_observation_noise",
        "observation_noise_mode",
        "observation_noise_namespace",
        "world_family_intervention_hash",
        "world_family_intervention_version",
        "world_id",
        "world_provider",
    }
)


def sanitize_agent_info(payload: Any) -> Any:
    """Recursively remove evaluator-only identity from Agent-visible payloads."""

    if isinstance(payload, Mapping):
        return {
            key: sanitize_agent_info(value)
            for key, value in payload.items()
            if str(key) not in _PRIVATE_AGENT_IDENTITY_KEYS and not str(key).startswith("_hidden_")
        }
    if isinstance(payload, list):
        return [sanitize_agent_info(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(sanitize_agent_info(value) for value in payload)
    return payload


_PUBLIC_STATE_DELTA_KEYS = frozenset(
    {
        "delta_time_s",
        "delta_cost",
        "delta_risk",
        "delta_temperature_K",
        "delta_volume_L",
        "configured_potential_V",
        "configured_current_mA",
    }
)


def _agent_state_delta_summary(env: Any, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project an operation record onto declared public process telemetry."""

    if env.debug_truth:
        return deepcopy(dict(summary))
    return {
        key: deepcopy(value)
        for key, value in summary.items()
        if str(key) in _PUBLIC_STATE_DELTA_KEYS
    }


def build_public_observation_contract_summary(env: Any) -> dict[str, Any]:
    contract = env.observation_contract
    return {
        "success_metrics": list(contract.success_metrics),
        "score_family": contract.score_family,
        "required_observation_keys": list(contract.required_observation_keys),
        "instrument_observable_keys": {
            instrument_id: list(keys)
            for instrument_id, keys in sorted(contract.instrument_observable_keys.items())
        },
        "public_species_label_policy": [
            "reactant_public",
            "target_public",
            "impurity_public",
            "degradation_public",
        ],
        "mapping_visibility_policy": "hidden mechanism-to-species mapping is not public",
        "contract_hash": contract.contract_hash,
    }


def build_constitution_summary(env: Any) -> dict[str, Any]:
    state_report = env.constitution.check_state(env._state)
    return {
        "name": "PhysicalConstitutionChecklist",
        "passed": state_report.passed,
        "checks": _public_constitution_checks(
            env,
            state_report.to_list(),
        ),
        "rules": [
            "material_conservation",
            "nonnegative_state",
            "species_registry_membership",
            "state_numeric_values_finite",
            "unit_consistency",
            "yield_upper_bound",
            "energy_balance",
            "phase_mass_balance",
            "observation_non_omniscient",
            "observation_values_finite_and_bounded",
            "observation_mask_consistent",
            "observation_signal_and_accounting_integrity",
            "measurement_has_cost",
            "action_preconditions",
            "safety_constraints",
            "public_private_reproducibility",
        ],
    }


def render_env(env: Any) -> Any:
    """Render a concise visible campaign summary."""

    last_operation = (
        None if env._last_operation_record is None else env._last_operation_record.operation_type
    )
    lines = [
        "ChemWorld",
        f"  task: {env.task_id or 'ad-hoc'}",
        f"  scenario: {None if env.scenario_spec is None else env.scenario_spec.scenario_id}",
        f"  campaign: {env._campaign_id}",
        f"  step: {env._step_count}/{env.budget}",
        f"  experiment: {env._experiment_index}",
        f"  last_operation: {last_operation}",
        (
            "  ledger: "
            f"time_s={env._state.ledger.time_s:.1f}, "
            f"cost={env._state.ledger.cost:.3f}, "
            f"risk={env._state.ledger.risk:.3f}, "
            f"sample_L={env._state.ledger.sample_consumed_L:.6f}"
        ),
    ]
    visible = {
        key: float(value[0])
        for key, value in env._last_observation.items()
        if np.isfinite(float(value[0]))
    }
    lines.append(f"  visible_observation: {visible}")
    rendered = "\n".join(lines)
    if env.render_mode == "human":
        print(rendered)
        return None
    return rendered


def build_step_info(
    env: Any,
    operation_record: OperationRecord,
    observation: Any,
) -> dict[str, Any]:
    values = observation.values
    checks = _public_constitution_checks(
        env,
        operation_record.constitution_checks,
    )
    constitution_failed = any(not bool(check.get("passed", False)) for check in checks)
    precondition_failed = not all(operation_record.preconditions.values())
    observed_keys = [key for key, observed in observation.observed_mask.items() if observed]
    if precondition_failed:
        reward_source = "failed_precondition"
    elif operation_record.operation_type == "measure":
        reward_source = f"instrument:{operation_record.instrument}"
    elif any(key in observed_keys for key in ("yield", "selectivity", "conversion")):
        reward_source = "carried_observation_with_public_ledger"
    else:
        reward_source = "public_ledger_only"
    score = value_or_default(values, "score")
    failed_preconditions = [
        key for key, passed in operation_record.preconditions.items() if not passed
    ]
    constraint_flags = {
        "unsafe": value_or_default(values, "safety_risk") >= env.safety_limit,
        "unsafe_by_task_limit": (value_or_default(values, "safety_risk") >= env.safety_limit),
        "high_cost": value_or_default(values, "cost") >= 0.75,
        "low_selectivity": value_or_default(values, "selectivity") <= 0.35,
        "degradation_detected": value_or_default(values, "degradation_warning") >= 0.28,
        "constitution_failed": constitution_failed,
        "precondition_failed": precondition_failed,
        "phase_mass_balance_failed": any(
            check.get("name") == "phase_mass_balance" and not check.get("passed", False)
            for check in checks
        ),
    }
    cost_signal, cost_components = safety_cost_from_flags(constraint_flags)
    payload = {
        "step": env._step_count,
        "budget": env.budget,
        "remaining_budget": max(env.budget - env._step_count, 0),
        "campaign_id": env._campaign_id,
        "episode_mode": env.episode_mode,
        "experiment_index": env._experiment_index,
        "operation_id": env._operation_id,
        "experiment_ended": False,
        "experiment_summaries": deepcopy(env._experiment_summaries),
        "sample_inventory": env._state.samples.public_summary(),
        "material_routing": env._state.samples.routing_summary(),
        "component_network": public_network(env._state),
        "process_control": public_control(env._state),
        "continuous_streams": public_streams(env._state),
        "spectral_configurations": public_spectral_settings(env._state),
        "control_trace": (
            control_settings(env._state).get("last_samples", [])
            if operation_record.operation_type == "advance_control"
            else []
        ),
        "task_id": env.task_id,
        "scenario_id": None if env.scenario_spec is None else env.scenario_spec.scenario_id,
        "initial_state_id": env.scenario_spec.initial_state_id,
        "world_law_id": env.world.family_version,
        "world_split": env.world_split,
        "objective": env.objective,
        "safety_limit": env.safety_limit,
        "task_contract_hash": (None if env.task_spec is None else env.task_spec.contract_hash),
        "runtime_profile_hash": env.runtime.profile.profile_hash,
        "scoring_contract_hash": env.scoring_contract.contract_hash,
        "observation_contract_hash": env.observation_contract.contract_hash,
        "operation_type": operation_record.operation_type,
        "operation_allowed_by_task": operation_record.operation_type in env.allowed_operations,
        "instrument_allowed_by_task": (
            operation_record.operation_type != "measure"
            or operation_record.instrument in env.allowed_instruments
        ),
        "preconditions": deepcopy(operation_record.preconditions),
        "state_delta_summary": _agent_state_delta_summary(
            env, operation_record.state_delta_summary
        ),
        "constitution_checks": checks,
        "instrument": operation_record.instrument,
        "instrument_source": observation.instrument_id,
        "observed_keys": observed_keys,
        "observed_mask": deepcopy(observation.observed_mask),
        "raw_signal": deepcopy(observation.raw_signal),
        "processed_estimate": deepcopy(observation.processed_estimate),
        "uncertainty": deepcopy(observation.uncertainty),
        "measurement_cost": operation_record.measurement_cost,
        "sample_consumed": operation_record.sample_consumed_L,
        "observed_reward": score,
        "leaderboard_score": (
            score
            if operation_record.instrument == "final_assay" and not precondition_failed
            else None
        ),
        "reward_source": reward_source,
        "cost": cost_signal,
        "cost_components": cost_components,
        "constraint_budget_remaining": max(1.0 - cost_signal, 0.0),
        "error_message": (
            None
            if not failed_preconditions
            else f"Action precondition failed: {', '.join(failed_preconditions)}"
        ),
        "constraint_flags": constraint_flags,
        "risk_signal_contract": risk_signal_contract(),
        "env_version": __version__,
        "world_family_version": env.world.family_version,
    }
    campaign_resources = env.public_campaign_resource_state()
    if campaign_resources is not None:
        payload["campaign_resources"] = campaign_resources
    return payload


def _public_constitution_checks(
    env: Any,
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove hidden-state identities and values from agent-facing checks."""

    if env.debug_truth:
        return [dict(check) for check in checks]
    public_checks: list[dict[str, object]] = []
    seen: set[tuple[str, bool]] = set()
    for check in checks:
        name = str(check.get("name", "constitution_check"))
        tokens = name.split(":")
        if len(tokens) >= 3 and tokens[:2] == ["nonnegative", "amount"]:
            name = "nonnegative:hidden_species_amount"
        elif tokens and tokens[0] in {
            "phase_amount_nonnegative",
            "phase_amount_finite_nonnegative",
        }:
            name = ":".join((*tokens[:-1], "hidden_species"))
        identity = (name, bool(check.get("passed", False)))
        if identity in seen:
            continue
        seen.add(identity)
        public_checks.append(
            {
                "name": name,
                "passed": bool(check.get("passed", False)),
                "message": "",
                "value": None,
                "tolerance": None,
            }
        )
    return public_checks


def annotate_constitution_rollback(info: dict[str, Any]) -> dict[str, Any]:
    """Preserve a rejected candidate-state failure in public step reporting."""

    if info.get("rollback_reason") != "constitution_failed":
        return info
    flags = {
        **dict(info.get("constraint_flags", {})),
        "constitution_failed": True,
    }
    cost_signal, cost_components = safety_cost_from_flags(flags)
    info.update(
        {
            "constraint_flags": flags,
            "cost": cost_signal,
            "cost_components": cost_components,
            "constraint_budget_remaining": max(1.0 - cost_signal, 0.0),
            "reward_source": "constitution_rollback",
            "leaderboard_score": None,
        }
    )
    if info.get("error_message") is None:
        info["error_message"] = "Transaction rolled back: constitution_failed"
    return info


__all__ = [
    "annotate_constitution_rollback",
    "build_constitution_summary",
    "build_evaluator_provenance",
    "build_public_mechanism_summary",
    "build_public_observation_contract_summary",
    "build_public_scenario_summary",
    "build_step_info",
    "build_task_info",
    "render_env",
    "sanitize_agent_info",
]
