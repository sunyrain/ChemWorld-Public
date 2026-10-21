"""Trajectory replay verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.data.validation import validate_records
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID
from chemworld.world.thermal_control import REACTION_THERMAL_CONTROL_ID


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    checked_steps: int
    max_abs_error: float
    mismatches: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "checked_steps": self.checked_steps,
            "max_abs_error": self.max_abs_error,
            "mismatches": self.mismatches,
        }


def _objective_from_record(record: dict[str, Any]) -> str:
    if "objective" in record:
        return str(record["objective"])
    parts = str(record["task_id"]).split(":")
    if len(parts) >= 3:
        return parts[2]
    return "balanced"


def _scalar_observation(observation: dict[str, Any]) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    for key, value in observation.items():
        scalar = float(value.reshape(-1)[0])
        payload[key] = scalar if isfinite(scalar) else None
    return payload


def _jsonish_mismatches(
    *,
    step: int,
    field: str,
    recorded: Any,
    replayed: Any,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compare replay metadata while tolerating deterministic float roundoff."""

    if isinstance(recorded, bool) or isinstance(replayed, bool):
        if recorded == replayed:
            return []
        return [
            {
                "step": step,
                "field": field,
                "recorded": recorded,
                "replayed": replayed,
                "abs_error": None,
            }
        ]
    if isinstance(recorded, float | int) and isinstance(replayed, float | int):
        error = abs(float(recorded) - float(replayed))
        if error <= tolerance:
            return []
        return [
            {
                "step": step,
                "field": field,
                "recorded": recorded,
                "replayed": replayed,
                "abs_error": error,
            }
        ]
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        mismatches: list[dict[str, Any]] = []
        recorded_keys = set(recorded)
        replayed_keys = set(replayed)
        for key in sorted(recorded_keys | replayed_keys):
            child_field = f"{field}.{key}"
            if key not in recorded or key not in replayed:
                mismatches.append(
                    {
                        "step": step,
                        "field": child_field,
                        "recorded": recorded.get(key),
                        "replayed": replayed.get(key),
                        "abs_error": None,
                    }
                )
                continue
            mismatches.extend(
                _jsonish_mismatches(
                    step=step,
                    field=child_field,
                    recorded=recorded[key],
                    replayed=replayed[key],
                    tolerance=tolerance,
                )
            )
        return mismatches
    if isinstance(recorded, list) and isinstance(replayed, list):
        mismatches = []
        if len(recorded) != len(replayed):
            mismatches.append(
                {
                    "step": step,
                    "field": f"{field}.length",
                    "recorded": len(recorded),
                    "replayed": len(replayed),
                    "abs_error": None,
                }
            )
        for index, recorded_item in enumerate(recorded[: len(replayed)]):
            mismatches.extend(
                _jsonish_mismatches(
                    step=step,
                    field=f"{field}.{index}",
                    recorded=recorded_item,
                    replayed=replayed[index],
                    tolerance=tolerance,
                )
            )
        return mismatches
    if recorded != replayed:
        return [
            {
                "step": step,
                "field": field,
                "recorded": recorded,
                "replayed": replayed,
                "abs_error": None,
            }
        ]
    return []


def verify_records(
    records: list[dict[str, Any]],
    *,
    tolerance: float = 1.0e-5,
    world_interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> VerificationResult:
    """Replay trajectory actions and compare deterministic observations/rewards."""

    validate_records(records)
    for record in records:
        semantics = {
            "reaction_thermal_control_id": REACTION_THERMAL_CONTROL_ID,
            "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
        }
        for field, current in semantics.items():
            if record.get(field) == current:
                continue
            return VerificationResult(
                verified=False,
                checked_steps=0,
                max_abs_error=0.0,
                mismatches=[
                    {
                        "step": record["step"],
                        "field": field,
                        "recorded": record.get(field),
                        "replayed": current,
                        "abs_error": None,
                        "reason": (
                            "This trajectory requires its original frozen runtime and uv.lock. "
                            "Current main does not replay historical physics; see "
                            "configs/current.json publication.frozen_release for paper evidence."
                        ),
                    }
                ],
            )
    first = records[0]
    composition_request = first.get("composition_request")
    benchmark_task_id = first.get("benchmark_task_id")
    env_kwargs: dict[str, Any]
    if isinstance(composition_request, dict):
        env_kwargs = {
            "composition": dict(composition_request),
            "seed": int(first["seed"]),
        }
    elif benchmark_task_id:
        env_kwargs = {
            "task_id": str(benchmark_task_id),
            "seed": int(first["seed"]),
        }
        if first.get("contract_profile") == "extended-research":
            env_kwargs["budget_override"] = int(first["budget"])
            env_kwargs["episode_mode_override"] = str(first["episode_mode"])
        optional_string_kwargs = {
            "full_process_contract_id": "full_process_contract_id",
            "electrochemical_workflow_mode": "electrochemical_workflow_mode",
            "electrochemical_material_family_id": ("electrochemical_material_family_id"),
            "crystallization_material_family_id": ("crystallization_material_family_id"),
            "scoring_contract_id": "scoring_contract_id",
        }
        for record_key, env_key in optional_string_kwargs.items():
            value = first.get(record_key)
            if isinstance(value, str) and value:
                env_kwargs[env_key] = value
        # New trajectories carry the evaluator-only normalized configuration
        # separately so a misindexed arm can be reconstructed without exposing
        # the transposition to the agent-facing material dossier.  Fall back
        # to the legacy mode-only field for older trajectories.
        material_information = first.get("material_information_config")
        if not isinstance(material_information, dict):
            legacy_material_information = first.get("material_information")
            material_information = (
                {"mode": legacy_material_information.get("mode")}
                if isinstance(legacy_material_information, dict)
                and isinstance(legacy_material_information.get("mode"), str)
                else None
            )
        if isinstance(material_information, dict):
            env_kwargs["material_information"] = dict(material_information)
    else:
        env_kwargs = {
            "world_split": first["world_split"],
            "budget": int(first.get("budget", len(records))),
            "objective": _objective_from_record(first),
            "seed": int(first["seed"]),
        }
    # Observation streams can be overridden for composed worlds as well as
    # benchmark tasks. Reconstruct the recorded stream independently of seed.
    for key in ("observation_noise_mode", "observation_noise_namespace"):
        value = first.get(key)
        if isinstance(value, str) and value:
            env_kwargs[key] = value
    observation_seed = first.get("observation_seed")
    if isinstance(observation_seed, int) and not isinstance(observation_seed, bool):
        env_kwargs["observation_seed_override"] = observation_seed
    campaign_resource_card = first.get("campaign_resource_card")
    if isinstance(first.get("research_brief"), dict):
        env_kwargs["research_brief"] = dict(first["research_brief"])
    if isinstance(campaign_resource_card, dict):
        env_kwargs["campaign_resource_card"] = dict(campaign_resource_card)
    if world_interventions:
        env_kwargs["world_interventions"] = list(world_interventions)
    env = gym.make(
        first["env_id"],
        **env_kwargs,
    )
    _, reset_info = env.reset(seed=int(first["seed"]))
    replay_env = cast(ChemWorldEnv, env.unwrapped)
    replay_provenance = replay_env.evaluator_provenance()
    recorded_mechanism_hash = first.get("mechanism_hash")
    replay_mechanism_hash = replay_provenance.get("mechanism_hash")
    early_mismatches: list[dict[str, Any]] = []
    recorded_intervention_hashes = {
        field: first.get(field)
        for field in (
            "world_family_intervention_hash",
            "mechanism_family_intervention_hash",
        )
        if first.get(field)
    }
    if recorded_intervention_hashes and not world_interventions:
        early_mismatches.append(
            {
                "step": 0,
                "field": "world_interventions",
                "recorded": recorded_intervention_hashes,
                "replayed": None,
                "abs_error": None,
                "reason": "exact intervention payload is required for replay",
            }
        )
    for field, recorded, replayed in (
        (
            "task_contract_hash",
            first.get("task_contract_hash"),
            reset_info.get("task_contract_hash"),
        ),
        (
            "runtime_profile_hash",
            first.get("runtime_profile_hash"),
            reset_info.get("runtime_profile_hash"),
        ),
        ("mechanism_hash", recorded_mechanism_hash, replay_mechanism_hash),
        (
            "scoring_contract_hash",
            first.get("scoring_contract_hash"),
            reset_info.get("scoring_contract_hash"),
        ),
        (
            "observation_contract_hash",
            first.get("observation_contract_hash"),
            reset_info.get("observation_contract_hash"),
        ),
        (
            "world_family_intervention_hash",
            first.get("world_family_intervention_hash"),
            replay_provenance.get("world_family_intervention_hash"),
        ),
        (
            "mechanism_family_intervention_hash",
            first.get("mechanism_family_intervention_hash"),
            replay_provenance.get("mechanism_family_intervention_hash"),
        ),
    ):
        if recorded and replayed != recorded:
            early_mismatches.append(
                {
                    "step": 0,
                    "field": field,
                    "recorded": recorded,
                    "replayed": replayed,
                    "abs_error": None,
                }
            )

    mismatches: list[dict[str, Any]] = list(early_mismatches)
    max_abs_error = 0.0
    checked_steps = 0
    try:
        for record in records:
            commands = record.get("initialization_replay")
            if commands is not None:
                from chemworld.runtime.lifecycle import replay_initializations

                try:
                    replay_initializations(replay_env, commands)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    mismatches.append({"step": record["step"],
                                       "field": "initialization_replay", "reason": str(exc)})
                    break
            observation, reward, terminated, truncated, info = env.step(record["action"])
            checked_steps += 1
            if commands is not None:
                try:
                    replay_initializations(replay_env, commands)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    mismatches.append({"step": record["step"],
                                       "field": "initialization_replay", "reason": str(exc)})
                    break
            replay_audit_info = {**info, **replay_provenance}
            replay_observation = _scalar_observation(observation)
            recorded_observation = record["observation"]
            if (
                campaign_resource_card
                and campaign_resource_card.get("hard_limits", {}).get("per_batch")
            ):
                for field in (
                    "batch_resource_preflight", "batch_resource_state",
                    "batch_budget_exhausted", "last_terminal_summary",
                ):
                    recorded_local = record.get("environment_outcome", {}).get(field)
                    replayed_local = info.get(field)
                    if field == "batch_resource_preflight":
                        # Receipt IDs include the new replay session identity.
                        # Compare all resource decisions and amounts exactly.
                        recorded_local = {k: v for k, v in (recorded_local or {}).items()
                                          if k != "event_id"}
                        replayed_local = {k: v for k, v in (replayed_local or {}).items()
                                          if k != "event_id"}
                    mismatches.extend(_jsonish_mismatches(
                        step=int(record["step"]), field=field,
                        recorded=recorded_local, replayed=replayed_local, tolerance=tolerance,
                    ))
            for inventory_field, default_inventory in (
                ("sample_inventory", []), ("material_routing", {})
            ):
                recorded_inventory = record.get("environment_outcome", {}).get(inventory_field)
                if recorded_inventory is None:
                    continue
                inventory_mismatches = _jsonish_mismatches(
                    step=int(record["step"]),
                    field=inventory_field,
                    recorded=recorded_inventory,
                    replayed=info.get(inventory_field, default_inventory),
                    tolerance=tolerance,
                )
                mismatches.extend(inventory_mismatches)
                max_abs_error = max(
                    max_abs_error,
                    max(
                        (
                            float(item["abs_error"])
                            for item in inventory_mismatches
                            if item.get("abs_error") is not None
                        ),
                        default=0.0,
                    ),
                )

            reward_error = abs(float(record["reward"]) - float(reward))
            max_abs_error = max(max_abs_error, reward_error)
            if reward_error > tolerance:
                mismatches.append(
                    {
                        "step": record["step"],
                        "field": "reward",
                        "recorded": float(record["reward"]),
                        "replayed": float(reward),
                        "abs_error": reward_error,
                    }
                )

            for key, replayed_value in replay_observation.items():
                recorded_raw = recorded_observation[key]
                if replayed_value is None or recorded_raw is None:
                    if replayed_value is not None or recorded_raw is not None:
                        mismatches.append(
                            {
                                "step": record["step"],
                                "field": f"observation.{key}",
                                "recorded": recorded_raw,
                                "replayed": replayed_value,
                                "abs_error": None,
                            }
                        )
                    continue
                recorded_value = float(recorded_raw)
                error = abs(recorded_value - replayed_value)
                max_abs_error = max(max_abs_error, error)
                if error > tolerance:
                    mismatches.append(
                        {
                            "step": record["step"],
                            "field": f"observation.{key}",
                            "recorded": recorded_value,
                            "replayed": replayed_value,
                            "abs_error": error,
                        }
                    )

            if bool(record["terminated"]) != terminated:
                mismatches.append(
                    {
                        "step": record["step"],
                        "field": "terminated",
                        "recorded": bool(record["terminated"]),
                        "replayed": terminated,
                        "abs_error": None,
                    }
                )
            if bool(record["truncated"]) != truncated:
                mismatches.append(
                    {
                        "step": record["step"],
                        "field": "truncated",
                        "recorded": bool(record["truncated"]),
                        "replayed": truncated,
                        "abs_error": None,
                    }
                )
            if record.get("operation_type") is not None:
                replay_operation = info.get("operation_type")
                if record["operation_type"] != replay_operation:
                    mismatches.append(
                        {
                            "step": record["step"],
                            "field": "operation_type",
                            "recorded": record["operation_type"],
                            "replayed": replay_operation,
                            "abs_error": None,
                        }
                    )
            replay_metadata_fields = (
                "task_contract_hash",
                "runtime_profile_hash",
                "mechanism_id",
                "mechanism_hash",
                "scoring_contract_hash",
                "observation_contract_hash",
                "world_family_intervention_hash",
                "mechanism_family_intervention_hash",
                "kernel_id",
                "kernel_version",
                "affected_ledgers",
                "world_events",
                "state_patches_summary",
                "transaction_status",
                "rollback_reason",
                "operation_outcome",
                "actual_duration_s",
                "outcome_diagnostics",
                "measurement_status",
                "terminal_assay",
                "state_delta_summary",
                "observed_mask",
                "raw_signal",
                "native_observation",
                "processed_estimate",
                "uncertainty",
                "measurement_cost",
                "sample_consumed",
            )
            if record.get("material_balance"):
                recorded_balance = record["material_balance"]
                replayed_balance = replay_env._last_operation_record.material_balance.to_dict(
                    include_hidden="exchanges" in recorded_balance
                )
                mismatches.extend(_jsonish_mismatches(
                    step=int(record["step"]), field="material_balance",
                    recorded=recorded_balance, replayed=replayed_balance, tolerance=tolerance,
                ))
            for field in replay_metadata_fields:
                if field not in record:
                    continue
                replayed_metadata = replay_audit_info.get(field)
                field_mismatches = _jsonish_mismatches(
                    step=int(record["step"]),
                    field=field,
                    recorded=record[field],
                    replayed=replayed_metadata,
                    tolerance=tolerance,
                )
                mismatches.extend(field_mismatches)
                for mismatch in field_mismatches:
                    abs_error = mismatch.get("abs_error")
                    if abs_error is not None:
                        max_abs_error = max(max_abs_error, float(abs_error))
            if record.get("constitution_checks"):
                # JSON maps can reorder compartments without changing physics.
                # Compare the full multiset, retaining duplicates and failures.
                def check_order(check):
                    return str(check.get("name")), str(check.get("passed"))

                recorded_checks = sorted(record["constitution_checks"], key=check_order)
                replay_checks = sorted(info.get("constitution_checks", []), key=check_order)
                if len(recorded_checks) != len(replay_checks):
                    mismatches.append(
                        {
                            "step": record["step"],
                            "field": "constitution_checks.length",
                            "recorded": len(recorded_checks),
                            "replayed": len(replay_checks),
                            "abs_error": None,
                        }
                    )
                for index, recorded_check in enumerate(recorded_checks[: len(replay_checks)]):
                    replay_check = replay_checks[index]
                    recorded_pair = (recorded_check.get("name"), recorded_check.get("passed"))
                    replay_pair = (replay_check.get("name"), replay_check.get("passed"))
                    if recorded_pair != replay_pair:
                        mismatches.append(
                            {
                                "step": record["step"],
                                "field": f"constitution_checks.{index}",
                                "recorded": recorded_pair,
                                "replayed": replay_pair,
                                "abs_error": None,
                            }
                        )
    finally:
        env.close()

    return VerificationResult(
        verified=not mismatches,
        checked_steps=checked_steps,
        max_abs_error=max_abs_error,
        mismatches=mismatches,
    )
