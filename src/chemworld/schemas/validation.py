"""JSON-friendly schema contracts without a runtime jsonschema dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from chemworld.physchem.reaction_network_specs import SUPPORTED_RATE_LAW_EQUATION_IDS
from chemworld.world.actions import ELECTROLYTE_PROFILES
from chemworld.world.operations import EXTENDED_INSTRUMENTS as INSTRUMENTS
from chemworld.world.operations import OPERATION_TYPES

PHASES = ("reactor_liquid", "aqueous", "organic")
MECHANISM_SCHEMA_VERSION = "chemworld_mechanism_v1"
TRAJECTORY_REQUIRED_KEYS = {
    "schema_version",
    "env_version",
    "world_family_version",
    "task_id",
    "env_id",
    "world_split",
    "benchmark_task_id",
    "objective",
    "budget",
    "episode_mode",
    "safety_limit",
    "task_contract_hash",
    "runtime_profile_hash",
    "mechanism_id",
    "mechanism_hash",
    "scoring_contract_hash",
    "observation_contract_hash",
    "kernel_id",
    "kernel_version",
    "affected_ledgers",
    "world_events",
    "state_patches_summary",
    "transaction_status",
    "rollback_reason",
    "world_id",
    "seed",
    "step",
    "run_id",
    "campaign_id",
    "experiment_index",
    "operation_id",
    "scenario_id",
    "initial_state_id",
    "kernel_maturity",
    "action",
    "environment_outcome",
    "agent_visible_observation",
    "evaluation_outcome",
    "observation",
    "reward",
    "terminated",
    "truncated",
    "constraint_flags",
    "constitution_checks",
    "agent_metadata",
    "instrument",
    "instrument_source",
    "measurement_cost",
    "observed_keys",
    "observed_mask",
    "raw_signal",
    "processed_estimate",
    "uncertainty",
    "observed_reward",
    "operation_type",
    "preconditions",
    "physics_maturity",
    "leaderboard_score",
    "proxy_allowed",
    "reward_source",
    "sample_consumed",
    "state_delta_summary",
    "timestamp",
    "explanation",
}

ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld event action",
    "type": "object",
    "required": ["operation"],
    "properties": {
        "operation": {
            "oneOf": [
                {"type": "string", "enum": list(OPERATION_TYPES)},
                {"type": "integer", "minimum": 0, "maximum": len(OPERATION_TYPES) - 1},
            ]
        },
        "payload": {"type": "object"},
        "instrument": {
            "oneOf": [
                {"type": "string", "enum": list(INSTRUMENTS)},
                {"type": "integer", "minimum": 0, "maximum": len(INSTRUMENTS) - 1},
            ]
        },
        "phase": {
            "oneOf": [
                {"type": "string", "enum": list(PHASES)},
                {"type": "integer", "minimum": 0, "maximum": len(PHASES) - 1},
            ]
        },
        "target_phase": {
            "oneOf": [
                {"type": "string", "enum": list(PHASES)},
                {"type": "integer", "minimum": 0, "maximum": len(PHASES) - 1},
            ]
        },
    },
    "additionalProperties": True,
}

OBSERVATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld observation",
    "type": "object",
    "additionalProperties": {"type": ["number", "null"]},
}

RECIPE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld recipe",
    "type": "object",
    "required": ["steps"],
    "properties": {
        "steps": {"type": "array", "items": ACTION_SCHEMA, "minItems": 1},
        "metadata": {"type": "object"},
    },
    "additionalProperties": True,
}

TRAJECTORY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld trajectory record",
    "type": "object",
    "required": sorted(TRAJECTORY_REQUIRED_KEYS),
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "chemworld-trajectory-0.2",
        },
        "run_id": {"type": "string", "minLength": 1},
        "environment_outcome": {"type": "object"},
        "agent_visible_observation": {"type": "object"},
        "evaluation_outcome": {"type": "object"},
    },
}

MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld submission manifest",
    "type": "object",
    "required": ["schema_version", "agent_name", "agent_family", "task_id", "seeds"],
}

TASK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld task spec",
    "type": "object",
    "required": [
        "task_contract_version",
        "task_id",
        "env_id",
        "world_law_id",
        "scenario_id",
        "initial_state_id",
        "world_split",
        "objective",
        "budget",
        "seeds",
        "threshold",
        "episode_mode",
        "allowed_operations",
        "allowed_instruments",
        "observation_policy",
        "termination_policy",
        "success_metrics",
        "safety_limit",
        "difficulty",
        "description",
        "tags",
        "kernel_maturity",
        "physics_maturity",
        "proxy_allowed",
        "contract_hash",
    ],
    "properties": {
        "task_contract_version": {"type": "string"},
        "task_id": {"type": "string", "minLength": 1},
        "env_id": {"type": "string", "const": "ChemWorld"},
        "world_law_id": {"type": "string", "minLength": 1},
        "scenario_id": {"type": "string", "minLength": 1},
        "initial_state_id": {"type": "string", "minLength": 1},
        "world_split": {
            "type": "string",
            "enum": ["public-dev", "public-test", "private-eval"],
        },
        "objective": {"type": "string", "minLength": 1},
        "budget": {"type": "integer", "minimum": 1},
        "seeds": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 0},
        },
        "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "episode_mode": {
            "type": "string",
            "enum": ["single_experiment", "campaign"],
        },
        "allowed_operations": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(OPERATION_TYPES)},
        },
        "allowed_instruments": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(INSTRUMENTS)},
        },
        "observation_policy": {"type": "string", "minLength": 1},
        "termination_policy": {"type": "string", "minLength": 1},
        "success_metrics": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "safety_limit": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "difficulty": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "tags": {"type": "array", "items": {"type": "string"}},
        "kernel_maturity": {
            "type": "object",
            "required": ["modules", "lowest_level", "proxy_allowed", "notes"],
        },
        "physics_maturity": {
            "type": "string",
            "enum": [
                "proxy",
                "lite",
                "reference_validated",
                "professional_candidate",
                "professional",
            ],
        },
        "proxy_allowed": {"type": "boolean"},
        "contract_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "additionalProperties": False,
}

SCENARIO_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld scenario spec",
    "type": "object",
    "required": ["scenario_id", "world_law_id", "family", "split"],
}

MECHANISM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChemWorld mechanism",
    "type": "object",
    "required": ["schema_version", "network_id", "species", "reactions"],
    "properties": {
        "schema_version": {"type": "string", "const": MECHANISM_SCHEMA_VERSION},
        "network_id": {"type": "string"},
        "metadata": {"type": "object"},
        "species": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["species_id", "formula"],
                "properties": {
                    "species_id": {"type": "string"},
                    "formula": {"type": "string"},
                    "phase": {"type": "string"},
                    "charge": {"type": "integer"},
                    "catalyst": {"type": "boolean"},
                    "observable_aliases": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
        },
        "reactions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["reaction_id", "rate_law"],
                "properties": {
                    "reaction_id": {"type": "string"},
                    "equation": {"type": "string"},
                    "stoichiometry": {"type": "object"},
                    "reversible": {"type": "boolean"},
                    "delta_h_J_per_mol": {"type": "number"},
                    "equilibrium_model_id": {"type": "string"},
                    "rate_law": {
                        "type": "object",
                        "required": ["rate_law_id", "equation_id"],
                        "properties": {
                            "rate_law_id": {"type": "string"},
                            "equation_id": {
                                "type": "string",
                                "enum": list(SUPPORTED_RATE_LAW_EQUATION_IDS),
                            },
                            "parameters": {"type": "object"},
                        },
                    },
                    "metadata": {"type": "object"},
                },
            },
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_action_schema(
    action: object,
    *,
    operation_types: tuple[str, ...] = OPERATION_TYPES,
) -> SchemaValidationResult:
    errors: list[str] = []
    if not isinstance(action, dict):
        return SchemaValidationResult(False, ("action must be an object",))
    action = cast(dict[str, Any], action)
    if "operation" not in action:
        errors.append("missing required field: operation")
    operation = action.get("operation")
    if isinstance(operation, str) and operation not in operation_types:
        errors.append(f"unknown operation: {operation}")
    if isinstance(operation, int) and not 0 <= operation < len(operation_types):
        errors.append(f"operation index outside valid range: {operation}")
    if not isinstance(operation, str | int):
        errors.append("operation must be a string name or integer index")
    reason = action.get("reason")
    if reason is not None and (
        not isinstance(reason, str) or not reason.strip() or len(reason) > 256
    ):
        errors.append("reason must be a non-empty string of at most 256 characters")
    if "payload" in action and not isinstance(action["payload"], dict):
        errors.append("payload must be an object when provided")
    instrument = action.get("instrument")
    if instrument is not None and not isinstance(instrument, str | int):
        errors.append("instrument must be a string name or integer index")
    if isinstance(instrument, str) and instrument not in INSTRUMENTS:
        errors.append(f"unknown instrument: {instrument}")
    if isinstance(instrument, int) and not 0 <= instrument < len(INSTRUMENTS):
        errors.append(f"instrument index outside valid range: {instrument}")
    electrolyte_profile = action.get("electrolyte_profile")
    if electrolyte_profile is not None:
        valid_profile = (
            isinstance(electrolyte_profile, str)
            and electrolyte_profile in ELECTROLYTE_PROFILES
        ) or (
            isinstance(electrolyte_profile, int)
            and not isinstance(electrolyte_profile, bool)
            and 0 <= electrolyte_profile < len(ELECTROLYTE_PROFILES)
        )
        if not valid_profile:
            errors.append(
                "electrolyte_profile must be a known profile id or integer index in [0, 3]"
            )
    for key in ("phase", "target_phase"):
        value = action.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value not in PHASES:
            errors.append(f"unknown {key}: {value}")
        elif isinstance(value, int) and not 0 <= value < len(PHASES):
            errors.append(f"{key} index outside valid range: {value}")
        elif not isinstance(value, str | int):
            errors.append(f"{key} must be a string name or integer index")
    for key in (
        "amount_mol",
        "volume_L",
        "catalyst_amount_mol",
        "target_temperature_K",
        "duration_s",
        "stirring_speed_rpm",
        "sample_volume_L",
        "wash_volume_L",
        "transfer_fraction",
        "seed_mass_g",
        "reflux_ratio",
        "flow_rate_mL_min",
        "residence_time_s",
        "potential_V",
        "current_mA",
    ):
        if key in action and not _is_number(action[key]):
            errors.append(f"{key} must be numeric")
    return SchemaValidationResult(not errors, tuple(errors))


def validate_recipe_schema(recipe: object) -> SchemaValidationResult:
    errors: list[str] = []
    if not isinstance(recipe, dict):
        return SchemaValidationResult(False, ("recipe must be an object",))
    recipe = cast(dict[str, Any], recipe)
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("recipe.steps must be a non-empty list")
        return SchemaValidationResult(False, tuple(errors))
    for index, step in enumerate(steps):
        result = validate_action_schema(step)
        errors.extend(f"steps[{index}]: {error}" for error in result.errors)
    return SchemaValidationResult(not errors, tuple(errors))


def validate_manifest_schema(manifest: dict[str, Any]) -> SchemaValidationResult:
    required = {"schema_version", "agent_name", "agent_family", "task_id", "seeds"}
    missing = sorted(required - manifest.keys())
    errors = [f"missing required field: {key}" for key in missing]
    if "seeds" in manifest and not isinstance(manifest["seeds"], list):
        errors.append("seeds must be a list")
    return SchemaValidationResult(not errors, tuple(errors))


def validate_task_schema(task: object) -> SchemaValidationResult:
    """Validate the executable fields required by a benchmark task contract."""

    if not isinstance(task, dict):
        return SchemaValidationResult(False, ("task must be an object",))
    payload = cast(dict[str, Any], task)
    errors = [
        f"missing required field: {key}"
        for key in TASK_SCHEMA["required"]
        if key not in payload
    ]
    budget = payload.get("budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        errors.append("budget must be a positive integer")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        errors.append("seeds must be a non-empty unique list of nonnegative integers")
    for key in ("threshold", "safety_limit"):
        value = payload.get(key)
        in_unit_interval = _is_number(value) and 0.0 <= float(
            cast(int | float, value)
        ) <= 1.0
        if not in_unit_interval:
            errors.append(f"{key} must be a number in [0, 1]")
    if payload.get("episode_mode") not in {"single_experiment", "campaign"}:
        errors.append("episode_mode must be single_experiment or campaign")
    operations = payload.get("allowed_operations")
    if (
        not isinstance(operations, list)
        or not operations
        or len(operations) != len(set(operations))
        or any(operation not in OPERATION_TYPES for operation in operations)
    ):
        errors.append("allowed_operations must be unique registered operations")
    instruments = payload.get("allowed_instruments")
    if (
        not isinstance(instruments, list)
        or not instruments
        or len(instruments) != len(set(instruments))
        or any(instrument not in INSTRUMENTS for instrument in instruments)
    ):
        errors.append("allowed_instruments must be unique registered instruments")
    success_metrics = payload.get("success_metrics")
    if (
        not isinstance(success_metrics, list)
        or not success_metrics
        or any(not isinstance(metric, str) or not metric for metric in success_metrics)
    ):
        errors.append("success_metrics must be a non-empty string list")
    contract_hash = payload.get("contract_hash")
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash)
    ):
        errors.append("contract_hash must be a lowercase SHA-256 digest")
    maturity = payload.get("kernel_maturity")
    if not isinstance(maturity, dict) or not maturity.get("modules"):
        errors.append("kernel_maturity must declare at least one module")
    elif bool(maturity.get("proxy_allowed")) != bool(payload.get("proxy_allowed")):
        errors.append("proxy_allowed must match kernel_maturity.proxy_allowed")
    return SchemaValidationResult(not errors, tuple(errors))


def validate_mechanism_schema(mechanism: object) -> SchemaValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(mechanism, dict):
        return SchemaValidationResult(False, ("mechanism must be an object",))
    payload = cast(dict[str, Any], mechanism)
    for key in MECHANISM_SCHEMA["required"]:
        if key not in payload:
            errors.append(f"missing required field: {key}")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version and schema_version != MECHANISM_SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version: {schema_version}; expected {MECHANISM_SCHEMA_VERSION}"
        )
    network_id = payload.get("network_id")
    if network_id is not None and not str(network_id):
        errors.append("network_id cannot be empty")

    species_payload = payload.get("species")
    species_ids: list[str] = []
    if not isinstance(species_payload, list) or not species_payload:
        errors.append("species must be a non-empty list")
    else:
        for index, species in enumerate(species_payload):
            if not isinstance(species, dict):
                errors.append(f"species[{index}] must be an object")
                continue
            species_id = species.get("species_id")
            formula = species.get("formula")
            if not isinstance(species_id, str) or not species_id:
                errors.append(f"species[{index}].species_id must be a non-empty string")
            else:
                species_ids.append(species_id)
            if not isinstance(formula, str) or not formula:
                errors.append(f"species[{index}].formula must be a non-empty string")
            aliases = species.get("observable_aliases", ())
            if aliases is not None and not isinstance(aliases, list):
                errors.append(f"species[{index}].observable_aliases must be a list")
        duplicate_species = _duplicates(species_ids)
        if duplicate_species:
            errors.append(f"duplicate species_id values: {duplicate_species}")

    reactions_payload = payload.get("reactions")
    reaction_ids: list[str] = []
    if not isinstance(reactions_payload, list) or not reactions_payload:
        errors.append("reactions must be a non-empty list")
    else:
        for index, reaction in enumerate(reactions_payload):
            if not isinstance(reaction, dict):
                errors.append(f"reactions[{index}] must be an object")
                continue
            reaction_id = reaction.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id:
                errors.append(f"reactions[{index}].reaction_id must be a non-empty string")
            else:
                reaction_ids.append(reaction_id)
            if "equation" not in reaction and "stoichiometry" not in reaction:
                errors.append(
                    f"reactions[{index}] must declare either equation or stoichiometry"
                )
            rate_law = reaction.get("rate_law")
            if not isinstance(rate_law, dict):
                errors.append(f"reactions[{index}].rate_law must be an object")
                continue
            equation_id = rate_law.get("equation_id")
            rate_law_id = rate_law.get("rate_law_id")
            if not isinstance(rate_law_id, str) or not rate_law_id:
                errors.append(
                    f"reactions[{index}].rate_law.rate_law_id must be a non-empty string"
                )
            if equation_id not in SUPPORTED_RATE_LAW_EQUATION_IDS:
                errors.append(
                    f"reactions[{index}].rate_law.equation_id is unsupported: {equation_id!r}"
                )
            if isinstance(equation_id, str) and equation_id.lower() in {"eval", "python"}:
                errors.append(
                    f"reactions[{index}].rate_law.equation_id cannot execute code"
                )
            parameters = rate_law.get("parameters", {})
            if parameters is not None and not isinstance(parameters, dict):
                errors.append(f"reactions[{index}].rate_law.parameters must be an object")
        duplicate_reactions = _duplicates(reaction_ids)
        if duplicate_reactions:
            errors.append(f"duplicate reaction_id values: {duplicate_reactions}")

    if payload.get("metadata") is None:
        warnings.append("metadata is absent; mechanism provenance will be sparse")
    return SchemaValidationResult(not errors, tuple(errors), tuple(warnings))


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


__all__ = [
    "ACTION_SCHEMA",
    "MANIFEST_SCHEMA",
    "MECHANISM_SCHEMA",
    "MECHANISM_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA",
    "RECIPE_SCHEMA",
    "SCENARIO_SCHEMA",
    "TASK_SCHEMA",
    "TRAJECTORY_REQUIRED_KEYS",
    "TRAJECTORY_SCHEMA",
    "SchemaValidationResult",
    "validate_action_schema",
    "validate_manifest_schema",
    "validate_mechanism_schema",
    "validate_recipe_schema",
    "validate_task_schema",
]
