"""Agent-facing public views for ChemWorld environments.

The functions in this module are intentionally thin adapters over the existing
task registry, operation validator, observation contracts, and campaign state.
The full-process sensor view exposes explicitly declared operational readouts,
never latent composition, process-quality summaries, or rate constants.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from typing import Any, cast

import numpy as np

from chemworld.data.logging import to_builtin
from chemworld.foundation import equipment_settings, selected_phase_id
from chemworld.foundation.samples import CONTAINER_IDS
from chemworld.materials import material_choice_labels
from chemworld.operation_validator import (
    ELECTROCHEMICAL_MIN_ADAPTED_CURRENT_DELTA_MA,
    ELECTROCHEMICAL_MIN_ADAPTED_POTENTIAL_DELTA_V,
)
from chemworld.physchem.crystallization_units import (
    DEFAULT_MAXIMUM_COOLING_RATE_K_S,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1,
)
from chemworld.runtime.component_network import public_network
from chemworld.runtime.continuous_streams import public_streams
from chemworld.runtime.control_program import public_control
from chemworld.runtime.scheduling import lab_instrument_reservation
from chemworld.runtime.spectral_settings import public_spectral_settings
from chemworld.world.control_contract import CONTROL_CHOICES
from chemworld.world.operations import (
    CAMPAIGN_CONTROL_OPERATIONS,
    OPERATION_TYPES,
    operation_contracts,
)
from chemworld.world.operations import PUBLIC_OBSERVATION_KEYS as OBSERVATION_KEYS

PUBLIC_ACTION_SCHEMA_VERSION = "chemworld-public-action-affordance-0.2"

# Synchronous public-view builders never execute actions. Reuse the bounded
# resource state and derived affordances only within that read, then discard
# them even on exceptions. Action execution never consumes this view context.
# ContextVar isolates concurrent callers; nested calls for another env/state
# receive their own scope. Nothing is cached by state identity across requests.
_RESOURCE_READ: ContextVar[tuple[Any, Any, Any, dict[str, Any]] | None] = ContextVar(
    "chemworld_public_resource_read", default=None
)


def _resource_read(function):
    @wraps(function)
    def read(env, *args, **kwargs):
        base = _base_env(env)
        ledger = getattr(base, "_campaign_resource_ledger", None)
        state = getattr(base, "_state", None)
        active = _RESOURCE_READ.get()
        if (
            active is not None and active[0] is base and active[1] is ledger and active[2] is state
        ):
            return function(env, *args, **kwargs)
        token = _RESOURCE_READ.set((base, ledger, state, {}))
        try:
            return function(env, *args, **kwargs)
        finally:
            _RESOURCE_READ.reset(token)

    return read


def _view_affordances(base):
    """One validator traversal for all views in the same synchronous request."""
    active = _RESOURCE_READ.get()
    if active is None or active[0] is not base or active[2] is not base._state:
        return base.operation_validator.operation_affordances(base._state)
    cache = active[3]
    if "affordances" not in cache:
        cache["affordances"] = base.operation_validator.operation_affordances(base._state)
    # Public view construction annotates preconditions with resource reasons.
    # Those annotations must not mutate the validator's physical result.
    return deepcopy(cache["affordances"])


def _campaign_snapshot(base: Any, ledger: Any) -> dict[str, Any]:
    """Return the bounded live state used by public affordance queries."""

    active = _RESOURCE_READ.get()
    if (
        active is None
        or active[0] is not base
        or active[1] is not ledger
        or active[2] is not getattr(base, "_state", None)
    ):
        return ledger.bounded_state()
    cache = active[3]
    if "snapshot" not in cache:
        cache["snapshot"] = ledger.bounded_state()
    return cache["snapshot"]


OPERATION_GROUPS: dict[str, tuple[str, ...]] = {
    "reaction_setup": ("add_solvent", "add_reagent", "add_component", "add_catalyst"),
    "reaction_control": ("heat", "wait", "quench", "terminate"),
    "sampling_and_measurement": ("sample", "measure"),
    "phase_setup": ("add_phase", "add_extractant"),
    "phase_partition": ("mix", "settle", "separate_phase"),
    "purification": ("transfer", "wash", "dry", "concentrate"),
    "equilibrium_characterization": ("measure", "terminate"),
}

TASK_PROMPT_PROFILES: dict[str, dict[str, Any]] = {
    "reaction-to-assay": {
        "task_goal": (
            "Run one complete reaction experiment from charging the reactor to a "
            "valid terminal final assay."
        ),
        "success_criteria": [
            "Reach a valid final_assay after a physically legal operation sequence.",
            "Improve final_assay_score while keeping the trajectory valid.",
            "Use intermediate instruments only when their information is worth the cost.",
        ],
        "constraints": [
            "Single-experiment task: final_assay ends the episode.",
            "Terminate or quench before requesting final_assay.",
            "Heating before adding solvent/reagent/catalyst is invalid.",
        ],
        "measurement_policy": (
            "HPLC, GC, and UV-vis are noisy intermediate tools; final_assay is the "
            "high-cost terminal measurement used for the task result."
        ),
        "recommended_strategy": [
            "Charge solvent and reagent first, then add catalyst if used.",
            "Choose a temperature/time pair, run heat or wait, then inspect with HPLC/UV-vis.",
            "Terminate and run final_assay only after the reaction has meaningful conversion.",
        ],
        "failure_modes": [
            "precondition failure",
            "budget exhaustion before final_assay",
            "process/procedure penalty at or above the task limit",
        ],
    },
    "reaction-to-purification": {
        "task_goal": (
            "Complete a reaction, extract product through phase separation, purify "
            "the material, and submit a final assay."
        ),
        "success_criteria": [
            "Maximize score while balancing purity, recovery, and cost.",
            "Keep process_mass_balance_error small through separation and purification.",
            "Use final_assay after downstream workup to make the result leaderboard eligible.",
        ],
        "constraints": [
            "Single-experiment task: final_assay ends the episode.",
            "Downstream purification should follow a terminated or quenched reaction.",
            "Phase operations require appropriate phase or extractant setup.",
        ],
        "measurement_policy": (
            "Intermediate HPLC/GC/UV-vis can guide reaction and workup choices. "
            "The final_assay should be run on the selected purified material."
        ),
        "recommended_strategy": [
            "First generate product under safe reaction conditions.",
            (
                "Add extractant or phase, mix, settle, and separate the phase "
                "expected to carry product."
            ),
            "Use wash/dry/concentrate to trade recovery against purity before final_assay.",
        ],
        "failure_modes": [
            "poor phase split",
            "low product recovery",
            "high impurity after purification",
            "mass-balance drift",
        ],
    },
    "electrochemical-conversion": {
        "task_goal": (
            "Use solvent-medium and electrolyte-profile comparisons plus pH/UV-vis "
            "diagnostics to adapt a second electrochemical regime."
        ),
        "success_criteria": [
            "Maximize selective_product_yield as the declared primary metric.",
            "Balance faradaic, transport, ohmic, and energy efficiency as secondary outcomes.",
            "Complete the probe, diagnostic, adapted-control, and final-assay lifecycle.",
        ],
        "constraints": [
            "Select one solvent and one electrolyte profile before the first electrolysis.",
            "After the probe, collect both ph_meter and uvvis diagnostics.",
            "The second setpoint must differ by at least 0.02 V or 1.0 mA.",
            "Only declared operation fields are accepted; cell physics is environment-owned.",
        ],
        "measurement_policy": (
            "ph_meter and uvvis expose noisy diagnostic consequences; final_assay is the "
            "leaderboard-grade outcome measurement."
        ),
        "recommended_strategy": [
            "Compare solvent and electrolyte choices relationally across experiments.",
            "Use the first regime as a probe, not as a duplicate production stage.",
            "Condition the second potential/current choice on the diagnostic response.",
        ],
        "failure_modes": [
            "repeating the probe setpoint",
            "terminating without the post-control UV-vis assay",
            "confounding solvent and electrolyte effects without matched controls",
        ],
    },
    "partition-discovery": {
        "task_goal": (
            "Learn the public behavior of an unknown product/solvent partition system "
            "through repeated finite-budget experiments."
        ),
        "success_criteria": [
            "Estimate phase_ratio from public observations.",
            "Identify when product is enriched in organic versus aqueous phase.",
            "Use measurements efficiently across campaign experiments.",
        ],
        "constraints": [
            (
                "Campaign task: one terminal assay or termination summarizes an "
                "experiment, not the whole campaign."
            ),
            "Partition coefficients and hidden phase amounts are not directly visible.",
            "Phase split actions require a valid phase setup, mixing, and settling sequence.",
        ],
        "measurement_policy": (
            "Use HPLC/GC/UV-vis/final_assay as public instruments to infer partition "
            "trends; do not assume hidden partition coefficients are known."
        ),
        "recommended_strategy": [
            "Vary solvent or extractant choices across experiments.",
            "Use mix and settle before separating phases.",
            (
                "Compare instrument summaries from organic and aqueous outcomes "
                "to build a local model."
            ),
        ],
        "failure_modes": [
            "insufficient phase setup",
            "separating before settling",
            "overusing costly measurements",
            "confusing noisy instrument estimates with hidden truth",
        ],
    },
    "equilibrium-characterization": {
        "task_goal": (
            "Characterize a bounded aqueous-equilibrium slice using pH-meter "
            "and final-assay observations."
        ),
        "success_criteria": [
            "Obtain informative pH-meter observations before final assay.",
            "Maximize equilibrium_confidence while keeping equilibrium_residual low.",
            "Use measurements efficiently under the finite campaign budget.",
        ],
        "constraints": [
            "pH is public only through ph_meter or final_assay observations.",
            "Hidden acidity constants and species amounts are never exposed.",
            "Final assay still requires terminate.",
        ],
        "measurement_policy": (
            "ph_meter is the low-cost primary characterization instrument. "
            "final_assay provides leaderboard-grade equilibrium metrics."
        ),
        "recommended_strategy": [
            "Charge solvent and a moderate reagent amount.",
            "Measure pH, perturb concentration, then measure pH again.",
            "Terminate and request final_assay after enough equilibrium evidence.",
        ],
        "failure_modes": [
            "measuring before material exists",
            "overusing final assay without intermediate evidence",
            "treating pH-normalized Gym scalar as hidden pKa",
        ],
    },
}

ELECTROCHEMICAL_AUTONOMOUS_OPEN_PROMPT_PROFILE: dict[str, Any] = {
    "task_goal": (
        "Autonomously design and execute electrochemical experiments by choosing "
        "materials, setpoints, electrolysis stages, and optional diagnostics to "
        "maximize the declared balanced leaderboard score."
    ),
    "success_criteria": [
        (
            "Optimize the full declared leaderboard contract; selective_product_yield "
            "has a gate and 0.30 weight but is not the sole objective."
        ),
        (
            "Balance selectivity, conversion, faradaic, transport, ohmic, and energy "
            "efficiency according to the published component weights."
        ),
        "Use repeated controls or measurements only when they advance the experiment.",
        "Complete each chosen experiment through terminate and final_assay.",
    ],
    "constraints": [
        "Catalyst and electrolyte identities are fixed; solvent composition changes with dosing.",
        "Terminate is legal only after at least one committed electrolysis.",
        "No diagnostic instrument, second stage, or setpoint change is mandatory.",
    ],
    "measurement_policy": (
        "ph_meter and uvvis are optional noisy diagnostics. final_assay is available "
        "only after terminate and completes the current experiment."
    ),
    "recommended_strategy": [
        "Use the current public affordances to identify physically legal operations.",
        "A valid final_assay after terminate completes the current experiment.",
        "All operations, including optional diagnostics, consume the finite campaign budget.",
    ],
    "failure_modes": [
        "terminating before electrolysis",
        "budget exhaustion before final_assay",
        "invalid catalyst or electrolyte changes after the corresponding fixture is charged",
    ],
}

HIDDEN_INFORMATION_POLICY = (
    "No hidden species amounts, rate constants, mechanism parameters, partition "
    "coefficients, hidden phase amounts, or private scenario parameters are exposed "
    "through agent-facing views."
)

RL_MISSING_VALUE = -1.0
RL_OBSERVATION_VALUE_BOUNDS = (RL_MISSING_VALUE, 1.0)
RL_MASK_BOUNDS = (0.0, 1.0)
RL_COST_BOUNDS = (0.0, 1.0)

SUBMISSION_REQUIREMENTS = [
    "trajectory JSONL",
    "agent manifest",
    "reproducible command or replay trace",
]


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _latest_info(env: Any) -> dict[str, Any]:
    base = _base_env(env)
    info = getattr(base, "_last_info", {})
    return deepcopy(info) if isinstance(info, dict) else {}


def _latest_observation(env: Any) -> dict[str, Any]:
    base = _base_env(env)
    observation = getattr(base, "_last_observation", {})
    return deepcopy(observation) if isinstance(observation, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value.reshape(-1)[0]) if isinstance(value, np.ndarray) else float(value)
    except (TypeError, ValueError, IndexError):
        return default
    return scalar if math.isfinite(scalar) else default


def _observed_float(value: Any) -> tuple[float, bool]:
    try:
        scalar = float(value.reshape(-1)[0]) if isinstance(value, np.ndarray) else float(value)
    except (TypeError, ValueError, IndexError):
        return -1.0, False
    return (scalar, True) if math.isfinite(scalar) else (-1.0, False)


def _field_schema(
    field: str,
    *,
    operation: str | None = None,
    task_id: str | None = None,
    modules=None,
) -> dict[str, Any]:
    from chemworld.world.domain_modules import DomainModuleRegistry

    registry = modules or DomainModuleRegistry.default()
    spec = registry.operation_specs().get(operation)
    declared = None if spec is None else next((f for f in spec.fields if f.name == field), None)
    payload = {
        "field": field,
        "unit": "text" if field == "reason" else "unitless",
        "required": True,
    }
    if declared is not None:
        payload = declared.public_schema()
    bounds = None if declared is None else (declared.runtime_bounds or declared.bounds)
    if bounds is not None:
        low, high = bounds
        payload["bounds"] = {"low": low, "high": high}
        payload["recommended_range"] = {"low": low, "high": high}
        payload["lower_bound_inclusive"] = not (
            field in {"amount_mol", "catalyst_amount_mol", "sample_volume_L"}
            or (
                operation in {"transfer_material", "route_material"}
                and field == "transfer_fraction"
            )
            or (field == "volume_L" and operation in {"add_solvent", "add_phase", "add_extractant"})
        )
        payload["upper_bound_inclusive"] = True
    choices = None if declared is None else (declared.runtime_choices or declared.choices or None)
    if choices is not None:
        payload["choices"] = list(choices)
        if field in {"container", "source_container", "destination_container"}:
            payload["choice_labels"] = {str(i): CONTAINER_IDS[i] for i in choices}
        elif field == "mixing":
            payload["choice_labels"] = {"0": "empty destination only", "1": "allow mixing"}
        elif field in CONTROL_CHOICES:
            payload["choice_labels"] = {
                str(i): name for i, name in enumerate(CONTROL_CHOICES[field])
            }
        labels = material_choice_labels(field, task_id=task_id)
        if labels:
            payload["choice_labels"] = labels
    return payload


def _locked_recipe_choice(base: Any, operation: str, field: str) -> Any | None:
    lock_contract = {
        "add_catalyst": ("catalyst", "batch_reactor", "catalyst_amount_mol"),
    }.get(operation)
    if lock_contract is None:
        return None
    category_field, equipment_id, charged_key = lock_contract
    if field != category_field:
        return None
    state = getattr(base, "_state", None)
    if state is None:
        return None
    settings = equipment_settings(state.equipment, equipment_id, fields=(charged_key, field))
    if float(settings.get(charged_key, 0.0)) <= 0.0:
        return None
    return settings.get(field)


@_resource_read
def action_schema(env: Any, operation: str) -> dict[str, Any]:
    """Return the public JSON-friendly schema for one operation."""

    base = _base_env(env)
    modules = getattr(base, "domain_modules", None)
    contracts = operation_contracts(
        include_campaign_controls=operation in CAMPAIGN_CONTROL_OPERATIONS,
        modules=modules,
    )
    if operation not in contracts:
        return {
            "schema_version": PUBLIC_ACTION_SCHEMA_VERSION,
            "operation": operation,
            "valid_operation_type": False,
            "installed": False,
            "task_allowed": False,
            "error": f"Unknown operation {operation!r}",
        }
    contract = contracts[operation]
    state = getattr(base, "_state", None)
    task_id = getattr(base, "task_id", None)
    fields = [
        _field_schema(field, operation=operation, task_id=task_id, modules=modules)
        for field in contract.required_fields
    ]
    if modules is not None and operation in modules.operation_specs():
        fields = [
            _field_schema(f.name, operation=operation, task_id=task_id, modules=modules)
            for f in modules.operation_specs()[operation].fields
        ]
    from chemworld.runtime.full_process_contract import population_active

    if operation == "distill" and state is not None and population_active(state):
        fields.append(
            {
                "field": "cut_fraction",
                "required": False,
                "type": "number",
                "bounds": {"low": 0.005, "high": 0.9},
                "description": "Requested fraction of total feed moles (including carrier) "
                "distilled; independent of "
                "duration. Actual cut is limited by duty. Omission uses legacy duration rule.",
            }
        )
    if operation == "measure":
        instrument_status = base.operation_validator.measurement_affordances(state)
        for row in instrument_status:
            reasons = _campaign_resource_rejection_reasons_for_action(
                base, {"operation": "measure", "instrument": row["instrument"]}
            )
            if reasons:
                row["valid"] = False
                row["preconditions"]["campaign_resources_available"] = False
                row["invalid_reasons"] = list(dict.fromkeys([
                    *row["invalid_reasons"], "campaign_resources_available",
                    *[f"campaign_resource:{reason}" for reason in reasons],
                ]))
        fields = [
            {
                **_field_schema(
                    "instrument",
                    operation=operation,
                    task_id=task_id,
                ),
                "choices": [row["instrument"] for row in instrument_status if row["valid"]],
                "installed_choices": [
                    row["instrument"] for row in instrument_status if row["installed"]
                ],
                "task_allowed_choices": [
                    row["instrument"] for row in instrument_status if row["task_allowed"]
                ],
                "choice_status": instrument_status,
                "state_dependent_choices": True,
                "resource_limited": any(
                    not row["preconditions"].get("campaign_resources_available", True)
                    for row in instrument_status if row["installed"] and row["task_allowed"]
                ),
            }
        ]
    electrical_fields = None
    if operation == 'set_sde_parameters' and state is not None:
        from chemworld.physchem.stochastic_sde import sde_model

        composition = getattr(getattr(base, 'compiled_composition', None), 'spec', None)
        physics = None if composition is None else composition.physics
        spec = None if physics is None else physics.models_of('stochastic_sde').get(state.vessel_id)
        if spec is not None:
            model = sde_model(physics.network, spec, state.process.stochastic_continuation)
            for field in fields:
                if field['field'] == 'values':
                    field.update(properties={name: {
                        'type': 'number', 'minimum': metadata['bounds'][0],
                        'maximum': metadata['bounds'][1], 'unit': metadata['unit'],
                        'dimension': metadata['dimension'],
                        'current_value': cast(Mapping[str, float], model.control_values)[name],
                    } for name, metadata in model.control_metadata.items()},
                    additionalProperties=False, partial_updates=True)
    if operation in {"configure_electrical_control", "queue_electrical_stage"}:
        from chemworld.runtime.control_program import electrical_device

        try:
            settings = electrical_device(state).settings["electrical_control"]
        except ValueError:
            settings = {}
        electrical_fields = [
            name for name, key in (("current_A", "limits_A"), ("potential_V", "limits_V"))
            if key in settings
        ]
        fields = [
            field for field in fields
            if field["field"] not in {"current_A", "potential_V"}
            or field["field"] in electrical_fields
        ]
        for field in fields:
            if field["field"] in electrical_fields:
                field["conditionally_required"] = "electrical_control_quantity"
    for field in fields:
        field_name = str(field["field"])
        bounds = field.get("bounds")
        if isinstance(bounds, dict) and state is not None:
            low, high = base.operation_validator.public_field_bounds(
                operation,
                field_name,
                state,
                low=float(bounds["low"]),
                high=float(bounds["high"]),
            )
            if low != float(bounds["low"]) or high != float(bounds["high"]):
                field["bounds"] = {"low": low, "high": high}
                field["recommended_range"] = {"low": low, "high": high}
                field["state_dependent_bounds"] = True
        choices = field.get("choices")
        if isinstance(choices, list) and state is not None and operation != "measure":
            public_choices = list(
                base.operation_validator.public_field_choices(
                    operation,
                    field_name,
                    state,
                    choices=tuple(choices),
                )
            )
            if public_choices != choices:
                field["choices"] = public_choices
                field["state_dependent_choices"] = True
            if field_name in {"vessel", "connection"}:
                network = public_network(state)
                entries = (
                    []
                    if network is None
                    else network["vessels" if field_name == "vessel" else "connections"]
                )
                field["choice_labels"] = {str(item["index"]): item["id"] for item in entries}
        locked = _locked_recipe_choice(base, operation, field_name)
        if locked is not None:
            field["choices"] = [locked]
            field["locked_for_current_experiment"] = True
    _apply_campaign_resource_schema(base, operation, fields)
    constraints: list[dict[str, Any]] = []
    if electrical_fields is not None:
        constraints.append({
            "id": "electrical_control_quantity", "kind": "exactly_one_field",
            "fields": electrical_fields,
            "description": "Select exactly one quantity supported by the connected source.",
        })
    if operation in {"select_vessel", "route_material"} and state is not None:
        constraints.append(
            {"id": "component_network", "kind": "material_ports", "network": public_network(state)}
        )
    if operation == "transfer_material" and state is not None:
        from chemworld.runtime.material_routing import material_routes

        constraints.append(
            {"id": "material_routing", "kind": "material_routes", "routes": material_routes(state)}
        )
    if (
        operation == "cool_crystallize"
        and state is not None
        and base.operation_validator.native_crystal_model(state) is not None
    ):
        constraints.append(
            {
                "id": "finite_crystal_thermal_control",
                "kind": "physical_control",
                "description": "Temperature is a setpoint. Actual cooling and phase transfer "
                "are limited by the connected device's finite power.",
            }
        )
    if (
        operation == "cool_crystallize"
        and state is not None
        and base.operation_validator.native_crystal_model(state) is None
    ):
        constraints.extend(
            [
                {
                    "id": "cool_crystallize_requires_reaction_or_seed",
                    "kind": "state_precondition",
                    "description": (
                        "Requires at least one committed reaction advance or a seed charge."
                    ),
                },
                {
                    "id": "payload_coupling:maximum_cooling_rate_K_s",
                    "kind": "cross_field_upper_bound",
                    "fields": ["target_temperature_K", "duration_s"],
                    "relation": (
                        "(current_temperature_K - target_temperature_K) / duration_s "
                        "<= maximum_cooling_rate_K_s"
                    ),
                    "parameters": {
                        "current_temperature_K": float(state.temperature_K),
                        "maximum_cooling_rate_K_s": DEFAULT_MAXIMUM_COOLING_RATE_K_S,
                    },
                },
            ]
        )
    if operation == "set_potential":
        constraints.append(
            {
                "id": "payload_coupling:potential_within_voltage_window",
                "kind": "absolute_value_upper_bound",
                "fields": ["potential_V"],
                "relation": "abs(potential_V) <= default_voltage_window_V",
                "parameters": {"default_voltage_window_V": 2.5},
            }
        )
        if state is not None:
            cell_settings = equipment_settings(
                state.equipment, "electrochemical_cell", fields=("setpoint_history",)
            )
            setpoint_history = tuple(cell_settings.get("setpoint_history", ()))
            if (
                len(setpoint_history) == 1
                and getattr(base, "electrochemical_workflow_mode", None)
                != ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
            ):
                constraints.append(
                    {
                        "id": "payload_adapts:electrochemical_setpoint",
                        "kind": "minimum_change_from_previous",
                        "fields": ["potential_V", "current_mA"],
                        "relation": (
                            "abs(potential_V - previous_potential_V) >= "
                            "minimum_potential_delta_V or "
                            "abs(current_mA - previous_current_mA) >= "
                            "minimum_current_delta_mA"
                        ),
                        "parameters": {
                            "previous_potential_V": float(dict(setpoint_history[0])["potential_V"]),
                            "previous_current_mA": float(dict(setpoint_history[0])["current_mA"]),
                            "minimum_potential_delta_V": (
                                ELECTROCHEMICAL_MIN_ADAPTED_POTENTIAL_DELTA_V
                            ),
                            "minimum_current_delta_mA": (
                                ELECTROCHEMICAL_MIN_ADAPTED_CURRENT_DELTA_MA
                            ),
                        },
                    }
                )
    from chemworld.runtime.component_network import operation_available

    return {
        "schema_version": PUBLIC_ACTION_SCHEMA_VERSION,
        "operation": operation,
        "valid_operation_type": True,
        "installed": state is None or operation_available(state, operation),
        "module": contract.module,
        "kind": contract.kind,
        "required_fields": list(contract.required_fields),
        "fields": fields,
        "preconditions": list(contract.preconditions),
        "constraints": constraints,
        "task_allowed": operation in getattr(base, "allowed_operations", set(OPERATION_TYPES)),
        "notes": [
            "Use validate_action(action) before executing.",
            "Installed capabilities and task permissions persist when current preconditions fail.",
            "Payload aliases are canonicalized by the environment action codec.",
            "Only declared fields are accepted; required_fields and constraints apply together.",
            *(
                [
                    "Storage domain: quenched finite-catalog mixed liquids and crystal slurries; "
                    "sealed isothermal storage has no reaction, evaporation or phase evolution.",
                    "Transfer is instantaneous and proportional in every phase; liquid heat "
                    "capacity determines adiabatic mixing temperature.",
                    "Each transfer uses a new disposable tool (cost 0.005); each new container "
                    "costs 0.02 and retires when emptied. Occupied destinations require mixing=1.",
                    "Addresses: active=0, container-01..16=1..16, filtrate-0001..0016=17..32. "
                    "Use the material_routes constraint for compatible source/destination bounds.",
                ]
                if operation in {"create_container", "transfer_material"}
                else []
            ),
        ],
    }


def validate_action(env: Any, action: dict[str, Any]) -> dict[str, Any]:
    """Validate an action against schema, task policy, instrument policy, and state."""

    base = _base_env(env)
    try:
        canonical = base.action_codec.canonicalize(action)
    except (TypeError, ValueError) as exc:
        return {
            "operation_type": str(action.get("operation", "invalid")),
            "valid": False,
            "dispatchable_to_runtime": False,
            "preconditions": {"action_schema_valid": False},
            "invalid_reasons": [str(exc)],
            "valid_operations": list(base.operation_validator.valid_operations(base._state)),
            "action_mask": list(base.operation_validator.action_mask(base._state)),
            "cost_penalty": 0.20,
            "safety_flags": {
                "operation_allowed_by_task": False,
                "precondition_failed": True,
                "action_schema_valid": False,
            },
            "canonical_action": to_builtin(action),
            "will_mutate_state": False,
        }
    validation = base.operation_validator.validate(canonical, base._state)
    payload = validation.to_dict()
    resource_reasons = _campaign_resource_rejection_reasons_for_action(
        base,
        canonical,
    )
    if resource_reasons:
        payload["valid"] = False
        payload["dispatchable_to_runtime"] = False
        payload["preconditions"] = {
            **payload.get("preconditions", {}),
            "campaign_resources_available": False,
        }
        payload["invalid_reasons"] = list(
            dict.fromkeys(
                [
                    *payload.get("invalid_reasons", []),
                    "campaign_resources_available",
                    *[f"campaign_resource:{reason}" for reason in resource_reasons],
                ]
            )
        )
        payload["safety_flags"] = {
            **payload.get("safety_flags", {}),
            "precondition_failed": True,
            "campaign_resource_rejected": True,
        }
    payload["canonical_action"] = to_builtin(canonical)
    payload["will_mutate_state"] = False
    return to_builtin(payload)


@_resource_read
def available_actions(env: Any, *, include_invalid: bool = False) -> list[dict[str, Any]]:
    """Return current operation affordances for agents and tool planners."""

    base = _base_env(env)
    affordances = _view_affordances(base)
    allowed = set(getattr(base, "allowed_operations", set(OPERATION_TYPES)))
    actions: list[dict[str, Any]] = []
    operation_types = tuple(getattr(base.operation_validator, "operation_types", OPERATION_TYPES))
    for operation in operation_types:
        if operation not in allowed and not include_invalid:
            continue
        if operation == "discard_batch":
            discard_available = getattr(
                base,
                "_campaign_batch_discard_available",
                lambda: False,
            )()
            if not discard_available and not include_invalid:
                continue
        affordance = affordances[operation]
        validation = affordance.to_dict()
        schema = action_schema(base, operation)
        resource_reasons = list(_campaign_resource_rejection_reasons_for_operation(base, operation))
        resource_reasons.extend(
            _campaign_schema_resource_rejection_reasons(base, operation, schema)
        )
        resource_reasons = list(dict.fromkeys(resource_reasons))
        is_valid = affordance.is_valid and not resource_reasons
        if operation == "measure" and schema["valid_operation_type"]:
            # The default candidate is a physical witness, not a resource-aware
            # choice. At least one fully bound instrument must remain executable.
            choices = schema["fields"][0]["choices"]
            is_valid = bool(choices)
            validation["preconditions"]["measurement_choice_available"] = is_valid
            validation["invalid_reasons"] = (
                [] if is_valid else list(dict.fromkeys([
                    *validation["invalid_reasons"], "measurement_choice_available",
                    *[
                        reason for row in schema["fields"][0]["choice_status"]
                        if row["installed"] and row["task_allowed"]
                        for reason in row["invalid_reasons"]
                    ],
                ]))
            )
        if resource_reasons:
            validation["preconditions"] = {
                **validation["preconditions"],
                "campaign_resources_available": False,
            }
            validation["invalid_reasons"] = list(
                dict.fromkeys(
                    [
                        *validation["invalid_reasons"],
                        "campaign_resources_available",
                        *[f"campaign_resource:{reason}" for reason in resource_reasons],
                    ]
                )
            )
        if not is_valid and not include_invalid:
            continue
        actions.append(
            {
                "operation": operation,
                "installed": schema["installed"],
                "task_allowed": schema["task_allowed"],
                "valid": is_valid,
                "invalid_reasons": validation["invalid_reasons"],
                "preconditions": validation["preconditions"],
                "schema": schema,
            }
        )
    return actions


@_resource_read
def sample_valid_action(
    env: Any,
    rng: np.random.Generator,
    *,
    operation: str | None = None,
    max_attempts: int = 100,
) -> dict[str, Any]:
    """Sample semantic payloads from bound schemas and validate before returning.

    Only the caller's RNG is consumed. This does not execute a physical action,
    and failed search does not imply that the feasible domain is empty.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    actions = [
        row for row in available_actions(env)
        if operation is None or row["operation"] == operation
    ]
    if not actions:
        raise ValueError("No currently available operation matches the request")
    for attempt in range(max_attempts):
        schema = actions[int(rng.integers(len(actions)))]["schema"]
        candidate: dict[str, Any] = {"operation": schema["operation"]}
        exclusive = {
            field for rule in schema["constraints"] if rule["kind"] == "exactly_one_field"
            for field in rule["fields"]
        }
        selected = {
            rule["fields"][int(rng.integers(len(rule["fields"])))]
            for rule in schema["constraints"]
            if rule["kind"] == "exactly_one_field" and rule["fields"]
        }
        for field in schema["fields"]:
            name = field["field"]
            if name in exclusive and name not in selected:
                continue
            # Try the minimal schema-valid payload first. Independent optional
            # fields may describe mutually exclusive extensions of a control.
            if (not field.get("required", True) and name not in selected
                    and (attempt == 0 or not rng.integers(2))):
                continue
            choices = field.get("choices")
            bounds = field.get("bounds")
            if choices:
                candidate[field["field"]] = choices[int(rng.integers(len(choices)))]
            elif isinstance(bounds, dict) and bounds["low"] <= bounds["high"]:
                candidate[field["field"]] = float(rng.uniform(bounds["low"], bounds["high"]))
            else:
                break
        else:
            result = validate_action(env, candidate)
            if result["valid"]:
                return result["canonical_action"]
    raise RuntimeError("No valid payload found within max_attempts; no action was executed")


@_resource_read
def resource_blocked_actions(env: Any) -> list[dict[str, Any]]:
    """Return physically valid operations blocked only by campaign resources."""

    base = _base_env(env)
    if _campaign_ledger(base) is None:
        return []
    valid = {name for name, result in _view_affordances(base).items() if result.is_valid}
    allowed = set(getattr(base, "allowed_operations", set(OPERATION_TYPES)))
    blocked: list[dict[str, Any]] = []
    operation_types = tuple(getattr(base.operation_validator, "operation_types", OPERATION_TYPES))
    for operation in operation_types:
        if operation not in valid or operation not in allowed:
            continue
        if (
            operation == "discard_batch"
            and not getattr(
                base,
                "_campaign_batch_discard_available",
                lambda: False,
            )()
        ):
            continue
        schema = action_schema(base, operation)
        reasons = list(_campaign_resource_rejection_reasons_for_operation(base, operation))
        reasons.extend(_campaign_schema_resource_rejection_reasons(base, operation, schema))
        reasons = list(dict.fromkeys(reasons))
        if not reasons:
            continue
        blocked.append(
            {
                "operation": operation,
                "status": "cannot_complete",
                "reason_codes": reasons,
                "message": (
                    f"Operation {operation} cannot be completed with the remaining "
                    "campaign resources: " + ", ".join(reasons) + "."
                ),
            }
        )
    return blocked


_CAMPAIGN_RESOURCE_STOCK_FIELDS: dict[str, tuple[str, str]] = {
    "add_component": ("reagent_mol", "amount_mol"),
    "add_reagent": ("reagent_mol", "amount_mol"),
    "add_solvent": ("solvent_L", "volume_L"),
    "resuspend_crystals": ("solvent_L", "volume_L"),
    "add_catalyst": ("catalyst_mol", "catalyst_amount_mol"),
    "seed_crystals": ("seed_g", "seed_mass_g"),
    "add_extractant": ("extractant_L", "volume_L"),
    "add_phase": ("phase_liquid_L", "volume_L"),
    "wash": ("wash_solvent_L", "wash_volume_L"),
}


def _campaign_ledger(base: Any) -> Any | None:
    ledger = getattr(base, "_campaign_resource_ledger", None)
    return ledger


def _campaign_remaining(base: Any) -> dict[str, Any]:
    ledger = _campaign_ledger(base)
    if ledger is None:
        return {}
    snapshot = _campaign_snapshot(base, ledger)
    state = snapshot.get("state", {})
    remaining = state.get("remaining", {})
    result = deepcopy(remaining) if isinstance(remaining, dict) else {}
    local = getattr(base, "_batch_resource_ledger", None)
    if local is not None:

        def intersect(global_values, local_values):
            for key, value in local_values.items():
                if isinstance(value, dict):
                    intersect(global_values.setdefault(key, {}), value)
                elif isinstance(value, (int, float)):
                    existing = global_values.get(key)
                    global_values[key] = min(existing, value) if existing is not None else value

        intersect(result, local.bounded_state()["state"]["remaining"])
    return result


def _campaign_duration_capacity_s(
    base: Any,
    operation: str,
) -> tuple[float | None, str | None]:
    """Return the executable duration ceiling from the owning resource ledger."""

    ledger = _campaign_ledger(base)
    if ledger is None:
        return None, None
    snapshot = _campaign_snapshot(base, ledger)
    state = snapshot.get("state", {})
    if not isinstance(state, dict):
        return None, None
    remaining = state.get("remaining", {})
    if not isinstance(remaining, dict):
        return None, None
    raw_remaining = remaining.get("process_time_s")
    if raw_remaining is None:
        return None, None
    capacity = max(float(raw_remaining), 0.0)
    limiting_reason = "process_time_limit"
    reserve = state.get("protected_closeout_reserve")
    if isinstance(reserve, dict):
        allowed = reserve.get("allowed_operation_classes", [])
        allowed_operations = (
            {str(item) for item in allowed} if isinstance(allowed, list | tuple) else set()
        )
        if operation not in allowed_operations:
            capacity = max(
                capacity - float(reserve.get("protected_process_time_s", 0.0)),
                0.0,
            )
            limiting_reason = "protected_closeout_process_time_reserve"
    return capacity, limiting_reason


def _campaign_starts_vessel(base: Any, operation: str) -> bool:
    return bool(
        operation != "discard_batch"
        and not getattr(base, "_campaign_resource_current_vessel_started", False)
    )


def _campaign_resource_rejection_reasons_for_action(
    base: Any,
    action: dict[str, Any],
) -> tuple[str, ...]:
    ledger = _campaign_ledger(base)
    if ledger is None:
        return ()
    operation = str(action.get("operation", ""))
    if not operation:
        return ()
    reasons = ledger.preview_rejection_reasons(
        action,
        starts_vessel=_campaign_starts_vessel(base, operation),
        instrument_reservation=lab_instrument_reservation(
            getattr(base, "_state", None), action, modules=getattr(base, "domain_modules", None)),
    )
    local = getattr(base, "_batch_resource_ledger", None)
    if local is not None:
        reasons += tuple(
            "per_batch:" + reason
            for reason in local.preview_rejection_reasons(
                action,
                starts_vessel=_campaign_starts_vessel(base, operation),
                instrument_reservation=lab_instrument_reservation(
                    getattr(base, "_state", None), action,
                    modules=getattr(base, "domain_modules", None),
                ),
            )
        )
    return reasons


def _campaign_resource_rejection_reasons_for_operation(
    base: Any,
    operation: str,
) -> tuple[str, ...]:
    ledger = _campaign_ledger(base)
    if ledger is None:
        return ()
    remaining = _campaign_remaining(base)
    reasons: list[str] = []
    if int(remaining.get("operation_attempts", 0)) < 1:
        reasons.append("operation_attempt_limit")
    operation_repeats = remaining.get("operation_repeats", {})
    if (
        operation in ledger.card.operation_repeat_limits
        and isinstance(operation_repeats, dict)
        and int(operation_repeats.get(operation, 0)) < 1
    ):
        reasons.append(f"operation_repeat_limit:{operation}")
    if _campaign_starts_vessel(base, operation) and int(remaining.get("vessel_starts", 0)) < 1:
        reasons.append("vessel_start_limit")
    stock_contract = _CAMPAIGN_RESOURCE_STOCK_FIELDS.get(operation)
    if stock_contract is not None:
        stock_id, _ = stock_contract
        if stock_id not in ledger.card.stock_limits:
            return tuple(dict.fromkeys(reasons))
        stocks = remaining.get("stocks", {})
        available = float(stocks.get(stock_id, 0.0)) if isinstance(stocks, dict) else 0.0
        if available <= 1.0e-12:
            reasons.append(f"stock_limit:{stock_id}")
    return tuple(dict.fromkeys(reasons))


def _campaign_schema_resource_rejection_reasons(
    base: Any,
    operation: str,
    schema: dict[str, Any],
) -> tuple[str, ...]:
    """Detect empty resource-constrained numeric domains in a public schema."""

    ledger = _campaign_ledger(base)
    if ledger is None:
        return ()
    reasons: list[str] = []
    if (operation == "measure" and schema["valid_operation_type"]
            and not schema["fields"][0]["choices"]):
        for row in schema["fields"][0]["choice_status"]:
            checks = row["preconditions"]
            if all(passed for key, passed in checks.items()
                   if key != "campaign_resources_available"):
                reasons.extend(
                    reason.removeprefix("campaign_resource:")
                    for reason in row["invalid_reasons"]
                    if reason.startswith("campaign_resource:")
                )
    stock_contract = _CAMPAIGN_RESOURCE_STOCK_FIELDS.get(operation)
    remaining = _campaign_remaining(base)
    stock_id = ""
    field_name = ""
    available: float | None = None
    if stock_contract is not None:
        stock_id, field_name = stock_contract
        if stock_id in ledger.card.stock_limits:
            stocks = remaining.get("stocks", {})
            available = float(stocks.get(stock_id, 0.0)) if isinstance(stocks, dict) else 0.0
    duration_capacity_s, duration_reason = _campaign_duration_capacity_s(
        base,
        operation,
    )
    for field in schema.get("fields", []):
        if not isinstance(field, dict):
            continue
        bounds = field.get("bounds")
        if not isinstance(bounds, dict):
            continue
        low = float(bounds.get("low", 0.0))
        high = float(bounds.get("high", 0.0))
        if (
            available is not None
            and field.get("field") == field_name
            and (available <= 1.0e-12 or high < low - 1.0e-12)
        ):
            reasons.append(f"stock_limit:{stock_id}")
        if (
            duration_capacity_s is not None
            and field.get("field") == "duration_s"
            and high < low - 1.0e-12
        ):
            reasons.append(duration_reason or "process_time_limit")
    return tuple(dict.fromkeys(reasons))


def _apply_campaign_resource_schema(
    base: Any,
    operation: str,
    fields: list[dict[str, Any]],
) -> None:
    """Narrow G2 schemas to the remaining public campaign envelope."""

    ledger = _campaign_ledger(base)
    if ledger is None:
        return
    remaining = _campaign_remaining(base)
    stock_contract = _CAMPAIGN_RESOURCE_STOCK_FIELDS.get(operation)
    stock_remaining: float | None = None
    if stock_contract is not None:
        stock_id, _ = stock_contract
        if stock_id in ledger.card.stock_limits:
            stocks = remaining.get("stocks", {})
            if isinstance(stocks, dict):
                stock_remaining = max(float(stocks.get(stock_id, 0.0)), 0.0)
                if operation == "add_reagent":
                    stock_remaining /= ledger.reagent_charge_molar_multiplier
    for field in fields:
        field_name = str(field.get("field", ""))
        if (
            stock_contract is not None
            and stock_remaining is not None
            and field_name == stock_contract[1]
        ):
            bounds = field.get("bounds")
            if isinstance(bounds, dict) and stock_remaining is not None:
                old_high = float(bounds.get("high", stock_remaining))
                new_high = min(old_high, stock_remaining)
                field["bounds"] = {
                    "low": float(bounds.get("low", 0.0)),
                    "high": new_high,
                }
                field["recommended_range"] = {
                    "low": float(bounds.get("low", 0.0)),
                    "high": new_high,
                }
                field["state_dependent_bounds"] = True
                field["resource_limited"] = new_high < old_high
        if field_name == "duration_s":
            duration_capacity_s, _ = _campaign_duration_capacity_s(base, operation)
            bounds = field.get("bounds")
            if duration_capacity_s is not None and isinstance(bounds, dict):
                old_high = float(bounds.get("high", duration_capacity_s))
                new_high = min(old_high, duration_capacity_s)
                field["bounds"] = {
                    "low": float(bounds.get("low", 0.0)),
                    "high": new_high,
                }
                field["recommended_range"] = {
                    "low": float(bounds.get("low", 0.0)),
                    "high": new_high,
                }
                field["state_dependent_bounds"] = True
                field["resource_limited"] = new_high < old_high


def _task_spec_value(base: Any, name: str, fallback: Any = None) -> Any:
    spec = getattr(base, "task_spec", None)
    if spec is None:
        return fallback
    return getattr(spec, name, fallback)


def _allowed_operation_groups(allowed_operations: list[str]) -> dict[str, list[str]]:
    allowed = set(allowed_operations)
    groups: dict[str, list[str]] = {}
    grouped: set[str] = set()
    for group, operations in OPERATION_GROUPS.items():
        present = [operation for operation in operations if operation in allowed]
        if present:
            groups[group] = present
            grouped.update(present)
    other = sorted(allowed - grouped)
    if other:
        groups["other"] = other
    return groups


def _default_task_prompt_profile(info: dict[str, Any], base: Any) -> dict[str, Any]:
    description = _task_spec_value(
        base,
        "description",
        info.get("description") or "Run the ChemWorld task under the public contract.",
    )
    return {
        "task_goal": description,
        "success_criteria": [
            "Optimize the public task score under the stated budget and constraints.",
            "Use only task-allowed operations and instruments.",
            "Submit a replayable trajectory with public observations and constraint flags.",
        ],
        "constraints": [
            f"Episode mode: {info.get('episode_mode')}.",
            f"Safety limit: {info.get('safety_limit')}.",
            "Measurements are noisy and consume time, sample, or cost.",
        ],
        "measurement_policy": (
            "Use available instruments to obtain public observations. Instrument "
            "outputs are partial, noisy, and cost-aware."
        ),
        "recommended_strategy": [
            "Validate actions before executing them.",
            "Use measurements to update a local world model.",
            "Balance score, cost, risk, and remaining budget.",
        ],
        "failure_modes": [
            "invalid operation",
            "precondition failure",
            "budget exhaustion",
            "unsafe or high-cost trajectory",
        ],
    }


def _task_prompt_profile(info: dict[str, Any], base: Any) -> dict[str, Any]:
    if info.get("scoring_contract", {}).get("score_family") == "unscored":
        return {
            "task_goal": info.get("objective") or "Explore the declared physical environment.",
            "success_criteria": [
                "Pursue the user's research objective with the declared "
                "operations and instruments.",
                "The user defines any reward, evaluation and stopping decision within the budget.",
            ],
            "constraints": [
                f"Episode mode: {info.get('episode_mode')}.",
                "Operations must satisfy the current action schemas and available resources.",
                "Measurements may consume time, sample or cost; history reads do not acquire data.",
            ],
            "measurement_policy": (
                "Use native observation channels with their units, quality and sample times. "
                "Missing measurements are not zero; cached observations are not new evidence."
            ),
            "recommended_strategy": [
                "Choose experiments and measurements for the user's question.",
                "Use acquired observations and remaining resources to decide what to do next.",
            ],
            "failure_modes": ["invalid operation", "precondition failure", "resource exhaustion"],
        }
    task_id = str(info.get("task_id") or "")
    if (
        task_id == "electrochemical-conversion"
        and getattr(base, "electrochemical_workflow_mode", None)
        == ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
    ):
        return ELECTROCHEMICAL_AUTONOMOUS_OPEN_PROMPT_PROFILE
    if task_id in TASK_PROMPT_PROFILES:
        return TASK_PROMPT_PROFILES[task_id]
    return _default_task_prompt_profile(info, base)


def experiment_lifecycle_contract(
    episode_mode: Any,
    *,
    termination_policy: str = "final-assay-or-budget",
    terminal_instruments: tuple[str, ...] = ("final_assay",),
) -> dict[str, str]:
    """Describe experiment completion without prescribing a search strategy."""

    final_effect = (
        "A valid final_assay completes the current experiment and, while campaign "
        "budget remains, resets the environment to a fresh experiment."
        if episode_mode == "campaign"
        else "A valid final_assay completes the experiment and ends the episode."
    )
    contract = {
        "terminate_action_template": '{"operation":"terminate"}',
        "final_assay_action_template": ('{"operation":"measure","instrument":"final_assay"}'),
        "terminate_effect": (
            "terminate marks the current process as terminated; it does not by itself "
            "complete the experiment."
        ),
        "final_assay_precondition": (
            "measure with instrument=final_assay is valid only after terminate."
        ),
        "final_assay_effect": final_effect,
        "intermediate_measurement_effect": (
            "Measurements with other instruments provide evidence but do not complete "
            "the experiment."
        ),
        "discard_batch_effect": (
            "In campaign mode, discard_batch is an explicit Agent decision that ends "
            "the current open batch without returning consumed resources; a fresh batch "
            "is exposed only when the campaign ledger still permits it."
            if episode_mode == "campaign"
            else "discard_batch is unavailable outside campaign mode."
        ),
        "discard_batch_action_template": (
            '{"operation":"discard_batch","reason":"<participant rationale>"}'
            if episode_mode == "campaign"
            else "unavailable outside campaign mode"
        ),
    }
    if termination_policy == "explicit-stop-or-budget":
        contract.update(
            terminate_effect="terminate completes this experiment and retains its equipment state.",
            final_assay_action_template="No terminal assay is required by this task.",
            final_assay_precondition="No terminal assay is required by this task.",
            final_assay_effect="Observation collection can end directly with terminate.",
        )
    elif terminal_instruments != ("final_assay",):
        instrument = terminal_instruments[0] if terminal_instruments else None
        contract.update(
            final_assay_action_template=json.dumps(
                {"operation": "measure", "instrument": instrument}, separators=(",", ":")
            ),
            final_assay_precondition=(
                f"Declared terminal instruments {', '.join(terminal_instruments)} "
                "require terminate."
            ),
            final_assay_effect=(
                "A valid terminal assay completes the experiment; physical state remains frozen."
            ),
        )
    return contract


def _render_task_prompt_text(
    *,
    task_id: str,
    objective: Any,
    budget: Any,
    episode_mode: Any,
    safety_limit: Any,
    profile: dict[str, Any],
    allowed_operations: list[str],
    allowed_instruments: list[str],
    success_metrics: list[str],
    operation_groups: dict[str, list[str]],
    experiment_lifecycle: dict[str, str],
    unscored: bool,
    submission_requirements: list[str],
) -> str:
    group_lines = [
        f"  - {group}: {', '.join(operations)}" for group, operations in operation_groups.items()
    ]
    lines = [
        f"{'Environment' if unscored else 'Task'}: {task_id or 'ad-hoc ChemWorld task'}",
        f"Goal: {profile['task_goal']}",
        (
            f"Objective: {objective}; budget: {budget} operations; "
            f"episode_mode: {episode_mode}; safety_limit: {safety_limit}."
        ),
        "Hidden information policy: " + HIDDEN_INFORMATION_POLICY,
        "Environment use:" if unscored else "Success criteria:",
        *[f"  - {item}" for item in profile["success_criteria"]],
        "Constraints:",
        *[f"  - {item}" for item in profile["constraints"]],
        "Experiment lifecycle:",
        *[f"  - {item}" for item in experiment_lifecycle.values()],
        "Allowed tools:",
        f"  - instruments: {', '.join(allowed_instruments)}",
        "  - operation groups:",
        *group_lines,
        f"  - all operations: {', '.join(allowed_operations)}",
        (
            "Reward and evaluation: user-defined. Replay checks do not assess agent performance."
            if unscored
            else "Success metrics: "
            + (", ".join(success_metrics) if success_metrics else "task score") + "."
        ),
        f"Measurement policy: {profile['measurement_policy']}",
        "Recommended strategy:",
        *[f"  - {item}" for item in profile["recommended_strategy"]],
    ]
    if submission_requirements:
        lines.append("Submission requirements: " + ", ".join(submission_requirements) + ".")
    return "\n".join(lines)


def task_prompt(env: Any) -> dict[str, Any]:
    """Build a natural-language task prompt plus structured task facts."""

    base = _base_env(env)
    info = base.task_info()
    unscored = info.get("scoring_contract", {}).get("score_family") == "unscored"
    submission_requirements = [] if unscored else list(SUBMISSION_REQUIREMENTS)
    allowed_operations = list(info.get("allowed_operations", []))
    allowed_instruments = list(info.get("allowed_instruments", []))
    success_metrics = list(
        info.get("scoring_contract", {}).get("success_metrics")
        or _task_spec_value(base, "success_metrics", ())
        or []
    )
    operation_groups = _allowed_operation_groups(allowed_operations)
    profile = _task_prompt_profile(info, base)
    experiment_lifecycle = experiment_lifecycle_contract(
        info.get("episode_mode"),
        termination_policy=_task_spec_value(base, "termination_policy", "final-assay-or-budget"),
        terminal_instruments=tuple(
            name
            for name in allowed_instruments
            if base.constitution.instruments[name].requires_terminated
        ),
    )
    task_id = str(info.get("task_id") or "")
    text = _render_task_prompt_text(
        task_id=task_id,
        objective=info.get("objective"),
        budget=info.get("budget"),
        episode_mode=info.get("episode_mode"),
        safety_limit=info.get("safety_limit"),
        profile=profile,
        allowed_operations=allowed_operations,
        allowed_instruments=allowed_instruments,
        success_metrics=success_metrics,
        operation_groups=operation_groups,
        experiment_lifecycle=experiment_lifecycle,
        unscored=unscored,
        submission_requirements=submission_requirements,
    )
    return {
        "text": text + "\nRisk signal: " + info["risk_signal_contract"]["interpretation"],
        "runtime_semantics_id": info.get("runtime_semantics_id"),
        "risk_signal_contract": deepcopy(info["risk_signal_contract"]),
        "task_id": info.get("task_id"),
        "objective": info.get("objective"),
        "budget": info.get("budget"),
        "episode_mode": info.get("episode_mode"),
        "description": _task_spec_value(base, "description", info.get("description")),
        "world_law_id": info.get("world_law_id"),
        "scenario_id": info.get("scenario_id"),
        "allowed_operations": info.get("allowed_operations", []),
        "allowed_instruments": info.get("allowed_instruments", []),
        "success_metrics": success_metrics,
        "safety_limit": info.get("safety_limit"),
        "threshold": None if unscored else _task_spec_value(
            base, "threshold", info.get("threshold")
        ),
        "termination_policy": _task_spec_value(
            base,
            "termination_policy",
            info.get("termination_policy"),
        ),
        "observation_policy": _task_spec_value(
            base,
            "observation_policy",
            info.get("observation_policy"),
        ),
        "task_goal": profile["task_goal"],
        "constraints": list(profile["constraints"]),
        "success_criteria": list(profile["success_criteria"]),
        "allowed_tools": {
            "operations": allowed_operations,
            "operation_groups": operation_groups,
            "instruments": allowed_instruments,
        },
        "material_catalog": info.get("material_catalog", {}),
        "material_information": info.get("material_information"),
        **(
            {"campaign_resources": deepcopy(info["campaign_resources"])}
            if "campaign_resources" in info
            else {}
        ),
        "electrochemical_workflow_mode": info.get("electrochemical_workflow_mode"),
        "measurement_policy": profile["measurement_policy"],
        "experiment_lifecycle": experiment_lifecycle,
        "recommended_strategy": list(profile["recommended_strategy"]),
        "failure_modes": list(profile["failure_modes"]),
        "hidden_information_policy": HIDDEN_INFORMATION_POLICY,
        "submission_requirements": submission_requirements,
        "prompt_version": (
            "chemworld-environment-prompt-0.1" if unscored else "chemworld-agent-task-prompt-0.3"
        ),
    }


def campaign_state(env: Any) -> dict[str, Any]:
    """Return visible campaign progress and best-so-far state."""

    base = _base_env(env)
    summaries = deepcopy(getattr(base, "_experiment_summaries", []))
    completed_summaries = [summary for summary in summaries if summary.get("final_assay") is True]
    scored = [
        _safe_float(summary.get("leaderboard_score"), default=-1.0)
        for summary in completed_summaries
        if summary.get("leaderboard_score") is not None
    ]
    info = _latest_info(base)
    current_score = info.get("leaderboard_score")
    current_score_already_summarized = bool(
        info.get("experiment_ended") and getattr(base, "episode_mode", None) == "campaign"
    )
    if current_score is not None and not current_score_already_summarized:
        scored.append(_safe_float(current_score, default=-1.0))
    best_score = max(scored) if scored else None
    remaining_budget = max(
        int(getattr(base, "budget", 0)) - int(getattr(base, "_step_count", 0)),
        0,
    )
    payload = {
        "campaign_id": getattr(base, "_campaign_id", None),
        "task_id": getattr(base, "task_id", None),
        "scenario_id": getattr(getattr(base, "scenario_spec", None), "scenario_id", None),
        "episode_mode": getattr(base, "episode_mode", None),
        "experiment_index": int(getattr(base, "_experiment_index", 0)),
        "operation_count": int(getattr(base, "_step_count", 0)),
        "remaining_budget": remaining_budget,
        "budget": int(getattr(base, "budget", 0)),
        "final_assay_count": len(completed_summaries)
        + (1 if current_score is not None and not current_score_already_summarized else 0),
        "best_score": best_score,
        "last_terminal_summary": summaries[-1] if summaries else None,
        "experiment_summaries": to_builtin(summaries),
        "completed_batches": to_builtin(completed_summaries),
        "discarded_batches": to_builtin(
            [summary for summary in summaries if summary.get("outcome") == "discarded"]
        ),
        "done": bool(getattr(base, "_done", False)),
    }
    resource_state_factory = getattr(
        base,
        "public_campaign_resource_state",
        None,
    )
    if callable(resource_state_factory):
        resources = resource_state_factory()
        if resources is not None:
            payload["campaign_resources"] = resources
    declared_process_factory = getattr(
        base,
        "public_declared_process_resource_state",
        None,
    )
    if callable(declared_process_factory):
        declared_process = declared_process_factory()
        if declared_process is not None:
            payload["declared_process_resources"] = declared_process
    return payload


def rl_observation_view(
    observation: dict[str, Any],
    info: dict[str, Any] | None = None,
    *,
    include_cost: bool = True,
) -> dict[str, Any]:
    """Return a NaN-safe vector observation and binary observed mask."""

    values: list[float] = []
    mask: list[float] = []
    for key in OBSERVATION_KEYS:
        value, observed = _observed_float(observation.get(key))
        values.append(value if observed else RL_MISSING_VALUE)
        mask.append(1.0 if observed else 0.0)
    if include_cost:
        values.append(_safe_float((info or {}).get("cost"), default=0.0))
        mask.append(1.0)
    spec = rl_observation_spec(include_cost=include_cost)
    return {
        "mode": "rl",
        "schema_version": "chemworld-rl-view-0.2",
        "keys": spec["keys"],
        "vector": values,
        "mask": mask,
        "cost": _safe_float((info or {}).get("cost"), default=0.0),
        "bounds": spec["value_bounds"],
        "mask_bounds": spec["mask_bounds"],
        "missing_value": RL_MISSING_VALUE,
        "dtype": "float32",
        "nan_safe": True,
        "constraint_flags": to_builtin((info or {}).get("constraint_flags", {})),
    }


def rl_observation_spec(*, include_cost: bool = True) -> dict[str, Any]:
    """Return the stable RL vector contract used by views and wrappers."""

    keys = [*OBSERVATION_KEYS, *(["cost_signal"] if include_cost else [])]
    low = [RL_OBSERVATION_VALUE_BOUNDS[0] for _ in OBSERVATION_KEYS]
    high = [RL_OBSERVATION_VALUE_BOUNDS[1] for _ in OBSERVATION_KEYS]
    if include_cost:
        low.append(RL_COST_BOUNDS[0])
        high.append(RL_COST_BOUNDS[1])
    return {
        "schema_version": "chemworld-rl-view-0.2",
        "keys": list(keys),
        "value_bounds": {
            "low": low,
            "high": high,
        },
        "mask_bounds": {
            "low": [RL_MASK_BOUNDS[0] for _ in keys],
            "high": [RL_MASK_BOUNDS[1] for _ in keys],
        },
        "missing_value": RL_MISSING_VALUE,
        "dtype": "float32",
        "nan_safe": True,
    }


def _peak_group(peak: dict[str, Any]) -> str:
    for key in ("public_role", "role", "group", "species_role", "species_id"):
        value = str(peak.get(key, "")).lower()
        if value:
            if "reactant" in value:
                return "reactant"
            if "byproduct" in value or "impurity" in value:
                return "impurity"
            if "degradation" in value:
                return "degradation"
            if "target" in value or "product" in value:
                return "target"
    return "unknown"


def _find_peak_tables(payload: Any) -> list[list[dict[str, Any]]]:
    if isinstance(payload, dict):
        tables: list[list[dict[str, Any]]] = []
        for key, value in payload.items():
            if key in {"peaks", "peak_table", "chromatogram"} and isinstance(value, list):
                dict_rows = [row for row in value if isinstance(row, dict)]
                if dict_rows:
                    tables.append(dict_rows)
            else:
                tables.extend(_find_peak_tables(value))
        return tables
    if isinstance(payload, list):
        tables = []
        for item in payload:
            tables.extend(_find_peak_tables(item))
        return tables
    return []


def _area_value(peak: dict[str, Any]) -> float:
    for key in ("area_fraction", "peak_fraction", "area", "height", "intensity"):
        if key in peak:
            return max(_safe_float(peak[key], default=0.0), 0.0)
    return 0.0


def _spectral_fit(info: dict[str, Any]) -> dict[str, Any]:
    """Expose the existing public fit separately from endpoint proxy estimates."""
    raw = info.get("raw_signal", {})
    fit = raw.get("processed_estimates", {}) if isinstance(raw, dict) else {}
    return to_builtin(fit) if isinstance(fit, dict) else {}


def spectra_summary(info: dict[str, Any]) -> dict[str, Any]:
    """Summarize public raw signals without exposing hidden mechanism identities."""

    raw_signal = info.get("raw_signal", {})
    raw_signal_dict = raw_signal if isinstance(raw_signal, dict) else {}
    spectra = raw_signal_dict.get("spectra", {})
    channels = raw_signal_dict.get("channels", [])
    if not isinstance(channels, list):
        channels = []
    if isinstance(spectra, dict) and not channels:
        channels = sorted(str(key) for key in spectra)
    axis = raw_signal_dict.get("axis", {})
    signal = raw_signal_dict.get("raw_signal", {})
    axis_values = axis.get("values", []) if isinstance(axis, dict) else []
    signal_values = signal.get("values", []) if isinstance(signal, dict) else []
    sample_count = (
        len(axis_values)
        if isinstance(axis_values, list)
        and isinstance(signal_values, list)
        and len(axis_values) == len(signal_values)
        else 0
    )
    peak_tables = _find_peak_tables(raw_signal)
    grouped: dict[str, float] = {
        "target": 0.0,
        "reactant": 0.0,
        "impurity": 0.0,
        "degradation": 0.0,
        "unknown": 0.0,
    }
    dominant_group: str | None = None
    dominant_fraction = 0.0
    total_area = 0.0
    for table in peak_tables:
        for peak in table:
            area = _area_value(peak)
            group = _peak_group(peak)
            grouped[group] = grouped.get(group, 0.0) + area
            total_area += area
    if total_area > 0.0:
        for group, value in list(grouped.items()):
            grouped[group] = value / total_area
        top_group, top_fraction = max(grouped.items(), key=lambda item: item[1])
        dominant_group = top_group
        dominant_fraction = top_fraction

    processed = info.get("processed_estimate", {})
    observed = info.get("observed_keys", [])
    warnings: list[str] = []
    if grouped["reactant"] > grouped["target"] and any(
        key in observed for key in ("yield", "purity", "recovery")
    ):
        warnings.append("reactant_peak_dominates_public_spectrum")
    if grouped["impurity"] + grouped["degradation"] > 0.35:
        warnings.append("impurity_or_degradation_peaks_are_large")

    return {
        "instrument": info.get("instrument_source") or info.get("instrument"),
        "packet_kind": raw_signal_dict.get("kind"),
        "has_spectral_packet": bool(
            sample_count
            or peak_tables
            or spectra
            or channels
            or raw_signal_dict.get("kind") == "ph_meter_signal"
        ),
        "sample_count": sample_count,
        "axis_unit": axis.get("unit") if isinstance(axis, dict) else None,
        "spectral_fit": _spectral_fit(info),
        "channels": [str(channel) for channel in channels],
        "channel_count": len(channels),
        "peak_table_count": len(peak_tables),
        "observed_keys": list(observed),
        "peak_group_fractions": grouped,
        "dominant_peak": {"group": dominant_group, "fraction": dominant_fraction},
        "processed_estimate": to_builtin(processed),
        "uncertainty": to_builtin(info.get("uncertainty", {})),
        "warnings": warnings,
    }


def _recovery_suggestion(
    env: Any,
    info: dict[str, Any],
) -> str | None:
    failure = _failure_summary(info)
    transaction = failure["transaction_status"]
    if not (
        failure["precondition_failed"]
        or failure["campaign_resource_rejected"]
        or failure["constitution_failed"]
        or failure["error_message"]
        or transaction not in (None, "committed")
        or info.get("operation_outcome") == "rejected"
    ):
        return None
    if getattr(_base_env(env), "_done", False):
        return "This session has ended; call reset() before submitting another action."
    if failure["campaign_resource_rejected"]:
        return (
            "Inspect resource_rejection_reasons and remaining resources; check "
            "available_actions() and the chosen action_schema() before retrying."
        )
    if failure["constitution_failed"] or transaction == "rolled_back":
        return (
            "The action was rolled back. Inspect the failure summary and revise the "
            "action parameters; validate_action() can check a proposed retry without executing it."
        )
    return (
        "Inspect failed_preconditions, check available_actions(), then consult the chosen "
        "action_schema() and validate_action() before retrying."
    )


def _visible_metrics(observation: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in OBSERVATION_KEYS:
        value, observed = _observed_float(observation.get(key))
        if observed:
            metrics[key] = value
    return metrics


def _format_metric_map(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={metrics[key]:.3f}" for key in keys if key in metrics]
    return ", ".join(parts) if parts else "none"


def _instrument_summary(info: dict[str, Any]) -> dict[str, Any]:
    instrument = info.get("instrument_source") or info.get("instrument")
    return {
        "operation_type": info.get("operation_type"),
        "instrument": instrument,
        "is_measurement": info.get("operation_type") == "measure",
        "observed_keys": to_builtin(info.get("observed_keys", [])),
        "measurement_cost": _safe_float(info.get("measurement_cost"), default=0.0),
        "sample_consumed": _safe_float(info.get("sample_consumed"), default=0.0),
        "reward_source": info.get("reward_source"),
    }


def _final_assay_summary(info: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    instrument = info.get("instrument_source") or info.get("instrument")
    is_final_assay = info.get("operation_type") == "measure" and bool(
        info.get("terminal_assay", instrument == "final_assay")
    )
    leaderboard_score = info.get("leaderboard_score")
    return {
        "is_final_assay": is_final_assay,
        "leaderboard_score": (
            None if leaderboard_score is None else _safe_float(leaderboard_score)
        ),
        "leaderboard_eligible": bool(is_final_assay and leaderboard_score is not None),
        "experiment_ended": bool(info.get("experiment_ended", False)),
        "episode_done": bool(campaign.get("done", False)),
        "final_assay_count": campaign.get("final_assay_count", 0),
        "last_terminal_summary": to_builtin(campaign.get("last_terminal_summary")),
    }


def _campaign_progress(campaign: dict[str, Any]) -> dict[str, Any]:
    budget = max(int(campaign.get("budget") or 0), 0)
    operation_count = max(int(campaign.get("operation_count") or 0), 0)
    remaining = max(int(campaign.get("remaining_budget") or 0), 0)
    progress_fraction = operation_count / budget if budget > 0 else 0.0
    best_score = campaign.get("best_score")
    return {
        "campaign_id": campaign.get("campaign_id"),
        "task_id": campaign.get("task_id"),
        "episode_mode": campaign.get("episode_mode"),
        "experiment_index": campaign.get("experiment_index"),
        "operation_count": operation_count,
        "budget": budget,
        "remaining_budget": remaining,
        "progress_fraction": progress_fraction,
        "final_assay_count": campaign.get("final_assay_count", 0),
        "best_score": None if best_score is None else _safe_float(best_score),
        "done": bool(campaign.get("done", False)),
    }


def _failure_summary(info: dict[str, Any]) -> dict[str, Any]:
    flags = info.get("constraint_flags", {})
    preconditions = info.get("preconditions", {})
    failed_preconditions = [
        str(key) for key, passed in preconditions.items() if isinstance(passed, bool) and not passed
    ]
    return {
        "precondition_failed": bool(flags.get("precondition_failed", False)),
        "campaign_resource_rejected": bool(flags.get("campaign_resource_rejected", False)),
        "constitution_failed": bool(flags.get("constitution_failed", False)),
        "transaction_status": info.get("transaction_status"),
        "rollback_reason": info.get("rollback_reason"),
        # The operation can reject an otherwise valid payload during execution.
        # Publish its control error, not the accompanying private model diagnostics.
        "error_message": info.get("error_message")
        or info.get("outcome_diagnostics", {}).get("control_error"),
        "resource_rejection_reasons": to_builtin(
            info.get("campaign_resource_rejection_reasons", [])
        ),
        "failed_preconditions": failed_preconditions,
    }


def action_failure_feedback(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Summarize a finalized public outcome without building schemas or executing actions."""
    return {
        "failure_summary": _failure_summary(info),
        "recovery_suggestion": _recovery_suggestion(env, info),
    }


@_resource_read
def lab_report_view(
    env: Any,
    observation: dict[str, Any],
    info: dict[str, Any],
    *,
    _actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a compact public lab report for LLMs and students."""

    actions = available_actions(env) if _actions is None else _actions
    summary = spectra_summary(info)
    campaign = campaign_state(env)
    progress = _campaign_progress(campaign)
    metrics = _visible_metrics(observation)
    instrument = _instrument_summary(info)
    final_assay = _final_assay_summary(info, campaign)
    failure = _failure_summary(info)
    score = _safe_float(observation.get("score"), default=0.0)
    cost = _safe_float(observation.get("cost"), default=0.0)
    risk = _safe_float(observation.get("safety_risk"), default=0.0)
    resource_rejected = failure["campaign_resource_rejected"]
    transaction = failure["transaction_status"]
    if resource_rejected:
        status = "cannot_complete"
    elif failure["precondition_failed"]:
        status = "failed_precondition"
    elif transaction is not None and transaction != "committed":
        status = str(transaction)
    elif failure["constitution_failed"] or info.get("operation_outcome") == "rejected":
        status = "rejected"
    else:
        status = "accepted"
    operation = info.get("operation_type") or "none"
    opening = (
        f"Operation {operation} cannot be completed with the remaining campaign resources."
        if resource_rejected
        else f"Operation {operation} was {status}."
    )
    lines = [
        opening,
        f"Visible score={score:.3f}, cost={cost:.3f}, safety_risk={risk:.3f} "
        "(mixed process/procedure penalty, not a physical hazard probability).",
        (
            "Key public metrics: "
            + _format_metric_map(
                metrics,
                (
                    "yield",
                    "selectivity",
                    "conversion",
                    "purity",
                    "recovery",
                    "phase_ratio",
                    "pH_normalized",
                    "acid_dissociation_fraction",
                    "precipitation_signal",
                    "equilibrium_residual",
                    "equilibrium_confidence",
                    "score",
                ),
            )
            + "."
        ),
        f"Observed keys: {', '.join(summary['observed_keys']) or 'none'}.",
        (
            f"Campaign progress: step {progress['operation_count']}/{progress['budget']}, "
            f"experiment {progress['experiment_index']}, "
            f"remaining {progress['remaining_budget']}, "
            f"final_assays {progress['final_assay_count']}."
        ),
    ]
    if status != "accepted":
        lines.insert(
            1,
            "The requested operation was not executed. "
            "Attempt and penalty accounting still applies.",
        )
        if failure["error_message"]:
            lines.insert(2, "Reason: " + str(failure["error_message"]))
    if instrument["instrument"] is not None:
        lines.append(
            f"Instrument: {instrument['instrument']} "
            f"(measurement_cost={instrument['measurement_cost']:.3f}, "
            f"sample={instrument['sample_consumed']:.6f})."
        )
    if final_assay["is_final_assay"]:
        score_text = (
            "none"
            if final_assay["leaderboard_score"] is None
            else f"{float(final_assay['leaderboard_score']):.3f}"
        )
        lines.append(
            "Final assay: leaderboard_eligible="
            f"{final_assay['leaderboard_eligible']}, score={score_text}."
        )
    if summary["has_spectral_packet"]:
        if summary["sample_count"]:
            lines.append(
                f"Spectra packet: {summary['packet_kind']} with {summary['sample_count']} "
                f"sampled points (axis: {summary['axis_unit']})."
            )
            if summary["spectral_fit"]:
                lines.append(
                    "Public reporter concentration estimates, fit standard errors and "
                    "missingness are available in spectral_fit; they are separate from "
                    "endpoint proxy metrics."
                )
        else:
            channels = ", ".join(summary["channels"]) or "public peaks"
            lines.append(
                f"Spectra packet: {summary['packet_kind'] or 'instrument_signal'} "
                f"with {summary['channel_count']} channels ({channels})."
            )
    dominant = summary["dominant_peak"]
    if dominant["group"] is not None:
        lines.append(
            f"Dominant public spectral group: {dominant['group']} "
            f"({float(dominant['fraction']):.2f})."
        )
    if summary["warnings"]:
        lines.append("Spectral warnings: " + ", ".join(summary["warnings"]) + ".")
    if failure["resource_rejection_reasons"]:
        lines.append("Resource limits: " + ", ".join(failure["resource_rejection_reasons"]) + ".")
    recovery = _recovery_suggestion(env, info)
    if recovery is not None:
        lines.append(f"Recovery suggestion: {recovery}")
    if campaign["best_score"] is not None:
        lines.append(f"Best score so far: {float(campaign['best_score']):.3f}.")
    return {
        "mode": "lab_report",
        "report_version": "chemworld-lab-report-0.2",
        "text": "\n".join(lines),
        "status": status,
        "operation_type": info.get("operation_type"),
        "visible_metrics": metrics,
        "instrument_summary": instrument,
        "final_assay_summary": final_assay,
        "spectra_summary": summary,
        "campaign_state": campaign,
        "campaign_progress": progress,
        "failure_summary": failure,
        "next_action_hints": [entry["operation"] for entry in actions[:5]],
        "recovery_suggestion": recovery,
        "constraint_flags": to_builtin(info.get("constraint_flags", {})),
    }


def full_process_operational_state(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Expose process controls and sensor readings, never composition or model parameters.

    This is an additive development interface for the three full-process tasks.
    It deliberately separates current controls from carried analytical results.
    """
    base = _base_env(env)
    if getattr(base, "task_id", None) not in {
        "reaction-to-purification",
        "reaction-to-crystallization",
        "reaction-to-distillation",
    }:
        return {}
    state = getattr(base, "_state", None)
    if state is None:
        return {}
    selected = selected_phase_id(state.phases)
    fresh = (
        info.get("operation_type") == "measure"
        and info.get("transaction_status") == "committed"
        and info.get("instrument") is not None
    )
    process = state.process
    has_previous = bool(process is not None and process.last_observation)
    phases = {} if state.phases is None else state.phases.phases
    from chemworld.runtime.full_process_contract import (
        CRYSTAL_RECOVERY_DEFINITION,
        active,
        sample_domain,
    )

    resolved = active(state)
    return {
        "schema_version": "chemworld-full-process-operational-state-0.1",
        "sensor_contract": "Ideal temperature and volume readouts; analytical quality requires "
        "a paid instrument. Phase identities and selection are operation bookkeeping.",
        "temperature_K": float(state.temperature_K),
        "active_volume_L": float(state.volume_L),
        "process_time_s": float(state.ledger.time_s),
        "quenched": bool(state.quenched),
        "terminated": bool(state.terminated),
        "selected_phase": selected,
        "phases": {
            str(key): {
                "phase_type": phase.phase_type,
                "volume_L": float(phase.volume_L),
                "selected": bool(phase.selected),
                "settled": bool(phase.settled),
            }
            for key, phase in phases.items()
        },
        "measurement_context": {
            "fresh_measurement_this_step": fresh,
            "status": "measured_this_step"
            if fresh
            else ("carried_not_remeasured" if has_previous else "not_measured"),
            "instrument_this_step": info.get("instrument") if fresh else None,
            "selected_phase_at_measurement": selected if fresh else None,
            "scope_note": "Current phase selection does not relabel an earlier reading. "
            "Reaction metrics and downstream metrics follow their own instrument contracts; "
            "they are not all assays of the same material.",
            "purification_default_metric_phase": (
                "organic" if selected is None and "organic" in phases else None
            ),
        },
        "sampling_contract": (
            "Only the selected liquid phase is depleted. Solid and mother_liquor form one "
            "representative analytical slurry group, including after filtration; they are not "
            "separately stored bottles. "
            + (
                CRYSTAL_RECOVERY_DEFINITION
                if base.task_id == "reaction-to-crystallization"
                else "Recovery/yield denominators are original reactant charge."
            )
            if resolved
            else "Current runtime proportionally depletes all inventories during "
            "sampling, including saved phases. It is not isolated receiver sampling."
        ),
        "sample_domain": list(sample_domain(state)) if resolved else None,
        "receiver_selection_contract": (
            "collect_fraction=0 selects an existing saved collected_fraction without mixing "
            "new distillate. Positive fractions add current distillate to that saved receiver."
            if resolved and base.task_id == "reaction-to-distillation"
            else None
        ),
        "particle_measurement_contract": (
            "Paid particle_size measurement is available after crystals exist: non-destructive, "
            "120 seconds, cost 0.04, noisy number-weighted d50 and fines below 20 um."
            if resolved and base.task_id == "reaction-to-crystallization"
            else "Crystal size and fines are measured only by final_assay in the current task; "
            "HPLC does not provide an online particle-size measurement."
            if base.task_id == "reaction-to-crystallization"
            else None
        ),
    }


@_resource_read
def tool_json_view(env: Any, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Return a structured public observation bundle for tool agents."""

    actions = available_actions(env)
    from chemworld.world.instruments import public_native_observation

    native = info.get("native_observation")
    if native is not None:
        native = public_native_observation(
            native,
            _base_env(env).constitution,
            debug_truth=getattr(_base_env(env), "debug_truth", False),
        )
        from chemworld.envs.spaces import to_observation
        from chemworld.foundation.observations import Observation

        reading = Observation.from_native_record(native)
        observation = to_observation(reading.values)
        info = {
            **info,
            "raw_signal": reading.raw_signal,
            "processed_estimate": reading.processed_estimate,
            "uncertainty": reading.uncertainty,
            "observed_mask": reading.observed_mask,
            "observed_keys": [k for k, seen in reading.observed_mask.items() if seen],
        }
    material_state = getattr(_base_env(env), "_state", None)
    operational = full_process_operational_state(env, info)
    research = getattr(_base_env(env), "research_brief", None)
    if research is not None:
        from chemworld.research_brief import public_research_brief, research_operational_state

        operational = research_operational_state(_base_env(env), info)
    return {
        "mode": "tool_json",
        "task": {
            "task_id": getattr(_base_env(env), "task_id", None),
            "objective": getattr(_base_env(env), "objective", None),
        },
        "observation": to_builtin(observation),
        "raw_signal": to_builtin(info.get("raw_signal", {})),
        "native_observation": to_builtin(native),
        "spectral_fit": _spectral_fit(info),
        "processed_estimate": to_builtin(info.get("processed_estimate", {})),
        "uncertainty": to_builtin(info.get("uncertainty", {})),
        "observed_keys": to_builtin(info.get("observed_keys", [])),
        "observed_mask": to_builtin(info.get("observed_mask", {})),
        "cost": _safe_float(info.get("cost"), default=0.0),
        "cost_components": to_builtin(info.get("cost_components", {})),
        "constraints": to_builtin(info.get("constraint_flags", {})),
        "campaign_state": campaign_state(env),
        "sample_inventory": (
            material_state.samples.public_summary() if material_state is not None else []
        ),
        "material_routing": (
            material_state.samples.routing_summary() if material_state is not None else {}
        ),
        "component_network": public_network(material_state) if material_state is not None else None,
        "process_control": public_control(material_state) if material_state is not None else None,
        "continuous_streams": public_streams(material_state)
        if material_state is not None
        else None,
        "spectral_configurations": public_spectral_settings(material_state)
        if material_state is not None
        else {},
        "available_actions": actions,
        "resource_blocked_actions": resource_blocked_actions(env),
        "lab_report": lab_report_view(env, observation, info, _actions=actions),
        **({"operational_state": operational} if operational else {}),
        **({"research_brief": public_research_brief(research)} if research is not None else {}),
    }


def observation_view(
    env: Any,
    mode: str = "tool_json",
    observation: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an agent-facing observation view."""

    base = _base_env(env)
    obs = _latest_observation(base) if observation is None else observation
    step_info = _latest_info(base) if info is None else info
    if mode == "rl":
        return rl_observation_view(obs, step_info)
    if mode == "tool_json":
        return tool_json_view(base, obs, step_info)
    if mode == "lab_report":
        return lab_report_view(base, obs, step_info)
    raise ValueError("mode must be one of 'rl', 'tool_json', or 'lab_report'")


@_resource_read
def agent_view_bundle(
    env: Any,
    observation: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    """Return all standard agent-facing views for trajectory export."""

    tool_view = observation_view(env, "tool_json", observation, info)
    return {
        "rl": observation_view(env, "rl", observation, info),
        "tool_json": tool_view,
        # Preserve independent returned dictionaries without recomputing views.
        "lab_report": deepcopy(tool_view["lab_report"]),
    }


__all__ = [
    "action_failure_feedback",
    "action_schema",
    "agent_view_bundle",
    "available_actions",
    "campaign_state",
    "lab_report_view",
    "observation_view",
    "resource_blocked_actions",
    "rl_observation_spec",
    "rl_observation_view",
    "sample_valid_action",
    "spectra_summary",
    "task_prompt",
    "tool_json_view",
    "validate_action",
]
