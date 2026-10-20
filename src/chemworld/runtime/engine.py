"""Unified experimental session, independent of Gym action and observation spaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any
from uuid import uuid4

import numpy as np

from chemworld.action_codec import ActionCodec
from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceLedger,
    campaign_resource_event_id,
)
from chemworld.foundation import equipment_settings
from chemworld.foundation.observations import Observation
from chemworld.foundation.state import (
    OperationRecord,
    ProcessLedger,
    WorldState,
    process_with_last_observation,
)
from chemworld.materials import (
    normalize_static_material_information_config,
    static_material_information_dossier,
)
from chemworld.operation_validator import OperationValidation, OperationValidator
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE,
    ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1,
    normalize_electrochemical_workflow_mode,
)
from chemworld.runtime import (
    ChemWorldObservationKernel,
    MechanismSpeciesView,
    make_chemworld_constitution,
)
from chemworld.runtime.executor import OperationExecutor
from chemworld.runtime.full_process_contract import (
    FULL_PROCESS_CONTRACT,
    FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    FULL_PROCESS_POPULATION_CONTRACT,
    FULL_PROCESS_SEED_CONTRACT,
    FULL_PROCESS_TASKS,
    FULL_PROCESS_THERMAL_CONTRACT,
)
from chemworld.runtime.observation_noise import (
    ObservationNoiseCoordinate,
    keyed_noise_provenance,
    keyed_observation_rng,
)
from chemworld.runtime.observation_values import empty_reading, value_or_default
from chemworld.runtime.reports import (
    annotate_constitution_rollback,
    build_constitution_summary,
    build_evaluator_provenance,
    build_step_info,
    build_task_info,
    render_env,
    sanitize_agent_info,
)
from chemworld.runtime.session_snapshot import (
    SessionSnapshot,
    capture_session,
    declaration,
    restore_continuation,
)
from chemworld.tasks import default_kernel_maturity, get_task
from chemworld.world.composition import (
    CompiledWorldComposition,
    compile_world_composition,
)
from chemworld.world.crystallization_material_family import (
    apply_crystallization_material_family,
    normalize_crystallization_material_family,
)
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.electrochemical_material_family import (
    apply_electrochemical_material_family,
    normalize_electrochemical_material_family,
)
from chemworld.world.observation_contracts import TaskObservationContract
from chemworld.world.operations import (
    CAMPAIGN_OPERATION_TYPES,
    INSTRUMENTS,
    OPERATION_TYPES,
    REACTION_OPERATIONS,
)
from chemworld.world.phase_kernel import (
    INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1,
)
from chemworld.world.process_time_budget import ProcessTimeBudgetPolicy
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario
from chemworld.world.scoring import (
    PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
    TASK_DERIVED_SCORING_CONTRACT,
    TaskScoringContract,
)

DEFAULT_SCENARIO_ID = "reaction-to-assay"
__all__ = ["ChemWorldRuntime", "SessionResult", "SessionSnapshot"]


@dataclass(frozen=True)
class SessionResult:
    """A completed operation and its already-settled public outcome."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ChemWorldRuntime:
    """One owner of validation, physical execution, measurements and resource settlement."""

    metadata: dict[str, Any] = {"render_modes": ["ansi", "human"], "render_fps": 4}  # noqa: RUF012

    def __init__(
        self,
        *,
        world_split: str = "public-dev",
        budget: int = 30,
        objective: str = "balanced",
        seed: int = 0,
        task_id: str | None = None,
        composition: Mapping[str, Any] | CompiledWorldComposition | None = None,
        domain_modules: DomainModuleRegistry | None = None,
        budget_override: int | None = None,
        episode_mode_override: str | None = None,
        safety_limit_override: float | None = None,
        observation_seed_override: int | None = None,
        observation_noise_mode: str = "sequential",
        observation_noise_namespace: str = "chemworld-default-observation",
        world_interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE),
        electrochemical_material_family_id: str | None = None,
        crystallization_material_family_id: str | None = None,
        full_process_contract_id: str | None = None,
        material_information: Mapping[str, Any] | None = None,
        research_brief: Mapping[str, Any] | None = None,
        campaign_resource_card: (Mapping[str, Any] | CampaignResourceCard | None) = None,
        scoring_contract_id: str = TASK_DERIVED_SCORING_CONTRACT,
        debug_truth: bool = False,
        render_mode: str | None = None,
    ) -> None:
        initialization = {
            key: value for key, value in locals().items() if key not in {"self", "domain_modules"}
        }
        if composition is not None:
            declaration = getattr(composition, "spec", composition)
            initialization["composition"] = (
                declaration.to_dict() if hasattr(declaration, "to_dict") else dict(declaration)
            )
        if isinstance(campaign_resource_card, CampaignResourceCard):
            initialization["campaign_resource_card"] = campaign_resource_card.to_dict()
        self._initialization = deepcopy(initialization)
        self.domain_modules = domain_modules or DomainModuleRegistry.default()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode!r}")
        if task_id is not None and composition is not None:
            raise ValueError("task_id and composition are mutually exclusive")
        self.compiled_composition = (
            None
            if composition is None
            else compile_world_composition(composition, modules=self.domain_modules)
        )
        if self.compiled_composition is not None:
            self.domain_modules = self.compiled_composition.spec.modules
        self._declared_process_time_policy = self._load_process_time_policy()
        self.task_spec = (
            self.compiled_composition.task_spec
            if self.compiled_composition is not None
            else get_task(task_id)
            if task_id
            else None
        )
        self.task_id = None if self.task_spec is None else self.task_spec.task_id
        from chemworld.research_brief import normalize_research_brief

        self.research_brief = normalize_research_brief(
            research_brief,
            task_id=self.task_id,
            scoring_contract_id=scoring_contract_id,
        )
        self.runtime_task_profile_id = (
            self.compiled_composition.runtime_task_profile_id
            if self.compiled_composition is not None
            else self.task_id
        )
        if self.task_spec is not None:
            world_split = self.task_spec.world_split
            budget = self.task_spec.budget
            objective = self.task_spec.objective
        if budget_override is not None:
            budget = int(budget_override)
        if budget <= 0:
            raise ValueError("budget must be positive")
        if episode_mode_override not in {None, "single_experiment", "campaign"}:
            raise ValueError("episode_mode_override must be single_experiment, campaign, or None")
        if safety_limit_override is not None and (
            not isfinite(float(safety_limit_override))
            or not 0.0 < float(safety_limit_override) <= 1.0
        ):
            raise ValueError("safety_limit_override must be finite and in (0, 1]")
        if observation_noise_mode not in {"sequential", "keyed"}:
            raise ValueError("observation_noise_mode must be sequential or keyed")
        if not observation_noise_namespace.strip():
            raise ValueError("observation_noise_namespace must be non-empty")

        self.world_split = world_split
        self.budget = budget
        self.official_budget = self.task_spec.budget if self.task_spec is not None else budget
        self.objective = objective
        self.seed = seed
        self.observation_seed_override = (
            None if observation_seed_override is None else int(observation_seed_override)
        )
        self.observation_noise_mode = observation_noise_mode
        self.observation_noise_namespace = observation_noise_namespace
        if full_process_contract_id not in {
            None,
            FULL_PROCESS_CONTRACT,
            FULL_PROCESS_POPULATION_CONTRACT,
            FULL_PROCESS_SEED_CONTRACT,
            FULL_PROCESS_FREE_RESEARCH_CONTRACT,
            FULL_PROCESS_THERMAL_CONTRACT,
        }:
            raise ValueError("unsupported full_process_contract_id")
        self.full_process_contract_id = full_process_contract_id
        if full_process_contract_id is not None:
            if self.task_id not in FULL_PROCESS_TASKS or self.task_spec is None:
                raise ValueError("full process contract requires a native full-process task")
            self.task_spec = replace(
                self.task_spec,
                tags=(*self.task_spec.tags, full_process_contract_id),
                allowed_instruments=(*self.task_spec.allowed_instruments, "particle_size")
                if self.task_id == "reaction-to-crystallization"
                else self.task_spec.allowed_instruments,
            )
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.electrochemical_material_family_id = normalize_electrochemical_material_family(
            electrochemical_material_family_id
        )
        self.crystallization_material_family_id = normalize_crystallization_material_family(
            crystallization_material_family_id
        )
        material_family_id = (
            self.electrochemical_material_family_id
            if self.runtime_task_profile_id == "electrochemical-conversion"
            else self.crystallization_material_family_id
            if self.runtime_task_profile_id == "reaction-to-crystallization"
            else None
        )
        self.material_information_config = normalize_static_material_information_config(
            material_information,
            task_ids=(
                () if self.runtime_task_profile_id is None else (self.runtime_task_profile_id,)
            ),
            material_family_id=material_family_id,
        )
        self.material_information_condition = str(self.material_information_config["mode"])
        self._material_information_dossier = static_material_information_dossier(
            self.material_information_config,
            task_id=str(self.runtime_task_profile_id or ""),
            material_family_id=material_family_id,
        )
        self.material_information_sha256 = (
            None
            if self._material_information_dossier is None
            else hashlib.sha256(
                json.dumps(
                    self._material_information_dossier,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        self.campaign_resource_card = self._normalize_campaign_resource_card(campaign_resource_card)
        self.scoring_contract_id = str(scoring_contract_id)
        self.debug_truth = debug_truth
        self.world_interventions = tuple(world_interventions or ())
        self.render_mode = render_mode
        self.allowed_operations = (
            set(self.task_spec.allowed_operations)
            if self.task_spec is not None
            else set(REACTION_OPERATIONS)
        )
        self.allowed_instruments = (
            set(self.task_spec.allowed_instruments)
            if self.task_spec is not None
            else set(INSTRUMENTS)
        )
        self.kernel_maturity = (
            self.task_spec.kernel_maturity
            if self.task_spec is not None
            else default_kernel_maturity(
                tuple(sorted(self.allowed_operations)),
                allowed_instruments=tuple(sorted(self.allowed_instruments)),
            )
        )
        self.episode_mode = (
            self.task_spec.episode_mode if self.task_spec is not None else "single_experiment"
        )
        if episode_mode_override is not None:
            self.episode_mode = episode_mode_override
        self.campaign_controls_enabled = bool(
            self.episode_mode == "campaign" and self.campaign_resource_card is not None
        )
        if self.campaign_controls_enabled:
            self.allowed_operations.add("discard_batch")
        self.contract_profile = (
            "extended-research"
            if budget_override is not None or episode_mode_override is not None
            else "official"
        )
        self.safety_limit = self.task_spec.safety_limit if self.task_spec is not None else 0.65
        if safety_limit_override is not None:
            self.safety_limit = float(safety_limit_override)
        self.scoring_contract = TaskScoringContract.from_success_metrics(
            objective=objective,
            success_metrics=(
                self.task_spec.success_metrics if self.task_spec is not None else ("score",)
            ),
            contract_id=self.scoring_contract_id,
        )
        self.action_codec = ActionCodec(
            modules=self.domain_modules,
            feed_ids=(
                tuple(sorted(self.compiled_composition.spec.physics.feed_sources))
                if self.compiled_composition is not None
                and self.compiled_composition.spec.physics is not None
                and self.compiled_composition.spec.physics.feed_sources
                else None
            ),
            operation_types=(
                CAMPAIGN_OPERATION_TYPES if self.campaign_controls_enabled else OPERATION_TYPES
            ),
            **(
                {
                    "vessels": tuple(v["id"] for v in self.compiled_composition.spec.vessels),
                    "connections": tuple(
                        c["id"] for c in self.compiled_composition.spec.connections
                    ),
                }
                if self.compiled_composition is not None
                else {}
            ),
        )
        self.scenario_generator = (
            self.compiled_composition.spec.physics
            if self.compiled_composition is not None
            and self.compiled_composition.spec.physics is not None
            else DefaultScenarioGenerator()
        )
        self.scenario_spec = (
            self.compiled_composition.scenario_spec
            if self.compiled_composition is not None
            else get_scenario(self.task_spec.scenario_id, split=world_split)
            if self.task_spec is not None
            else get_scenario(DEFAULT_SCENARIO_ID, split=world_split)
        )
        self.scenario_instance = self.scenario_generator.generate(
            self.scenario_spec,
            seed,
            self.world_interventions,
        )
        self.scenario_instance = apply_electrochemical_material_family(
            self.scenario_instance,
            self.electrochemical_material_family_id,
        )
        self.scenario_instance = apply_crystallization_material_family(
            self.scenario_instance,
            self.crystallization_material_family_id,
        )
        self.world = self.scenario_instance.parameters
        self.constitution = make_chemworld_constitution(
            self.scenario_instance.compiled_mechanism,
            modules=self.domain_modules,
            include_particle_size=full_process_contract_id is not None,
        )
        self.observation_contract = self._make_observation_contract()
        self.runtime = self._make_runtime()
        self.operation_validator = self._make_operation_validator()
        self.observation_kernel = ChemWorldObservationKernel(
            self.constitution,
            objective,
            self.scenario_instance.compiled_mechanism,
            self.scoring_contract,
            self.observation_contract,
            modules=self.domain_modules,
            observation_noise_multiplier=self.world.domain_parameter(
                "observation_noise_multiplier"
            ),
        )
        self._rng = np.random.default_rng(self._observation_seed(seed))
        self._observation_occurrences: dict[tuple[int, str, str], int] = {}
        self._last_observation_noise_provenance: dict[str, Any] = {
            "mode": self.observation_noise_mode,
            "status": "not_observed",
        }
        self._step_count = 0
        self._experiment_index = 0
        self._operation_id = 0
        self._done = False
        self._state = self._fresh_initial_state()
        self._last_reading = empty_reading()
        self._last_operation_record: OperationRecord | None = None
        self._last_info: dict[str, Any] = {}
        self._campaign_id = self._make_campaign_id()
        self._experiment_summaries: list[dict[str, Any]] = []
        self._lifecycle_history: list[dict[str, Any]] = []
        self._campaign_resource_ledger: CampaignResourceLedger | None = None
        self._campaign_resource_current_vessel_started = False
        self._last_campaign_resource_receipt: dict[str, Any] | None = None
        self._current_batch_resource_baseline: dict[str, Any] | None = None
        self._campaign_terminal = False
        self._campaign_terminal_reason: str | None = None
        self._right_censored_open_batch = False
        self._declared_operation_counts: dict[str, int] = dict.fromkeys(
            (
                self._declared_process_time_policy.operation_repeat_limits
                if self._declared_process_time_policy is not None
                else {}
            ),
            0,
        )
        self._reset_campaign_resource_ledger()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        scope = (options or {}).get("scope", "world")
        if scope != "world":
            if seed is not None or (options or {}).get("scenario_id") is not None:
                raise ValueError("partial initialization cannot change the world or noise seed")
            return self._initialize_entities(scope, options or {})
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self._observation_seed(self.seed))
        self._observation_occurrences = {}
        self._last_observation_noise_provenance = {
            "mode": self.observation_noise_mode,
            "status": "not_observed",
        }
        if options and options.get("scenario_id"):
            if self.compiled_composition is not None:
                raise ValueError("scenario_id reset overrides are unavailable for composed worlds")
            self.scenario_spec = get_scenario(str(options["scenario_id"]), split=self.world_split)
        self.scenario_instance = self.scenario_generator.generate(
            self.scenario_spec,
            self.seed,
            self.world_interventions,
        )
        self.scenario_instance = apply_electrochemical_material_family(
            self.scenario_instance,
            self.electrochemical_material_family_id,
        )
        self.scenario_instance = apply_crystallization_material_family(
            self.scenario_instance,
            self.crystallization_material_family_id,
        )
        self.world = self.scenario_instance.parameters
        self._state = self._fresh_initial_state()
        self.constitution = make_chemworld_constitution(
            self.scenario_instance.compiled_mechanism,
            modules=self.domain_modules,
            include_particle_size=self.full_process_contract_id is not None,
        )
        self.observation_contract = self._make_observation_contract()
        self.runtime = self._make_runtime()
        self.operation_validator = self._make_operation_validator()
        self.observation_kernel = ChemWorldObservationKernel(
            self.constitution,
            self.objective,
            self.scenario_instance.compiled_mechanism,
            self.scoring_contract,
            self.observation_contract,
            modules=self.domain_modules,
            observation_noise_multiplier=self.world.domain_parameter(
                "observation_noise_multiplier"
            ),
        )
        self._step_count = 0
        self._experiment_index = 0
        self._operation_id = 0
        self._done = False
        self._last_reading = empty_reading()
        self._last_operation_record = None
        self._last_info = {}
        self._campaign_id = self._make_campaign_id()
        self._experiment_summaries = []
        self._lifecycle_history = []
        self._campaign_terminal = False
        self._campaign_terminal_reason = None
        self._right_censored_open_batch = False
        self._declared_operation_counts = dict.fromkeys(
            (
                self._declared_process_time_policy.operation_repeat_limits
                if self._declared_process_time_policy is not None
                else {}
            ),
            0,
        )
        self._reset_campaign_resource_ledger()
        return self.observe(), self.task_info()

    def _initialize_entities(self, scope, options):
        """Evaluator-only reset boundary; never an uncharged Agent operation."""
        from chemworld.foundation.lifecycle import initialize_entities, replace_equipment
        from chemworld.foundation.state_ledgers import EquipmentRecord
        from chemworld.foundation.state_ownership import StateSlots

        if self._state.lab_clock is not None:
            raise ValueError(
                "shared laboratory lifecycle requires full reset; "
                "partial resets cannot split clocks"
            )

        if scope == "equipment":
            replacement = options["replacement"]
            if isinstance(replacement, Mapping):
                replacement = EquipmentRecord(**replacement)
            slots = options.get("slots", StateSlots())
            if isinstance(slots, Mapping):
                slots = StateSlots.from_dict(dict(slots))
            transition = replace_equipment(
                self._state,
                options["equipment_id"],
                replacement,
                slots=slots,
            )
        else:
            physics = (
                self.compiled_composition.spec.physics
                if self.compiled_composition is not None
                else None
            )
            fresh_initial = self._fresh_initial_state()
            transition = initialize_entities(
                self._state,
                fresh_initial,
                scope=scope,
                vessel_ids=options.get("vessel_ids"),
                preserve_vessels=(set() if physics is None else physics.finite_supply_vessels)
                | self.runtime.domain_services.reaction_thermal.advancer.preserved_source_vessels,
            )
            if physics is not None:
                from chemworld.runtime.condensed_material_routing import (
                    reconcile_condensed_initialization,
                )

                transition = reconcile_condensed_initialization(
                    self._state, fresh_initial, transition, physics
                )
                from chemworld.runtime.crystal_thermal import reconcile_crystal_initialization

                transition = reconcile_crystal_initialization(
                    self._state, fresh_initial, transition, physics
                )
        if (
            scope != "equipment"
            and physics is not None
            and (physics.column_models or physics.concentration_models)
        ):
            # Material replacement retains stationary body H. Re-equilibrate
            # new fluid plus retained apparatus by the same PH closure, without
            # advancing the experiment clock or introducing a duplicate n.
            from chemworld.runtime.concentration import install_concentration, read_concentration
            from chemworld.runtime.dynamic_column import install_column, read_column

            active = transition.state
            locals_ = {
                active.vessel_id: active.replace(inactive_vessels={}),
                **active.inactive_vessels,
            }
            initialized = {}
            models = {**physics.column_models, **physics.concentration_models}
            for name in set(transition.receipt["vessels"]) & set(models):
                local, model = locals_[name], models[name]
                before_h = sum(s.values["enthalpy_J"] for s in local.state_slots.slots)
                if name in physics.column_models:
                    local = install_column(local, model, read_column(local, model))
                else:
                    local = install_concentration(local, model, read_concentration(local, model))
                locals_[name] = local
                initialized[name] = {
                    "total_enthalpy_before_J": before_h,
                    "total_enthalpy_after_J": sum(
                        s.values["enthalpy_J"] for s in local.state_slots.slots
                    ),
                    "temperature_K": local.temperature_K,
                }
            active = locals_.pop(active.vessel_id)
            transition = replace(
                transition,
                state=active.replace(inactive_vessels=locals_),
                receipt={
                    **transition.receipt,
                    **(
                        {
                            "column_equilibration": {
                                k: v for k, v in initialized.items() if k in physics.column_models
                            }
                        }
                        if physics.column_models
                        else {}
                    ),
                    **(
                        {
                            "concentration_equilibration": {
                                k: v
                                for k, v in initialized.items()
                                if k in physics.concentration_models
                            }
                        }
                        if physics.concentration_models
                        else {}
                    ),
                },
            )
        advancer = self.runtime.domain_services.reaction_thermal.advancer
        transition = advancer.reconcile_initialization(transition)
        report = self.constitution.check_state(transition.state)
        if not report.passed:
            raise ValueError("entity initialization produced an invalid physical state")
        self._state = transition.state
        self._lifecycle_history.append(
            {
                "after_operation_id": self._operation_id,
                "experiment_index": self._experiment_index,
                **transition.receipt,
            }
        )
        self._last_reading = empty_reading()
        self._last_operation_record = None
        self._last_info = {"lifecycle_initialization": {"scope": scope}}
        return self.observe(), self.task_info()

    def execute(
        self,
        action: Any,
    ) -> SessionResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        # Malformed or unknown agent actions are benchmark outcomes, not runner
        # crashes. Preserve their public payload so the central validator can
        # emit a replayable invalid transaction with no physical state mutation.
        # Valid aliases and numeric Gym actions still take the canonical path.
        raw_action = dict(action) if isinstance(action, Mapping) else {"operation": "invalid"}
        try:
            action = self.action_codec.canonicalize(raw_action)
        except (IndexError, OverflowError, TypeError, ValueError):
            action = raw_action
        previous_state = self._state
        resource_preflight = None
        batch_preflight = None
        resource_event_id: str | None = None
        resource_starts_vessel = False
        operation_value = action.get("operation")
        is_campaign_discard = (
            isinstance(operation_value, str) and operation_value == "discard_batch"
        )
        if self._campaign_resource_ledger is not None:
            resource_starts_vessel = (
                not self._campaign_resource_current_vessel_started and not is_campaign_discard
            )
            resource_event_id = campaign_resource_event_id(
                self._campaign_id,
                self._operation_id + 1,
            )
            resource_preflight = self._campaign_resource_ledger.preflight(
                resource_event_id,
                action,
                starts_vessel=resource_starts_vessel,
            )
            if self._batch_resource_ledger is not None:
                batch_preflight = self._batch_resource_ledger.preflight(
                    resource_event_id,
                    action,
                    starts_vessel=resource_starts_vessel,
                )
        validation = self.operation_validator.validate(action, self._state)
        declared_process_preflight = self._declared_process_time_preflight(action)
        if is_campaign_discard and not self._campaign_batch_discard_available():
            validation = self._domain_failure_validation(
                validation,
                "campaign_batch_discard_available",
            )
        campaign_resource_rejected = bool(
            resource_preflight is not None and not resource_preflight.allowed
        )
        resource_rejection_reasons = (
            resource_preflight.rejection_reasons if resource_preflight is not None else ()
        )
        if batch_preflight is not None and not batch_preflight.allowed:
            campaign_resource_rejected = True
            resource_rejection_reasons += tuple(
                "per_batch:" + reason for reason in batch_preflight.rejection_reasons
            )
        if campaign_resource_rejected:
            validation = self._campaign_resource_failure_validation(
                validation,
                resource_rejection_reasons,
            )
        if (
            declared_process_preflight is not None
            and declared_process_preflight.get("allowed") is not True
        ):
            validation = self._declared_process_failure_validation(
                validation,
                tuple(
                    str(reason)
                    for reason in declared_process_preflight.get("rejection_reasons", ())
                ),
            )
            runtime_result = self.runtime.apply_invalid_transaction(
                self._state,
                action,
                validation,
            )
        elif is_campaign_discard and validation.is_valid:
            runtime_result = self.runtime.apply_campaign_control_transaction(
                self._state,
                action,
                validation,
            )
        elif validation.dispatchable_to_runtime:
            try:
                runtime_result = (
                    self.runtime.apply_transaction(self._state, action)
                    if validation.is_valid
                    else self.runtime.apply_precondition_failure_transaction(
                        self._state,
                        action,
                        validation,
                    )
                )
            except (ArithmeticError, ValueError):
                # Physically undefined proposals are part of an exploratory agent's
                # action distribution, not a reason to terminate the entire Gym job.
                # Convert only domain/numerical errors into a replayable failed
                # transaction. Programming errors such as KeyError and TypeError
                # remain visible to developers.
                validation = self._domain_failure_validation(
                    validation,
                    "runtime_domain_valid",
                )
                runtime_result = self.runtime.apply_invalid_transaction(
                    self._state,
                    action,
                    validation,
                )
        else:
            runtime_result = self.runtime.apply_invalid_transaction(
                self._state,
                action,
                validation,
            )
        self._state = runtime_result.state
        operation_record = runtime_result.operation_record
        runtime_info = runtime_result.info_payload()
        if declared_process_preflight is not None:
            runtime_info["declared_process_time_preflight"] = deepcopy(declared_process_preflight)
            runtime_info["declared_process_resources"] = (
                self.public_declared_process_resource_state()
            )
        if campaign_resource_rejected:
            runtime_info.update(
                {
                    "transaction_status": "campaign_resource_rejected",
                    "rollback_reason": "campaign_resource_rejected",
                    "campaign_resource_rejected": True,
                    "campaign_resource_rejection_reasons": list(resource_rejection_reasons),
                }
            )
        preconditions_passed = all(operation_record.preconditions.values())
        operation_committed = (
            preconditions_passed and runtime_result.kernel_result.transaction_status == "committed"
        )
        observation_checks: list[dict[str, object]] = []
        observation_rng_state = deepcopy(self._rng.bit_generator.state)
        observation_rng = self._rng
        noise_counter_key: tuple[int, str, str] | None = None
        noise_provenance: dict[str, Any] = {
            "mode": self.observation_noise_mode,
            "status": "not_observed",
        }
        if operation_committed and not is_campaign_discard:
            observation_state = self._state
            if operation_record.is_instrument_measurement:
                # The measurement execution selects the specimen at its declared
                # sampling time. The post-execution host can have evolved further.
                specimen = runtime_result.kernel_result.measurement_state or previous_state
                observation_state = specimen.replace(
                    ledger=self._state.ledger,
                    equipment=self._state.equipment,
                )
                if self.full_process_contract_id is not None:
                    observation_state = observation_state.replace(
                        process=previous_state.process,
                        equipment=previous_state.equipment,
                    )
            if self.observation_noise_mode == "keyed":
                operation_type = str(action.get("operation") or "unknown")
                instrument = str(operation_record.instrument or action.get("instrument") or "none")
                noise_counter_key = (
                    self._experiment_index,
                    operation_type,
                    instrument,
                )
                replicate_index = self._observation_occurrences.get(
                    noise_counter_key,
                    0,
                )
                coordinate = ObservationNoiseCoordinate(
                    namespace=self.observation_noise_namespace,
                    base_observation_seed=self._observation_seed(self.seed),
                    experiment_index=self._experiment_index,
                    operation_type=operation_type,
                    instrument=instrument,
                    replicate_index=replicate_index,
                )
                observation_rng = keyed_observation_rng(coordinate)
                noise_provenance = {
                    **keyed_noise_provenance(coordinate),
                    "status": "candidate",
                }
            try:
                acquisition = runtime_result.kernel_result.diagnostics.get("acquisition")
                if acquisition is not None and not acquisition["acquisition_complete"]:
                    raise ValueError("physical acquisition stopped before readout")
                observation = self.observation_kernel.observe(
                    observation_state,
                    action,
                    observation_rng,
                )
                if acquisition is not None and acquisition["clock_semantics"] != (
                    "instantaneous_snapshot"
                ):
                    record = observation.native_record()
                    for channel in record["channels"]:
                        channel.update(
                            {
                                key: acquisition[key]
                                for key in ("sample_id", "sampled_at", "available_at")
                            }
                        )
                    observation = type(observation).from_native_record(record)
            except (ArithmeticError, ValueError) as exc:
                self._rng.bit_generator.state = observation_rng_state
                if operation_record.is_instrument_measurement:
                    runtime_result = self.runtime.invalidate_measurement(
                        runtime_result,
                        reason=f"readout_domain_error: {type(exc).__name__}",
                    )
                else:
                    validation = self._domain_failure_validation(
                        validation,
                        "observation_domain_valid",
                    )
                    runtime_result = self.runtime.apply_invalid_transaction(
                        previous_state,
                        action,
                        validation,
                    )
                    self._state = runtime_result.state
                    operation_record = runtime_result.operation_record
                    preconditions_passed = False
                    operation_committed = False
                observation = self.observation_kernel.failed_observation()
                noise_provenance["status"] = "invalid" if operation_committed else "rolled_back"
            else:
                candidate_observation_report = self.constitution.check_observation(
                    observation,
                    debug_truth=self.debug_truth,
                )
                if candidate_observation_report.passed:
                    observation_checks = candidate_observation_report.to_list()
                    noise_provenance["status"] = "committed"
                    if noise_counter_key is not None:
                        self._observation_occurrences[noise_counter_key] = (
                            self._observation_occurrences.get(noise_counter_key, 0) + 1
                        )
                    if operation_record.is_instrument_measurement:
                        runtime_result = replace(
                            runtime_result,
                            kernel_result=replace(
                                runtime_result.kernel_result,
                                measurement_status="valid",
                            ),
                        )
                else:
                    self._rng.bit_generator.state = observation_rng_state
                    observation_checks = candidate_observation_report.to_list()
                    if operation_record.is_instrument_measurement:
                        # The packet is withheld, but actual withdrawal, instrument
                        # use and its irreversible consumption remain committed.
                        runtime_result = self.runtime.invalidate_measurement(
                            runtime_result,
                            reason="readout_contract_failed",
                        )
                    else:
                        validation = self._domain_failure_validation(
                            validation,
                            "observation_domain_valid",
                        )
                        runtime_result = self.runtime.apply_invalid_transaction(
                            previous_state,
                            action,
                            validation,
                        )
                        self._state = runtime_result.state
                        operation_record = runtime_result.operation_record
                        preconditions_passed = False
                        operation_committed = False
                    runtime_info = runtime_result.info_payload()
                    observation = self.observation_kernel.failed_observation()
                    noise_provenance["status"] = "invalid" if operation_committed else "rolled_back"
        else:
            observation = self.observation_kernel.failed_observation()
        if runtime_result.kernel_result.measurement_status == "invalid" and noise_counter_key:
            self._observation_occurrences[noise_counter_key] = (
                self._observation_occurrences.get(noise_counter_key, 0) + 1
            )
        if operation_committed:
            operation = str(action.get("operation", ""))
            if operation in self._declared_operation_counts:
                self._declared_operation_counts[operation] += 1
        runtime_info = runtime_result.info_payload()
        if declared_process_preflight is not None:
            runtime_info["declared_process_time_preflight"] = deepcopy(declared_process_preflight)
            runtime_info["declared_process_resources"] = (
                self.public_declared_process_resource_state()
            )
        if campaign_resource_rejected:
            runtime_info.update(
                {
                    "transaction_status": "campaign_resource_rejected",
                    "rollback_reason": "campaign_resource_rejected",
                    "campaign_resource_rejected": True,
                    "campaign_resource_rejection_reasons": list(resource_rejection_reasons),
                }
            )
        self._last_observation_noise_provenance = deepcopy(noise_provenance)
        if not observation_checks:
            observation_checks = self.constitution.check_observation(
                observation,
                debug_truth=self.debug_truth,
            ).to_list()
        operation_record = replace(
            operation_record,
            constitution_checks=[
                *operation_record.constitution_checks,
                *observation_checks,
            ],
        )
        observation_values = observation.values
        if (
            operation_committed
            and operation_record.is_instrument_measurement
            and runtime_result.kernel_result.measurement_status == "valid"
        ):
            self._state = self._state.replace(
                process=process_with_last_observation(
                    self._state.process,
                    observation,
                )
            )
        previous_process = previous_state.process or ProcessLedger()

        resource_outcome_delta = None
        if self._campaign_resource_ledger is not None:
            if resource_event_id is None or resource_preflight is None:
                raise RuntimeError("campaign resource preflight receipt is missing")
            resource_outcome = {
                "operation_committed": operation_committed,
                "transaction_status": runtime_info["transaction_status"],
                "material_stock_consumption": dict(
                    operation_record.material_balance.stock_consumption
                ),
                "campaign_resource_report_delta": (
                    self._campaign_resource_report_delta(
                        previous_state,
                        self._state,
                        observation.values,
                    )
                ),
            }
            resource_outcome_delta = self._campaign_resource_ledger.record_outcome(
                resource_event_id,
                action,
                resource_outcome,
                starts_vessel=resource_starts_vessel,
            )
            if self._batch_resource_ledger is not None:
                self._batch_resource_ledger.record_outcome(
                    resource_event_id,
                    action,
                    resource_outcome,
                    starts_vessel=resource_starts_vessel,
                )
            if resource_outcome_delta.vessel_starts:
                self._campaign_resource_current_vessel_started = True
            self._last_campaign_resource_receipt = {
                "event_id": resource_event_id,
                "preflight": resource_preflight.to_dict(),
                "outcome_delta": resource_outcome_delta.to_dict(),
                "operation_committed": operation_committed,
                "transaction_status": runtime_info["transaction_status"],
                "rejected": campaign_resource_rejected,
                "rejection_reasons": list(resource_preflight.rejection_reasons),
            }

        self._step_count += 1
        self._operation_id += 1
        successful_final_assay = (
            operation_committed
            and operation_record.is_final_assay
            and runtime_result.kernel_result.measurement_status == "valid"
        )
        campaign_final_assay = successful_final_assay and self.episode_mode == "campaign"
        explicit_stop = (
            self.task_spec is not None
            and self.task_spec.termination_policy == "explicit-stop-or-budget"
        )
        campaign_explicit_stop = (
            explicit_stop
            and operation_committed
            and operation_record.operation_type == "terminate"
            and self.episode_mode == "campaign"
        )
        campaign_completed = campaign_final_assay or campaign_explicit_stop
        campaign_discard = (
            operation_committed and is_campaign_discard and self.episode_mode == "campaign"
        )
        batch_budget_exhausted = bool(
            self._batch_resource_ledger is not None
            and self._batch_resource_ledger.operation_attempts
            >= self._batch_resource_ledger.card.operation_attempt_limit
            and not campaign_completed
            and not campaign_discard
        )
        campaign_batch_closed = campaign_completed or campaign_discard or batch_budget_exhausted
        budget_exhausted = self._step_count >= self.budget
        operation_attempts_exhausted = bool(
            self._campaign_resource_ledger is not None
            and self._campaign_resource_ledger.operation_attempts
            >= self._campaign_resource_ledger.card.operation_attempt_limit
        )
        truncated = (budget_exhausted or operation_attempts_exhausted) and not campaign_batch_closed
        terminated = (
            operation_committed
            and operation_record.operation_type == "terminate"
            and not campaign_explicit_stop
            if explicit_stop
            else successful_final_assay and not campaign_final_assay
        )
        truncated = truncated and not terminated
        self._last_reading = deepcopy(observation)
        # Environment reward is an event-gated public score delta.  Only a
        # successful instrument measurement creates new public information;
        # process actions may retain the last observation for Markov state but
        # must never earn that cached absolute score again.
        reward = 0.0
        if (
            operation_committed
            and operation_record.is_instrument_measurement
            and runtime_result.kernel_result.measurement_status == "valid"
        ):
            previous_score = (
                value_or_default(previous_process.last_observation, "score")
                if previous_process.last_observation
                else 0.0
            )
            reward = value_or_default(observation_values, "score") - previous_score
        self._last_operation_record = operation_record
        info = self._info(operation_record, observation)
        info.update(runtime_info)
        if resource_preflight is not None and resource_outcome_delta is not None:
            info["campaign_resource_preflight"] = resource_preflight.to_dict()
            info["campaign_resource_outcome_delta"] = resource_outcome_delta.to_dict()
        if batch_preflight is not None:
            info["batch_resource_preflight"] = batch_preflight.to_dict()
            info["batch_resource_state"] = self._batch_resource_ledger.snapshot()["state"]
        if campaign_resource_rejected:
            info["constraint_flags"] = {
                **info.get("constraint_flags", {}),
                "campaign_resource_rejected": True,
            }
            info["error_message"] = (
                "Operation cannot be completed with the remaining campaign resources: "
                + ", ".join(resource_rejection_reasons)
            )
        # Operation records are assembled from the retained rollback state,
        # which is constitution-safe by construction.  The report adapter
        # preserves any failed candidate-state check as a public outcome.
        info = annotate_constitution_rollback(info)
        info["observed_reward"] = float(reward)
        info["environment_reward"] = {
            "schema_version": "chemworld-environment-reward-0.2",
            "semantics": "fresh_measurement_score_delta",
            "fresh_measurement": bool(
                operation_committed
                and operation_record.is_instrument_measurement
                and runtime_result.kernel_result.measurement_status == "valid"
            ),
            "cached_observation_rewarded": False,
            "score_delta": float(reward),
        }
        if self.debug_truth:
            info["truth"] = self._state.to_dict(include_hidden=True)
        else:
            info = sanitize_agent_info(info)
        if campaign_batch_closed:
            info["experiment_ended"] = True
            info["experiment_completed"] = bool(campaign_completed)
            info["batch_discarded"] = bool(campaign_discard)
            if self._batch_resource_ledger is not None:
                info["batch_budget_exhausted"] = batch_budget_exhausted
            lifecycle_experiment_index = self._experiment_index + 1
            completed_ordinal = (
                sum(
                    item.get("outcome") == "completed"
                    for item in self._experiment_summaries
                    if isinstance(item, Mapping)
                )
                + 1
                if campaign_completed
                else None
            )
            terminal_summary = {
                "experiment_index": self._experiment_index,
                "lifecycle_experiment_index": lifecycle_experiment_index,
                "experiment_index_base": 1,
                "batch_id": f"batch-{lifecycle_experiment_index:04d}",
                "completed_ordinal": completed_ordinal,
                "terminal_step": self._step_count,
                "outcome": (
                    "completed"
                    if campaign_completed
                    else "operation_budget_exhausted"
                    if batch_budget_exhausted
                    else "discarded"
                ),
                "leaderboard_score": (info["leaderboard_score"] if campaign_final_assay else None),
                "safety_risk": value_or_default(observation_values, "safety_risk"),
                "cost": value_or_default(observation_values, "cost"),
                "final_assay": bool(campaign_final_assay),
                "discard_reason": (str(action.get("reason", "")) if campaign_discard else None),
                "resource_delta": self._batch_resource_delta(
                    self._current_batch_resource_baseline,
                    (
                        self._campaign_resource_ledger.snapshot()["state"]
                        if self._campaign_resource_ledger is not None
                        else None
                    ),
                ),
            }
            sample_outcome = observation.raw_signal.get("sample_outcome")
            if isinstance(sample_outcome, dict):
                terminal_summary["sample_outcome"] = deepcopy(sample_outcome)
            self._experiment_summaries.append(deepcopy(terminal_summary))
            self._experiment_index += 1
            self._campaign_resource_current_vessel_started = False
            info["experiment_summaries"] = deepcopy(self._experiment_summaries)
            info["last_terminal_summary"] = deepcopy(terminal_summary)
            info["next_experiment_index"] = self._experiment_index
            can_start_next, blockers = self._campaign_next_batch_availability()
            if budget_exhausted:
                can_start_next = False
                blockers = ["environment_budget_exhausted"]
            if can_start_next:
                new_state = self._fresh_initial_state()
                from chemworld.foundation.lifecycle import initialize_entities

                physics = (
                    self.compiled_composition.spec.physics
                    if self.compiled_composition is not None
                    else None
                )
                advancer = self.runtime.domain_services.reaction_thermal.advancer
                transition = initialize_entities(
                    self._state,
                    new_state,
                    preserve_vessels=(set() if physics is None else physics.finite_supply_vessels)
                    | advancer.preserved_source_vessels,
                )
                transition = advancer.reconcile_initialization(transition)
                new_state = transition.state
                self._lifecycle_history.append(
                    {
                        "after_operation_id": self._operation_id,
                        "experiment_index": self._experiment_index,
                        **transition.receipt,
                    }
                )
                info["lifecycle_initialization"] = {
                    "scope": "batch",
                    "vessels": sorted(transition.receipt["vessels"]),
                    "preserved_vessels": transition.receipt["preserved_vessels"],
                }
                retained_solvents = self._state.solvent_accounting
                from chemworld.foundation.solvents import SolventInventory

                closed_batch_solvents = SolventInventory()
                if self._state.phases is not None:
                    for phase in self._state.phases.phases.values():
                        closed_batch_solvents = closed_batch_solvents + phase.solvents
                for local in self._state.inactive_vessels.values():
                    if local.phases is not None:
                        for phase in local.phases.phases.values():
                            closed_batch_solvents = closed_batch_solvents + phase.solvents
                self._state = new_state.replace(
                    solvent_accounting=replace(
                        retained_solvents,
                        initial=retained_solvents.initial + new_state.solvent_accounting.initial,
                        removed=retained_solvents.removed + closed_batch_solvents,
                    ),
                )
                self._current_batch_resource_baseline = deepcopy(
                    self._campaign_resource_ledger.snapshot()["state"]
                    if self._campaign_resource_ledger is not None
                    else None
                )
                self._reset_batch_resource_ledger()
                info["next_experiment_ready"] = True
            else:
                info["next_experiment_ready"] = False
                self._campaign_terminal = True
                self._campaign_terminal_reason = (
                    "campaign_resources_exhausted_after_batch_close"
                    if blockers
                    else "campaign_terminal"
                )
                info["campaign_terminal"] = True
                info["campaign_terminal_reason"] = self._campaign_terminal_reason
                info["campaign_terminal_blockers"] = blockers
                terminated = True
            if self._campaign_resource_ledger is not None:
                info["campaign_resources"] = self.public_campaign_resource_state()
        elif (
            self.episode_mode == "campaign"
            and self._campaign_resource_ledger is not None
            and operation_attempts_exhausted
        ):
            self._campaign_terminal = True
            self._campaign_terminal_reason = "operation_attempt_limit_exhausted"
            self._right_censored_open_batch = bool(self._campaign_resource_current_vessel_started)
            info["campaign_terminal"] = True
            info["campaign_terminal_reason"] = self._campaign_terminal_reason
            info["right_censored_open_batch"] = self._right_censored_open_batch
            truncated = True
        elif self._campaign_terminal:
            info["campaign_terminal"] = True
            info["campaign_terminal_reason"] = self._campaign_terminal_reason
        self._done = terminated or truncated
        self._last_info = deepcopy(info)
        return SessionResult(self.observe(), reward, terminated, truncated, info)

    @property
    def _last_observation(self) -> dict[str, float | None]:
        return self.observe().values

    def observe(self) -> Observation:
        """Read the latest committed public packet without sampling or charging."""
        return deepcopy(self._last_reading)

    def lab_status(self) -> dict[str, Any]:
        """Read queues and released offline results without acquisition or RNG draws."""
        from chemworld.runtime.scheduling import public_lab

        return public_lab(self._state)

    def snapshot(self) -> SessionSnapshot:
        """Capture a private, JSON-serializable continuation without acquiring evidence."""
        return capture_session(self)

    @classmethod
    def from_snapshot(cls, snapshot, *, domain_modules=None):
        """Create a session from saved state, using explicitly installed module code."""
        snapshot = (
            snapshot
            if isinstance(snapshot, SessionSnapshot)
            else SessionSnapshot.from_dict(snapshot)
        )
        instance = cls(**deepcopy(snapshot.payload["configuration"]), domain_modules=domain_modules)
        options = (
            {"scenario_id": snapshot.payload["scenario_id"]}
            if instance.compiled_composition is None
            else None
        )
        ChemWorldRuntime.reset(instance, seed=instance.seed, options=options)
        restore_continuation(instance, snapshot)
        return instance

    def restore(self, snapshot: SessionSnapshot) -> None:
        """Restore a compatible session atomically; no history is executed or charged."""
        config = snapshot.payload["configuration"]
        if {k: v for k, v in config.items() if k != "seed"} != {
            k: v for k, v in declaration(self._initialization).items() if k != "seed"
        }:
            raise ValueError("snapshot belongs to a different experimental session configuration")
        candidate = ChemWorldRuntime.from_snapshot(snapshot, domain_modules=self.domain_modules)
        self.__dict__.update(candidate.__dict__)

    def close(self) -> None:
        """No external handles are owned by the in-process physical session."""

    def task_info(self) -> dict[str, Any]:
        return build_task_info(self)

    def material_information_dossier(self) -> dict[str, Any] | None:
        """Return a defensive copy of the public, nominal-only dossier."""

        return deepcopy(self._material_information_dossier)

    def campaign_resource_snapshot(self) -> dict[str, Any] | None:
        """Return the full evaluator-owned, replayable resource ledger.

        This accessor intentionally includes the event history and card
        metadata. Agent-facing reports use :meth:`public_campaign_resource_state`
        instead so their size stays constant as a campaign grows.
        """

        if self._campaign_resource_ledger is None:
            return None
        return deepcopy(self._campaign_resource_ledger.snapshot())

    def public_campaign_resource_state(
        self,
        *,
        include_card: bool = False,
    ) -> dict[str, Any] | None:
        """Return a bounded, hidden-world-safe campaign resource view."""

        if self._campaign_resource_ledger is None:
            return None
        snapshot = self._campaign_resource_ledger.snapshot()
        payload: dict[str, Any] = {
            "schema_version": "chemworld-public-campaign-resource-state-0.1",
            "state": deepcopy(snapshot["state"]),
            "ledger_sha256": snapshot["ledger_sha256"],
            "last_event_id": snapshot["last_event_id"],
            "current_experiment": {
                "experiment_index": self._experiment_index,
                "vessel_started": (self._campaign_resource_current_vessel_started),
            },
            "campaign_terminal": self._campaign_terminal,
            "campaign_terminal_reason": self._campaign_terminal_reason,
            "latest_receipt": deepcopy(self._last_campaign_resource_receipt),
        }
        lifecycle_reserve = self._campaign_lifecycle_reserve()
        if self._batch_resource_ledger is not None:
            payload["current_batch_resources"] = {
                "hard_limits": self._batch_resource_ledger.card.to_dict()["hard_limits"],
                "state": self._batch_resource_ledger.snapshot()["state"],
                "assigned_rounds": self.campaign_resource_card.vessel_start_limit,
                "closed_rounds": self._experiment_index,
                "unused_quota_carries_over": False,
            }
        if lifecycle_reserve is not None:
            payload["lifecycle_reserve"] = lifecycle_reserve
        if include_card:
            card = self.campaign_resource_card
            if card is None:
                raise RuntimeError("campaign resource card is missing")
            payload["card"] = card.to_dict()
        return payload

    def _campaign_lifecycle_reserve(self) -> dict[str, Any] | None:
        """Publish an advisory, non-reserving closeout feasibility projection.

        The campaign pool remains globally shared and preflight never hides a
        future-vessel allocation.  This projection gives an autonomous agent
        the arithmetic it needs to choose whether another discretionary step
        is worth risking the remaining planned batch lifecycle.
        """

        ledger = self._campaign_resource_ledger
        if (
            ledger is None
            or self.task_id != "electrochemical-conversion"
            or self.electrochemical_workflow_mode != ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
        ):
            return None
        state = ledger.snapshot()["state"]
        remaining = state["remaining"]
        remaining_attempts = int(remaining["operation_attempts"])
        remaining_vessels = int(remaining["vessel_starts"])
        current_open = bool(self._campaign_resource_current_vessel_started)
        future_unstarted = remaining_vessels if current_open else max(remaining_vessels - 1, 0)
        current_final_assay_operations = 0
        current_discard_operations = 0
        if current_open:
            baseline = self._current_batch_resource_baseline or {}
            baseline_stocks = baseline.get("stocks_used", {})
            current_stocks = state.get("stocks_used", {})
            if not isinstance(baseline_stocks, Mapping):
                baseline_stocks = {}
            if not isinstance(current_stocks, Mapping):
                current_stocks = {}
            solvent_added = float(current_stocks.get("solvent_L", 0.0)) - float(
                baseline_stocks.get("solvent_L", 0.0)
            )
            reagent_added = float(current_stocks.get("reagent_mol", 0.0)) - float(
                baseline_stocks.get("reagent_mol", 0.0)
            )
            cell = equipment_settings(self._state.equipment, "electrochemical_cell")
            has_setpoint = bool(tuple(cell.get("setpoint_history", ())))
            has_electrolysis = bool(tuple(cell.get("electrolysis_history", ())))
            current_final_assay_operations = sum(
                (
                    int(solvent_added <= self.constitution.tolerance),
                    int(reagent_added <= self.constitution.tolerance),
                    int(not has_setpoint),
                    int(not has_electrolysis),
                    int(not self._state.terminated),
                    1,  # final_assay itself
                )
            )
            current_discard_operations = 1
        elif remaining_vessels > 0:
            current_final_assay_operations = 6
            current_discard_operations = 2

        future_final_assay_reserve = 6 * future_unstarted
        future_discard_reserve = 2 * future_unstarted
        final_assay_floor = current_final_assay_operations + future_final_assay_reserve
        discard_floor = current_discard_operations + future_discard_reserve
        return {
            "schema_version": "chemworld-campaign-lifecycle-reserve-0.1",
            "policy": "advisory_only_agent_controlled_no_hidden_allocation",
            "minimum_fresh_batch_operations": {
                "to_final_assay": 6,
                "to_explicit_discard": 2,
            },
            "current_batch": {
                "open": current_open,
                "minimum_operations_to_final_assay": (current_final_assay_operations),
                "minimum_operations_to_explicit_discard": (current_discard_operations),
            },
            "future_unstarted_batches": future_unstarted,
            "minimum_future_batch_operation_reserve": {
                "for_final_assays": future_final_assay_reserve,
                "for_explicit_discards": future_discard_reserve,
            },
            "recommended_remaining_attempt_floor": {
                "to_final_assay_all_planned_batches": final_assay_floor,
                "to_close_all_planned_batches_with_discards_allowed": (discard_floor),
            },
            "discretionary_attempts_before_final_assay_floor": max(
                remaining_attempts - final_assay_floor,
                0,
            ),
            "remaining_operation_attempts": remaining_attempts,
        }

    def evaluator_provenance(self) -> dict[str, Any]:
        """Return private replay identity for the official evaluator/logger."""

        return build_evaluator_provenance(self)

    def observation_noise_provenance(self) -> dict[str, Any]:
        """Return evaluator-only noise identity for the most recent transition."""

        return deepcopy(self._last_observation_noise_provenance)

    def task_prompt(self) -> dict[str, Any]:
        from chemworld.agent_interface import task_prompt

        return task_prompt(self)

    def available_actions(self, *, include_invalid: bool = False) -> list[dict[str, Any]]:
        from chemworld.agent_interface import available_actions

        return available_actions(self, include_invalid=include_invalid)

    def resource_blocked_actions(self) -> list[dict[str, Any]]:
        from chemworld.agent_interface import resource_blocked_actions

        return resource_blocked_actions(self)

    def action_schema(self, operation: str) -> dict[str, Any]:
        from chemworld.agent_interface import action_schema

        return action_schema(self, operation)

    def validate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        from chemworld.agent_interface import validate_action

        return validate_action(self, action)

    def observation_view(self, mode: str = "tool_json") -> dict[str, Any]:
        from chemworld.agent_interface import observation_view

        return observation_view(self, mode)

    def campaign_state(self) -> dict[str, Any]:
        from chemworld.agent_interface import campaign_state

        return campaign_state(self)

    def constitution_summary(self) -> dict[str, Any]:
        return build_constitution_summary(self)

    def render(self) -> Any:
        return render_env(self)

    @staticmethod
    def _normalize_campaign_resource_card(
        card: Mapping[str, Any] | CampaignResourceCard | None,
    ) -> CampaignResourceCard | None:
        if card is None or isinstance(card, CampaignResourceCard):
            return card
        if not isinstance(card, Mapping):
            raise TypeError(
                "campaign_resource_card must be a mapping, CampaignResourceCard, or None"
            )
        payload = deepcopy(dict(card))
        if "hard_limits" in payload:
            return CampaignResourceCard.from_dict(payload)
        payload.pop("card_sha256", None)
        return CampaignResourceCard(**payload)

    def _reset_campaign_resource_ledger(self) -> None:
        self._campaign_resource_ledger = (
            None
            if self.campaign_resource_card is None
            else CampaignResourceLedger(
                self.campaign_resource_card,
                reagent_charge_molar_multiplier=(
                    self.operation_validator.reagent_charge_molar_multiplier
                    if "add_reagent" in self.allowed_operations
                    else 1.0
                ),
            )
        )
        self._campaign_resource_current_vessel_started = False
        self._last_campaign_resource_receipt = None
        self._current_batch_resource_baseline = (
            None
            if self._campaign_resource_ledger is None
            else deepcopy(self._campaign_resource_ledger.snapshot()["state"])
        )
        self._reset_batch_resource_ledger()

    def _reset_batch_resource_ledger(self) -> None:
        card = self.campaign_resource_card
        self._batch_resource_ledger = (
            CampaignResourceLedger(
                card.batch_card(),
                reagent_charge_molar_multiplier=(
                    self.operation_validator.reagent_charge_molar_multiplier
                    if "add_reagent" in self.allowed_operations
                    else 1.0
                ),
            )
            if card is not None and card.per_batch_limits
            else None
        )

    def _campaign_batch_discard_available(self) -> bool:
        if not self.campaign_controls_enabled:
            return False
        if not self._campaign_resource_current_vessel_started:
            return False
        ledger = self._campaign_resource_ledger
        if ledger is None:
            return False
        # ``step`` charges the operation attempt in resource preflight before
        # validating campaign controls.  Equality therefore means the current
        # discard is consuming the final available attempt, not that the
        # action was submitted after the budget was already exhausted.
        return ledger.operation_attempts <= ledger.card.operation_attempt_limit

    def _campaign_next_batch_availability(self) -> tuple[bool, list[str]]:
        ledger = self._campaign_resource_ledger
        # Legacy campaign tasks predate the explicit G2 resource ledger and
        # remain budget-delimited multi-experiment episodes.  A missing card
        # therefore means "no additional ledger gate", not "end campaign".
        # When a card is present its limits govern every campaign and expose
        # ``discard_batch`` as the explicit closeout for an abandoned vessel.
        if ledger is None:
            return True, []
        state = ledger.snapshot()["state"]
        remaining = state.get("remaining", {})
        blockers: list[str] = []
        if (
            ledger.card.per_batch_limits
            and self._experiment_index >= ledger.card.vessel_start_limit
        ):
            blockers.append("assigned_round_limit")
        if int(remaining.get("operation_attempts", 0)) < 1:
            blockers.append("operation_attempt_limit")
        if int(remaining.get("vessel_starts", 0)) < 1:
            blockers.append("vessel_start_limit")
        if (
            self.task_spec is None or self.task_spec.termination_policy != "explicit-stop-or-budget"
        ) and int(remaining.get("final_assays", 0)) < 1:
            blockers.append("final_assay_limit")
        stocks = remaining.get("stocks", {})
        if self.runtime_task_profile_id == "electrochemical-conversion":
            for stock_id in ("solvent_L", "reagent_mol"):
                if float(stocks.get(stock_id, 0.0)) <= self.constitution.tolerance:
                    blockers.append(f"stock_limit:{stock_id}")
        return not blockers, blockers

    @staticmethod
    def _batch_resource_delta(
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            return {}
        delta: dict[str, Any] = {}
        for key in (
            "operation_attempts",
            "vessel_starts",
            "final_assays",
            "discarded_batches",
            "nonfinal_instrument_uses",
        ):
            delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("instrument_uses", "stocks_used"):
            before_map = before.get(key, {})
            after_map = after.get(key, {})
            if isinstance(before_map, Mapping) and isinstance(after_map, Mapping):
                delta[key] = {
                    str(item): (
                        int(after_map.get(item, 0)) - int(before_map.get(item, 0))
                        if key == "instrument_uses"
                        else float(after_map.get(item, 0.0)) - float(before_map.get(item, 0.0))
                    )
                    for item in sorted(set(before_map) | set(after_map))
                    if (
                        int(after_map.get(item, 0)) - int(before_map.get(item, 0))
                        if key == "instrument_uses"
                        else float(after_map.get(item, 0.0)) - float(before_map.get(item, 0.0))
                    )
                }
        before_report = before.get("report_only", {})
        after_report = after.get("report_only", {})
        if isinstance(before_report, Mapping) and isinstance(after_report, Mapping):
            delta["report_only"] = {
                key: float(after_report.get(key, 0.0)) - float(before_report.get(key, 0.0))
                for key in (
                    "process_time_s",
                    "sample_consumed_L",
                    "physical_cost",
                    "accumulated_risk",
                )
            }
        return delta

    @staticmethod
    def _campaign_resource_report_delta(
        previous_state: WorldState,
        current_state: WorldState,
        observation_values: Mapping[str, float | None],
    ) -> dict[str, float]:
        def positive_delta(previous: float, current: float) -> float:
            delta = float(current) - float(previous)
            return delta if isfinite(delta) and delta > 0.0 else 0.0

        observed_risk = observation_values.get("safety_risk")
        if observed_risk is None or isinstance(observed_risk, bool):
            normalized_observed_risk = 0.0
        else:
            candidate_risk = float(observed_risk)
            normalized_observed_risk = (
                candidate_risk if isfinite(candidate_risk) and candidate_risk >= 0.0 else 0.0
            )
        return {
            "process_time_s": positive_delta(
                previous_state.ledger.time_s,
                current_state.ledger.time_s,
            ),
            "sample_consumed_L": positive_delta(
                previous_state.ledger.sample_consumed_L,
                current_state.ledger.sample_consumed_L,
            ),
            "physical_cost": positive_delta(
                previous_state.ledger.cost,
                current_state.ledger.cost,
            ),
            "accumulated_risk": positive_delta(
                previous_state.ledger.risk,
                current_state.ledger.risk,
            ),
            "observed_risk": normalized_observed_risk,
        }

    def _make_campaign_id(self) -> str:
        # Public episode identity must not be a reversible encoding of the
        # hidden-world seed. Replay identity is carried separately in private
        # evaluator provenance.
        return f"episode-{uuid4().hex}"

    def _load_process_time_policy(self) -> ProcessTimeBudgetPolicy | None:
        if self.compiled_composition is None:
            return None
        raw = self.compiled_composition.spec.task.resources.get("process_time_policy")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("composed task process_time_policy must be an object")
        return ProcessTimeBudgetPolicy.from_dict(raw)

    def _declared_process_time_preflight(
        self,
        action: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        policy = self._declared_process_time_policy
        if policy is None:
            return None
        operation = str(action.get("operation", ""))
        used_before = float(self._state.ledger.time_s)
        proposed = policy.proposed_time_s(action)
        if operation == "measure":
            instrument = self.runtime.modules.instrument_specs().get(str(action.get("instrument")))
            if instrument is not None:
                proposed = instrument.contract.physical_duration_s
        repeat_limit = policy.operation_repeat_limits.get(operation)
        operation_count_before = int(self._declared_operation_counts.get(operation, 0))
        reasons: list[str] = []
        if repeat_limit is not None and operation_count_before + 1 > repeat_limit:
            reasons.append("operation_repeat_limit")
        if used_before + proposed > policy.process_time_limit_s + 1.0e-9:
            reasons.append("process_time_limit")
        return {
            "schema_version": "chemworld-declared-process-time-preflight-0.1",
            "allowed": not reasons,
            "rejection_reasons": sorted(set(reasons)),
            "operation": operation,
            "operation_count_before": operation_count_before,
            "operation_repeat_limit": repeat_limit,
            "used_before_s": used_before,
            "proposed_delta_s": proposed,
            "limit_s": policy.process_time_limit_s,
            "remaining_before_s": max(
                policy.process_time_limit_s - used_before,
                0.0,
            ),
        }

    @staticmethod
    def _declared_process_failure_validation(
        validation: OperationValidation,
        rejection_reasons: tuple[str, ...],
    ) -> OperationValidation:
        resource_reasons = tuple(
            f"declared_process_resource:{reason}" for reason in rejection_reasons
        )
        return replace(
            validation,
            is_valid=False,
            preconditions={
                **validation.preconditions,
                "declared_process_resources_available": False,
            },
            invalid_reasons=tuple(
                dict.fromkeys(
                    (
                        *validation.invalid_reasons,
                        "declared_process_resources_available",
                        *resource_reasons,
                    )
                )
            ),
            cost_penalty=max(validation.cost_penalty, 0.10),
            safety_flags={
                **validation.safety_flags,
                "precondition_failed": True,
                "declared_process_resource_rejected": True,
            },
        )

    def public_declared_process_resource_state(self) -> dict[str, Any] | None:
        """Return bounded public process-time usage and repeat headroom."""

        policy = self._declared_process_time_policy
        if policy is None:
            return None
        used = float(self._state.ledger.time_s)
        return {
            "schema_version": "chemworld-public-declared-process-resources-0.1",
            "policy_id": policy.policy_id,
            "pattern_id": policy.pattern_id,
            "used_s": used,
            "limit_s": policy.process_time_limit_s,
            "remaining_s": max(policy.process_time_limit_s - used, 0.0),
            "operation_counts": dict(self._declared_operation_counts),
            "operation_remaining": {
                operation: max(
                    limit - self._declared_operation_counts.get(operation, 0),
                    0,
                )
                for operation, limit in policy.operation_repeat_limits.items()
            },
            "formula": (
                "timed_stage_max_s + implicit_stage_reserve_s + repeat_allowance_s; "
                "quench/transfer allowances are reserved prospectively"
            ),
        }

    def _make_runtime(self) -> OperationExecutor:
        partition_nominal_pair_contract = (
            INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1
            if self.scoring_contract.contract_id == PARTITION_S0_EXTRACTION_EFFICIENCY_V3
            else None
        )
        return OperationExecutor(
            world=self.world,
            constitution=self.constitution,
            task_spec=self.task_spec,
            compiled_mechanism=self.scenario_instance.compiled_mechanism,
            debug_truth=self.debug_truth,
            partition_nominal_pair_contract=partition_nominal_pair_contract,
            modules=self.domain_modules,
            composition=None
            if self.compiled_composition is None
            else self.compiled_composition.spec,
        )

    def _make_operation_validator(self) -> OperationValidator:
        species_view = MechanismSpeciesView(self.scenario_instance.compiled_mechanism)
        unit_charge = (
            species_view.reagent_charge_amounts(
                self.scenario_instance.initial_state,
                limiting_amount_mol=1.0,
            )
            if "add_reagent" in self.allowed_operations
            else {}
        )
        reagent_charge_molar_multiplier = sum(unit_charge.values())
        return OperationValidator(
            constitution=self.constitution,
            allowed_operations=self.allowed_operations,
            allowed_instruments=self.allowed_instruments,
            task_id=(
                None
                if self.compiled_composition is not None
                and len(self.compiled_composition.spec.vessels) > 1
                else self.runtime_task_profile_id
            ),
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
            target_species=species_view.target_species,
            reagent_charge_molar_multiplier=reagent_charge_molar_multiplier,
            feed_count=len(self.action_codec.feed_ids or species_view.feed_species)
            if "add_component" in self.allowed_operations
            else 0,
            liquid_supply_temperature_K=self.world.environment_temperature_K,
            operation_types=self.action_codec.operation_types,
            action_codec=self.action_codec,
            material_router=self.runtime.domain_services.material_routing,
            authored_field_bounds=(
                {}
                if self.compiled_composition is None
                else self.compiled_composition.compatibility.operation_field_bounds
            ),
        )

    @staticmethod
    def _domain_failure_validation(
        validation: OperationValidation,
        failure_key: str,
    ) -> OperationValidation:
        return replace(
            validation,
            is_valid=False,
            preconditions={**validation.preconditions, failure_key: False},
            invalid_reasons=tuple(dict.fromkeys((*validation.invalid_reasons, failure_key))),
            cost_penalty=max(validation.cost_penalty, 0.10),
            safety_flags={
                **validation.safety_flags,
                "precondition_failed": True,
                failure_key: False,
            },
        )

    @staticmethod
    def _campaign_resource_failure_validation(
        validation: OperationValidation,
        rejection_reasons: tuple[str, ...],
    ) -> OperationValidation:
        resource_reasons = tuple(f"campaign_resource:{reason}" for reason in rejection_reasons)
        return replace(
            validation,
            is_valid=False,
            preconditions={
                **validation.preconditions,
                "campaign_resources_available": False,
            },
            invalid_reasons=tuple(
                dict.fromkeys(
                    (
                        *validation.invalid_reasons,
                        "campaign_resources_available",
                        *resource_reasons,
                    )
                )
            ),
            cost_penalty=max(validation.cost_penalty, 0.10),
            safety_flags={
                **validation.safety_flags,
                "precondition_failed": True,
                "campaign_resource_rejected": True,
            },
        )

    def _make_observation_contract(self) -> TaskObservationContract:
        return TaskObservationContract.from_task(
            success_metrics=(
                self.task_spec.success_metrics if self.task_spec is not None else ("score",)
            ),
            scoring_contract=self.scoring_contract,
            allowed_instruments=tuple(sorted(self.allowed_instruments)),
            instruments=self.constitution.instruments,
            mechanism_observable_mapping=(
                self.scenario_instance.compiled_mechanism.observable_mapping
            ),
        )

    def _fresh_initial_state(self) -> WorldState:
        from chemworld.runtime.solvent_transport import initialize_solvents

        state = deepcopy(self.scenario_instance.initial_state)
        if self.full_process_contract_id is not None:
            state = state.replace(
                metadata={
                    **state.metadata,
                    "full_process_contract_id": self.full_process_contract_id,
                    "full_process_task_id": self.task_id,
                }
            )
        if self.compiled_composition is not None:
            from chemworld.runtime.component_network import initialize_network

            state = initialize_network(state, self.compiled_composition)
        if self.compiled_composition is None or self.compiled_composition.spec.physics is None:
            from chemworld.runtime.disposable_sorbent import install_legacy_media_supplies

            state = install_legacy_media_supplies(state)
        from chemworld.runtime.scheduling import initialize_lab

        state = initialize_lab(
            state, None if self.compiled_composition is None else self.compiled_composition.spec,
            self._rng,
        )
        if state.lab_clock is not None and self.observation_noise_mode != "sequential":
            raise ValueError("shared lab uses its persistent sequential detector stream")
        return initialize_solvents(state)

    def _observation_seed(self, world_seed: int) -> int:
        """Resolve observation noise independently from hidden-world generation.

        The default preserves the historical one-seed behavior. Evaluators may
        override only the observation stream so paired no-change resets retain the
        same hidden physical laws without receiving identical measurement noise.
        """

        if self.observation_seed_override is None:
            return int(world_seed)
        return self.observation_seed_override

    def _info(self, operation_record: OperationRecord, observation: Any) -> dict[str, Any]:
        return build_step_info(self, operation_record, observation)
