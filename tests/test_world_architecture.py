from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import chemworld  # noqa: F401
from chemworld.cli import main
from chemworld.data.datasets import dataset_card, export_dataset, flatten_record
from chemworld.data.logging import load_jsonl
from chemworld.foundation.state import (
    PhaseLedger,
    PhaseRecord,
    ProcessLedger,
    SpeciesLedger,
    VesselLedger,
    VesselRecord,
    WorldState,
    equipment_settings,
    instrument_completed,
    instrument_equipment_id,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.operation_validator import OperationValidator
from chemworld.physchem.mechanism_library import get_mechanism_card, list_mechanism_cards
from chemworld.runtime.domain_services import (
    ChemWorldDomainServices,
    DomainServiceRegistry,
    make_chemworld_constitution,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult, RuntimeContext
from chemworld.runtime.kernel_registry import (
    OperationKernelRegistry,
    ServiceOperationKernel,
)
from chemworld.runtime.mechanisms import compile_mechanism, compile_mechanism_for_scenario
from chemworld.runtime.observation_services import ChemWorldObservationKernel
from chemworld.runtime.profiles import TaskRuntimeProfile
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.transactions import TransactionManager
from chemworld.schemas import (
    ACTION_SCHEMA,
    MANIFEST_SCHEMA,
    OBSERVATION_SCHEMA,
    RECIPE_SCHEMA,
    SCENARIO_SCHEMA,
    TASK_SCHEMA,
    TRAJECTORY_SCHEMA,
    load_schema_file,
    validate_action_schema,
    validate_recipe_schema,
)
from chemworld.tasks import get_task, list_tasks
from chemworld.validation import validate_action
from chemworld.world import (
    get_scenario_card,
    instrument_contracts,
    list_scenarios,
    load_chemworld_parameters,
    world_law_spec,
)
from chemworld.world.observation_kernel import raw_signal
from chemworld.world.operations import OPERATION_TYPES
from chemworld.world.parameters import WORLD_FAMILY_VERSION
from chemworld.world.phase_kernel import partition_split
from chemworld.world.reaction_kernel import (
    integrate_compiled_reaction_ode,
)
from chemworld.world.reaction_reference import integrate_seven_slot_reference_ode
from chemworld.world.recipes import compile_recipe, validate_recipe
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario
from chemworld.world.separation_kernel import downstream_truth_values
from chemworld.world.state_factory import initial_chemworld_state
from chemworld.world.thermal_kernel import pressure_and_risk


def test_native_execution_sources_do_not_depend_on_assert_statements() -> None:
    findings: list[str] = []
    # Protect the native execution boundary, not unrelated study scripts or
    # agent-side proof assertions. Optimized-Python behavior is tested separately.
    paths = [
        path
        for name in ("foundation", "runtime", "world", "physchem", "envs")
        for path in (Path("src/chemworld") / name).rglob("*.py")
    ]
    paths.extend(
        Path("src/chemworld") / name for name in ("operation_validator.py", "agent_interface.py")
    )
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(
            f"{path.as_posix()}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )
    assert findings == []


def test_world_law_contains_professional_contracts() -> None:
    spec = world_law_spec().to_dict()
    assert spec["law_version"] == WORLD_FAMILY_VERSION
    assert spec["ontology_registry"]["substance_registry_policy"] == ("scenario_compiled_mechanism")
    assert "reaction" in spec["module_versions"]
    assert "thermal_energy_balance" in spec["transition_kernel_registry"]
    assert "crystallization" in spec["transition_kernel_registry"]
    assert "distillation" in spec["transition_kernel_registry"]
    assert "continuous_flow" in spec["transition_kernel_registry"]
    assert "electrochemistry" in spec["transition_kernel_registry"]
    assert "hplc" in spec["instrument_registry"]
    assert "material_conservation" in spec["constitution_rules"]
    assert spec["backend"]["backend_id"] == "semi_mechanistic"
    module_ids = {module["module_id"] for module in spec["ontology_registry"]["modules"]}
    assert {
        "crystallization",
        "distillation",
        "continuous_flow",
        "electrochemistry",
    } <= module_ids
    assert spec["module_versions"]["reaction"] == "0.3"
    assert spec["module_versions"]["observation"] == "0.4"


def test_runtime_ontology_is_mechanism_owned_not_fixed_species_default() -> None:
    ontology_source = Path("src/chemworld/world/ontology.py").read_text(encoding="utf-8")
    state_factory_source = Path("src/chemworld/world/state_factory.py").read_text(encoding="utf-8")
    reference_source = Path("src/chemworld/world/reaction_reference.py").read_text(encoding="utf-8")

    assert "SPECIES =" not in ontology_source
    assert '"A"' not in ontology_source
    assert '"P"' not in ontology_source
    assert "from chemworld.world.ontology import SPECIES" not in state_factory_source
    assert "species_ids or SPECIES" not in state_factory_source
    assert "REFERENCE_REACTION_SPECIES" in reference_source


def test_env_constitution_substances_follow_compiled_mechanism() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=3)
    env.reset(seed=3)
    compiled_species = set(env.unwrapped.scenario_instance.compiled_mechanism.species_index)
    constitution_species = set(env.unwrapped.constitution.substances)

    assert constitution_species == compiled_species


def test_world_layer_does_not_import_batch_core() -> None:
    world_dir = Path("src/chemworld/world")
    offenders = [
        path
        for path in world_dir.glob("*.py")
        if "chemworld.core.batch_reactor" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_env_and_runtime_do_not_import_removed_batch_runtime() -> None:
    roots = (Path("src/chemworld/envs"), Path("src/chemworld/runtime"))
    offenders = [
        path
        for root in roots
        for path in root.glob("*.py")
        if "chemworld.core.batch_reactor" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_runtime_env_and_world_use_world_action_catalog() -> None:
    assert not Path("src/chemworld/core/actions.py").exists()


def test_core_package_removed_from_source_and_tutorials() -> None:
    roots = (
        Path("src/chemworld"),
        Path("notebooks/tutorials"),
    )
    offenders: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".ipynb"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "chemworld.core." in text or "from chemworld.core" in text:
                offenders.append(path)
    core_sources = list(Path("src/chemworld/core").glob("*.py"))

    assert core_sources == []
    assert offenders == []


def test_tutorial_helpers_do_not_import_removed_batch_runtime() -> None:
    tutorial_helpers = Path("notebooks/tutorials/tutorial_utils.py").read_text(encoding="utf-8")

    assert "chemworld.core.batch_reactor" not in tutorial_helpers
    assert "from chemworld.world.recipes import recipe_to_event_sequence" in tutorial_helpers


def test_runtime_does_not_use_legacy_species_constants() -> None:
    roots = (Path("src/chemworld/envs"), Path("src/chemworld/runtime"), Path("src/chemworld/eval"))
    legacy_usage: list[tuple[Path, int, str]] = []

    for root in roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.name
                    if isinstance(node, ast.alias)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else ""
                )
                # Historical format IDs and explicitly selected low-order models
                # are valid. Fixed reference-species identities are not runtime defaults.
                if name in {
                    "LEGACY_SPECIES",
                    "LEGACY_SPECIES_INDEX",
                    "LEGACY_REACTION_SPECIES",
                    "REFERENCE_REACTION_SPECIES",
                }:
                    legacy_usage.append((path, node.lineno, name))

    assert [(path.as_posix(), lineno, name) for path, lineno, name in legacy_usage] == []


def test_downstream_world_modules_do_not_use_legacy_species_fallbacks() -> None:
    checked_paths = (
        Path("src/chemworld/world/species_roles.py"),
        Path("src/chemworld/world/separation_kernel.py"),
    )
    offenders = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in checked_paths
        if "LEGACY_" in path.read_text(encoding="utf-8")
    }
    assert offenders == {}


def test_chemworld_env_delegates_process_operation_dispatch_to_runtime() -> None:
    source = Path("src/chemworld/envs/chemworld_env.py").read_text(encoding="utf-8")

    assert "self.execute(action)" in source
    assert "def execute(" not in source
    assert ".domain_services" not in source
    assert "operation_record.operation_type" not in source
    for operation in OPERATION_TYPES:
        assert f'operation_record.operation_type == "{operation}"' not in source
        assert f'action["operation"] == "{operation}"' not in source
        assert f"action['operation'] == '{operation}'" not in source


def test_runtime_observation_service_is_separate_from_state_changing_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    observation_services = Path("src/chemworld/runtime/observation_services.py").read_text(
        encoding="utf-8"
    )

    assert "class ChemWorldObservationKernel" not in domain_services
    assert "class ChemWorldObservationKernel" in observation_services
    assert "score_observation" not in domain_services


def test_runtime_operation_recorder_is_separate_from_state_changing_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    record_services = Path("src/chemworld/runtime/record_services.py").read_text(encoding="utf-8")

    assert "def _record" not in domain_services
    assert "class ChemWorldOperationRecorder" in record_services
    assert "material delta allowed or phase-ledger conserved" not in domain_services


def test_runtime_reaction_thermal_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    reaction_thermal_services = Path(
        "src/chemworld/runtime/reaction_thermal_services.py"
    ).read_text(encoding="utf-8")

    assert "def _integrate" not in domain_services
    assert "def _with_risk_and_pressure" not in domain_services
    assert "integrate_reaction_ode" not in domain_services
    assert "pressure_and_risk" not in domain_services
    assert "class ChemWorldReactionThermalServices" in reaction_thermal_services
    assert "integrate_compiled_reaction_ode" in reaction_thermal_services
    assert "pressure_and_risk" in reaction_thermal_services


def test_runtime_reaction_thermal_requires_compiled_mechanism_without_fallback_map() -> None:
    reaction_thermal_services = Path(
        "src/chemworld/runtime/reaction_thermal_services.py"
    ).read_text(encoding="utf-8")
    species_services = Path("src/chemworld/runtime/species.py").read_text(encoding="utf-8")
    reaction_kernel = Path("src/chemworld/world/reaction_kernel.py").read_text(encoding="utf-8")

    assert "reaction_backend_species_map" not in reaction_thermal_services
    assert "def reaction_backend_species_map" not in species_services
    assert "Reaction advancement requires a compiled mechanism" in (reaction_thermal_services)
    assert "compiled_mechanism is required" in reaction_kernel


def test_runtime_does_not_import_seven_slot_reference_fixture() -> None:
    roots = (Path("src/chemworld/envs"), Path("src/chemworld/runtime"), Path("src/chemworld/eval"))
    offenders = [
        path.as_posix()
        for root in roots
        for path in root.glob("*.py")
        if "reaction_reference" in path.read_text(encoding="utf-8")
        or "integrate_seven_slot_reference_ode" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_runtime_phase_separation_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    phase_separation_services = Path(
        "src/chemworld/runtime/phase_separation_services.py"
    ).read_text(encoding="utf-8")
    phase_ledger_services = Path("src/chemworld/runtime/phase_ledger_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _phase_ledger" not in domain_services
    assert "def _mix_phases" not in domain_services
    assert "def _separate_phase" not in domain_services
    assert "partition_split" not in domain_services
    assert "downstream_truth_values" not in phase_separation_services
    assert "class ChemWorldPhaseSeparationServices" in phase_separation_services
    assert "partition_split" in phase_separation_services
    assert "class ChemWorldPhaseLedgerServices" in phase_ledger_services
    assert "downstream_truth_values" in phase_ledger_services


def test_chemworld_env_spaces_are_separate_from_control_loop() -> None:
    env_source = Path("src/chemworld/envs/chemworld_env.py").read_text(encoding="utf-8")
    space_source = Path("src/chemworld/envs/spaces.py").read_text(encoding="utf-8")

    assert "from gymnasium import spaces" not in env_source
    assert "spaces.Box" not in env_source
    assert "spaces.Discrete" not in env_source
    assert "make_action_space(modules=self.domain_modules)" in env_source
    assert "make_observation_space()" in env_source
    assert "def make_action_space" in space_source
    assert "class NullableScalarBox" in space_source


def test_chemworld_env_reports_are_separate_from_control_loop() -> None:
    env_source = Path("src/chemworld/envs/chemworld_env.py").read_text(encoding="utf-8")
    report_source = Path("src/chemworld/runtime/reports.py").read_text(encoding="utf-8")

    assert "def build_task_info" in report_source
    assert "def build_step_info" in report_source
    assert "def render_env" in report_source
    assert "instrument_contracts" not in env_source
    assert "semi_mechanistic_backend_spec" not in env_source
    assert "safety_cost_from_flags" not in env_source
    session_source = Path("src/chemworld/runtime/engine.py").read_text(encoding="utf-8")
    assert "return build_task_info(self)" in session_source
    assert "return build_step_info(self, operation_record, observation)" in session_source
    assert "return render_env(self)" in session_source


def test_runtime_mechanism_manifest_and_validation_are_separate_from_compiler_facade() -> None:
    mechanisms_source = Path("src/chemworld/runtime/mechanisms.py").read_text(encoding="utf-8")
    manifest_source = Path("src/chemworld/runtime/mechanism_manifest.py").read_text(
        encoding="utf-8"
    )
    validation_source = Path("src/chemworld/runtime/mechanism_validation.py").read_text(
        encoding="utf-8"
    )

    assert "class CompiledMechanism" not in mechanisms_source
    assert "class MechanismManifest" not in mechanisms_source
    assert "class MechanismValidationReport" not in mechanisms_source
    assert "def validate_mechanism_file" not in mechanisms_source
    assert "def mechanism_hash" not in mechanisms_source
    assert "def compile_mechanism" in mechanisms_source
    assert "class CompiledMechanism" in manifest_source
    assert "class MechanismManifest" in manifest_source
    assert "class MechanismValidationReport" in manifest_source
    assert "def validate_mechanism_file" in validation_source
    assert "def validate_compiled_role_contract" in validation_source


def test_runtime_kernel_profile_contracts_and_registry_are_separate() -> None:
    profiles_source = Path("src/chemworld/runtime/profiles.py").read_text(encoding="utf-8")
    contracts_source = Path("src/chemworld/runtime/kernel_contracts.py").read_text(encoding="utf-8")
    registry_source = Path("src/chemworld/runtime/kernel_registry.py").read_text(encoding="utf-8")

    assert not Path("src/chemworld/runtime/kernels.py").exists()
    assert "class TaskRuntimeProfile" in profiles_source
    assert "def profile_hash" in profiles_source
    assert "class RuntimeContext" in contracts_source
    assert "class KernelResult" in contracts_source
    assert "class OperationKernel" in contracts_source
    assert "class ServiceOperationKernel" in registry_source
    assert "class OperationKernelRegistry" in registry_source
    assert "def affected_ledgers" in registry_source


def test_runtime_domain_service_registry_and_constitution_factory_are_separate() -> None:
    domain_source = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    registry_source = Path("src/chemworld/runtime/domain_service_registry.py").read_text(
        encoding="utf-8"
    )
    factory_source = Path("src/chemworld/runtime/constitution_factory.py").read_text(
        encoding="utf-8"
    )

    assert "class DomainServiceContract" not in domain_source
    assert "class DomainServiceRegistry" not in domain_source
    assert "def make_chemworld_constitution" not in domain_source
    assert "class ChemWorldDomainServices" in domain_source
    assert "class DomainServiceContract" in registry_source
    assert "class DomainServiceRegistry" in registry_source
    assert "def make_chemworld_constitution" in factory_source


def test_foundation_constitution_reports_state_checks_and_preconditions_are_separate() -> None:
    constitution_source = Path("src/chemworld/foundation/constitution.py").read_text(
        encoding="utf-8"
    )
    reports_source = Path("src/chemworld/foundation/constitution_reports.py").read_text(
        encoding="utf-8"
    )
    state_checks_source = Path("src/chemworld/foundation/constitution_state_checks.py").read_text(
        encoding="utf-8"
    )
    preconditions_source = Path("src/chemworld/foundation/constitution_preconditions.py").read_text(
        encoding="utf-8"
    )

    assert "class CheckResult" not in constitution_source
    assert "class ConstitutionReport" not in constitution_source
    assert "def check_operation_preconditions" not in constitution_source
    assert "def check_typed_ledgers" not in constitution_source
    assert "class PhysicalConstitution" in constitution_source
    assert "class CheckResult" in reports_source
    assert "class ConstitutionReport" in reports_source
    assert "def check_typed_ledgers" in state_checks_source
    assert "def check_operation_preconditions" in preconditions_source


def test_foundation_state_ledgers_and_helpers_are_separate_from_state_facade() -> None:
    state_source = Path("src/chemworld/foundation/state.py").read_text(encoding="utf-8")
    ledgers_source = Path("src/chemworld/foundation/state_ledgers.py").read_text(encoding="utf-8")
    helpers_source = Path("src/chemworld/foundation/state_helpers.py").read_text(encoding="utf-8")

    assert "class SpeciesLedger" not in state_source
    assert "class PhaseLedger" not in state_source
    assert "class EquipmentLedger" not in state_source
    assert "def equipment_settings" not in state_source
    assert "def scale_phase_ledger" not in state_source
    assert "class WorldState" in state_source
    assert "class SpeciesLedger" in ledgers_source
    assert "class PhaseLedger" in ledgers_source
    assert "class EquipmentLedger" in ledgers_source
    assert "def process_with_metrics" in ledgers_source
    assert "def equipment_settings" in helpers_source
    assert "def scale_phase_ledger" in helpers_source


def test_runtime_electrochemical_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    electrochemical_services = Path("src/chemworld/runtime/electrochemical_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _set_potential" not in domain_services
    assert "def _electrolyze" not in domain_services
    assert "run_electrolysis" not in domain_services
    assert "ElectrodeReactionSpec" not in domain_services
    assert "class ChemWorldElectrochemicalServices" in electrochemical_services
    assert "run_electrolysis" in electrochemical_services
    assert "ElectrodeReactionSpec" in electrochemical_services


def test_runtime_instrument_cost_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    instrument_cost_services = Path("src/chemworld/runtime/instrument_cost_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _apply_measurement_cost" not in domain_services
    assert "instrument_name" not in domain_services
    assert "final_assay_done" not in domain_services
    assert "class ChemWorldInstrumentCostServices" in instrument_cost_services
    assert "instrument_name" in instrument_cost_services
    assert "upsert_equipment_record" in instrument_cost_services
    assert "instrument_equipment_id" in instrument_cost_services
    assert "sample_consumed_L=state.ledger.sample_consumed_L + volume" in instrument_cost_services


def test_runtime_crystallization_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    crystallization_services = Path("src/chemworld/runtime/crystallization_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _seed_crystals" not in domain_services
    assert "def _cool_crystallize" not in domain_services
    assert "def _filter_crystals" not in domain_services
    assert "crystal_seeded" not in domain_services
    assert "crystal_product_mol" not in domain_services
    assert "class ChemWorldCrystallizationServices" in crystallization_services
    assert "def seed_crystals" in crystallization_services
    assert "def cool_crystallize" in crystallization_services
    assert "def filter_crystals" in crystallization_services
    assert "upsert_equipment_record" in crystallization_services
    assert "equipment_settings" in crystallization_services
    assert "crystal_product_mol" not in crystallization_services


def test_runtime_distillation_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    distillation_services = Path("src/chemworld/runtime/distillation_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _distill" not in domain_services
    assert "def _collect_fraction" not in domain_services
    assert "vle_shortcut_distillation" not in domain_services
    assert "distillate_product_mol" not in domain_services
    assert "class ChemWorldDistillationServices" in distillation_services
    assert "def distill" in distillation_services
    assert "def collect_fraction" in distillation_services
    assert "run_duty_limited_distillation" in distillation_services
    assert "distillate_product_mol" not in distillation_services


def test_runtime_flow_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    flow_services = Path("src/chemworld/runtime/flow_services.py").read_text(encoding="utf-8")

    assert "def _set_flow_rate" not in domain_services
    assert "def _run_flow" not in domain_services
    assert "flow_rate_mL_min" not in domain_services
    assert "flow_conversion" not in domain_services
    assert "class ChemWorldFlowServices" in flow_services
    assert "def set_flow_rate" in flow_services
    assert "def run_flow" in flow_services
    assert "flow_rate_mL_min" in flow_services
    assert "flow_conversion" in flow_services
    assert "PFRModel" in flow_services
    assert "PFRGeometrySpec" in flow_services
    assert "reaction_thermal.integrate" not in flow_services


def test_runtime_primitive_service_is_separate_from_domain_services() -> None:
    domain_services = Path("src/chemworld/runtime/domain_services.py").read_text(encoding="utf-8")
    primitive_services = Path("src/chemworld/runtime/primitive_services.py").read_text(
        encoding="utf-8"
    )

    assert "def _add_reagent" not in domain_services
    assert "def _add_solvent" not in domain_services
    assert "def _add_catalyst" not in domain_services
    assert "def _sample" not in domain_services
    assert "def _quench" not in domain_services
    assert "def _evaporate" not in domain_services
    assert "def _penalize_invalid" not in domain_services
    assert "CATALYSTS" not in domain_services
    assert "SOLVENTS" not in domain_services
    assert "np.clip" not in domain_services
    assert "class ChemWorldPrimitiveOperationServices" in primitive_services
    assert "def add_reagent" in primitive_services
    assert "def add_solvent" in primitive_services
    assert "def add_catalyst" in primitive_services
    assert "def sample" in primitive_services
    assert "def quench" in primitive_services
    assert "def evaporate" in primitive_services
    assert "def penalize_invalid" in primitive_services


def test_reaction_network_specs_are_separate_from_engine() -> None:
    reaction_network = Path("src/chemworld/physchem/reaction_network.py").read_text(
        encoding="utf-8"
    )
    specs = Path("src/chemworld/physchem/reaction_network_specs.py").read_text(encoding="utf-8")

    assert "class SpeciesSpec" not in reaction_network
    assert "class RateLawSpec" not in reaction_network
    assert "class ReactionSpec" not in reaction_network
    assert "def parse_reaction_equation" not in reaction_network
    assert "def _merge_side" not in reaction_network
    assert "_TERM_RE" not in reaction_network
    assert "def species_from_dict" not in reaction_network
    assert "def reaction_from_dict" not in reaction_network
    assert "reaction_network_specs" in reaction_network
    assert "class SpeciesSpec" in specs
    assert "class RateLawSpec" in specs
    assert "class ReactionSpec" in specs
    assert "def parse_reaction_equation" in specs
    assert "def species_from_dict" in specs
    assert "def reaction_from_dict" in specs


def test_reaction_rate_laws_are_separate_from_network_engine() -> None:
    reaction_network = Path("src/chemworld/physchem/reaction_network.py").read_text(
        encoding="utf-8"
    )
    rate_laws = Path("src/chemworld/physchem/reaction_rate_laws.py").read_text(encoding="utf-8")

    assert "def mass_action_rate" not in reaction_network
    assert "def _mass_action_rate" not in reaction_network
    assert "def arrhenius_k" not in reaction_network
    assert "def _arrhenius_k" not in reaction_network
    assert "def reverse_rate_constant(" not in reaction_network
    assert "def _reverse_rate_constant" not in reaction_network
    assert "def reverse_params" not in reaction_network
    assert "def _reverse_params" not in reaction_network
    assert "def positive_reaction_parameter" not in reaction_network
    assert "def _positive_reaction_parameter" not in reaction_network
    assert "def with_reaction_parameter" not in reaction_network
    assert "def _with_reaction_parameter" not in reaction_network
    assert "def reaction_by_id" not in reaction_network
    assert "def _reaction_by_id" not in reaction_network
    assert "def float_param" not in reaction_network
    assert "def _float_param" not in reaction_network
    assert "reaction_rate_laws" in reaction_network
    assert "def evaluate_rate_law" in rate_laws
    assert "def mass_action_rate" in rate_laws
    assert "def arrhenius_k" in rate_laws
    assert "def reverse_rate_constant" in rate_laws
    assert "def positive_reaction_parameter" in rate_laws
    assert "def with_reaction_parameter" in rate_laws


def test_reaction_reference_cases_are_separate_from_network_engine() -> None:
    reaction_network = Path("src/chemworld/physchem/reaction_network.py").read_text(
        encoding="utf-8"
    )
    reference_cases = Path("src/chemworld/physchem/reaction_reference_cases.py").read_text(
        encoding="utf-8"
    )

    assert "class ReactionODEReferenceCase" not in reaction_network
    assert "class ReactionODEReferenceResult" not in reaction_network
    assert "def cantera_comparable_reaction_cases" not in reaction_network
    assert "def integrate_reaction_ode_reference_case" not in reaction_network
    assert "def evaluate_reaction_ode_reference_case" not in reaction_network
    assert "reaction_reference_cases" in reaction_network
    assert "class ReactionODEReferenceCase" in reference_cases
    assert "class ReactionODEReferenceResult" in reference_cases
    assert "def cantera_comparable_reaction_cases" in reference_cases
    assert "def integrate_reaction_ode_reference_case" in reference_cases
    assert "def evaluate_reaction_ode_reference_case" in reference_cases


def test_runtime_profile_requires_current_task_kernels_only() -> None:
    task = get_task("reaction-to-assay")
    profile = TaskRuntimeProfile.from_task(task)
    registry = OperationKernelRegistry.default()

    assert profile.world_law_id == WORLD_FAMILY_VERSION
    assert "measure" in profile.required_kernels
    assert "add_extractant" not in profile.required_kernels
    assert "instrument_cost" in profile.required_domain_services
    assert "phase_separation" not in profile.required_domain_services
    assert "observation" in profile.required_capabilities
    assert all(registry.has(operation) for operation in profile.required_kernels)
    assert profile.is_operation_allowed("measure")
    assert not profile.is_operation_allowed("add_extractant")
    assert profile.is_domain_service_required("instrument_cost")
    assert not profile.is_domain_service_required("phase_separation")


def test_mechanism_compiler_supports_non_fixed_species_networks() -> None:
    compiled = compile_mechanism_for_scenario("electrochemical-conversion")

    assert compiled.mechanism_id == "electrochemical_conversion"
    assert "Ox" in compiled.species_index
    assert "Red" in compiled.species_index
    assert compiled.score_spec.reactant_species == "Ox"
    assert compiled.score_spec.product_species == "Red"
    assert compiled.mechanism_hash
    assert len(compiled.stoichiometric_matrix) == len(compiled.species_index)
    assert all(
        len(row) == len(compiled.network.reactions) for row in compiled.stoichiometric_matrix
    )


def test_mechanism_compiler_validates_runtime_role_contracts() -> None:
    for card in list_mechanism_cards():
        compiled = compile_mechanism(card)
        assert compiled.score_spec.initial_limiting_species
        assert compiled.score_spec.target_species

    for task in list_tasks():
        compiled = compile_mechanism_for_scenario(task.scenario_id)
        assert compiled.score_spec.impurity_species

    card = get_mechanism_card("simple_batch_reaction")
    bad_target = replace(card, target_species=("MissingProduct",))
    with pytest.raises(ValueError, match="target species not in mechanism"):
        compile_mechanism(bad_target)

    bad_initial = replace(card, initial_amounts_mol={"A": 0.0})
    with pytest.raises(ValueError, match="at least one positive species"):
        compile_mechanism(bad_initial)

    bad_impurity = replace(card, impurity_species=())
    with pytest.raises(ValueError, match="impurity_species cannot be empty"):
        compile_mechanism(bad_impurity, require_runtime_roles=True)


def test_runtime_species_view_uses_mechanism_roles_without_fixed_names() -> None:
    compiled = compile_mechanism_for_scenario("electrochemical-conversion")
    view = MechanismSpeciesView(compiled)
    state = WorldState(
        species_amounts={
            "Ox": 0.006,
            "Red": 0.003,
            "SideRed": 0.001,
        },
        volume_L=0.05,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="electrochemical_cell",
        species=SpeciesLedger(initial_amounts_mol={"Ox": 0.010}),
    )

    truth = view.truth_values(state)

    assert view.reactant_species(state) == "Ox"
    assert view.primary_target_species == "Red"
    assert view.primary_impurity_species == "SideRed"
    assert view.target_amount(state) == 0.003
    assert view.impurity_amount(state) == 0.001
    assert truth["yield"] == pytest.approx(0.3)
    assert truth["conversion"] == pytest.approx(0.4)
    assert truth["selectivity"] == pytest.approx(0.75)


def test_scenario_initial_state_uses_compiled_mechanism_species() -> None:
    scenario = get_scenario("electrochemical-conversion")
    instance = DefaultScenarioGenerator().generate(scenario, seed=0)

    assert set(instance.initial_state.species_amounts) == set(
        instance.compiled_mechanism.species_index
    )
    assert "A" not in instance.initial_state.species_amounts
    assert set(instance.initial_state.species.initial_amounts_mol) == set(
        instance.compiled_mechanism.initial_amount_policy
    )
    assert all(
        amount_mol == pytest.approx(0.0)
        for amount_mol in instance.initial_state.species.initial_amounts_mol.values()
    )
    assert "initial_Ox_mol" not in instance.initial_state.metadata
    assert "initial_reactant_mol" not in instance.initial_state.metadata
    assert instance.initial_state.metadata["mechanism_id"] == "electrochemical_conversion"
    assert (
        instance.initial_state.metadata["mechanism_hash"]
        == instance.compiled_mechanism.mechanism_hash
    )


def test_compiled_runtime_reaction_does_not_require_fixed_species_state() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0)
    try:
        env.reset(seed=0)
        state = env.unwrapped._state
        assert "A" not in state.species_amounts
        assert "P" not in state.species_amounts
        assert "Acid" in state.species_amounts
        assert "Ester" in state.species_amounts

        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
        ):
            env.step(action)

        heated = env.unwrapped._state
        assert "A" not in heated.species_amounts
        assert "P" not in heated.species_amounts
        assert "Cat_active" not in heated.species_amounts
        assert heated.species_amounts["Acid"] < 0.010
        assert heated.species_amounts["Ester"] > 0.0
        assert heated.phases is not None
        assert heated.phases.total_amounts_mol() == pytest.approx(heated.species_amounts)
    finally:
        env.close()


def test_reagent_charge_uses_mechanism_initial_amount_policy() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.028, "solvent": 2})
        _, _, _, _, info = env.step({"operation": "add_reagent", "amount_mol": 0.010})
        state = env.unwrapped._state

        assert info["transaction_status"] == "committed"
        assert state.species_amounts["Acid"] == pytest.approx(0.010)
        assert state.species_amounts["Alcohol"] == pytest.approx(0.0125)
        assert state.species.initial_amounts_mol["Acid"] == pytest.approx(0.010)
        assert "initial_Acid_mol" not in state.metadata
        assert "initial_reactant_mol" not in state.metadata
        assert env.unwrapped.constitution.check_state(state).passed
    finally:
        env.close()


def test_reagent_charge_does_not_implicitly_add_catalyst() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 0})
        _, _, _, _, reagent_info = env.step({"operation": "add_reagent", "amount_mol": 0.010})
        reagent_state = env.unwrapped._state

        assert reagent_info["transaction_status"] == "committed"
        assert reagent_state.species_amounts["A"] == pytest.approx(0.010)
        assert reagent_state.species_amounts["Cat_active"] == pytest.approx(0.0)

        _, _, _, _, catalyst_info = env.step(
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.00020,
                "catalyst": 0,
            }
        )
        catalyst_state = env.unwrapped._state
        assert catalyst_info["transaction_status"] == "committed"
        assert catalyst_state.species_amounts["Cat_active"] == pytest.approx(0.00020)
    finally:
        env.close()


def test_uncharged_catalyst_category_does_not_modify_reaction_rates() -> None:
    env = gym.make("ChemWorld", task_id="reaction-safety-constrained", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 0})
        env.step({"operation": "add_reagent", "amount_mol": 0.015})
        base = env.unwrapped
        state = base._state
        from chemworld.world.reaction_kernel import _hidden_reaction_modifier

        stirring_factor = 0.70 + 0.30 * (1.0 - np.exp(-675.0 / 420.0))
        factors = [
            _hidden_reaction_modifier(
                base.world,
                catalyst=catalyst,
                solvent=0,
                reaction_index=1,
                stirring_factor=stirring_factor,
                state=state,
            )
            for catalyst in range(4)
        ]

        assert factors == pytest.approx([factors[0]] * 4)
        assert factors[0] == pytest.approx(
            float(base.world.solvent_effects[0, 1]) * stirring_factor
        )
    finally:
        env.close()


def test_zero_dose_catalyst_categories_produce_identical_reaction_trajectories() -> None:
    terminal_states = []
    for catalyst in (0, 3):
        env = gym.make("ChemWorld", task_id="reaction-safety-constrained", seed=0)
        try:
            env.reset(seed=0)
            for action in (
                {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
                {"operation": "add_reagent", "amount_mol": 0.015},
            ):
                _observation, _reward, _terminated, _truncated, info = env.step(action)
                assert info["transaction_status"] == "committed"
            base = env.unwrapped
            state = base._state
            base._state = state.replace(
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id="batch_reactor",
                    equipment_type="batch_reactor",
                    attached_vessel_id=state.vessel_id,
                    status="configured",
                    settings={
                        **equipment_settings(state.equipment, "batch_reactor"),
                        "catalyst": catalyst,
                        "catalyst_amount_mol": 0.0,
                    },
                )
            )
            _observation, _reward, _terminated, _truncated, info = env.step(
                {
                    "operation": "heat",
                    "target_temperature_K": 385.0,
                    "duration_s": 1_500.0,
                    "stirring_speed_rpm": 675.0,
                }
            )
            assert info["transaction_status"] == "committed"
            terminal_states.append(env.unwrapped._state)
        finally:
            env.close()

    left, right = terminal_states
    assert left.species_amounts == pytest.approx(right.species_amounts)
    assert left.temperature_K == pytest.approx(right.temperature_K)
    assert left.pressure_Pa == pytest.approx(right.pressure_Pa)
    assert left.ledger.to_dict() == pytest.approx(right.ledger.to_dict())


def test_domain_services_apply_mechanism_roles_for_reagent_and_electrolysis() -> None:
    compiled = compile_mechanism_for_scenario("electrochemical-conversion")
    services = ChemWorldDomainServices(
        load_chemworld_parameters("public-dev", seed=1),
        make_chemworld_constitution(),
        compiled,
    )
    state = WorldState(
        species_amounts={"Ox": 0.0, "Red": 0.0, "SideRed": 0.0},
        volume_L=0.05,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="electrochemical_cell",
    ).replace(
        equipment=upsert_equipment_record(
            None,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id="electrochemical_cell",
            status="configured",
            settings={"solvent": 0, "catalyst": 0},
        )
    )

    charged, add_record = services.apply_operation(
        state,
        {"operation": "add_reagent", "amount_mol": 0.010},
    )
    configured, potential_record = services.apply_operation(
        charged,
        {
            "operation": "set_potential",
            "potential_V": 1.35,
            "current_mA": 80.0,
            "electrolyte_profile": 1,
        },
    )
    converted, electro_record = services.apply_operation(
        configured,
        {"operation": "electrolyze", "duration_s": 900.0},
    )

    assert add_record.preconditions["not_terminated"]
    assert potential_record.preconditions["not_terminated"]
    assert charged.species_amounts["Ox"] == pytest.approx(0.010)
    assert "A" not in charged.species_amounts
    assert charged.species.initial_amounts_mol["Ox"] == pytest.approx(0.010)
    assert "potential_V" not in configured.metadata
    assert equipment_settings(configured.equipment, "electrochemical_cell")["potential_V"] == 1.35
    assert (
        equipment_settings(configured.equipment, "electrochemical_cell")[
            "electrolyte_conductivity_S_m"
        ]
        > 0.0
    )
    assert converted.species_amounts["Ox"] < configured.species_amounts["Ox"]
    assert converted.species_amounts["Red"] > configured.species_amounts["Red"]
    assert converted.species_amounts["SideRed"] >= configured.species_amounts["SideRed"]
    assert abs(electro_record.state_delta_summary["actual_current_A"]) > 0.0
    assert "interfacial_potential_V" in electro_record.state_delta_summary
    assert "ohmic_loss_J" in electro_record.state_delta_summary


def test_observation_kernel_scores_non_fixed_mechanism_species() -> None:
    compiled = compile_mechanism_for_scenario("electrochemical-conversion")
    kernel = ChemWorldObservationKernel(
        make_chemworld_constitution(),
        objective="balanced",
        compiled_mechanism=compiled,
    )
    state = WorldState(
        species_amounts={
            "Ox": 0.004,
            "Red": 0.004,
            "SideRed": 0.002,
        },
        volume_L=0.05,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="electrochemical_cell",
        species=SpeciesLedger(initial_amounts_mol={"Ox": 0.010}),
    )

    truth = kernel._truth_values(state)

    assert truth["yield"] == pytest.approx(0.4)
    assert truth["conversion"] == pytest.approx(0.6)
    assert truth["byproduct_signal"] == pytest.approx(0.2)
    assert truth["degradation_warning"] == pytest.approx(0.0)


def test_env_runtime_v2_info_contains_kernel_transaction_and_mechanism() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        _, reset_info = env.reset(seed=0)
        _, _, _, _, step_info = env.step(
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}
        )

        provenance = env.unwrapped.evaluator_provenance()
        assert provenance["mechanism_id"] == "simple_batch_reaction"
        assert provenance["mechanism_hash"]
        assert reset_info["mechanism_summary"]["instance_identity_visible"] is False
        assert reset_info["mechanism_summary"]["topology_summary_visible"] is False
        assert "mechanism_manifest" not in reset_info
        assert "reactions" not in reset_info
        assert step_info["kernel_id"] == "chemworld.operation.add_solvent"
        assert step_info["transaction_status"] == "committed"
        assert step_info["rollback_reason"] is None
        assert "process" in step_info["affected_ledgers"]
        assert step_info["world_events"][0]["event_type"] == "operation_applied"
        assert step_info["world_events"][0]["payload"]["domain_service_id"] == (
            "primitive_operations"
        )
        runtime = env.unwrapped.runtime.to_dict()
        assert "instrument_cost" in runtime["profile"]["required_domain_services"]
        assert "phase_separation" not in runtime["profile"]["required_domain_services"]
        assert runtime["mechanism_summary"]["mechanism_id"] == "simple_batch_reaction"
        assert "compiled_mechanism" not in runtime
        assert runtime["domain_services"]["operation_service_map"]["heat"] == ("reaction_thermal")
        assert runtime["domain_services"]["operation_service_map"]["measure"] == ("instrument_cost")
    finally:
        env.close()


def test_domain_service_registry_covers_operations_once() -> None:
    registry = DomainServiceRegistry.default()
    registry.validate_operation_coverage()
    payload = registry.to_dict()
    operation_service_map = payload["operation_service_map"]

    from chemworld.world.domain_modules import DomainModuleRegistry

    declared = DomainModuleRegistry.default().operation_specs()
    assert set(operation_service_map) == set(declared)
    assert set(OPERATION_TYPES) <= set(operation_service_map)
    assert sum(len(contract.operations) for contract in registry.contracts) == len(declared)
    assert payload["services"]["phase_separation"]["operations"] == [
        "add_phase",
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
    ]
    assert registry.service_id_for_operation("electrolyze") == "electrochemistry"


def test_domain_service_registry_validates_task_profile() -> None:
    profile = TaskRuntimeProfile.from_task(get_task("reaction-to-purification"))
    assay_profile = TaskRuntimeProfile.from_task(get_task("reaction-to-assay"))
    registry = DomainServiceRegistry.default()

    registry.validate_profile(profile)
    registry.validate_operation_coverage(operations=assay_profile.required_kernels)
    assert registry.service_ids_for_operations(profile.required_kernels) == (
        profile.required_domain_services
    )
    assert "phase_separation" in profile.required_domain_services
    assert "separation" in profile.required_capabilities

    broken_registry = DomainServiceRegistry(
        tuple(
            replace(
                contract,
                operations=tuple(
                    operation for operation in contract.operations if operation != "add_extractant"
                ),
            )
            if contract.service_id == "phase_separation"
            else contract
            for contract in registry.contracts
        )
    )
    broken_registry.validate_contract_integrity()
    broken_registry.validate_profile(assay_profile)
    broken_registry.validate_operation_coverage(operations=assay_profile.required_kernels)
    with pytest.raises(ValueError, match="Missing domain service operation coverage"):
        broken_registry.validate_profile(profile)
    with pytest.raises(ValueError, match="Invalid domain service registry operation coverage"):
        broken_registry.validate_operation_coverage()


def test_domain_services_runtime_construction_is_task_scoped() -> None:
    registry = DomainServiceRegistry.default()
    assay_profile = TaskRuntimeProfile.from_task(get_task("reaction-to-assay"))
    purification_profile = TaskRuntimeProfile.from_task(get_task("reaction-to-purification"))
    assay_only_registry = DomainServiceRegistry(
        tuple(
            replace(
                contract,
                operations=tuple(
                    operation for operation in contract.operations if operation != "add_extractant"
                ),
            )
            if contract.service_id == "phase_separation"
            else contract
            for contract in registry.contracts
        )
    )
    services = ChemWorldDomainServices(
        load_chemworld_parameters("public-dev", seed=0),
        make_chemworld_constitution(),
        compile_mechanism_for_scenario("reaction-to-assay"),
        service_registry=assay_only_registry,
    )

    services.validate_profile(assay_profile)
    with pytest.raises(ValueError, match="Missing domain service operation coverage"):
        services.validate_profile(purification_profile)


def test_operation_kernel_registry_validates_profile_capabilities() -> None:
    profile = TaskRuntimeProfile.from_task(get_task("reaction-to-assay"))
    broken_registry = OperationKernelRegistry(
        [
            ServiceOperationKernel(operation_type=operation, module="general")
            for operation in OPERATION_TYPES
        ]
    )

    with pytest.raises(ValueError, match="Missing required operation kernel capabilities"):
        broken_registry.validate_profile(profile)


def test_service_kernel_operation_record_matches_rollback_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = compile_mechanism_for_scenario("reaction-to-assay")
    scenario = get_scenario("reaction-to-assay")
    instance = DefaultScenarioGenerator().generate(scenario, seed=0)
    constitution = make_chemworld_constitution()
    services = ChemWorldDomainServices(
        load_chemworld_parameters("public-dev", seed=0),
        constitution,
        compiled,
    )
    context = RuntimeContext(
        task_spec=None,
        profile=TaskRuntimeProfile.from_task(None),
        compiled_mechanism=compiled,
        domain_services=services,
        transaction_manager=TransactionManager(constitution),
    )
    before = instance.initial_state
    action = {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}

    def apply_invalid_candidate(
        state: WorldState,
        candidate_action: dict[str, object],
    ) -> ModelProviderResult:
        candidate = state.replace(volume_L=999.0)
        return ModelProviderResult(
            outputs={"preconditions": {"not_terminated": True}},
            diagnostics={}, state=candidate,
        )

    monkeypatch.setattr(services, "evaluate_operation", apply_invalid_candidate)

    result = ServiceOperationKernel("add_solvent", "reaction").apply(
        before,
        action,
        context,
    )

    assert result.transaction_status == "rolled_back"
    assert result.rollback_reason == "constitution_failed"
    assert result.state.volume_L == pytest.approx(before.volume_L)
    assert result.operation_record.state_delta_summary["delta_volume_L"] == pytest.approx(0.0)
    assert result.state_delta_summary["delta_cost"] == pytest.approx(
        result.state.ledger.cost - before.ledger.cost
    )
    assert result.state_delta_summary == result.operation_record.state_delta_summary
    assert result.patches[-1].patch_type == "rollback_penalty"
    assert result.patches[-1].affected_ledgers == ("process",)
    assert any(event.event_type == "transaction_rollback" for event in result.events)


def test_transaction_rollback_penalty_preserves_process_evidence_cache() -> None:
    initial = initial_chemworld_state()
    state = initial.replace(
        ledger=initial.ledger.with_updates(
            time_s=12.0,
            cost=0.4,
            risk=0.2,
            sample_consumed_L=0.001,
        ),
        process=ProcessLedger(
            time_s=12.0,
            cost=0.4,
            risk=0.2,
            sample_consumed_L=0.001,
            waste_L=0.003,
            metrics={"purity": 0.81},
            last_observation={"score": 0.73, "purity": 0.81},
            last_observed_mask={"score": True, "purity": True},
        ),
    )

    result = TransactionManager(make_chemworld_constitution()).rollback(
        state=state,
        operation_type="measure",
        rollback_reason="constitution_failed",
        failed_checks=("injected_failure",),
    )

    assert result.transaction_status == "rolled_back"
    assert result.state.process is not None
    assert result.state.process.time_s == pytest.approx(12.0)
    assert result.state.process.sample_consumed_L == pytest.approx(0.001)
    assert result.state.process.waste_L == pytest.approx(0.003)
    assert result.state.process.metrics == {"purity": pytest.approx(0.81)}
    assert result.state.process.last_observation == {
        "score": pytest.approx(0.73),
        "purity": pytest.approx(0.81),
    }
    assert result.state.process.last_observed_mask == {
        "score": True,
        "purity": True,
    }
    assert result.state.process.cost == pytest.approx(state.process.cost + 0.03)
    assert result.state.process.risk == pytest.approx(state.process.risk + 0.08)


def test_typed_phase_ledger_tracks_state_replacements() -> None:
    state = WorldState(
        species_amounts={"A": 1.0, "P": 0.0},
        volume_L=0.01,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="reactor",
    )
    updated = state.replace(species_amounts={"A": 0.2, "P": 0.8})

    assert updated.phases is not None
    assert updated.phases.total_amounts_mol() == {"A": 0.2, "P": 0.8}
    assert updated.process is not None
    assert updated.process.time_s == updated.ledger.time_s


def test_downstream_truth_uses_mechanism_role_species_not_fixed_slots() -> None:
    state = WorldState(
        species_amounts={"reactant_Q": 0.0, "target_X": 0.006, "impurity_Y": 0.001},
        volume_L=0.030,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="reactor",
        process=process_with_metrics(
            ProcessLedger(),
            pre_separation_product_mol=0.006,
            solvent_loss=0.03,
        ),
        phases=PhaseLedger(
            {
                "organic": PhaseRecord(
                    phase_id="organic",
                    vessel_id="reactor",
                    phase_type="organic",
                    volume_L=0.020,
                    species_amounts_mol={"target_X": 0.0045, "impurity_Y": 0.0002},
                    selected=True,
                ),
                "aqueous": PhaseRecord(
                    phase_id="aqueous",
                    vessel_id="reactor",
                    phase_type="aqueous",
                    volume_L=0.010,
                    species_amounts_mol={"target_X": 0.0015, "impurity_Y": 0.0008},
                    selected=False,
                ),
            }
        ),
    )

    truth = downstream_truth_values(
        state,
        product_amount_mol=0.006,
        impurity_amount_mol=0.001,
        initial_product_mol=0.006,
        target_species=("target_X",),
        impurity_species=("impurity_Y",),
    )

    assert truth["product_in_organic"] == pytest.approx(0.75)
    assert truth["product_in_aqueous"] == pytest.approx(0.25)
    assert truth["recovery"] == pytest.approx(0.75)
    assert truth["purity"] == pytest.approx(0.0045 / 0.0047)
    assert truth["solvent_loss"] == pytest.approx(0.03)
    assert truth["process_mass_balance_error"] == pytest.approx(0.0)


def test_downstream_truth_ignores_phase_material_metadata_fallback() -> None:
    state = WorldState(
        species_amounts={"target_X": 0.0, "impurity_Y": 0.0},
        volume_L=0.030,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="reactor",
        phases=PhaseLedger(
            {
                "organic": PhaseRecord(
                    phase_id="organic",
                    vessel_id="reactor",
                    phase_type="organic",
                    volume_L=0.020,
                    species_amounts_mol={},
                    selected=True,
                    metadata={"product_mol": 0.009, "impurity_mol": 0.001},
                )
            }
        ),
    )

    truth = downstream_truth_values(state, initial_product_mol=0.010)

    assert truth["product_in_organic"] == pytest.approx(0.0)
    assert truth["purity"] == pytest.approx(0.0)
    assert truth["recovery"] == pytest.approx(0.0)


def test_runtime_phase_separation_uses_typed_phase_ledger_as_primary_state() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    actions = (
        {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
        {
            "operation": "heat",
            "target_temperature_K": 385.0,
            "duration_s": 1500.0,
            "stirring_speed_rpm": 720.0,
        },
        {"operation": "wait", "duration_s": 900.0, "stirring_speed_rpm": 720.0},
        {"operation": "quench"},
        {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
        {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
        {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
        {"operation": "settle", "duration_s": 420.0},
        {"operation": "separate_phase", "target_phase": "organic"},
    )
    try:
        env.reset(seed=0)
        for action in actions:
            before = env.unwrapped._state
            _, _, _, _, info = env.step(action)
            assert not info["constraint_flags"]["precondition_failed"], action
            assert info["transaction_status"] == "committed"

        state = env.unwrapped._state
        assert "phase_ledger" not in state.metadata
        assert "phase_system" not in state.metadata
        assert "phase_settled" not in state.metadata
        assert "selected_phase" not in state.metadata
        assert state.phases is not None
        assert set(before.phases.phases) == {"organic", "aqueous"}
        assert set(state.phases.phases) == {"organic"}
        assert state.phases.phases["organic"].selected is True
        assert state.phases.phases["organic"].settled is True
        assert before.phases.phases["aqueous"].settled is True
        outlet = state.metadata["removed_phase_inventory_history"][-1]
        assert outlet["operation"] == "separate_phase"
        removed = outlet["inventories"]["aqueous"]
        assert removed["species_amounts_mol"] == pytest.approx(
            {
                key: value
                for key, value in before.phases.phases["aqueous"].species_amounts_mol.items()
                if value > 0
            }
        )
        assert state.process.waste_L - before.process.waste_L == pytest.approx(removed["volume_L"])
        assert {
            key: state.species_amounts.get(key, 0) + removed["species_amounts_mol"].get(key, 0)
            for key in before.species_amounts
        } == pytest.approx(before.species_amounts)
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert state.metadata["extraction_model_id"] == ("chemworld_stability_aware_lle_vnext")
        assert state.metadata["extraction_converged"] is True
        assert state.metadata["extraction_material_balance_error_mol"] < 1.0e-10
        assert state.vessels is not None
        assert set(state.vessels.vessels[state.vessel_id].phase_ids) == set(state.phases.phases)
        assert env.unwrapped.constitution.check_state(state).passed

        before_wash = state
        _, _, _, _, wash_info = env.step({"operation": "wash", "wash_volume_L": 0.008})
        state = env.unwrapped._state
        assert wash_info["transaction_status"] == "committed"
        assert state.metadata["wash_model_id"] == ("chemworld_stability_aware_lle_vnext")
        assert state.metadata["wash_converged"] is True
        assert state.metadata["wash_material_balance_error_mol"] < 1.0e-10
        assert state.phases is not None
        assert set(state.phases.phases) == {"organic"}
        wash_outlet = state.metadata["removed_phase_inventory_history"][-1]
        assert wash_outlet["operation"] == "wash"
        wash_removed = wash_outlet["inventories"]["wash_aqueous"]
        assert state.process.waste_L - before_wash.process.waste_L == pytest.approx(
            wash_removed["volume_L"]
        )
        assert state.volume_L + wash_removed["volume_L"] == pytest.approx(
            before_wash.volume_L + 0.008
        )
        assert {
            key: state.species_amounts.get(key, 0) + wash_removed["species_amounts_mol"].get(key, 0)
            for key in before_wash.species_amounts
        } == pytest.approx(before_wash.species_amounts)
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert env.unwrapped.constitution.check_state(state).passed

        env.step({"operation": "terminate"})
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "final_assay"})
        assert info["transaction_status"] == "committed"
        state = env.unwrapped._state
        assert state.phases is not None
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
    finally:
        env.close()


def test_constitution_rejects_primary_phase_status_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "phase_system": True,
            "phase_settled": True,
            "selected_phase": "organic",
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(check.name == "metadata_no_primary_phase_status" for check in report.failures())


def test_constitution_rejects_primary_vessel_bound_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "max_volume_L": 0.02,
            "max_temperature_K": 330.0,
            "max_pressure_Pa": 150_000.0,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(check.name == "metadata_no_primary_vessel_bounds" for check in report.failures())


def test_typed_vessel_bounds_drive_constitution_and_validator() -> None:
    vessel = VesselRecord(
        vessel_id="batch_reactor",
        vessel_type="micro_batch_reactor",
        max_volume_L=0.020,
        max_temperature_K=330.0,
        max_pressure_Pa=150_000.0,
        phase_ids=("reactor_liquid",),
        temperature_K=320.0,
        pressure_Pa=101_325.0,
    )
    state = initial_chemworld_state().replace(
        volume_L=0.019,
        temperature_K=320.0,
        vessels=VesselLedger({"batch_reactor": vessel}),
    )
    constitution = make_chemworld_constitution()
    validator = OperationValidator(
        constitution=constitution,
        allowed_operations={"add_solvent", "heat"},
    )

    assert state.vessels.vessels["batch_reactor"].max_volume_L == pytest.approx(0.020)
    assert state.vessels.vessels["batch_reactor"].max_temperature_K == pytest.approx(330.0)
    assert constitution.check_state(state).passed

    too_large = state.replace(volume_L=0.021)
    too_hot = state.replace(temperature_K=331.0)
    large_report = constitution.check_state(too_large)
    hot_report = constitution.check_state(too_hot)

    assert not large_report.passed
    assert any(check.name == "vessel_volume_bound" for check in large_report.failures())
    assert not hot_report.passed
    assert any(check.name == "vessel_temperature_bound" for check in hot_report.failures())

    solvent_result = validator.validate(
        {"operation": "add_solvent", "volume_L": 0.005, "solvent": 1},
        state,
    )
    heat_result = validator.validate(
        {
            "operation": "heat",
            "target_temperature_K": 335.0,
            "duration_s": 60.0,
            "stirring_speed_rpm": 600.0,
        },
        state,
    )

    assert "payload_bounds:total_volume_L" in solvent_result.invalid_reasons
    assert "payload_bounds:target_temperature_K" in heat_result.invalid_reasons


def test_constitution_rejects_primary_instrument_status_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "final_assay_done": True,
            "final_assay_time_s": 0.0,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(check.name == "metadata_no_primary_instrument_status" for check in report.failures())


def test_constitution_rejects_primary_crystallizer_seed_status_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "crystal_seeded": True,
            "crystal_seed_mass_g": 0.006,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(
        check.name == "metadata_no_primary_crystallizer_seed_status" for check in report.failures()
    )


def test_constitution_rejects_primary_crystallization_output_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "crystallization_active": True,
            "crystal_product_mol": 0.002,
            "crystal_impurity_mol": 0.0001,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(
        check.name == "metadata_no_primary_crystallization_output" for check in report.failures()
    )


def test_constitution_rejects_primary_distillation_output_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "distillation_active": True,
            "distillate_product_mol": 0.002,
            "distillate_impurity_mol": 0.0001,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(
        check.name == "metadata_no_primary_distillation_output" for check in report.failures()
    )


def test_constitution_rejects_primary_downstream_operation_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "crystals_filtered": True,
            "distillation_model": "vle_shortcut_distillation",
            "distillation_kernel": {"stage_efficiency": 0.62},
            "fraction_collected": True,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(
        check.name == "metadata_no_primary_downstream_operation_status"
        for check in report.failures()
    )


def test_constitution_rejects_primary_process_metric_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "last_observation": {"yield": 0.4},
            "last_observed_mask": {"yield": True},
            "pre_separation_product_mol": 0.006,
            "crystal_yield": 0.6,
            "distillate_purity": 0.9,
            "solvent_loss": 0.2,
            "flow_conversion": 0.7,
            "electrochemical_selectivity": 0.8,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(check.name == "metadata_no_primary_process_metrics" for check in report.failures())


def test_constitution_rejects_phase_local_process_metric_metadata() -> None:
    state = initial_chemworld_state().replace(
        phases=PhaseLedger(
            {
                "organic": PhaseRecord(
                    phase_id="organic",
                    vessel_id="batch_reactor",
                    phase_type="organic",
                    volume_L=0.010,
                    species_amounts_mol={},
                    metadata={
                        "product_mol": 0.002,
                        "impurity_mol": 0.0001,
                        "solvent_loss": 0.04,
                    },
                )
            }
        )
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(
        check.name == "phase_metadata_no_primary_material_or_process_state:organic"
        for check in report.failures()
    )


def test_constitution_rejects_primary_initial_amount_metadata() -> None:
    state = initial_chemworld_state().replace(
        metadata={
            **initial_chemworld_state().metadata,
            "initial_A_mol": 0.010,
            "initial_reactant_mol": 0.010,
        }
    )
    report = make_chemworld_constitution().check_state(state)

    assert not report.passed
    assert any(check.name == "metadata_no_primary_initial_amounts" for check in report.failures())


def test_runtime_observation_cache_uses_typed_process_ledger() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=4)
    try:
        env.reset(seed=4)
        env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 1})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        observation, _, _, _, measure_info = env.step(
            {"operation": "measure", "instrument": "hplc"}
        )
        measured_yield = float(observation["yield"][0])
        state_after_measure = env.unwrapped._state

        assert measure_info["transaction_status"] == "committed"
        assert "last_observation" not in state_after_measure.metadata
        assert "last_observed_mask" not in state_after_measure.metadata
        assert state_after_measure.process.last_observation["yield"] == pytest.approx(
            measured_yield
        )
        assert state_after_measure.process.last_observed_mask["yield"] is True
        assert env.unwrapped.constitution.check_state(state_after_measure).passed

        followup_observation, _, _, _, wait_info = env.step(
            {
                "operation": "wait",
                "duration_s": 30.0,
                "stirring_speed_rpm": 600.0,
            }
        )
        state_after_wait = env.unwrapped._state

        assert wait_info["transaction_status"] == "committed"
        assert float(followup_observation["yield"][0]) == pytest.approx(measured_yield)
        assert "last_observation" not in state_after_wait.metadata
        assert "last_observed_mask" not in state_after_wait.metadata
        assert state_after_wait.process.last_observation["yield"] == pytest.approx(measured_yield)
        assert env.unwrapped.constitution.check_state(state_after_wait).passed
    finally:
        env.close()


def test_runtime_flow_and_electrochemical_setup_use_typed_equipment_ledger() -> None:
    flow_env = gym.make("ChemWorld", task_id="flow-reaction-optimization", seed=0)
    electro_env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        flow_env.reset(seed=0)
        flow_env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 2})
        flow_env.step({"operation": "add_reagent", "amount_mol": 0.010})
        _, _, _, _, flow_info = flow_env.step(
            {"operation": "set_flow_rate", "flow_rate_mL_min": 1.2, "residence_time_s": 900.0}
        )
        flow_state = flow_env.unwrapped._state
        flow_settings = equipment_settings(flow_state.equipment, "flow_reactor")
        assert flow_info["transaction_status"] == "committed"
        assert "equipment" in flow_info["affected_ledgers"]
        assert "flow_rate_mL_min" not in flow_state.metadata
        assert "residence_time_s" not in flow_state.metadata
        assert flow_settings == {
            "flow_rate_mL_min": 1.2,
            "residence_time_s": 900.0,
            "requested_residence_time_s": 900.0,
            "minimum_run_duration_s": 900.0,
        }
        assert flow_env.unwrapped.constitution.check_state(flow_state).passed
        _, _, _, _, flow_run_info = flow_env.step(
            {"operation": "run_flow", "target_temperature_K": 382.0, "duration_s": 1800.0}
        )
        flow_state = flow_env.unwrapped._state
        assert "process" in flow_run_info["affected_ledgers"]
        assert "flow_conversion" not in flow_state.metadata
        assert "flow_campaign_time_s" not in flow_state.metadata
        assert flow_state.process is not None
        assert flow_state.process.metrics["flow_conversion"] >= 0.0
        assert flow_state.process.metrics["flow_campaign_time_s"] == pytest.approx(1800.0)
        flow_settings = equipment_settings(flow_state.equipment, "flow_reactor")
        assert flow_settings["flow_model_id"] == "pfr"
        assert flow_settings["runtime_adapter_id"] == ("chemworld_geometry_resolved_pfr_v1")
        assert flow_settings["pressure_drop_Pa"] >= 0.0
        assert flow_settings["reynolds_number"] > 0.0
        assert flow_settings["material_balance_error_mol"] < 1.0e-9
        assert flow_settings["solver_diagnostic"]["success"] is True
        assert flow_env.unwrapped.constitution.check_state(flow_state).passed

        electro_env.reset(seed=0)
        electro_env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 0})
        electro_env.step({"operation": "add_reagent", "amount_mol": 0.010})
        _, _, _, _, electro_info = electro_env.step(
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
            }
        )
        electro_state = electro_env.unwrapped._state
        electro_settings = equipment_settings(electro_state.equipment, "electrochemical_cell")
        assert electro_info["transaction_status"] == "committed"
        assert "equipment" in electro_info["affected_ledgers"]
        assert "potential_V" not in electro_state.metadata
        assert "current_mA" not in electro_state.metadata
        assert electro_settings["potential_V"] == pytest.approx(1.15)
        assert electro_settings["current_mA"] == pytest.approx(75.0)
        assert electro_settings["electrolyte_conductivity_S_m"] > 0.0
        assert electro_settings["electrode_gap_m"] > 0.0
        assert electro_settings["electrode_area_m2"] > 0.0
        assert electro_settings["contact_resistance_ohm"] >= 0.0
        assert electro_settings["voltage_window_V"] > 0.0
        assert electro_env.unwrapped.constitution.check_state(electro_state).passed
        _, _, _, _, electrolysis_info = electro_env.step(
            {"operation": "electrolyze", "duration_s": 1800.0}
        )
        electro_state = electro_env.unwrapped._state
        assert "process" in electrolysis_info["affected_ledgers"]
        assert "electrochemical_selectivity" not in electro_state.metadata
        assert "energy_efficiency" not in electro_state.metadata
        assert electro_state.process is not None
        assert 0.0 <= electro_state.process.metrics["electrochemical_selectivity"] <= 1.0
        assert 0.0 <= electro_state.process.metrics["energy_efficiency"] <= 1.0
        assert electro_state.process.metrics["charge_C"] > 0.0
        assert electro_state.process.metrics["total_resistance_ohm"] > 0.0
        assert electro_state.process.metrics["ohmic_loss_J"] >= 0.0
        assert electro_env.unwrapped.constitution.check_state(electro_state).passed
    finally:
        flow_env.close()
        electro_env.close()


def test_runtime_reactor_settings_use_typed_equipment_ledger() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        env.reset(seed=0)
        _, _, _, _, solvent_info = env.step(
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2}
        )
        state = env.unwrapped._state
        settings = equipment_settings(state.equipment, "batch_reactor")
        assert "equipment" in solvent_info["affected_ledgers"]
        assert settings["solvent"] == 2
        assert "solvent" not in state.metadata

        _, _, _, _, catalyst_info = env.step(
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1}
        )
        state = env.unwrapped._state
        settings = equipment_settings(state.equipment, "batch_reactor")
        assert "equipment" in catalyst_info["affected_ledgers"]
        assert settings["catalyst"] == 1
        assert "catalyst" not in state.metadata

        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        _, _, _, _, heat_info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": 380.0,
                "duration_s": 600.0,
                "stirring_speed_rpm": 740.0,
            }
        )
        state = env.unwrapped._state
        settings = equipment_settings(state.equipment, "batch_reactor")
        assert "equipment" in heat_info["affected_ledgers"]
        assert settings["stirring_speed_rpm"] == pytest.approx(740.0)
        assert "stirring_speed_rpm" not in state.metadata
        assert env.unwrapped.constitution.check_state(state).passed

        env.step({"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012})
        env.step({"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018})
        extractor_settings = equipment_settings(
            env.unwrapped._state.equipment,
            "liquid_liquid_extractor",
        )
        assert extractor_settings["extractant"] == 3
        assert extractor_settings["extractant_volume_L"] == pytest.approx(0.018)
        _, _, _, _, mix_info = env.step(
            {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0}
        )
        state = env.unwrapped._state
        mixer_settings = equipment_settings(state.equipment, "phase_mixer")
        assert "equipment" in mix_info["affected_ledgers"]
        assert mixer_settings["stirring_speed_rpm"] == pytest.approx(850.0)
        assert "stirring_speed_rpm" not in state.metadata
        assert env.unwrapped.constitution.check_state(state).passed
    finally:
        env.close()


def test_runtime_final_assay_status_uses_typed_instrument_equipment() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "terminate"},
        ):
            env.step(action)
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "final_assay"})
        state = env.unwrapped._state
        instrument_id = instrument_equipment_id("final_assay")
        settings = equipment_settings(state.equipment, instrument_id)

        assert info["transaction_status"] == "committed"
        assert "equipment" in info["affected_ledgers"]
        assert instrument_completed(state.equipment, "final_assay")
        assert settings["instrument_id"] == "final_assay"
        assert settings["use_count"] == 1
        assert "final_assay_done" not in state.metadata
        assert "final_assay_time_s" not in state.metadata

        preconditions = env.unwrapped.constitution.check_preconditions(
            "measure",
            state,
            {"instrument": "final_assay"},
        )
        assert not preconditions["measure_final_not_repeated"]
    finally:
        env.close()


def test_runtime_crystallizer_seed_status_uses_typed_equipment_ledger() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.024, "solvent": 1})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        env.step({"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1})
        env.step(
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            }
        )
        env.step({"operation": "measure", "instrument": "hplc"})
        _, _, _, _, seed_info = env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        state = env.unwrapped._state
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")

        assert seed_info["transaction_status"] == "committed"
        assert seed_info["constraint_flags"]["constitution_failed"] is False
        assert "equipment" in seed_info["affected_ledgers"]
        assert crystallizer_settings["crystal_seeded"] is True
        assert crystallizer_settings["crystal_seed_mass_g"] == pytest.approx(0.006)
        assert "crystal_seeded" not in state.metadata
        assert "crystal_seed_mass_g" not in state.metadata
        assert env.unwrapped.constitution.check_state(state).passed

        first_seed_target_mol = crystallizer_settings["seed_target_mol"]
        first_solid_target_mol = state.phases.phases["solid"].species_amounts_mol["P"]
        _, _, _, _, second_seed_info = env.step(
            {"operation": "seed_crystals", "seed_mass_g": 0.004}
        )
        state = env.unwrapped._state
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")
        assert second_seed_info["transaction_status"] == "committed"
        assert crystallizer_settings["crystal_seed_mass_g"] == pytest.approx(0.010)
        assert crystallizer_settings["seed_charge_count"] == 2
        assert crystallizer_settings["seed_target_mol"] > first_seed_target_mol
        assert state.phases.phases["solid"].species_amounts_mol["P"] > first_solid_target_mol
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert env.unwrapped.constitution.check_state(state).passed

        _, _, _, _, cool_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1200.0,
            }
        )
        state = env.unwrapped._state
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")
        assert cool_info["transaction_status"] == "committed"
        execution = env.unwrapped.runtime.domain_services.crystallization.last_provider_execution
        assert crystallizer_settings["crystallization_model_id"] == execution["model_id"]
        assert execution["success"] is True
        assert execution["model_id"] == "finite-crystal-cohorts-temperature-v1"
        crystal = env.unwrapped.runtime.domain_services.crystallization.thermal.crystal
        assert crystal.model(state).solubility.model_id == (
            "runtime_vanthoff_material_solubility_v2"
        )
        assert crystallizer_settings["material_balance_error_mol"] < 1.0e-10
        assert crystallizer_settings["csd_d10_m"] > 0.0
        assert crystallizer_settings["csd_d50_m"] >= crystallizer_settings["csd_d10_m"]
        assert crystallizer_settings["csd_d90_m"] >= crystallizer_settings["csd_d50_m"]
        assert state.phases is not None
        assert {"solid", "mother_liquor"} <= set(state.phases.phases)
        solid_before_filter = state.phases.phases["solid"]
        solid_product_before_filter = solid_before_filter.species_amounts_mol["P"]
        solid_impurity_before_filter = solid_before_filter.species_amounts_mol["B"]
        assert solid_product_before_filter > 0.0
        # Cooling retains the slurry selection; only filtration selects the cake.
        assert solid_before_filter.selected is False
        assert state.phases.phases["mother_liquor"].selected is True
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert "crystal_seeded" not in state.metadata
        assert "crystal_seed_mass_g" not in state.metadata
        assert "crystallization_active" not in state.metadata
        assert "crystal_product_mol" not in state.metadata
        assert "crystal_impurity_mol" not in state.metadata
        assert "crystal_yield" not in state.metadata
        assert "crystal_purity" not in state.metadata
        assert "crystal_size" not in state.metadata
        assert "pre_separation_product_mol" not in state.metadata
        assert state.process.metrics["pre_separation_product_mol"] > 0.0
        assert state.process.metrics["crystal_yield"] > 0.0
        assert state.process.metrics["crystal_purity"] > 0.0
        assert state.process.metrics["crystal_size"] > 0.0
        assert env.unwrapped.constitution.check_preconditions(
            "filter_crystals",
            state,
            {},
        )["filter_requires_crystallization"]
        assert env.unwrapped.constitution.check_state(state).passed

        env.step({"operation": "measure", "instrument": "hplc"})
        state = env.unwrapped._state
        solid_before_filter = state.phases.phases["solid"]
        solid_product_before_filter = solid_before_filter.species_amounts_mol["P"]
        solid_impurity_before_filter = solid_before_filter.species_amounts_mol["B"]
        _, _, _, _, filter_info = env.step({"operation": "filter_crystals"})
        state = env.unwrapped._state
        solid_after_filter = state.phases.phases["solid"]
        filter_settings = equipment_settings(state.equipment, "crystal_filter")
        assert filter_info["transaction_status"] == "committed"
        assert solid_after_filter.selected is True
        assert solid_after_filter.species_amounts_mol["P"] == pytest.approx(
            solid_product_before_filter * 0.96
        )
        assert solid_after_filter.species_amounts_mol["B"] == pytest.approx(
            solid_impurity_before_filter * 0.92
        )
        assert "crystallization_active" not in state.metadata
        assert "crystal_product_mol" not in state.metadata
        assert "crystal_impurity_mol" not in state.metadata
        assert "crystal_yield" not in state.metadata
        assert "crystal_purity" not in state.metadata
        assert "crystal_size" not in state.metadata
        assert "purity" not in state.metadata
        assert "recovery" not in state.metadata
        assert "pre_separation_product_mol" not in state.metadata
        assert "solvent_loss" not in state.metadata
        assert "crystals_filtered" not in state.metadata
        assert filter_settings["crystals_filtered"] is True
        assert filter_settings["filtered_product_mol"] > 0.0
        assert state.process.metrics["pre_separation_product_mol"] > 0.0
        assert state.process.metrics["solvent_loss"] > 0.0
        assert state.process.metrics["crystal_yield"] > 0.0
        assert state.process.metrics["crystal_purity"] > 0.0
        assert state.process.metrics["purity"] > 0.0
        assert state.process.metrics["recovery"] > 0.0
        assert env.unwrapped.constitution.check_state(state).passed
    finally:
        env.close()


def test_runtime_distillation_outputs_use_typed_phase_ledger() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0)
    try:
        env.reset(seed=0)
        species_view = MechanismSpeciesView(env.unwrapped.scenario_instance.compiled_mechanism)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "wait", "duration_s": 900.0, "stirring_speed_rpm": 720.0},
            {"operation": "evaporate", "target_temperature_K": 335.0, "duration_s": 600.0},
        ):
            env.step(action)
        state = env.unwrapped._state
        assert "solvent_loss" not in state.metadata
        assert state.process.metrics["solvent_loss"] > 0.0
        assert env.unwrapped.constitution.check_state(state).passed
        feed_volume_before_distill = state.volume_L

        _, _, _, _, distill_info = env.step(
            {
                "operation": "distill",
                "target_temperature_K": 360.0,
                "duration_s": 1500.0,
                "reflux_ratio": 2.0,
            }
        )
        state = env.unwrapped._state
        assert distill_info["transaction_status"] == "committed"
        assert state.phases is not None
        assert {"distillate", "bottoms"} <= set(state.phases.phases)
        distillation_settings = equipment_settings(state.equipment, "distillation_column")
        target_species = species_view.target_species_for_state(state)
        impurity_species = species_view.impurity_species_for_state(state)
        distillate_before_collect = state.phases.phases["distillate"]
        product_before_collect = sum(
            distillate_before_collect.species_amounts_mol[species_id]
            for species_id in target_species
        )
        impurity_before_collect = sum(
            distillate_before_collect.species_amounts_mol[species_id]
            for species_id in impurity_species
        )
        assert product_before_collect > 0.0
        assert distillate_before_collect.selected is True
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert sum(phase.volume_L for phase in state.phases.phases.values()) == pytest.approx(
            feed_volume_before_distill
        )
        assert "distillation_active" not in state.metadata
        assert "distillate_product_mol" not in state.metadata
        assert "distillate_impurity_mol" not in state.metadata
        assert "distillate_purity" not in state.metadata
        assert "distillate_recovery" not in state.metadata
        assert "pre_separation_product_mol" not in state.metadata
        assert "solvent_loss" not in state.metadata
        assert "distillation_model" not in state.metadata
        assert "distillation_kernel" not in state.metadata
        assert distillation_settings["distillation_model"] == (
            "chemworld_duty_limited_distillation_vnext"
        )
        assert "distillation_kernel" in distillation_settings
        assert state.process.metrics["pre_separation_product_mol"] > 0.0
        assert state.process.metrics["solvent_loss"] > 0.0
        assert state.process.metrics["distillate_purity"] > 0.0
        assert state.process.metrics["distillate_recovery"] > 0.0
        assert env.unwrapped.constitution.check_preconditions(
            "collect_fraction",
            state,
            {},
        )["collect_fraction_requires_distillation"]
        assert env.unwrapped.constitution.check_state(state).passed

        _, _, _, _, collect_info = env.step(
            {"operation": "collect_fraction", "transfer_fraction": 0.92}
        )
        state = env.unwrapped._state
        distillate_after_collect = state.phases.phases["distillate"]
        collected_fraction = state.phases.phases["collected_fraction"]
        fraction_settings = equipment_settings(state.equipment, "distillation_column")
        assert collect_info["transaction_status"] == "committed"
        assert sum(
            distillate_after_collect.species_amounts_mol[species_id]
            for species_id in target_species
        ) == pytest.approx(product_before_collect * 0.08)
        assert sum(
            distillate_after_collect.species_amounts_mol[species_id]
            for species_id in impurity_species
        ) == pytest.approx(impurity_before_collect * 0.08)
        assert sum(
            collected_fraction.species_amounts_mol[species_id] for species_id in target_species
        ) == pytest.approx(product_before_collect * 0.92)
        assert sum(
            collected_fraction.species_amounts_mol[species_id] for species_id in impurity_species
        ) == pytest.approx(impurity_before_collect * 0.92)
        assert collected_fraction.selected is True
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert sum(phase.volume_L for phase in state.phases.phases.values()) == pytest.approx(
            feed_volume_before_distill
        )
        assert "distillation_active" not in state.metadata
        assert "distillate_product_mol" not in state.metadata
        assert "distillate_impurity_mol" not in state.metadata
        assert "distillate_purity" not in state.metadata
        assert "distillate_recovery" not in state.metadata
        assert "purity" not in state.metadata
        assert "recovery" not in state.metadata
        assert "pre_separation_product_mol" not in state.metadata
        assert "solvent_loss" not in state.metadata
        assert "fraction_collected" not in state.metadata
        assert fraction_settings["fraction_collected"] is True
        assert fraction_settings["transfer_fraction"] == pytest.approx(0.92)
        assert state.process.metrics["pre_separation_product_mol"] > 0.0
        assert state.process.metrics["solvent_loss"] > 0.0
        assert state.process.metrics["distillate_purity"] > 0.0
        assert state.process.metrics["distillate_recovery"] > 0.0
        assert state.process.metrics["purity"] > 0.0
        assert state.process.metrics["recovery"] > 0.0
        assert env.unwrapped.constitution.check_state(state).passed
    finally:
        env.close()


def test_world_reference_reaction_fixture_is_executable_not_metadata_only() -> None:
    world = load_chemworld_parameters("public-dev", seed=1)
    state = initial_chemworld_state().replace(
        species_amounts={
            "A": 0.01,
            "P": 0.0,
            "B": 0.0,
            "D": 0.0,
            "E": 0.0,
            "Cat_active": 0.0002,
            "Cat_dead": 0.0,
        },
        volume_L=0.025,
        species=SpeciesLedger(initial_amounts_mol={"A": 0.01}),
        equipment=upsert_equipment_record(
            initial_chemworld_state().equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id="batch_reactor",
            status="configured",
            settings={"solvent": 1, "catalyst": 1, "stirring_speed_rpm": 700.0},
        ),
    )
    result = integrate_seven_slot_reference_ode(
        state=state,
        world=world,
        duration_s=900.0,
        target_temperature_K=380.0,
        heat=True,
        stirring_speed_rpm=700.0,
    )
    assert result is not None
    assert result.species_amounts["A"] < state.species_amounts["A"]
    assert result.species_amounts["P"] >= 0.0
    pressure, risk = pressure_and_risk(
        state=state.replace(
            species_amounts=result.species_amounts,
            temperature_K=result.temperature_K,
        ),
        solvent_risks=world.solvent_risks,
    )
    assert pressure > 0.0
    assert 0.0 <= risk <= 1.0
    split = partition_split(
        product_mol=0.005,
        impurity_mol=0.001,
        solvent=1,
        temperature_K=298.15,
        duration_s=180.0,
        stirring_speed_rpm=700.0,
        organic_volume_L=0.015,
        aqueous_volume_L=0.025,
    )
    assert split["organic_product_mol"] + split["aqueous_product_mol"] == 0.005


def test_compiled_reaction_kernel_uses_mechanism_species_not_fixed_slots() -> None:
    compiled = compile_mechanism_for_scenario("reaction-to-distillation")
    world = load_chemworld_parameters("public-dev", seed=1)
    state = initial_chemworld_state(
        species_ids=tuple(compiled.species_index),
        species_roles=compiled.species_roles,
        initial_amounts_mol=compiled.initial_amount_policy,
        initial_limiting_species=compiled.score_spec.initial_limiting_species,
    ).replace(
        species_amounts={
            "Acid": 0.010,
            "Alcohol": 0.012,
            "Ester": 0.0,
            "Water": 0.0,
            "Ether": 0.0,
            "Ester_vapor": 0.0,
            "Alcohol_vapor": 0.0,
            "Water_vapor": 0.0,
        },
        volume_L=0.028,
        equipment=upsert_equipment_record(
            initial_chemworld_state().equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id="batch_reactor",
            status="configured",
            settings={"solvent": 1, "catalyst": 1, "stirring_speed_rpm": 700.0},
        ),
    )

    result = integrate_compiled_reaction_ode(
        state=state,
        world=world,
        compiled_mechanism=compiled,
        duration_s=1200.0,
        target_temperature_K=380.0,
        heat=True,
        stirring_speed_rpm=700.0,
    )

    assert result is not None
    assert "A" not in result.species_amounts
    assert "P" not in result.species_amounts
    assert result.species_amounts["Acid"] < state.species_amounts["Acid"]
    assert result.species_amounts["Ester"] > 0.0
    assert result.heat_reaction_J != 0.0


def test_compiled_reaction_kernel_rejects_missing_mechanism() -> None:
    world = load_chemworld_parameters("public-dev", seed=1)
    state = initial_chemworld_state().replace(volume_L=0.025)

    with pytest.raises(ValueError, match="compiled_mechanism is required"):
        integrate_compiled_reaction_ode(
            state=state,
            world=world,
            compiled_mechanism=None,
            duration_s=1200.0,
            target_temperature_K=380.0,
            heat=True,
            stirring_speed_rpm=700.0,
        )


def test_scenarios_are_first_class_and_share_world_law() -> None:
    scenarios = list_scenarios()
    assert scenarios
    assert {scenario.world_law_id for scenario in scenarios} == {WORLD_FAMILY_VERSION}
    card = get_scenario_card("reaction-to-purification")
    assert card["family"] == "reaction_separation"
    assert "separation" in card["allowed_module_tags"]
    crystallization_card = get_scenario_card("reaction-to-crystallization")
    assert crystallization_card["family"] == "reaction_crystallization"
    assert "crystallization" in crystallization_card["allowed_module_tags"]
    assert {task.world_law_id for task in list_tasks()} == {WORLD_FAMILY_VERSION}


def test_instrument_contracts_expose_observation_layers() -> None:
    contracts = instrument_contracts()
    final_assay = contracts["final_assay"].to_dict()
    assert final_assay["requires_terminated"]
    assert final_assay["destructive"]
    assert "processed_estimate_schema" in final_assay
    assert "spectra" in final_assay["raw_signal_schema"]["properties"]
    assert "yield" in final_assay["observable_keys"]


def test_virtual_instrument_signals_are_plot_ready() -> None:
    values = {
        "yield": 0.62,
        "selectivity": 0.82,
        "conversion": 0.76,
        "byproduct_signal": 0.12,
        "degradation_warning": 0.06,
        "purity": 0.88,
        "recovery": 0.70,
        "phase_ratio": 0.55,
        "impurity_signal": 0.08,
        "solvent_loss": 0.04,
        "process_mass_balance_error": 0.02,
        "distillate_purity": 0.91,
        "flow_conversion": 0.0,
        "energy_efficiency": 0.0,
    }

    hplc = raw_signal("hplc", values)
    assert hplc["kind"] == "hplc_chromatogram"
    assert len(hplc["time_min"]) == len(hplc["intensity"]) >= 100
    assert hplc["peaks"][1]["assignment"] == "target_product_proxy"

    uvvis = raw_signal("uvvis", values)
    assert uvvis["kind"] == "uvvis_spectrum"
    assert len(uvvis["wavelength_nm"]) == len(uvvis["absorbance"]) >= 100

    final_packet = raw_signal("final_assay", values)
    assert final_packet["kind"] == "final_assay_packet"
    assert {"hplc", "gc", "uvvis", "ir", "nmr"} <= set(final_packet["spectra"])


def test_spectra_do_not_expose_fixed_reaction_species_labels() -> None:
    spectra_source = Path("src/chemworld/world/spectra.py").read_text(encoding="utf-8")
    spectroscopy_source = Path("src/chemworld/physchem/spectroscopy.py").read_text(encoding="utf-8")

    assert "A_proxy" not in spectra_source
    assert "P_proxy" not in spectra_source
    assert 'startswith("P")' not in spectra_source
    assert 'startswith(("B", "D", "E", "S"))' not in spectra_source
    assert 'lowered.startswith(("b", "s", "d", "e"))' not in spectroscopy_source


def test_action_and_recipe_public_validation() -> None:
    action = {"operation": "measure", "instrument": "final_assay"}
    assert validate_action_schema(action).valid
    task = get_task("reaction-to-assay")
    result = validate_action(action, task.to_dict())
    assert result.valid
    blocked = validate_action(
        {"operation": "add_extractant", "volume_L": 0.01},
        get_task("reaction-optimization-standard").to_dict(),
    )
    assert not blocked.valid
    recipe = {"steps": [{"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}]}
    assert validate_recipe_schema(recipe).valid


def test_operation_contracts_classify_macros_and_domains() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        task_info = env.unwrapped.task_info()
        contracts = task_info["operation_contracts"]

        assert contracts["heat"]["kind"] == "primitive"
        assert contracts["wash"]["kind"] == "macro"
        assert contracts["dry"]["kind"] == "macro"
        assert contracts["concentrate"]["kind"] == "macro"
        assert contracts["distill"]["kind"] == "domain"
        assert contracts["terminate"]["kind"] == "terminal"
    finally:
        env.close()


def test_recipe_compiler_expands_macros_and_checks_task_policy() -> None:
    recipe = {
        "steps": [
            {"operation": "wash", "volume_L": 0.012, "solvent": 2},
            {"operation": "dry", "duration_s": 240.0},
            {"operation": "concentrate", "duration_s": 360.0},
        ]
    }
    purification_task = get_task("reaction-to-purification").to_dict()

    compiled = compile_recipe(recipe, task_info=purification_task)

    assert [step["operation"] for step in compiled] == [
        "add_phase",
        "mix",
        "settle",
        "separate_phase",
        "dry",
        "concentrate",
    ]
    assert compiled[0] == {
        "operation": "add_phase",
        "phase": "aqueous",
        "volume_L": 0.012,
    }
    assert compiled[4] == {"operation": "dry"}
    assert validate_recipe(recipe, task_info=purification_task).valid

    reaction_only_task = get_task("reaction-to-assay").to_dict()
    blocked = validate_recipe(recipe, task_info=reaction_only_task)
    assert not blocked.valid
    assert "operation not allowed by task: add_phase" in blocked.errors[0]


def test_packaged_schema_files_match_runtime_constants() -> None:
    expected = {
        "action": ACTION_SCHEMA,
        "manifest": MANIFEST_SCHEMA,
        "observation": OBSERVATION_SCHEMA,
        "recipe": RECIPE_SCHEMA,
        "scenario": SCENARIO_SCHEMA,
        "task": TASK_SCHEMA,
        "trajectory": TRAJECTORY_SCHEMA,
    }
    for schema_id, schema in expected.items():
        assert load_schema_file(schema_id) == schema


def test_world_state_does_not_expose_mutable_references() -> None:
    source_metadata = {"nested": {"value": 1}}
    state = WorldState(
        species_amounts={"A": 1.0},
        volume_L=0.01,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="reactor",
        metadata=source_metadata,
    )
    source_metadata["nested"]["value"] = 99
    exported = state.to_dict()
    exported["metadata"]["nested"]["value"] = 123
    assert state.metadata["nested"]["value"] == 1


def test_env_info_logs_campaign_experiment_operation_and_render() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0, render_mode="ansi")
    try:
        _, info = env.reset(seed=0)
        assert info["scenario"]["scenario_id"] == "reaction-to-assay"
        _, _, _, _, step_info = env.step(
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}
        )
        assert step_info["campaign_id"]
        assert step_info["experiment_index"] == 0
        assert step_info["operation_id"] == 1
        assert "cost_components" in step_info
        task_info = env.unwrapped.task_info()
        assert task_info["operation_contracts"]["add_extractant"]["module"] == "separation"
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "ChemWorld" in rendered
        assert "ledger:" in rendered
    finally:
        env.close()


def test_validator_rejects_invalid_payload_bounds() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        _, _ = env.reset(seed=0)
        before = env.unwrapped._state
        bad = env.unwrapped.operation_validator.validate(
            {"operation": "add_solvent", "volume_L": 0.2, "solvent": 1},
            env.unwrapped._state,
        )
        assert not bad.is_valid
        assert not bad.dispatchable_to_runtime
        assert "payload_bounds:volume_L" in bad.invalid_reasons
        assert "payload_bounds:total_volume_L" in bad.invalid_reasons
        _, _, _, _, info = env.step({"operation": "add_solvent", "volume_L": 0.2, "solvent": 1})
        after = env.unwrapped._state
        assert info["transaction_status"] == "validation_failed"
        assert info["world_events"][0]["event_type"] == "validation_failed"
        assert after.species_amounts == before.species_amounts
        assert after.volume_L == pytest.approx(before.volume_L)
    finally:
        env.close()


def test_validator_rejects_private_electrochemical_payload_fields() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        _, _ = env.reset(seed=0)
        bad = env.unwrapped.operation_validator.validate(
            {
                "operation": "set_potential",
                "potential_V": 1.2,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
                "electrolyte_conductivity_S_m": 0.0,
                "electrode_gap_m": 0.20,
                "electrode_area_m2": 0.0,
                "contact_resistance_ohm": -0.1,
                "voltage_window_V": 0.0,
            },
            env.unwrapped._state,
        )

        assert not bad.is_valid
        assert "payload_fields_declared" in bad.invalid_reasons
        assert not bad.dispatchable_to_runtime
    finally:
        env.close()


def test_scenario_generator_uses_initial_state_seed_and_profile() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        _, info = env.reset(seed=0)
        assert "initial_state_seed" not in info["scenario"]
        assert "parameter_profile" not in info["scenario"]
        assert env.unwrapped.scenario_spec.initial_state_seed == 7
        assert env.unwrapped.scenario_spec.parameter_profile == "downstream_processing"
        assert env.unwrapped._state.metadata["scenario_id"] == "reaction-to-purification"
        assert "initial_condition_jitter" in env.unwrapped._state.metadata
    finally:
        env.close()


def test_cli_scenarios_validation_render_and_dataset(tmp_path, capsys) -> None:
    action_path = tmp_path / "action.json"
    action_path.write_text(
        json.dumps({"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}),
        encoding="utf-8-sig",
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps({"steps": [{"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}]}),
        encoding="utf-8",
    )
    trajectory = tmp_path / "run.jsonl"
    exported = tmp_path / "dataset.jsonl"

    main(["scenarios", "list"])
    main(["scenarios", "show", "reaction-optimization"])
    main(["validate-action", "--task", "reaction-to-assay", "--action", str(action_path)])
    main(["validate-recipe", "--task", "reaction-to-assay", "--recipe", str(recipe_path)])
    main(["render", "--task", "reaction-to-assay", "--actions", str(action_path)])
    main(["run", "--task", "reaction-to-assay", "--agent", "random", "--output", str(trajectory)])
    main(
        [
            "datasets",
            "export",
            "--submission",
            str(trajectory),
            "--format",
            "jsonl",
            "--output",
            str(exported),
        ]
    )
    main(["datasets", "card", "--dataset", str(exported)])
    output = capsys.readouterr().out
    assert "reaction-optimization" in output
    assert "compiled_step_count" in output
    assert "ChemWorld" in output
    assert exported.exists()
    records = load_jsonl(exported)
    assert records[0]["campaign_id"]
    card = dataset_card(exported)
    flattened = export_dataset(trajectory, output=tmp_path / "copy.jsonl", format="jsonl")
    assert card["record_count"] == len(records)
    assert card["schema_version"] == "chemworld-dataset-card-0.4"
    assert card["trajectory_schema_versions"] == ["chemworld-trajectory-0.2"]
    assert card["protocol_hashes"]["task_contract_hashes"] == [records[0]["task_contract_hash"]]
    assert card["protocol_hashes"]["runtime_profile_hashes"] == [records[0]["runtime_profile_hash"]]
    assert card["protocol_hashes"]["mechanism_hashes"] == [records[0]["mechanism_hash"]]
    assert card["protocol_hashes"]["scoring_contract_hashes"] == [
        records[0]["scoring_contract_hash"]
    ]
    assert card["protocol_hashes"]["observation_contract_hashes"] == [
        records[0]["observation_contract_hash"]
    ]
    assert card["replay_verification"]["verified"]
    assert card["replay_verification"]["checked_steps"] == len(records)
    assert card["privacy"]["status"] == "synthetic_or_submission_provided"
    assert card["provenance"]["generator"] == "ChemWorld"
    assert card["provenance"]["replay_verified"] is True
    assert card["provenance"]["protocol_hashes"] == card["protocol_hashes"]
    assert flattened.record_count == len(records)
    flattened_record = flatten_record(records[0])
    assert flattened_record["task_id"] == records[0]["task_id"]
    assert flattened_record["run_id"] == records[0]["run_id"]
    assert json.loads(flattened_record["environment_outcome"])
    assert json.loads(flattened_record["agent_visible_observation"])
    assert json.loads(flattened_record["evaluation_outcome"])
    assert flattened_record["task_contract_hash"] == records[0]["task_contract_hash"]
    assert flattened_record["runtime_profile_hash"] == records[0]["runtime_profile_hash"]
