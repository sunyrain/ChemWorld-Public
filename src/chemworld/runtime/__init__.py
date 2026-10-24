"""Transactional runtime public surface for ChemWorld.

The package exports a stable facade, but imports are intentionally lazy. Scenario
generation only needs the mechanism compiler; eager importing the full runtime
would pull in action validation while the action codec is still initializing.
"""

from typing import Any

__all__ = [
    "AqueousMineralInventoryResult",
    "AqueousPhaseInventoryResult",
    "CalphadPhaseInventoryResult",
    "CalphadPhaseTarget",
    "CalphadThermalSpec",
    "ChemWorldCrystallizationServices",
    "ChemWorldDistillationServices",
    "ChemWorldDomainServices",
    "ChemWorldElectrochemicalServices",
    "ChemWorldFlowServices",
    "ChemWorldInstrumentCostServices",
    "ChemWorldObservationKernel",
    "ChemWorldOperationRecorder",
    "ChemWorldPhaseLedgerServices",
    "ChemWorldPhaseSeparationServices",
    "ChemWorldPrimitiveOperationServices",
    "ChemWorldReactionThermalServices",
    "ChemWorldRuntime",
    "CompiledMechanism",
    "DomainServiceContract",
    "DomainServiceRegistry",
    "KernelPlan",
    "KernelResult",
    "MechanismManifest",
    "MechanismSpeciesView",
    "MechanismValidationReport",
    "MolecularCascadeAdapter",
    "MolecularDesignControl",
    "MolecularFlashSeparatorAdapter",
    "MolecularFlashThermalSpec",
    "MolecularInventoryInterface",
    "MolecularPhaseEquilibriumSpec",
    "MolecularPhaseInventoryResult",
    "MolecularPhaseTarget",
    "MolecularProcessDesignResult",
    "MolecularProcessDesignSpec",
    "MolecularProcessPath",
    "MolecularProcessReceiverSpec",
    "MolecularSteadyProcessPath",
    "MolecularUtilityPrices",
    "OperationExecutor",
    "OperationKernel",
    "OperationKernelRegistry",
    "RuntimeContext",
    "RuntimeStepResult",
    "ScoreSpec",
    "ServiceOperationKernel",
    "SessionResult",
    "SessionSnapshot",
    "StatePatch",
    "TaskRuntimeProfile",
    "TransactionManager",
    "TransactionResult",
    "WorldEvent",
    "advance_molecular_process_receiver",
    "analyze_energy_grain_world_state",
    "analyze_gas_network_world_state",
    "analyze_sbml_finite_time_world",
    "analyze_sbml_flux_modes",
    "analyze_sbml_world_state",
    "analyze_treatment_world_economics",
    "build_molecular_process_receiver",
    "calphad_inventory_enthalpy_j",
    "compile_mechanism",
    "compile_mechanism_for_scenario",
    "equilibrate_aqueous_mineral_inventory",
    "equilibrate_aqueous_phase_inventory",
    "equilibrate_calphad_phase_inventory",
    "equilibrate_molecular_phase_inventory",
    "evaluate_molecular_design",
    "make_chemworld_constitution",
    "mechanism_hash",
    "mechanism_id_for_scenario",
    "molecular_flash_inventory_enthalpy_j",
    "molecular_interface_from_ledger",
    "molecular_phase_stream",
    "optimize_molecular_process",
    "optimized_process_receiver",
    "read_membrane_distillation",
    "select_calphad_diagram_inventory_point",
    "select_calphad_ternary_inventory_point",
    "validate_calphad_thermal_state",
    "validate_mechanism_file",
    "validate_molecular_flash_state",
]

_MECHANISM_EXPORTS = {
    "CompiledMechanism",
    "MechanismManifest",
    "MechanismValidationReport",
    "ScoreSpec",
    "compile_mechanism",
    "compile_mechanism_for_scenario",
    "mechanism_hash",
    "mechanism_id_for_scenario",
    "validate_mechanism_file",
}


def __getattr__(name: str) -> Any:
    if name == "analyze_sbml_finite_time_world":
        from chemworld.runtime.sbml_finite_time import analyze_sbml_finite_time_world

        return analyze_sbml_finite_time_world
    if name == "select_calphad_ternary_inventory_point":
        from chemworld.runtime.calphad_ternary_inventory import (
            select_calphad_ternary_inventory_point,
        )

        return select_calphad_ternary_inventory_point
    if name == "analyze_energy_grain_world_state":
        from chemworld.runtime.energy_grain_analysis import analyze_energy_grain_world_state

        return analyze_energy_grain_world_state
    if name == "analyze_gas_network_world_state":
        from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state

        return analyze_gas_network_world_state
    if name == "MolecularFlashSeparatorAdapter":
        from chemworld.runtime.molecular_flash_separator import MolecularFlashSeparatorAdapter

        return MolecularFlashSeparatorAdapter
    if name == "MolecularCascadeAdapter":
        from chemworld.runtime.molecular_cascade import MolecularCascadeAdapter

        return MolecularCascadeAdapter
    if name == "analyze_sbml_flux_modes":
        from chemworld.runtime.sbml_flux_modes import analyze_sbml_flux_modes

        return analyze_sbml_flux_modes
    if name in {
        "MolecularDesignControl", "MolecularUtilityPrices", "MolecularProcessDesignSpec",
        "MolecularProcessDesignResult", "evaluate_molecular_design",
        "optimize_molecular_process", "optimized_process_receiver",
    }:
        from chemworld.runtime import molecular_process_design

        return getattr(molecular_process_design, name)
    if name == "select_calphad_diagram_inventory_point":
        from chemworld.runtime.calphad_phase_diagram import select_calphad_diagram_inventory_point

        return select_calphad_diagram_inventory_point
    if name in {
        "CalphadThermalSpec", "calphad_inventory_enthalpy_j", "validate_calphad_thermal_state",
    }:
        from chemworld.runtime import calphad_thermal

        return getattr(calphad_thermal, name)
    if name == "MolecularProcessReceiverSpec":
        from chemworld.runtime.molecular_process_runtime import MolecularProcessReceiverSpec

        return MolecularProcessReceiverSpec
    if name in {
        "MolecularFlashThermalSpec", "molecular_flash_inventory_enthalpy_j",
        "validate_molecular_flash_state",
    }:
        from chemworld.runtime import molecular_flash_thermal

        return getattr(molecular_flash_thermal, name)
    if name in {"AqueousMineralInventoryResult", "equilibrate_aqueous_mineral_inventory"}:
        from chemworld.runtime import aqueous_mineral_equilibrium

        return getattr(aqueous_mineral_equilibrium, name)
    if name == "read_membrane_distillation":
        from chemworld.runtime.membrane_distillation import read_membrane_distillation

        return read_membrane_distillation
    if name == "analyze_treatment_world_economics":
        from chemworld.runtime.zero_order_treatment import analyze_treatment_world_economics

        return analyze_treatment_world_economics
    if name == "analyze_sbml_world_state":
        from chemworld.runtime.sbml_analysis import analyze_sbml_world_state

        return analyze_sbml_world_state
    if name in {
        "CalphadPhaseInventoryResult", "CalphadPhaseTarget",
        "equilibrate_calphad_phase_inventory",
    }:
        from chemworld.runtime import calphad_phase_inventory

        return getattr(calphad_phase_inventory, name)
    if name in {
        "MolecularProcessPath", "MolecularSteadyProcessPath", "build_molecular_process_receiver",
        "advance_molecular_process_receiver",
    }:
        from chemworld.runtime import molecular_process_path

        return getattr(molecular_process_path, name)
    if name in {"MolecularInventoryInterface", "molecular_interface_from_ledger"}:
        from chemworld.runtime import molecular_interface

        return getattr(molecular_interface, name)
    if name in {"AqueousPhaseInventoryResult", "equilibrate_aqueous_phase_inventory"}:
        from chemworld.runtime import aqueous_phase_equilibrium

        return getattr(aqueous_phase_equilibrium, name)
    if name in {
        "MolecularPhaseTarget", "MolecularPhaseEquilibriumSpec", "MolecularPhaseInventoryResult",
        "equilibrate_molecular_phase_inventory", "molecular_phase_stream",
    }:
        from chemworld.runtime import molecular_phase_inventory

        return getattr(molecular_phase_inventory, name)
    if name in _MECHANISM_EXPORTS:
        from chemworld.runtime import mechanisms

        return getattr(mechanisms, name)
    if name in {"WorldEvent", "StatePatch", "TransactionManager", "TransactionResult"}:
        from chemworld.runtime import transactions

        return getattr(transactions, name)
    if name == "MechanismSpeciesView":
        from chemworld.runtime import species

        return getattr(species, name)
    if name == "ChemWorldOperationRecorder":
        from chemworld.runtime import record_services

        return getattr(record_services, name)
    if name == "ChemWorldCrystallizationServices":
        from chemworld.runtime import crystallization_services

        return getattr(crystallization_services, name)
    if name == "ChemWorldDistillationServices":
        from chemworld.runtime import distillation_services

        return getattr(distillation_services, name)
    if name == "ChemWorldElectrochemicalServices":
        from chemworld.runtime import electrochemical_services

        return getattr(electrochemical_services, name)
    if name == "ChemWorldFlowServices":
        from chemworld.runtime import flow_services

        return getattr(flow_services, name)
    if name == "ChemWorldInstrumentCostServices":
        from chemworld.runtime import instrument_cost_services

        return getattr(instrument_cost_services, name)
    if name == "ChemWorldPhaseSeparationServices":
        from chemworld.runtime import phase_separation_services

        return getattr(phase_separation_services, name)
    if name == "ChemWorldPhaseLedgerServices":
        from chemworld.runtime import phase_ledger_services

        return getattr(phase_ledger_services, name)
    if name == "ChemWorldPrimitiveOperationServices":
        from chemworld.runtime import primitive_services

        return getattr(primitive_services, name)
    if name == "ChemWorldReactionThermalServices":
        from chemworld.runtime import reaction_thermal_services

        return getattr(reaction_thermal_services, name)
    if name == "ChemWorldDomainServices":
        from chemworld.runtime import domain_services

        return getattr(domain_services, name)
    if name in {"DomainServiceContract", "DomainServiceRegistry"}:
        from chemworld.runtime import domain_service_registry

        return getattr(domain_service_registry, name)
    if name in {
        "KernelPlan",
        "KernelResult",
        "OperationKernel",
        "RuntimeContext",
    }:
        from chemworld.runtime import kernel_contracts

        return getattr(kernel_contracts, name)
    if name in {"OperationKernelRegistry", "ServiceOperationKernel"}:
        from chemworld.runtime import kernel_registry

        return getattr(kernel_registry, name)
    if name == "TaskRuntimeProfile":
        from chemworld.runtime import profiles

        return getattr(profiles, name)
    if name in {"ChemWorldRuntime", "SessionResult", "SessionSnapshot"}:
        from chemworld.runtime import engine

        return getattr(engine, name)
    if name in {"OperationExecutor", "RuntimeStepResult"}:
        from chemworld.runtime import executor

        return getattr(executor, name)
    if name == "ChemWorldObservationKernel":
        from chemworld.runtime import observation_services

        return getattr(observation_services, name)
    if name == "make_chemworld_constitution":
        from chemworld.runtime import constitution_factory

        return getattr(constitution_factory, name)
    raise AttributeError(f"module 'chemworld.runtime' has no attribute {name!r}")
