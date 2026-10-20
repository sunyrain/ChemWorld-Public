"""Transactional runtime public surface for ChemWorld.

The package exports a stable facade, but imports are intentionally lazy. Scenario
generation only needs the mechanism compiler; eager importing the full runtime
would pull in action validation while the action codec is still initializing.
"""

from typing import Any

__all__ = [
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
    "compile_mechanism",
    "compile_mechanism_for_scenario",
    "make_chemworld_constitution",
    "mechanism_hash",
    "mechanism_id_for_scenario",
    "validate_mechanism_file",
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
