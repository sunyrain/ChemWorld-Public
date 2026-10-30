"""WF-10 provider and adapter proposal for rate-law contract diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.model_provider import ModelProviderResult
from chemworld.physchem.reaction_network_specs import ReactionSpec
from chemworld.physchem.reaction_rate_contracts import (
    CANTERA_COMMIT,
    RMG_PY_COMMIT,
    audit_reaction_rate_contract,
)

OWNED_PATHS = (
    "src/chemworld/physchem/reaction_rate_contracts.py",
    "src/chemworld/physchem/reaction_adapter_manifest.py",
    "tests/test_reaction_rate_contracts.py",
)
INTEGRATION_OPERATIONS = ("heat", "wait", "run_flow")
RUNTIME_INTEGRATION_OWNED_PATHS = (
    "src/chemworld/runtime/reaction_thermal_services.py",
    "src/chemworld/world/reaction_kernel.py",
    "src/chemworld/world/reaction_reference.py",
    "src/chemworld/physchem/reaction_adapter_manifest.py",
    "src/chemworld/physchem/reactor_cards.py",
    "tests/test_reaction_reactor_runtime_integration.py",
)


def reaction_rate_provider_contract() -> ModelProviderContract:
    """Return the WF-00-compatible diagnostic provider contract."""

    return ModelProviderContract(
        model_id="chemworld_arrhenius_unit_contract_vnext",
        module_id="reaction_kinetics",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.REFERENCE,
        provider_path=("chemworld.physchem.reaction_adapter_manifest.ReactionRateContractProvider"),
        input_fields=("reaction",),
        output_fields=("contract",),
        units={
            "reaction": "ReactionSpec",
            "contract": "JSON dimension contract",
        },
        validity_checks=(
            "reaction is a ReactionSpec",
            "rate law is in the supported Arrhenius family",
            "pre-exponential factors are finite and positive",
            "temperature exponents and activation energies are finite",
            "concentration/activity basis and standard concentration are explicit",
            "explicit reaction orders are finite and nonnegative",
            "reversible, third-body, and falloff declarations are complete",
        ),
        diagnostic_fields=(
            "passed",
            "equation_id",
            "forward_order",
            "effective_forward_order",
            "kinetic_basis",
            "violations",
        ),
        failure_policy=(
            "return an unsuccessful ModelProviderResult with explicit declaration "
            "violations; never infer missing units or mutate the mechanism"
        ),
        provenance=(
            (f"Cantera {CANTERA_COMMIT}: Arrhenius and falloff unit conventions"),
            f"RMG-Py {RMG_PY_COMMIT}: Arrhenius quantity schema",
            "WF-10 analytical dimensional identities",
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class ReactionRateContractProvider:
    """Diagnostic provider that audits one reaction declaration."""

    @property
    def model_contract(self) -> ModelProviderContract:
        return reaction_rate_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        reaction = inputs.get("reaction")
        if not isinstance(reaction, ReactionSpec):
            return ("reaction must be a ReactionSpec",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        domain_violations = self.validate_domain(inputs)
        if domain_violations:
            return ModelProviderResult(
                outputs={},
                diagnostics={
                    "passed": False,
                    "equation_id": None,
                    "forward_order": None,
                    "effective_forward_order": None,
                    "kinetic_basis": None,
                    "violations": list(domain_violations),
                },
                success=False,
                failure_reason="; ".join(domain_violations),
                provenance=self.model_contract.provenance,
            )
        reaction = inputs["reaction"]
        report = audit_reaction_rate_contract(reaction)
        diagnostics = {
            "passed": report.passed,
            "equation_id": report.equation_id,
            "forward_order": report.forward_order,
            "effective_forward_order": report.effective_forward_order,
            "kinetic_basis": report.kinetic_basis,
            "violations": list(report.violations),
        }
        if not report.passed:
            return ModelProviderResult(
                outputs={"contract": report.to_dict()},
                diagnostics=diagnostics,
                success=False,
                failure_reason="; ".join(report.violations),
                provenance=report.provenance,
            )
        return ModelProviderResult(
            outputs={"contract": report.to_dict()},
            diagnostics=diagnostics,
            provenance=report.provenance,
        )


def reaction_rate_adapter_manifest() -> ModelAdapterManifest:
    """Return the claim-bound proposal for later WF-110 intake."""

    return ModelAdapterManifest(
        adapter_id="wf-10-arrhenius-unit-contract",
        adapter_version="0.1",
        owner_workstream="wf-10-rate-law-unit-contracts",
        provider_contract=reaction_rate_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="proposal",
    )


def reaction_reactor_runtime_provider_contract() -> ModelProviderContract:
    """Return the validated mechanism-plus-dynamic-batch runtime contract."""

    return ModelProviderContract(
        model_id="chemworld_validated_reaction_reactor_runtime_v1",
        module_id="reaction_reactor_runtime",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.RUNTIME,
        provider_path="chemworld.world.reaction_kernel.integrate_compiled_reaction_ode",
        input_fields=(
            "world_state",
            "compiled_mechanism",
            "duration_s",
            "target_temperature_K",
            "heat_boundary",
            "stirring_speed_rpm",
        ),
        output_fields=(
            "species_amounts",
            "temperature_K",
            "material_energy_ledgers",
            "solver_diagnostic",
            "reactor_diagnostic",
            "trajectory_digest",
        ),
        units={
            "species_amounts": "mol",
            "temperature_K": "K",
            "duration_s": "s",
            "material_energy_ledgers": "mol; J",
            "solver_diagnostic": "JSON",
            "reactor_diagnostic": "JSON",
            "trajectory_digest": "sha256",
        },
        validity_checks=(
            "compiled mechanism passes element and charge conservation",
            "duration, temperature, stirring, volume, pressure, and amounts are finite and bounded",
            "dynamic batch solver converges under the declared tolerance policy",
            "trajectory remains nonnegative and inside temperature/pressure/runaway domain",
            "material, invariant, element, charge, and energy residuals close",
        ),
        diagnostic_fields=(
            "provider_id",
            "reactor_model_id",
            "network_id",
            "mechanism_hash",
            "termination_reason",
            "material_balance_error_mol",
            "maximum_conservation_drift_mol",
            "element_inventory_residuals_mol",
            "charge_inventory_residual_mol",
            "energy_balance_residual_J",
            "trajectory_digest",
        ),
        failure_policy=(
            "raise before commit on invalid domain, solver failure, nonnegativity, "
            "runaway, or conservation/energy drift; the transaction manager rolls back"
        ),
        provenance=(
            "chemworld.physchem.reaction_network.ReactionNetworkSpec",
            "chemworld.physchem.batch_reactors.DynamicBatchReactorModel",
            "chemworld.physchem.reactor_shared reactor validity and ledger contracts",
        ),
        intended_operations=("heat", "wait"),
    )


def reaction_reactor_runtime_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id="foundation-reaction-reactor-runtime-integration",
        adapter_version="1.0",
        owner_workstream="foundation-reaction-reactor-runtime-integration",
        provider_contract=reaction_reactor_runtime_provider_contract(),
        owned_paths=RUNTIME_INTEGRATION_OWNED_PATHS,
        integration_operations=("heat", "wait"),
        target_world_law="chemworld-physical-chemistry-v0.5",
        status="integrated",
        replaces_model_ids=(
            "chemworld_reaction_network_lite",
            "chemworld_reactor_lite",
        ),
    )


__all__ = [
    "INTEGRATION_OPERATIONS",
    "OWNED_PATHS",
    "RUNTIME_INTEGRATION_OWNED_PATHS",
    "ReactionRateContractProvider",
    "reaction_rate_adapter_manifest",
    "reaction_rate_provider_contract",
    "reaction_reactor_runtime_adapter_manifest",
    "reaction_reactor_runtime_provider_contract",
]
