"""Integrated WF-30 provider for finite-capacity sorbent drying."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.physchem.drying_units import (
    DRYING_MODEL_ID,
    IDAES_ADSORPTION_PATH,
    IDAES_COMMIT,
    SorbentDryingRequest,
    simulate_sorbent_drying,
)
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult

OWNER_WORKSTREAM = "wf-30-drying-sorbent"
OWNED_PATHS = (
    "src/chemworld/physchem/drying_units.py",
    "src/chemworld/physchem/sorbent_kinetics.py",
    "src/chemworld/physchem/drying_adapter_manifest.py",
    "tests/test_drying_units.py",
    "tests/test_sorbent_regeneration.py",
)


def sorbent_drying_provider_contract() -> ModelProviderContract:
    return ModelProviderContract(
        model_id=DRYING_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.drying_adapter_manifest.SorbentDryingProvider"
        ),
        input_fields=("request",),
        output_fields=("drying_result",),
        units={
            "request": "SorbentDryingRequest with explicit L, mol, kg, and s units",
            "drying_result": "JSON liquid/sorbent material ledger",
        },
        validity_checks=(
            "request is a SorbentDryingRequest",
            "wet-liquid and sorbent inventories are finite, nonnegative, and component explicit",
            "initial loading does not exceed the shared site capacity",
            "liquid and retained volumes remain within the declared contactor capacity",
            "the residual drying-component endpoint lies in the closed unit interval",
            (
                "competitive equilibrium converges and component/volume closure "
                "remains within tolerance"
            ),
        ),
        diagnostic_fields=(
            "material_balance_error_mol",
            "volume_balance_error_L",
            "equilibrium_residual",
            "equilibrium_iterations",
            "residual_drying_component_fraction",
            "endpoint_met",
            "product_recovery",
            "warnings",
        ),
        failure_policy=(
            "reject invalid domain inputs or return a failed result; never discard "
            "initial loading, sorbed material, retained liquid, or dried-liquid inventory"
        ),
        provenance=(
            "WF-30 competitive shared-site sorbent and explicit spent-solid ledger",
            "single-component quadratic and SciPy nonlinear reference checks",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_ADSORPTION_PATH} extended-isotherm "
                "convention boundary; not an IDAES liquid-drying backend"
            ),
        ),
        intended_operations=("dry",),
    )


class SorbentDryingProvider:
    @property
    def model_contract(self) -> ModelProviderContract:
        return sorbent_drying_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(inputs.get("request"), SorbentDryingRequest):
            return ("request must be a SorbentDryingRequest",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return ModelProviderResult(
                outputs={},
                diagnostics={
                    "material_balance_error_mol": None,
                    "volume_balance_error_L": None,
                    "equilibrium_residual": None,
                    "equilibrium_iterations": None,
                    "residual_drying_component_fraction": None,
                    "endpoint_met": None,
                    "product_recovery": None,
                    "warnings": list(violations),
                },
                warnings=violations,
                success=False,
                failure_reason="; ".join(violations),
                provenance=self.model_contract.provenance,
            )
        request = inputs["request"]
        try:
            result = simulate_sorbent_drying(request)
        except (RuntimeError, ValueError) as error:
            message = str(error)
            return ModelProviderResult(
                outputs={},
                diagnostics={
                    "material_balance_error_mol": None,
                    "volume_balance_error_L": None,
                    "equilibrium_residual": None,
                    "equilibrium_iterations": None,
                    "residual_drying_component_fraction": None,
                    "endpoint_met": None,
                    "product_recovery": None,
                    "warnings": [message],
                },
                warnings=(message,),
                success=False,
                failure_reason=message,
                provenance=self.model_contract.provenance,
            )
        return ModelProviderResult(
            outputs={"drying_result": result.to_dict()},
            diagnostics={
                "material_balance_error_mol": result.material_balance_error_mol,
                "volume_balance_error_L": result.volume_balance_error_L,
                "equilibrium_residual": result.equilibrium_residual,
                "equilibrium_iterations": result.equilibrium_iterations,
                "residual_drying_component_fraction": (
                    result.residual_drying_component_fraction
                ),
                "endpoint_met": result.endpoint_met,
                "product_recovery": result.product_recovery,
                "warnings": list(result.warnings),
            },
            warnings=result.warnings,
            provenance=result.provenance,
        )


def sorbent_drying_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id="wf-30-drying-sorbent",
        adapter_version="0.2",
        owner_workstream=OWNER_WORKSTREAM,
        provider_contract=sorbent_drying_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=("dry",),
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
        replaces_model_ids=(),
    )


__all__ = [
    "OWNED_PATHS",
    "OWNER_WORKSTREAM",
    "SorbentDryingProvider",
    "sorbent_drying_adapter_manifest",
    "sorbent_drying_provider_contract",
]
