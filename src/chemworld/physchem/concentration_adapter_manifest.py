"""Integrated WF-30 provider for bounded vacuum concentration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.physchem.concentration_units import (
    CONCENTRATION_MODEL_ID,
    IDAES_COMMIT,
    IDAES_FLASH_PATH,
    VacuumConcentrationRequest,
    simulate_vacuum_concentration,
)
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.model_provider import ModelProviderResult

OWNER_WORKSTREAM = "wf-30-vacuum-concentration"
OWNED_PATHS = (
    "src/chemworld/physchem/concentration_units.py",
    "src/chemworld/physchem/concentration_adapter_manifest.py",
    "tests/test_concentration_units.py",
)


def vacuum_concentration_provider_contract() -> ModelProviderContract:
    return ModelProviderContract(
        model_id=CONCENTRATION_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.concentration_adapter_manifest."
            "VacuumConcentrationProvider"
        ),
        input_fields=("request",),
        output_fields=("concentration_result",),
        units={
            "request": "VacuumConcentrationRequest with explicit mol/L/K/Pa/W/s units",
            "concentration_result": "JSON liquid/condensate/vent and energy ledger",
        },
        validity_checks=(
            "request is a VacuumConcentrationRequest",
            "feed and component property profiles have exactly matching component ids",
            "every property profile evaluation temperature equals the operating temperature",
            "pressure, temperature, power, duration, and volume satisfy equipment limits",
            "operating temperature remains within every present component thermal limit",
            "ODE succeeds without heater overspend and component/volume/energy ledgers close",
        ),
        diagnostic_fields=(
            "material_balance_error_mol",
            "volume_balance_error_L",
            "energy_balance_error_J",
            "solvent_remaining_fraction",
            "endpoint_met",
            "target_recovery",
            "heat_duty_J",
            "termination_reason",
            "warnings",
        ),
        failure_policy=(
            "reject invalid domains or return a failed result; never discard "
            "liquid, recovered condensate, vent loss, sensible heat, latent heat, "
            "endpoint, or volatile-target diagnostics"
        ),
        provenance=(
            "WF-30 differential batch gamma-Raoult and Rayleigh identities",
            "power/rate constrained sensible-latent energy ledger",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_FLASH_PATH} material, energy, "
                "heat-duty, and pressure convention boundary"
            ),
        ),
        intended_operations=("concentrate",),
    )


class VacuumConcentrationProvider:
    @property
    def model_contract(self) -> ModelProviderContract:
        return vacuum_concentration_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(inputs.get("request"), VacuumConcentrationRequest):
            return ("request must be a VacuumConcentrationRequest",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return _failed_result("; ".join(violations), self.model_contract.provenance)
        request = inputs["request"]
        try:
            result = simulate_vacuum_concentration(request)
        except (RuntimeError, ValueError) as error:
            return _failed_result(str(error), self.model_contract.provenance)
        return ModelProviderResult(
            outputs={"concentration_result": result.to_dict()},
            diagnostics={
                "material_balance_error_mol": result.material_balance_error_mol,
                "volume_balance_error_L": result.volume_balance_error_L,
                "energy_balance_error_J": result.energy_balance_error_J,
                "solvent_remaining_fraction": result.solvent_remaining_fraction,
                "endpoint_met": result.endpoint_met,
                "target_recovery": result.target_recovery,
                "heat_duty_J": result.heat_duty_J,
                "termination_reason": result.termination_reason,
                "warnings": list(result.warnings),
            },
            warnings=result.warnings,
            provenance=result.provenance,
        )


def vacuum_concentration_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id="wf-30-vacuum-concentration",
        adapter_version="0.2",
        owner_workstream=OWNER_WORKSTREAM,
        provider_contract=vacuum_concentration_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=("concentrate",),
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
        replaces_model_ids=(),
    )


def _failed_result(
    failure_reason: str,
    provenance: tuple[str, ...],
) -> ModelProviderResult:
    return ModelProviderResult(
        outputs={},
        diagnostics={
            "material_balance_error_mol": None,
            "volume_balance_error_L": None,
            "energy_balance_error_J": None,
            "solvent_remaining_fraction": None,
            "endpoint_met": None,
            "target_recovery": None,
            "heat_duty_J": None,
            "termination_reason": None,
            "warnings": [failure_reason],
        },
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=provenance,
    )


__all__ = [
    "OWNED_PATHS",
    "OWNER_WORKSTREAM",
    "VacuumConcentrationProvider",
    "vacuum_concentration_adapter_manifest",
    "vacuum_concentration_provider_contract",
]
