"""Integrated WF-40 provider for stability-aware LLE."""

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
from chemworld.physchem.phase_equilibrium_units import (
    PHASE_EQUILIBRIUM_MODEL_ID,
    PHASEPY_COMMIT,
    PHASEPY_LLE_PATH,
    PHASEPY_STABILITY_PATH,
    StabilityAwareExtractionRequest,
    simulate_stability_aware_extraction,
)

OWNER_WORKSTREAM = "wf-40-lle-stability-coupling"
OWNED_PATHS = (
    "src/chemworld/physchem/phase_equilibrium_units.py",
    "src/chemworld/physchem/phase_equilibrium_adapter_manifest.py",
    "tests/test_phase_equilibrium_units.py",
)
INTEGRATION_OPERATIONS = ("mix", "wash")
REPLACED_MODEL_IDS = (
    "activity_corrected_extraction_train_v1",
    "lle_phase_stability_diagnostic_v1",
)


def stability_aware_lle_provider_contract() -> ModelProviderContract:
    return ModelProviderContract(
        model_id=PHASE_EQUILIBRIUM_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.PROFESSIONAL_CANDIDATE,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.phase_equilibrium_adapter_manifest.StabilityAwareLLEProvider"
        ),
        input_fields=("request",),
        output_fields=("phase_equilibrium_result",),
        units={
            "request": "StabilityAwareExtractionRequest with explicit mol/L/K units",
            "phase_equilibrium_result": (
                "JSON stage stability, phase allocation, entrainment, and material ledger"
            ),
        },
        validity_checks=(
            "request is a StabilityAwareExtractionRequest",
            "feed, distribution, and optional stability activity models share exact component ids",
            "intrinsic and activity-corrected distribution coefficients remain in [1e-12, 1e12]",
            "every contact lies within the declared volume, stage-count, and temperature domain",
            "every contact passes the two-liquid TPD-style stability gate",
            "distribution iterations converge and stage/train component ledgers close",
        ),
        diagnostic_fields=(
            "material_balance_error_mol",
            "maximum_stage_material_balance_error_mol",
            "minimum_tpd_like",
            "maximum_distribution_residual",
            "target_recovery",
            "target_purity",
            "impurity_rejection",
            "entrained_volume_L",
            "all_stages_two_liquid",
            "all_stages_converged",
            "warnings",
        ),
        failure_policy=(
            "return an explicit failed provider result for invalid, single-liquid, "
            "non-converged, out-of-domain, or non-conservative contacts; never "
            "fabricate a split or silently accept an incomplete ledger"
        ),
        provenance=(
            "WF-40 analytical ideal-contact and multistage recovery identities",
            "coupled TPD-style stability, gamma-corrected distribution, and "
            "directional entrainment ledgers",
            (
                f"PhasePy {PHASEPY_COMMIT}:{PHASEPY_STABILITY_PATH} and "
                f"{PHASEPY_LLE_PATH} convention boundary"
            ),
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class StabilityAwareLLEProvider:
    @property
    def model_contract(self) -> ModelProviderContract:
        return stability_aware_lle_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(inputs.get("request"), StabilityAwareExtractionRequest):
            return ("request must be a StabilityAwareExtractionRequest",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return _failed_result("; ".join(violations), self.model_contract.provenance)
        request = inputs["request"]
        try:
            result = simulate_stability_aware_extraction(request)
        except (RuntimeError, ValueError) as error:
            return _failed_result(str(error), self.model_contract.provenance)
        return ModelProviderResult(
            outputs={"phase_equilibrium_result": result.to_dict()},
            diagnostics={
                "material_balance_error_mol": result.material_balance_error_mol,
                "maximum_stage_material_balance_error_mol": (
                    result.maximum_stage_material_balance_error_mol
                ),
                "minimum_tpd_like": result.minimum_tpd_like,
                "maximum_distribution_residual": (result.maximum_distribution_residual),
                "target_recovery": result.target_recovery,
                "target_purity": result.target_purity,
                "impurity_rejection": result.impurity_rejection,
                "entrained_volume_L": result.entrained_volume_L,
                "all_stages_two_liquid": result.all_stages_two_liquid,
                "all_stages_converged": result.all_stages_converged,
                "warnings": list(result.warnings),
            },
            warnings=result.warnings,
            provenance=result.provenance,
        )


def stability_aware_lle_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id=OWNER_WORKSTREAM,
        adapter_version="0.2",
        owner_workstream=OWNER_WORKSTREAM,
        provider_contract=stability_aware_lle_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
        replaces_model_ids=REPLACED_MODEL_IDS,
    )


def _failed_result(
    failure_reason: str,
    provenance: tuple[str, ...],
) -> ModelProviderResult:
    return ModelProviderResult(
        outputs={},
        diagnostics={
            "material_balance_error_mol": None,
            "maximum_stage_material_balance_error_mol": None,
            "minimum_tpd_like": None,
            "maximum_distribution_residual": None,
            "target_recovery": None,
            "target_purity": None,
            "impurity_rejection": None,
            "entrained_volume_L": None,
            "all_stages_two_liquid": False,
            "all_stages_converged": False,
            "warnings": [failure_reason],
        },
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=provenance,
    )


__all__ = [
    "INTEGRATION_OPERATIONS",
    "OWNED_PATHS",
    "OWNER_WORKSTREAM",
    "REPLACED_MODEL_IDS",
    "StabilityAwareLLEProvider",
    "stability_aware_lle_adapter_manifest",
    "stability_aware_lle_provider_contract",
]
