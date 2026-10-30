"""Integrated WF-60 provider for bounded distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.physchem.distillation_units import (
    DISTILLATION_ENGINE_MODEL_ID,
    DUTY_LIMITED_DISTILLATION_MODEL_ID,
    IDAES_COMMIT,
    IDAES_CONDENSER_PATH,
    IDAES_REBOILER_PATH,
    IDAES_TRAY_COLUMN_PATH,
    DutyLimitedDistillationRequest,
    simulate_duty_limited_distillation,
)
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.model_provider import ModelProviderResult

OWNER_WORKSTREAM = "wf-60-duty-limited-distillation"
OWNED_PATHS = (
    "src/chemworld/physchem/distillation_units.py",
    "src/chemworld/physchem/distillation_adapter_manifest.py",
    "tests/test_distillation_units.py",
)
INTEGRATION_OPERATIONS = ("distill",)
REPLACED_MODEL_IDS = (DISTILLATION_ENGINE_MODEL_ID,)


def duty_limited_distillation_provider_contract() -> ModelProviderContract:
    return ModelProviderContract(
        model_id=DUTY_LIMITED_DISTILLATION_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.PROFESSIONAL_CANDIDATE,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.distillation_adapter_manifest.DutyLimitedDistillationProvider"
        ),
        input_fields=("request",),
        output_fields=("distillation_result",),
        units={
            "request": "DutyLimitedDistillationRequest with explicit mol/Pa/K/s/W units",
            "distillation_result": (
                "JSON distillate/bottoms, VLE/FUG, energy, vapor, and capacity ledger"
            ),
        },
        validity_checks=(
            "request is a DutyLimitedDistillationRequest",
            "feed and temperature-evaluated component profiles have exact matching ids",
            "light key is more volatile than heavy key under the supplied VLE condition",
            "pressure, temperature, reflux, duration, batch amount, cut, and "
            "residual bottoms satisfy the column card",
            "operating condition reaches the declared feed-composition bubble "
            "pressure before vapor production",
            "reboiler, condenser, and internal vapor traffic remain within installed capacities",
            "component and sensible/latent energy ledgers close within tolerance",
        ),
        diagnostic_fields=(
            "material_balance_error_mol",
            "energy_balance_error_J",
            "requested_distillate_cut_fraction",
            "actual_distillate_cut_fraction",
            "cut_endpoint_met",
            "limiting_constraint",
            "bubble_pressure_margin_Pa",
            "minimum_thermal_margin_K",
            "light_key_distillate_purity",
            "light_key_recovery",
            "heavy_key_bottoms_purity",
            "heavy_key_recovery",
            "total_reboiler_duty_J",
            "condenser_duty_J",
            "internal_vapor_rate_mol_s",
            "fenske_stage_residual",
            "minimum_reflux_ratio",
            "required_theoretical_stages",
            "installed_equilibrium_stage_margin",
            "warnings",
        ),
        failure_policy=(
            "reject invalid property or equipment domains; return explicit no-cut "
            "results for incomplete heating or below-bubble operation; reduce a "
            "feasible cut to the tightest declared capacity without losing material, "
            "energy, key-recovery, or convergence diagnostics"
        ),
        provenance=(
            "ChemWorld reference-validated VLE/Fenske shortcut split engine",
            "WF-60 sensible/latent, total-condenser, and internal-vapor capacity envelope",
            "binary Fenske/Underwood diagnostics when their declared domain is satisfied",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_TRAY_COLUMN_PATH}, "
                f"{IDAES_CONDENSER_PATH}, and {IDAES_REBOILER_PATH} convention boundary"
            ),
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class DutyLimitedDistillationProvider:
    @property
    def model_contract(self) -> ModelProviderContract:
        return duty_limited_distillation_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(inputs.get("request"), DutyLimitedDistillationRequest):
            return ("request must be a DutyLimitedDistillationRequest",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return _failed_result("; ".join(violations), self.model_contract.provenance)
        request = inputs["request"]
        try:
            result = simulate_duty_limited_distillation(request)
        except (RuntimeError, ValueError) as error:
            return _failed_result(str(error), self.model_contract.provenance)
        return ModelProviderResult(
            outputs={"distillation_result": result.to_dict()},
            diagnostics={
                "material_balance_error_mol": result.material_balance_error_mol,
                "energy_balance_error_J": result.energy_balance_error_J,
                "requested_distillate_cut_fraction": (result.requested_distillate_cut_fraction),
                "actual_distillate_cut_fraction": (result.actual_distillate_cut_fraction),
                "cut_endpoint_met": result.cut_endpoint_met,
                "limiting_constraint": result.limiting_constraint,
                "bubble_pressure_margin_Pa": result.bubble_pressure_margin_Pa,
                "minimum_thermal_margin_K": result.minimum_thermal_margin_K,
                "light_key_distillate_purity": (result.light_key_distillate_purity),
                "light_key_recovery": result.light_key_recovery,
                "heavy_key_bottoms_purity": result.heavy_key_bottoms_purity,
                "heavy_key_recovery": result.heavy_key_recovery,
                "total_reboiler_duty_J": result.total_reboiler_duty_J,
                "condenser_duty_J": result.condenser_duty_J,
                "internal_vapor_rate_mol_s": result.internal_vapor_rate_mol_s,
                "fenske_stage_residual": result.fenske_stage_residual,
                "minimum_reflux_ratio": result.minimum_reflux_ratio,
                "required_theoretical_stages": (result.required_theoretical_stages),
                "installed_equilibrium_stage_margin": (result.installed_equilibrium_stage_margin),
                "warnings": list(result.warnings),
            },
            warnings=result.warnings,
            provenance=result.provenance,
        )


def duty_limited_distillation_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id=OWNER_WORKSTREAM,
        adapter_version="0.2",
        owner_workstream=OWNER_WORKSTREAM,
        provider_contract=duty_limited_distillation_provider_contract(),
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
            "energy_balance_error_J": None,
            "requested_distillate_cut_fraction": None,
            "actual_distillate_cut_fraction": None,
            "cut_endpoint_met": False,
            "limiting_constraint": None,
            "bubble_pressure_margin_Pa": None,
            "minimum_thermal_margin_K": None,
            "light_key_distillate_purity": None,
            "light_key_recovery": None,
            "heavy_key_bottoms_purity": None,
            "heavy_key_recovery": None,
            "total_reboiler_duty_J": None,
            "condenser_duty_J": None,
            "internal_vapor_rate_mol_s": None,
            "fenske_stage_residual": None,
            "minimum_reflux_ratio": None,
            "required_theoretical_stages": None,
            "installed_equilibrium_stage_margin": None,
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
    "DutyLimitedDistillationProvider",
    "duty_limited_distillation_adapter_manifest",
    "duty_limited_distillation_provider_contract",
]
