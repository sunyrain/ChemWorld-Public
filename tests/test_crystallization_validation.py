from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.crystallization_adapter_manifest import (
    OWNED_PATHS,
    CrystallizationConvergenceProvider,
    crystallization_convergence_adapter_manifest,
    crystallization_convergence_provider_contract,
)
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.crystallization_validation import (
    IDAES_BALANCE_PATH,
    IDAES_COMMIT,
    CrystallizationConvergenceSpec,
    CrystallizationGridCase,
    audit_crystallization_convergence,
    crystallization_convergence_model_card,
)
from chemworld.physchem.maturity import ModelAdapterManifest, validate_model_card


def _case(
    *,
    nucleation_coefficient: float = 2.0e7,
    seed_mass_g: float = 0.05,
) -> CrystallizationGridCase:
    return CrystallizationGridCase(
        feed_amounts_mol={"target": 1.0, "impurity": 0.10},
        target_component="target",
        impurity_component="impurity",
        solvent_volume_L=1.0,
        initial_temperature_K=320.0,
        final_temperature_K=280.0,
        duration_s=1800.0,
        solubility_curve=SolubilityCurveSpec(
            model_id="vanthoff_target_solubility",
            reference_solubility_mol_L=1.0,
            reference_temperature_K=320.0,
            dissolution_enthalpy_J_mol=20_000.0,
            minimum_temperature_K=275.0,
            maximum_temperature_K=325.0,
            provenance_id="synthetic-vanthoff-solubility-case",
        ),
        kinetics=CrystallizationKineticsSpec(
            model_id="compact_target_population_balance",
            primary_nucleation_coefficient_per_L_s=nucleation_coefficient,
            primary_nucleation_exponent=2.0,
            growth_coefficient_m_s=2.0e-8,
            growth_exponent=1.0,
            crystal_density_kg_m3=1200.0,
            target_molecular_weight_kg_mol=0.100,
            nucleus_diameter_m=8.0e-6,
            impurity_occlusion_mol_per_mol=0.02,
            supersaturation_occlusion_factor=0.5,
            fines_threshold_m=20.0e-6,
            provenance_id="synthetic-nucleation-growth-case",
        ),
        seed_mass_g=seed_mass_g,
        seed_diameter_m=100.0e-6,
    )


def test_refinement_audit_converges_and_closes_all_ledgers() -> None:
    report = audit_crystallization_convergence(_case())

    assert [point.time_steps for point in report.grid_points] == [60, 120, 240]
    assert report.material_closed
    assert report.step_ledger_closed
    assert report.particle_count_ledger_closed
    assert report.grid_converged
    assert report.passed
    assert report.warnings == ()
    assert max(point.material_balance_error_mol for point in report.grid_points) < 1e-10
    assert max(point.particle_count_ledger_relative_error for point in report.grid_points) < 1e-12


def test_strict_grid_policy_reports_nonconvergence_without_input_failure() -> None:
    spec = CrystallizationConvergenceSpec(
        base_time_steps=8,
        refinement_factors=(1, 2),
        max_recovery_relative_delta=1e-15,
        max_crystallized_relative_delta=1e-15,
        max_d50_relative_delta=1e-15,
    )
    report = audit_crystallization_convergence(_case(), spec=spec)

    assert not report.grid_converged
    assert not report.passed
    assert "time_grid_not_converged" in report.warnings
    assert report.material_closed
    assert report.step_ledger_closed


def test_zero_population_keeps_ledgers_closed_and_is_explicit() -> None:
    report = audit_crystallization_convergence(_case(nucleation_coefficient=0.0, seed_mass_g=0.0))

    assert report.material_closed
    assert report.step_ledger_closed
    assert report.particle_count_ledger_closed
    assert report.grid_converged
    assert report.passed
    assert report.grid_points[-1].total_particle_count == 0.0
    assert "finest_grid_has_no_crystal_population" in report.warnings


def test_case_and_result_hashes_are_deterministic() -> None:
    left = audit_crystallization_convergence(_case())
    right = audit_crystallization_convergence(_case())

    assert left.to_dict() == right.to_dict()
    assert left.case_sha256 == right.case_sha256
    assert [point.result_sha256 for point in left.grid_points] == [
        point.result_sha256 for point in right.grid_points
    ]
    assert len(left.case_sha256) == 64


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"base_time_steps": 1}, "base_time_steps"),
        ({"refinement_factors": (1,)}, "at least two"),
        ({"refinement_factors": (2, 4)}, "begin with one"),
        ({"refinement_factors": (1, 4, 2)}, "strictly increasing"),
        ({"max_d50_relative_delta": 0.0}, "finite and positive"),
    ],
)
def test_convergence_spec_rejects_invalid_policies(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CrystallizationConvergenceSpec(**changes)


def test_provider_distinguishes_invalid_inputs_from_failed_audits() -> None:
    provider = CrystallizationConvergenceProvider()
    invalid = provider.evaluate({"case": object(), "audit_spec": None})
    strict = CrystallizationConvergenceSpec(
        base_time_steps=8,
        refinement_factors=(1, 2),
        max_recovery_relative_delta=1e-15,
        max_crystallized_relative_delta=1e-15,
        max_d50_relative_delta=1e-15,
    )
    nonconverged = provider.evaluate({"case": _case(), "audit_spec": strict})

    assert not invalid.success
    assert "CrystallizationGridCase" in (invalid.failure_reason or "")
    assert nonconverged.success
    assert nonconverged.diagnostics["passed"] is False
    assert nonconverged.outputs["report"]["passed"] is False


def test_underlying_domain_failure_becomes_provider_failure() -> None:
    provider = CrystallizationConvergenceProvider()
    invalid_case = replace(_case(), final_temperature_K=330.0)
    result = provider.evaluate({"case": invalid_case, "audit_spec": None})

    assert not result.success
    assert "cannot exceed" in (result.failure_reason or "")


def test_provider_contract_and_adapter_manifest_are_hash_bound() -> None:
    contract = crystallization_convergence_provider_contract()
    manifest = crystallization_convergence_adapter_manifest()
    payload = manifest.to_dict()

    assert contract.role.value == "diagnostic"
    assert contract.intended_operations == ("cool_crystallize",)
    assert manifest.owned_paths == OWNED_PATHS
    assert ModelAdapterManifest.from_dict(payload).to_dict() == payload
    tampered = json.loads(json.dumps(payload))
    tampered["provider_contract"]["model_id"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelAdapterManifest.from_dict(tampered)


def test_model_card_limits_reference_claim_to_audit_identities() -> None:
    card = crystallization_convergence_model_card()

    assert validate_model_card(card) == []
    assert card.maturity.value == "reference_validated"
    assert any("not to PBM parameters" in note for note in card.model_limit_notes)
    assert any("neither raised nor lowered" in note for note in card.model_limit_notes)


def test_pinned_idaes_reference_boundary_is_auditable() -> None:
    root = Path(__file__).parents[1]
    reference_path = root / "reference_repos" / "idaes-pse" / IDAES_BALANCE_PATH

    assert IDAES_COMMIT == "eed7cebc3d99be616ee7ead203cecaee9f81ac01"
    if not reference_path.is_file():
        pytest.skip("optional IDAES reference checkout is unavailable")
    assert any(
        "not a crystallization kinetics backend" in item
        for item in crystallization_convergence_provider_contract().provenance
    )
