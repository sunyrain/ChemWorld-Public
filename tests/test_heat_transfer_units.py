from __future__ import annotations

import pytest

from chemworld.physchem import (
    FoulingEvolutionSpec,
    HeatTransferEquipmentSpec,
    MaturityLevel,
    PhaseChangeBoundarySpec,
    equipment_heat_transfer,
    transport_model_cards,
    validate_model_card,
)
from chemworld.physchem.heat_transfer_units import HeatTransferSurface


def _equipment(
    surface_type: HeatTransferSurface = "jacket",
    *,
    correction: float = 1.0,
    coverage: float = 1.0,
) -> HeatTransferEquipmentSpec:
    return HeatTransferEquipmentSpec(
        equipment_id=f"test_{surface_type}",
        surface_type=surface_type,
        area_m2=2.0,
        clean_overall_u_W_m2_K=300.0,
        geometry_correction_factor=correction,
        jacket_coverage_fraction=coverage,
        provenance_id="synthetic-equipment-card",
    )


def test_fouling_resistance_evolves_and_reduces_heat_transfer() -> None:
    fouling = FoulingEvolutionSpec(
        model_id="asymptotic_fouling",
        initial_resistance_m2_K_W=0.0,
        asymptotic_resistance_m2_K_W=0.002,
        rate_constant_per_s=1.0e-4,
        provenance_id="synthetic-fouling-case",
    )
    clean = equipment_heat_transfer(
        _equipment(),
        process_temperature_K=300.0,
        utility_temperature_K=360.0,
        process_heat_capacity_J_K=20_000.0,
        duration_s=60.0,
        fouling=fouling,
        elapsed_fouling_time_s=0.0,
    )
    fouled = equipment_heat_transfer(
        _equipment(),
        process_temperature_K=300.0,
        utility_temperature_K=360.0,
        process_heat_capacity_J_K=20_000.0,
        duration_s=60.0,
        fouling=fouling,
        elapsed_fouling_time_s=20_000.0,
    )

    assert fouling.resistance_at(20_000.0) > fouling.resistance_at(0.0)
    assert fouled.effective_overall_u_W_m2_K < clean.effective_overall_u_W_m2_K
    assert fouled.heat_energy_J < clean.heat_energy_J


def test_jacket_coil_and_shell_corrections_are_explicit() -> None:
    jacket = equipment_heat_transfer(
        _equipment("jacket", coverage=0.5),
        process_temperature_K=300.0,
        utility_temperature_K=350.0,
        process_heat_capacity_J_K=50_000.0,
        duration_s=30.0,
    )
    coil = equipment_heat_transfer(
        _equipment("coil", correction=1.2),
        process_temperature_K=300.0,
        utility_temperature_K=350.0,
        process_heat_capacity_J_K=50_000.0,
        duration_s=30.0,
    )
    shell = equipment_heat_transfer(
        _equipment("shell", correction=0.75),
        process_temperature_K=300.0,
        utility_temperature_K=350.0,
        process_heat_capacity_J_K=50_000.0,
        duration_s=30.0,
    )

    assert jacket.surface_correction_factor == pytest.approx(0.5)
    assert coil.surface_correction_factor == pytest.approx(1.2)
    assert shell.surface_correction_factor == pytest.approx(0.75)
    assert coil.conductance_W_K > shell.conductance_W_K > jacket.conductance_W_K


def test_boiling_path_closes_sensible_and_latent_energy_ledger() -> None:
    result = equipment_heat_transfer(
        _equipment("coil"),
        process_temperature_K=360.0,
        utility_temperature_K=410.0,
        process_heat_capacity_J_K=10_000.0,
        duration_s=600.0,
        phase_change=PhaseChangeBoundarySpec(
            mode="boiling",
            saturation_temperature_K=373.15,
            latent_heat_J_mol=40_650.0,
            available_phase_change_mol=2.0,
            provenance_id="water-like-boiling-boundary",
        ),
    )

    assert result.phase_changed_mol > 0.0
    assert result.latent_energy_J > 0.0
    assert result.sensible_energy_J > 0.0
    assert result.heat_energy_J == pytest.approx(
        result.sensible_energy_J + result.latent_energy_J
    )
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=1.0e-12)
    assert result.final_temperature_K >= 373.15


def test_condensation_path_has_negative_heat_and_balanced_latent_duty() -> None:
    result = equipment_heat_transfer(
        _equipment("shell", correction=0.85),
        process_temperature_K=390.0,
        utility_temperature_K=290.0,
        process_heat_capacity_J_K=12_000.0,
        duration_s=500.0,
        phase_change=PhaseChangeBoundarySpec(
            mode="condensation",
            saturation_temperature_K=373.15,
            latent_heat_J_mol=40_650.0,
            available_phase_change_mol=1.5,
            provenance_id="water-like-condensation-boundary",
        ),
    )

    assert result.phase_changed_mol > 0.0
    assert result.latent_energy_J < 0.0
    assert result.heat_energy_J < 0.0
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=1.0e-12)
    assert result.final_temperature_K <= 373.15


def test_phase_crossing_without_enabled_model_emits_warning() -> None:
    result = equipment_heat_transfer(
        _equipment(),
        process_temperature_K=350.0,
        utility_temperature_K=450.0,
        process_heat_capacity_J_K=5_000.0,
        duration_s=300.0,
        phase_change=PhaseChangeBoundarySpec(
            mode="none",
            saturation_temperature_K=373.15,
            latent_heat_J_mol=40_650.0,
            available_phase_change_mol=1.0,
            provenance_id="warning-only-boundary",
        ),
    )

    assert result.final_temperature_K > 373.15
    assert "boiling_possible_but_phase_change_not_enabled" in result.warnings
    assert result.phase_changed_mol == 0.0


def test_heat_transfer_contract_rejects_inconsistent_phase_direction() -> None:
    with pytest.raises(ValueError, match="above saturation"):
        equipment_heat_transfer(
            _equipment(),
            process_temperature_K=350.0,
            utility_temperature_K=360.0,
            process_heat_capacity_J_K=10_000.0,
            duration_s=60.0,
            phase_change=PhaseChangeBoundarySpec(
                mode="boiling",
                saturation_temperature_K=373.15,
                latent_heat_J_mol=40_650.0,
                available_phase_change_mol=1.0,
                provenance_id="invalid-boiling-boundary",
            ),
        )


def test_equipment_heat_transfer_model_card_is_auditable() -> None:
    card = {
        item.model_id: item for item in transport_model_cards()
    }["equipment_phase_change_heat_transfer_v1"]

    assert card.maturity is MaturityLevel.REFERENCE_VALIDATED
    assert validate_model_card(card) == []
