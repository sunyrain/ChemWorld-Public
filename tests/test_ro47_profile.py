"""Local membrane and actual finite inventory closure tests."""

from dataclasses import replace

import pytest

from chemworld.physchem.distributed_ro import (
    advance_distributed_ro,
    distributed_ro_inventory,
    distributed_ro_profile,
    local_ro_flux,
)
from chemworld.physchem.transport import PumpSpec
from reference.ro47_fixtures import feed, spec


@pytest.mark.parametrize("temperature", [290, 310, 330])
def test_pure_water_has_one_pressure_drive(temperature):
    model = spec(temperature=temperature)
    flux = local_ro_flux(model, 0, 5e5)
    a = model.water_permeability.evaluate(temperature, 5e5, 0)
    assert flux["water_flux_m_s"] == pytest.approx(a * 4e5, rel=1e-10)
    assert flux["salt_flux_mol_m2_s"] == 0


def test_profile_recovery_pressure_and_original_ions():
    model = spec()
    p = distributed_ro_profile(model, feed(model), 10 / 6e7)
    assert 0 < p["water_recovery"] < 1
    assert p["stations"][-1]["bulk_concentration_mol_m3"] > 30
    assert p["pressure_drop_Pa"] > 0
    assert p["backpressure_dissipation_W"] > 0
    for k in model.properties.component_ids:
        assert p["permeate_mol_s"][k] + p["retentate_mol_s"][k] == pytest.approx(
            p["inlet_mol_s"][k]
        )
    assert p["permeate_mol_s"]["Na+"] == p["permeate_mol_s"]["Cl-"]


@pytest.mark.parametrize("prefilled", [False, True])
def test_finite_three_tanks_have_original_n_h_and_energy_closure(prefilled):
    model = spec()
    initial = distributed_ro_inventory(
        model,
        feed(model),
        permeate=feed(model, 70, 0.2) if prefilled else None,
        retentate=feed(model, 10, 0.3) if prefilled else None,
    )
    result, diagnostic = advance_distributed_ro(model, initial, 10 / 6e7, 60)
    assert max(map(abs, result.component_residuals_mol.values())) < 1e-10
    assert abs(result.enthalpy_residual_J) < 1e-5
    assert diagnostic["shaft_work_J"] - diagnostic["motor_loss_J"] + diagnostic[
        "bath_heat_into_liquid_J"
    ] == pytest.approx(diagnostic["liquid_enthalpy_change_J"], abs=1e-9)
    for n, h in zip(result.final_state.amounts_mol, result.final_state.enthalpies_J, strict=True):
        assert model.properties.enthalpy(n, 310, 1e5) == pytest.approx(h, abs=2e-5)


def test_finite_retained_outlet_capacity_and_blocked_pump_stop():
    model = replace(spec(), retentate_capacity_L=0.001)
    result, _ = advance_distributed_ro(
        model, distributed_ro_inventory(model, feed(model)), 10 / 6e7, 60
    )
    assert result.stopping_event == "retentate_full"
    assert 0 < result.actual_duration_s < 60
    blocked = replace(spec(), pump=PumpSpec(100, 1, 0.75))
    result, d = advance_distributed_ro(
        blocked, distributed_ro_inventory(blocked, feed(blocked)), 10 / 6e7, 60
    )
    assert result.stopping_event == "no_forward_pump_drive"
    assert result.actual_duration_s == d["shaft_work_J"] == 0


def test_local_limits_reject_instead_of_extrapolating():
    model = spec()
    for c, p in ((201, 5e5), (30, 90000)):
        with pytest.raises(ValueError):
            local_ro_flux(model, c, p)
    perfect = replace(model, salt_permeability=None)
    assert local_ro_flux(perfect, 30, 1e5)["water_flux_m_s"] == 0


def test_flux_bracket_can_approach_an_interface_boundary_without_extrapolation():
    model = spec()
    unrestricted = local_ro_flux(model, 30, 5e5)
    bounded = replace(
        model,
        maximum_interface_concentration_mol_m3=unrestricted["interface_concentration_mol_m3"]
        * 1.001,
    )
    assert local_ro_flux(bounded, 30, 5e5)["water_flux_m_s"] == pytest.approx(
        unrestricted["water_flux_m_s"], rel=1e-8
    )
