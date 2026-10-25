"""Real mean inventories, porous charge fields and the single coupled residual."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import F
from chemworld.physchem.battery_newman_tobias import (
    AveragedElectrodeSpec,
    BatteryNewmanTobiasSpec,
    advance_newman_tobias,
    newman_tobias_control_mode,
    newman_tobias_fields,
    newman_tobias_spatial_profile,
    newman_tobias_system,
    validate_newman_tobias_state,
)
from chemworld.physchem.battery_porous_electrical import PorousElectrodeGeometrySpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def model(**changes):
    args = {"geometry": PorousElectrodeGeometrySpec(cells=(3, 2, 4)), "maximum_current_A": 0.4}
    args.update(changes)
    return BatteryNewmanTobiasSpec(**args)


def program(value=0.2, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def test_mean_inventory_and_json_codec_never_create_radial_or_salt_states():
    spec = model()
    assert tuple(spec.initial_state().amounts_mol) == ("negative:Li", "positive:Li")
    assert spec.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    assert not hasattr(spec.negative, "cells") and not hasattr(spec.negative, "diffusivity_m2_s")
    _ = spec.electrical_maps  # Cached derived maps are not serialized.
    payload = spec.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = BatteryNewmanTobiasSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored == spec and restored.initial_state() == spec.initial_state()
    assert spec.volume_l == pytest.approx(0.0045)


@pytest.mark.parametrize("current", [0.0, 0.2, -0.2])
def test_actual_spatial_bv_and_exact_charge_boundary_conditions(current):
    spec = model()
    f = newman_tobias_fields(spec, spec.initial_state(), current_A=current)
    q, g = f["reaction_current_A"], spec.geometry
    assert sum(q[:3]) == pytest.approx(current, abs=1e-14)
    assert sum(q[3:]) == pytest.approx(-current, abs=1e-14)
    np.testing.assert_allclose(
        np.diff(f["electrolyte_face_current_A_m2"]) * g.area_m2,
        np.r_[q[:3], 0, 0, q[3:]],
        atol=2e-16,
    )
    assert f["electrolyte_face_current_A_m2"][0] == 0
    assert f["electrolyte_face_current_A_m2"][-1] == pytest.approx(0, abs=1e-13)
    assert f["negative_solid_face_current_A_m2"][-1] == pytest.approx(0, abs=1e-13)
    assert f["positive_solid_face_current_A_m2"][0] == pytest.approx(0, abs=1e-13)
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=2e-11)
    np.testing.assert_array_equal(f["surface_fraction"], f["average_fraction"])
    if current:
        assert np.ptp(q[:3]) > 1e-7  # Real porous distribution, not uniform SPM.
        assert (f["voltage_V"] - 3.83) * current < 0
    else:
        assert f["voltage_V"] == pytest.approx(3.83, abs=1e-12)
        np.testing.assert_allclose(f["solid_potential_V"][:3], 0, atol=1e-13)
        np.testing.assert_allclose(f["electrolyte_potential_V"], -0.13, atol=1e-13)


def test_spatial_profile_is_json_safe_and_keeps_unique_inventory_boundaries():
    spec = model()
    state = spec.initial_state()
    before = state
    profile = newman_tobias_spatial_profile(spec, state, current_A=0.2)
    assert profile["model_family"] == "newman-tobias-isothermal-v1"
    assert profile["thickness_cells"] == [3, 2, 4]
    assert len(profile["cell_labels"]) == 7
    assert set(profile["reaction_current_A"]) == set(profile["cell_labels"])
    assert profile["charge_balance_A"]["negative_reaction_sum_A"] == pytest.approx(0.2)
    assert profile["charge_balance_A"]["positive_reaction_sum_A"] == pytest.approx(-0.2)
    assert len(profile["electrolyte_face_current_A_m2"]) == 10
    assert len(profile["electrolyte_potential_V"]) == 9
    assert len(profile["solid_face_current_A_m2"]["negative"]) == 4
    assert len(profile["solid_face_current_A_m2"]["positive"]) == 5
    json.dumps(profile, allow_nan=False)
    assert state == before


@pytest.mark.parametrize("current", [0.2, -0.2, 0.0])
def test_shared_time_faradaic_and_terminal_work_are_real(current):
    spec = model()
    before = spec.initial_state()
    result = advance_newman_tobias(spec, before, program(current), 3)
    after = result.final_state
    assert after.time_s == 3
    assert after.coordinates["discharged_charge_C"] == pytest.approx(3 * current, abs=2e-11)
    for name, sign in (("negative", -1), ("positive", 1)):
        assert F * (after.amounts_mol[name + ":Li"] - before.amounts_mol[name + ":Li"]) == (
            pytest.approx(sign * 3 * current, abs=2e-9)
        )
    assert sum(after.amounts_mol.values()) == pytest.approx(
        sum(before.amounts_mol.values()), abs=2e-16
    )
    if current:
        assert after.coordinates["delivered_work_J"] * current > 0
    else:
        assert after.amounts_mol == before.amounts_mol
        assert after.coordinates["delivered_work_J"] == 0
    assert after.temperature_K == spec.temperature_K
    assert "no NT heat closure" in result.solver_diagnostic["energy_scope"]


@pytest.mark.parametrize(
    "target,mode",
    [(3.8, "constant_voltage"), (2.6, "positive_current_limit"), (4.4, "negative_current_limit")],
)
def test_actual_finite_cv_and_consistent_control_change(target, mode):
    spec = model()
    # The accepted CC algebraic solution must not be reused after switching to CV.
    state = advance_newman_tobias(spec, spec.initial_state(), program(), 1).final_state
    control = program(target, "V")
    f = newman_tobias_fields(spec, state, program=control)
    assert newman_tobias_control_mode(spec, state, control)[0] == mode
    if mode == "constant_voltage":
        assert f["voltage_V"] == pytest.approx(target, abs=1e-11)
        assert abs(f["current_A"]) < spec.maximum_current_A
    else:
        expected = spec.maximum_current_A * (1 if mode.startswith("positive") else -1)
        assert f["current_A"] == pytest.approx(expected, abs=2e-12)
    final = advance_newman_tobias(spec, state, control, 2).final_state
    assert final.time_s == 3


@pytest.mark.parametrize("value,unit", [(0.2, "A"), (0.0, "A"), (3.8, "V"), (2.6, "V")])
def test_complete_analytic_jacobian_matches_actual_shared_rhs(value, unit):
    spec, control = model(resistance_ohm=0.1), program(value, unit)
    state = spec.initial_state()
    f = newman_tobias_fields(spec, state, program=control)
    closure = newman_tobias_system(spec, control).algebraic
    nn = spec.geometry.cells[0]
    algebraic = [
        *f["reaction_current_A"][: nn - 1],
        *f["reaction_current_A"][nn:-1],
        f["electrolyte_potential_V"][0],
        f["current_A"],
        f["voltage_V"],
    ]
    # Offset is a boundary value: reconstruct it from the exact linear map.
    from chemworld.physchem.battery_porous_electrical import PHI_E, linear_electrical

    guess = dict(zip(spec.algebraic_names, algebraic, strict=True))
    guess[PHI_E] = 0.0
    offset_zero = linear_electrical(spec.geometry, spec.reaction_keys, spec.resistance_ohm, guess)[
        2
    ]
    algebraic[-3] = f["electrolyte_potential_V"][0] - offset_zero[0]
    names = (*spec.inventory_keys, "temperature", *spec.coordinate_keys, *spec.algebraic_names)
    values = np.r_[list(state.amounts_mol.values()), state.temperature_K, 0.0, 0.0, algebraic]
    scales = np.r_[0.03, 0.03, 300, 1000, 4000, [0.4] * len(spec.reaction_keys), 1, 0.4, 1]

    def view(y):
        return CoupledStateView(
            dict(zip(spec.inventory_keys, y[:2], strict=True)),
            y[2],
            spec.volume_l,
            coordinates=dict(zip(spec.coordinate_keys, y[3:5], strict=True)),
            algebraic_values=dict(zip(spec.algebraic_names, y[5:], strict=True)),
        )

    system = newman_tobias_system(spec, control)

    def rhs(y):
        candidate = view(y)
        rates = assemble_coupled_rates(
            candidate,
            thermodynamics=system.thermodynamics,
            processes=system.processes,
            heat_flux=system.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in spec.inventory_keys],
            0.0,
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -closure.residual(candidate),
        ]

    actual = system.residual_rhs_jacobian(view(values), names).toarray() * scales
    expected = finite_jacobian(rhs, values, scales, relative_step=1e-6) * scales
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-8)


def test_high_conductivity_limit_matches_independent_uniform_butler_volmer():
    base = model()
    spec = replace(
        base,
        geometry=replace(base.geometry, conductivity_S_m=1e10, solid_conductivity_S_m=(1e10, 1e10)),
    )
    current = 0.2
    expected = spec.ocv("positive", 0.3) - spec.ocv("negative", 0.7)
    for name, sign, theta in (("negative", -1, 0.7), ("positive", -1, 0.3)):
        electrode = getattr(spec, name)
        j0 = electrode.exchange_current_scale_A_m2 * np.sqrt(theta * (1 - theta))
        expected += (
            sign
            * 2
            * 8.31446261815324
            * spec.temperature_K
            / F
            * np.arcsinh(current / (2 * electrode.area_m2 * j0))
        )
    f = newman_tobias_fields(spec, spec.initial_state(), current_A=current)
    assert f["voltage_V"] == pytest.approx(expected, abs=2e-10)
    np.testing.assert_allclose(f["reaction_current_A"][:3], current / 3, atol=1e-10)


def test_actual_inventory_cutoff_then_reverse_and_json_continuation():
    spec = model(initial_negative_fraction=0.0101, minimum_voltage_V=-10, maximum_voltage_V=10)
    initial = spec.initial_state()
    expected_time = F * spec.negative.capacity_mol * 0.0001 / 0.2
    result = advance_newman_tobias(spec, initial, program(), 10)
    state = result.final_state
    assert state.time_s == pytest.approx(expected_time, abs=2e-6)
    assert state.amounts_mol["negative:Li"] / spec.negative.capacity_mol == pytest.approx(
        0.01, abs=1e-10
    )
    resumed = advance_newman_tobias(spec, state, program(-0.1), 1).final_state
    assert resumed.amounts_mol["negative:Li"] > state.amounts_mol["negative:Li"]
    restored = replace(
        state,
        amounts_mol=json.loads(json.dumps(dict(state.amounts_mol))),
        coordinates=json.loads(json.dumps(dict(state.coordinates))),
    )
    assert advance_newman_tobias(spec, restored, program(-0.1), 1).final_state == resumed


@pytest.mark.parametrize(
    "change",
    [
        {"negative": {"cells": 1}},
        {"geometry": {"cells": (1, 2, 2)}},
        {"negative": AveragedElectrodeSpec(active_volume_m3=2e-6)},
        {"temperature_K": 0},
        {"electrolyte_concentration_mol_m3": 0},
        {"resistance_ohm": -1},
        {"initial_negative_fraction": 0.01},
    ],
)
def test_declared_physical_domain_refuses_invalid_or_fake_spatial_states(change):
    with pytest.raises((ValueError, TypeError)):
        model(**change)


def test_state_rejects_fake_salt_and_faradaic_drift():
    spec = model()
    state = spec.initial_state()
    with pytest.raises(ValueError, match="exactly two"):
        validate_newman_tobias_state(
            spec, replace(state, amounts_mol={**state.amounts_mol, "salt": 1.0})
        )
    with pytest.raises(ValueError, match="Faraday"):
        validate_newman_tobias_state(
            spec, replace(state, coordinates={**state.coordinates, "discharged_charge_C": 1.0})
        )
    with pytest.raises(ValueError, match="thermal"):
        newman_tobias_system(spec, program(), thermal_device=object())


def test_sourced_ocv_explicit_family_identity_and_codec():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    source = InputSource(
        DataSourceProvenance("nt-test", 0, "Synthetic OCV", "1", "CC0-1.0"), "synthetic"
    )
    curve = ElectrodeOCV(
        "negative", source, polynomial_V=(0.3, -0.2), supported_models=("NewmanTobias",)
    )
    spec = model(negative_ocv=curve)
    assert newman_tobias_fields(spec, spec.initial_state(), current_A=0)["voltage_V"] == (
        pytest.approx(3.80, abs=1e-12)
    )
    assert BatteryNewmanTobiasSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    with pytest.raises(ValueError, match="explicitly support"):
        model(negative_ocv=replace(curve, supported_models=("SPM",)))


def test_nonzero_contact_resistance_has_actual_voltage_and_work_effect():
    first, second = model(), model(resistance_ohm=0.2)
    i = 0.2
    a = newman_tobias_fields(first, first.initial_state(), current_A=i)
    b = newman_tobias_fields(second, second.initial_state(), current_A=i)
    assert a["voltage_V"] - b["voltage_V"] == pytest.approx(i * 0.2, abs=1e-12)
    wa = advance_newman_tobias(first, first.initial_state(), program(i), 2).final_state.coordinates[
        "delivered_work_J"
    ]
    wb = advance_newman_tobias(
        second, second.initial_state(), program(i), 2
    ).final_state.coordinates["delivered_work_J"]
    assert wa - wb == pytest.approx(i * i * 0.2 * 2, abs=2e-9)


def test_requested_current_on_immutable_view_does_not_reuse_another_operating_point():
    spec = model()
    state = advance_newman_tobias(spec, spec.initial_state(), program(0.2), 1).final_state
    view = CoupledStateView(
        state.amounts_mol,
        state.temperature_K,
        state.volume_L,
        state.time_s,
        coordinates=state.coordinates,
        algebraic_values=state.algebraic_values,
    )
    reversed_field = newman_tobias_fields(spec, view, current_A=-0.2)
    assert reversed_field["current_A"] == pytest.approx(-0.2, abs=1e-13)
    assert (
        reversed_field["voltage_V"] > newman_tobias_fields(spec, state, current_A=0.2)["voltage_V"]
    )


def test_voltage_cutoff_stops_actual_time_and_charge_before_requested_duration():
    base = model()
    initial_voltage = newman_tobias_fields(base, base.initial_state(), current_A=0.2)["voltage_V"]
    spec = replace(base, minimum_voltage_V=initial_voltage - 0.0001)
    result = advance_newman_tobias(spec, spec.initial_state(), program(0.2), 10)
    final = result.final_state
    assert 0 < final.time_s < 10
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0.2 * final.time_s, abs=1e-10)
    assert newman_tobias_fields(spec, final, current_A=0.2)["voltage_V"] == pytest.approx(
        spec.minimum_voltage_V, abs=2e-9
    )
