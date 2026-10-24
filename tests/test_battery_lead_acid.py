"""Finite six-species LOQS: source physics, true events and full shared Jacobian."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_lead_acid import (
    LOQS_KEYS,
    LOQS_STOICHIOMETRY,
    BatteryLeadAcidSpec,
    LeadAcidGeometrySpec,
    LeadAcidMaterialSpec,
    advance_lead_acid,
    lead_acid_control_mode,
    lead_acid_fields,
    lead_acid_readout,
    lead_acid_system,
    validate_lead_acid_state,
)
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, R
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def program(value=0.2, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def model(**changes):
    base = BatteryLeadAcidSpec(maximum_current_A=0.5)
    # Nonzero products permit both directions in the independent Jacobian tests.
    n = np.asarray(base.initial_amounts_mol) + np.asarray(LOQS_STOICHIOMETRY) * 0.002
    return replace(base, initial_amounts_mol=tuple(n), **changes)


def test_initial_zero_products_are_real_and_codec_keeps_all_six_materials():
    spec = BatteryLeadAcidSpec()
    assert spec.initial_amounts_mol[1] == spec.initial_amounts_mol[3] == 0
    assert len(spec.inventory_species) == 6 and len(spec.species_formulas) == 5
    assert "Li" not in spec.species_formulas.values()
    payload = spec.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = BatteryLeadAcidSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored == spec and restored.initial_state() == spec.initial_state()
    assert lead_acid_fields(spec, spec.initial_state(), current_A=0)["porosities"] == (
        pytest.approx((0.5, 0.6, 0.5), abs=2e-15)
    )
    reading = lead_acid_readout(spec, spec.initial_state(), 0.2)
    assert reading.amounts_mol == dict(spec.initial_state().amounts_mol)
    assert not hasattr(reading, "negative_surface_fraction")


@pytest.mark.parametrize("current", [0.2, -0.2, 0.0])
def test_two_electron_bv_and_bode_molality_are_independently_evaluated(current):
    spec = model()
    f = lead_acid_fields(spec, spec.initial_state(), current_A=current)
    ca, cw = f["acid_concentration_mol_m3"], f["water_concentration_mol_m3"]
    m = ca / (cw * spec.material.water_molar_mass_kg_mol)
    log = np.log10(m)
    un = -np.polynomial.polynomial.polyval(log, [0.294, 0.074, 0.030, 0.031, 0.012])
    up = np.polynomial.polynomial.polyval(log, [1.628, 0.074, 0.033, 0.043, 0.022])
    cn = spec.material.reference_acid_concentration_mol_m3
    wn = spec.material.reference_water_concentration_mol_m3
    jn = 0.06 * ca / cn
    jp = 0.004 * (ca / cn) ** 2 * cw / wn
    an, ap = spec.geometry.reaction_areas_m2
    en = R * spec.temperature_K / F * np.arcsinh(current / (2 * an * jn))
    ep = R * spec.temperature_K / F * np.arcsinh(-current / (2 * ap * jp))
    assert f["voltage_V"] == pytest.approx(up - un + ep - en, abs=1e-14)
    assert 2 * jn * np.sinh(F * en / (R * spec.temperature_K)) == pytest.approx(current / an)
    assert 2 * jp * np.sinh(F * ep / (R * spec.temperature_K)) == pytest.approx(-current / ap)
    assert current * (f["voltage_V"] - up + un) <= 1e-15


@pytest.mark.parametrize("current", [0.2, -0.2, 0.0])
def test_six_real_inventories_volume_and_work_use_one_clock(current):
    spec = model()
    before = spec.initial_state()
    result = advance_lead_acid(spec, before, program(current), 5.0)
    after = result.final_state
    q = 5 * current
    assert after.time_s == 5
    np.testing.assert_allclose(
        [after.amounts_mol[k] for k in LOQS_KEYS],
        np.asarray(spec.initial_amounts_mol) + np.asarray(LOQS_STOICHIOMETRY) * q / (2 * F),
        rtol=0,
        atol=5e-16,
    )
    validate_lead_acid_state(spec, after)
    f = lead_acid_fields(spec, after, current_A=current)
    ve, vw = spec.material.molar_volumes_m3_mol[-2:]
    assert ve * after.amounts_mol[LOQS_KEYS[4]] + vw * after.amounts_mol[LOQS_KEYS[5]] == (
        pytest.approx(f["total_pore_volume_m3"], abs=1e-20)
    )
    assert after.coordinates["discharged_charge_C"] == pytest.approx(q, abs=1e-13)
    assert max(abs(v) for v in result.conservation_residuals.values()) < 1e-10
    assert after.temperature_K == spec.temperature_K
    assert result.energy_balance_residual_J is None
    assert "no chemical/thermal closure" in result.solver_diagnostic["energy_scope"]
    if current:
        v0 = lead_acid_fields(spec, before, current_A=current)["voltage_V"]
        assert after.coordinates["delivered_work_J"] == pytest.approx(
            q * (v0 + f["voltage_V"]) / 2, abs=2e-7
        )
    else:
        assert after.amounts_mol == before.amounts_mol
        assert after.coordinates["delivered_work_J"] == 0


@pytest.mark.parametrize(
    "target,mode",
    [(2.0, "constant_voltage"), (1.1, "positive_current_limit"), (2.9, "negative_current_limit")],
)
def test_real_cv_or_finite_current_limit_and_terminal_work(target, mode):
    spec = model()
    ctrl = program(target, "V")
    state = advance_lead_acid(spec, spec.initial_state(), program(), 1).final_state
    f = lead_acid_fields(spec, state, program=ctrl)
    assert lead_acid_control_mode(spec, state, ctrl)[0] == mode
    if mode == "constant_voltage":
        assert f["voltage_V"] == pytest.approx(target, abs=1e-12)
        assert abs(f["current_A"]) < spec.maximum_current_A
    else:
        assert f["current_A"] == pytest.approx(
            spec.maximum_current_A * (1 if mode == "positive_current_limit" else -1), abs=1e-12
        )
    after = advance_lead_acid(spec, state, ctrl, 3).final_state
    assert after.time_s == 4
    if mode == "constant_voltage":
        assert after.coordinates["delivered_work_J"] - state.coordinates["delivered_work_J"] == (
            pytest.approx(
                target
                * (
                    after.coordinates["discharged_charge_C"]
                    - state.coordinates["discharged_charge_C"]
                ),
                abs=2e-10,
            )
        )


@pytest.mark.parametrize(
    "value,unit", [(0.2, "A"), (-0.2, "A"), (0.0, "A"), (2.0, "V"), (1.1, "V"), (2.9, "V")]
)
def test_full_six_inventory_analytic_jacobian_matches_true_residual(value, unit):
    spec, ctrl = model(), program(value, unit)
    state = spec.initial_state()
    f = lead_acid_fields(spec, state, program=ctrl)
    names = (*LOQS_KEYS, "temperature", *spec.coordinate_keys, *BATTERY_ALGEBRAIC)
    values = np.r_[
        spec.initial_amounts_mol, spec.temperature_K, 0.0, 0.0, f["current_A"], f["voltage_V"]
    ]
    scales = np.r_[0.1, 0.01, 0.1, 0.01, 0.05, 0.5, 300.0, 5000.0, 10000.0, 0.5, 2.0]
    system = lead_acid_system(spec, ctrl)

    def view(y):
        return CoupledStateView(
            dict(zip(LOQS_KEYS, y[:6], strict=True)),
            y[6],
            spec.volume_l,
            coordinates=dict(zip(spec.coordinate_keys, y[7:9], strict=True)),
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y[9:], strict=True)),
        )

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
            [rates.amount_rates_mol_s[k] for k in LOQS_KEYS],
            0.0,
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -system.algebraic.residual(candidate),
        ]

    jac = system.residual_rhs_jacobian(view(values), names).toarray() * scales
    numerical = finite_jacobian(rhs, values, scales, relative_step=1e-6) * scales
    np.testing.assert_allclose(jac, numerical, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("index", [0, 2])
def test_true_solid_depletion_preserves_zero_state_and_reverses(index):
    base = model()
    n = list(base.initial_amounts_mol)
    # Keep pore volume fixed by transferring the removed solid volume into inert.
    old = n[index]
    n[index] = 1e-6
    material_volume = base.material.molar_volumes_m3_mol[index // 2]
    inert = list(base.geometry.inactive_solid_volumes_m3)
    inert[index // 2] += (old - n[index]) * material_volume
    spec = replace(
        base,
        geometry=replace(base.geometry, inactive_solid_volumes_m3=tuple(inert)),
        initial_amounts_mol=tuple(n),
    )
    result = advance_lead_acid(spec, spec.initial_state(), program(), 4)
    after = result.final_state
    assert result.stopping_event == "lead-acid:" + LOQS_KEYS[index] + ":depleted"
    assert after.time_s == pytest.approx(2 * F * n[index] / 0.2, abs=1e-8)
    assert after.amounts_mol[LOQS_KEYS[index]] == pytest.approx(0, abs=1e-15)
    stopped = advance_lead_acid(spec, after, program(), 1)
    assert stopped.actual_duration_s == 0
    restored = replace(
        after,
        amounts_mol=json.loads(json.dumps(dict(after.amounts_mol))),
        coordinates=json.loads(json.dumps(dict(after.coordinates))),
    )
    reverse = advance_lead_acid(spec, after, program(-0.1), 1)
    assert reverse.final_state.amounts_mol[LOQS_KEYS[index]] > after.amounts_mol[LOQS_KEYS[index]]
    assert advance_lead_acid(spec, restored, program(-0.1), 1).final_state == reverse.final_state


def test_zero_initial_product_does_not_prevent_discharge_but_prevents_charge():
    spec = BatteryLeadAcidSpec()
    initial = spec.initial_state()
    stopped = advance_lead_acid(spec, initial, program(-0.2), 2)
    assert stopped.actual_duration_s == 0
    assert stopped.stopping_event in {
        "lead-acid:negative:PbSO4:depleted",
        "lead-acid:positive:PbSO4:depleted",
    }
    forward = advance_lead_acid(spec, initial, program(0.2), 2)
    assert forward.actual_duration_s == 2
    assert forward.final_state.amounts_mol["negative:PbSO4"] > 0


def test_product_depletion_and_zero_porosity_are_real_recoverable_boundaries():
    base = model()
    # Return the real finite charge previously used to create both sulfate stocks.
    result = advance_lead_acid(base, base.initial_state(), program(-0.2), 2100)
    assert result.stopping_event in {
        "lead-acid:negative:PbSO4:depleted",
        "lead-acid:positive:PbSO4:depleted",
    }
    assert result.actual_duration_s == pytest.approx(2 * F * 0.002 / 0.2, abs=2e-7)
    assert advance_lead_acid(base, result.final_state, program(0.1), 1).actual_duration_s == 1
    # A tiny negative pore fraction makes epsilon=0 occur before acid or solids.
    n = list(base.initial_amounts_mol)
    vn, vs, vp = base.geometry.layer_volumes_m3
    pb, pbo2, sulfate, acid, water = base.material.molar_volumes_m3_mol
    pore_n = vn * 1e-5
    inert = (vn - pore_n - pb * n[0] - sulfate * n[1], base.geometry.inactive_solid_volumes_m3[1])
    pore_p = vp - inert[1] - pbo2 * n[2] - sulfate * n[3]
    pore = pore_n + vs * base.geometry.separator_porosity + pore_p
    n[4] = 4000 * pore
    n[5] = (pore - acid * n[4]) / water
    spec = replace(
        base,
        geometry=replace(base.geometry, inactive_solid_volumes_m3=inert),
        initial_amounts_mol=tuple(n),
        minimum_porosity=0,
    )
    result = advance_lead_acid(spec, spec.initial_state(), program(), 10)
    assert result.stopping_event == "lead-acid:negative:minimum-porosity"
    assert lead_acid_fields(spec, result.final_state, current_A=0)["porosities"][
        0
    ] == pytest.approx(0, abs=1e-12)
    assert advance_lead_acid(spec, result.final_state, program(-0.1), 1).actual_duration_s == 1


def test_actual_acid_and_porosity_cutoffs_are_derived_from_real_material():
    base = model()
    initial = base.initial_state()
    f = lead_acid_fields(base, initial, current_A=0.2)
    acid = f["acid_concentration_mol_m3"]
    spec = replace(base, minimum_acid_concentration_mol_m3=acid - 0.1)
    result = advance_lead_acid(spec, spec.initial_state(), program(), 10)
    assert result.stopping_event == "lead-acid:minimum-acid"
    assert lead_acid_fields(spec, result.final_state, current_A=0.2)[
        "acid_concentration_mol_m3"
    ] == (pytest.approx(acid - 0.1, abs=1e-8))
    assert advance_lead_acid(spec, result.final_state, program(-0.1), 1).actual_duration_s == 1
    lower = min(f["porosities"][0], f["porosities"][2]) - 1e-6
    spec = replace(base, minimum_porosity=lower)
    result = advance_lead_acid(spec, spec.initial_state(), program(), 10)
    assert result.stopping_event.endswith("minimum-porosity")
    assert advance_lead_acid(spec, result.final_state, program(-0.1), 1).actual_duration_s == 1


def test_voltage_stop_and_no_undeclared_heat_or_ohmic_drop():
    base = model()
    initial_v = lead_acid_fields(base, base.initial_state(), current_A=0.2)["voltage_V"]
    spec = replace(base, minimum_voltage_V=initial_v - 1e-5)
    result = advance_lead_acid(spec, spec.initial_state(), program(), 100)
    assert result.stopping_event == "battery:minimum-voltage"
    assert 0 < result.actual_duration_s < 100
    with pytest.raises(ValueError, match="isothermal"):
        lead_acid_system(spec, program(), thermal_device=object())
    assert spec.series_resistance_ohm == 0


@pytest.mark.parametrize("kind", ["incompatible", "water", "faraday", "custom", "geometry"])
def test_invalid_volume_or_inventory_declarations_refuse_without_repair(kind):
    base = model()
    with pytest.raises(ValueError):
        if kind == "incompatible":
            LeadAcidMaterialSpec(molar_volumes_m3_mol=(18e-6, 26e-6, 48e-6, 45e-6, 18e-6))
        elif kind == "water":
            n = list(base.initial_amounts_mol)
            n[5] += 0.01
            replace(base, initial_amounts_mol=tuple(n))
        elif kind == "faraday":
            state = base.initial_state()
            amounts = dict(state.amounts_mol)
            amounts[LOQS_KEYS[0]] -= 1e-5
            validate_lead_acid_state(base, replace(state, amounts_mol=amounts))
        elif kind == "custom":
            BatteryLeadAcidSpec(geometry=LeadAcidGeometrySpec(area_m2=0.02))
        else:
            LeadAcidGeometrySpec(inactive_solid_volumes_m3=(1, 1))
