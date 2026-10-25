"""One thickness field, local inverse BV and the shared coupled energy residual."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_heat_stack import (
    BatteryHeatStackSpec,
    HeatStackCollector,
    HeatStackLayer,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BatterySPMSpec,
    ParticleSpec,
    battery_readout,
)
from chemworld.physchem.battery_spatial_thermal import (
    BatterySpatialThermalSPMSpec,
    advance_spatial_thermal,
    spatial_thermal_control_mode,
    spatial_thermal_field_ledger,
    spatial_thermal_fields,
    spatial_thermal_internal_energy_j,
    spatial_thermal_readout,
    spatial_thermal_system,
    validate_spatial_thermal_state,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def source():
    return InputSource(
        DataSourceProvenance(
            "spatial-thermal-test", 0, "Authored synthetic pouch thermal parameters", "1", "CC0-1.0"
        ),
        "synthetic",
    )


def model(*, cooling=True, **kwargs):
    stack = BatteryHeatStackSpec(
        HeatStackLayer(1e-4, 2e6, 0.002, 2, source()),
        HeatStackLayer(2.5e-5, 1.5e6, 0.0005, 2, source()),
        HeatStackLayer(1e-4, 2.5e6, 0.003, 3, source()),
        HeatStackCollector(1e-5, 3e6, 2 if cooling else 0, 1, 0.01, source()),
        HeatStackCollector(1.5e-5, 2.8e6, 1 if cooling else 0, 2, 0.02, source()),
        0.1,
        0.1,
        0.1 if cooling else 0,
        298.15,
        source(),
    )
    if not cooling:
        stack = replace(
            stack,
            negative_collector=replace(stack.negative_collector, tab_heat_transfer_W_m2_K=0),
            positive_collector=replace(stack.positive_collector, tab_heat_transfer_W_m2_K=0),
        )
    particle = ParticleSpec(
        radius_m=5e-6,
        diffusivity_m2_s=1e-13,
        active_volume_m3=6e-7,
        cells=5,
        exchange_current_scale_A_m2=4,
    )
    return BatterySpatialThermalSPMSpec(
        particle,
        replace(particle, exchange_current_scale_A_m2=2),
        stack,
        ElectrodeOCV("negative", source(), (0.2,)),
        ElectrodeOCV("positive", source(), (4.0,)),
        initial_negative_fraction=0.6,
        initial_positive_fraction=0.35,
        **kwargs,
    )


def program(value=0, unit="A", start=0):
    return ScalarControlProgram(((start, value),), unit)


def test_geometry_codec_and_only_capacity_mean_plus_signed_temperature_differences():
    s = model()
    stack = s.heat_stack
    values = np.linspace(295, 315, len(stack.node_ids))
    s = replace(s, initial_temperatures_K=tuple(values))
    decoded = BatterySpatialThermalSPMSpec.from_dict(json.loads(json.dumps(s.to_dict())))
    assert decoded == s
    assert decoded.initial_state() == s.initial_state()
    assert s.volume_l == pytest.approx(1000 * 0.01 * (2e-4 + 2.5e-5 + 2.5e-5))
    assert len(s.thermal_coordinate_keys) == len(values) - 1
    state = s.initial_state()
    np.testing.assert_allclose(stack.temperatures(state.temperature_K, state.coordinates), values)
    assert any(v < 0 for k, v in state.coordinates.items() if k in s.thermal_coordinate_keys)
    with pytest.raises(ValueError):
        stack.heat_capacities_J_K[0] = 0


@pytest.mark.parametrize("nonuniform", [False, True])
def test_initial_configuration_is_json_native_and_roundtrips_exactly(nonuniform):
    s = model()
    if nonuniform:
        s = replace(s, initial_temperatures_K=tuple(np.linspace(300, 310, 9)))
    payload = s.to_dict()
    decoded_payload = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded_payload == payload
    assert BatterySpatialThermalSPMSpec.from_dict(decoded_payload).to_dict() == payload
    assert payload["initial_temperatures_K"] is None or isinstance(
        payload["initial_temperatures_K"], list
    )


@pytest.mark.parametrize("cells", [6, 12])
def test_high_conductivity_uniform_temperature_has_exactly_zero_internal_heat(cells):
    s = model(cooling=False)
    stack = replace(
        s.heat_stack,
        **{
            name: replace(getattr(s.heat_stack, name), cells=cells, conductivity_W_m_K=100.0)
            for name in ("negative", "separator", "positive")
        },
    )
    s = replace(s, heat_stack=stack, temperature_K=310.0)
    state = s.initial_state()
    assert all(state.coordinates[k] == 0 for k in s.thermal_coordinate_keys)
    np.testing.assert_array_equal(stack.conductive_heat_W(state.coordinates), 0)
    fields = spatial_thermal_fields(s, state, current_A=0)
    np.testing.assert_array_equal(fields["local_temperature_rates_K_s"], 0)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *BATTERY_ALGEBRAIC)
    view = CoupledStateView(
        state.amounts_mol,
        state.temperature_K,
        state.volume_L,
        coordinates=state.coordinates,
        algebraic_values={BATTERY_ALGEBRAIC[0]: 0.0, BATTERY_ALGEBRAIC[1]: 3.8},
    )
    jac = spatial_thermal_system(s, program()).residual_rhs_jacobian(view, names).toarray()
    np.testing.assert_array_equal(jac[:, names.index("temperature")], 0)


def test_high_conductivity_internal_heat_is_invariant_to_absolute_temperature_translation():
    s = model(cooling=False)
    stack = replace(
        s.heat_stack,
        **{
            name: replace(getattr(s.heat_stack, name), conductivity_W_m_K=100.0)
            for name in ("negative", "separator", "positive")
        },
    )
    s = replace(s, heat_stack=stack)
    state = s.initial_state()
    differences = dict(
        zip(
            s.thermal_coordinate_keys,
            np.linspace(-0.001, 0.001, len(s.thermal_coordinate_keys)),
            strict=True,
        )
    )
    state = replace(state, coordinates={**state.coordinates, **differences})
    shifted = replace(state, temperature_K=state.temperature_K + 30)
    before = spatial_thermal_fields(s, state, current_A=0)
    after = spatial_thermal_fields(s, shifted, current_A=0)
    np.testing.assert_array_equal(
        before["local_temperature_rates_K_s"], after["local_temperature_rates_K_s"]
    )
    heat = stack.conductive_heat_W(state.coordinates)
    assert abs(sum(heat)) < 2e-12


def test_high_conductivity_short_cooling_uses_same_stiff_residual_and_matrix_limit():
    s = model()
    stack = replace(
        s.heat_stack,
        **{
            name: replace(getattr(s.heat_stack, name), conductivity_W_m_K=100.0)
            for name in ("negative", "separator", "positive")
        },
    )
    s = replace(s, heat_stack=stack, temperature_K=310.0)
    system = spatial_thermal_system(s, program())
    system = replace(system, policy=replace(system.policy, rtol=1e-9, atol=1e-11))
    result = system.advance(s.initial_state(), 0.2)
    operator = (
        stack.conduction_matrix_W_K - np.diag(stack.loss_conductance_W_K)
    ) / stack.heat_capacities_J_K[:, None]
    expected = stack.ambient_temperature_K + expm(0.2 * operator) @ np.full(
        len(stack.node_ids), 310 - stack.ambient_temperature_K
    )
    actual = stack.temperatures(result.final_state.temperature_K, result.final_state.coordinates)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=0)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("invalid", ["active-volume", "nonconstant", "initial-hot", "source"])
def test_explicit_scientific_scope_rejects_invalid_declarations(invalid):
    s = model()
    with pytest.raises(ValueError):
        if invalid == "active-volume":
            replace(s, negative=replace(s.negative, active_volume_m3=1.1e-6))
        elif invalid == "nonconstant":
            replace(s, negative_ocv=ElectrodeOCV("negative", source(), (0.2, -0.1)))
        elif invalid == "initial-hot":
            replace(s, initial_temperatures_K=(500,) * len(s.heat_stack.node_ids))
        else:
            replace(s.heat_stack.negative, source=None)


def test_conduction_network_and_rest_match_independent_linear_matrix_exponential():
    s = model(cooling=False)
    stack = s.heat_stack
    initial_t = np.array([308, 305, 300, 310, 299, 302, 303, 300, 297.0])
    s = replace(s, initial_temperatures_K=tuple(initial_t))
    # Construct the series half-cell conductances independently of production matrices.
    dx = np.repeat([1e-4 / 2, 2.5e-5 / 2, 1e-4 / 3], [2, 2, 3])
    conductivity = np.repeat([0.002, 0.0005, 0.003], [2, 2, 3])
    half = dx / (2 * 0.01 * conductivity)
    conductance = 1 / np.r_[half[0], half[:-1] + half[1:], half[-1]]
    operator = np.zeros((9, 9))
    for k, g in enumerate(conductance):
        operator[k : k + 2, k : k + 2] += g * np.array([[-1, 1], [1, -1]])
    capacity = np.r_[0.3, [1.0] * 2, [0.1875] * 2, [2.5 / 3] * 3, 0.42]
    expected = expm(operator / capacity[:, None] * 2) @ initial_t
    system = spatial_thermal_system(s, program())
    result = replace(system, policy=replace(system.policy, rtol=1e-10, atol=1e-13)).advance(
        s.initial_state(), 2
    )
    actual = stack.temperatures(result.final_state.temperature_K, result.final_state.coordinates)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=0)
    assert result.final_state.temperature_K == pytest.approx(s.temperature_K, abs=2e-10)
    assert result.final_state.coordinates["delivered_work_J"] == pytest.approx(0, abs=1e-13)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("current", [-0.5, 0, 0.5])
def test_uniform_temperature_instantaneous_readout_is_original_spm_limit(current):
    s = model()
    old = BatterySPMSpec(
        negative=s.negative,
        positive=s.positive,
        negative_ocv=s.negative_ocv,
        positive_ocv=s.positive_ocv,
        initial_negative_fraction=s.initial_negative_fraction,
        initial_positive_fraction=s.initial_positive_fraction,
        series_resistance_ohm=0,
    )
    a = spatial_thermal_readout(s, s.initial_state(), current)
    b = battery_readout(old, old.initial_state(), current)
    assert a.voltage_V == pytest.approx(b.voltage_V, abs=1e-14)
    assert a.negative_overpotential_V == pytest.approx(b.negative_overpotential_V, abs=1e-14)


def test_nonuniform_temperature_controls_local_bv_and_heat_not_only_bulk_mean():
    s = model()
    t = np.full(len(s.heat_stack.node_ids), 298.15)
    t[s.heat_stack.electrode_slice("negative")] += 30
    t[s.heat_stack.electrode_slice("positive")] -= 10
    nonuniform = replace(s, initial_temperatures_K=tuple(t))
    bulk = replace(s, temperature_K=nonuniform.temperature_K)
    a = spatial_thermal_fields(nonuniform, nonuniform.initial_state(), current_A=0.5)
    b = spatial_thermal_fields(bulk, bulk.initial_state(), current_A=0.5)
    assert abs(a["voltage_V"] - b["voltage_V"]) > 1e-4
    heat = sum(a["generated_heat_W"])
    assert heat == pytest.approx(0.5 * (3.8 - a["voltage_V"]), abs=1e-14)
    assert np.min(a["generated_heat_W"]) >= 0


def test_thermal_field_ledger_is_json_safe_and_keeps_environment_balance():
    s = model(cooling=True, initial_temperatures_K=tuple(np.linspace(295, 315, 9)))
    ledger = spatial_thermal_field_ledger(s, s.initial_state(), current_A=.5)
    restored = json.loads(json.dumps(ledger, allow_nan=False))
    assert len(restored["temperature_K"]) == len(s.heat_stack.node_ids)
    assert len(restored["local_temperature_rates_K_s"]) == len(s.heat_stack.node_ids)
    assert restored["ambient_temperature_K"] == pytest.approx(s.heat_stack.ambient_temperature_K)
    assert restored["boundary_heat_loss_W"] > 0
    assert restored["current_A"] == pytest.approx(.5)
    assert restored["minimum_temperature_K"] == pytest.approx(min(restored["temperature_K"]))
    assert restored["maximum_temperature_K"] == pytest.approx(max(restored["temperature_K"]))


@pytest.mark.parametrize("current", [-0.5, 0.5])
def test_cc_total_energy_terminal_work_and_boundary_receipt_close_once(current):
    s = model()
    initial = s.initial_state()
    result = advance_spatial_thermal(s, initial, program(current), 4)
    final = result.final_state
    delta = spatial_thermal_internal_energy_j(s, final) - spatial_thermal_internal_energy_j(
        s, initial
    )
    work = final.coordinates["delivered_work_J"]
    assert delta + work + final.heat_loss_J == pytest.approx(0, abs=2e-8)
    assert final.boundary_heat_J == pytest.approx(0, abs=1e-13)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(4 * current, abs=1e-10)
    assert np.ptp(s.heat_stack.temperatures(final.temperature_K, final.coordinates)) > 0.001
    assert "thickness-field" in result.solver_diagnostic["energy_scope"]


@pytest.mark.parametrize("value,unit", [(0, "A"), (0.5, "A"), (3.79, "V"), (2.6, "V")])
def test_full_shared_residual_jacobian_has_all_local_temperature_and_electrical_derivatives(
    value, unit
):
    s = model(initial_temperatures_K=tuple(np.linspace(295, 315, 9)))
    initial, p = s.initial_state(), program(value, unit)
    f = spatial_thermal_fields(s, initial, program=p)
    receipts = ("boundary_heat", "chemical_energy", "heat_loss", "work", "energy:battery:terminal")
    names = (*s.inventory_keys, "temperature", *receipts, *s.coordinate_keys, *BATTERY_ALGEBRAIC)
    n, nc = len(s.inventory_keys), len(s.coordinate_keys)
    y = np.r_[
        list(initial.amounts_mol.values()),
        initial.temperature_K,
        np.zeros(len(receipts)),
        list(initial.coordinates.values()),
        f["current_A"],
        f["voltage_V"],
    ]
    scales = np.r_[[1e-3] * n, 300, [10] * len(receipts), 1000, 4000, [10] * (nc - 2), 2, 1]
    offset = n + 1 + len(receipts)

    def view(z):
        return CoupledStateView(
            dict(zip(s.inventory_keys, z[:n], strict=True)),
            z[n],
            s.volume_l,
            coordinates=dict(zip(s.coordinate_keys, z[offset : offset + nc], strict=True)),
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, z[offset + nc :], strict=True)),
        )

    system = spatial_thermal_system(s, p)

    def rhs(z):
        v = view(z)
        rates = assemble_coupled_rates(
            v,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            isothermal=False,
            heat_flux=system.heat_flux,
            energy_ports=system.energy_ports,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in s.inventory_keys],
            rates.temperature_rate_K_s,
            rates.heater_W,
            rates.chemical_energy_W,
            rates.loss_W,
            rates.work_W,
            rates.port_energy_W["battery:terminal"],
            [rates.coordinate_rates[k] for k in s.coordinate_keys],
            -np.asarray(system.algebraic.residual(v)),
        ]

    actual = system.residual_rhs_jacobian(view(y), names).toarray()
    lower, upper = np.full(len(y), -np.inf), np.full(len(y), np.inf)
    lower[-2], upper[-2] = -s.maximum_current_A, s.maximum_current_A
    expected = finite_jacobian(rhs, y, scales, lower=lower, upper=upper)
    row_scale = np.r_[[1e-3] * n, 300, [4] * len(receipts), 1, 4, [10] * (nc - 2), 1, 1]
    np.testing.assert_allclose(
        actual * scales / row_scale[:, None],
        expected * scales / row_scale[:, None],
        atol=3e-5,
        rtol=5e-6,
    )


@pytest.mark.parametrize(
    "target,mode", [(3.79, "constant_voltage"), (2.6, "positive_current_limit")]
)
def test_cv_uses_real_temperature_field_and_finite_source(target, mode):
    s = model(maximum_current_A=0.4)
    p = program(target, "V")
    result = advance_spatial_thermal(s, s.initial_state(), p, 2)
    f = spatial_thermal_fields(s, result.final_state, program=p)
    assert spatial_thermal_control_mode(s, result.final_state, p)[0] == mode
    if mode == "constant_voltage":
        assert f["voltage_V"] == pytest.approx(target, abs=1e-9)
        assert 0 < f["current_A"] < 0.4
    else:
        assert f["current_A"] == pytest.approx(0.4, abs=1e-12)


def test_local_temperature_event_stops_before_bulk_reaches_bound():
    s = model(cooling=False)
    s = replace(s, heat_stack=replace(s.heat_stack, maximum_temperature_K=298.18))
    result = advance_spatial_thermal(s, s.initial_state(), program(1), 10)
    t = s.heat_stack.temperatures(result.final_state.temperature_K, result.final_state.coordinates)
    assert result.stopping_event.endswith(":maximum-temperature")
    assert result.final_state.time_s < 10
    assert max(t) == pytest.approx(298.18, abs=1e-8)
    assert result.final_state.temperature_K < 298.18 - 0.003
    validate_spatial_thermal_state(s, result.final_state)


def test_invalid_local_profile_and_undeclared_external_heat_contact_refuse():
    s = model()
    state = s.initial_state()
    coordinates = {**state.coordinates, s.thermal_coordinate_keys[1]: 100}
    with pytest.raises(ValueError, match="local temperature"):
        validate_spatial_thermal_state(s, replace(state, coordinates=coordinates))
    with pytest.raises(ValueError, match="external thermal contact"):
        spatial_thermal_system(s, program(), thermal_device=object())


def test_cold_environment_outside_operating_bounds_drives_real_local_cutoff():
    s = model()
    stack = replace(
        s.heat_stack,
        ambient_temperature_K=250,
        minimum_temperature_K=298.1,
        negative_collector=replace(s.heat_stack.negative_collector, face_heat_transfer_W_m2_K=200),
    )
    s = replace(s, heat_stack=stack)
    result = advance_spatial_thermal(s, s.initial_state(), program(), 2)
    assert result.stopping_event == "battery:thermal:negative-collector:minimum-temperature"
    assert 0 < result.final_state.time_s < 2
    local = stack.temperatures(result.final_state.temperature_K, result.final_state.coordinates)
    assert min(local) == pytest.approx(298.1, abs=1e-8)
    assert result.final_state.temperature_K > 298.14
    assert result.final_state.heat_loss_J > 0
