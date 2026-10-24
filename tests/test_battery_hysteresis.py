"""Real OCP state evolution, common residual derivatives and bounded CV behavior."""

import json
from dataclasses import replace
from math import exp

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_chemical_energy_j,
    battery_control_current,
    battery_dissipation_w,
    battery_readout,
    battery_spec_from_dict,
    battery_system,
)
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_ocp import ElectrodeOCPSpec, hysteresis_key
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def source():
    return InputSource(
        DataSourceProvenance(
            "ocp-native-test", 0, "Authored synthetic OCP regression inputs", "1", "CC0-1.0"
        ),
        "synthetic",
    )


def ocp(name="negative", mode="axen", h=0.2, *, reverse=False, flat=False):
    a, b = (0.2, -0.1) if name == "negative" else (4.2, -0.8)
    b = 0 if flat else b
    width = -0.08 if reverse else 0.08
    branches = [
        ElectrodeOCV(name, source(), (a + offset, b), supported_models=("SPM", "MPM"))
        for offset in (-width / 2, width / 2)
    ]
    values = (
        {
            "initial_h": (h, "dimensionless"),
            "lithiation_decay": (20.0, "dimensionless"),
            "delithiation_decay": (40.0, "dimensionless"),
        }
        if mode == "axen"
        else {"collector_area_m2": (0.02, "m^2"), "sigmoid_slope_m2_A": (100.0, "m^2/A")}
    )
    table = ScientificTable(
        "ocp-test",
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (-1000, 1000), "synthetic fixed parameter")
            for k, (_, u) in values.items()
        ),
        (source(),),
        (
            ScientificRow(
                "row",
                source().provenance.source_id,
                "g",
                name,
                "parameters",
                {
                    k: InputValue(v, u, None, "Fixed synthetic input")
                    for k, (v, u) in values.items()
                },
            ),
        ),
    )
    return ElectrodeOCPSpec(mode, *branches, table, "row")


def model(*, mode="axen", mpm=False, h=0.2):
    p = ParticleSpec(cells=4)
    options = {
        "negative": p,
        "positive": p,
        "negative_ocp": ocp(h=h, mode=mode),
        "positive_ocp": ocp("positive", mode, -h),
    }
    if mpm:
        d = ParticleSizeDistribution.from_weights(
            (8e-6, 12e-6), (0.4, 0.6), basis="volume", source=source()
        )
        return BatteryMPMSpec(**options, negative_distribution=d, positive_distribution=d)
    return BatterySPMSpec(**options)


def program(current):
    return ScalarControlProgram(((0.0, current),), "A")


@pytest.mark.parametrize("mpm", [False, True])
def test_source_codec_unique_state_and_unchanged_legacy_payload(mpm):
    s = model(mpm=mpm)
    restored = battery_spec_from_dict(json.loads(json.dumps(s.to_dict(), allow_nan=False)))
    assert restored == s
    assert restored.initial_state() == s.initial_state()
    assert s.coordinate_keys == (
        "discharged_charge_C",
        "delivered_work_J",
        "battery:negative:hysteresis",
        "battery:positive:hysteresis",
    )
    assert s.negative_ocp.parameters.to_dict() == restored.negative_ocp.parameters.to_dict()
    old = replace(s, negative_ocp=None, positive_ocp=None)
    assert "negative_ocp" not in old.to_dict()
    assert len(old.coordinate_keys) == 2
    with pytest.raises(ValueError, match="fields"):
        ElectrodeOCPSpec.from_dict({**s.negative_ocp.to_dict(), "hidden_gain": 3})


@pytest.mark.parametrize("current", [0.2, -0.2])
@pytest.mark.parametrize("mpm", [False, True])
def test_axen_actual_charge_analytic_state_rest_reversal_and_total_electrode_capacity(current, mpm):
    s = model(mpm=mpm)
    initial = s.initial_state()
    total = sum(initial.amounts_mol.values())
    system = battery_system(s, program(current))
    a = replace(system, policy=replace(system.policy, rtol=1e-10, atol=1e-13)).advance(
        initial, 3.0).final_state
    for name, sign in (("negative", 1), ("positive", -1)):
        q = sign * current
        direction = np.sign(q)
        k = 40 if q > 0 else 20
        key = hysteresis_key(name)
        expected = direction + (initial.coordinates[key] - direction) * exp(
            -k * abs(q) * 3 / F / getattr(s, name).capacity_mol
        )
        assert a.coordinates[key] == pytest.approx(expected, abs=3e-10)
    assert sum(a.amounts_mol.values()) == pytest.approx(total, abs=1e-12)
    assert a.coordinates["discharged_charge_C"] == pytest.approx(3 * current, abs=1e-10)
    rest = advance_battery(s, a, program(0), 2.0).final_state
    for key in s.hysteresis_keys:
        assert rest.coordinates[key] == a.coordinates[key]
    reverse = advance_battery(s, rest, program(-current), 3.0).final_state
    assert reverse.coordinates["discharged_charge_C"] == pytest.approx(0, abs=1e-10)
    assert reverse.coordinates[s.hysteresis_keys[0]] != pytest.approx(
        initial.coordinates[s.hysteresis_keys[0]], abs=1e-6
    )


@pytest.mark.parametrize("h,current", [(-1.0, -0.2), (1.0, 0.2)])
@pytest.mark.parametrize("mpm", [False, True])
def test_valid_hysteresis_endpoints_are_not_false_stopping_events(h, current, mpm):
    s = model(mpm=mpm, h=h)
    result = advance_battery(s, s.initial_state(), program(current), 1.0)
    assert result.final_state.time_s == pytest.approx(1.0)
    assert result.final_state.coordinates[s.hysteresis_keys[0]] == h
    assert result.final_state.coordinates[s.hysteresis_keys[1]] == -h


@pytest.mark.parametrize("h", [-1.001, 1.001])
def test_illegal_accepted_h_state_is_rejected_not_clipped(h):
    s = model()
    a = s.initial_state()
    a = replace(a, coordinates={**a.coordinates, s.hysteresis_keys[0]: h})
    with pytest.raises(ValueError, match="outside"):
        advance_battery(s, a, program(0.2), 1.0)
    with pytest.raises(ValueError, match="outside"):
        battery_readout(s, a, 0.2)


@pytest.mark.parametrize("mpm", [False, True])
def test_sigmoid_uses_total_collector_density_and_actual_cv(mpm):
    s = model(mode="current_sigmoid", mpm=mpm)
    state = s.initial_state()
    # A small actual current keeps the independently evaluable sigmoid in transition.
    current = 2e-5
    value = s.operational_ocp("negative", 0.7, current, state)
    weight = 0.5 * (1 + np.tanh(-100 * current / 0.02))
    assert value == pytest.approx(
        weight * (0.2 - 0.04 - 0.1 * 0.7) + (1 - weight) * (0.2 + 0.04 - 0.1 * 0.7)
    )
    target = battery_readout(s, state, current).voltage_V
    cv = ScalarControlProgram(((0.0, target),), "V")
    assert battery_control_current(s, state, cv) == pytest.approx(current, abs=1e-10)
    final = advance_battery(s, state, cv, 0.2).final_state
    i = battery_control_current(s, final, cv)
    assert battery_readout(s, final, i).voltage_V == pytest.approx(target, abs=1e-9)


@pytest.mark.parametrize("mpm", [False, True])
def test_nonmonotone_sigmoid_cv_rejected_cc_retains_declared_branches(mpm):
    s = replace(
        model(mode="current_sigmoid", mpm=mpm),
        negative_ocp=ocp(mode="current_sigmoid", reverse=True),
    )
    state = s.initial_state()
    assert s.negative_ocp.derivatives("negative", 0.7, 0)[2] < 0
    assert np.isfinite(battery_readout(s, state, 0).voltage_V)
    with pytest.raises(ValueError, match="monotonicity"):
        battery_system(s, ScalarControlProgram(((0.0, 3.65),), "V"))


@pytest.mark.parametrize("mode", ["axen", "current_sigmoid"])
@pytest.mark.parametrize("mpm", [False, True])
def test_complete_common_residual_jacobian_matches_real_physics(mode, mpm):
    s = model(mode=mode, mpm=mpm)
    current = 0.02 if mode == "axen" else 2e-5
    coupled = battery_system(s, program(current))
    state = s.initial_state()
    material, coords = s.inventory_keys, s.coordinate_keys
    closure = coupled.algebraic
    names = tuple(v.name for v in closure.variables)
    identities = (*material, *coords, *names)
    initial = {**state.amounts_mol, **state.coordinates, **closure.initial_values(state)}
    y = np.array([initial[k] for k in identities])
    scales = np.array([max(abs(x), 1e-5) for x in y])
    scales[-len(names):] = [v.scale for v in closure.variables]

    def view(z):
        values = dict(zip(identities, z, strict=True))
        return CoupledStateView(
            {k: values[k] for k in material},
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates={k: values[k] for k in coords},
            algebraic_values={k: values[k] for k in names},
        )

    def rhs(z):
        candidate = view(z)
        rates = assemble_coupled_rates(
            candidate,
            processes=coupled.processes,
            thermodynamics=coupled.thermodynamics,
            heat_flux=coupled.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in material],
            [rates.coordinate_rates[k] for k in coords],
            -np.array(closure.residual(candidate)),
        ]

    analytic = coupled.residual_rhs_jacobian(view(y), identities).toarray()
    # This narrow sigmoid requires a genuinely local derivative step.
    numeric = finite_jacobian(rhs, y, scales, relative_step=1e-8)
    assert np.max(abs(analytic - numeric) / np.maximum(1.0, abs(analytic))) < 2e-5


def test_single_exact_old_behavior_and_equilibrium_curve_not_replaced():
    old = BatterySPMSpec(negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4))
    single = replace(old, negative_ocp=ElectrodeOCPSpec(), positive_ocp=ElectrodeOCPSpec())
    assert advance_battery(old, old.initial_state(), program(0.2), 1.0).final_state == (
        advance_battery(single, single.initial_state(), program(0.2), 1.0).final_state
    )
    directional = replace(old, negative_ocp=ocp())
    assert directional.ocv("negative", 0.7) == old.ocv("negative", 0.7)
    assert battery_chemical_energy_j(directional, directional.initial_state()) == (
        battery_chemical_energy_j(old, old.initial_state())
    )
    with pytest.raises(ValueError, match="hysteresis energy"):
        battery_dissipation_w(directional, directional.initial_state(), 0.2)
    with pytest.raises(ValueError, match="isothermal"):
        replace(directional, thermal=BatteryThermalSpec())


def test_flat_branch_cv_is_still_strictly_passive_through_butler_volmer():
    s = replace(
        model(mode="current_sigmoid"),
        negative_ocp=ocp(mode="current_sigmoid", flat=True),
        positive_ocp=ocp("positive", "current_sigmoid", flat=True),
    )
    target = battery_readout(s, s.initial_state(), 0.001).voltage_V
    assert battery_control_current(
        s, s.initial_state(), ScalarControlProgram(((0.0, target),), "V")
    ) == pytest.approx(0.001, abs=1e-10)
