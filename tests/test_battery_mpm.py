"""Many-particle current closure, true inventories and common runtime behavior."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_mpm import BatteryMPMSpec, mpm_fields
from chemworld.physchem.battery_particles import (
    ParticleSizeDistribution,
    lognormal_size_distribution,
)
from chemworld.physchem.battery_soh import soh_from_battery
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualEvaluator, finite_jacobian
from test_battery_materials import curve


def psd(radii=(5e-6, 15e-6), weights=(0.5, 0.5), basis="volume"):
    return ParticleSizeDistribution.from_weights(radii, weights, basis=basis, source=curve().source)


def model(**kwargs):
    return BatteryMPMSpec(
        negative=ParticleSpec(cells=6),
        positive=ParticleSpec(cells=6),
        negative_distribution=psd(),
        positive_distribution=psd(),
        **kwargs,
    )


def program(value=0.2, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def test_weight_bases_and_real_particle_capacity_area_and_json():
    d = psd(radii=(1e-6, 2e-6), weights=(1, 1), basis="number")
    assert d.volume_weights == pytest.approx((1 / 9, 8 / 9))
    assert d.weights("number") == pytest.approx((0.5, 0.5))
    assert d.weights("area") == pytest.approx((0.2, 0.8))
    s = replace(model(), negative_distribution=d)
    bins = [b for b in s.particle_bins if b[0] == "negative"]
    assert sum(p.capacity_mol for _, _, p, _, _ in bins) == pytest.approx(s.negative.capacity_mol)
    assert sum(p.area_m2 for _, _, p, _, _ in bins) == pytest.approx(
        3 * s.negative.active_volume_m3 * (1 / 9 / 1e-6 + 8 / 9 / 2e-6)
    )
    restored = battery_spec_from_dict(json.loads(json.dumps(s.to_dict(), allow_nan=False)))
    assert restored == s
    assert restored.initial_state() == s.initial_state()


def test_table_original_units_and_lognormal_parameters_survive():
    source = curve().source
    table = ScientificTable(
        "sizes",
        "parameters",
        (
            QuantityColumn("radius", "um", "parameter", (0, 100), "radius"),
            QuantityColumn("weight", "dimensionless", "parameter", (0, 100), "relative number"),
        ),
        (source,),
        tuple(
            ScientificRow(
                str(i),
                "synthetic-ocv",
                "size",
                "material-" + str(i),
                "parameters",
                {
                    "radius": InputValue(r, "um", 0.1, "synthetic uncertainty"),
                    "weight": InputValue(1.0, "dimensionless", None, "not supplied"),
                },
            )
            for i, r in enumerate((1.0, 2.0))
        ),
    )
    d = ParticleSizeDistribution.from_parameter_table(
        table, radius_column="radius", weight_column="weight", basis="number"
    )
    assert d.radii_m == pytest.approx((1e-6, 2e-6))
    assert d.parameter_rows[0][1].unit == "um"
    assert d.parameter_rows[0][1].uncertainty == 0.1
    assert ParticleSizeDistribution.from_dict(d.to_dict()) == d
    lognormal = lognormal_size_distribution(
        median_radius_m=1e-5,
        log_standard_deviation=0.3,
        lower_radius_m=3e-6,
        upper_radius_m=25e-6,
        nodes=8,
        source=source,
    )
    assert len(lognormal.radii_m) == 8
    assert sum(lognormal.volume_weights) == pytest.approx(1.0)
    assert ParticleSizeDistribution.from_dict(lognormal.to_dict()) == lognormal
    assert dict(lognormal.distribution_parameters)["median_radius"].value == 1e-5


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.2])
def test_single_size_is_same_physics_as_spm(current):
    particles = {"negative": ParticleSpec(cells=6), "positive": ParticleSpec(cells=6)}
    mpm, spm = BatteryMPMSpec(**particles), BatterySPMSpec(**particles)
    mf = mpm_fields(mpm, mpm.initial_state(), current_A=current)
    sr = battery_readout(spm, spm.initial_state(), current)
    assert mf["voltage_V"] == pytest.approx(sr.voltage_V, abs=1e-11)
    result = advance_battery(mpm, mpm.initial_state(), program(current), 2.0)
    original = advance_battery(spm, spm.initial_state(), program(current), 2.0)
    assert result.outcome == original.outcome == "success"
    assert battery_readout(mpm, result.final_state, current).voltage_V == pytest.approx(
        battery_readout(spm, original.final_state, current).voltage_V, abs=1e-8
    )


@pytest.mark.parametrize("current", [-0.2, 0.2])
def test_shared_potential_determines_each_size_current_and_conserves_charge(current):
    s = model()
    initial = s.initial_state()
    f = mpm_fields(s, initial, current_A=current)
    q = f["reaction_current_A"]
    assert sum(q[:2]) == pytest.approx(current, abs=1e-15)
    assert sum(q[2:]) == pytest.approx(-current, abs=1e-15)
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=1e-11)
    assert abs(q[0] / current - 0.5) > 0.01  # not prescribed by volume fractions
    result = advance_battery(s, initial, program(current), 3.0)
    assert result.outcome == "success"
    assert max(abs(v) for v in result.conservation_residuals.values()) < 1e-8
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        3 * current, abs=1e-9
    )
    assert soh_from_battery(s, result.final_state).spec.known_capacity_C == pytest.approx(F * 0.03)


def test_rest_can_redistribute_charge_between_sizes_without_terminal_current():
    s = model()
    initial = s.initial_state()
    amounts = dict(initial.amounts_mol)
    for name, i, p, keys, _ in s.particle_bins:
        if name == "negative":
            theta = 0.6 if i == 0 else 0.8
            amounts.update(
                zip(
                    keys,
                    p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3 * theta,
                    strict=True,
                )
            )
    state = replace(initial, amounts_mol=amounts)
    f = mpm_fields(s, state, current_A=0.0)
    assert f["reaction_current_A"][0] == pytest.approx(-f["reaction_current_A"][1])
    assert abs(f["reaction_current_A"][0]) > 1e-3
    result = advance_battery(s, state, program(0.0), 2.0)
    assert result.outcome == "success"
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(0.0, abs=1e-10)
    assert result.final_state.amounts_mol != state.amounts_mol


@pytest.mark.parametrize("voltage,limited", [(3.8, False), (2.6, True), (4.4, True)])
def test_cv_and_finite_source(voltage, limited):
    s = model(maximum_current_A=0.3)
    f = mpm_fields(s, s.initial_state(), program=program(voltage, "V"))
    if limited:
        assert abs(f["current_A"]) == pytest.approx(0.3)
    else:
        assert f["voltage_V"] == pytest.approx(voltage, abs=1e-10)
    result = advance_battery(s, s.initial_state(), program(voltage, "V"), 1.0)
    assert result.outcome == "success"


def test_true_analytic_residual_and_readonly_impedance():
    s = model()
    system, y = battery_impedance_system(s, s.initial_state())
    ev = ResidualEvaluator(system)
    actual = ev.derivative("state", 0.0, y, np.zeros_like(y)).toarray()
    expected = finite_jacobian(lambda z: ev.value(0.0, z, np.zeros_like(z)), y, system.scales)
    np.testing.assert_allclose(
        actual * system.scales[None, :] / system.residual_scales[:, None],
        expected * system.scales[None, :] / system.residual_scales[:, None],
        atol=2e-5,
        rtol=5e-6,
    )
    before = s.initial_state()
    spectrum = battery_impedance(s, [0.001, 0.01], state=before)
    assert spectrum.success
    assert all(z.real > 0 and z.imag < 0 for z in spectrum.impedance_ohm)
    assert before == s.initial_state()


@pytest.mark.parametrize("value,unit", [(0.2, "A"), (3.8, "V"), (2.6, "V"), (4.4, "V")])
def test_current_cv_and_source_limit_analytic_derivatives(value, unit):
    from chemworld.physchem.battery_mpm import _closure_jacobian, mpm_algebraic
    from chemworld.physchem.coupled_system import CoupledStateView

    s = model(maximum_current_A=0.3)
    p = program(value, unit)
    initial = s.initial_state()
    f = mpm_fields(s, initial, program=p)
    names = (*s.inventory_keys, "temperature", *s.algebraic_names)
    y = np.r_[
        [initial.amounts_mol[k] for k in s.inventory_keys],
        initial.temperature_K,
        [f["algebraic_values"][k] for k in s.algebraic_names],
    ]
    n = len(s.inventory_keys)
    scales = np.r_[[1e-3] * n, 300.0, [0.3] * len(s.reaction_keys), 1.0, 0.3, 1.0]

    def view(z):
        return CoupledStateView(
            dict(zip(s.inventory_keys, z[:n], strict=True)),
            z[n],
            initial.volume_L,
            coordinates=initial.coordinates,
            algebraic_values=dict(zip(s.algebraic_names, z[n + 1 :], strict=True)),
        )

    closure = mpm_algebraic(s, p)
    numeric = finite_jacobian(lambda z: closure.residual(view(z)), y, scales)
    analytic = _closure_jacobian(s, view(y), p, names).toarray()
    np.testing.assert_allclose(analytic * scales, numeric * scales, atol=2e-5, rtol=1e-5)


def test_same_chemical_energy_and_particle_losses_balance_terminal_work():
    from chemworld.physchem.battery_models import (
        battery_chemical_potentials,
        battery_dissipation_w,
        battery_system,
    )
    from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates

    s = model(negative_ocv=replace(curve(), supported_models=("MPM",)))
    initial = s.initial_state()
    f = mpm_fields(s, initial, current_A=0.2)
    view = CoupledStateView(
        initial.amounts_mol,
        initial.temperature_K,
        initial.volume_L,
        coordinates=initial.coordinates,
        algebraic_values=f["algebraic_values"],
    )
    system = battery_system(s, program(0.2))
    rates = assemble_coupled_rates(
        view,
        processes=system.processes,
        thermodynamics=system.thermodynamics,
        heat_flux=system.heat_flux,
        ports=system.ports,
        energy_ports=system.energy_ports,
        isothermal=True,
    )
    mu = battery_chemical_potentials(s, initial)
    dg = sum(mu[k] * rates.amount_rates_mol_s[k] for k in s.inventory_keys)
    losses = battery_dissipation_w(s, view, 0.2)
    assert min(losses.values()) >= -1e-12
    assert dg + 0.2 * f["voltage_V"] + sum(losses.values()) == pytest.approx(0.0, abs=1e-10)


def test_cutoff_checks_each_size_not_only_the_population_mean():
    s = model()
    initial = s.initial_state()
    amounts = dict(initial.amounts_mol)
    for name, i, p, keys, _ in s.particle_bins:
        if name == "negative" and i == 0:
            amounts.update(
                zip(
                    keys,
                    p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3 * 0.009,
                    strict=True,
                )
            )
    state = replace(initial, amounts_mol=amounts)
    result = advance_battery(s, state, program(0.0), 0.0)
    # With a connected population, rest can move charge, so explicitly inspect
    # the true current-coupled surface boundary, not just the average inventory.
    from chemworld.physchem.battery_models import battery_system

    system = battery_system(s, program(0.0))
    values = {event.event_id: event.value(state) for event in system.events}
    assert values["battery:negative:depleted"] <= 0
    assert result.outcome == "stopped"
    assert battery_readout(s, state, 0.0).negative_average_fraction > s.surface_margin
    assert values["battery:negative:depleted"] < (
        battery_readout(s, state, 0.0).negative_average_fraction - s.surface_margin
    )


def test_narrow_low_weight_tails_resolve_without_discarding_particle_inventories():
    # G02 fails at initial consistency, before any time advancement. Every
    # quadrature bin must remain a real inventory, including the tiny tails.
    distribution = lognormal_size_distribution(
        median_radius_m=1e-5,
        log_standard_deviation=0.12,
        lower_radius_m=5e-6,
        upper_radius_m=17e-6,
        nodes=32,
        source=curve().source,
        basis="area",
    )
    spec = BatteryMPMSpec(
        negative=ParticleSpec(cells=64),
        positive=ParticleSpec(cells=64),
        negative_distribution=distribution,
        positive_distribution=distribution,
    )
    initial = spec.initial_state()
    result = advance_battery(spec, initial, program(0.25), 0.0)
    assert result.outcome == "success"
    assert result.final_state.amounts_mol == initial.amounts_mol
    assert len(result.final_state.amounts_mol) == 2 * 32 * 64
    fields = mpm_fields(spec, initial, current_A=0.25)
    assert max(abs(value) for value in fields["bv_residual_V"]) < 1e-10
    assert len(fields["reaction_current_A"]) == 64
    assert sum(fields["reaction_current_A"][:32]) == pytest.approx(0.25, abs=1e-12)
    assert sum(fields["reaction_current_A"][32:]) == pytest.approx(-0.25, abs=1e-12)


def test_current_coordinates_keep_small_bins_and_pivot_each_electrode_independently():
    from chemworld.physchem.battery_mpm import mpm_algebraic

    spec = replace(
        model(),
        negative_distribution=psd((5e-6, 10e-6, 15e-6), (1e-9, 0.999998999, 1e-6)),
        positive_distribution=psd((5e-6, 15e-6), (0.8, 0.2)),
    )
    assert tuple(pivot for pivot, _, _ in spec.current_coordinates) == (1, 0)
    assert spec.reaction_keys == (
        "battery:negative:size-current:0_A",
        "battery:negative:size-current:2_A",
        "battery:positive:size-current:1_A",
    )
    closure = mpm_algebraic(spec, program())
    for name, (_, free, scales) in zip(
        ("negative", "positive"), spec.current_coordinates, strict=True
    ):
        areas = np.array([p.area_m2 for n, _, p, _, _ in spec.particle_bins if n == name])
        assert scales == pytest.approx(spec.maximum_current_A * areas[list(free)] / areas.sum())
    assert all(variable.scale > 0 for variable in closure.variables)
    for current in (-0.2, 0.0, 0.2):
        fields = mpm_fields(spec, spec.initial_state(), current_A=current)
        assert len(fields["reaction_current_A"]) == 5
        assert sum(fields["reaction_current_A"][:3]) == pytest.approx(current, abs=1e-12)
        assert sum(fields["reaction_current_A"][3:]) == pytest.approx(-current, abs=1e-12)
        np.testing.assert_allclose(fields["bv_residual_V"], 0, atol=1e-10)


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_world_meter_replay_and_cross_process_resume(tmp_path, noise):
    from chemworld.examples.battery_mpm import run, workflow
    from test_native_domains_end_to_end import test_snapshot_resume_in_middle_of_real_workflow

    result = run(tmp_path / (noise + ".jsonl"), spec=model(), noise=noise)
    assert result["steps"] == result["replay_steps"] == 9
    test_snapshot_resume_in_middle_of_real_workflow(
        "MPM", noise, tmp_path, recipe=workflow(model()), sort_keys=True
    )


@pytest.mark.parametrize(
    "radii,weights",
    [
        ((0, 1), (0.5, 0.5)),
        ((2, 1), (0.5, 0.5)),
        ((1, 2), (0.5, -0.5)),
        ((1, 2), (0.2, 0.2)),
        ((1, 1), (0.5, 0.5)),
    ],
)
def test_invalid_distributions_reject(radii, weights):
    with pytest.raises(ValueError):
        ParticleSizeDistribution(radii, weights, curve().source)
