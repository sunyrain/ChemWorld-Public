"""The DFN solves local reactions together and retains one real inventory."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_dfn import BatteryDFNSpec, dfn_fields
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_control_current,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram


def model(**kwargs):
    return BatteryDFNSpec(negative=ParticleSpec(cells=6), positive=ParticleSpec(cells=6),
                          electrolyte=BatteryElectrolyteSpec(cells=(3, 2, 3)), **kwargs)


def program(value, unit="A"):
    return ScalarControlProgram(((0, value),), unit)


def test_dfn_defaults_match_qualified_mesh_without_rewriting_explicit_inputs():
    spec = BatteryDFNSpec()
    assert spec.negative.cells == spec.positive.cells == 64
    assert spec.electrolyte.cells == (24, 24, 24)
    assert len(spec.inventory_keys) == 2 * 24 * 64 + 3 * 24
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    explicit = model()
    decoded = battery_spec_from_dict(json.loads(json.dumps(explicit.to_dict())))
    assert decoded == explicit
    assert decoded.electrolyte.cells == (3, 2, 3)
    assert decoded.negative.cells == decoded.positive.cells == 6
    assert BatterySPMSpec().negative.cells == 40
    assert BatteryElectrolyteSpec().cells == (20, 20, 20)


def test_explicit_dfn_codec_and_distinct_spatial_inventory():
    spec = model()
    decoded = battery_spec_from_dict(json.loads(json.dumps(spec.to_dict(), sort_keys=True)))
    assert decoded == spec
    assert len(spec.particle_bins) == 6
    assert len(spec.inventory_keys) == 6 * 6 + 8
    assert sum(spec.initial_state().amounts_mol.values()) == pytest.approx(
        spec.negative.capacity_mol * 0.7 + spec.positive.capacity_mol * 0.3
        + sum(spec.electrolyte.initial_amounts().values()))
    with pytest.raises(ValueError, match="unknown battery"):
        battery_spec_from_dict({**spec.to_dict(), "model_family": "spm-fallback"})
    for change in ({"electrolyte": None}, {"numerical_form": "ode"},
                   {"thermal": BatteryThermalSpec()}):
        with pytest.raises(ValueError):
            replace(spec, **change)


def test_rest_satisfies_local_bv_current_boundaries_and_gauge():
    spec = model()
    f = dfn_fields(spec, spec.initial_state(), current_A=0)
    assert f["voltage_V"] == pytest.approx(3.83)
    np.testing.assert_allclose(f["reaction_current_A"], 0, atol=1e-12)
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=1e-12)
    np.testing.assert_allclose(f["solid_potential_V"][:3], 0, atol=1e-12)
    np.testing.assert_allclose(f["electrolyte_potential_V"], -0.13, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_salt_cutoff_includes_neumann_boundary_without_mutating_inventory(side):
    spec = model()
    e = spec.electrolyte
    state = spec.initial_state()
    amounts = dict(state.amounts_mol)
    widths = np.diff(e.mesh.faces_m)
    indices = (0, 1) if side == "left" else (-1, -2)
    h0, h1 = (widths[i] for i in indices)
    # c(x)=980+1e10*x^2 measured inward from an insulating collector.
    averages = (980 + 1e10 * h0**2 / 3,
                980 + 1e10 * ((h0 + h1)**3 - h0**3) / (3 * h1))
    for i, average in zip(indices, averages, strict=True):
        amounts[e.keys[i]] = average * e.diffusion.storage_volumes_m3[i]
    state = replace(state, amounts_mol=amounts)
    before = dict(state.amounts_mol)
    assert min(e.concentrations(state)) > 980
    assert e.minimum_resolved_concentration(state) == pytest.approx(980, abs=1e-10)
    assert dict(state.amounts_mol) == before


@pytest.mark.parametrize("current", [-0.25, 0.25])
def test_nonuniform_reaction_and_exact_cellwise_charge_balance(current):
    spec = model()
    f = dfn_fields(spec, spec.initial_state(), current_A=current)
    q = f["reaction_current_A"]
    assert sum(q[:3]) == pytest.approx(current, abs=1e-14)
    assert sum(q[3:]) == pytest.approx(-current, abs=1e-14)
    assert np.ptp(q[:3]) > 1e-6
    faces = f["electrolyte_face_current_A_m2"]
    np.testing.assert_allclose(np.diff(faces) * spec.electrolyte.area_m2,
                               np.r_[q[:3], 0, 0, q[3:]], atol=1e-14)
    assert faces[0] == 0
    assert faces[-1] == pytest.approx(0, abs=1e-12)
    assert f["negative_solid_face_current_A_m2"][-1] == pytest.approx(0, abs=1e-12)
    assert f["positive_solid_face_current_A_m2"][0] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=1e-11)
    assert (f["voltage_V"] - 3.83) * current < 0
    reading = battery_readout(spec, spec.initial_state(), current)
    assert reading.electrolyte_ohmic_V * current < 0
    assert reading.solid_ohmic_V * current < 0
    up = (spec.positive_ocv_intercept_V
          + spec.positive_ocv_slope_V * reading.positive_surface_fraction)
    un = (spec.negative_ocv_intercept_V
          + spec.negative_ocv_slope_V * reading.negative_surface_fraction)
    assert reading.voltage_V == pytest.approx(
        up - un + reading.positive_overpotential_V - reading.negative_overpotential_V
        + reading.concentration_overpotential_V + reading.electrolyte_ohmic_V
        + reading.solid_ohmic_V - current * spec.series_resistance_ohm, abs=1e-10)


@pytest.mark.parametrize("target,limited", [(3.8, False), (2.6, True), (4.4, True)])
def test_cv_closure_and_real_source_limits(target, limited):
    spec = model(maximum_current_A=0.4)
    f = dfn_fields(spec, spec.initial_state(), program=program(target, "V"))
    if limited:
        assert abs(f["current_A"]) == pytest.approx(0.4)
        assert (f["voltage_V"] - target) * f["current_A"] > 0
    else:
        assert f["voltage_V"] == pytest.approx(target, abs=1e-10)
        assert abs(f["current_A"]) < 0.4
    assert battery_control_current(spec, spec.initial_state(), program(target, "V")) == (
        pytest.approx(f["current_A"]))


@pytest.mark.parametrize("current", [0.0, 0.2, -0.2])
def test_coupled_advance_retains_salt_lithium_and_faradaic_charge(current):
    spec = model()
    initial = spec.initial_state()
    result = advance_battery(spec, initial, program(current), 10)
    assert result.outcome == "success"
    final = result.final_state
    for name, value in result.conservation_residuals.items():
        assert abs(value) < (1e-8 if name.endswith("_C") else 1e-12)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(current * 10, abs=1e-8)
    for name, sign in (("negative", -1), ("positive", 1)):
        keys = [k for n, _, _, ks, _ in spec.particle_bins if n == name for k in ks]
        difference = sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in keys)
        assert difference == pytest.approx(sign * current * 10 / F, abs=1e-12)
    if current:
        c = spec.electrolyte.concentrations(final)
        assert (c[0] - c[-1]) * current > 0
    else:
        assert final.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-14)
        assert battery_readout(spec, final, 0).voltage_V == pytest.approx(3.83)


@pytest.mark.parametrize("value,unit", [(0.2, "A"), (3.8, "V"), (2.6, "V"), (4.4, "V")])
def test_full_physical_rhs_jacobian_matches_independent_scaled_differences(value, unit):
    from chemworld.physchem.battery_dfn import dfn_system
    from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
    from chemworld.physchem.residual_system import finite_jacobian

    spec = model(maximum_current_A=0.4)
    system = dfn_system(spec, program(value, unit))
    initial = spec.initial_state()
    keys = spec.inventory_keys
    algebraic_names = spec.algebraic_names
    names = (*keys, "temperature", "boundary_heat", "chemical_energy", "heat_loss", "work",
             "discharged_charge_C", "delivered_work_J", *algebraic_names)
    count = len(names) - len(algebraic_names)
    guess = system.algebraic.initial_values(initial)
    values = np.r_[[initial.amounts_mol[k] for k in keys], initial.temperature_K, np.zeros(6),
                   [guess[k] for k in algebraic_names]]
    scales = np.r_[[system.inventory_scales_mol[k] for k in keys], 300, np.ones(6),
                   [v.scale for v in system.algebraic.variables]]

    def view(vector):
        return CoupledStateView(dict(zip(keys, vector[:len(keys)], strict=True)),
                                vector[len(keys)], initial.volume_L, coordinates={
                                    "discharged_charge_C": vector[count - 2],
                                    "delivered_work_J": vector[count - 1]},
                                algebraic_values=dict(zip(algebraic_names, vector[count:],
                                                          strict=True)))

    def physical_rhs(vector):
        state = view(vector)
        rates = assemble_coupled_rates(state, processes=system.processes,
                                      thermodynamics=system.thermodynamics,
                                      heat_flux=system.heat_flux, isothermal=True)
        return np.r_[[rates.amount_rates_mol_s[k] for k in keys], np.zeros(5),
                     [rates.coordinate_rates[k] for k in
                      ("discharged_charge_C", "delivered_work_J")],
                     -np.asarray(system.algebraic.residual(state))]

    actual = system.residual_rhs_jacobian(view(values), names).toarray() * scales
    expected = finite_jacobian(physical_rhs, values, scales, relative_step=1e-6) * scales
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def test_cached_sparse_assembly_matches_uncached_form_for_states_and_orderings():
    from scipy import sparse

    from chemworld.physchem.battery_dfn import _closure_jacobian, dfn_system
    from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC
    from chemworld.physchem.coupled_system import CoupledStateView
    from chemworld.physchem.finite_volume import ConservativeDiffusion

    spec = model()
    control = program(.2)
    system = dfn_system(spec, control)
    initial = spec.initial_state()
    names = (*spec.inventory_keys, "temperature", "discharged_charge_C", "delivered_work_J",
             *spec.algebraic_names)

    def uncached(state, identities):
        # Retained direct assembly is a bookkeeping oracle, not independent physics.
        index = {key: i for i, key in enumerate(identities)}
        jac = sparse.lil_matrix((len(identities), len(identities)))
        ac = [index[key] for key in spec.algebraic_names]
        qmap, _ = spec.electrical_maps
        for row, (_, _, particle, keys, _) in enumerate(spec.particle_bins):
            cols = [index[key] for key in keys]
            jac[np.ix_(cols, cols)] = ConservativeDiffusion(
                particle.mesh, particle.diffusivity_m2_s).inventory_jacobian_s_inv
            jac[index[keys[-1]], ac] = -qmap[row] / F
        e = spec.electrolyte
        cols = [index[key] for key in e.keys]
        jac[np.ix_(cols, cols)] = e.diffusion.inventory_jacobian_s_inv
        nn, ns, _ = e.cells
        source = np.vstack((qmap[:nn], np.zeros((ns, len(ac))), qmap[nn:]))
        jac[np.ix_(cols, ac)] = source * (1 - e.cation_transference) / F
        ci, cv = (index[key] for key in BATTERY_ALGEBRAIC)
        jac[index["discharged_charge_C"], ci] = 1
        jac[index["delivered_work_J"], ci] = state.algebraic_values[BATTERY_ALGEBRAIC[1]]
        jac[index["delivered_work_J"], cv] = state.algebraic_values[BATTERY_ALGEBRAIC[0]]
        jac[ac, :] = -_closure_jacobian(spec, state, control, identities)
        return jac.toarray()

    for shift in (0, .01, 0):
        amounts = {key: value * (1 + shift * (i % 3 - 1))
                   for i, (key, value) in enumerate(initial.amounts_mol.items())}
        changed = replace(initial, amounts_mol=amounts)
        guess = system.algebraic.initial_values(changed)
        guess[BATTERY_ALGEBRAIC[1]] += shift
        state = CoupledStateView(changed.amounts_mol, changed.temperature_K, changed.volume_L,
                                 coordinates=changed.coordinates, algebraic_values=guess)
        for identities in (names, names[::-1]):
            expected = uncached(state, identities)
            actual = system.residual_rhs_jacobian(state, identities)
            np.testing.assert_array_equal(actual.toarray(), expected)
            actual.data[:] = 0  # Returning a matrix must not expose the cached geometry.
            np.testing.assert_array_equal(
                system.residual_rhs_jacobian(state, identities).toarray(), expected)


@pytest.mark.parametrize("target,limit,mode", [
    (3.75, 2, "constant_voltage"), (3.7, 0.01, "positive_current_limit"),
    (4.1, 0.01, "negative_current_limit"),
])
def test_cv_time_advance_uses_measured_voltage_and_actual_limited_charge(target, limit, mode):
    spec = model(maximum_current_A=limit)
    control = program(target, "V")
    result = advance_battery(spec, spec.initial_state(), control, 2)
    assert result.outcome == "success"
    assert result.solver_diagnostic["battery_control_mode"] == mode
    current = battery_control_current(spec, result.final_state, control)
    if mode == "constant_voltage":
        assert battery_readout(spec, result.final_state, current).voltage_V == pytest.approx(
            target, abs=1e-10)
    else:
        assert abs(current) == pytest.approx(limit)
        assert abs(result.final_state.coordinates["discharged_charge_C"]) == pytest.approx(
            2 * limit)


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_native_public_meter_sorted_snapshot_and_equipment_material_ownership(noise):
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.battery import unpack_battery
    from chemworld.world.battery_world import battery_world_spec

    world = battery_world_spec(model())
    runtime = ChemWorldRuntime(composition=world, observation_noise_mode=noise)
    restored = ChemWorldRuntime(composition=world, observation_noise_mode=noise)
    try:
        runtime.reset(seed=13)
        for action in ({"operation": "configure_electrical_control", "current_A": 0.2},
                       {"operation": "wait", "duration_s": 2}):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored.restore(json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True)))
        for action in ({"operation": "wait", "duration_s": 1},
                       {"operation": "measure", "instrument": "battery_meter"}):
            a, b = runtime.execute(action), restored.execute(action)
            assert a.info["transaction_status"] == b.info["transaction_status"] == "committed"
        assert runtime.snapshot().to_dict() == restored.snapshot().to_dict()
        channels = {c.channel_id: c for c in a.observation.channels}
        assert channels["terminal_potential"].unit == "V"
        assert channels["electrical_current"].value == 0.2
        slot = unpack_battery(runtime._state)[1]
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert unpack_battery(runtime._state)[1] == slot
        assert unpack_battery(runtime._state)[2].model_id == "dfn-isothermal-v1"
    finally:
        runtime.close()
        restored.close()


def test_numerical_failure_cannot_partially_commit_dfn_inventory_or_clock(monkeypatch):
    import chemworld.runtime.battery as battery
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.battery_world import battery_world_spec

    runtime = ChemWorldRuntime(composition=battery_world_spec(model()))
    try:
        runtime.reset(seed=4)
        runtime.execute({"operation": "configure_electrical_control", "current_A": 0.2})
        before = runtime._state

        def fail(*args, **kwargs):
            raise RuntimeError("injected DFN numerical failure")

        monkeypatch.setattr(battery, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_dfn_sorted_snapshot_continues_in_a_new_process(noise, tmp_path):
    from chemworld.examples.battery_spme import workflow
    from test_native_domains_end_to_end import test_snapshot_resume_in_middle_of_real_workflow

    test_snapshot_resume_in_middle_of_real_workflow(
        "dfn", noise, tmp_path, recipe=workflow(model()), sort_keys=True)


@pytest.mark.parametrize("unit,value", [("A", 0.2), ("V", 3.75)])
def test_analytic_and_numeric_common_residual_have_same_physical_solution(unit, value):
    from chemworld.physchem.battery_dfn import dfn_system

    spec = model()
    system = dfn_system(spec, program(value, unit))
    analytic = system.advance(spec.initial_state(), 3)
    numeric = replace(system, residual_rhs_jacobian=None).advance(spec.initial_state(), 3)
    assert analytic.final_state.amounts_mol == pytest.approx(
        numeric.final_state.amounts_mol, rel=1e-8, abs=1e-13)
    assert analytic.final_state.coordinates == pytest.approx(
        numeric.final_state.coordinates, rel=1e-8, abs=1e-9)
    assert analytic.solver_diagnostic["residual"]["derivative_modes"]["state"] == "analytic"


@pytest.mark.parametrize("cutoff", ["voltage", "salt", "surface"])
def test_earliest_physical_cutoff_keeps_actual_time_and_resources(cutoff):
    spec = model()
    if cutoff == "voltage":
        start_v = battery_readout(spec, spec.initial_state(), 0.2).voltage_V
        spec = replace(spec, minimum_voltage_V=start_v - 0.001)
        event = "battery:minimum-voltage"
    elif cutoff == "salt":
        spec = replace(spec, electrolyte=replace(spec.electrolyte,
            minimum_concentration_mol_m3=990, diffusivity_m2_s=1e-12))
        event = "battery:electrolyte:depleted"
    else:
        spec = replace(spec, positive=replace(spec.positive, diffusivity_m2_s=1e-16))
        event = "battery:positive:saturated"
    result = advance_battery(spec, spec.initial_state(), program(0.2), 2000)
    assert result.outcome == "stopped"
    assert result.stopping_event == event
    assert 0 < result.actual_duration_s < 2000
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        0.2 * result.actual_duration_s, rel=1e-8)
    assert min(result.final_state.amounts_mol.values()) >= 0


def test_rest_allows_local_redistribution_with_zero_external_current():
    spec = model()
    initial = spec.initial_state()
    amounts = dict(initial.amounts_mol)
    _, _, _, first, _ = spec.particle_bins[0]
    _, _, _, second, _ = spec.particle_bins[1]
    for a, b in zip(first, second, strict=True):
        transfer = 0.03 * amounts[a]
        amounts[a] -= transfer
        amounts[b] += transfer
    nonuniform = replace(initial, amounts_mol=amounts)
    f = dfn_fields(spec, nonuniform, current_A=0)
    assert np.ptp(f["reaction_current_A"][:3]) > 1e-3
    assert sum(f["reaction_current_A"][:3]) == pytest.approx(0, abs=1e-12)
    final = advance_battery(spec, nonuniform, program(0), 10).final_state
    assert final.amounts_mol[first[-1]] > nonuniform.amounts_mol[first[-1]]
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_dfn_offline_example_records_and_replays_every_action(noise, tmp_path):
    from chemworld.examples.battery_dfn import run

    result = run(tmp_path / "dfn.jsonl", noise=noise, spec=model())
    assert result["model"] == "dfn-isothermal-v1"
    assert result["steps"] == result["replay_steps"] == 9


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_dfn_cutoff_is_the_same_commit_time_for_every_connected_domain(noise):
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.battery import unpack_battery
    from chemworld.runtime.scheduling import local_states

    base = model()
    voltage = battery_readout(base, base.initial_state(), 0.02).voltage_V
    spec = replace(base, minimum_voltage_V=voltage - 0.001)
    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=spec),
                               observation_noise_mode=noise)
    try:
        runtime.reset(seed=7)
        before = local_states(runtime._state)
        result = runtime.execute({"operation": "wait", "duration_s": 100})
        assert result.info["operation_outcome"] == "stopped", result.info
        elapsed = runtime._state.ledger.time_s
        assert 0 < elapsed < 100
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, elapsed)
        advancer = runtime.runtime.domain_services.reaction_thermal.advancer
        for key, local in local_states(runtime._state).items():
            expected = advancer.batch(before[key], {"duration_s": elapsed}, heat=False)
            assert local.state_slots == expected.state.state_slots
            assert local.species_amounts == expected.state.species_amounts
        _, _, restored_spec, battery, _ = unpack_battery(local_states(runtime._state)["cell"])
        assert restored_spec.model_id == "dfn-isothermal-v1"
        assert spec.electrolyte.concentrations(battery)[0] > 1000
    finally:
        runtime.close()


def test_dfn_sdist_wheel_works_outside_checkout_without_reference_backend(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=root, check=True,
                   capture_output=True, text=True)
    wheel = next(tmp_path.glob("*.whl"))
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import chemworld
assert sys.argv[1] in chemworld.__file__
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec
model = BatteryDFNSpec(negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4),
    electrolyte=BatteryElectrolyteSpec(cells=(2,2,2)))
session = ChemWorldRuntime(composition=battery_world_spec(model))
session.reset(seed=4)
for action in ({'operation':'configure_electrical_control','current_A':0.2},
               {'operation':'wait','duration_s':2}):
    assert session.execute(action).info['transaction_status'] == 'committed'
assert model.electrolyte.concentrations(unpack_battery(session._state)[3])[0] > 1000
assert 'pybamm' not in sys.modules
session.close()
print('native packaged DFN OK')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(wheel)], cwd=tmp_path,
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "native packaged DFN OK" in result.stdout
