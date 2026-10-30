"""SPMe is a declared physical option in the same battery and runtime."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_control_current,
    battery_readout,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


def model(**kwargs):
    return BatterySPMSpec(
        negative=ParticleSpec(cells=6), positive=ParticleSpec(cells=6),
        electrolyte=BatteryElectrolyteSpec(cells=(3, 3, 3)), **kwargs,
    )


def program(current):
    return ScalarControlProgram(((0, current),), "A")


def test_porous_diffusion_uses_storage_volume_and_interface_resistance():
    mesh = DiffusionMesh("planar", (0, 1, 3), area_m2=2)
    operator = ConservativeDiffusion(mesh, (2, 4), storage_fractions=(0.5, 0.25))
    np.testing.assert_array_equal(operator.storage_volumes_m3, [1, 1])
    # Interface resistance 0.5/2 + 1/4, area=2; concentrations 3 and 1.
    result = operator.evaluate([3, 1])
    np.testing.assert_allclose(result.face_rates_mol_s, [0, 8, 0])
    np.testing.assert_allclose(result.amount_rates_mol_s, [-8, 8])
    np.testing.assert_allclose(operator.inventory_jacobian_s_inv @ [3, 1], [-8, 8])
    blocked = ConservativeDiffusion(mesh, (0, 4), storage_fractions=(0.5, 0.25))
    np.testing.assert_array_equal(blocked.evaluate([3, 1]).amount_rates_mol_s, [0, 0])


@pytest.mark.parametrize("values", [(1,), (0, 1), (float("nan"), 1), (1.1, 1)])
def test_invalid_porous_storage_rejected(values):
    with pytest.raises(ValueError, match="storage fractions"):
        ConservativeDiffusion(DiffusionMesh.planar(1, 2), 1, storage_fractions=values)


@pytest.mark.parametrize("change", [
    {"model_version": "unknown"}, {"potential_form": "magic"}, {"cation_transference": 1},
    {"cation_transference": float("nan")}, {"cells": (2, True, 2)},
    {"porosities": (0.3, 0.5, 1)}, {"minimum_concentration_mol_m3": 1000},
    {"conductivity_S_m": 0}, {"diffusivity_m2_s": -1}, {"lengths_m": (1, 2)},
])
def test_invalid_electrolyte_spec_rejected(change):
    with pytest.raises(ValueError):
        BatteryElectrolyteSpec(**change)


def test_model_round_trip_and_unimplemented_couplings_do_not_fall_back():
    spec = model()
    assert BatterySPMSpec(**json.loads(json.dumps(spec.to_dict()))) == spec
    assert "electrolyte" not in BatterySPMSpec().to_dict()
    assert spec.model_id == "spme-isothermal-v1"
    with pytest.raises(ValueError, match="isothermal"):
        replace(spec, thermal=BatteryThermalSpec())
    with pytest.raises(ValueError, match="active particle volume"):
        replace(spec, negative=replace(spec.negative, active_volume_m3=1e-3))


@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_uniform_rest_is_stationary_and_lithium_is_not_duplicated(numerical_form):
    spec = model(numerical_form=numerical_form)
    initial = spec.initial_state()
    final = advance_battery(spec, initial, program(0), 20).final_state
    np.testing.assert_allclose(spec.electrolyte.concentrations(final), 1000, atol=1e-10)
    assert final.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-15)
    assert battery_readout(spec, final, 0).voltage_V == pytest.approx(3.83)
    expected = (spec.negative.capacity_mol * 0.7 + spec.positive.capacity_mol * 0.3
                + sum(spec.electrolyte.initial_amounts().values()))
    assert sum(initial.amounts_mol.values()) == pytest.approx(expected)


@pytest.mark.parametrize("current", [-0.2, 0.2])
def test_real_gradient_charge_salt_and_voltage_contributions(current):
    spec = model()
    start = spec.initial_state()
    result = advance_battery(spec, start, program(current), 50)
    assert result.outcome == "success"
    final = result.final_state
    c = spec.electrolyte.concentrations(final)
    assert (c[0] - c[-1]) * current > 0
    assert abs(result.conservation_residuals["electrolyte_salt_mol"]) < 1e-12
    assert abs(result.conservation_residuals["lithium_mol"]) < 1e-12
    for name, particle, sign in (("negative", spec.negative, -1),
                                 ("positive", spec.positive, 1)):
        difference = sum(final.amounts_mol[k] - start.amounts_mol[k] for k in particle.keys(name))
        assert difference == pytest.approx(sign * current * 50 / F, abs=1e-12)
    reading = battery_readout(spec, final, current)
    assert reading.concentration_overpotential_V * current < 0
    assert reading.electrolyte_ohmic_V * current < 0
    assert reading.solid_ohmic_V * current < 0
    data = spec.electrolyte.potential(final, current)
    assert np.all(data["ionic_current_A_m2"] * current > 0)
    assert np.isfinite(data["potential_V"]).all()
    integrated = replace(spec, electrolyte=replace(spec.electrolyte, potential_form="integrated"))
    assert battery_readout(integrated, final, current) == reading


def test_rest_relaxes_salt_without_replenishing_material():
    spec = model()
    first = advance_battery(spec, spec.initial_state(), program(0.2), 50).final_state
    final = advance_battery(spec, first, program(0), 300).final_state
    assert np.ptp(spec.electrolyte.concentrations(final)) < np.ptp(
        spec.electrolyte.concentrations(first))
    assert final.coordinates == first.coordinates
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(first.amounts_mol.values()), abs=1e-12)


@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_electrolyte_cutoff_retains_actual_time_and_positive_inventory(numerical_form):
    base = model(numerical_form=numerical_form)
    spec = replace(base, electrolyte=replace(base.electrolyte,
        minimum_concentration_mol_m3=990, diffusivity_m2_s=1e-12))
    result = advance_battery(spec, spec.initial_state(), program(0.3), 100)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:electrolyte:depleted"
    assert 0 < result.actual_duration_s < 100
    assert np.min(spec.electrolyte.concentrations(result.final_state)) == pytest.approx(
        990, abs=1e-7)
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        0.3 * result.actual_duration_s, rel=1e-9)


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_same_runtime_public_meter_sorted_snapshot_and_material_lifetime(noise):
    world = battery_world_spec(model())
    session = ChemWorldRuntime(composition=world, observation_noise_mode=noise)
    restored = ChemWorldRuntime(composition=world, observation_noise_mode=noise)
    try:
        session.reset(seed=13)
        for action in ({"operation": "configure_electrical_control", "current_A": 0.2},
                       {"operation": "wait", "duration_s": 10}):
            result = session.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        saved = json.loads(json.dumps(session.snapshot().to_dict(), sort_keys=True))
        restored.restore(saved)
        for action in ({"operation": "wait", "duration_s": 5},
                       {"operation": "measure", "instrument": "battery_meter"}):
            a, b = session.execute(action), restored.execute(action)
            assert a.info["transaction_status"] == b.info["transaction_status"] == "committed"
        assert session.snapshot().to_dict() == restored.snapshot().to_dict()
        channels = {c.channel_id: c for c in a.observation.channels}
        assert channels["terminal_potential"].unit == "V"
        assert channels["electrical_current"].value == 0.2
        slot = unpack_battery(session._state)[1]
        session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert unpack_battery(session._state)[1] == slot
    finally:
        session.close()
        restored.close()


@pytest.mark.parametrize("voltage,limit,mode", [
    (3.75, 2, "constant_voltage"), (3.7, 0.01, "positive_current_limit"),
    (4.1, 0.01, "negative_current_limit"),
])
def test_spme_voltage_control_and_both_limits_use_the_common_residual(voltage, limit, mode):
    spec = model(numerical_form="residual", maximum_current_A=limit)
    control = ScalarControlProgram(((0, voltage),), "V")
    before = spec.initial_state()
    result = advance_battery(spec, before, control, 2)
    ode = advance_battery(replace(spec, numerical_form="ode"), before, control, 2)
    assert result.final_state.amounts_mol == pytest.approx(
        ode.final_state.amounts_mol, rel=1e-7, abs=1e-12)
    assert result.final_state.coordinates == pytest.approx(
        ode.final_state.coordinates, rel=1e-7, abs=1e-8)
    assert result.solver_diagnostic["battery_control_mode"] == mode
    current = battery_control_current(spec, result.final_state, control)
    assert abs(current) <= limit
    if mode == "constant_voltage":
        assert battery_readout(spec, result.final_state, current).voltage_V == pytest.approx(
            voltage, abs=1e-10)
    assert all(abs(v) < 1e-9 for v in result.conservation_residuals.values())


def test_spme_numerical_failure_rolls_back_salt_particles_time_and_charge(monkeypatch):
    import chemworld.runtime.battery as battery

    runtime = ChemWorldRuntime(composition=battery_world_spec(model()))
    try:
        runtime.reset(seed=4)
        runtime.execute({"operation": "configure_electrical_control", "current_A": 0.2})
        before = runtime._state

        def fail(*args, **kwargs):
            raise RuntimeError("injected SPMe numerical failure")

        monkeypatch.setattr(battery, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_spme_sorted_snapshot_continues_in_a_new_process(noise, tmp_path):
    from chemworld.examples.battery_spme import workflow
    from test_native_domains_end_to_end import (
        test_snapshot_resume_in_middle_of_real_workflow,
    )

    test_snapshot_resume_in_middle_of_real_workflow(
        "spme", noise, tmp_path, recipe=workflow(model()), sort_keys=True)


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_spme_offline_example_logs_and_replays_all_nine_actions(noise, tmp_path):
    from chemworld.examples.battery_spme import run

    report = run(tmp_path / "spme.jsonl", noise=noise, spec=model())
    assert report["steps"] == report["replay_steps"] == 9


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_spme_shares_the_actual_clock_and_earliest_cutoff_with_other_domains(noise):
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
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
            for actual_slot, expected_slot in zip(
                local.state_slots.slots, expected.state.state_slots.slots, strict=True,
            ):
                if actual_slot.spec.state_type == "battery-lithium-and-charge-v1":
                    assert actual_slot.spec == expected_slot.spec
                    assert actual_slot.initial_values == expected_slot.initial_values
                    # Event location and integration directly to that time use
                    # different adaptive meshes. Use the solver's 1e-8 rtol;
                    # identical-input replay is checked separately above.
                    assert actual_slot.values == pytest.approx(
                        expected_slot.values, rel=1e-8, abs=0,
                    )
                else:
                    assert actual_slot == expected_slot
            assert local.species_amounts == expected.state.species_amounts
        battery = unpack_battery(local_states(runtime._state)["cell"])
        assert battery_readout(spec, battery[3], 0.02).voltage_V == pytest.approx(
            spec.minimum_voltage_V, abs=1e-8,
        )
        assert battery[3].coordinates["discharged_charge_C"] == pytest.approx(
            0.02 * elapsed, rel=1e-10, abs=0,
        )
        assert abs(spec.electrolyte.concentrations(battery[3])[0] - 1000) > 0
    finally:
        runtime.close()


def test_spme_sdist_wheel_runs_without_reference_library_outside_checkout(tmp_path):
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
from chemworld.examples.battery_spme import workflow
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
model = BatterySPMSpec(negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4),
    electrolyte=BatteryElectrolyteSpec(cells=(3,3,3)))
world, actions = workflow(model)
session = ChemWorldRuntime(composition=world)
session.reset(seed=4)
for action in actions[:2]:
    assert session.execute(action).info['transaction_status'] == 'committed'
assert model.electrolyte.concentrations(unpack_battery(session._state)[3])[0] > 1000
assert 'pybamm' not in sys.modules
session.close()
print('native packaged SPMe OK')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(wheel)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert "native packaged SPMe OK" in result.stdout
