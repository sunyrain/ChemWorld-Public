"""Finite-inventory electrochemistry, independently checked electrical identities."""

from dataclasses import replace
from itertools import pairwise
from math import exp, log, pi, sqrt

import numpy as np
import pytest

from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledCoordinate,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.transient_electrode import (
    F,
    TransientElectrodeSpec,
    advance_electrode,
    electrode_readout,
    electrode_system,
)


def control(value):
    return ScalarControlProgram(((0.0, value),), "V")


def assert_balances(spec, report):
    initial, final = report.initial_state, report.final_state
    o_keys = spec.keys(spec.oxidized_species)
    before, after = initial.coordinates, final.coordinates
    delta = {key: after[key] - before[key] for key in after}
    amount_scale = sum(initial.amounts_mol.values())
    assert abs(sum(final.amounts_mol.values()) - amount_scale) <= 1e-12 + 1e-7 * amount_scale
    qf = F * sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in o_keys)
    assert delta["faradaic_charge_C"] == pytest.approx(qf, abs=1e-9, rel=1e-7)
    assert delta["applied_charge_C"] == pytest.approx(
        delta["double_layer_charge_C"] + delta["faradaic_charge_C"], abs=1e-9, rel=1e-7
    )
    assert min(final.amounts_mol.values()) >= 0
    assert report.energy_balance_residual_J is None  # no invented complete-cell thermal claim


def test_signed_native_coordinates_share_one_solver_and_conservation():
    process = CoupledProcess(
        "capacitor",
        lambda s: ProcessContribution(
            coordinate_rates={"q": -2 * s.coordinates["q"], "receipt": -2 * s.coordinates["q"]}
        ),
    )
    system = CoupledSystem(
        ConstantCvThermodynamics({}, support_heat_capacity_J_K=1),
        (process,),
        coordinates=(CoupledCoordinate("q", "C", 1), CoupledCoordinate("receipt", "C", 1)),
        conservation_laws={"charge": {"q": 1, "receipt": -1}},
        isothermal=True,
    )
    result = system.advance(CoupledState({}, 298.15, 1, coordinates={"q": -1, "receipt": 0}), 2)
    assert result.final_state.coordinates["q"] == pytest.approx(-exp(-4), rel=1e-6)
    assert result.final_state.coordinates["receipt"] == pytest.approx(1 - exp(-4), rel=1e-6)
    assert result.solver_diagnostic["solver_calls"] == 1
    with pytest.raises(ValueError, match="declared coordinates"):
        system.advance(CoupledState({}, 298.15, 1), 1)


@pytest.mark.parametrize("potential", [-0.1, 0.0, 0.1])
def test_three_potential_steps(potential):
    spec = TransientElectrodeSpec()
    result = advance_electrode(
        spec, spec.initial_state(), control(potential), 10, evaluation_times_s=(0, 1, 5, 10)
    )
    assert result.actual_duration_s == 10
    assert_balances(spec, result)
    read = electrode_readout(spec, result.final_state, potential)
    if potential:
        assert read.faradaic_current_A * potential > 0
    else:
        assert abs(read.applied_current_A) < 1e-12


@pytest.mark.parametrize("ratio", [0.25, 1.0, 4.0])
def test_nernst_equilibrium_at_actual_face(ratio):
    spec = TransientElectrodeSpec()
    potential = R_J_PER_MOL_K * spec.temperature_K / F * log(ratio)
    state = spec.initial_state(ratio, 1, potential_V=potential)
    assert abs(electrode_readout(spec, state, potential).faradaic_current_A) < 1e-12
    result = advance_electrode(spec, state, control(potential), 1)
    assert (
        max(abs(result.final_state.amounts_mol[k] - v) for k, v in state.amounts_mol.items())
        < 1e-16
    )


def test_blocking_electrode_has_independent_exact_rc_response():
    spec = TransientElectrodeSpec(standard_rate_m_s=0)
    tau = spec.capacitance_f * spec.series_resistance_ohm
    result = advance_electrode(
        spec,
        spec.initial_state(),
        control(-0.1),
        5 * tau,
        evaluation_times_s=(tau, 2 * tau, 5 * tau),
    )
    for state in result.trajectory:
        read = electrode_readout(spec, state, -0.1)
        assert read.interfacial_potential_V == pytest.approx(
            -0.1 * (1 - exp(-state.time_s / tau)), abs=1e-8
        )
        assert read.applied_current_A == pytest.approx(
            -0.1 / spec.series_resistance_ohm * exp(-state.time_s / tau), abs=1e-9
        )
        assert read.faradaic_current_A == 0
    assert_balances(spec, result)


def test_cottrell_early_time_window_records_each_error():
    spec = TransientElectrodeSpec(cells=160, standard_rate_m_s=0.01, series_resistance_ohm=0.01)
    times = (0.1, 0.2, 0.5, 1.0)
    result = advance_electrode(
        spec, spec.initial_state(1, 0), control(-0.4), 1, evaluation_times_s=times
    )
    errors = []
    for state in result.trajectory:
        if state.time_s == 0:
            continue
        expected = -F * spec.area_m2 * sqrt(spec.diffusivity_O_m2_s / (pi * state.time_s))
        observed = electrode_readout(spec, state, -0.4).faradaic_current_A
        errors.append(abs(observed / expected - 1))
    print("Cottrell relative errors:", errors)
    assert len(errors) == len(times)
    assert max(errors) < 0.02
    assert_balances(spec, result)


def test_inventory_and_current_mesh_convergence():
    traces, profiles = {}, {}
    for cells in (20, 40, 80, 160):
        spec = TransientElectrodeSpec(cells=cells)
        result = advance_electrode(
            spec, spec.initial_state(), control(0.1), 10, evaluation_times_s=(1, 5, 10)
        )
        traces[cells] = np.array(
            [
                electrode_readout(spec, s, 0.1).faradaic_current_A
                for s in result.trajectory
                if s.time_s > 0
            ]
        )
        profiles[cells] = np.array(
            [result.final_state.amounts_mol[k] for k in spec.keys(spec.oxidized_species)]
        )
        assert_balances(spec, result)
    current_errors, profile_errors = [], []
    for cells in (20, 40, 80):
        current_errors.append(float(np.max(abs(traces[cells] / traces[160] - 1))))
        # Compare extensive cell amounts to exact sums of the fine cells.
        reference = profiles[160].reshape(cells, 160 // cells).sum(axis=1)
        profile_errors.append(
            float(np.linalg.norm(profiles[cells] - reference) / np.linalg.norm(reference))
        )
    print("mesh current errors:", current_errors, "profile errors:", profile_errors)
    assert all(b < a for a, b in pairwise(current_errors))
    assert all(b < a for a, b in pairwise(profile_errors))
    assert current_errors[-1] < 0.02


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.1])
def test_three_scans_reverse_current_without_losing_charge(rate):
    spec = TransientElectrodeSpec()
    program = ScalarControlProgram(
        ((0, 0), (0.1 / rate, 0.1), (0.3 / rate, -0.1), (0.4 / rate, 0)), "V"
    )
    result = advance_electrode(
        spec,
        spec.initial_state(),
        program,
        0.4 / rate,
        evaluation_times_s=(0.1 / rate, 0.3 / rate, 0.4 / rate),
    )
    points = [p for p in result.trajectory if p.time_s > 0]
    assert electrode_readout(spec, points[0], 0.1).faradaic_current_A > 0
    assert electrode_readout(spec, points[1], -0.1).faradaic_current_A < 0
    assert_balances(spec, result)


def test_continuation_retains_concentration_profile_and_charge():
    spec = TransientElectrodeSpec()
    program = ScalarControlProgram(((0, 0), (10, 0.1)), "V")
    initial = spec.initial_state()
    whole = advance_electrode(spec, initial, program, 10)
    first = advance_electrode(spec, initial, program, 4)
    second = advance_electrode(spec, first.final_state, program, 6)
    assert second.final_state.time_s == whole.final_state.time_s
    for key, value in whole.final_state.amounts_mol.items():
        assert second.final_state.amounts_mol[key] == pytest.approx(value, rel=2e-6, abs=1e-16)
    for key, value in whole.final_state.coordinates.items():
        assert second.final_state.coordinates[key] == pytest.approx(value, rel=2e-6, abs=1e-10)


def test_invalid_electrode_or_control_fails_explicitly():
    with pytest.raises(ValueError):
        TransientElectrodeSpec(cells=True)
    with pytest.raises(ValueError):
        TransientElectrodeSpec(alpha_cathodic=0.6)
    with pytest.raises(ValueError):
        electrode_system(TransientElectrodeSpec(), ScalarControlProgram(((0, 0.1),), "A"))
    spec = TransientElectrodeSpec()
    bad = replace(spec.initial_state(), temperature_K=310)
    with pytest.raises(ValueError, match="isothermal"):
        advance_electrode(spec, bad, control(0), 1)


def test_native_electrode_uses_shared_wait_measurement_and_snapshot(tmp_path):
    from chemworld.runtime.engine import ChemWorldRuntime
    from chemworld.runtime.session_snapshot import SessionSnapshot
    from chemworld.runtime.transient_electrode import unpack_electrode
    from chemworld.world.electrode_world import electrode_world_spec

    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=19)
    initial = unpack_electrode(session._state)[3]
    configured = session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
    assert configured.info["transaction_status"] == "committed"
    result = session.execute({"operation": "wait", "duration_s": 10})
    assert result.info["transaction_status"] == "committed"
    assert session._state.ledger.time_s == 10
    assert unpack_electrode(session._state)[3].amounts_mol != initial.amounts_mol
    measured = session.execute({"operation": "measure", "instrument": "electrical_meter"})
    assert measured.info["transaction_status"] == "committed"
    assert session._state.ledger.time_s == 11
    assert {c.channel_id: c.unit for c in measured.observation.channels} == {
        "terminal_potential": "V",
        "electrical_current": "A",
    }
    path = tmp_path / "electrode.json"
    session.snapshot().save(path)
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
    next_action = {"operation": "wait", "duration_s": 2}
    left, right = session.execute(next_action), restored.execute(next_action)
    assert left.info == right.info
    assert session._state.to_dict() == restored._state.to_dict()


def test_finite_closed_layer_relaxes_to_nernst_with_actual_depletion():
    spec = TransientElectrodeSpec(cells=20)
    initial = spec.initial_state()
    potential = 0.03
    result = advance_electrode(spec, initial, control(potential), 12000)
    final = result.final_state
    ratio = exp(F * potential / (R_J_PER_MOL_K * spec.temperature_K))
    expected_o = 2 * ratio / (1 + ratio)
    for key, volume in zip(
        spec.keys(spec.oxidized_species), spec.mesh.cell_volumes_m3, strict=True
    ):
        assert final.amounts_mol[key] / volume == pytest.approx(expected_o, rel=2e-5)
    assert abs(electrode_readout(spec, final, potential).faradaic_current_A) < 1e-10
    assert_balances(spec, result)


def test_electrode_material_and_equipment_lifetimes_remain_distinct():
    from chemworld.foundation.state_ownership import StateSlots
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.transient_electrode import initial_charge_slot, unpack_electrode
    from chemworld.world.electrode_world import electrode_world_spec

    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=2)
    initial_phases = session._state.phases
    session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
    session.execute({"operation": "wait", "duration_s": 10})
    record, slot, spec, previous, _ = unpack_electrode(session._state)
    session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    restored_record, restored_slot, _, replaced_material, _ = unpack_electrode(session._state)
    assert restored_record == record
    assert restored_slot.values == slot.values
    assert session._state.phases == initial_phases
    assert replaced_material.amounts_mol != previous.amounts_mol
    # Replacement is an explicit experiment-boundary operation. It must neither
    # clone the old charge nor reset the liquid profile to uniform a second time.
    session.execute({"operation": "wait", "duration_s": 2})
    phases_before = session._state.phases
    replacement = replace(record, equipment_id="replacement-electrode")
    fresh = initial_charge_slot(session._state, replacement, spec.initial_state().coordinates)
    session.reset(
        options={
            "scope": "equipment",
            "equipment_id": record.equipment_id,
            "replacement": replacement,
            "slots": StateSlots((fresh,)),
        }
    )
    assert session._state.phases == phases_before
    new_record, new_slot, _, _, _ = unpack_electrode(session._state)
    assert new_record.equipment_id == "replacement-electrode"
    assert new_slot.values == spec.initial_state().coordinates
    continued = session.execute({"operation": "wait", "duration_s": 2})
    assert continued.info["transaction_status"] == "committed"
    assert unpack_electrode(session._state)[1].values["applied_charge_C"] != 0


def test_numerical_failure_rolls_back_electrolyte_charge_and_time(monkeypatch):
    import chemworld.runtime.transient_electrode as runtime
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.electrode_world import electrode_world_spec

    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=5)
    session.execute({"operation": "configure_electrical_control", "potential_V": -0.1})
    before = session._state

    def failed(*args, **kwargs):
        raise RuntimeError("injected numerical failure")

    monkeypatch.setattr(runtime, "advance_electrode", failed)
    result = session.execute({"operation": "wait", "duration_s": 2})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert session._state.phases == before.phases
    assert session._state.state_slots == before.state_slots
    assert session._state.ledger.time_s == before.ledger.time_s


def test_physical_potential_stop_keeps_actual_endpoint():
    spec = TransientElectrodeSpec()
    state = spec.initial_state(potential_V=spec.maximum_potential_V)
    result = advance_electrode(spec, state, control(0), 10)
    assert result.outcome == "stopped"
    assert result.stopping_event == "electrode:positive-potential-limit"
    assert result.actual_duration_s == 0
    assert result.final_state == state


def test_runtime_ramp_program_and_measurement_keep_charge_history():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.transient_electrode import unpack_electrode
    from chemworld.world.electrode_world import electrode_world_spec

    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=3)
    currents = []
    for target in (0.1, -0.1, 0):
        config = session.execute(
            {
                "operation": "configure_electrical_control",
                "potential_V": target,
                "ramp_duration_s": 5,
            }
        )
        assert config.info["transaction_status"] == "committed"
        before_time = session._state.ledger.time_s
        waited = session.execute({"operation": "wait", "duration_s": 5})
        assert waited.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == before_time + 5
        measured = session.execute({"operation": "measure", "instrument": "electrical_meter"})
        assert measured.info["transaction_status"] == "committed"
        values = {c.channel_id: c.value for c in measured.observation.channels}
        assert set(values) == {"terminal_potential", "electrical_current"}
        assert values["terminal_potential"] == target
        currents.append(values["electrical_current"])
        physical = unpack_electrode(session._state)[3]
        q = physical.coordinates
        assert q["applied_charge_C"] == pytest.approx(
            q["faradaic_charge_C"] + q["double_layer_charge_C"], abs=1e-10
        )
    assert currents[0] > 0 > currents[1]


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_agent_electrode_history_replays_and_has_no_profile_observations(tmp_path, noise):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records
    from chemworld.world.electrode_world import electrode_world_spec

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {"operation": "configure_electrical_control", "potential_V": -0.1},
                {"operation": "wait", "duration_s": 3},
                {"operation": "measure", "instrument": "electrical_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "electrode.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=electrode_world_spec().to_dict(),
        world_split="public-dev",
        budget=40,
        objective="balanced",
        seed=4,
        observation_noise_mode=noise,
        output_path=path,
    )
    records = load_jsonl(path)
    assert len(records) == 4
    report = verify_records(records, tolerance=1e-8)
    assert report.verified, report.mismatches


def test_serialized_electrode_continues_in_another_process(tmp_path):
    import json
    import subprocess
    import sys

    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.electrode_world import electrode_world_spec

    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=7)
    session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
    session.execute({"operation": "wait", "duration_s": 3})
    snapshot, output = tmp_path / "before.json", tmp_path / "after.json"
    session.snapshot().save(snapshot)
    expected = session.execute({"operation": "measure", "instrument": "electrical_meter"})
    code = (
        "import json,sys; from pathlib import Path; "
        "from chemworld.runtime import ChemWorldRuntime,SessionSnapshot; "
        "s=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1])); "
        "r=s.execute({'operation':'measure','instrument':'electrical_meter'}); "
        "Path(sys.argv[2]).write_text(json.dumps({'state':s._state.to_dict(),"
        "'observation':r.observation.native_record()},allow_nan=False))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code, str(snapshot), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text()) == json.loads(
        json.dumps(
            {"state": session._state.to_dict(), "observation": expected.observation.native_record()}
        )
    )
