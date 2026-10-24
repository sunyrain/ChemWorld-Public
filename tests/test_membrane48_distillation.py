"""Actual finite thermal membrane and original runtime semantics."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.membrane_distillation import (
    advance_membrane_distillation,
    membrane_distillation_inventory,
    membrane_distillation_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.membrane_distillation import (
    MembraneDistillationRuntime,
    read_membrane_distillation,
)
from chemworld.world.membrane_distillation_world import membrane_distillation_world_spec
from reference.membrane48_fixtures import amounts, devices, spec


@pytest.mark.parametrize("temperatures", [(320, 300), (300, 320)])
def test_signed_transfer_preserves_all_original_ions_and_energy(temperatures):
    model = spec("Davies")
    ns = (amounts(model), amounts(model, concentration=0))
    initial = membrane_distillation_inventory(model, *ns, *temperatures)
    heater, cooler = devices(False)
    result, receipt = advance_membrane_distillation(
        model, initial, heater=heater, cooler=cooler, duration_s=30
    )
    for i in range(2):
        for key in ("Na+", "Cl-"):
            assert result.final_state.amounts_mol[i][key] == initial.amounts_mol[i][key]
    assert sum(result.final_state.enthalpies_J) == pytest.approx(
        sum(initial.enthalpies_J), abs=2e-7
    )
    assert receipt["water_hot_to_cold_mol"] * (temperatures[0] - temperatures[1]) > 0
    assert receipt["membrane_energy_hot_to_cold_J"] * (temperatures[0] - temperatures[1]) > 0
    assert abs(receipt["membrane_energy_closure_J"]) < 2e-7


def test_common_reference_and_knudsen_limits_are_required():
    model = spec()
    wrong = replace(
        model.vapor,
        standard_state=replace(model.vapor.standard_state, reference_basis_id="unrelated"),
    )
    with pytest.raises(ValueError, match="reference"):
        replace(model, vapor=wrong)
    with pytest.raises(ValueError, match="Knudsen"):
        membrane_distillation_point(
            replace(model, pore_radius_m=1e-5), (amounts(model), amounts(model)), (320, 300)
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_heat_pause_passive_transfer_and_exact_snapshot(reverse):
    model = spec()
    heater, cooler = devices()
    temperatures = (300, 320) if reverse else (320, 300)
    runtime = ChemWorldRuntime(
        composition=membrane_distillation_world_spec(
            model,
            amounts(model),
            amounts(model, concentration=0),
            hot_temperature_K=temperatures[0],
            cold_temperature_K=temperatures[1],
            heater=heater,
            cooler=cooler,
        )
    )
    restored = None
    try:
        runtime.reset(seed=48)
        for action in (
            {"operation": "heat", "duration_s": 2, "target_temperature_K": 325},
            {"operation": "pause_control"},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        before = runtime._state
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            result = session.execute({"operation": "wait", "duration_s": 3})
            assert result.info["transaction_status"] == "committed", result.info
            state = session._state
            assert state.ledger.time_s == 5
            assert state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
            assert state.inactive_vessels["cold"].thermal.vessels["cold"].heat_loss_J > 0
            assert state.thermal.vessels["reactor"].heat_loss_J == 0
            assert state.thermal.vessels["reactor"].energy_jacket_J > 0
            assert state.species_amounts["H2O"] != before.species_amounts["H2O"]
            assert read_membrane_distillation(state)["instrument_id"] is None
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_nonexistent_flow_and_program_are_rejected_without_time():
    model = spec()
    heater, cooler = devices()
    composition = membrane_distillation_world_spec(
        model,
        amounts(model),
        amounts(model),
        hot_temperature_K=320,
        cold_temperature_K=300,
        heater=heater,
        cooler=cooler,
    )
    states = composition.physics.initial_states
    state = states["reactor"].replace(inactive_vessels={"cold": states["cold"]})
    adapter = MembraneDistillationRuntime(composition)
    for action, program in (
        ({"operation": "run_flow", "duration_s": 2}, None),
        ({"operation": "wait", "duration_s": 2}, object()),
    ):
        result = adapter.advance(state, action, jacket_program=program)
        assert result.outcome == "rejected"
        assert result.actual_duration_s == 0
    with pytest.raises(ValueError, match="no declared pump"):
        adapter.configure_flow(state, {"flow_rate_mL_min": 1})


def test_finite_receiver_capacity_stops_at_actual_time():
    model = spec()
    hot, cold = amounts(model), amounts(model, concentration=0)
    volume = 1000 * model.properties.volume(cold, 300, 1e5)
    model = replace(model, capacities_L=(0.04, volume + 1e-7))
    initial = membrane_distillation_inventory(model, hot, cold, 320, 300)
    heater, cooler = devices(False)
    result, receipt = advance_membrane_distillation(
        model, initial, heater=heater, cooler=cooler, duration_s=30
    )
    assert result.stopping_event == "capacity_1"
    assert 0 < result.actual_duration_s < 30
    assert receipt["final"]["volumes_L"][1] == pytest.approx(model.capacities_L[1], abs=1e-10)


def test_declared_shared_wait_owns_both_clocks_and_heat_once():
    model = spec()
    heater, cooler = devices(False)
    runtime = ChemWorldRuntime(
        composition=membrane_distillation_world_spec(
            model,
            amounts(model),
            amounts(model, concentration=0),
            hot_temperature_K=320,
            cold_temperature_K=300,
            heater=heater,
            cooler=cooler,
            shared_clock=True,
        )
    )
    restored = None
    try:
        runtime.reset(seed=48)
        result = runtime.execute({"operation": "wait", "duration_s": 2})
        assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            result = session.execute({"operation": "wait", "duration_s": 3})
            assert result.info["transaction_status"] == "committed", result.info
            state = session._state
            assert state.ledger.time_s == 5
            assert set(state.vessel_elapsed_s.values()) == {5}
            assert state.thermal.vessels["reactor"].energy_jacket_J == pytest.approx(
                state.ledger.energy_jacket_J
            )
            assert state.thermal.vessels["reactor"].heat_loss_J == 0
            assert state.inactive_vessels["cold"].thermal.vessels[
                "cold"
            ].heat_loss_J == pytest.approx(state.ledger.heat_loss_J)
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_inconsistent_local_heat_map_rolls_back_real_session(monkeypatch):
    model = spec()
    heater, cooler = devices()
    original = MembraneDistillationRuntime.advance

    def broken(self, state, action, **kwargs):
        result = original(self, state, action, **kwargs)
        return replace(
            result,
            vessel_thermal_deltas={
                state.vessel_id: {"energy_jacket_J": 9999.0},
                model.cold_vessel_id: {"heat_loss_J": 0.0},
            },
        )

    monkeypatch.setattr(MembraneDistillationRuntime, "advance", broken)
    runtime = ChemWorldRuntime(
        composition=membrane_distillation_world_spec(
            model,
            amounts(model),
            amounts(model, concentration=0),
            hot_temperature_K=320,
            cold_temperature_K=300,
            heater=heater,
            cooler=cooler,
        )
    )
    try:
        runtime.reset(seed=48)
        before = runtime._state
        result = runtime.execute({"operation": "wait", "duration_s": 3})
        assert result.info["transaction_status"] != "committed", result.info
        after = runtime._state
        assert after.ledger.time_s == before.ledger.time_s
        assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        assert after.ledger.heat_loss_J == before.ledger.heat_loss_J
        assert after.species_amounts == before.species_amounts
        assert after.state_slots == before.state_slots
        assert after.inactive_vessels["cold"].to_dict() == before.inactive_vessels["cold"].to_dict()
    finally:
        runtime.close()
