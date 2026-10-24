"""Closed gas/condensate histories on the original world inventory and clock."""

import json
from dataclasses import replace

import pytest
from tests.test_humid_air_volume import thermal_inputs

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.physical_world import CompositionPhysics


def inputs(phase):
    dew = 300. if phase == "liquid" else 240.
    thermal, saturated = thermal_inputs(dew, 101325., phase, extra=0)
    initial = replace(saturated, temperature_K=dew+10)
    low, high, stop = (270., 473., 460.) if phase == "liquid" else (193., 273.16, 272.)
    model = ThermalSystemSpec(thermal, pressure_law="humid_air", loss_conductance_W_K=0,
        minimum_temperature_K=low, maximum_temperature_K=high, stop_temperature_K=stop,
        maximum_pressure_Pa=5e6)
    device = ThermalDeviceSpec(conductance_W_K=100, maximum_power_W=2, maximum_cooling_power_W=2,
        setpoint_K=dew-15, minimum_setpoint_K=low, maximum_setpoint_K=high)
    pressure = thermal.pressure_pa(initial)
    state = WorldState(
        dict(initial.amounts_mol), initial.volume_L, initial.temperature_K,
        pressure, "fluid", "reactor",
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
               "time": "s", "cost": "currency", "risk": "risk"},
        equipment=EquipmentLedger({"heater": EquipmentRecord("heater", "thermal_device",
            attached_vessel_id="reactor", settings={"thermal_device": device.to_dict()})}),
        vessels=VesselLedger({"reactor": VesselRecord(
            "reactor", "humid-air-vessel", initial.volume_L,
            high, 5e6, ("fluid",), initial.temperature_K, pressure)}))
    network = ReactionNetworkSpec("humid-air-stock", tuple(
        SpeciesSpec(c.identifier, c.formula, phase="gas") for c in thermal.components), ())
    payload = {
        "schema_version": "chemworld-world-composition-0.1", "composition_id": "humid-condensation",
        "components": [{"kind": "thermal", "parameters": {}}],
        "vessels": [{"id": "reactor", "capacity_L": initial.volume_L}],
        "physics": CompositionPhysics(network, {"reactor": state},
                                       models={"thermal": {"reactor": model}}).to_dict(),
        "task": {"objective": "cool and reheat the same humid-air inventory", "budget": 30,
                 "operations": ["heat", "wait", "pause_control", "resume_control", "terminate"],
                 "instruments": []},
    }
    return payload, model, initial, dew


def execute(runtime, operation, **kwargs):
    print(f"humid world {operation} starts at {runtime._state.ledger.time_s:.6f} s; "
          f"requested {kwargs}", flush=True)
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", json.dumps({key: result.info.get(key)
        for key in ("outcome_diagnostics", "rollback_reason", "world_events")}, default=str)
    print(f"humid world {operation} committed at {runtime._state.ledger.time_s:.6f} s", flush=True)
    return result


def packed(state):
    return CoupledState(state.species_amounts, state.temperature_K, state.volume_L)


@pytest.mark.parametrize("phase", ["liquid", "ice"])
def test_cooling_condenses_and_heating_evaporates_on_same_clock(phase, tmp_path, record_property):
    payload, model, initial, dew = inputs(phase)
    thermo = model.thermodynamics
    high_u = thermo.internal_energy_j(initial)
    low_u = thermo.internal_energy_j(replace(initial, temperature_K=dew-10))
    duration = (high_u-low_u)/2
    assert duration > 0
    runtime = ChemWorldRuntime(composition=json.loads(json.dumps(payload)))
    restored = None
    history = []
    try:
        runtime.reset(seed=25)
        assert thermo.equilibrium_state(packed(runtime._state)).condensed_water_mass_kg == 0
        heat, elapsed = 0., 0.
        for direction in (-1, 1):
            execute(runtime, "heat", target_temperature_K=dew+direction*15, duration_s=duration/2)
            if restored is not None:
                restored.close()
            path = tmp_path / f"{phase}-{direction}.json"
            runtime.snapshot().save(path)
            restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
            execute(runtime, "wait", duration_s=duration/2)
            execute(restored, "wait", duration_s=duration/2)
            heat += direction*2*duration
            elapsed += duration
            state = runtime._state
            current = thermo.equilibrium_state(packed(state))
            error = thermo.internal_energy_j(packed(state))-high_u-heat
            history.append({"direction": direction, "elapsed_s": state.ledger.time_s,
                            "temperature_K": state.temperature_K,
                            "energy_error_J": error, "heat_J": heat,
                            "equilibrium": current.to_dict()})
            assert state.species_amounts == dict(initial.amounts_mol)
            assert state.volume_L == initial.volume_L
            assert state.ledger.time_s == pytest.approx(elapsed, abs=1e-10)
            assert state.temperature_K == pytest.approx(dew+direction*10, abs=1e-5)
            assert state.ledger.energy_jacket_J == pytest.approx(heat, abs=1e-4)
            assert abs(error) <= 1e-4
            assert (current.condensed_water_mass_kg > 0) == (direction == -1)
            assert restored._state.to_dict() == state.to_dict()
        record_property("humid_air_history", json.dumps(history, allow_nan=False))
    finally:
        runtime.close()
        if restored is not None:
            restored.close()
