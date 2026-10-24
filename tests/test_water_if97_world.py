"""Water energy and pressure reach the existing finite equipment and world."""

import json
from dataclasses import replace

import pytest

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
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.water_if97_solvers import water_if97_saturation, water_if97_tp
from chemworld.physchem.water_if97_thermal import IF97IsochoricThermodynamics
from chemworld.physchem.water_if97_volume import water_if97_trho
from chemworld.physchem.water_saturation import water_saturation_pressure
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.physical_world import CompositionPhysics


def inputs(kind):
    component = ComponentSpec("water", "H2O", cas_number="7732-18-5")
    if kind == "coexistence":
        point = water_if97_saturation(water_saturation_pressure(500), .35)
        phase = "fluid"
    else:
        point = water_if97_tp(300, 3e6) if kind == "liquid" else water_if97_tp(500, .1e6)
        phase = "liquid" if kind == "liquid" else "gas"
    volume = 2 * (component.molecular_weight_g_mol / 1000) / point.mass_density_kg_m3 * 1000
    thermal = IF97IsochoricThermodynamics(component, volume, 5)
    model = ThermalSystemSpec(thermal, pressure_law="if97", loss_conductance_W_K=0,
        minimum_temperature_K=273.15, maximum_temperature_K=620, stop_temperature_K=610,
        maximum_pressure_Pa=80e6)
    device = ThermalDeviceSpec(conductance_W_K=100, maximum_power_W=10, setpoint_K=600,
        minimum_setpoint_K=273.15, maximum_setpoint_K=620)
    state = WorldState(
        {"water": 2}, volume, point.temperature_K, point.pressure_Pa, phase, "reactor",
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
               "time": "s", "cost": "currency", "risk": "risk"},
        equipment=EquipmentLedger({"heater": EquipmentRecord("heater", "thermal_device",
            attached_vessel_id="reactor", settings={"thermal_device": device.to_dict()})}),
        vessels=VesselLedger({"reactor": VesselRecord("reactor", "pure-water-vessel", volume,
            620, 80e6, (phase,), point.temperature_K, point.pressure_Pa)}))
    network = ReactionNetworkSpec("pure-water",
                                  (SpeciesSpec("water", "H2O", phase="gas"),), ())
    payload = {
        "schema_version": "chemworld-world-composition-0.1", "composition_id": "if97-water-heating",
        "components": [{"kind": "thermal", "parameters": {}}],
        "vessels": [{"id": "reactor", "capacity_L": volume}],
        "physics": CompositionPhysics(network, {"reactor": state},
                                       models={"thermal": {"reactor": model}}).to_dict(),
        "task": {"objective": "heat a declared closed pure water inventory", "budget": 20,
                 "operations": ["heat", "wait", "pause_control", "resume_control", "terminate"],
                 "instruments": []},
    }
    return payload, model


def packed(state):
    return CoupledState(state.species_amounts, state.temperature_K, state.volume_L)


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("kind", ["liquid", "vapor", "coexistence"])
def test_closed_water_history_and_snapshot_preserve_energy_and_inventory(kind, tmp_path):
    payload, model = inputs(kind)
    assert ThermalSystemSpec.from_dict(model.to_dict()).to_dict() == model.to_dict()
    runtime = ChemWorldRuntime(composition=json.loads(json.dumps(payload)))
    restored = None
    try:
        runtime.reset(seed=25)
        initial = runtime._state
        energy = model.thermodynamics.internal_energy_j(packed(initial))
        execute(runtime, "heat", target_temperature_K=600, duration_s=2)
        snapshot = tmp_path / "water.json"
        runtime.snapshot().save(snapshot)
        restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snapshot))
        elapsed, expected_heat = 2, 20
        for operation, parameters, seconds, heat in (
            ("wait", {"duration_s": 2}, 2, 20), ("pause_control", {}, 0, 0),
            ("wait", {"duration_s": 1}, 1, 0), ("resume_control", {}, 0, 0),
            ("wait", {"duration_s": 2}, 2, 20),
        ):
            execute(runtime, operation, **parameters)
            execute(restored, operation, **parameters)
            elapsed += seconds
            expected_heat += heat
            state = runtime._state
            assert state.species_amounts == {"water": 2}
            assert state.volume_L == initial.volume_L
            assert state.ledger.time_s == elapsed
            assert state.ledger.energy_jacket_J == pytest.approx(expected_heat, abs=1e-4)
            assert model.thermodynamics.internal_energy_j(packed(state)) - energy == pytest.approx(
                expected_heat, abs=1e-4)
            assert state.pressure_Pa == pytest.approx(
                model.thermodynamics.pressure_pa(packed(state)), rel=2e-8)
            assert restored._state.to_dict() == state.to_dict()
        assert runtime._state.temperature_K > initial.temperature_K
        if kind == "coexistence":
            final_water = model.thermodynamics.water_state(packed(runtime._state))
            assert final_water.vapor_mass_fraction > .35
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def test_if97_pressure_limit_uses_the_same_energy_state():
    payload, model = inputs("liquid")
    volume = model.thermodynamics.volume_L
    limit = model.thermodynamics.pressure_pa(CoupledState({"water": 2}, 300.1, volume))
    payload["physics"]["models"]["thermal"]["bindings"]["reactor"]["maximum_pressure_Pa"] = limit
    runtime = ChemWorldRuntime(composition=payload)
    try:
        runtime.reset(seed=25)
        result = execute(runtime, "heat", target_temperature_K=600, duration_s=2)
        assert result.info["operation_outcome"] == "stopped"
        assert runtime._state.temperature_K == pytest.approx(300.1, abs=1e-6)
        assert runtime._state.pressure_Pa == pytest.approx(limit, rel=2e-8)
        assert 0 < runtime._state.ledger.time_s < 2
    finally:
        runtime.close()


def test_material_reset_keeps_only_declared_support_energy():
    payload, model = inputs("vapor")
    runtime = ChemWorldRuntime(composition=payload)
    try:
        runtime.reset(seed=25)
        initial = runtime._state
        execute(runtime, "wait", duration_s=2)
        retained_temperature = runtime._state.temperature_K
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        thermal = model.thermodynamics
        expected = (thermal.internal_energy_j(packed(initial)) + thermal.support_heat_capacity_J_K
                    * (retained_temperature - initial.temperature_K))
        assert thermal.internal_energy_j(packed(runtime._state)) == pytest.approx(
            expected, abs=1e-4)
        assert runtime._state.volume_L == initial.volume_L
        assert runtime._state.species_amounts == initial.species_amounts
    finally:
        runtime.close()


def test_thermal_water_rejects_changed_stock_volume_and_other_pressure_laws():
    _, model = inputs("liquid")
    thermal = model.thermodynamics
    for state in (CoupledState({"water": 0}, 300, thermal.volume_L),
                  CoupledState({"water": 2, "NaCl": 1}, 300, thermal.volume_L),
                  CoupledState({"water": 2}, 300, thermal.volume_L * 2)):
        with pytest.raises(ValueError):
            thermal.validate_state(state)
    for pressure_law in ("fixed_pressure", "ideal_gas", "helmholtz"):
        with pytest.raises(ValueError, match="same EOS"):
            replace(model, pressure_law=pressure_law)
    with pytest.raises(ValueError, match="temperature domain"):
        replace(model, minimum_temperature_K=270)
    for T, rho in ((250, 1000), (300, 5000), (647.096, 322), (300, 0)):
        with pytest.raises(ValueError):
            water_if97_trho(T, rho)
