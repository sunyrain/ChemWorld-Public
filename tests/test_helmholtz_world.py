"""Same-EOS fixed-volume material runs through original native thermal actions."""

import json
from dataclasses import replace

import pytest
from tests.test_helmholtz_flash import mixture

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_equilibrium_thermal import (
    HelmholtzEquilibriumIsochoricThermodynamics,
)
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.physical_world import CompositionPhysics


def inputs(*, equilibrium=False):
    fluid = HelmholtzFluid("synthetic-Ar", "", .039948, 8.3, 300, 1000, 100, 1500, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})),
        (HelmholtzTerm("ResidualHelmholtzPower",
            {"n": [.05], "d": [2], "t": [1], "l": [0]}),), "synthetic consumer equation")
    thermal = HelmholtzIsochoricThermodynamics(fluid, "Ar", 20, 5, 298.15)
    volume, temperature, phase, minimum = 20, 500, "gas", 300
    if equilibrium:
        phase_model, coexistence = mixture()
        volume, temperature, phase, minimum = (
            2000/coexistence.molar_density_mol_m3, 240, "fluid", 200)
        thermal = HelmholtzEquilibriumIsochoricThermodynamics(phase_model, "Ar", volume, 5)
    model = ThermalSystemSpec(thermal, pressure_law="helmholtz", loss_conductance_W_K=0,
        minimum_temperature_K=minimum, maximum_temperature_K=650, stop_temperature_K=640,
        maximum_pressure_Pa=1e6)
    pressure = thermal.pressure_pa(CoupledState({"Ar": 2}, temperature, volume))
    device = ThermalDeviceSpec(conductance_W_K=100, maximum_power_W=60, setpoint_K=600,
        minimum_setpoint_K=minimum, maximum_setpoint_K=650)
    state = WorldState({"Ar": 2}, volume, temperature, pressure, phase, "reactor",
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
            "time": "s", "cost": "currency", "risk": "risk"},
        equipment=EquipmentLedger({"heater": EquipmentRecord("heater", "thermal_device",
            attached_vessel_id="reactor", settings={"thermal_device": device.to_dict()})}),
        vessels=VesselLedger({"reactor": VesselRecord("reactor", "pure-fluid-vessel", volume,
            650, 1e6, (phase,), temperature, pressure)}))
    network = ReactionNetworkSpec("synthetic-pure-fluid",
        (SpeciesSpec("Ar", "Ar", phase="gas"),), ())
    payload = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "helmholtz-thermal-consumer",
        "components": [{"kind": "thermal", "parameters": {}}],
        "vessels": [{"id": "reactor", "capacity_L": volume}],
        "physics": CompositionPhysics(network, {"reactor": state},
            thermal_models={"reactor": model}).to_dict(),
        "task": {"objective": "heat a declared homogeneous pure fluid", "budget": 20,
            "operations": ["heat", "wait", "pause_control", "resume_control", "terminate"],
            "instruments": []},
    }
    return payload, model


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_heat_wait_pressure_energy_and_snapshot_keep_one_world(tmp_path):
    payload, model = inputs()
    runtime = ChemWorldRuntime(composition=json.loads(json.dumps(payload)))
    runtime.reset(seed=24)
    initial = runtime._state
    result = execute(runtime, "heat", target_temperature_K=600, duration_s=5)
    state = runtime._state
    expected = 500 + 300 / (2 * 1.5 * 8.3 + 5)
    assert state.temperature_K == pytest.approx(expected, abs=1e-7)
    expected_pressure = 100 * 8.3 * expected + 2 * 8.3 * 300 * .05 * 100**3 / 1000**2
    assert state.pressure_Pa == pytest.approx(expected_pressure)
    assert state.species_amounts == initial.species_amounts
    assert state.volume_L == initial.volume_L and state.ledger.time_s == 5
    assert state.ledger.energy_jacket_J == pytest.approx(300)
    assert abs(result.info["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-7
    assert ThermalSystemSpec.from_dict(model.to_dict()).to_dict() == model.to_dict()
    snapshot = tmp_path / "world.json"
    runtime.snapshot().save(snapshot)
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snapshot))
    execute(runtime, "wait", duration_s=5)
    execute(restored, "wait", duration_s=5)
    assert restored._state.temperature_K == runtime._state.temperature_K
    assert restored._state.pressure_Pa == runtime._state.pressure_Pa
    assert restored._state.ledger.to_dict() == runtime._state.ledger.to_dict()
    execute(runtime, "pause_control")
    before = runtime._state
    execute(runtime, "wait", duration_s=3)
    assert runtime._state.temperature_K == before.temperature_K
    assert runtime._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J


def test_helmholtz_native_pressure_event_stops_on_same_eos():
    payload, model = inputs()
    limit = model.thermodynamics.pressure_pa(CoupledState({"Ar": 2}, 505, 20))
    # The existing model declaration owns the pressure limit, not the controller.
    physics = CompositionPhysics.from_dict(payload["physics"])
    raw = physics.to_dict()
    raw["models"]["thermal"]["bindings"]["reactor"]["maximum_pressure_Pa"] = limit
    payload["physics"] = raw
    runtime = ChemWorldRuntime(composition=payload)
    runtime.reset(seed=24)
    result = execute(runtime, "wait", duration_s=5)
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.temperature_K == pytest.approx(505, abs=1e-6)
    assert runtime._state.pressure_Pa == pytest.approx(limit)
    assert runtime._state.ledger.time_s < 5


def test_eos_pressure_cannot_be_combined_with_another_caloric_law():
    _, model = inputs()
    with pytest.raises(ValueError, match="same EOS"):
        replace(model, pressure_law="ideal_gas")
    with pytest.raises(ValueError, match="temperature domain"):
        replace(model, maximum_temperature_K=1600)


def test_material_replacement_retains_body_energy_at_same_fixed_volume():
    payload, model = inputs()
    runtime = ChemWorldRuntime(composition=payload)
    runtime.reset(seed=24)
    execute(runtime, "wait", duration_s=5)
    retained_temperature = runtime._state.temperature_K
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    expected = 500 + 5 * (retained_temperature-500) / (2*1.5*8.3+5)
    assert runtime._state.temperature_K == pytest.approx(expected, abs=1e-7)
    assert runtime._state.volume_L == 20 and runtime._state.species_amounts == {"Ar": 2}
    expected_pressure = model.thermodynamics.pressure_pa(CoupledState({"Ar": 2}, expected, 20))
    assert runtime._state.pressure_Pa == pytest.approx(expected_pressure)


def test_equilibrium_latent_energy_and_snapshot_use_original_world(tmp_path):
    payload, model = inputs(equilibrium=True)
    thermal = model.thermodynamics
    assert ThermalSystemSpec.from_dict(model.to_dict()).to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="same EOS"):
        replace(model, pressure_law="ideal_gas")
    runtime = ChemWorldRuntime(composition=json.loads(json.dumps(payload)))
    runtime.reset(seed=24)

    def packed():
        state = runtime._state
        return CoupledState(state.species_amounts, state.temperature_K, state.volume_L)

    start = packed()
    execute(runtime, "heat", target_temperature_K=600, duration_s=5)
    final = packed()
    phases = thermal.molar_state(final)
    assert isinstance(phases, HelmholtzTwoPhaseState)
    assert phases.vapor_fraction > .35 and final.temperature_K > 240
    assert thermal.internal_energy_j(final)-thermal.internal_energy_j(start) == pytest.approx(
        300, abs=2e-4)
    assert runtime._state.pressure_Pa == pytest.approx(phases.pressure_Pa)
    assert runtime._state.ledger.time_s == 5
    assert runtime._state.species_amounts == {"Ar": 2}
    snapshot = tmp_path / "equilibrium.json"
    runtime.snapshot().save(snapshot)
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snapshot))
    execute(runtime, "wait", duration_s=2)
    execute(restored, "wait", duration_s=2)
    assert restored._state.to_dict() == runtime._state.to_dict()
