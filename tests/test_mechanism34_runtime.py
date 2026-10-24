"""Expanded thermo reaches actual thermal worlds and exact session continuation."""

import json

import pytest

from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_mechanism_world import gas_mechanism_world
from test_mechanism34 import cases


@pytest.mark.parametrize("model", cases.THERMOS)
def test_new_thermochemistry_world_heating_and_snapshot_continuation(model):
    network = convert_mechanism(cases.payload(thermo=model))
    world = gas_mechanism_world(network, {"A": 1e-4})
    original = ChemWorldRuntime(composition=world)
    restored = None
    try:
        original.reset(seed=71)
        first = original.execute(
            {
                "operation": "heat",
                "target_temperature_K": 750,
                "duration_s": 1,
                "stirring_speed_rpm": 600,
            }
        )
        assert first.info["transaction_status"] == "committed", first.info
        assert abs(first.info["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-5
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(original.snapshot().to_dict()))
        )
        action = {"operation": "wait", "duration_s": 1, "stirring_speed_rpm": 600}
        for runtime in (original, restored):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
            assert abs(result.info["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-5
            assert runtime._state.ledger.time_s == 2
            assert runtime._state.temperature_K > 700
            assert sum(runtime._state.species_amounts.values()) == pytest.approx(1e-4, abs=1e-12)
        assert original._state.to_dict() == restored._state.to_dict()
    finally:
        original.close()
        if restored is not None:
            restored.close()


def test_legacy_nasa7_thermal_declaration_remains_readable():
    network = convert_mechanism(cases.payload())
    model = IdealGasThermodynamics(network.imported_gas_thermochemistry)
    legacy = model.to_dict()
    legacy["model"] = "nasa7-ideal-gas-v1"
    assert IdealGasThermodynamics.from_dict(legacy).to_dict() == model.to_dict()
