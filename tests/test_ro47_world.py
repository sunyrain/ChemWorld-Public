"""Original membrane strategy, finite vessel ledger and exact snapshot continuation."""

import json
from dataclasses import replace

import pytest

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.distributed_ro import DistributedRORuntime
from chemworld.world.distributed_ro_world import distributed_ro_world_spec
from reference.ro47_fixtures import feed, spec


@pytest.mark.parametrize("temperature", [290, 310, 330])
def test_original_world_transaction_and_snapshot(temperature):
    model = spec(temperature=temperature)
    runtime = ChemWorldRuntime(composition=distributed_ro_world_spec(model, feed(model)))
    restored = None
    try:
        runtime.reset(seed=47)
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "wait", "duration_s": 2},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            result = session.execute({"operation": "run_flow", "duration_s": 3})
            assert result.info["transaction_status"] == "committed", result.info
            state = session._state
            assert state.ledger.time_s == 5
            assert state.inactive_vessels["permeate"].species_amounts["H2O"] > 0
            assert state.inactive_vessels["retentate"].species_amounts["H2O"] > 0
            assert state.ledger.flow_pump_work_J > 0
            assert state.phases.phases["liquid"].species_amounts_mol == state.species_amounts
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def test_undeclared_temperature_and_old_meter_do_not_commit_or_fabricate_readings():
    model = spec()
    runtime = ChemWorldRuntime(composition=distributed_ro_world_spec(model, feed(model)))
    try:
        runtime.reset(seed=47)
        state = runtime._state.to_dict()
        result = runtime.execute(
            {"operation": "wait", "duration_s": 2, "target_temperature_K": 320}
        )
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.species_amounts == state["species_amounts"]
        result = runtime.execute({"operation": "measure", "instrument": "membrane_meter"})
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.ledger.time_s == 0
    finally:
        runtime.close()


def test_original_network_cannot_silently_change_ion_identity():
    model = spec()
    composition = distributed_ro_world_spec(model, feed(model))
    network = composition.physics.network
    altered = replace(
        network,
        species=tuple(
            replace(s, charge=0) if s.species_id == "Na+" else s for s in network.species
        ),
    )
    bad = replace(composition, physics=replace(composition.physics, network=altered))
    with pytest.raises(ValueError, match="species/charge"):
        DistributedRORuntime(bad).validate_world()
