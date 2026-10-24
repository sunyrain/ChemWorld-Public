"""Equipment boundaries preserve an independent hybrid's complete Poisson state."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_msmr import default_spec
from chemworld.examples.native_domains import _rename
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.reaction_network import SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.runtime.scheduling import local_states
from chemworld.world.battery_world import battery_world_spec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world
from reference.stochastic_hybrid51_cases import author


def _world():
    network, hybrid, amounts = author()
    # The global chemical declaration includes equipment-owned Li. Its zero
    # amount in the hybrid is no reservoir; bind the continuation only after
    # extending the network, so its exact inventory/signature contract holds.
    network = replace(network, species=(*network.species, SpeciesSpec("Li", "Li")))
    payload = stochastic_hybrid_world(network, hybrid, {**amounts, "Li": 0.},
                                     seed=17).to_dict()
    battery = _rename(battery_world_spec(default_spec(cells=2)).to_dict(), "reactor", "cell")
    cell = battery["physics"]["initial_states"]["cell"]
    cell["equipment"]["battery"]["settings"]["electrical_control"]["program"] = (
        ScalarControlProgram(((0., .12),), "A").to_dict())
    payload["composition_id"] = "hybrid-independent-msmr-lifecycle"
    payload["vessels"].extend(battery["vessels"])
    payload["components"][0].update(id="reactor:stochastic", vessel="reactor")
    payload["components"].extend(c for c in battery["components"] if c["kind"] == "battery_spm")
    payload["components"].extend({
        "id": name + ":clock", "vessel": name, "kind": "lab_schedule",
        "parameters": {"duration_range_s": [1e-6, 14400]},
    } for name in ("reactor", "cell"))
    payload["physics"]["initial_states"]["cell"] = cell
    payload["physics"]["lab_clock"] = LabClockSpec(vessel_capacity=2).to_dict()
    payload["task"].update(operations=["select_vessel", "configure_electrical_control",
                                       "wait", "terminate"], budget=80)
    return WorldCompositionSpec.from_dict(payload)


def _execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.fixture
def running_world():
    runtime = ChemWorldRuntime(composition=_world())
    runtime.reset(seed=4)
    _execute(runtime, {"operation": "wait", "duration_s": 1.})
    states = local_states(runtime._state)
    assert runtime._state.vessel_elapsed_s == {"reactor": 1., "cell": 1.}
    memory = states["reactor"].process.stochastic_continuation
    assert memory["time_s"] == 1.
    assert memory["pending"]["channel_ids"] == ["activate", "deactivate"]
    assert all(e > h >= 0 for h, e in zip(memory["pending"]["hazards"],
                                         memory["pending"]["targets"], strict=True))
    assert unpack_battery(states["cell"])[3].coordinates["discharged_charge_C"] == (
        pytest.approx(.12, abs=1e-8))
    try:
        yield runtime
    finally:
        runtime.close()


def _hybrid_physical(runtime):
    state = local_states(runtime._state)["reactor"]
    return deepcopy({key: getattr(state, key) for key in (
        "species_amounts", "phases", "temperature_K", "pressure_Pa", "volume_L",
        "state_slots", "equipment", "process", "samples",
    )})


@pytest.mark.parametrize("active_vessel", ["reactor", "cell"])
def test_hybrid_preserves_pending_risks_when_other_vessel_battery_is_replaced(
    running_world, active_vessel,
):
    runtime = running_world
    _execute(runtime, {"operation": "select_vessel", "vessel": active_vessel})
    before = _hybrid_physical(runtime)
    ledger, clock = runtime._state.ledger, runtime._state.lab_clock
    vessel_times = deepcopy(runtime._state.vessel_elapsed_s)
    old_history = deepcopy(runtime._lifecycle_history)
    control = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    record, _, old, previous, _ = unpack_battery(local_states(runtime._state)["cell"])
    new = replace(old, initial_negative_fraction=.5)
    replacement = replace(record, equipment_id="cell:replacement-msmr", settings={
        **record.settings, "physical_model": new.to_dict()})
    fresh = initial_battery_slot(replacement, new)
    restored = None
    try:
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots((fresh,))})
        cell = local_states(runtime._state)["cell"]
        installed, slot, spec, physical, _ = unpack_battery(cell)
        assert installed == replacement and record.equipment_id not in cell.equipment.equipment
        assert slot == fresh and spec == new
        assert physical.amounts_mol == new.initial_state().amounts_mol
        assert physical.amounts_mol != previous.amounts_mol
        assert physical.time_s == 1.
        assert _hybrid_physical(runtime) == before
        assert runtime._state.ledger == ledger
        assert runtime._state.lab_clock == clock
        assert runtime._state.vessel_elapsed_s == vessel_times
        assert runtime._lifecycle_history[:-1] == old_history
        receipt = runtime._lifecycle_history[-1]
        assert receipt["removed_device"]["attached_vessel_id"] == "cell"
        assert receipt["introduced_device"]["attached_vessel_id"] == "cell"
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for instance in (runtime, restored, control):
            _execute(instance, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
        # The unrelated replacement consumes neither random draws nor pending
        # channel risk, including their effect on the next actual wait.
        assert _hybrid_physical(runtime) == _hybrid_physical(control)
    finally:
        control.close()
        if restored is not None:
            restored.close()


def test_hybrid_owned_equipment_replacement_rejects_without_mutation(running_world):
    runtime = running_world
    _execute(runtime, {"operation": "select_vessel", "vessel": "cell"})
    hybrid = local_states(runtime._state)["reactor"]
    record = next(r for r in hybrid.equipment.equipment.values()
                  if r.equipment_type == "batch_reactor")
    replacement = replace(record, equipment_id="reactor:replacement-device")
    before = deepcopy(runtime._state.to_dict())
    history = deepcopy(runtime._lifecycle_history)
    with pytest.raises(ValueError, match="hybrid partial lifecycle"):
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots()})
    assert runtime._state.to_dict() == before
    assert runtime._lifecycle_history == history
