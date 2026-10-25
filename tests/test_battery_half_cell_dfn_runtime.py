"""Dynamic-salt half cells use the original equipment, queue and meter."""

import json
from collections.abc import Mapping
from dataclasses import fields

import pytest

from chemworld.physchem.battery_half_cell import METAL_KEY
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_models import (
    F,
    battery_chemical_energy_j,
    battery_dissipation_w,
    battery_particle_flux,
    battery_sei_affinity_j_mol,
    battery_system,
)
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec
from test_battery_half_cell_dfn import model


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_dynamic_salt_half_cell_queue_json_meter_and_reset_keep_real_inventories():
    spec = model()
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=4)
    _, slot, decoded, initial, _ = unpack_battery(runtime._state)
    assert decoded == spec
    assert slot.spec.owner_kind == "equipment"
    assert set(initial.amounts_mol) == set(spec.inventory_keys)
    for current, duration in ((.03, 2.), (0., 1.), (-.02, 2.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(
        json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True)))
    for active in (runtime, restored):
        stopped = execute(active, {"operation": "wait", "duration_s": 5.})
        assert stopped.info["process_control"]["status"] == "completed"
        assert stopped.info["actual_duration_s"] == pytest.approx(4.)
        measured = execute(active, {"operation": "measure", "instrument": "battery_meter"})
        assert measured.info["actual_duration_s"] == pytest.approx(1.)
        assert {c.channel_id for c in measured.observation.channels} == {
            "terminal_potential", "electrical_current"}
    assert runtime._state.to_dict() == restored._state.to_dict()
    _, slot, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == pytest.approx(6.)
    charge = final.coordinates["discharged_charge_C"]
    assert charge == pytest.approx(.02, abs=1e-10)
    assert final.amounts_mol[METAL_KEY] - initial.amounts_mol[METAL_KEY] == pytest.approx(
        -charge / F, abs=1e-12)
    working = [k for k in spec.inventory_keys if k.startswith("positive:")]
    salt = [k for k in spec.inventory_keys if k.startswith("electrolyte:")]
    assert sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in working) == (
        pytest.approx(charge / F, abs=1e-12))
    assert sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in salt) == (
        pytest.approx(0., abs=1e-12))
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-9)
    values = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == values


def test_dynamic_salt_half_cell_refuses_undeclared_analysis_and_thermal_combinations():
    spec = model()
    state = spec.initial_state()
    program = ScalarControlProgram(((0., 0.),), "A")
    requests = (
        lambda: battery_impedance(spec, (1.,), state=state),
        lambda: soh_from_battery(spec, state),
        lambda: battery_spec_at_soc(spec, None, .5),
        lambda: battery_particle_flux(spec, state, "positive", 0.),
        lambda: battery_chemical_energy_j(spec, state),
        lambda: battery_dissipation_w(spec, state, 0.),
        lambda: battery_sei_affinity_j_mol(spec, state, 0.),
        lambda: battery_world_spec(spec, thermal_device=object()),
        lambda: battery_system(spec, program, thermal_device=object()),
    )
    def snapshot():
        return {field.name: dict(value) if isinstance(value, Mapping) else value
                for field in fields(state) for value in (getattr(state, field.name),)}

    before = snapshot()
    for request in requests:
        with pytest.raises(ValueError, match=r"(DFN|half.cell|Half.cell)"):
            request()
        assert snapshot() == before
