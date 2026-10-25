"""Local collector inventories retain one equipment lifetime and shared clock."""

import json

import pytest

from chemworld.physchem.battery_collector_spm import (
    collector_spm_fields,
    default_collector_spm_spec,
)
from chemworld.physchem.battery_models import F
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session():
    spec = default_collector_spm_spec(cells=4, collector_cells=3)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=4)
    return runtime


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_collector_meter_time_and_terminal_charge_reach_actual_local_inventories():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 0.03})
    execute(runtime, {"operation": "wait", "duration_s": 2.0})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, slot, _, final, _ = unpack_battery(runtime._state)
    charge = 0.03 * (final.time_s - initial.time_s)
    assert final.time_s > 2.0  # The meter consumes shared laboratory time too.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(charge, abs=1e-9)
    for electrode, sign in (("negative", -1), ("positive", 1)):
        delta = sum(
            final.amounts_mol[k] - initial.amounts_mol[k]
            for k in spec.inventory_keys
            if k.startswith(electrode + ":")
        )
        assert delta == pytest.approx(sign * charge / F, abs=1e-11)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-11
    )
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-8
    )
    assert len(slot.values) == len(spec.inventory_keys) + 2
    assert slot.spec.owner_kind == "equipment"
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential",
        "electrical_current",
    }
    fields = collector_spm_fields(spec, final, current_A=0.03)
    assert fields["positive_reaction_current_A"] == pytest.approx(
        -fields["negative_reaction_current_A"], abs=1e-12
    )


def test_collector_midqueue_json_resume_is_exact_and_material_reset_keeps_device():
    runtime = session()
    for current, duration in ((0.03, 3.0), (0.0, 2.0), (-0.02, 4.0)):
        execute(
            runtime,
            {
                "operation": "queue_electrical_stage",
                "current_A": current,
                "stage_duration_s": duration,
            },
        )
    execute(runtime, {"operation": "wait", "duration_s": 1.0})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.0})
    assert runtime._state.to_dict() == restored._state.to_dict()
    _, slot, _, final, _ = unpack_battery(runtime._state)
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    assert unpack_battery(runtime._state)[3].amounts_mol == final.amounts_mol


def test_collector_undeclared_heat_is_rejected_without_mutating_state():
    runtime = session()
    before = runtime._state.to_dict()
    spec = unpack_battery(runtime._state)[2]
    with pytest.raises(ValueError, match=r"collector.*thermal"):
        battery_world_spec(spec, thermal_device=object())
    assert runtime._state.to_dict() == before
