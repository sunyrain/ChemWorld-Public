"""Actual node temperatures, source enthalpy and original world queue integration."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.physchem.gas_network_energy import gas_reaction_energy_key
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import gas_accounting_slot, unpack_gas_network
from chemworld.world.gas_reactor_network import gas_reactor_network_world


@pytest.fixture(scope="module")
def reference():
    directory = str(Path(__file__).parent / "reference")
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("run_gas_network35_energy")
    finally:
        sys.path.remove(directory)


@pytest.mark.parametrize("mode", ("inert", "adiabatic", "heating", "cooling"))
def test_temperatures_flow_enthalpy_and_physical_reaction_energy_match_independent_u(
    reference, mode
):
    record = reference.Record(mode)
    reference.physical(record, reference.cases.case("loop", mode))
    assert all(c["passed"] for c in record.checks), [c for c in record.checks if not c["passed"]]


@pytest.mark.parametrize(
    "name", ("constant-power", "newton-cooling", "two-body-conduction", "closed-exotherm")
)
def test_analytic_energy_limits(reference, name):
    record = reference.Record(name)
    reference.physical(record, reference.cases.analytic(name), analytic=True)
    assert all(c["passed"] for c in record.checks), [c for c in record.checks if not c["passed"]]


@pytest.mark.parametrize("name", ("inventory", "high-T", "low-T", "pressure"))
def test_earliest_inventory_temperature_or_pressure_event(reference, name):
    record = reference.Record(name)
    reference.physical(record, reference.cases.event(name)[0], event=True)
    assert all(c["passed"] for c in record.checks), [c for c in record.checks if not c["passed"]]


@pytest.mark.parametrize("kind", ("queue", "json", "process", "rollback"))
def test_original_world_selected_member_queue_replay_and_rollback(reference, kind, tmp_path):
    record = reference.Record(kind)
    reference.world(record, kind, tmp_path)
    assert all(c["passed"] for c in record.checks), [
        c["label"] for c in record.checks if not c["passed"]
    ]


def test_inert_temperature_coordinate_work_is_not_reported_as_reaction_heat(reference):
    card = reference.cases.case("loop", "inert")
    spec, network, _ = reference.cases.native(card)
    state = pack_gas_reactor_network_state(spec, network, card["amounts"])
    result, diagnostic = advance_gas_reactor_network(spec, network, state, 1)
    assert any(
        row["temperature_K"] != spec.initial_temperatures_K[name]
        for name, row in diagnostic["nodes"].items()
    )
    assert all(row["chemical_energy_change_J"] == 0 for row in diagnostic["nodes"].values())
    assert all(
        result.final_state.coordinates[gas_reaction_energy_key(n.vessel_id)] == 0
        for n in spec.nodes
    )
    slot = gas_accounting_slot(spec, network, "reactor:gas-network")
    assert all(f.unit != "K" for f in slot.spec.fields)


def test_energy_codec_and_original_isothermal_codec_are_distinct(reference):
    spec, _, _ = reference.cases.native(reference.cases.case())
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    old = replace(spec, energy_nodes=(), heat_links=())
    assert set(old.to_dict()) == {
        "model",
        "nodes",
        "connections",
        "temperature_K",
        "anchor_vessel_id",
    }
    assert old.to_dict()["model"] == "finite-isothermal-gas-network-v1"
    assert GasReactorNetworkSpec.from_dict(old.to_dict()) == old
    payload = spec.to_dict()
    payload["model"] = old.to_dict()["model"]
    with pytest.raises(ValueError, match="model version"):
        GasReactorNetworkSpec.from_dict(payload)


@pytest.mark.parametrize(
    "defect", ("missing-node", "anchor-temperature", "zero-capacity", "bad-thermo")
)
def test_rejects_incomplete_or_inconsistent_energy_authoring(reference, defect):
    spec, network, _ = reference.cases.native(reference.cases.case())
    with pytest.raises(ValueError):
        if defect == "missing-node":
            replace(spec, energy_nodes=spec.energy_nodes[:1])
        elif defect == "anchor-temperature":
            replace(spec, temperature_K=500)
        elif defect == "zero-capacity":
            GasNodeEnergySpec("reactor", 400, 0)
        else:
            gas_reactor_network_system(
                replace(
                    spec,
                    energy_nodes=tuple(
                        replace(n, maximum_temperature_K=2100) for n in spec.energy_nodes
                    ),
                ),
                network,
            )


@pytest.mark.parametrize("defect", ("enabled", "isothermal", "second", "member"))
def test_fixed_anchor_heater_contract_rejects_extra_or_moved_devices(reference, defect):
    card = reference.cases.case(mode="heating")
    spec, network, heater = reference.cases.native(card)
    if defect in {"enabled", "isothermal"}:
        with pytest.raises(ValueError, match="starts disabled"):
            gas_reactor_network_world(
                spec,
                network,
                card["amounts"],
                anchor_thermal_device=(
                    heater
                    if defect == "enabled"
                    else replace(heater, enabled=False, mode="isothermal")
                ),
            )
        return
    composition = gas_reactor_network_world(
        spec, network, card["amounts"], anchor_thermal_device=replace(heater, enabled=False)
    )
    states = dict(composition.physics.initial_states)
    original = states["reactor"].equipment.equipment["reactor:heater"]
    target = "reactor" if defect == "second" else "source"
    records = dict(states[target].equipment.equipment)
    records["extra"] = replace(original, equipment_id="extra", attached_vessel_id=target)
    states[target] = states[target].replace(equipment=EquipmentLedger(records))
    anchor = states["reactor"].replace(inactive_vessels={"source": states["source"]})
    with pytest.raises(ValueError, match=r"one anchor|fixed"):
        unpack_gas_network(anchor, network)


def test_target_outside_actual_device_range_does_not_queue(reference):
    runtime = reference.session()
    try:
        before = runtime._state.lab_clock.to_dict()
        result = runtime.execute(
            {"operation": "queue_lab_heat", "target_temperature_K": 1600, "duration_s": 1}
        )
        assert result.info["operation_outcome"] == "rejected"
        assert runtime._state.lab_clock.to_dict() == before
        assert runtime._state.ledger.time_s == 0
    finally:
        runtime.close()


def test_thermal_world_without_heater_keeps_wait_but_has_no_heat_queue(reference):
    card = reference.cases.case()
    spec, network, _ = reference.cases.native(card)
    composition = gas_reactor_network_world(spec, network, card["amounts"])
    assert "queue_lab_heat" not in composition.task.operations
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=7)
        reference.execute(runtime, "wait", duration_s=0.1)
        assert runtime._state.temperature_K != spec.temperature_K
    finally:
        runtime.close()
