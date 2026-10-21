"""Explicit entity boundaries within the one physical laboratory clock."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.examples.multi_device_lab import multi_device_lab_world
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states


def budget():
    return CampaignResourceCard(
        card_id="shared-lifecycle", operation_attempt_limit=160, vessel_start_limit=1,
        final_assay_limit=1, nonfinal_instrument_use_limit=4,
        stock_limits={"solvent_L": .03}, per_instrument_limits={"programmed_hplc": 4},
    )


def session(noise="sequential", *, resource_card=None):
    runtime = ChemWorldRuntime(composition=multi_device_lab_world(), observation_noise_mode=noise,
                               campaign_resource_card=resource_card)
    runtime.reset(seed=11)
    return runtime


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        "failed": {key: value for key, value in result.info.get("preconditions", {}).items()
                   if not value},
        **{key: result.info.get(key) for key in ("transaction_status", "rollback_reason",
                                               "outcome_diagnostics")},
    }
    return result


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("scope", ("material", "vessel"))
@pytest.mark.parametrize("target", ("reactor", "sample"))
def test_idle_replacement_preserves_clock_resources_equipment_and_published_reading(
    noise, scope, target,
):
    runtime = session(noise)
    try:
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "wait", duration_s=60)
        before = runtime._state
        local = local_states(before)[target]
        runtime.reset(options={"scope": scope, "vessel_ids": [target]})
        after = runtime._state
        assert after.ledger == before.ledger
        assert after.lab_clock == before.lab_clock
        assert after.vessel_elapsed_s == before.vessel_elapsed_s
        assert after.samples.samples == before.samples.samples
        assert local_states(after)[target].equipment == local.equipment
        assert local_states(after)[target].state_slots == local.state_slots
        assert runtime._operation_id == 2
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "wait", duration_s=60)
        assert runtime._state.lab_clock.completed[0] == before.lab_clock.completed[0]
        record = next(r for frame in local_states(runtime._state).values()
                      for r in frame.equipment.equipment.values() if r.equipment_id == "hplc-a")
        assert record.settings["use_count"] == 2
        assert runtime._state.ledger.time_s == 120
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("scope", ("material", "vessel", "equipment"))
@pytest.mark.parametrize("running", (False, True))
def test_related_queued_and_running_jobs_block_entity_change_without_mutation(
    noise, scope, running,
):
    runtime = session(noise)
    try:
        execute(runtime, "queue_lab_measure", device="hplc-a")
        if running:
            execute(runtime, "wait", duration_s=.25)
        before = runtime.snapshot().to_dict()
        options = {"scope": scope, "vessel_ids": ["reactor"]}
        if scope == "equipment":
            record = local_states(runtime._state)["reactor"].equipment.equipment["heater-a"]
            options = {"scope": scope, "equipment_id": "heater-a",
                       "replacement": replace(record, equipment_id="new-heater").to_dict()}
        with pytest.raises(ValueError, match=r"queued/running jobs 0:hplc-a:reactor"):
            runtime.reset(options=options)
        assert runtime.snapshot().to_dict() == before
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_unrelated_analysis_continues_across_an_idle_vessel_material_boundary(noise):
    runtime = session(noise)
    try:
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "wait", duration_s=.25)
        before = runtime._state
        runtime.reset(options={"scope": "material", "vessel_ids": ["sample"]})
        assert runtime._state.lab_clock == before.lab_clock
        assert runtime._state.equipment == before.equipment
        assert runtime._state.state_slots == before.state_slots
        execute(runtime, "wait", duration_s=60)
        assert runtime._state.ledger.time_s == 60
        assert runtime._state.lab_clock.completed[0]["available_at"] == 60
        assert runtime._state.ledger.cost == pytest.approx(.05)
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0002)
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_new_analyzer_identity_owns_fresh_state_and_snapshot_future(noise, tmp_path):
    runtime = session(noise, resource_card=budget())
    restored = None
    try:
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "wait", duration_s=60)
        before = runtime._state
        fresh = runtime._fresh_initial_state()
        record = fresh.equipment.equipment["hplc-a"]
        replacement = replace(record, equipment_id="replacement-hplc")
        slots = StateSlots(tuple(replace(slot, spec=replace(
            slot.spec, owner_id=replacement.equipment_id,
            slot_id=slot.spec.slot_id.replace("hplc-a", replacement.equipment_id),
        )) for slot in fresh.state_slots.slots if slot.spec.owner_id == "hplc-a"))
        runtime.reset(options={
            "scope": "equipment", "equipment_id": "hplc-a",
            "replacement": replacement.to_dict(), "slots": slots.to_dict(),
        })
        after = runtime._state
        assert after.ledger == before.ledger and after.lab_clock == before.lab_clock
        assert "hplc-a" not in after.equipment.equipment
        assert after.equipment.equipment[replacement.equipment_id].settings.get("use_count", 0) == 0
        assert all(slot.spec.owner_id != "hplc-a" for slot in after.state_slots.slots)
        assert all(value == 0 for slot in after.state_slots.slots
                   if slot.spec.owner_id == replacement.equipment_id
                   for value in slot.values.values())
        restored = ChemWorldRuntime.from_snapshot(runtime.snapshot())
        for operation, fields in (
            ("queue_lab_measure", {"device": replacement.equipment_id}),
            ("wait", {"duration_s": 60}),
        ):
            assert execute(runtime, operation, **fields) == execute(restored, operation, **fields)
        assert runtime._state.to_dict() == restored._state.to_dict()
        assert runtime._state.lab_clock.completed[0] == before.lab_clock.completed[0]
        assert runtime._state.lab_clock.completed[1]["device"] == replacement.equipment_id
        assert_new_process_future(runtime, tmp_path, {"operation": "wait", "duration_s": 1})
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_two_sorbent_cycles_preserve_loading_and_use_actual_finite_wash(noise, tmp_path):
    from chemworld.examples.shared_lifecycle_lab import sorbent_maintenance_lab

    runtime = ChemWorldRuntime(composition=sorbent_maintenance_lab(), observation_noise_mode=noise,
                               campaign_resource_card=budget())
    runtime.reset(seed=2)

    def loading():
        return {key: value for slot in runtime._state.state_slots.slots
                for key, value in slot.values.items() if key.startswith("loading:")}

    try:
        for cycle in (1, 2):
            if cycle == 2:
                retained, ledger = loading(), runtime._state.ledger
                wash = runtime._state.inactive_vessels["wash_supply"].to_dict()
                runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
                assert loading() == retained
                assert runtime._state.ledger == ledger
                assert runtime._state.inactive_vessels["wash_supply"].to_dict() == wash
            previous = sum(loading().values())
            execute(runtime, "wait", duration_s=300)
            loaded = loading()
            assert sum(loaded.values()) > previous
            execute(runtime, "route_material", connection="reactor-to-waste",
                    transfer_fraction=1, mixing=1)
            assert loading() == loaded and runtime._state.volume_L == 0
            execute(runtime, "add_solvent", solvent=0, volume_L=.01)
            assert runtime._state.ledger.time_s == 360 * (cycle - 1) + 300
            execute(runtime, "queue_lab_heat", target_temperature_K=350, duration_s=60)
            execute(runtime, "wait", duration_s=60)
            assert sum(loading().values()) < sum(loaded.values())
            for component in ("A", "B"):
                assert runtime._state.species_amounts[component] == pytest.approx(
                    loaded["loading:" + component] - loading()["loading:" + component], abs=1e-12,
                )
            assert runtime._state.inactive_vessels["wash_supply"].volume_L == pytest.approx(
                .03 - .01 * cycle, abs=1e-12,
            )
            assert runtime._state.vessel_elapsed_s == dict.fromkeys(
                runtime._state.vessel_elapsed_s, 360 * cycle,
            )
        before = runtime.snapshot().to_dict()
        with pytest.raises(ValueError, match="cannot replenish a finite supply"):
            runtime.reset(options={"scope": "material", "vessel_ids": ["wash_supply"]})
        assert runtime.snapshot().to_dict() == before
        assert_new_process_future(runtime, tmp_path, {"operation": "wait", "duration_s": 1})
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_two_battery_cycles_preserve_actual_lithium_age_temperature_and_control(noise, tmp_path):
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.physchem.battery_degradation import SEISpec
    from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec
    from chemworld.runtime.battery import initial_battery_slot, unpack_battery

    world = reactive_closed_lab(battery_spec=BatterySPMSpec(
        thermal=BatteryThermalSpec(), sei=SEISpec(),
    ))
    world = replace(world, task=replace(world.task, operations=(
        *world.task.operations, "configure_electrical_control",
    )))
    runtime = ChemWorldRuntime(composition=world, observation_noise_mode=noise,
                               campaign_resource_card=budget())
    runtime.reset(seed=4)
    try:
        execute(runtime, "select_vessel", vessel="cell")
        initial_carbonate = unpack_battery(runtime._state)[1].values["sei:carbonate"]
        for cycle in (1, 2):
            for current in (.02, -.02):
                execute(runtime, "configure_electrical_control", current_A=current)
                execute(runtime, "wait", duration_s=10)
            before = runtime._state
            slot = unpack_battery(before)[1]
            assert slot.values["sei:carbonate"] > initial_carbonate
            runtime.reset(options={"scope": "material", "vessel_ids": ["cell"]})
            assert unpack_battery(runtime._state)[1] == slot
            assert runtime._state.temperature_K == before.temperature_K
            assert runtime._state.equipment == before.equipment
            assert runtime._state.ledger == before.ledger
            assert runtime._state.lab_clock == before.lab_clock
            assert runtime._state.ledger.time_s == 20 * cycle
        before = runtime._state
        resource_events = runtime.campaign_resource_snapshot()
        template = local_states(runtime._fresh_initial_state())["cell"]
        record = template.equipment.equipment["battery"]
        replacement = replace(record, equipment_id="replacement-cell")
        model = BatterySPMSpec(**replacement.settings["physical_model"])
        runtime.reset(options={
            "scope": "equipment", "equipment_id": "battery",
            "replacement": replacement.to_dict(),
            "slots": StateSlots((initial_battery_slot(replacement, model),)).to_dict(),
        })
        record, slot, _, _, _ = unpack_battery(runtime._state)
        assert record.equipment_id == "replacement-cell"
        assert "battery" not in runtime._state.equipment.equipment
        assert slot.values["sei:carbonate"] == initial_carbonate
        assert slot.values["sei:carbonate"] < unpack_battery(before)[1].values["sei:carbonate"]
        assert runtime._state.temperature_K == model.temperature_K
        assert runtime._state.ledger == before.ledger
        assert runtime.campaign_resource_snapshot() == resource_events
        assert_new_process_future(runtime, tmp_path, {"operation": "wait", "duration_s": 1})
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_two_membrane_cycles_retain_fouling_and_consume_actual_rinse_supply(noise, tmp_path):
    from chemworld.examples.open_shared_labs import open_shared_lab
    from chemworld.runtime.membranes import FOULING_STATE

    runtime = ChemWorldRuntime(
        composition=open_shared_lab("membrane"), observation_noise_mode=noise,
        campaign_resource_card=budget(),
    )
    runtime.reset(seed=3)

    def bound():
        return next(s.values["bound_mol"] for s in runtime._state.state_slots.slots
                    if s.spec.state_type == FOULING_STATE)

    try:
        for cycle in (1, 2):
            if cycle == 2:
                before = runtime._state
                runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
                assert bound() == next(s.values["bound_mol"] for s in before.state_slots.slots
                                       if s.spec.state_type == FOULING_STATE)
                assert runtime._state.equipment == before.equipment
                assert runtime._state.inactive_vessels == before.inactive_vessels
                assert runtime._state.ledger == before.ledger
            execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="process")
            old_bound = bound()
            execute(runtime, "wait", duration_s=5)
            loaded = bound()
            assert loaded > old_bound
            wash = runtime._state.inactive_vessels["wash"].volume_L
            waste = runtime._state.inactive_vessels["waste"].species_amounts["foulant"]
            execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="rinse")
            execute(runtime, "wait", duration_s=5)
            assert bound() < loaded
            assert runtime._state.inactive_vessels["wash"].volume_L < wash
            assert runtime._state.inactive_vessels["waste"].species_amounts["foulant"] > waste
            assert runtime._state.ledger.time_s == 10 * cycle
        before = runtime.snapshot().to_dict()
        with pytest.raises(ValueError, match="cannot replenish a finite supply"):
            runtime.reset(options={"scope": "vessel", "vessel_ids": ["wash"]})
        assert runtime.snapshot().to_dict() == before
        assert_new_process_future(runtime, tmp_path, {"operation": "wait", "duration_s": 1})
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_actual_archived_sample_survives_local_material_and_vessel_changes(noise):
    from chemworld.examples.shared_lifecycle_lab import sample_archive_lab

    runtime = ChemWorldRuntime(composition=sample_archive_lab(), observation_noise_mode=noise)
    runtime.reset(seed=9)
    try:
        execute(runtime, "create_container", container="container-01", capacity_L=.01)
        source_volume = runtime._state.volume_L
        execute(runtime, "transfer_material", source_container="active",
                destination_container="container-01", transfer_fraction=.05, mixing=0)
        assert runtime._state.samples.samples["container-01"].volume_L == pytest.approx(
            source_volume * .05, abs=1e-12,
        )
        before = runtime._state.samples.samples
        for scope in ("material", "vessel"):
            outlet = sum(phase.solvents.volume_L for phase in runtime._state.phases.phases.values())
            runtime.reset(options={"scope": scope, "vessel_ids": ["reactor"]})
            assert runtime._state.samples.samples == before
            receipt = runtime._lifecycle_history[-1]
            assert sum(receipt["replacement_carrier_inlet_L"].values()) == pytest.approx(
                source_volume, abs=1e-12,
            )
            assert sum(receipt["replacement_carrier_outlet_L"].values()) == pytest.approx(
                outlet, abs=1e-12,
            )
            execute(runtime, "wait", duration_s=1)
            assert runtime._state.samples.samples == before
    finally:
        runtime.close()


def assert_new_process_future(runtime, directory, action):
    source, output = directory / "lifecycle.json", directory / "future.json"
    runtime.snapshot().save(source)
    before = runtime.campaign_resource_snapshot()
    restored = ChemWorldRuntime.from_snapshot(runtime.snapshot())
    assert restored.campaign_resource_snapshot() == before
    expected = execute(runtime, action["operation"], **{
        key: value for key, value in action.items() if key != "operation"
    })
    assert restored.execute(action) == expected
    restored.close()
    code = (
        "import json,sys; from pathlib import Path; "
        "from chemworld.runtime import ChemWorldRuntime,SessionSnapshot; "
        "r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1])); "
        "result=r.execute(json.loads(sys.argv[3])); "
        "assert result.info['transaction_status']=='committed'; "
        "Path(sys.argv[2]).write_text(json.dumps({'state':r._state.to_dict(),"
        "'resources':r.campaign_resource_snapshot(),'history':r._lifecycle_history},allow_nan=False))"
    )
    subprocess.run([sys.executable, "-c", code, str(source), str(output), json.dumps(action)],
                   check=True, capture_output=True, text=True, env=os.environ.copy())
    assert json.loads(output.read_text()) == json.loads(json.dumps({
        "state": runtime._state.to_dict(), "resources": runtime.campaign_resource_snapshot(),
        "history": runtime._lifecycle_history,
    }))


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("running", (False, True))
def test_a_connected_system_rejects_local_changes_when_any_member_has_a_job(noise, running):
    from dataclasses import asdict

    from chemworld.examples.open_shared_labs import open_shared_lab
    from chemworld.foundation import WorldState
    from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger
    from chemworld.physchem.chromatography_methods import EmpiricalChromatographyAnalyteSpec
    from chemworld.physchem.membranes import MembraneSpec
    from chemworld.runtime.chromatography import MEMORY_TYPE, initial_memory
    from chemworld.runtime.membranes import membrane_record
    from chemworld.world.composition import WorldCompositionSpec

    payload = open_shared_lab("membrane").to_dict()
    for component in payload["components"]:
        if component["kind"] == "observation" and component["vessel"] == "reactor":
            component["parameters"]["instruments"].append("programmed_hplc")
    # The empirical fixture declares exactly the actual sampled species. This
    # checks finite aliquot occupation, not the accuracy of a salt detector.
    bottle = WorldState.from_dict(payload["physics"]["initial_states"]["bottle"])
    record = bottle.equipment.equipment["instrument:programmed_hplc"]
    analytes = tuple(replace(
        EmpiricalChromatographyAnalyteSpec(**item), analyte_id=identity,
        provenance_id="synthetic-aliquot-admission-fixture-not-salt-detector-validation",
    ) for item, identity in zip(record.settings["chromatography"]["analytes"],
                               ("water", "nacl", "foulant"), strict=True))
    record = replace(record, settings={**record.settings, "chromatography": {
        **record.settings["chromatography"], "analytes": [asdict(a) for a in analytes],
    }})
    bottle = bottle.replace(
        species_amounts={**bottle.species_amounts,
                         **dict.fromkeys(("water", "nacl", "foulant"), 0.0)},
        phases=PhaseLedger({key: replace(phase, species_amounts_mol={
            **phase.species_amounts_mol, **dict.fromkeys(("water", "nacl", "foulant"), 0.0),
        }) for key, phase in bottle.phases.phases.items()}),
        equipment=EquipmentLedger({**bottle.equipment.equipment, record.equipment_id: record}),
        state_slots=StateSlots((
            *(slot for slot in bottle.state_slots.slots if slot.spec.state_type != MEMORY_TYPE),
            initial_memory("programmed_hplc", "bottle", analytes),
        )),
    )
    payload["physics"]["initial_states"]["bottle"] = bottle.to_dict()
    caloric = payload["physics"]["models"]["thermal"]["bindings"]["bottle"]["thermodynamics"]
    for identity in ("water", "nacl", "foulant"):
        caloric["species"][identity] = {"u_ref_J_mol": 0.0, "cv_J_mol_K": 30.0}

    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload),
                               observation_noise_mode=noise)
    runtime.reset(seed=3)
    try:
        execute(runtime, "queue_lab_measure")
        if running:
            execute(runtime, "wait", duration_s=.25)
            memory = next(slot for slot in runtime._state.state_slots.slots
                          if slot.spec.state_type == MEMORY_TYPE)
            assert memory.values["water"] > 0
        before = runtime.snapshot().to_dict()
        record = membrane_record(runtime._state)
        model = MembraneSpec.from_dict(record.settings["physical_model"])
        for target in ("reactor", model.collector_vessel_id, model.fouling.waste_vessel_id):
            with pytest.raises(ValueError, match="queued/running jobs"):
                runtime.reset(options={"scope": "material", "vessel_ids": [target]})
        with pytest.raises(ValueError, match="queued/running jobs"):
            runtime.reset(options={
                "scope": "equipment", "equipment_id": record.equipment_id,
                "replacement": replace(record, equipment_id="replacement-membrane").to_dict(),
            })
        assert runtime.snapshot().to_dict() == before
    finally:
        runtime.close()


@pytest.mark.parametrize("violation", ("identity", "type", "instrument_contract"))
def test_incompatible_device_replacement_has_a_specific_reason_and_no_state_change(violation):
    runtime = session()
    try:
        record = runtime._state.equipment.equipment["hplc-a"]
        changes = {"equipment_id": "replacement-hplc"}
        if violation == "identity":
            changes["equipment_id"] = "hplc-b"
            reason = "new unique identity"
        elif violation == "type":
            changes["equipment_type"] = "thermal_device"
            reason = "installed equipment type"
        else:
            changes["settings"] = {**record.settings, "instrument_id": "programmed_gc"}
            reason = "declared instrument contract"
        before = runtime.snapshot().to_dict()
        with pytest.raises(ValueError, match=reason):
            runtime.reset(options={
                "scope": "equipment", "equipment_id": record.equipment_id,
                "replacement": replace(record, **changes).to_dict(),
            })
        assert runtime.snapshot().to_dict() == before
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_replacing_stochastic_material_retains_rng_and_shared_physical_time(noise):
    from chemworld.examples.closed_shared_labs import reactive_closed_lab

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(), observation_noise_mode=noise)
    runtime.reset(seed=5)
    try:
        execute(runtime, "wait", duration_s=.25)
        execute(runtime, "select_vessel", vessel="ssa")
        before = runtime._state
        continuation = before.process.stochastic_continuation
        runtime.reset(options={"scope": "material", "vessel_ids": ["ssa"]})
        after = runtime._state.process.stochastic_continuation
        assert after["rng_state"] == continuation["rng_state"]
        assert after["sampling_rng_state"] == continuation["sampling_rng_state"]
        assert after["events_committed"] == continuation["events_committed"]
        assert after["pending"] is None and after["signature"] is None
        assert after["time_s"] == runtime._state.ledger.time_s == .25
        assert runtime._state.lab_clock == before.lab_clock
        execute(runtime, "wait", duration_s=1)
        assert runtime._state.process.stochastic_continuation["time_s"] == 1.25
    finally:
        runtime.close()


@pytest.mark.parametrize("extra", ({"scheduling": "automatic"}, {"unexpected": True}))
def test_manual_transfer_metadata_is_explicit_and_strict(extra):
    from chemworld.examples.shared_lifecycle_lab import sorbent_maintenance_lab
    from chemworld.world.composition import WorldCompositionError, WorldCompositionSpec

    payload = sorbent_maintenance_lab().to_dict()
    payload["connections"][0].update(extra)
    with pytest.raises(WorldCompositionError, match="optional scheduling=manual"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


def test_model_driven_flow_connection_cannot_be_reused_as_a_manual_transfer():
    from chemworld.agent_interface import action_schema, validate_action
    from chemworld.examples.open_shared_labs import open_shared_lab

    world = open_shared_lab("transport")
    world = replace(world, task=replace(world.task, operations=(*world.task.operations,
                                                               "route_material")))
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=2)
    try:
        edge = world.connections[0]
        components = {component.id: component for component in world.components}
        execute(runtime, "select_vessel", vessel=components[edge["source"]["component"]].vessel)
        before = runtime._state
        schema = action_schema(runtime, "route_material")
        assert next(field["choices"] for field in schema["fields"]
                    if field["field"] == "connection") == []
        action = {"operation": "route_material", "connection": edge["id"],
                  "transfer_fraction": .1, "mixing": 1}
        assert not validate_action(runtime, action)["valid"]
        result = runtime.execute(action)
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.lab_clock == before.lab_clock
        assert runtime._state.ledger.time_s == before.ledger.time_s
        assert runtime._state.species == before.species
        assert runtime._state.equipment == before.equipment
        assert runtime._state.inactive_vessels == before.inactive_vessels
    finally:
        runtime.close()


@pytest.mark.parametrize("thermal", (False, True))
def test_membrane_does_not_ignore_extra_dynamic_or_thermal_owned_inventory(thermal):
    from chemworld.examples.open_shared_labs import open_shared_lab
    from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
    from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
    from chemworld.runtime.membranes import unpack_membrane

    runtime = ChemWorldRuntime(composition=open_shared_lab("membrane"))
    runtime.reset(seed=2)
    try:
        state = runtime._state
        record = EquipmentRecord("extra-device", "auxiliary_device", state.vessel_id)
        field = (StateFieldSpec("capacity", "J/K") if thermal
                 else StateFieldSpec("stock", "mol", species_id="water"))
        slot = StateSlot(StateSlotSpec(
            "extra-device:stock", "equipment", record.equipment_id, "extra-owned-state",
            (field,), heat_capacity_field="capacity" if thermal else None,
            time_evolution="static" if thermal else "coupled",
        ), state.vessel_id, {field.name: 1.0})
        candidate = state.replace(
            equipment=EquipmentLedger({**state.equipment.equipment, record.equipment_id: record}),
            state_slots=StateSlots((*state.state_slots.slots, slot)),
        )
        with pytest.raises(ValueError, match="caloric ownership"):
            unpack_membrane(candidate)
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("scope", ("material", "vessel"))
def test_hot_retained_body_and_new_material_share_the_declared_energy_reference(noise, scope):
    runtime = session(noise)
    try:
        execute(runtime, "queue_lab_heat", device="heater-a",
                target_temperature_K=320, duration_s=2)
        execute(runtime, "wait", duration_s=2)
        before = runtime._state
        initial = runtime.compiled_composition.spec.physics.initial_states["reactor"]
        assert before.temperature_K > initial.temperature_K
        # The authored fixture has a 20 J/K vessel and 30 J/(mol K) solutes;
        # the empty detector memory has no additional retained heat capacity.
        fluid_capacity = 30 * sum(initial.species_amounts.values())
        expected = (20 * before.temperature_K + fluid_capacity * initial.temperature_K) / (
            20 + fluid_capacity
        ) if scope == "material" else initial.temperature_K
        runtime.reset(options={"scope": scope, "vessel_ids": ["reactor"]})
        assert runtime._state.temperature_K == pytest.approx(expected, abs=1e-10)
        assert runtime._state.ledger == before.ledger
        assert runtime._state.equipment == before.equipment
        assert runtime._state.lab_clock == before.lab_clock
    finally:
        runtime.close()
