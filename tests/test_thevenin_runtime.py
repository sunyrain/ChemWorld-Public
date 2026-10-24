"""Persistent circuit ownership, real timed observations and original runtime replay."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.battery_thevenin import (
    CELL_T,
    SOC,
    WORK,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thevenin import METER, unpack_thevenin
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.thevenin_world import thevenin_world_spec
from test_battery_thevenin import fixture


def session(spec=None, **options):
    result = ChemWorldRuntime(composition=thevenin_world_spec(spec or fixture()), **options)
    result.reset(seed=7)
    return result


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_one_equipment_slot_owns_actual_circuit_and_in_situ_measurement_time():
    runtime = session()
    initial = runtime._state
    assert initial.phase == "formal" and initial.volume_L == 0.
    assert not initial.species_amounts and initial.liquid_heat_capacity_J_L_K is None
    _, slot, model, physical, _ = unpack_thevenin(initial)
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    before = runtime._state
    reading = execute(runtime, {"operation": "measure", "instrument": METER})
    _, _, _, after, program = unpack_thevenin(runtime._state)
    assert runtime._state.ledger.time_s == 3.
    assert before.ledger.time_s == 2. and initial.ledger.time_s == 0.
    assert after.coordinates[SOC] == pytest.approx(.7-1.5/100, abs=1e-9)
    assert after.coordinates[CELL_T] > physical.coordinates[CELL_T]
    assert runtime._state.temperature_K == initial.temperature_K
    fields = thevenin_fields(model, after, program)
    channels = reading.observation.channels
    assert {c.channel_id for c in channels} == {"terminal_potential", "electrical_current"}
    assert channels[0].value == pytest.approx(fields["voltage_V"], abs=1e-8)
    assert channels[1].value == pytest.approx(.5, abs=1e-9)
    assert runtime._state.ledger.sample_consumed_L == 0.
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(-after.coordinates[WORK])
    before_cached = runtime._state.to_dict()
    runtime.observe()
    assert runtime._state.to_dict() == before_cached


def test_actual_control_queue_switches_on_terminal_event_and_survives_snapshot():
    runtime = session()
    execute(runtime, {"operation": "queue_electrical_stage", "current_A": .5,
                      "stage_duration_s": 5., "stop_voltage_below_V": 3.60})
    execute(runtime, {"operation": "queue_electrical_stage", "current_A": 0.,
                      "stage_duration_s": 1.})
    execute(runtime, {"operation": "wait", "duration_s": .1})
    saved = json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True, allow_nan=False))
    restored = ChemWorldRuntime.from_snapshot(saved)
    for current in (runtime, restored):
        execute(current, {"operation": "wait", "duration_s": 4.9})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, _, _, final, _ = unpack_thevenin(runtime._state)
    assert record.settings["electrical_control"]["stage_index"] == 2
    assert 0. < final.coordinates["discharged_charge_C"] < 2.5
    # The existing finite controller stops the common slice once its queue is
    # complete; it does not silently spend the unused wait budget at rest.
    assert 1. < runtime._state.ledger.time_s < 5.
    assert runtime._state.lab_clock.stopped


def test_material_initialization_does_not_recharge_or_cool_retained_circuit():
    runtime = session()
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    before = runtime._state
    record, slot, _, _, _ = unpack_thevenin(before)
    runtime.reset(options={"scope": "material", "vessel_ids": [before.vessel_id]})
    retained_record, retained_slot, _, _, _ = unpack_thevenin(runtime._state)
    assert retained_record == record and retained_slot == slot
    assert runtime._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("defect", ("owner", "coordinate", "bulk", "thermal_capacity"))
def test_invalid_snapshot_state_is_not_silently_reinitialized(defect):
    runtime = session()
    payload = runtime.snapshot().to_dict()
    # Use the actual native state schema; no repair or defaults for missing state.
    if defect == "bulk":
        payload["world_state"]["volume_L"] = .01
    elif defect == "thermal_capacity":
        payload["world_state"]["liquid_heat_capacity_J_L_K"] = 100.
    else:
        slots = list(payload["world_state"]["state_slots"].values())
        if defect == "owner":
            slots[0]["spec"]["owner_id"] = "other-circuit"
        else:
            slots[0]["values"][SOC] = 1.5
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_real_agent_trajectory_replay_and_fresh_process_continuation(tmp_path, noise):
    from chemworld.examples.native_thevenin import run, workflow
    from test_native_domains_end_to_end import (
        test_snapshot_resume_in_middle_of_real_workflow as resume,
    )

    report = run(tmp_path / (noise+".jsonl"), noise=noise)
    assert report["steps"] == report["replay_steps"] == 9
    resume("Thevenin", noise, tmp_path, recipe=workflow(), sort_keys=True)


def two_circuit_world():
    from chemworld.examples.native_domains import _rename

    first = thevenin_world_spec(fixture()).to_dict()
    second = _rename(_rename(first, "circuit_workspace", "second_workspace"),
                     "circuit", "second_circuit")
    for component in second["components"]:
        component["id"] = "second:"+component["id"]
    first["vessels"] += second["vessels"]
    first["components"] += second["components"]
    first["physics"]["initial_states"].update(second["physics"]["initial_states"])
    first["physics"]["advance_bindings"].update(second["physics"]["advance_bindings"])
    for vessel, identity, current in (("circuit_workspace", "circuit", .5),
                                       ("second_workspace", "second_circuit", -.2)):
        first["physics"]["initial_states"][vessel]["equipment"][identity][
            "settings"]["electrical_control"]["program"]["setpoints"] = [[0., current]]
    return WorldCompositionSpec.from_dict(first)


def test_meter_advances_both_actual_circuits_once_not_only_the_selected_vessel():
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=two_circuit_world())
    runtime.reset(seed=7)
    result = execute(runtime, {"operation": "measure", "instrument": METER})
    local = local_states(runtime._state)
    assert runtime._state.vessel_elapsed_s == dict.fromkeys(local, 1.)
    for vessel, current in (("circuit_workspace", .5), ("second_workspace", -.2)):
        physical = unpack_thevenin(local[vessel])[3]
        assert physical.coordinates[SOC] == pytest.approx(.7-current/100, abs=1e-9)
        assert physical.coordinates["discharged_charge_C"] == pytest.approx(current, abs=1e-9)
    assert result.info["actual_duration_s"] == 1.
    assert runtime._state.ledger.cost == .01


def test_other_circuit_failure_rolls_back_the_complete_measurement_candidate(monkeypatch):
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from chemworld.runtime.thevenin import TheveninRuntime

    runtime = ChemWorldRuntime(composition=two_circuit_world())
    runtime.reset(seed=7)
    initial = runtime._state.to_dict()
    rng = json.dumps(runtime._rng.bit_generator.state, sort_keys=True)
    advance = TheveninRuntime.advance

    def fail_second(self, state, action, **kwargs):
        candidate = advance(self, state, action, **kwargs)
        if state.vessel_id == "second_workspace":
            return ModelProviderResult({}, {"test_fixture": "after_actual_candidate"},
                success=False, outcome="numerical_failure", failure_reason="injected_failure")
        return candidate

    monkeypatch.setattr(TheveninRuntime, "advance", fail_second)
    result = runtime.execute({"operation": "measure", "instrument": METER})
    assert result.info["transaction_status"] == "rolled_back"
    after = runtime._state.to_dict()
    # The original failure contract charges attempted-operation cost/risk, not
    # the rejected acquisition or physical time. Preserve those consequences.
    for ledger in ("ledger", "process"):
        assert after[ledger]["cost"] > initial[ledger]["cost"]
        assert after[ledger]["risk"] > initial[ledger]["risk"]
        for field in ("cost", "risk"):
            after[ledger][field] = initial[ledger][field]
    assert after == initial
    assert json.dumps(runtime._rng.bit_generator.state, sort_keys=True) == rng


@pytest.mark.parametrize("minimum_soc", (0., .2))
def test_measurement_keeps_the_actual_physical_stop_and_marks_incomplete_acquisition(minimum_soc):
    runtime = session(fixture(capacity_C=.1, minimum_soc=minimum_soc))
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    result = execute(runtime, {"operation": "measure", "instrument": METER})
    assert result.info["operation_outcome"] == "stopped"
    expected = (.7-minimum_soc)*.1/.5
    assert runtime._state.ledger.time_s == pytest.approx(expected, abs=1e-8)
    assert result.info["actual_duration_s"] == pytest.approx(expected, abs=1e-8)
    assert runtime._state.lab_clock.stopped
    assert unpack_thevenin(runtime._state)[3].coordinates[SOC] == pytest.approx(
        minimum_soc, abs=1e-9)


def read_offline_fixture(state, action, rng):
    from chemworld.world.electrode_instruments import electrical_observation, read_thevenin

    reading = read_thevenin(state, action, rng)
    return electrical_observation("test_offline_circuit",
                                  reading.channels[0].value, reading.channels[1].value)


def queued_measurement_session(*, card=None):
    from chemworld.foundation.lab_clock import LabClockSpec
    from chemworld.foundation.state_ledgers import EquipmentRecord
    from chemworld.world.domain_modules import DomainModuleRegistry, DomainModuleSpec
    from chemworld.world.electrode_instruments import electrical_instrument_spec

    offline = electrical_instrument_spec("test_offline_circuit", read_offline_fixture)
    offline = replace(offline, contract=replace(offline.contract,
        input_state_schema={"required_phase": "formal", "sample_consumption_L": 0.},
        clock_semantics="offline_snapshot", latency_s=0., analysis_time_s=.3))
    modules = DomainModuleRegistry((*DomainModuleRegistry.default().modules,
        DomainModuleSpec("test_offline_fixture", instruments=(offline,))))
    payload = thevenin_world_spec(fixture()).to_dict()
    state = payload["physics"]["initial_states"]["circuit_workspace"]
    analyzer = EquipmentRecord("offline_analyzer", "instrument", "circuit_workspace",
                               settings={"instrument_id": "test_offline_circuit"})
    state["equipment"]["offline_analyzer"] = analyzer.to_dict()
    payload["physics"]["lab_clock"] = LabClockSpec(
        analyzers={"offline_analyzer": "test_offline_circuit"}).to_dict()
    for component in payload["components"]:
        if component["kind"] == "observation":
            component["parameters"]["instruments"].append("test_offline_circuit")
    payload["task"]["instruments"].append("test_offline_circuit")
    payload["task"]["operations"].append("queue_lab_measure")
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload, modules=modules),
                               domain_modules=modules, campaign_resource_card=card)
    runtime.reset(seed=7)
    return runtime


def test_timed_acquisition_crosses_an_actual_queued_analysis_boundary_once():
    runtime = queued_measurement_session()
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    execute(runtime, {"operation": "queue_lab_measure", "device": "offline_analyzer"})
    result = execute(runtime, {"operation": "measure", "instrument": METER})
    assert runtime._state.ledger.time_s == 1.
    assert runtime._state.ledger.analysis_time_s == pytest.approx(.3)
    assert runtime._state.ledger.cost == .02
    completed = runtime._state.lab_clock.completed
    assert len(completed) == 1 and completed[0]["available_at"] == .3
    assert result.info["actual_duration_s"] == 1.
    assert unpack_thevenin(runtime._state)[3].coordinates[SOC] == pytest.approx(.695, abs=1e-9)
    assert len(result.info["outcome_diagnostics"]["shared_acquisition_slices"]) == 2


@pytest.mark.parametrize("limit,accepted", ((4, False), (5, True)))
def test_fifo_reservation_and_actual_budget_include_direct_and_queued_starts(limit, accepted):
    from chemworld.campaign_resources import CampaignResourceCard
    from chemworld.runtime.scheduling import lab_instrument_reservation

    card = CampaignResourceCard(card_id="timed-circuit-resource-test", operation_attempt_limit=80,
        vessel_start_limit=1, final_assay_limit=1, nonfinal_instrument_use_limit=limit,
        stock_limits={})
    runtime = queued_measurement_session(card=card)
    for _ in range(5):
        execute(runtime, {"operation": "queue_lab_measure", "device": "offline_analyzer"})
    action = {"operation": "measure", "instrument": METER}
    assert lab_instrument_reservation(runtime._state, action, modules=runtime.domain_modules) == {
        METER: 1, "test_offline_circuit": 4}  # Fifth cannot start before t=1.2.
    result = runtime.execute(action)
    ledger = runtime._campaign_resource_ledger
    if accepted:
        assert result.info["transaction_status"] == "committed"
        assert ledger.instrument_uses == {METER: 1, "test_offline_circuit": 4}
        assert ledger.nonfinal_instrument_uses == 5
        assert len(runtime._state.lab_clock.completed) == 3
        assert len(runtime._state.lab_clock.running) == 1
        assert len(runtime._state.lab_clock.queued) == 1
        assert runtime._state.ledger.cost == pytest.approx(.05)
        assert runtime._state.ledger.analysis_time_s == pytest.approx(1.)
    else:
        assert result.info["transaction_status"] != "committed"
        assert ledger.instrument_uses == {} and ledger.nonfinal_instrument_uses == 0
        assert runtime._state.ledger.time_s == 0.
        assert len(runtime._state.lab_clock.queued) == 5


def test_later_measurement_slice_failure_preserves_queued_job_and_detector_rng(monkeypatch):
    from chemworld.runtime.kernel_contracts import ModelProviderResult

    runtime = queued_measurement_session()
    execute(runtime, {"operation": "queue_lab_measure", "device": "offline_analyzer"})
    initial = runtime._state.to_dict()
    service = runtime.runtime.domain_services.instrument_cost
    original, calls = service.shared_clock_advance, []

    def fail_second(candidate, action):
        calls.append(candidate.ledger.time_s)
        if len(calls) == 2:
            return ModelProviderResult({}, {"fixture": "second-shared-slice"}, success=False,
                outcome="numerical_failure", failure_reason="injected_later_failure")
        return original(candidate, action)

    monkeypatch.setattr(service, "shared_clock_advance", fail_second)
    result = runtime.execute({"operation": "measure", "instrument": METER})
    assert calls == [0., .3]
    assert result.info["transaction_status"] == "rolled_back"
    after = runtime._state.to_dict()
    for ledger in ("ledger", "process"):
        for field in ("cost", "risk"):
            after[ledger][field] = initial[ledger][field]
    assert after == initial  # Includes original queue and lab detector RNG.


def test_diffusion_profile_has_one_owner_and_material_reset_preserves_clock_and_resources():
    from chemworld.campaign_resources import CampaignResourceCard
    from chemworld.examples.native_thevenin import diffusion_spec

    spec = diffusion_spec(cells=8)
    card = CampaignResourceCard(card_id="diffusion-single-clock", operation_attempt_limit=20,
        vessel_start_limit=3, final_assay_limit=1, nonfinal_instrument_use_limit=2,
        stock_limits={})
    runtime = session(spec, campaign_resource_card=card)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    reading = execute(runtime, {"operation": "measure", "instrument": METER})
    before = runtime._state
    record, slot, model, physical, program = unpack_thevenin(before)
    assert len(before.state_slots.slots) == 1
    assert set(slot.values) == {c.name for c in spec.coordinate_declarations}
    assert len(spec.diffusion.coordinate_keys) == 7
    assert any(slot.values[key] != 0 for key in spec.diffusion.coordinate_keys)
    fields = thevenin_fields(model, physical, program)
    assert fields["surface_soc"] < fields["soc"]
    assert sum(w*z for w, z in zip(model.diffusion.weights, fields["distributed_soc"],
                                   strict=True)) == pytest.approx(fields["soc"], abs=1e-15)
    assert reading.info["actual_duration_s"] == 1.
    assert before.ledger.time_s == physical.time_s == 3.
    assert physical.coordinates[SOC] == pytest.approx(.694, abs=1e-10)
    assert before.ledger.flow_electrical_work_J == pytest.approx(-physical.coordinates[WORK])
    assert before.ledger.sample_consumed_L == 0.
    assert runtime._campaign_resource_ledger.instrument_uses == {METER: 1}
    assert runtime._campaign_resource_ledger.nonfinal_instrument_uses == 1
    assert {c.channel_id for c in reading.observation.channels} == {
        "terminal_potential", "electrical_current"}  # No hidden SOC/profile truth channel.
    runtime.reset(options={"scope": "material", "vessel_ids": [before.vessel_id]})
    retained_record, retained_slot, _, _, _ = unpack_thevenin(runtime._state)
    assert retained_record == record and retained_slot == slot
    assert runtime._state.ledger.time_s == 3.
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True, allow_nan=False))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    assert unpack_thevenin(restored._state)[1] == slot
    for instance in (runtime, restored):
        execute(instance, {"operation": "configure_electrical_control", "current_A": 0.})
        execute(instance, {"operation": "wait", "duration_s": 1.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    assert runtime._state.ledger.time_s == 4.
    assert unpack_thevenin(runtime._state)[1].values != slot.values


def test_diffusion_surface_stop_commits_only_partial_measurement_time_and_retains_profile():
    from chemworld.examples.native_thevenin import diffusion_spec

    spec = replace(fixture(0, initial_soc=.005), diffusion=diffusion_spec(cells=8).diffusion)
    program = ScalarControlProgram(((0., .5),), "A")
    expected = thevenin_system(spec, program).advance(spec.initial_state(), 1.)
    assert 0 < expected.actual_duration_s < 1.
    runtime = session(spec)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    result = execute(runtime, {"operation": "measure", "instrument": METER})
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["measurement_status"] in {"failed", "invalid", "not_observed"}
    assert runtime._state.lab_clock.stopped
    actual = unpack_thevenin(runtime._state)[3]
    assert actual.time_s == pytest.approx(expected.actual_duration_s, abs=1e-10)
    assert runtime._state.ledger.time_s == actual.time_s
    assert result.info["actual_duration_s"] == actual.time_s
    assert actual.coordinates[SOC] > 0.
    assert spec.diffusion.surface_soc(actual.coordinates[SOC], actual.coordinates) == pytest.approx(
        0., abs=2e-14)
    assert all(z > 0 for z in spec.diffusion.profile(actual.coordinates[SOC], actual.coordinates))
    # Runtime state/model serialization can change the finite-difference/Newton
    # roundoff path; compare the physical coordinates within solver accuracy.
    assert dict(actual.coordinates) == pytest.approx(
        dict(expected.final_state.coordinates), rel=1e-8, abs=1e-10)
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(-actual.coordinates[WORK])
    assert runtime._state.ledger.sample_consumed_L == 0.


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_diffusion_agent_replay_and_sorted_snapshot_fresh_process_continuation(tmp_path, noise):
    from chemworld.examples.native_thevenin import diffusion_spec, run, workflow
    from test_native_domains_end_to_end import (
        test_snapshot_resume_in_middle_of_real_workflow as resume,
    )

    spec = diffusion_spec(cells=8)
    report = run(tmp_path / (noise+"-diffusion.jsonl"), noise=noise, spec=spec,
                 include_impedance_analysis=True)
    assert report["steps"] == report["replay_steps"] == 10
    # The checkpoint is after CC acquisition and entering rest: this restores a
    # genuine nonuniform profile, not merely a fresh uniform specification.
    resume("Thevenin-diffusion", noise, tmp_path,
           recipe=workflow(spec, include_impedance_analysis=True), sort_keys=True)


@pytest.mark.parametrize("defect", ("missing", "internal", "surface"))
def test_diffusion_snapshot_rejects_missing_or_physically_invalid_spatial_state(defect):
    from chemworld.examples.native_thevenin import diffusion_spec

    spec = diffusion_spec(cells=8)
    runtime = session(spec)
    payload = runtime.snapshot().to_dict()
    values = next(iter(payload["world_state"]["state_slots"].values()))["values"]
    if defect == "missing":
        del values[spec.diffusion.coordinate_keys[0]]
    elif defect == "internal":
        values[spec.diffusion.coordinate_keys[0]] = .5
        values[spec.diffusion.coordinate_keys[1]] = -.1
    else:
        values[spec.diffusion.coordinate_keys[0]] = -.3
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(json.loads(json.dumps(payload, allow_nan=False)))
