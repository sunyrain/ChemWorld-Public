"""Real identities, retained inventory and device-specific FIFO occupancy."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.examples.multi_device_lab import multi_device_lab_world
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states


def session(noise="sequential", *, resource_card=None, **options):
    runtime = ChemWorldRuntime(
        composition=multi_device_lab_world(**options), observation_noise_mode=noise,
        observation_seed_override=12,
        campaign_resource_card=resource_card,
    )
    runtime.reset(seed=7)
    return runtime


def budget(*, total=4, per_instrument=4):
    return CampaignResourceCard(
        card_id="identified-queue-budget", operation_attempt_limit=80,
        vessel_start_limit=1, final_assay_limit=1,
        nonfinal_instrument_use_limit=total, stock_limits={},
        per_instrument_limits={"programmed_hplc": per_instrument},
    )


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        key: result.info.get(key) for key in (
            "transaction_status", "rollback_reason", "outcome_diagnostics", "preconditions",
        )
    }
    return result


def enqueue_four(runtime):
    for index, vessel in enumerate(("reactor", "sample", "vial-c", "vial-d")):
        suffix = "a" if index % 2 == 0 else "b"
        execute(runtime, "select_vessel", vessel=vessel)
        execute(runtime, "queue_lab_heat", device="heater-" + suffix,
                target_temperature_K=320, duration_s=2)
        execute(runtime, "queue_lab_measure", device="hplc-" + suffix)


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_two_actual_devices_per_kind_serve_four_systems_in_fifo_waves(noise):
    runtime = session(noise, resource_card=budget())
    try:
        enqueue_four(runtime)
        assert runtime._campaign_resource_ledger.nonfinal_instrument_uses == 0
        execute(runtime, "wait", duration_s=100)
        assert runtime._campaign_resource_ledger.instrument_uses == {"programmed_hplc": 2}
        assert runtime._state.ledger.time_s == 2
        assert len(runtime._state.lab_clock.running) == 2
        assert runtime._state.ledger.analysis_time_s == 4
        assert [job["sequence"] for job in runtime._state.lab_clock.completed] == [0, 2]
        for _ in range(3):
            execute(runtime, "wait", duration_s=100)
        assert runtime._state.ledger.time_s == 120
        assert runtime._state.ledger.analysis_time_s == 240
        assert runtime._state.ledger.cost == pytest.approx(.2)
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0008)
        assert runtime._campaign_resource_ledger.instrument_uses == {"programmed_hplc": 4}
        assert [job["sequence"] for job in runtime._state.lab_clock.completed] == [
            0, 2, 4, 6, 1, 3, 5, 7,
        ]
        records = [record for local in local_states(runtime._state).values()
                   for record in local.equipment.equipment.values()]
        assert {record.equipment_id for record in records} == {
            "heater-a", "heater-b", "hplc-a", "hplc-b",
        }
        for record in records:
            if record.equipment_id.startswith("hplc"):
                assert record.settings["use_count"] == 2
                assert record.settings["elapsed_time_s"] == 120
                assert record.status == "used"
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(
            runtime._state.vessel_elapsed_s, 120,
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_two_calibrations_and_retention_slots_remain_with_their_real_device(noise):
    runtime = session(noise, noise_sd=0)
    try:
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "queue_lab_measure", device="hplc-b")
        execute(runtime, "wait", duration_s=60)
        jobs = runtime._state.lab_clock.completed
        assert [job["device"] for job in jobs] == ["hplc-a", "hplc-b"]
        assert all(job["observation"]["sampled_at"] == {"clock_id": "lab", "time_s": 0}
                   for job in jobs)
        for job in jobs:
            acquisition = job["observation"]["raw_template"]["acquisition"]
            assert acquisition["device_id"] == job["device"]
            concentrations = [channel["value"] for channel in job["observation"]["channels"]
                              if channel["channel_id"].endswith("_concentration")]
            assert concentrations == pytest.approx([.005] * 3, abs=1e-12)
        areas = [{channel["channel_id"]: channel["value"]
                  for channel in job["observation"]["channels"]
                  if channel["channel_id"].endswith("_area")} for job in jobs]
        assert areas[1] == pytest.approx(
            {key: 2 * value for key, value in areas[0].items()}, abs=1e-12,
        )
        local = local_states(runtime._state)["reactor"]
        for identity in ("hplc-a", "hplc-b"):
            memory = [slot for slot in local.state_slots.slots if slot.spec.owner_id == identity]
            assert len(memory) == 1
            assert sum(memory[0].values.values()) > 0
        assert local.equipment.equipment["hplc-a"].settings["calibration"] != \
            local.equipment.equipment["hplc-b"].settings["calibration"]
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_heaters_serialize_one_port_and_park_only_an_idle_actual_device(noise):
    runtime = session(noise)
    try:
        execute(runtime, "select_vessel", vessel="sample")
        execute(runtime, "queue_lab_heat", device="heater-a",
                target_temperature_K=320, duration_s=2)
        execute(runtime, "wait", duration_s=.25)
        local = local_states(runtime._state)["sample"]
        assert local.equipment.equipment["heater-a"].status == "busy"
        assert local.equipment.equipment["heater-b"].status == "parked"
        execute(runtime, "queue_lab_heat", device="heater-b",
                target_temperature_K=320, duration_s=2)
        execute(runtime, "wait", duration_s=100)
        assert runtime._state.ledger.time_s == 2
        assert [job["device"] for job in runtime._state.lab_clock.queued] == ["heater-b"]
        first_energy = runtime._state.ledger.energy_jacket_J
        execute(runtime, "wait", duration_s=100)
        assert runtime._state.ledger.time_s == 4
        local = local_states(runtime._state)["sample"]
        assert local.equipment.equipment["heater-a"].status == "parked"
        assert local.equipment.equipment["heater-b"].attached_vessel_id == "sample"
        assert runtime._state.ledger.energy_jacket_J - first_energy > first_energy
        for identity, factor in (("heater-a", .9), ("heater-b", 1.1)):
            assert local.equipment.equipment[identity].settings["thermal_device"][
                "calibration_factor"] == factor
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ("queue_lab_heat", "queue_lab_measure"))
@pytest.mark.parametrize("value", (None, "missing", True, ["heater-a", "heater-b"]))
def test_ambiguous_or_unknown_device_never_starts_sampling_or_a_clock(operation, value):
    runtime = session()
    try:
        before = runtime._state
        action = {"operation": operation}
        if value is not None:
            action["device"] = value
        if operation == "queue_lab_heat":
            action.update(target_temperature_K=320, duration_s=2)
        result = runtime.execute(action)
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.lab_clock == before.lab_clock
        assert runtime._state.ledger.time_s == 0
        assert runtime._state.ledger.analysis_time_s == 0
        assert runtime._state.ledger.sample_consumed_L == 0
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.inactive_vessels == before.inactive_vessels
    finally:
        runtime.close()


def test_single_device_input_converts_to_the_same_canonical_collection():
    legacy = LabClockSpec("heater", "programmed_hplc")
    explicit = LabClockSpec(
        heater_ids=("heater",), analyzers={"instrument:programmed_hplc": "programmed_hplc"},
    )
    assert legacy.to_dict() == explicit.to_dict()
    assert LabClockSpec(**json.loads(json.dumps(legacy.to_dict()))) == explicit
    assert legacy.heater_id == "heater" and legacy.analyzer == "programmed_hplc"


@pytest.mark.parametrize("profile", (
    {"heater_ids": ("a", "a")},
    {"heater_ids": ("a",), "analyzers": {"a": "programmed_hplc"}},
    {"analyzers": ({"device_id": "a", "instrument": "programmed_hplc"},) * 2},
))
def test_two_names_cannot_alias_one_declared_device(profile):
    with pytest.raises(ValueError, match=r"identit|alias"):
        LabClockSpec(**profile)


def test_public_device_choices_follow_real_ownership_and_current_task_permission():
    from chemworld.agent_interface import action_schema, available_actions

    runtime = session()
    try:
        before = runtime.snapshot().to_dict()

        def device_choices(operation):
            row = action_schema(runtime, operation)
            return next(field["choices"] for field in row["fields"] if field["field"] == "device")

        assert device_choices("queue_lab_heat") == ["heater-a", "heater-b"]
        assert device_choices("queue_lab_measure") == ["hplc-a", "hplc-b"]
        with pytest.raises(ValueError, match="event JSON"):
            runtime.action_codec.encode_vector({
                "operation": "queue_lab_measure", "device": "hplc-a",
            })
        assert runtime.snapshot().to_dict() == before
        runtime.operation_validator.allowed_instruments = set()
        assert device_choices("queue_lab_measure") == []
        assert "queue_lab_measure" not in {row["operation"] for row in available_actions(runtime)}
        runtime.operation_validator.allowed_instruments = {"programmed_hplc"}
        assert device_choices("queue_lab_measure") == ["hplc-a", "hplc-b"]
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_multi_device_pending_snapshot_and_polling_resume_exactly_in_new_process(noise, tmp_path):
    runtime = session(noise, resource_card=budget())
    try:
        enqueue_four(runtime)
        execute(runtime, "wait", duration_s=.25)
        assert len(runtime._state.lab_clock.running) == len(runtime._state.lab_clock.queued) == 4
        before = runtime.snapshot().to_dict()
        assert runtime._campaign_resource_ledger.instrument_uses == {"programmed_hplc": 2}
        for _ in range(10):
            runtime.observe()
            assert all("observation" not in row for row in runtime.lab_status()["running"])
        assert runtime.snapshot().to_dict() == before
        source, output = tmp_path / "session.json", tmp_path / "continued.json"
        runtime.snapshot().save(source)
        for _ in range(4):
            execute(runtime, "wait", duration_s=100)
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            "results=[r.execute({'operation':'wait','duration_s':100}) for _ in range(4)]; "
            "assert all(x.info['transaction_status']=='committed' for x in results); "
            f"Path({str(output)!r}).write_text(json.dumps({{"
            "'state':r._state.to_dict(),'resources':r._campaign_resource_ledger.snapshot()}))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        assert json.loads(output.read_text()) == json.loads(json.dumps({
            "state": runtime._state.to_dict(),
            "resources": runtime._campaign_resource_ledger.snapshot(),
        }))
        assert runtime._campaign_resource_ledger.instrument_uses == {"programmed_hplc": 4}
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_failed_detector_still_consumes_its_actual_sample_and_occupancy(noise, monkeypatch):
    from chemworld.foundation import instrument_equipment_id

    runtime = session(noise, resource_card=budget(total=2, per_instrument=2))
    try:
        modules = runtime.runtime.domain_services.lab_schedule.modules
        definitions = modules.instrument_specs()
        original = definitions["programmed_hplc"].native_observer

        def observer(state, action, rng):
            if instrument_equipment_id(action["instrument"], state=state) == "hplc-b":
                raise ValueError("injected second-device detector failure")
            return original(state, action, rng)

        changed = {**definitions, "programmed_hplc": replace(
            definitions["programmed_hplc"], native_observer=observer,
        )}
        monkeypatch.setattr(type(modules), "instrument_specs", lambda self, **kwargs: changed)
        execute(runtime, "queue_lab_measure", device="hplc-a")
        execute(runtime, "queue_lab_measure", device="hplc-b")
        execute(runtime, "wait", duration_s=1)
        assert runtime.lab_status()["completed"] == []
        assert all("readout_error" not in row for row in runtime.lab_status()["running"])
        execute(runtime, "wait", duration_s=100)
        jobs = runtime._state.lab_clock.completed
        assert jobs[0]["measurement_status"] == "valid"
        assert jobs[1]["measurement_status"] == "invalid"
        assert "observation" not in jobs[1]
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0004)
        assert runtime._state.ledger.analysis_time_s == 120
        assert runtime._state.ledger.cost == pytest.approx(.1)
        records = local_states(runtime._state)["reactor"].equipment.equipment
        assert records["hplc-a"].settings["use_count"] == 1
        assert records["hplc-b"].settings["use_count"] == 1
        assert runtime._campaign_resource_ledger.instrument_uses == {"programmed_hplc": 2}
        assert runtime._campaign_resource_ledger.nonfinal_instrument_uses == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("constraint", ("total", "per_instrument"))
def test_simultaneous_starts_reserve_both_type_uses_before_any_physics(noise, constraint):
    from chemworld.agent_interface import validate_action

    runtime = session(noise, resource_card=budget(**{constraint: 1}))
    try:
        for device in ("hplc-a", "hplc-b"):
            execute(runtime, "queue_lab_measure", device=device)
        ledger = runtime._campaign_resource_ledger
        assert ledger.nonfinal_instrument_uses == 0
        assert runtime._state.ledger.sample_consumed_L == runtime._state.ledger.cost == 0
        before = runtime._state.to_dict()
        snapshot = runtime.snapshot().to_dict()
        action = {"operation": "wait", "duration_s": 60}
        validation = validate_action(runtime, action)
        assert not validation["valid"]
        assert runtime.snapshot().to_dict() == snapshot
        result = runtime.execute(action)
        assert result.info["transaction_status"] != "committed"
        reason = ("nonfinal_instrument_use_limit" if constraint == "total"
                  else "per_instrument_limit:programmed_hplc")
        assert reason in result.info["campaign_resource_rejection_reasons"]
        after = runtime._state.to_dict()
        # Existing rejected-action penalty is .01 cost/.08 risk; no acquisition debit.
        for section in ("process", "ledger"):
            before[section]["cost"] = .01
            before[section]["risk"] = .08
        assert after == before
        assert ledger.nonfinal_instrument_uses == 0
        assert ledger.operation_attempts == 3
        assert ledger.verify_integrity()
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_failed_shared_slice_does_not_charge_reserved_acquisitions(noise, monkeypatch):
    from chemworld.runtime.kernel_contracts import ModelProviderResult

    runtime = session(noise, resource_card=budget())
    try:
        enqueue_four(runtime)
        before = runtime._state
        solver = runtime.runtime.domain_services.reaction_thermal.advancer
        original = solver.batch

        def failure(state, action, **kwargs):
            if state.vessel_id == "vial-d":
                return ModelProviderResult(
                    {}, state=None, diagnostics={"injected": True},
                    success=False, outcome="numerical_failure",
                    failure_reason="injected_after_two_device_acquisitions",
                )
            return original(state, action, **kwargs)

        monkeypatch.setattr(solver, "batch", failure)
        result = runtime.execute({"operation": "wait", "duration_s": 60})
        assert result.info["operation_outcome"] == "numerical_failure"
        after = runtime._state
        assert after.lab_clock == before.lab_clock
        assert after.inactive_vessels == before.inactive_vessels
        assert after.equipment == before.equipment
        assert after.state_slots == before.state_slots
        assert after.ledger.time_s == after.ledger.analysis_time_s == 0
        assert after.ledger.sample_consumed_L == 0
        ledger = runtime._campaign_resource_ledger
        assert ledger.nonfinal_instrument_uses == 0
        assert ledger.instrument_uses == {}
        assert ledger.verify_integrity()
    finally:
        runtime.close()
