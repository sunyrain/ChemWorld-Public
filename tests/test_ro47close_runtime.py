"""Fixed Block R: 27 actual RO runtime/restart/boundary/rejection cases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from tests.reference.ro47close_fresh_resume import continue_world, normalized

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.distributed_ro import unpack_distributed_ro
from chemworld.runtime.seawater_ro import unpack_seawater_ro
from chemworld.world.distributed_ro_world import distributed_ro_world_spec
from chemworld.world.seawater_ro_world import seawater_ro_world_spec
from reference.ro47close_fixtures import (
    mcas_feed,
    mcas_spec,
    multisolute_feed,
    multisolute_spec,
    seawater_spec,
)

MODES = ("common-g", "mcas-ideal-volume", "seawater")


def short_outcome(info):
    return {
        key: info.get(key)
        for key in (
            "transaction_status",
            "operation_outcome",
            "rollback_reason",
            "invalid_reasons",
            "outcome_diagnostics",
        )
    }


def require_committed(info):
    assert info["transaction_status"] == "committed", short_outcome(info)


def record(name, payload):
    """Optional exact evidence outside Git; each full pytest run has a fresh root."""
    target = os.environ.get("CHEMWORLD_RO47_RUNTIME_OUTPUT")
    if target:
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (name + ".json")).open("x", encoding="utf-8") as handle:
            json.dump(normalized(payload), handle, indent=2, allow_nan=False)


def build(mode, *, shared=False, stop=None):
    if mode == "seawater":
        spec = replace(seawater_spec(), membrane_area_m2=0.001, membrane_length_m=0.05)
        feed = {"H2O": 0.965, "TDS": 0.035}
        builder = seawater_ro_world_spec
    else:
        spec = multisolute_spec() if mode == "common-g" else mcas_spec()
        if mode == "mcas-ideal-volume":
            spec = replace(
                spec, membrane_area_m2=0.001, channel=replace(spec.channel, length_m=0.05)
            )
        feed = multisolute_feed(spec) if mode == "common-g" else mcas_feed(spec)
        builder = distributed_ro_world_spec
    if stop == "feed":
        initial_volume = (
            spec.properties.volume(feed, spec.temperature_K, spec.reservoir_pressure_Pa) * 1000
        )
        # Declared pickup retains all but exactly 1.5 s at requested 10 mL/min.
        spec = replace(spec, pickup_volume_L=initial_volume - 10 / 60000 * 1.5)
    elif stop:
        spec = replace(spec, **{stop + "_capacity_L": 1e-6})
    runtime = ChemWorldRuntime(composition=builder(spec, feed, shared_clock=shared))
    runtime.reset(seed=47)
    return spec, runtime


def inventory(runtime, mode):
    _, spec, state = (unpack_seawater_ro if mode == "seawater" else unpack_distributed_ro)(
        runtime._state
    )
    rows = state.masses_kg if mode == "seawater" else state.amounts_mol
    return spec, state, {key: sum(row[key] for row in rows) for key in rows[0]}


def check_conservation_and_owned_clock(runtime, mode, totals):
    spec, state, final_totals = inventory(runtime, mode)
    assert final_totals == pytest.approx(totals, rel=1e-9, abs=1e-10)
    tanks = (runtime._state, *runtime._state.inactive_vessels.values())
    assert len(tanks) == 3
    for tank in tanks:
        assert tank.ledger.time_s == pytest.approx(state.time_s, abs=1e-8)
        assert all(n >= 0 for n in tank.species_amounts.values())
        if mode != "seawater":
            assert sum(
                spec.properties.charges[k] * tank.species_amounts[k]
                for k in spec.properties.solute_ids
            ) == pytest.approx(0, abs=1e-10)
        else:
            assert set(tank.species_amounts) == {"H2O"}
            assert any(
                slot.spec.state_type == "seawater_tds_mass_v1" for slot in tank.state_slots.slots
            )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shared", (False, True))
def test_original_local_shared_full_json_and_fresh_process(mode, shared):
    spec, runtime = build(mode, shared=shared)
    restored = None
    payload = {"mode": mode, "shared": shared, "declaration": spec.to_dict()}
    try:
        _, _, totals = inventory(runtime, mode)
        # Local wait retains the existing 1 s API minimum. Shared waits preserve
        # the original failed .5/.75 s case rather than moving its physical inputs.
        prefix_wait_s, short_wait_s = (0.5, 0.75) if shared else (1, 1)
        payload["prefix"] = []
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "wait", "duration_s": prefix_wait_s},
        ):
            outcome = runtime.execute(action)
            payload["prefix"].append({"action": action, "info": outcome.info})
            require_committed(outcome.info)
        before = normalized(runtime.snapshot().to_dict())
        payload["before"] = before
        restored = ChemWorldRuntime.from_snapshot(before)
        original = payload["original"] = continue_world(runtime, short_wait_s)
        replay = payload["replay"] = continue_world(restored, short_wait_s)
        child = subprocess.run(
            [sys.executable, "-m", "tests.reference.ro47close_fresh_resume"],
            input=json.dumps({"snapshot": before, "short_wait_s": short_wait_s}),
            capture_output=True,
            text=True,
            timeout=45,
        )
        payload.update(
            child_returncode=child.returncode, child_stdout=child.stdout, child_stderr=child.stderr
        )
        assert child.returncode == 0, child.stderr
        fresh = json.loads(child.stdout)
        for continuation in (original, replay, fresh):
            for row in continuation["actions"]:
                require_committed(row["info"])
            assert len(continuation["actions"]) == 3
        assert bool(original == replay == fresh), "full continuation snapshots/outcomes differ"
        check_conservation_and_owned_clock(runtime, mode, totals)
        assert runtime._state.ledger.time_s == pytest.approx(prefix_wait_s + short_wait_s + 1.25)
        assert runtime._state.ledger.flow_pump_work_J > 0
        assert runtime._state.ledger.cost > 0
    finally:
        record(f"normal-{mode}-{shared}", payload)
        runtime.close()
        if restored:
            restored.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stop", ("feed", "permeate", "retentate"))
def test_three_finite_shared_clock_stops(mode, stop):
    spec, runtime = build(mode, shared=True, stop=stop)
    payload = {"mode": mode, "stop": stop, "declaration": spec.to_dict()}
    try:
        _, _, totals = inventory(runtime, mode)
        require_committed(
            runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 10}).info
        )
        before = normalized(runtime.snapshot().to_dict())
        outcome = runtime.execute({"operation": "wait", "duration_s": 2})
        payload.update(before=before, info=outcome.info, after=runtime.snapshot().to_dict())
        require_committed(outcome.info)
        assert outcome.info["operation_outcome"] == "stopped", short_outcome(outcome.info)
        expected_event = (
            "feed_pickup_exhausted"
            if stop == "feed"
            else stop + ("_receiver_full" if mode == "seawater" else "_full")
        )
        assert expected_event in str(outcome.info["outcome_diagnostics"])
        assert 0 < runtime._state.ledger.time_s < 2
        check_conservation_and_owned_clock(runtime, mode, totals)
        if stop == "feed":
            assert runtime._state.volume_L == pytest.approx(spec.pickup_volume_L, abs=1e-9)
            assert runtime._state.ledger.time_s == pytest.approx(1.5, abs=2e-7)
        else:
            assert runtime._state.inactive_vessels[stop].volume_L == pytest.approx(1e-6, abs=1e-10)
    finally:
        record(f"stop-{mode}-{stop}", payload)
        runtime.close()


def forged_snapshot_rejected(mode, mutation, tag):
    _, runtime = build(mode, shared=True)
    restored = None
    payload = {"mode": mode, "mutation": tag}
    try:
        original = normalized(runtime.snapshot().to_dict())
        altered = deepcopy(original)
        mutation(altered["world_state"])
        payload["forged_snapshot"] = altered
        try:
            restored = ChemWorldRuntime.from_snapshot(altered)
        except (ValueError, KeyError, TypeError) as error:
            payload["rejection"] = type(error).__name__ + ": " + str(error)
        else:
            before = normalized(restored.snapshot().to_dict())
            outcome = restored.execute({"operation": "wait", "duration_s": 1})
            payload["first_action"] = outcome.info
            after = normalized(restored.snapshot().to_dict())
            payload.update(before=before, after=after)
            assert outcome.info["transaction_status"] != "committed", short_outcome(outcome.info)
            assert outcome.info["actual_duration_s"] == 0
            assert outcome.info["sample_delta"] == 0
            # Rejected operations charge the original procedural penalty. Every
            # physical field, all three inventories, H, clocks and RNG stay exact.
            before_state, after_state = (
                deepcopy(before["world_state"]),
                deepcopy(after["world_state"]),
            )
            for name in ("cost", "risk"):
                delta = after_state["ledger"][name] - before_state["ledger"][name]
                assert delta == pytest.approx(outcome.info[name + "_delta"])
                after_state["ledger"][name] = before_state["ledger"][name]
                if before_state["process"] is not None:
                    after_state["process"][name] = before_state["process"][name]
            assert bool(after_state == before_state), "rejection changed physical world state"
            for key in ("configuration", "execution", "rng_state"):
                assert bool(after[key] == before[key]), "rejection changed " + key
        assert bool(normalized(runtime.snapshot().to_dict()) == original), "source world mutated"
    finally:
        record(f"forged-{mode}-{tag}", payload)
        runtime.close()
        if restored:
            restored.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ("species", "enthalpy", "schema"))
def test_forged_original_species_enthalpy_and_declaration_are_rejected(mode, kind):
    def mutate(state):
        if kind == "species":
            state["species_amounts"]["forged_solute"] = 1.0
        elif kind == "enthalpy":
            slot = next(s for s in state["state_slots"].values() if "enthalpy_J" in s["values"])
            slot["values"]["enthalpy_J"] += 100
        else:
            state["equipment"]["membrane"]["settings"]["physical_model"]["schema"] = (
                "forged-ro-v999"
            )

    forged_snapshot_rejected(mode, mutate, kind)


@pytest.mark.parametrize("kind", ("mass", "identity", "negative"))
def test_seawater_tds_conserved_mass_and_owner_cannot_be_forged(kind):
    def mutate(state):
        slot = next(s for s in state["state_slots"].values() if "tds_mass_kg" in s["values"])
        if kind == "identity":
            slot["spec"]["state_type"] = "forged_tds"
        elif kind == "negative":
            slot["values"]["tds_mass_kg"] = -0.01
        else:
            slot["values"]["tds_mass_kg"] += 0.01

    forged_snapshot_rejected("seawater", mutate, "tds-" + kind)
