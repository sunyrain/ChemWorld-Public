"""Source-critical reactions, true calendar caps and complete numerical continuation."""

import copy
import importlib
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict
from xml.etree import ElementTree as ET

import pytest

from chemworld.physchem.stochastic_kinetics import (
    EventChannel,
    EventProcessSpec,
    _signature,
    advance_event_channels,
    initial_continuation,
)
from chemworld.physchem.stochastic_tau_leap import PENDING_SCHEMA, TauLeapPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sbml_world import sbml_world
from test_sbml import model
from test_sbml51k_events import clock, counts, diagnostics, event, owned, parsed, plain, wait
from test_sbml_stochastic51 import CORE, TAU, source


def policy(kind):
    return TauLeapPolicy(epsilon=0.1 if kind == "coarse" else 0.01,
                         max_tau_s=0.1 if kind == "coarse" else 0.005,
                         ssa_threshold=0.01)


@contextmanager
def session(original, kind, seed=37, controlled_parameters=()):
    world = sbml_world(original, numerical_form="stochastic", solver_method=TAU,
                      tau_policy=policy(kind), maximum_events=100000,
                      controlled_parameters=controlled_parameters, budget=100,
                      readout_relative_sd=0, seed=seed)
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=seed)
        yield runtime
    finally:
        runtime.close()


def stopped_at(deadline=0.037):
    return parsed(original=source(counts={"A": 1000, "B": 0, "C": 3}, coefficient=1.0),
                  events=event(guard=clock(deadline), values=(("k", "<cn>0</cn>"),)))


def controlled_deadline():
    original = source(counts={"A": 1000, "B": 0, "C": 3}, coefficient=1.0)
    tree = ET.fromstring(original.to_xml())
    parameters = tree.find(".//{" + CORE + "}listOfParameters")
    ET.SubElement(parameters, "{" + CORE + "}parameter", id="due", value="0.2",
                  constant="true", units="second")
    original = model(ET.tostring(tree, encoding="unicode"),
                     items_per_mole=original.items_per_mole)
    time = ('<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">'
            'time</csymbol>')
    return parsed(original=original, events=event(
        guard="<apply><geq/>" + time + "<ci>due</ci></apply>",
        values=(("k", "<cn>0</cn>"),)))


def settings(node):
    if isinstance(node, dict):
        if ("coordinates" in node and isinstance(node.get("continuation"), dict)
                and node["continuation"].get("format") == "chemworld_sbml_integer_events_v1"):
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = settings(child)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("kind", ("coarse", "fine"))
@pytest.mark.parametrize("seed", (17, 37))
def test_state_guard_protects_first_actual_reaction_with_permissive_tau(kind, seed):
    original = parsed(original=source(coefficient=1.0), events=event(
        guard="<apply><lt/><ci>A</ci><cn>79.5</cn></apply>",
        values=(("k", "<cn>0</cn>"),)))
    with session(original, kind, seed) as runtime:
        result = wait(runtime, 2.0)
        assert counts(runtime) == {"A": 79, "B": 1, "C": 3}
        report = diagnostics(result)
        work = report["stochastic_work"]
        assert report["source_events_executed"] == 1
        assert work["ssa_steps"] == 1 and work["poisson_draws"] == 0
        spec = runtime.compiled_composition.spec.physics.models_of("sbml")["formal_workspace"]
        assert spec.guard_critical_reactions == {"r"}


@pytest.mark.parametrize("kind", ("coarse", "fine"))
@pytest.mark.parametrize("seed", (17, 37))
def test_time_only_profile_keeps_poisson_and_never_proposes_past_source_deadline(
    kind, seed, monkeypatch
):
    module = importlib.import_module("chemworld.physchem.stochastic_tau_leap")
    original_advance = module.advance_tau_leap
    recorded = []

    def observed(*args, **kwargs):
        result = original_advance(*args, **kwargs)
        deadline = kwargs.get("calendar_deadline_s")
        if deadline is not None:
            assert all(batch.end_time_s <= deadline for batch in result.batches)
            packet = result.continuation["pending"]
            assert (packet is None or packet["mode"] != "leap"
                    or packet["end_time_s"] <= deadline)
        recorded.append((deadline, result))
        return result

    monkeypatch.setattr(module, "advance_tau_leap", observed)
    with session(stopped_at(), kind, seed) as runtime:
        result = wait(runtime, 0.037)
        assert any(deadline == 0.037 for deadline, _ in recorded)
        assert diagnostics(result)["stochastic_work"]["poisson_draws"] > 0
        assert diagnostics(result)["stochastic_work"]["accepted_leaps"] > 0
        assert diagnostics(result)["source_events_executed"] == 1
        assert counts(runtime)["B"] > 0
        before = owned(runtime)
        result = wait(runtime, 1.0)
        after = owned(runtime)
        assert after["coordinates"] == before["coordinates"]
        assert after["continuation"]["process"]["rng_state"]\
            == before["continuation"]["process"]["rng_state"]
        assert diagnostics(result)["stochastic_work"]["random_draws"] == 0


@pytest.mark.parametrize("kind", ("coarse", "fine"))
@pytest.mark.parametrize("seed", (17, 37))
def test_observation_splits_and_json_restore_preserve_same_absolute_calendar(kind, seed):
    with ExitStack() as stack:
        full = stack.enter_context(session(stopped_at(), kind, seed))
        initial = plain(full.snapshot().to_dict())
        split = stack.enter_context(closing(ChemWorldRuntime.from_snapshot(initial)))
        wait(full, 1.0)
        wait(split, 0.011)
        packet = owned(split)["continuation"]["process"]["pending"]
        assert packet is not None and packet["mode"] == "leap"
        assert split._state.ledger.time_s < packet["end_time_s"] <= 0.037
        checkpoint = plain(split.snapshot().to_dict())
        restored = stack.enter_context(closing(ChemWorldRuntime.from_snapshot(checkpoint)))
        for duration in (0.006, 0.02, 0.963):
            expected, actual = wait(split, duration), wait(restored, duration)
            assert plain(actual.info) == plain(expected.info)
            assert plain(restored._state.to_dict()) == plain(split._state.to_dict())
        assert owned(split) == owned(full)
        assert split._state.ledger.time_s == full._state.ledger.time_s == 1.0


@pytest.mark.parametrize("kind", ("coarse", "fine"))
@pytest.mark.parametrize("seed", (17, 37))
def test_real_calendar_control_invalidates_future_packet_at_zero_time_without_draw(kind, seed):
    with session(controlled_deadline(), kind, seed, controlled_parameters=("due",)) as runtime:
        wait(runtime, 1e-6)
        before = owned(runtime)
        clock_before = runtime._state.ledger.time_s
        packet = before["continuation"]["process"]["pending"]
        assert packet is not None and packet["mode"] == "leap"
        new_due = (clock_before+packet["end_time_s"])/2
        result = runtime.execute({"operation": "set_model_parameters", "values": {"due": new_due}})
        assert result.info["transaction_status"] == "committed", result.info
        after = owned(runtime)
        assert after["continuation"]["process"]["pending"] is None
        assert after["continuation"]["process"]["rng_state"]\
            == before["continuation"]["process"]["rng_state"]
        assert runtime._state.ledger.time_s == clock_before
        assert diagnostics(result)["source_events_executed"] == 0
        result = wait(runtime, new_due-clock_before)
        assert diagnostics(result)["source_events_executed"] == 1
        assert owned(runtime)["coordinates"]["P:k"] == 0


@pytest.mark.parametrize("kind", ("coarse", "fine"))
def test_snapshot_rejects_feasible_but_multi_firing_source_critical_leap(kind):
    original = parsed(original=source(coefficient=1.0), events=event(
        guard="<apply><lt/><ci>A</ci><cn>79.5</cn></apply>",
        values=(("k", "<cn>0</cn>"),)))
    with session(original, kind) as runtime:
        snapshot = plain(runtime.snapshot().to_dict())
        memory = settings(snapshot["world_state"])["continuation"]
        process = memory["process"]
        process["pending"] = {"schema": PENDING_SCHEMA, "algorithm": TAU,
            "policy": policy(kind).to_dict(), "mode": "leap", "channel_ids": ["r"],
            "signature": process["signature"], "start_time_s": 0.0,
            "end_time_s": policy(kind).max_tau_s, "firings": [2]}
        with pytest.raises(ValueError):
            ChemWorldRuntime.from_snapshot(snapshot)


@pytest.mark.parametrize("kind", ("coarse", "fine"))
def test_snapshot_rejects_otherwise_valid_packet_crossing_new_known_calendar(kind):
    with session(controlled_deadline(), kind, controlled_parameters=("due",)) as runtime:
        wait(runtime, 1e-6)
        snapshot = plain(runtime.snapshot().to_dict())
        payload = settings(snapshot["world_state"])
        process = payload["continuation"]["process"]
        packet = process["pending"]
        assert packet is not None and packet["mode"] == "leap"
        # Leave the packet's own feasible interval and firing data unchanged.
        # Only forge a new live controlled input whose calendar excludes it.
        payload["coordinates"]["P:due"] = (process["time_s"]+packet["end_time_s"])/2
        spec = runtime.compiled_composition.spec.physics.models_of("sbml")["formal_workspace"]
        integer, coordinates = spec.integer_kinetics, payload["coordinates"]
        signature = _signature(integer.counts(coordinates, process["time_s"]),
            integer.compile(coordinates, process["time_s"]),
            integer.policy(coordinates, TAU, spec.maximum_events, spec.tau_policy))
        process["signature"] = packet["signature"] = signature
        with pytest.raises(ValueError):
            ChemWorldRuntime.from_snapshot(snapshot)


def kernel_fixture():
    channels = (EventChannel("critical", (("A", -1), ("B", 1)), (("A", 1),), 1.0),
                EventChannel("other", (("C", -1), ("D", 1)), (("C", 1),), 1.0))
    spec = EventProcessSpec(algorithm=TAU, max_events_per_advance=100000,
                           tau_policy=policy("coarse"), context="source-event-functional")
    counts = {"A": 1000, "B": 0, "C": 1000, "D": 0}
    memory = initial_continuation(17, algorithm=TAU)
    return channels, tuple(counts), spec, counts, memory


@pytest.mark.parametrize("seed", (17, 37))
def test_only_selected_critical_channel_fires_once_while_other_channel_keeps_poisson(seed):
    channels, species, spec, counts, _ = kernel_fixture()
    memory = initial_continuation(seed, algorithm=TAU)
    result = advance_event_channels(channels, species, spec, counts, memory, 0.1,
        stop_after_boundary=True, critical_reaction_ids=frozenset({"critical"}))
    assert result.counts["A"] == 999 and result.counts["B"] == 1
    assert result.counts["C"]+result.counts["D"] == 1000
    assert len(result.batches) == 1 and result.events == ()
    assert result.work["poisson_draws"] == 1 and result.work["ssa_steps"] == 0


@pytest.mark.parametrize("options", (
    {"critical_reaction_ids": None}, {"critical_reaction_ids": True},
    {"critical_reaction_ids": {"critical"}}, {"critical_reaction_ids": frozenset({1})},
    {"critical_reaction_ids": frozenset({"missing"})},
    {"calendar_deadline_s": True}, {"calendar_deadline_s": "tomorrow"},
    {"calendar_deadline_s": float("nan")}, {"calendar_deadline_s": float("inf")},
    {"calendar_deadline_s": -0.1}, {"calendar_deadline_s": 0.005},
    {"fluctuation_scaled_tau": None}, {"fluctuation_scaled_tau": 0},
    {"fluctuation_scaled_tau": 1}, {"fluctuation_scaled_tau": "true"},
))
def test_invalid_critical_or_calendar_policy_is_rejected_without_changing_inputs(options):
    channels, species, spec, counts, memory = kernel_fixture()
    before = copy.deepcopy((counts, memory))
    with pytest.raises(ValueError):
        advance_event_channels(channels, species, spec, counts, memory, 0.01, **options)
    assert (counts, memory) == before


@pytest.mark.parametrize("critical", (frozenset(), frozenset({"critical"})))
def test_zero_duration_policy_rebinding_keeps_original_random_stream(critical):
    channels, species, spec, counts, memory = kernel_fixture()
    result = advance_event_channels(channels, species, spec, counts, memory, 0,
        critical_reaction_ids=critical, calendar_deadline_s=0.037)
    assert result.counts == counts
    assert result.continuation["rng_state"] == memory["rng_state"]
    assert result.work["random_draws"] == 0


def test_calendar_only_rebinding_discards_a_valid_future_packet_without_new_draws():
    channels, species, spec, counts, memory = kernel_fixture()
    first = advance_event_channels(channels, species, spec, counts, memory, 1e-6,
                                   calendar_deadline_s=0.2)
    packet = first.continuation["pending"]
    assert packet is not None and packet["mode"] == "leap"
    deadline = (first.continuation["time_s"]+packet["end_time_s"])/2
    rebound = advance_event_channels(channels, species, spec, first.counts,
                                    first.continuation, 0, calendar_deadline_s=deadline)
    assert rebound.continuation["pending"] is None
    assert rebound.continuation["rng_state"] == first.continuation["rng_state"]
    assert rebound.counts == first.counts and rebound.work["random_draws"] == 0
    assert rebound.work["calendar_replanned_packets"] == 1


def test_calendar_invalidation_does_not_hide_malformed_old_packet():
    channels, species, spec, counts, memory = kernel_fixture()
    first = advance_event_channels(channels, species, spec, counts, memory, 1e-6,
                                   calendar_deadline_s=0.2)
    forged = copy.deepcopy(first.continuation)
    forged["pending"]["schema"] = "forged"
    before = copy.deepcopy(forged)
    with pytest.raises(ValueError, match="declaration"):
        advance_event_channels(channels, species, spec, first.counts, forged, 0,
                               calendar_deadline_s=0.002)
    assert forged == before


def test_new_keyword_defaults_leave_the_original_compiled_kernel_result_exact():
    channels, species, spec, counts, memory = kernel_fixture()
    original = advance_event_channels(channels, species, spec, counts, memory, 1.0)
    explicit = advance_event_channels(channels, species, spec, counts, memory, 1.0,
        critical_reaction_ids=frozenset(), calendar_deadline_s=None)
    assert asdict(explicit) == asdict(original)
