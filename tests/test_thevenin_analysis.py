"""Opt-in analysis uses the original world, resource ledger, records and replay."""

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace

import numpy as np
import pytest

from chemworld.agents.base import BaseAgent
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.cli import main
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.thevenin_frequency import thevenin_impedance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thevenin import METER, unpack_thevenin
from chemworld.world.thevenin_analysis import COST, FREQUENCIES_HZ, THEVENIN_ANALYSIS
from chemworld.world.thevenin_world import thevenin_world_spec
from test_battery_thevenin import fixture
from test_thevenin_runtime import execute


def session(spec=None, **kwargs):
    result = ChemWorldRuntime(composition=thevenin_world_spec(
        fixture() if spec is None else spec, include_impedance_analysis=True), **kwargs)
    result.reset(seed=7)
    return result


def analyze():
    return {"operation": "measure", "instrument": THEVENIN_ANALYSIS}


def test_opt_in_analysis_has_exact_native_axis_no_hidden_readout_or_physical_advance():
    runtime = session()
    before = runtime.snapshot().to_dict()
    initial = unpack_thevenin(runtime._state)[3]
    result = execute(runtime, analyze())
    after = runtime.snapshot().to_dict()
    assert dict(unpack_thevenin(runtime._state)[3].coordinates) == dict(initial.coordinates)
    assert runtime._state.ledger.time_s == initial.time_s == 0.
    assert runtime._state.ledger.cost == COST and runtime._state.ledger.sample_consumed_L == 0.
    assert runtime._state.ledger.analysis_time_s == 0.
    assert before["rng_state"] == after["rng_state"]
    assert result.info["actual_duration_s"] == 0.
    channels = result.observation.channels
    assert {c.channel_id for c in channels} == {"impedance_real", "impedance_imag"}
    assert all(c.unit == "Ohm" and c.shape == (13,) and c.axis.values == FREQUENCIES_HZ
               and c.axis.unit == "Hz" and c.uncertainty is None for c in channels)
    expected = np.asarray(thevenin_impedance(fixture(), FREQUENCIES_HZ).impedance_ohm)
    np.testing.assert_array_equal(channels[0].value, expected.real)
    np.testing.assert_array_equal(channels[1].value, expected.imag)
    contract = runtime.domain_modules.instrument_specs()[THEVENIN_ANALYSIS].contract
    assert contract.clock_semantics == "instantaneous_snapshot"
    assert "not a physical sinusoidal acquisition" in contract.synthetic_boundary
    assert contract.axis_contract["values"] == list(FREQUENCIES_HZ)
    cached = runtime._state.to_dict()
    runtime.observe()
    assert cached == runtime._state.to_dict()


def test_old_world_does_not_silently_gain_a_new_analysis_contract():
    runtime = ChemWorldRuntime(composition=thevenin_world_spec(fixture()))
    runtime.reset(seed=7)
    before = dict(unpack_thevenin(runtime._state)[3].coordinates)
    result = runtime.execute(analyze())
    assert result.info["transaction_status"] != "committed"
    assert before == dict(unpack_thevenin(runtime._state)[3].coordinates)
    with pytest.raises(ValueError, match="explicitly enable"):
        thevenin_world_spec(fixture(), include_impedance_analysis="yes")


@pytest.mark.parametrize("control", ({"current_A": .5}, {"potential_V": 3.7}))
def test_analysis_never_substitutes_a_different_live_control_source(control):
    runtime = session()
    execute(runtime, {"operation": "configure_electrical_control", **control})
    before = dict(unpack_thevenin(runtime._state)[3].coordinates)
    result = runtime.execute(analyze())
    assert result.info["measurement_status"] in {"failed", "invalid", "not_observed"}
    assert before == dict(unpack_thevenin(runtime._state)[3].coordinates)
    assert runtime._state.ledger.time_s == 0.


def test_actual_hot_device_does_not_get_replaced_by_equilibrium_for_analysis():
    runtime = session(fixture(initial_cell_temperature_K=310.))
    before = dict(unpack_thevenin(runtime._state)[3].coordinates)
    result = runtime.execute(analyze())
    assert result.info["measurement_status"] in {"failed", "invalid", "not_observed"}
    assert before == dict(unpack_thevenin(runtime._state)[3].coordinates)
    assert runtime._state.ledger.time_s == 0.


def test_analysis_is_an_original_budgeted_instrument_use_not_a_free_truth_query():
    card = CampaignResourceCard(card_id="analysis-use-test", operation_attempt_limit=10,
        vessel_start_limit=1, final_assay_limit=1, nonfinal_instrument_use_limit=1, stock_limits={})
    runtime = session(campaign_resource_card=card)
    execute(runtime, analyze())
    result = runtime.execute(analyze())
    assert result.info["transaction_status"] != "committed"
    ledger = runtime._campaign_resource_ledger
    assert ledger.instrument_uses == {THEVENIN_ANALYSIS: 1} and ledger.nonfinal_instrument_uses == 1
    assert runtime._state.ledger.time_s == 0.


def test_offline_example_analysis_is_explicit_and_original_default_remains_nine_steps(tmp_path):
    from chemworld.examples.native_thevenin import run, workflow

    world, actions = workflow(include_impedance_analysis=True)
    assert len(workflow()[1]) == 9 and len(actions) == 10
    assert THEVENIN_ANALYSIS in world.to_dict()["task"]["instruments"]
    assert actions[0] == analyze()
    report = run(tmp_path / "offline-analysis.jsonl", include_impedance_analysis=True)
    assert report["steps"] == report["replay_steps"] == 10


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_full_agent_cli_replay_and_sorted_snapshot_new_process_continuation(tmp_path, noise):
    from test_native_domains_end_to_end import (
        test_snapshot_resume_in_middle_of_real_workflow as resume,
    )

    world = thevenin_world_spec(fixture(), include_impedance_analysis=True)
    actions = (analyze(), {"operation": "wait", "duration_s": .1},
        {"operation": "measure", "instrument": METER}, analyze(), analyze(),
        {"operation": "configure_electrical_control", "current_A": .2},
        {"operation": "wait", "duration_s": 1.},
        {"operation": "measure", "instrument": METER}, {"operation": "terminate"})

    class Sequence(BaseAgent):
        def act(self, history):
            return actions[len(history)]

    path = tmp_path / "agent.jsonl"
    run_agent(env_id="ChemWorld", agent=Sequence(), composition=world.to_dict(),
        world_split="public-dev", budget=80, objective="balanced", seed=7,
        observation_noise_mode=noise, observation_seed=12, output_path=path)
    records = load_jsonl(path)
    assert len(records) == 9 and all(r["transaction_status"] == "committed" for r in records)
    assert verify_records(records, tolerance=1e-8).verified
    composition = tmp_path / "composition.json"
    composition.write_text(json.dumps(world.to_dict(), sort_keys=True), encoding="utf8")
    inputs = []
    for index, action in enumerate(actions):
        action_path = tmp_path / f"action-{index}.json"
        action_path.write_text(json.dumps(action), encoding="utf8")
        inputs.append(str(action_path))
    cli_path = tmp_path / "cli.jsonl"
    with redirect_stdout(io.StringIO()):
        main(["render", "--composition", str(composition), "--seed", "7", "--noise-mode", noise,
              "--observation-seed", "12", "--actions", *inputs,
              "--format", "json", "--trajectory", str(cli_path)])
    cli_records = load_jsonl(cli_path)
    assert len(cli_records) == 9 and verify_records(cli_records, tolerance=1e-8).verified
    resume("Thevenin-analysis", noise, tmp_path, recipe=(world, actions), sort_keys=True)


def test_stationary_diffusion_analysis_uses_actual_spatial_state_without_spending_time():
    from chemworld.examples.native_thevenin import diffusion_spec

    spec = diffusion_spec(cells=8)
    runtime = session(spec)
    snapshot = runtime.snapshot().to_dict()
    before = unpack_thevenin(runtime._state)[3]
    result = execute(runtime, analyze())
    after = unpack_thevenin(runtime._state)[3]
    assert before == after
    assert runtime.snapshot().to_dict()["rng_state"] == snapshot["rng_state"]
    assert result.info["actual_duration_s"] == runtime._state.ledger.time_s == 0.
    assert runtime._state.ledger.cost == COST
    expected = np.asarray(thevenin_impedance(spec, FREQUENCIES_HZ).impedance_ohm)
    channels = result.observation.channels
    assert {c.channel_id for c in channels} == {"impedance_real", "impedance_imag"}
    np.testing.assert_array_equal(channels[0].value, expected.real)
    np.testing.assert_array_equal(channels[1].value, expected.imag)
    lumped = np.asarray(thevenin_impedance(
        replace(spec, diffusion=None), FREQUENCIES_HZ).impedance_ohm)
    assert expected[0].real > lumped[0].real


def test_nonuniform_diffusion_rest_profile_is_not_replaced_by_uniform_state_for_analysis():
    from chemworld.examples.native_thevenin import diffusion_spec

    spec = diffusion_spec(cells=8)
    original = session(spec)
    payload = json.loads(json.dumps(original.snapshot().to_dict(), sort_keys=True, allow_nan=False))
    # Isolate spatial relaxation: current, RC and both body temperatures remain
    # at their stationary values; only a valid, zero-mean SOC deviation changes.
    slot = next(iter(payload["world_state"]["state_slots"].values()))
    slot["values"][spec.diffusion.coordinate_keys[0]] = .001
    runtime = ChemWorldRuntime.from_snapshot(payload)
    before = unpack_thevenin(runtime._state)[3]
    assert spec.diffusion.surface_soc(before.coordinates["thevenin:soc"], before.coordinates) != .7
    result = runtime.execute(analyze())
    assert result.info["measurement_status"] in {"failed", "invalid", "not_observed"}
    assert unpack_thevenin(runtime._state)[3] == before
    assert runtime._state.ledger.time_s == 0.
