"""Behavior across readout failures, budget scopes and shared lifecycle preparation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.examples.native_concentration_research import (
    initial_concentration_state,
    synthetic_concentration_spec,
)
from chemworld.foundation import Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.concentration_world import concentration_world_spec
from chemworld.world.surface_world import surface_world_spec


def resource_card(*, batch_measurements=4):
    local = CampaignResourceCard(
        card_id="local", operation_attempt_limit=20, vessel_start_limit=1,
        final_assay_limit=1, nonfinal_instrument_use_limit=batch_measurements,
        stock_limits={"solvent_L": .04, "reagent_mol": .02},
    )
    return CampaignResourceCard(
        card_id="campaign", operation_attempt_limit=40, vessel_start_limit=2,
        final_assay_limit=2, nonfinal_instrument_use_limit=8,
        stock_limits={"solvent_L": .08, "reagent_mol": .04},
        per_batch_limits=local.to_dict()["hard_limits"],
    )


def liquid_session(noise="sequential", **card_options):
    session = ChemWorldRuntime(
        task_id="reaction-to-assay", episode_mode_override="campaign",
        campaign_resource_card=resource_card(**card_options), observation_noise_mode=noise,
    )
    session.reset(seed=5)
    for action in [
        {"operation": "add_solvent", "volume_L": .02, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": .01},
    ]:
        assert session.execute(action).info["transaction_status"] == "committed"
    return session


def failing_observer(mode):
    def observe(state, action, rng):
        rng.normal(size=3)
        if mode == "exception":
            raise ValueError("controlled readout domain failure")
        return Observation(
            channels=(ObservationChannel("invalid", "dimensionless", 2, valid_range=(0, 1)),),
        )
    return observe


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("failure", ["exception", "contract"])
def test_failed_readout_keeps_sample_and_settles_both_scopes_once(monkeypatch, noise, failure):
    session = liquid_session(noise)
    before = session._state
    random_state = deepcopy(session._rng.bit_generator.state)
    with monkeypatch.context() as patch:
        patch.setattr(session.observation_kernel, "observe", failing_observer(failure))
        result = session.execute({"operation": "measure", "instrument": "hplc"})
    assert result.info["measurement_status"] == "invalid"
    assert result.info["transaction_status"] == "committed"
    assert result.info["outcome_diagnostics"]["observation_failure"] == (
        "readout_domain_error: ValueError" if failure == "exception" else "readout_contract_failed"
    )
    assert session._state.volume_L < before.volume_L
    assert session._state.ledger.sample_consumed_L > before.ledger.sample_consumed_L
    assert session._state.process.last_observation_record == before.process.last_observation_record
    assert session._rng.bit_generator.state == random_state
    assert session.observation_noise_provenance()["status"] == "invalid"
    key = (0, "measure", "hplc")
    assert session._observation_occurrences.get(key, 0) == (1 if noise == "keyed" else 0)
    snapshots = [
        ledger.snapshot() for ledger in (
            session._campaign_resource_ledger, session._batch_resource_ledger,
        )
    ]
    assert snapshots[0]["events"][-1]["outcome"] == snapshots[1]["events"][-1]["outcome"]
    for snapshot in snapshots:
        state = snapshot["state"]
        assert state["operation_attempts"] == 3
        assert state["nonfinal_instrument_uses"] == 1
        assert state["report_only"]["sample_consumed_L"] == pytest.approx(
            session._state.ledger.sample_consumed_L,
        )
    for _ in range(3):
        session.observe()
        session.task_info()
        session.snapshot()
        session.action_schema("measure")
        session.public_campaign_resource_state()
    assert snapshots == [
        ledger.snapshot() for ledger in (
            session._campaign_resource_ledger, session._batch_resource_ledger,
        )
    ]
    result = session.execute({"operation": "measure", "instrument": "hplc"})
    assert result.info["measurement_status"] == "valid"
    if noise == "keyed":
        assert session.observation_noise_provenance()["coordinate"]["replicate_index"] == 1


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("failure", ["exception", "contract"])
def test_nonmeasurement_observation_failure_rolls_back_physics_and_noise(
    monkeypatch, noise, failure,
):
    session = liquid_session(noise)
    before = session._state
    random_state = deepcopy(session._rng.bit_generator.state)
    occurrences = dict(session._observation_occurrences)
    with monkeypatch.context() as patch:
        patch.setattr(session.observation_kernel, "observe", failing_observer(failure))
        result = session.execute({"operation": "add_solvent", "volume_L": .001, "solvent": 0})
    assert result.info["transaction_status"] != "committed"
    assert session._state.species_amounts == before.species_amounts
    assert session._state.volume_L == before.volume_L
    assert session._state.ledger.time_s == before.ledger.time_s
    assert session._rng.bit_generator.state == random_state
    assert session._observation_occurrences == occurrences
    assert session.observation_noise_provenance()["status"] == "rolled_back"
    for ledger in (session._campaign_resource_ledger, session._batch_resource_ledger):
        delta = ledger.snapshot()["events"][-1]["outcome"]["delta"]
        assert delta["stocks"] == {}
        assert delta["report_only"]["process_time_s"] == 0
        assert delta["report_only"]["physical_cost"] == 0
        assert ledger.operation_attempts == 3


def test_batch_only_rejection_is_reported_in_the_settled_receipt():
    session = liquid_session(batch_measurements=0)
    before = session._state
    result = session.execute({"operation": "measure", "instrument": "hplc"})
    assert result.info["transaction_status"] == "campaign_resource_rejected"
    receipt = session.public_campaign_resource_state()["latest_receipt"]
    assert receipt["rejected"] and receipt["preflight"]["allowed"]
    assert receipt["rejection_reasons"] == ["per_batch:nonfinal_instrument_use_limit"]
    assert session._state.species_amounts == before.species_amounts
    assert session._state.equipment == before.equipment
    assert session._state.ledger.sample_consumed_L == before.ledger.sample_consumed_L
    assert session._campaign_resource_ledger.nonfinal_instrument_uses == 0
    assert session._batch_resource_ledger.nonfinal_instrument_uses == 0


def test_stopped_acquisition_settles_actual_time_in_each_budget_scope():
    spec = surface_world_spec(include_surface=False)
    models = dict(spec.physics.models_of('thermal'))
    models["reactor"] = replace(models["reactor"], minimum_temperature_K=349.99)
    components = tuple(
        replace(c, parameters={**c.parameters, "temperature_range_K": [349.99, 430]})
        if c.kind == "thermal" and c.vessel == "reactor" else c for c in spec.components
    )
    spec = replace(
        spec,
        components=components,
        physics=replace(spec.physics, models={**spec.physics.models, "thermal": models}),
    )
    session = ChemWorldRuntime(composition=spec, campaign_resource_card=resource_card())
    session.reset(seed=12)
    assert session.execute({"operation": "pause_control"}).info["transaction_status"] == "committed"
    result = session.execute({"operation": "measure", "instrument": "gas_composition"})
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["measurement_status"] == "invalid"
    assert 0 < result.info["actual_duration_s"] < 5
    for ledger in (session._campaign_resource_ledger, session._batch_resource_ledger):
        assert ledger.process_time_s == session._state.ledger.time_s
        assert ledger.process_time_s == result.info["actual_duration_s"]
        assert ledger.nonfinal_instrument_uses == 1


def test_campaign_and_explicit_batch_reset_share_retained_thermal_closure():
    model = synthetic_concentration_spec(30000)
    spec = concentration_world_spec(model, initial_concentration_state(model))
    session = ChemWorldRuntime(
        composition=spec, episode_mode_override="campaign",
    )
    session.reset(seed=21)
    assert session.execute({"operation": "concentrate", "duration_s": 1}).info[
        "transaction_status"
    ] == "committed"
    manual = ChemWorldRuntime.from_snapshot(session.snapshot())
    manual.reset(options={"scope": "batch"})
    result = session.execute({"operation": "terminate"})
    assert result.info["next_experiment_ready"]
    assert session._state.temperature_K == manual._state.temperature_K
    assert session._state.species_amounts == manual._state.species_amounts
    assert session._state.state_slots == manual._state.state_slots
    assert session._state.equipment == manual._state.equipment
    assert session._state.ledger == manual._state.ledger
    assert session._lifecycle_history[-1]["concentration_equilibration"] == (
        manual._lifecycle_history[-1]["concentration_equilibration"]
    )
