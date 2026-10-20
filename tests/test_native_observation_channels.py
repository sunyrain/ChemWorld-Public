"""Physical readings, actual acquisition/cache paths and fixed Gym compatibility."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import gymnasium as gym
import numpy as np
import pytest

import chemworld  # noqa: F401
from chemworld.agent_interface import tool_json_view
from chemworld.data.logging import load_jsonl
from chemworld.envs.spaces import OBSERVATION_KEYS, to_observation
from chemworld.eval.runner import make_agent, run_agent
from chemworld.eval.verify import verify_records
from chemworld.foundation import (
    ChannelAxis,
    Instrument,
    Observation,
    ObservationChannel,
    ObservationTime,
    ScalarProjection,
    WorldState,
)
from chemworld.foundation.observations import LEGACY_PROJECTION_VERSION
from chemworld.foundation.state_ledgers import ProcessLedger
from chemworld.world.domain_modules import (
    ComponentCapability,
    DomainModuleRegistry,
    DomainModuleSpec,
    InstrumentSpec,
)
from chemworld.world.instruments import instrument_contracts, public_native_observation


@pytest.fixture
def env(request):
    instance = gym.make("ChemWorld", task_id=getattr(request, "param", "reaction-to-assay"), seed=7)
    instance.reset(seed=7)
    yield instance
    instance.close()


def checked(env, *channels, projection=()):
    obs = Observation(channels=channels, projection=projection)
    report = env.unwrapped.constitution.check_observation(obs)
    assert report.passed, report.failures()
    return obs


def test_native_voltage_pressure_and_curve_keep_physical_values(env):
    curve = ObservationChannel(
        "voltage_trace",
        "V",
        (4.2, 4.1, 4.0),
        axis=ChannelAxis("elapsed", "s", (0, 1, 2)),
        valid_range=(0, 5),
        uncertainty=0.01,
        sampled_at=ObservationTime("sample", 5),
        available_at=ObservationTime("world", 17),
        sample_id="sample-1",
    )
    obs = checked(
        env,
        ObservationChannel("voltage", "V", 4.2),
        ObservationChannel("pressure", "Pa", 100000),
        curve,
    )
    assert obs.values == {}
    assert obs.channel("voltage").value == 4.2
    assert obs.channel("pressure").value == 100000
    assert obs.channel("voltage_trace").shape == (3,)
    restored = Observation.from_native_record(json.loads(json.dumps(obs.native_record())))
    assert restored.channels == obs.channels
    assert restored.channel("voltage_trace").sampled_at.clock_id == "sample"
    assert restored.channel("voltage_trace").available_at.time_s == 17


@pytest.mark.parametrize(
    ("value", "quality", "kwargs", "projected", "observed"),
    [
        (0.0, "valid", {}, 0.0, True),
        (None, "missing", {}, None, False),
        (float("nan"), "missing", {}, None, False),
        (None, "not_observed", {}, None, False),
        (0.01, "below_detection", {"detection_limit": 0.02}, None, False),
        (5.0, "saturated", {}, 1.0, True),
        (5.5, "out_of_range", {}, None, False),
    ],
)
def test_quality_preserves_zero_missing_detection_and_range(
    env,
    value,
    quality,
    kwargs,
    projected,
    observed,
):
    obs = checked(
        env,
        ObservationChannel("voltage", "V", value, quality, valid_range=(0, 5), **kwargs),
        projection=(ScalarProjection("voltage", "voltage", 0, 5),),
    )
    assert obs.values["voltage"] == projected
    assert obs.observed_mask["voltage"] is observed
    native = json.loads(json.dumps(obs.native_record(), allow_nan=False))
    assert native["channels"][0]["quality"] == quality
    processed = checked(env, replace(obs.channels[0], role="processed"))
    assert processed.processed_estimate["voltage"] == obs.channels[0].value


@pytest.mark.parametrize(
    "channel",
    [
        ObservationChannel("voltage", "V", float("nan")),
        ObservationChannel("voltage", "V", float("inf"), "invalid"),
        ObservationChannel("voltage", "V", 5.5, valid_range=(0, 5)),
        ObservationChannel("voltage", "V", 0.2, "below_detection", detection_limit=0.1),
        ObservationChannel("voltage", "unknown_unit", 4.2),
        ObservationChannel("voltage", "V", 4.2, uncertainty=-0.1),
        ObservationChannel("voltage", "V", (4.2, 4.1)),
        ObservationChannel("voltage", "V", (4.2, 4.1), axis=ChannelAxis("t", "s", (1, 1))),
        ObservationChannel("voltage", "V", (4.2, 4.1), axis=ChannelAxis("t", "s", (1, 2, 3))),
        ObservationChannel(
            "voltage",
            "V",
            4.2,
            sampled_at=ObservationTime(time_s=5),
            available_at=ObservationTime(time_s=4),
        ),
        ObservationChannel("processed", "V", 4.2, role="processed", derived_from=("absent",)),
    ],
)
def test_invalid_native_readings_fail_same_constitution_and_tool_checks(env, channel):
    obs = Observation(channels=(channel,))
    assert not env.unwrapped.constitution.check_observation(obs).passed
    with pytest.raises(ValueError, match="Invalid public"):
        public_native_observation(obs.native_record(), env.unwrapped.constitution)


@pytest.mark.parametrize("identity", ["true_yield", "rate_constants", "solver_error", "A"])
def test_native_channel_identity_cannot_smuggle_hidden_truth(env, identity):
    # A is a real private species identifier in this actual world.
    if identity == "A":
        identity = next(iter(env.unwrapped.constitution.substances))
    obs = Observation(channels=(ObservationChannel(identity, "dimensionless", 0.2),))
    assert not env.unwrapped.constitution.check_observation(obs).passed
    with pytest.raises(ValueError):
        tool_json_view(env, {}, {"native_observation": obs.native_record()})


def test_scalar_compatibility_is_derived_and_gym_order_is_fixed(env):
    obs = checked(
        env,
        ObservationChannel("cost", "currency", 4.2),
        ObservationChannel("score", "dimensionless", 0.37),
        projection=(ScalarProjection("score", "score"), ScalarProjection("cost", "cost")),
    )
    assert obs.values == {"score": 0.37, "cost": 1.0}
    obs.values["score"] = 0.9
    assert obs.values["score"] == 0.37
    with pytest.raises(FrozenInstanceError):
        obs.channels[0].value = 0.0
    vector = to_observation(obs.values)
    assert tuple(vector) == OBSERVATION_KEYS
    assert float(vector["score"][0]) == pytest.approx(0.37)
    assert np.isnan(vector["yield"][0])
    assert obs.projection_version == LEGACY_PROJECTION_VERSION
    with pytest.raises(ValueError, match="sources"):
        Observation(values={"score": 0.2}, channels=obs.channels)
    bad = Observation(
        channels=obs.channels, projection=obs.projection, projection_version="future-version"
    )
    assert not env.unwrapped.constitution.check_observation(bad).passed


@pytest.mark.parametrize(
    ("env", "instrument"),
    [
        (
            "equilibrium-characterization" if instrument == "ph_meter" else "reaction-to-assay",
            instrument,
        )
        for instrument in ("hplc", "uvvis", "ph_meter", "nmr", "ir", "ms")
    ],
    indirect=["env"],
)
def test_actual_acquired_native_signal_cache_and_snapshot_have_one_owner(env, instrument):
    env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 1})
    env.step({"operation": "add_reagent", "amount_mol": 0.010})
    _, _, _, _, info = env.step({"operation": "measure", "instrument": instrument})
    assert info["measurement_status"] == "valid"
    cached = env.unwrapped._state.process.last_observation_record
    assert cached is not None
    signal = cached.channel("acquired_signal" if instrument == "ph_meter" else "acquired_trace")
    assert signal.sampled_at.time_s == env.unwrapped._state.ledger.time_s
    assert signal.available_at == signal.sampled_at
    assert signal.sample_id
    assert not signal.errors()
    packet = cached.native_record()
    assert "$channel_value" in json.dumps(packet["raw_template"])
    assert all(
        c.derived_from
        if instrument in {"nmr", "ir", "ms"}
        else c.acquisition_relation == "legacy_parallel_readout"
        for c in cached.channels
        if c.role == "processed" and c.channel_id in cached.processed_estimate
    )
    restored = WorldState.from_dict(json.loads(json.dumps(env.unwrapped._state.to_dict())))
    assert restored.process.last_observation_record.native_record() == packet
    assert "last_observation" not in restored.process.to_dict()
    tool = tool_json_view(env, {}, info)
    assert tool["native_observation"] == packet
    _, _, _, _, after = env.step(
        {"operation": "wait", "duration_s": 2.0, "stirring_speed_rpm": 400.0}
    )
    assert after["transaction_status"] == "committed"
    assert after["native_observation"]["channels"][0] == packet["channels"][0]
    assert env.unwrapped._state.process.last_observation_record.native_record() == packet


def pressure_observer(state, action, rng):
    """A real pressure readout; no new pressure dynamics are claimed here."""
    return Observation(
        channels=(
            ObservationChannel(
                "pressure_readout",
                "Pa",
                state.pressure_Pa,
                uncertainty=1.0,
                sampled_at=ObservationTime(time_s=state.ledger.time_s),
                sample_id="active-vessel",
            ),
        ),
        instrument_id="pressure_probe",
        cost=0.01,
    )


def test_declared_native_instrument_runs_through_real_measurement_transaction():
    contract = replace(
        instrument_contracts()["hplc"],
        instrument_id="pressure_probe",
        observable_keys=(),
        cost=0.01,
        sample_consumption_L=0.0,
    )
    module = DomainModuleSpec(
        "native_pressure",
        instruments=(
            InstrumentSpec(
                Instrument("pressure_probe", "Pressure probe", (), 0.01, 0.0, {}),
                contract,
                native_observer=pressure_observer,
            ),
        ),
        components=(
            (
                "native_pressure",
                ComponentCapability(
                    operations=("measure",),
                ),
            ),
        ),
    )
    modules = DomainModuleRegistry((*DomainModuleRegistry.default().modules, module))
    request = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "pressure-bench",
        "world_split": "public-dev",
        "components": [
            {"kind": kind, "role": kind, "parameters": {}}
            for kind in ("reaction", "thermal", "observation", "native_pressure")
        ],
        "task": {
            "budget": 8,
            "description": "Native pressure acquisition.",
            "instruments": ["pressure_probe", "final_assay"],
        },
    }
    instance = gym.make("ChemWorld", composition=request, domain_modules=modules, seed=4)
    try:
        instance.reset(seed=4)
        instance.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 1})
        before = instance.unwrapped._state
        _, _, _, _, info = instance.step({"operation": "measure", "instrument": "pressure_probe"})
        assert info["transaction_status"] == "committed", (
            info.get("rollback_reason"),
            info.get("preconditions"),
            info.get("error_message"),
        )
        assert info["measurement_status"] == "valid"
        cached = instance.unwrapped._state.process.last_observation_record
        assert cached.channel("pressure_readout").value == before.pressure_Pa
        assert cached.channel("pressure_readout").unit == "Pa"
        execution = instance.unwrapped._state.equipment.equipment[
            "instrument:pressure_probe"
        ].settings["execution_history"][-1]
        assert execution["model_id"] == "native-observer:pressure_probe"
        assert execution["maturity"] == "development"
        assert instance.unwrapped._state.ledger.cost - before.ledger.cost == pytest.approx(0.01)
        assert instance.unwrapped._state.ledger.sample_consumed_L == before.ledger.sample_consumed_L
        assert tuple(to_observation(cached.values)) == OBSERVATION_KEYS
        reverse = DomainModuleRegistry(tuple(reversed(modules.modules)))
        assert reverse.instrument_specs()["pressure_probe"].native_observer is pressure_observer
    finally:
        instance.close()


def test_old_snapshot_cache_is_ingested_once_and_has_derived_views():
    ledger = ProcessLedger(last_observation={"score": 0.3}, last_observed_mask={"score": True})
    assert ledger.last_observation_record.channel("score").value == 0.3
    ledger.last_observation["score"] = 0.9
    assert ledger.last_observation["score"] == 0.3
    assert ProcessLedger(**ledger.to_dict()).to_dict() == ledger.to_dict()


def test_two_same_time_measurements_have_distinct_samples_and_tool_uses_native_views(env):
    env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 1})
    env.step({"operation": "add_reagent", "amount_mol": 0.010})
    _, _, _, _, first = env.step({"operation": "measure", "instrument": "hplc"})
    _, _, _, _, second = env.step({"operation": "measure", "instrument": "hplc"})
    a, b = first["native_observation"]["channels"][0], second["native_observation"]["channels"][0]
    assert a["sampled_at"] == b["sampled_at"]
    assert a["sample_id"] != b["sample_id"]
    altered = {**second, "raw_signal": {"rate_constants": float("inf")},
               "processed_estimate": {"yield": 0.999}, "observed_mask": {}}
    view = tool_json_view(env, {"yield": [0.999]}, altered)
    native = Observation.from_native_record(second["native_observation"])
    assert view["raw_signal"] == native.raw_signal
    assert view["processed_estimate"] == native.processed_estimate
    assert view["observed_mask"] == native.observed_mask


def test_actual_currency_remains_native_while_legacy_cost_and_score_stay_bounded(env):
    base = env.unwrapped
    state = base._state.replace(ledger=base._state.ledger.with_updates(cost=4.2))
    kernel = base.observation_kernel
    obs = kernel.observe(state, {"operation": "terminate"}, np.random.default_rng(1))
    assert obs.channel("cost").value == 4.2
    assert obs.values["cost"] == 1.0
    assert obs.values["score"] == kernel._score({**obs.values, "cost": 1.0})


def test_native_metadata_is_logged_and_exact_replay_rejects_tampering(tmp_path):
    path = tmp_path / "native.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    assert all(row["native_observation"] for row in records)
    assert verify_records(records, tolerance=0).verified
    altered = deepcopy(records)
    altered[0]["native_observation"]["channels"][0]["unit"] = "Pa"
    result = verify_records(altered, tolerance=0)
    assert not result.verified
    assert any(str(m["field"]).startswith("native_observation") for m in result.mismatches)
