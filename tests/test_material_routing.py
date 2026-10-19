from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.action_codec import ActionCodec
from chemworld.agent_interface import action_schema, observation_view
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.eval.verify import verify_records
from chemworld.foundation import equipment_settings
from chemworld.runtime.material_routing import _active
from test_recrystallization_cycles import cycle_actions, total_inventory


def create(slot, capacity=0.1):
    return {"operation": "create_container", "container": slot, "capacity_L": capacity}


def transfer(source, target, fraction=1.0, mixing=0):
    return {
        "operation": "transfer_material",
        "source_container": source,
        "destination_container": target,
        "transfer_fraction": fraction,
        "mixing": mixing,
    }


def make_env(seed=0):
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=seed,
        full_process_contract_id="phase-resolved-process-v5",
        episode_mode_override="single_experiment",
        budget_override=80,
    )
    env.reset(seed=seed)
    return env


def material_totals(state):
    samples = [_active(state), *state.samples.samples.values()]
    initial = {}
    number, third = 0.0, 0.0
    for sample in samples:
        if sample.contents:
            for s, n in sample.contents.species.initial_amounts_mol.items():
                initial[s] = initial.get(s, 0) + n
            for n, d in equipment_settings(sample.contents.equipment, "crystallizer").get(
                "population_cohorts", ()
            ):
                number += n
                third += n * d**3
    return {
        "amounts": total_inventory(state),
        "initial": initial,
        "volume": sum(s.volume_L for s in samples),
        "seed": sum(s.seed_target_mol for s in samples),
        "particles": number,
        "third_moment": third,
    }


def routing_actions():
    return [
        *cycle_actions()[1:8],
        create(1),
        create(2),
        transfer("active", "container-01", 0.25),
        transfer("active", "container-02", 1 / 3),
        transfer(2, 1),  # explicit negative: destination contains another aliquot
        transfer(2, 1, mixing=1),
        transfer(1, 0, mixing=1),
        {
            "operation": "heat",
            "target_temperature_K": 380.0,
            "duration_s": 900.0,
            "stirring_speed_rpm": 600.0,
        },
        {"operation": "cool_crystallize", "target_temperature_K": 260.0, "duration_s": 7200.0},
        {"operation": "filter_crystals"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def run_routing(path, seed):
    env = make_env(seed)
    task = {**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()}
    rows = []
    try:
        with TrajectoryLogger(path) as logger:
            for step, action in enumerate(routing_actions(), 1):
                before = env.unwrapped._state
                obs, reward, terminated, truncated, info = env.step(action)
                after = env.unwrapped._state
                logger.log(
                    task_info=task,
                    step=step,
                    action=action,
                    observation=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={},
                )
                committed = info["transaction_status"] == "committed"
                rows.append(
                    {
                        "step": step,
                        "committed": committed,
                        "invalid_reasons": info.get("invalid_reasons", []),
                    }
                )
                assert committed == (step != 12), (
                    step,
                    info.get("preconditions"),
                    info.get("rollback_reason"),
                )
                if action["operation"] == "transfer_material":
                    left, right = material_totals(before), material_totals(after)
                    for key in (
                        "amounts",
                        "initial",
                        "volume",
                        "seed",
                        "particles",
                        "third_moment",
                    ):
                        assert right[key] == pytest.approx(left[key], abs=1e-12, rel=1e-10), key
                    errors = {
                        key: max(
                            (
                                abs(right[key].get(s, 0) - left[key].get(s, 0))
                                for s in left[key].keys() | right[key].keys()
                            ),
                            default=0,
                        )
                        for key in ("amounts", "initial")
                    }
                    errors.update({key: abs(right[key] - left[key]) for key in ("volume", "seed")})
                    errors.update(
                        {
                            key: abs(right[key] - left[key]) / max(abs(left[key]), 1e-300)
                            for key in ("particles", "third_moment")
                        }
                    )
                    rows[-1]["conservation_errors"] = errors
                    assert errors["amounts"] <= 1e-10
                    assert max(errors[k] for k in ("initial", "volume", "seed")) <= 1e-12
                    assert max(errors[k] for k in ("particles", "third_moment")) <= 1e-10
                    assert after.ledger.time_s == before.ledger.time_s
                    assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
                    assert (
                        after.samples.transfer_tools_used - before.samples.transfer_tools_used
                        == int(committed)
                    )
                    if committed:
                        assert after.ledger.cost - before.ledger.cost == pytest.approx(0.005)
                    else:
                        assert after.samples == before.samples
                        assert after.phases == before.phases
                if step == 14:
                    assert len(after.samples.active_lineage) == 1
                    assert all(s.retired for s in after.samples.samples.values())
                if step == 15:
                    assert after.samples == before.samples
                    assert after.metadata["last_energy_transition"]["phase_change_heat_J"] > 0
                assert not truncated
        assert terminated
        assert env.unwrapped._state.samples.containers_used == 2
        assert env.unwrapped._state.samples.transfer_tools_used == 4
        records = load_jsonl(path)
        replay = verify_records(records, tolerance=0)
        assert replay.verified, replay.to_dict()
        return {
            "seed": seed,
            "actions": rows,
            "replay": replay.to_dict(),
            "routing": env.unwrapped._state.samples.routing_summary(),
        }
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_merge_reload_and_process(tmp_path, seed):
    run_routing(tmp_path / "trajectory.jsonl", seed)


def test_capacity_active_chemistry_identity_and_retired_container_are_atomic():
    env = make_env()
    try:
        for action in cycle_actions()[1:4]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        assert env.step(create(1, 0.01))[-1]["transaction_status"] == "committed"
        for action in [
            transfer(0, 1, 0.1),
            create(1),
            create(0),
            create(2, float("nan")),
            create(50),
            transfer(0, 1, float("inf")),
        ]:
            before = env.unwrapped._state
            assert env.step(action)[-1]["transaction_status"] != "committed"
            after = env.unwrapped._state
            assert before.samples == after.samples
            assert before.phases == after.phases
        env.step({"operation": "quench"})
        assert env.step(transfer(0, 1))[-1]["transaction_status"] != "committed"
        assert env.step(transfer(0, 1, 0.2))[-1]["transaction_status"] == "committed"
        assert env.step(transfer(1, 0, mixing=1))[-1]["transaction_status"] == "committed"
        before = env.unwrapped._state
        assert env.step(transfer(0, 1, 0.1))[-1]["transaction_status"] != "committed"
        assert env.unwrapped._state.samples == before.samples
    finally:
        env.close()


def test_routing_has_lossless_numeric_carrier_and_public_ids():
    codec = ActionCodec()
    for action in [create("container-01"), transfer("active", "container-02", 0.25, 1)]:
        canonical = codec.canonicalize(action)
        decoded = codec.decode_vector(codec.encode_vector(action))
        assert decoded.keys() == canonical.keys()
        for key in decoded:
            assert decoded[key] == (
                canonical[key] if key == "operation" else pytest.approx(canonical[key])
            )
    env = make_env()
    try:
        schema = action_schema(env, "transfer_material")
        field = next(f for f in schema["fields"] if f["field"] == "destination_container")
        assert field["choice_labels"]["1"] == "container-01"
        env.step(create(1))
        view = observation_view(env, "tool_json")
        assert view["sample_inventory"][0]["sample_id"] == "container-01"
        assert "contents" not in view["sample_inventory"][0]
        assert "species_amounts_mol" not in view["sample_inventory"][0]
        assert view["material_routing"]["containers_used"] == 1
    finally:
        env.close()


def test_replay_detects_changed_routing_provenance(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    run_routing(path, 0)
    records = deepcopy(load_jsonl(path))
    records[-1]["environment_outcome"]["material_routing"]["transfers"][0]["source"] = "fake"
    result = verify_records(records)
    assert not result.verified
    assert any("material_routing" in row["field"] for row in result.mismatches)


def test_stored_filtrate_reloads_with_its_original_charge_and_can_be_processed():
    env = make_env()
    try:
        for action in cycle_actions()[1:10]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        initial = env.unwrapped._state.species.initial_amounts_mol
        seed = env.unwrapped._state.samples.samples["filtrate-0001"].seed_target_mol
        for action in [create(1), transfer(0, 1), transfer("filtrate-0001", "active")]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        loaded = env.unwrapped._state
        assert loaded.species.initial_amounts_mol == initial
        assert (
            equipment_settings(loaded.equipment, "crystallizer")["dissolved_seed_target_mol"]
            == seed
        )
        assert loaded.samples.samples["filtrate-0001"].retired
        assert (
            env.step(
                {
                    "operation": "heat",
                    "target_temperature_K": 350,
                    "duration_s": 60,
                    "stirring_speed_rpm": 600,
                }
            )[-1]["transaction_status"]
            == "committed"
        )
        assert (
            env.step({"operation": "measure", "instrument": "hplc"})[-1]["transaction_status"]
            == "committed"
        )
        assert (
            env.unwrapped._state.samples.samples["container-01"]
            == loaded.samples.samples["container-01"]
        )
    finally:
        env.close()


def test_storage_and_adiabatic_reunion_conserve_sensible_heat_and_instrument_usage():
    env = make_env()
    try:
        for action in [*cycle_actions()[1:8], create(1), transfer(0, 1, 0.25)]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        stored = env.unwrapped._state.samples.samples["container-01"]
        assert (
            env.step(
                {
                    "operation": "heat",
                    "target_temperature_K": 380,
                    "duration_s": 900,
                    "stirring_speed_rpm": 600,
                }
            )[-1]["transaction_status"]
            == "committed"
        )
        assert (
            env.step({"operation": "measure", "instrument": "hplc"})[-1]["transaction_status"]
            == "committed"
        )
        before = env.unwrapped._state
        assert before.samples.samples["container-01"] == stored
        thermal_before = (
            before.volume_L * before.temperature_K + stored.volume_L * stored.temperature_K
        )
        assert env.step(transfer(1, 0, mixing=1))[-1]["transaction_status"] == "committed"
        after = env.unwrapped._state
        assert after.volume_L * after.temperature_K == pytest.approx(thermal_before, abs=1e-12)
        assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        assert equipment_settings(after.equipment, "instrument:hplc")["use_count"] == 1
        assert (
            env.step({"operation": "measure", "instrument": "hplc"})[-1]["transaction_status"]
            == "committed"
        )
        assert (
            equipment_settings(env.unwrapped._state.equipment, "instrument:hplc")["use_count"] == 2
        )
    finally:
        env.close()


def test_operation_mask_tracks_available_routes():
    env = make_env()
    try:
        assert "transfer_material" not in env.unwrapped.operation_validator.valid_operations(
            env.unwrapped._state
        )
        for action in [*cycle_actions()[1:6], create(1)]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        assert "transfer_material" in env.unwrapped.operation_validator.valid_operations(
            env.unwrapped._state
        )
    finally:
        env.close()


def test_reuniting_filtrates_does_not_double_original_charge_reference():
    env = make_env()
    try:
        for action in cycle_actions()[1:14]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        before = env.unwrapped._state
        initial = before.species.initial_amounts_mol
        material = total_inventory(before)
        for action in [
            create(1),
            transfer("filtrate-0001", 1),
            transfer("filtrate-0002", 1, mixing=1),
        ]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        sample = env.unwrapped._state.samples.samples["container-01"]
        assert sample.contents.species.initial_amounts_mol == pytest.approx(initial)
        assert len(sample.contents.reference_sources) == 1
        assert total_inventory(env.unwrapped._state) == pytest.approx(material)
    finally:
        env.close()


def test_campaign_sources_stay_distinct_across_batch_reset():
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        episode_mode_override="campaign",
        budget_override=80,
    )
    try:
        env.reset(seed=0)
        for action in [*cycle_actions()[1:], *cycle_actions()[1:10]]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        first = env.unwrapped._state.samples.samples["filtrate-0001"]
        second = env.unwrapped._state.samples.samples["filtrate-0003"]
        assert first.lineage != second.lineage
        expected = {s: n * 2 for s, n in first.contents.species.initial_amounts_mol.items()}
        for action in [
            create(1),
            transfer("filtrate-0001", 1),
            transfer("filtrate-0003", 1, mixing=1),
        ]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        combined = env.unwrapped._state.samples.samples["container-01"]
        assert len(combined.contents.reference_sources) == 2
        assert combined.contents.species.initial_amounts_mol == pytest.approx(expected)
    finally:
        env.close()
