from __future__ import annotations

from copy import deepcopy

import pytest

from chemworld.agent_interface import action_schema
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation.gas import GasBoundary
from chemworld.physchem.gas_control import MAX_GAS_FLOW_MOL_S, advance_gas
from chemworld.runtime.control_program import control_settings, public_control


def config(**updates):
    return {
        "operation": "configure_control",
        "target_temperature_K": 350.0,
        "pressure_Pa": 150000.0,
        "atmosphere": "nitrogen",
        "ramp_rate_K_s": 0.25,
        "control_interval_s": 10.0,
        "headspace_L": 0.02,
    } | updates


def actions(kind="constant"):
    result = [
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": 0.015},
        config(),
    ]
    if kind == "stages":
        for temperature, atmosphere in ((335.0, "nitrogen"), (300.0, "argon")):
            result.append(
                {
                    "operation": "queue_control_stage",
                    "target_temperature_K": temperature,
                    "pressure_Pa": 150000.0,
                    "atmosphere": atmosphere,
                    "ramp_rate_K_s": 0.3,
                    "stage_duration_s": 120.0,
                }
            )
        result.append({"operation": "advance_control", "duration_s": 240.0})
    elif kind == "feedback":
        result += [
            {
                "operation": "set_control_feedback",
                "feedback_sensor": "temperature",
                "feedback_threshold": 310.0,
                "feedback_direction": "above",
                "feedback_response": "pause",
            },
            {"operation": "advance_control", "duration_s": 240.0},
            {
                "operation": "set_control_feedback",
                "feedback_sensor": "off",
                "feedback_threshold": 310.0,
                "feedback_direction": "above",
                "feedback_response": "pause",
            },
            {"operation": "resume_control"},
            {"operation": "advance_control", "duration_s": 60.0},
        ]
    else:
        result += [{"operation": "advance_control", "duration_s": 240.0}]
    return [
        *result,
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def step(env, action):
    result = env.step(action)
    info = result[-1]
    assert info["transaction_status"] == "committed", (
        action,
        info.get("rollback_reason"),
        {k: v for k, v in info.get("preconditions", {}).items() if not v},
    )
    assert env.constitution.check_state(env._state).passed
    settings = control_settings(env._state)
    if settings:
        gas = GasBoundary.from_dict(settings["gas"])
        assert gas.material_residual_mol <= 1e-10
        assert abs(gas.energy_residual_j) <= 1e-7
        assert env._state.pressure_Pa == gas.pressure_pa
    return result


def make_env(seed=0):
    env = ChemWorldEnv(task_id="reaction-to-assay", seed=seed, budget_override=80)
    env.reset(seed=seed)
    return env


@pytest.mark.parametrize("kind", ["constant", "stages", "feedback"])
@pytest.mark.parametrize("seed", [0, 1])
def test_control_lifecycle(kind, seed):
    env = make_env(seed)
    for action in actions(kind):
        _, _, terminated, truncated, _ = step(env, action)
    assert terminated and not truncated
    settings = control_settings(env._state)
    if kind == "stages":
        assert settings["stage_index"] == 2
        assert settings["atmosphere"] == 1
    if kind == "feedback":
        assert len(settings["events"]) == 1
        assert settings["events"][0]["temperature_K"] >= 310
        assert settings["events"][0]["time_s"] < 240
    assert settings["elapsed_s"] == env._state.ledger.time_s
    assert env._state.ledger.gas_pump_work_J > 0


@pytest.mark.parametrize("parts", [(240,), (120, 120), (37, 63, 140)])
def test_control_segmentation_conserves_physics_and_cost(parts):
    states = []
    for chunks in ((240,), parts):
        env = make_env()
        for action in actions()[:3]:
            step(env, action)
        for duration in chunks:
            step(env, {"operation": "advance_control", "duration_s": duration})
        states.append(env._state)
    left, right = states
    assert left.temperature_K == pytest.approx(right.temperature_K, abs=1e-4)
    assert left.species_amounts == pytest.approx(right.species_amounts, abs=1e-8)
    assert left.ledger.energy_jacket_J == pytest.approx(right.ledger.energy_jacket_J, rel=1e-5)
    assert left.ledger.cost == pytest.approx(right.ledger.cost, abs=2e-7)
    assert control_settings(left)["sensor_count"] == control_settings(right)["sensor_count"] == 24
    assert left.ledger.time_s == right.ledger.time_s == 240


def test_pause_and_resume_preserves_cursor_and_gas():
    env = make_env()
    for action in actions("stages")[:5]:
        step(env, action)
    step(env, {"operation": "advance_control", "duration_s": 60.0})
    step(env, {"operation": "pause_control"})
    before = env._state
    saved = deepcopy(control_settings(before))
    info = env.step({"operation": "advance_control", "duration_s": 30.0})[-1]
    assert info["transaction_status"] != "committed"
    assert control_settings(env._state) == saved
    assert env._state.species_amounts == before.species_amounts
    step(env, {"operation": "resume_control"})
    step(env, {"operation": "advance_control", "duration_s": 60.0})
    assert control_settings(env._state)["stage_index"] == 1
    assert control_settings(env._state)["elapsed_s"] == 120


def test_control_quantized_feedback_and_next_stage():
    env = make_env()
    for action in actions("stages")[:5]:
        step(env, action)
    step(
        env,
        {
            "operation": "set_control_feedback",
            "feedback_sensor": "temperature",
            "feedback_threshold": 300.0,
            "feedback_direction": "above",
            "feedback_response": "next_stage",
        },
    )
    step(env, {"operation": "advance_control", "duration_s": 240.0})
    settings = control_settings(env._state)
    assert settings["status"] == "paused"
    assert len(settings["events"]) == 2
    assert all(event["time_s"] % 10 == 0 for event in settings["events"])
    assert public_control(env._state)["sensor_model"]["temperature_resolution_K"] == 0.01


def test_gas_sealed_limit_and_finite_feed_balance():
    gas = GasBoundary.initialize(0.02, 300, 101325)
    sealed = advance_gas(
        gas,
        duration_s=60,
        final_temperature_K=330,
        pressure_setpoint_Pa=150000,
        atmosphere=0,
        purge_flow_mol_s=0,
        maximum_flow_mol_s=0,
    )
    assert sealed.amounts_mol == gas.amounts_mol
    assert sealed.pressure_pa == pytest.approx(gas.pressure_pa * 1.1)
    assert sealed.pump_work_J == 0
    assert abs(sealed.energy_residual_j) < 1e-10
    purged = advance_gas(
        gas, duration_s=60, final_temperature_K=300, pressure_setpoint_Pa=200000, atmosphere=1
    )
    assert 0 < purged.added_mol[2] <= 60 * MAX_GAS_FLOW_MOL_S + 1e-12
    assert purged.amounts_mol[1] < gas.amounts_mol[1]
    assert purged.pump_work_J > 0


def test_control_time_step_refinement():
    states = []
    for interval in (40, 20, 10, 5):
        env = make_env()
        for action in [*actions()[:2], config(control_interval_s=interval)]:
            step(env, action)
        step(env, {"operation": "advance_control", "duration_s": 240.0})
        states.append(env._state)
    reference = states[-1]
    errors = [abs(s.pressure_Pa - reference.pressure_Pa) for s in states[:-1]]
    assert errors[2] < errors[1] < errors[0]
    assert max(abs(s.temperature_K - reference.temperature_K) for s in states) < 1e-4


@pytest.mark.parametrize(
    "changes",
    [
        {"pressure_Pa": -1},
        {"headspace_L": 0},
        {"atmosphere": "hydrogen"},
        {"ramp_rate_K_s": float("nan")},
        {"control_interval_s": True},
    ],
)
def test_invalid_controller_payload_preserves_physics(changes):
    env = make_env()
    step(env, actions()[0])
    before = env._state
    info = env.step(config(**changes))[-1]
    assert info["transaction_status"] != "committed"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.equipment == before.equipment


def test_controller_gym_encoding_and_schema():
    env = make_env()
    event = config()
    decoded = env.action_codec.decode_vector(env.action_codec.encode_vector(event))
    assert decoded["atmosphere"] == 0
    assert decoded["headspace_L"] == pytest.approx(0.02)
    assert decoded["operation"] == "configure_control"
    schema = action_schema(env, "configure_control")
    assert {f["field"] for f in schema["fields"]} == set(event) - {"operation"}


def test_event_history_recovers_paused_program_and_continues(tmp_path):
    import json

    prefix = [
        *actions("stages")[:5],
        {"operation": "advance_control", "duration_s": 57.0},
        {"operation": "pause_control"},
    ]
    original = make_env()
    for action in prefix:
        step(original, action)
    history = tmp_path / "controller-history.json"
    history.write_text(json.dumps(prefix), encoding="utf-8")
    recovered = make_env()
    for action in json.loads(history.read_text(encoding="utf-8")):
        step(recovered, action)
    assert recovered._state.to_dict() == original._state.to_dict()
    for action in (
        {"operation": "resume_control"},
        {"operation": "advance_control", "duration_s": 183.0},
    ):
        step(original, action)
        step(recovered, action)
    assert recovered._state.to_dict() == original._state.to_dict()
    assert control_settings(recovered._state)["stage_index"] == 2
    assert control_settings(recovered._state)["sensor_count"] == 24


def test_numeric_gym_payload_and_quenched_thermal_control():
    import numpy as np

    env = make_env()
    for action in actions()[:2]:
        step(env, action)
    step(env, {"operation": "quench"})
    before = env._state.species_amounts
    event = config()
    event["atmosphere"] = 0
    for key, value in list(event.items()):
        if isinstance(value, float):
            event[key] = np.asarray([value], dtype=np.float32)
    step(env, event)
    step(env, {"operation": "advance_control", "duration_s": np.asarray([40.0], dtype=np.float32)})
    assert env._state.species_amounts == before
    assert env._state.temperature_K > 298.15


def test_controller_state_is_local_to_component_vessel():
    components = [
        {"id": f"{v}-{kind}", "kind": kind, "vessel": v}
        for v in ("a", "b")
        for kind in ("reaction", "thermal", "observation")
    ]
    request = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "control-two",
        "components": components,
        "task": {"budget": 80},
    }
    env = ChemWorldEnv(composition=request)
    env.reset()
    for action in actions()[:3]:
        step(env, action)
    step(env, {"operation": "advance_control", "duration_s": 30.0})
    saved = control_settings(env._state)
    step(env, {"operation": "select_vessel", "vessel": "b"})
    assert public_control(env._state) is None
    for action in [*actions()[:2], config(pressure_Pa=110000.0, atmosphere="argon")]:
        step(env, action)
    step(env, {"operation": "advance_control", "duration_s": 20.0})
    step(env, {"operation": "select_vessel", "vessel": "a"})
    assert control_settings(env._state) == saved
    assert env._state.vessel_elapsed_s == {"a": 30.0, "b": 20.0}


def test_resource_rejection_keeps_controller_cursor_unchanged():
    from chemworld.campaign_resources import CampaignResourceCard

    card = CampaignResourceCard(
        card_id="control-resource",
        operation_attempt_limit=20,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=2,
        stock_limits={"solvent_L": 0.1, "reagent_mol": 0.1},
        process_time_limit_s=40.0,
    )
    env = ChemWorldEnv(task_id="reaction-to-assay", budget_override=20, campaign_resource_card=card)
    env.reset()
    for action in actions()[:3]:
        step(env, action)
    saved = deepcopy(control_settings(env._state))
    before = env._state.species_amounts
    info = env.step({"operation": "advance_control", "duration_s": 60.0})[-1]
    assert info["transaction_status"] == "campaign_resource_rejected"
    assert control_settings(env._state) == saved
    assert env._state.species_amounts == before
    step(env, {"operation": "advance_control", "duration_s": 30.0})
    assert env._state.ledger.time_s == 30


def test_stage_can_be_added_after_a_setpoint_hold():
    env = make_env()
    for action in actions()[:3]:
        step(env, action)
    step(env, {"operation": "advance_control", "duration_s": 60.0})
    stage = actions("stages")[3]
    step(env, stage)
    step(env, {"operation": "advance_control", "duration_s": 120.0})
    assert control_settings(env._state)["stage_index"] == 1
