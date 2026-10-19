from __future__ import annotations

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.runtime.continuous_streams import settings
from chemworld.runtime.solvent_transport import total_solvents


def request(kind="cstr"):
    components = [
        {"id": v + "-" + k, "kind": k, "vessel": v}
        for v in ("tank", "feed-a", "feed-b", "collector")
        for k in ("reaction", "thermal", "observation")
    ]
    components.append({"id": "flow", "kind": "continuous_flow", "vessel": "tank"})
    if kind == "electrochemical":
        components.append({"id": "cell", "kind": "electrochemistry", "vessel": "tank"})
    connections = [
        {
            "id": name,
            "source": {"component": source + "-reaction", "port": "out"},
            "target": {"component": target + "-reaction", "port": "in"},
            "unit": "mol",
        }
        for name, source, target in (
            ("a-in", "feed-a", "tank"),
            ("b-in", "feed-b", "tank"),
            ("out", "tank", "collector"),
        )
    ]
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "streams-" + kind,
        "components": components,
        "connections": connections,
        "task": {"budget": 100},
    }


def advance(duration=60, current=0):
    return {
        "operation": "advance_flow",
        "duration_s": duration,
        "target_temperature_K": 320.0,
        "current_mA": current,
    }


def valve(connection, rate):
    return {"operation": "set_flow_stream", "connection": connection, "flow_rate_mL_min": rate}


def actions(kind="cstr"):
    result = []
    for vessel, volume, amount, solvent in (
        ("tank", 0.01, 0.001, 0),
        ("feed-a", 0.03, 0.015, 0),
        ("feed-b", 0.03, 0.006, 1),
    ):
        result += [
            {"operation": "select_vessel", "vessel": vessel},
            {"operation": "add_solvent", "volume_L": volume, "solvent": solvent},
            {"operation": "add_reagent", "amount_mol": amount},
        ]
    result += [{"operation": "select_vessel", "vessel": "tank"}, valve("a-in", 1), valve("b-in", 1)]
    if kind != "semibatch":
        result.append(valve("out", 2))
    result.append(advance(120, 50 if kind == "electrochemical" else 0))
    if kind == "cstr":
        result += [valve("a-in", 0), valve("b-in", 2), advance(120)]
    if kind != "semibatch":
        result.append({"operation": "select_vessel", "vessel": "collector"})
    return [
        *result,
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def make_env(kind="cstr", seed=0):
    env = ChemWorldEnv(composition=request(kind), seed=seed)
    env.reset(seed=seed)
    return env


def step(env, action):
    before = env._state
    result = env.step(action)
    info = result[-1]
    assert info["transaction_status"] == "committed", (
        action,
        info.get("rollback_reason"),
        {k: v for k, v in info.get("preconditions", {}).items() if not v},
    )
    assert env.constitution.check_state(env._state).passed
    if action["operation"] == "advance_flow":
        assert env.constitution.check_material_conservation(before, env._state).passed
        assert total_solvents(before).volumes_L == pytest.approx(
            total_solvents(env._state).volumes_L, abs=1e-10
        )
        diagnostics = settings(env._state)["last_step"]["diagnostics"]
        assert abs(diagnostics["energy_residual_J"]) < 1e-6
        assert abs(diagnostics["charge_residual_C"]) < 1e-8
    return result


@pytest.mark.parametrize("kind", ["semibatch", "cstr", "electrochemical"])
@pytest.mark.parametrize("seed", [0, 1])
def test_lifecycle(kind, seed):
    env = make_env(kind, seed)
    for action in actions(kind):
        _, _, terminal, truncated, _ = step(env, action)
    assert terminal and not truncated


@pytest.mark.parametrize("kind", ["cstr", "electrochemical"])
def test_call_segmentation(kind):
    states = []
    for chunks in ((120,), (37, 83)):
        env = make_env(kind)
        for action in actions(kind)[:13]:
            step(env, action)
        for duration in chunks:
            step(env, advance(duration, 50 if kind == "electrochemical" else 0))
        states.append(env._state)
    a, b = states
    assert a.species_amounts == pytest.approx(b.species_amounts, abs=1e-8)
    assert a.temperature_K == pytest.approx(b.temperature_K, abs=1e-4)
    assert a.ledger.cost == pytest.approx(b.ledger.cost, abs=1e-7)
    assert a.inactive_vessels["collector"].species_amounts == pytest.approx(
        b.inactive_vessels["collector"].species_amounts, abs=1e-8
    )


def test_exhaustion_rolls_back():
    env = make_env()
    for action in actions()[:13]:
        step(env, action)
    before = env._state
    result = env.step(advance(5000))[-1]
    assert result["transaction_status"] != "committed"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.inactive_vessels == before.inactive_vessels
    assert settings(env._state) == settings(before)


def test_zero_flow_and_valve_switching_inventory():
    env = make_env()
    for action in actions()[:13]:
        step(env, action)
    for name in ("a-in", "b-in", "out"):
        step(env, valve(name, 0))
    before = env._state
    step(env, advance(30))
    assert env._state.volume_L == pytest.approx(before.volume_L)
    assert env._state.inactive_vessels == before.inactive_vessels
    assert settings(env._state)["last_step"]["residence_time_s"] is None
    step(env, valve("a-in", 1))
    step(env, valve("out", 1))
    step(env, advance(60))
    assert env._state.inactive_vessels["feed-a"].volume_L == pytest.approx(0.029)
    assert env._state.inactive_vessels["feed-b"].volume_L == pytest.approx(0.03)
    collector = env._state.inactive_vessels["collector"]
    assert collector.volume_L == pytest.approx(0.001)
    assert any("feed-a" in source for source in collector.samples.active_reference_sources)
    assert settings(env._state)["last_step"]["residence_time_s"] == pytest.approx(600)


def test_charge_tracks_product_and_cell_energy():
    from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL

    env = make_env("electrochemical")
    for action in actions("electrochemical")[:14]:
        step(env, action)
    state = env._state
    product = env.runtime.domain_services.species_view.primary_target_species
    amount = (
        state.species_amounts[product]
        + state.inactive_vessels["collector"].species_amounts[product]
    )
    led = settings(state)["totals"]
    assert led["charge_C"] == pytest.approx(6)
    assert amount == pytest.approx(6 * 0.9 * 0.9 / (2 * FARADAY_C_PER_MOL), abs=1e-10)
    assert led["electrical_work_J"] == pytest.approx(
        led["chemical_work_J"] + led["cell_heat_J"], abs=1e-8
    )
    assert led["cell_heat_J"] >= led["ohmic_heat_J"] > 0


def test_different_feed_temperature_has_explicit_enthalpy():
    env = make_env()
    for action in actions()[:9]:
        step(env, action)
    step(
        env,
        {
            "operation": "heat",
            "duration_s": 90,
            "target_temperature_K": 350,
            "stirring_speed_rpm": 600,
        },
    )
    for action in actions()[9:14]:
        step(env, action)
    assert settings(env._state)["last_step"]["diagnostics"]["inlet_enthalpy_J"] > 0


@pytest.mark.parametrize(
    "action",
    [valve("a-in", -1), valve("a-in", True), valve("missing", 1), advance(0), advance(60, 10)],
)
def test_invalid_payload_preserves_all_vessel_inventory(action):
    env = make_env()
    for item in actions()[:13]:
        step(env, item)
    before = env._state
    assert env.step(action)[-1]["transaction_status"] != "committed"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.inactive_vessels == before.inactive_vessels


def test_collector_capacity_rejection():
    env = make_env()
    for action in actions()[:13]:
        step(env, action)
    step(env, {"operation": "select_vessel", "vessel": "collector"})
    step(env, {"operation": "add_solvent", "volume_L": 0.075, "solvent": 0})
    step(env, {"operation": "add_solvent", "volume_L": 0.024, "solvent": 0})
    step(env, {"operation": "select_vessel", "vessel": "tank"})
    before = env._state
    assert env.step(advance(120))[-1]["transaction_status"] != "committed"
    assert env._state.inactive_vessels == before.inactive_vessels


def test_resource_rejection_and_gym_transport():
    import numpy as np

    from chemworld.campaign_resources import CampaignResourceCard

    card = CampaignResourceCard(
        card_id="stream-resource",
        operation_attempt_limit=100,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=2,
        stock_limits={"solvent_L": 0.1, "reagent_mol": 0.1},
        process_time_limit_s=40,
    )
    env = ChemWorldEnv(composition=request(), campaign_resource_card=card)
    env.reset()
    for action in actions()[:13]:
        step(env, action)
    before = env._state
    assert env.step(advance(60))[-1]["transaction_status"] == "campaign_resource_rejected"
    assert env._state.inactive_vessels == before.inactive_vessels
    event = advance(30)
    for key in ("duration_s", "target_temperature_K", "current_mA"):
        event[key] = np.asarray([event[key]], dtype=np.float32)
    step(env, event)


def test_incompatible_feed_stop_state_is_rejected():
    env = make_env()
    for action in actions()[:13]:
        step(env, action)
    step(env, {"operation": "select_vessel", "vessel": "feed-a"})
    step(env, {"operation": "quench"})
    step(env, {"operation": "select_vessel", "vessel": "tank"})
    before = env._state
    assert env.step(advance())[-1]["transaction_status"] != "committed"
    assert env._state.inactive_vessels == before.inactive_vessels


def test_two_incompatible_catalyst_feeds_are_rejected():
    env = make_env()
    for action in actions()[:13]:
        step(env, action)
    for name, catalyst in (("feed-a", 0), ("feed-b", 1)):
        step(env, {"operation": "select_vessel", "vessel": name})
        step(
            env, {"operation": "add_catalyst", "catalyst": catalyst, "catalyst_amount_mol": 0.0001}
        )
    step(env, {"operation": "select_vessel", "vessel": "tank"})
    before = env._state
    assert env.step(advance())[-1]["transaction_status"] != "committed"
    assert env._state.inactive_vessels == before.inactive_vessels
