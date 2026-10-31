"""Posthoc ledger validation must replay the action admitted by the real runtime."""

from copy import deepcopy

import gymnasium as gym
import pytest
from tests.fixtures.process_recipes import verify_resources

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.runtime.full_process_contract import FULL_PROCESS_FREE_RESEARCH_CONTRACT


def test_requested_integer_control_values_verify_against_real_environment_receipts():
    card = CampaignResourceCard(
        card_id="posthoc-action-canonicalization",
        operation_attempt_limit=12,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=1,
        stock_limits={},
    )
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-purification",
        seed=0,
        budget_override=12,
        episode_mode_override="campaign",
        campaign_resource_card=card,
        full_process_contract_id=FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    )
    actions = [
        {
            "operation": "configure_control",
            "atmosphere": 0,
            "control_interval_s": 10,
            "headspace_L": 0.03,
            "pressure_Pa": 101325,
            "ramp_rate_K_s": 0.5,
            "target_temperature_K": 350,
        },
        {"operation": "configure_control", "target_temperature_K": 351.0},
        {"operation": "invalid-operation"},
    ]
    records = []
    try:
        env.reset(seed=0)
        for action in actions:
            env.step(action)
            records.append(
                {
                    "action": deepcopy(action),
                    "campaign_resource_card": card.to_dict(),
                    "agent_view": {
                        "tool_json": {"campaign_state": deepcopy(env.unwrapped.campaign_state())}
                    },
                }
            )
        # Demonstrate the int/float distinction really occurred in production code.
        admitted = env.unwrapped.campaign_resource_snapshot()["events"][0]["action"]
        assert type(actions[0]["target_temperature_K"]) is int
        assert type(admitted["target_temperature_K"]) is float
    finally:
        env.close()
    unchanged = deepcopy(records)
    assert verify_resources(records) == {"checked_receipts": 3, "verified": True}
    assert records == unchanged
    tampered = deepcopy(records)
    tampered[0]["action"]["target_temperature_K"] = 355
    with pytest.raises(AssertionError):
        verify_resources(tampered)
