"""Behavioral safeguards for bounded, nonpersistent public-view reuse."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle, available_actions, observation_view
from chemworld.campaign_resources import CampaignResourceCard, CampaignResourceIntegrityError
from chemworld.foundation.state_helpers import equipment_setting_truth, equipment_settings
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.tasks import list_tasks


def test_projected_settings_remain_defensive_without_traversing_history():
    equipment = EquipmentLedger(
        {
            "crystallizer": EquipmentRecord(
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id="reactor",
                status="ready",
                settings={"dose": {"value": [1]}, "execution_history": [{"large": [2]}]},
            )
        }
    )
    copied = equipment_settings(equipment, "crystallizer", fields=("dose", "absent"))
    copied["dose"]["value"].append(3)
    assert equipment_settings(equipment, "crystallizer")["dose"] == {"value": [1]}
    assert "absent" not in copied

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("unrequested history must not be traversed")

    equipment.equipment["crystallizer"].settings["execution_history"] = [CannotCopy()]
    assert equipment_settings(equipment, "crystallizer", fields=("dose",)) == {
        "dose": {"value": [1]}
    }
    assert equipment_setting_truth(equipment, "crystallizer", "execution_history")
    assert not equipment_setting_truth(equipment, "missing", "execution_history")
    assert equipment_settings(None, "crystallizer", fields=("dose",)) == {}


def test_resource_snapshot_scope_refreshes_after_spend_and_exception():
    card = CampaignResourceCard(
        card_id="read-scope",
        operation_attempt_limit=40,
        vessel_start_limit=2,
        final_assay_limit=2,
        nonfinal_instrument_use_limit=2,
        stock_limits={"solvent_L": 0.028, "reagent_mol": 0.02},
    )
    env = gym.make(
        "ChemWorld",
        task_id="partition-discovery",
        seed=0,
        budget_override=40,
        episode_mode_override="campaign",
        campaign_resource_card=card,
    )
    try:
        obs, info = env.reset(seed=0)
        base = env.unwrapped
        ledger = base._campaign_resource_ledger
        with (
            patch.object(ledger, "snapshot", wraps=ledger.snapshot) as snapshots,
            patch.object(ledger, "bounded_state", wraps=ledger.bounded_state) as bounded,
        ):
            agent_view_bundle(env, obs, info)
            # Public reads use bounded state, never serialize/replay full history.
            assert snapshots.call_count == 0
            assert 0 < bounded.call_count <= 3
        obs, _, _, _, info = env.step({"operation": "add_solvent", "solvent": 2, "volume_L": 0.028})
        assert "add_solvent" not in {a["operation"] for a in available_actions(env)}
        # Full snapshots retain their integrity check; bounded reads do not audit
        # historical events. A failed read must still clear the request scope.
        attempts = ledger.operation_attempts
        ledger.operation_attempts += 1
        with pytest.raises(CampaignResourceIntegrityError):
            ledger.snapshot()
        ledger.operation_attempts = attempts
        with (
            patch.object(ledger, "bounded_state", side_effect=RuntimeError("view unavailable")),
            pytest.raises(RuntimeError, match="view unavailable"),
        ):
            available_actions(env)
        assert "add_solvent" not in {a["operation"] for a in available_actions(env)}
        obs, info = env.reset(seed=0)
        assert "add_solvent" in {a["operation"] for a in available_actions(env)}
        assert agent_view_bundle(env, obs, info)["tool_json"]["available_actions"] == (
            available_actions(env)
        )
    finally:
        env.close()


@pytest.mark.parametrize("campaign", [False, True])
def test_bundle_reuses_affordances_but_execution_revalidates(campaign):
    options = {}
    if campaign:
        options = {
            "episode_mode_override": "campaign",
            "campaign_resource_card": CampaignResourceCard(
                card_id="view-execution-boundary",
                operation_attempt_limit=40,
                vessel_start_limit=2,
                final_assay_limit=2,
                nonfinal_instrument_use_limit=2,
                stock_limits={},
            ),
        }
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0, **options)
    try:
        obs, info = env.reset(seed=0)
        validator = env.unwrapped.operation_validator
        with patch.object(
            validator, "operation_affordances", wraps=validator.operation_affordances
        ) as traversals:
            agent_view_bundle(env, obs, info)
            assert traversals.call_count == 1
            # Same state object still starts a fresh request.
            agent_view_bundle(env, obs, info)
            assert traversals.call_count == 2
        with patch.object(validator, "_preconditions", wraps=validator._preconditions) as checks:
            result = env.step(
                {"operation": "heat", "target_temperature_K": 360.0, "duration_s": 60.0}
            )[-1]
            assert checks.call_count > 0
        assert result["transaction_status"] == "rolled_back"
        assert not result["preconditions"]["has_volume"]
    finally:
        env.close()


@pytest.mark.parametrize("task", [task.task_id for task in list_tasks()])
def test_views_refresh_after_actions_config_changes_and_reset(task):
    env = gym.make("ChemWorld", task_id=task, seed=0)
    try:
        observation, info = env.reset(seed=0)
        base = env.unwrapped
        for action in (
            None,
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.01},
            {"operation": "unknown_operation"},
        ):
            if action is not None:
                observation, _, _, _, info = env.step(action)
            before = deepcopy(base._state.to_dict())
            with patch.object(
                base.operation_validator,
                "_preconditions",
                wraps=base.operation_validator._preconditions,
            ) as checks:
                bundle = agent_view_bundle(env, observation, info)
                # At most two full traversals, independent of allowed-op count.
                assert checks.call_count <= 2 * len(base.operation_validator.operation_types)
            assert before == base._state.to_dict()
            assert bundle == {
                mode: observation_view(env, mode, observation, info)
                for mode in ("rl", "tool_json", "lab_report")
            }
            report = deepcopy(bundle["tool_json"]["lab_report"])
            bundle["lab_report"]["next_action_hints"].clear()
            assert bundle["tool_json"]["lab_report"] == report
            if action and action["operation"] == "unknown_operation":
                assert info["transaction_status"] != "committed"

        # State identity is unchanged, but configuration changed. No identity cache.
        base.operation_validator.allowed_operations.remove("add_solvent")
        assert "add_solvent" not in {a["operation"] for a in available_actions(env)}
        base.operation_validator.allowed_operations.add("add_solvent")
        observation, info = env.reset(seed=0)
        assert agent_view_bundle(env, observation, info)["tool_json"]["available_actions"] == (
            available_actions(env)
        )
    finally:
        env.close()
