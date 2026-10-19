from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import gymnasium as gym
import pytest

from chemworld.campaign_resources import CampaignResourceCard, CampaignResourceLedger
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1,
)


def _card(**overrides: Any) -> CampaignResourceCard:
    values: dict[str, Any] = {
        "card_id": "integration-card",
        "operation_attempt_limit": 40,
        "vessel_start_limit": 3,
        "final_assay_limit": 3,
        "nonfinal_instrument_use_limit": 4,
        "stock_limits": {
            "reagent_mol": 0.08,
            "solvent_L": 0.12,
        },
        "per_instrument_limits": {"uvvis": 2},
        "metadata": {
            "task_id": "electrochemical-conversion",
            "envelope_kind": "public-test-envelope",
        },
    }
    values.update(overrides)
    return CampaignResourceCard(**values)


def _make_electrochemical_env(
    card: CampaignResourceCard | dict[str, Any],
    *,
    budget: int = 40,
) -> Any:
    return gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=0,
        budget_override=budget,
        episode_mode_override="campaign",
        electrochemical_workflow_mode=(ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1),
        campaign_resource_card=card,
    )


@pytest.mark.parametrize("as_mapping", (False, True))
def test_env_accepts_card_or_mapping_and_keeps_public_views_bounded(
    as_mapping: bool,
) -> None:
    card = _card()
    supplied = card.to_dict() if as_mapping else card
    env = _make_electrochemical_env(supplied)
    try:
        _, task_info = env.reset(seed=0)
        task_resources = task_info["campaign_resources"]
        assert task_resources["card"]["card_id"] == card.card_id
        assert task_resources["state"]["operation_attempts"] == 0
        assert "events" not in task_resources
        assert task_resources["card"] == card.to_dict()
        serialized_task_info = json.dumps(task_info, sort_keys=True)
        assert "world_id" not in serialized_task_info
        assert "mechanism_hash" not in serialized_task_info
        evaluator_provenance = env.unwrapped.evaluator_provenance()
        assert evaluator_provenance["observation_seed"] == 0
        assert evaluator_provenance["campaign_resource_card_sha256"] == card.card_sha256

        _, _, _, _, info = env.step(
            {
                "operation": "add_solvent",
                "volume_L": 0.020,
                "solvent": 1,
            }
        )
        public_state = env.unwrapped.campaign_state()["campaign_resources"]
        for payload in (info["campaign_resources"], public_state):
            assert "events" not in payload
            assert "card" not in payload
            assert payload["latest_receipt"]["operation_committed"] is True
        full_snapshot = env.unwrapped.campaign_resource_snapshot()
        assert len(full_snapshot["events"]) == 1
        assert full_snapshot["card"]["metadata"]["envelope_kind"] == "public-test-envelope"
    finally:
        env.close()


def test_stock_rejection_charges_attempt_without_physical_mutation() -> None:
    card = _card(
        stock_limits={"solvent_L": 0.020, "reagent_mol": 0.08},
    )
    env = _make_electrochemical_env(card)
    try:
        env.reset(seed=0)
        _, _, _, _, first = env.step(
            {
                "operation": "add_solvent",
                "volume_L": 0.020,
                "solvent": 1,
            }
        )
        assert first["transaction_status"] == "committed"
        physical_before = {
            "volume_L": env.unwrapped._state.volume_L,
            "temperature_K": env.unwrapped._state.temperature_K,
            "pressure_Pa": env.unwrapped._state.pressure_Pa,
            "phase": env.unwrapped._state.phase,
            "species_amounts": deepcopy(env.unwrapped._state.species_amounts),
        }

        _, _, _, _, rejected = env.step(
            {
                "operation": "add_solvent",
                "volume_L": 0.001,
                "solvent": 1,
            }
        )
        physical_after = {
            "volume_L": env.unwrapped._state.volume_L,
            "temperature_K": env.unwrapped._state.temperature_K,
            "pressure_Pa": env.unwrapped._state.pressure_Pa,
            "phase": env.unwrapped._state.phase,
            "species_amounts": deepcopy(env.unwrapped._state.species_amounts),
        }

        assert rejected["transaction_status"] == ("campaign_resource_rejected")
        assert rejected["rollback_reason"] == "campaign_resource_rejected"
        assert rejected["campaign_resource_rejected"] is True
        assert rejected["campaign_resource_rejection_reasons"] == ["stock_limit:solvent_L"]
        assert rejected["preconditions"]["campaign_resources_available"] is False
        assert rejected["campaign_resource_preflight"]["attempt_charged"] is True
        assert rejected["campaign_resource_outcome_delta"]["stocks"] == {}
        assert physical_after == physical_before

        resource_state = rejected["campaign_resources"]["state"]
        assert resource_state["operation_attempts"] == 2
        assert resource_state["vessel_starts"] == 1
        assert resource_state["stocks_used"]["solvent_L"] == pytest.approx(0.020)
    finally:
        env.close()


def test_campaign_process_time_and_repeat_limits_are_hard_and_replayable() -> None:
    card = _card(
        process_time_limit_s=200.0,
        operation_repeat_limits={"electrolyze": 1},
    )
    restored = CampaignResourceCard.from_dict(card.to_dict())
    assert restored == card
    ledger = CampaignResourceLedger(card)
    action = {"operation": "electrolyze", "duration_s": 180.0}
    preflight = ledger.preflight("event-1", action)
    assert preflight.allowed is True
    assert preflight.proposed_delta.process_time_s == 180.0
    ledger.record_outcome(
        "event-1",
        action,
        {
            "transaction_status": "committed",
            "campaign_resource_report_delta": {"process_time_s": 180.0},
        },
    )
    reasons = ledger.preview_rejection_reasons({"operation": "electrolyze", "duration_s": 30.0})
    assert reasons == (
        "operation_repeat_limit:electrolyze",
        "process_time_limit",
    )
    state = ledger.snapshot()["state"]
    assert state["operation_committed_counts"] == {"electrolyze": 1}
    assert state["remaining"]["process_time_s"] == 20.0
    assert state["remaining"]["operation_repeats"] == {"electrolyze": 0}
    assert CampaignResourceLedger.from_snapshot(ledger.snapshot()).snapshot() == ledger.snapshot()


def test_public_duration_schema_tracks_remaining_campaign_process_time() -> None:
    env = _make_electrochemical_env(
        _card(process_time_limit_s=200.0),
        budget=10,
    )
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.020, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "set_potential",
                "potential_V": 1.0,
                "current_mA": 60.0,
                "electrolyte_profile": 1,
            },
        ):
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"

        ledger = env.unwrapped._campaign_resource_ledger
        consumed = {"operation": "electrolyze", "duration_s": 150.0}
        preflight = ledger.preflight("duration-schema-history", consumed)
        assert preflight.allowed is True
        ledger.record_outcome(
            "duration-schema-history",
            consumed,
            {
                "transaction_status": "committed",
                "campaign_resource_report_delta": {"process_time_s": 150.0},
            },
        )

        schema = env.unwrapped.action_schema("electrolyze")
        duration = next(
            field for field in schema["fields"] if field["field"] == "duration_s"
        )
        assert duration["bounds"] == {"low": 1.0, "high": 50.0}
        assert duration["state_dependent_bounds"] is True
        assert duration["resource_limited"] is True
        accepted = env.unwrapped.validate_action(
            {"operation": "electrolyze", "duration_s": 50.0}
        )
        rejected = env.unwrapped.validate_action(
            {"operation": "electrolyze", "duration_s": 50.001}
        )
        assert accepted["valid"] is True
        assert rejected["valid"] is False
        assert "campaign_resource:process_time_limit" in rejected["invalid_reasons"]
    finally:
        env.close()


def test_public_duration_schema_preserves_protected_closeout_time() -> None:
    card = _card(
        operation_attempt_limit=10,
        process_time_limit_s=100.0,
        metadata={
            "task_id": "electrochemical-conversion",
            "process_time_policy": {"protected_reserve_s": 20.0},
            "closeout_policy": {
                "policy": "protected_closeout_reserve_enforced",
                "planned_batches": 2,
                "final_assay_path_operations_per_batch": 2,
                "final_assay_path_total_operation_reserve": 4,
                "discard_path_operations_per_batch": 1,
                "discard_path_total_operation_reserve": 2,
                "allowed_operation_classes": [
                    "discard_batch",
                    "final_assay",
                    "quench",
                    "terminate",
                    "transfer",
                ],
            },
        },
    )
    env = _make_electrochemical_env(card, budget=10)
    try:
        env.reset(seed=0)
        schema = env.unwrapped.action_schema("electrolyze")
        duration = next(
            field for field in schema["fields"] if field["field"] == "duration_s"
        )
        assert duration["bounds"] == {"low": 1.0, "high": 80.0}
        assert duration["state_dependent_bounds"] is True
        assert duration["resource_limited"] is True
    finally:
        env.close()


def test_crystallization_implicit_quench_and_filter_time_match_reservations() -> None:
    card = CampaignResourceCard(
        card_id="crystallization-implicit-time-test",
        operation_attempt_limit=12,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=2,
        stock_limits={
            "reagent_mol": 0.04,
            "solvent_L": 0.08,
            "catalyst_mol": 0.01,
            "seed_g": 0.05,
        },
        process_time_limit_s=5000.0,
        implicit_operation_time_s={"filter_crystals": 480.0, "quench": 120.0},
        operation_repeat_limits={"filter_crystals": 1, "quench": 1},
    )
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        budget_override=12,
        episode_mode_override="campaign",
        campaign_resource_card=card,
    )
    try:
        env.reset(seed=0)
        actions = (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.00035,
                "catalyst": 1,
            },
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "quench"},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "seed_crystals", "seed_mass_g": 0.006},
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1800.0,
            },
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "filter_crystals"},
        )
        receipts: dict[str, dict[str, Any]] = {}
        for action in actions:
            _obs, _reward, _terminated, _truncated, info = env.step(action)
            assert info["transaction_status"] == "committed"
            receipts[str(action["operation"])] = info["campaign_resources"][
                "latest_receipt"
            ]

        quench = receipts["quench"]
        assert quench["preflight"]["proposed_delta"]["report_only"][
            "process_time_s"
        ] == 120.0
        assert 0.0 < quench["outcome_delta"]["report_only"]["process_time_s"] <= 120.0
        filtering = receipts["filter_crystals"]
        assert filtering["preflight"]["proposed_delta"]["report_only"][
            "process_time_s"
        ] == 480.0
        assert filtering["outcome_delta"]["report_only"]["process_time_s"] == 480.0
    finally:
        env.close()


def test_non_electrochemical_campaign_can_discard_unassayable_started_batch() -> None:
    card = CampaignResourceCard(
        card_id="crystallization-discard-recovery-test",
        operation_attempt_limit=6,
        vessel_start_limit=2,
        final_assay_limit=2,
        nonfinal_instrument_use_limit=0,
        stock_limits={"catalyst_mol": 0.01},
    )
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        budget_override=6,
        episode_mode_override="campaign",
        campaign_resource_card=card,
    )
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, started = env.step(
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.00035,
                "catalyst": 1,
            }
        )
        assert started["transaction_status"] == "committed"
        assert terminated is False
        assert truncated is False
        assert "discard_batch" in {
            action["operation"] for action in env.unwrapped.available_actions()
        }

        _, _, terminated, truncated, rejected = env.step({"operation": "terminate"})
        assert rejected["transaction_status"] == "rolled_back"
        assert rejected["rollback_reason"] == "precondition_failed"
        assert rejected["preconditions"]["final_assay_sample_available"] is False
        assert rejected["preconditions"][
            "flagship_crystallization_requires_isolated_crystals"
        ] is False
        assert terminated is False
        assert truncated is False
        assert "discard_batch" in {
            action["operation"] for action in env.unwrapped.available_actions()
        }

        _, _, terminated, truncated, discarded = env.step(
            {"operation": "discard_batch", "reason": "no assayable sample"}
        )
        assert discarded["transaction_status"] == "committed"
        assert discarded["experiment_ended"] is True
        assert discarded["experiment_completed"] is False
        assert discarded["batch_discarded"] is True
        assert discarded["next_experiment_ready"] is True
        assert terminated is False
        assert truncated is False
        assert "add_solvent" in {
            action["operation"] for action in env.unwrapped.available_actions()
        }
    finally:
        env.close()


def test_resource_ledger_narrows_stock_affordances_and_validation() -> None:
    env = _make_electrochemical_env(_card(stock_limits={"solvent_L": 0.020, "reagent_mol": 0.08}))
    try:
        env.reset(seed=0)
        initial = env.unwrapped.available_actions()
        solvent = next(item for item in initial if item["operation"] == "add_solvent")
        volume_field = next(
            field for field in solvent["schema"]["fields"] if field["field"] == "volume_L"
        )
        assert volume_field["bounds"]["high"] == pytest.approx(0.020)

        env.step({"operation": "add_solvent", "volume_L": 0.020, "solvent": 1})
        assert "add_solvent" not in {
            item["operation"] for item in env.unwrapped.available_actions()
        }
        validation = env.unwrapped.validate_action(
            {"operation": "add_solvent", "volume_L": 0.001, "solvent": 1}
        )
        assert validation["valid"] is False
        assert validation["preconditions"]["campaign_resources_available"] is False
        assert validation["invalid_reasons"] == [
            "campaign_resources_available",
            "campaign_resource:stock_limit:solvent_L",
        ]
    finally:
        env.close()


def test_resource_ledger_filters_instrument_choices_and_terminal_tokens() -> None:
    env = _make_electrochemical_env(
        _card(
            nonfinal_instrument_use_limit=1,
            per_instrument_limits={"uvvis": 0, "ph_meter": 1},
        )
    )
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.020, "solvent": 1})
        schema = env.unwrapped.action_schema("measure")
        instrument_field = next(
            field for field in schema["fields"] if field["field"] == "instrument"
        )
        assert instrument_field["choices"] == ["ph_meter"]

        env.step({"operation": "measure", "instrument": "ph_meter"})
        assert "measure" not in {item["operation"] for item in env.unwrapped.available_actions()}
        invalid = env.unwrapped.available_actions(include_invalid=True)
        measure = next(item for item in invalid if item["operation"] == "measure")
        assert "campaign_resource:nonfinal_instrument_use_limit" in measure["invalid_reasons"]
    finally:
        env.close()


def test_resource_ledger_exhausted_attempts_exposes_no_executable_actions() -> None:
    env = _make_electrochemical_env(_card(operation_attempt_limit=1), budget=5)
    try:
        env.reset(seed=0)
        env.step({"operation": "not-a-real-operation"})
        assert env.unwrapped.available_actions() == []
        invalid = env.unwrapped.available_actions(include_invalid=True)
        assert invalid
        assert all(
            "campaign_resource:operation_attempt_limit" in item["invalid_reasons"]
            for item in invalid
        )
    finally:
        env.close()


def test_repeat_exhaustion_is_not_advertised_and_reports_cannot_complete() -> None:
    env = _make_electrochemical_env(
        _card(
            process_time_limit_s=1_000.0,
            operation_repeat_limits={"electrolyze": 1},
        ),
        budget=12,
    )
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.020, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "set_potential",
                "potential_V": 1.0,
                "current_mA": 60.0,
                "electrolyte_profile": 1,
            },
            {"operation": "electrolyze", "duration_s": 60.0},
        ):
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"

        assert "electrolyze" not in {
            item["operation"] for item in env.unwrapped.available_actions()
        }
        blocked = env.unwrapped.resource_blocked_actions()
        electrolyze = next(
            item for item in blocked if item["operation"] == "electrolyze"
        )
        assert electrolyze["status"] == "cannot_complete"
        assert electrolyze["reason_codes"] == [
            "operation_repeat_limit:electrolyze"
        ]
        assert "cannot be completed" in electrolyze["message"]

        _, _, _, _, rejected = env.step(
            {"operation": "electrolyze", "duration_s": 60.0}
        )
        assert rejected["transaction_status"] == "campaign_resource_rejected"
        assert rejected["campaign_resource_rejection_reasons"] == [
            "operation_repeat_limit:electrolyze"
        ]
        assert rejected["error_message"].startswith(
            "Operation cannot be completed with the remaining campaign resources"
        )
        report = env.unwrapped.observation_view("lab_report")
        assert report["status"] == "cannot_complete"
        assert report["failure_summary"]["resource_rejection_reasons"] == [
            "operation_repeat_limit:electrolyze"
        ]
        assert "cannot be completed" in report["text"]
    finally:
        env.close()


def test_public_lifecycle_reserve_is_advisory_and_tracks_closeout_feasibility() -> None:
    env = _make_electrochemical_env(
        _card(
            operation_attempt_limit=28,
            vessel_start_limit=2,
            final_assay_limit=2,
            nonfinal_instrument_use_limit=6,
            stock_limits={"reagent_mol": 0.16, "solvent_L": 0.32},
        ),
        budget=28,
    )
    try:
        env.reset(seed=0)
        initial = env.unwrapped.campaign_state()["campaign_resources"]["lifecycle_reserve"]
        assert initial["policy"] == ("advisory_only_agent_controlled_no_hidden_allocation")
        assert initial["current_batch"]["minimum_operations_to_final_assay"] == 6
        assert initial["future_unstarted_batches"] == 1
        assert initial["minimum_future_batch_operation_reserve"]["for_final_assays"] == 6
        assert (
            initial["recommended_remaining_attempt_floor"]["to_final_assay_all_planned_batches"]
            == 12
        )
        assert initial["discretionary_attempts_before_final_assay_floor"] == 16

        env.step({"operation": "add_solvent", "volume_L": 0.020, "solvent": 1})
        after_solvent = env.unwrapped.campaign_state()["campaign_resources"]["lifecycle_reserve"]
        assert after_solvent["current_batch"]["minimum_operations_to_final_assay"] == 5
        assert (
            after_solvent["recommended_remaining_attempt_floor"][
                "to_final_assay_all_planned_batches"
            ]
            == 11
        )
        assert after_solvent["discretionary_attempts_before_final_assay_floor"] == 16
    finally:
        env.close()


def test_invalid_env_action_uses_attempt_but_no_physical_resources() -> None:
    env = _make_electrochemical_env(_card())
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step({"operation": "not-a-real-operation", "amount_mol": 0.04})
        state = info["campaign_resources"]["state"]
        assert info["transaction_status"] == "validation_failed"
        assert state["operation_attempts"] == 1
        assert state["vessel_starts"] == 0
        assert state["final_assays"] == 0
        assert state["stocks_used"] == {}
        assert info["campaign_resource_outcome_delta"]["stocks"] == {}
    finally:
        env.close()


def test_vessels_final_assays_nonfinal_measurements_and_stocks_are_distinct() -> None:
    env = _make_electrochemical_env(_card())
    first_experiment = (
        {
            "operation": "add_solvent",
            "volume_L": 0.026,
            "solvent": 2,
        },
        {"operation": "add_reagent", "amount_mol": 0.010},
        {
            "operation": "set_potential",
            "potential_V": 1.15,
            "current_mA": 75.0,
            "electrolyte_profile": 2,
        },
        {"operation": "electrolyze", "duration_s": 120.0},
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    )
    try:
        env.reset(seed=0)
        final_info: dict[str, Any] = {}
        for action in first_experiment:
            _, _, _, _, final_info = env.step(action)
            assert final_info["transaction_status"] == "committed"

        first_state = final_info["campaign_resources"]["state"]
        assert first_state["vessel_starts"] == 1
        assert first_state["final_assays"] == 1
        assert first_state["nonfinal_instrument_uses"] == 1
        assert first_state["instrument_uses"] == {"uvvis": 1}
        assert first_state["stocks_used"] == pytest.approx(
            {"reagent_mol": 0.010, "solvent_L": 0.026}
        )
        assert final_info["campaign_resources"]["current_experiment"] == {
            "experiment_index": 1,
            "vessel_started": False,
        }
        assert final_info["campaign_resource_outcome_delta"]["final_assays"] == 1
        assert final_info["campaign_resource_outcome_delta"]["nonfinal_instrument_uses"] == 0

        _, _, _, _, second = env.step(
            {
                "operation": "add_solvent",
                "volume_L": 0.020,
                "solvent": 0,
            }
        )
        assert second["transaction_status"] == "committed"
        assert second["campaign_resources"]["state"]["vessel_starts"] == 2
        assert second["campaign_resource_outcome_delta"]["vessel_starts"] == 1
        assert second["campaign_resources"]["current_experiment"] == {
            "experiment_index": 1,
            "vessel_started": True,
        }
    finally:
        env.close()


def test_final_assay_with_no_remaining_vessel_does_not_create_phantom_batch() -> None:
    env = _make_electrochemical_env(
        _card(vessel_start_limit=1, final_assay_limit=1),
        budget=20,
    )
    try:
        env.reset(seed=0)
        actions = (
            {"operation": "add_solvent", "volume_L": 0.020, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "set_potential",
                "potential_V": 1.10,
                "current_mA": 70.0,
                "electrolyte_profile": 1,
            },
            {"operation": "electrolyze", "duration_s": 120.0},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        )
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert info["campaign_terminal"] is True
        assert info["next_experiment_ready"] is False
        assert info["campaign_resources"]["state"]["vessel_starts"] == 1
        assert env.unwrapped.available_actions() == []
    finally:
        env.close()


def test_agent_can_discard_started_batch_and_ledger_keeps_consumed_stock() -> None:
    env = _make_electrochemical_env(_card(), budget=10)
    try:
        env.reset(seed=0)
        _, _, _, _, first = env.step({"operation": "add_solvent", "volume_L": 0.020, "solvent": 1})
        assert (
            "discard_batch" not in {item["operation"] for item in env.unwrapped.available_actions()}
            or first["campaign_resources"]["current_experiment"]["vessel_started"] is True
        )
        _, _, terminated, truncated, info = env.step(
            {"operation": "discard_batch", "reason": "diagnostic branch abandoned"}
        )
        assert terminated is False
        assert truncated is False
        assert info["experiment_ended"] is True
        assert info["experiment_completed"] is False
        assert info["batch_discarded"] is True
        state = info["campaign_resources"]["state"]
        assert state["discarded_batches"] == 1
        assert state["final_assays"] == 0
        assert state["vessel_starts"] == 1
        assert state["stocks_used"] == pytest.approx({"solvent_L": 0.020})
        assert info["next_experiment_ready"] is True
    finally:
        env.close()


def test_agent_can_discard_batch_on_final_available_operation_attempt() -> None:
    env = _make_electrochemical_env(
        _card(
            operation_attempt_limit=2,
            vessel_start_limit=1,
            final_assay_limit=1,
        ),
        budget=2,
    )
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.020, "solvent": 1})
        _, _, terminated, truncated, info = env.step(
            {"operation": "discard_batch", "reason": "last-attempt closeout"}
        )

        assert terminated is True
        assert truncated is False
        assert info["transaction_status"] == "committed"
        assert info["experiment_ended"] is True
        assert info["batch_discarded"] is True
        assert info["campaign_resources"]["state"]["operation_attempts"] == 2
        assert info["campaign_resources"]["state"]["closed_batches"] == 1
    finally:
        env.close()


def test_public_resource_state_does_not_repeat_event_history() -> None:
    env = _make_electrochemical_env(_card(operation_attempt_limit=30))
    try:
        env.reset(seed=0)
        public_sizes: list[int] = []
        for _ in range(20):
            _, _, _, _, info = env.step({"operation": "not-a-real-operation"})
            resource_view = info["campaign_resources"]
            assert "events" not in resource_view
            assert "card" not in resource_view
            assert len(resource_view["latest_receipt"]) == 7
            public_sizes.append(len(json.dumps(resource_view, sort_keys=True)))

        full_snapshot = env.unwrapped.campaign_resource_snapshot()
        assert len(full_snapshot["events"]) == 20
        assert max(public_sizes) < 6_000
        assert max(public_sizes) - min(public_sizes) < 100
    finally:
        env.close()


def test_runner_reconciles_usage_learned_during_agent_update() -> None:
    class DeferredUsageAgent:
        name = "deferred-usage-test"

        def reset(self, task_info: dict[str, Any], seed: int) -> None:
            del seed
            self.task_info = task_info
            self.input_tokens = 0
            self.output_tokens = 0

        def act(self, history: list[Any]) -> dict[str, Any]:
            del history
            return {
                "operation": "add_solvent",
                "volume_L": 0.020,
                "solvent": 1,
            }

        def update(
            self,
            action: dict[str, Any],
            observation: dict[str, float | None],
            reward: float,
            info: dict[str, Any],
        ) -> None:
            del action, observation, reward, info
            self.input_tokens = 321
            self.output_tokens = 45

        def manifest(self) -> dict[str, Any]:
            return {}

        def method_resource_usage(self) -> dict[str, Any]:
            return {
                "schema_version": "chemworld-method-resource-usage-0.1",
                "accounting_complete": True,
                "usage_source": "test",
                "model_call_count": int(self.input_tokens > 0),
                "input_token_count": self.input_tokens,
                "output_token_count": self.output_tokens,
                "monetary_cost_usd": 0.0,
                "training_environment_step_count": 0,
                "cpu_time_s": 0.0,
                "gpu_time_s": 0.0,
                "model_provenance": {},
            }

    card = _card(
        operation_attempt_limit=2,
        vessel_start_limit=1,
        final_assay_limit=1,
    )
    agent = DeferredUsageAgent()
    records = run_agent(
        env_id="ChemWorld",
        agent=agent,
        world_split="public-test",
        budget=1,
        objective="balanced",
        seed=0,
        task_id="partition-discovery",
        budget_override=1,
        campaign_resource_card=card,
        method_resource_limits={
            "operation_limit": 1,
            "model_call_limit": 1,
            "input_token_limit": 500,
            "output_token_limit": 100,
        },
    )

    assert agent.task_info["campaign_resources"]["card"]["card_id"] == (card.card_id)
    usage = records[-1].method_resources["agent_usage"]
    assert usage["model_call_count"] == 1
    assert usage["input_token_count"] == 321
    assert usage["output_token_count"] == 45
    assert records[-1].method_resources["operation_count"] == 1
    assert records[-1].info["campaign_resources"]["state"]["operation_attempts"] == 1


def test_legacy_env_omits_campaign_resource_fields() -> None:
    env = gym.make(
        "ChemWorld",
        task_id="partition-discovery",
        seed=0,
        budget_override=1,
    )
    try:
        _, task_info = env.reset(seed=0)
        assert "campaign_resources" not in task_info
        _, _, _, _, info = env.step(
            {
                "operation": "add_solvent",
                "volume_L": 0.020,
                "solvent": 1,
            }
        )
        assert "campaign_resources" not in info
        assert "campaign_resources" not in env.unwrapped.campaign_state()
        assert env.unwrapped.campaign_resource_snapshot() is None
    finally:
        env.close()


def test_runner_closes_optional_agent_on_normal_exit() -> None:
    class ClosingAgent:
        name = "closing-agent"

        def reset(self, task_info: dict[str, Any], seed: int) -> None:
            del task_info, seed
            self.closed = False

        def act(self, history: list[Any]) -> dict[str, Any]:
            del history
            return {"operation": "not-a-real-operation"}

        def update(
            self,
            action: dict[str, Any],
            observation: dict[str, float | None],
            reward: float,
            info: dict[str, Any],
        ) -> None:
            del action, observation, reward, info

        def manifest(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

    agent = ClosingAgent()
    run_agent(
        env_id="ChemWorld",
        agent=agent,
        world_split="public-test",
        budget=1,
        objective="balanced",
        seed=0,
        task_id="partition-discovery",
        budget_override=1,
    )
    assert agent.closed is True


def test_agent_close_error_does_not_mask_primary_runner_error() -> None:
    class FailingAgent:
        name = "failing-close-agent"

        def reset(self, task_info: dict[str, Any], seed: int) -> None:
            del task_info, seed

        def act(self, history: list[Any]) -> dict[str, Any]:
            del history
            raise ValueError("primary decision failure")

        def update(
            self,
            action: dict[str, Any],
            observation: dict[str, float | None],
            reward: float,
            info: dict[str, Any],
        ) -> None:
            del action, observation, reward, info

        def manifest(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            raise RuntimeError("secondary close failure")

    with pytest.raises(ValueError, match="primary decision failure"):
        run_agent(
            env_id="ChemWorld",
            agent=FailingAgent(),
            world_split="public-test",
            budget=1,
            objective="balanced",
            seed=0,
            task_id="partition-discovery",
            budget_override=1,
        )


@pytest.mark.parametrize(
    "material_information",
    [
        {"mode": "anonymous_nominal_properties"},
        {
            "mode": "anonymous_misindexed_properties",
            "target_field": "solvent",
            "descriptor_permutation": [0, 3, 2, 1],
        },
    ],
)
def test_extended_autonomous_material_trajectory_exactly_replays(
    tmp_path: Any,
    material_information: dict[str, Any],
) -> None:
    class SequenceAgent:
        name = "autonomous-replay-sequence"

        def reset(self, task_info: dict[str, Any], seed: int) -> None:
            del task_info, seed
            self.index = 0
            self.actions = [
                {
                    "operation": "add_solvent",
                    "volume_L": 0.025,
                    "solvent": 1,
                },
                {"operation": "add_reagent", "amount_mol": 0.012},
                {
                    "operation": "set_potential",
                    "potential_V": 1.10,
                    "current_mA": 70.0,
                    "electrolyte_profile": 2,
                },
                {"operation": "electrolyze", "duration_s": 180.0},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]

        def act(self, history: list[Any]) -> dict[str, Any]:
            del history
            action = self.actions[self.index]
            self.index += 1
            return action

        def update(
            self,
            action: dict[str, Any],
            observation: dict[str, float | None],
            reward: float,
            info: dict[str, Any],
        ) -> None:
            del action, observation, reward, info

        def manifest(self) -> dict[str, Any]:
            return {}

    card = CampaignResourceCard(
        card_id="exact-replay-card",
        operation_attempt_limit=6,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=0,
        stock_limits={"reagent_mol": 0.04, "solvent_L": 0.08},
        metadata={"task_id": "electrochemical-conversion"},
    )
    trajectory = tmp_path / "trajectory.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=SequenceAgent(),
        world_split="public-test",
        budget=6,
        objective="balanced",
        seed=0,
        observation_seed=123456,
        task_id="electrochemical-conversion",
        output_path=trajectory,
        budget_override=6,
        episode_mode_override="campaign",
        campaign_resource_card=card,
        material_information=material_information,
        electrochemical_material_family_id="nominal-prior-latent-v2",
        electrochemical_workflow_mode="autonomous_open_v1",
        scoring_contract_id="electrochemical-s0-balanced-efficiency-v2",
        observation_noise_mode="keyed",
        observation_noise_namespace="campaign-resource-replay-test",
        method_resource_limits={
            "operation_limit": 6,
            "complete_experiment_limit": 1,
        },
    )

    records = load_jsonl(trajectory)
    replay = verify_records(records)
    assert replay.verified, replay.mismatches
    assert records[0]["material_information"] == {"mode": material_information["mode"]}
    assert records[0]["material_information_config"] == material_information
    assert records[0]["campaign_resource_card"] == card.to_dict()
    assert records[0]["campaign_resource_card_sha256"] == card.card_sha256
    assert records[0]["electrochemical_material_family_id"] == "nominal-prior-latent-v2"
    assert records[0]["observation_seed"] == 123456
