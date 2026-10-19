from __future__ import annotations

import copy

import pytest

from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
    campaign_resource_event_id,
    derive_campaign_resource_delta,
    generous_electrochemical_max_envelope_card,
)
from chemworld.eval.provenance import canonical_json_sha256


def _small_card(**overrides: object) -> CampaignResourceCard:
    payload: dict[str, object] = {
        "card_id": "test-card",
        "operation_attempt_limit": 6,
        "vessel_start_limit": 2,
        "final_assay_limit": 2,
        "nonfinal_instrument_use_limit": 2,
        "stock_limits": {"reagent_mol": 0.05, "solvent_L": 0.10},
        "per_instrument_limits": {"uvvis": 1},
    }
    payload.update(overrides)
    return CampaignResourceCard(**payload)


def _committed(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "transaction_status": "committed",
        "operation_committed": True,
    }
    payload.update(overrides)
    return payload


def _rejected() -> dict[str, object]:
    return {
        "transaction_status": "rejected",
        "operation_committed": False,
    }


def test_generous_electrochemical_card_is_frozen_and_roundtrips() -> None:
    card = generous_electrochemical_max_envelope_card()

    assert card.operation_attempt_limit == 84
    assert card.vessel_start_limit == 6
    assert card.final_assay_limit == 6
    assert card.nonfinal_instrument_use_limit == 18
    assert card.stock_limits["reagent_mol"] >= 0.040 * 6
    assert card.stock_limits["solvent_L"] >= 0.080 * 6
    assert CampaignResourceCard.from_dict(card.to_dict()).card_sha256 == card.card_sha256
    assert card.card_sha256 == generous_electrochemical_max_envelope_card().card_sha256

    with pytest.raises(TypeError):
        card.stock_limits["reagent_mol"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        card.metadata["task_id"] = "changed"  # type: ignore[index]


def test_implicit_operation_time_is_reserved_debited_and_roundtrips() -> None:
    card = _small_card(
        process_time_limit_s=600.0,
        implicit_operation_time_s={"filter_crystals": 480.0},
        operation_repeat_limits={"filter_crystals": 1},
    )
    restored = CampaignResourceCard.from_dict(card.to_dict())
    assert restored == card
    assert restored.implicit_operation_time_s == {"filter_crystals": 480.0}
    with pytest.raises(TypeError):
        restored.implicit_operation_time_s["filter_crystals"] = 1.0  # type: ignore[index]

    ledger = CampaignResourceLedger(restored)
    action = {"operation": "filter_crystals"}
    preflight = ledger.preflight("filter-1", action)
    assert preflight.allowed is True
    assert preflight.proposed_delta.process_time_s == 480.0
    delta = ledger.record_outcome(
        "filter-1",
        action,
        _committed(
            campaign_resource_report_delta={"process_time_s": 480.0}
        ),
    )
    assert delta.process_time_s == 480.0
    assert ledger.snapshot()["state"]["remaining"]["process_time_s"] == 120.0
    assert ledger.preview_rejection_reasons(action) == (
        "operation_repeat_limit:filter_crystals",
        "process_time_limit",
    )


def test_process_time_reservation_ignores_only_floating_point_tail() -> None:
    card = _small_card(
        process_time_limit_s=14_400.0,
        implicit_operation_time_s={"cool_crystallize": 14_400.0},
        operation_repeat_limits={"cool_crystallize": 1},
    )
    ledger = CampaignResourceLedger(card)
    action = {"operation": "cool_crystallize", "duration_s": 14_400.0}
    event_id = campaign_resource_event_id("float-tail", 1)
    preflight = ledger.preflight(event_id, action)
    assert preflight.allowed is True
    ledger.record_outcome(
        event_id,
        action,
        _committed(
            campaign_resource_report_delta={"process_time_s": 14_400.000000000004}
        ),
    )

    rejecting = CampaignResourceLedger(card)
    over_event_id = campaign_resource_event_id("real-overage", 1)
    rejecting.preflight(over_event_id, action)
    with pytest.raises(CampaignResourceIntegrityError, match="process-time reservation"):
        rejecting.record_outcome(
            over_event_id,
            action,
            _committed(
                campaign_resource_report_delta={"process_time_s": 14_400.0 + 2.0e-6}
            ),
        )


def test_implicit_operation_time_requires_a_process_time_limit() -> None:
    with pytest.raises(ValueError, match="requires process_time_limit_s"):
        _small_card(implicit_operation_time_s={"quench": 120.0})


def test_card_hash_detects_mutation_and_limits_are_validated() -> None:
    card = _small_card()
    tampered = card.to_dict()
    tampered["hard_limits"]["stocks"]["reagent_mol"] = 99.0
    with pytest.raises(CampaignResourceIntegrityError, match="card hash"):
        CampaignResourceCard.from_dict(tampered)

    with pytest.raises(ValueError, match="final_assay_limit"):
        _small_card(vessel_start_limit=1, final_assay_limit=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        _small_card(operation_attempt_limit=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _small_card(stock_limits={"reagent_mol": float("nan")})


def test_invalid_action_attempt_consumes_only_an_operation_slot() -> None:
    ledger = CampaignResourceLedger(_small_card())
    event_id = campaign_resource_event_id("campaign-a", 1)

    preflight = ledger.preflight(
        event_id,
        {"operation": "unknown", "malformed_field": 12},
        starts_vessel=True,
    )
    assert preflight.allowed is True
    assert preflight.attempt_charged is True
    ledger.record_outcome(
        event_id,
        {"operation": "unknown", "malformed_field": 12},
        _rejected(),
        starts_vessel=True,
    )

    state = ledger.snapshot()["state"]
    assert state["operation_attempts"] == 1
    assert state["vessel_starts"] == 0
    assert state["final_assays"] == 0
    assert state["stocks_used"] == {}


def test_preflight_rejects_stock_overrun_but_charges_the_attempt() -> None:
    ledger = CampaignResourceLedger(_small_card(stock_limits={"reagent_mol": 0.04}))
    first = campaign_resource_event_id("campaign-stock", 1)
    second = campaign_resource_event_id("campaign-stock", 2)
    action = {"operation": "add_reagent", "amount_mol": 0.04}

    assert ledger.preflight(first, action).allowed is True
    ledger.record_outcome(first, action, _committed())

    rejected = ledger.preflight(
        second,
        {"operation": "add_reagent", "amount_mol": 0.001},
    )
    assert rejected.allowed is False
    assert rejected.attempt_charged is True
    assert rejected.rejection_reasons == ("stock_limit:reagent_mol",)
    ledger.record_outcome(
        second,
        {"operation": "add_reagent", "amount_mol": 0.001},
        _rejected(),
    )

    state = ledger.snapshot()["state"]
    assert state["operation_attempts"] == 2
    assert state["stocks_used"] == {"reagent_mol": 0.04}
    assert state["remaining"]["stocks"]["reagent_mol"] == pytest.approx(0.0)


def test_resource_rejected_action_cannot_be_committed() -> None:
    ledger = CampaignResourceLedger(_small_card(stock_limits={"reagent_mol": 0.0}))
    event_id = campaign_resource_event_id("campaign-rejected", 1)
    action = {"operation": "add_reagent", "amount_mol": 0.01}

    assert ledger.preflight(event_id, action).allowed is False
    with pytest.raises(CampaignResourceIntegrityError, match="cannot have a committed"):
        ledger.record_outcome(event_id, action, _committed())


def test_instrument_vessel_and_final_assay_resources_are_separate() -> None:
    ledger = CampaignResourceLedger(_small_card())
    start_id = campaign_resource_event_id("campaign-instruments", 1)
    uvvis_id = campaign_resource_event_id("campaign-instruments", 2)
    extra_uvvis_id = campaign_resource_event_id("campaign-instruments", 3)
    final_id = campaign_resource_event_id("campaign-instruments", 4)

    start_action = {"operation": "add_solvent", "volume_L": 0.02, "solvent": 0}
    assert ledger.preflight(start_id, start_action, starts_vessel=True).allowed
    ledger.record_outcome(
        start_id,
        start_action,
        _committed(),
        starts_vessel=True,
    )

    uvvis = {"operation": "measure", "instrument": "uvvis"}
    assert ledger.preflight(uvvis_id, uvvis).allowed
    ledger.record_outcome(uvvis_id, uvvis, _committed(sample_consumed=0.0002))
    denied = ledger.preflight(extra_uvvis_id, uvvis)
    assert denied.allowed is False
    assert denied.rejection_reasons == ("per_instrument_limit:uvvis",)
    ledger.record_outcome(extra_uvvis_id, uvvis, _rejected())

    final = {"operation": "measure", "instrument": "final_assay"}
    assert ledger.preflight(final_id, final).allowed
    ledger.record_outcome(final_id, final, _committed(sample_consumed=0.0005))

    state = ledger.snapshot()["state"]
    assert state["vessel_starts"] == 1
    assert state["nonfinal_instrument_uses"] == 1
    assert state["instrument_uses"] == {"uvvis": 1}
    assert state["final_assays"] == 1
    assert state["report_only"]["sample_consumed_L"] == pytest.approx(0.0007)


def test_discard_batch_is_a_committed_lifecycle_debit_without_refunds() -> None:
    ledger = CampaignResourceLedger(_small_card())
    start_id = campaign_resource_event_id("campaign-discard", 1)
    discard_id = campaign_resource_event_id("campaign-discard", 2)
    start = {"operation": "add_solvent", "volume_L": 0.02, "solvent": 0}
    discard = {"operation": "discard_batch", "reason": "failed probe"}

    ledger.preflight(start_id, start, starts_vessel=True)
    ledger.record_outcome(start_id, start, _committed(), starts_vessel=True)
    ledger.preflight(discard_id, discard)
    delta = ledger.record_outcome(discard_id, discard, _committed())

    assert delta.discarded_batches == 1
    state = ledger.snapshot()["state"]
    assert state["vessel_starts"] == 1
    assert state["final_assays"] == 0
    assert state["discarded_batches"] == 1
    assert state["closed_batches"] == 1
    assert state["stocks_used"] == {"solvent_L": 0.02}
    assert CampaignResourceLedger.from_snapshot(ledger.snapshot()).snapshot() == ledger.snapshot()


def test_reporting_axes_accumulate_without_becoming_one_weighted_cost() -> None:
    ledger = CampaignResourceLedger(_small_card())
    event_id = campaign_resource_event_id("campaign-report", 1)
    action = {"operation": "electrolyze", "duration_s": 120.0}
    outcome = _committed(
        campaign_resource_report_delta={
            "process_time_s": 125.0,
            "sample_consumed_L": 0.0001,
            "physical_cost": 0.17,
            "accumulated_risk": 0.08,
            "observed_risk": 0.31,
        }
    )

    ledger.preflight(event_id, action)
    delta = ledger.record_outcome(event_id, action, outcome)
    state = ledger.snapshot()["state"]["report_only"]

    assert delta.process_time_s == 125.0
    assert state == {
        "process_time_s": 125.0,
        "sample_consumed_L": 0.0001,
        "physical_cost": 0.17,
        "accumulated_risk": 0.08,
        "peak_risk": 0.31,
    }


def test_preflight_and_outcome_replays_are_idempotent_and_conflicts_fail() -> None:
    ledger = CampaignResourceLedger(_small_card())
    event_id = campaign_resource_event_id("campaign-idempotent", 1)
    action = {"operation": "add_reagent", "amount_mol": 0.01}

    first = ledger.preflight(event_id, action)
    assert ledger.preflight(event_id, dict(action)) == first
    assert ledger.snapshot()["state"]["operation_attempts"] == 1
    with pytest.raises(CampaignResourceIntegrityError, match="different action"):
        ledger.preflight(
            event_id,
            {"operation": "add_reagent", "amount_mol": 0.02},
        )

    outcome = _committed(physical_cost_delta=0.1)
    first_delta = ledger.record_outcome(event_id, action, outcome)
    assert ledger.record_outcome(event_id, action, dict(outcome)) == first_delta
    assert ledger.snapshot()["state"]["stocks_used"]["reagent_mol"] == 0.01
    with pytest.raises(CampaignResourceIntegrityError, match="different resource outcome"):
        ledger.record_outcome(
            event_id,
            action,
            _committed(physical_cost_delta=0.2),
        )


def test_operation_limit_rejection_does_not_overdraw_the_hard_counter() -> None:
    ledger = CampaignResourceLedger(_small_card(operation_attempt_limit=1))
    first = campaign_resource_event_id("campaign-operations", 1)
    second = campaign_resource_event_id("campaign-operations", 2)

    ledger.preflight(first, {"operation": "invalid"})
    ledger.record_outcome(first, {"operation": "invalid"}, _rejected())
    denied = ledger.preflight(second, {"operation": "invalid"})
    ledger.record_outcome(second, {"operation": "invalid"}, _rejected())

    assert denied.allowed is False
    assert denied.attempt_charged is False
    assert denied.rejection_reasons == ("operation_attempt_limit",)
    assert ledger.snapshot()["state"]["operation_attempts"] == 1


def test_protected_closeout_reserve_rejects_exploration_without_spending_reserve() -> None:
    card = _small_card(
        operation_attempt_limit=5,
        process_time_limit_s=100.0,
        implicit_operation_time_s={"quench": 5.0},
        metadata={
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
    ledger = CampaignResourceLedger(card)
    first = {"operation": "electrolyze", "duration_s": 80.0}
    assert ledger.preflight("explore-1", first).allowed is True
    ledger.record_outcome(
        "explore-1",
        first,
        _committed(campaign_resource_report_delta={"process_time_s": 80.0}),
    )

    rejected = ledger.preflight(
        "explore-2", {"operation": "electrolyze", "duration_s": 1.0}
    )
    assert rejected.allowed is False
    assert rejected.attempt_charged is False
    assert rejected.rejection_reasons == (
        "protected_closeout_operation_reserve",
        "protected_closeout_process_time_reserve",
    )
    assert ledger.snapshot()["state"]["operation_attempts"] == 1

    closeout = {"operation": "quench"}
    accepted = ledger.preflight("closeout-1", closeout)
    assert accepted.allowed is True
    assert accepted.attempt_charged is True
    ledger.record_outcome(
        "closeout-1",
        closeout,
        _committed(campaign_resource_report_delta={"process_time_s": 5.0}),
    )
    state = ledger.snapshot()["state"]
    assert state["protected_closeout_reserve"]["process_time_consumed_s"] == 5.0
    assert state["report_only"]["protected_reserve_consumed_s"] == 5.0
    assert state["report_only"]["reserve_consumption_by_operation_class"] == {
        "quench": {"operation_attempts": 1, "process_time_s": 5.0}
    }
    assert CampaignResourceLedger.from_snapshot(ledger.snapshot()).snapshot() == (
        ledger.snapshot()
    )


def test_protected_closeout_contract_fails_when_formula_is_inconsistent() -> None:
    card = _small_card(
        operation_attempt_limit=6,
        process_time_limit_s=100.0,
        metadata={
            "process_time_policy": {"protected_reserve_s": 20.0},
            "closeout_policy": {
                "policy": "protected_closeout_reserve_enforced",
                "planned_batches": 2,
                "final_assay_path_operations_per_batch": 2,
                "final_assay_path_total_operation_reserve": 3,
                "discard_path_operations_per_batch": 1,
                "discard_path_total_operation_reserve": 2,
            },
        },
    )
    with pytest.raises(ValueError, match="operation reserve differs"):
        CampaignResourceLedger(card)


def test_snapshot_hash_roundtrip_and_integrity_checks() -> None:
    ledger = CampaignResourceLedger(_small_card())
    event_id = campaign_resource_event_id("campaign-snapshot", 1)
    action = {"operation": "add_solvent", "volume_L": 0.025, "solvent": 1}
    ledger.preflight(event_id, action, starts_vessel=True)
    ledger.record_outcome(event_id, action, _committed(), starts_vessel=True)

    snapshot = ledger.snapshot()
    restored = CampaignResourceLedger.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.verify_integrity() is True

    tampered = copy.deepcopy(snapshot)
    tampered["state"]["stocks_used"]["solvent_L"] = 0.09
    with pytest.raises(CampaignResourceIntegrityError, match="ledger hash"):
        CampaignResourceLedger.from_snapshot(tampered)

    rehashed = copy.deepcopy(tampered)
    rehashed.pop("ledger_sha256")
    tampered["ledger_sha256"] = canonical_json_sha256(rehashed)
    with pytest.raises(CampaignResourceIntegrityError, match="monotone state mismatch"):
        CampaignResourceLedger.from_snapshot(tampered)

    falsified_decision = copy.deepcopy(snapshot)
    falsified_decision["events"][0]["preflight"]["allowed"] = False
    falsified_decision.pop("ledger_sha256")
    falsified_decision["ledger_sha256"] = canonical_json_sha256(falsified_decision)
    with pytest.raises(
        CampaignResourceIntegrityError,
        match="preflight allowed decision mismatch",
    ):
        CampaignResourceLedger.from_snapshot(falsified_decision)


def test_delta_derivation_is_deterministic_and_rejects_negative_reports() -> None:
    action = {"operation": "add_solvent", "volume_L": 0.025, "solvent": 2}
    proposal = derive_campaign_resource_delta(action, starts_vessel=True)
    rejected = derive_campaign_resource_delta(
        action,
        _rejected(),
        starts_vessel=True,
    )

    assert proposal.stocks == {"solvent_L": 0.025}
    assert proposal.vessel_starts == 1
    assert rejected.operation_attempts == 1
    assert rejected.stocks == {}
    assert rejected.vessel_starts == 0
    assert campaign_resource_event_id("fixed", 2) == campaign_resource_event_id("fixed", 2)

    with pytest.raises(ValueError, match="finite and non-negative"):
        derive_campaign_resource_delta(
            {"operation": "electrolyze", "duration_s": 10.0},
            _committed(
                campaign_resource_report_delta={
                    "physical_cost": -1.0,
                }
            ),
        )
