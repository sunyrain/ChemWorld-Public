"""Immutable history, live-state checks and unchanged canonical hash boundaries."""

import copy

import pytest

from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
    canonical_json_sha256,
)


def ledger():
    return CampaignResourceLedger(CampaignResourceCard("prefix", 100, 0, 0, 0, {}))


def test_hash_matches_full_canonical_payload_at_every_phase_and_restore():
    book = ledger()
    action = {"operation": "pause_control"}
    outcome = {"transaction_status": "committed", "operation_committed": True}
    for i in range(12):
        book.preflight(str(i), action)
        assert book.bounded_state()["ledger_sha256"] == canonical_json_sha256(
            book._snapshot_payload()
        )
        book.record_outcome(str(i), action, outcome)
        assert book.bounded_state()["ledger_sha256"] == canonical_json_sha256(
            book._snapshot_payload()
        )
    snapshot = book.snapshot()
    restored = CampaignResourceLedger.from_snapshot(snapshot)
    for item in (book, restored):
        item.preflight("next", action)
        item.record_outcome("next", action, outcome)
    assert restored.snapshot() == book.snapshot()
    old = copy.deepcopy(snapshot)
    old.pop("history_root_sha256")
    old.pop("ledger_sha256")
    old["ledger_sha256"] = canonical_json_sha256(old)
    assert CampaignResourceLedger.from_snapshot(old).snapshot() == old


def test_receipts_cannot_be_changed_through_mapping_order_or_nested_values():
    book = ledger()
    book.preflight("one", {"operation": "pause_control"})
    with pytest.raises(TypeError):
        book._events["one"]["preflight"]["allowed"] = False
    with pytest.raises(TypeError):
        book._events["one"] = {}
    with pytest.raises(TypeError):
        book._event_order[0] = "other"
    recent = book.recent_events(1)
    recent[0]["preflight"]["allowed"] = False
    assert book.recent_events(1)[0]["preflight"]["allowed"]


@pytest.mark.parametrize("field,value", [("operation_attempts", 0), ("process_time_s", 8.0)])
def test_changed_live_counters_are_not_hidden_by_validated_history(field, value):
    book = ledger()
    action = {"operation": "pause_control"}
    book.preflight("one", action)
    book.record_outcome("one", action, {"transaction_status": "committed"})
    setattr(book, field, value)
    with pytest.raises(CampaignResourceIntegrityError, match="state mismatch"):
        book.preflight("two", action)


def test_pending_receipts_are_not_mistaken_for_a_completed_prefix():
    book = ledger()
    action = {"operation": "pause_control"}
    outcome = {"transaction_status": "committed"}
    book.preflight("one", action)
    book.preflight("two", action)
    book.record_outcome("two", action, outcome)
    assert book.verify_integrity()
    book.record_outcome("one", action, outcome)
    assert book.verify_integrity()
    assert book.operation_attempts == 2
    assert book.operation_committed_counts == {"pause_control": 2}
    assert book.bounded_state()["ledger_sha256"] == canonical_json_sha256(book._snapshot_payload())
    # The original chronological-vs-receipt-order root rule is retained; this
    # optimization does not silently invent a new interleaved snapshot format.
    with pytest.raises(CampaignResourceIntegrityError, match="history root mismatch"):
        book.snapshot()
