"""Settlement of one executed event into the session's resource budget scopes.

The session owns both ledgers. These functions retain no state and never execute
physics: campaign and batch limits consume the same physical outcome.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.campaign_resources import (
    CampaignResourceLedger,
    CampaignResourcePreflight,
    campaign_resource_event_id,
)
from chemworld.foundation.state import WorldState
from chemworld.runtime.scheduling import lab_instrument_reservation


@dataclass(frozen=True)
class ResourceReservation:
    event_id: str
    starts_vessel: bool
    campaign: CampaignResourcePreflight
    batch: CampaignResourcePreflight | None

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.campaign.rejection_reasons + (
            tuple("per_batch:" + reason for reason in self.batch.rejection_reasons)
            if self.batch is not None else ()
        )

    @property
    def rejected(self) -> bool:
        return not self.campaign.allowed or (
            self.batch is not None and not self.batch.allowed
        )


def reserve_resources(
    campaign: CampaignResourceLedger | None,
    batch: CampaignResourceLedger | None,
    action: Mapping[str, Any],
    *,
    campaign_id: str,
    operation_index: int,
    vessel_started: bool,
    state: WorldState,
    modules=None,
) -> ResourceReservation | None:
    if campaign is None:
        return None
    operation = action.get("operation")
    discard = isinstance(operation, str) and operation == "discard_batch"
    starts_vessel = not vessel_started and not discard
    event_id = campaign_resource_event_id(campaign_id, operation_index)
    instruments = lab_instrument_reservation(state, action, modules=modules)
    return ResourceReservation(
        event_id,
        starts_vessel,
        campaign.preflight(event_id, action, starts_vessel=starts_vessel,
                           instrument_reservation=instruments),
        batch.preflight(event_id, action, starts_vessel=starts_vessel,
                        instrument_reservation=instruments) if batch else None,
    )


def physical_resource_report(
    previous: WorldState,
    current: WorldState,
    observation_values: Mapping[str, float | None],
) -> dict[str, float]:
    """Measure actual experience, including partial stops and withheld readouts."""
    def positive_delta(before: float, after: float) -> float:
        delta = float(after) - float(before)
        return delta if isfinite(delta) and delta > 0.0 else 0.0

    risk = observation_values.get("safety_risk")
    observed_risk = 0.0
    if risk is not None and not isinstance(risk, bool):
        candidate = float(risk)
        if isfinite(candidate) and candidate >= 0.0:
            observed_risk = candidate
    return {
        "process_time_s": positive_delta(previous.ledger.time_s, current.ledger.time_s),
        "sample_consumed_L": positive_delta(
            previous.ledger.sample_consumed_L, current.ledger.sample_consumed_L,
        ),
        "physical_cost": positive_delta(previous.ledger.cost, current.ledger.cost),
        "accumulated_risk": positive_delta(previous.ledger.risk, current.ledger.risk),
        "observed_risk": observed_risk,
    }


def settle_resources(
    reservation: ResourceReservation,
    campaign: CampaignResourceLedger,
    batch: CampaignResourceLedger | None,
    action: Mapping[str, Any],
    *,
    committed: bool,
    transaction_status: str,
    stock_consumption: Mapping[str, float],
    previous: WorldState,
    current: WorldState,
    observation_values: Mapping[str, float | None],
):
    outcome = {
        "operation_committed": committed,
        "transaction_status": transaction_status,
        "material_stock_consumption": dict(stock_consumption),
        "campaign_resource_report_delta": physical_resource_report(
            previous, current, observation_values,
        ),
    }
    if (action.get("operation") in ("wait", "measure") and previous.lab_clock is not None
            and current.lab_clock is not None):
        before = {job["sequence"] for job in (
            *previous.lab_clock.running, *previous.lab_clock.completed,
        )}
        counts = {}
        for job in (*current.lab_clock.running, *current.lab_clock.completed):
            if job["sequence"] not in before and job["kind"] == "measure":
                counts[job["instrument"]] = counts.get(job["instrument"], 0) + 1
        if action.get("operation") == "measure" and counts and committed:
            instrument = action["instrument"]
            counts[instrument] = counts.get(instrument, 0)+1
        if action.get("operation") == "wait" or counts:
            outcome["instrument_uses"] = counts
    delta = campaign.record_outcome(
        reservation.event_id, action, outcome, starts_vessel=reservation.starts_vessel,
    )
    if batch is not None:
        batch.record_outcome(
            reservation.event_id, action, outcome, starts_vessel=reservation.starts_vessel,
        )
    return delta, {
        "event_id": reservation.event_id,
        "preflight": reservation.campaign.to_dict(),
        "outcome_delta": delta.to_dict(),
        "operation_committed": committed,
        "transaction_status": transaction_status,
        "rejected": reservation.rejected,
        "rejection_reasons": list(reservation.rejection_reasons),
    }
