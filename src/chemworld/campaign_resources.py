"""Campaign-persistent physical resource cards and an auditable ledger.

This module is deliberately independent of the environment and runner.  A
caller first reserves one operation attempt with :meth:`preflight`, then
records the environment outcome with :meth:`record_outcome`.  Physical stocks,
instrument uses, vessels, and final assays are debited only for committed
outcomes; a well-formed but environment-invalid proposal still consumes its
operation attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from chemworld.data.logging import to_builtin


def canonical_json_sha256(payload: Any) -> str:
    """Hash canonical JSON without importing the higher-level eval package."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CAMPAIGN_RESOURCE_CARD_VERSION = "chemworld-campaign-resource-card-0.1"
CAMPAIGN_RESOURCE_DELTA_VERSION = "chemworld-campaign-resource-delta-0.1"
CAMPAIGN_RESOURCE_LEDGER_VERSION = "chemworld-campaign-resource-ledger-0.1"
RESOURCE_FLOAT_ABS_TOL = 1.0e-9
PROTECTED_CLOSEOUT_POLICY = "protected_closeout_reserve_enforced"
DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS = frozenset(
    {"discard_batch", "final_assay", "quench", "terminate", "transfer"}
)

# Public action-space upper bounds.  They are copied here rather than importing
# agent-facing prompt code, so the evaluator resource contract stays below all
# agent adapters.
ELECTROCHEMICAL_REAGENT_ACTION_UPPER_MOL = 0.040
ELECTROCHEMICAL_SOLVENT_ACTION_UPPER_L = 0.080

_STOCK_FIELDS_BY_OPERATION: dict[str, tuple[str, str]] = {
    "add_component": ("reagent_mol", "amount_mol"),
    "add_reagent": ("reagent_mol", "amount_mol"),
    "add_solvent": ("solvent_L", "volume_L"),
    "resuspend_crystals": ("solvent_L", "volume_L"),
    "add_catalyst": ("catalyst_mol", "catalyst_amount_mol"),
    "seed_crystals": ("seed_g", "seed_mass_g"),
    "add_extractant": ("extractant_L", "volume_L"),
    "add_phase": ("phase_liquid_L", "volume_L"),
    "wash": ("wash_solvent_L", "wash_volume_L"),
}


class CampaignResourceError(RuntimeError):
    """Base class for campaign resource failures."""


class CampaignResourceIntegrityError(CampaignResourceError):
    """Raised when an event replay or serialized ledger is inconsistent."""


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, *, name: str) -> int:
    number = _nonnegative_int(value, name=name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return to_builtin(value)


def _normalized_float_map(
    values: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, float]:
    normalized = {
        str(key): _finite_nonnegative(value, name=f"{name}.{key}")
        for key, value in (values or {}).items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalized_int_map(
    values: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, int]:
    normalized = {
        str(key): _nonnegative_int(value, name=f"{name}.{key}")
        for key, value in (values or {}).items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class CampaignResourceCard:
    """Immutable hard limits for one persistent experimental campaign."""

    card_id: str
    operation_attempt_limit: int
    vessel_start_limit: int
    final_assay_limit: int
    nonfinal_instrument_use_limit: int
    stock_limits: Mapping[str, float]
    per_instrument_limits: Mapping[str, int | None] = field(default_factory=dict)
    process_time_limit_s: float | None = None
    implicit_operation_time_s: Mapping[str, float] = field(default_factory=dict)
    operation_repeat_limits: Mapping[str, int] = field(default_factory=dict)
    per_batch_limits: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CAMPAIGN_RESOURCE_CARD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_RESOURCE_CARD_VERSION:
            raise ValueError("unsupported campaign resource card schema")
        if not isinstance(self.card_id, str) or not self.card_id.strip():
            raise ValueError("card_id must be a non-empty string")
        _positive_int(
            self.operation_attempt_limit,
            name="operation_attempt_limit",
        )
        for name in (
            "vessel_start_limit",
            "final_assay_limit",
            "nonfinal_instrument_use_limit",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.final_assay_limit > self.vessel_start_limit:
            raise ValueError("final_assay_limit cannot exceed vessel_start_limit")
        object.__setattr__(
            self,
            "stock_limits",
            _normalized_float_map(self.stock_limits, name="stock_limits"),
        )
        instrument_limits: dict[str, int | None] = {}
        for instrument, limit in self.per_instrument_limits.items():
            instrument_id = str(instrument)
            instrument_limits[instrument_id] = (
                None
                if limit is None
                else _nonnegative_int(limit, name=f"per_instrument_limits.{instrument_id}")
            )
        object.__setattr__(
            self,
            "per_instrument_limits",
            MappingProxyType(dict(sorted(instrument_limits.items()))),
        )
        if self.process_time_limit_s is not None:
            process_time_limit = _finite_nonnegative(
                self.process_time_limit_s,
                name="process_time_limit_s",
            )
            if process_time_limit <= 0.0:
                raise ValueError("process_time_limit_s must be positive when supplied")
            object.__setattr__(self, "process_time_limit_s", process_time_limit)
        object.__setattr__(
            self,
            "implicit_operation_time_s",
            _normalized_float_map(
                self.implicit_operation_time_s,
                name="implicit_operation_time_s",
            ),
        )
        if self.implicit_operation_time_s and self.process_time_limit_s is None:
            raise ValueError("implicit_operation_time_s requires process_time_limit_s")
        object.__setattr__(
            self,
            "operation_repeat_limits",
            _normalized_int_map(
                self.operation_repeat_limits,
                name="operation_repeat_limits",
            ),
        )
        if self.per_batch_limits:
            # A normal ledger enforces each local package. Nested packages are
            # deliberately unsupported; historical cards serialize unchanged.
            if "per_batch" in self.per_batch_limits:
                raise ValueError("nested per-batch resource packages are unsupported")
            local = self.batch_card()
            if local.vessel_start_limit != 1 or local.final_assay_limit != 1:
                raise ValueError("a per-batch package requires one vessel and one final assay")
        object.__setattr__(self, "per_batch_limits", _deep_freeze(self.per_batch_limits))
        frozen_metadata = _deep_freeze(self.metadata)
        # Fail early if metadata is not canonical-JSON serializable.
        canonical_json_sha256(_deep_thaw(frozen_metadata))
        object.__setattr__(self, "metadata", frozen_metadata)

    @property
    def card_sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        hard_limits: dict[str, Any] = {
            "operation_attempts": self.operation_attempt_limit,
            "vessel_starts": self.vessel_start_limit,
            "final_assays": self.final_assay_limit,
            "nonfinal_instrument_uses": self.nonfinal_instrument_use_limit,
            "stocks": dict(self.stock_limits),
            "per_instrument": dict(self.per_instrument_limits),
        }
        if self.process_time_limit_s is not None:
            hard_limits["process_time_s"] = self.process_time_limit_s
        if self.implicit_operation_time_s:
            hard_limits["implicit_operation_time_s"] = dict(self.implicit_operation_time_s)
        if self.operation_repeat_limits:
            hard_limits["operation_repeats"] = dict(self.operation_repeat_limits)
        if self.per_batch_limits:
            hard_limits["per_batch"] = _deep_thaw(self.per_batch_limits)
        return {
            "schema_version": self.schema_version,
            "card_id": self.card_id,
            "hard_limits": hard_limits,
            "metadata": _deep_thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["card_sha256"] = self.card_sha256
        return payload

    def batch_card(self) -> CampaignResourceCard:
        """Construct the independent local envelope without sharing unused quota."""
        if not self.per_batch_limits:
            raise ValueError("card has no per-batch package")
        return CampaignResourceCard.from_dict({
            "schema_version": CAMPAIGN_RESOURCE_CARD_VERSION,
            "card_id": self.card_id + "-per-batch",
            "hard_limits": _deep_thaw(self.per_batch_limits),
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignResourceCard:
        if payload.get("schema_version") != CAMPAIGN_RESOURCE_CARD_VERSION:
            raise ValueError("unsupported campaign resource card schema")
        hard_limits = payload.get("hard_limits")
        if not isinstance(hard_limits, Mapping):
            raise ValueError("campaign resource card hard_limits must be an object")
        card = cls(
            card_id=str(payload.get("card_id", "")),
            operation_attempt_limit=int(hard_limits["operation_attempts"]),
            vessel_start_limit=int(hard_limits["vessel_starts"]),
            final_assay_limit=int(hard_limits["final_assays"]),
            nonfinal_instrument_use_limit=int(hard_limits["nonfinal_instrument_uses"]),
            stock_limits=dict(hard_limits.get("stocks", {})),
            per_instrument_limits=dict(hard_limits.get("per_instrument", {})),
            process_time_limit_s=(
                None
                if hard_limits.get("process_time_s") is None
                else float(hard_limits["process_time_s"])
            ),
            implicit_operation_time_s=dict(hard_limits.get("implicit_operation_time_s", {})),
            operation_repeat_limits=dict(hard_limits.get("operation_repeats", {})),
            per_batch_limits=dict(hard_limits.get("per_batch", {})),
            metadata=dict(payload.get("metadata", {})),
        )
        supplied_hash = payload.get("card_sha256")
        if supplied_hash is not None and supplied_hash != card.card_sha256:
            raise CampaignResourceIntegrityError("campaign resource card hash mismatch")
        return card


@dataclass(frozen=True)
class CampaignResourceDelta:
    """One proposed or realized resource delta."""

    operation_attempts: int = 1
    vessel_starts: int = 0
    final_assays: int = 0
    discarded_batches: int = 0
    nonfinal_instrument_uses: int = 0
    instrument_uses: Mapping[str, int] = field(default_factory=dict)
    stocks: Mapping[str, float] = field(default_factory=dict)
    process_time_s: float = 0.0
    sample_consumed_L: float = 0.0
    physical_cost: float = 0.0
    accumulated_risk: float = 0.0
    observed_risk: float = 0.0
    schema_version: str = CAMPAIGN_RESOURCE_DELTA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_RESOURCE_DELTA_VERSION:
            raise ValueError("unsupported campaign resource delta schema")
        for name in (
            "operation_attempts",
            "vessel_starts",
            "final_assays",
            "discarded_batches",
            "nonfinal_instrument_uses",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "instrument_uses",
            _normalized_int_map(self.instrument_uses, name="instrument_uses"),
        )
        object.__setattr__(
            self,
            "stocks",
            _normalized_float_map(self.stocks, name="stocks"),
        )
        for name in (
            "process_time_s",
            "sample_consumed_L",
            "physical_cost",
            "accumulated_risk",
            "observed_risk",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_attempts": self.operation_attempts,
            "vessel_starts": self.vessel_starts,
            "final_assays": self.final_assays,
            "discarded_batches": self.discarded_batches,
            "nonfinal_instrument_uses": self.nonfinal_instrument_uses,
            "instrument_uses": dict(self.instrument_uses),
            "stocks": dict(self.stocks),
            "report_only": {
                "process_time_s": self.process_time_s,
                "sample_consumed_L": self.sample_consumed_L,
                "physical_cost": self.physical_cost,
                "accumulated_risk": self.accumulated_risk,
                "observed_risk": self.observed_risk,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignResourceDelta:
        report = payload.get("report_only", {})
        if not isinstance(report, Mapping):
            raise ValueError("campaign resource report_only delta must be an object")
        return cls(
            operation_attempts=int(payload.get("operation_attempts", 0)),
            vessel_starts=int(payload.get("vessel_starts", 0)),
            final_assays=int(payload.get("final_assays", 0)),
            discarded_batches=int(payload.get("discarded_batches", 0)),
            nonfinal_instrument_uses=int(payload.get("nonfinal_instrument_uses", 0)),
            instrument_uses=dict(payload.get("instrument_uses", {})),
            stocks=dict(payload.get("stocks", {})),
            process_time_s=float(report.get("process_time_s", 0.0)),
            sample_consumed_L=float(report.get("sample_consumed_L", 0.0)),
            physical_cost=float(report.get("physical_cost", 0.0)),
            accumulated_risk=float(report.get("accumulated_risk", 0.0)),
            observed_risk=float(report.get("observed_risk", 0.0)),
            schema_version=str(payload.get("schema_version", CAMPAIGN_RESOURCE_DELTA_VERSION)),
        )


@dataclass(frozen=True)
class CampaignResourcePreflight:
    """Serializable result of reserving one operation attempt."""

    event_id: str
    allowed: bool
    attempt_charged: bool
    rejection_reasons: tuple[str, ...]
    proposed_delta: CampaignResourceDelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "allowed": self.allowed,
            "attempt_charged": self.attempt_charged,
            "rejection_reasons": list(self.rejection_reasons),
            "proposed_delta": self.proposed_delta.to_dict(),
        }


def _positive_action_float(action: Mapping[str, Any], field_name: str) -> float:
    value = action.get(field_name)
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0.0 else 0.0


def _is_committed(outcome: Mapping[str, Any]) -> bool:
    explicit = outcome.get("operation_committed")
    if isinstance(explicit, bool):
        return explicit
    explicit = outcome.get("committed")
    if isinstance(explicit, bool):
        return explicit
    return str(outcome.get("transaction_status", "")).lower() in {
        "committed",
        "success",
        "succeeded",
    }


def _report_value(
    report: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *keys: str,
    default: float = 0.0,
) -> float:
    for source in (report, outcome):
        for key in keys:
            if key in source:
                return _finite_nonnegative(source[key], name=key)
    return default


def derive_campaign_resource_delta(
    action: Mapping[str, Any],
    outcome: Mapping[str, Any] | None = None,
    *,
    starts_vessel: bool = False,
    reagent_charge_molar_multiplier: float = 1.0,
) -> CampaignResourceDelta:
    """Derive proposed or committed resources from one action/outcome pair.

    With ``outcome=None`` the returned physical quantities are reservations used
    by preflight.  With an outcome, physical quantities are zero unless the
    transaction was committed.  ``operation_attempts`` remains one in both
    cases; the ledger charges it during preflight only.
    """

    operation = str(action.get("operation", "invalid"))
    committed = outcome is None or _is_committed(outcome)
    stocks: dict[str, float] = {}
    stock_contract = _STOCK_FIELDS_BY_OPERATION.get(operation)
    if committed and stock_contract is not None:
        stock_id, field_name = stock_contract
        amount = _positive_action_float(action, field_name)
        if operation == "add_reagent":
            amount *= reagent_charge_molar_multiplier
        if amount > 0.0:
            stocks[stock_id] = amount

    instrument = str(action.get("instrument", "")) if operation == "measure" else ""
    is_final = committed and instrument == "final_assay"
    is_discard = committed and operation == "discard_batch"
    is_nonfinal = committed and bool(instrument) and not is_final
    instrument_uses = {instrument: 1} if is_nonfinal else {}

    process_time_s = 0.0
    sample_consumed_L = 0.0
    physical_cost = 0.0
    accumulated_risk = 0.0
    observed_risk = 0.0
    if outcome is not None and committed:
        raw_report = outcome.get("campaign_resource_report_delta", {})
        report = raw_report if isinstance(raw_report, Mapping) else {}
        process_time_s = _report_value(
            report,
            outcome,
            "process_time_s",
            "process_time_delta_s",
            "time_delta_s",
            default=_positive_action_float(action, "duration_s"),
        )
        sample_consumed_L = _report_value(
            report,
            outcome,
            "sample_consumed_L",
            "sample_consumed_delta_L",
            "sample_consumed",
            default=(
                _positive_action_float(action, "sample_volume_L") if operation == "sample" else 0.0
            ),
        )
        physical_cost = _report_value(
            report,
            outcome,
            "physical_cost",
            "physical_cost_delta",
            "cost_delta",
        )
        accumulated_risk = _report_value(
            report,
            outcome,
            "accumulated_risk",
            "risk_delta",
        )
        observed_risk = _report_value(
            report,
            outcome,
            "observed_risk",
            "safety_risk",
            "risk",
        )

    return CampaignResourceDelta(
        operation_attempts=1,
        vessel_starts=int(committed and starts_vessel),
        final_assays=int(is_final),
        discarded_batches=int(is_discard),
        nonfinal_instrument_uses=int(is_nonfinal),
        instrument_uses=instrument_uses,
        stocks=stocks,
        process_time_s=process_time_s,
        sample_consumed_L=sample_consumed_L,
        physical_cost=physical_cost,
        accumulated_risk=accumulated_risk,
        observed_risk=observed_risk,
    )


def campaign_resource_event_id(campaign_id: str, operation_attempt_index: int) -> str:
    """Return a deterministic opaque event ID for one campaign attempt."""

    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string")
    if (
        isinstance(operation_attempt_index, bool)
        or not isinstance(operation_attempt_index, int)
        or operation_attempt_index <= 0
    ):
        raise ValueError("operation_attempt_index must be a positive integer")
    digest = canonical_json_sha256(
        {
            "campaign_id": campaign_id,
            "operation_attempt_index": operation_attempt_index,
        }
    )
    return f"campaign-resource-{operation_attempt_index:06d}-{digest[:16]}"


class CampaignResourceLedger:
    """Persistent, idempotent two-phase ledger for a single resource card."""

    def __init__(
        self, card: CampaignResourceCard, *, reagent_charge_molar_multiplier: float = 1.0
    ) -> None:
        self.card = card
        self.reagent_charge_molar_multiplier = _finite_nonnegative(
            reagent_charge_molar_multiplier, name="reagent_charge_molar_multiplier"
        )
        if self.reagent_charge_molar_multiplier <= 0:
            raise ValueError("Reagent multiplier must be positive")
        self._protected_closeout_contract = self._load_protected_closeout_contract()
        self._events: dict[str, dict[str, Any]] = {}
        self._event_order: list[str] = []
        self.operation_attempts = 0
        self.vessel_starts = 0
        self.final_assays = 0
        self.discarded_batches = 0
        self.nonfinal_instrument_uses = 0
        self.instrument_uses: dict[str, int] = {}
        self.operation_committed_counts: dict[str, int] = {}
        self.stocks_used: dict[str, float] = {}
        self.process_time_s = 0.0
        self.sample_consumed_L = 0.0
        self.physical_cost = 0.0
        self.accumulated_risk = 0.0
        self.peak_risk = 0.0

    def preflight(
        self,
        event_id: str,
        action: Mapping[str, Any],
        *,
        starts_vessel: bool = False,
    ) -> CampaignResourcePreflight:
        """Reserve an attempt and reject actions outside the remaining hard envelope."""

        normalized_action = self._normalize_action(action)
        event_id = self._normalize_event_id(event_id)
        action_sha256 = self._action_sha256(normalized_action, starts_vessel)
        existing = self._events.get(event_id)
        if existing is not None:
            if existing["action_sha256"] != action_sha256:
                raise CampaignResourceIntegrityError(
                    "event_id was replayed with a different action or vessel-start flag"
                )
            return self._preflight_from_event(existing)

        proposed = self._proposed_delta(
            normalized_action,
            starts_vessel=starts_vessel,
        )
        attempt_available = self.operation_attempts < self.card.operation_attempt_limit
        reasons = self._hard_rejection_reasons(
            proposed,
            operation=str(normalized_action.get("operation", "invalid")),
            protected_closeout_action=self._is_protected_closeout_action(normalized_action),
            attempt_charged=attempt_available,
            operation_attempts=self.operation_attempts,
            vessel_starts=self.vessel_starts,
            final_assays=self.final_assays,
            discarded_batches=self.discarded_batches,
            nonfinal_instrument_uses=self.nonfinal_instrument_uses,
            instrument_uses=self.instrument_uses,
            operation_committed_counts=self.operation_committed_counts,
            stocks_used=self.stocks_used,
            process_time_s=self.process_time_s,
        )
        attempt_charged = attempt_available and not any(
            reason.startswith("protected_closeout_") for reason in reasons
        )
        if attempt_charged:
            self.operation_attempts += 1
        preflight = CampaignResourcePreflight(
            event_id=event_id,
            allowed=not reasons,
            attempt_charged=attempt_charged,
            rejection_reasons=tuple(reasons),
            proposed_delta=proposed,
        )
        self._event_order.append(event_id)
        self._events[event_id] = {
            "event_id": event_id,
            "ordinal": len(self._event_order),
            "action": normalized_action,
            "starts_vessel": bool(starts_vessel),
            "action_sha256": action_sha256,
            "preflight": preflight.to_dict(),
            "outcome": None,
        }
        self.verify_integrity()
        return preflight

    def preview_rejection_reasons(
        self,
        action: Mapping[str, Any],
        *,
        starts_vessel: bool = False,
    ) -> tuple[str, ...]:
        """Return hard-envelope rejection reasons without reserving an attempt.

        Agent-facing affordance generation must be able to consult the same
        resource policy as :meth:`preflight` without mutating the ledger.  The
        preview intentionally has no event id and does not append an event or
        increment ``operation_attempts``; the authoritative reservation still
        happens exactly once in :meth:`preflight` during ``env.step``.
        """

        normalized_action = self._normalize_action(action)
        proposed = self._proposed_delta(
            normalized_action,
            starts_vessel=starts_vessel,
        )
        attempt_charged = self.operation_attempts < self.card.operation_attempt_limit
        return tuple(
            self._hard_rejection_reasons(
                proposed,
                operation=str(normalized_action.get("operation", "invalid")),
                protected_closeout_action=self._is_protected_closeout_action(normalized_action),
                attempt_charged=attempt_charged,
                operation_attempts=self.operation_attempts,
                vessel_starts=self.vessel_starts,
                final_assays=self.final_assays,
                discarded_batches=self.discarded_batches,
                nonfinal_instrument_uses=self.nonfinal_instrument_uses,
                instrument_uses=self.instrument_uses,
                operation_committed_counts=self.operation_committed_counts,
                stocks_used=self.stocks_used,
                process_time_s=self.process_time_s,
            )
        )

    def record_outcome(
        self,
        event_id: str,
        action: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        starts_vessel: bool = False,
    ) -> CampaignResourceDelta:
        """Commit outcome-owned physical/reporting resources exactly once."""

        event_id = self._normalize_event_id(event_id)
        event = self._events.get(event_id)
        if event is None:
            raise CampaignResourceIntegrityError("outcome has no matching preflight event")
        normalized_action = self._normalize_action(action)
        if event["action_sha256"] != self._action_sha256(
            normalized_action,
            starts_vessel,
        ):
            raise CampaignResourceIntegrityError(
                "outcome action does not match its preflight reservation"
            )
        actual = derive_campaign_resource_delta(
            normalized_action,
            outcome,
            starts_vessel=starts_vessel,
            reagent_charge_molar_multiplier=self.reagent_charge_molar_multiplier,
        )
        committed = _is_committed(outcome)
        outcome_view = {
            "committed": committed,
            "delta": actual.to_dict(),
        }
        outcome_sha256 = canonical_json_sha256(outcome_view)
        if event["outcome"] is not None:
            if event["outcome"]["outcome_sha256"] != outcome_sha256:
                raise CampaignResourceIntegrityError(
                    "event_id was replayed with a different resource outcome"
                )
            return CampaignResourceDelta.from_dict(event["outcome"]["delta"])

        preflight = self._preflight_from_event(event)
        if committed and not preflight.allowed:
            raise CampaignResourceIntegrityError(
                "a resource-rejected action cannot have a committed outcome"
            )
        operation = str(normalized_action.get("operation", "invalid"))
        self._verify_actual_within_reservation(
            actual,
            preflight.proposed_delta,
            operation=operation,
        )
        self._apply_outcome_delta(
            actual,
            operation=(operation if committed else None),
        )
        event["outcome"] = {
            "committed": committed,
            "outcome_sha256": outcome_sha256,
            "delta": actual.to_dict(),
        }
        self.verify_integrity()
        return actual

    def snapshot(self) -> dict[str, Any]:
        self.verify_integrity()
        payload = {
            "schema_version": CAMPAIGN_RESOURCE_LEDGER_VERSION,
            "card": self.card.to_dict(),
            "reagent_charge_molar_multiplier": self.reagent_charge_molar_multiplier,
            "state": self._state_payload(),
            "events": [_deep_thaw(self._events[event_id]) for event_id in self._event_order],
            "last_event_id": self._event_order[-1] if self._event_order else None,
        }
        payload["ledger_sha256"] = canonical_json_sha256(payload)
        return payload

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> CampaignResourceLedger:
        if payload.get("schema_version") != CAMPAIGN_RESOURCE_LEDGER_VERSION:
            raise ValueError("unsupported campaign resource ledger schema")
        supplied_hash = payload.get("ledger_sha256")
        unhashed = dict(payload)
        unhashed.pop("ledger_sha256", None)
        if supplied_hash != canonical_json_sha256(unhashed):
            raise CampaignResourceIntegrityError("campaign resource ledger hash mismatch")
        card_payload = payload.get("card")
        state = payload.get("state")
        events = payload.get("events")
        if not isinstance(card_payload, Mapping) or not isinstance(state, Mapping):
            raise ValueError("campaign resource ledger card/state must be objects")
        if not isinstance(events, list):
            raise ValueError("campaign resource ledger events must be a list")
        ledger = cls(
            CampaignResourceCard.from_dict(card_payload),
            reagent_charge_molar_multiplier=float(payload["reagent_charge_molar_multiplier"]),
        )
        ledger.operation_attempts = int(state.get("operation_attempts", 0))
        ledger.vessel_starts = int(state.get("vessel_starts", 0))
        ledger.final_assays = int(state.get("final_assays", 0))
        ledger.discarded_batches = int(state.get("discarded_batches", 0))
        ledger.nonfinal_instrument_uses = int(state.get("nonfinal_instrument_uses", 0))
        ledger.instrument_uses = {
            str(key): int(value) for key, value in dict(state.get("instrument_uses", {})).items()
        }
        ledger.operation_committed_counts = {
            str(key): int(value)
            for key, value in dict(state.get("operation_committed_counts", {})).items()
        }
        ledger.stocks_used = {
            str(key): float(value) for key, value in dict(state.get("stocks_used", {})).items()
        }
        report = state.get("report_only", {})
        if not isinstance(report, Mapping):
            raise ValueError("campaign resource ledger report_only must be an object")
        ledger.process_time_s = float(report.get("process_time_s", 0.0))
        ledger.sample_consumed_L = float(report.get("sample_consumed_L", 0.0))
        ledger.physical_cost = float(report.get("physical_cost", 0.0))
        ledger.accumulated_risk = float(report.get("accumulated_risk", 0.0))
        ledger.peak_risk = float(report.get("peak_risk", 0.0))
        for raw_event in events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("campaign resource event must be an object")
            event = _deep_thaw(raw_event)
            event_id = ledger._normalize_event_id(str(event.get("event_id", "")))
            if event_id in ledger._events:
                raise CampaignResourceIntegrityError("duplicate campaign resource event_id")
            ledger._event_order.append(event_id)
            ledger._events[event_id] = event
        if payload.get("last_event_id") != (
            ledger._event_order[-1] if ledger._event_order else None
        ):
            raise CampaignResourceIntegrityError("campaign resource last_event_id mismatch")
        ledger.verify_integrity()
        if ledger.snapshot()["ledger_sha256"] != supplied_hash:
            raise CampaignResourceIntegrityError(
                "campaign resource snapshot is not canonically replayable"
            )
        return ledger

    def verify_integrity(self) -> bool:
        """Recompute all monotone state from event receipts and enforce hard limits."""

        attempts = 0
        vessels = 0
        finals = 0
        discarded = 0
        nonfinal = 0
        instruments: dict[str, int] = {}
        operation_committed_counts: dict[str, int] = {}
        stocks: dict[str, float] = {}
        process_time_s = 0.0
        sample_consumed_L = 0.0
        physical_cost = 0.0
        accumulated_risk = 0.0
        peak_risk = 0.0
        for ordinal, event_id in enumerate(self._event_order, start=1):
            event = self._events.get(event_id)
            if event is None or event.get("event_id") != event_id:
                raise CampaignResourceIntegrityError("campaign resource event index mismatch")
            if event.get("ordinal") != ordinal:
                raise CampaignResourceIntegrityError("campaign resource event ordinal mismatch")
            action = self._normalize_action(event.get("action", {}))
            starts_vessel = event.get("starts_vessel") is True
            if event.get("action_sha256") != self._action_sha256(action, starts_vessel):
                raise CampaignResourceIntegrityError("campaign resource action hash mismatch")
            preflight = self._preflight_from_event(event)
            expected_proposed = self._proposed_delta(
                action,
                starts_vessel=starts_vessel,
            )
            expected_attempt_available = attempts < self.card.operation_attempt_limit
            expected_reasons = self._hard_rejection_reasons(
                expected_proposed,
                operation=str(action.get("operation", "invalid")),
                protected_closeout_action=self._is_protected_closeout_action(action),
                attempt_charged=expected_attempt_available,
                operation_attempts=attempts,
                vessel_starts=vessels,
                final_assays=finals,
                discarded_batches=discarded,
                nonfinal_instrument_uses=nonfinal,
                instrument_uses=instruments,
                operation_committed_counts=operation_committed_counts,
                stocks_used=stocks,
                process_time_s=process_time_s,
            )
            if preflight.event_id != event_id:
                raise CampaignResourceIntegrityError(
                    "campaign resource preflight event_id mismatch"
                )
            if preflight.proposed_delta != expected_proposed:
                raise CampaignResourceIntegrityError(
                    "campaign resource preflight proposal mismatch"
                )
            expected_attempt_charged = expected_attempt_available and not any(
                reason.startswith("protected_closeout_") for reason in expected_reasons
            )
            if preflight.attempt_charged != expected_attempt_charged:
                raise CampaignResourceIntegrityError(
                    "campaign resource preflight attempt decision mismatch"
                )
            if preflight.rejection_reasons != tuple(expected_reasons):
                raise CampaignResourceIntegrityError(
                    "campaign resource preflight rejection decision mismatch"
                )
            if preflight.allowed != (not expected_reasons):
                raise CampaignResourceIntegrityError(
                    "campaign resource preflight allowed decision mismatch"
                )
            if preflight.attempt_charged:
                attempts += 1
            outcome = event.get("outcome")
            if outcome is None:
                continue
            if not isinstance(outcome, Mapping):
                raise CampaignResourceIntegrityError("campaign resource outcome is invalid")
            actual = CampaignResourceDelta.from_dict(outcome.get("delta", {}))
            outcome_view = {
                "committed": outcome.get("committed") is True,
                "delta": actual.to_dict(),
            }
            if outcome.get("outcome_sha256") != canonical_json_sha256(outcome_view):
                raise CampaignResourceIntegrityError("campaign resource outcome hash mismatch")
            if actual.operation_attempts != 1:
                raise CampaignResourceIntegrityError(
                    "campaign resource outcome attempt delta must be one"
                )
            if outcome_view["committed"] and not preflight.allowed:
                raise CampaignResourceIntegrityError(
                    "resource-rejected event was recorded as committed"
                )
            if not outcome_view["committed"] and self._has_committed_debit(actual):
                raise CampaignResourceIntegrityError(
                    "uncommitted campaign resource outcome has a physical or report debit"
                )
            self._verify_actual_within_reservation(
                actual,
                preflight.proposed_delta,
                operation=str(action.get("operation", "invalid")),
            )
            if outcome_view["committed"]:
                operation = str(action.get("operation", "invalid"))
                operation_committed_counts[operation] = (
                    operation_committed_counts.get(operation, 0) + 1
                )
            vessels += actual.vessel_starts
            finals += actual.final_assays
            discarded += actual.discarded_batches
            nonfinal += actual.nonfinal_instrument_uses
            for instrument, count in actual.instrument_uses.items():
                instruments[instrument] = instruments.get(instrument, 0) + count
            for stock_id, amount in actual.stocks.items():
                stocks[stock_id] = stocks.get(stock_id, 0.0) + amount
            process_time_s += actual.process_time_s
            sample_consumed_L += actual.sample_consumed_L
            physical_cost += actual.physical_cost
            accumulated_risk += actual.accumulated_risk
            peak_risk = max(peak_risk, actual.observed_risk)
        expected: dict[str, Any] = {
            "operation_attempts": attempts,
            "vessel_starts": vessels,
            "final_assays": finals,
            "discarded_batches": discarded,
            "nonfinal_instrument_uses": nonfinal,
            "instrument_uses": dict(sorted(instruments.items())),
            "operation_committed_counts": dict(sorted(operation_committed_counts.items())),
            "stocks_used": dict(sorted(stocks.items())),
            "process_time_s": process_time_s,
            "sample_consumed_L": sample_consumed_L,
            "physical_cost": physical_cost,
            "accumulated_risk": accumulated_risk,
            "peak_risk": peak_risk,
        }
        observed: dict[str, Any] = {
            "operation_attempts": self.operation_attempts,
            "vessel_starts": self.vessel_starts,
            "final_assays": self.final_assays,
            "discarded_batches": self.discarded_batches,
            "nonfinal_instrument_uses": self.nonfinal_instrument_uses,
            "instrument_uses": dict(sorted(self.instrument_uses.items())),
            "operation_committed_counts": dict(sorted(self.operation_committed_counts.items())),
            "stocks_used": dict(sorted(self.stocks_used.items())),
            "process_time_s": self.process_time_s,
            "sample_consumed_L": self.sample_consumed_L,
            "physical_cost": self.physical_cost,
            "accumulated_risk": self.accumulated_risk,
            "peak_risk": self.peak_risk,
        }
        for key in (
            "operation_attempts",
            "vessel_starts",
            "final_assays",
            "discarded_batches",
            "nonfinal_instrument_uses",
            "instrument_uses",
            "operation_committed_counts",
            "stocks_used",
        ):
            if observed[key] != expected[key]:
                raise CampaignResourceIntegrityError(
                    f"campaign resource monotone state mismatch: {key}"
                )
        for key in (
            "process_time_s",
            "sample_consumed_L",
            "physical_cost",
            "accumulated_risk",
            "peak_risk",
        ):
            if not math.isclose(
                float(observed[key]),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise CampaignResourceIntegrityError(
                    f"campaign resource report state mismatch: {key}"
                )
        self._verify_state_within_limits()
        return True

    def _load_protected_closeout_contract(self) -> dict[str, Any] | None:
        closeout = self.card.metadata.get("closeout_policy")
        closeout = closeout if isinstance(closeout, Mapping) else {}
        if closeout.get("policy") != PROTECTED_CLOSEOUT_POLICY:
            return None
        process = self.card.metadata.get("process_time_policy")
        process = process if isinstance(process, Mapping) else {}
        planned_batches = _positive_int(
            closeout.get("planned_batches"),
            name="closeout_policy.planned_batches",
        )
        operations_per_batch = _positive_int(
            closeout.get("final_assay_path_operations_per_batch"),
            name="closeout_policy.final_assay_path_operations_per_batch",
        )
        operation_reserve = _positive_int(
            closeout.get("final_assay_path_total_operation_reserve"),
            name="closeout_policy.final_assay_path_total_operation_reserve",
        )
        if operation_reserve != planned_batches * operations_per_batch:
            raise ValueError(
                "protected closeout operation reserve differs from its per-batch formula"
            )
        if operation_reserve >= self.card.operation_attempt_limit:
            raise ValueError("protected closeout operation reserve leaves no exploration attempt")
        discard_operations_per_batch = _positive_int(
            closeout.get("discard_path_operations_per_batch"),
            name="closeout_policy.discard_path_operations_per_batch",
        )
        discard_operation_reserve = _positive_int(
            closeout.get("discard_path_total_operation_reserve"),
            name="closeout_policy.discard_path_total_operation_reserve",
        )
        if discard_operation_reserve != planned_batches * discard_operations_per_batch:
            raise ValueError(
                "protected discard operation reserve differs from its per-batch formula"
            )
        if self.card.process_time_limit_s is None:
            raise ValueError("protected closeout requires a process-time hard limit")
        protected_process_time = _finite_nonnegative(
            process.get("protected_reserve_s"),
            name="process_time_policy.protected_reserve_s",
        )
        if (
            protected_process_time <= 0.0
            or protected_process_time >= self.card.process_time_limit_s
        ):
            raise ValueError(
                "protected process-time reserve must be positive and below the hard limit"
            )
        raw_allowed = closeout.get("allowed_operation_classes")
        if raw_allowed is None:
            allowed = DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS
        elif isinstance(raw_allowed, (list, tuple, frozenset)) and all(
            isinstance(item, str) and item for item in raw_allowed
        ):
            allowed = frozenset(raw_allowed)
        else:
            raise ValueError("closeout_policy.allowed_operation_classes must contain operation IDs")
        if not {"discard_batch", "terminate", "final_assay"}.issubset(allowed):
            raise ValueError(
                "protected closeout operations must include discard, terminate and final assay"
            )
        return {
            "planned_batches": planned_batches,
            "operations_per_batch": operations_per_batch,
            "operation_reserve": operation_reserve,
            "discard_operations_per_batch": discard_operations_per_batch,
            "discard_operation_reserve": discard_operation_reserve,
            "protected_process_time_s": protected_process_time,
            "exploration_process_time_ceiling_s": (
                self.card.process_time_limit_s - protected_process_time
            ),
            "allowed_operation_classes": tuple(sorted(allowed)),
        }

    def _is_protected_closeout_action(self, action: Mapping[str, Any]) -> bool:
        contract = self._protected_closeout_contract
        if contract is None:
            return False
        operation = str(action.get("operation", "invalid"))
        operation_class = (
            "final_assay"
            if operation == "measure" and action.get("instrument") == "final_assay"
            else operation
        )
        return operation_class in contract["allowed_operation_classes"]

    def _required_protected_operation_reserve(
        self,
        *,
        final_assays: int,
        discarded_batches: int,
    ) -> int:
        contract = self._protected_closeout_contract
        if contract is None:
            return 0
        outstanding = max(
            int(contract["planned_batches"]) - final_assays - discarded_batches,
            0,
        )
        return outstanding * int(contract["operations_per_batch"])

    @staticmethod
    def _protected_operation_class(action: Mapping[str, Any]) -> str:
        operation = str(action.get("operation", "invalid"))
        if operation == "measure" and action.get("instrument") == "final_assay":
            return "final_assay"
        return operation

    def _protected_reserve_consumption(self) -> dict[str, Any]:
        contract = self._protected_closeout_contract
        if contract is None:
            return {
                "operation_attempts": 0,
                "process_time_s": 0.0,
                "by_operation_class": {},
            }
        nonreserved_attempts = self.card.operation_attempt_limit - int(
            contract["operation_reserve"]
        )
        process_ceiling = float(contract["exploration_process_time_ceiling_s"])
        charged_attempts = 0
        elapsed_process_time = 0.0
        operation_consumed = 0
        process_consumed = 0.0
        by_class: dict[str, dict[str, float | int]] = {}
        for event_id in self._event_order:
            event = self._events[event_id]
            action = event.get("action")
            action = action if isinstance(action, Mapping) else {}
            preflight = self._preflight_from_event(event)
            protected = self._is_protected_closeout_action(action)
            operation_class = self._protected_operation_class(action)
            entry = by_class.setdefault(
                operation_class,
                {"operation_attempts": 0, "process_time_s": 0.0},
            )
            if preflight.attempt_charged:
                charged_attempts += 1
                if protected and charged_attempts > nonreserved_attempts:
                    operation_consumed += 1
                    entry["operation_attempts"] = int(entry["operation_attempts"]) + 1
            outcome = event.get("outcome")
            if not isinstance(outcome, Mapping) or outcome.get("committed") is not True:
                continue
            delta = CampaignResourceDelta.from_dict(outcome.get("delta", {}))
            before = elapsed_process_time
            elapsed_process_time += delta.process_time_s
            overlap = max(
                elapsed_process_time - max(before, process_ceiling),
                0.0,
            )
            if protected and overlap > 0.0:
                process_consumed += overlap
                entry["process_time_s"] = float(entry["process_time_s"]) + overlap
        return {
            "operation_attempts": operation_consumed,
            "process_time_s": process_consumed,
            "by_operation_class": {
                key: value
                for key, value in sorted(by_class.items())
                if value["operation_attempts"] or value["process_time_s"]
            },
        }

    def _hard_rejection_reasons(
        self,
        proposed: CampaignResourceDelta,
        *,
        operation: str,
        protected_closeout_action: bool,
        attempt_charged: bool,
        operation_attempts: int,
        vessel_starts: int,
        final_assays: int,
        discarded_batches: int,
        nonfinal_instrument_uses: int,
        instrument_uses: Mapping[str, int],
        operation_committed_counts: Mapping[str, int],
        stocks_used: Mapping[str, float],
        process_time_s: float,
    ) -> list[str]:
        reasons: list[str] = []
        if not attempt_charged:
            reasons.append("operation_attempt_limit")
        contract = self._protected_closeout_contract
        if contract is not None and attempt_charged and not protected_closeout_action:
            required_attempts = self._required_protected_operation_reserve(
                final_assays=final_assays,
                discarded_batches=discarded_batches,
            )
            remaining_after = self.card.operation_attempt_limit - operation_attempts - 1
            if remaining_after < required_attempts:
                reasons.append("protected_closeout_operation_reserve")
        if vessel_starts + proposed.vessel_starts > self.card.vessel_start_limit:
            reasons.append("vessel_start_limit")
        if final_assays + proposed.final_assays > self.card.final_assay_limit:
            reasons.append("final_assay_limit")
        if (
            nonfinal_instrument_uses + proposed.nonfinal_instrument_uses
            > self.card.nonfinal_instrument_use_limit
        ):
            reasons.append("nonfinal_instrument_use_limit")
        repeat_limit = self.card.operation_repeat_limits.get(operation)
        if (
            repeat_limit is not None
            and operation_committed_counts.get(operation, 0) + 1 > repeat_limit
        ):
            reasons.append(f"operation_repeat_limit:{operation}")
        if (
            self.card.process_time_limit_s is not None
            and process_time_s + proposed.process_time_s
            > self.card.process_time_limit_s + RESOURCE_FLOAT_ABS_TOL
        ):
            reasons.append("process_time_limit")
        if (
            contract is not None
            and not protected_closeout_action
            and process_time_s + proposed.process_time_s
            > float(contract["exploration_process_time_ceiling_s"]) + RESOURCE_FLOAT_ABS_TOL
        ):
            reasons.append("protected_closeout_process_time_reserve")
        for instrument, count in proposed.instrument_uses.items():
            instrument_limit = self.card.per_instrument_limits.get(instrument)
            if (
                instrument_limit is not None
                and instrument_uses.get(instrument, 0) + count > instrument_limit
            ):
                reasons.append(f"per_instrument_limit:{instrument}")
        for stock_id, amount in proposed.stocks.items():
            stock_limit = self.card.stock_limits.get(stock_id)
            if (
                stock_limit is not None
                and stocks_used.get(stock_id, 0.0) + amount > stock_limit + 1.0e-12
            ):
                reasons.append(f"stock_limit:{stock_id}")
        return sorted(set(reasons))

    @staticmethod
    def _has_committed_debit(delta: CampaignResourceDelta) -> bool:
        return bool(
            delta.vessel_starts
            or delta.final_assays
            or delta.discarded_batches
            or delta.nonfinal_instrument_uses
            or delta.instrument_uses
            or delta.stocks
            or delta.process_time_s
            or delta.sample_consumed_L
            or delta.physical_cost
            or delta.accumulated_risk
            or delta.observed_risk
        )

    def _state_payload(self) -> dict[str, Any]:
        stocks_remaining = {
            stock_id: max(limit - self.stocks_used.get(stock_id, 0.0), 0.0)
            for stock_id, limit in self.card.stock_limits.items()
        }
        instrument_remaining = {
            instrument: (
                None if limit is None else max(limit - self.instrument_uses.get(instrument, 0), 0)
            )
            for instrument, limit in self.card.per_instrument_limits.items()
        }
        payload: dict[str, Any] = {
            "operation_attempts": self.operation_attempts,
            "vessel_starts": self.vessel_starts,
            "final_assays": self.final_assays,
            "discarded_batches": self.discarded_batches,
            "closed_batches": self.final_assays + self.discarded_batches,
            "nonfinal_instrument_uses": self.nonfinal_instrument_uses,
            "instrument_uses": dict(sorted(self.instrument_uses.items())),
            "operation_committed_counts": dict(sorted(self.operation_committed_counts.items())),
            "stocks_used": dict(sorted(self.stocks_used.items())),
            "remaining": {
                "operation_attempts": max(
                    self.card.operation_attempt_limit - self.operation_attempts,
                    0,
                ),
                "vessel_starts": max(
                    self.card.vessel_start_limit - self.vessel_starts,
                    0,
                ),
                "final_assays": max(
                    self.card.final_assay_limit - self.final_assays,
                    0,
                ),
                "nonfinal_instrument_uses": max(
                    self.card.nonfinal_instrument_use_limit - self.nonfinal_instrument_uses,
                    0,
                ),
                "stocks": stocks_remaining,
                "per_instrument": instrument_remaining,
                "process_time_s": (
                    None
                    if self.card.process_time_limit_s is None
                    else max(self.card.process_time_limit_s - self.process_time_s, 0.0)
                ),
                "operation_repeats": {
                    operation: max(
                        limit - self.operation_committed_counts.get(operation, 0),
                        0,
                    )
                    for operation, limit in self.card.operation_repeat_limits.items()
                },
            },
            "report_only": {
                "process_time_s": self.process_time_s,
                "sample_consumed_L": self.sample_consumed_L,
                "physical_cost": self.physical_cost,
                "accumulated_risk": self.accumulated_risk,
                "peak_risk": self.peak_risk,
            },
        }
        contract = self._protected_closeout_contract
        if contract is not None:
            consumption = self._protected_reserve_consumption()
            payload["protected_closeout_reserve"] = {
                "policy": PROTECTED_CLOSEOUT_POLICY,
                "allowed_operation_classes": list(contract["allowed_operation_classes"]),
                "planned_batches": contract["planned_batches"],
                "outstanding_batches": max(
                    int(contract["planned_batches"]) - self.final_assays - self.discarded_batches,
                    0,
                ),
                "required_operation_attempts": (
                    self._required_protected_operation_reserve(
                        final_assays=self.final_assays,
                        discarded_batches=self.discarded_batches,
                    )
                ),
                "protected_process_time_s": contract["protected_process_time_s"],
                "process_time_consumed_s": consumption["process_time_s"],
                "operation_attempts_consumed": consumption["operation_attempts"],
                "consumption_by_operation_class": consumption["by_operation_class"],
            }
            payload["report_only"]["protected_reserve_consumed_s"] = consumption["process_time_s"]
            payload["report_only"]["reserve_consumption_by_operation_class"] = consumption[
                "by_operation_class"
            ]
        return payload

    def _apply_outcome_delta(
        self,
        delta: CampaignResourceDelta,
        *,
        operation: str | None,
    ) -> None:
        self.vessel_starts += delta.vessel_starts
        self.final_assays += delta.final_assays
        self.discarded_batches += delta.discarded_batches
        self.nonfinal_instrument_uses += delta.nonfinal_instrument_uses
        for instrument, count in delta.instrument_uses.items():
            self.instrument_uses[instrument] = self.instrument_uses.get(instrument, 0) + count
        if operation is not None:
            self.operation_committed_counts[operation] = (
                self.operation_committed_counts.get(operation, 0) + 1
            )
        for stock_id, amount in delta.stocks.items():
            self.stocks_used[stock_id] = self.stocks_used.get(stock_id, 0.0) + amount
        self.process_time_s += delta.process_time_s
        self.sample_consumed_L += delta.sample_consumed_L
        self.physical_cost += delta.physical_cost
        self.accumulated_risk += delta.accumulated_risk
        self.peak_risk = max(self.peak_risk, delta.observed_risk)
        self._verify_state_within_limits()

    def _verify_actual_within_reservation(
        self,
        actual: CampaignResourceDelta,
        proposed: CampaignResourceDelta,
        *,
        operation: str,
    ) -> None:
        for name in (
            "vessel_starts",
            "final_assays",
            "discarded_batches",
            "nonfinal_instrument_uses",
        ):
            if getattr(actual, name) > getattr(proposed, name):
                raise CampaignResourceIntegrityError(
                    f"outcome exceeded preflight reservation: {name}"
                )
        for instrument, count in actual.instrument_uses.items():
            if count > proposed.instrument_uses.get(instrument, 0):
                raise CampaignResourceIntegrityError(
                    f"outcome exceeded instrument reservation: {instrument}"
                )
        for stock_id, amount in actual.stocks.items():
            if amount > proposed.stocks.get(stock_id, 0.0) + RESOURCE_FLOAT_ABS_TOL:
                raise CampaignResourceIntegrityError(
                    f"outcome exceeded stock reservation: {stock_id}"
                )
        if (
            self.card.process_time_limit_s is not None
            and actual.process_time_s > proposed.process_time_s + RESOURCE_FLOAT_ABS_TOL
        ):
            raise CampaignResourceIntegrityError(
                "outcome exceeded process-time reservation: "
                f"operation={operation}, actual_s={actual.process_time_s}, "
                f"reserved_s={proposed.process_time_s}"
            )

    def _proposed_delta(
        self,
        action: Mapping[str, Any],
        *,
        starts_vessel: bool,
    ) -> CampaignResourceDelta:
        proposed = derive_campaign_resource_delta(
            action,
            starts_vessel=starts_vessel,
            reagent_charge_molar_multiplier=self.reagent_charge_molar_multiplier,
        )
        if self.card.process_time_limit_s is None:
            return proposed
        operation = str(action.get("operation", "invalid"))
        requested_duration_s = _positive_action_float(action, "duration_s")
        reserved_time_s = (
            requested_duration_s
            if requested_duration_s > 0.0
            else float(self.card.implicit_operation_time_s.get(operation, 0.0))
        )
        return replace(
            proposed,
            process_time_s=reserved_time_s,
        )

    def _verify_state_within_limits(self) -> None:
        checks = (
            ("operation_attempt_limit", self.operation_attempts, self.card.operation_attempt_limit),
            ("vessel_start_limit", self.vessel_starts, self.card.vessel_start_limit),
            ("final_assay_limit", self.final_assays, self.card.final_assay_limit),
            (
                "nonfinal_instrument_use_limit",
                self.nonfinal_instrument_uses,
                self.card.nonfinal_instrument_use_limit,
            ),
        )
        exceeded = [name for name, observed, limit in checks if observed > limit]
        for instrument, instrument_limit in self.card.per_instrument_limits.items():
            if (
                instrument_limit is not None
                and self.instrument_uses.get(instrument, 0) > instrument_limit
            ):
                exceeded.append(f"per_instrument_limit:{instrument}")
        for stock_id, stock_limit in self.card.stock_limits.items():
            if self.stocks_used.get(stock_id, 0.0) > stock_limit + RESOURCE_FLOAT_ABS_TOL:
                exceeded.append(f"stock_limit:{stock_id}")
        if (
            self.card.process_time_limit_s is not None
            and self.process_time_s > self.card.process_time_limit_s + RESOURCE_FLOAT_ABS_TOL
        ):
            exceeded.append("process_time_limit")
        for operation, repeat_limit in self.card.operation_repeat_limits.items():
            if self.operation_committed_counts.get(operation, 0) > repeat_limit:
                exceeded.append(f"operation_repeat_limit:{operation}")
        if exceeded:
            raise CampaignResourceIntegrityError(
                "campaign resource hard limit exceeded: " + ", ".join(exceeded)
            )

    @staticmethod
    def _normalize_event_id(event_id: str) -> str:
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        return event_id

    @staticmethod
    def _normalize_action(action: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise TypeError("campaign resource action must be a mapping")
        normalized = to_builtin(dict(action))
        if not isinstance(normalized, dict):
            raise TypeError("campaign resource action must normalize to an object")
        canonical_json_sha256(normalized)
        return normalized

    @staticmethod
    def _action_sha256(action: Mapping[str, Any], starts_vessel: bool) -> str:
        return canonical_json_sha256(
            {
                "action": dict(action),
                "starts_vessel": bool(starts_vessel),
            }
        )

    @staticmethod
    def _preflight_from_event(event: Mapping[str, Any]) -> CampaignResourcePreflight:
        payload = event.get("preflight")
        if not isinstance(payload, Mapping):
            raise CampaignResourceIntegrityError("campaign resource preflight is missing")
        reasons = payload.get("rejection_reasons", [])
        if not isinstance(reasons, list):
            raise CampaignResourceIntegrityError("preflight rejection_reasons must be a list")
        proposed = payload.get("proposed_delta")
        if not isinstance(proposed, Mapping):
            raise CampaignResourceIntegrityError("preflight proposed_delta must be an object")
        return CampaignResourcePreflight(
            event_id=str(payload.get("event_id", "")),
            allowed=payload.get("allowed") is True,
            attempt_charged=payload.get("attempt_charged") is True,
            rejection_reasons=tuple(str(item) for item in reasons),
            proposed_delta=CampaignResourceDelta.from_dict(proposed),
        )


def generous_electrochemical_max_envelope_card(
    *,
    experiment_count: int = 6,
    operation_attempt_limit: int = 84,
    nonfinal_instrument_use_limit: int = 18,
    stock_action_envelopes_per_experiment: float = 1.0,
    card_id: str = "electrochemical-k6-generous-max-envelope-v1",
) -> CampaignResourceCard:
    """Return the development G2 max envelope requested for electrochemistry.

    Each stock is the public per-action upper bound times the experiment count
    and a declared envelope multiplier.  A multiplier above one permits
    semibatch allocation without imposing equal per-vessel slots.
    """

    if experiment_count <= 0:
        raise ValueError("experiment_count must be positive")
    stock_envelopes = _finite_nonnegative(
        stock_action_envelopes_per_experiment,
        name="stock_action_envelopes_per_experiment",
    )
    if stock_envelopes <= 0.0:
        raise ValueError("stock_action_envelopes_per_experiment must be positive")
    return CampaignResourceCard(
        card_id=card_id,
        operation_attempt_limit=operation_attempt_limit,
        vessel_start_limit=experiment_count,
        final_assay_limit=experiment_count,
        nonfinal_instrument_use_limit=nonfinal_instrument_use_limit,
        stock_limits={
            "reagent_mol": (
                ELECTROCHEMICAL_REAGENT_ACTION_UPPER_MOL * experiment_count * stock_envelopes
            ),
            "solvent_L": (
                ELECTROCHEMICAL_SOLVENT_ACTION_UPPER_L * experiment_count * stock_envelopes
            ),
        },
        metadata={
            "task_id": "electrochemical-conversion",
            "envelope_kind": "generous_max_envelope",
            "experiment_count": experiment_count,
            "stock_action_envelopes_per_experiment": stock_envelopes,
            "source_action_upper_bounds": {
                "add_reagent.amount_mol": ELECTROCHEMICAL_REAGENT_ACTION_UPPER_MOL,
                "add_solvent.volume_L": ELECTROCHEMICAL_SOLVENT_ACTION_UPPER_L,
            },
            "debit_semantics": (
                "operation attempts at preflight; physical resources on committed outcome"
            ),
        },
    )


__all__ = [
    "CAMPAIGN_RESOURCE_CARD_VERSION",
    "CAMPAIGN_RESOURCE_DELTA_VERSION",
    "CAMPAIGN_RESOURCE_LEDGER_VERSION",
    "CampaignResourceCard",
    "CampaignResourceDelta",
    "CampaignResourceError",
    "CampaignResourceIntegrityError",
    "CampaignResourceLedger",
    "CampaignResourcePreflight",
    "campaign_resource_event_id",
    "derive_campaign_resource_delta",
    "generous_electrochemical_max_envelope_card",
]
