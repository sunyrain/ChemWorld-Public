"""Work II adapter for the frozen Work I process-profile surface.

The first-paper profile is a 19-coordinate descriptive vector.  Work II keeps
the coordinate identities and operational meanings, but rebuilds denominators
from the actually planned discovery campaign rather than assuming six fixed
lifecycles.  Evaluator-owned truth and blind replays are never accepted as
participant records.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.action_codec import ActionCodec
from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
)
from chemworld.data.records import trajectory_value
from chemworld.eval.policy_validity_contract import (
    AXES,
    ENDPOINT_CONTEXT,
    METRICS,
    profile_contract_sha256,
)
from chemworld.eval.provenance import canonical_json_sha256

WORK_II_PROCESS_PROFILE_VERSION = "chemworld-work-ii-process-profile-0.1"
WORK_II_RESOURCE_REPLAY_VERSION = "chemworld-work-ii-resource-replay-0.1"
WORK_II_HIDDEN_BOUNDARY_AUDIT_VERSION = "chemworld-work-ii-hidden-boundary-audit-0.1"
WORK_II_EXECUTION_AUDIT_VERSION = "chemworld-work-ii-execution-audit-0.1"

_PROCESS_EXCLUSIONS = frozenset({"measure", "terminate", "discard_batch"})
_TERMINAL_STATES = frozenset({"completed", "right_censored", "failed"})
_EVALUATOR_ROLES = frozenset(
    {
        "blind_evaluator",
        "evaluator",
        "evaluator_only",
        "held_out_evaluator",
        "held_out_truth",
    }
)
_VISIBLE_RECORD_FIELDS = (
    "agent_view",
    "agent_visible_observation",
)
_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "cell_key_sha256",
        "evaluator_truth",
        "evaluator_truth_report_sha256",
        "formal_cell",
        "formal_preflight_sha256",
        "mechanism_family_intervention_hash",
        "mechanism_hash",
        "mechanism_id",
        "prior_arm",
        "private_identity",
        "private_world_id",
        "truth",
        "world_cluster_id",
        "world_family_intervention_hash",
        "world_id",
        "world_seed",
    }
)
_FORBIDDEN_PROTOCOL_TOKENS = ("aligned_nominal", "misindexed_nominal")
_RETENTION_FRACTION = 0.9
_FLOAT_TOLERANCE = 1.0e-12
_RESOURCE_ACTION_CODEC = ActionCodec()


class WorkIIProcessProfileError(ValueError):
    """Raised when participant evidence cannot support a process profile."""


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkIIProcessProfileError(f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkIIProcessProfileError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkIIProcessProfileError(f"{label} must be finite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkIIProcessProfileError(f"{label} must be a non-negative integer")
    return value


def _step(record: Mapping[str, Any], fallback: int) -> int:
    value = record.get("step", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkIIProcessProfileError("participant record step must be a positive integer")
    return value


def _operation(record: Mapping[str, Any]) -> str:
    action = _mapping(record.get("action"), "participant record action")
    operation = action.get("operation")
    if not isinstance(operation, str) or not operation:
        raise WorkIIProcessProfileError("participant action lacks operation")
    return operation


def _instrument(record: Mapping[str, Any]) -> str | None:
    action = _mapping(record.get("action"), "participant record action")
    if action.get("operation") != "measure":
        return None
    value = action.get("instrument")
    if not isinstance(value, str) or not value:
        raise WorkIIProcessProfileError("measurement action lacks instrument")
    return value


def _committed(record: Mapping[str, Any]) -> bool:
    return record.get("transaction_status") == "committed"


def _participant_role(record: Mapping[str, Any]) -> None:
    for key in ("execution_role", "operation_role", "trajectory_role"):
        value = record.get(key)
        if isinstance(value, str) and value.lower() in _EVALUATOR_ROLES:
            raise WorkIIProcessProfileError(
                f"evaluator-owned record entered participant profile via {key}"
            )


def _public_resource_state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    agent_view = _mapping(trajectory_value(record, "agent_view", None), "record.agent_view")
    tool_json = _mapping(agent_view.get("tool_json"), "record.agent_view.tool_json")
    campaign = _mapping(
        tool_json.get("campaign_state"),
        "record.agent_view.tool_json.campaign_state",
    )
    return _mapping(
        campaign.get("campaign_resources"),
        "record.agent_view.tool_json.campaign_state.campaign_resources",
    )


def _record_index(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": _step(record, index),
            "record_sha256": canonical_json_sha256(record),
        }
        for index, record in enumerate(records, start=1)
    ]


def _canonical_resource_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the runtime action aliases before rebuilding resource events.

    Participant trajectories retain the action selected by the agent.  The
    environment's action codec may canonicalize aliases (for example,
    ``amount_mol`` to ``catalyst_amount_mol`` for ``add_catalyst``) before the
    campaign ledger sees them.  Replay must use that same canonical payload;
    otherwise a valid runtime event can fail resource replay even though the
    physical and public ledgers are internally consistent.  Truly malformed
    actions are left untouched so the replay remains fail-closed.
    """

    raw = dict(action)
    try:
        return _RESOURCE_ACTION_CODEC.canonicalize(raw)
    except (IndexError, TypeError, ValueError, OverflowError):
        return raw


def replay_work_ii_campaign_resources(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild every participant resource event from its public two-phase receipt."""

    if not records:
        raise WorkIIProcessProfileError("resource replay requires participant records")
    first_card = _mapping(records[0].get("campaign_resource_card"), "campaign resource card")
    for record in records:
        if _mapping(record.get("campaign_resource_card"), "campaign resource card") != first_card:
            raise WorkIIProcessProfileError("campaign resource card changed within one cell")
    try:
        card = CampaignResourceCard.from_dict(first_card)
        ledger = CampaignResourceLedger(card)
    except (KeyError, TypeError, ValueError) as error:
        raise WorkIIProcessProfileError(f"invalid campaign resource card: {error}") from error
    card_sha256 = card.card_sha256
    for record in records:
        if record.get("campaign_resource_card_sha256") != card_sha256:
            raise WorkIIProcessProfileError(
                "participant record campaign resource-card binding mismatch"
            )

    errors: list[str] = []
    seen_events: set[str] = set()
    resource_event_steps: list[int] = []
    non_campaign_record_steps: list[int] = []
    recorded_ledger_sha256: str | None = None
    for index, record in enumerate(records, start=1):
        _participant_role(record)
        step = _step(record, index)
        try:
            public = _public_resource_state(record)
            receipt = _mapping(public.get("latest_receipt"), "latest resource receipt")
            event_id = receipt.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise WorkIIProcessProfileError("latest resource receipt lacks event_id")
            if event_id in seen_events:
                if record.get("transaction_status") == "not_executed_resource_limit":
                    non_campaign_record_steps.append(step)
                    continue
                raise WorkIIProcessProfileError("resource event_id is duplicated")
            seen_events.add(event_id)
            resource_event_steps.append(step)
            raw_action = _mapping(record.get("action"), "resource event action")
            action = _canonical_resource_action(raw_action)
            recorded_preflight = _mapping(receipt.get("preflight"), "resource preflight")
            proposed = _mapping(
                recorded_preflight.get("proposed_delta"),
                "resource proposed delta",
            )
            starts_vessel = proposed.get("vessel_starts") == 1
            replayed_preflight = ledger.preflight(
                event_id,
                action,
                starts_vessel=starts_vessel,
            ).to_dict()
            if replayed_preflight != dict(recorded_preflight):
                errors.append(f"step {step}: resource preflight mismatch")

            recorded_delta = _mapping(receipt.get("outcome_delta"), "resource outcome delta")
            report_only = _mapping(
                recorded_delta.get("report_only"),
                "resource outcome report-only delta",
            )
            outcome = {
                "transaction_status": receipt.get(
                    "transaction_status",
                    record.get("transaction_status"),
                ),
                "campaign_resource_report_delta": dict(report_only),
            }
            replayed_delta = ledger.record_outcome(
                event_id,
                action,
                outcome,
                starts_vessel=starts_vessel,
            ).to_dict()
            if replayed_delta != dict(recorded_delta):
                errors.append(f"step {step}: resource outcome delta mismatch")
            expected_committed = record.get("transaction_status") == "committed"
            if receipt.get("operation_committed") is not expected_committed:
                errors.append(f"step {step}: resource committed flag mismatch")
            if receipt.get("transaction_status") != record.get("transaction_status"):
                errors.append(f"step {step}: resource transaction status mismatch")
            expected_rejected = recorded_preflight.get("allowed") is not True
            if receipt.get("rejected") is not expected_rejected:
                errors.append(f"step {step}: resource rejection flag mismatch")
            replayed_snapshot = ledger.snapshot()
            recorded_ledger_sha256 = str(public.get("ledger_sha256", ""))
            if replayed_snapshot["ledger_sha256"] != recorded_ledger_sha256:
                errors.append(f"step {step}: resource ledger hash mismatch")
            if public.get("state") != replayed_snapshot["state"]:
                errors.append(f"step {step}: resource state mismatch")
            if public.get("last_event_id") != event_id:
                errors.append(f"step {step}: resource last-event binding mismatch")
        except (
            CampaignResourceIntegrityError,
            KeyError,
            TypeError,
            ValueError,
            WorkIIProcessProfileError,
        ) as error:
            errors.append(f"step {step}: {error}")

    snapshot = ledger.snapshot()
    report: dict[str, Any] = {
        "schema_version": WORK_II_RESOURCE_REPLAY_VERSION,
        "status": "passed" if not errors else "failed",
        "participant_record_count": len(records),
        "resource_event_count": len(resource_event_steps),
        "resource_event_steps": resource_event_steps,
        "non_campaign_record_steps": non_campaign_record_steps,
        "campaign_resource_card_sha256": card_sha256,
        "recorded_ledger_sha256": recorded_ledger_sha256,
        "rebuilt_ledger_sha256": snapshot["ledger_sha256"],
        "rebuilt_state": snapshot["state"],
        "errors": errors,
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def _walk_visible(
    value: Any,
    *,
    path: str,
    forbidden_values: set[str],
    leaks: list[dict[str, str]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if str(key).lower() in _FORBIDDEN_VISIBLE_KEYS:
                leaks.append({"path": child_path, "reason": "forbidden_key"})
            _walk_visible(
                child,
                path=child_path,
                forbidden_values=forbidden_values,
                leaks=leaks,
            )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, child in enumerate(value):
            _walk_visible(
                child,
                path=f"{path}/{index}",
                forbidden_values=forbidden_values,
                leaks=leaks,
            )
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in _FORBIDDEN_PROTOCOL_TOKENS):
            leaks.append({"path": path, "reason": "forbidden_protocol_token"})
        if value in forbidden_values:
            leaks.append({"path": path, "reason": "forbidden_identity_value"})


def audit_work_ii_hidden_boundary(
    records: Sequence[Mapping[str, Any]],
    *,
    hidden_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit only participant-visible projections for evaluator/private leakage."""

    forbidden_values = {
        value
        for value in (
            str(item) for item in (hidden_identity or {}).values() if isinstance(item, str)
        )
        if len(value) >= 8
    }
    leaks: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        try:
            _participant_role(record)
        except WorkIIProcessProfileError as error:
            leaks.append({"path": f"record-{index}", "reason": str(error)})
        for field in _VISIBLE_RECORD_FIELDS:
            if field in record:
                _walk_visible(
                    record[field],
                    path=f"record-{index}/{field}",
                    forbidden_values=forbidden_values,
                    leaks=leaks,
                )
    report: dict[str, Any] = {
        "schema_version": WORK_II_HIDDEN_BOUNDARY_AUDIT_VERSION,
        "status": "passed" if not leaks else "failed",
        "participant_record_count": len(records),
        "visible_projection_fields": list(_VISIBLE_RECORD_FIELDS),
        "forbidden_visible_keys": sorted(_FORBIDDEN_VISIBLE_KEYS),
        "evaluator_owned_operation_count": 0,
        "leak_count": len(leaks),
        "leaks": leaks,
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def _trajectory_values(scores: Sequence[float]) -> dict[str, dict[str, Any]]:
    if not scores:
        return {
            metric_id: {
                "value": None,
                "calculation": {},
                "registered_denominator_count": 0,
                "null_reason": "no completed final assay",
            }
            for metric_id in (
                "global_best_discovery_fraction",
                "online_incumbent_retention_rate",
                "maximum_absolute_incumbent_drawdown",
                "loss_episode_recovery_rate",
                "terminal_to_global_best_ratio",
            )
        }
    best = max(scores)
    best_index = next(
        index for index, score in enumerate(scores) if abs(score - best) <= _FLOAT_TOLERANCE
    )
    discovery_denominator = max(len(scores) - 1, 0)
    discovery = best_index / discovery_denominator if discovery_denominator else 0.0
    if len(scores) == 1:
        return {
            "global_best_discovery_fraction": {
                "value": discovery,
                "calculation": {
                    "first_global_best_zero_based_ordinal": best_index,
                    "normalizer": discovery_denominator,
                },
                "registered_denominator_count": 1,
                "null_reason": None,
            },
            "online_incumbent_retention_rate": {
                "value": None,
                "calculation": {},
                "registered_denominator_count": 0,
                "null_reason": "fewer than two completed final assays",
            },
            "maximum_absolute_incumbent_drawdown": {
                "value": None,
                "calculation": {},
                "registered_denominator_count": 0,
                "null_reason": "fewer than two completed final assays",
            },
            "loss_episode_recovery_rate": {
                "value": None,
                "calculation": {},
                "registered_denominator_count": 0,
                "null_reason": "no loss episode",
            },
            "terminal_to_global_best_ratio": {
                "value": scores[-1] / best if best > 0.0 else None,
                "calculation": {"terminal_score": scores[-1], "global_best_score": best},
                "registered_denominator_count": 1,
                "null_reason": None if best > 0.0 else "global best score is not positive",
            },
        }

    incumbent = scores[0]
    retained_count = 0
    drawdowns: list[float] = []
    loss_episodes = 0
    recovered = 0
    recovery_threshold: float | None = None
    for score in scores[1:]:
        threshold = _RETENTION_FRACTION * incumbent
        retained = score + _FLOAT_TOLERANCE >= threshold
        retained_count += int(retained)
        drawdowns.append(max(0.0, incumbent - score))
        if recovery_threshold is not None:
            if score + _FLOAT_TOLERANCE >= recovery_threshold:
                recovered += 1
                recovery_threshold = None
        elif not retained:
            loss_episodes += 1
            recovery_threshold = threshold
        if score > incumbent + _FLOAT_TOLERANCE:
            incumbent = score
    opportunities = len(scores) - 1
    return {
        "global_best_discovery_fraction": {
            "value": discovery,
            "calculation": {
                "first_global_best_zero_based_ordinal": best_index,
                "normalizer": discovery_denominator,
            },
            "registered_denominator_count": len(scores),
            "null_reason": None,
        },
        "online_incumbent_retention_rate": {
            "value": retained_count / opportunities,
            "calculation": {
                "retained_count": retained_count,
                "retention_opportunity_count": opportunities,
                "retention_fraction": _RETENTION_FRACTION,
            },
            "registered_denominator_count": opportunities,
            "null_reason": None,
        },
        "maximum_absolute_incumbent_drawdown": {
            "value": max(drawdowns),
            "calculation": {"retention_opportunity_count": opportunities},
            "registered_denominator_count": opportunities,
            "null_reason": None,
        },
        "loss_episode_recovery_rate": {
            "value": recovered / loss_episodes if loss_episodes else None,
            "calculation": {
                "recovered_loss_episode_count": recovered,
                "loss_episode_count": loss_episodes,
            },
            "registered_denominator_count": loss_episodes,
            "null_reason": None if loss_episodes else "no loss episode",
        },
        "terminal_to_global_best_ratio": {
            "value": scores[-1] / best if best > 0.0 else None,
            "calculation": {"terminal_score": scores[-1], "global_best_score": best},
            "registered_denominator_count": len(scores),
            "null_reason": None if best > 0.0 else "global best score is not positive",
        },
    }


def build_work_ii_process_profile(
    records: Sequence[Mapping[str, Any]],
    resource_replay: Mapping[str, Any],
    *,
    planned_experiment_count: int,
    maximum_experiment_count: int | None = None,
    terminal_state: str,
) -> dict[str, Any]:
    """Build the 19-coordinate participant-only discovery-campaign profile."""

    if planned_experiment_count <= 0:
        raise WorkIIProcessProfileError("planned_experiment_count must be positive")
    maximum_experiment_count = (
        planned_experiment_count
        if maximum_experiment_count is None
        else maximum_experiment_count
    )
    if maximum_experiment_count < planned_experiment_count:
        raise WorkIIProcessProfileError(
            "maximum_experiment_count must cover planned_experiment_count"
        )
    if terminal_state not in _TERMINAL_STATES:
        raise WorkIIProcessProfileError("terminal_state is outside the formal contract")
    if resource_replay.get("status") != "passed":
        raise WorkIIProcessProfileError("process profile requires passed resource replay")
    allowed_steps = {
        _nonnegative_int(item, "resource event step")
        for item in resource_replay.get("resource_event_steps", [])
    }
    participant_records = [
        record
        for index, record in enumerate(records, start=1)
        if _step(record, index) in allowed_steps
    ]
    for record in participant_records:
        _participant_role(record)

    closed_lifecycles: list[dict[str, Any]] = []
    open_rows: list[Mapping[str, Any]] = []
    for record in participant_records:
        open_rows.append(record)
        operation = _operation(record)
        instrument = _instrument(record)
        terminal_kind: str | None = None
        if _committed(record) and operation == "measure" and instrument == "final_assay":
            terminal_kind = "final_assay"
        elif _committed(record) and operation == "discard_batch":
            terminal_kind = "discard"
        if terminal_kind is not None:
            closed_lifecycles.append(
                {
                    "terminal_kind": terminal_kind,
                    "rows": open_rows,
                }
            )
            open_rows = []
    if len(closed_lifecycles) > maximum_experiment_count:
        raise WorkIIProcessProfileError("closed lifecycle count exceeds the finite start capacity")

    terminal_steps = [_step(lifecycle["rows"][-1], 1) for lifecycle in closed_lifecycles]
    final_lifecycles = [
        item for item in closed_lifecycles if item["terminal_kind"] == "final_assay"
    ]
    discard_lifecycles = [item for item in closed_lifecycles if item["terminal_kind"] == "discard"]
    measured_lifecycles = 0
    nonfinal_measurement_steps: list[int] = []
    first_measurement_fractions: list[float] = []
    continued_lifecycles = 0
    post_measure_process_steps: list[int] = []
    for lifecycle in closed_lifecycles:
        rows = lifecycle["rows"]
        measurements = [
            index
            for index, record in enumerate(rows)
            if _committed(record)
            and _operation(record) == "measure"
            and _instrument(record) != "final_assay"
        ]
        if not measurements:
            continue
        measured_lifecycles += 1
        nonfinal_measurement_steps.extend(_step(rows[index], index + 1) for index in measurements)
        first_index = measurements[0]
        first_measurement_fractions.append(first_index / len(rows))
        later = [
            (row_index, record)
            for row_index, record in enumerate(rows[first_index + 1 : -1], first_index + 1)
            if _committed(record) and _operation(record) not in _PROCESS_EXCLUSIONS
        ]
        if later:
            continued_lifecycles += 1
            post_measure_process_steps.extend(
                _step(record, row_index + 1) for row_index, record in later
            )

    closed = len(closed_lifecycles)
    final_count = len(final_lifecycles)
    discard_count = len(discard_lifecycles)
    scores = [
        _finite(trajectory_value(item["rows"][-1], "leaderboard_score", None), "leaderboard score")
        for item in final_lifecycles
    ]
    final_score_steps = [
        _step(item["rows"][-1], index + 1) for index, item in enumerate(final_lifecycles)
    ]
    state = _mapping(resource_replay.get("rebuilt_state"), "rebuilt resource state")
    report_only = _mapping(state.get("report_only"), "rebuilt resource report_only")
    attempted = _nonnegative_int(state.get("operation_attempts"), "operation attempts")
    committed_steps = [
        _step(record, index)
        for index, record in enumerate(participant_records, start=1)
        if _committed(record)
    ]
    all_steps = [_step(record, index) for index, record in enumerate(participant_records, 1)]

    values: dict[str, dict[str, Any]] = {
        "closed_lifecycle_fraction": {
            "value": closed / maximum_experiment_count,
            "calculation": {
                "closed_lifecycle_count": closed,
                "maximum_lifecycle_count": maximum_experiment_count,
            },
            "registered_denominator_count": maximum_experiment_count,
            "null_reason": None,
            "source_steps": terminal_steps,
        },
        "assay_fraction": {
            "value": final_count / closed if closed else None,
            "calculation": {
                "final_assay_count": final_count,
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": terminal_steps,
        },
        "discard_fraction": {
            "value": discard_count / closed if closed else None,
            "calculation": {
                "discard_count": discard_count,
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": terminal_steps,
        },
        "measured_lifecycle_fraction": {
            "value": measured_lifecycles / closed if closed else None,
            "calculation": {
                "measured_lifecycle_count": measured_lifecycles,
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": sorted({*terminal_steps, *nonfinal_measurement_steps}),
        },
        "nonfinal_instrument_uses_per_closed_lifecycle": {
            "value": len(nonfinal_measurement_steps) / closed if closed else None,
            "calculation": {
                "nonfinal_instrument_use_count": len(nonfinal_measurement_steps),
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": sorted({*terminal_steps, *nonfinal_measurement_steps}),
        },
        "mean_first_measurement_operation_fraction": {
            "value": (
                sum(first_measurement_fractions) / len(first_measurement_fractions)
                if first_measurement_fractions
                else None
            ),
            "calculation": {
                "first_measurement_operation_fractions": first_measurement_fractions,
                "measured_lifecycle_count": measured_lifecycles,
            },
            "registered_denominator_count": measured_lifecycles,
            "null_reason": (
                None
                if measured_lifecycles
                else "no closed lifecycle contains a committed non-final measurement"
            ),
            "source_steps": sorted({*terminal_steps, *nonfinal_measurement_steps}),
        },
        "continued_after_measurement_fraction": {
            "value": continued_lifecycles / closed if closed else None,
            "calculation": {
                "continued_after_measurement_lifecycle_count": continued_lifecycles,
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": sorted(
                {
                    *terminal_steps,
                    *nonfinal_measurement_steps,
                    *post_measure_process_steps,
                }
            ),
        },
        "post_measure_process_operations_per_closed_lifecycle": {
            "value": len(post_measure_process_steps) / closed if closed else None,
            "calculation": {
                "post_measure_process_operation_count": len(post_measure_process_steps),
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": sorted(
                {
                    *terminal_steps,
                    *nonfinal_measurement_steps,
                    *post_measure_process_steps,
                }
            ),
        },
        "threshold_eligible_fraction": {
            "value": None,
            "calculation": {},
            "registered_denominator_count": 0,
            "null_reason": "no preregistered Work I diagnostic threshold in Work II",
            "source_steps": [],
        },
        "threshold_decision_concordance": {
            "value": None,
            "calculation": {},
            "registered_denominator_count": 0,
            "null_reason": "no preregistered Work I diagnostic threshold in Work II",
            "source_steps": [],
        },
        "attempted_operations_per_closed_lifecycle": {
            "value": attempted / closed if closed else None,
            "calculation": {
                "operation_attempt_count": attempted,
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": all_steps,
        },
        "committed_operations_per_closed_lifecycle": {
            "value": len(committed_steps) / closed if closed else None,
            "calculation": {
                "committed_operation_count": len(committed_steps),
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": committed_steps,
        },
        "total_cost_per_closed_lifecycle": {
            "value": (
                _finite(report_only.get("physical_cost"), "physical cost") / closed
                if closed
                else None
            ),
            "calculation": {
                "total_physical_cost": _finite(report_only.get("physical_cost"), "physical cost"),
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": all_steps,
        },
        "total_risk_per_closed_lifecycle": {
            "value": (
                _finite(report_only.get("accumulated_risk"), "accumulated risk") / closed
                if closed
                else None
            ),
            "calculation": {
                "total_accumulated_risk": _finite(
                    report_only.get("accumulated_risk"), "accumulated risk"
                ),
                "closed_lifecycle_count": closed,
            },
            "registered_denominator_count": closed,
            "null_reason": None if closed else "no lifecycle is closed",
            "source_steps": all_steps,
        },
    }
    for metric_id, payload in _trajectory_values(scores).items():
        values[metric_id] = {**payload, "source_steps": final_score_steps}

    axes: dict[str, dict[str, Any]] = {str(axis["axis_id"]): {} for axis in AXES}
    for spec in METRICS:
        payload = values[spec.metric_id]
        axes[spec.axis_id][spec.metric_id] = {
            "value": payload["value"],
            "unit": spec.unit,
            "registered_denominator_id": spec.denominator,
            "registered_denominator_count": payload["registered_denominator_count"],
            "applicable": payload["null_reason"] is None,
            "null_reason": payload["null_reason"],
            "calculation": payload["calculation"],
            "source_steps": payload["source_steps"],
        }

    index = _record_index(participant_records)
    profile: dict[str, Any] = {
        "schema_version": WORK_II_PROCESS_PROFILE_VERSION,
        "source_work_i_profile_contract_sha256": profile_contract_sha256(),
        "terminal_state": terminal_state,
        "participant_only": True,
        "evaluator_owned_operation_count": 0,
        "planned_experiment_count": planned_experiment_count,
        "maximum_experiment_count": maximum_experiment_count,
        "counts": {
            "participant_record_count": len(participant_records),
            "participant_operation_attempt_count": attempted,
            "committed_operation_count": len(committed_steps),
            "closed_lifecycle_count": closed,
            "final_assay_count": final_count,
            "discard_count": discard_count,
            "open_lifecycle_record_count": len(open_rows),
            "measured_lifecycle_count": measured_lifecycles,
        },
        "construct_axes": axes,
        "endpoint_context": {
            "mean_assayed_score": {
                "value": sum(scores) / len(scores) if scores else None,
                "unit": next(
                    item.unit for item in ENDPOINT_CONTEXT if item.metric_id == "mean_assayed_score"
                ),
                "denominator_count": final_count,
                "source_steps": final_score_steps,
            },
            "best_assayed_score": {
                "value": max(scores) if scores else None,
                "unit": next(
                    item.unit for item in ENDPOINT_CONTEXT if item.metric_id == "best_assayed_score"
                ),
                "denominator_count": final_count,
                "source_steps": final_score_steps,
            },
        },
        "record_index": index,
        "participant_trajectory_sha256": canonical_json_sha256(participant_records),
        "resource_replay_report_sha256": resource_replay.get("report_sha256"),
    }
    profile["profile_sha256"] = _self_hash(profile, "profile_sha256")
    errors = validate_work_ii_process_profile(profile)
    if errors:
        raise WorkIIProcessProfileError("; ".join(errors))
    return profile


def validate_work_ii_process_profile(profile: Mapping[str, Any]) -> list[str]:
    """Validate the adapted metric surface, trace index, and self-hash."""

    errors: list[str] = []
    if profile.get("schema_version") != WORK_II_PROCESS_PROFILE_VERSION:
        errors.append("unexpected process profile schema")
    if profile.get("profile_sha256") != _self_hash(profile, "profile_sha256"):
        errors.append("process profile self-hash mismatch")
    if profile.get("source_work_i_profile_contract_sha256") != profile_contract_sha256():
        errors.append("process profile does not bind the frozen Work I contract")
    if profile.get("participant_only") is not True:
        errors.append("process profile is not participant-only")
    if profile.get("evaluator_owned_operation_count") != 0:
        errors.append("evaluator operations entered the participant profile")
    planned = profile.get("planned_experiment_count")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
        errors.append("process profile planned experiment count is invalid")
    maximum = profile.get("maximum_experiment_count", planned)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not isinstance(planned, int)
        or maximum < planned
    ):
        errors.append("process profile maximum experiment count is invalid")
    if profile.get("terminal_state") not in _TERMINAL_STATES:
        errors.append("process profile terminal state is invalid")
    counts = profile.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("process profile counts are missing")
        counts = {}
    count_fields = (
        "participant_record_count",
        "participant_operation_attempt_count",
        "committed_operation_count",
        "closed_lifecycle_count",
        "final_assay_count",
        "discard_count",
        "open_lifecycle_record_count",
        "measured_lifecycle_count",
    )
    if any(
        isinstance(counts.get(field), bool)
        or not isinstance(counts.get(field), int)
        or int(counts.get(field, -1)) < 0
        for field in count_fields
    ):
        errors.append("process profile counts contain an invalid denominator")
    else:
        if (
            counts["final_assay_count"] + counts["discard_count"]
            != counts["closed_lifecycle_count"]
        ):
            errors.append("process profile terminal counts do not reconcile")
        if isinstance(maximum, int) and counts["closed_lifecycle_count"] > maximum:
            errors.append("process profile closed count exceeds finite start capacity")
        if counts["committed_operation_count"] > counts["participant_operation_attempt_count"]:
            errors.append("process profile committed count exceeds attempts")
        if counts["measured_lifecycle_count"] > counts["closed_lifecycle_count"]:
            errors.append("process profile measured count exceeds closed lifecycles")
    axes = profile.get("construct_axes")
    if not isinstance(axes, Mapping):
        errors.append("process profile axes are missing")
        return errors
    expected_axes = {str(axis["axis_id"]) for axis in AXES}
    if set(axes) != expected_axes:
        errors.append("process profile axis surface differs from Work I")
    observed_metrics: set[str] = set()
    specs = {spec.metric_id: spec for spec in METRICS}
    for axis_id, raw_axis in axes.items():
        if not isinstance(raw_axis, Mapping):
            errors.append(f"process profile axis {axis_id} is malformed")
            continue
        observed_metrics.update(str(key) for key in raw_axis)
        for metric_id, raw_metric in raw_axis.items():
            if not isinstance(raw_metric, Mapping):
                errors.append(f"process coordinate {metric_id} is malformed")
                continue
            value = raw_metric.get("value")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                errors.append(f"process coordinate {metric_id} is non-finite")
            spec = specs.get(str(metric_id))
            if spec is not None:
                if raw_metric.get("unit") != spec.unit:
                    errors.append(f"process coordinate {metric_id} unit mismatch")
                if raw_metric.get("registered_denominator_id") != spec.denominator:
                    errors.append(f"process coordinate {metric_id} denominator identity mismatch")
                if (
                    value is not None
                    and spec.lower_bound is not None
                    and float(value) < (spec.lower_bound - _FLOAT_TOLERANCE)
                ):
                    errors.append(f"process coordinate {metric_id} is below its bound")
                if (
                    value is not None
                    and spec.upper_bound is not None
                    and float(value) > (spec.upper_bound + _FLOAT_TOLERANCE)
                ):
                    errors.append(f"process coordinate {metric_id} is above its bound")
            denominator_count = raw_metric.get("registered_denominator_count")
            if (
                isinstance(denominator_count, bool)
                or not isinstance(denominator_count, int)
                or denominator_count < 0
            ):
                errors.append(f"process coordinate {metric_id} denominator is invalid")
            if raw_metric.get("applicable") is (value is None):
                errors.append(f"process coordinate {metric_id} applicability mismatch")
            if not isinstance(raw_metric.get("source_steps"), list):
                errors.append(f"process coordinate {metric_id} lacks source steps")
    if observed_metrics != {spec.metric_id for spec in METRICS}:
        errors.append("process profile must contain exactly 19 Work I coordinates")
    for metric_id in ("threshold_eligible_fraction", "threshold_decision_concordance"):
        raw_metric = axes.get("evidence_conditioned_action", {}).get(metric_id)
        if not isinstance(raw_metric, Mapping) or (
            raw_metric.get("value") is not None
            or raw_metric.get("applicable") is not False
            or raw_metric.get("registered_denominator_count") != 0
        ):
            errors.append(f"Work II coordinate {metric_id} must remain inapplicable")
    index = profile.get("record_index")
    if not isinstance(index, list):
        errors.append("process profile record index is missing")
    else:
        indexed_steps = {item.get("step") for item in index if isinstance(item, Mapping)}
        if len(indexed_steps) != len(index) or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in indexed_steps
        ):
            errors.append("process profile record index has invalid or duplicate steps")
        if isinstance(counts, Mapping) and len(index) != counts.get("participant_record_count"):
            errors.append("process profile record index count mismatch")
        if any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("record_sha256"), str)
            or len(item["record_sha256"]) != 64
            for item in index
        ):
            errors.append("process profile record index has an invalid record hash")
        for raw_axis in axes.values():
            if not isinstance(raw_axis, Mapping):
                continue
            for raw_metric in raw_axis.values():
                if isinstance(raw_metric, Mapping) and any(
                    step not in indexed_steps for step in raw_metric.get("source_steps", [])
                ):
                    errors.append("process coordinate references an unknown participant step")
    return list(dict.fromkeys(errors))


def build_work_ii_execution_audit(
    records: Sequence[Mapping[str, Any]],
    exact_replay: Mapping[str, Any],
    process_profile: Mapping[str, Any],
    resource_replay: Mapping[str, Any],
    hidden_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine physical replay, resource replay, profile, and leakage gates."""

    checks = {
        "physical_exact_replay": exact_replay.get("verified") is True
        and exact_replay.get("checked_steps") == len(records)
        and exact_replay.get("mismatches") == [],
        "resource_exact_replay": resource_replay.get("status") == "passed"
        and resource_replay.get("participant_record_count") == len(records),
        "hidden_boundary": hidden_boundary.get("status") == "passed"
        and hidden_boundary.get("evaluator_owned_operation_count") == 0,
        "process_profile": not validate_work_ii_process_profile(process_profile),
        "participant_denominator_excludes_evaluator": process_profile.get(
            "evaluator_owned_operation_count"
        )
        == 0,
    }
    report: dict[str, Any] = {
        "schema_version": WORK_II_EXECUTION_AUDIT_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "participant_record_count": len(records),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "physical_exact_replay_sha256": canonical_json_sha256(exact_replay),
        "resource_replay_report_sha256": resource_replay.get("report_sha256"),
        "hidden_boundary_report_sha256": hidden_boundary.get("report_sha256"),
        "process_profile_sha256": process_profile.get("profile_sha256"),
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def build_work_ii_execution_artifacts(
    records: Sequence[Mapping[str, Any]],
    exact_replay: Mapping[str, Any],
    *,
    planned_experiment_count: int,
    maximum_experiment_count: int | None = None,
    terminal_state: str,
    hidden_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed while retaining all independently constructible audit artifacts."""

    hidden_boundary = audit_work_ii_hidden_boundary(
        records,
        hidden_identity=hidden_identity,
    )
    construction_errors: list[str] = []
    try:
        resource_replay = replay_work_ii_campaign_resources(records)
    except WorkIIProcessProfileError as error:
        construction_errors.append(str(error))
        resource_replay = {
            "schema_version": WORK_II_RESOURCE_REPLAY_VERSION,
            "status": "failed",
            "participant_record_count": len(records),
            "resource_event_count": 0,
            "resource_event_steps": [],
            "non_campaign_record_steps": [],
            "errors": [str(error)],
        }
        resource_replay["report_sha256"] = _self_hash(
            resource_replay,
            "report_sha256",
        )
    process_profile: dict[str, Any] | None = None
    try:
        process_profile = build_work_ii_process_profile(
            records,
            resource_replay,
            planned_experiment_count=planned_experiment_count,
            maximum_experiment_count=maximum_experiment_count,
            terminal_state=terminal_state,
        )
    except WorkIIProcessProfileError as error:
        construction_errors.append(str(error))

    if process_profile is not None:
        execution_audit = build_work_ii_execution_audit(
            records,
            exact_replay,
            process_profile,
            resource_replay,
            hidden_boundary,
        )
    else:
        checks = {
            "physical_exact_replay": exact_replay.get("verified") is True
            and exact_replay.get("checked_steps") == len(records)
            and exact_replay.get("mismatches") == [],
            "resource_exact_replay": resource_replay.get("status") == "passed",
            "hidden_boundary": hidden_boundary.get("status") == "passed",
            "process_profile": False,
            "participant_denominator_excludes_evaluator": False,
        }
        execution_audit = {
            "schema_version": WORK_II_EXECUTION_AUDIT_VERSION,
            "status": "failed",
            "passed": False,
            "participant_record_count": len(records),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "construction_errors": construction_errors,
            "physical_exact_replay_sha256": canonical_json_sha256(exact_replay),
            "resource_replay_report_sha256": resource_replay.get("report_sha256"),
            "hidden_boundary_report_sha256": hidden_boundary.get("report_sha256"),
            "process_profile_sha256": None,
        }
        execution_audit["report_sha256"] = _self_hash(
            execution_audit,
            "report_sha256",
        )
    return {
        "process_profile": process_profile,
        "resource_replay": resource_replay,
        "hidden_boundary_audit": hidden_boundary,
        "execution_audit": execution_audit,
    }


__all__ = [
    "WORK_II_EXECUTION_AUDIT_VERSION",
    "WORK_II_HIDDEN_BOUNDARY_AUDIT_VERSION",
    "WORK_II_PROCESS_PROFILE_VERSION",
    "WORK_II_RESOURCE_REPLAY_VERSION",
    "WorkIIProcessProfileError",
    "audit_work_ii_hidden_boundary",
    "build_work_ii_execution_artifacts",
    "build_work_ii_execution_audit",
    "build_work_ii_process_profile",
    "replay_work_ii_campaign_resources",
    "validate_work_ii_process_profile",
]
