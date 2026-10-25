"""Participant outcome names and compact readiness compatibility helpers."""

from __future__ import annotations

from typing import Any, Literal, cast

ParticipantOutcomeId = Literal["O1", "O2", "O3", "O4", "O5"]
ResultStatus = Literal["not_executed", "positive", "negative", "mixed", "inconclusive"]

PARTICIPANT_OUTCOMES: dict[ParticipantOutcomeId, dict[str, str]] = {
    "O1": {
        "name": "Change detection",
        "question": "Did the Agent detect and attribute failure of the old world model?",
    },
    "O2": {
        "name": "Feedback use",
        "question": "Did experimental feedback causally change belief and experiment choice?",
    },
    "O3": {
        "name": "Adaptation and recovery",
        "question": "Did the Agent recover task performance after the change?",
    },
    "O4": {
        "name": "Procedural autonomy",
        "question": "Could the Agent manage the operation-level experiment lifecycle?",
    },
    "O5": {
        "name": "Resource efficiency",
        "question": (
            "What experiments, measurements, risk, calls, tokens, cost, and time were used?"
        ),
    },
}

HISTORICAL_PARTICIPANT_GATE_TO_OUTCOME: dict[str, ParticipantOutcomeId] = {
    "B": "O1",
    "C": "O2",
    "D": "O3",
    "E": "O4",
}


def participant_outcome_id(legacy_gate_or_outcome: str) -> ParticipantOutcomeId:
    """Resolve old Gate B-E labels while new artifacts migrate to O1-O5."""

    normalized = legacy_gate_or_outcome.strip().upper().removeprefix("GATE ")
    if normalized in PARTICIPANT_OUTCOMES:
        return cast(ParticipantOutcomeId, normalized)
    try:
        return HISTORICAL_PARTICIPANT_GATE_TO_OUTCOME[normalized]
    except KeyError as error:
        raise ValueError(f"unknown participant outcome label: {legacy_gate_or_outcome}") from error


def compact_readiness(
    *,
    environment_ready: bool,
    methods_frozen: bool,
    execution_complete: bool,
    release_ready: bool,
    result_status: ResultStatus = "not_executed",
) -> dict[str, Any]:
    """Build the compressed project status without performance-as-gate semantics."""

    values = {
        "environment_ready": environment_ready,
        "methods_frozen": methods_frozen,
        "execution_complete": execution_complete,
        "release_ready": release_ready,
    }
    if not all(isinstance(value, bool) for value in values.values()):
        raise ValueError("readiness fields must be booleans")
    if methods_frozen and not environment_ready:
        raise ValueError("methods cannot freeze before environment readiness")
    if execution_complete and not methods_frozen:
        raise ValueError("execution cannot complete before methods are frozen")
    if release_ready and not execution_complete:
        raise ValueError("release cannot be ready before execution is complete")
    allowed_statuses = {"not_executed", "positive", "negative", "mixed", "inconclusive"}
    if result_status not in allowed_statuses:
        raise ValueError(f"unsupported result_status: {result_status}")
    if execution_complete == (result_status == "not_executed"):
        raise ValueError("result_status must agree with execution_complete")
    return {
        "schema_version": "chemworld-compact-readiness-0.1",
        **values,
        "result_status": result_status,
        "participant_performance_controls_release_readiness": False,
    }


__all__ = [
    "HISTORICAL_PARTICIPANT_GATE_TO_OUTCOME",
    "PARTICIPANT_OUTCOMES",
    "ParticipantOutcomeId",
    "ResultStatus",
    "compact_readiness",
    "participant_outcome_id",
]
