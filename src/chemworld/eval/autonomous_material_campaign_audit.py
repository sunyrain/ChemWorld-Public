"""Offline audit for paired autonomous material-information campaigns.

The intended matrix has five physical worlds and two information conditions:
opaque anonymous material identifiers and a nominal-property dossier.  This
reader is intentionally independent of the campaign runner.  It accepts a
small alias set for evolving artifact names, but it fails closed when a
pairing identity, resource-ledger snapshot, or exact replay receipt is absent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from chemworld.campaign_resources import (
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
)
from chemworld.data.logging import load_jsonl, to_builtin
from chemworld.data.records import trajectory_value
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)

AUTONOMOUS_MATERIAL_CAMPAIGN_AUDIT_VERSION = (
    "chemworld-autonomous-material-campaign-audit-0.3"
)

TRAJECTORY_RETENTION_FRACTION = 0.90
_SCORE_COMPARISON_TOLERANCE = 1e-12

OPAQUE_ARM = "opaque"
NOMINAL_ARM = "nominal"
EXPECTED_ARMS = (OPAQUE_ARM, NOMINAL_ARM)
DEFAULT_WORLD_SEEDS = (0, 1, 2, 3, 4)

_MATERIAL_FIELDS = ("solvent", "electrolyte_profile", "catalyst", "extractant")
_SETPOINT_FIELDS = ("potential_V", "current_mA", "electrolyte_profile")
_CONTROL_FIELDS = {
    "set_potential": _SETPOINT_FIELDS,
    "electrolyze": ("duration_s",),
}
_SCORE_COMPONENT_ALIASES = {
    "selective_product_yield": ("selective_product_yield", "yield"),
    "electrochemical_selectivity": (
        "electrochemical_selectivity",
        "selectivity",
    ),
    "electrochemical_conversion": (
        "electrochemical_conversion",
        "conversion",
    ),
    "faradaic_efficiency": ("faradaic_efficiency", "faradaic"),
    "transport_efficiency": ("transport_efficiency", "transport"),
    "ohmic_efficiency": ("ohmic_efficiency", "ohmic"),
    "energy_efficiency": ("energy_efficiency", "energy"),
    "safety_risk": ("safety_risk", "risk"),
    "cost": ("cost", "cost_signal"),
}

_RESOURCE_SNAPSHOT_KEYS = (
    "campaign_resource_ledger_snapshot",
    "campaign_resource_snapshot",
    "resource_ledger_snapshot",
    "campaign_resources",
)
_RESOURCE_PATH_KEYS = (
    "campaign_resource_ledger_path",
    "campaign_resource_snapshot_path",
    "resource_ledger_path",
    "resource_snapshot_path",
)
_REPLAY_KEYS = (
    "exact_replay",
    "replay_audit",
    "trajectory_replay",
    "postrun_replay",
)
_REPLAY_PATH_KEYS = (
    "exact_replay_path",
    "replay_audit_path",
    "trajectory_replay_path",
    "postrun_audit_path",
)
_EXPERIMENT_TOOL_INTEGRITY_KEYS = (
    "experiment_tool_integrity_verified_after_session",
    "mcp_tool_integrity_verified_after_session",
    "lab_tool_integrity_verified_after_session",
)


class AutonomousMaterialCampaignAuditError(ValueError):
    """Raised when the matrix cannot support a paired audited result."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise AutonomousMaterialCampaignAuditError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutonomousMaterialCampaignAuditError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise AutonomousMaterialCampaignAuditError(
            f"{label} must contain a JSON object: {path}"
        )
    return payload


def _dig(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first_value(
    sources: Sequence[Mapping[str, Any]],
    paths: Sequence[str],
) -> Any:
    for source in sources:
        for path in paths:
            value = _dig(source, path)
            if value is not None:
                return value
    return None


def _required_value(
    sources: Sequence[Mapping[str, Any]],
    paths: Sequence[str],
    *,
    label: str,
    cell_id: str,
) -> Any:
    value = _first_value(sources, paths)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: missing required {label}; accepted fields: "
            + ", ".join(paths)
        )
    return value


def _arm_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("mode", value.get("condition", value.get("arm")))
    normalized = str(value or "").strip().lower().replace("-", "_")
    if any(token in normalized for token in ("nominal", "provided", "known")):
        return NOMINAL_ARM
    if any(token in normalized for token in ("opaque", "unknown", "blind")):
        return OPAQUE_ARM
    raise AutonomousMaterialCampaignAuditError(
        f"unsupported material-information arm: {value!r}"
    )


def _manifest_cells(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        manifest.get("cells"),
        manifest.get("runs"),
        _dig(manifest, "matrix.cells"),
        _dig(manifest, "execution.cells"),
    )
    raw_cells = next((item for item in candidates if isinstance(item, list)), None)
    if raw_cells is None:
        raise AutonomousMaterialCampaignAuditError(
            "matrix manifest is missing a cells/runs list"
        )
    cells: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            raise AutonomousMaterialCampaignAuditError(
                f"matrix cell {index} must be an object"
            )
        cells.append(dict(raw))
    return cells


def _resolve_path(
    value: str | Path,
    *,
    bases: Sequence[Path],
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return bases[0] / path


def _run_file(
    cell: Mapping[str, Any],
    run_dir: Path,
    manifest_dir: Path,
    *,
    explicit_keys: Sequence[str],
    conventional_names: Sequence[str],
    label: str,
) -> Path:
    explicit = _first_value((cell,), explicit_keys)
    if explicit is not None:
        path = _resolve_path(explicit, bases=(run_dir, manifest_dir))
        if not path.exists():
            raise AutonomousMaterialCampaignAuditError(
                f"{cell.get('cell_id', run_dir.name)}: missing {label}: {path}"
            )
        return path
    for name in conventional_names:
        path = run_dir / name
        if path.exists():
            return path
    raise AutonomousMaterialCampaignAuditError(
        f"{cell.get('cell_id', run_dir.name)}: missing {label} in {run_dir}; "
        f"tried {', '.join(conventional_names)}"
    )


def _config_payload_sha256(config: Mapping[str, Any]) -> str:
    payload = dict(config)
    payload.pop("config_sha256", None)
    return canonical_json_sha256(payload)


def _validate_config_hashes(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    computed = _config_payload_sha256(config)
    declared = config.get("config_sha256")
    if declared is not None and declared != computed:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run_config config_sha256 mismatch"
        )
    authoritative = str(declared or computed)
    summary_hash = summary.get("config_sha256")
    if summary_hash is not None and str(summary_hash) != authoritative:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run_summary/run_config config_sha256 mismatch"
        )
    cell_hash = cell.get("config_sha256")
    if cell_hash is not None and str(cell_hash) != authoritative:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: matrix/run_config config_sha256 mismatch"
        )
    return authoritative


def _validate_trajectory_hash(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    trajectory_path: Path,
) -> str:
    observed = file_sha256(trajectory_path)
    for label, expected in (
        ("run_summary", summary.get("trajectory_sha256")),
        ("matrix", cell.get("trajectory_sha256")),
    ):
        if expected is not None and str(expected) != observed:
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: {label}/trajectory sha256 mismatch"
            )
    return observed


def _constant_record_value(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[str],
    *,
    label: str,
    cell_id: str,
    required: bool = True,
) -> Any:
    present = [
        _first_value((record,), paths)
        for record in records
        if _first_value((record,), paths) is not None
    ]
    if not present:
        if required:
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: trajectory is missing {label}"
            )
        return None
    first = present[0]
    if any(value != first for value in present[1:]):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: trajectory {label} changes within the cell"
        )
    return first


def _resource_snapshot_from_sources(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: Path,
    manifest_dir: Path,
) -> tuple[dict[str, Any], str]:
    for source_name, source in (
        ("matrix cell", cell),
        ("run summary", summary),
        ("run config", config),
    ):
        for key in _RESOURCE_SNAPSHOT_KEYS:
            value = source.get(key)
            if isinstance(value, Mapping):
                snapshot = dict(value)
                if isinstance(snapshot.get("snapshot"), Mapping):
                    snapshot = dict(snapshot["snapshot"])
                return snapshot, f"embedded:{source_name}.{key}"
        for key in _RESOURCE_PATH_KEYS:
            value = source.get(key)
            if value is not None:
                path = _resolve_path(value, bases=(run_dir, manifest_dir))
                return (
                    _load_json_object(path, label="campaign resource snapshot"),
                    str(path),
                )
    for name in (
        "campaign_resource_ledger.json",
        "campaign_resource_snapshot.json",
        "resource_ledger.json",
    ):
        path = run_dir / name
        if path.exists():
            return _load_json_object(path, label="campaign resource snapshot"), str(path)
    raise AutonomousMaterialCampaignAuditError(
        f"{cell_id}: missing campaign resource ledger snapshot; expected an "
        f"embedded {_RESOURCE_SNAPSHOT_KEYS[0]} or one of "
        "campaign_resource_ledger.json/campaign_resource_snapshot.json"
    )


def _verified_flag(payload: Mapping[str, Any]) -> bool | None:
    candidates = (
        payload.get("verified"),
        payload.get("all_verified"),
        payload.get("exact_replay_verified"),
        _dig(payload, "replay.verified"),
        _dig(payload, "summary.all_verified"),
    )
    for value in candidates:
        if isinstance(value, bool):
            return value
    return None


def _replay_receipt_from_sources(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_dir: Path,
    manifest_dir: Path,
) -> tuple[dict[str, Any], str]:
    for source_name, source in (("matrix cell", cell), ("run summary", summary)):
        for key in _REPLAY_KEYS:
            value = source.get(key)
            if isinstance(value, Mapping):
                return dict(value), f"embedded:{source_name}.{key}"
        for key in _REPLAY_PATH_KEYS:
            value = source.get(key)
            if value is not None:
                path = _resolve_path(value, bases=(run_dir, manifest_dir))
                return _load_json_object(path, label="exact replay audit"), str(path)
    for name in ("exact_replay.json", "replay_audit.json", "postrun_audit.json"):
        path = run_dir / name
        if path.exists():
            return _load_json_object(path, label="exact replay audit"), str(path)
    raise AutonomousMaterialCampaignAuditError(
        f"{cell_id}: missing exact trajectory replay receipt; expected an "
        "embedded replay_audit/exact_replay object or replay_audit.json"
    )


def _status_committed(record: Mapping[str, Any]) -> bool:
    return str(
        _first_value(
            (record,),
            ("transaction_status", "environment_outcome.transaction_status"),
        )
        or ""
    ).lower() == "committed"


def _action(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("action", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _operation(record: Mapping[str, Any]) -> str:
    action = _action(record)
    return str(action.get("operation", record.get("operation_type", "unknown")))


def _instrument(record: Mapping[str, Any]) -> str | None:
    action = _action(record)
    raw = action.get("instrument", record.get("instrument"))
    return None if raw is None else str(raw)


def _leaderboard_score(record: Mapping[str, Any]) -> float | None:
    raw = _first_value(
        (record,),
        ("evaluation_outcome.leaderboard_score", "leaderboard_score"),
    )
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AutonomousMaterialCampaignAuditError(
            f"invalid final leaderboard score: {raw!r}"
        )
    return value


def _batch_indices(records: Sequence[Mapping[str, Any]]) -> list[int]:
    inferred = 0
    result: list[int] = []
    for record in records:
        raw = record.get("experiment_index")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            index = raw
            inferred = max(inferred, index)
        else:
            index = inferred
        result.append(index)
        final_assay_closed = (
            _instrument(record) == "final_assay" and _status_committed(record)
        )
        discard_closed = (
            _operation(record) == "discard_batch" and _status_committed(record)
        )
        if final_assay_closed or discard_closed:
            inferred = index + 1
    return result


def _running_best(scores: Sequence[float]) -> list[float]:
    best = 0.0
    curve: list[float] = []
    for score in scores:
        best = max(best, float(score))
        curve.append(best)
    return curve


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _score_components(record: Mapping[str, Any]) -> dict[str, float]:
    observation_sources = (
        trajectory_value(record, "observation", None),
        _dig(record, "agent_visible_observation.observation"),
        _dig(record, "evaluation_outcome.score_components"),
    )
    sources = [
        source for source in observation_sources if isinstance(source, Mapping)
    ]
    sources.extend(
        source
        for source in (
            record.get("evaluation_outcome"),
            record.get("environment_outcome"),
            record,
        )
        if isinstance(source, Mapping)
    )
    components: dict[str, float] = {}
    for canonical_name, aliases in _SCORE_COMPONENT_ALIASES.items():
        value = _first_value(sources, aliases)
        numeric = _finite_float(value)
        if numeric is not None:
            components[canonical_name] = numeric
    return components


def _final_assay_outcomes(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
    *,
    expected_batches: int,
    cell_id: str,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    by_batch: dict[int, dict[str, Any]] = {}
    for attempt_index, (record, batch_index) in enumerate(
        zip(records, batch_indices, strict=True),
        start=1,
    ):
        if (
            _operation(record) != "measure"
            or _instrument(record) != "final_assay"
            or not _status_committed(record)
        ):
            continue
        score = _leaderboard_score(record)
        if score is None:
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: committed final assay is missing leaderboard score"
            )
        if batch_index in by_batch:
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: multiple committed final assays for batch {batch_index}"
            )
        by_batch[batch_index] = {
            "batch_index": batch_index,
            "operation_attempt_index": attempt_index,
            "step": int(record.get("step", attempt_index)),
            "score": score,
            "components": _score_components(record),
        }
    discarded_batches = {
        batch_index
        for record, batch_index in zip(records, batch_indices, strict=True)
        if _operation(record) == "discard_batch" and _status_committed(record)
    }
    expected = set(range(expected_batches))
    observed_closed = set(by_batch) | discarded_batches
    invalid_partition = bool(set(by_batch) & discarded_batches)
    invalid_indices = not observed_closed.issubset(expected)
    nonprefix_closed = observed_closed != set(range(len(observed_closed)))
    missing_closed = observed_closed != expected
    if (
        invalid_partition
        or invalid_indices
        or nonprefix_closed
        or (missing_closed and not allow_incomplete)
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: closed batches={sorted(observed_closed)}, "
            f"expected={sorted(expected)}"
        )
    return [by_batch[index] for index in sorted(by_batch)]


def _operation_attempt_running_best(
    records: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Return the post-outcome incumbent at every submitted operation attempt."""

    best = 0.0
    curve: list[float] = []
    for record in records:
        if (
            _status_committed(record)
            and _operation(record) == "measure"
            and _instrument(record) == "final_assay"
        ):
            score = _leaderboard_score(record)
            if score is not None:
                best = max(best, score)
        curve.append(best)
    return curve


def _field_change(before: Any, after: Any) -> dict[str, Any]:
    changed = before != after
    payload: dict[str, Any] = {
        "before": to_builtin(before),
        "after": to_builtin(after),
        "changed": changed,
    }
    before_numeric = _finite_float(before)
    after_numeric = _finite_float(after)
    if before_numeric is not None and after_numeric is not None:
        delta = after_numeric - before_numeric
        payload["signed_delta"] = delta
        payload["absolute_delta"] = abs(delta)
    return payload


def _control_values(action: Mapping[str, Any], operation: str) -> dict[str, Any]:
    return {
        field: to_builtin(action.get(field))
        for field in _CONTROL_FIELDS[operation]
    }


def _diagnostic_adaptation_metrics(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
) -> dict[str, Any]:
    """Align each committed nonfinal diagnostic to the next committed control event."""

    per_batch_attempt_index: Counter[int] = Counter()
    batch_attempt_indices: list[int] = []
    committed_rank_by_batch: Counter[int] = Counter()
    committed_ranks: list[int | None] = []
    for record, batch_index in zip(records, batch_indices, strict=True):
        per_batch_attempt_index[batch_index] += 1
        batch_attempt_indices.append(per_batch_attempt_index[batch_index])
        if _status_committed(record):
            committed_rank_by_batch[batch_index] += 1
            committed_ranks.append(committed_rank_by_batch[batch_index])
        else:
            committed_ranks.append(None)

    alignments: list[dict[str, Any]] = []
    for diagnostic_index, (diagnostic, batch_index) in enumerate(
        zip(records, batch_indices, strict=True)
    ):
        if (
            not _status_committed(diagnostic)
            or _operation(diagnostic) != "measure"
            or _instrument(diagnostic) == "final_assay"
        ):
            continue
        target_index: int | None = None
        for candidate_index in range(diagnostic_index + 1, len(records)):
            if batch_indices[candidate_index] != batch_index:
                break
            candidate = records[candidate_index]
            if (
                _status_committed(candidate)
                and _operation(candidate) in _CONTROL_FIELDS
            ):
                target_index = candidate_index
                break
        diagnostic_attempt = diagnostic_index + 1
        row: dict[str, Any] = {
            "diagnostic_step": int(
                diagnostic.get("step", diagnostic_attempt)
            ),
            "diagnostic_operation_attempt_index": diagnostic_attempt,
            "diagnostic_operation_index_in_batch": batch_attempt_indices[
                diagnostic_index
            ],
            "batch_index": batch_index,
            "instrument": _instrument(diagnostic),
            "matched_next_control": target_index is not None,
        }
        if target_index is None:
            row["unmatched_reason"] = (
                "no later committed set_potential or electrolyze in the same batch"
            )
            alignments.append(row)
            continue

        target = records[target_index]
        target_operation = _operation(target)
        target_action = _action(target)
        prior_index: int | None = None
        for candidate_index in range(diagnostic_index - 1, -1, -1):
            if batch_indices[candidate_index] != batch_index:
                break
            candidate = records[candidate_index]
            if (
                _status_committed(candidate)
                and _operation(candidate) == target_operation
            ):
                prior_index = candidate_index
                break
        target_values = _control_values(target_action, target_operation)
        comparison = (
            {}
            if prior_index is None
            else {
                field: _field_change(
                    _control_values(
                        _action(records[prior_index]),
                        target_operation,
                    )[field],
                    target_values[field],
                )
                for field in _CONTROL_FIELDS[target_operation]
            }
        )
        changed_fields = [
            field
            for field, detail in comparison.items()
            if detail["changed"]
        ]
        diagnostic_committed_rank = committed_ranks[diagnostic_index]
        target_committed_rank = committed_ranks[target_index]
        if diagnostic_committed_rank is None or target_committed_rank is None:
            raise AssertionError("aligned diagnostic and control must be committed")
        row.update(
            {
                "next_control_step": int(
                    target.get("step", target_index + 1)
                ),
                "next_control_operation_attempt_index": target_index + 1,
                "next_control_operation_index_in_batch": batch_attempt_indices[
                    target_index
                ],
                "next_control_operation": target_operation,
                "operation_attempt_lag": target_index - diagnostic_index,
                "intervening_operation_attempt_count": (
                    target_index - diagnostic_index - 1
                ),
                "committed_operation_lag": (
                    target_committed_rank - diagnostic_committed_rank
                ),
                "comparison_available": prior_index is not None,
                "comparison_reference_step": (
                    None
                    if prior_index is None
                    else int(
                        records[prior_index].get("step", prior_index + 1)
                    )
                ),
                "next_control_values": target_values,
                "field_changes": comparison,
                "changed_fields": changed_fields,
                "changed_field_count": len(changed_fields),
                "any_control_field_changed": bool(changed_fields),
            }
        )
        alignments.append(row)

    matched = [row for row in alignments if row["matched_next_control"]]
    comparable = [row for row in matched if row["comparison_available"]]
    return {
        "definition": (
            "Each committed non-final measure is aligned to the first later "
            "committed set_potential or electrolyze in the same batch. "
            "operation_attempt_lag counts the target attempt as one; "
            "intervening_operation_attempt_count excludes both endpoints. "
            "Control changes compare the target with the most recent earlier "
            "committed event of the same control type in that batch."
        ),
        "diagnostic_event_count": len(alignments),
        "matched_event_count": len(matched),
        "unmatched_event_count": len(alignments) - len(matched),
        "comparable_event_count": len(comparable),
        "changed_control_event_count": sum(
            bool(row["any_control_field_changed"]) for row in comparable
        ),
        "next_control_operation_counts": dict(
            sorted(
                Counter(
                    str(row["next_control_operation"]) for row in matched
                ).items()
            )
        ),
        "events": alignments,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(float(value) for value in values) if values else None


def _discovery_retention_recovery_metrics(
    final_outcomes: Sequence[Mapping[str, Any]],
    *,
    retention_fraction: float = TRAJECTORY_RETENTION_FRACTION,
) -> dict[str, Any]:
    """Describe online discovery, incumbent loss, and recovery over final assays."""

    if not 0.0 < retention_fraction <= 1.0:
        raise ValueError("retention_fraction must be in (0, 1]")
    points = [
        {
            "final_assay_ordinal": ordinal,
            "batch_number": int(outcome["batch_index"]) + 1,
            "score": float(outcome["score"]),
        }
        for ordinal, outcome in enumerate(final_outcomes, start=1)
    ]
    if not points:
        return {
            "definition": (
                "No committed final assay was available, so discovery, retention, "
                "drawdown, and recovery are undefined."
            ),
            "retention_fraction": retention_fraction,
            "score_observation_count": 0,
            "global_best_score": None,
            "global_best_first_final_assay_ordinal": None,
            "global_best_first_batch_number": None,
            "global_best_discovery_fraction": None,
            "incumbent_update_count": 0,
            "incumbent_updates": [],
            "online_retention_opportunity_count": 0,
            "online_retained_count": 0,
            "online_retention_rate": None,
            "post_global_best_observation_count": 0,
            "post_global_best_retained_count": 0,
            "post_global_best_retention_rate": None,
            "maximum_absolute_drawdown_from_prior_incumbent": None,
            "maximum_relative_drawdown_from_prior_incumbent": None,
            "terminal_to_global_best_ratio": None,
            "loss_episode_count": 0,
            "recovered_loss_episode_count": 0,
            "unresolved_loss_episode_count": 0,
            "recovery_rate": None,
            "mean_recovery_delay_final_assays": None,
            "mean_recovery_delay_batches": None,
            "loss_episodes": [],
            "per_final_assay": [],
        }

    incumbent: float | None = None
    incumbent_updates: list[dict[str, Any]] = []
    per_final_assay: list[dict[str, Any]] = []
    for point in points:
        score = float(point["score"])
        pre_incumbent = incumbent
        if pre_incumbent is None:
            threshold = None
            retained = None
            absolute_drawdown = None
            relative_drawdown = None
            new_incumbent = True
        else:
            threshold = retention_fraction * pre_incumbent
            retained = score + _SCORE_COMPARISON_TOLERANCE >= threshold
            absolute_drawdown = max(0.0, pre_incumbent - score)
            relative_drawdown = (
                absolute_drawdown / pre_incumbent
                if pre_incumbent > 0.0
                else 0.0
            )
            new_incumbent = score > (
                pre_incumbent + _SCORE_COMPARISON_TOLERANCE
            )
        if new_incumbent:
            incumbent_updates.append(
                {
                    **point,
                    "improvement_over_prior_incumbent": (
                        None
                        if pre_incumbent is None
                        else score - pre_incumbent
                    ),
                }
            )
            incumbent = score
        per_final_assay.append(
            {
                **point,
                "pre_assay_incumbent": pre_incumbent,
                "retention_threshold": threshold,
                "retained_prior_incumbent": retained,
                "absolute_drawdown_from_prior_incumbent": absolute_drawdown,
                "relative_drawdown_from_prior_incumbent": relative_drawdown,
                "new_incumbent": new_incumbent,
                "post_assay_incumbent": incumbent,
            }
        )

    global_best = max(float(point["score"]) for point in points)
    global_best_index = next(
        index
        for index, point in enumerate(points)
        if abs(float(point["score"]) - global_best)
        <= _SCORE_COMPARISON_TOLERANCE
    )
    global_best_point = points[global_best_index]
    post_global_best = points[global_best_index + 1 :]
    global_retention_threshold = retention_fraction * global_best
    post_global_retained_count = sum(
        float(point["score"]) + _SCORE_COMPARISON_TOLERANCE
        >= global_retention_threshold
        for point in post_global_best
    )
    online_rows = per_final_assay[1:]
    online_retained_count = sum(
        row["retained_prior_incumbent"] is True for row in online_rows
    )
    absolute_drawdowns = [
        float(row["absolute_drawdown_from_prior_incumbent"])
        for row in online_rows
    ]
    relative_drawdowns = [
        float(row["relative_drawdown_from_prior_incumbent"])
        for row in online_rows
    ]

    loss_episodes: list[dict[str, Any]] = []
    open_episode: dict[str, Any] | None = None
    for row in online_rows:
        score = float(row["score"])
        if open_episode is not None:
            if score + _SCORE_COMPARISON_TOLERANCE >= float(
                open_episode["recovery_threshold"]
            ):
                open_episode.update(
                    {
                        "recovered": True,
                        "recovery_final_assay_ordinal": row[
                            "final_assay_ordinal"
                        ],
                        "recovery_batch_number": row["batch_number"],
                        "recovery_score": score,
                        "recovery_delay_final_assays": int(
                            row["final_assay_ordinal"]
                        )
                        - int(open_episode["loss_start_final_assay_ordinal"]),
                        "recovery_delay_batches": int(row["batch_number"])
                        - int(open_episode["loss_start_batch_number"]),
                        "recovery_time_right_censored": False,
                    }
                )
                loss_episodes.append(open_episode)
                open_episode = None
            else:
                open_episode["lowest_score"] = min(
                    float(open_episode["lowest_score"]), score
                )
                reference = float(open_episode["reference_incumbent"])
                open_episode["maximum_absolute_drawdown"] = max(
                    float(open_episode["maximum_absolute_drawdown"]),
                    reference - score,
                )
                open_episode["maximum_relative_drawdown"] = (
                    float(open_episode["maximum_absolute_drawdown"])
                    / reference
                    if reference > 0.0
                    else 0.0
                )
            continue
        if row["retained_prior_incumbent"] is False:
            reference = float(row["pre_assay_incumbent"])
            absolute_drawdown = float(
                row["absolute_drawdown_from_prior_incumbent"]
            )
            open_episode = {
                "loss_start_final_assay_ordinal": row["final_assay_ordinal"],
                "loss_start_batch_number": row["batch_number"],
                "loss_start_score": score,
                "reference_incumbent": reference,
                "recovery_threshold": retention_fraction * reference,
                "lowest_score": score,
                "maximum_absolute_drawdown": absolute_drawdown,
                "maximum_relative_drawdown": (
                    absolute_drawdown / reference if reference > 0.0 else 0.0
                ),
            }
    if open_episode is not None:
        open_episode.update(
            {
                "recovered": False,
                "recovery_final_assay_ordinal": None,
                "recovery_batch_number": None,
                "recovery_score": None,
                "recovery_delay_final_assays": None,
                "recovery_delay_batches": None,
                "recovery_time_right_censored": True,
            }
        )
        loss_episodes.append(open_episode)

    recovered_episodes = [
        episode for episode in loss_episodes if episode["recovered"]
    ]
    return {
        "definition": (
            "A final assay retains the pre-assay incumbent when its score is "
            f"at least {retention_fraction:.0%} of that incumbent. A loss "
            "episode begins below that threshold; its reference incumbent is "
            "frozen, and recovery is the first later final assay reaching the "
            "same threshold. Global-best discovery uses the first occurrence "
            "of the observed campaign maximum. Unrecovered terminal episodes "
            "have right-censored recovery time."
        ),
        "retention_fraction": retention_fraction,
        "score_observation_count": len(points),
        "global_best_score": global_best,
        "global_best_first_final_assay_ordinal": global_best_point[
            "final_assay_ordinal"
        ],
        "global_best_first_batch_number": global_best_point["batch_number"],
        "global_best_discovery_fraction": (
            global_best_index / (len(points) - 1) if len(points) > 1 else 0.0
        ),
        "incumbent_update_count": len(incumbent_updates),
        "incumbent_updates": incumbent_updates,
        "online_retention_opportunity_count": len(online_rows),
        "online_retained_count": online_retained_count,
        "online_retention_rate": (
            online_retained_count / len(online_rows) if online_rows else None
        ),
        "post_global_best_observation_count": len(post_global_best),
        "post_global_best_retained_count": post_global_retained_count,
        "post_global_best_retention_rate": (
            post_global_retained_count / len(post_global_best)
            if post_global_best
            else None
        ),
        "maximum_absolute_drawdown_from_prior_incumbent": (
            max(absolute_drawdowns) if absolute_drawdowns else 0.0
        ),
        "maximum_relative_drawdown_from_prior_incumbent": (
            max(relative_drawdowns) if relative_drawdowns else 0.0
        ),
        "terminal_to_global_best_ratio": (
            float(points[-1]["score"]) / global_best
            if global_best > 0.0
            else 1.0
        ),
        "loss_episode_count": len(loss_episodes),
        "recovered_loss_episode_count": len(recovered_episodes),
        "unresolved_loss_episode_count": (
            len(loss_episodes) - len(recovered_episodes)
        ),
        "recovery_rate": (
            len(recovered_episodes) / len(loss_episodes)
            if loss_episodes
            else None
        ),
        "mean_recovery_delay_final_assays": _mean_or_none(
            [
                float(episode["recovery_delay_final_assays"])
                for episode in recovered_episodes
            ]
        ),
        "mean_recovery_delay_batches": _mean_or_none(
            [
                float(episode["recovery_delay_batches"])
                for episode in recovered_episodes
            ]
        ),
        "loss_episodes": loss_episodes,
        "per_final_assay": per_final_assay,
    }


def _diagnostic_control_to_final_metrics(
    final_outcomes: Sequence[Mapping[str, Any]],
    diagnostic_adaptation: Mapping[str, Any],
) -> dict[str, Any]:
    """Align diagnostic-triggered control changes with that batch's final score."""

    events = diagnostic_adaptation.get("events", [])
    event_rows = [event for event in events if isinstance(event, Mapping)]
    ordered_outcomes = sorted(
        final_outcomes,
        key=lambda outcome: int(outcome["batch_index"]),
    )
    prior_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for ordinal, outcome in enumerate(ordered_outcomes, start=1):
        batch_index = int(outcome["batch_index"])
        score = float(outcome["score"])
        batch_events = [
            event
            for event in event_rows
            if int(event.get("batch_index", -1)) == batch_index
        ]
        comparable = [
            event
            for event in batch_events
            if event.get("matched_next_control") is True
            and event.get("comparison_available") is True
        ]
        changed = [
            event
            for event in comparable
            if event.get("any_control_field_changed") is True
        ]
        previous_score = prior_scores[-1] if prior_scores else None
        pre_batch_incumbent = max(prior_scores) if prior_scores else None
        rows.append(
            {
                "final_assay_ordinal": ordinal,
                "batch_number": batch_index + 1,
                "final_score": score,
                "diagnostic_event_count": len(batch_events),
                "comparable_control_event_count": len(comparable),
                "changed_control_event_count": len(changed),
                "any_diagnostic_aligned_control_change": bool(changed),
                "previous_final_score": previous_score,
                "pre_batch_incumbent": pre_batch_incumbent,
                "score_delta_vs_previous_final": (
                    None if previous_score is None else score - previous_score
                ),
                "score_delta_vs_pre_batch_incumbent": (
                    None
                    if pre_batch_incumbent is None
                    else score - pre_batch_incumbent
                ),
                "positive_delta_vs_previous_final": (
                    None
                    if previous_score is None
                    else score
                    > previous_score + _SCORE_COMPARISON_TOLERANCE
                ),
                "new_incumbent": (
                    None
                    if pre_batch_incumbent is None
                    else score
                    > pre_batch_incumbent + _SCORE_COMPARISON_TOLERANCE
                ),
            }
        )
        prior_scores.append(score)

    eligible_changed = [
        row
        for row in rows
        if row["any_diagnostic_aligned_control_change"]
        and row["previous_final_score"] is not None
    ]
    eligible_unchanged = [
        row
        for row in rows
        if row["comparable_control_event_count"] > 0
        and not row["any_diagnostic_aligned_control_change"]
        and row["previous_final_score"] is not None
    ]

    def conversion_summary(
        selected: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        positive_count = sum(
            row["positive_delta_vs_previous_final"] is True
            for row in selected
        )
        incumbent_count = sum(row["new_incumbent"] is True for row in selected)
        return {
            "eligible_batch_count": len(selected),
            "positive_next_final_delta_count": positive_count,
            "positive_next_final_delta_rate": (
                positive_count / len(selected) if selected else None
            ),
            "new_incumbent_count": incumbent_count,
            "new_incumbent_rate": (
                incumbent_count / len(selected) if selected else None
            ),
            "mean_next_final_delta_vs_previous": _mean_or_none(
                [float(row["score_delta_vs_previous_final"]) for row in selected]
            ),
            "mean_next_final_delta_vs_pre_batch_incumbent": _mean_or_none(
                [
                    float(row["score_delta_vs_pre_batch_incumbent"])
                    for row in selected
                ]
            ),
        }

    return {
        "definition": (
            "The analysis unit is a batch, not a diagnostic event. A batch is "
            "classified as changed when at least one committed non-final "
            "diagnostic is aligned to a later comparable control whose fields "
            "changed. Conversion is the fraction of eligible changed batches "
            "whose final score exceeds the preceding batch's final score. This "
            "is temporal alignment, not a causal effect estimate."
        ),
        "changed_control": conversion_summary(eligible_changed),
        "comparable_without_change": conversion_summary(eligible_unchanged),
        "per_final_assay": rows,
    }


def _material_id(field: str, value: Any) -> str:
    prefixes = {
        "solvent": "solvent-S",
        "electrolyte_profile": "electrolyte-E",
        "catalyst": "catalyst-C",
        "extractant": "extractant-X",
    }
    if isinstance(value, bool):
        return f"{field}:{value}"
    if isinstance(value, int):
        return f"{prefixes.get(field, field + '-')}{value}"
    return str(value)


def _verified_archived_material_reference(
    run_dir: Path,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    archive_root = run_dir / "codex_workspace"
    reference_path = archive_root / "reference" / "material_information.json"
    manifest_path = run_dir / "codex_workspace_manifest.json"
    if not reference_path.is_file() or not manifest_path.is_file():
        return None, None
    try:
        manifest = _load_json_object(
            manifest_path,
            label="Codex workspace manifest",
        )
    except AutonomousMaterialCampaignAuditError as error:
        return None, str(error)
    declared_in_summary = summary.get("codex_workspace_archive")
    if (
        isinstance(declared_in_summary, Mapping)
        and canonical_json_sha256(declared_in_summary)
        != canonical_json_sha256(manifest)
    ):
        return None, "run summary/workspace manifest mismatch"
    entries = manifest.get("files")
    if not isinstance(entries, list) or not all(
        isinstance(entry, Mapping) for entry in entries
    ):
        return None, "workspace manifest files list is absent or malformed"
    if canonical_json_sha256(entries) != manifest.get("tree_sha256"):
        return None, "workspace manifest tree_sha256 mismatch"
    target_entry = next(
        (
            entry
            for entry in entries
            if entry.get("path") == "reference/material_information.json"
        ),
        None,
    )
    if target_entry is None:
        return None, "workspace manifest omits material information reference"
    if file_sha256(reference_path) != target_entry.get("sha256"):
        return None, "archived material information reference sha256 mismatch"
    byte_count = target_entry.get("byte_count")
    if (
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and reference_path.stat().st_size != byte_count
    ):
        return None, "archived material information reference size mismatch"
    try:
        return (
            _load_json_object(
                reference_path,
                label="archived material information reference",
            ),
            None,
        )
    except AutonomousMaterialCampaignAuditError as error:
        return None, str(error)


def _recover_material_reference(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_dir: Path,
    arm: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Recover only self-hash-checked or archive-manifest-checked public material data."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for source, raw in (
        ("run_config.material_information", config.get("material_information")),
        ("run_summary.material_information", summary.get("material_information")),
        (
            "run_summary.environment_contract.public_contract.material_information",
            _dig(
                summary,
                "environment_contract.public_contract.material_information",
            ),
        ),
    ):
        if isinstance(raw, Mapping):
            candidates.append((source, raw))

    archived, archive_error = _verified_archived_material_reference(
        run_dir,
        summary,
    )
    catalog: dict[str, Any] | None = None
    catalog_source: str | None = None
    if archived is not None:
        archived_information = archived.get("material_information")
        if isinstance(archived_information, Mapping):
            candidates.append(
                (
                    "verified_archive.reference/material_information.json",
                    archived_information,
                )
            )
        archived_catalog = archived.get("material_catalog")
        if isinstance(archived_catalog, Mapping):
            catalog = dict(archived_catalog)
            catalog_source = (
                "verified_archive.reference/material_information.json"
            )

    valid: list[tuple[str, dict[str, Any], str]] = []
    issues: list[str] = []
    if archive_error is not None:
        issues.append(archive_error)
    for source, information in candidates:
        raw_dossier = information.get("dossier")
        if raw_dossier is None:
            continue
        if not isinstance(raw_dossier, Mapping):
            issues.append(f"{source}: dossier is not an object")
            continue
        declared_hash = information.get("dossier_sha256")
        observed_hash = canonical_json_sha256(raw_dossier)
        if not isinstance(declared_hash, str) or declared_hash != observed_hash:
            issues.append(f"{source}: dossier_sha256 is absent or mismatched")
            continue
        valid.append((source, dict(raw_dossier), observed_hash))

    hashes = {item[2] for item in valid}
    if len(hashes) > 1:
        issues.append("validated material dossier candidates disagree")
        valid = []
    dossier = valid[0][1] if valid else None
    selected_source: str | None = valid[0][0] if valid else None
    dossier_sha256 = valid[0][2] if valid else None
    if dossier is None:
        reason = (
            "opaque arm intentionally exposes no nominal-property dossier"
            if arm == OPAQUE_ARM and not issues
            else (
                "; ".join(issues)
                if issues
                else "no self-hash-checked or archive-verified dossier was found"
            )
        )
    else:
        reason = None
    metadata = {
        "nominal_dossier_available": dossier is not None,
        "nominal_dossier_source": selected_source,
        "nominal_dossier_sha256": dossier_sha256,
        "nominal_dossier_unavailable_reason": reason,
        "material_catalog_available": catalog is not None,
        "material_catalog_source": catalog_source,
        "recovery_issues": issues,
        "safety_policy": (
            "Nominal descriptor ranks are computed only from a dossier whose "
            "declared canonical SHA-256 matches its content, or from the "
            "manifest-verified host reference archive."
        ),
    }
    return dossier, catalog, metadata


def _catalog_ids(
    catalog: Mapping[str, Any] | None,
    dossier: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if catalog is not None:
        for field, catalog_key in (
            ("solvent", "solvents"),
            ("electrolyte_profile", "electrolyte_profiles"),
            ("catalyst", "catalysts"),
            ("extractant", "extractants"),
        ):
            rows = catalog.get(catalog_key)
            if isinstance(rows, list):
                ids = [
                    str(row["anonymous_material_id"])
                    for row in rows
                    if isinstance(row, Mapping)
                    and row.get("anonymous_material_id") is not None
                ]
                if ids:
                    result[field] = ids
    choices = dossier.get("choices") if isinstance(dossier, Mapping) else None
    if isinstance(choices, Mapping):
        for field, rows in choices.items():
            if not isinstance(rows, list):
                continue
            ids = [
                str(row["anonymous_material_id"])
                for row in rows
                if isinstance(row, Mapping)
                and row.get("anonymous_material_id") is not None
            ]
            if ids:
                result.setdefault(str(field), ids)
    return result


def _dense_rank(value: float, values: Sequence[float], *, reverse: bool) -> int:
    ordered = sorted({float(item) for item in values}, reverse=reverse)
    return ordered.index(float(value)) + 1


def _nominal_descriptor_rank_metrics(
    dossier: Mapping[str, Any] | None,
    first_choices_by_batch: Sequence[Mapping[str, str]],
    *,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    if dossier is None:
        return {
            "available": False,
            "unavailable_reason": recovery[
                "nominal_dossier_unavailable_reason"
            ],
            "dossier_source": None,
            "dossier_sha256": None,
        }
    choices = dossier.get("choices")
    if not isinstance(choices, Mapping):
        return {
            "available": False,
            "unavailable_reason": "validated dossier has no choices object",
            "dossier_source": recovery["nominal_dossier_source"],
            "dossier_sha256": recovery["nominal_dossier_sha256"],
        }
    fields: dict[str, Any] = {}
    for field, raw_rows in choices.items():
        if not isinstance(raw_rows, list):
            continue
        rows = [row for row in raw_rows if isinstance(row, Mapping)]
        row_by_id = {
            str(row["anonymous_material_id"]): row
            for row in rows
            if row.get("anonymous_material_id") is not None
        }
        numeric_values: dict[str, list[float]] = {}
        for row in rows:
            properties = row.get("nominal_properties")
            if not isinstance(properties, Mapping):
                continue
            for descriptor, raw_value in properties.items():
                value = _finite_float(raw_value)
                if value is not None:
                    numeric_values.setdefault(str(descriptor), []).append(value)
        selections: list[dict[str, Any]] = []
        unmatched: list[dict[str, Any]] = []
        for batch_index, batch_choices in enumerate(first_choices_by_batch):
            material_id = batch_choices.get(str(field))
            if material_id is None:
                continue
            selected_row = row_by_id.get(material_id)
            if selected_row is None:
                unmatched.append(
                    {
                        "batch_index": batch_index,
                        "material_id": material_id,
                    }
                )
                continue
            properties = selected_row.get("nominal_properties")
            descriptor_rows: dict[str, Any] = {}
            if isinstance(properties, Mapping):
                for descriptor, raw_value in properties.items():
                    value = _finite_float(raw_value)
                    population = numeric_values.get(str(descriptor), [])
                    if value is None or not population:
                        continue
                    descriptor_rows[str(descriptor)] = {
                        "value": value,
                        "ascending_dense_rank": _dense_rank(
                            value,
                            population,
                            reverse=False,
                        ),
                        "descending_dense_rank": _dense_rank(
                            value,
                            population,
                            reverse=True,
                        ),
                    }
            selections.append(
                {
                    "batch_index": batch_index,
                    "material_id": material_id,
                    "descriptors": descriptor_rows,
                }
            )
        fields[str(field)] = {
            "option_count": len(row_by_id),
            "descriptor_names": sorted(numeric_values),
            "selections": selections,
            "unmatched_selections": unmatched,
        }
    return {
        "available": True,
        "dossier_source": recovery["nominal_dossier_source"],
        "dossier_sha256": recovery["nominal_dossier_sha256"],
        "rank_definition": (
            "Ranks are dense ranks within each material field and descriptor; "
            "ascending rank 1 is the lowest declared nominal value and "
            "descending rank 1 is the highest. No descriptor direction or "
            "cross-descriptor utility is inferred."
        ),
        "fields": fields,
    }


def _material_metrics(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
    *,
    expected_batches: int,
    dossier: Mapping[str, Any] | None,
    catalog: Mapping[str, Any] | None,
    reference_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    by_field: dict[str, list[str]] = {field: [] for field in _MATERIAL_FIELDS}
    per_batch: list[dict[str, Any]] = []
    for attempt_index, (record, batch_index) in enumerate(
        zip(records, batch_indices, strict=True),
        start=1,
    ):
        if not _status_committed(record):
            continue
        action = _action(record)
        for field in _MATERIAL_FIELDS:
            if field not in action:
                continue
            material = _material_id(field, action[field])
            by_field[field].append(material)
            selections.append(
                {
                    "step": int(record.get("step", len(selections) + 1)),
                    "operation_attempt_index": attempt_index,
                    "batch_index": batch_index,
                    "field": field,
                    "material_id": material,
                    "operation": _operation(record),
                }
            )
    all_selection_signatures: list[str] = []
    first_choice_signatures: list[str] = []
    first_choices_by_batch: list[dict[str, str]] = []
    for batch_index in range(expected_batches):
        batch_rows = [
            row for row in selections if row["batch_index"] == batch_index
        ]
        values = {
            field: sorted(
                {
                    str(row["material_id"])
                    for row in batch_rows
                    if row["field"] == field
                }
            )
            for field in _MATERIAL_FIELDS
        }
        first_choices = {
            field: str(
                next(
                    row["material_id"]
                    for row in batch_rows
                    if row["field"] == field
                )
            )
            for field in _MATERIAL_FIELDS
            if any(row["field"] == field for row in batch_rows)
        }
        all_signature = "|".join(
            f"{field}={','.join(values[field])}"
            for field in _MATERIAL_FIELDS
            if values[field]
        )
        first_signature = "|".join(
            f"{field}={first_choices[field]}"
            for field in _MATERIAL_FIELDS
            if field in first_choices
        )
        per_batch.append(
            {
                "batch_index": batch_index,
                "selections": values,
                "first_choices": first_choices,
                "all_selected_ids_signature": all_signature,
                "first_choice_signature": first_signature,
            }
        )
        all_selection_signatures.append(all_signature)
        first_choice_signatures.append(first_signature)
        first_choices_by_batch.append(first_choices)

    reference_ids = _catalog_ids(catalog, dossier)
    coverage_by_field: dict[str, Any] = {}
    for field in _MATERIAL_FIELDS:
        sequence = [
            batch[field] for batch in first_choices_by_batch if field in batch
        ]
        if not sequence:
            continue
        unique_ids = list(dict.fromkeys(sequence))
        available_ids = reference_ids.get(field)
        available_count = len(set(available_ids)) if available_ids else None
        coverage_by_field[field] = {
            "first_choice_sequence": sequence,
            "observed_batch_count": len(sequence),
            "unique_selected_ids": unique_ids,
            "unique_selected_count": len(unique_ids),
            "available_option_count": available_count,
            "available_option_ids": (
                sorted(set(available_ids)) if available_ids else None
            ),
            "material_space_coverage_fraction": (
                None
                if available_count is None or available_count == 0
                else len(set(sequence)) / available_count
            ),
            "adjacent_switch_count": sum(
                left != right for left, right in pairwise(sequence)
            ),
            "adjacent_repeat_count": sum(
                left == right for left, right in pairwise(sequence)
            ),
            "revisit_after_switch_count": sum(
                current in set(sequence[:index])
                and current != sequence[index - 1]
                for index, current in enumerate(sequence[1:], start=1)
            ),
        }
    nonempty_first_signatures = [
        signature for signature in first_choice_signatures if signature
    ]
    unique_first_signatures = list(dict.fromkeys(nonempty_first_signatures))
    joint_revisit_after_switch_count = sum(
        current in set(nonempty_first_signatures[:index])
        and current != nonempty_first_signatures[index - 1]
        for index, current in enumerate(nonempty_first_signatures[1:], start=1)
    )
    return {
        "selections": selections,
        "selection_count": len(selections),
        "by_field": {
            field: {
                "count": len(values),
                "unique_count": len(set(values)),
                "counts": dict(sorted(Counter(values).items())),
            }
            for field, values in by_field.items()
            if values
        },
        "per_batch": per_batch,
        "predeclared_endpoints": {
            "definition": (
                "A batch material policy uses the first committed selection "
                "of each material field in that batch. Coverage, switches, "
                "and revisits are computed from these first-choice sequences."
            ),
            "first_batch_choices": (
                first_choices_by_batch[0] if first_choices_by_batch else {}
            ),
            "first_two_batch_choices": first_choices_by_batch[:2],
            "coverage_by_field": coverage_by_field,
            "joint_first_choice_policy": {
                "sequence": nonempty_first_signatures,
                "unique_policy_count": len(unique_first_signatures),
                "unique_policies": unique_first_signatures,
                "adjacent_switch_count": sum(
                    left != right
                    for left, right in pairwise(nonempty_first_signatures)
                ),
                "adjacent_repeat_count": sum(
                    left == right
                    for left, right in pairwise(nonempty_first_signatures)
                ),
                "revisit_after_switch_count": joint_revisit_after_switch_count,
            },
        },
        "material_reference_recovery": dict(reference_recovery),
        "nominal_descriptor_ranks": _nominal_descriptor_rank_metrics(
            dossier,
            first_choices_by_batch,
            recovery=reference_recovery,
        ),
        "unique_batch_material_policy_count": len(
            {
                signature
                for signature in all_selection_signatures
                if signature
            }
        ),
        "batch_material_policy_switch_count": sum(
            left != right
            for left, right in pairwise(
                [
                    signature
                    for signature in all_selection_signatures
                    if signature
                ]
            )
        ),
    }


def _measurement_metrics(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    operation_in_batch: Counter[int] = Counter()
    process_time_by_batch: dict[int, float] = {}
    for record, batch_index in zip(records, batch_indices, strict=True):
        operation_in_batch[batch_index] += 1
        process_time_by_batch.setdefault(batch_index, 0.0)
        action = _action(record)
        instrument = _instrument(record)
        if _operation(record) == "measure" and instrument != "final_assay":
            entries.append(
                {
                    "step": int(record.get("step", len(entries) + 1)),
                    "batch_index": batch_index,
                    "operation_index_in_batch": operation_in_batch[batch_index],
                    "process_time_s_before": process_time_by_batch[batch_index],
                    "instrument": instrument,
                    "committed": _status_committed(record),
                }
            )
        raw_delta = _first_value(
            (record,),
            (
                "state_delta_summary.delta_time_s",
                "environment_outcome.state_delta_summary.delta_time_s",
            ),
        )
        if raw_delta is not None and math.isfinite(float(raw_delta)):
            process_time_by_batch[batch_index] += max(float(raw_delta), 0.0)
        if action.get("operation") == "electrolyze":
            duration = action.get("duration_s")
            if raw_delta is None and duration is not None:
                process_time_by_batch[batch_index] += max(float(duration), 0.0)
    committed = [entry for entry in entries if entry["committed"]]
    return {
        "attempt_count": len(entries),
        "committed_count": len(committed),
        "invalid_count": len(entries) - len(committed),
        "instrument_counts": dict(
            sorted(Counter(str(item["instrument"]) for item in committed).items())
        ),
        "timing": entries,
        "per_batch_committed_counts": {
            str(index): sum(
                item["committed"] and item["batch_index"] == index for item in entries
            )
            for index in sorted(set(batch_indices))
        },
    }


def _setpoint_metrics(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    prior_by_batch: dict[int, dict[str, Any]] = {}
    changed_count = 0
    repeated_count = 0
    for record, batch_index in zip(records, batch_indices, strict=True):
        action = _action(record)
        if action.get("operation") != "set_potential" or not _status_committed(record):
            continue
        setpoint = {
            key: to_builtin(action.get(key))
            for key in ("potential_V", "current_mA", "electrolyte_profile")
        }
        prior = prior_by_batch.get(batch_index)
        changed_fields = (
            [] if prior is None else [key for key in setpoint if setpoint[key] != prior[key]]
        )
        if prior is not None:
            if changed_fields:
                changed_count += 1
            else:
                repeated_count += 1
        entry = {
            "step": int(record.get("step", len(entries) + 1)),
            "batch_index": batch_index,
            **setpoint,
            "is_initial_for_batch": prior is None,
            "changed_fields": changed_fields,
        }
        entries.append(entry)
        prior_by_batch[batch_index] = setpoint
    return {
        "setpoint_operation_count": len(entries),
        "within_batch_change_count": changed_count,
        "within_batch_repeat_count": repeated_count,
        "sequence": entries,
    }


def _batch_policy_rows(
    records: Sequence[Mapping[str, Any]],
    batch_indices: Sequence[int],
    final_outcomes: Sequence[Mapping[str, Any]],
    *,
    expected_batches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcome_by_batch = {
        int(outcome["batch_index"]): outcome for outcome in final_outcomes
    }
    batch_rows: list[dict[str, Any]] = []
    for batch_index in range(expected_batches):
        selected = [
            record
            for record, index in zip(records, batch_indices, strict=True)
            if index == batch_index
        ]
        committed = [record for record in selected if _status_committed(record)]
        first_material_choices: dict[str, str] = {}
        material_selection_sequence: list[dict[str, Any]] = []
        diagnostic_policy: list[dict[str, Any]] = []
        for operation_index, record in enumerate(selected, start=1):
            if not _status_committed(record):
                continue
            action = _action(record)
            selected_materials = {
                field: _material_id(field, action[field])
                for field in _MATERIAL_FIELDS
                if field in action
            }
            if selected_materials:
                material_selection_sequence.append(selected_materials)
                for field, material_id in selected_materials.items():
                    first_material_choices.setdefault(field, material_id)
            if (
                _operation(record) == "measure"
                and _instrument(record) != "final_assay"
            ):
                diagnostic_policy.append(
                    {
                        "instrument": _instrument(record),
                        "operation_index_in_batch": operation_index,
                    }
                )
        setpoint_policy = [
            {
                key: to_builtin(_action(record).get(key))
                for key in _SETPOINT_FIELDS
            }
            for record in committed
            if _operation(record) == "set_potential"
        ]
        duration_policy = [
            _action(record).get("duration_s")
            for record in committed
            if _operation(record) == "electrolyze"
        ]
        outcome = outcome_by_batch.get(batch_index)
        batch_rows.append(
            {
                "batch_index": batch_index,
                "operation_count": len(selected),
                "invalid_operation_count": len(selected) - len(committed),
                "operation_signature": [_operation(record) for record in committed],
                "material_first_choices": first_material_choices,
                "material_selection_sequence": material_selection_sequence,
                "diagnostic_policy": diagnostic_policy,
                "setpoint_policy": setpoint_policy,
                "electrolysis_duration_policy": duration_policy,
                "final_score": (
                    None if outcome is None else float(outcome["score"])
                ),
                "final_score_components": (
                    {} if outcome is None else dict(outcome["components"])
                ),
            }
        )
    shifts: list[dict[str, Any]] = []
    for previous, current in pairwise(batch_rows):
        material_fields = sorted(
            set(previous["material_first_choices"])
            | set(current["material_first_choices"])
        )
        material_field_changes = {
            field: _field_change(
                previous["material_first_choices"].get(field),
                current["material_first_choices"].get(field),
            )
            for field in material_fields
        }
        setpoint_fields = sorted(
            {
                field
                for row in (
                    *previous["setpoint_policy"],
                    *current["setpoint_policy"],
                )
                for field in row
            }
        )
        setpoint_changed_fields = [
            field
            for field in setpoint_fields
            if [row.get(field) for row in previous["setpoint_policy"]]
            != [row.get(field) for row in current["setpoint_policy"]]
        ]
        diagnostic_changed = (
            previous["diagnostic_policy"] != current["diagnostic_policy"]
        )
        duration_changed = (
            previous["electrolysis_duration_policy"]
            != current["electrolysis_duration_policy"]
        )
        changed = [
            dimension
            for dimension, is_changed in (
                (
                    "material_first_choices",
                    any(
                        detail["changed"]
                        for detail in material_field_changes.values()
                    ),
                ),
                ("setpoint_policy", bool(setpoint_changed_fields)),
                ("diagnostic_policy", diagnostic_changed),
                ("electrolysis_duration_policy", duration_changed),
                (
                    "operation_signature",
                    previous["operation_signature"]
                    != current["operation_signature"],
                ),
            )
            if is_changed
        ]
        shifts.append(
            {
                "from_batch_index": previous["batch_index"],
                "to_batch_index": current["batch_index"],
                "antecedent_final_outcome": {
                    "score": previous["final_score"],
                    "components": previous["final_score_components"],
                },
                "next_batch_final_outcome": {
                    "score": current["final_score"],
                    "components": current["final_score_components"],
                },
                "material_policy_change": {
                    "changed": any(
                        detail["changed"]
                        for detail in material_field_changes.values()
                    ),
                    "changed_fields": [
                        field
                        for field, detail in material_field_changes.items()
                        if detail["changed"]
                    ],
                    "field_changes": material_field_changes,
                    "previous": previous["material_first_choices"],
                    "next": current["material_first_choices"],
                },
                "setpoint_policy_change": {
                    "changed": bool(setpoint_changed_fields),
                    "changed_fields": setpoint_changed_fields,
                    "previous": previous["setpoint_policy"],
                    "next": current["setpoint_policy"],
                },
                "diagnostic_policy_change": {
                    "changed": diagnostic_changed,
                    "previous": previous["diagnostic_policy"],
                    "next": current["diagnostic_policy"],
                },
                "electrolysis_duration_policy_change": {
                    "changed": duration_changed,
                    "previous": previous["electrolysis_duration_policy"],
                    "next": current["electrolysis_duration_policy"],
                },
                "changed_dimensions": changed,
                "changed_dimension_count": len(changed),
                "score_delta": (
                    None
                    if previous["final_score"] is None
                    or current["final_score"] is None
                    else current["final_score"] - previous["final_score"]
                ),
            }
        )
    return (
        batch_rows,
        shifts,
    )


def _usage_metrics(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    usage = _first_value(
        (summary,),
        ("method_resources", "method_resource_usage", "resources.method"),
    )
    if not isinstance(usage, Mapping) and records:
        usage = _first_value(
            (records[-1],),
            ("method_resources.agent_usage", "method_resources"),
        )
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    if isinstance(usage.get("agent_usage"), Mapping):
        usage = {**usage, **dict(usage["agent_usage"])}
    receipts = summary.get("provider_receipts", [])
    receipt_rows = [item for item in receipts if isinstance(item, Mapping)] if isinstance(
        receipts, list
    ) else []
    tool_events = [
        event
        for receipt in receipt_rows
        for event in (
            receipt.get("tool_events", [])
            if isinstance(receipt.get("tool_events", []), list)
            else []
        )
        if isinstance(event, Mapping)
    ]
    classifications = Counter(
        str(event.get("classification", "unclassified"))
        for event in tool_events
    )
    material_reference_reads = [
        event
        for event in tool_events
        if event.get("classification") == "file_read"
        and any(
            str(path).replace("\\", "/").lower().endswith(
                "reference/material_information.json"
            )
            for path in (
                event.get("referenced_relative_paths", [])
                if isinstance(
                    event.get("referenced_relative_paths", []),
                    list,
                )
                else []
            )
        )
    ]
    material_mcp_reads = [
        event
        for event in tool_events
        if event.get("classification") == "material_information_read"
        and event.get("server") == "chemworld_lab"
    ]
    task_contract_reads = [
        event
        for event in tool_events
        if event.get("classification") == "file_read"
        and any(
            str(path).replace("\\", "/").lower().endswith(
                "reference/task_contract.json"
            )
            for path in (
                event.get("referenced_relative_paths", [])
                if isinstance(
                    event.get("referenced_relative_paths", []),
                    list,
                )
                else []
            )
        )
    ]
    artifact_access_rows = [
        access
        for receipt in receipt_rows
        for access in (
            receipt.get("artifact_access", [])
            if isinstance(receipt.get("artifact_access", []), list)
            else []
        )
        if isinstance(access, Mapping)
    ]
    named_counts = {
        name: int(classifications.get(name, 0))
        for name in (
            "lab_step",
            "status_read",
            "history_read",
            "artifact_inspect",
            "material_information_read",
            "file_read",
            "file_write",
            "other",
            "unclassified",
        )
    }
    return {
        "input_tokens": int(usage.get("input_token_count", 0)),
        "output_tokens": int(usage.get("output_token_count", 0)),
        "model_calls": int(usage.get("model_call_count", len(receipt_rows))),
        "provider_usage_pending": usage.get("provider_usage_pending") is True,
        "tool_event_count": len(tool_events),
        "tool_event_classification_counts": dict(
            sorted(classifications.items())
        ),
        **{f"{name}_count": count for name, count in named_counts.items()},
        "material_information_file_read_count": len(material_reference_reads),
        "material_information_mcp_read_count": len(material_mcp_reads),
        "task_contract_file_read_count": len(task_contract_reads),
        "artifact_access_receipt_count": len(artifact_access_rows),
        "material_information_reference_adherence": {
            "expected": True,
            "observed": bool(material_reference_reads or material_mcp_reads),
            "read_count": len(material_reference_reads) + len(material_mcp_reads),
            "transport_counts": {
                "file": len(material_reference_reads),
                "mcp": len(material_mcp_reads),
            },
            "interpretation": (
                "This is prompt/interface adherence only. It is never used "
                "to assign the material-information arm."
            ),
        },
        "provider_receipt_count": len(receipt_rows),
    }


def _receipt_experiment_tool_integrity(
    receipt: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    present = [
        key for key in _EXPERIMENT_TOOL_INTEGRITY_KEYS if key in receipt
    ]
    return (
        bool(present) and all(receipt.get(key) is True for key in present),
        present,
    )


def _provider_session_qualification(
    summary: Mapping[str, Any],
    *,
    cell_id: str,
    expected_experiments: int,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    decision_audit = summary.get("provider_decision_audit")
    if isinstance(decision_audit, Mapping):
        return _provider_decision_qualification(
            summary,
            cell_id=cell_id,
        )

    receipts = summary.get("provider_receipts")
    method_resources = summary.get("method_resources")
    declared_audit = summary.get("provider_session_audit")
    if not isinstance(receipts, list) or not all(
        isinstance(item, Mapping) for item in receipts
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider receipts are absent or malformed"
        )
    if not isinstance(method_resources, Mapping):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: method resource accounting is absent"
        )
    if not isinstance(declared_audit, Mapping):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider session audit is absent"
        )
    receipt_failures: list[dict[str, Any]] = []
    incomplete_terminal_sessions: list[int] = []
    integrity_fields_seen: set[str] = set()
    for index, receipt in enumerate(receipts, start=1):
        integrity_verified, integrity_fields = (
            _receipt_experiment_tool_integrity(receipt)
        )
        integrity_fields_seen.update(integrity_fields)
        incomplete_terminal = (
            allow_incomplete
            and receipt.get("terminal_reason") == "budget_exhausted"
            and receipt.get("final_payload_status") == "budget_exhausted"
        )
        if incomplete_terminal:
            incomplete_terminal_sessions.append(index)
        failed = [
            label
            for label, passed in {
                "status": receipt.get("status") == "completed",
                "return_code": receipt.get("return_code") == 0,
                "terminal_reason": (
                    receipt.get("terminal_reason")
                    in {"experiment_complete", "batch_discarded"}
                    or incomplete_terminal
                ),
                "final_payload_valid": (
                    receipt.get("final_payload_valid") is True
                ),
                "final_payload_status": (
                    receipt.get("final_payload_status")
                    == "experiment_complete"
                    or incomplete_terminal
                ),
                "usage_complete": receipt.get("usage_complete") is True,
                "lab_tool_integrity_verified_after_session": (
                    integrity_verified
                ),
            }.items()
            if not passed
        ]
        if failed:
            receipt_failures.append(
                {"experiment_index": index, "failed": failed}
            )
    method_failures = [
        label
        for label, passed in {
            "provider_usage_pending": (
                method_resources.get("provider_usage_pending") is False
            ),
            "provider_usage_accounting_complete": (
                method_resources.get(
                    "provider_usage_accounting_complete"
                )
                is True
            ),
            "provider_token_accounting_complete": (
                method_resources.get(
                    "provider_token_accounting_complete"
                )
                is True
            ),
            "provider_call_accounting_complete": (
                method_resources.get(
                    "provider_call_accounting_complete"
                )
                is True
            ),
            "model_call_count": (
                method_resources.get("model_call_count")
                == expected_experiments
            ),
        }.items()
        if not passed
    ]
    declared_passed = (
        declared_audit.get("passed") is True
        and declared_audit.get("target_experiment_count")
        == expected_experiments
        and declared_audit.get("receipt_count") == expected_experiments
        and declared_audit.get("receipt_count_matches_target") is True
        and declared_audit.get("all_receipts_passed") is True
        and declared_audit.get("all_method_resource_checks_passed") is True
    )
    declared_incomplete_consistent = (
        allow_incomplete
        and summary.get("run_status") == "operation_budget_exhausted_incomplete"
        and declared_audit.get("target_experiment_count") == expected_experiments
        and declared_audit.get("receipt_count") == expected_experiments
        and declared_audit.get("receipt_count_matches_target") is True
        and declared_audit.get("all_method_resource_checks_passed") is True
        and incomplete_terminal_sessions
    )
    run_status_allowed = summary.get("run_status") == "completed" or (
        allow_incomplete
        and summary.get("run_status") == "operation_budget_exhausted_incomplete"
    )
    if (
        not run_status_allowed
        or len(receipts) != expected_experiments
        or receipt_failures
        or method_failures
        or not (declared_passed or declared_incomplete_consistent)
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider session qualification failed; "
            f"receipt_count={len(receipts)}, "
            f"receipt_failures={receipt_failures[:2]}, "
            f"method_failures={method_failures}, "
            f"declared_audit_passed={declared_passed}"
        )
    return {
        "verified": True,
        "qualification_kind": "experiment_session",
        "receipt_count": len(receipts),
        "model_call_count": int(method_resources["model_call_count"]),
        "all_receipts_completed": True,
        "all_usage_accounting_complete": True,
        "all_lab_tool_integrity_verified_after_session": True,
        "all_experiment_tool_integrity_verified_after_session": True,
        "integrity_receipt_fields": sorted(integrity_fields_seen),
        "lifecycle_qualified": declared_passed,
        "right_censored": not declared_passed,
        "incomplete_terminal_sessions": incomplete_terminal_sessions,
    }


def _provider_decision_qualification(
    summary: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    """Qualify direct providers that make one decision per primitive operation."""

    receipts = summary.get("provider_receipts")
    method_resources = summary.get("method_resources")
    declared_audit = summary.get("provider_decision_audit")
    if not isinstance(receipts, list) or not all(
        isinstance(item, Mapping) for item in receipts
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider receipts are absent or malformed"
        )
    if not isinstance(method_resources, Mapping):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: method resource accounting is absent"
        )
    if not isinstance(declared_audit, Mapping):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider decision audit is absent"
        )

    decisions = declared_audit.get("decisions")
    if not isinstance(decisions, list) or not all(
        isinstance(item, Mapping) for item in decisions
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider decision rows are absent or malformed"
        )
    operation_count_raw = _first_value(
        (summary,),
        ("behavior.operation_count", "operation_count"),
    )
    if operation_count_raw is None:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: operation count is absent"
        )
    operation_count = int(operation_count_raw)
    logical_indices = [
        int(receipt.get("logical_decision_index", -1))
        for receipt in receipts
    ]
    grouped_receipt_counts = Counter(logical_indices)
    expected_indices = list(range(1, operation_count + 1))
    decision_indices = [
        int(decision.get("operation_index", -1)) for decision in decisions
    ]
    decision_attempt_counts_match = all(
        int(decision.get("attempt_count", -1))
        == grouped_receipt_counts[int(decision.get("operation_index", -1))]
        for decision in decisions
    )
    successful_final_attempts = all(
        bool(decision.get("attempts"))
        and isinstance(decision["attempts"][-1], Mapping)
        and decision["attempts"][-1].get("status") == "succeeded"
        for decision in decisions
    )
    method_checks = {
        "provider_usage_accounting_complete": method_resources.get(
            "provider_usage_accounting_complete"
        )
        is True,
        "provider_token_accounting_complete": method_resources.get(
            "provider_token_accounting_complete"
        )
        is True,
        "provider_call_accounting_complete": method_resources.get(
            "provider_call_accounting_complete"
        )
        is True,
        "model_call_count_matches_receipts": int(
            method_resources.get("model_call_count", -1)
        )
        == len(receipts),
    }
    declared_checks = {
        "passed": declared_audit.get("passed") is True,
        "all_decisions_passed": declared_audit.get("all_decisions_passed")
        is True,
        "all_method_resource_checks_passed": declared_audit.get(
            "all_method_resource_checks_passed"
        )
        is True,
        "logical_indices_match_operations": declared_audit.get(
            "logical_indices_match_operations"
        )
        is True,
        "logical_decision_count": int(
            declared_audit.get("logical_decision_count", -1)
        )
        == operation_count,
        "target_operation_count": int(
            declared_audit.get("target_operation_count", -1)
        )
        == operation_count,
        "receipt_count": int(declared_audit.get("receipt_count", -1))
        == len(receipts),
        "decision_indices": decision_indices == expected_indices,
        "receipt_indices": sorted(set(logical_indices)) == expected_indices,
        "decision_attempt_counts": decision_attempt_counts_match,
        "successful_final_attempts": successful_final_attempts,
    }
    if (
        summary.get("run_status") != "completed"
        or not all(method_checks.values())
        or not all(declared_checks.values())
    ):
        failed = [
            key
            for key, passed in {**method_checks, **declared_checks}.items()
            if not passed
        ]
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider decision qualification failed; failed={failed}"
        )
    return {
        "verified": True,
        "qualification_kind": "primitive_operation_decision",
        "receipt_count": len(receipts),
        "model_call_count": int(method_resources["model_call_count"]),
        "logical_decision_count": operation_count,
        "all_receipts_completed": True,
        "all_usage_accounting_complete": True,
        "all_lab_tool_integrity_verified_after_session": None,
        "all_experiment_tool_integrity_verified_after_session": None,
        "integrity_receipt_fields": [],
        "lifecycle_qualified": True,
        "right_censored": False,
        "incomplete_terminal_sessions": [],
    }


def _file_metrics(run_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    agent_files = [
        path
        for path in files
        if any(
            token in {part.lower() for part in path.relative_to(run_dir).parts}
            for token in ("agent", "artifacts", "experiment_documents")
        )
        or "notebook" in path.name.lower()
    ]
    return {
        "total_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "agent_or_artifact_file_count": len(agent_files),
        "agent_or_artifact_bytes": sum(path.stat().st_size for path in agent_files),
        "suffix_counts": dict(
            sorted(Counter(path.suffix.lower() or "<none>" for path in files).items())
        ),
    }


def _identity_payload(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    resource_card_sha256: str,
    run_config_sha256: str,
) -> dict[str, Any]:
    first = records[0]
    sources = (cell, summary, config, first)
    seed = int(
        _required_value(
            sources,
            ("world_seed", "seed", "task.world_seed", "cell.world_seed"),
            label="world_seed",
            cell_id=cell_id,
        )
    )
    trajectory_seed = int(
        _constant_record_value(
            records,
            ("world_seed", "seed"),
            label="world_seed",
            cell_id=cell_id,
        )
    )
    if trajectory_seed != seed:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: matrix/config world_seed does not match trajectory"
        )
    configured_seed = _first_value(
        (config, summary),
        ("task.world_seed", "world_seed", "seed"),
    )
    if configured_seed is not None and int(configured_seed) != seed:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run config/summary world_seed mismatch"
        )
    pair_config_hash = _first_value(
        (cell, config, summary, manifest),
        (
            "paired_config_sha256",
            "pair_config_sha256",
            "base_config_sha256",
            "protocol_sha256",
            "matrix_config_sha256",
        ),
    )
    if pair_config_hash is None:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: missing pair/base config hash; full config_sha256 is "
            "arm-specific and is audited separately"
        )
    material_identity_paths = {
        "material_family_id": (
            "electrochemical_material_family_id",
            "material_family_id",
        ),
        "material_family_sha256": (
            "electrochemical_material_family_sha256",
            "material_family_sha256",
        ),
        "material_instance_sha256": (
            "electrochemical_material_instance_sha256",
            "material_instance_sha256",
        ),
    }
    observed_material_identity: dict[str, str] = {}
    for label, paths in material_identity_paths.items():
        runtime_value = _first_value(
            (first, summary),
            (
                *paths,
                *(f"evaluator_provenance.{path}" for path in paths),
                *(f"physics.{path}" for path in paths),
            ),
        )
        if runtime_value is None:
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: missing runtime-recorded {label}; config declaration "
                "alone cannot prove physical pairing"
            )
        trajectory_value = _constant_record_value(
            records,
            paths,
            label=label,
            cell_id=cell_id,
            required=False,
        )
        if trajectory_value is not None and str(trajectory_value) != str(runtime_value):
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: trajectory/summary {label} mismatch"
            )
        declared_value = _first_value(
            (cell, config),
            (
                *paths,
                *(f"task.{path}" for path in paths),
                *(f"physics.{path}" for path in paths),
            ),
        )
        if declared_value is not None and str(declared_value) != str(runtime_value):
            raise AutonomousMaterialCampaignAuditError(
                f"{cell_id}: configured/runtime {label} mismatch"
            )
        observed_material_identity[label] = str(runtime_value)
    declared_card_hash = _first_value(
        (cell, config, summary),
        (
            "resource_card_sha256",
            "campaign_resource_card_sha256",
            "resource_card.card_sha256",
            "campaign_resources.card_sha256",
        ),
    )
    if (
        declared_card_hash is not None
        and str(declared_card_hash) != resource_card_sha256
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: declared/runtime resource card hash mismatch"
        )
    return {
        "world_seed": seed,
        "world_id": str(
            _constant_record_value(
                records,
                ("world_id",),
                label="world_id",
                cell_id=cell_id,
            )
        ),
        "world_family_version": str(
            _constant_record_value(
                records,
                ("world_family_version",),
                label="world_family_version",
                cell_id=cell_id,
            )
        ),
        "mechanism_hash": str(
            _constant_record_value(
                records,
                ("mechanism_hash",),
                label="mechanism_hash",
                cell_id=cell_id,
            )
        ),
        **observed_material_identity,
        "scoring_contract_hash": str(
            _constant_record_value(
                records,
                (
                    "scoring_contract_hash",
                    "evaluation_outcome.scoring_contract_hash",
                ),
                label="scoring_contract_hash",
                cell_id=cell_id,
            )
        ),
        "workflow_mode": str(
            _required_value(
                sources,
                (
                    "electrochemical_workflow_mode",
                    "workflow_mode",
                    "task.electrochemical_workflow_mode",
                    "method.electrochemical_workflow_mode",
                ),
                label="workflow_mode",
                cell_id=cell_id,
            )
        ),
        "observation_noise_mode": str(
            _required_value(
                sources,
                (
                    "observation_noise_mode",
                    "task.observation_noise_mode",
                    "pairing.observation_noise_mode",
                ),
                label="observation_noise_mode",
                cell_id=cell_id,
            )
        ),
        "observation_noise_namespace": str(
            _required_value(
                sources,
                (
                    "observation_noise_namespace",
                    "task.observation_noise_namespace",
                    "pairing.observation_noise_namespace",
                ),
                label="observation_noise_namespace",
                cell_id=cell_id,
            )
        ),
        "observation_seed": int(
            _required_value(
                sources,
                (
                    "observation_seed",
                    "observation_seed_override",
                    "task.observation_seed",
                    "pairing.observation_seed",
                ),
                label="observation_seed",
                cell_id=cell_id,
            )
        ),
        "resource_card_sha256": resource_card_sha256,
        "code_hash": str(
            _required_value(
                sources,
                (
                    "code_sha256",
                    "source_sha256",
                    "git_commit",
                    "source.git_commit",
                    "source.commit",
                ),
                label="code hash/git commit",
                cell_id=cell_id,
            )
        ),
        "pair_config_sha256": str(pair_config_hash),
        "run_config_sha256": run_config_sha256,
    }


def _audit_resource_ledger(
    *,
    cell_id: str,
    snapshot: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    expected_batches: int,
    allow_incomplete: bool = False,
) -> tuple[dict[str, Any], CampaignResourceLedger]:
    try:
        ledger = CampaignResourceLedger.from_snapshot(snapshot)
    except (CampaignResourceIntegrityError, KeyError, TypeError, ValueError) as error:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: campaign resource ledger replay failed: {error}"
        ) from error
    state = ledger.snapshot()["state"]
    vessel_starts = int(state["vessel_starts"])
    if (
        vessel_starts > expected_batches
        or (vessel_starts != expected_batches and not allow_incomplete)
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: resource ledger vessel_starts={state['vessel_starts']}, "
            f"expected {expected_batches}"
        )
    closed_batches = int(state["final_assays"]) + int(
        state.get("discarded_batches", 0)
    )
    if (
        closed_batches > expected_batches
        or closed_batches > vessel_starts
        or (closed_batches != expected_batches and not allow_incomplete)
    ):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: resource ledger closed_batches="
            f"{closed_batches}, "
            f"expected {expected_batches}"
        )
    events = snapshot.get("events")
    if not isinstance(events, list) or len(events) != len(records):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: resource ledger/trajectory event count mismatch"
        )
    alignment_mismatches: list[dict[str, Any]] = []
    for index, (event, record) in enumerate(zip(events, records, strict=True), start=1):
        if not isinstance(event, Mapping):
            alignment_mismatches.append({"event": index, "field": "event_object"})
            continue
        if to_builtin(event.get("action")) != to_builtin(_action(record)):
            alignment_mismatches.append({"event": index, "field": "action"})
        outcome = event.get("outcome")
        if not isinstance(outcome, Mapping):
            alignment_mismatches.append({"event": index, "field": "outcome"})
        elif (outcome.get("committed") is True) != _status_committed(record):
            alignment_mismatches.append({"event": index, "field": "committed"})
    if alignment_mismatches:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: resource ledger/trajectory alignment failed: "
            f"{alignment_mismatches[:3]}"
        )
    return (
        {
            "verified": True,
            "ledger_sha256": snapshot["ledger_sha256"],
            "card_sha256": ledger.card.card_sha256,
            "operation_attempts": state["operation_attempts"],
            "vessel_starts": state["vessel_starts"],
            "final_assays": state["final_assays"],
            "discarded_batches": state.get("discarded_batches", 0),
            "closed_batches": closed_batches,
            "expected_batches": expected_batches,
            "right_censored": closed_batches < expected_batches,
            "nonfinal_instrument_uses": state["nonfinal_instrument_uses"],
            "stocks_used": state["stocks_used"],
            "report_only": state["report_only"],
            "trajectory_event_alignment_verified": True,
        },
        ledger,
    )


def _audit_cell(
    *,
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    expected_batches: int,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    arm_raw = _first_value(
        (cell,),
        (
            "arm",
            "condition",
            "material_information_mode",
            "material_information",
            "task.material_information",
        ),
    )
    arm = _arm_name(arm_raw)
    seed_raw = _first_value((cell,), ("world_seed", "seed", "cell.world_seed"))
    if seed_raw is None:
        raise AutonomousMaterialCampaignAuditError("matrix cell is missing world_seed")
    seed = int(seed_raw)
    cell_id = str(cell.get("cell_id", f"world-{seed}-{arm}"))
    run_dir_raw = _required_value(
        (cell,),
        (
            "run_dir",
            "run_root",
            "output_dir",
            "path",
            "authoritative_attempt_dir",
        ),
        label="run_dir",
        cell_id=cell_id,
    )
    run_dir = _resolve_path(run_dir_raw, bases=(manifest_dir,))
    if not run_dir.is_dir():
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run_dir does not exist: {run_dir}"
        )
    config_path = _run_file(
        cell,
        run_dir,
        manifest_dir,
        explicit_keys=("config_path", "run_config_path"),
        conventional_names=("run_config.json", "config.json"),
        label="run config",
    )
    summary_path = _run_file(
        cell,
        run_dir,
        manifest_dir,
        explicit_keys=("summary_path", "run_summary_path"),
        conventional_names=("run_summary.json", "summary.json", "report.json"),
        label="run summary",
    )
    trajectory_path = _run_file(
        cell,
        run_dir,
        manifest_dir,
        explicit_keys=("trajectory_path", "trajectory"),
        conventional_names=("trajectory.jsonl", "run.jsonl"),
        label="trajectory",
    )
    config = _load_json_object(config_path, label="run config")
    summary = _load_json_object(summary_path, label="run summary")
    cell_right_censored = summary.get("run_status") != "completed"
    if cell_right_censored and not allow_incomplete:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run_status={summary.get('run_status')!r}, expected 'completed'"
        )
    configured_arm_raw = _first_value(
        (config, summary),
        (
            "material_information",
            "material_information_mode",
            "task.material_information",
            "task.material_information_mode",
        ),
    )
    if configured_arm_raw is None:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run artifacts are missing material-information mode"
        )
    configured_arm = _arm_name(configured_arm_raw)
    if configured_arm != arm:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: matrix/run material-information arm mismatch"
        )
    records = load_jsonl(trajectory_path)
    if not records:
        raise AutonomousMaterialCampaignAuditError(f"{cell_id}: empty trajectory")
    run_config_sha256 = _validate_config_hashes(
        cell_id=cell_id,
        cell=cell,
        config=config,
        summary=summary,
    )
    trajectory_sha256 = _validate_trajectory_hash(
        cell_id=cell_id,
        cell=cell,
        summary=summary,
        trajectory_path=trajectory_path,
    )
    snapshot, snapshot_source = _resource_snapshot_from_sources(
        cell_id=cell_id,
        cell=cell,
        summary=summary,
        config=config,
        run_dir=run_dir,
        manifest_dir=manifest_dir,
    )
    resource_audit, ledger = _audit_resource_ledger(
        cell_id=cell_id,
        snapshot=snapshot,
        records=records,
        expected_batches=expected_batches,
        allow_incomplete=cell_right_censored,
    )
    replay, replay_source = _replay_receipt_from_sources(
        cell_id=cell_id,
        cell=cell,
        summary=summary,
        run_dir=run_dir,
        manifest_dir=manifest_dir,
    )
    if _verified_flag(replay) is not True:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: exact trajectory replay receipt is not verified"
        )
    replay_trajectory_sha256 = _required_value(
        (replay,),
        ("trajectory_sha256", "trajectory.sha256"),
        label="replay-bound trajectory_sha256",
        cell_id=cell_id,
    )
    if str(replay_trajectory_sha256) != trajectory_sha256:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: exact replay receipt/trajectory sha256 mismatch"
        )
    replayed_operation_count = _required_value(
        (replay,),
        (
            "checked_steps",
            "trajectory_record_count",
            "replayed_operation_count",
        ),
        label="replayed operation count",
        cell_id=cell_id,
    )
    if int(replayed_operation_count) != len(records):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: exact replay receipt/trajectory operation count mismatch"
        )
    replay_ledger_sha256 = _required_value(
        (replay,),
        (
            "campaign_resource_ledger_sha256",
            "resource_ledger_sha256",
        ),
        label="replay-bound campaign resource ledger sha256",
        cell_id=cell_id,
    )
    if str(replay_ledger_sha256) != str(snapshot.get("ledger_sha256")):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: exact replay receipt/resource ledger sha256 mismatch"
        )
    identity = _identity_payload(
        cell_id=cell_id,
        cell=cell,
        config=config,
        summary=summary,
        records=records,
        manifest=manifest,
        resource_card_sha256=ledger.card.card_sha256,
        run_config_sha256=run_config_sha256,
    )
    if identity["world_seed"] != seed:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: matrix and artifact world_seed mismatch"
        )
    batch_indices = _batch_indices(records)
    final_outcomes = _final_assay_outcomes(
        records,
        batch_indices,
        expected_batches=expected_batches,
        cell_id=cell_id,
        allow_incomplete=cell_right_censored,
    )
    scores = [float(outcome["score"]) for outcome in final_outcomes]
    batch_running_best_curve = _running_best(scores)
    operation_running_best_curve = _operation_attempt_running_best(records)
    operation_budget = ledger.card.operation_attempt_limit
    if len(operation_running_best_curve) > operation_budget:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: trajectory exceeds declared operation-attempt budget"
        )
    budget_normalized_curve = (
        [
            *operation_running_best_curve,
            *(
                [operation_running_best_curve[-1]]
                * (operation_budget - len(operation_running_best_curve))
            ),
        ]
        if operation_running_best_curve
        else []
    )
    invalid_steps = [
        int(record.get("step", index))
        for index, record in enumerate(records, start=1)
        if not _status_committed(record)
    ]
    operations = [_operation(record) for record in records]
    dossier, catalog, reference_recovery = _recover_material_reference(
        config=config,
        summary=summary,
        run_dir=run_dir,
        arm=arm,
    )
    material = _material_metrics(
        records,
        batch_indices,
        expected_batches=expected_batches,
        dossier=dossier,
        catalog=catalog,
        reference_recovery=reference_recovery,
    )
    measurements = _measurement_metrics(records, batch_indices)
    setpoints = _setpoint_metrics(records, batch_indices)
    diagnostic_adaptation = _diagnostic_adaptation_metrics(
        records,
        batch_indices,
    )
    trajectory_learning = {
        "discovery_retention_recovery": (
            _discovery_retention_recovery_metrics(final_outcomes)
        ),
        "diagnostic_control_to_final": _diagnostic_control_to_final_metrics(
            final_outcomes,
            diagnostic_adaptation,
        ),
    }
    batches, policy_shifts = _batch_policy_rows(
        records,
        batch_indices,
        final_outcomes,
        expected_batches=expected_batches,
    )
    provider_sessions = _provider_session_qualification(
        summary,
        cell_id=cell_id,
        expected_experiments=expected_batches,
        allow_incomplete=cell_right_censored,
    )
    usage = _usage_metrics(summary, records)
    if usage["provider_usage_pending"]:
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: provider token usage is still pending"
        )
    summary_operation_count = _first_value(
        (summary,),
        ("behavior.operation_count", "operation_count"),
    )
    if summary_operation_count is not None and int(summary_operation_count) != len(records):
        raise AutonomousMaterialCampaignAuditError(
            f"{cell_id}: run summary/trajectory operation count mismatch"
        )
    return {
        "cell_id": cell_id,
        "world_seed": seed,
        "arm": arm,
        "material_information_mode": str(
            arm_raw.get("mode") if isinstance(arm_raw, Mapping) else arm_raw
        ),
        "run_dir": str(run_dir),
        "artifacts": {
            "config_path": str(config_path),
            "config_sha256": run_config_sha256,
            "summary_path": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": trajectory_sha256,
            "resource_snapshot_source": snapshot_source,
            "replay_source": replay_source,
        },
        "identity": identity,
        "completion": {
            "expected_vessels": expected_batches,
            "completed_vessels": len(final_outcomes),
            "discarded_vessels": int(resource_audit.get("discarded_batches", 0)),
            "closed_vessels": int(resource_audit.get("closed_batches", 0)),
            "completion_rate": len(final_outcomes) / expected_batches,
            "closed_rate": int(resource_audit.get("closed_batches", 0))
            / expected_batches,
            "complete": int(resource_audit.get("closed_batches", 0))
            == expected_batches,
            "right_censored": cell_right_censored,
            "run_status": summary.get("run_status"),
        },
        "operations": {
            "count": len(records),
            "invalid_count": len(invalid_steps),
            "invalid_rate": len(invalid_steps) / len(records),
            "invalid_steps": invalid_steps,
            "action_counts": dict(sorted(Counter(operations).items())),
            "sequence": operations,
        },
        "scores": {
            "final_score_sequence": scores,
            "final_assay_outcomes": final_outcomes,
            "final_assay_score_components": {
                component: [
                    outcome["components"].get(component)
                    for outcome in final_outcomes
                ]
                for component in sorted(
                    {
                        component
                        for outcome in final_outcomes
                        for component in outcome["components"]
                    }
                )
            },
            "best_final_score": max(scores) if scores else None,
            "final_score_mean": statistics.fmean(scores) if scores else None,
            "batch_final_assay_running_best_sequence": (
                batch_running_best_curve
            ),
            "batch_final_assay_running_best_auc": (
                statistics.fmean(batch_running_best_curve)
                if batch_running_best_curve
                else None
            ),
            "batch_final_assay_running_best_auc_definition": (
                "Discrete arithmetic mean of the post-assay running-best "
                "leaderboard score over the ordered committed final assays; "
                "one equally weighted point per completed batch."
            ),
            "operation_attempt_running_best_sequence": (
                operation_running_best_curve
            ),
            "operation_attempt_running_best_auc": statistics.fmean(
                operation_running_best_curve
            ),
            "operation_attempt_running_best_auc_definition": (
                "Discrete arithmetic mean over every submitted primitive "
                "operation attempt in the realized trajectory. At attempt t, "
                "the value is the best committed final-assay score observed "
                "through the outcome of t; it is 0 before the first committed "
                "final assay. Invalid and resource-rejected attempts are "
                "included."
            ),
            "budget_normalized_operation_attempt_running_best_auc": (
                statistics.fmean(budget_normalized_curve)
                if budget_normalized_curve
                else None
            ),
            "budget_normalized_operation_attempt_running_best_auc_definition": (
                "The operation-attempt running-best curve is right-padded "
                "with its terminal incumbent to the declared "
                f"{operation_budget}-attempt campaign budget, then averaged "
                "over that fixed budget."
            ),
            "operation_attempt_budget": operation_budget,
        },
        "materials": material,
        "measurements": measurements,
        "setpoints": setpoints,
        "diagnostic_adaptation": diagnostic_adaptation,
        "trajectory_learning": trajectory_learning,
        "batches": batches,
        "cross_batch_policy_shifts": policy_shifts,
        "method_usage": usage,
        "provider_sessions": provider_sessions,
        "files": _file_metrics(run_dir),
        "resource_ledger": resource_audit,
        "exact_replay": {
            "verified": True,
            "receipt_sha256": canonical_json_sha256(replay),
        },
    }


_PAIR_IDENTITY_FIELDS = (
    "world_seed",
    "world_id",
    "world_family_version",
    "mechanism_hash",
    "material_family_id",
    "material_family_sha256",
    "material_instance_sha256",
    "scoring_contract_hash",
    "workflow_mode",
    "observation_noise_mode",
    "observation_noise_namespace",
    "observation_seed",
    "resource_card_sha256",
    "code_hash",
    "pair_config_sha256",
)


def _paired_delta(nominal: Mapping[str, Any], opaque: Mapping[str, Any]) -> dict[str, Any]:
    nominal_scores = nominal["scores"]["final_score_sequence"]
    opaque_scores = opaque["scores"]["final_score_sequence"]
    scalar_paths = {
        "completion_rate": "completion.completion_rate",
        "operation_count": "operations.count",
        "invalid_count": "operations.invalid_count",
        "best_final_score": "scores.best_final_score",
        "final_score_mean": "scores.final_score_mean",
        "batch_final_assay_running_best_auc": (
            "scores.batch_final_assay_running_best_auc"
        ),
        "operation_attempt_running_best_auc": (
            "scores.operation_attempt_running_best_auc"
        ),
        "budget_normalized_operation_attempt_running_best_auc": (
            "scores.budget_normalized_operation_attempt_running_best_auc"
        ),
        "material_first_choice_policy_diversity": (
            "materials.predeclared_endpoints.joint_first_choice_policy."
            "unique_policy_count"
        ),
        "measurement_count": "measurements.committed_count",
        "setpoint_change_count": "setpoints.within_batch_change_count",
        "diagnostic_adaptation_change_count": (
            "diagnostic_adaptation.changed_control_event_count"
        ),
        "global_best_discovery_fraction": (
            "trajectory_learning.discovery_retention_recovery."
            "global_best_discovery_fraction"
        ),
        "online_incumbent_retention_rate": (
            "trajectory_learning.discovery_retention_recovery."
            "online_retention_rate"
        ),
        "maximum_absolute_incumbent_drawdown": (
            "trajectory_learning.discovery_retention_recovery."
            "maximum_absolute_drawdown_from_prior_incumbent"
        ),
        "terminal_to_global_best_ratio": (
            "trajectory_learning.discovery_retention_recovery."
            "terminal_to_global_best_ratio"
        ),
        "loss_episode_count": (
            "trajectory_learning.discovery_retention_recovery."
            "loss_episode_count"
        ),
        "recovered_loss_episode_count": (
            "trajectory_learning.discovery_retention_recovery."
            "recovered_loss_episode_count"
        ),
        "unresolved_loss_episode_count": (
            "trajectory_learning.discovery_retention_recovery."
            "unresolved_loss_episode_count"
        ),
        "input_tokens": "method_usage.input_tokens",
        "output_tokens": "method_usage.output_tokens",
        "tool_event_count": "method_usage.tool_event_count",
        "lab_step_count": "method_usage.lab_step_count",
        "status_read_count": "method_usage.status_read_count",
        "history_read_count": "method_usage.history_read_count",
        "artifact_inspect_count": "method_usage.artifact_inspect_count",
        "material_information_file_read_count": (
            "method_usage.material_information_file_read_count"
        ),
        "file_count": "files.total_count",
    }
    deltas: dict[str, float | None] = {}
    for name, path in scalar_paths.items():
        nominal_value = _dig(nominal, path)
        opaque_value = _dig(opaque, path)
        deltas[name] = (
            None
            if nominal_value is None or opaque_value is None
            else float(nominal_value) - float(opaque_value)
        )
    nominal_components = nominal["scores"]["final_assay_score_components"]
    opaque_components = opaque["scores"]["final_assay_score_components"]
    component_names = sorted(
        set(nominal_components) | set(opaque_components)
    )
    component_deltas = {
        component: [
            (
                None
                if left is None or right is None
                else float(left) - float(right)
            )
            for left, right in zip(
                nominal_components.get(component, [None] * len(nominal_scores)),
                opaque_components.get(component, [None] * len(opaque_scores)),
                strict=False,
            )
        ]
        for component in component_names
    }
    return {
        "world_seed": nominal["world_seed"],
        "nominal_cell_id": nominal["cell_id"],
        "opaque_cell_id": opaque["cell_id"],
        "nominal_minus_opaque": {
            **deltas,
            "final_score_sequence": [
                float(left) - float(right)
                for left, right in zip(
                    nominal_scores, opaque_scores, strict=False
                )
            ],
            "paired_final_assay_count": min(
                len(nominal_scores), len(opaque_scores)
            ),
            "nominal_final_assay_count": len(nominal_scores),
            "opaque_final_assay_count": len(opaque_scores),
            "final_assay_score_components": component_deltas,
        },
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(float(value) for value in values)


def _trajectory_learning_arm_aggregate(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    discovery_rows = [
        cell["trajectory_learning"]["discovery_retention_recovery"]
        for cell in cells
    ]
    loss_episodes = [
        episode
        for row in discovery_rows
        for episode in row["loss_episodes"]
    ]
    recovered = [episode for episode in loss_episodes if episode["recovered"]]
    control_rows = [
        row
        for cell in cells
        for row in cell["trajectory_learning"][
            "diagnostic_control_to_final"
        ]["per_final_assay"]
    ]
    changed_rows = [
        row
        for row in control_rows
        if row["any_diagnostic_aligned_control_change"]
        and row["previous_final_score"] is not None
    ]
    unchanged_rows = [
        row
        for row in control_rows
        if row["comparable_control_event_count"] > 0
        and not row["any_diagnostic_aligned_control_change"]
        and row["previous_final_score"] is not None
    ]

    def observed_mean(field: str) -> float | None:
        return _mean_or_none(
            [
                float(value)
                for row in discovery_rows
                if (value := row.get(field)) is not None
            ]
        )

    def pooled_conversion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        positive_count = sum(
            row["positive_delta_vs_previous_final"] is True for row in rows
        )
        incumbent_count = sum(row["new_incumbent"] is True for row in rows)
        return {
            "eligible_batch_count": len(rows),
            "positive_next_final_delta_count": positive_count,
            "positive_next_final_delta_rate": (
                positive_count / len(rows) if rows else None
            ),
            "new_incumbent_count": incumbent_count,
            "new_incumbent_rate": (
                incumbent_count / len(rows) if rows else None
            ),
            "mean_next_final_delta_vs_previous": _mean_or_none(
                [float(row["score_delta_vs_previous_final"]) for row in rows]
            ),
            "mean_next_final_delta_vs_pre_batch_incumbent": _mean_or_none(
                [
                    float(row["score_delta_vs_pre_batch_incumbent"])
                    for row in rows
                ]
            ),
        }

    return {
        "cell_count": len(cells),
        "retention_fraction": TRAJECTORY_RETENTION_FRACTION,
        "mean_global_best_discovery_fraction": observed_mean(
            "global_best_discovery_fraction"
        ),
        "mean_incumbent_update_count": observed_mean(
            "incumbent_update_count"
        ),
        "mean_online_retention_rate": observed_mean("online_retention_rate"),
        "mean_maximum_absolute_drawdown": observed_mean(
            "maximum_absolute_drawdown_from_prior_incumbent"
        ),
        "mean_terminal_to_global_best_ratio": observed_mean(
            "terminal_to_global_best_ratio"
        ),
        "loss_episode_count": len(loss_episodes),
        "recovered_loss_episode_count": len(recovered),
        "unresolved_loss_episode_count": len(loss_episodes) - len(recovered),
        "pooled_recovery_rate": (
            len(recovered) / len(loss_episodes) if loss_episodes else None
        ),
        "mean_recovery_delay_final_assays": _mean_or_none(
            [
                float(episode["recovery_delay_final_assays"])
                for episode in recovered
            ]
        ),
        "mean_recovery_delay_batches": _mean_or_none(
            [float(episode["recovery_delay_batches"]) for episode in recovered]
        ),
        "diagnostic_control_to_final": {
            "changed_control": pooled_conversion(changed_rows),
            "comparable_without_change": pooled_conversion(unchanged_rows),
        },
    }


def audit_autonomous_material_campaign(
    manifest_path: str | Path,
    *,
    expected_world_seeds: Sequence[int] | None = None,
    expected_vessels_per_cell: int = 6,
    allow_incomplete_cells: bool = False,
) -> dict[str, Any]:
    """Validate and summarize a 5-world x 2-information-condition matrix."""

    manifest_file = Path(manifest_path)
    manifest = _load_json_object(manifest_file, label="matrix manifest")
    manifest_dir = manifest_file.resolve().parent
    expected_seeds = tuple(
        int(value)
        for value in (
            expected_world_seeds
            if expected_world_seeds is not None
            else manifest.get("world_seeds", DEFAULT_WORLD_SEEDS)
        )
    )
    if (
        len(expected_seeds) != 5
        or len(set(expected_seeds)) != 5
        or expected_vessels_per_cell <= 0
    ):
        raise AutonomousMaterialCampaignAuditError(
            "audit requires exactly five unique world seeds and a positive vessel count"
        )
    raw_cells = _manifest_cells(manifest)
    if len(raw_cells) != len(expected_seeds) * len(EXPECTED_ARMS):
        raise AutonomousMaterialCampaignAuditError(
            f"matrix has {len(raw_cells)} cells; expected "
            f"{len(expected_seeds) * len(EXPECTED_ARMS)}"
        )
    cells = [
        _audit_cell(
            cell=cell,
            manifest=manifest,
            manifest_dir=manifest_dir,
            expected_batches=expected_vessels_per_cell,
            allow_incomplete=allow_incomplete_cells,
        )
        for cell in raw_cells
    ]
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    for cell in cells:
        key = (int(cell["world_seed"]), str(cell["arm"]))
        if key in keyed:
            raise AutonomousMaterialCampaignAuditError(
                f"duplicate matrix cell for seed={key[0]}, arm={key[1]}"
            )
        keyed[key] = cell
    expected_keys = {
        (seed, arm) for seed in expected_seeds for arm in EXPECTED_ARMS
    }
    if set(keyed) != expected_keys:
        missing = sorted(expected_keys - set(keyed))
        extra = sorted(set(keyed) - expected_keys)
        raise AutonomousMaterialCampaignAuditError(
            f"matrix arm/seed coverage mismatch; missing={missing}, extra={extra}"
        )

    pair_rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        opaque = keyed[(seed, OPAQUE_ARM)]
        nominal = keyed[(seed, NOMINAL_ARM)]
        mismatches = [
            field
            for field in _PAIR_IDENTITY_FIELDS
            if opaque["identity"][field] != nominal["identity"][field]
        ]
        if mismatches:
            raise AutonomousMaterialCampaignAuditError(
                f"world seed {seed}: nominal/opaque physical pairing mismatch: "
                + ", ".join(mismatches)
            )
        pair_rows.append(
            {
                **_paired_delta(nominal, opaque),
                "identity_match": True,
                "matched_identity_fields": list(_PAIR_IDENTITY_FIELDS),
            }
        )

    ordered_cells = [
        keyed[(seed, arm)] for seed in expected_seeds for arm in EXPECTED_ARMS
    ]
    incomplete_cell_ids = [
        str(cell["cell_id"])
        for cell in ordered_cells
        if cell["completion"]["right_censored"]
    ]
    arm_aggregates: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        arm_cells = [keyed[(seed, arm)] for seed in expected_seeds]
        arm_aggregates[arm] = {
            "cell_count": len(arm_cells),
            "mean_completion_rate": _mean(
                [cell["completion"]["completion_rate"] for cell in arm_cells]
            ),
            "mean_operation_count": _mean(
                [cell["operations"]["count"] for cell in arm_cells]
            ),
            "mean_invalid_count": _mean(
                [cell["operations"]["invalid_count"] for cell in arm_cells]
            ),
            "mean_best_final_score": _mean(
                [cell["scores"]["best_final_score"] for cell in arm_cells]
            ),
            "mean_batch_final_assay_running_best_auc": _mean(
                [
                    cell["scores"][
                        "batch_final_assay_running_best_auc"
                    ]
                    for cell in arm_cells
                ]
            ),
            "mean_operation_attempt_running_best_auc": _mean(
                [
                    cell["scores"]["operation_attempt_running_best_auc"]
                    for cell in arm_cells
                ]
            ),
            "mean_budget_normalized_operation_attempt_running_best_auc": _mean(
                [
                    cell["scores"][
                        "budget_normalized_operation_attempt_running_best_auc"
                    ]
                    for cell in arm_cells
                ]
            ),
            "mean_measurement_count": _mean(
                [cell["measurements"]["committed_count"] for cell in arm_cells]
            ),
            "mean_input_tokens": _mean(
                [cell["method_usage"]["input_tokens"] for cell in arm_cells]
            ),
            "mean_output_tokens": _mean(
                [cell["method_usage"]["output_tokens"] for cell in arm_cells]
            ),
            "trajectory_learning": _trajectory_learning_arm_aggregate(
                arm_cells
            ),
        }
    paired_metric_names = (
        "completion_rate",
        "best_final_score",
        "final_score_mean",
        "batch_final_assay_running_best_auc",
        "operation_attempt_running_best_auc",
        "budget_normalized_operation_attempt_running_best_auc",
        "operation_count",
        "invalid_count",
        "measurement_count",
        "material_first_choice_policy_diversity",
        "setpoint_change_count",
        "diagnostic_adaptation_change_count",
        "global_best_discovery_fraction",
        "online_incumbent_retention_rate",
        "maximum_absolute_incumbent_drawdown",
        "terminal_to_global_best_ratio",
        "loss_episode_count",
        "recovered_loss_episode_count",
        "unresolved_loss_episode_count",
        "input_tokens",
        "output_tokens",
        "tool_event_count",
        "lab_step_count",
        "status_read_count",
        "history_read_count",
        "artifact_inspect_count",
        "material_information_file_read_count",
        "file_count",
    )
    paired_descriptive: dict[str, Any] = {}
    for name in paired_metric_names:
        values = [
            float(value)
            for row in pair_rows
            if (value := row["nominal_minus_opaque"][name]) is not None
        ]
        paired_descriptive[name] = {
            "n_pairs": len(pair_rows),
            "n_observed_pairs": len(values),
            "mean_nominal_minus_opaque": _mean(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    report: dict[str, Any] = {
        "schema_version": AUTONOMOUS_MATERIAL_CAMPAIGN_AUDIT_VERSION,
        "status": (
            "completed_audited_descriptive_matrix_with_right_censoring"
            if incomplete_cell_ids
            else "completed_audited_descriptive_matrix"
        ),
        "formal_result": False,
        "manifest": {
            "path": str(manifest_file),
            "sha256": canonical_json_sha256(manifest),
            "schema_version": manifest.get("schema_version"),
        },
        "matrix": {
            "world_seeds": list(expected_seeds),
            "arms": list(EXPECTED_ARMS),
            "cell_count": len(ordered_cells),
            "expected_vessels_per_cell": expected_vessels_per_cell,
            "all_cells_complete": all(
                cell["completion"]["complete"] for cell in ordered_cells
            ),
            "right_censored_cell_count": len(incomplete_cell_ids),
            "right_censored_cell_ids": incomplete_cell_ids,
            "all_resource_ledgers_verified": all(
                cell["resource_ledger"]["verified"] for cell in ordered_cells
            ),
            "all_exact_replays_verified": all(
                cell["exact_replay"]["verified"] for cell in ordered_cells
            ),
            "all_provider_sessions_verified": all(
                cell["provider_sessions"]["verified"]
                for cell in ordered_cells
            ),
            "all_pairs_physically_matched": all(
                row["identity_match"] for row in pair_rows
            ),
        },
        "cells": ordered_cells,
        "arm_descriptive_aggregates": arm_aggregates,
        "paired_worlds": pair_rows,
        "paired_descriptive_aggregates": paired_descriptive,
        "interpretation": {
            "analysis_unit": "paired physical world",
            "paired_estimand": "nominal minus opaque",
            "n_pairs": len(pair_rows),
            "descriptive_only": True,
            "confirmatory_claim_allowed": False,
            "caveat": (
                "n=5 paired worlds is a small exploratory sample. Report effect "
                "heterogeneity and paired descriptive deltas; do not treat these "
                "cells as an independently powered confirmatory test or infer a "
                "general material-information effect."
                + (
                    " In addition, one or more cells are right-censored after "
                    "autonomous lifecycle/resource failure; paired sequence "
                    "contrasts use only jointly observed final assays, and all "
                    "aggregate results remain developmental."
                    if incomplete_cell_ids
                    else ""
                )
            ),
        },
    }
    report["audit_sha256"] = canonical_json_sha256(report)
    return report


def render_autonomous_material_campaign_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render the audited matrix as a compact UTF-8 Chinese Markdown report."""

    def optional(value: Any, spec: str) -> str:
        return "—" if value is None else format(float(value), spec)

    lines = [
        "# 自主逐操作材料信息配对实验审计",
        "",
        (
            "状态: 10 个 cell 均通过物理配对、资源账本重放、轨迹精确重放"
            "和 provider 决策/会话完整性审计。"
            if report["matrix"]["all_cells_complete"]
            else "状态: 矩阵未完整完成。"
        ),
        "",
        "| seed | arm | 完成率 | 操作/无效 | final sequence | best | "
        "batch AUC | attempt AUC | fixed-budget AUC | 诊断 | "
        "材料首选策略数 | input/output tokens | lab/status/history/artifact |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        scores = ", ".join(
            f"{value:.3f}"
            for value in cell["scores"]["final_score_sequence"]
        )
        material_policy_count = cell["materials"][
            "predeclared_endpoints"
        ]["joint_first_choice_policy"]["unique_policy_count"]
        lines.append(
            f"| {cell['world_seed']} | {cell['arm']} | "
            f"{cell['completion']['completion_rate']:.0%} | "
            f"{cell['operations']['count']}/{cell['operations']['invalid_count']} | "
            f"{scores} | {cell['scores']['best_final_score']:.4f} | "
            f"{cell['scores']['batch_final_assay_running_best_auc']:.4f} | "
            f"{cell['scores']['operation_attempt_running_best_auc']:.4f} | "
            f"{cell['scores']['budget_normalized_operation_attempt_running_best_auc']:.4f} | "
            f"{cell['measurements']['committed_count']} | "
            f"{material_policy_count} | "
            f"{cell['method_usage']['input_tokens']}/"
            f"{cell['method_usage']['output_tokens']} | "
            f"{cell['method_usage']['lab_step_count']}/"
            f"{cell['method_usage']['status_read_count']}/"
            f"{cell['method_usage']['history_read_count']}/"
            f"{cell['method_usage']['artifact_inspect_count']} |"
        )
    lines.extend(
        [
            "",
            "## 发现—保留—恢复轨迹",
            "",
            "| seed | arm | 最佳首次批次 | incumbent 更新 | 在线保留率 | "
            "最佳后保留率 | 最大回撤 | 终点/最佳 | loss:恢复/未恢复 | "
            "诊断改控后正增量 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in report["cells"]:
        discovery = cell["trajectory_learning"][
            "discovery_retention_recovery"
        ]
        conversion = cell["trajectory_learning"][
            "diagnostic_control_to_final"
        ]["changed_control"]
        post_best_retention = discovery["post_global_best_retention_rate"]
        conversion_rate = conversion["positive_next_final_delta_rate"]
        lines.append(
            f"| {cell['world_seed']} | {cell['arm']} | "
            f"{discovery['global_best_first_batch_number']} | "
            f"{discovery['incumbent_update_count']} | "
            f"{optional(discovery['online_retention_rate'], '.0%')} | "
            f"{('—' if post_best_retention is None else f'{post_best_retention:.0%}')} | "
            f"{optional(discovery['maximum_absolute_drawdown_from_prior_incumbent'], '.4f')} | "
            f"{optional(discovery['terminal_to_global_best_ratio'], '.0%')} | "
            f"{discovery['loss_episode_count']}:"
            f"{discovery['recovered_loss_episode_count']}/"
            f"{discovery['unresolved_loss_episode_count']} | "
            f"{('—' if conversion_rate is None else f'{conversion_rate:.0%}')} "
            f"({conversion['positive_next_final_delta_count']}/"
            f"{conversion['eligible_batch_count']}) |"
        )
    lines.extend(
        [
            "",
            "## 两臂轨迹汇总",
            "",
            "| arm | 平均最佳发现进度 | 平均在线保留率 | 平均最大回撤 | "
            "平均终点/最佳 | loss:恢复/未恢复 | 诊断改控后正增量 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in EXPECTED_ARMS:
        aggregate = report["arm_descriptive_aggregates"][arm][
            "trajectory_learning"
        ]
        conversion = aggregate["diagnostic_control_to_final"][
            "changed_control"
        ]
        conversion_rate = conversion["positive_next_final_delta_rate"]
        lines.append(
            f"| {arm} | {optional(aggregate['mean_global_best_discovery_fraction'], '.0%')} | "
            f"{optional(aggregate['mean_online_retention_rate'], '.0%')} | "
            f"{optional(aggregate['mean_maximum_absolute_drawdown'], '.4f')} | "
            f"{optional(aggregate['mean_terminal_to_global_best_ratio'], '.0%')} | "
            f"{aggregate['loss_episode_count']}:"
            f"{aggregate['recovered_loss_episode_count']}/"
            f"{aggregate['unresolved_loss_episode_count']} | "
            f"{('—' if conversion_rate is None else f'{conversion_rate:.0%}')} "
            f"({conversion['positive_next_final_delta_count']}/"
            f"{conversion['eligible_batch_count']}) |"
        )
    lines.extend(
        [
            "",
            "## 每个世界的 nominal - opaque 配对差",
            "",
            "| seed | Δcompletion | Δbest | Δbatch AUC | Δattempt AUC | "
            "Δfixed-budget AUC | Δoperations | Δinvalid | "
            "Δmeasurements | Δmaterial diversity | Δinput tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["paired_worlds"]:
        delta = row["nominal_minus_opaque"]
        lines.append(
            f"| {row['world_seed']} | {delta['completion_rate']:+.1%} | "
            f"{delta['best_final_score']:+.4f} | "
            f"{delta['batch_final_assay_running_best_auc']:+.4f} | "
            f"{delta['operation_attempt_running_best_auc']:+.4f} | "
            f"{delta['budget_normalized_operation_attempt_running_best_auc']:+.4f} | "
            f"{delta['operation_count']:+.0f} | "
            f"{delta['invalid_count']:+.0f} | "
            f"{delta['measurement_count']:+.0f} | "
            f"{delta['material_first_choice_policy_diversity']:+.0f} | "
            f"{delta['input_tokens']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "## 每个世界的 nominal - opaque 轨迹差",
            "",
            "| seed | Δ最佳发现进度 | Δ在线保留率 | Δ最大回撤 | "
            "Δ终点/最佳 | Δloss | Δ恢复 | Δ未恢复 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["paired_worlds"]:
        delta = row["nominal_minus_opaque"]
        lines.append(
            f"| {row['world_seed']} | "
            f"{optional(delta['global_best_discovery_fraction'], '+.0%')} | "
            f"{optional(delta['online_incumbent_retention_rate'], '+.0%')} | "
            f"{optional(delta['maximum_absolute_incumbent_drawdown'], '+.4f')} | "
            f"{optional(delta['terminal_to_global_best_ratio'], '+.0%')} | "
            f"{optional(delta['loss_episode_count'], '+.0f')} | "
            f"{optional(delta['recovered_loss_episode_count'], '+.0f')} | "
            f"{optional(delta['unresolved_loss_episode_count'], '+.0f')} |"
        )
    lines.extend(
        [
            "",
            "## 指标定义与审计边界",
            "",
            f"- {report['interpretation']['caveat']}",
            "- batch AUC: 每次已提交 final assay 后的 running-best "
            "分数离散均值, 每个 batch 等权。",
            "- attempt AUC: 每次原子操作尝试结果返回后的 running-best "
            "分数离散均值; 首次 final assay 前取 0, 无效和资源拒绝操作也计入。",
            "- fixed-budget AUC: 将 attempt 曲线以最终 incumbent 右侧填充至"
            "固定操作预算后求均值。",
            "- 发现进度: 全局最佳首次出现位置映射到 0—1; 0 表示首批, "
            "1 表示末批。该值越大只表示发现越晚, 不表示更优。",
            "- 在线保留: 每个后续 final score 达到此前 incumbent 的 90% "
            "即视为保留。低于阈值开启 loss episode; 恢复使用开启时冻结的"
            " incumbent 阈值, 终局未恢复的时延按右删失记录。",
            "- 诊断改控转化率: 至少一个诊断对齐控制发生可比较字段变化的"
            " batch 中, final score 高于上一 batch 的比例。它是时间对齐描述, "
            "不是因果效应。",
            "- 材料首选、名义描述符秩、诊断后控制变化、跨 batch "
            "适应和分项终点评分均保存在 JSON 明细中。",
            "- 对材料信息文件的读取仅表示接口遵循, 不用于判定实验 arm。",
            f"- 审计内容哈希: `{report['audit_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_autonomous_material_campaign_audit(
    manifest_path: str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    expected_world_seeds: Sequence[int] | None = None,
    expected_vessels_per_cell: int = 6,
    allow_incomplete_cells: bool = False,
) -> dict[str, Any]:
    """Audit a matrix and atomically write JSON plus Markdown."""

    report = audit_autonomous_material_campaign(
        manifest_path,
        expected_world_seeds=expected_world_seeds,
        expected_vessels_per_cell=expected_vessels_per_cell,
        allow_incomplete_cells=allow_incomplete_cells,
    )
    write_json_atomic(Path(json_path), report)
    markdown_file = Path(markdown_path)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text(
        render_autonomous_material_campaign_markdown(report),
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--expected-vessels", type=int, default=6)
    parser.add_argument(
        "--allow-incomplete-cells",
        action="store_true",
        help=(
            "Write a development-only right-censored report for audited "
            "resource/lifecycle failures; formal fail-closed behavior remains default."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = write_autonomous_material_campaign_audit(
        args.manifest,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        expected_vessels_per_cell=args.expected_vessels,
        allow_incomplete_cells=args.allow_incomplete_cells,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "cell_count": report["matrix"]["cell_count"],
                "audit_sha256": report["audit_sha256"],
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTONOMOUS_MATERIAL_CAMPAIGN_AUDIT_VERSION",
    "AutonomousMaterialCampaignAuditError",
    "audit_autonomous_material_campaign",
    "render_autonomous_material_campaign_markdown",
    "write_autonomous_material_campaign_audit",
]
