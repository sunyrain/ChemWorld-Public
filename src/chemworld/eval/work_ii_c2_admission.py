"""Fail-closed admission contract for the complete Work II C2 programme."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_ae_formal_cohort import (
    validate_ae_public_cells,
    validate_formal_ae_qualification,
)
from chemworld.eval.work_ii_execution_mode import (
    ExecutionMode,
    validate_execution_envelope,
)

C2_ADMISSION_PLAN_VERSION = "chemworld-work-ii-c2-admission-plan-0.1"
C2_ADMISSION_REPORT_VERSION = "chemworld-work-ii-c2-admission-report-0.1"
C2_TASK_ADMISSION_RECEIPT_VERSION = "chemworld-work-ii-c2-task-admission-receipt-0.2"
C2_OUTCOME_BLIND_SELECTION_VERSION = (
    "chemworld-work-ii-c2-outcome-blind-selection-0.3"
)
C2_SELECTION_PROTOCOL_VERSION = "chemworld-work-ii-c2-selection-protocol-0.1"
C2_LOCI = ("A_P", "A_S")
C2_REQUIRED_TASK_COUNTS = {"A_P": 2, "A_S": 2}
C2_REQUIRED_ROUNDS = {"A_P": 10, "A_S": 12}
C2_TASK_STAGE_ORDER = ("Q1", "Q2", "D1")
C2_REQUIRED_CHECKPOINTS = {
    "A_P": (0, 2, 4, 7, 10),
    "A_S": (0, 3, 6, 9, 12),
}
C2_CAMPAIGN_LOCUS_NAMES = {
    "A_P": {"A_P", "parametric", "parametric_dynamical"},
    "A_S": {"A_S", "structural", "structural_mechanistic"},
}
C2_STAGE_SCHEMA_TOKENS = {
    "Q1": ("qualification", "mechanism-oracle"),
    "Q2": ("matched-prior",),
    "D1": ("initial-model-pilot-evaluation",),
}
C2_PUBLIC_AE_CELL_COUNT = 75
C2_DYNAMIC_EVIDENCE_ROOT = "workstreams/flagship_tasks/reports"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def c2_admission_sha256(report: Mapping[str, Any]) -> str:
    return _self_hash(report, "admission_sha256")


def c2_task_admission_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return _self_hash(receipt, "receipt_sha256")


def c2_outcome_blind_selection_sha256(record: Mapping[str, Any]) -> str:
    return _self_hash(record, "selection_sha256")


def c2_selection_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return _self_hash(protocol, "protocol_sha256")


def build_c2_selection_protocol(
    *, locus: str, candidate_roster: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the protected, evidence-independent candidate ranking protocol.

    Eligibility is deliberately absent: it is derived later from terminal
    qualification state.  Writing this object under ``configs/benchmark`` and
    committing it before qualification evidence is collected is what freezes the
    roster and rule; a dynamic selection record cannot create that assertion.
    """

    if locus not in C2_LOCI:
        raise ValueError(f"unsupported C2 locus: {locus}")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    ranks: set[int] = set()
    for raw in candidate_roster:
        task_id = raw.get("task_id")
        rank = raw.get("frozen_rank")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in identities
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or rank in ranks
        ):
            raise ValueError("C2 selection protocol roster is malformed or not uniquely ranked")
        identities.add(task_id)
        ranks.add(rank)
        rows.append({"task_id": task_id, "frozen_rank": rank})
    if len(rows) < C2_REQUIRED_TASK_COUNTS[locus]:
        raise ValueError("C2 selection protocol has too few candidates")
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError("C2 selection protocol ranks must be contiguous from one")
    protocol: dict[str, Any] = {
        "schema_version": C2_SELECTION_PROTOCOL_VERSION,
        "locus": locus,
        "status": "frozen_before_eligible_terminal_receipts",
        "formal_participant_outcomes_observed_at_freeze": 0,
        "eligible_terminal_task_receipts_observed_at_freeze": 0,
        "candidate_roster": sorted(rows, key=lambda row: int(row["frozen_rank"])),
        "selection_rule": {
            "method": "terminally_eligible_then_ascending_frozen_rank",
            "required_selected_task_count": C2_REQUIRED_TASK_COUNTS[locus],
            "all_candidates_require_terminal_eligibility_disposition": True,
            "formal_participant_outcomes_permitted": False,
        },
    }
    protocol["protocol_sha256"] = c2_selection_protocol_sha256(protocol)
    return protocol


def _selection_protocol_errors(protocol: Mapping[str, Any], *, locus: str) -> list[str]:
    errors: list[str] = []
    if protocol.get("schema_version") != C2_SELECTION_PROTOCOL_VERSION:
        errors.append("unexpected C2 selection-protocol schema")
    if protocol.get("protocol_sha256") != c2_selection_protocol_sha256(protocol):
        errors.append("C2 selection protocol self-hash mismatch")
    if (
        protocol.get("locus") != locus
        or protocol.get("status") != "frozen_before_eligible_terminal_receipts"
        or protocol.get("formal_participant_outcomes_observed_at_freeze") != 0
        or protocol.get("eligible_terminal_task_receipts_observed_at_freeze") != 0
    ):
        errors.append("C2 selection protocol does not prove a pre-evidence freeze")
    roster = protocol.get("candidate_roster")
    if not isinstance(roster, list):
        errors.append("C2 selection protocol lacks its candidate roster")
        return errors
    try:
        rebuilt = build_c2_selection_protocol(
            locus=locus,
            candidate_roster=[
                dict(row) if isinstance(row, Mapping) else {} for row in roster
            ],
        )
    except (TypeError, ValueError) as error:
        errors.append(f"C2 selection protocol cannot be rebuilt: {error}")
    else:
        if dict(protocol) != rebuilt:
            errors.append("C2 selection protocol differs from deterministic rebuild")
    return errors


def build_c2_outcome_blind_selection_record(
    root: Path,
    *,
    locus: str,
    task_id: str,
    selection_protocol_path: Path,
    terminal_eligibility: Mapping[str, Mapping[str, Any]],
    selection_slot: int,
) -> dict[str, Any]:
    """Freeze one task choice without using formal participant outcomes.

    The protected protocol owns candidate identities, ranks and the rule.  The
    dynamic input may only add terminal eligibility dispositions.  This builder
    never asserts that a caller-supplied roster or rule was pre-frozen.
    """

    if locus not in C2_LOCI:
        raise ValueError(f"unsupported C2 locus: {locus}")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("C2 selection task_id must be non-empty")
    protocol_path = _inside_root(root, selection_protocol_path, label="selection protocol")
    protected_root = (root.resolve() / "configs/benchmark").resolve()
    if not protocol_path.is_file() or not protocol_path.is_relative_to(protected_root):
        raise ValueError("C2 selection protocol must be a protected configs/benchmark file")
    protocol = _load_object(protocol_path)
    protocol_errors = _selection_protocol_errors(protocol, locus=locus)
    if protocol_errors:
        raise ValueError("invalid C2 selection protocol: " + "; ".join(protocol_errors))
    selection_rule = protocol["selection_rule"]
    selected_count = selection_rule.get("required_selected_task_count")
    if (
        isinstance(selection_slot, bool)
        or not isinstance(selection_slot, int)
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count != C2_REQUIRED_TASK_COUNTS[locus]
        or not 1 <= selection_slot <= selected_count
    ):
        raise ValueError("C2 selection rule has an invalid frozen slot contract")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    ranks: set[int] = set()
    protocol_roster = protocol["candidate_roster"]
    if set(terminal_eligibility) != {
        str(row["task_id"]) for row in protocol_roster
    }:
        raise ValueError("terminal eligibility must dispose every frozen candidate exactly once")
    for raw in protocol_roster:
        row = dict(raw)
        candidate_task = row.get("task_id")
        rank = row.get("frozen_rank")
        disposition = terminal_eligibility.get(str(candidate_task), {})
        if (
            not isinstance(candidate_task, str)
            or not candidate_task
            or candidate_task in identities
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or rank in ranks
            or not isinstance(disposition.get("terminal_qualification_passed"), bool)
            or not isinstance(disposition.get("disposition"), str)
            or disposition.get("disposition") not in {"eligible", "ineligible_retained"}
            or (disposition.get("terminal_qualification_passed") is True)
            != (disposition.get("disposition") == "eligible")
        ):
            raise ValueError("C2 candidate roster is malformed or not uniquely ranked")
        identities.add(candidate_task)
        ranks.add(rank)
        rows.append(
            {
                "task_id": candidate_task,
                "frozen_rank": rank,
                "terminal_qualification_passed": disposition[
                    "terminal_qualification_passed"
                ],
                "disposition": disposition["disposition"],
            }
        )
    eligible = sorted(
        (row for row in rows if row["terminal_qualification_passed"]),
        key=lambda row: int(row["frozen_rank"]),
    )
    if (
        len(eligible) < selected_count
        or eligible[selection_slot - 1]["task_id"] != task_id
    ):
        raise ValueError("selected C2 task does not occupy its eligible frozen slot")
    record: dict[str, Any] = {
        "schema_version": C2_OUTCOME_BLIND_SELECTION_VERSION,
        "locus": locus,
        "task_id": task_id,
        "selected_before_formal_participant_outcomes": True,
        "formal_participant_outcomes_observed": 0,
        "formal_participant_outcomes_used": False,
        "selection_rule": dict(selection_rule),
        "candidate_roster": sorted(rows, key=lambda row: int(row["frozen_rank"])),
        "selection_slot": selection_slot,
        "required_selected_task_count": selected_count,
        "selected_frozen_rank": eligible[selection_slot - 1]["frozen_rank"],
        "selection_protocol_binding": {
            "path": protocol_path.relative_to(root.resolve()).as_posix(),
            "sha256": file_sha256(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
        },
    }
    record["selection_sha256"] = c2_outcome_blind_selection_sha256(record)
    return record


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _release_execution_context_errors(
    root: Path,
    execution_context: object,
    *,
    label: str,
) -> list[str]:
    """Require one release-authorized envelope without rehashing the source tree."""

    if not isinstance(execution_context, Mapping):
        return [f"{label} execution context is missing"]
    errors = [
        f"{label}: {error}"
        for error in validate_execution_envelope(root, execution_context)
    ]
    if (
        execution_context.get("execution_mode") != ExecutionMode.RELEASE.value
        or execution_context.get("release_eligible") is not True
        or execution_context.get("c2_admission_authorized") is not True
    ):
        errors.append(f"{label} is not release-authorized for C2 admission")
    return errors


def _execution_cohort_key(execution_context: object) -> tuple[str, str] | None:
    if not isinstance(execution_context, Mapping):
        return None
    freeze_id = execution_context.get("freeze_id")
    tested_commit = execution_context.get("tested_commit")
    if not isinstance(freeze_id, str) or not _is_commit(tested_commit):
        return None
    return freeze_id, str(tested_commit)


def _binding(root: Path, path: Path, *, embedded_hash: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(path),
    }
    if embedded_hash is not None:
        value["embedded_sha256"] = embedded_hash
    return value


def _inside_root(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes repository") from error
    return resolved


def _dynamic_evidence_path_errors(root: Path, path: Path, *, label: str) -> list[str]:
    """Keep generated evidence separate from frozen execution inputs."""

    resolved = _inside_root(root, path, label=label)
    dynamic_root = (root.resolve() / C2_DYNAMIC_EVIDENCE_ROOT).resolve()
    try:
        resolved.relative_to(dynamic_root)
    except ValueError:
        return [
            f"{label} must be under {C2_DYNAMIC_EVIDENCE_ROOT} to preserve the "
            "generated-evidence boundary"
        ]
    return []


def _embedded_hash_contract(report: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    fields = [field for field in ("summary_sha256", "report_sha256") if field in report]
    if len(fields) != 1:
        return None, ["stage report must contain exactly one supported self-hash field"]
    field = fields[0]
    observed = report.get(field)
    expected = _self_hash(report, field)
    if not isinstance(observed, str) or observed != expected:
        return None, [f"stage report {field} mismatch"]
    return observed, []


def _stage_status_errors(
    report: Mapping[str, Any],
    *,
    stage: str,
    task_id: str,
) -> list[str]:
    errors: list[str] = []
    if stage not in C2_TASK_STAGE_ORDER:
        return [f"unsupported C2 task-admission stage: {stage}"]
    if report.get("execution_mode") == ExecutionMode.DEVELOPMENT.value:
        errors.append(f"{stage} development report cannot support terminal admission")
    if report.get("release_eligible") is False:
        errors.append(f"{stage} non-release report cannot support terminal admission")
    schema = report.get("schema_version")
    qualification_schema = report.get("qualification_schema_version")
    if not isinstance(schema, str) or not any(
        token in schema or (isinstance(qualification_schema, str) and token in qualification_schema)
        for token in C2_STAGE_SCHEMA_TOKENS[stage]
    ):
        errors.append(f"{stage} report has an unsupported schema")
    if report.get("task_id") != task_id:
        errors.append(f"{stage} report task does not match campaign task")
    if report.get("formal_result") is not False:
        errors.append(f"{stage} report does not preserve the non-formal boundary")
    if stage == "Q1":
        q0 = report.get("q0")
        if not isinstance(q0, Mapping) or q0.get("passed") is not True:
            errors.append("Q1 report does not embed a passed Q0 reachability audit")
        if report.get("qualification_passed") is not True:
            errors.append("Q1 report did not pass")
        worlds = report.get("worlds")
        if (
            report.get("world_seeds") != [0, 1, 2, 3, 4]
            or not isinstance(worlds, list)
            or len(worlds) != 5
            or any(not _q1_world_passed(world, seed) for seed, world in enumerate(worlds))
        ):
            errors.append("Q1 report is not a five-world terminal pass")
    elif stage == "Q2":
        if report.get("qualification_passed") is not True:
            errors.append("Q2 report did not pass")
        worlds = report.get("worlds")
        if (
            report.get("world_seeds") != [0, 1, 2, 3, 4]
            or not isinstance(worlds, list)
            or len(worlds) != 5
            or any(
                not isinstance(world, Mapping)
                or world.get("world_seed") != seed
                or world.get("qualification_passed") is not True
                for seed, world in enumerate(worlds)
            )
        ):
            errors.append("Q2 report is not a five-world terminal pass")
        if report.get("provider_call_count") != 0:
            errors.append("Q2 report is not provider-free")
    elif stage == "D1":
        if report.get("status") != "passed":
            errors.append("D1 report did not pass")
        denominators = report.get("denominators")
        denominators = denominators if isinstance(denominators, Mapping) else {}
        if (
            denominators.get("participant_cell_count") != 3
            or denominators.get("participant_completed_cell_count") != 3
            or denominators.get("participant_terminal_trajectory_count") != 3
            or denominators.get("participant_platform_failure_count") != 0
        ):
            errors.append("D1 report is not a complete clean three-arm terminal pilot")
        action_layer = report.get("action_layer")
        action_layer = action_layer if isinstance(action_layer, Mapping) else {}
        if action_layer.get("status") != "participant_interpretable":
            errors.append(
                "D1 action layer is not participant-interpretable and cannot support "
                "terminal admission"
            )
        if action_layer.get("submitted_recommendations_replaced") is not False:
            errors.append("D1 action layer replaced submitted recommendations")
    provider_calls = report.get("provider_call_count")
    if stage in {"Q0", "Q1", "Q2"} and provider_calls not in {None, 0}:
        errors.append(f"{stage} report is not provider-free")
    return errors


def _q1_world_passed(world: object, seed: int) -> bool:
    if not isinstance(world, Mapping) or world.get("world_seed") != seed:
        return False
    analysis = world.get("analysis")
    return isinstance(analysis, Mapping) and analysis.get("passed") is True


def _stage_evidence_row(
    root: Path,
    *,
    stage: str,
    path: Path,
    task_id: str,
    require_release_context: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    path = _inside_root(root, path, label=f"{stage} evidence")
    errors = _dynamic_evidence_path_errors(root, path, label=f"{stage} evidence")
    report: dict[str, Any] = {}
    embedded_hash: str | None = None
    if not path.is_file():
        errors.append(f"{stage} report is missing")
    else:
        report = _load_object(path)
        embedded_hash, hash_errors = _embedded_hash_contract(report)
        errors.extend(hash_errors)
        errors.extend(_stage_status_errors(report, stage=stage, task_id=task_id))
        execution_context = report.get("execution_context")
        if require_release_context or execution_context is not None:
            errors.extend(
                _release_execution_context_errors(
                    root,
                    execution_context,
                    label=f"{stage} report",
                )
            )
    row: dict[str, Any] = {
        "stage": stage,
        "execution_context": (
            dict(report["execution_context"])
            if isinstance(report.get("execution_context"), Mapping)
            else None
        ),
        "report_binding": (
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "embedded_sha256": embedded_hash,
            }
            if path.is_file()
            else {"path": path.relative_to(root).as_posix()}
        ),
        "passed": not errors,
        "validation_errors": errors,
    }
    return row, errors


def _campaign_errors(
    config: Mapping[str, Any],
    *,
    locus: str,
    task_id: str,
) -> list[str]:
    errors: list[str] = []
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    intervention = config.get("intervention")
    intervention = intervention if isinstance(intervention, Mapping) else {}
    rounds = C2_REQUIRED_ROUNDS[locus]
    if config.get("task_id") != task_id:
        errors.append("campaign task does not match requested task")
    if config.get("formal_result") is not False:
        errors.append("campaign does not preserve the non-formal admission boundary")
    if intervention.get("locus") not in C2_CAMPAIGN_LOCUS_NAMES[locus]:
        errors.append("campaign locus does not match requested C2 locus")
    if (
        campaign.get("complete_experiments") != rounds
        or campaign.get("checkpoint_complete_experiments")
        != list(C2_REQUIRED_CHECKPOINTS[locus])
    ):
        errors.append("campaign rounds/checkpoints do not match the C2 locus")
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    if qualification.get("q2_passed") is not True:
        errors.append("campaign is not bound to a passed Q2 design")
    return errors


def _selection_errors(
    record: Mapping[str, Any],
    *,
    root: Path,
    locus: str,
    task_id: str,
) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != C2_OUTCOME_BLIND_SELECTION_VERSION:
        errors.append("unexpected outcome-blind selection schema")
    if record.get("selection_sha256") != c2_outcome_blind_selection_sha256(record):
        errors.append("outcome-blind selection self-hash mismatch")
    if (
        record.get("locus") != locus
        or record.get("task_id") != task_id
        or record.get("selected_before_formal_participant_outcomes") is not True
        or record.get("formal_participant_outcomes_observed") != 0
        or record.get("formal_participant_outcomes_used") is not False
    ):
        errors.append("selection record does not prove outcome-blind task selection")
    selection_rule = record.get("selection_rule")
    roster = record.get("candidate_roster")
    protocol_binding = record.get("selection_protocol_binding")
    if (
        not isinstance(selection_rule, Mapping)
        or not isinstance(roster, list)
        or not isinstance(protocol_binding, Mapping)
    ):
        errors.append("selection record lacks its protected protocol binding or dispositions")
    else:
        relative = protocol_binding.get("path")
        try:
            protocol_path = _inside_root(
                root, root / str(relative), label="selection protocol"
            )
            protocol = _load_object(protocol_path)
            if (
                not protocol_path.is_relative_to((root / "configs/benchmark").resolve())
                or protocol_binding.get("sha256") != file_sha256(protocol_path)
                or protocol_binding.get("protocol_sha256")
                != protocol.get("protocol_sha256")
            ):
                errors.append("selection record protected protocol binding is stale")
            errors.extend(_selection_protocol_errors(protocol, locus=locus))
            terminal_eligibility = {
                str(row.get("task_id")): {
                    "terminal_qualification_passed": row.get(
                        "terminal_qualification_passed"
                    ),
                    "disposition": row.get("disposition"),
                }
                for row in roster
                if isinstance(row, Mapping)
            }
            rebuilt = build_c2_outcome_blind_selection_record(
                root,
                locus=locus,
                task_id=task_id,
                selection_protocol_path=protocol_path,
                terminal_eligibility=terminal_eligibility,
                selection_slot=int(record.get("selection_slot", 0)),
            )
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"selection record cannot be deterministically rebuilt: {error}")
        else:
            if dict(record) != rebuilt:
                errors.append("selection record differs from deterministic outcome-blind rebuild")
    return errors


def validate_c2_outcome_blind_selection_pair(
    records: Sequence[Mapping[str, Any]], *, locus: str
) -> list[str]:
    """Require one shared roster/rule and the exact two frozen slots per locus."""

    if locus not in C2_LOCI:
        return [f"unsupported C2 locus: {locus}"]
    errors: list[str] = []
    if len(records) != C2_REQUIRED_TASK_COUNTS[locus]:
        return [f"{locus} requires exactly two outcome-blind selection records"]

    slots: list[object] = []
    task_ids: list[object] = []
    rosters: list[object] = []
    rules: list[object] = []
    protocol_bindings: list[object] = []
    for record in records:
        slots.append(record.get("selection_slot"))
        task_ids.append(record.get("task_id"))
        rosters.append(record.get("candidate_roster"))
        rule = record.get("selection_rule")
        rules.append(dict(rule) if isinstance(rule, Mapping) else None)
        protocol_bindings.append(record.get("selection_protocol_binding"))
    if set(slots) != {1, 2} or len(slots) != len(set(slots)):
        errors.append(f"{locus} outcome-blind selection slots must be exactly {{1,2}}")
    if (
        any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        or len(set(task_ids)) != C2_REQUIRED_TASK_COUNTS[locus]
    ):
        errors.append(f"{locus} outcome-blind selected task identities must be distinct")
    if (
        not isinstance(rosters[0], list)
        or not isinstance(rosters[1], list)
        or rosters[0] != rosters[1]
    ):
        errors.append(
            f"{locus} selection records do not share the exact candidate roster"
        )
    if (
        rules[0] is None or rules[1] is None or rules[0] != rules[1]
    ):
        errors.append(f"{locus} selection records do not share one protected rule")
    if (
        not isinstance(protocol_bindings[0], Mapping)
        or not isinstance(protocol_bindings[1], Mapping)
        or protocol_bindings[0] != protocol_bindings[1]
    ):
        errors.append(f"{locus} selection records do not bind one protected protocol")
    return errors


def _selection_pair_summary(
    records: Sequence[Mapping[str, Any]], *, locus: str
) -> tuple[dict[str, Any], list[str]]:
    errors = validate_c2_outcome_blind_selection_pair(records, locus=locus)
    common_roster = (
        records[0].get("candidate_roster")
        if len(records) == 2
        and records[0].get("candidate_roster") == records[1].get("candidate_roster")
        else None
    )
    rules = [
        (dict(rule) if isinstance(rule, Mapping) else None)
        for record in records
        for rule in (record.get("selection_rule"),)
    ]
    common_rule = rules[0] if len(rules) == 2 and rules[0] == rules[1] else None
    summary = {
        "required_selection_slots": [1, 2],
        "observed_selection_slots": sorted(
            slot
            for slot in (record.get("selection_slot") for record in records)
            if isinstance(slot, int) and not isinstance(slot, bool)
        ),
        "selected_task_ids": [record.get("task_id") for record in records],
        "shared_candidate_roster_sha256": (
            canonical_json_sha256(common_roster) if common_roster is not None else None
        ),
        "shared_selection_rule_without_slot_sha256": (
            canonical_json_sha256(common_rule) if common_rule is not None else None
        ),
        "shared_selection_protocol_binding": (
            records[0].get("selection_protocol_binding")
            if len(records) == 2
            and records[0].get("selection_protocol_binding")
            == records[1].get("selection_protocol_binding")
            else None
        ),
        "selection_record_sha256": [
            record.get("selection_sha256") for record in records
        ],
        "passed": not errors,
        "validation_errors": errors,
    }
    return summary, errors


def build_c2_task_admission_receipt(
    root: Path,
    *,
    locus: str,
    task_id: str,
    campaign_config_path: Path,
    stage_report_paths: Mapping[str, Path],
    selection_record_path: Path,
) -> dict[str, Any]:
    """Build a deterministic terminal-task receipt from independently validated evidence."""

    root = root.resolve()
    if locus not in C2_LOCI:
        raise ValueError(f"unsupported C2 locus: {locus}")
    if set(stage_report_paths) != set(C2_TASK_STAGE_ORDER):
        raise ValueError("stage evidence roster must contain exactly Q1, Q2 and D1")
    campaign_path = _inside_root(root, campaign_config_path, label="campaign config")
    selection_path = _inside_root(root, selection_record_path, label="selection record")
    errors: list[str] = []
    errors.extend(
        _dynamic_evidence_path_errors(root, campaign_path, label="campaign config")
    )
    errors.extend(
        _dynamic_evidence_path_errors(root, selection_path, label="selection record")
    )
    campaign: dict[str, Any] = {}
    if campaign_path.is_file():
        campaign = _load_object(campaign_path)
        errors.extend(_campaign_errors(campaign, locus=locus, task_id=task_id))
    else:
        errors.append("campaign config is missing")

    selection: dict[str, Any] = {}
    if selection_path.is_file():
        selection = _load_object(selection_path)
        errors.extend(
            _selection_errors(
                selection,
                root=root,
                locus=locus,
                task_id=task_id,
            )
        )
    else:
        errors.append("outcome-blind selection record is missing")

    stages: list[dict[str, Any]] = []
    for stage in C2_TASK_STAGE_ORDER:
        row, stage_errors = _stage_evidence_row(
            root,
            stage=stage,
            path=stage_report_paths[stage],
            task_id=task_id,
        )
        stages.append(row)
        errors.extend(stage_errors)

    passed = not errors
    receipt: dict[str, Any] = {
        "schema_version": C2_TASK_ADMISSION_RECEIPT_VERSION,
        "status": (
            "passed_terminal_task_admission" if passed else "not_ready_fail_closed"
        ),
        "formal_result": False,
        "terminal_qualification_passed": passed,
        "locus": locus,
        "task_id": task_id,
        "complete_experiments_per_cell": C2_REQUIRED_ROUNDS[locus],
        "campaign_config_binding": (
            _binding(root, campaign_path) if campaign_path.is_file() else None
        ),
        "stage_evidence_order": list(C2_TASK_STAGE_ORDER),
        "stage_evidence": stages,
        "outcome_blind_selection_binding": (
            _binding(
                root,
                selection_path,
                embedded_hash=str(selection.get("selection_sha256", "")),
            )
            if selection_path.is_file()
            else None
        ),
        "participant_outcomes_used_for_selection": False,
        "formal_participant_outcomes_observed": 0,
        "validation_errors": errors,
    }
    receipt["receipt_sha256"] = c2_task_admission_receipt_sha256(receipt)
    return receipt


def _schedule_binding(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "block": "A_E",
        "public_schedule_cell_count": len(cells),
        "public_schedule_sha256": canonical_json_sha256(list(cells)),
        "schedule_owner": "formal_preflight.cells",
    }


def _plan_errors(plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != C2_ADMISSION_PLAN_VERSION:
        errors.append("unexpected C2 admission plan schema")
    if (
        plan.get("program_scope") != "C2"
        or plan.get("status") not in {"not_ready_fail_closed", "candidate_evidence_frozen"}
        or plan.get("formal_execution_allowed") is not False
    ):
        errors.append("C2 admission plan does not preserve its non-execution boundary")
    required = plan.get("required_blocks")
    required = required if isinstance(required, Mapping) else {}
    if set(required) != {"A_E", "A_P", "A_S"}:
        errors.append("C2 admission plan does not contain exactly A_E, A_P and A_S")
    ae = required.get("A_E")
    ae = ae if isinstance(ae, Mapping) else {}
    if ae.get("public_schedule_cell_count") != C2_PUBLIC_AE_CELL_COUNT:
        errors.append("C2 admission plan changed the 75-cell A_E public subblock")
    for locus in C2_LOCI:
        block = required.get(locus)
        block = block if isinstance(block, Mapping) else {}
        if (
            block.get("required_terminal_task_count") != C2_REQUIRED_TASK_COUNTS[locus]
            or block.get("complete_experiments_per_cell") != C2_REQUIRED_ROUNDS[locus]
            or not isinstance(block.get("task_admission_receipt_paths"), list)
        ):
            errors.append(f"C2 admission plan changed the frozen {locus} contract")
    calibration = plan.get("resource_calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    calibration_manifest_path = calibration.get("manifest_path")
    if calibration.get("work_item") != "W2-26" or not isinstance(
        calibration_manifest_path, str
    ):
        errors.append("C2 admission plan changed the W2-26 contract")
    elif not calibration_manifest_path.startswith(f"{C2_DYNAMIC_EVIDENCE_ROOT}/"):
        errors.append("W2-26 execution manifest must use the dynamic evidence root")
    freeze = plan.get("freeze_contract")
    freeze = freeze if isinstance(freeze, Mapping) else {}
    expected_freeze = {
        "all_blocks_share_one_runtime_commit": True,
        "clean_worktree_required_at_admission": True,
        "partial_A_E_launch_forbidden": True,
        "participant_outcomes_before_admission": 0,
        "outcome_based_task_selection_forbidden": True,
    }
    if dict(freeze) != expected_freeze:
        errors.append("C2 admission plan changed the shared-freeze contract")
    return errors


def _task_receipt_errors(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    locus: str,
    require_release_context: bool = False,
) -> list[str]:
    errors: list[str] = []
    task_id = receipt.get("task_id")
    if receipt.get("schema_version") != C2_TASK_ADMISSION_RECEIPT_VERSION:
        errors.append(f"{locus} task admission has an unexpected schema")
    if receipt.get("receipt_sha256") != c2_task_admission_receipt_sha256(receipt):
        errors.append(f"{locus} task admission receipt self-hash mismatch: {task_id}")
    if (
        receipt.get("status") != "passed_terminal_task_admission"
        or receipt.get("formal_result") is not False
        or receipt.get("terminal_qualification_passed") is not True
        or receipt.get("participant_outcomes_used_for_selection") is not False
        or receipt.get("formal_participant_outcomes_observed") != 0
        or receipt.get("locus") != locus
        or receipt.get("complete_experiments_per_cell") != C2_REQUIRED_ROUNDS[locus]
        or not isinstance(task_id, str)
        or not task_id
    ):
        errors.append(f"{locus} task admission is not a terminal outcome-blind pass: {task_id}")
    stage_order = receipt.get("stage_evidence_order")
    stages = receipt.get("stage_evidence")
    stages = stages if isinstance(stages, list) else []
    stage_cohorts: list[tuple[str, str]] = []
    if stage_order != list(C2_TASK_STAGE_ORDER) or [
        row.get("stage") for row in stages if isinstance(row, Mapping)
    ] != list(C2_TASK_STAGE_ORDER):
        errors.append(f"{locus} task admission stage roster is not frozen: {task_id}")
    for row in stages:
        if not isinstance(row, Mapping):
            errors.append(f"{locus} task admission has malformed stage evidence: {task_id}")
            continue
        stage = str(row.get("stage"))
        if stage not in C2_TASK_STAGE_ORDER:
            errors.append(f"{locus} task admission has unsupported stage: {task_id}.{stage}")
            continue
        report_binding = row.get("report_binding")
        report_binding = report_binding if isinstance(report_binding, Mapping) else {}
        relative = report_binding.get("path")
        if not isinstance(relative, str):
            errors.append(f"{locus} task admission lacks {stage} evidence: {task_id}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"{locus} task admission {stage} evidence escapes repository")
            continue
        rebuilt, stage_errors = _stage_evidence_row(
            root,
            stage=stage,
            path=path,
            task_id=str(task_id),
            require_release_context=require_release_context,
        )
        if dict(row) != rebuilt:
            errors.append(f"{locus} task admission {stage} evidence is stale: {task_id}")
        errors.extend(f"{stage}: {error}" for error in stage_errors)
        cohort = _execution_cohort_key(rebuilt.get("execution_context"))
        if cohort is not None:
            stage_cohorts.append(cohort)
    if require_release_context and (
        len(stage_cohorts) != len(C2_TASK_STAGE_ORDER)
        or len(set(stage_cohorts)) != 1
    ):
        errors.append(
            f"{locus} task admission stages do not share one release freeze and commit: "
            f"{task_id}"
        )
    selection_binding = receipt.get("outcome_blind_selection_binding")
    selection_binding = (
        selection_binding if isinstance(selection_binding, Mapping) else {}
    )
    selection_relative = selection_binding.get("path")
    if not isinstance(selection_relative, str):
        errors.append(f"{locus} task admission lacks outcome-blind selection: {task_id}")
    else:
        selection_path = (root / selection_relative).resolve()
        try:
            selection_path.relative_to(root)
        except ValueError:
            errors.append(f"{locus} task selection binding escapes repository: {task_id}")
        else:
            errors.extend(
                _dynamic_evidence_path_errors(
                    root, selection_path, label="selection record"
                )
            )
            if not selection_path.is_file():
                errors.append(f"{locus} task selection record is missing: {task_id}")
            else:
                selection = _load_object(selection_path)
                errors.extend(
                    _selection_errors(
                        selection,
                        root=root,
                        locus=locus,
                        task_id=str(task_id),
                    )
                )
                expected_selection = _binding(
                    root,
                    selection_path,
                    embedded_hash=str(selection.get("selection_sha256", "")),
                )
                if dict(selection_binding) != expected_selection:
                    errors.append(f"{locus} task selection binding is stale: {task_id}")
    for label in ("campaign_config_binding",):
        binding = receipt.get(label)
        binding = binding if isinstance(binding, Mapping) else {}
        relative = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            errors.append(f"{locus} task admission lacks {label}: {task_id}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"{locus} task admission binding escapes repository: {task_id}")
            continue
        errors.extend(
            _dynamic_evidence_path_errors(root, path, label="campaign config")
        )
        if not path.is_file() or file_sha256(path) != digest:
            errors.append(f"{locus} task admission binding is stale: {task_id}.{label}")
            continue
        config = _load_object(path)
        errors.extend(_campaign_errors(config, locus=locus, task_id=str(task_id)))
    if receipt.get("validation_errors") != []:
        errors.append(f"{locus} task admission retains validation errors: {task_id}")
    return errors


def _ae_qualification_errors(
    root: Path,
    report_path: Path,
    design: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not report_path.is_file():
        return None, ["A_E prior qualification report is missing"]
    report = _load_object(report_path)
    errors = validate_formal_ae_qualification(root, report_path, design)
    if report.get("status") != "passed":
        errors.append("A_E prior qualification did not pass")
    errors.extend(
        _release_execution_context_errors(
            root,
            report.get("execution_context"),
            label="A_E prior qualification",
        )
    )
    return report, errors


def _resource_calibration_errors(
    root: Path,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not manifest_path.is_file():
        return None, ["W2-26 resource calibration manifest is missing"]
    if not summary_path.is_file():
        return None, ["W2-26 resource calibration summary is missing"]
    manifest = _load_object(manifest_path)
    summary = _load_object(summary_path)
    from chemworld.eval.work_ii_resource_calibration_v02 import (
        EXPECTED_PATTERN_KEYS,
        validate_manifest,
        validate_summary,
    )

    errors = validate_manifest(root, manifest)
    errors.extend(validate_summary(summary, manifest=manifest))
    if (
        summary.get("status") != "passed"
        or summary.get("calibration_passed") is not True
        or summary.get("method_qualification_may_be_authorized") is not True
    ):
        errors.append("W2-26 resource calibration did not pass")

    expected_keys = set(EXPECTED_PATTERN_KEYS)
    reserve_fraction_by_locus = {"A_E": 0.15, "A_P": 0.15, "A_S": 0.20}
    cards = summary.get("resource_card_proposals")
    cards = cards if isinstance(cards, list) else []
    observed_keys: list[tuple[str, str, int]] = []
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            errors.append(f"W2-26 resource card {index} is malformed")
            continue
        identity = card.get("card_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        locus = identity.get("locus")
        task_id = identity.get("task_id")
        rounds = identity.get("rounds")
        if (
            not isinstance(locus, str)
            or not isinstance(task_id, str)
            or isinstance(rounds, bool)
            or not isinstance(rounds, int)
        ):
            errors.append(f"W2-26 resource card {index} has an invalid identity")
            continue
        key = (locus, task_id, rounds)
        observed_keys.append(key)
        if card.get("protected_closeout_reserve_enforced") is not True:
            errors.append(f"W2-26 resource card does not enforce closeout: {key}")
        formula_binding = identity.get("resource_formula_binding")
        formula_binding = (
            formula_binding if isinstance(formula_binding, Mapping) else {}
        )
        formula = formula_binding.get("formula")
        formula = formula if isinstance(formula, Mapping) else {}
        process_formula = formula.get("process_time_formula")
        process_formula = (
            process_formula if isinstance(process_formula, Mapping) else {}
        )
        expected_fraction = reserve_fraction_by_locus.get(locus)
        if (
            expected_fraction is None
            or process_formula.get("protected_reserve_fraction")
            != expected_fraction
        ):
            errors.append(f"W2-26 resource card closeout fraction differs: {key}")
    if (
        len(observed_keys) != len(EXPECTED_PATTERN_KEYS)
        or len(set(observed_keys)) != len(observed_keys)
        or set(observed_keys) != expected_keys
    ):
        errors.append("W2-26 requires the exact nine task resource cards")
    return summary, errors


def build_c2_admission_report(
    root: Path,
    plan_path: Path,
    design_path: Path,
    ae_public_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a truthful current C2 admission report without executing experiments."""

    root = root.resolve()
    plan_path = plan_path.resolve()
    design_path = design_path.resolve()
    plan = _load_object(plan_path)
    design = _load_object(design_path)
    blockers: list[str] = []
    evidence_errors: list[str] = _plan_errors(plan)
    evidence_cohorts: list[tuple[str, str]] = []

    required = plan.get("required_blocks")
    required = required if isinstance(required, Mapping) else {}
    task_rows: dict[str, list[dict[str, Any]]] = {locus: [] for locus in C2_LOCI}
    selection_records: dict[str, list[dict[str, Any]]] = {
        locus: [] for locus in C2_LOCI
    }
    selection_pair_summaries: dict[str, dict[str, Any]] = {}
    for locus in C2_LOCI:
        block = required.get(locus)
        block = block if isinstance(block, Mapping) else {}
        paths = block.get("task_admission_receipt_paths")
        paths = paths if isinstance(paths, list) else []
        if len(paths) != C2_REQUIRED_TASK_COUNTS[locus]:
            blockers.append(
                f"{locus} requires exactly {C2_REQUIRED_TASK_COUNTS[locus]} "
                "terminal task admissions"
            )
        for relative in paths:
            if not isinstance(relative, str):
                evidence_errors.append(f"{locus} task admission path is invalid")
                continue
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                evidence_errors.append(f"{locus} task admission path escapes repository")
                continue
            evidence_errors.extend(
                _dynamic_evidence_path_errors(
                    root, path, label="task admission receipt"
                )
            )
            if not path.is_file():
                evidence_errors.append(f"{locus} task admission receipt is missing: {relative}")
                continue
            receipt = _load_object(path)
            receipt_errors = _task_receipt_errors(
                root,
                receipt,
                locus=locus,
                require_release_context=True,
            )
            stages = receipt.get("stage_evidence")
            stages = stages if isinstance(stages, list) else []
            stage_cohorts = {
                cohort
                for row in stages
                if isinstance(row, Mapping)
                for cohort in (_execution_cohort_key(row.get("execution_context")),)
                if cohort is not None
            }
            if len(stage_cohorts) == 1:
                evidence_cohorts.append(next(iter(stage_cohorts)))
            evidence_errors.extend(receipt_errors)
            selection_binding = receipt.get("outcome_blind_selection_binding")
            selection_binding = (
                selection_binding if isinstance(selection_binding, Mapping) else {}
            )
            selection_relative = selection_binding.get("path")
            if isinstance(selection_relative, str):
                selection_path = (root / selection_relative).resolve()
                try:
                    selection_path.relative_to(root)
                except ValueError:
                    pass
                else:
                    if selection_path.is_file():
                        selection_records[locus].append(_load_object(selection_path))
            task_rows[locus].append(
                {
                    "task_id": receipt.get("task_id"),
                    "receipt_binding": _binding(
                        root,
                        path,
                        embedded_hash=str(receipt.get("receipt_sha256", "")),
                    ),
                    "passed": not receipt_errors,
                }
            )
        task_ids = [row.get("task_id") for row in task_rows[locus]]
        if len(task_ids) != len(set(task_ids)):
            evidence_errors.append(f"{locus} task admission roster contains duplicates")
        pair_summary, pair_errors = _selection_pair_summary(
            selection_records[locus], locus=locus
        )
        selection_pair_summaries[locus] = pair_summary
        evidence_errors.extend(pair_errors)

    ae_block = required.get("A_E")
    ae_block = ae_block if isinstance(ae_block, Mapping) else {}
    ae_path_value = ae_block.get("prior_qualification_report_path")
    ae_report: dict[str, Any] | None = None
    ae_errors: list[str] = []
    if not isinstance(ae_path_value, str) or not ae_path_value:
        blockers.append("A_E prior distinguishability qualification is missing")
    else:
        ae_path = (root / ae_path_value).resolve()
        ae_report, ae_errors = _ae_qualification_errors(
            root,
            ae_path,
            design,
        )
        evidence_errors.extend(ae_errors)
        ae_context = ae_report.get("execution_context") if ae_report else None
        ae_cohort = _execution_cohort_key(ae_context)
        if ae_cohort is not None:
            evidence_cohorts.append(ae_cohort)

    calibration = plan.get("resource_calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    manifest_value = calibration.get("manifest_path")
    summary_value = calibration.get("summary_path")
    calibration_summary: dict[str, Any] | None = None
    calibration_errors: list[str] = []
    if not isinstance(summary_value, str) or not summary_value:
        blockers.append("W2-26 resource calibration is missing")
    elif not isinstance(manifest_value, str) or not manifest_value:
        evidence_errors.append("W2-26 resource calibration manifest path is invalid")
    else:
        evidence_errors.extend(
            _dynamic_evidence_path_errors(
                root,
                (root / manifest_value).resolve(),
                label="W2-26 execution manifest",
            )
        )
        evidence_errors.extend(
            _dynamic_evidence_path_errors(
                root,
                (root / summary_value).resolve(),
                label="W2-26 calibration summary",
            )
        )
        calibration_summary, calibration_errors = _resource_calibration_errors(
            root,
            (root / manifest_value).resolve(),
            (root / summary_value).resolve(),
        )
        evidence_errors.extend(calibration_errors)
    expected_release_contexts = 5
    shared_cohorts = set(evidence_cohorts)
    if (
        len(evidence_cohorts) != expected_release_contexts
        or len(shared_cohorts) != 1
    ):
        blockers.append(
            "A_E, two A_P and two A_S do not prove one shared release freeze and commit"
        )
    release_cohort = next(iter(shared_cohorts)) if len(shared_cohorts) == 1 else None
    release_freeze_id = release_cohort[0] if release_cohort is not None else None
    runtime_commit = release_cohort[1] if release_cohort is not None else None

    schedule = _schedule_binding(ae_public_cells)
    if design.get("schema_version") == "chemworld-work-ii-formal-design-0.2":
        evidence_errors.extend(validate_ae_public_cells(root, design, ae_public_cells))
    if (
        len(ae_public_cells) != C2_PUBLIC_AE_CELL_COUNT
        or ae_block.get("public_schedule_cell_count") != C2_PUBLIC_AE_CELL_COUNT
    ):
        evidence_errors.append("A_E public schedule is not the frozen 75-cell subblock")

    blockers.extend(f"invalid evidence: {error}" for error in evidence_errors)
    ready = not blockers
    report: dict[str, Any] = {
        "schema_version": C2_ADMISSION_REPORT_VERSION,
        "status": "ready_for_formal_authorization" if ready else "not_ready_fail_closed",
        "program_scope": "C2",
        "formal_result": False,
        "formal_execution_allowed": ready,
        "plan_binding": _binding(root, plan_path),
        "design_binding": {
            "path": design_path.relative_to(root).as_posix(),
            "sha256": canonical_json_sha256(design),
        },
        "shared_release_freeze_id": release_freeze_id,
        "shared_runtime_commit": runtime_commit,
        "blocks": {
            "A_E": {
                "public_schedule": schedule,
                "prior_qualification_binding": (
                    None
                    if ae_report is None or not isinstance(ae_path_value, str)
                    else _binding(
                        root,
                        (root / ae_path_value).resolve(),
                        embedded_hash=str(ae_report.get("report_sha256", "")),
                    )
                ),
                "passed": ae_report is not None and not ae_errors,
            },
            "A_P": {
                "required_terminal_task_count": 2,
                "task_admissions": task_rows["A_P"],
                "outcome_blind_selection_pair": selection_pair_summaries["A_P"],
                "passed": len(task_rows["A_P"]) == 2
                and all(row["passed"] for row in task_rows["A_P"])
                and selection_pair_summaries["A_P"]["passed"] is True,
            },
            "A_S": {
                "required_terminal_task_count": 2,
                "task_admissions": task_rows["A_S"],
                "outcome_blind_selection_pair": selection_pair_summaries["A_S"],
                "passed": len(task_rows["A_S"]) == 2
                and all(row["passed"] for row in task_rows["A_S"])
                and selection_pair_summaries["A_S"]["passed"] is True,
            },
        },
        "resource_calibration": {
            "work_item": "W2-26",
            "summary_binding": (
                None
                if calibration_summary is None or not isinstance(summary_value, str)
                else _binding(
                    root,
                    (root / summary_value).resolve(),
                    embedded_hash=str(calibration_summary.get("summary_sha256", "")),
                )
            ),
            "passed": calibration_summary is not None and not calibration_errors,
        },
        "blocking_requirements": blockers,
        "evidence_validation_errors": evidence_errors,
    }
    report["admission_sha256"] = c2_admission_sha256(report)
    return report


def validate_c2_admission_report(
    root: Path,
    report: Mapping[str, Any],
    plan_path: Path,
    design_path: Path,
    ae_public_cells: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Rebuild every evidence binding and reject forged ready-state reports."""

    errors: list[str] = []
    if report.get("schema_version") != C2_ADMISSION_REPORT_VERSION:
        errors.append("unexpected C2 admission report schema")
    if report.get("admission_sha256") != c2_admission_sha256(report):
        errors.append("C2 admission report self-hash mismatch")
    expected = build_c2_admission_report(root, plan_path, design_path, ae_public_cells)
    if dict(report) != expected:
        errors.append("C2 admission report differs from deterministic evidence rebuild")
    ready = report.get("status") == "ready_for_formal_authorization"
    if ready != (report.get("formal_execution_allowed") is True):
        errors.append("C2 admission report has an inconsistent authorization state")
    if ready and (
        report.get("blocking_requirements") != []
        or report.get("evidence_validation_errors") != []
    ):
        errors.append("C2 admission report claims readiness without complete clean evidence")
    return errors


__all__ = [
    "C2_ADMISSION_PLAN_VERSION",
    "C2_ADMISSION_REPORT_VERSION",
    "C2_DYNAMIC_EVIDENCE_ROOT",
    "C2_OUTCOME_BLIND_SELECTION_VERSION",
    "C2_SELECTION_PROTOCOL_VERSION",
    "C2_TASK_ADMISSION_RECEIPT_VERSION",
    "C2_TASK_STAGE_ORDER",
    "build_c2_admission_report",
    "build_c2_outcome_blind_selection_record",
    "build_c2_selection_protocol",
    "build_c2_task_admission_receipt",
    "c2_admission_sha256",
    "c2_outcome_blind_selection_sha256",
    "c2_selection_protocol_sha256",
    "c2_task_admission_receipt_sha256",
    "validate_c2_admission_report",
    "validate_c2_outcome_blind_selection_pair",
]
