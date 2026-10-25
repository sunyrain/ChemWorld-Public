"""Sealed one-shot private-confirmation preflight for Work II.

Private world identities stay outside Git.  The public repository contains only
their canonical seal commitment and this validator.  A private preflight can be
materialized only after the public formal analysis is complete and hash-bound;
the resulting schedule must remain under the ignored ``runs/private`` tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_confirmatory import validate_confirmatory_analysis
from chemworld.eval.work_ii_formal import (
    EXPECTED_PRIVATE_CONFIRMATION_CONTRACT,
    FORMAL_ARMS,
    validate_formal_preflight,
)

WORK_II_PRIVATE_SEAL_VERSION = "chemworld-work-ii-private-world-seal-0.1"
WORK_II_PRIVATE_PREFLIGHT_VERSION = "chemworld-work-ii-private-confirmation-preflight-0.1"
WORK_II_PRIVATE_CELL_VERSION = "chemworld-work-ii-private-confirmation-cell-0.1"
PRIVATE_BLOCKING_REQUIREMENTS = (
    "separate private currency ceiling is not approved",
    "private one-shot execution command lacks user signoff",
    "private execution runner and transfer analysis lack final release receipt",
)


class WorkIIPrivateConfirmationError(ValueError):
    """Raised when a private seal or preflight crosses its frozen boundary."""


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkIIPrivateConfirmationError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    raise WorkIIPrivateConfirmationError(f"{label} must be a list")


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def validate_private_seal(
    design: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> list[str]:
    """Validate the external seal without returning or logging private identities."""

    errors: list[str] = []
    if set(seal) != {"schema_version", "design_id", "seal_nonce", "task_world_seeds"}:
        errors.append("private seal fields differ from the frozen schema")
    if seal.get("schema_version") != WORK_II_PRIVATE_SEAL_VERSION:
        errors.append("unexpected private seal schema")
    if seal.get("design_id") != design.get("design_id"):
        errors.append("private seal design identity mismatch")
    nonce = seal.get("seal_nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        errors.append("private seal nonce is invalid")

    cohort = design.get("world_cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    private = cohort.get("private_confirmation")
    private = private if isinstance(private, Mapping) else {}
    public = cohort.get("public_formal")
    public = public if isinstance(public, Mapping) else {}
    development = cohort.get("development_and_qualification")
    development = development if isinstance(development, Mapping) else {}
    tasks = design.get("tasks")
    tasks = tasks if isinstance(tasks, list) else []
    expected_tasks = [str(row.get("task_id")) for row in tasks if isinstance(row, Mapping)]
    raw_seeds = seal.get("task_world_seeds")
    raw_seeds = raw_seeds if isinstance(raw_seeds, Mapping) else {}
    if set(raw_seeds) != set(expected_tasks):
        errors.append("private seal task roster mismatch")

    namespace_start = private.get("namespace_start")
    namespace_size = private.get("namespace_size")
    worlds_per_task = private.get("worlds_per_task")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (namespace_start, namespace_size, worlds_per_task)
    ):
        errors.append("private seal namespace contract is invalid")
        namespace_start, namespace_size, worlds_per_task = 0, 0, 0
    assert isinstance(namespace_start, int)
    assert isinstance(namespace_size, int)
    assert isinstance(worlds_per_task, int)
    namespace_end = namespace_start + namespace_size
    private_flat: list[int] = []
    for task_id in expected_tasks:
        values = raw_seeds.get(task_id)
        if not isinstance(values, list) or len(values) != worlds_per_task:
            errors.append("private seal world denominator mismatch")
            continue
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append("private seal contains a non-integer world identity")
                continue
            private_flat.append(value)
            if not namespace_start <= value < namespace_end:
                errors.append("private seal identity is outside its namespace")

    public_seeds = {
        int(seed)
        for values in _object(public.get("task_world_seeds"), "public task worlds").values()
        for seed in _sequence(values, "public task worlds")
    }
    development_seeds = {
        int(seed)
        for seed in _sequence(development.get("world_seeds"), "development worlds")
    }
    if (
        len(private_flat) != len(set(private_flat))
        or set(private_flat) & public_seeds
        or set(private_flat) & development_seeds
    ):
        errors.append("private seal identities are not unique and split-disjoint")
    commitment = private.get("sealed_identity_commitment_sha256")
    if canonical_json_sha256(seal) != commitment:
        errors.append("private seal commitment mismatch")
    return errors


def _private_cell_key(cell: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in cell.items() if key != "cell_key_sha256"}
    )


def build_private_confirmation_preflight(
    *,
    public_manifest: Mapping[str, Any],
    public_analysis: Mapping[str, Any],
    design: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the sealed private schedule after public analysis is frozen."""

    errors = validate_formal_preflight(public_manifest)
    if errors:
        raise WorkIIPrivateConfirmationError(
            "public formal manifest is invalid: " + "; ".join(errors)
        )
    if (
        public_manifest.get("formal_execution_allowed") is not True
        or public_manifest.get("status") != "passed_execution_authorized"
    ):
        raise WorkIIPrivateConfirmationError(
            "private unseal requires an authorized public manifest"
        )
    analysis_errors = validate_confirmatory_analysis(public_analysis)
    if analysis_errors:
        raise WorkIIPrivateConfirmationError(
            "public confirmatory analysis is invalid: " + "; ".join(analysis_errors)
        )
    if public_analysis.get("formal_preflight_sha256") != public_manifest.get(
        "preflight_sha256"
    ):
        raise WorkIIPrivateConfirmationError(
            "public confirmatory analysis does not bind the authorized manifest"
        )
    design_binding = _object(public_manifest.get("design_binding"), "design binding")
    if canonical_json_sha256(design) != design_binding.get("sha256"):
        raise WorkIIPrivateConfirmationError("private preflight design binding drifted")
    seal_errors = validate_private_seal(design, seal)
    if seal_errors:
        raise WorkIIPrivateConfirmationError("invalid private seal: " + "; ".join(seal_errors))
    private_contract = public_manifest.get("private_confirmation_contract")
    if private_contract != EXPECTED_PRIVATE_CONFIRMATION_CONTRACT:
        raise WorkIIPrivateConfirmationError("private confirmation contract drifted")

    templates: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_cell in _sequence(public_manifest.get("cells"), "public cells"):
        public_cell = _object(raw_cell, "public cell")
        templates.setdefault(
            (str(public_cell["task_id"]), str(public_cell["prior_arm"])), public_cell
        )
    raw_seeds = _object(seal.get("task_world_seeds"), "private task worlds")
    task_rows = [_object(row, "design task") for row in _sequence(design.get("tasks"), "tasks")]
    cells: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_rows, start=1):
        task_id = str(task["task_id"])
        for world_index, world_seed in enumerate(
            _sequence(raw_seeds.get(task_id), f"{task_id} private worlds"),
            start=1,
        ):
            cluster_id = f"work-ii-private-{task_index:02d}-{world_index:02d}"
            for arm_index, arm in enumerate(FORMAL_ARMS, start=1):
                template = templates[(task_id, arm)]
                cell: dict[str, Any] = {
                    "schema_version": WORK_II_PRIVATE_CELL_VERSION,
                    "schedule_index": len(cells) + 1,
                    "cell_id": f"{cluster_id}-arm-{arm_index:02d}",
                    "world_cluster_id": cluster_id,
                    "task_id": task_id,
                    "world_index": world_index,
                    "world_seed": int(world_seed),
                    "world_split": "private_confirmation",
                    "prior_arm": arm,
                    "campaign_config_path": template["campaign_config_path"],
                    "campaign_config_sha256": template["campaign_config_sha256"],
                    "checkpoint_contract_sha256": template["checkpoint_contract_sha256"],
                    "participant_execution_contract_sha256": template[
                        "participant_execution_contract_sha256"
                    ],
                    "law_summary_evaluation_contract_sha256": template[
                        "law_summary_evaluation_contract_sha256"
                    ],
                    "private_confirmation_contract_sha256": template[
                        "private_confirmation_contract_sha256"
                    ],
                    "complete_experiment_count": template["complete_experiment_count"],
                    "belief_checkpoint_count": template["belief_checkpoint_count"],
                    "held_out_query_count_per_snapshot": template[
                        "held_out_query_count_per_snapshot"
                    ],
                    "held_out_query_metric_count_per_snapshot": template[
                        "held_out_query_metric_count_per_snapshot"
                    ],
                    "provider_session_limit": template["provider_session_limit"],
                    "provider_attempt_limit": template["provider_attempt_limit"],
                    "provider_repeat": template["provider_repeat"],
                    "participant_final_recommendation_count": template[
                        "participant_final_recommendation_count"
                    ],
                    "blind_validation_target_count": template[
                        "blind_validation_target_count"
                    ],
                    "blind_replicates_per_target": template["blind_replicates_per_target"],
                    "blind_validation_execution_count": template[
                        "blind_validation_execution_count"
                    ],
                    "terminal_states": deepcopy(template["terminal_states"]),
                    "public_template_cell_key_sha256": template["cell_key_sha256"],
                }
                cell["cell_key_sha256"] = _private_cell_key(cell)
                cells.append(cell)

    report: dict[str, Any] = {
        "schema_version": WORK_II_PRIVATE_PREFLIGHT_VERSION,
        "status": "passed_private_execution_blocked",
        "formal_result": False,
        "private_confirmation_result": False,
        "private_execution_allowed": False,
        "blocking_requirements": list(PRIVATE_BLOCKING_REQUIREMENTS),
        "provider_calls_executed": 0,
        "public_formal_manifest_sha256": public_manifest["preflight_sha256"],
        "public_confirmatory_analysis_sha256": public_analysis["report_sha256"],
        "design_sha256": design_binding["sha256"],
        "private_seal_commitment_sha256": canonical_json_sha256(seal),
        "private_identity_schedule_sha256": canonical_json_sha256(raw_seeds),
        "private_confirmation_contract": deepcopy(EXPECTED_PRIVATE_CONFIRMATION_CONTRACT),
        "private_confirmation_contract_sha256": canonical_json_sha256(
            EXPECTED_PRIVATE_CONFIRMATION_CONTRACT
        ),
        "expected_counts": {
            "tasks": 5,
            "independent_task_world_clusters": 25,
            "participant_cells": 75,
            "complete_experiments": sum(
                int(cell["complete_experiment_count"]) for cell in cells
            ),
            "belief_checkpoints": sum(
                int(cell["belief_checkpoint_count"]) for cell in cells
            ),
            "provider_sessions": 75,
            "provider_attempts_initial_planned": 75,
            "provider_attempts_hard_cap": 150,
            "evaluator_truth_executions": 100,
            "blind_validation_executions": 450,
        },
        "identity_boundary": {
            "seal_nonce_present": False,
            "identities_present": True,
            "output_must_remain_under_ignored_runs_private": True,
            "participant_receives_public_analysis": False,
        },
        "cells": cells,
    }
    report["preflight_sha256"] = _self_hash(report, "preflight_sha256")
    validation_errors = validate_private_confirmation_preflight(
        report,
        public_manifest=public_manifest,
        public_analysis=public_analysis,
        design=design,
        seal=seal,
    )
    if validation_errors:
        raise WorkIIPrivateConfirmationError(
            "built private preflight is invalid: " + "; ".join(validation_errors)
        )
    return report


def validate_private_confirmation_preflight(
    report: Mapping[str, Any],
    *,
    public_manifest: Mapping[str, Any],
    public_analysis: Mapping[str, Any],
    design: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> list[str]:
    """Validate a private schedule against all public and sealed bindings."""

    errors: list[str] = []
    if report.get("schema_version") != WORK_II_PRIVATE_PREFLIGHT_VERSION:
        errors.append("unexpected private confirmation preflight schema")
    if report.get("preflight_sha256") != _self_hash(report, "preflight_sha256"):
        errors.append("private confirmation preflight self-hash mismatch")
    if (
        report.get("status") != "passed_private_execution_blocked"
        or report.get("private_execution_allowed") is not False
        or report.get("blocking_requirements") != list(PRIVATE_BLOCKING_REQUIREMENTS)
    ):
        errors.append("private confirmation preflight crossed its authorization boundary")
    if (
        report.get("formal_result") is not False
        or report.get("private_confirmation_result") is not False
        or report.get("provider_calls_executed") != 0
    ):
        errors.append("private confirmation preflight contains outcomes or provider calls")
    if report.get("public_formal_manifest_sha256") != public_manifest.get("preflight_sha256"):
        errors.append("private preflight public manifest binding mismatch")
    if report.get("public_confirmatory_analysis_sha256") != public_analysis.get("report_sha256"):
        errors.append("private preflight public analysis binding mismatch")
    if report.get("design_sha256") != canonical_json_sha256(design):
        errors.append("private preflight design binding mismatch")
    if report.get("private_seal_commitment_sha256") != canonical_json_sha256(seal):
        errors.append("private preflight seal commitment mismatch")
    if validate_private_seal(design, seal):
        errors.append("private preflight uses an invalid private seal")
    if report.get("private_confirmation_contract") != EXPECTED_PRIVATE_CONFIRMATION_CONTRACT:
        errors.append("private preflight contract mismatch")
    cells = report.get("cells")
    cells = cells if isinstance(cells, list) else []
    if len(cells) != 75:
        errors.append("private preflight cell denominator mismatch")
    ids = [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
    keys = [cell.get("cell_key_sha256") for cell in cells if isinstance(cell, Mapping)]
    clusters = {
        cell.get("world_cluster_id") for cell in cells if isinstance(cell, Mapping)
    }
    if len(set(ids)) != 75 or len(set(keys)) != 75 or len(clusters) != 25:
        errors.append("private preflight identities are not unique")
    expected_counts = report.get("expected_counts")
    expected_counts = expected_counts if isinstance(expected_counts, Mapping) else {}
    observed_counts = {
        "tasks": len(
            {
                str(cell.get("task_id"))
                for cell in cells
                if isinstance(cell, Mapping)
            }
        ),
        "independent_task_world_clusters": len(clusters),
        "participant_cells": len(cells),
        "complete_experiments": sum(
            int(cell.get("complete_experiment_count", 0))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "belief_checkpoints": sum(
            int(cell.get("belief_checkpoint_count", 0))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "provider_sessions": sum(
            int(cell.get("provider_session_limit", 0))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "provider_attempts_initial_planned": len(cells),
        "provider_attempts_hard_cap": sum(
            int(cell.get("provider_attempt_limit", 0))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "evaluator_truth_executions": len(clusters) * 4,
        "blind_validation_executions": sum(
            int(cell.get("blind_validation_execution_count", 0))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
    }
    if dict(expected_counts) != observed_counts:
        errors.append("private preflight expected counts differ from its exact cell schedule")
    seal_seeds = _object(seal.get("task_world_seeds"), "private task worlds")
    expected_triplets = {
        (str(task_id), int(seed), arm)
        for task_id, values in seal_seeds.items()
        for seed in _sequence(values, "private task worlds")
        for arm in FORMAL_ARMS
    }
    observed_triplets = {
        (str(cell.get("task_id")), int(cell.get("world_seed", -1)), str(cell.get("prior_arm")))
        for cell in cells
        if isinstance(cell, Mapping)
    }
    if observed_triplets != expected_triplets:
        errors.append("private preflight cells differ from the sealed identity schedule")
    for cell in cells:
        if not isinstance(cell, Mapping):
            errors.append("private preflight contains a malformed cell")
            continue
        if cell.get("world_split") != "private_confirmation":
            errors.append("private preflight contains a non-private cell")
        if cell.get("cell_key_sha256") != _private_cell_key(cell):
            errors.append("private confirmation cell self-hash mismatch")
    boundary = report.get("identity_boundary")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    if (
        boundary.get("seal_nonce_present") is not False
        or boundary.get("identities_present") is not True
        or boundary.get("output_must_remain_under_ignored_runs_private") is not True
        or boundary.get("participant_receives_public_analysis") is not False
        or "seal_nonce" in report
    ):
        errors.append("private preflight identity boundary is invalid")
    return errors


def write_private_preflight_once(root: Path, output: Path, report: Mapping[str, Any]) -> None:
    """Write a private preflight once, only beneath the ignored private run root."""

    private_root = (root.resolve() / "runs" / "private").resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(private_root):
        raise WorkIIPrivateConfirmationError(
            "private preflight output must stay under runs/private"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "PRIVATE_BLOCKING_REQUIREMENTS",
    "WORK_II_PRIVATE_CELL_VERSION",
    "WORK_II_PRIVATE_PREFLIGHT_VERSION",
    "WORK_II_PRIVATE_SEAL_VERSION",
    "WorkIIPrivateConfirmationError",
    "build_private_confirmation_preflight",
    "validate_private_confirmation_preflight",
    "validate_private_seal",
    "write_private_preflight_once",
]
