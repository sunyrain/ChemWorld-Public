"""Manifest-bound, resumable public evaluator orchestration for Work II.

This module never launches a participant provider.  It consumes an authorized
formal execution manifest plus the immutable terminal participant-cell store,
then publishes evaluator truth packs per task/world cluster and blind replay
packs per completed participant cell.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    git_source_commit,
    git_worktree_dirty,
    repository_tree_sha256,
    write_json_atomic,
)
from chemworld.eval.work_ii_blind import (
    blind_execution_directory_name,
    build_blind_evaluation_plan,
    effective_blind_evaluator_contract,
    execute_blind_evaluation_plan,
    validate_blind_evaluation_plan,
    validate_blind_evaluation_report,
)
from chemworld.eval.work_ii_formal import (
    FORMAL_ARMS,
    WorkIIFormalCellStore,
    validate_formal_bindings,
    validate_formal_preflight,
)
from chemworld.eval.work_ii_truth import (
    build_evaluator_truth_plan,
    execute_evaluator_truth_plan,
    validate_evaluator_truth_plan,
    validate_evaluator_truth_report,
)

WORK_II_FORMAL_EVALUATOR_ORCHESTRATION_VERSION = (
    "chemworld-work-ii-formal-evaluator-orchestration-0.1"
)
WORK_II_FORMAL_EVALUATOR_UNIT_VERSION = (
    "chemworld-work-ii-formal-evaluator-unit-receipt-0.1"
)
WORK_II_FORMAL_EVALUATOR_SUMMARY_VERSION = (
    "chemworld-work-ii-formal-evaluator-summary-0.1"
)

FORMAL_EVALUATOR_SOURCE_ROOTS = (
    "pyproject.toml",
    "uv.lock",
    "scripts/run_work_ii_formal_evaluators.py",
    "src/chemworld/eval/provenance.py",
    "src/chemworld/eval/work_ii_blind.py",
    "src/chemworld/eval/work_ii_formal.py",
    "src/chemworld/eval/work_ii_formal_evaluators.py",
    "src/chemworld/eval/work_ii_truth.py",
)

C2_FORMAL_ROSTER_CONTRACT = {
    "A_E": {"task_count": 5, "worlds_per_task": 5, "arms_per_world": 3},
    "A_P": {"task_count": 2, "worlds_per_task": 5, "arms_per_world": 3},
    "A_S": {"task_count": 2, "worlds_per_task": 5, "arms_per_world": 3},
}

_SAFE_UNIT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish an immutable JSON object without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"immutable artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _relative_exclusion(root: Path, output_root: Path) -> tuple[str, ...]:
    try:
        return (output_root.resolve().relative_to(root.resolve()).as_posix(),)
    except ValueError:
        return ()


def build_formal_evaluator_source_binding(
    root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Bind the evaluator implementation and require an otherwise clean tree."""

    root = root.resolve()
    exclusions = _relative_exclusion(root, output_root)
    if git_worktree_dirty(root, excluded_prefixes=exclusions):
        raise ValueError("formal evaluator execution requires a clean immutable worktree")
    binding: dict[str, Any] = {
        "source_commit": git_source_commit(root),
        "material_roots": list(FORMAL_EVALUATOR_SOURCE_ROOTS),
        "material_tree_sha256": repository_tree_sha256(
            root,
            relative_roots=FORMAL_EVALUATOR_SOURCE_ROOTS,
        ),
    }
    binding["source_binding_sha256"] = canonical_json_sha256(binding)
    return binding


def _safe_unit_id(value: Any, *, label: str) -> str:
    rendered = str(value)
    if not _SAFE_UNIT_ID.fullmatch(rendered) or rendered in {".", ".."}:
        raise ValueError(f"unsafe {label}: {rendered!r}")
    return rendered


def _resolve_execution_binding(
    execution_root: Path,
    binding: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding is missing")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError(f"{label} binding is incomplete")
    candidate = (execution_root / relative).resolve()
    try:
        candidate.relative_to(execution_root)
    except ValueError as error:
        raise ValueError(f"{label} binding escapes the execution root") from error
    if not candidate.is_file() or file_sha256(candidate) != digest:
        raise ValueError(f"{label} artifact is missing or changed")
    return candidate, dict(binding)


def _load_campaign_config(
    root: Path,
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = cell.get("campaign_config_path")
    expected = cell.get("campaign_config_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"{cell.get('cell_id')}: campaign config binding is incomplete")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{cell.get('cell_id')}: campaign config escapes repository") from error
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{cell.get('cell_id')}: campaign config binding drifted")
    config = _load_object(path)
    if config.get("task_id") != cell.get("task_id"):
        raise ValueError(f"{cell.get('cell_id')}: campaign task binding drifted")
    return config, {
        "path": relative,
        "file_sha256": expected,
        "canonical_json_sha256": canonical_json_sha256(config),
    }


def _cluster_schedule(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for raw_cell in cells:
        cell = dict(raw_cell)
        locus = cell.get("c2_locus")
        if locus not in C2_FORMAL_ROSTER_CONTRACT:
            raise ValueError(
                f"{cell.get('cell_id')}: c2_locus must be exactly A_E, A_P or A_S"
            )
        cluster_id = _safe_unit_id(cell.get("world_cluster_id"), label="cluster ID")
        if cluster_id not in grouped:
            order.append(cluster_id)
        grouped[cluster_id].append(cell)
    clusters: list[dict[str, Any]] = []
    for cluster_id in order:
        cluster_cells = grouped[cluster_id]
        first = cluster_cells[0]
        if len(cluster_cells) != len(FORMAL_ARMS) or {
            str(cell.get("prior_arm")) for cell in cluster_cells
        } != set(FORMAL_ARMS):
            raise ValueError(f"{cluster_id}: formal arm triplet is incomplete")
        stable_fields = (
            "c2_locus",
            "task_id",
            "world_seed",
            "campaign_config_path",
            "campaign_config_sha256",
        )
        if any(
            cell.get(field) != first.get(field)
            for cell in cluster_cells[1:]
            for field in stable_fields
        ):
            raise ValueError(f"{cluster_id}: shared truth identity drifts across arms")
        clusters.append(
            {
                "c2_locus": first["c2_locus"],
                "world_cluster_id": cluster_id,
                "task_id": first["task_id"],
                "world_seed": first["world_seed"],
                "cells": cluster_cells,
                "cell_key_sha256": [cell["cell_key_sha256"] for cell in cluster_cells],
            }
        )
    observed_loci = {str(cluster["c2_locus"]) for cluster in clusters}
    if observed_loci != set(C2_FORMAL_ROSTER_CONTRACT):
        raise ValueError("formal C2 schedule must contain exactly A_E, A_P and A_S")
    for locus, contract in C2_FORMAL_ROSTER_CONTRACT.items():
        locus_clusters = [cluster for cluster in clusters if cluster["c2_locus"] == locus]
        task_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cluster in locus_clusters:
            task_clusters[str(cluster["task_id"])].append(cluster)
        expected_tasks = int(contract["task_count"])
        expected_worlds = int(contract["worlds_per_task"])
        expected_arms = int(contract["arms_per_world"])
        expected_clusters = expected_tasks * expected_worlds
        expected_cells = expected_clusters * expected_arms
        if (
            len(task_clusters) != expected_tasks
            or len(locus_clusters) != expected_clusters
            or sum(len(cluster["cells"]) for cluster in locus_clusters) != expected_cells
        ):
            raise ValueError(
                f"{locus}: formal roster must contain {expected_tasks} tasks, "
                f"{expected_clusters} clusters and {expected_cells} cells"
            )
        for task_id, task_worlds in task_clusters.items():
            seeds = {int(cluster["world_seed"]) for cluster in task_worlds}
            if len(task_worlds) != expected_worlds or len(seeds) != expected_worlds:
                raise ValueError(
                    f"{locus}/{task_id}: roster must contain {expected_worlds} "
                    "distinct public worlds"
                )
    return clusters


def _blind_cell_denominators(
    cell: Mapping[str, Any],
    global_contract: Mapping[str, Any],
) -> tuple[int, int, int]:
    effective = effective_blind_evaluator_contract(cell, global_contract)
    return (
        len(effective["blind_targets_per_cell"]),
        int(effective["blind_replicates_per_target"]),
        int(effective["blind_validation_execution_count"]),
    )


def _roster_summary(
    cells: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for locus, contract in C2_FORMAL_ROSTER_CONTRACT.items():
        locus_cells = [cell for cell in cells if cell["c2_locus"] == locus]
        locus_clusters = [cluster for cluster in clusters if cluster["c2_locus"] == locus]
        summary[locus] = {
            **contract,
            "observed_task_count": len({str(cell["task_id"]) for cell in locus_cells}),
            "observed_cluster_count": len(locus_clusters),
            "observed_cell_count": len(locus_cells),
            "schedule_sha256": canonical_json_sha256(locus_cells),
        }
    return summary


def _validate_completed_session(
    execution_root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = terminal_receipt["result"]
    if terminal_receipt.get("state") != "completed" or result.get("completed") is not True:
        raise ValueError(f"{cell['cell_id']}: participant completion binding is inconsistent")
    summary_path, summary_binding = _resolve_execution_binding(
        execution_root,
        result.get("summary"),
        label=f"{cell['cell_id']} summary",
    )
    plan_path, plan_binding = _resolve_execution_binding(
        execution_root,
        result.get("blind_evaluation_plan"),
        label=f"{cell['cell_id']} blind plan",
    )
    summary = _load_object(summary_path)
    plan = _load_object(plan_path)
    summary_plan_binding = summary.get("blind_evaluation_plan")
    if (
        summary.get("formal_result") is not True
        or summary.get("formal_preflight_sha256") != manifest.get("preflight_sha256")
        or summary.get("formal_cell") != dict(cell)
        or summary.get("completed") is not True
        or not isinstance(summary.get("qualification"), Mapping)
        or summary["qualification"].get("passed") is not True
        or not isinstance(summary.get("exact_replay"), Mapping)
        or summary["exact_replay"].get("verified") is not True
        or not isinstance(summary_plan_binding, Mapping)
        or summary_plan_binding.get("sha256") != plan_binding["sha256"]
        or summary_plan_binding.get("plan_sha256") != plan.get("plan_sha256")
    ):
        raise ValueError(f"{cell['cell_id']}: completed session is not evaluator-admissible")
    plan_errors = validate_blind_evaluation_plan(plan)
    if plan_errors:
        raise ValueError(
            f"{cell['cell_id']}: invalid committed blind plan: " + "; ".join(plan_errors)
        )
    effective_contract = effective_blind_evaluator_contract(
        cell,
        manifest["blind_evaluator_contract"],
    )
    expected_plan = build_blind_evaluation_plan(cell, summary, effective_contract)
    if plan != expected_plan:
        raise ValueError(f"{cell['cell_id']}: committed blind plan is not deterministic")
    _, _, scheduled_executions = _blind_cell_denominators(
        cell,
        manifest["blind_evaluator_contract"],
    )
    if plan.get("blind_execution_count") != scheduled_executions:
        raise ValueError(f"{cell['cell_id']}: blind plan denominator differs from schedule")
    return summary, plan, {
        "summary": summary_binding,
        "blind_evaluation_plan": plan_binding,
        "terminal_receipt_sha256": terminal_receipt["receipt_sha256"],
    }


def _validate_trajectory_binding(unit_root: Path, binding: Any, *, label: str) -> None:
    if binding is None:
        return
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label}: malformed trajectory binding")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError(f"{label}: incomplete trajectory binding")
    path = (unit_root / relative).resolve()
    try:
        path.relative_to(unit_root)
    except ValueError as error:
        raise ValueError(f"{label}: trajectory escapes evaluator unit") from error
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"{label}: trajectory is missing or changed")


def _truth_pack(
    unit_root: Path,
    expected_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_object(unit_root / "plan.json")
    report = _load_object(unit_root / "report.json")
    if plan != dict(expected_plan):
        raise ValueError(f"{unit_root.name}: evaluator truth plan binding drifted")
    errors = validate_evaluator_truth_report(report, plan)
    if errors:
        raise ValueError(f"{unit_root.name}: invalid evaluator truth report: " + "; ".join(errors))
    plan_queries = plan.get("queries")
    receipts = report.get("receipts")
    if not isinstance(plan_queries, list) or not isinstance(receipts, list):
        raise ValueError(f"{unit_root.name}: evaluator truth denominator is malformed")
    failed_count = 0
    for query, receipt in zip(plan_queries, receipts, strict=True):
        if not isinstance(query, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError(f"{unit_root.name}: malformed evaluator truth receipt")
        if (
            receipt.get("execution_id") != query.get("execution_id")
            or receipt.get("query_id") != query.get("query_id")
            or receipt.get("metric_ids") != query.get("metric_ids")
            or receipt.get("action_plan_sha256") != query.get("action_plan_sha256")
            or receipt.get("status") not in {"completed", "failed"}
        ):
            raise ValueError(f"{unit_root.name}: evaluator truth receipt binding drifted")
        failed_count += receipt.get("status") == "failed"
        _validate_trajectory_binding(
            unit_root,
            receipt.get("trajectory"),
            label=str(receipt.get("execution_id")),
        )
    if report.get("failed_truth_query_count") != failed_count:
        raise ValueError(f"{unit_root.name}: evaluator truth failure count drifted")
    return plan, report


def _blind_pack(
    unit_root: Path,
    expected_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = _load_object(unit_root / "plan.json")
    report = _load_object(unit_root / "report.json")
    if plan != dict(expected_plan):
        raise ValueError(f"{unit_root.name}: blind evaluator plan binding drifted")
    receipts: list[dict[str, Any]] = []
    expected_receipt_paths: set[Path] = set()
    for execution in plan["executions"]:
        receipt_path = (
            unit_root
            / "executions"
            / blind_execution_directory_name(execution)
            / "receipt.json"
        )
        expected_receipt_paths.add(receipt_path.resolve())
        receipt = _load_object(receipt_path)
        if (
            receipt.get("execution_id") != execution.get("execution_id")
            or receipt.get("target") != execution.get("target")
            or receipt.get("replicate_index") != execution.get("replicate_index")
            or receipt.get("paired_noise_id_sha256")
            != execution.get("paired_noise_id_sha256")
            or receipt.get("action_plan_sha256") != execution.get("action_plan_sha256")
            or receipt.get("status") not in {"completed", "failed"}
        ):
            raise ValueError(f"{unit_root.name}: blind evaluator receipt binding drifted")
        _validate_trajectory_binding(
            unit_root,
            receipt.get("trajectory"),
            label=str(receipt.get("execution_id")),
        )
        receipts.append(receipt)
    observed_paths = {
        path.resolve() for path in (unit_root / "executions").glob("*/receipt.json")
    }
    if observed_paths != expected_receipt_paths:
        raise ValueError(f"{unit_root.name}: blind evaluator receipt coverage drifted")
    receipt_views: list[Mapping[str, Any]] = []
    receipt_views.extend(receipts)
    errors = validate_blind_evaluation_report(report, plan, receipt_views)
    if errors:
        raise ValueError(f"{unit_root.name}: invalid blind evaluator report: " + "; ".join(errors))
    return plan, report, receipts


def _unit_receipt(
    unit_root: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_object(unit_root / "orchestration_receipt.json")
    if (
        receipt.get("schema_version") != WORK_II_FORMAL_EVALUATOR_UNIT_VERSION
        or receipt.get("receipt_sha256") != _self_hash(receipt, "receipt_sha256")
        or any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("plan_file_sha256") != file_sha256(unit_root / "plan.json")
        or receipt.get("report_file_sha256") != file_sha256(unit_root / "report.json")
    ):
        raise ValueError(f"{unit_root.name}: evaluator unit receipt binding drifted")
    return receipt


def _audit_unit_directories(root: Path, expected: set[str]) -> None:
    if not root.exists():
        return
    observed = {path.name for path in root.iterdir() if path.is_dir()}
    pending = sorted(name for name in observed if name.startswith(".pending-"))
    unexpected = sorted(observed - expected)
    if pending:
        raise ValueError(
            "incomplete evaluator staging directories require audit: "
            + ", ".join(pending)
        )
    if unexpected:
        raise ValueError("unexpected evaluator unit directories: " + ", ".join(unexpected))


def _publish_truth_unit(
    output_root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    unit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    unit_id = _safe_unit_id(plan["world_cluster_id"], label="truth unit ID")
    parent = output_root / "truth"
    final = parent / unit_id
    pending = parent / f".pending-{unit_id}"
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists() or pending.exists():
        raise FileExistsError(f"refusing to overwrite evaluator truth unit: {unit_id}")
    execute_evaluator_truth_plan(plan, config, pending)
    _, report = _truth_pack(pending, plan)
    receipt = {
        **dict(unit_receipt),
        "schema_version": WORK_II_FORMAL_EVALUATOR_UNIT_VERSION,
        "unit_type": "evaluator_truth_cluster",
        "unit_id": unit_id,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": file_sha256(pending / "plan.json"),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": file_sha256(pending / "report.json"),
        "status": report["status"],
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    _write_json_once(pending / "orchestration_receipt.json", receipt)
    pending.rename(final)
    return report


def _publish_blind_unit(
    output_root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    unit_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unit_id = _safe_unit_id(plan["cell_key_sha256"], label="blind unit ID")
    parent = output_root / "blind"
    final = parent / unit_id
    pending = parent / f".pending-{unit_id}"
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists() or pending.exists():
        raise FileExistsError(f"refusing to overwrite blind evaluator unit: {unit_id}")
    execute_blind_evaluation_plan(plan, config, pending)
    _, report, receipts = _blind_pack(pending, plan)
    receipt = {
        **dict(unit_receipt),
        "schema_version": WORK_II_FORMAL_EVALUATOR_UNIT_VERSION,
        "unit_type": "blind_evaluator_session",
        "unit_id": unit_id,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": file_sha256(pending / "plan.json"),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": file_sha256(pending / "report.json"),
        "status": report["status"],
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    _write_json_once(pending / "orchestration_receipt.json", receipt)
    pending.rename(final)
    return report, receipts


def execute_formal_evaluators(
    root: Path,
    manifest: Mapping[str, Any],
    execution_root: Path,
    output_root: Path,
    *,
    resume: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute or resume all public truth and eligible blind evaluator units."""

    root = root.resolve()
    execution_root = execution_root.resolve()
    output_root = output_root.resolve()
    manifest = dict(manifest)
    errors = validate_formal_preflight(manifest)
    if errors:
        raise ValueError("invalid formal execution manifest: " + "; ".join(errors))
    if manifest.get("formal_execution_allowed") is not True:
        raise ValueError("formal evaluator execution requires an authorized manifest")
    binding_errors = validate_formal_bindings(root, manifest)
    if binding_errors:
        raise ValueError("formal manifest binding validation failed: " + "; ".join(binding_errors))
    execution_manifest_path = execution_root / "execution_manifest.json"
    execution_manifest = _load_object(execution_manifest_path)
    if execution_manifest != manifest:
        raise ValueError("participant execution manifest differs from evaluator manifest")

    cells = manifest.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, Mapping) for cell in cells):
        raise ValueError("formal evaluator manifest has a malformed schedule")
    typed_cells = [dict(cell) for cell in cells]
    clusters = _cluster_schedule(typed_cells)
    roster = _roster_summary(typed_cells, clusters)
    global_blind_contract = manifest.get("blind_evaluator_contract")
    if not isinstance(global_blind_contract, Mapping):
        raise ValueError("formal evaluator manifest lacks the global blind contract")
    blind_cell_denominators = {
        str(cell["cell_key_sha256"]): _blind_cell_denominators(
            cell,
            global_blind_contract,
        )
        for cell in typed_cells
    }

    store = WorkIIFormalCellStore(execution_root / "store", manifest)
    audit = store.audit()
    if audit.get("complete") is not True:
        raise ValueError("participant formal store must be terminal-complete before evaluation")
    terminal_receipts = {
        str(cell["cell_key_sha256"]): store.load_terminal(str(cell["cell_key_sha256"]))
        for cell in typed_cells
    }

    config_by_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    truth_plans: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for cluster in clusters:
        first = cluster["cells"][0]
        config_key = str(first["campaign_config_path"])
        if config_key not in config_by_path:
            config_by_path[config_key] = _load_campaign_config(root, first)
        config, config_binding = config_by_path[config_key]
        plan = build_evaluator_truth_plan(
            cluster,
            config,
            formal_result=True,
            formal_preflight_sha256=str(manifest["preflight_sha256"]),
        )
        plan_errors = validate_evaluator_truth_plan(plan)
        if plan_errors:
            raise ValueError(
                f"{cluster['world_cluster_id']}: invalid truth plan: " + "; ".join(plan_errors)
            )
        truth_plans[str(cluster["world_cluster_id"])] = (plan, config, config_binding)

    blind_plans: dict[
        str,
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    for cell in typed_cells:
        key = str(cell["cell_key_sha256"])
        terminal = terminal_receipts[key]
        if terminal["state"] != "completed":
            if terminal["result"].get("completed") is True:
                raise ValueError(f"{cell['cell_id']}: non-completed receipt claims completion")
            continue
        _, plan, session_binding = _validate_completed_session(
            execution_root,
            manifest,
            cell,
            terminal,
        )
        config_key = str(cell["campaign_config_path"])
        if config_key not in config_by_path:
            config_by_path[config_key] = _load_campaign_config(root, cell)
        config, config_binding = config_by_path[config_key]
        blind_plans[key] = (plan, config, config_binding, session_binding)

    truth_query_count = sum(int(plan["truth_query_count"]) for plan, _, _ in truth_plans.values())
    truth_metric_count = sum(
        int(plan["truth_query_metric_count"]) for plan, _, _ in truth_plans.values()
    )
    blind_target_count = sum(values[0] for values in blind_cell_denominators.values())
    blind_execution_count = sum(values[2] for values in blind_cell_denominators.values())
    launchable_blind_execution_count = sum(
        int(plan["blind_execution_count"]) for plan, _, _, _ in blind_plans.values()
    )

    source_binding = build_formal_evaluator_source_binding(root, output_root=output_root)
    orchestration_manifest: dict[str, Any] = {
        "schema_version": WORK_II_FORMAL_EVALUATOR_ORCHESTRATION_VERSION,
        "formal_result": True,
        "participant_provider_calls_allowed": False,
        "formal_preflight_sha256": manifest["preflight_sha256"],
        "execution_manifest": {
            "canonical_json_sha256": canonical_json_sha256(manifest),
            "file_sha256": file_sha256(execution_manifest_path),
        },
        "schedule": {
            "cell_count": len(typed_cells),
            "cluster_count": len(clusters),
            "cells_sha256": canonical_json_sha256(typed_cells),
            "c2_roster": roster,
            "terminal_receipts_sha256": canonical_json_sha256(
                [
                    terminal_receipts[str(cell["cell_key_sha256"])]["receipt_sha256"]
                    for cell in typed_cells
                ]
            ),
        },
        "participant_store_audit_sha256": audit["audit_sha256"],
        "source_binding": source_binding,
        "expected_counts": {
            "truth_clusters": len(clusters),
            "truth_executions": truth_query_count,
            "truth_query_metrics": truth_metric_count,
            "participant_sessions": len(typed_cells),
            "eligible_blind_sessions": len(blind_plans),
            "blind_targets": blind_target_count,
            "blind_executions": blind_execution_count,
            "launchable_blind_executions": launchable_blind_execution_count,
        },
    }
    orchestration_manifest["orchestration_manifest_sha256"] = _self_hash(
        orchestration_manifest,
        "orchestration_manifest_sha256",
    )

    if output_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite evaluator output root: {output_root}")
    if not output_root.exists() and resume:
        raise FileNotFoundError("missing-only evaluator resume requires an existing output root")
    if output_root.exists():
        stored = _load_object(output_root / "orchestration_manifest.json")
        if stored != orchestration_manifest:
            raise ValueError("existing evaluator orchestration manifest binding drifted")
    else:
        output_root.mkdir(parents=True)
        _write_json_once(output_root / "orchestration_manifest.json", orchestration_manifest)

    orchestration_hash = orchestration_manifest["orchestration_manifest_sha256"]
    truth_unit_ids = set(truth_plans)
    blind_unit_ids = set(blind_plans)
    _audit_unit_directories(output_root / "truth", truth_unit_ids)
    _audit_unit_directories(output_root / "blind", blind_unit_ids)

    truth_reports: dict[str, dict[str, Any]] = {}
    blind_reports: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    reused_truth = 0
    reused_blind = 0

    def emit(payload: Mapping[str, Any]) -> None:
        if progress is not None:
            progress(payload)

    emit(
        {
            "event": "formal_evaluators_started" if not resume else "formal_evaluators_resumed",
            "truth_cluster_total": len(truth_plans),
            "eligible_blind_session_total": len(blind_plans),
            "participant_session_total": len(typed_cells),
        }
    )

    # Validate every existing unit before publishing any missing unit.
    for cluster_id, (plan, _, config_binding) in truth_plans.items():
        unit_root = output_root / "truth" / cluster_id
        if not unit_root.exists():
            continue
        _, report = _truth_pack(unit_root, plan)
        _unit_receipt(
            unit_root,
            expected={
                "unit_type": "evaluator_truth_cluster",
                "unit_id": cluster_id,
                "orchestration_manifest_sha256": orchestration_hash,
                "config_binding": config_binding,
                "schedule_unit_sha256": canonical_json_sha256(
                    next(
                        cluster
                        for cluster in clusters
                        if cluster["world_cluster_id"] == cluster_id
                    )
                ),
            },
        )
        truth_reports[cluster_id] = report
        reused_truth += 1
        emit(
            {
                "event": "truth_cluster_reused",
                "world_cluster_id": cluster_id,
                "completed": len(truth_reports),
                "total": len(truth_plans),
            }
        )
    for key, (plan, _, config_binding, session_binding) in blind_plans.items():
        unit_root = output_root / "blind" / key
        if not unit_root.exists():
            continue
        _, report, receipts = _blind_pack(unit_root, plan)
        _unit_receipt(
            unit_root,
            expected={
                "unit_type": "blind_evaluator_session",
                "unit_id": key,
                "orchestration_manifest_sha256": orchestration_hash,
                "config_binding": config_binding,
                "session_binding": session_binding,
                "schedule_unit_sha256": canonical_json_sha256(
                    next(cell for cell in typed_cells if cell["cell_key_sha256"] == key)
                ),
            },
        )
        blind_reports[key] = (report, receipts)
        reused_blind += 1
        emit(
            {
                "event": "blind_session_reused",
                "cell_key_sha256": key,
                "completed": len(blind_reports),
                "total": len(blind_plans),
            }
        )

    for cluster in clusters:
        cluster_id = str(cluster["world_cluster_id"])
        if cluster_id in truth_reports:
            continue
        plan, config, config_binding = truth_plans[cluster_id]
        emit(
            {
                "event": "truth_cluster_started",
                "world_cluster_id": cluster_id,
                "completed": len(truth_reports),
                "total": len(truth_plans),
            }
        )
        report = _publish_truth_unit(
            output_root,
            plan,
            config,
            {
                "orchestration_manifest_sha256": orchestration_hash,
                "config_binding": config_binding,
                "schedule_unit_sha256": canonical_json_sha256(cluster),
            },
        )
        truth_reports[cluster_id] = report
        emit(
            {
                "event": "truth_cluster_finished",
                "world_cluster_id": cluster_id,
                "status": report["status"],
                "completed": len(truth_reports),
                "total": len(truth_plans),
            }
        )
    for cell in typed_cells:
        key = str(cell["cell_key_sha256"])
        if key not in blind_plans or key in blind_reports:
            continue
        plan, config, config_binding, session_binding = blind_plans[key]
        emit(
            {
                "event": "blind_session_started",
                "cell_id": cell["cell_id"],
                "completed": len(blind_reports),
                "total": len(blind_plans),
            }
        )
        report, receipts = _publish_blind_unit(
            output_root,
            plan,
            config,
            {
                "orchestration_manifest_sha256": orchestration_hash,
                "config_binding": config_binding,
                "session_binding": session_binding,
                "schedule_unit_sha256": canonical_json_sha256(cell),
            },
        )
        blind_reports[key] = (report, receipts)
        emit(
            {
                "event": "blind_session_finished",
                "cell_id": cell["cell_id"],
                "status": report["status"],
                "completed": len(blind_reports),
                "total": len(blind_plans),
            }
        )

    failures: list[dict[str, Any]] = []
    for cell in typed_cells:
        terminal = terminal_receipts[str(cell["cell_key_sha256"])]
        if terminal["state"] == "completed":
            continue
        failures.append(
            {
                "stage": "participant_session",
                "cell_id": cell["cell_id"],
                "cell_key_sha256": cell["cell_key_sha256"],
                "state": terminal["state"],
                "reason_code": terminal["reason_code"],
                "affected_blind_execution_count": blind_cell_denominators[
                    str(cell["cell_key_sha256"])
                ][2],
            }
        )
    for cluster_id, report in truth_reports.items():
        for receipt in report["receipts"]:
            if receipt["status"] == "failed":
                failures.append(
                    {
                        "stage": "evaluator_truth",
                        "world_cluster_id": cluster_id,
                        "execution_id": receipt["execution_id"],
                        "query_id": receipt["query_id"],
                        "failure_type": receipt["failure_type"],
                        "failure_message": receipt["failure_message"],
                    }
                )
    for key, (_, receipts) in blind_reports.items():
        for receipt in receipts:
            if receipt["status"] == "failed":
                failures.append(
                    {
                        "stage": "blind_evaluator",
                        "cell_key_sha256": key,
                        "execution_id": receipt["execution_id"],
                        "target": receipt["target"],
                        "replicate_index": receipt["replicate_index"],
                        "failure_type": receipt["failure_type"],
                        "failure_message": receipt["failure_message"],
                    }
                )

    truth_completed = sum(
        int(report["completed_truth_query_count"]) for report in truth_reports.values()
    )
    blind_completed = sum(
        int(report["completed_execution_count"]) for report, _ in blind_reports.values()
    )
    blind_launched = sum(
        int(report["scheduled_execution_count"]) for report, _ in blind_reports.values()
    )
    participant_state_counts = {
        state: sum(receipt["state"] == state for receipt in terminal_receipts.values())
        for state in ("completed", "right_censored", "failed")
    }
    summary: dict[str, Any] = {
        "schema_version": WORK_II_FORMAL_EVALUATOR_SUMMARY_VERSION,
        "formal_result": True,
        "status": "completed" if not failures else "terminal_with_retained_failures",
        "formal_preflight_sha256": manifest["preflight_sha256"],
        "orchestration_manifest_sha256": orchestration_hash,
        "resume": bool(resume),
        "provider_call_count": 0,
        "participant_feedback_emitted": False,
        "participant_operation_denominator_impact": 0,
        "denominators": {
            "participant_session_count": len(typed_cells),
            "participant_terminal_session_count": len(terminal_receipts),
            "participant_state_counts": participant_state_counts,
            "c2_roster": roster,
            "truth_cluster_count": len(truth_reports),
            "truth_scheduled_execution_count": truth_query_count,
            "truth_completed_execution_count": truth_completed,
            "truth_failed_execution_count": truth_query_count - truth_completed,
            "truth_scheduled_query_metric_count": truth_metric_count,
            "truth_completed_query_metric_count": sum(
                int(report["completed_truth_query_metric_count"])
                for report in truth_reports.values()
            ),
            "blind_session_count": len(typed_cells),
            "blind_evaluable_session_count": len(blind_plans),
            "blind_scheduled_target_count": blind_target_count,
            "blind_scheduled_execution_count": blind_execution_count,
            "blind_launched_execution_count": blind_launched,
            "blind_completed_execution_count": blind_completed,
            "blind_failed_or_unstarted_execution_count": blind_execution_count
            - blind_completed,
        },
        "resume_audit": {
            "reused_truth_cluster_count": reused_truth,
            "new_truth_cluster_count": len(truth_reports) - reused_truth,
            "reused_blind_session_count": reused_blind,
            "new_blind_session_count": len(blind_reports) - reused_blind,
        },
        "failure_count": len(failures),
        "failures": failures,
    }
    summary["summary_sha256"] = _self_hash(summary, "summary_sha256")
    write_json_atomic(output_root / "summary.json", summary)
    emit(
        {
            "event": "formal_evaluators_finished",
            "status": summary["status"],
            "truth_completed_execution_count": truth_completed,
            "blind_completed_execution_count": blind_completed,
            "failure_count": len(failures),
        }
    )
    return summary


__all__ = [
    "FORMAL_EVALUATOR_SOURCE_ROOTS",
    "WORK_II_FORMAL_EVALUATOR_ORCHESTRATION_VERSION",
    "WORK_II_FORMAL_EVALUATOR_SUMMARY_VERSION",
    "WORK_II_FORMAL_EVALUATOR_UNIT_VERSION",
    "build_formal_evaluator_source_binding",
    "execute_formal_evaluators",
]
