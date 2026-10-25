"""External cluster sharding for provider-free Work II A-E v0.3 phases.

The scientific plan remains the sole source of execution identities, seeds, recipes,
thresholds, and denominators.  A shard owns complete ``(locus_id, world_index)``
clusters (24 executions) and writes only beneath its own external directory.  The
merger accepts results only after every planned execution is present exactly once.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.work_ii_ae_prior_qualification_v03 import (
    RECEIPT_VERSION,
    AEPriorQualificationV03Error,
    build_development_summary,
    build_phase_report,
    execute_one,
    select_screen_loci,
    validate_next_receipt,
    validate_plan,
    validate_receipt_denominator,
)

SHARD_MANIFEST_VERSION = "chemworld-work-ii-ae-v03-shard-manifest-0.1"
SHARD_SUMMARY_VERSION = "chemworld-work-ii-ae-v03-shard-summary-0.1"


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AEPriorQualificationV03Error(f"required shard artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AEPriorQualificationV03Error(f"{path} must contain one object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AEPriorQualificationV03Error(f"refusing to replace shard artifact: {path}")
    write_json_atomic(path, dict(payload))


def _cluster_assignments(
    plan: Mapping[str, Any], shard_count: int, *, start_execution_index: int = 0
) -> list[list[dict[str, Any]]]:
    """Return deterministic whole-world clusters assigned round-robin to shards."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise AEPriorQualificationV03Error("shard count must be a positive integer")
    if (
        isinstance(start_execution_index, bool)
        or not isinstance(start_execution_index, int)
        or start_execution_index < 0
    ):
        raise AEPriorQualificationV03Error(
            "shard start execution index must be a nonnegative integer"
        )
    errors = validate_plan(plan)
    if errors:
        raise AEPriorQualificationV03Error("invalid sharded plan: " + "; ".join(errors))
    executions = plan.get("executions")
    executions = executions if isinstance(executions, list) else []
    if any(row.get("execution_index") != index for index, row in enumerate(executions)):
        raise AEPriorQualificationV03Error(
            "plan execution indexes are not the canonical complete sequence"
        )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    first_index: dict[tuple[str, int], int] = {}
    for row in executions:
        key = (str(row.get("locus_id")), int(row.get("world_index", -1)))
        grouped[key].append(row)
        first_index.setdefault(key, int(row["execution_index"]))
    clusters = sorted(grouped.items(), key=lambda item: first_index[item[0]])
    expected_coordinates = {
        (anchor, category, replicate)
        for replicate in range(3)
        for anchor in range(2)
        for category in range(4)
    }
    for key, rows in clusters:
        observed = {
            (
                int(row.get("anchor_id", -1)),
                int(row.get("target_category", -1)),
                int(row.get("replicate", -1)),
            )
            for row in rows
        }
        if len(rows) != 24 or observed != expected_coordinates:
            raise AEPriorQualificationV03Error(
                f"A-E v0.3 cluster {key} is not the exact 2x4x3 unit"
            )
        if len({row.get("world_seed") for row in rows}) != 1:
            raise AEPriorQualificationV03Error(
                f"A-E v0.3 cluster {key} mixes world seeds"
            )
    if start_execution_index > len(executions):
        raise AEPriorQualificationV03Error(
            "shard start execution index exceeds the plan denominator"
        )
    prefix_clusters = [
        rows for _key, rows in clusters if first_index[_key] < start_execution_index
    ]
    if sum(len(rows) for rows in prefix_clusters) != start_execution_index:
        raise AEPriorQualificationV03Error(
            "shard start execution index is not a complete cluster boundary"
        )
    clusters = [
        (key, rows)
        for key, rows in clusters
        if first_index[key] >= start_execution_index
    ]
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for ordinal, (_key, rows) in enumerate(clusters):
        assigned[ordinal % shard_count].extend(rows)
    return assigned


def _shard_name(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index:05d}-of-{shard_count:05d}"


def _platform_failure_marker(shard_base: Path) -> Path:
    return shard_base / "platform-failure.json"


def _write_first_platform_failure(shard_base: Path, payload: Mapping[str, Any]) -> None:
    """Preserve the first cross-worker failure without replacing another worker's record."""

    marker = _platform_failure_marker(shard_base)
    rendered = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        pass


def execute_shard(
    root: Path,
    plan: Mapping[str, Any],
    shard_base: Path,
    *,
    shard_index: int,
    shard_count: int,
    start_execution_index: int = 0,
    executor: Callable[..., dict[str, Any]] = execute_one,
) -> dict[str, Any]:
    """Execute one write-once set of complete A-E v0.3 world clusters."""

    assignments = _cluster_assignments(
        plan, shard_count, start_execution_index=start_execution_index
    )
    if not 0 <= shard_index < shard_count:
        raise AEPriorQualificationV03Error("shard index is outside the shard set")
    shard_base = shard_base.resolve()
    shard_base.mkdir(parents=True, exist_ok=True)
    shard_root = shard_base / _shard_name(shard_index, shard_count)
    if shard_root.exists():
        raise AEPriorQualificationV03Error(
            f"shard output already exists and is immutable: {shard_root}"
        )
    shard_root.mkdir()
    rows = assignments[shard_index]
    manifest: dict[str, Any] = {
        "schema_version": SHARD_MANIFEST_VERSION,
        "development_only": True,
        "phase": plan["phase"],
        "plan_sha256": plan["plan_sha256"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "imported_prefix_count": start_execution_index,
        "cluster_keys": [
            {"locus_id": row["locus_id"], "world_index": row["world_index"]}
            for row in rows[::24]
        ],
        "execution_indexes": [int(row["execution_index"]) for row in rows],
        "execution_count": len(rows),
        "provider_call_count": 0,
    }
    manifest["shard_manifest_sha256"] = _self_hash(
        manifest, "shard_manifest_sha256"
    )
    _write_once(shard_root / "shard-manifest.json", manifest)
    completed = 0
    started = time.monotonic()
    for row in rows:
        if _platform_failure_marker(shard_base).exists():
            raise AEPriorQualificationV03Error(
                "another shard reported a platform failure; the phase must restart"
            )
        index = int(row["execution_index"])
        execution_root = shard_root / "executions" / str(index)
        receipt = executor(
            root,
            plan,
            row,
            shard_root,
            execution_root=execution_root,
        )
        receipt_errors = validate_next_receipt(plan, index, receipt)
        if receipt_errors:
            raise AEPriorQualificationV03Error(
                "invalid shard receipt: " + "; ".join(receipt_errors)
            )
        receipt_path = shard_root / "receipts" / f"{index}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _write_once(receipt_path, receipt)
        completed += 1
        elapsed = time.monotonic() - started
        throughput_per_minute = 60.0 * completed / elapsed if elapsed > 0 else 0.0
        eta_seconds = (
            60.0 * (len(rows) - completed) / throughput_per_minute
            if throughput_per_minute > 0
            else None
        )
        print(
            json.dumps(
                {
                    "stage": plan["phase"],
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "completed": completed,
                    "total": len(rows),
                    "global_execution_index": index,
                    "elapsed_s": round(elapsed, 3),
                    "throughput_per_min": round(throughput_per_minute, 3),
                    "eta_s": round(eta_seconds, 1) if eta_seconds is not None else None,
                    "platform_failures": int(receipt["status"] == "platform_failure"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if receipt["status"] == "platform_failure":
            failure = {
                "phase": plan["phase"],
                "plan_sha256": plan["plan_sha256"],
                "shard_index": shard_index,
                "execution_index": index,
                "execution_id": receipt["execution_id"],
                "failure": receipt["failure"],
                "restart_required_from_execution_zero": True,
            }
            _write_first_platform_failure(shard_base, failure)
            summary = {
                "schema_version": SHARD_SUMMARY_VERSION,
                "status": "platform_failure",
                "plan_sha256": plan["plan_sha256"],
                "shard_manifest_sha256": manifest["shard_manifest_sha256"],
                "completed_receipts": completed,
                "expected_receipts": len(rows),
                "platform_failure_count": 1,
            }
            summary["shard_summary_sha256"] = _self_hash(
                summary, "shard_summary_sha256"
            )
            _write_once(shard_root / "shard-summary.json", summary)
            return summary
    summary = {
        "schema_version": SHARD_SUMMARY_VERSION,
        "status": "completed",
        "plan_sha256": plan["plan_sha256"],
        "shard_manifest_sha256": manifest["shard_manifest_sha256"],
        "completed_receipts": completed,
        "expected_receipts": len(rows),
        "platform_failure_count": 0,
    }
    summary["shard_summary_sha256"] = _self_hash(summary, "shard_summary_sha256")
    _write_once(shard_root / "shard-summary.json", summary)
    return summary


def _collect_imported_prefix(
    plan: Mapping[str, Any],
    imported_prefix: Path | None,
    *,
    imported_prefix_count: int,
) -> dict[int, tuple[dict[str, Any], Path]]:
    """Validate a stopped serial prefix without modifying or adopting its tail."""

    if (imported_prefix is None) != (imported_prefix_count == 0):
        raise AEPriorQualificationV03Error(
            "imported prefix path and positive count must be provided together"
        )
    if imported_prefix is None:
        return {}
    root = imported_prefix.resolve()
    source_plan = _load_object(root / "plan.json")
    if source_plan != plan:
        raise AEPriorQualificationV03Error(
            "imported serial prefix plan differs from the shard plan"
        )
    receipt_root = root / "receipts"
    members = list(receipt_root.iterdir()) if receipt_root.is_dir() else []
    if any(not path.is_file() or path.suffix != ".json" for path in members):
        raise AEPriorQualificationV03Error(
            "imported serial prefix receipts contain unexpected members"
        )
    try:
        members.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise AEPriorQualificationV03Error(
            "imported serial prefix has a nonnumeric receipt filename"
        ) from error
    expected_indexes = list(range(imported_prefix_count))
    if [int(path.stem) for path in members] != expected_indexes:
        raise AEPriorQualificationV03Error(
            "imported serial receipts are not the declared complete prefix"
        )
    observed: dict[int, tuple[dict[str, Any], Path]] = {}
    for index, receipt_path in enumerate(members):
        receipt = _load_object(receipt_path)
        errors = validate_next_receipt(plan, index, receipt)
        if errors:
            raise AEPriorQualificationV03Error(
                f"invalid imported receipt {index}: " + "; ".join(errors)
            )
        if (
            receipt.get("schema_version") != RECEIPT_VERSION
            or receipt.get("execution_index") != index
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("provider_call_count") != 0
            or receipt.get("status") != "completed"
            or receipt.get("failure") is not None
            or receipt.get("exact_replay", {}).get("verified") is not True
        ):
            raise AEPriorQualificationV03Error(
                f"imported receipt {index} is not a completed provider-free replay"
            )
        trajectory_binding = receipt.get("trajectory")
        trajectory_binding = (
            trajectory_binding if isinstance(trajectory_binding, Mapping) else {}
        )
        relative = PurePosixPath(str(trajectory_binding.get("path", "")))
        if relative.parts != ("executions", str(index), "trajectory.jsonl"):
            raise AEPriorQualificationV03Error(
                f"imported receipt {index} trajectory path is not canonical"
            )
        trajectory = (root / Path(*relative.parts)).resolve()
        if (
            not trajectory.is_relative_to(root)
            or not trajectory.is_file()
            or file_sha256(trajectory) != trajectory_binding.get("sha256")
        ):
            raise AEPriorQualificationV03Error(
                f"imported receipt {index} trajectory binding is invalid"
            )
        observed[index] = (receipt, trajectory)
    return observed


def collect_shard_receipts(
    plan: Mapping[str, Any],
    shard_base: Path,
    *,
    shard_count: int,
    imported_prefix: Path | None = None,
    imported_prefix_count: int = 0,
) -> list[dict[str, Any]]:
    """Validate a complete shard set and return canonical standard-output receipts."""

    assignments = _cluster_assignments(
        plan, shard_count, start_execution_index=imported_prefix_count
    )
    shard_base = shard_base.resolve()
    if _platform_failure_marker(shard_base).exists():
        raise AEPriorQualificationV03Error(
            "shard set contains a platform failure and cannot be merged"
        )
    expected_shard_names = {
        _shard_name(shard_index, shard_count) for shard_index in range(shard_count)
    }
    observed_shard_names = {
        path.name
        for path in shard_base.glob("shard-*")
        if path.is_dir()
    }
    if observed_shard_names != expected_shard_names:
        raise AEPriorQualificationV03Error(
            "shard directory set differs from the declared shard count"
        )
    observed = _collect_imported_prefix(
        plan, imported_prefix, imported_prefix_count=imported_prefix_count
    )
    for shard_index, expected_rows in enumerate(assignments):
        shard_root = shard_base / _shard_name(shard_index, shard_count)
        manifest = _load_object(shard_root / "shard-manifest.json")
        summary = _load_object(shard_root / "shard-summary.json")
        expected_indexes = [int(row["execution_index"]) for row in expected_rows]
        expected_cluster_keys = [
            {"locus_id": row["locus_id"], "world_index": row["world_index"]}
            for row in expected_rows[::24]
        ]
        if (
            manifest.get("schema_version") != SHARD_MANIFEST_VERSION
            or manifest.get("development_only") is not True
            or manifest.get("phase") != plan.get("phase")
            or manifest.get("shard_manifest_sha256")
            != _self_hash(manifest, "shard_manifest_sha256")
            or manifest.get("plan_sha256") != plan.get("plan_sha256")
            or manifest.get("shard_index") != shard_index
            or manifest.get("shard_count") != shard_count
            or manifest.get("imported_prefix_count") != imported_prefix_count
            or manifest.get("cluster_keys") != expected_cluster_keys
            or manifest.get("execution_indexes") != expected_indexes
            or manifest.get("execution_count") != len(expected_indexes)
            or manifest.get("provider_call_count") != 0
        ):
            raise AEPriorQualificationV03Error(
                f"shard {shard_index} manifest differs from deterministic assignment"
            )
        if (
            summary.get("schema_version") != SHARD_SUMMARY_VERSION
            or summary.get("shard_summary_sha256")
            != _self_hash(summary, "shard_summary_sha256")
            or summary.get("status") != "completed"
            or summary.get("plan_sha256") != plan.get("plan_sha256")
            or summary.get("shard_manifest_sha256")
            != manifest.get("shard_manifest_sha256")
            or summary.get("completed_receipts") != len(expected_indexes)
            or summary.get("expected_receipts") != len(expected_indexes)
            or summary.get("platform_failure_count") != 0
        ):
            raise AEPriorQualificationV03Error(
                f"shard {shard_index} is not a complete terminal shard"
            )
        receipt_root = shard_root / "receipts"
        receipt_members = list(receipt_root.iterdir()) if receipt_root.is_dir() else []
        if any(
            not path.is_file() or path.suffix != ".json" for path in receipt_members
        ):
            raise AEPriorQualificationV03Error(
                f"shard {shard_index} receipt directory has unexpected members"
            )
        receipt_files = receipt_members
        try:
            receipt_files.sort(key=lambda path: int(path.stem))
        except ValueError as error:
            raise AEPriorQualificationV03Error(
                f"shard {shard_index} has a nonnumeric receipt filename"
            ) from error
        if [int(path.stem) for path in receipt_files] != expected_indexes:
            raise AEPriorQualificationV03Error(
                f"shard {shard_index} has missing, extra, or reordered receipts"
            )
        for receipt_path in receipt_files:
            index = int(receipt_path.stem)
            if index in observed:
                raise AEPriorQualificationV03Error(
                    f"execution index {index} appears in more than one shard"
                )
            receipt = _load_object(receipt_path)
            errors = validate_next_receipt(plan, index, receipt)
            if errors:
                raise AEPriorQualificationV03Error(
                    f"invalid shard receipt {index}: " + "; ".join(errors)
                )
            if (
                receipt.get("schema_version") != RECEIPT_VERSION
                or receipt.get("execution_index") != index
                or receipt.get("plan_sha256") != plan.get("plan_sha256")
                or receipt.get("provider_call_count") != 0
                or receipt.get("status") != "completed"
                or receipt.get("failure") is not None
                or receipt.get("exact_replay", {}).get("verified") is not True
            ):
                raise AEPriorQualificationV03Error(
                    f"shard receipt {index} is not a completed provider-free replay"
                )
            trajectory_binding = receipt.get("trajectory")
            trajectory_binding = (
                trajectory_binding if isinstance(trajectory_binding, Mapping) else {}
            )
            relative = PurePosixPath(str(trajectory_binding.get("path", "")))
            if relative.parts != ("executions", str(index), "trajectory.jsonl"):
                raise AEPriorQualificationV03Error(
                    f"shard receipt {index} trajectory path is not canonical"
                )
            trajectory = (shard_root / Path(*relative.parts)).resolve()
            if (
                not trajectory.is_relative_to(shard_root)
                or not trajectory.is_file()
                or file_sha256(trajectory) != trajectory_binding.get("sha256")
            ):
                raise AEPriorQualificationV03Error(
                    f"shard receipt {index} trajectory binding is invalid"
                )
            canonical_receipt = deepcopy(receipt)
            canonical_receipt["trajectory"] = {
                "path": f"executions/{index}/trajectory.jsonl",
                "sha256": trajectory_binding["sha256"],
            }
            canonical_receipt["receipt_sha256"] = canonical_json_sha256(
                {
                    key: value
                    for key, value in canonical_receipt.items()
                    if key != "receipt_sha256"
                }
            )
            canonical_errors = validate_next_receipt(plan, index, canonical_receipt)
            if canonical_errors:
                raise AEPriorQualificationV03Error(
                    f"canonical merged receipt {index} is invalid: "
                    + "; ".join(canonical_errors)
                )
            observed[index] = (canonical_receipt, trajectory)
    expected_global = list(range(len(plan.get("executions", []))))
    if sorted(observed) != expected_global:
        missing = sorted(set(expected_global) - set(observed))
        extra = sorted(set(observed) - set(expected_global))
        raise AEPriorQualificationV03Error(
            f"shard merge denominator differs; missing={missing[:10]} extra={extra[:10]}"
        )
    return [
        {"receipt": observed[index][0], "trajectory_source": observed[index][1]}
        for index in expected_global
    ]


def materialize_merged_output(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_base: Path,
    output: Path,
    *,
    shard_count: int,
    imported_prefix: Path | None = None,
    imported_prefix_count: int = 0,
    fit_report: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    screen_report: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
    report_builder: Callable[..., dict[str, Any]] = build_phase_report,
) -> dict[str, Any]:
    """Atomically materialize the standard serial-layout phase output from shards."""

    plan_errors = validate_plan(plan)
    if plan_errors:
        raise AEPriorQualificationV03Error(
            "cannot merge invalid plan: " + "; ".join(plan_errors)
        )
    contract_binding = plan.get("contract_binding")
    contract_binding = contract_binding if isinstance(contract_binding, Mapping) else {}
    if contract_binding.get("canonical_sha256") != canonical_json_sha256(contract):
        raise AEPriorQualificationV03Error("sharded plan is detached from its contract")
    collected = collect_shard_receipts(
        plan,
        shard_base,
        shard_count=shard_count,
        imported_prefix=imported_prefix,
        imported_prefix_count=imported_prefix_count,
    )
    receipts = [item["receipt"] for item in collected]
    denominator_errors = validate_receipt_denominator(plan, receipts)
    if denominator_errors:
        raise AEPriorQualificationV03Error(
            "merged receipt denominator is invalid: " + "; ".join(denominator_errors)
        )
    phase = str(plan["phase"])
    if phase != "classifier_fit" and not isinstance(fit_report, Mapping):
        raise AEPriorQualificationV03Error(f"{phase} merge requires the fit report")
    report = report_builder(contract, plan, receipts, fit_report=fit_report)
    selected_output: dict[str, Any] | None = None
    development_summary: dict[str, Any] | None = None
    if phase == "prospective_screen":
        selected_output = select_screen_loci(contract, report["locus_results"])
    elif phase == "confirmation":
        if not all(
            isinstance(value, Mapping)
            for value in (fit_report, validation_report, screen_report, selection)
        ):
            raise AEPriorQualificationV03Error(
                "confirmation merge requires every upstream report and selection"
            )
        development_summary = build_development_summary(
            contract,
            fit_report,
            validation_report,
            screen_report,
            selection,
            report,
        )
    output = output.resolve()
    if output.exists():
        raise AEPriorQualificationV03Error(
            f"refusing to replace merged phase output: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.merging-", dir=output.parent)
    ).resolve()
    try:
        write_json_atomic(temporary / "plan.json", dict(plan))
        for item in collected:
            receipt = item["receipt"]
            index = int(receipt["execution_index"])
            destination = temporary / "executions" / str(index) / "trajectory.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["trajectory_source"], destination)
            if file_sha256(destination) != receipt["trajectory"]["sha256"]:
                raise AEPriorQualificationV03Error(
                    f"merged trajectory {index} changed during copy"
                )
            receipt_path = temporary / "receipts" / f"{index}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(receipt_path, receipt)
        write_json_atomic(temporary / "report.json", report)
        if selected_output is not None:
            write_json_atomic(temporary / "selection.json", selected_output)
        if development_summary is not None:
            write_json_atomic(temporary / "summary.json", development_summary)
        os.rename(temporary, output)
    except Exception:
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
    return {
        "status": report["status"],
        "phase": phase,
        "primary_executions": len(receipts),
        "exact_replays": sum(
            row["exact_replay"]["verified"] is True for row in receipts
        ),
        "shard_count": shard_count,
        "output": str(output),
    }


__all__ = [
    "SHARD_MANIFEST_VERSION",
    "SHARD_SUMMARY_VERSION",
    "collect_shard_receipts",
    "execute_shard",
    "materialize_merged_output",
]
