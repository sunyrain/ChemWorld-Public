"""Immutable execution and evidence-chain contracts for Work II D1 triplets."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256, write_json_atomic

D1_PROVIDER_ATTEMPT_LIMIT = 2
D1_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
D1_STORE_VERSION = "chemworld-work-ii-d1-terminal-store-0.1"
D1_ATTEMPT_VERSION = "chemworld-work-ii-d1-provider-attempt-0.1"
D1_TERMINAL_VERSION = "chemworld-work-ii-d1-terminal-receipt-0.1"
D1_ADMISSION_VERSION = "chemworld-work-ii-d1-admission-receipt-0.1"
D1_TERMINAL_STATES = frozenset({"completed", "right_censored", "failed"})
D1_INFRASTRUCTURE_FAILURES = {
    "provider_process_launch_failed": "provider_process_launch",
    "child_reported_preoperation_infrastructure_failure": "child_provider_session",
    "missing_terminal_summary_zero_committed_operations": "child_terminal_materialization",
    "unreadable_terminal_summary_zero_committed_operations": "child_terminal_materialization",
}

D1_EXECUTION_CONTRACT = {
    "accepted_terminal_cells_are_immutable": True,
    "missing_infrastructure_only_resume": True,
    "persisted_committed_operation_forbids_replacement": True,
    "provider_attempt_limit_per_cell": D1_PROVIDER_ATTEMPT_LIMIT,
    "systemic_guard_counter": "committed_operation_count",
    "evidence_chain_order": ["participant", "truth", "blind", "admission"],
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _terminal_admission_mode_errors(report: Mapping[str, Any], *, label: str) -> list[str]:
    """Reject evidence explicitly produced outside the release-eligible boundary."""

    errors: list[str] = []
    if report.get("execution_mode") == "development":
        errors.append(f"{label} development report cannot support terminal admission")
    if report.get("release_eligible") is False:
        errors.append(f"{label} non-release report cannot support terminal admission")
    return errors


def _inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    path = (root.resolve() / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes repository")
    return path


def _binding(
    root: Path,
    path: Path,
    *,
    embedded: tuple[str, object] | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"cannot bind missing repository artifact: {resolved}")
    result: dict[str, Any] = {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(resolved),
    }
    if embedded is not None:
        result[embedded[0]] = embedded[1]
    return result


def build_d1_qualification_evidence_binding(
    root: Path,
    *,
    source_config_path: Path,
    qualification_package_path: Path,
    qualification_plan_path: Path,
) -> dict[str, Any]:
    """Bind the exact Q2 package and the record/plan that authorized D1."""

    root = root.resolve()
    source_path = source_config_path.resolve()
    package_path = qualification_package_path.resolve()
    plan_path = qualification_plan_path.resolve()
    source = _load(source_path)
    package = _load(package_path)
    plan = _load(plan_path)
    schema = str(package.get("schema_version", ""))
    if package.get("execution_context") != source.get("execution_context"):
        raise ValueError("D1 qualification package is not from the source release freeze")
    package_hash = package.get("package_sha256")
    if package_hash != _self_hash(package, "package_sha256"):
        raise ValueError("D1 qualification package self-hash mismatch")

    if "constitutive-structural" in schema:
        kind = "A_S_q2_package_and_plan"
        intervention = source.get("intervention")
        intervention = intervention if isinstance(intervention, Mapping) else {}
        candidate_id = intervention.get("candidate_id")
        candidate_laws = package.get("candidate_laws")
        candidate_laws = candidate_laws if isinstance(candidate_laws, Mapping) else {}
        candidate = candidate_laws.get(candidate_id)
        candidate = candidate if isinstance(candidate, Mapping) else {}
        world_evidence = candidate.get("world_evidence")
        world_evidence = world_evidence if isinstance(world_evidence, list) else []
        if (
            not isinstance(candidate_id, str)
            or candidate.get("task_id") != source.get("task_id")
        ):
            raise ValueError("A-S Q2 package candidate task differs from source config")
        if (
            package.get("all_five_world_cohorts_passed") is not True
            or len(world_evidence) != 5
            or {row.get("world_seed") for row in world_evidence if isinstance(row, Mapping)}
            != set(range(5))
            or any(
                not isinstance(row, Mapping) or row.get("passed") is not True
                for row in world_evidence
            )
        ):
            raise ValueError("A-S Q2 package did not pass all five registered worlds")
        package_plan = package.get("plan_binding")
        if not isinstance(package_plan, Mapping):
            raise ValueError("A-S Q2 package lacks its qualification plan binding")
        expected_plan = _binding(
            root,
            plan_path,
            embedded=("plan_sha256", plan.get("plan_sha256")),
        )
        if dict(package_plan) != expected_plan:
            raise ValueError("A-S Q2 package qualification plan binding is stale")
        if plan.get("plan_sha256") != _self_hash(plan, "plan_sha256"):
            raise ValueError("A-S qualification plan self-hash mismatch")
        if plan.get("execution_context") != source.get("execution_context"):
            raise ValueError("A-S qualification plan is not from the source release freeze")
        plan_binding = expected_plan
    else:
        kind = "A_P_q2_package_and_generation_record"
        if package.get("task_id") != source.get("task_id"):
            raise ValueError("D1 qualification package task differs from source config")
        if package.get("qualification_passed") is not True:
            raise ValueError("D1 qualification package did not pass")
        if plan.get("task_id") != source.get("task_id"):
            raise ValueError("A-P Q2 generation record task differs from source config")
        if plan.get("qualification_passed") is not True:
            raise ValueError("A-P Q2 generation record did not pass")
        if plan.get("execution_context") != source.get("execution_context"):
            raise ValueError("A-P Q2 generation record is not from the source release freeze")
        if plan.get("summary_sha256") != _self_hash(plan, "summary_sha256"):
            raise ValueError("A-P Q2 generation record self-hash mismatch")
        generated_package = plan.get("generated_package")
        generated_d1 = plan.get("generated_d1_config")
        if (
            not isinstance(generated_package, Mapping)
            or generated_package.get("path") != package_path.relative_to(root).as_posix()
            or generated_package.get("sha256") != file_sha256(package_path)
        ):
            raise ValueError("A-P Q2 generation record does not bind the package")
        if (
            not isinstance(generated_d1, Mapping)
            or generated_d1.get("path") != source_path.relative_to(root).as_posix()
            or generated_d1.get("sha256") != file_sha256(source_path)
        ):
            raise ValueError("A-P Q2 generation record does not bind the source D1 config")
        plan_binding = _binding(
            root,
            plan_path,
            embedded=("summary_sha256", plan["summary_sha256"]),
        )

    result = {
        "schema_version": "chemworld-work-ii-d1-qualification-evidence-binding-0.1",
        "kind": kind,
        "qualification_package": _binding(
            root, package_path, embedded=("package_sha256", package_hash)
        ),
        "qualification_plan": plan_binding,
    }
    result["binding_sha256"] = _self_hash(result, "binding_sha256")
    return result


def validate_d1_qualification_evidence(root: Path, config: Mapping[str, Any]) -> list[str]:
    value = config.get("qualification_evidence")
    if not isinstance(value, Mapping):
        return ["D1 config lacks exact qualification package/plan bindings"]
    errors: list[str] = []
    if value.get("binding_sha256") != _self_hash(value, "binding_sha256"):
        errors.append("D1 qualification evidence binding self-hash mismatch")
    try:
        basis = config.get("development_resource_basis")
        basis = basis if isinstance(basis, Mapping) else {}
        source = _inside(root, basis.get("source_config"), label="D1 source")
        package = _inside(
            root,
            value.get("qualification_package", {}).get("path"),
            label="D1 qualification package",
        )
        plan = _inside(
            root,
            value.get("qualification_plan", {}).get("path"),
            label="D1 qualification plan",
        )
        for label, path, binding in (
            ("package", package, value.get("qualification_package")),
            ("plan", plan, value.get("qualification_plan")),
        ):
            if (
                not path.is_file()
                or not isinstance(binding, Mapping)
                or binding.get("sha256") != file_sha256(path)
            ):
                errors.append(f"D1 qualification {label} file binding is stale")
        expected = build_d1_qualification_evidence_binding(
            root,
            source_config_path=source,
            qualification_package_path=package,
            qualification_plan_path=plan,
        )
        if dict(value) != expected:
            errors.append("D1 qualification evidence differs from deterministic rebuild")
        if _load(source).get("execution_context") != config.get("execution_context"):
            errors.append("D1 execution config crossed its qualification release freeze")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"D1 qualification evidence cannot be rebuilt: {error}")
    return errors


class D1CellStore:
    """Write-once D1 terminal store with one missing-infrastructure resume."""

    def __init__(
        self,
        root: Path,
        *,
        config_path: Path,
        task_id: str,
        world_seeds: Sequence[int],
        arms: Sequence[str],
    ) -> None:
        self.root = root.resolve()
        self.output_root = self.root.parent
        self.terminals = self.root / "terminal"
        self.provider_attempts = self.root / "provider_attempts"
        self.infrastructure_attempts = self.root / "infrastructure_attempts"
        self.config_path = config_path.resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        self.config_binding = {
            "path": str(self.config_path),
            "sha256": file_sha256(self.config_path),
        }
        cells = [
            {
                "task_id": task_id,
                "world_seed": int(seed),
                "prior_arm": str(arm),
                "provider_attempt_limit": D1_PROVIDER_ATTEMPT_LIMIT,
            }
            for seed in world_seeds
            for arm in arms
        ]
        for cell in cells:
            cell["cell_key_sha256"] = canonical_json_sha256(
                {**cell, "config_sha256": self.config_binding["sha256"]}
            )
        self.cells = {str(cell["cell_key_sha256"]): cell for cell in cells}
        self.by_identity = {
            (int(cell["world_seed"]), str(cell["prior_arm"])): key
            for key, cell in self.cells.items()
        }
        manifest = {
            "schema_version": D1_STORE_VERSION,
            "execution_contract": D1_EXECUTION_CONTRACT,
            "config_binding": self.config_binding,
            "cells": cells,
        }
        manifest["store_sha256"] = _self_hash(manifest, "store_sha256")
        self.manifest = manifest
        self.manifest_path = self.root / "store.json"
        if self.manifest_path.exists():
            if _load(self.manifest_path) != manifest:
                raise ValueError("D1 immutable store manifest changed")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            write_json_atomic(self.manifest_path, manifest)

    def key(self, world_seed: int, arm: str) -> str:
        try:
            return self.by_identity[(int(world_seed), str(arm))]
        except KeyError as error:
            raise ValueError(f"unknown D1 cell: seed={world_seed}, arm={arm}") from error

    def record_provider_attempt_launch(self, key: str, *, attempt_id: str) -> Path:
        cell = self.cells[key]
        existing = sorted((self.provider_attempts / key).glob("*.json"))
        if len(existing) >= D1_PROVIDER_ATTEMPT_LIMIT:
            raise ValueError(f"D1 cell exhausted provider attempt cap=2: {cell}")
        if existing and not self._has_retryable_infrastructure(
            key, str(_load(existing[-1])["attempt_id"])
        ):
            raise ValueError("D1 resume is allowed only after recorded missing infrastructure")
        payload = {
            "schema_version": D1_ATTEMPT_VERSION,
            "state": "provider_process_launch_authorized",
            "cell_key_sha256": key,
            "attempt_id": attempt_id,
            "attempt_index": len(existing) + 1,
            "attempt_limit": D1_PROVIDER_ATTEMPT_LIMIT,
            "store_sha256": self.manifest["store_sha256"],
        }
        payload["attempt_sha256"] = _self_hash(payload, "attempt_sha256")
        attempt_root = self.provider_attempts / key
        attempt_root.mkdir(parents=True, exist_ok=True)
        slot = attempt_root / f".attempt-{payload['attempt_index']}.claim"
        try:
            with slot.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(attempt_id + "\n")
        except FileExistsError as error:
            raise ValueError(
                f"D1 provider attempt index {payload['attempt_index']} is already claimed"
            ) from error
        target = attempt_root / f"{attempt_id}.json"
        self._write_once(target, payload)
        return target

    def record_infrastructure_failure(
        self,
        key: str,
        *,
        attempt_id: str,
        error_type: str,
        error_message: str,
        reason_code: str,
        committed_operation_count: int,
        log_path: Path,
        attempt_evidence_paths: Mapping[str, Path] | None = None,
    ) -> Path:
        if committed_operation_count != 0:
            raise ValueError("D1 infrastructure resume cannot follow a committed operation")
        if reason_code not in D1_INFRASTRUCTURE_FAILURES:
            raise ValueError("D1 infrastructure failure reason is outside the frozen taxonomy")
        launch = self.provider_attempts / key / f"{attempt_id}.json"
        if not launch.is_file():
            raise ValueError("D1 infrastructure failure lacks its provider launch receipt")
        payload = {
            "schema_version": D1_ATTEMPT_VERSION,
            "state": "retryable_missing_infrastructure",
            "cell_key_sha256": key,
            "attempt_id": attempt_id,
            "committed_operation_count": 0,
            "error_type": error_type,
            "error_message": error_message[:2000],
            "reason_code": reason_code,
            "reason_stage": D1_INFRASTRUCTURE_FAILURES[reason_code],
            "log": _binding(self.output_root, log_path),
            "attempt_evidence": {
                str(label): _binding(self.output_root, path)
                for label, path in (attempt_evidence_paths or {}).items()
            },
            "store_sha256": self.manifest["store_sha256"],
        }
        payload["attempt_sha256"] = _self_hash(payload, "attempt_sha256")
        target = self.infrastructure_attempts / key / f"{attempt_id}.json"
        self._write_once(target, payload)
        return target

    def write_terminal(
        self,
        key: str,
        *,
        attempt_id: str,
        state: str,
        result_root: Path,
        committed_operation_count: int,
    ) -> Path:
        if state not in D1_TERMINAL_STATES:
            raise ValueError("invalid D1 terminal state")
        launch = self.provider_attempts / key / f"{attempt_id}.json"
        if not launch.is_file():
            raise ValueError("D1 terminal lacks its provider launch receipt")
        cell = self.cells[key]
        view = self.output_root / f"seed-{cell['world_seed']}" / str(cell["prior_arm"])
        if view.exists():
            raise FileExistsError(f"refusing to overwrite D1 terminal view: {view}")
        shutil.copytree(result_root, view)
        result = {
            name: _binding(self.output_root, path)
            for name, path in {
                "summary": view / "summary.json",
                "trajectory": view / "trajectory.jsonl",
                "report": view / "report.json",
            }.items()
            if path.is_file()
        }
        payload = {
            "schema_version": D1_TERMINAL_VERSION,
            "cell_key_sha256": key,
            "cell": cell,
            "attempt_id": attempt_id,
            "state": state,
            "committed_operation_count": int(committed_operation_count),
            "result": result,
            "result_sha256": canonical_json_sha256(result),
            "store_sha256": self.manifest["store_sha256"],
        }
        payload["receipt_sha256"] = _self_hash(payload, "receipt_sha256")
        target = self.terminals / f"{key}.json"
        self._write_once(target, payload)
        return target

    def pending(self, *, resume: bool) -> list[dict[str, Any]]:
        audit = self.audit()
        if audit["invalid_receipts"]:
            raise ValueError("D1 store contains invalid receipts")
        if audit["terminal_count"] and not resume:
            raise ValueError("D1 store contains terminal cells; use missing-only resume")
        pending: list[dict[str, Any]] = []
        terminal = set(audit["terminal_cell_key_sha256"])
        for key, cell in self.cells.items():
            if key in terminal:
                continue
            count = int(audit["provider_attempt_counts_by_cell_key_sha256"].get(key, 0))
            if count >= D1_PROVIDER_ATTEMPT_LIMIT:
                raise ValueError(f"D1 cell exhausted provider attempt cap=2: {cell}")
            if count and not self._all_launches_retryable(key):
                raise ValueError("D1 missing cell is not eligible for infrastructure-only resume")
            pending.append(dict(cell))
        return pending

    def audit(self) -> dict[str, Any]:
        invalid: list[str] = []
        terminal: dict[str, dict[str, Any]] = {}
        attempts: dict[str, set[int]] = {}
        attempt_ids: set[tuple[str, str]] = set()
        for path in sorted(self.provider_attempts.glob("*/*.json")):
            try:
                row = _load(path)
                key = str(row["cell_key_sha256"])
                index = int(row["attempt_index"])
                if (
                    row.get("schema_version") != D1_ATTEMPT_VERSION
                    or row.get("state") != "provider_process_launch_authorized"
                    or key not in self.cells
                    or path.parent.name != key
                    or path.stem != row.get("attempt_id")
                    or index not in {1, 2}
                    or index in attempts.setdefault(key, set())
                    or row.get("attempt_limit") != D1_PROVIDER_ATTEMPT_LIMIT
                    or row.get("store_sha256") != self.manifest["store_sha256"]
                    or row.get("attempt_sha256") != _self_hash(row, "attempt_sha256")
                ):
                    raise ValueError("invalid attempt")
                attempts[key].add(index)
                attempt_ids.add((key, str(row["attempt_id"])))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        for key, indices in attempts.items():
            if indices != set(range(1, len(indices) + 1)):
                invalid.append((self.provider_attempts / key).as_posix())
        for path in sorted(self.infrastructure_attempts.glob("*/*.json")):
            try:
                row = _load(path)
                key = str(row["cell_key_sha256"])
                attempt_id = str(row["attempt_id"])
                log = row.get("log")
                attempt_evidence = row.get("attempt_evidence", {})
                if (
                    row.get("schema_version") != D1_ATTEMPT_VERSION
                    or row.get("state") != "retryable_missing_infrastructure"
                    or row.get("committed_operation_count") != 0
                    or row.get("reason_code") not in D1_INFRASTRUCTURE_FAILURES
                    or row.get("reason_stage")
                    != D1_INFRASTRUCTURE_FAILURES.get(str(row.get("reason_code")))
                    or (key, attempt_id) not in attempt_ids
                    or path.parent.name != key
                    or path.stem != attempt_id
                    or not self._binding_valid(log)
                    or not isinstance(attempt_evidence, Mapping)
                    or set(attempt_evidence) - {"summary", "trajectory"}
                    or any(
                        not self._binding_valid(binding)
                        for binding in attempt_evidence.values()
                    )
                    or row.get("store_sha256") != self.manifest["store_sha256"]
                    or row.get("attempt_sha256") != _self_hash(row, "attempt_sha256")
                ):
                    raise ValueError("invalid infrastructure receipt")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        for path in sorted(self.terminals.glob("*.json")):
            try:
                row = _load(path)
                key = str(row["cell_key_sha256"])
                result = row.get("result")
                if (
                    row.get("schema_version") != D1_TERMINAL_VERSION
                    or key not in self.cells
                    or path.stem != key
                    or key in terminal
                    or row.get("cell") != self.cells[key]
                    or (key, str(row.get("attempt_id"))) not in attempt_ids
                    or row.get("state") not in D1_TERMINAL_STATES
                    or not isinstance(result, Mapping)
                    or set(result) != {"summary", "trajectory", "report"}
                    or any(not self._binding_valid(item) for item in result.values())
                    or row.get("store_sha256") != self.manifest["store_sha256"]
                    or row.get("result_sha256") != canonical_json_sha256(result)
                    or row.get("receipt_sha256") != _self_hash(row, "receipt_sha256")
                ):
                    raise ValueError("invalid terminal receipt")
                terminal[key] = row
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        report = {
            "expected_cell_count": len(self.cells),
            "terminal_count": len(terminal),
            "terminal_cell_key_sha256": sorted(terminal),
            "missing_cell_key_sha256": sorted(set(self.cells) - set(terminal)),
            "provider_attempt_count": sum(len(rows) for rows in attempts.values()),
            "provider_attempt_counts_by_cell_key_sha256": {
                key: len(rows) for key, rows in attempts.items()
            },
            "invalid_receipts": sorted(set(invalid)),
        }
        report["audit_sha256"] = canonical_json_sha256(report)
        return report

    def _all_launches_retryable(self, key: str) -> bool:
        launches = sorted((self.provider_attempts / key).glob("*.json"))
        return bool(launches) and all(
            self._has_retryable_infrastructure(key, str(_load(path)["attempt_id"]))
            for path in launches
        )

    def _has_retryable_infrastructure(self, key: str, attempt_id: str) -> bool:
        path = self.infrastructure_attempts / key / f"{attempt_id}.json"
        return path.is_file() and _load(path).get("committed_operation_count") == 0

    def _binding_valid(self, value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        try:
            path = _inside(self.output_root, value.get("path"), label="D1 result")
        except ValueError:
            return False
        return path.is_file() and value.get("sha256") == file_sha256(path)

    @staticmethod
    def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite D1 evidence: {path}")
        write_json_atomic(path, dict(payload))


def build_d1_admission_receipt(
    root: Path,
    *,
    config_path: Path,
    participant_root: Path,
    truth_report_path: Path,
    blind_report_paths: Sequence[Path],
    evaluation_report_path: Path,
) -> dict[str, Any]:
    """Bind participant terminals to truth, blind, and the final D1 admission decision."""

    root = root.resolve()
    config = _load(config_path)
    matrix_path = participant_root.resolve() / "matrix_report.json"
    matrix = _load(matrix_path)
    truth = _load(truth_report_path)
    evaluation = _load(evaluation_report_path)
    blind = [_load(path) for path in blind_report_paths]
    errors: list[str] = []
    for label, report in (
        ("participant matrix", matrix),
        ("truth evaluator", truth),
        ("D1 evaluation", evaluation),
    ):
        errors.extend(_terminal_admission_mode_errors(report, label=label))
    for index, report in enumerate(blind):
        errors.extend(
            _terminal_admission_mode_errors(
                report,
                label=f"blind evaluator {index}",
            )
        )
    terminal_bindings = matrix.get("terminal_receipt_bindings")
    terminal_receipts: list[dict[str, Any]] = []
    if not isinstance(terminal_bindings, list) or len(terminal_bindings) != 3:
        errors.append("participant matrix lacks three immutable terminal receipts")
    else:
        for item in terminal_bindings:
            if not isinstance(item, Mapping):
                errors.append("participant terminal binding is malformed")
                continue
            try:
                path = _inside(
                    participant_root.resolve(),
                    item.get("path"),
                    label="participant terminal",
                )
                receipt = _load(path)
                if item.get("sha256") != file_sha256(path) or item.get(
                    "receipt_sha256"
                ) != receipt.get("receipt_sha256"):
                    errors.append("participant terminal receipt binding is stale")
                terminal_receipts.append(receipt)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                errors.append("participant terminal receipt cannot be read")
    terminal_cells = [row.get("cell") for row in terminal_receipts]
    terminal_arms = [row.get("prior_arm") for row in terminal_cells if isinstance(row, Mapping)]
    terminal_worlds = {row.get("world_seed") for row in terminal_cells if isinstance(row, Mapping)}
    terminal_tasks = {row.get("task_id") for row in terminal_cells if isinstance(row, Mapping)}
    if (
        len(terminal_receipts) != 3
        or len({row.get("receipt_sha256") for row in terminal_receipts}) != 3
        or set(terminal_arms) != set(D1_ARMS)
        or len(terminal_arms) != len(set(terminal_arms))
        or terminal_worlds != {matrix.get("world_seeds", [None])[0]}
        or terminal_tasks != {config.get("task_id")}
    ):
        errors.append("participant terminals do not exactly cover one task/world three-arm triplet")
    if validate_d1_qualification_evidence(root, config):
        errors.append("D1 config qualification evidence is invalid")
    if truth.get("status") != "completed" or truth.get("task_id") != config.get("task_id"):
        errors.append("truth evaluator did not complete for the D1 task")
    if truth.get("report_sha256") != _self_hash(truth, "report_sha256"):
        errors.append("truth evaluator report self-hash mismatch")
    blind_keys = [row.get("cell_key_sha256") for row in blind]
    if (
        len(blind) != 3
        or len(set(blind_keys)) != 3
        or any(row.get("report_sha256") != _self_hash(row, "report_sha256") for row in blind)
    ):
        errors.append("blind evaluator reports are incomplete or invalid")
    for path, row in zip(blind_report_paths, blind, strict=True):
        try:
            plan = _load(path.resolve().with_name("plan.json"))
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("blind evaluator lacks its participant-bound plan")
            continue
        if (
            row.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("plan_sha256") != _self_hash(plan, "plan_sha256")
            or plan.get("task_id") != config.get("task_id")
            or plan.get("world_seed") != matrix.get("world_seeds", [None])[0]
        ):
            errors.append("blind evaluator plan does not bind the D1 participant cell")
    if evaluation.get("participant_run", {}).get("matrix_report_sha256") != file_sha256(
        matrix_path
    ):
        errors.append("D1 evaluation does not bind the participant matrix")
    if evaluation.get("truth_report_sha256") != truth.get("report_sha256"):
        errors.append("D1 evaluation does not bind the truth evaluator")
    if evaluation.get("report_sha256") != _self_hash(evaluation, "report_sha256"):
        errors.append("D1 evaluation report self-hash mismatch")
    evaluation_cells = evaluation.get("cells")
    evaluation_cells = evaluation_cells if isinstance(evaluation_cells, list) else []
    if {row.get("cell_key_sha256") for row in evaluation_cells if isinstance(row, Mapping)} != set(
        blind_keys
    ):
        errors.append("D1 evaluation/blind participant cell bindings differ")
    expected_blind_status = {str(row.get("cell_key_sha256")): row.get("status") for row in blind}
    if any(
        expected_blind_status.get(str(row.get("cell_key_sha256"))) != "completed"
        or row.get("blind_evaluation_status") != "completed"
        for row in evaluation_cells
        if isinstance(row, Mapping)
    ):
        errors.append("D1 evaluation does not bind three completed blind reports")
    action = evaluation.get("action_layer")
    action = action if isinstance(action, Mapping) else {}
    passed = (
        not errors
        and evaluation.get("status") == "passed"
        and action.get("status") == "participant_interpretable"
        and action.get("submitted_recommendations_replaced") is False
    )
    participant_receipt_bindings = []
    if (
        isinstance(terminal_bindings, list)
        and len(terminal_bindings) == len(terminal_receipts) == 3
    ):
        participant_receipt_bindings = [
            _binding(
                root,
                _inside(
                    participant_root.resolve(),
                    item["path"],
                    label="participant terminal",
                ),
                embedded=("receipt_sha256", terminal["receipt_sha256"]),
            )
            for item, terminal in zip(terminal_bindings, terminal_receipts, strict=True)
        ]
    receipt = {
        "schema_version": D1_ADMISSION_VERSION,
        "status": "passed_terminal_d1_admission" if passed else "failed_retained",
        "task_id": config.get("task_id"),
        "world_seed": matrix.get("world_seeds", [None])[0],
        "evidence_order": ["participant", "truth", "blind", "admission"],
        "config_binding": _binding(root, config_path),
        "participant_matrix_binding": _binding(root, matrix_path),
        "participant_terminal_receipts": participant_receipt_bindings,
        "truth_report_binding": _binding(
            root, truth_report_path, embedded=("report_sha256", truth.get("report_sha256"))
        ),
        "blind_report_bindings": [
            _binding(root, path, embedded=("report_sha256", row.get("report_sha256")))
            for path, row in zip(blind_report_paths, blind, strict=True)
        ],
        "evaluation_report_binding": _binding(
            root,
            evaluation_report_path,
            embedded=("report_sha256", evaluation.get("report_sha256")),
        ),
        "validation_errors": errors,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return receipt


__all__ = [
    "D1_ADMISSION_VERSION",
    "D1_ARMS",
    "D1_EXECUTION_CONTRACT",
    "D1_PROVIDER_ATTEMPT_LIMIT",
    "D1CellStore",
    "build_d1_admission_receipt",
    "build_d1_qualification_evidence_binding",
    "validate_d1_qualification_evidence",
]
