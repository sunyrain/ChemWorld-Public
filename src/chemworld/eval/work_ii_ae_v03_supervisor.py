"""Provider-free supervisor for the Work II A-E v0.3 development pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from chemworld.eval.provenance import file_sha256
from chemworld.eval.work_ii_ae_prior_qualification_v03 import (
    build_development_summary,
    build_phase_plan,
    select_screen_loci,
    validate_plan,
    validate_report,
)

PHASE_ORDER = (
    "classifier_fit",
    "classifier_validation",
    "prospective_screen",
    "confirmation",
)
SUPERVISOR_VERSION = "chemworld-work-ii-ae-v03-supervisor-0.1"


class AEV03SupervisorError(RuntimeError):
    """The local phase chain or supervisor state is unsafe to advance."""


@dataclass(frozen=True)
class PhaseEvidence:
    phase: str
    output: Path
    plan: dict[str, Any]
    receipts: list[dict[str, Any]]
    report: dict[str, Any]
    selection: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class SupervisorDecision:
    status: str
    phase: str
    reason: str
    next_phase: str | None = None
    next_output: Path | None = None
    command: tuple[str, ...] = ()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AEV03SupervisorError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AEV03SupervisorError(f"{label} must contain one JSON object")
    return value


def _load_receipts(output: Path) -> list[dict[str, Any]]:
    receipt_root = output / "receipts"
    if not receipt_root.is_dir():
        raise AEV03SupervisorError("terminal phase lacks a receipt directory")
    files = list(receipt_root.iterdir())
    if any(not path.is_file() or path.suffix != ".json" for path in files):
        raise AEV03SupervisorError("receipt directory contains unexpected members")
    try:
        files.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise AEV03SupervisorError("receipt filenames must be numeric indexes") from error
    if [int(path.stem) for path in files] != list(range(len(files))):
        raise AEV03SupervisorError("receipt indexes are not a complete prefix from zero")
    return [_load_object(path, "receipt") for path in files]


def _validate_trajectory_bindings(output: Path, receipts: list[dict[str, Any]]) -> None:
    resolved_output = output.resolve()
    bound_paths: set[Path] = set()
    for receipt in receipts:
        binding = receipt.get("trajectory")
        if not isinstance(binding, dict):
            raise AEV03SupervisorError("completed receipt lacks trajectory binding")
        relative = PurePosixPath(str(binding.get("path", "")))
        index = int(receipt.get("execution_index", -1))
        canonical = relative.parts == ("executions", str(index), "trajectory.jsonl")
        resumed = (
            len(relative.parts) == 4
            and relative.parts[0] == "resume-executions"
            and relative.parts[1].isdigit()
            and relative.parts[2] == str(index)
            and relative.parts[3] == "trajectory.jsonl"
        )
        if not canonical and not resumed:
            raise AEV03SupervisorError("trajectory path is outside registered layouts")
        trajectory = (resolved_output / Path(*relative.parts)).resolve()
        if not trajectory.is_relative_to(resolved_output) or not trajectory.is_file():
            raise AEV03SupervisorError("bound trajectory is missing or outside output")
        if file_sha256(trajectory) != binding.get("sha256"):
            raise AEV03SupervisorError("bound trajectory content has changed")
        bound_paths.add(trajectory)
    materialized = {
        path.resolve()
        for root_name in ("executions", "resume-executions")
        for path in (output / root_name).rglob("trajectory.jsonl")
        if path.is_file()
    }
    unbound = materialized - bound_paths
    if len(unbound) > 1:
        raise AEV03SupervisorError(
            "terminal output contains more than one unbound canonical trajectory"
        )
    receipt_bindings = {
        int(receipt["execution_index"]): PurePosixPath(str(receipt["trajectory"]["path"]))
        for receipt in receipts
    }
    for historical in unbound:
        relative = historical.relative_to(resolved_output)
        parts = relative.parts
        canonical = (
            len(parts) == 3
            and parts[0] == "executions"
            and parts[1].isdigit()
            and parts[2] == "trajectory.jsonl"
        )
        index = int(parts[1]) if canonical else -1
        receipt_path = receipt_bindings.get(index, PurePosixPath())
        receipt_is_resumed = (
            len(receipt_path.parts) == 4
            and receipt_path.parts[0] == "resume-executions"
            and receipt_path.parts[1].isdigit()
            and receipt_path.parts[2] == str(index)
            and receipt_path.parts[3] == "trajectory.jsonl"
        )
        if not canonical or not receipt_is_resumed:
            raise AEV03SupervisorError(
                "terminal output contains an unsafe unbound trajectory"
            )


def _phase_plan_kwargs(upstream: dict[str, PhaseEvidence]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    fit = upstream.get("classifier_fit")
    validation = upstream.get("classifier_validation")
    screen = upstream.get("prospective_screen")
    if fit is not None:
        kwargs.update(
            fit_report=fit.report,
            fit_plan=fit.plan,
            fit_receipts=fit.receipts,
        )
    if validation is not None:
        kwargs.update(
            validation_report=validation.report,
            validation_plan=validation.plan,
            validation_receipts=validation.receipts,
        )
    if screen is not None:
        kwargs.update(
            screen_report=screen.report,
            screen_plan=screen.plan,
            screen_receipts=screen.receipts,
            selection=screen.selection,
        )
    return kwargs


def validate_terminal_output(
    *,
    root: Path,
    contract_path: Path,
    output: Path,
    phase: str,
    upstream: dict[str, PhaseEvidence],
) -> PhaseEvidence:
    """Deeply rebuild one terminal phase from plan, receipts, and trajectories."""

    output = output.resolve()
    if phase not in PHASE_ORDER:
        raise AEV03SupervisorError(f"unknown phase: {phase}")
    if (output / "platform-failure.json").exists():
        raise AEV03SupervisorError(f"{phase} ended in platform failure")
    plan = _load_object(output / "plan.json", f"{phase} plan")
    report = _load_object(output / "report.json", f"{phase} report")
    receipts = _load_receipts(output)
    expected_plan = build_phase_plan(
        root,
        contract_path,
        phase,
        **_phase_plan_kwargs(upstream),
    )
    if plan != expected_plan:
        raise AEV03SupervisorError(f"{phase} plan differs from deterministic rebuild")
    plan_errors = validate_plan(plan)
    if plan_errors:
        raise AEV03SupervisorError(f"invalid {phase} plan: " + "; ".join(plan_errors))
    _validate_trajectory_bindings(output, receipts)
    contract = _load_object(contract_path, "A-E v0.3 contract")
    fit_report = upstream.get("classifier_fit")
    report_errors = validate_report(
        contract,
        plan,
        receipts,
        report,
        fit_report=fit_report.report if fit_report and phase != "classifier_fit" else None,
    )
    if report_errors:
        raise AEV03SupervisorError(
            f"invalid {phase} report: " + "; ".join(report_errors)
        )
    selection = None
    summary = None
    if phase == "prospective_screen":
        selection = _load_object(output / "selection.json", "screen selection")
        expected_selection = select_screen_loci(contract, report["locus_results"])
        if selection != expected_selection:
            raise AEV03SupervisorError("screen selection differs from frozen rule")
    elif phase == "confirmation":
        summary = _load_object(output / "summary.json", "confirmation summary")
        expected_summary = build_development_summary(
            contract,
            upstream["classifier_fit"].report,
            upstream["classifier_validation"].report,
            upstream["prospective_screen"].report,
            upstream["prospective_screen"].selection,
            report,
        )
        if summary != expected_summary:
            raise AEV03SupervisorError("confirmation summary differs from reconstruction")
    return PhaseEvidence(phase, output, plan, receipts, report, selection, summary)


def next_phase_command(
    *,
    python: str,
    runner: Path,
    contract_path: Path,
    phase: str,
    output: Path,
    upstream: dict[str, PhaseEvidence],
) -> tuple[str, ...]:
    command = [
        python,
        str(runner),
        "--contract",
        str(contract_path),
        "--phase",
        phase,
        "--output",
        str(output),
    ]
    fit = upstream.get("classifier_fit")
    validation = upstream.get("classifier_validation")
    screen = upstream.get("prospective_screen")
    if fit:
        command.extend(
            [
                "--fit-plan",
                str(fit.output / "plan.json"),
                "--fit-receipts",
                str(fit.output / "receipts"),
                "--fit-report",
                str(fit.output / "report.json"),
            ]
        )
    if validation:
        command.extend(
            [
                "--validation-plan",
                str(validation.output / "plan.json"),
                "--validation-receipts",
                str(validation.output / "receipts"),
                "--validation-report",
                str(validation.output / "report.json"),
            ]
        )
    if screen:
        command.extend(
            [
                "--screen-plan",
                str(screen.output / "plan.json"),
                "--screen-receipts",
                str(screen.output / "receipts"),
                "--screen-report",
                str(screen.output / "report.json"),
                "--selection",
                str(screen.output / "selection.json"),
            ]
        )
    return tuple(command)


def validate_external_root(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if resolved == root.resolve() or resolved.is_relative_to(root.resolve()):
        raise AEV03SupervisorError("supervisor outputs/logs must stay outside the repository")
    return resolved


__all__ = [
    "PHASE_ORDER",
    "SUPERVISOR_VERSION",
    "AEV03SupervisorError",
    "PhaseEvidence",
    "SupervisorDecision",
    "next_phase_command",
    "validate_external_root",
    "validate_terminal_output",
]
