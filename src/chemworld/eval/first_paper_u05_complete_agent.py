"""Frozen complete-agent instrument-use qualification for the first paper.

The sole new experimental unit is one ``InteractiveCodexExperimentAgent``
session in the first frozen unseen reaction--distillation composition.  U04 is
resolved as existing evidence and is never rerun here.  Every submitted action
is audited; sampling is not a qualification gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle
from chemworld.agents.base import HistoryRecord
from chemworld.agents.interactive_codex_experiment import (
    InteractiveCodexExperimentAgent,
    ProcessFactory,
)
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import load_jsonl, observation_to_json, to_builtin
from chemworld.eval.composition_qualification import _leakage_findings
from chemworld.eval.cross_world_infrastructure_qualification import (
    _post_termination_validation_receipt,
    _resource_receipt_summary,
)
from chemworld.eval.report_io import git_output as _git
from chemworld.eval.report_io import write_report_outputs
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records

REPORT_SCHEMA_VERSION = "chemworld-first-paper-agent-instrument-use-report-0.1"
QUALIFICATION_ID = "first-paper-agent-instrument-use-v1"
METHOD_ID = "first_paper_u05_interactive_codex_sol_medium_v1"

EXPERIMENT_NOTE = Path(
    "workstreams/arxiv_v1/experiments/first-paper-agent-instrument-use.md"
)
TODO_PATH = Path("workstreams/arxiv_v1/FIRST_PAPER_TODOLIST.md")
CURRENT_CONFIG = Path("configs/current.json")
EVALUATOR_SOURCE = Path("src/chemworld/eval/first_paper_u05_complete_agent.py")
RUNNER_SOURCE = Path("src/chemworld/eval/runner.py")
AGENT_SOURCE = Path("src/chemworld/agents/interactive_codex_experiment.py")
EXECUTION_SCRIPT = Path("scripts/run_first_paper_u05_complete_agent.py")

U04_NODE_ID = "work_i_world_fork_qualification"
U05_NODE_ID = "first_paper_composition_qualification"
FROZEN_COMPOSITION_ID = "qualification-reaction-distillation-observation-coverage-0001"
FROZEN_CASE_ID = "qualification-reaction-distillation-observation-case-0001"
FROZEN_PATTERN = "reaction-distillation-observation"
FROZEN_REQUEST_SHA256 = (
    "687007fb2fe9e7cb7bde1eff10219469fecc73903648d2fa34fec17c10694b4f"
)
FROZEN_RUNTIME_TASK_CONTRACT_HASH = (
    "9b775c56b1cfe07dc75afc355d4815077913b27cbeedf20d32fb21d9dadf9f14"
)
FROZEN_PUBLIC_TASK_SUBOBJECT_HASH = (
    "2d89a69f68d910dc8593a6ccfad698b108114a5295d18a4c362aad59155c497d"
)
FROZEN_GENERATION_SEED = 105
FROZEN_GENERATION_INDEX = 0
FROZEN_WORLD_SEED = 0
FROZEN_AGENT_SEED = 0

FROZEN_MODEL = "gpt-5.6-sol"
FROZEN_REASONING_EFFORT = "medium"
FROZEN_CODEX_CLI_VERSION = "codex-cli 0.145.0"
FROZEN_OPERATION_LIMIT = 16
FROZEN_COMPLETE_EXPERIMENT_LIMIT = 1
FROZEN_MODEL_CALL_LIMIT = 1
FROZEN_AUDITED_CUMULATIVE_INPUT_BASELINE = 517_000
FROZEN_CUMULATIVE_INPUT_HEADROOM = 123_000
FROZEN_INPUT_TOKEN_LIMIT = (
    FROZEN_AUDITED_CUMULATIVE_INPUT_BASELINE
    + FROZEN_CUMULATIVE_INPUT_HEADROOM
)
FROZEN_UNCACHED_INPUT_TOKEN_LIMIT = 192_000
FROZEN_OUTPUT_TOKEN_LIMIT = 64_000
FROZEN_REQUEST_TIMEOUT_S = 600.0
FROZEN_FINALIZATION_TIMEOUT_S = 300.0
FROZEN_WALL_TIME_RESERVE_S = 600.0
FROZEN_WALL_TIME_LIMIT_S = (
    FROZEN_OPERATION_LIMIT * FROZEN_REQUEST_TIMEOUT_S
    + FROZEN_FINALIZATION_TIMEOUT_S
    + FROZEN_WALL_TIME_RESERVE_S
)
FROZEN_PRE_ACTION_RESTART_LIMIT = 0
FROZEN_MAX_TOOL_OUTPUT_BYTES = 16_384
FROZEN_HISTORY_EVENT_LIMIT = 16
FROZEN_HISTORY_BYTE_LIMIT = 65_536

_EXPECTED_CLAIM = "Claim: Codex /root — U04/U05/E02-INSTRUMENT-USE — DOING"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

PreflightRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


class CompleteAgentQualificationError(ValueError):
    """Raised when a frozen input or launch binding fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompleteAgentQualificationError(f"JSON root must be an object: {path}")
    return payload


def _resolve_repository_path(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CompleteAgentQualificationError(
            f"repository binding escapes the repository root: {relative_path}"
        ) from exc
    return candidate


def _source_binding(root: Path, relative_path: str | Path) -> dict[str, Any]:
    path = _resolve_repository_path(root, relative_path)
    if not path.is_file():
        raise CompleteAgentQualificationError(f"bound source is missing: {relative_path}")
    return {
        "path": Path(relative_path).as_posix(),
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }


def validate_launch_preconditions(
    repository_root: str | Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Validate the frozen claim and bind the exact execution sources."""

    root = Path(repository_root).resolve()
    errors: list[str] = []
    try:
        git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompleteAgentQualificationError(
            "repository_root is not a Git worktree"
        ) from exc
    if git_root != root:
        errors.append(f"repository_root is not the Git root: {root}")

    note_path = _resolve_repository_path(root, EXPERIMENT_NOTE)
    todo_path = _resolve_repository_path(root, TODO_PATH)
    if not note_path.is_file():
        errors.append("frozen experiment note is missing")
    else:
        note = note_path.read_text(encoding="utf-8")
        required_note_fragments = (
            "FROZEN BEFORE DATA GENERATION",
            "Codex `/root`",
            METHOD_ID,
            FROZEN_COMPOSITION_ID,
            FROZEN_CASE_ID,
            FROZEN_REQUEST_SHA256,
            FROZEN_RUNTIME_TASK_CONTRACT_HASH,
            FROZEN_PUBLIC_TASK_SUBOBJECT_HASH,
            "generation seed 105",
            "generation index 0",
            "world seed 0",
            "pre-action restart limit 为 0",
            "环境操作上限 16",
                "provider session 与 logical Codex turn 上限各 1",
        )
        missing = [item for item in required_note_fragments if item not in note]
        if missing:
            errors.append("experiment note drifted: " + ", ".join(missing))
    if not todo_path.is_file():
        errors.append("first-paper TODO is missing")
    elif _EXPECTED_CLAIM not in todo_path.read_text(encoding="utf-8"):
        errors.append("active U04/U05/E02 claim is missing or not DOING")

    branch = _git(root, "branch", "--show-current")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and branch != "main":
        errors.append(f"formal execution requires main, found {branch or 'detached HEAD'}")
    if require_clean and dirty:
        errors.append("formal execution requires a clean worktree")
    if errors:
        raise CompleteAgentQualificationError("; ".join(errors))

    return {
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "branch": branch,
        "worktree_clean": not bool(dirty),
        "experiment_note": _source_binding(root, EXPERIMENT_NOTE),
        "todo": _source_binding(root, TODO_PATH),
        "evaluator": _source_binding(root, EVALUATOR_SOURCE),
        "runner": _source_binding(root, RUNNER_SOURCE),
        "interactive_agent": _source_binding(root, AGENT_SOURCE),
        "execution_script": _source_binding(root, EXECUTION_SCRIPT),
    }


def _current_node_binding(
    root: Path,
    current: Mapping[str, Any],
    node_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dag = current.get("evidence_dag")
    nodes = dag.get("nodes") if isinstance(dag, Mapping) else None
    node = nodes.get(node_id) if isinstance(nodes, Mapping) else None
    if not isinstance(node, Mapping):
        raise CompleteAgentQualificationError(f"current evidence node is missing: {node_id}")
    for field, expected in (
        ("artifact_state", "current"),
        ("freshness", "fresh"),
        ("gate_state", "passed"),
    ):
        if node.get(field) != expected:
            raise CompleteAgentQualificationError(
                f"current evidence node {node_id} has {field}={node.get(field)!r}"
            )
    relative_path = node.get("path")
    expected_sha = node.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
        raise CompleteAgentQualificationError(
            f"current evidence node {node_id} has an incomplete path/SHA binding"
        )
    path = _resolve_repository_path(root, relative_path)
    if not path.is_file():
        raise CompleteAgentQualificationError(
            f"current evidence artifact is missing for {node_id}: {relative_path}"
        )
    actual_sha = _sha256_path(path)
    if actual_sha != expected_sha:
        raise CompleteAgentQualificationError(
            f"current evidence SHA drifted for {node_id}: {actual_sha} != {expected_sha}"
        )
    return (
        {
            "node_id": node_id,
            "path": relative_path,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "artifact_state": node["artifact_state"],
            "freshness": node["freshness"],
            "gate_state": node["gate_state"],
            "binding_verified": True,
        },
        _read_json_object(path),
    )


def resolve_existing_evidence(repository_root: str | Path) -> dict[str, Any]:
    """Resolve U04 and U05 only through ``configs/current.json``."""

    root = Path(repository_root).resolve()
    current_path = _resolve_repository_path(root, CURRENT_CONFIG)
    current = _read_json_object(current_path)
    u04_binding, u04_report = _current_node_binding(root, current, U04_NODE_ID)
    u05_binding, u05_report = _current_node_binding(root, current, U05_NODE_ID)

    if not (
        u04_report.get("passed") is True
        and int(u04_report.get("pair_count", -1)) > 0
        and int(u04_report.get("trace_count", -1)) > 0
        and int(u04_report.get("provider_call_count", -1)) == 0
    ):
        raise CompleteAgentQualificationError("current U04 fork evidence is not passing")

    generated = u05_report.get("generated_qualification")
    cases = generated.get("cases") if isinstance(generated, Mapping) else None
    if not isinstance(generated, Mapping) or not isinstance(cases, list):
        raise CompleteAgentQualificationError(
            "current composition qualification lacks generated cases"
        )
    unseen_pattern = generated.get("unseen_pattern")
    unseen_cases = [
        case
        for case in cases
        if isinstance(case, Mapping) and case.get("pattern") == unseen_pattern
    ]
    if unseen_pattern != FROZEN_PATTERN or not unseen_cases:
        raise CompleteAgentQualificationError("current unseen composition pattern drifted")
    first = unseen_cases[0]
    identity_checks = {
        "composition_id": FROZEN_COMPOSITION_ID,
        "case_id": FROZEN_CASE_ID,
        "generation_seed": FROZEN_GENERATION_SEED,
        "generation_index": FROZEN_GENERATION_INDEX,
        "composition_request_sha256": FROZEN_REQUEST_SHA256,
    }
    drifted = {
        field: {"expected": expected, "observed": first.get(field)}
        for field, expected in identity_checks.items()
        if first.get(field) != expected
    }
    request = first.get("composition_request")
    if not isinstance(request, Mapping):
        drifted["composition_request"] = {
            "expected": "object",
            "observed": type(request).__name__,
        }
        normalized_request: dict[str, Any] = {}
    else:
        normalized_request = copy.deepcopy(dict(request))
        actual_request_sha = _sha256_value(normalized_request)
        if actual_request_sha != FROZEN_REQUEST_SHA256:
            drifted["actual_composition_request_sha256"] = {
                "expected": FROZEN_REQUEST_SHA256,
                "observed": actual_request_sha,
            }
    if u05_report.get("status") != "passed":
        drifted["qualification_status"] = {
            "expected": "passed",
            "observed": u05_report.get("status"),
        }
    if first.get("passed") is not True:
        drifted["case_passed"] = {"expected": True, "observed": first.get("passed")}
    replay = first.get("exact_replay")
    if not isinstance(replay, Mapping) or replay.get("verified") is not True:
        drifted["case_exact_replay"] = {
            "expected": True,
            "observed": replay.get("verified") if isinstance(replay, Mapping) else None,
        }
    compile_receipt = first.get("compile_receipt")
    observed_public_task_hash = (
        compile_receipt.get("task_contract_sha256")
        if isinstance(compile_receipt, Mapping)
        else None
    )
    if observed_public_task_hash != FROZEN_PUBLIC_TASK_SUBOBJECT_HASH:
        drifted["public_compiled_task_subobject_hash"] = {
            "expected": FROZEN_PUBLIC_TASK_SUBOBJECT_HASH,
            "observed": observed_public_task_hash,
        }
    if drifted:
        raise CompleteAgentQualificationError(
            "current frozen U05 first unseen case drifted: "
            + json.dumps(drifted, ensure_ascii=False, sort_keys=True)
        )

    return {
        "current_registry": {
            "path": CURRENT_CONFIG.as_posix(),
            "sha256": _sha256_path(current_path),
        },
        "U04": {
            "evidence_role": "single-private-component controlled world forks",
            "binding": u04_binding,
            "passed": True,
            "pair_count": int(u04_report["pair_count"]),
            "trace_count": int(u04_report["trace_count"]),
            "provider_call_count": int(u04_report["provider_call_count"]),
            "protocol_id": u04_report.get("protocol_id"),
        },
        "U05": {
            "evidence_role": "first row of the frozen unseen generated composition batch",
            "binding": u05_binding,
            "passed": True,
            "case_id": FROZEN_CASE_ID,
            "composition_id": FROZEN_COMPOSITION_ID,
            "pattern": FROZEN_PATTERN,
            "composition_request_sha256": FROZEN_REQUEST_SHA256,
            "generation_seed": FROZEN_GENERATION_SEED,
            "generation_index": FROZEN_GENERATION_INDEX,
            "source_action_count": int(first.get("action_count", 0)),
            "source_exact_replay_verified": True,
            "public_compiled_task_subobject_hash": observed_public_task_hash,
            "composition_request": normalized_request,
        },
    }


def _default_preflight_runner(
    command: Sequence[str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )


def _command_receipt(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    return {
        "return_code": int(completed.returncode),
        "stdout_byte_count": len(stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": hashlib.sha256(
            stdout.encode("utf-8", errors="replace")
        ).hexdigest(),
        "stderr_byte_count": len(stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": hashlib.sha256(
            stderr.encode("utf-8", errors="replace")
        ).hexdigest(),
        "body_retained": False,
    }


def collect_provider_preflight(
    repository_root: str | Path,
    *,
    codex_executable: str | None = None,
    runner: PreflightRunner | None = None,
) -> dict[str, Any]:
    """Check the frozen Codex CLI version/login without retaining command output."""

    root = Path(repository_root).resolve()
    executable = codex_executable or shutil.which("codex")
    if not executable:
        raise CompleteAgentQualificationError("Codex CLI is not available on PATH")
    command_runner = runner or _default_preflight_runner
    try:
        version_result = command_runner([str(executable), "--version"], root)
        login_result = command_runner([str(executable), "login", "status"], root)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompleteAgentQualificationError("Codex CLI preflight could not run") from exc
    version_text = "\n".join(
        value
        for value in (version_result.stdout, version_result.stderr)
        if isinstance(value, str) and value.strip()
    ).strip()
    observed_version = version_text.splitlines()[0].strip() if version_text else ""
    version_matches = (
        version_result.returncode == 0 and observed_version == FROZEN_CODEX_CLI_VERSION
    )
    login_passed = login_result.returncode == 0
    receipt = {
        "schema_version": "chemworld-first-paper-codex-preflight-0.1",
        "expected_cli_version": FROZEN_CODEX_CLI_VERSION,
        "observed_cli_version": observed_version,
        "cli_version_matches": version_matches,
        "cached_chatgpt_login_status": "passed" if login_passed else "failed",
        "version_command": _command_receipt(version_result),
        "login_status_command": _command_receipt(login_result),
        "verified": bool(version_matches and login_passed),
    }
    if receipt["verified"] is not True:
        raise CompleteAgentQualificationError("frozen Codex CLI preflight failed")
    return receipt


def _validate_injected_provider_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(dict(value))
    if not (
        receipt.get("expected_cli_version") == FROZEN_CODEX_CLI_VERSION
        and receipt.get("observed_cli_version") == FROZEN_CODEX_CLI_VERSION
        and receipt.get("cli_version_matches") is True
        and receipt.get("cached_chatgpt_login_status") == "passed"
        and receipt.get("verified") is True
    ):
        raise CompleteAgentQualificationError("injected provider preflight is not verified")
    return receipt


def _campaign_resource_card(request: Mapping[str, Any]) -> dict[str, Any]:
    task = request.get("task")
    resources = task.get("resources") if isinstance(task, Mapping) else None
    if not isinstance(resources, Mapping):
        raise CompleteAgentQualificationError("frozen composition task resources are missing")
    operation_limit = int(resources.get("operation_budget", -1))
    final_assays = int(resources.get("final_assays", -1))
    instrument_uses = int(resources.get("instrument_uses", -1))
    if (
        operation_limit != FROZEN_OPERATION_LIMIT
        or final_assays != 1
        or instrument_uses < final_assays
    ):
        raise CompleteAgentQualificationError("frozen composition resource budget drifted")
    card = CampaignResourceCard(
        card_id="first-paper-u05-complete-agent",
        operation_attempt_limit=operation_limit,
        vessel_start_limit=1,
        final_assay_limit=final_assays,
        nonfinal_instrument_use_limit=instrument_uses - final_assays,
        stock_limits={
            "reagent_mol": operation_limit * 0.04,
            "solvent_L": operation_limit * 0.08,
            "catalyst_mol": operation_limit * 0.005,
        },
        per_instrument_limits={
            "hplc": instrument_uses - final_assays,
            "gc": instrument_uses - final_assays,
        },
        metadata={
            "qualification_id": QUALIFICATION_ID,
            "composition_id": FROZEN_COMPOSITION_ID,
            "world_seed": FROZEN_WORLD_SEED,
            "scope": "single_complete_agent_lifecycle",
        },
    )
    return card.to_dict()


def _method_resource_limits() -> dict[str, Any]:
    return {
        "operation_limit": FROZEN_OPERATION_LIMIT,
        "complete_experiment_limit": FROZEN_COMPLETE_EXPERIMENT_LIMIT,
        "wall_time_limit_s": FROZEN_WALL_TIME_LIMIT_S,
        "model_call_limit": FROZEN_MODEL_CALL_LIMIT,
        "input_token_limit": FROZEN_INPUT_TOKEN_LIMIT,
        "uncached_input_token_limit": FROZEN_UNCACHED_INPUT_TOKEN_LIMIT,
        "output_token_limit": FROZEN_OUTPUT_TOKEN_LIMIT,
        "monetary_cost_limit_usd": None,
        "training_environment_step_limit": 0,
        "cpu_time_limit_s": None,
        "gpu_time_limit_s": None,
        "checkpoint_complete_experiments": [1],
    }


def _method_resource_policy() -> dict[str, Any]:
    return {
        "wall_time": {
            "formula": (
                "operation_limit * request_timeout_s + "
                "finalization_timeout_s + reserve_s"
            ),
            "operation_limit": FROZEN_OPERATION_LIMIT,
            "request_timeout_s": FROZEN_REQUEST_TIMEOUT_S,
            "finalization_timeout_s": FROZEN_FINALIZATION_TIMEOUT_S,
            "reserve_s": FROZEN_WALL_TIME_RESERVE_S,
            "limit_s": FROZEN_WALL_TIME_LIMIT_S,
        },
        "input_tokens": {
            "cumulative_input_semantics": (
                "complete persistent Codex turn, including cache-hit context "
                "recounted across backend responses"
            ),
            "audited_cumulative_input_baseline": (
                FROZEN_AUDITED_CUMULATIVE_INPUT_BASELINE
            ),
            "cumulative_input_headroom": FROZEN_CUMULATIVE_INPUT_HEADROOM,
            "cumulative_input_limit": FROZEN_INPUT_TOKEN_LIMIT,
            "uncached_input_limit": FROZEN_UNCACHED_INPUT_TOKEN_LIMIT,
            "cache_hit_is_reused_input_not_repeated_output": True,
        },
        "mcp_context": {
            "max_tool_output_bytes": FROZEN_MAX_TOOL_OUTPUT_BYTES,
            "history_event_limit": FROZEN_HISTORY_EVENT_LIMIT,
            "history_byte_limit": FROZEN_HISTORY_BYTE_LIMIT,
        },
    }


def _declared_resource_limits(
    composition_request: Mapping[str, Any],
) -> dict[str, float | int]:
    task = composition_request.get("task")
    resources = task.get("resources") if isinstance(task, Mapping) else None
    if not isinstance(resources, Mapping):
        raise CompleteAgentQualificationError(
            "frozen composition declared resources are missing"
        )
    limits: dict[str, float | int] = {
        "operation_attempts": int(resources.get("operation_budget", -1)),
        "instrument_uses": int(resources.get("instrument_uses", -1)),
        "final_assays": int(resources.get("final_assays", -1)),
        "sample_consumed_L": float(resources.get("sample_volume_L", -1.0)),
        "process_time_s": float(resources.get("time_s", -1.0)),
    }
    if any(float(value) < 0.0 for value in limits.values()):
        raise CompleteAgentQualificationError(
            "frozen composition declared resource limits are invalid"
        )
    return limits


def _declared_resource_budget_receipt(
    *,
    composition_request: Mapping[str, Any],
    environment_resource_receipt: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    limits = _declared_resource_limits(composition_request)
    outcome = environment_resource_receipt.get("outcome_delta")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    report_only = outcome.get("report_only")
    report_only = report_only if isinstance(report_only, Mapping) else {}
    observed = {
        "operation_attempts": int(outcome.get("operation_attempts", -1)),
        "instrument_uses": int(outcome.get("nonfinal_instrument_uses", -1))
        + int(outcome.get("final_assays", -1)),
        "final_assays": int(outcome.get("final_assays", -1)),
        "sample_consumed_L": float(report_only.get("sample_consumed_L", -1.0)),
        "process_time_s": float(report_only.get("process_time_s", -1.0)),
    }
    tolerance = 1.0e-12
    checks = {
        key: observed[key] >= 0.0 and observed[key] <= float(limit) + tolerance
        for key, limit in limits.items()
    }
    exceeded = [key for key, passed in checks.items() if not passed]
    cumulative: dict[str, float] = {
        "operation_attempts": 0.0,
        "instrument_uses": 0.0,
        "final_assays": 0.0,
        "sample_consumed_L": 0.0,
        "process_time_s": 0.0,
    }
    first_exceeded_step: dict[str, int] = {}
    for index, action_receipt in enumerate(actions or (), start=1):
        delta = action_receipt.get("resource_outcome_delta")
        delta = delta if isinstance(delta, Mapping) else {}
        report_delta = delta.get("report_only")
        report_delta = report_delta if isinstance(report_delta, Mapping) else {}
        cumulative["operation_attempts"] += float(delta.get("operation_attempts", 0))
        cumulative["instrument_uses"] += float(
            int(delta.get("nonfinal_instrument_uses", 0))
            + int(delta.get("final_assays", 0))
        )
        cumulative["final_assays"] += float(delta.get("final_assays", 0))
        cumulative["sample_consumed_L"] += float(
            report_delta.get("sample_consumed_L", 0.0)
        )
        cumulative["process_time_s"] += float(
            report_delta.get("process_time_s", 0.0)
        )
        for key, limit in limits.items():
            if (
                key not in first_exceeded_step
                and cumulative[key] > float(limit) + tolerance
            ):
                first_exceeded_step[key] = index
    return {
        "schema_version": "chemworld-first-paper-declared-resource-budget-0.1",
        "declared_limits": limits,
        "observed_usage": observed,
        "checks": checks,
        "exceeded_resources": exceeded,
        "checked_action_count": len(actions or ()),
        "first_exceeded_step": first_exceeded_step,
        "passed": not exceeded,
    }


def _runtime_contract_binding(
    *,
    composition_request: Mapping[str, Any],
    campaign_card: Mapping[str, Any],
) -> dict[str, Any]:
    env = gym.make(
        "ChemWorld",
        composition=copy.deepcopy(dict(composition_request)),
        seed=FROZEN_WORLD_SEED,
        campaign_resource_card=copy.deepcopy(dict(campaign_card)),
    )
    try:
        env.reset(seed=FROZEN_WORLD_SEED)
        base: Any = env.unwrapped
        task_info = base.task_info()
    finally:
        env.close()
    observed = task_info.get("task_contract_hash")
    if observed != FROZEN_RUNTIME_TASK_CONTRACT_HASH:
        raise CompleteAgentQualificationError(
            "frozen runtime task contract hash drifted: "
            f"{observed} != {FROZEN_RUNTIME_TASK_CONTRACT_HASH}"
        )
    required_hashes = {
        key: task_info.get(key)
        for key in (
            "runtime_profile_hash",
            "scoring_contract_hash",
            "observation_contract_hash",
        )
    }
    missing = [key for key, value in required_hashes.items() if not isinstance(value, str)]
    if missing:
        raise CompleteAgentQualificationError(
            "runtime contract binding is missing hashes: " + ", ".join(missing)
        )
    return {
        "task_id": task_info.get("task_id"),
        "task_contract_hash": observed,
        "expected_task_contract_hash": FROZEN_RUNTIME_TASK_CONTRACT_HASH,
        "task_contract_hash_matches": True,
        **required_hashes,
    }


def _history_payload(record: HistoryRecord, trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **to_builtin(asdict(record)),
        "agent_trace": to_builtin(trace),
    }


def _failure(failure_class: str, **details: Any) -> dict[str, Any]:
    return {"class": failure_class, **to_builtin(details)}


def _exception_receipt(error: BaseException) -> dict[str, Any]:
    message = str(error)
    encoded = message.encode("utf-8", errors="replace")
    return {
        "exception_type": type(error).__name__,
        "message_byte_count": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "message_body_retained": False,
    }


def _sanitize_provider_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema_version",
        "session_id",
        "thread_id",
        "status",
        "return_code",
        "terminal_reason",
        "failure_type",
        "model_id",
        "reasoning_effort",
        "usage",
        "usage_complete",
        "prompt_byte_count",
        "prompt_sha256",
        "tool_events",
        "event_counts",
        "provider_errors",
        "final_payload_valid",
        "final_payload_status",
        "stderr_byte_count",
        "stderr_sha256",
        "artifact_access",
        "mcp_tool_calls",
        "experiment_tool_transport",
        "mcp_tool_integrity_verified_after_session",
        "experiment_tool_integrity_verified_after_session",
        "lab_tool_integrity_verified_after_session",
        "private_reasoning_retained",
    )
    return {
        key: copy.deepcopy(receipt[key])
        for key in allowed
        if key in receipt
    }


def _sanitize_method_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(dict(usage))
    sanitized.pop("monetary_cost_usd", None)
    provenance = sanitized.get("model_provenance")
    if isinstance(provenance, dict):
        provenance.pop("provider_base_url", None)
        pricing = provenance.get("pricing")
        if isinstance(pricing, dict):
            pricing["reported_monetary_cost"] = "unavailable_not_zero"
    return sanitized


def _trace_for_action(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trace = record.get("agent_trace")
    if not isinstance(trace, list):
        return None
    candidates = [
        row
        for row in trace
        if isinstance(row, Mapping) and row.get("status") == "codex_lab_tool_action"
    ]
    return candidates[-1] if candidates else None


def _mcp_step_binding(
    *,
    audit: Mapping[str, Any],
    step: int,
    action: Mapping[str, Any],
    request_id: str,
) -> dict[str, Any]:
    keys = audit.get("argument_keys")
    expected_payload: dict[str, Any] = {"expected_step": step, "action": dict(action)}
    if isinstance(keys, list) and "request_id" in keys:
        expected_payload["request_id"] = request_id
    expected_sha = _sha256_value(expected_payload)
    observed_sha = audit.get("arguments_sha256")
    return {
        "tool": audit.get("tool"),
        "argument_keys": copy.deepcopy(keys) if isinstance(keys, list) else [],
        "expected_arguments_sha256": expected_sha,
        "observed_arguments_sha256": observed_sha,
        "verified": audit.get("tool") == "step" and observed_sha == expected_sha,
    }


def _public_input_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    context = record.get("decision_context")
    public_view = record.get("public_view")
    return {
        "decision_context_sha256": _sha256_value(context),
        "public_view_sha256": _sha256_value(public_view),
        "remaining_operations": (
            context.get("remaining_operations") if isinstance(context, Mapping) else None
        ),
        "available_operation_count": (
            len(context.get("available_operations", []))
            if isinstance(context, Mapping)
            and isinstance(context.get("available_operations"), list)
            else None
        ),
    }


def _available_operations(public_view: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    tool = public_view.get("tool_json")
    available = tool.get("available_actions") if isinstance(tool, Mapping) else None
    operations: set[str] = set()
    instruments: set[str] = set()
    for raw in available if isinstance(available, list) else ():
        if not isinstance(raw, Mapping):
            continue
        operation = raw.get("operation")
        if isinstance(operation, str):
            operations.add(operation)
        if operation != "measure":
            continue
        fields = raw.get("fields")
        schema = raw.get("schema")
        if not isinstance(fields, list) and isinstance(schema, Mapping):
            fields = schema.get("fields")
        for field in fields if isinstance(fields, list) else ():
            if not isinstance(field, Mapping) or field.get("field") != "instrument":
                continue
            choices = field.get("choices")
            if isinstance(choices, list):
                instruments.update(str(choice) for choice in choices)
        compact_instrument = raw.get("instrument")
        if isinstance(compact_instrument, Mapping):
            choices = compact_instrument.get("choices")
            if isinstance(choices, list):
                instruments.update(str(choice) for choice in choices)
    return operations, instruments


def _remaining_monitor_resources(public_view: Mapping[str, Any]) -> dict[str, Any]:
    tool = public_view.get("tool_json")
    campaign = tool.get("campaign_state") if isinstance(tool, Mapping) else None
    remaining_budget = (
        campaign.get("remaining_budget") if isinstance(campaign, Mapping) else None
    )
    raw_resources = (
        campaign.get("campaign_resources") if isinstance(campaign, Mapping) else None
    )
    state = raw_resources.get("state") if isinstance(raw_resources, Mapping) else None
    remaining = state.get("remaining") if isinstance(state, Mapping) else None
    return {
        "remaining_budget": remaining_budget,
        "operation_attempts": (
            remaining.get("operation_attempts") if isinstance(remaining, Mapping) else None
        ),
        "final_assays": (
            remaining.get("final_assays") if isinstance(remaining, Mapping) else None
        ),
        "nonfinal_instrument_uses": (
            remaining.get("nonfinal_instrument_uses")
            if isinstance(remaining, Mapping)
            else None
        ),
    }


class _StepFailFastMonitor:
    """Audit every committed action immediately after the runner records it."""

    def __init__(
        self,
        *,
        composition_request: Mapping[str, Any],
        campaign_card: Mapping[str, Any],
        require_agent_trace: bool = False,
    ) -> None:
        self._env = gym.make(
            "ChemWorld",
            world_split="public-test",
            budget=FROZEN_OPERATION_LIMIT,
            objective="balanced",
            composition=copy.deepcopy(dict(composition_request)),
            seed=FROZEN_WORLD_SEED,
            campaign_resource_card=copy.deepcopy(dict(campaign_card)),
        )
        self._env.reset(seed=FROZEN_WORLD_SEED)
        self._expected_step = 1
        self._terminate_count = 0
        self._final_assay_count = 0
        self._require_agent_trace = require_agent_trace
        self._session_id: str | None = None
        self._declared_limits = _declared_resource_limits(composition_request)
        self._observed_usage = {
            "operation_attempts": 0,
            "instrument_uses": 0,
            "final_assays": 0,
            "sample_consumed_L": 0.0,
            "process_time_s": 0.0,
        }
        self.events: list[dict[str, Any]] = []

    def close(self) -> None:
        self._env.close()

    def observe(
        self,
        record: HistoryRecord,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        step = self._expected_step
        failures: list[str] = []
        action = record.action
        info = record.info
        base: Any = self._env.unwrapped
        validation = base.validate_action(action)
        shadow_observation, _reward, terminated, truncated, replay_info = self._env.step(
            action
        )
        transaction = info.get("transaction_status")
        preflight = info.get("campaign_resource_preflight")
        outcome = info.get("campaign_resource_outcome_delta")
        checks = info.get("constitution_checks")
        checks = checks if isinstance(checks, list) else []
        events = info.get("world_events")
        events = events if isinstance(events, list) else []

        if record.step != step:
            failures.append("step_not_contiguous")
        if validation.get("valid") is not True:
            failures.append("schema_or_precondition_invalid")
        if replay_info.get("transaction_status") != "committed":
            failures.append("shadow_transaction_not_committed")
        if transaction != replay_info.get("transaction_status"):
            failures.append("transaction_shadow_mismatch")
        if transaction != "committed":
            failures.append("transaction_not_committed")
        if info.get("rollback_reason") is not None:
            failures.append("rollback_present")
        if not isinstance(preflight, Mapping) or not isinstance(outcome, Mapping):
            failures.append("campaign_resource_receipt_missing")
        elif (
            preflight.get("allowed") is not True
            or preflight.get("attempt_charged") is not True
            or bool(preflight.get("rejection_reasons"))
        ):
            failures.append("campaign_resource_rejected")
        recorded_preflight_semantics = (
            {
                key: copy.deepcopy(preflight.get(key))
                for key in (
                    "allowed",
                    "attempt_charged",
                    "rejection_reasons",
                    "proposed_delta",
                )
            }
            if isinstance(preflight, Mapping)
            else None
        )
        replay_preflight = replay_info.get("campaign_resource_preflight")
        replay_preflight_semantics = (
            {
                key: copy.deepcopy(replay_preflight.get(key))
                for key in (
                    "allowed",
                    "attempt_charged",
                    "rejection_reasons",
                    "proposed_delta",
                )
            }
            if isinstance(replay_preflight, Mapping)
            else None
        )
        if recorded_preflight_semantics != replay_preflight_semantics:
            failures.append("campaign_resource_preflight_shadow_mismatch")
        if outcome != replay_info.get("campaign_resource_outcome_delta"):
            failures.append("campaign_resource_outcome_shadow_mismatch")
        if isinstance(outcome, Mapping):
            report_only = outcome.get("report_only")
            report_only = report_only if isinstance(report_only, Mapping) else {}
            self._observed_usage["operation_attempts"] += int(
                outcome.get("operation_attempts", 0)
            )
            self._observed_usage["instrument_uses"] += int(
                outcome.get("nonfinal_instrument_uses", 0)
            ) + int(outcome.get("final_assays", 0))
            self._observed_usage["final_assays"] += int(
                outcome.get("final_assays", 0)
            )
            self._observed_usage["sample_consumed_L"] += float(
                report_only.get("sample_consumed_L", 0.0)
            )
            self._observed_usage["process_time_s"] += float(
                report_only.get("process_time_s", 0.0)
            )
        for key, limit in self._declared_limits.items():
            if float(self._observed_usage[key]) > float(limit) + 1.0e-12:
                failures.append(f"declared_resource_exceeded:{key}")
        if any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks
        ):
            failures.append("constitution_check_failed")
        event_matches = any(
            isinstance(event, Mapping)
            and event.get("operation_type") == action.get("operation")
            and event.get("event_type") == "operation_applied"
            for event in events
        )
        if not event_matches:
            failures.append("world_event_mismatch")
        if checks != replay_info.get("constitution_checks"):
            failures.append("constitution_checks_shadow_mismatch")
        if events != replay_info.get("world_events"):
            failures.append("world_events_shadow_mismatch")
        if record.observation != observation_to_json(shadow_observation):
            failures.append("public_observation_shadow_mismatch")
        if int(record.method_resources.get("operation_count", -1)) != step:
            failures.append("method_operation_count_mismatch")

        action_trace = _trace_for_action({"agent_trace": trace or []})
        if self._require_agent_trace:
            trace_session = (
                action_trace.get("session_id")
                if isinstance(action_trace, Mapping)
                else None
            )
            trace_valid = bool(
                isinstance(action_trace, Mapping)
                and isinstance(trace_session, str)
                and isinstance(action_trace.get("request_id"), str)
                and action_trace.get("expected_step") == step
                and action_trace.get("action") == action
                and action_trace.get("action_payload_sha256") == _sha256_value(action)
            )
            if not trace_valid:
                failures.append("agent_session_action_trace_mismatch")
            elif self._session_id is None:
                self._session_id = trace_session
            elif trace_session != self._session_id:
                failures.append("agent_session_changed_mid_lifecycle")

        if action.get("operation") == "terminate" and transaction == "committed":
            self._terminate_count += 1
        if (
            action.get("operation") == "measure"
            and action.get("instrument") == "final_assay"
            and transaction == "committed"
        ):
            self._final_assay_count += 1

        operations, instruments = _available_operations(record.public_view)
        remaining = _remaining_monitor_resources(record.public_view)
        budget_remaining = remaining.get("remaining_budget")
        attempts_remaining = remaining.get("operation_attempts")
        final_assays_remaining = remaining.get("final_assays")
        complete = self._final_assay_count == 1 and bool(terminated) and not bool(truncated)
        if complete:
            closure_reserve = 0
        elif self._terminate_count > 0:
            closure_reserve = 1
            if "final_assay" not in instruments:
                failures.append("final_assay_not_immediately_reachable")
        else:
            terminate_reachable = "terminate" in operations
            setup_reachable = bool(
                operations.intersection({"add_reagent", "add_catalyst", "add_solvent"})
            )
            closure_reserve = 2 if terminate_reachable else 3
            if not terminate_reachable and not setup_reachable:
                failures.append("terminate_setup_not_reachable")
        for label, observed in (
            ("remaining_budget", budget_remaining),
            ("remaining_operation_attempts", attempts_remaining),
        ):
            if not isinstance(observed, int) or isinstance(observed, bool):
                failures.append(f"{label}_missing")
            elif observed < closure_reserve:
                failures.append(f"{label}_below_closeout_reserve")
        if not complete and (
            not isinstance(final_assays_remaining, int)
            or isinstance(final_assays_remaining, bool)
            or final_assays_remaining < 1
        ):
            failures.append("final_assay_resource_not_reserved")

        event = {
            "schema_version": "chemworld-first-paper-step-monitor-0.1",
            "qualification_id": QUALIFICATION_ID,
            "step": step,
            "operation": action.get("operation"),
            "instrument": action.get("instrument"),
            "transaction_status": transaction,
            "remaining_operations": {
                "runner_budget": budget_remaining,
                "campaign_attempts": attempts_remaining,
                "closure_reserve_required": closure_reserve,
            },
            "remaining_resources": {
                "final_assays": final_assays_remaining,
                "nonfinal_instrument_uses": remaining.get(
                    "nonfinal_instrument_uses"
                ),
            },
            "declared_resource_budget": {
                "limits": copy.deepcopy(self._declared_limits),
                "observed_usage": copy.deepcopy(self._observed_usage),
            },
            "lifecycle": {
                "terminate_count": self._terminate_count,
                "final_assay_count": self._final_assay_count,
                "complete": complete,
            },
            "agent_session_id": self._session_id,
            "status": "passed" if not failures else "failed",
            "failures": failures,
        }
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
        self._expected_step += 1
        if failures:
            raise CompleteAgentQualificationError(
                f"fail-fast step monitor rejected recorded action {step}: "
                + ", ".join(failures)
            )


def _audit_records(
    *,
    records: list[dict[str, Any]],
    histories: list[dict[str, Any]],
    composition_request: dict[str, Any],
    campaign_card: dict[str, Any],
    provider_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    action_receipts: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    resource_steps: list[dict[str, Any]] = []
    termination_probe: dict[str, Any] | None = None
    evaluation_receipt: dict[str, Any] | None = None

    sanitized_sessions = [_sanitize_provider_receipt(row) for row in provider_receipts]
    session = sanitized_sessions[0] if len(sanitized_sessions) == 1 else {}
    session_id = session.get("session_id") if isinstance(session.get("session_id"), str) else None
    raw_mcp = session.get("mcp_tool_calls")
    mcp_calls = raw_mcp if isinstance(raw_mcp, list) else []
    mcp_steps = [row for row in mcp_calls if isinstance(row, Mapping) and row.get("tool") == "step"]

    if len(records) != len(histories):
        failures.append(
            _failure(
                "trajectory_history_count_mismatch",
                trajectory_record_count=len(records),
                history_record_count=len(histories),
            )
        )

    env = gym.make(
        "ChemWorld",
        world_split="public-test",
        budget=FROZEN_OPERATION_LIMIT,
        objective="balanced",
        composition=copy.deepcopy(composition_request),
        seed=FROZEN_WORLD_SEED,
        campaign_resource_card=copy.deepcopy(campaign_card),
    )
    try:
        initial_observation, reset_info = env.reset(seed=FROZEN_WORLD_SEED)
        base: Any = env.unwrapped
        resource_before = base.campaign_resource_snapshot()
        leakage.extend(_leakage_findings(env, reset_info, "reset_info"))
        leakage.extend(
            _leakage_findings(
                env,
                agent_view_bundle(env, initial_observation, {}),
                "initial_agent_view",
            )
        )
        actions: list[dict[str, Any]] = []
        for index, history in enumerate(histories, start=1):
            action_failures: list[dict[str, Any]] = []
            action = history.get("action")
            action = copy.deepcopy(dict(action)) if isinstance(action, Mapping) else {}
            actions.append(action)
            validation = base.validate_action(action)
            step_resource_before = base.campaign_resource_snapshot()
            _observation, reward, terminated, truncated, info = env.step(action)
            step_resource_after = base.campaign_resource_snapshot()
            public_resource = base.public_campaign_resource_state()
            transaction_status = info.get("transaction_status")
            committed = transaction_status == "committed"
            recorded_preflight = history.get("info", {}).get("campaign_resource_preflight")
            recorded_outcome = history.get("info", {}).get("campaign_resource_outcome_delta")
            replay_preflight = info.get("campaign_resource_preflight")
            replay_outcome = info.get("campaign_resource_outcome_delta")
            resource_step = {
                "step": index,
                "operation": action.get("operation"),
                "instrument": action.get("instrument"),
                "transaction_status": transaction_status,
                "operation_committed": committed,
                "preflight": copy.deepcopy(recorded_preflight),
                "outcome_delta": copy.deepcopy(recorded_outcome),
            }
            resource_steps.append(resource_step)
            step_resource = _resource_receipt_summary(
                [resource_step],
                before_snapshot=step_resource_before,
                after_snapshot=step_resource_after,
                public_state=public_resource,
            )
            step_resource.pop("step_receipts", None)

            if history.get("step") != index:
                action_failures.append(
                    _failure(
                        "noncontiguous_step",
                        step=index,
                        observed=history.get("step"),
                    )
                )
            if validation.get("valid") is not True:
                action_failures.append(
                    _failure(
                        "schema_or_precondition_invalid",
                        step=index,
                        invalid_reasons=validation.get("invalid_reasons", []),
                    )
                )
            if transaction_status != "committed":
                action_failures.append(
                    _failure(
                        "transaction_not_committed",
                        step=index,
                        observed=transaction_status,
                        rollback_reason=info.get("rollback_reason"),
                    )
                )
            if not isinstance(recorded_preflight, Mapping) or not isinstance(
                recorded_outcome, Mapping
            ):
                action_failures.append(_failure("campaign_resource_receipt_missing", step=index))
            else:
                recorded_preflight_semantics = {
                    key: copy.deepcopy(recorded_preflight.get(key))
                    for key in (
                        "allowed",
                        "attempt_charged",
                        "rejection_reasons",
                        "proposed_delta",
                    )
                }
                replay_preflight_semantics = {
                    key: copy.deepcopy(replay_preflight.get(key))
                    for key in (
                        "allowed",
                        "attempt_charged",
                        "rejection_reasons",
                        "proposed_delta",
                    )
                } if isinstance(replay_preflight, Mapping) else None
                if recorded_preflight_semantics != replay_preflight_semantics:
                    action_failures.append(
                        _failure("campaign_resource_preflight_replay_mismatch", step=index)
                    )
                if recorded_outcome != replay_outcome:
                    action_failures.append(
                        _failure("campaign_resource_outcome_replay_mismatch", step=index)
                    )
                if recorded_preflight.get("allowed") is not True or recorded_preflight.get(
                    "rejection_reasons"
                ):
                    action_failures.append(
                        _failure("campaign_resource_rejection", step=index)
                    )
            if step_resource.get("resource_reconciled") is not True:
                action_failures.append(
                    _failure(
                        "step_resource_reconciliation_failed",
                        step=index,
                        mismatches=step_resource.get("reconciliation_mismatches", []),
                    )
                )

            checks = info.get("constitution_checks")
            checks = copy.deepcopy(checks) if isinstance(checks, list) else []
            failed_checks = [
                check
                for check in checks
                if isinstance(check, Mapping) and check.get("passed") is not True
            ]
            if failed_checks:
                action_failures.append(
                    _failure(
                        "constitution_check_failed",
                        step=index,
                        check_names=sorted(str(check.get("name")) for check in failed_checks),
                    )
                )
            events = info.get("world_events")
            events = copy.deepcopy(events) if isinstance(events, list) else []
            event_matches = any(
                isinstance(event, Mapping)
                and event.get("operation_type") == action.get("operation")
                and event.get("event_type") == "operation_applied"
                for event in events
            )
            if not event_matches:
                action_failures.append(_failure("world_event_propagation_failed", step=index))

            public_payloads = (
                ("decision_context", history.get("decision_context")),
                ("public_view", history.get("public_view")),
                ("public_observation", history.get("observation")),
            )
            step_leakage: list[dict[str, Any]] = []
            for label, payload in public_payloads:
                step_leakage.extend(
                    _leakage_findings(env, payload, f"action-{index}.{label}")
                )
            leakage.extend(step_leakage)
            if step_leakage:
                action_failures.append(
                    _failure(
                        "public_private_leakage",
                        step=index,
                        finding_count=len(step_leakage),
                    )
                )

            trace = _trace_for_action(history)
            trace_session = trace.get("session_id") if isinstance(trace, Mapping) else None
            request_id = trace.get("request_id") if isinstance(trace, Mapping) else None
            trace_valid = bool(
                isinstance(trace, Mapping)
                and trace_session == session_id
                and isinstance(request_id, str)
                and trace.get("expected_step") == index
                and trace.get("action") == action
                and trace.get("action_payload_sha256") == _sha256_value(action)
            )
            if not trace_valid:
                action_failures.append(_failure("agent_session_action_binding_failed", step=index))

            mcp_binding: dict[str, Any]
            if index <= len(mcp_steps) and isinstance(request_id, str):
                mcp_binding = _mcp_step_binding(
                    audit=mcp_steps[index - 1],
                    step=index,
                    action=action,
                    request_id=request_id,
                )
            else:
                mcp_binding = {"verified": False}
            if mcp_binding.get("verified") is not True:
                action_failures.append(_failure("mcp_step_binding_failed", step=index))

            method_resources = history.get("method_resources")
            if not isinstance(method_resources, Mapping):
                action_failures.append(_failure("method_resource_receipt_missing", step=index))
                method_resources = {}
            elif int(method_resources.get("operation_count", -1)) != index:
                action_failures.append(
                    _failure(
                        "method_operation_count_mismatch",
                        step=index,
                        observed=method_resources.get("operation_count"),
                    )
                )

            if action.get("operation") == "terminate" and committed:
                termination_probe = _post_termination_validation_receipt(base, actions)
            if action.get("operation") == "measure" and action.get("instrument") == "final_assay":
                evaluation_receipt = {
                    "reward": float(reward),
                    "leaderboard_score": info.get("leaderboard_score"),
                    "reward_source": info.get("reward_source"),
                    "experiment_completed": info.get("experiment_completed"),
                    "experiment_ended": info.get("experiment_ended"),
                    "transaction_status": transaction_status,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "public_observation": copy.deepcopy(history.get("observation")),
                }

            receipt = {
                "step": index,
                "public_input": _public_input_summary(history),
                "action": action,
                "action_sha256": _sha256_value(action),
                "schema_validation": {
                    "valid": validation.get("valid"),
                    "invalid_reasons": copy.deepcopy(validation.get("invalid_reasons", [])),
                    "preconditions": copy.deepcopy(validation.get("preconditions", {})),
                    "canonical_action": copy.deepcopy(validation.get("canonical_action")),
                },
                "transaction": {
                    "status": transaction_status,
                    "rollback_reason": info.get("rollback_reason"),
                    "operation_committed": committed,
                },
                "constitution_checks": checks,
                "world_events": events,
                "resource_preflight": copy.deepcopy(recorded_preflight),
                "resource_outcome_delta": copy.deepcopy(recorded_outcome),
                "resource_reconciliation": step_resource,
                "public_observation": copy.deepcopy(history.get("observation")),
                "public_observation_sha256": _sha256_value(history.get("observation")),
                "method_resources": copy.deepcopy(dict(method_resources)),
                "provider_binding": {
                    "session_id": trace_session,
                    "request_id": request_id,
                    "accepted_action_verified": trace_valid,
                    "mcp_step": mcp_binding,
                },
                "leakage_findings": step_leakage,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "passed": not action_failures,
                "failures": action_failures,
            }
            action_receipts.append(receipt)
            failures.extend(action_failures)

        resource_after = base.campaign_resource_snapshot()
        public_resource = base.public_campaign_resource_state()
        resource_receipt = _resource_receipt_summary(
            resource_steps,
            before_snapshot=resource_before,
            after_snapshot=resource_after,
            public_state=public_resource,
        )
        resource_receipt.pop("step_receipts", None)
    except Exception as exc:
        failures.append(_failure("independent_receipt_replay_exception", **_exception_receipt(exc)))
        resource_receipt = {
            "resource_reconciled": False,
            "reconciliation_mismatches": ["independent_receipt_replay_exception"],
        }
    finally:
        env.close()

    if leakage and not any(row["class"] == "public_private_leakage" for row in failures):
        failures.append(_failure("public_private_leakage", finding_count=len(leakage)))
    return {
        "actions": action_receipts,
        "failures": failures,
        "environment_resource_receipt": resource_receipt,
        "termination_probe": termination_probe,
        "evaluation_receipt": evaluation_receipt,
        "leakage_findings": leakage,
        "provider_receipts": sanitized_sessions,
        "mcp_call_count": len(mcp_calls),
        "mcp_step_count": len(mcp_steps),
        "session_id": session_id,
    }


def _provider_accounting(
    *,
    receipts: list[dict[str, Any]],
    method_usage: Mapping[str, Any],
    action_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    session = receipts[0] if len(receipts) == 1 else {}
    usage = session.get("usage") if isinstance(session, Mapping) else None
    usage = copy.deepcopy(dict(usage)) if isinstance(usage, Mapping) else {}
    mcp_calls = session.get("mcp_tool_calls") if isinstance(session, Mapping) else None
    mcp_calls = mcp_calls if isinstance(mcp_calls, list) else []
    mcp_steps = [
        row for row in mcp_calls if isinstance(row, Mapping) and row.get("tool") == "step"
    ]
    accepted = action_count

    if len(receipts) != 1:
        failures.append(
            _failure("provider_session_count_mismatch", expected=1, observed=len(receipts))
        )
    required_session_checks = {
        "status_completed": session.get("status") == "completed",
        "return_code_zero": session.get("return_code") == 0,
        "terminal_reason_complete": session.get("terminal_reason") == "experiment_complete",
        "model_exact": session.get("model_id") == FROZEN_MODEL,
        "reasoning_effort_exact": session.get("reasoning_effort")
        == FROZEN_REASONING_EFFORT,
        "usage_complete": session.get("usage_complete") is True,
        "provider_errors_empty": session.get("provider_errors") == [],
        "final_payload_valid": session.get("final_payload_valid") is True,
        "final_payload_status_complete": session.get("final_payload_status")
        == "experiment_complete",
        "mcp_integrity": session.get("mcp_tool_integrity_verified_after_session") is True,
        "experiment_tool_integrity": session.get(
            "experiment_tool_integrity_verified_after_session"
        )
        is True,
        "lab_tool_integrity": session.get("lab_tool_integrity_verified_after_session") is True,
        "private_reasoning_not_retained": session.get("private_reasoning_retained") is False,
    }
    failed_session_checks = [name for name, passed in required_session_checks.items() if not passed]
    if failed_session_checks:
        failures.append(
            _failure("provider_session_receipt_failed", checks=failed_session_checks)
        )

    input_tokens = int(usage.get("prompt_tokens", -1))
    uncached_input_tokens = int(usage.get("prompt_cache_miss_tokens", -1))
    output_tokens = int(usage.get("completion_tokens", -1))
    token_checks = {
        "input_nonnegative": input_tokens >= 0,
        "uncached_input_nonnegative": uncached_input_tokens >= 0,
        "output_nonnegative": output_tokens >= 0,
        "input_within_limit": 0 <= input_tokens <= FROZEN_INPUT_TOKEN_LIMIT,
        "uncached_input_within_limit": (
            0 <= uncached_input_tokens <= FROZEN_UNCACHED_INPUT_TOKEN_LIMIT
        ),
        "output_within_limit": 0 <= output_tokens <= FROZEN_OUTPUT_TOKEN_LIMIT,
        "total_reconciled": int(usage.get("total_tokens", -1))
        == input_tokens + output_tokens,
        "cache_reconciled": int(usage.get("prompt_cache_hit_tokens", -1))
            + int(usage.get("prompt_cache_miss_tokens", -1))
            == input_tokens,
        "cache_hit_nonnegative": int(usage.get("prompt_cache_hit_tokens", -1)) >= 0,
        "cache_miss_nonnegative": int(usage.get("prompt_cache_miss_tokens", -1)) >= 0,
    }
    failed_token_checks = [name for name, passed in token_checks.items() if not passed]
    if failed_token_checks:
        failures.append(_failure("provider_token_accounting_failed", checks=failed_token_checks))

    sanitized_method = _sanitize_method_usage(method_usage)
    method_checks = {
        "provider_usage_not_pending": method_usage.get("provider_usage_pending") is False,
        "provider_usage_complete": method_usage.get("provider_usage_accounting_complete") is True,
        "provider_calls_complete": method_usage.get("provider_call_accounting_complete") is True,
        "provider_tokens_complete": method_usage.get("provider_token_accounting_complete") is True,
        "provider_cache_complete": method_usage.get("provider_cache_accounting_complete") is True,
        "provider_session_count_exact": int(
            method_usage.get(
                "provider_session_count",
                method_usage.get("model_call_count", -1),
            )
        )
        == 1,
        "logical_codex_turn_count_exact": int(
            method_usage.get(
                "logical_codex_turn_count",
                method_usage.get("model_call_count", -1),
            )
        )
        == 1,
        "legacy_model_call_count_alias_exact": int(
            method_usage.get("model_call_count", -1)
        )
        == 1,
        "input_tokens_match": int(method_usage.get("input_token_count", -1)) == input_tokens,
        "uncached_input_tokens_match": (
            method_usage.get("uncached_input_token_count") is None
            or int(method_usage.get("uncached_input_token_count", -1))
            == uncached_input_tokens
        ),
        "output_tokens_match": int(method_usage.get("output_token_count", -1))
        == output_tokens,
    }
    failed_method_checks = [name for name, passed in method_checks.items() if not passed]
    if failed_method_checks:
        failures.append(_failure("method_provider_accounting_failed", checks=failed_method_checks))
    if len(mcp_steps) != action_count:
        failures.append(
            _failure(
                "mcp_step_count_mismatch",
                expected=action_count,
                observed=len(mcp_steps),
            )
        )

    return (
        {
            "session_count": len(receipts),
            "provider_session_count": len(receipts),
            "logical_codex_turn_count": method_usage.get(
                "logical_codex_turn_count",
                method_usage.get("model_call_count"),
            ),
            "model_call_count": method_usage.get("model_call_count"),
            "backend_model_response_count": method_usage.get(
                "backend_model_response_count"
            ),
            "accepted_action_count": accepted,
            "mcp_tool_call_count": len(mcp_calls),
            "mcp_step_count": len(mcp_steps),
            "usage": usage,
            "token_limits": {
                "input": FROZEN_INPUT_TOKEN_LIMIT,
                "uncached_input": FROZEN_UNCACHED_INPUT_TOKEN_LIMIT,
                "output": FROZEN_OUTPUT_TOKEN_LIMIT,
            },
            "input_token_breakdown": {
                "cumulative_input_tokens": input_tokens,
                "cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
                "cache_miss_tokens": uncached_input_tokens,
                "cache_hit_ratio": (
                    int(usage.get("prompt_cache_hit_tokens", 0)) / input_tokens
                    if input_tokens > 0
                    else 0.0
                ),
                "cache_means_reused_input_context_not_repeated_output": True,
            },
            "session_checks": required_session_checks,
            "token_checks": token_checks,
            "method_checks": method_checks,
            "method_resource_usage": sanitized_method,
            "pricing": "cached ChatGPT subscription; per-run USD price unavailable",
            "passed": not failures,
        },
        failures,
    )


def _lifecycle_receipt(
    actions: list[dict[str, Any]],
    *,
    method_usage: Mapping[str, Any],
    termination_probe: Mapping[str, Any] | None,
    evaluation_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    committed = [
        row
        for row in actions
        if row.get("transaction", {}).get("status") == "committed"
    ]
    terminates = [row for row in committed if row.get("action", {}).get("operation") == "terminate"]
    finals = [
        row
        for row in committed
        if row.get("action", {}).get("operation") == "measure"
        and row.get("action", {}).get("instrument") == "final_assay"
    ]
    final = finals[-1] if finals else None
    final_method = final.get("method_resources") if isinstance(final, Mapping) else None
    complete_count = (
        int(final_method.get("complete_experiment_count", -1))
        if isinstance(final_method, Mapping)
        else -1
    )
    checks = {
        "action_count_in_range": 1 <= len(actions) <= FROZEN_OPERATION_LIMIT,
        "all_actions_committed": len(committed) == len(actions),
        "at_least_one_terminate": len(terminates) >= 1,
        "exactly_one_final_assay": len(finals) == 1,
        "complete_experiment_count_exact": complete_count == 1,
        "final_assay_terminated": bool(final and final.get("terminated") is True),
        "final_assay_not_truncated": bool(final and final.get("truncated") is False),
        "post_termination_probe_passed": bool(
            isinstance(termination_probe, Mapping)
            and termination_probe.get("passed") is True
        ),
        "evaluation_receipt_present": isinstance(evaluation_receipt, Mapping),
        "no_right_censoring": not any(
            bool(row.get("truncated"))
            or bool(row.get("transaction", {}).get("status") != "committed")
            for row in actions
        ),
        "provider_usage_finalized": method_usage.get("provider_usage_pending") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        failures.append(_failure("lifecycle_closure_failed", checks=failed))
    return (
        {
            "submitted_action_count": len(actions),
            "committed_action_count": len(committed),
            "rollback_count": sum(
                row.get("transaction", {}).get("status") == "rolled_back" for row in actions
            ),
            "committed_terminate_count": len(terminates),
            "committed_final_assay_count": len(finals),
            "complete_experiment_count": complete_count,
            "right_censored": not checks["no_right_censoring"],
            "checks": checks,
            "post_termination_validation": copy.deepcopy(termination_probe),
            "evaluation_receipt": copy.deepcopy(evaluation_receipt),
            "passed": not failures,
        },
        failures,
    )


def _receipt_completeness(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    actions = report.get("actions")
    actions = actions if isinstance(actions, list) else []
    required_action_fields = {
        "step",
        "public_input",
        "action",
        "action_sha256",
        "schema_validation",
        "transaction",
        "constitution_checks",
        "world_events",
        "resource_preflight",
        "resource_outcome_delta",
        "resource_reconciliation",
        "public_observation",
        "method_resources",
        "provider_binding",
        "leakage_findings",
        "failures",
    }
    for expected_step, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping):
            errors.append({"step": expected_step, "requirement": "action_receipt_object"})
            continue
        for field in sorted(required_action_fields - set(action)):
            errors.append({"step": expected_step, "requirement": field})
        if action.get("step") != expected_step:
            errors.append({"step": expected_step, "requirement": "contiguous_step"})
        if action.get("schema_validation", {}).get("valid") is not True:
            errors.append({"step": expected_step, "requirement": "schema_valid"})
        if action.get("transaction", {}).get("status") != "committed":
            errors.append({"step": expected_step, "requirement": "transaction_committed"})
        if action.get("resource_reconciliation", {}).get("resource_reconciled") is not True:
            errors.append({"step": expected_step, "requirement": "resource_reconciled"})
        if action.get("provider_binding", {}).get("accepted_action_verified") is not True:
            errors.append({"step": expected_step, "requirement": "provider_action_binding"})
        if action.get("provider_binding", {}).get("mcp_step", {}).get("verified") is not True:
            errors.append({"step": expected_step, "requirement": "mcp_step_binding"})
    required_report_paths = {
        "runtime_task_contract": report.get("frozen_experiment", {})
        .get("runtime_contract_binding", {})
        .get("task_contract_hash_matches"),
        "step_monitor": report.get("step_monitor", {}).get("all_passed"),
        "provider_accounting": report.get("provider_accounting", {}).get("passed"),
        "lifecycle": report.get("lifecycle", {}).get("passed"),
        "environment_resources": report.get("environment_resource_receipt", {}).get(
            "resource_reconciled"
        ),
        "declared_resource_budget": report.get("declared_resource_budget", {}).get(
            "passed"
        ),
        "exact_replay": report.get("exact_replay", {}).get("verified"),
        "zero_leakage": report.get("public_boundary", {}).get("finding_count") == 0,
    }
    for requirement, passed in required_report_paths.items():
        if passed is not True:
            errors.append({"requirement": requirement})
    return (
        {"passed": not errors, "error_count": len(errors), "errors": errors},
        [_failure("missing_or_failed_receipt", **error) for error in errors],
    )


def _sanitization_findings(value: Any, *, temp_root: Path | None = None) -> list[str]:
    findings: list[str] = []
    forbidden_keys = {
        "final_payload_summary",
        "raw_provider_response",
        "raw_provider_payload",
        "private_reasoning",
        "reasoning_body",
        "prompt_body",
    }
    temp_text = str(temp_root.resolve()) if temp_root is not None else None

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key) in forbidden_keys:
                    findings.append(f"forbidden_key:{child_path}")
                visit(child, child_path)
        elif isinstance(item, list | tuple):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            normalized = item.replace("\\", "/")
            if temp_text and temp_text.replace("\\", "/") in normalized:
                findings.append(f"temporary_path:{path}")
            if _WINDOWS_ABSOLUTE_PATH.match(item) or item.startswith("/"):
                findings.append(f"absolute_path:{path}")

    visit(value, "")
    return sorted(set(findings))


def _build_execution_report(
    *,
    source_binding: dict[str, Any],
    existing_evidence: dict[str, Any],
    provider_preflight: dict[str, Any],
    scratch: Path,
    process_factory: ProcessFactory | None,
) -> dict[str, Any]:
    composition_request = copy.deepcopy(existing_evidence["U05"]["composition_request"])
    campaign_card = _campaign_resource_card(composition_request)
    runtime_contract = _runtime_contract_binding(
        composition_request=composition_request,
        campaign_card=campaign_card,
    )
    method_limits = _method_resource_limits()
    trajectory_path = scratch / "trajectory.jsonl"
    workspace = scratch / "interactive-workspace"
    histories: list[dict[str, Any]] = []
    run_failure: dict[str, Any] | None = None

    agent = InteractiveCodexExperimentAgent(
        workspace=workspace,
        role_id=METHOD_ID,
        model=FROZEN_MODEL,
        reasoning_effort=FROZEN_REASONING_EFFORT,
        process_factory=process_factory,
        request_timeout_s=FROZEN_REQUEST_TIMEOUT_S,
        finalization_timeout_s=FROZEN_FINALIZATION_TIMEOUT_S,
        pre_action_restart_limit=FROZEN_PRE_ACTION_RESTART_LIMIT,
        max_tool_output_bytes=FROZEN_MAX_TOOL_OUTPUT_BYTES,
        history_event_limit=FROZEN_HISTORY_EVENT_LIMIT,
        history_byte_limit=FROZEN_HISTORY_BYTE_LIMIT,
    )
    step_monitor = _StepFailFastMonitor(
        composition_request=composition_request,
        campaign_card=campaign_card,
        require_agent_trace=True,
    )

    def retain(record: HistoryRecord, trace: list[dict[str, Any]]) -> None:
        histories.append(_history_payload(record, trace))
        step_monitor.observe(record, trace)

    try:
        try:
            run_agent(
                env_id="ChemWorld",
                agent=agent,
                world_split="public-test",
                budget=FROZEN_OPERATION_LIMIT,
                objective="balanced",
                seed=FROZEN_WORLD_SEED,
                agent_seed=FROZEN_AGENT_SEED,
                composition=composition_request,
                output_path=trajectory_path,
                step_callback=retain,
                campaign_resource_card=campaign_card,
                method_resource_limits=method_limits,
            )
        except BaseException as exc:  # retain the sole formal attempt as a failed result
            run_failure = _exception_receipt(exc)
    finally:
        step_monitor.close()

    try:
        records = load_jsonl(trajectory_path) if trajectory_path.is_file() else []
    except (OSError, json.JSONDecodeError) as exc:
        records = []
        run_failure = run_failure or _exception_receipt(exc)
    provider_receipts = agent.provider_receipts()
    method_usage = agent.method_resource_usage()
    audit = _audit_records(
        records=records,
        histories=histories,
        composition_request=composition_request,
        campaign_card=campaign_card,
        provider_receipts=provider_receipts,
    )

    failures = list(audit["failures"])
    if run_failure is not None:
        failures.append(_failure("runner_or_provider_exception", **run_failure))
    try:
        exact_replay = verify_records(records, tolerance=0.0).to_dict()
    except Exception as exc:
        exact_replay = {
            "verified": False,
            "checked_steps": 0,
            "max_abs_error": None,
            "mismatches": [_exception_receipt(exc)],
        }
    if not (
        exact_replay.get("verified") is True
        and exact_replay.get("max_abs_error") == 0.0
        and int(exact_replay.get("checked_steps", -1)) == len(records)
    ):
        failures.append(_failure("exact_replay_failed"))

    provider_accounting, provider_failures = _provider_accounting(
        receipts=audit["provider_receipts"],
        method_usage=method_usage,
        action_count=len(audit["actions"]),
    )
    failures.extend(provider_failures)
    lifecycle, lifecycle_failures = _lifecycle_receipt(
        audit["actions"],
        method_usage=method_usage,
        termination_probe=audit["termination_probe"],
        evaluation_receipt=audit["evaluation_receipt"],
    )
    failures.extend(lifecycle_failures)
    if audit["environment_resource_receipt"].get("resource_reconciled") is not True:
        failures.append(_failure("environment_resource_reconciliation_failed"))
    declared_resource_budget = _declared_resource_budget_receipt(
        composition_request=composition_request,
        environment_resource_receipt=audit["environment_resource_receipt"],
        actions=audit["actions"],
    )
    if declared_resource_budget["passed"] is not True:
        failures.append(
            _failure(
                "declared_resource_budget_exceeded",
                resources=declared_resource_budget["exceeded_resources"],
            )
        )
    monitor_passed = bool(step_monitor.events) and len(step_monitor.events) == len(
        audit["actions"]
    ) and all(event.get("status") == "passed" for event in step_monitor.events)
    if not monitor_passed:
        failures.append(
            _failure(
                "step_monitor_failed",
                event_count=len(step_monitor.events),
                action_count=len(audit["actions"]),
            )
        )

    trajectory_bytes = trajectory_path.stat().st_size if trajectory_path.is_file() else 0
    trajectory_sha = _sha256_path(trajectory_path) if trajectory_path.is_file() else None
    action_count = len(audit["actions"])
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "status": "failed",
        "source_binding": source_binding,
        "provider_preflight": provider_preflight,
        "existing_evidence": {
            "current_registry": existing_evidence["current_registry"],
            "U04": existing_evidence["U04"],
            "U05": {
                key: copy.deepcopy(value)
                for key, value in existing_evidence["U05"].items()
                if key != "composition_request"
            },
        },
        "frozen_experiment": {
            "independent_unit": "one complete agent system in one frozen generated world",
            "method_id": METHOD_ID,
            "model": FROZEN_MODEL,
            "reasoning_effort": FROZEN_REASONING_EFFORT,
            "agent_seed": FROZEN_AGENT_SEED,
            "world_seed": FROZEN_WORLD_SEED,
            "composition_id": FROZEN_COMPOSITION_ID,
            "case_id": FROZEN_CASE_ID,
            "generation_seed": FROZEN_GENERATION_SEED,
            "generation_index": FROZEN_GENERATION_INDEX,
            "composition_request": composition_request,
            "composition_request_sha256": FROZEN_REQUEST_SHA256,
            "runtime_contract_binding": runtime_contract,
            "public_compiled_task_subobject_hash": FROZEN_PUBLIC_TASK_SUBOBJECT_HASH,
            "campaign_resource_card": campaign_card,
            "method_resource_limits": method_limits,
            "method_resource_policy": _method_resource_policy(),
            "pre_action_restart_limit": FROZEN_PRE_ACTION_RESTART_LIMIT,
            "request_timeout_s": FROZEN_REQUEST_TIMEOUT_S,
            "finalization_timeout_s": FROZEN_FINALIZATION_TIMEOUT_S,
        },
        "counting_rule": {
            "independent_unit_count": 1,
            "action_monitoring": "complete census; sampling is not a pass gate",
            "performance_threshold": None,
            "model_ranking": False,
        },
        "denominators": {
            "lifecycle_count": 1,
            "provider_session_count": 1,
            "model_call_count": 1,
            "submitted_action_count": action_count,
            "trajectory_record_count": len(records),
        },
        "summary": {
            "submitted_action_count": action_count,
            "trajectory_record_count": len(records),
            "committed_action_count": lifecycle["committed_action_count"],
            "rollback_count": lifecycle["rollback_count"],
            "committed_terminate_count": lifecycle["committed_terminate_count"],
            "committed_final_assay_count": lifecycle["committed_final_assay_count"],
            "provider_session_count": provider_accounting["session_count"],
            "mcp_step_count": provider_accounting["mcp_step_count"],
            "public_private_leakage_count": len(audit["leakage_findings"]),
            "trajectory_bytes": trajectory_bytes,
            "trajectory_sha256": trajectory_sha,
            "step_monitor_event_count": len(step_monitor.events),
        },
        "step_monitor": {
            "mode": "complete_per-action_fail-fast_census",
            "event_count": len(step_monitor.events),
            "all_passed": monitor_passed,
            "events": step_monitor.events,
        },
        "provider_accounting": provider_accounting,
        "provider_session_receipts": audit["provider_receipts"],
        "lifecycle": lifecycle,
        "environment_resource_receipt": audit["environment_resource_receipt"],
        "declared_resource_budget": declared_resource_budget,
        "public_boundary": {
            "finding_count": len(audit["leakage_findings"]),
            "findings": audit["leakage_findings"],
            "raw_provider_payload_retained": False,
            "private_reasoning_retained": False,
            "temporary_workspace_retained": False,
            "final_payload_summary_retained": False,
        },
        "exact_replay": exact_replay,
        "actions": audit["actions"],
        "failures": failures,
        "claim_boundary": [
            "One complete-agent virtual-instrument usability demonstration only.",
            "The endpoint score is descriptive and is not a performance threshold or ranking.",
            "No physical-laboratory validity or general-agent claim is made.",
        ],
    }
    completeness, completeness_failures = _receipt_completeness(report)
    report["receipt_completeness"] = completeness
    report["failures"].extend(completeness_failures)
    report["failure_class_counts"] = dict(
        sorted(Counter(row["class"] for row in report["failures"]).items())
    )
    report["status"] = "passed" if not report["failures"] else "failed"
    sanitization_findings = _sanitization_findings(report, temp_root=scratch)
    report["sanitization"] = {
        "passed": not sanitization_findings,
        "finding_count": len(sanitization_findings),
        "findings": sanitization_findings,
    }
    if sanitization_findings:
        report["failures"].append(
            _failure("report_sanitization_failed", findings=sanitization_findings)
        )
        report["failure_class_counts"] = dict(
            sorted(Counter(row["class"] for row in report["failures"]).items())
        )
        report["status"] = "failed"
    return report


def build_report(
    *,
    repository_root: str | Path,
    require_clean: bool = True,
    scratch_dir: str | Path | None = None,
    process_factory: ProcessFactory | None = None,
    provider_preflight: Mapping[str, Any] | None = None,
    preflight_runner: PreflightRunner | None = None,
    codex_executable: str | None = None,
) -> dict[str, Any]:
    """Execute and fully audit the sole frozen U05/E02 agent lifecycle.

    ``process_factory`` and ``preflight_runner`` are dependency-injection seams
    for offline tests.  Formal execution leaves them unset.
    """

    root = Path(repository_root).resolve()
    source_binding = validate_launch_preconditions(root, require_clean=require_clean)
    existing_evidence = resolve_existing_evidence(root)
    preflight = (
        _validate_injected_provider_preflight(provider_preflight)
        if provider_preflight is not None
        else collect_provider_preflight(
            root,
            codex_executable=codex_executable,
            runner=preflight_runner,
        )
    )
    if scratch_dir is not None:
        scratch = Path(scratch_dir).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        return _build_execution_report(
            source_binding=source_binding,
            existing_evidence=existing_evidence,
            provider_preflight=preflight,
            scratch=scratch,
            process_factory=process_factory,
        )
    with tempfile.TemporaryDirectory(prefix="chemworld-first-paper-u05-") as temporary:
        return _build_execution_report(
            source_binding=source_binding,
            existing_evidence=existing_evidence,
            provider_preflight=preflight,
            scratch=Path(temporary),
            process_factory=process_factory,
        )


def amend_report_with_declared_resource_audit(
    report: Mapping[str, Any],
    *,
    original_report_sha256: str,
    original_markdown_sha256: str,
    original_result_commit: str,
    amendment_commit: str,
) -> dict[str, Any]:
    """Add a fail-closed declared-resource audit without rerunning the provider."""

    amended = copy.deepcopy(dict(report))
    if amended.get("qualification_id") != QUALIFICATION_ID:
        raise CompleteAgentQualificationError("cannot amend an unrelated report")
    if amended.get("postrun_amendment") is not None:
        raise CompleteAgentQualificationError("postrun amendment already exists")
    frozen = amended.get("frozen_experiment")
    composition = frozen.get("composition_request") if isinstance(frozen, Mapping) else None
    resource_receipt = amended.get("environment_resource_receipt")
    actions = amended.get("actions")
    if not isinstance(composition, Mapping) or not isinstance(resource_receipt, Mapping):
        raise CompleteAgentQualificationError(
            "formal report lacks composition or environment resource receipts"
        )
    action_rows = (
        [row for row in actions if isinstance(row, Mapping)]
        if isinstance(actions, list)
        else []
    )
    declared = _declared_resource_budget_receipt(
        composition_request=composition,
        environment_resource_receipt=resource_receipt,
        actions=action_rows,
    )
    original_failures = amended.get("failures")
    retained_failures = [
        copy.deepcopy(row)
        for row in original_failures
        if isinstance(row, Mapping)
        and row.get("class")
        not in {
            "declared_resource_budget_exceeded",
            "missing_or_failed_receipt",
            "report_sanitization_failed",
        }
    ] if isinstance(original_failures, list) else []
    if declared["passed"] is not True:
        retained_failures.append(
            _failure(
                "declared_resource_budget_exceeded",
                resources=declared["exceeded_resources"],
                first_exceeded_step=declared["first_exceeded_step"],
            )
        )
    amended["declared_resource_budget"] = declared
    amended["postrun_amendment"] = {
        "schema_version": "chemworld-first-paper-postrun-amendment-0.1",
        "amended_on": "2026-08-05",
        "reason": (
            "The original evaluator reconciled the resource ledger but did not compare "
            "observed process time and sample use against the composition-declared caps."
        ),
        "original_report_sha256": original_report_sha256,
        "original_markdown_sha256": original_markdown_sha256,
        "original_result_commit": original_result_commit,
        "amendment_commit": amendment_commit,
        "provider_rerun": False,
        "action_or_provider_data_changed": False,
        "original_status": amended.get("status"),
        "original_failure_class_counts": copy.deepcopy(
            amended.get("failure_class_counts", {})
        ),
    }
    amended["failures"] = retained_failures
    completeness, completeness_failures = _receipt_completeness(amended)
    amended["receipt_completeness"] = completeness
    amended["failures"].extend(completeness_failures)
    sanitization_findings = _sanitization_findings(amended)
    amended["sanitization"] = {
        "passed": not sanitization_findings,
        "finding_count": len(sanitization_findings),
        "findings": sanitization_findings,
    }
    if sanitization_findings:
        amended["failures"].append(
            _failure("report_sanitization_failed", findings=sanitization_findings)
        )
    amended["failure_class_counts"] = dict(
        sorted(Counter(row["class"] for row in amended["failures"]).items())
    )
    amended["status"] = "failed" if amended["failures"] else "passed"
    return amended


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, denominator-complete human-readable summary."""

    summary = report["summary"]
    provider = report["provider_accounting"]
    lifecycle = report["lifecycle"]
    replay = report["exact_replay"]
    resource = report["environment_resource_receipt"]
    declared = report.get("declared_resource_budget", {})
    declared_limits = declared.get("declared_limits", {})
    observed_usage = declared.get("observed_usage", {})
    lines = [
        "# First-paper complete-agent instrument use",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "## Complete census",
        "",
        "| Quantity | Observed | Required |",
        "| --- | ---: | ---: |",
        "| Lifecycles | 1 | 1 |",
        f"| Submitted actions | {summary['submitted_action_count']} | 1--16 |",
        f"| Trajectory records | {summary['trajectory_record_count']} | submitted actions |",
        f"| Committed actions | {summary['committed_action_count']} | submitted actions |",
        f"| Rollbacks | {summary['rollback_count']} | 0 |",
        f"| Committed terminate | {summary['committed_terminate_count']} | >=1 |",
        f"| Committed final assay | {summary['committed_final_assay_count']} | 1 |",
        (
            "| Provider sessions/model calls | "
            f"{provider['session_count']}/{provider['model_call_count']} | 1/1 |"
        ),
        f"| MCP step calls | {provider['mcp_step_count']} | submitted actions |",
        f"| Public/private leakage findings | {summary['public_private_leakage_count']} | 0 |",
        "",
        "All submitted actions are inspected; sampling is not used as a pass gate.",
        "",
        "## Closure and replay",
        "",
        f"- Lifecycle closed: `{str(lifecycle['passed']).lower()}`.",
        (
            "- Environment resources reconciled: "
            f"`{str(resource.get('resource_reconciled')).lower()}`."
        ),
        (
            "- Declared process-time budget: "
            f"`{observed_usage.get('process_time_s')}` / "
            f"`{declared_limits.get('process_time_s')}` s; "
            f"passed `{str(declared.get('passed')).lower()}`."
        ),
        (
            "- Declared sample budget: "
            f"`{observed_usage.get('sample_consumed_L')}` / "
            f"`{declared_limits.get('sample_consumed_L')}` L."
        ),
        (
            f"- Exact replay: `{str(replay.get('verified')).lower()}`; "
            f"max absolute error `{replay.get('max_abs_error')}`."
        ),
        (
            f"- Input/output tokens: `{provider['usage'].get('prompt_tokens')}` / "
            f"`{provider['usage'].get('completion_tokens')}`."
        ),
        (
            "- Per-run USD price: unavailable for cached ChatGPT subscription login; "
            "no measured zero is reported."
        ),
        "",
        "## Existing U04 evidence",
        "",
        (
            "- Current fork pairs/traces/provider calls: "
            f"`{report['existing_evidence']['U04']['pair_count']}` / "
            f"`{report['existing_evidence']['U04']['trace_count']}` / "
            f"`{report['existing_evidence']['U04']['provider_call_count']}`."
        ),
        "",
        "## Failures",
        "",
    ]
    failures = report.get("failures", [])
    if failures:
        lines.extend(
            (
                f"- `{row.get('class', 'unclassified')}`: "
                f"`{json.dumps(row, ensure_ascii=False, sort_keys=True)}`"
            )
            for row in failures
            if isinstance(row, Mapping)
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any],
    *,
    output_path: str | Path,
    markdown_path: str | Path | None = None,
    allow_existing: bool = False,
) -> tuple[Path, Path]:
    """Write JSON and Markdown together, refusing replacement by default."""

    return write_report_outputs(
        report,
        render_markdown=render_markdown,
        output_path=output_path,
        markdown_path=markdown_path,
        allow_existing=allow_existing,
    )


__all__ = [
    "CURRENT_CONFIG",
    "EXPERIMENT_NOTE",
    "FROZEN_AGENT_SEED",
    "FROZEN_AUDITED_CUMULATIVE_INPUT_BASELINE",
    "FROZEN_CASE_ID",
    "FROZEN_COMPOSITION_ID",
    "FROZEN_CUMULATIVE_INPUT_HEADROOM",
    "FROZEN_FINALIZATION_TIMEOUT_S",
    "FROZEN_GENERATION_INDEX",
    "FROZEN_GENERATION_SEED",
    "FROZEN_HISTORY_BYTE_LIMIT",
    "FROZEN_HISTORY_EVENT_LIMIT",
    "FROZEN_INPUT_TOKEN_LIMIT",
    "FROZEN_MAX_TOOL_OUTPUT_BYTES",
    "FROZEN_OPERATION_LIMIT",
    "FROZEN_PUBLIC_TASK_SUBOBJECT_HASH",
    "FROZEN_REQUEST_SHA256",
    "FROZEN_RUNTIME_TASK_CONTRACT_HASH",
    "FROZEN_UNCACHED_INPUT_TOKEN_LIMIT",
    "FROZEN_WALL_TIME_LIMIT_S",
    "FROZEN_WALL_TIME_RESERVE_S",
    "FROZEN_WORLD_SEED",
    "METHOD_ID",
    "QUALIFICATION_ID",
    "REPORT_SCHEMA_VERSION",
    "CompleteAgentQualificationError",
    "amend_report_with_declared_resource_audit",
    "build_report",
    "collect_provider_preflight",
    "render_markdown",
    "resolve_existing_evidence",
    "validate_launch_preconditions",
    "write_outputs",
]
