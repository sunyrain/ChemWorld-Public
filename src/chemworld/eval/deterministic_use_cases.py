"""Frozen deterministic use-case qualification for the first paper.

The independent units are the eight cases frozen in
``workstreams/arxiv_v1/experiments/first-paper-deterministic-use-cases.md``.
Every submitted action is inspected; sampling is never used as a pass gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

import chemworld
from chemworld.agent_interface import agent_view_bundle
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.eval.cross_world_infrastructure_qualification import (
    _leakage_findings,
    _negative_ghost_state_receipt,
    _post_termination_validation_receipt,
    _recipe_resource_card,
    _resource_receipt_summary,
    _rng_snapshot,
    _world_state_sections,
    recipe_cases,
)
from chemworld.eval.report_io import git_output as _git
from chemworld.eval.report_io import write_report_outputs
from chemworld.eval.verify import verify_records
from chemworld.tasks import TASK_CONTRACT_VERSION, TaskSpec, get_task

REPORT_SCHEMA_VERSION = "chemworld-first-paper-deterministic-use-cases-report-0.1"
QUALIFICATION_ID = "first-paper-deterministic-use-cases-v1"

EXPECTED_CASE_COUNT = 8
EXPECTED_SUBMITTED_ACTION_COUNT = 89
EXPECTED_COMMITTED_ACTION_COUNT = 88
EXPECTED_ROLLBACK_COUNT = 1
EXPECTED_FINAL_ASSAY_COUNT = 8

EXPERIMENT_NOTE = Path(
    "workstreams/arxiv_v1/experiments/first-paper-deterministic-use-cases.md"
)
TODO_PATH = Path("workstreams/arxiv_v1/FIRST_PAPER_TODOLIST.md")
CURRENT_CONFIG = Path("configs/current.json")
REFERENCE_PATHS = Path("examples/world-authoring/use-case-reference-paths-v0.1.json")

_TASK_SOURCE_PATHS = (
    Path("src/chemworld/tasks.py"),
    Path("src/chemworld/agents/task_recipes.py"),
)

_REGISTERED_CASES = (
    ("U01", "U01", "reaction-to-crystallization", 0, 12),
    ("U06-flow", "U06", "flow-reaction-optimization", 0, 8),
    ("U06-electro", "U06", "electrochemical-conversion", 0, 11),
    ("U06-distillation", "U06", "reaction-to-distillation", 0, 12),
    ("U06-partition", "U06", "partition-discovery", 0, 10),
    ("U06-crystallization", "U06", "reaction-to-crystallization", 1, 12),
)

EXPECTED_ACTION_SHA256 = {
    "U01": "928e98b92859afb6968aa07e1021c222403f699baafb10bdd29b3ce9e7b81e95",
    "U02": "cb3446cb33ab3c4e101a6cc2ce96be651958b6e91b49d3b8d223ddffc3de1758",
    "U03/E01": "a0990c28168f348a669ca9db080d095a60a76ab6cb8d31fa7aac995bbe579b4d",
    "U06-flow": "d3c6776528074b758d7130ecd8cd56f67a9c52a8cf06a2e631f3679b01bea7f0",
    "U06-electro": "d5c0426bb93779aa0c2daf67ccdc1a9274508d82a8f20d5b08eb46f0cea693b6",
    "U06-distillation": (
        "d583f38de8f309f48dfada9c08414795b90c0619328a4bb5898a95a12d8ff3e7"
    ),
    "U06-partition": "1dadd67878bfece878e8d8edf76261897577790964cbb5d67a2e30f0ad2cdcba",
    "U06-crystallization": (
        "928e98b92859afb6968aa07e1021c222403f699baafb10bdd29b3ce9e7b81e95"
    ),
}

_U04_NODE_ID = "work_i_world_fork_qualification"
_U05_NODE_ID = "first_paper_composition_qualification"
_U05_COMPOSITION_ID = "qualification-reaction-distillation-observation-coverage-0001"
_U05_REQUEST_SHA256 = "687007fb2fe9e7cb7bde1eff10219469fecc73903648d2fa34fec17c10694b4f"
_U05_GENERATION_SEED = 105
_U05_GENERATION_INDEX = 0
_COMPLETED_CLAIM = "Claim: Codex /root — U01-U03/U06/E01-DET — DONE"


class DeterministicUseCaseProtocolError(ValueError):
    """Raised when a frozen input or current evidence binding has drifted."""


@dataclass(frozen=True)
class _CasePlan:
    case_id: str
    use_case_id: str
    public_identity: str
    seed: int
    task: TaskSpec
    composition_request: dict[str, Any] | None
    normalized_request: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    expected_validation: tuple[bool, ...]
    expected_transactions: tuple[str, ...]
    expected_failure: dict[str, Any] | None
    source_kind: str
    source_bindings: tuple[dict[str, Any], ...]
    recipe_binding: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
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
        raise DeterministicUseCaseProtocolError(f"JSON root must be an object: {path}")
    return payload


def _resolve_repository_path(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DeterministicUseCaseProtocolError(
            f"repository binding escapes the repository root: {relative_path}"
        ) from exc
    return candidate


def _source_binding(root: Path, relative_path: str | Path) -> dict[str, Any]:
    path = _resolve_repository_path(root, relative_path)
    if not path.is_file():
        raise DeterministicUseCaseProtocolError(f"bound source is missing: {relative_path}")
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
    """Validate the frozen claim and record the execution source binding."""

    root = Path(repository_root).resolve()
    errors: list[str] = []
    try:
        git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeterministicUseCaseProtocolError("repository_root is not a Git worktree") from exc
    if git_root != root:
        errors.append(f"repository_root is not the Git root: {root}")

    note_path = _resolve_repository_path(root, EXPERIMENT_NOTE)
    todo_path = _resolve_repository_path(root, TODO_PATH)
    if not note_path.is_file():
        errors.append("frozen experiment note is missing")
    else:
        note_text = note_path.read_text(encoding="utf-8")
        if "FROZEN BEFORE DATA GENERATION" not in note_text:
            errors.append("experiment note is not frozen before data generation")
    if not todo_path.is_file():
        errors.append("first-paper TODO is missing")
    else:
        todo_text = todo_path.read_text(encoding="utf-8")
        if _COMPLETED_CLAIM not in todo_text:
            errors.append("completed deterministic use-case claim is missing or not DONE")

    branch = _git(root, "branch", "--show-current")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and branch != "main":
        errors.append(f"formal execution requires main, found {branch or 'detached HEAD'}")
    if require_clean and dirty:
        errors.append("formal execution requires a clean worktree")
    if errors:
        raise DeterministicUseCaseProtocolError("; ".join(errors))
    return {
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "branch": branch,
        "worktree_clean": not bool(dirty),
        "task_contract_version": TASK_CONTRACT_VERSION,
        "experiment_note": _source_binding(root, EXPERIMENT_NOTE),
        "todo": _source_binding(root, TODO_PATH),
    }


def _midpoint_case(task: TaskSpec) -> dict[str, Any]:
    matches = [case for case in recipe_cases(task) if case.get("case_id") == "midpoint"]
    if len(matches) != 1:
        raise DeterministicUseCaseProtocolError(
            f"expected one midpoint recipe for {task.task_id}, found {len(matches)}"
        )
    return matches[0]


def _registered_case_plan(
    root: Path,
    *,
    case_id: str,
    use_case_id: str,
    task_id: str,
    seed: int,
    expected_action_count: int,
) -> _CasePlan:
    task = get_task(task_id)
    midpoint = _midpoint_case(task)
    actions = tuple(copy.deepcopy(midpoint["compiled_actions"]))
    if len(actions) != expected_action_count:
        raise DeterministicUseCaseProtocolError(
            f"frozen action count drifted for {case_id}: "
            f"{len(actions)} != {expected_action_count}"
        )
    source_bindings = tuple(_source_binding(root, path) for path in _TASK_SOURCE_PATHS)
    normalized_request = {
        "request_kind": "registered_task_midpoint_recipe",
        "task_id": task.task_id,
        "world_seed": seed,
        "recipe_case_id": "midpoint",
        "recipe_vector": copy.deepcopy(midpoint["vector"]),
        "recipe_vector_sha256": midpoint["vector_sha256"],
    }
    return _CasePlan(
        case_id=case_id,
        use_case_id=use_case_id,
        public_identity=task.task_id,
        seed=seed,
        task=task,
        composition_request=None,
        normalized_request=normalized_request,
        actions=actions,
        expected_validation=tuple(True for _ in actions),
        expected_transactions=tuple("committed" for _ in actions),
        expected_failure=None,
        source_kind="registered_task_midpoint",
        source_bindings=source_bindings,
        recipe_binding={
            "case_id": "midpoint",
            "kind": midpoint["kind"],
            "vector": copy.deepcopy(midpoint["vector"]),
            "vector_sha256": midpoint["vector_sha256"],
            "compiled_actions_sha256": midpoint["compiled_actions_sha256"],
        },
    )


def _composed_case_plans(root: Path) -> list[_CasePlan]:
    sidecar = _read_json_object(_resolve_repository_path(root, REFERENCE_PATHS))
    if sidecar.get("status") != "frozen_prelaunch_specification":
        raise DeterministicUseCaseProtocolError("use-case reference paths are not frozen")
    rows = sidecar.get("cases")
    if not isinstance(rows, list):
        raise DeterministicUseCaseProtocolError("use-case reference paths cases must be a list")
    by_id = {
        str(row.get("use_case_id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("use_case_id"), str)
    }
    if set(by_id) != {"U02", "U03"}:
        raise DeterministicUseCaseProtocolError("frozen sidecar must contain exactly U02 and U03")

    plans: list[_CasePlan] = []
    for use_case_id, case_id in (("U02", "U02"), ("U03", "U03/E01")):
        row = by_id[use_case_id]
        request_relative = row.get("composition_request")
        if not isinstance(request_relative, str):
            raise DeterministicUseCaseProtocolError(
                f"composition request path is missing for {use_case_id}"
            )
        request_path = _resolve_repository_path(root, request_relative)
        request = _read_json_object(request_path)
        compiled = chemworld.compile_world_composition(request)
        actions = tuple(copy.deepcopy(row.get("actions", [])))
        expected_validation = tuple(bool(value) for value in row.get("expected_validation", []))
        expected_transactions = tuple(
            str(value) for value in row.get("expected_transactions", [])
        )
        expected_count = int(row.get("submitted_action_count", -1))
        if not (
            len(actions)
            == len(expected_validation)
            == len(expected_transactions)
            == expected_count
        ):
            raise DeterministicUseCaseProtocolError(
                f"frozen sidecar denominator drifted for {use_case_id}"
            )
        if request.get("composition_id") != row.get("composition_id"):
            raise DeterministicUseCaseProtocolError(
                f"composition identity drifted for {use_case_id}"
            )
        if compiled.compatibility.pattern != row.get("expected_pattern"):
            raise DeterministicUseCaseProtocolError(
                f"composition compatibility pattern drifted for {use_case_id}"
            )
        if compiled.task_spec.objective != row.get("objective"):
            raise DeterministicUseCaseProtocolError(f"objective drifted for {use_case_id}")
        plans.append(
            _CasePlan(
                case_id=case_id,
                use_case_id=use_case_id,
                public_identity=str(request["composition_id"]),
                seed=int(row["seed"]),
                task=compiled.task_spec,
                composition_request=copy.deepcopy(request),
                normalized_request=compiled.spec.to_dict(),
                actions=actions,
                expected_validation=expected_validation,
                expected_transactions=expected_transactions,
                expected_failure=copy.deepcopy(row.get("expected_failure")),
                source_kind="public_world_composition_sidecar",
                source_bindings=(
                    _source_binding(root, REFERENCE_PATHS),
                    _source_binding(root, request_relative),
                ),
                recipe_binding={
                    "sidecar_use_case_id": use_case_id,
                    "expected_pattern": row["expected_pattern"],
                    "objective": row["objective"],
                    "expected_final": copy.deepcopy(row.get("expected_final")),
                },
            )
        )
    return plans


def _build_case_plans(root: Path) -> list[_CasePlan]:
    plans = [
        _registered_case_plan(
            root,
            case_id=case_id,
            use_case_id=use_case_id,
            task_id=task_id,
            seed=seed,
            expected_action_count=expected_action_count,
        )
        for case_id, use_case_id, task_id, seed, expected_action_count in _REGISTERED_CASES
    ]
    composed = _composed_case_plans(root)
    plans[1:1] = composed
    if [plan.case_id for plan in plans] != [
        "U01",
        "U02",
        "U03/E01",
        "U06-flow",
        "U06-electro",
        "U06-distillation",
        "U06-partition",
        "U06-crystallization",
    ]:
        raise DeterministicUseCaseProtocolError("frozen case order drifted")

    submitted = sum(len(plan.actions) for plan in plans)
    committed = sum(
        status == "committed" for plan in plans for status in plan.expected_transactions
    )
    rollbacks = sum(
        status == "rolled_back" for plan in plans for status in plan.expected_transactions
    )
    final_assays = sum(
        action.get("operation") == "measure" and action.get("instrument") == "final_assay"
        for plan in plans
        for action in plan.actions
    )
    denominators = (len(plans), submitted, committed, rollbacks, final_assays)
    expected = (
        EXPECTED_CASE_COUNT,
        EXPECTED_SUBMITTED_ACTION_COUNT,
        EXPECTED_COMMITTED_ACTION_COUNT,
        EXPECTED_ROLLBACK_COUNT,
        EXPECTED_FINAL_ASSAY_COUNT,
    )
    if denominators != expected:
        raise DeterministicUseCaseProtocolError(
            f"frozen denominators drifted: observed={denominators}, expected={expected}"
        )
    for plan in plans:
        observed_hash = _sha256_value(list(plan.actions))
        expected_hash = EXPECTED_ACTION_SHA256[plan.case_id]
        if observed_hash != expected_hash:
            raise DeterministicUseCaseProtocolError(
                f"frozen action list drifted for {plan.case_id}: {observed_hash}"
            )
    return plans


def build_case_specs(repository_root: str | Path) -> list[dict[str, Any]]:
    """Return the JSON-safe frozen case matrix without executing environments."""

    root = Path(repository_root).resolve()
    return [_case_spec(plan) for plan in _build_case_plans(root)]


def _case_spec(plan: _CasePlan) -> dict[str, Any]:
    return {
        "case_id": plan.case_id,
        "use_case_id": plan.use_case_id,
        "public_identity": plan.public_identity,
        "seed": plan.seed,
        "source_kind": plan.source_kind,
        "source_bindings": copy.deepcopy(list(plan.source_bindings)),
        "normalized_request": copy.deepcopy(plan.normalized_request),
        "normalized_task_contract": plan.task.to_dict(),
        "task_contract_hash": plan.task.contract_hash,
        "recipe_binding": copy.deepcopy(plan.recipe_binding),
        "submitted_action_count": len(plan.actions),
        "actions": copy.deepcopy(list(plan.actions)),
        "actions_sha256": _sha256_value(list(plan.actions)),
        "expected_validation": list(plan.expected_validation),
        "expected_transactions": list(plan.expected_transactions),
        "expected_failure": copy.deepcopy(plan.expected_failure),
    }


def _current_node_binding(
    root: Path,
    current: dict[str, Any],
    node_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dag = current.get("evidence_dag")
    nodes = dag.get("nodes") if isinstance(dag, Mapping) else None
    node = nodes.get(node_id) if isinstance(nodes, Mapping) else None
    if not isinstance(node, Mapping):
        raise DeterministicUseCaseProtocolError(f"current evidence node is missing: {node_id}")
    required_state = {
        "artifact_state": "current",
        "freshness": "fresh",
        "gate_state": "passed",
    }
    for field, expected in required_state.items():
        if node.get(field) != expected:
            raise DeterministicUseCaseProtocolError(
                f"current evidence node {node_id} has {field}={node.get(field)!r}"
            )
    relative_path = node.get("path")
    expected_sha = node.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
        raise DeterministicUseCaseProtocolError(
            f"current evidence node {node_id} has an incomplete path/SHA binding"
        )
    path = _resolve_repository_path(root, relative_path)
    if not path.is_file():
        raise DeterministicUseCaseProtocolError(
            f"current evidence artifact is missing for {node_id}: {relative_path}"
        )
    actual_sha = _sha256_path(path)
    if actual_sha != expected_sha:
        raise DeterministicUseCaseProtocolError(
            f"current evidence SHA drifted for {node_id}: {actual_sha} != {expected_sha}"
        )
    binding = {
        "node_id": node_id,
        "path": relative_path,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "artifact_state": node["artifact_state"],
        "freshness": node["freshness"],
        "gate_state": node["gate_state"],
        "binding_verified": True,
    }
    return binding, _read_json_object(path)


def resolve_existing_evidence(repository_root: str | Path) -> dict[str, Any]:
    """Resolve U04/U05 exclusively through ``configs/current.json`` and verify SHA."""

    root = Path(repository_root).resolve()
    current_path = _resolve_repository_path(root, CURRENT_CONFIG)
    current = _read_json_object(current_path)
    current_sha = _sha256_path(current_path)
    u04_binding, u04_report = _current_node_binding(root, current, _U04_NODE_ID)
    u05_binding, u05_report = _current_node_binding(root, current, _U05_NODE_ID)

    u04_passed = bool(
        u04_report.get("passed") is True
        and int(u04_report.get("pair_count", -1)) > 0
        and int(u04_report.get("trace_count", -1)) > 0
        and int(u04_report.get("provider_call_count", -1)) == 0
    )
    if not u04_passed:
        raise DeterministicUseCaseProtocolError("current U04 fork evidence is not passing")

    generated = u05_report.get("generated_qualification")
    raw_cases = generated.get("cases") if isinstance(generated, Mapping) else None
    matches = [
        row
        for row in raw_cases or []
        if isinstance(row, Mapping) and row.get("composition_id") == _U05_COMPOSITION_ID
    ]
    if len(matches) != 1:
        raise DeterministicUseCaseProtocolError(
            "current U05 composition evidence does not contain exactly one frozen first case"
        )
    u05_case = matches[0]
    u05_passed = bool(
        u05_report.get("status") == "passed"
        and u05_case.get("composition_request_sha256") == _U05_REQUEST_SHA256
        and int(u05_case.get("generation_seed", -1)) == _U05_GENERATION_SEED
        and int(u05_case.get("generation_index", -1)) == _U05_GENERATION_INDEX
        and u05_case.get("passed") is True
        and isinstance(u05_case.get("exact_replay"), Mapping)
        and u05_case["exact_replay"].get("verified") is True
    )
    if not u05_passed:
        raise DeterministicUseCaseProtocolError("current U05 frozen composition evidence drifted")

    return {
        "current_registry": {
            "path": CURRENT_CONFIG.as_posix(),
            "sha256": current_sha,
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
            "evidence_role": "frozen unseen generated composition, first generation row",
            "binding": u05_binding,
            "passed": True,
            "case_id": u05_case.get("case_id"),
            "composition_id": u05_case.get("composition_id"),
            "composition_request_sha256": u05_case.get("composition_request_sha256"),
            "generation_seed": int(u05_case["generation_seed"]),
            "generation_index": int(u05_case["generation_index"]),
            "action_count": int(u05_case.get("action_count", 0)),
            "exact_replay_verified": True,
        },
    }


def _finite_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_nonfinite_paths(item, child))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            paths.extend(_nonfinite_paths(item, child))
    elif not _finite_number(value):
        paths.append(prefix or "<root>")
    return paths


def _failure(
    failure_class: str,
    *,
    case_id: str,
    step: int | None = None,
    **details: Any,
) -> dict[str, Any]:
    row = {"class": failure_class, "case_id": case_id, **details}
    if step is not None:
        row["step"] = step
    return row


def _make_environment(plan: _CasePlan, resource_card: dict[str, Any]) -> gym.Env[Any, Any]:
    if plan.composition_request is not None:
        return gym.make(
            "ChemWorld",
            composition=copy.deepcopy(plan.composition_request),
            seed=plan.seed,
            campaign_resource_card=copy.deepcopy(resource_card),
        )
    return gym.make(
        plan.task.env_id,
        **plan.task.env_kwargs(seed=plan.seed),
        campaign_resource_card=copy.deepcopy(resource_card),
    )


def _contract_binding(plan: _CasePlan, task_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_contract_hash": task_info.get("task_contract_hash"),
        "expected_task_contract_hash": plan.task.contract_hash,
        "task_contract_hash_matches": (
            task_info.get("task_contract_hash") == plan.task.contract_hash
        ),
        "runtime_profile_hash": task_info.get("runtime_profile_hash"),
        "scoring_contract_hash": task_info.get("scoring_contract_hash"),
        "observation_contract_hash": task_info.get("observation_contract_hash"),
    }


def _execute_case(plan: _CasePlan, *, scratch_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    case_spec = _case_spec(plan)
    actions = copy.deepcopy(list(plan.actions))
    resource_card = _recipe_resource_card(
        plan.task,
        plan.seed,
        {"case_id": plan.case_id, "compiled_actions": actions},
    )
    resource_card["card_id"] = f"first-paper-deterministic-{plan.case_id.replace('/', '-') }"
    resource_card["metadata"] = {
        "qualification_id": QUALIFICATION_ID,
        "case_id": plan.case_id,
        "public_identity": plan.public_identity,
        "world_seed": plan.seed,
        "scope": "frozen_deterministic_use_case",
        "expected_rollback_count": sum(
            status == "rolled_back" for status in plan.expected_transactions
        ),
    }
    trajectory_path = scratch_dir / f"{plan.case_id.replace('/', '-')}.jsonl"
    env: gym.Env[Any, Any] | None = None
    failures: list[dict[str, Any]] = []
    leakage_findings: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []
    resource_steps: list[dict[str, Any]] = []
    evaluation_receipt: dict[str, Any] = {}
    rollback_recovery_receipt: dict[str, Any] | None = None
    post_termination_receipt: dict[str, Any] = {
        "validate_only": True,
        "passed": False,
        "failure": "terminate action was not reached",
    }
    exact_replay: dict[str, Any] = {
        "verified": False,
        "checked_steps": 0,
        "max_abs_error": None,
        "mismatches": [],
    }
    resource_receipt: dict[str, Any] = {
        "resource_reconciled": False,
        "reconciliation_mismatches": ["execution_did_not_complete"],
    }
    contract_binding: dict[str, Any] = {
        "task_contract_hash": None,
        "expected_task_contract_hash": plan.task.contract_hash,
        "task_contract_hash_matches": False,
        "runtime_profile_hash": None,
        "scoring_contract_hash": None,
        "observation_contract_hash": None,
    }
    initial_public_observation: dict[str, Any] = {}
    final_terminated = False
    final_truncated = False
    right_censored = False

    try:
        env = _make_environment(plan, resource_card)
        observation, reset_info = env.reset(seed=plan.seed)
        base: Any = env.unwrapped
        resource_before = base.campaign_resource_snapshot()
        task_info = base.task_info()
        contract_binding = _contract_binding(plan, task_info)
        missing_contract_hashes = [
            key
            for key in (
                "runtime_profile_hash",
                "scoring_contract_hash",
                "observation_contract_hash",
            )
            if not isinstance(contract_binding.get(key), str)
            or not contract_binding.get(key)
        ]
        if not contract_binding["task_contract_hash_matches"]:
            failures.append(_failure("task_contract_hash_mismatch", case_id=plan.case_id))
        if missing_contract_hashes:
            failures.append(
                _failure(
                    "contract_hash_missing",
                    case_id=plan.case_id,
                    fields=missing_contract_hashes,
                )
            )
        logging_task_info = {**task_info, **base.evaluator_provenance()}
        initial_public_observation = observation_to_json(observation)
        leakage_findings.extend(_leakage_findings(env, reset_info, "reset_info"))
        initial_view = agent_view_bundle(env, observation, {})
        leakage_findings.extend(
            _leakage_findings(env, initial_view, "initial_agent_view")
        )

        with TrajectoryLogger(trajectory_path) as logger:
            for step, action in enumerate(actions, start=1):
                expected_valid = plan.expected_validation[step - 1]
                expected_status = plan.expected_transactions[step - 1]
                step_failures: list[dict[str, Any]] = []
                state_before = _world_state_sections(env)
                rng_before = _rng_snapshot(env)
                step_resource_before = base.campaign_resource_snapshot()
                try:
                    validation = base.validate_action(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                except Exception as exc:  # preserve a receipt for every submitted action
                    step_failures.append(
                        _failure(
                            "action_execution_exception",
                            case_id=plan.case_id,
                            step=step,
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                    step_receipts.append(
                        {
                            "step": step,
                            "action": copy.deepcopy(action),
                            "action_sha256": _sha256_value(action),
                            "expected_validation": expected_valid,
                            "expected_transaction_status": expected_status,
                            "schema_validation": None,
                            "transaction": None,
                            "constitution_checks": [],
                            "world_events": [],
                            "resource_preflight": None,
                            "resource_outcome_delta": None,
                            "resource_reconciliation": None,
                            "public_observation": None,
                            "public_observation_sha256": None,
                            "leakage_findings": [],
                            "passed": False,
                            "failures": step_failures,
                        }
                    )
                    failures.extend(step_failures)
                    continue

                state_after = _world_state_sections(env)
                step_resource_after = base.campaign_resource_snapshot()
                public_resource_state = base.public_campaign_resource_state()
                rng_preserved = _rng_snapshot(env) == rng_before
                observed_valid = bool(validation.get("valid"))
                transaction_status = info.get("transaction_status")
                committed = transaction_status == "committed"
                preflight = copy.deepcopy(info.get("campaign_resource_preflight"))
                outcome_delta = copy.deepcopy(info.get("campaign_resource_outcome_delta"))
                resource_step = {
                    "step": step,
                    "operation": action.get("operation"),
                    "instrument": action.get("instrument"),
                    "transaction_status": transaction_status,
                    "operation_committed": committed,
                    "preflight": preflight,
                    "outcome_delta": outcome_delta,
                }
                resource_steps.append(resource_step)
                step_resource = _resource_receipt_summary(
                    [resource_step],
                    before_snapshot=step_resource_before,
                    after_snapshot=step_resource_after,
                    public_state=public_resource_state,
                )
                step_resource.pop("step_receipts", None)

                if observed_valid is not expected_valid:
                    step_failures.append(
                        _failure(
                            "validation_outcome_mismatch",
                            case_id=plan.case_id,
                            step=step,
                            expected=expected_valid,
                            observed=observed_valid,
                        )
                    )
                if transaction_status != expected_status:
                    step_failures.append(
                        _failure(
                            "transaction_outcome_mismatch",
                            case_id=plan.case_id,
                            step=step,
                            expected=expected_status,
                            observed=transaction_status,
                        )
                    )
                if not isinstance(preflight, Mapping) or not isinstance(outcome_delta, Mapping):
                    step_failures.append(
                        _failure(
                            "campaign_resource_receipt_missing",
                            case_id=plan.case_id,
                            step=step,
                        )
                    )
                if step_resource.get("resource_reconciled") is not True:
                    step_failures.append(
                        _failure(
                            "step_resource_reconciliation_failed",
                            case_id=plan.case_id,
                            step=step,
                            mismatches=step_resource.get("reconciliation_mismatches", []),
                        )
                    )

                raw_checks = info.get("constitution_checks", [])
                constitution_checks = (
                    copy.deepcopy(raw_checks) if isinstance(raw_checks, list) else []
                )
                failed_checks = [
                    check
                    for check in constitution_checks
                    if isinstance(check, Mapping) and check.get("passed") is False
                ]
                if failed_checks:
                    step_failures.append(
                        _failure(
                            "constitution_check_failed",
                            case_id=plan.case_id,
                            step=step,
                            check_names=sorted(
                                str(check.get("name")) for check in failed_checks
                            ),
                        )
                    )
                raw_events = info.get("world_events", [])
                world_events = copy.deepcopy(raw_events) if isinstance(raw_events, list) else []
                event_matches = any(
                    isinstance(event, Mapping)
                    and event.get("operation_type") == action.get("operation")
                    and (
                        expected_status != "committed"
                        or event.get("event_type") == "operation_applied"
                    )
                    for event in world_events
                )
                if not event_matches:
                    step_failures.append(
                        _failure(
                            "world_event_propagation_failed",
                            case_id=plan.case_id,
                            step=step,
                        )
                    )

                public_observation = observation_to_json(observation)
                nonfinite = _nonfinite_paths(public_observation)
                if nonfinite or not math.isfinite(float(reward)):
                    step_failures.append(
                        _failure(
                            "nonfinite_measurement",
                            case_id=plan.case_id,
                            step=step,
                            fields=nonfinite,
                            reward_finite=math.isfinite(float(reward)),
                        )
                    )
                public_view = agent_view_bundle(env, observation, info)
                step_leakage = _leakage_findings(
                    env,
                    public_view,
                    f"{plan.case_id}.step-{step}",
                )
                leakage_findings.extend(step_leakage)
                if step_leakage:
                    step_failures.append(
                        _failure(
                            "public_private_leakage",
                            case_id=plan.case_id,
                            step=step,
                            finding_count=len(step_leakage),
                        )
                    )

                ghost_receipt: dict[str, Any] | None = None
                if expected_status == "rolled_back":
                    ghost_receipt = _negative_ghost_state_receipt(
                        action=action,
                        info=info,
                        state_before=state_before,
                        state_after=state_after,
                        rng_preserved=rng_preserved,
                        resource_before=step_resource_before,
                        resource_after=step_resource_after,
                        public_resource_state=public_resource_state,
                    )
                    rollback_recovery_receipt = ghost_receipt
                    expected_failure = plan.expected_failure or {}
                    if info.get("rollback_reason") != expected_failure.get("rollback_reason"):
                        step_failures.append(
                            _failure(
                                "rollback_reason_mismatch",
                                case_id=plan.case_id,
                                step=step,
                                expected=expected_failure.get("rollback_reason"),
                                observed=info.get("rollback_reason"),
                            )
                        )
                    ghost_checks = {
                        "physical_state_preserved": ghost_receipt.get("physical", {}).get(
                            "preserved"
                        ),
                        "observation_rng_preserved": ghost_receipt.get(
                            "observation_rng", {}
                        ).get("preserved"),
                        "ledger_penalty_reconciled": ghost_receipt.get("ledger", {}).get(
                            "ghost_state_preserved"
                        ),
                        "process_penalty_reconciled": ghost_receipt.get("process", {}).get(
                            "ghost_state_preserved"
                        ),
                        "failure_events_reconciled": ghost_receipt.get("events", {}).get(
                            "reconciled"
                        ),
                        "resource_reconciled": ghost_receipt.get("resource", {}).get(
                            "resource_reconciled"
                        ),
                    }
                    for check_name, passed in ghost_checks.items():
                        if passed is not True:
                            step_failures.append(
                                _failure(
                                    f"rollback_{check_name}_failed",
                                    case_id=plan.case_id,
                                    step=step,
                                )
                            )

                if action.get("operation") == "terminate" and committed:
                    post_termination_receipt = _post_termination_validation_receipt(base, actions)
                if (
                    action.get("operation") == "measure"
                    and action.get("instrument") == "final_assay"
                ):
                    final_terminated = bool(terminated)
                    final_truncated = bool(truncated)
                    evaluation_receipt = {
                        "reward": float(reward),
                        "environment_reward": info.get("environment_reward"),
                        "observed_reward": info.get("observed_reward"),
                        "leaderboard_score": info.get("leaderboard_score"),
                        "reward_source": info.get("reward_source"),
                        "experiment_completed": info.get("experiment_completed"),
                        "experiment_ended": info.get("experiment_ended"),
                        "transaction_status": transaction_status,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "public_observation": public_observation,
                    }
                if (terminated or truncated) and step != len(actions):
                    step_failures.append(
                        _failure(
                            "lifecycle_ended_before_frozen_path",
                            case_id=plan.case_id,
                            step=step,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                        )
                    )
                right_censored = right_censored or bool(
                    info.get("right_censored_open_batch", False)
                )

                logger.log(
                    task_info=logging_task_info,
                    step=step,
                    action=action,
                    observation=public_observation,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    agent_metadata={
                        "agent_id": "frozen_deterministic_use_case",
                        "policy_randomness": "none",
                        "case_id": plan.case_id,
                    },
                    agent_view=public_view,
                )
                receipt = {
                    "step": step,
                    "action": copy.deepcopy(action),
                    "action_sha256": _sha256_value(action),
                    "expected_validation": expected_valid,
                    "expected_transaction_status": expected_status,
                    "schema_validation": {
                        "valid": observed_valid,
                        "invalid_reasons": copy.deepcopy(
                            validation.get("invalid_reasons", [])
                        ),
                        "preconditions": copy.deepcopy(validation.get("preconditions", {})),
                        "canonical_action": copy.deepcopy(
                            validation.get("canonical_action")
                        ),
                        "canonical_action_sha256": _sha256_value(
                            validation.get("canonical_action")
                        ),
                    },
                    "transaction": {
                        "status": transaction_status,
                        "rollback_reason": info.get("rollback_reason"),
                        "operation_committed": committed,
                        "preconditions": copy.deepcopy(info.get("preconditions", {})),
                    },
                    "transaction_status": transaction_status,
                    "rollback_reason": info.get("rollback_reason"),
                    "operation_committed": committed,
                    "state_transition": {
                        "state_delta_summary": copy.deepcopy(
                            info.get("state_delta_summary", {})
                        ),
                        "state_patches_summary": copy.deepcopy(
                            info.get("state_patches_summary", [])
                        ),
                        "before_sha256": _sha256_value(state_before),
                        "after_sha256": _sha256_value(state_after),
                    },
                    "constitution_checks": constitution_checks,
                    "constitution_failed_check_names": sorted(
                        str(check.get("name")) for check in failed_checks
                    ),
                    "world_events": world_events,
                    "event_propagation_matches_operation": event_matches,
                    "resource_preflight": preflight,
                    "resource_outcome_delta": outcome_delta,
                    "resource_outcome": outcome_delta,
                    "resource_reconciliation": step_resource,
                    "rollback_recovery_receipt": ghost_receipt,
                    "ghost_state": ghost_receipt,
                    "public_observation": public_observation,
                    "public_observation_sha256": _sha256_value(public_observation),
                    "agent_view_sha256": _sha256_value(public_view),
                    "leakage_findings": step_leakage,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "passed": not step_failures,
                    "failures": step_failures,
                }
                step_receipts.append(receipt)
                failures.extend(step_failures)

        resource_after = base.campaign_resource_snapshot()
        public_resource_state = base.public_campaign_resource_state()
        resource_receipt = _resource_receipt_summary(
            resource_steps,
            before_snapshot=resource_before,
            after_snapshot=resource_after,
            public_state=public_resource_state,
        )
        resource_receipt.pop("step_receipts", None)
        if resource_receipt.get("resource_reconciled") is not True:
            failures.append(
                _failure(
                    "case_resource_reconciliation_failed",
                    case_id=plan.case_id,
                    mismatches=resource_receipt.get("reconciliation_mismatches", []),
                )
            )
        if post_termination_receipt.get("passed") is not True:
            failures.append(
                _failure(
                    "post_termination_validate_only_rejection_failed",
                    case_id=plan.case_id,
                    receipt=post_termination_receipt,
                )
            )
        records = load_jsonl(trajectory_path)
        if records:
            exact_replay = verify_records(records, tolerance=0.0).to_dict()
        if exact_replay.get("verified") is not True:
            failures.append(
                _failure(
                    "exact_replay_failed",
                    case_id=plan.case_id,
                    mismatch_count=len(exact_replay.get("mismatches", [])),
                )
            )
    except Exception as exc:
        failures.append(
            _failure(
                "case_execution_exception",
                case_id=plan.case_id,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
        )
    finally:
        if env is not None:
            env.close()

    committed_count = sum(
        receipt.get("transaction", {}).get("status") == "committed"
        for receipt in step_receipts
        if isinstance(receipt.get("transaction"), Mapping)
    )
    rollback_count = sum(
        receipt.get("transaction", {}).get("status") == "rolled_back"
        for receipt in step_receipts
        if isinstance(receipt.get("transaction"), Mapping)
    )
    final_assay_count = sum(
        receipt.get("action", {}).get("operation") == "measure"
        and receipt.get("action", {}).get("instrument") == "final_assay"
        and receipt.get("transaction", {}).get("status") == "committed"
        for receipt in step_receipts
        if isinstance(receipt.get("transaction"), Mapping)
    )
    terminate_count = sum(
        receipt.get("action", {}).get("operation") == "terminate"
        and receipt.get("transaction", {}).get("status") == "committed"
        for receipt in step_receipts
        if isinstance(receipt.get("transaction"), Mapping)
    )
    lifecycle_closed = bool(
        len(step_receipts) == len(actions)
        and final_assay_count == 1
        and terminate_count == 1
        and final_terminated
        and not final_truncated
        and not right_censored
    )
    if not lifecycle_closed:
        failures.append(
            _failure(
                "lifecycle_not_closed",
                case_id=plan.case_id,
                step_receipt_count=len(step_receipts),
                submitted_action_count=len(actions),
                committed_final_assay_count=final_assay_count,
                committed_terminate_count=terminate_count,
                final_terminated=final_terminated,
                final_truncated=final_truncated,
                right_censored=right_censored,
            )
        )
    if leakage_findings and not any(
        failure["class"] == "public_private_leakage" for failure in failures
    ):
        failures.append(
            _failure(
                "public_private_leakage",
                case_id=plan.case_id,
                finding_count=len(leakage_findings),
            )
        )

    expected_rollback_step = (
        int(plan.expected_failure["step"]) if plan.expected_failure is not None else None
    )
    recovery_receipt = {
        "applicable": expected_rollback_step is not None,
        "expected_rollback_step": expected_rollback_step,
        "observed_rollback_count": rollback_count,
        "subsequent_expected_commit_count": (
            len(actions) - expected_rollback_step if expected_rollback_step is not None else 0
        ),
        "subsequent_observed_commit_count": (
            sum(
                receipt.get("step", 0) > expected_rollback_step
                and receipt.get("transaction", {}).get("status") == "committed"
                for receipt in step_receipts
                if expected_rollback_step is not None
                and isinstance(receipt.get("transaction"), Mapping)
            )
            if expected_rollback_step is not None
            else 0
        ),
        "rollback_recovery_receipt": rollback_recovery_receipt,
    }
    recovery_receipt["passed"] = (
        True
        if expected_rollback_step is None
        else bool(
            rollback_count == 1
            and recovery_receipt["subsequent_observed_commit_count"]
            == recovery_receipt["subsequent_expected_commit_count"]
            and isinstance(rollback_recovery_receipt, Mapping)
            and rollback_recovery_receipt.get("ghost_state_preserved") is True
        )
    )
    if recovery_receipt["passed"] is not True:
        failures.append(
            _failure("rollback_recovery_path_failed", case_id=plan.case_id)
        )

    elapsed = time.perf_counter() - started
    case_report = {
        **case_spec,
        "identity": plan.public_identity,
        "action_list_sha256": case_spec["actions_sha256"],
        "contract_binding": contract_binding,
        "resource_card": resource_card,
        "initial_public_observation": initial_public_observation,
        "checked_action_count": len(step_receipts),
        "committed_action_count": committed_count,
        "rollback_count": rollback_count,
        "rolled_back_action_count": rollback_count,
        "committed_final_assay_count": final_assay_count,
        "final_assay_count": final_assay_count,
        "step_receipts": step_receipts,
        "resource_receipt": resource_receipt,
        "recovery_receipt": recovery_receipt,
        "termination_receipt": {
            "closed": lifecycle_closed,
            "committed_terminate_count": terminate_count,
            "committed_final_assay_count": final_assay_count,
            "final_terminated": final_terminated,
            "final_truncated": final_truncated,
            "terminated": final_terminated,
            "truncated": final_truncated,
            "right_censored_open_batch": right_censored,
            "post_termination_validation": post_termination_receipt,
        },
        "evaluation_receipt": evaluation_receipt,
        "leakage_findings": leakage_findings,
        "public_private_leakage_count": len(leakage_findings),
        "exact_replay": exact_replay,
        "provider_call_count": 0,
        "elapsed_s": elapsed,
        "trajectory_bytes": (
            trajectory_path.stat().st_size if trajectory_path.exists() else 0
        ),
        "passed": not failures,
        "failures": failures,
    }
    return case_report


def _receipt_completeness_errors(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def missing(case_id: str, requirement: str, *, step: int | None = None) -> None:
        row: dict[str, Any] = {"case_id": case_id, "requirement": requirement}
        if step is not None:
            row["step"] = step
        errors.append(row)

    required_step_fields = {
        "action",
        "action_sha256",
        "expected_validation",
        "expected_transaction_status",
        "schema_validation",
        "transaction",
        "constitution_checks",
        "world_events",
        "resource_preflight",
        "resource_outcome_delta",
        "resource_reconciliation",
        "public_observation",
        "leakage_findings",
        "failures",
    }
    for case in cases:
        case_id = str(case.get("case_id"))
        receipts = case.get("step_receipts", [])
        if not isinstance(receipts, list) or len(receipts) != int(
            case.get("submitted_action_count", -1)
        ):
            missing(case_id, "one_step_receipt_per_submitted_action")
            receipts = receipts if isinstance(receipts, list) else []
        for expected_step, receipt in enumerate(receipts, start=1):
            if not isinstance(receipt, Mapping):
                missing(case_id, "step_receipt_object", step=expected_step)
                continue
            for field in sorted(required_step_fields - set(receipt)):
                missing(case_id, field, step=expected_step)
            if receipt.get("step") != expected_step:
                missing(case_id, "contiguous_step_number", step=expected_step)
            resource = receipt.get("resource_reconciliation")
            if not isinstance(resource, Mapping) or resource.get("resource_reconciled") is not True:
                missing(case_id, "step_resource_reconciliation", step=expected_step)
            if not isinstance(receipt.get("constitution_checks"), list):
                missing(case_id, "constitution_checks_list", step=expected_step)
            if not isinstance(receipt.get("world_events"), list):
                missing(case_id, "world_events_list", step=expected_step)
            if receipt.get("expected_transaction_status") == "rolled_back":
                ghost = receipt.get("rollback_recovery_receipt")
                if not isinstance(ghost, Mapping) or ghost.get("ghost_state_preserved") is not True:
                    missing(case_id, "expected_rollback_ghost_state_receipt", step=expected_step)
        resource = case.get("resource_receipt")
        if not isinstance(resource, Mapping) or resource.get("resource_reconciled") is not True:
            missing(case_id, "case_resource_receipt")
        termination = case.get("termination_receipt")
        if not isinstance(termination, Mapping) or termination.get("closed") is not True:
            missing(case_id, "lifecycle_closure")
        elif not isinstance(termination.get("post_termination_validation"), Mapping) or (
            termination["post_termination_validation"].get("passed") is not True
        ):
            missing(case_id, "post_termination_validation")
        if not case.get("evaluation_receipt"):
            missing(case_id, "evaluation_receipt")
        replay = case.get("exact_replay")
        if not isinstance(replay, Mapping) or replay.get("verified") is not True:
            missing(case_id, "exact_replay")
        if int(case.get("trajectory_bytes", 0)) <= 0:
            missing(case_id, "trajectory_bytes")
        if int(case.get("public_private_leakage_count", -1)) != 0:
            missing(case_id, "zero_public_private_leakage")
        for field in (
            "task_contract_hash",
            "runtime_profile_hash",
            "scoring_contract_hash",
            "observation_contract_hash",
        ):
            binding = case.get("contract_binding", {})
            if not isinstance(binding, Mapping) or not isinstance(binding.get(field), str):
                missing(case_id, field)
    return errors


def _failure_class_counts(
    cases: list[dict[str, Any]],
    completeness_errors: list[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        for failure in case.get("failures", []):
            if isinstance(failure, Mapping):
                counts[str(failure.get("class", "unclassified"))] += 1
            else:
                counts[str(failure)] += 1
    if completeness_errors:
        counts["missing_receipt"] += len(completeness_errors)
    return dict(sorted(counts.items()))


def _build_report_with_scratch(
    *,
    root: Path,
    scratch: Path,
    source_binding: dict[str, Any],
    existing_evidence: dict[str, Any],
    plans: list[_CasePlan],
) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=True)
    cases = [_execute_case(plan, scratch_dir=scratch) for plan in plans]
    completeness_errors = _receipt_completeness_errors(cases)
    failure_counts = _failure_class_counts(cases, completeness_errors)
    submitted = sum(int(case["submitted_action_count"]) for case in cases)
    checked = sum(int(case["checked_action_count"]) for case in cases)
    committed = sum(int(case["committed_action_count"]) for case in cases)
    rollbacks = sum(int(case["rollback_count"]) for case in cases)
    final_assays = sum(int(case["committed_final_assay_count"]) for case in cases)
    leakage_count = sum(int(case["public_private_leakage_count"]) for case in cases)
    passed_cases = sum(case.get("passed") is True for case in cases)
    provider_call_count = 0
    exact_counts = bool(
        len(cases) == passed_cases == EXPECTED_CASE_COUNT
        and submitted == checked == EXPECTED_SUBMITTED_ACTION_COUNT
        and committed == EXPECTED_COMMITTED_ACTION_COUNT
        and rollbacks == EXPECTED_ROLLBACK_COUNT
        and final_assays == EXPECTED_FINAL_ASSAY_COUNT
    )
    overall_pass = bool(
        exact_counts
        and leakage_count == 0
        and provider_call_count == 0
        and existing_evidence["U04"]["passed"] is True
        and existing_evidence["U05"]["passed"] is True
        and not completeness_errors
        and not failure_counts
    )
    all_failures = [
        copy.deepcopy(failure)
        for case in cases
        for failure in case.get("failures", [])
    ]
    all_failures.extend(
        {
            "class": "missing_receipt",
            **copy.deepcopy(error),
        }
        for error in completeness_errors
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "status": "passed" if overall_pass else "failed",
        "source_binding": source_binding,
        "claim_boundary": [
            "Deterministic virtual-instrument use-case qualification only.",
            "The eight frozen cases are examples, not a benchmark or exhaustive task-space claim.",
            (
                "No provider was called and no agent-intelligence or "
                "physical-laboratory claim is made."
            ),
        ],
        "counting_rule": {
            "independent_unit": "one frozen deterministic use case",
            "action_monitoring": (
                "complete census of all 89 submitted actions; no sampling pass gate"
            ),
            "statistics": "deterministic exact counts only",
        },
        "provider_call_count": provider_call_count,
        "denominators": {
            "case_count": EXPECTED_CASE_COUNT,
            "submitted_action_count": EXPECTED_SUBMITTED_ACTION_COUNT,
            "committed_action_count": EXPECTED_COMMITTED_ACTION_COUNT,
            "rolled_back_action_count": EXPECTED_ROLLBACK_COUNT,
            "final_assay_count": EXPECTED_FINAL_ASSAY_COUNT,
        },
        "summary": {
            "cases": {"passed": passed_cases, "denominator": len(cases)},
            "submitted_actions": {
                "checked": checked,
                "denominator": submitted,
                "expected": EXPECTED_SUBMITTED_ACTION_COUNT,
            },
            "committed_actions": {
                "observed": committed,
                "expected": EXPECTED_COMMITTED_ACTION_COUNT,
            },
            "rolled_back_actions": {
                "observed": rollbacks,
                "expected": EXPECTED_ROLLBACK_COUNT,
            },
            "committed_final_assays": {
                "observed": final_assays,
                "expected": EXPECTED_FINAL_ASSAY_COUNT,
            },
            "public_private_leakage_count": leakage_count,
            "missing_receipt_count": len(completeness_errors),
            "failure_class_counts": failure_counts,
            "exact_denominators_passed": exact_counts,
        },
        "receipt_completeness": {
            "passed": not completeness_errors,
            "error_count": len(completeness_errors),
            "errors": completeness_errors,
        },
        "existing_evidence": existing_evidence,
        "cases": cases,
        "failures": all_failures,
    }


def build_report(
    *,
    repository_root: str | Path,
    require_clean: bool = True,
    scratch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen eight-case block and return a complete machine report."""

    root = Path(repository_root).resolve()
    source_binding = validate_launch_preconditions(root, require_clean=require_clean)
    existing_evidence = resolve_existing_evidence(root)
    plans = _build_case_plans(root)
    if scratch_dir is not None:
        return _build_report_with_scratch(
            root=root,
            scratch=Path(scratch_dir).resolve(),
            source_binding=source_binding,
            existing_evidence=existing_evidence,
            plans=plans,
        )
    with tempfile.TemporaryDirectory(prefix="chemworld-deterministic-use-cases-") as tmp:
        return _build_report_with_scratch(
            root=root,
            scratch=Path(tmp),
            source_binding=source_binding,
            existing_evidence=existing_evidence,
            plans=plans,
        )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# First-paper deterministic use cases",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "## Exact full-census counts",
        "",
        "| Quantity | Observed/checked | Denominator/expected |",
        "| --- | ---: | ---: |",
        f"| Cases passed | {summary['cases']['passed']} | {summary['cases']['denominator']} |",
        (
            f"| Submitted actions checked | {summary['submitted_actions']['checked']} | "
            f"{summary['submitted_actions']['denominator']} |"
        ),
        (
            f"| Committed actions | {summary['committed_actions']['observed']} | "
            f"{summary['committed_actions']['expected']} |"
        ),
        (
            f"| Rolled-back actions | {summary['rolled_back_actions']['observed']} | "
            f"{summary['rolled_back_actions']['expected']} |"
        ),
        (
            f"| Committed final assays | {summary['committed_final_assays']['observed']} | "
            f"{summary['committed_final_assays']['expected']} |"
        ),
        "",
        "All submitted actions are inspected. Sampling is not used as a qualification gate.",
        "",
        "## Case results",
        "",
        (
            "| Case | Public identity | Seed | Actions | Commit | Rollback | "
            "Final assay | Replay | Resource | Leakage | Status |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | `{case['public_identity']}` | {case['seed']} | "
            f"{case['checked_action_count']}/{case['submitted_action_count']} | "
            f"{case['committed_action_count']} | {case['rollback_count']} | "
            f"{case['committed_final_assay_count']} | "
            f"{'pass' if case['exact_replay']['verified'] else 'fail'} | "
            f"{'pass' if case['resource_receipt'].get('resource_reconciled') else 'fail'} | "
            f"{case['public_private_leakage_count']} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    u03 = next(case for case in report["cases"] if case["case_id"] == "U03/E01")
    rollback = u03["recovery_receipt"]
    lines.extend(
        [
            "",
            "## U03 failure and recovery",
            "",
            f"Expected rollback step: `{rollback['expected_rollback_step']}`; "
            f"observed rollbacks: `{rollback['observed_rollback_count']}`; "
            f"subsequent commits: `{rollback['subsequent_observed_commit_count']}/"
            f"{rollback['subsequent_expected_commit_count']}`; "
            f"receipt: `{'passed' if rollback['passed'] else 'failed'}`.",
            "",
            "## Existing evidence reused through current bindings",
            "",
            "| Use case | Evidence | Binding SHA verified | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for use_case_id in ("U04", "U05"):
        evidence = report["existing_evidence"][use_case_id]
        lines.append(
            f"| {use_case_id} | {evidence['evidence_role']} | "
            f"{'yes' if evidence['binding']['binding_verified'] else 'no'} | "
            f"{'PASS' if evidence['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Provider calls: `{report['provider_call_count']}`.",
            f"Public/private leakage findings: `{summary['public_private_leakage_count']}`.",
            f"Missing receipts: `{summary['missing_receipt_count']}`.",
            "",
            "## Failure classes",
            "",
        ]
    )
    failures = summary["failure_class_counts"]
    if failures:
        lines.extend(f"- `{name}`: {count}" for name, count in failures.items())
    else:
        lines.append("None.")
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any],
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
    "EXPECTED_ACTION_SHA256",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_COMMITTED_ACTION_COUNT",
    "EXPECTED_FINAL_ASSAY_COUNT",
    "EXPECTED_ROLLBACK_COUNT",
    "EXPECTED_SUBMITTED_ACTION_COUNT",
    "QUALIFICATION_ID",
    "REPORT_SCHEMA_VERSION",
    "DeterministicUseCaseProtocolError",
    "build_case_specs",
    "build_report",
    "render_markdown",
    "resolve_existing_evidence",
    "validate_launch_preconditions",
    "write_outputs",
]
