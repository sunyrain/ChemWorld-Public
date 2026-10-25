"""Host-owned STDIO MCP bridge for one interactive ChemWorld experiment.

The process is launched by Codex as a configured MCP server, outside the
model-generated shell-command path.  It exposes only bounded public state and
file-IPC requests; the authoritative environment remains in the parent runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemworld.agents.interaction import DecisionAuditRecord
from chemworld.agents.public_history import public_history_page
from chemworld.eval.work_ii_prior_discovery import (
    WORK_II_LAW_BASES,
    WORK_II_LAW_LINKS,
    WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
    WORK_II_SNAPSHOT_SCHEMA_VERSION,
    parse_work_ii_belief_snapshot,
    parse_work_ii_belief_snapshot_header,
    parse_work_ii_law_summary,
    parse_work_ii_prediction_page,
)

MCP_SERVER_VERSION = "chemworld-experiment-codex-mcp-0.13"
IPC_VERSION = "chemworld-experiment-codex-ipc-0.2"
CAMPAIGN_PROGRESS_VERSION = "chemworld-campaign-progress-0.2"
SERVER_NAME = "chemworld_lab"
SUPPORTED_TOOLS = (
    "material_information",
    "status",
    "history",
    "inspect_artifact",
    "belief_snapshot_status",
    "commit_belief_snapshot",
    "terminal_action_readout",
    "commit_final_recommendation",
    "step",
)
ATOMIC_REPLACE_RETRY_LIMIT = 40
ATOMIC_REPLACE_RETRY_INTERVAL_S = 0.025

BELIEF_SNAPSHOT_SHAPE_GUIDE = (
    "Required nested argument shape (names are literal; brackets mean JSON arrays): "
    "snapshot={schema_version,snapshot_id,stage,"
    "prior_assessment={nominal_information_available,reliability_probability,"
    "suspected_misindexed_fields,rationale},"
    "predictions:[{query_id,metrics:[{metric_id,mean,interval_lower,interval_upper,"
    "confidence}]}],"
    "law_summary={schema_version,summary_id,feature_ids,"
    "metric_laws:[{metric_id,intercept,link,lower_bound,upper_bound,"
    "terms:[{term_id,basis,input_ids,coefficient,category_value?}]}],"
    "evidence_ids,applicability,limitations,confidence},"
    "evidence_ids,next_experiment_intent,overall_confidence}. "
    "predictions and metrics are both arrays; do not flatten metric names or values into a "
    "prediction object. metric_laws is an array; each law uses metric_id plus executable "
    "numeric fields, not metric/relation prose. An initial law may use terms:[]; when terms "
    "are present, input_ids is an array. Use the exact required IDs, counts, enums, and types "
    "from inputSchema."
)

STAGED_BELIEF_SNAPSHOT_GUIDE = (
    "Use action=begin, then append_prediction_page and append_law_page in the exact "
    "host-published page order, then action=finalize. Partial drafts never count as "
    "checkpoints and step remains blocked until finalize. The staged inputSchema is the sole "
    "participant-facing authority. The host assembles snapshot.schema_version="
    f"{WORK_II_SNAPSHOT_SCHEMA_VERSION}; submit snapshot_header.law_summary.schema_version="
    f"{WORK_II_LAW_SUMMARY_SCHEMA_VERSION} exactly."
)

BELIEF_DRAFT_VERSION = "chemworld-work-ii-belief-snapshot-draft-0.1"
PREDICTION_PAGE_SIZE = 4
LAW_PAGE_SIZE = 2


def _encode(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_encode(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once(path: Path, value: Any) -> None:
    """Durably create one immutable protocol artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_encode(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Retry only transient sharing violations; preserve fail-closed semantics."""

    for retry in range(ATOMIC_REPLACE_RETRY_LIMIT + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if retry >= ATOMIC_REPLACE_RETRY_LIMIT:
                raise
            time.sleep(ATOMIC_REPLACE_RETRY_INTERVAL_S)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_encode(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


class ChemWorldMCPServer:
    """Minimal, dependency-free MCP server with a deliberately narrow surface."""

    def _free_research(self, descriptor: dict[str, Any]) -> bool:
        if descriptor.get("session_scope") != "campaign":
            return False
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        return contract.get("free_research") is True

    def _knowledge_delivery(self, descriptor: dict[str, Any]) -> bool:
        if not self._free_research(descriptor):
            return False
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        return contract.get("final_recommendation_required") is False

    def __init__(self, workspace: Path) -> None:
        self.root = workspace.resolve(strict=True)
        self.agent = (self.root / "agent").resolve(strict=True)
        self.public = (self.root / "public").resolve(strict=True)
        self.reference = (self.root / "reference").resolve(strict=True)
        self.ipc = (self.root / ".ipc").resolve(strict=True)
        self._terminal_outcome: dict[str, Any] | None = None
        for path in (self.agent, self.public, self.reference, self.ipc):
            path.relative_to(self.root)

    def run(self) -> int:
        for raw_line in sys.stdin.buffer:
            if not raw_line.strip():
                continue
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("JSON-RPC request must be an object")
                response = self._dispatch(request)
            except Exception as error:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": type(error).__name__,
                    },
                }
            if response is not None:
                sys.stdout.buffer.write(_encode(response) + b"\n")
                sys.stdout.buffer.flush()
        return 0

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "notifications/initialized" or str(method).startswith("notifications/"):
            return None
        if method == "initialize":
            params = request.get("params")
            protocol = params.get("protocolVersion") if isinstance(params, dict) else "2025-06-18"
            descriptor = self._descriptor()
            campaign = descriptor.get("session_scope") == "campaign"
            action_readout = descriptor.get("terminal_action_readout_required") is True
            return self._result(
                request_id,
                {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": MCP_SERVER_VERSION},
                    "instructions": (
                        "Use chemworld_lab.step for every physical operation. First call "
                        "material_information once. Use status, history, and inspect_artifact "
                        "only for bounded public evidence. Never fabricate an outcome. The "
                        "inputSchema values returned by tools/list are the sole executable "
                        "authority for tool arguments; do not try to read or infer schemas from "
                        "host filesystem paths. "
                        + (
                            "Commit every required belief checkpoint with "
                            "commit_belief_snapshot. An experiment_ended outcome closes only "
                            "the current batch; continue in the same session when "
                            "campaign_ended=false. After campaign_ended=true, commit the final "
                            "checkpoint if due, "
                            + (
                                "call terminal_action_readout to reveal the fixed held-out "
                                "candidate operations, then rank and commit exactly one selection "
                                "with commit_final_recommendation, "
                                if action_readout
                                else "submit the short research handoff without selecting a batch, "
                                if self._knowledge_delivery(descriptor)
                                else "commit exactly one participant-owned completed-batch "
                                "selection with commit_final_recommendation, "
                            )
                            + "then submit exactly one JSON object "
                            "matching the final response schema, with no prose or Markdown fence."
                            if campaign
                            else "After a step returns experiment_ended=true, call no more "
                            "tools and submit the final response for that experiment."
                        )
                    ),
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "logging/setLevel":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": self._tool_definitions()})
        if method == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict):
                return self._error(request_id, -32602, "invalid tools/call params")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return self._error(request_id, -32602, "invalid tools/call params")
            return self._result(request_id, self._call_tool(name, arguments))
        return self._error(request_id, -32601, "method not found")

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def _descriptor(self) -> dict[str, Any]:
        descriptor = _read_object(self.ipc / "active_session.json")
        if descriptor.get("schema_version") != IPC_VERSION:
            raise ValueError("unsupported active experiment session")
        return descriptor

    @staticmethod
    def _page_plan(contract: dict[str, Any]) -> dict[str, Any]:
        configured = contract.get("snapshot_submission_protocol")
        if isinstance(configured, dict):
            prediction_pages = configured.get("prediction_pages")
            law_pages = configured.get("law_pages")
            submission_order = configured.get("submission_order")
        else:
            query_contract = contract.get("query_metric_contract", {})
            query_ids = list(query_contract) if isinstance(query_contract, dict) else []
            prediction_pages = [
                {
                    "page_id": f"predictions-{index // PREDICTION_PAGE_SIZE + 1:03d}",
                    "query_metric_contract": {
                        query_id: query_contract[query_id]
                        for query_id in query_ids[index : index + PREDICTION_PAGE_SIZE]
                    },
                }
                for index in range(0, len(query_ids), PREDICTION_PAGE_SIZE)
            ]
            metrics = contract.get("allowed_metric_ids", [])
            metrics = metrics if isinstance(metrics, list) else []
            law_pages = [
                {
                    "page_id": f"laws-{index // LAW_PAGE_SIZE + 1:03d}",
                    "metric_ids": metrics[index : index + LAW_PAGE_SIZE],
                }
                for index in range(0, len(metrics), LAW_PAGE_SIZE)
            ]
            submission_order = [page["page_id"] for page in (*prediction_pages, *law_pages)]
        if not isinstance(prediction_pages, list) or not isinstance(law_pages, list):
            raise ValueError("belief snapshot page plan is invalid")
        pages = [*prediction_pages, *law_pages]
        page_ids = [page.get("page_id") for page in pages if isinstance(page, dict)]
        if (
            len(page_ids) != len(pages)
            or not all(isinstance(item, str) and item for item in page_ids)
            or len(set(page_ids)) != len(page_ids)
            or submission_order != page_ids
        ):
            raise ValueError("belief snapshot page IDs/order are invalid")
        return {
            "protocol": "staged_pages_v1",
            "prediction_pages": prediction_pages,
            "law_pages": law_pages,
            "submission_order": page_ids,
            "prediction_page_count": len(prediction_pages),
            "law_page_count": len(law_pages),
            "total_page_count": len(pages),
            "participant_payload_auto_repair": False,
            "partial_draft_counts_as_checkpoint": False,
        }

    def _belief_stage_context(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        stages = contract.get("snapshot_stages")
        required_counts = contract.get("checkpoint_complete_experiments")
        if (
            not isinstance(stages, list)
            or not stages
            or not isinstance(required_counts, list)
            or len(required_counts) != len(stages)
        ):
            raise ValueError("checkpoint experiment-count schedule is invalid")
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        session_root = self.ipc / "sessions" / session_id
        snapshot_root = session_root / "belief_snapshots"
        committed = len(list(snapshot_root.glob("*.json"))) if snapshot_root.exists() else 0
        if committed >= len(stages):
            raise RuntimeError("all required belief checkpoints are already committed")
        completed_count, observed_evidence = self._completed_experiment_state()
        stage = str(stages[committed])
        required_count = int(required_counts[committed])
        draft_root = session_root / "belief_snapshot_drafts" / f"{committed + 1:02d}-{stage}"
        return {
            "contract": contract,
            "stage": stage,
            "required_count": required_count,
            "completed_count": completed_count,
            "observed_evidence": observed_evidence,
            "committed": committed,
            "stage_count": len(stages),
            "snapshot_root": snapshot_root,
            "draft_root": draft_root,
            "page_plan": self._page_plan(contract),
        }

    def _belief_submission_state(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        if self._free_research(descriptor):
            return {
                "protocol": "free_research",
                "all_checkpoints_committed": True,
                "required_checkpoint_count": 0,
                "instruction": (
                    "Explore freely; scientific questions follow the completed campaign."
                ),
            }
        try:
            context = self._belief_stage_context(descriptor)
        except RuntimeError as error:
            if "already committed" not in str(error):
                raise
            return {
                "protocol": "staged_pages_v1",
                "all_checkpoints_committed": True,
                "snapshot_schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
                "law_summary_schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
                "participant_payload_auto_repair": False,
                "partial_draft_counts_as_checkpoint": False,
            }
        plan = context["page_plan"]
        draft_root = context["draft_root"]
        accepted: list[str] = []
        if (draft_root / "manifest.json").is_file():
            for ordinal, page_id in enumerate(plan["submission_order"], start=1):
                if (draft_root / "fragments" / f"{ordinal:03d}-{page_id}.json").is_file():
                    accepted.append(page_id)
                else:
                    break
        next_page = (
            plan["submission_order"][len(accepted)]
            if len(accepted) < len(plan["submission_order"])
            else None
        )
        return {
            **plan,
            "snapshot_schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
            "law_summary_schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
            "stage": context["stage"],
            "checkpoint_complete_experiment_count": context["required_count"],
            "draft_started": (draft_root / "manifest.json").is_file(),
            "accepted_page_ids": accepted,
            "accepted_page_count": len(accepted),
            "next_page_id": next_page,
            "ready_to_finalize": next_page is None and (draft_root / "manifest.json").is_file(),
            "finalized_checkpoint_count": context["committed"],
        }

    def _audit(
        self,
        descriptor: dict[str, Any],
        tool: str,
        arguments: Any,
        *,
        started_at: str,
        duration_ms: float,
        status: str,
        result: Any,
        error_type: str | None,
        error_code: str | None,
        error_field_path: str | None,
        error_detail: str | None,
    ) -> None:
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        _append_jsonl(
            self.ipc / "sessions" / session_id / "mcp_tool_calls.jsonl",
            {
                "schema_version": MCP_SERVER_VERSION,
                "tool": tool,
                "started_at": started_at,
                "duration_ms": round(duration_ms, 3),
                "status": status,
                "error_type": error_type,
                "error_code": error_code,
                "error_field_path": error_field_path,
                "error_detail": error_detail,
                "error_detail_byte_count": (
                    len(error_detail.encode("utf-8", errors="replace"))
                    if error_detail is not None
                    else 0
                ),
                "error_detail_sha256": (
                    hashlib.sha256(error_detail.encode("utf-8", errors="replace")).hexdigest()
                    if error_detail is not None
                    else None
                ),
                "arguments_sha256": hashlib.sha256(_encode(arguments)).hexdigest(),
                "argument_keys": sorted(arguments) if isinstance(arguments, dict) else [],
                "result_sha256": hashlib.sha256(_encode(result)).hexdigest(),
            },
        )

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in SUPPORTED_TOOLS:
            return self._tool_error("unsupported_tool")
        descriptor = self._descriptor()
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        status = "completed"
        error_type: str | None = None
        error_code: str | None = None
        error_field_path: str | None = None
        error_detail: str | None = None
        try:
            if name == "material_information":
                payload = _read_object(self.reference / "material_information.json")
                if descriptor.get("session_scope") == "campaign":
                    payload = {
                        **payload,
                        "belief_snapshot_submission": self._belief_submission_state(descriptor),
                    }
            elif name == "status":
                payload = self._status()
            elif name == "history":
                payload = self._history(arguments)
            elif name == "inspect_artifact":
                payload = self._inspect(descriptor, arguments)
            elif name == "belief_snapshot_status":
                payload = self._belief_snapshot_status(descriptor)
            elif name == "commit_belief_snapshot":
                payload = self._commit_belief_snapshot(descriptor, arguments)
            elif name == "terminal_action_readout":
                payload = self._terminal_action_readout(descriptor)
            elif name == "commit_final_recommendation":
                payload = self._commit_final_recommendation(descriptor, arguments)
            else:
                payload = self._step(descriptor, arguments)
            cap = int(descriptor["max_tool_output_bytes"])
            encoded = _encode(payload)
            if len(encoded) > cap:
                raise ValueError("tool output exceeds configured byte cap")
            result = {
                "content": [{"type": "text", "text": encoded.decode("utf-8")}],
                "isError": False,
            }
        except Exception as error:
            detail = str(error).strip()
            status = "failed"
            error_type = type(error).__name__
            error_detail = detail[:1000] if detail else error_type
            error_code = self._error_code(error_type, error_detail)
            if name == "commit_belief_snapshot" and error_code == "validation_error":
                error_code = "invalid_belief_snapshot"
            error_field_path = self._error_field_path(error_detail)
            error_payload = self._actionable_error_payload(
                descriptor=descriptor,
                tool=name,
                arguments=arguments,
                error_type=error_type,
                error_code=error_code,
                field_path=error_field_path,
                detail=error_detail,
            )
            error_code = str(error_payload["error_code"])
            error_field_path = error_payload.get("field_path")
            result = self._tool_error(error_payload)
        self._audit(
            descriptor,
            name,
            arguments,
            started_at=started_at,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
            result=result,
            error_type=error_type,
            error_code=error_code,
            error_field_path=error_field_path,
            error_detail=error_detail,
        )
        return result

    @staticmethod
    def _error_code(error_type: str, detail: str) -> str:
        lowered = detail.lower()
        if "tool output exceeds configured byte cap" in lowered:
            return "platform_tool_output_limit"
        if (
            "checkpoint contract has no snapshot stages" in lowered
            or "checkpoint experiment-count schedule is invalid" in lowered
        ):
            return "platform_checkpoint_contract_invalid"
        if "session changed before the operation completed" in lowered:
            return "ipc_session_changed"
        if "campaign_already_ended_submit_final_response" in lowered:
            return "invalid_step_after_campaign_terminal"
        if (
            "required belief checkpoint" in lowered
            or "all required belief checkpoints must be committed" in lowered
        ):
            return "missing_required_belief_checkpoint"
        if "checkpoint is not due" in lowered:
            return "invalid_checkpoint_timing"
        if "decision_audit" in lowered and "required" in lowered:
            return "missing_decision_audit"
        if "expected_step" in lowered:
            return "invalid_expected_step"
        if "belief" in lowered or "snapshot" in lowered:
            return "invalid_belief_snapshot"
        if "final_recommendation" in lowered or "selected_experiment_index" in lowered:
            return "invalid_final_recommendation"
        if error_type == "PermissionError":
            return "atomic_replace_permission_error"
        if error_type == "ValueError":
            return "validation_error"
        return "tool_execution_error"

    @staticmethod
    def _error_field_path(detail: str) -> str | None:
        lowered = detail.lower()
        if "required belief checkpoint" in lowered:
            return "checkpoint"
        if "checkpoint is not due" in lowered:
            return "snapshot.stage"
        if "belief_snapshot fields do not match" in lowered:
            return "snapshot"
        if "prediction denominator" in lowered:
            return "snapshot.predictions"
        if "law_summary" in lowered:
            match = re.search(r"law_summary(?:\.[A-Za-z0-9_]+(?:\[\d+\])?)*", detail)
            return f"snapshot.{match.group(0)}" if match else "snapshot.law_summary"
        match = re.search(
            r"(?:snapshot\.|decision_audit\.|prior_assessment\.|law_summary\."
            r"|predictions\[|action\.|expected_step|selected_experiment_index)"
            r"[A-Za-z0-9_\.\[\]-]*",
            detail,
        )
        return match.group(0).rstrip(".") if match else None

    @staticmethod
    def _tool_error(error: str | dict[str, Any]) -> dict[str, Any]:
        payload = {"ok": False, "error": error} if isinstance(error, str) else error
        text = _encode(payload).decode("utf-8")
        return {"content": [{"type": "text", "text": text}], "isError": True}

    def _actionable_error_payload(
        self,
        *,
        descriptor: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        error_type: str,
        error_code: str,
        field_path: str | None,
        detail: str,
    ) -> dict[str, Any]:
        exposed_detail = (
            f"{error_type}: {detail}"
            if tool in {"step", "commit_belief_snapshot", "commit_final_recommendation"} and detail
            else error_type
        )
        payload: dict[str, Any] = {
            "ok": False,
            "error": exposed_detail,
            "error_code": error_code,
            "field_path": field_path,
        }
        if descriptor.get("session_scope") != "campaign":
            if tool == "step":
                payload.update(self._step_repair_context(arguments, detail))
            return payload
        if error_code in {
            "invalid_checkpoint_timing",
            "missing_required_belief_checkpoint",
        }:
            closeout = self._campaign_closeout_state(descriptor)
            payload.update(
                {
                    "checkpoint_due": closeout["checkpoint_due"],
                    "next_stage": closeout["next_checkpoint_stage"],
                    "completed": closeout["complete_experiment_count"],
                    "required": closeout["next_checkpoint_complete_experiment_count"],
                    "expected": {
                        "next_stage": closeout["next_checkpoint_stage"],
                        "complete_experiment_count": closeout[
                            "next_checkpoint_complete_experiment_count"
                        ],
                    },
                    "observed": {
                        "submitted_stage": self._submitted_snapshot_stage(arguments),
                        "complete_experiment_count": closeout["complete_experiment_count"],
                        "committed_checkpoint_count": closeout["committed_checkpoint_count"],
                    },
                    "recovery_action": (
                        "Call commit_belief_snapshot for next_stage using its tools/list "
                        "inputSchema before calling step or commit_final_recommendation."
                        if closeout["checkpoint_due"]
                        else "Continue physical experiments until completed equals required; "
                        "then call commit_belief_snapshot for next_stage."
                    ),
                }
            )
        if tool == "commit_belief_snapshot" and error_code in {
            "invalid_belief_snapshot",
            "validation_error",
        }:
            payload.update(self._belief_snapshot_repair_context(arguments, detail))
        if tool == "step" and "expected" not in payload:
            payload.update(self._step_repair_context(arguments, detail))
        return payload

    @staticmethod
    def _submitted_snapshot_stage(arguments: dict[str, Any]) -> Any:
        snapshot = arguments.get("snapshot")
        if isinstance(snapshot, dict):
            return snapshot.get("stage")
        header = arguments.get("snapshot_header")
        return header.get("stage") if isinstance(header, dict) else None

    def _belief_snapshot_repair_context(
        self,
        arguments: dict[str, Any],
        detail: str,
    ) -> dict[str, Any]:
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        if isinstance(contract.get("snapshot_submission_protocol"), dict):
            state = self._belief_submission_state(self._descriptor())
            schema = self._staged_belief_snapshot_tool_schema()
            if not state.get("draft_started"):
                next_action = "begin"
            elif state.get("next_page_id") is not None:
                next_action = str(state["next_page_id"])
            elif state.get("ready_to_finalize") is True:
                next_action = "finalize"
            else:
                next_action = "status"
            action = arguments.get("action")
            branch = next(
                (
                    candidate
                    for candidate in schema["oneOf"]
                    if candidate["properties"]["action"].get("const") == action
                    and (
                        action not in {"append_prediction_page", "append_law_page"}
                        or candidate["properties"]["page_id"].get("const")
                        == arguments.get("page_id")
                    )
                ),
                None,
            )
            context: dict[str, Any] = {
                "schema_authority": "tools/list -> commit_belief_snapshot.inputSchema",
                "expected": {
                    "snapshot_schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
                    "law_summary_schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
                    "next_action": next_action,
                },
                "submission_state": state,
                "observed": {
                    "action": action,
                    "page_id": arguments.get("page_id"),
                    "argument_keys": sorted(arguments),
                },
                "schema_fragment": deepcopy(branch or schema),
                "participant_payload_auto_repair": False,
                "recovery_action": (
                    f"Submit only the current staged action ({next_action}) using the root "
                    "inputSchema and submission_state. Accepted pages cannot be replaced."
                ),
            }
            lowered = detail.lower()
            if "law_summary.schema_version does not match" in lowered:
                header = arguments.get("snapshot_header")
                header = header if isinstance(header, dict) else {}
                law_summary = header.get("law_summary")
                law_summary = law_summary if isinstance(law_summary, dict) else {}
                context.update(
                    {
                        "field_path": "snapshot_header.law_summary.schema_version",
                        "expected": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
                        "observed": law_summary.get("schema_version"),
                        "schema_fragment": {
                            "const": WORK_II_LAW_SUMMARY_SCHEMA_VERSION
                        },
                    }
                )
            elif "stage does not match" in lowered:
                header = arguments.get("snapshot_header")
                header = header if isinstance(header, dict) else {}
                stage_schema = schema["properties"]["snapshot_header"]["properties"]["stage"]
                context.update(
                    {
                        "field_path": "snapshot_header.stage",
                        "expected": stage_schema.get("const"),
                        "observed": header.get("stage"),
                        "schema_fragment": deepcopy(stage_schema),
                    }
                )
            elif "unknown feature id" in lowered:
                context.update(
                    {
                        "field_path": "snapshot_header.law_summary.feature_ids",
                        "expected": {
                            "type": "array",
                            "items_enum": [
                                str(item) for item in contract.get("allowed_feature_ids", [])
                            ],
                            "min_items": 1,
                            "unique_items": True,
                        },
                        "observed": (
                            arguments["snapshot_header"]["law_summary"].get("feature_ids")
                            if isinstance(arguments.get("snapshot_header"), dict)
                            and isinstance(
                                arguments["snapshot_header"].get("law_summary"), dict
                            )
                            else None
                        ),
                        "recovery_action": (
                            "Replace every feature ID with an exact value from "
                            "expected.items_enum; do not invent aliases or display names."
                        ),
                    }
                )
            elif "limitations must be a string list" in lowered:
                context.update(
                    {
                        "field_path": "snapshot_header.law_summary.limitations",
                        "expected": {"type": "array", "items": {"type": "string"}},
                        "recovery_action": (
                            "Submit limitations as a JSON array of strings; use [] when none."
                        ),
                    }
                )
            elif "suspected_misindexed_fields must be a string list" in lowered:
                context.update(
                    {
                        "field_path": (
                            "snapshot_header.prior_assessment.suspected_misindexed_fields"
                        ),
                        "expected": {
                            "type": "array",
                            "items_enum": [
                                str(item) for item in contract.get("allowed_prior_fields", [])
                            ],
                        },
                        "recovery_action": (
                            "Submit suspected_misindexed_fields as an array of exact public prior "
                            "field IDs; use [] when none."
                        ),
                    }
                )
            elif "prediction metrics must be a list" in lowered:
                context.update(
                    {
                        "field_path": "predictions[].metrics",
                        "expected": {
                            "type": "array",
                            "metric_ids": [
                                str(item) for item in contract.get("allowed_metric_ids", [])
                            ],
                        },
                        "recovery_action": (
                            "Submit metrics as the exact ordered JSON array published for the "
                            "current prediction page."
                        ),
                    }
                )
            return context
        snapshot = arguments.get("snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        schema = self._belief_snapshot_schema()
        context: dict[str, Any] = {
            "schema_authority": (
                "tools/list -> commit_belief_snapshot.inputSchema.properties.snapshot"
            ),
            "expected": {"contract": "authoritative snapshot tools/list inputSchema"},
            "observed": {
                "type": type(arguments.get("snapshot")).__name__,
                "fields": sorted(snapshot),
            },
            "schema_fragment": {
                "type": "object",
                "required": schema["required"],
                "additionalProperties": False,
            },
            "recovery_action": (
                "Rebuild the rejected field from the authoritative tools/list inputSchema; "
                "do not inspect host filesystem paths."
            ),
        }
        lowered = detail.lower()
        if "belief_snapshot fields do not match" in lowered:
            context.update(
                {
                    "field_path": "snapshot",
                    "expected": {"required_fields": schema["required"]},
                    "observed": {"fields": sorted(snapshot)},
                    "schema_fragment": {
                        "required": schema["required"],
                        "additionalProperties": False,
                    },
                }
            )
        elif "schema_version does not match" in lowered:
            law_summary = snapshot.get("law_summary")
            law_summary = law_summary if isinstance(law_summary, dict) else {}
            is_law = lowered.startswith("law_summary")
            expected_version = (
                WORK_II_LAW_SUMMARY_SCHEMA_VERSION if is_law else WORK_II_SNAPSHOT_SCHEMA_VERSION
            )
            context.update(
                {
                    "field_path": (
                        "snapshot.law_summary.schema_version"
                        if is_law
                        else "snapshot.schema_version"
                    ),
                    "expected": expected_version,
                    "observed": (
                        law_summary.get("schema_version")
                        if is_law
                        else snapshot.get("schema_version")
                    ),
                    "schema_fragment": {"const": expected_version},
                }
            )
        elif "prediction denominator" in lowered:
            predictions = snapshot.get("predictions")
            context.update(
                {
                    "field_path": "snapshot.predictions",
                    "expected": {
                        "count": len(contract["query_metric_contract"]),
                        "query_ids": list(contract["query_metric_contract"]),
                    },
                    "observed": {
                        "count": len(predictions) if isinstance(predictions, list) else None,
                        "query_ids": self._object_ids(predictions, "query_id"),
                    },
                    "schema_fragment": self._array_contract_fragment(
                        schema["properties"]["predictions"]
                    ),
                }
            )
        elif "metric_laws" in lowered or "exact held-out metric set" in lowered:
            law_summary = snapshot.get("law_summary")
            law_summary = law_summary if isinstance(law_summary, dict) else {}
            metric_laws = law_summary.get("metric_laws")
            metric_schema = schema["properties"]["law_summary"]["properties"]["metric_laws"]
            item_schema = metric_schema["items"]
            field_path = "snapshot.law_summary.metric_laws"
            item_match = re.search(r"metric_laws\[(\d+)\]", detail)
            if item_match:
                field_path += f"[{item_match.group(1)}]"
            context.update(
                {
                    "field_path": field_path,
                    "expected": {
                        "count": len(contract["allowed_metric_ids"]),
                        "metric_ids": contract["allowed_metric_ids"],
                    },
                    "observed": {
                        "count": len(metric_laws) if isinstance(metric_laws, list) else None,
                        "metric_ids": self._object_ids(metric_laws, "metric_id"),
                    },
                    "schema_fragment": {
                        **self._array_contract_fragment(metric_schema),
                        "item_required": item_schema["required"],
                        "allowed_links": item_schema["properties"]["link"]["enum"],
                    },
                }
            )
        elif "stage does not match" in lowered:
            closeout = self._campaign_closeout_state(self._descriptor())
            context.update(
                {
                    "field_path": "snapshot.stage",
                    "expected": closeout["next_checkpoint_stage"],
                    "observed": snapshot.get("stage"),
                    "schema_fragment": schema["properties"]["stage"],
                }
            )
        return context

    def _step_repair_context(
        self,
        arguments: dict[str, Any],
        detail: str,
    ) -> dict[str, Any]:
        lowered = detail.lower()
        if "expected_step" in lowered:
            fragment = {"type": "integer", "minimum": 1}
            return {
                "field_path": "expected_step",
                "expected": fragment,
                "observed": arguments.get("expected_step"),
                "schema_fragment": fragment,
                "recovery_action": (
                    "Resubmit step with the current positive integer expected_step from the "
                    "latest public outcome or status."
                ),
            }
        if (
            "decision_audit" in lowered
            or "decision audit" in lowered
            or "expected_information_gain" in lowered
            or "expected information gain" in lowered
            or "belief_update_rule" in lowered
            or "belief update rule" in lowered
        ):
            audit = arguments.get("decision_audit")
            field_match = re.search(r"decision_audit\.([A-Za-z0-9_]+)", detail)
            if field_match is None:
                normalized = lowered.replace(" ", "_")
                field_match = re.search(
                    r"(expected_information_gain|belief_update_rule|uncertainty|"
                    r"adaptation_source|expected_effect|diagnostic_target)",
                    normalized,
                )
            audit_schema = self._decision_audit_schema()
            if field_match:
                field = field_match.group(1)
                fragment = audit_schema["properties"].get(field, {})
                observed = audit.get(field) if isinstance(audit, dict) else None
                return {
                    "field_path": f"decision_audit.{field}",
                    "expected": fragment,
                    "observed": observed,
                    "schema_fragment": fragment,
                    "recovery_action": (
                        "Correct only this public decision-audit field and resubmit the same "
                        "intended physical operation."
                    ),
                }
            return {
                "field_path": "decision_audit",
                "expected": {
                    "type": "object",
                    "required_fields": audit_schema["required"],
                },
                "observed": {
                    "type": type(audit).__name__,
                    "fields": sorted(audit) if isinstance(audit, dict) else None,
                },
                "schema_fragment": {
                    "type": "object",
                    "required": audit_schema["required"],
                    "additionalProperties": False,
                },
                "recovery_action": (
                    "Add every required bounded decision_audit field from the step inputSchema "
                    "and resubmit the same intended physical operation."
                ),
            }
        if "action.operation" in lowered:
            fragment = {"type": "string", "minLength": 1}
            action = arguments.get("action")
            observed = action.get("operation") if isinstance(action, dict) else None
            return {
                "field_path": "action.operation",
                "expected": fragment,
                "observed": observed,
                "schema_fragment": fragment,
                "recovery_action": (
                    "Choose an available operation from the current public state and resubmit."
                ),
            }
        return {
            "expected": {"contract": "step tools/list inputSchema"},
            "observed": {"argument_fields": sorted(arguments)},
            "recovery_action": (
                "Correct the rejected field using the authoritative step tools/list inputSchema."
            ),
        }

    @staticmethod
    def _object_ids(value: Any, field: str) -> list[Any] | None:
        if not isinstance(value, list):
            return None
        return [item.get(field) if isinstance(item, dict) else None for item in value]

    @staticmethod
    def _array_contract_fragment(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            key: schema[key]
            for key in (
                "type",
                "minItems",
                "maxItems",
                "x-chemworld-required-ids",
                "x-chemworld-query-metric-contract",
            )
            if key in schema
        }

    def _history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": MCP_SERVER_VERSION,
            **public_history_page(
                self.public,
                limit=arguments.get("limit", 5),
                offset=arguments.get("offset"),
                max_bytes=int(self._descriptor()["max_tool_output_bytes"]) - 128,
            ),
        }

    def _action_readout_contract(self) -> dict[str, Any] | None:
        path = self.reference / "terminal_action_readout_contract.json"
        return _read_object(path) if path.is_file() else None

    def _all_checkpoints_committed(self, descriptor: dict[str, Any]) -> bool:
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        stages = contract.get("snapshot_stages")
        if not isinstance(stages, list) or not stages:
            return False
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        snapshot_root = self.ipc / "sessions" / session_id / "belief_snapshots"
        committed = len(list(snapshot_root.glob("*.json"))) if snapshot_root.exists() else 0
        return committed == len(stages)

    def _terminal_action_readout(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        if descriptor.get("session_scope") != "campaign":
            raise RuntimeError("terminal action readout requires campaign scope")
        contract = self._action_readout_contract()
        if descriptor.get("terminal_action_readout_required") is not True or contract is None:
            return {
                "schema_version": MCP_SERVER_VERSION,
                "available": False,
                "reason": "terminal_action_readout_not_configured",
            }
        terminal = self._campaign_terminal_observed()
        checkpoints = self._all_checkpoints_committed(descriptor)
        if not terminal or not checkpoints:
            return {
                "schema_version": MCP_SERVER_VERSION,
                "available": False,
                "reason": (
                    "campaign_not_terminal"
                    if not terminal
                    else "final_belief_checkpoint_not_committed"
                ),
                "campaign_terminal": terminal,
                "all_belief_checkpoints_committed": checkpoints,
                "candidate_contents_revealed": False,
            }
        return {
            **deepcopy(contract),
            "available": True,
            "campaign_terminal": True,
            "all_belief_checkpoints_committed": True,
            "candidate_outcomes_hidden": True,
            "final_belief_checkpoint_locked_before_reveal": True,
        }

    def _status(self) -> dict[str, Any]:
        descriptor = self._descriptor()
        campaign = descriptor.get("session_scope") == "campaign"
        action_readout = descriptor.get("terminal_action_readout_required") is True
        action_contract = self._action_readout_contract() if action_readout else None
        action_prediction_mode = (
            str(action_contract.get("prediction_mode", "full_metrics"))
            if isinstance(action_contract, dict)
            else "full_metrics"
        )
        if self._terminal_outcome is None:
            current = _read_object(self.public / "current.json")
            if not campaign:
                return current
            return {
                **current,
                "campaign_closeout": self._campaign_closeout_state(descriptor),
            }
        terminal = {
            "schema_version": MCP_SERVER_VERSION,
            "experiment_ended": True,
            "terminal_outcome": self._terminal_outcome,
            "instruction": (
                "Commit the final belief checkpoint if due, call "
                + (
                    "terminal_action_readout after the checkpoint is committed, then call "
                    "commit_final_recommendation exactly once and submit the final response; "
                    if action_readout
                    else "submit the short research handoff without selecting a batch; "
                    if self._knowledge_delivery(descriptor)
                    else "commit_final_recommendation exactly once, then submit the final "
                    "response; "
                )
                + "do not call step again."
                if campaign
                else "Submit the final response now; do not call step again."
            ),
            "final_response_contract": self._final_response_contract(
                campaign=campaign,
                terminal_action_readout=action_readout,
                terminal_prediction_mode=action_prediction_mode,
                final_recommendation_required=not self._knowledge_delivery(descriptor),
            ),
        }
        if campaign:
            terminal["campaign_closeout"] = self._campaign_closeout_state(descriptor)
        return terminal

    def _campaign_closeout_state(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        completed_count, _ = self._completed_experiment_state()
        progress = self._campaign_progress()
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        stages = contract.get("snapshot_stages", [])
        required_counts = contract.get("checkpoint_complete_experiments", [])
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        snapshot_root = self.ipc / "sessions" / session_id / "belief_snapshots"
        committed = len(list(snapshot_root.glob("*.json"))) if snapshot_root.exists() else 0
        has_next = committed < len(stages) and committed < len(required_counts)
        next_stage = str(stages[committed]) if has_next else None
        next_required = int(required_counts[committed]) if has_next else None
        return {
            "campaign_ended": self._campaign_terminal_observed(),
            "closed_batch_count": (
                int(progress["closed_batch_count"]) if progress is not None else completed_count
            ),
            "complete_experiment_count": completed_count,
            "completed_experiment_indices": (
                list(progress["completed_experiment_indices"])
                if progress is not None
                else list(range(1, completed_count + 1))
            ),
            "completed_batch_ids": (
                list(progress["completed_batch_ids"])
                if progress is not None
                else [f"batch-{index:04d}" for index in range(1, completed_count + 1)]
            ),
            "committed_checkpoint_count": committed,
            "required_checkpoint_count": len(stages),
            "checkpoint_due": has_next and completed_count == next_required,
            "next_checkpoint_stage": next_stage,
            "next_checkpoint_complete_experiment_count": next_required,
            "final_recommendation_committed": (
                self.ipc / "sessions" / session_id / "final_recommendation.json"
            ).is_file(),
            "belief_snapshot_submission": self._belief_submission_state(descriptor),
            **(
                {
                    "terminal_action_readout_required": True,
                    "terminal_action_readout_available": (
                        self._campaign_terminal_observed() and committed == len(stages)
                    ),
                }
                if descriptor.get("terminal_action_readout_required") is True
                else {}
            ),
        }

    @staticmethod
    def _final_response_contract(
        *,
        campaign: bool,
        terminal_action_readout: bool = False,
        terminal_prediction_mode: str = "full_metrics",
        final_recommendation_required: bool = True,
    ) -> dict[str, Any]:
        return {
            "format": "json_object_only",
            "required_keys": ["status", "summary"],
            "status": "campaign_complete" if campaign else "experiment_complete",
            "summary_max_length": 3000 if campaign else 2000,
            "final_recommendation_contract": (
                (
                    {
                        **(
                            {
                                "candidate_predictions": (
                                    "every_revealed_candidate_and_metric"
                                )
                            }
                            if terminal_prediction_mode == "full_metrics"
                            else {"candidate_predictions": "not_submitted"}
                        ),
                        "ranking": "all_candidate_query_ids_exactly_once",
                        "selected_action_query_id": "ranking_first_item",
                        "selection_rationale_max_length": 2000,
                        "mechanism_application_summary_max_length": 1200,
                        "revealed_after_final_belief_checkpoint": True,
                    }
                    if terminal_action_readout
                    else {
                        "selected_experiment_index": (
                            "1-based_lifecycle_index_from_completed_experiment_indices"
                        ),
                        "selection_rationale_max_length": 2000,
                        "committed_before_blind_evaluation": True,
                    }
                )
                if campaign and final_recommendation_required
                else None
            ),
            "prose_or_markdown_allowed": False,
        }

    def _belief_snapshot_status(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Expose the exact next staged payload without weakening scientific validation."""

        if descriptor.get("session_scope") != "campaign":
            raise RuntimeError("belief checkpoints are available only in campaign sessions")
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        state = self._belief_submission_state(descriptor)
        allowed_feature_ids = [str(item) for item in contract.get("allowed_feature_ids", [])]
        allowed_metric_ids = [str(item) for item in contract.get("allowed_metric_ids", [])]
        allowed_prior_fields = [str(item) for item in contract.get("allowed_prior_fields", [])]
        evidence_catalog = [str(item) for item in contract.get("evidence_catalog", [])]
        _, observed_evidence = self._completed_experiment_state()
        next_page_id = state.get("next_page_id")
        plan = self._page_plan(contract)
        if not state.get("draft_started"):
            next_action = "begin"
            next_payload_contract: dict[str, Any] = {
                "required_argument_fields": ["action", "snapshot_header"],
                "snapshot_header_required_fields": [
                    "snapshot_id",
                    "stage",
                    "prior_assessment",
                    "law_summary",
                    "evidence_ids",
                    "next_experiment_intent",
                    "overall_confidence",
                ],
                "prior_assessment_required_fields": [
                    "nominal_information_available",
                    "reliability_probability",
                    "suspected_misindexed_fields",
                    "rationale",
                ],
                "law_summary_required_fields": [
                    "schema_version",
                    "summary_id",
                    "feature_ids",
                    "evidence_ids",
                    "applicability",
                    "limitations",
                    "confidence",
                ],
                "limitations_type": "array[string]",
                "suspected_misindexed_fields_type": "array[string]",
            }
        elif next_page_id is not None:
            prediction_pages = self._prediction_page_map(plan)
            law_pages = self._law_page_map(plan)
            if next_page_id in prediction_pages:
                next_action = "append_prediction_page"
                next_payload_contract = {
                    "required_argument_fields": ["action", "page_id", "predictions"],
                    "page_id": next_page_id,
                    "query_metric_contract": deepcopy(
                        prediction_pages[str(next_page_id)]["query_metric_contract"]
                    ),
                    "prediction_required_fields": ["query_id", "metrics"],
                    "metric_required_fields": [
                        "metric_id",
                        "mean",
                        "interval_lower",
                        "interval_upper",
                        "confidence",
                    ],
                }
            else:
                next_action = "append_law_page"
                next_payload_contract = {
                    "required_argument_fields": ["action", "page_id", "metric_laws"],
                    "page_id": next_page_id,
                    "metric_ids": list(law_pages[str(next_page_id)]["metric_ids"]),
                    "metric_law_required_fields": [
                        "metric_id",
                        "intercept",
                        "link",
                        "lower_bound",
                        "upper_bound",
                        "terms",
                    ],
                    "law_term_required_fields": [
                        "term_id",
                        "basis",
                        "input_ids",
                        "coefficient",
                    ],
                    "law_term_optional_fields": ["category_value"],
                    "law_bases": sorted(WORK_II_LAW_BASES),
                    "law_links": sorted(WORK_II_LAW_LINKS),
                }
        elif state.get("ready_to_finalize") is True:
            next_action = "finalize"
            next_payload_contract = {"required_argument_fields": ["action"]}
        else:
            next_action = "none"
            next_payload_contract = {"required_argument_fields": []}
        return {
            "schema_version": MCP_SERVER_VERSION,
            "next_action": next_action,
            "submission_state": state,
            "next_payload_contract": next_payload_contract,
            "active_stage": state.get("stage"),
            "allowed_feature_ids": allowed_feature_ids,
            "allowed_metric_ids": allowed_metric_ids,
            "allowed_prior_fields": allowed_prior_fields,
            "allowed_evidence_ids": evidence_catalog,
            "observed_evidence_ids": [
                item for item in evidence_catalog if item in observed_evidence
            ],
            "snapshot_schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
            "law_summary_schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
            "participant_payload_auto_repair": False,
        }

    def _commit_belief_snapshot(
        self,
        descriptor: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if descriptor.get("session_scope") != "campaign":
            raise RuntimeError("belief checkpoints are available only in campaign sessions")
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        if "action" in arguments:
            return self._commit_staged_belief_snapshot(descriptor, arguments)
        if isinstance(contract.get("snapshot_submission_protocol"), dict):
            raise ValueError(
                "action is required: use begin, the published pages in order, then finalize"
            )
        snapshot = arguments.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        stages = contract.get("snapshot_stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("checkpoint contract has no snapshot stages")
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        root = self.ipc / "sessions" / session_id / "belief_snapshots"
        existing = sorted(root.glob("*.json")) if root.exists() else []
        if len(existing) >= len(stages):
            raise RuntimeError("all required belief checkpoints are already committed")
        expected_stage = str(stages[len(existing)])
        required_counts = contract.get("checkpoint_complete_experiments")
        if not isinstance(required_counts, list) or len(required_counts) != len(stages):
            raise ValueError("checkpoint experiment-count schedule is invalid")
        completed_count, observed_evidence = self._completed_experiment_state()
        if completed_count != int(required_counts[len(existing)]):
            raise RuntimeError(
                "checkpoint is not due at the current completed-experiment count: "
                f"expected {required_counts[len(existing)]}, observed {completed_count}"
            )
        parsed = parse_work_ii_belief_snapshot(
            snapshot,
            expected_stage=expected_stage,
            query_metric_contract={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(contract["query_metric_contract"]).items()
            },
            allowed_feature_ids=tuple(str(item) for item in contract["allowed_feature_ids"]),
            allowed_metric_ids=tuple(str(item) for item in contract["allowed_metric_ids"]),
            allowed_prior_fields=tuple(str(item) for item in contract["allowed_prior_fields"]),
            evidence_catalog=tuple(str(item) for item in contract["evidence_catalog"]),
            nominal_information_available=bool(contract["nominal_information_available"]),
        )
        cited = set(parsed.evidence_ids) | set(parsed.law_summary.evidence_ids)
        if not cited.issubset(observed_evidence):
            raise ValueError("belief snapshot cites evidence that has not yet been observed")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{len(existing) + 1:02d}-{expected_stage}.json"
        _atomic_json(path, parsed.to_dict())
        result = {
            "ok": True,
            "schema_version": MCP_SERVER_VERSION,
            "stage": expected_stage,
            "complete_experiment_count": completed_count,
            "committed_checkpoint_count": len(existing) + 1,
            "remaining_checkpoint_count": len(stages) - len(existing) - 1,
        }
        if result["remaining_checkpoint_count"] == 0:
            result["final_response_contract"] = self._final_response_contract(campaign=True)
        return result

    @staticmethod
    def _exact_argument_fields(
        arguments: dict[str, Any], expected: set[str], *, label: str
    ) -> None:
        if set(arguments) != expected:
            raise ValueError(f"{label} fields do not match the staged contract")

    @staticmethod
    def _prediction_page_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(page["page_id"]): page
            for page in plan["prediction_pages"]
            if isinstance(page, dict)
        }

    @staticmethod
    def _law_page_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(page["page_id"]): page for page in plan["law_pages"] if isinstance(page, dict)}

    def _draft_payloads(self, context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        root = context["draft_root"] / "fragments"
        payloads: dict[str, dict[str, Any]] = {}
        if not root.is_dir():
            return payloads
        for path in sorted(root.glob("*.json")):
            value = _read_object(path)
            page_id = value.get("page_id")
            if not isinstance(page_id, str) or page_id in payloads:
                raise ValueError("belief snapshot draft fragments are invalid")
            payloads[page_id] = value
        return payloads

    def _assemble_staged_snapshot(
        self,
        context: dict[str, Any],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        plan = context["page_plan"]
        payloads = self._draft_payloads(context)
        predictions: list[dict[str, Any]] = []
        laws: list[dict[str, Any]] = []
        for page in plan["prediction_pages"]:
            page_id = str(page["page_id"])
            if page_id in payloads:
                predictions.extend(payloads[page_id]["predictions"])
            else:
                raise ValueError(f"belief snapshot page is missing: {page_id}")
        for page in plan["law_pages"]:
            page_id = str(page["page_id"])
            if page_id in payloads:
                laws.extend(payloads[page_id]["metric_laws"])
            else:
                raise ValueError(f"belief snapshot page is missing: {page_id}")
        header = manifest["snapshot_header"]
        law_header = header["law_summary"]
        return {
            "schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": header["snapshot_id"],
            "stage": header["stage"],
            "prior_assessment": header["prior_assessment"],
            "predictions": predictions,
            "law_summary": {**law_header, "metric_laws": laws},
            "evidence_ids": header["evidence_ids"],
            "next_experiment_intent": header["next_experiment_intent"],
            "overall_confidence": header["overall_confidence"],
        }

    def _commit_staged_belief_snapshot(
        self, descriptor: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        action = arguments.get("action")
        if action not in {
            "begin",
            "append_prediction_page",
            "append_law_page",
            "finalize",
        }:
            raise ValueError("belief snapshot action is invalid")
        context = self._belief_stage_context(descriptor)
        if context["completed_count"] != context["required_count"]:
            raise RuntimeError(
                "checkpoint is not due at the current completed-experiment count: "
                f"expected {context['required_count']}, observed {context['completed_count']}"
            )
        manifest_path = context["draft_root"] / "manifest.json"
        if action == "begin":
            self._exact_argument_fields(
                arguments, {"action", "snapshot_header"}, label="belief snapshot begin"
            )
            header = arguments.get("snapshot_header")
            if not isinstance(header, dict):
                raise ValueError("snapshot_header must be an object")
            manifest = {
                "schema_version": BELIEF_DRAFT_VERSION,
                "stage": context["stage"],
                "complete_experiment_count": context["completed_count"],
                "page_plan": context["page_plan"],
                "snapshot_header": header,
                "participant_payload_auto_repair": False,
            }
            parsed_header = parse_work_ii_belief_snapshot_header(
                header,
                expected_stage=context["stage"],
                allowed_feature_ids=context["contract"]["allowed_feature_ids"],
                allowed_prior_fields=context["contract"]["allowed_prior_fields"],
                evidence_catalog=context["contract"]["evidence_catalog"],
                nominal_information_available=bool(
                    context["contract"]["nominal_information_available"]
                ),
            )
            cited = set(parsed_header["evidence_ids"]) | set(
                parsed_header["law_summary"]["evidence_ids"]
            )
            if not cited.issubset(context["observed_evidence"]):
                raise ValueError("belief snapshot cites evidence that has not yet been observed")
            if manifest_path.exists():
                existing = _read_object(manifest_path)
                if _encode(existing.get("snapshot_header")) == _encode(header):
                    return {
                        "ok": True,
                        "already_started": True,
                        **self._belief_submission_state(descriptor),
                    }
                raise ValueError(
                    "belief snapshot draft is already started with a different immutable header"
                )
            _write_json_once(manifest_path, manifest)
            return {
                "ok": True,
                "already_started": False,
                **self._belief_submission_state(descriptor),
            }
        if not manifest_path.is_file():
            raise ValueError("begin must be accepted before belief snapshot pages")
        manifest = _read_object(manifest_path)
        state = self._belief_submission_state(descriptor)
        if action == "finalize":
            self._exact_argument_fields(arguments, {"action"}, label="belief snapshot finalize")
            if not state["ready_to_finalize"]:
                raise ValueError("belief snapshot cannot finalize before every exact page")
            canonical = self._assemble_staged_snapshot(context, manifest)
            parsed = parse_work_ii_belief_snapshot(
                canonical,
                expected_stage=context["stage"],
                query_metric_contract={
                    str(key): tuple(str(item) for item in value)
                    for key, value in dict(context["contract"]["query_metric_contract"]).items()
                },
                allowed_feature_ids=tuple(
                    str(item) for item in context["contract"]["allowed_feature_ids"]
                ),
                allowed_metric_ids=tuple(
                    str(item) for item in context["contract"]["allowed_metric_ids"]
                ),
                allowed_prior_fields=tuple(
                    str(item) for item in context["contract"]["allowed_prior_fields"]
                ),
                evidence_catalog=tuple(
                    str(item) for item in context["contract"]["evidence_catalog"]
                ),
                nominal_information_available=bool(
                    context["contract"]["nominal_information_available"]
                ),
            )
            cited = set(parsed.evidence_ids) | set(parsed.law_summary.evidence_ids)
            if not cited.issubset(context["observed_evidence"]):
                raise ValueError("belief snapshot cites evidence that has not yet been observed")
            context["snapshot_root"].mkdir(parents=True, exist_ok=True)
            target = (
                context["snapshot_root"] / f"{context['committed'] + 1:02d}-{context['stage']}.json"
            )
            _write_json_once(target, parsed.to_dict())
            _write_json_once(
                context["draft_root"] / "finalization.json",
                {"status": "finalized", "stage": context["stage"]},
            )
            result = {
                "ok": True,
                "schema_version": MCP_SERVER_VERSION,
                "stage": context["stage"],
                "complete_experiment_count": context["completed_count"],
                "committed_checkpoint_count": context["committed"] + 1,
                "remaining_checkpoint_count": context["stage_count"] - context["committed"] - 1,
            }
            if result["remaining_checkpoint_count"] == 0:
                result["final_response_contract"] = self._final_response_contract(campaign=True)
            return result
        page_id = arguments.get("page_id")
        if not isinstance(page_id, str):
            raise ValueError("page_id must be a string")
        if page_id != state["next_page_id"]:
            if page_id in state["accepted_page_ids"]:
                raise ValueError("accepted belief snapshot page cannot be replaced")
            raise ValueError("belief snapshot page differs from the fixed next page")
        prediction_pages = self._prediction_page_map(context["page_plan"])
        law_pages = self._law_page_map(context["page_plan"])
        if action == "append_prediction_page":
            self._exact_argument_fields(
                arguments,
                {"action", "page_id", "predictions"},
                label="belief prediction page",
            )
            if page_id not in prediction_pages:
                raise ValueError("page_id is not a prediction page")
            predictions = arguments.get("predictions")
            if not isinstance(predictions, list):
                raise ValueError("predictions must be a list")
            expected = prediction_pages[page_id]["query_metric_contract"]
            observed = [item.get("query_id") for item in predictions if isinstance(item, dict)]
            if len(observed) != len(predictions) or observed != list(expected):
                raise ValueError("prediction page does not contain its exact ordered query IDs")
            parse_work_ii_prediction_page(
                predictions,
                query_metric_contract={
                    str(query_id): tuple(str(item) for item in metric_ids)
                    for query_id, metric_ids in expected.items()
                },
            )
            page = {"page_id": page_id, "predictions": arguments["predictions"]}
        else:
            self._exact_argument_fields(
                arguments,
                {"action", "page_id", "metric_laws"},
                label="belief law page",
            )
            if page_id not in law_pages:
                raise ValueError("page_id is not a law page")
            laws = arguments.get("metric_laws")
            if not isinstance(laws, list):
                raise ValueError("metric_laws must be a list")
            expected_metrics = law_pages[page_id]["metric_ids"]
            observed_metrics = [item.get("metric_id") for item in laws if isinstance(item, dict)]
            if len(observed_metrics) != len(laws) or observed_metrics != expected_metrics:
                raise ValueError("law page does not contain its exact ordered metric IDs")
            law_header = manifest["snapshot_header"]["law_summary"]
            parse_work_ii_law_summary(
                {**law_header, "metric_laws": laws},
                allowed_feature_ids=context["contract"]["allowed_feature_ids"],
                allowed_metric_ids=context["contract"]["allowed_metric_ids"],
                evidence_catalog=context["contract"]["evidence_catalog"],
                required_metric_ids=expected_metrics,
            )
            page = {"page_id": page_id, "metric_laws": arguments["metric_laws"]}
        ordinal = context["page_plan"]["submission_order"].index(page_id) + 1
        _write_json_once(
            context["draft_root"] / "fragments" / f"{ordinal:03d}-{page_id}.json",
            page,
        )
        return {"ok": True, **self._belief_submission_state(descriptor)}

    @staticmethod
    def _action_readout_recommendation(
        contract: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        candidates = contract.get("candidate_queries")
        metric_ids = contract.get("metric_ids")
        if (
            contract.get("schema_version")
            != "chemworld-work-ii-terminal-action-readout-contract-0.1"
            or not isinstance(candidates, list)
            or not candidates
            or not isinstance(metric_ids, list)
            or not metric_ids
        ):
            raise ValueError("terminal action readout contract is malformed")
        query_ids = [
            str(item.get("query_id"))
            for item in candidates
            if isinstance(item, dict) and isinstance(item.get("query_id"), str)
        ]
        metrics = [str(item) for item in metric_ids]
        prediction_mode = str(contract.get("prediction_mode", "full_metrics"))
        if len(query_ids) != len(candidates) or len(set(query_ids)) != len(query_ids):
            raise ValueError("terminal action candidate IDs are malformed")
        if len(set(metrics)) != len(metrics):
            raise ValueError("terminal action metric IDs are malformed")

        predictions: list[dict[str, Any]] = []
        if prediction_mode == "full_metrics":
            raw_predictions = arguments.get("candidate_predictions")
            if not isinstance(raw_predictions, list) or len(raw_predictions) != len(query_ids):
                raise ValueError("candidate_predictions must cover every terminal candidate")
            observed: set[str] = set()
            for raw in raw_predictions:
                if not isinstance(raw, dict):
                    raise ValueError("candidate prediction must be an object")
                query_id = raw.get("query_id")
                values = raw.get("metrics")
                if (
                    not isinstance(query_id, str)
                    or query_id not in query_ids
                    or query_id in observed
                ):
                    raise ValueError("candidate prediction query_id is invalid or duplicated")
                if not isinstance(values, dict) or set(values) != set(metrics):
                    raise ValueError("candidate prediction metric coverage is invalid")
                normalized_metrics: dict[str, float] = {}
                for metric_id in metrics:
                    value = values[metric_id]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0
                    ):
                        raise ValueError(
                            "candidate prediction metric must be finite in [0, 1]"
                        )
                    normalized_metrics[metric_id] = float(value)
                observed.add(query_id)
                predictions.append({"query_id": query_id, "metrics": normalized_metrics})
        elif prediction_mode == "ranking_only":
            if arguments.get("candidate_predictions") is not None:
                raise ValueError("ranking-only action readout forbids candidate_predictions")
        else:
            raise ValueError("terminal action prediction mode is invalid")

        ranking = arguments.get("ranking")
        if (
            not isinstance(ranking, list)
            or len(ranking) != len(query_ids)
            or any(not isinstance(item, str) for item in ranking)
            or set(ranking) != set(query_ids)
            or len(set(ranking)) != len(ranking)
        ):
            raise ValueError("ranking must contain every terminal candidate exactly once")
        selected = arguments.get("selected_action_query_id")
        if not isinstance(selected, str) or selected != ranking[0]:
            raise ValueError("selected_action_query_id must equal the first ranked candidate")
        rationale = arguments.get("selection_rationale")
        mechanism = arguments.get("mechanism_application_summary")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2000:
            raise ValueError("selection_rationale must be 1-2000 characters")
        if not isinstance(mechanism, str) or not mechanism.strip() or len(mechanism) > 1200:
            raise ValueError("mechanism_application_summary must be 1-1200 characters")
        recommendation = {
            "recommendation_type": "held_out_action_readout",
            "ranking": list(ranking),
            "selected_action_query_id": selected,
            "selection_rationale": rationale.strip(),
            "mechanism_application_summary": mechanism.strip(),
        }
        if prediction_mode == "full_metrics":
            by_id = {row["query_id"]: row for row in predictions}
            recommendation["candidate_predictions"] = [
                by_id[query_id] for query_id in query_ids
            ]
        return recommendation

    def _commit_final_recommendation(
        self,
        descriptor: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if descriptor.get("session_scope") != "campaign":
            raise RuntimeError("final recommendations are available only in campaign sessions")
        completed_count, _ = self._completed_experiment_state()
        completed_indices = self._completed_experiment_indices()
        if completed_count < 1 or not self._campaign_terminal_observed():
            raise RuntimeError("final recommendation is allowed only after campaign terminal")
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        stages = contract.get("snapshot_stages")
        if not isinstance(stages, list) or (not stages and not self._free_research(descriptor)):
            raise ValueError("checkpoint contract has no snapshot stages")
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        snapshot_root = self.ipc / "sessions" / session_id / "belief_snapshots"
        committed_snapshots = (
            len(list(snapshot_root.glob("*.json"))) if snapshot_root.exists() else 0
        )
        if committed_snapshots != len(stages):
            raise RuntimeError(
                "all required belief checkpoints must be committed before final recommendation"
            )
        action_readout_contract = (
            self._action_readout_contract()
            if descriptor.get("terminal_action_readout_required") is True
            else None
        )
        if action_readout_contract is not None:
            recommendation = self._action_readout_recommendation(
                action_readout_contract,
                arguments,
            )
        else:
            index = arguments.get("selected_experiment_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("selected_experiment_index must be an integer")
            if index not in completed_indices:
                raise ValueError(
                    "selected_experiment_index must be a lifecycle index from "
                    "campaign_closeout.completed_experiment_indices"
                )
            rationale = arguments.get("selection_rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("selection_rationale must be a non-empty string")
            if len(rationale) > 2000:
                raise ValueError("selection_rationale exceeds 2000 characters")
            recommendation = {
                "selected_experiment_index": index,
                "selection_rationale": rationale.strip(),
            }
        path = self.ipc / "sessions" / session_id / "final_recommendation.json"
        if path.exists():
            existing = _read_object(path)
            existing_recommendation = existing.get("recommendation")
            if _encode(existing_recommendation) != _encode(recommendation):
                raise RuntimeError("a different final recommendation is already committed")
            return {
                "ok": True,
                "schema_version": MCP_SERVER_VERSION,
                "already_committed": True,
                "recommendation_sha256": hashlib.sha256(_encode(recommendation)).hexdigest(),
                "instruction": "Submit the final response now; do not call more tools.",
            }
        record: dict[str, Any] = {
            "schema_version": MCP_SERVER_VERSION,
            "recommendation": recommendation,
            "recommendation_sha256": hashlib.sha256(_encode(recommendation)).hexdigest(),
            "complete_experiment_count": completed_count,
            "committed_checkpoint_count": committed_snapshots,
            "committed_after_campaign_terminal": True,
            "committed_before_blind_evaluation": True,
        }
        if action_readout_contract is not None:
            record.update(
                {
                    "recommendation_type": "held_out_action_readout",
                    "selected_action_query_id": recommendation[
                        "selected_action_query_id"
                    ],
                    "terminal_action_readout_contract_sha256": hashlib.sha256(
                        _encode(action_readout_contract)
                    ).hexdigest(),
                    "candidate_count": len(
                        action_readout_contract.get("candidate_queries", [])
                    ),
                    "committed_after_terminal_action_reveal": True,
                }
            )
        else:
            index = int(recommendation["selected_experiment_index"])
            record.update(
                {
                    "selected_batch_id": f"batch-{index:04d}",
                    "selected_completed_ordinal": completed_indices.index(index) + 1,
                }
            )
        _atomic_json(path, record)
        return {
            "ok": True,
            "schema_version": MCP_SERVER_VERSION,
            "already_committed": False,
            "recommendation_sha256": record["recommendation_sha256"],
            "instruction": "Submit the final response now; do not call more tools.",
        }

    def _campaign_terminal_observed(self) -> bool:
        if self._terminal_outcome is not None:
            return self._terminal_outcome.get("campaign_ended") is True
        progress = self._campaign_progress()
        if progress is not None:
            return progress["campaign_ended"] is True
        path = self.public / "history.jsonl"
        if not path.exists():
            return False
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line:
                continue
            value = json.loads(line)
            return isinstance(value, dict) and value.get("campaign_ended") is True
        return False

    def _completed_experiment_state(self) -> tuple[int, set[str]]:
        progress = self._campaign_progress()
        if progress is not None:
            return int(progress["completed_experiment_count"]), {
                str(item) for item in progress["observed_evidence_ids"]
            }
        rows: list[dict[str, Any]] = []
        path = self.public / "history.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        ended = [
            row
            for row in rows
            if row.get("experiment_ended") is True
            and row.get("experiment_completed") is True
        ]
        evidence = {
            str(row["evidence_id"]) for row in ended if isinstance(row.get("evidence_id"), str)
        }
        return len(ended), evidence

    def _completed_experiment_indices(self) -> list[int]:
        progress = self._campaign_progress()
        if progress is not None:
            return [int(item) for item in progress["completed_experiment_indices"]]
        path = self.public / "history.jsonl"
        indices: list[int] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("experiment_completed") is not True:
                    continue
                index = row.get("lifecycle_experiment_index")
                if isinstance(index, int) and not isinstance(index, bool) and index >= 1:
                    indices.append(index)
        return sorted(set(indices))

    def _campaign_progress(self) -> dict[str, Any] | None:
        descriptor = self._descriptor()
        if descriptor.get("session_scope") != "campaign":
            return None
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        path = self.ipc / "sessions" / session_id / "campaign_progress.json"
        if not path.is_file():
            return None
        progress = _read_object(path)
        closed = progress.get("closed_batch_count")
        count = progress.get("completed_experiment_count")
        indices = progress.get("completed_experiment_indices")
        batch_ids = progress.get("completed_batch_ids")
        evidence = progress.get("observed_evidence_ids")
        ended = progress.get("campaign_ended")
        closed_valid = isinstance(closed, int) and not isinstance(closed, bool) and closed >= 0
        count_valid = isinstance(count, int) and not isinstance(count, bool) and count >= 0
        expected_catalog = (
            [f"experiment-{index}-final-assay" for index in range(1, count + 1)]
            if count_valid
            else []
        )
        evidence_valid = isinstance(evidence, list) and all(
            isinstance(item, str) and item for item in evidence
        )
        indices_valid = (
            isinstance(indices, list)
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 1
                for item in indices
            )
            and indices == sorted(set(indices))
            and count_valid
            and len(indices) == count
            and closed_valid
            and all(item <= closed for item in indices)
        )
        batch_ids_valid = (
            isinstance(batch_ids, list)
            and indices_valid
            and batch_ids == [f"batch-{index:04d}" for index in indices]
        )
        if (
            progress.get("schema_version") != CAMPAIGN_PROGRESS_VERSION
            or not closed_valid
            or not count_valid
            or count > closed
            or not indices_valid
            or not batch_ids_valid
            or not evidence_valid
            or evidence != expected_catalog
            or not isinstance(ended, bool)
        ):
            raise ValueError("campaign progress ledger is invalid")
        return progress

    def _inspect(
        self,
        descriptor: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        artifact_id = self._leaf(str(arguments.get("artifact_id", "")), label="artifact_id")
        offset = max(self._integer(arguments.get("offset", 0), label="offset"), 0)
        cap = int(descriptor["max_tool_output_bytes"])
        requested = self._integer(arguments.get("limit", 4096), label="limit")
        limit = max(1, min(requested, max(256, cap // 2)))
        path = self.public / "artifacts" / f"{artifact_id}.json"
        text = path.read_text(encoding="utf-8")
        fragment = text[offset : offset + limit]
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        _append_jsonl(
            self.ipc / "sessions" / session_id / "artifact_access.jsonl",
            {
                "artifact_id": artifact_id,
                "offset": offset,
                "limit": limit,
                "returned_character_count": len(fragment),
                "total_character_count": len(text),
            },
        )
        return {
            "schema_version": MCP_SERVER_VERSION,
            "artifact_id": artifact_id,
            "offset": offset,
            "next_offset": offset + len(fragment),
            "total_character_count": len(text),
            "complete": offset + len(fragment) >= len(text),
            "encoding": "utf-8-json-text-fragment",
            "data": fragment,
        }

    def _step(
        self,
        descriptor: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self._terminal_outcome is not None:
            raise RuntimeError("campaign_already_ended_submit_final_response")
        if descriptor.get("session_scope") == "campaign":
            contract = _read_object(self.reference / "belief_checkpoint_contract.json")
            stages = contract.get("snapshot_stages", [])
            required_counts = contract.get("checkpoint_complete_experiments", [])
            root = self.ipc / "sessions" / str(descriptor["session_id"]) / "belief_snapshots"
            committed = len(list(root.glob("*.json"))) if root.exists() else 0
            completed_count, _ = self._completed_experiment_state()
            if (
                committed < len(stages)
                and committed < len(required_counts)
                and completed_count == int(required_counts[committed])
            ):
                raise RuntimeError(
                    f"required belief checkpoint {stages[committed]} must be committed before step"
                )
        expected_step = self._integer(arguments.get("expected_step"), label="expected_step")
        if expected_step < 1:
            raise ValueError("expected_step must be positive")
        action = arguments.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("operation"), str):
            raise ValueError("action.operation is required")
        decision_audit: dict[str, Any] | None = None
        if descriptor.get("session_scope") == "campaign" and not self._free_research(descriptor):
            raw_audit = arguments.get("decision_audit")
            if not isinstance(raw_audit, dict):
                raise ValueError("campaign step requires decision_audit")
            decision_audit = DecisionAuditRecord.from_payload(
                raw_audit,
                action=action,
            ).to_dict()
        session_id = self._leaf(str(descriptor["session_id"]), label="session_id")
        request_id_value = arguments.get("request_id")
        if request_id_value is None:
            request_id = hashlib.sha256(
                session_id.encode("utf-8")
                + b"|"
                + str(expected_step).encode("ascii")
                + b"|"
                + _encode(action)
            ).hexdigest()[:32]
        else:
            request_id = self._leaf(str(request_id_value), label="request_id")
        envelope = {
            "schema_version": IPC_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "expected_step": expected_step,
            "action": action,
        }
        if decision_audit is not None:
            envelope["decision_audit"] = decision_audit
        session_root = self.ipc / "sessions" / session_id
        request_path = session_root / "mcp_requests" / f"{request_id}.json"
        response_path = session_root / "responses" / f"{request_id}.json"
        if request_path.exists():
            if _encode(_read_object(request_path)) != _encode(envelope):
                raise ValueError("request_id was reused with another payload")
        else:
            _atomic_json(request_path, envelope)
        deadline = time.monotonic() + float(descriptor["response_timeout_s"])
        while time.monotonic() < deadline:
            if response_path.exists():
                outcome = _read_object(response_path)
                if outcome.get("campaign_ended") is True or (
                    descriptor.get("session_scope") != "campaign"
                    and outcome.get("experiment_ended") is True
                ):
                    self._terminal_outcome = outcome
                if descriptor.get("session_scope") == "campaign":
                    outcome = {
                        **outcome,
                        "campaign_closeout": self._campaign_closeout_state(descriptor),
                    }
                return outcome
            current = self._descriptor()
            if current.get("session_id") != session_id:
                raise RuntimeError("session changed before the operation completed")
            time.sleep(0.05)
        raise TimeoutError("runner did not return an operation outcome")

    @staticmethod
    def _leaf(value: str, *, label: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be a non-path string")
        return value

    @staticmethod
    def _integer(value: Any, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer")
        return value

    def _belief_snapshot_schema(self) -> dict[str, Any]:
        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        stages = [str(item) for item in contract.get("snapshot_stages", [])]
        query_contract = dict(contract.get("query_metric_contract", {}))
        query_ids = [str(item) for item in query_contract]
        metric_ids = [str(item) for item in contract.get("allowed_metric_ids", [])]
        feature_ids = [str(item) for item in contract.get("allowed_feature_ids", [])]
        prior_fields = [str(item) for item in contract.get("allowed_prior_fields", [])]
        evidence_ids = [str(item) for item in contract.get("evidence_catalog", [])]
        nominal = bool(contract.get("nominal_information_available"))
        probability = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        law_term_common = {
            "title": "Executable law term",
            "description": (
                "One typed law term. input_ids is always an array. category_value is required "
                "for categorical_level and conditional_* bases and omitted otherwise."
            ),
            "type": "object",
            "properties": {
                "term_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Participant-chosen non-empty identifier for this term.",
                },
                "basis": {
                    "enum": sorted(WORK_II_LAW_BASES),
                    "description": "Executable basis selected from this enum.",
                },
                "input_ids": {
                    "type": "array",
                    "description": (
                        "One or two feature IDs; conditional_* uses [categorical, numeric]. "
                        "Submit a JSON array, not a scalar."
                    ),
                    "items": {"enum": feature_ids},
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": True,
                },
                "coefficient": {"type": "number"},
                "category_value": {"type": ["string", "number"]},
            },
            "required": ["term_id", "basis", "input_ids", "coefficient"],
            "additionalProperties": False,
        }

        def metric_prediction(allowed_ids: list[str]) -> dict[str, Any]:
            return {
                "title": "Held-out metric prediction",
                "description": (
                    "Exactly these five fields: metric_id, mean, interval_lower, "
                    "interval_upper, and confidence."
                ),
                "type": "object",
                "properties": {
                    "metric_id": {"enum": allowed_ids},
                    "mean": {"type": "number"},
                    "interval_lower": {"type": "number"},
                    "interval_upper": {"type": "number"},
                    "confidence": probability,
                },
                "required": [
                    "metric_id",
                    "mean",
                    "interval_lower",
                    "interval_upper",
                    "confidence",
                ],
                "additionalProperties": False,
            }

        prediction_variants = []
        for query_id, raw_query_metrics in query_contract.items():
            query_metrics = [str(item) for item in raw_query_metrics]
            prediction_variants.append(
                {
                    "title": f"Held-out query prediction for {query_id}",
                    "description": (
                        "Exactly two fields: query_id and metrics. metrics must remain a nested "
                        "array of typed metric-prediction objects."
                    ),
                    "type": "object",
                    "properties": {
                        "query_id": {"const": str(query_id)},
                        "metrics": {
                            "type": "array",
                            "description": (
                                "Exactly one prediction for every required metric ID; "
                                "duplicate metric IDs are invalid."
                            ),
                            "minItems": len(query_metrics),
                            "maxItems": len(query_metrics),
                            "items": metric_prediction(query_metrics),
                            "x-chemworld-required-ids": query_metrics,
                        },
                    },
                    "required": ["query_id", "metrics"],
                    "additionalProperties": False,
                }
            )
        return {
            "title": "Work II belief snapshot",
            "description": BELIEF_SNAPSHOT_SHAPE_GUIDE,
            "type": "object",
            "properties": {
                "schema_version": {"const": WORK_II_SNAPSHOT_SCHEMA_VERSION},
                "snapshot_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "stage": {"enum": stages},
                "prior_assessment": {
                    "title": "Prior-information assessment",
                    "description": (
                        "Typed assessment of the supplied nominal information; all four listed "
                        "fields are required."
                    ),
                    "type": "object",
                    "properties": {
                        "nominal_information_available": {"const": nominal},
                        "reliability_probability": probability if nominal else {"type": "null"},
                        "suspected_misindexed_fields": {
                            "type": "array",
                            "items": {"enum": prior_fields},
                            "uniqueItems": True,
                        },
                        "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
                    },
                    "required": [
                        "nominal_information_available",
                        "reliability_probability",
                        "suspected_misindexed_fields",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
                "predictions": {
                    "type": "array",
                    "description": (
                        "A JSON array of {query_id, metrics:[...]} objects. Include exactly one "
                        "object for every required query ID and exactly its required metric IDs; "
                        "duplicate IDs and flattened named-metric fields are invalid."
                    ),
                    "minItems": len(query_ids),
                    "maxItems": len(query_ids),
                    "items": {"oneOf": prediction_variants},
                    "x-chemworld-required-ids": query_ids,
                    "x-chemworld-query-metric-contract": query_contract,
                },
                "law_summary": {
                    "title": "Executable law summary",
                    "description": (
                        "Typed executable laws. metric_laws is an array of law objects, never a "
                        "mapping or prose relationship."
                    ),
                    "type": "object",
                    "properties": {
                        "schema_version": {"const": WORK_II_LAW_SUMMARY_SCHEMA_VERSION},
                        "summary_id": {"type": "string", "minLength": 1, "maxLength": 200},
                        "feature_ids": {
                            "type": "array",
                            "description": (
                                "Non-empty JSON array of feature IDs used by this summary."
                            ),
                            "items": {"enum": feature_ids},
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "metric_laws": {
                            "type": "array",
                            "description": (
                                "Exactly one executable law for every required metric ID; "
                                "duplicate metric IDs are invalid."
                            ),
                            "minItems": len(metric_ids),
                            "maxItems": len(metric_ids),
                            "x-chemworld-required-ids": metric_ids,
                            "items": {
                                "title": "Executable metric law",
                                "description": (
                                    "Exactly one metric law with numeric intercept and bounds. "
                                    "terms is required and may be an empty array."
                                ),
                                "type": "object",
                                "properties": {
                                    "metric_id": {"enum": metric_ids},
                                    "intercept": {"type": "number"},
                                    "link": {"enum": sorted(WORK_II_LAW_LINKS)},
                                    "lower_bound": {"type": "number"},
                                    "upper_bound": {"type": "number"},
                                    "terms": {
                                        "type": "array",
                                        "description": (
                                            "JSON array of executable term objects; terms:[] is "
                                            "valid when no feature term is asserted."
                                        ),
                                        "maxItems": 64,
                                        "items": law_term_common,
                                    },
                                },
                                "required": [
                                    "metric_id",
                                    "intercept",
                                    "link",
                                    "lower_bound",
                                    "upper_bound",
                                    "terms",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"enum": evidence_ids},
                            "uniqueItems": True,
                        },
                        "applicability": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "limitations": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 200},
                            "maxItems": 16,
                            "uniqueItems": True,
                        },
                        "confidence": probability,
                    },
                    "required": [
                        "schema_version",
                        "summary_id",
                        "feature_ids",
                        "metric_laws",
                        "evidence_ids",
                        "applicability",
                        "limitations",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"enum": evidence_ids},
                    "uniqueItems": True,
                },
                "next_experiment_intent": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                },
                "overall_confidence": probability,
            },
            "required": [
                "schema_version",
                "snapshot_id",
                "stage",
                "prior_assessment",
                "predictions",
                "law_summary",
                "evidence_ids",
                "next_experiment_intent",
                "overall_confidence",
            ],
            "additionalProperties": False,
        }

    def _staged_belief_snapshot_tool_schema(self) -> dict[str, Any]:
        """Derive the staged participant grammar from the active public contract."""

        contract = _read_object(self.reference / "belief_checkpoint_contract.json")
        complete_schema = self._belief_snapshot_schema()
        complete_properties = complete_schema["properties"]
        plan = self._page_plan(contract)
        try:
            active_stage = self._belief_stage_context(self._descriptor())["stage"]
        except RuntimeError as error:
            if "already committed" not in str(error):
                raise
            active_stage = None
        probability = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        string_id = {"type": "string", "minLength": 1, "maxLength": 200}
        string_ids = {
            "type": "array",
            "items": string_id,
            "maxItems": 64,
            "uniqueItems": True,
        }
        metric_prediction = {
            "type": "object",
            "properties": {
                "metric_id": {
                    "enum": [str(item) for item in contract.get("allowed_metric_ids", [])]
                },
                "mean": {"type": "number"},
                "interval_lower": {"type": "number"},
                "interval_upper": {"type": "number"},
                "confidence": probability,
            },
            "required": [
                "metric_id",
                "mean",
                "interval_lower",
                "interval_upper",
                "confidence",
            ],
            "additionalProperties": False,
        }
        law_term = {
            "type": "object",
            "properties": {
                "term_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "basis": {"enum": sorted(WORK_II_LAW_BASES)},
                "input_ids": {
                    "type": "array",
                    "items": {
                        "enum": [str(item) for item in contract.get("allowed_feature_ids", [])]
                    },
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": True,
                },
                "coefficient": {"type": "number"},
                "category_value": {"type": ["string", "number"]},
            },
            "required": ["term_id", "basis", "input_ids", "coefficient"],
            "additionalProperties": False,
        }
        metric_law = {
            "type": "object",
            "properties": {
                "metric_id": {
                    "enum": [str(item) for item in contract.get("allowed_metric_ids", [])]
                },
                "intercept": {"type": "number"},
                "link": {"enum": sorted(WORK_II_LAW_LINKS)},
                "lower_bound": {"type": "number"},
                "upper_bound": {"type": "number"},
                "terms": {"type": "array", "items": law_term, "maxItems": 64},
            },
            "required": [
                "metric_id",
                "intercept",
                "link",
                "lower_bound",
                "upper_bound",
                "terms",
            ],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "enum": [
                        "begin",
                        "append_prediction_page",
                        "append_law_page",
                        "finalize",
                    ]
                },
                "snapshot_header": {
                    "type": "object",
                    "description": (
                        "Complete snapshot except predictions and law_summary.metric_laws."
                    ),
                    "properties": {
                        "snapshot_id": deepcopy(complete_properties["snapshot_id"]),
                        "stage": (
                            {"const": active_stage}
                            if active_stage is not None
                            else deepcopy(complete_properties["stage"])
                        ),
                        "prior_assessment": deepcopy(
                            complete_properties["prior_assessment"]
                        ),
                        "law_summary": {
                            "type": "object",
                            "description": "Complete law summary except metric_laws.",
                            "properties": {
                                "schema_version": {"const": WORK_II_LAW_SUMMARY_SCHEMA_VERSION},
                                "summary_id": string_id,
                                "feature_ids": {
                                    "type": "array",
                                    "items": {
                                        "enum": [
                                            str(item)
                                            for item in contract.get("allowed_feature_ids", [])
                                        ]
                                    },
                                    "maxItems": 64,
                                    "uniqueItems": True,
                                    "minItems": 1,
                                },
                                "evidence_ids": deepcopy(
                                    complete_properties["law_summary"]["properties"][
                                        "evidence_ids"
                                    ]
                                ),
                                "applicability": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 2000,
                                },
                                "limitations": {
                                    **string_ids,
                                    "maxItems": 16,
                                },
                                "confidence": probability,
                            },
                            "required": [
                                "schema_version",
                                "summary_id",
                                "feature_ids",
                                "evidence_ids",
                                "applicability",
                                "limitations",
                                "confidence",
                            ],
                            "additionalProperties": False,
                        },
                        "evidence_ids": deepcopy(complete_properties["evidence_ids"]),
                        "next_experiment_intent": deepcopy(
                            complete_properties["next_experiment_intent"]
                        ),
                        "overall_confidence": deepcopy(
                            complete_properties["overall_confidence"]
                        ),
                    },
                    "required": [
                        "snapshot_id",
                        "stage",
                        "prior_assessment",
                        "law_summary",
                        "evidence_ids",
                        "next_experiment_intent",
                        "overall_confidence",
                    ],
                    "additionalProperties": False,
                },
                "page_id": {"enum": list(plan["submission_order"])},
                "predictions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": PREDICTION_PAGE_SIZE,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query_id": {
                                "enum": [
                                    str(item)
                                    for item in dict(
                                        contract.get("query_metric_contract", {})
                                    )
                                ]
                            },
                            "metrics": {
                                "type": "array",
                                "items": metric_prediction,
                                "minItems": 1,
                                "maxItems": 32,
                            },
                        },
                        "required": ["query_id", "metrics"],
                        "additionalProperties": False,
                    },
                    "x-chemworld-page-contract": {
                        str(page["page_id"]): deepcopy(page["query_metric_contract"])
                        for page in plan["prediction_pages"]
                    },
                },
                "metric_laws": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": LAW_PAGE_SIZE,
                    "items": metric_law,
                    "x-chemworld-page-contract": {
                        str(page["page_id"]): list(page["metric_ids"])
                        for page in plan["law_pages"]
                    },
                },
            },
            "required": ["action"],
            "additionalProperties": False,
            "oneOf": [],
            "x-chemworld-submission-order": list(plan["submission_order"]),
        }
        # Keep the provider-facing grammar compact. The root properties above retain
        # the exact enums and page contracts, while the host performs the authoritative
        # page-specific validation. Expanding every query/metric combination into
        # nested oneOf branches makes large electrochemical contracts too complex for
        # some OpenAI-compatible providers to register as a function schema.
        schema["oneOf"] = [
            {
                "type": "object",
                "properties": {
                    "action": {"const": "begin"},
                    "snapshot_header": {"type": "object"},
                },
                "required": ["action", "snapshot_header"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"const": "append_prediction_page"},
                    "page_id": {"type": "string"},
                    "predictions": {"type": "array"},
                },
                "required": ["action", "page_id", "predictions"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"const": "append_law_page"},
                    "page_id": {"type": "string"},
                    "metric_laws": {"type": "array"},
                },
                "required": ["action", "page_id", "metric_laws"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"action": {"const": "finalize"}},
                "required": ["action"],
                "additionalProperties": False,
            },
        ]
        return schema

    @staticmethod
    def _decision_audit_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expected_effect": {"type": "string", "minLength": 1, "maxLength": 1000},
                "diagnostic_target": {"type": "string", "minLength": 1, "maxLength": 1000},
                "expected_information_gain": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "belief_update_rule": {
                    "type": "object",
                    "properties": {
                        "if_supported": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "if_not_supported": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                    },
                    "required": ["if_supported", "if_not_supported"],
                    "additionalProperties": False,
                },
                "uncertainty": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "adaptation_source": {
                    "enum": ["none", "measurement", "spectrum", "experiment_memory", "validator"]
                },
            },
            "required": [
                "expected_effect",
                "diagnostic_target",
                "expected_information_gain",
                "belief_update_rule",
                "uncertainty",
                "adaptation_source",
            ],
            "additionalProperties": False,
        }

    def _tool_definitions(self) -> list[dict[str, Any]]:
        descriptor = self._descriptor()
        campaign = descriptor.get("session_scope") == "campaign"
        typed_checkpoints = campaign and not self._free_research(descriptor)
        action_readout = descriptor.get("terminal_action_readout_required") is True
        action_contract = self._action_readout_contract() if action_readout else None
        action_prediction_mode = (
            str(action_contract.get("prediction_mode", "full_metrics"))
            if isinstance(action_contract, dict)
            else "full_metrics"
        )
        metric_ids = (
            [str(item) for item in action_contract.get("metric_ids", [])]
            if isinstance(action_contract, dict)
            else []
        )
        candidate_count = (
            len(action_contract.get("candidate_queries", []))
            if isinstance(action_contract, dict)
            and isinstance(action_contract.get("candidate_queries"), list)
            else 0
        )
        final_recommendation_schema: dict[str, Any]
        if action_readout:
            properties: dict[str, Any] = {
                "ranking": {
                    "type": "array",
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "selected_action_query_id": {"type": "string", "minLength": 1},
                "selection_rationale": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                },
                "mechanism_application_summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1200,
                },
            }
            required = [
                "ranking",
                "selected_action_query_id",
                "selection_rationale",
                "mechanism_application_summary",
            ]
            if action_prediction_mode == "full_metrics":
                properties["candidate_predictions"] = {
                    "type": "array",
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query_id": {"type": "string", "minLength": 1},
                            "metrics": {
                                "type": "object",
                                "properties": {
                                    metric_id: {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    }
                                    for metric_id in metric_ids
                                },
                                "required": metric_ids,
                                "additionalProperties": False,
                            },
                        },
                        "required": ["query_id", "metrics"],
                        "additionalProperties": False,
                    },
                }
                required.insert(0, "candidate_predictions")
            final_recommendation_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        else:
            final_recommendation_schema = {
                "type": "object",
                "properties": {
                    "selected_experiment_index": {"type": "integer", "minimum": 1},
                    "selection_rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                },
                "required": ["selected_experiment_index", "selection_rationale"],
                "additionalProperties": False,
            }
        snapshot_schema = (
            self._staged_belief_snapshot_tool_schema() if typed_checkpoints else {"type": "object"}
        )
        read_annotations = {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        return [
            {
                "name": "material_information",
                "description": "Read the bounded material information assigned to this cell once.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                "annotations": read_annotations,
            },
            {
                "name": "status",
                "description": "Read the latest bounded public experiment and campaign state.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                "annotations": read_annotations,
            },
            {
                "name": "history",
                "description": (
                    "Read public outcome history; defaults to recent events. Use offset=0 for "
                    "the first event, then next_offset to page through complete retained history."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                        "offset": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
                "annotations": read_annotations,
            },
            {
                "name": "inspect_artifact",
                "description": (
                    "Read one bounded fragment of a referenced public characterization artifact."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string", "minLength": 1},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["artifact_id"],
                    "additionalProperties": False,
                },
                "annotations": read_annotations,
            },
            {
                "name": "belief_snapshot_status",
                "description": (
                    "Read the exact next belief-snapshot action and its compact typed contract, "
                    "including legal feature, metric, prior-field, evidence, page, and stage IDs. "
                    "This call is read-only and is the preferred recovery path after a rejected "
                    "belief-snapshot fragment."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "annotations": read_annotations,
            },
            {
                "name": "commit_belief_snapshot",
                "description": (
                    "Stage and finalize the next typed Work II belief snapshot inside the active "
                    "campaign session. The input schema is derived from the active public "
                    "contract and enumerates its legal stages, feature/evidence IDs, page IDs, "
                    "query IDs, and metric IDs. material_information, begin, and status publish "
                    "the same fixed page_id to exact query/metric ID plan. "
                    "The host validates stage order, experiment-count location, exact page IDs, "
                    "evidence references, held-out predictions, and executable laws. No accepted "
                    "fragment can be replaced and no participant payload is auto-repaired. "
                    f"{STAGED_BELIEF_SNAPSHOT_GUIDE}"
                ),
                "inputSchema": snapshot_schema,
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            },
            {
                "name": "terminal_action_readout",
                "description": (
                    "Reveal the fixed held-out candidate operations only after campaign terminal "
                    "and after every required belief checkpoint is committed. Before that gate, "
                    "the tool returns only an unavailable status and never candidate contents. "
                    "Candidate outcomes and hidden ranks are never revealed."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "annotations": read_annotations,
            },
            {
                "name": "commit_final_recommendation",
                "description": (
                    (
                        "After terminal_action_readout, submit predictions for every candidate, "
                        "rank every candidate exactly once, and commit the first-ranked held-out "
                        "action. The already committed final belief checkpoint cannot be revised. "
                        if action_readout
                        else "After campaign terminal and the final belief checkpoint, commit "
                        "exactly one participant-selected completed experiment for "
                        "evaluator-owned blind replay. The index is the completed batch's "
                        "1-based lifecycle_experiment_index. "
                    )
                    + "The host stores the selection atomically; a repeated identical call is "
                    "idempotent and a differing second selection is rejected."
                ),
                "inputSchema": final_recommendation_schema,
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            },
            {
                "name": "step",
                "description": (
                    "Submit exactly one operation to the authoritative ChemWorld "
                    "runner and wait for its public outcome. In campaign scope, a batch "
                    "ending does not close this tool; campaign_ended=true closes it. Each campaign "
                    "outcome includes campaign_closeout; when checkpoint_due=true, commit "
                    "next_checkpoint_stage before calling step again."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expected_step": {"type": "integer", "minimum": 1},
                        "action": {
                            "type": "object",
                            "properties": {"operation": {"type": "string", "minLength": 1}},
                            "required": ["operation"],
                            "additionalProperties": True,
                        },
                        "request_id": {"type": "string", "minLength": 1},
                        "decision_audit": self._decision_audit_schema(),
                    },
                    "required": (
                        ["expected_step", "action", "decision_audit"]
                        if typed_checkpoints
                        else ["expected_step", "action"]
                    ),
                    "additionalProperties": False,
                },
                "annotations": {
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            },
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    return ChemWorldMCPServer(args.workspace).run()


if __name__ == "__main__":
    raise SystemExit(main())
