"""One Codex process controls one complete ChemWorld experiment through file IPC."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import IO, Any, Literal, Protocol
from uuid import uuid4

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.experiment_codex_ipc import (
    EXPERIMENT_CODEX_IPC_VERSION,
    ExperimentCodexIPCError,
    ExperimentCodexWorkspace,
    IPCRequest,
    diff_agent_snapshots,
)
from chemworld.agents.experiment_codex_mcp import MCP_SERVER_VERSION, SUPPORTED_TOOLS
from chemworld.agents.interaction import AgentDecisionContext, InteractionCapabilities
from chemworld.agents.prompt_context import compact_action, summarize_measurement
from chemworld.data.logging import to_builtin
from chemworld.providers.codex_subscription import (
    AUTH_SOURCE,
    HTTPS_PROVIDER_ID,
    MODEL_ACCESS_DATE,
    MODEL_SOURCE,
    SUPPORTED_MODELS,
)

INTERACTIVE_CODEX_EXPERIMENT_VERSION = "chemworld-interactive-codex-experiment-0.6"
DEFAULT_FINALIZATION_TIMEOUT_S = 300.0
TEMP_DIRECTORY_CLEANUP_RETRY_LIMIT = 20
TEMP_DIRECTORY_CLEANUP_RETRY_INTERVAL_S = 0.05

ELIGIBLE_ZERO_ACTION_INFRASTRUCTURE_PREDECESSOR = (
    "eligible_zero_action_infrastructure_predecessor"
)
TERMINAL_ACCEPTED_PRE_ACTION_RETRY_CLASSIFICATION = "terminal_accepted"
_PROCESS_EXITED_BEFORE_FIRST_REQUEST = "process_exited_before_first_request"
_PROCESS_EXITED_BEFORE_NEXT_ACTION = "process_exited_before_next_action"
_REQUEST_WAIT_TIMEOUT = "request_wait_timeout"


def _provider_error_category(message: str) -> str:
    """Classify a provider error without retaining its sensitive raw text."""

    normalized = message.casefold()
    categories = (
        ("rate_limit", ("rate limit", "too many requests", "status 429", "http 429")),
        (
            "quota_or_balance",
            (
                "quota",
                "insufficient balance",
                "insufficient credit",
                "billing",
                "payment",
            ),
        ),
        ("capacity", ("capacity", "overloaded", "temporarily unavailable")),
        (
            "authentication",
            ("unauthorized", "invalid api key", "authentication", "status 401"),
        ),
        ("permission", ("forbidden", "permission", "status 403")),
        ("model_unavailable", ("model not found", "model_not_found", "unknown model")),
        ("context_limit", ("context length", "maximum context", "too many tokens")),
        (
            "request_validation",
            ("bad request", "invalid request", "invalid_request", "status 400"),
        ),
        (
            "provider_server",
            (
                "bad gateway",
                "service unavailable",
                "gateway timeout",
                "status 500",
                "status 502",
                "status 503",
                "status 504",
                # DeepSeek may emit the undocumented 551 status for a
                # provider-side rejection while the account and a minimal
                # model request remain healthy. Treat it as transient server
                # infrastructure, never as a scientific or balance failure.
                "status 551",
                "http 551",
            ),
        ),
        (
            "network",
            ("connection", "dns", "timed out", "timeout", "tls", "socket"),
        ),
    )
    for category, markers in categories:
        if any(marker in normalized for marker in markers):
            return category
    return "provider_rejection_unclassified"


MCP_TOOL_FAILURE_TAXONOMY_VERSION = "chemworld-mcp-tool-failure-taxonomy-0.1"
MCP_TOOL_FAILURE_EPISODE_TAXONOMY_VERSION = (
    "chemworld-mcp-tool-failure-recovery-episode-taxonomy-0.2"
)
MCP_TOOL_FAILURE_EPISODE_TAXONOMY_LEGACY_VERSIONS = {
    "chemworld-mcp-tool-failure-recovery-episode-taxonomy-0.1",
}
MCP_DUPLICATE_FAILURE_BURST_WINDOW_MS = 1000.0
MCP_DEPENDENT_BATCH_CASCADE_WINDOW_MS = 1000.0
MCP_TOOL_FAILURE_CATEGORIES = (
    "provider_network",
    "transport_ipc_os",
    "agent_invalid",
    "unclassified",
)
MCPToolFailureCategory = Literal[
    "provider_network",
    "transport_ipc_os",
    "agent_invalid",
    "unclassified",
]


def _empty_mcp_tool_failure_counts() -> dict[str, int]:
    return dict.fromkeys(MCP_TOOL_FAILURE_CATEGORIES, 0)


def _classify_mcp_tool_failure(
    call: Mapping[str, Any],
) -> MCPToolFailureCategory | None:
    """Classify one retained MCP failure without changing the legacy denominator."""

    if call.get("status") == "completed":
        return None
    error_code = str(call.get("error_code") or "").strip().lower()
    error_type = str(call.get("error_type") or "").strip().lower()
    detail = str(call.get("error_detail") or "").strip().lower()
    if (
        error_code.startswith(("provider_", "network_", "http_", "api_"))
        or error_type
        in {
            "apiconnectionerror",
            "apierror",
            "apitimeouterror",
            "connectionerror",
            "connectionreseterror",
            "httperror",
            "httpstatuserror",
            "ratelimiterror",
        }
        or any(
            marker in detail
            for marker in (
                "provider connection",
                "provider request",
                "rate limit",
                "http status",
            )
        )
    ):
        return "provider_network"
    if (
        error_code
        in {
            "atomic_replace_permission_error",
            "ipc_error",
            "ipc_session_changed",
            "os_error",
            "platform_checkpoint_contract_invalid",
            "platform_tool_output_limit",
            "transport_error",
        }
        or error_type
        in {
            "brokenpipeerror",
            "eoferror",
            "experimentcodexipcerror",
            "fileexistserror",
            "filenotfounderror",
            "memoryerror",
            "oserror",
            "permissionerror",
            "timeouterror",
        }
        or any(
            marker in detail
            for marker in (
                "active_session.json",
                "broken pipe",
                "ipc",
                "paging file",
                "permission denied",
                "system resources",
                "transport",
                "winerror",
            )
        )
    ):
        return "transport_ipc_os"
    if (
        error_code.startswith(("invalid_", "missing_"))
        or error_code == "validation_error"
        or error_type in {"keyerror", "typeerror", "valueerror"}
        or any(
            marker in detail
            for marker in (
                "checkpoint is not due",
                "must be committed before",
                "required belief checkpoint",
                "required before step",
            )
        )
    ):
        return "agent_invalid"
    return "unclassified"


def _mcp_tool_failure_audit(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return raw failures plus feedback-aware recovery episodes from ordered calls.

    Raw calls are never removed.  A maximal consecutive burst of the same failed
    outcome within one second is additionally counted as one recovery episode.  A
    failed staged-snapshot ``begin`` followed in the same queued batch by dependent
    page/finalize calls is also one episode: the dependent calls could not react to
    the rejected begin.  This classification never accepts or repairs their payloads.
    """

    def started_at_ms(call: Mapping[str, Any]) -> float | None:
        value = call.get("started_at")
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0
        except ValueError:
            return None

    def duplicate_identity(
        call: Mapping[str, Any], category: MCPToolFailureCategory
    ) -> tuple[str, ...]:
        return (
            category,
            str(call.get("tool") or ""),
            str(call.get("error_type") or ""),
            str(call.get("error_code") or ""),
            str(call.get("error_field_path") or ""),
            str(call.get("error_detail_sha256") or call.get("error_detail") or ""),
            str(call.get("result_sha256") or ""),
        )

    def is_staged_begin_failure(call: Mapping[str, Any], category: MCPToolFailureCategory) -> bool:
        argument_keys = call.get("argument_keys")
        return bool(
            category == "agent_invalid"
            and call.get("tool") == "commit_belief_snapshot"
            and isinstance(argument_keys, list)
            and set(argument_keys) == {"action", "snapshot_header"}
        )

    def is_begin_dependency_failure(
        call: Mapping[str, Any], category: MCPToolFailureCategory
    ) -> bool:
        return bool(
            category == "agent_invalid"
            and call.get("tool") == "commit_belief_snapshot"
            and str(call.get("error_detail") or "").strip().lower()
            == "begin must be accepted before belief snapshot pages"
        )

    counts = _empty_mcp_tool_failure_counts()
    current_by_category = _empty_mcp_tool_failure_counts()
    maximum_by_category = _empty_mcp_tool_failure_counts()
    total = 0
    current_total = 0
    maximum_total = 0
    episode_counts = _empty_mcp_tool_failure_counts()
    current_episodes_by_category = _empty_mcp_tool_failure_counts()
    maximum_episodes_by_category = _empty_mcp_tool_failure_counts()
    episode_total = 0
    current_episode_total = 0
    maximum_episode_total = 0
    previous_failure_identity: tuple[str, ...] | None = None
    previous_failure_started_at_ms: float | None = None
    staged_begin_failure_started_at_ms: float | None = None
    staged_begin_cascade_size = 0
    dependent_batch_cascade_failure_count = 0
    dependent_batch_cascade_group_count = 0
    maximum_dependent_batch_cascade_size = 0
    dependent_batch_cascades: list[dict[str, Any]] = []
    active_dependent_batch_cascade: dict[str, Any] | None = None
    staged_begin_failure_call_index: int | None = None
    for call_index, call in enumerate(calls):
        category = _classify_mcp_tool_failure(call)
        if category is None:
            current_total = 0
            current_by_category = _empty_mcp_tool_failure_counts()
            current_episode_total = 0
            current_episodes_by_category = _empty_mcp_tool_failure_counts()
            previous_failure_identity = None
            previous_failure_started_at_ms = None
            staged_begin_failure_started_at_ms = None
            staged_begin_failure_call_index = None
            staged_begin_cascade_size = 0
            active_dependent_batch_cascade = None
            continue
        total += 1
        current_total += 1
        maximum_total = max(maximum_total, current_total)
        counts[category] += 1
        for name in MCP_TOOL_FAILURE_CATEGORIES:
            if name == category:
                current_by_category[name] += 1
            maximum_by_category[name] = max(maximum_by_category[name], current_by_category[name])
        identity = duplicate_identity(call, category)
        timestamp_ms = started_at_ms(call)
        same_dependent_batch_cascade = bool(
            staged_begin_failure_started_at_ms is not None
            and timestamp_ms is not None
            and 0.0
            <= timestamp_ms - staged_begin_failure_started_at_ms
            <= MCP_DEPENDENT_BATCH_CASCADE_WINDOW_MS
            and is_begin_dependency_failure(call, category)
        )
        same_duplicate_burst = bool(
            previous_failure_identity == identity
            and previous_failure_started_at_ms is not None
            and timestamp_ms is not None
            and 0.0
            <= timestamp_ms - previous_failure_started_at_ms
            <= MCP_DUPLICATE_FAILURE_BURST_WINDOW_MS
        )
        if same_dependent_batch_cascade:
            dependent_batch_cascade_failure_count += 1
            staged_begin_cascade_size += 1
            if staged_begin_cascade_size == 1:
                dependent_batch_cascade_group_count += 1
                active_dependent_batch_cascade = {
                    "classification": "dependent_batch_cascade",
                    "upstream_call_index": staged_begin_failure_call_index,
                    "dependent_call_indices": [],
                }
                dependent_batch_cascades.append(active_dependent_batch_cascade)
            if active_dependent_batch_cascade is not None:
                active_dependent_batch_cascade["dependent_call_indices"].append(call_index)
            maximum_dependent_batch_cascade_size = max(
                maximum_dependent_batch_cascade_size,
                staged_begin_cascade_size,
            )
        elif not same_duplicate_burst:
            episode_total += 1
            current_episode_total += 1
            maximum_episode_total = max(maximum_episode_total, current_episode_total)
            episode_counts[category] += 1
            for name in MCP_TOOL_FAILURE_CATEGORIES:
                if name == category:
                    current_episodes_by_category[name] += 1
                maximum_episodes_by_category[name] = max(
                    maximum_episodes_by_category[name],
                    current_episodes_by_category[name],
                )
        if not same_dependent_batch_cascade:
            if is_staged_begin_failure(call, category):
                staged_begin_failure_started_at_ms = timestamp_ms
                staged_begin_failure_call_index = call_index
                staged_begin_cascade_size = 0
                active_dependent_batch_cascade = None
            else:
                staged_begin_failure_started_at_ms = None
                staged_begin_failure_call_index = None
                staged_begin_cascade_size = 0
                active_dependent_batch_cascade = None
        previous_failure_identity = identity
        previous_failure_started_at_ms = timestamp_ms
    return {
        "schema_version": MCP_TOOL_FAILURE_TAXONOMY_VERSION,
        "recovered_mcp_tool_failure_count": total,
        "current_consecutive_mcp_tool_failure_count": current_total,
        "maximum_consecutive_mcp_tool_failure_count": maximum_total,
        "counts_by_category": counts,
        "current_consecutive_counts_by_category": current_by_category,
        "maximum_consecutive_counts_by_category": maximum_by_category,
        "recovery_episode_taxonomy": {
            "schema_version": MCP_TOOL_FAILURE_EPISODE_TAXONOMY_VERSION,
            "grouping_basis": (
                "maximal_consecutive_identical_failed_outcomes_without_an_"
                "intervening_success_within_the_frozen_duplicate_burst_window_or_"
                "dependent_staged_calls_after_an_unaccepted_begin_in_the_same_batch"
            ),
            "duplicate_failure_burst_window_ms": MCP_DUPLICATE_FAILURE_BURST_WINDOW_MS,
            "dependent_batch_cascade_window_ms": MCP_DEPENDENT_BATCH_CASCADE_WINDOW_MS,
            "dependent_batch_cascade_failure_count": (dependent_batch_cascade_failure_count),
            "dependent_batch_cascade_group_count": dependent_batch_cascade_group_count,
            "maximum_dependent_batch_cascade_size": maximum_dependent_batch_cascade_size,
            "dependent_batch_cascades": dependent_batch_cascades,
            "recovery_episode_count": episode_total,
            "current_consecutive_recovery_episode_count": current_episode_total,
            "maximum_consecutive_recovery_episode_count": maximum_episode_total,
            "counts_by_category": episode_counts,
            "current_consecutive_counts_by_category": current_episodes_by_category,
            "maximum_consecutive_counts_by_category": maximum_episodes_by_category,
        },
    }


def validated_mcp_tool_failure_budget(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate typed receipt counters, or use legacy counters when taxonomy is absent."""

    def nonnegative_integer(value: Any, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    legacy_total = nonnegative_integer(
        receipt.get("recovered_mcp_tool_failure_count"),
        label="recovered_mcp_tool_failure_count",
    )
    legacy_current = nonnegative_integer(
        receipt.get("current_consecutive_mcp_tool_failure_count", 0),
        label="current_consecutive_mcp_tool_failure_count",
    )
    legacy_maximum = nonnegative_integer(
        receipt.get("maximum_consecutive_mcp_tool_failure_count"),
        label="maximum_consecutive_mcp_tool_failure_count",
    )
    if legacy_current > legacy_maximum or legacy_maximum > legacy_total:
        raise ValueError("legacy MCP failure counters are inconsistent")
    typed_fields = (
        "mcp_tool_failure_taxonomy",
        "scientific_compliance_mcp_tool_failure_count",
        "current_consecutive_scientific_compliance_mcp_tool_failure_count",
        "maximum_consecutive_scientific_compliance_mcp_tool_failure_count",
    )
    present = [field in receipt for field in typed_fields]
    if not any(present):
        return {
            "source": "legacy_total_failures",
            "scientific_count": legacy_total,
            "scientific_current": legacy_current,
            "scientific_maximum": legacy_maximum,
            "scientific_episode_count": legacy_total,
            "scientific_episode_current": legacy_current,
            "scientific_episode_maximum": legacy_maximum,
            "transport_count": 0,
            "transport_maximum": 0,
            "provider_network_count": 0,
            "unclassified_count": 0,
        }
    if not all(present):
        raise ValueError("typed MCP failure receipt is incomplete")
    taxonomy = receipt["mcp_tool_failure_taxonomy"]
    if not isinstance(taxonomy, Mapping):
        raise ValueError("mcp_tool_failure_taxonomy must be an object")
    if taxonomy.get("schema_version") != MCP_TOOL_FAILURE_TAXONOMY_VERSION:
        raise ValueError("MCP tool failure taxonomy schema is unsupported")

    maps: dict[str, dict[str, int]] = {}
    for map_name in (
        "counts_by_category",
        "current_consecutive_counts_by_category",
        "maximum_consecutive_counts_by_category",
    ):
        raw = taxonomy.get(map_name)
        if not isinstance(raw, Mapping) or set(raw) != set(MCP_TOOL_FAILURE_CATEGORIES):
            raise ValueError(f"{map_name} must contain the complete MCP failure taxonomy")
        maps[map_name] = {
            category: nonnegative_integer(raw[category], label=f"{map_name}.{category}")
            for category in MCP_TOOL_FAILURE_CATEGORIES
        }
    counts = maps["counts_by_category"]
    currents = maps["current_consecutive_counts_by_category"]
    maximums = maps["maximum_consecutive_counts_by_category"]
    for category in MCP_TOOL_FAILURE_CATEGORIES:
        if currents[category] > maximums[category] or maximums[category] > counts[category]:
            raise ValueError(f"typed MCP failure counters are inconsistent for {category}")
    typed_total = nonnegative_integer(
        taxonomy.get("recovered_mcp_tool_failure_count"),
        label="mcp_tool_failure_taxonomy.recovered_mcp_tool_failure_count",
    )
    typed_current = nonnegative_integer(
        taxonomy.get("current_consecutive_mcp_tool_failure_count"),
        label="mcp_tool_failure_taxonomy.current_consecutive_mcp_tool_failure_count",
    )
    typed_maximum = nonnegative_integer(
        taxonomy.get("maximum_consecutive_mcp_tool_failure_count"),
        label="mcp_tool_failure_taxonomy.maximum_consecutive_mcp_tool_failure_count",
    )
    if (
        typed_total != sum(counts.values())
        or typed_total != legacy_total
        or typed_current != legacy_current
        or typed_maximum != legacy_maximum
    ):
        raise ValueError("typed and legacy MCP failure counters disagree")
    scientific_count = nonnegative_integer(
        receipt["scientific_compliance_mcp_tool_failure_count"],
        label="scientific_compliance_mcp_tool_failure_count",
    )
    scientific_current = nonnegative_integer(
        receipt["current_consecutive_scientific_compliance_mcp_tool_failure_count"],
        label="current_consecutive_scientific_compliance_mcp_tool_failure_count",
    )
    scientific_maximum = nonnegative_integer(
        receipt["maximum_consecutive_scientific_compliance_mcp_tool_failure_count"],
        label="maximum_consecutive_scientific_compliance_mcp_tool_failure_count",
    )
    if (
        scientific_count != counts["agent_invalid"]
        or scientific_current != currents["agent_invalid"]
        or scientific_maximum != maximums["agent_invalid"]
    ):
        raise ValueError("scientific MCP failure counters disagree with taxonomy")
    episode_fields = (
        "scientific_compliance_mcp_tool_failure_episode_count",
        "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count",
        "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count",
    )
    episode_present = [field in receipt for field in episode_fields]
    episode_taxonomy = taxonomy.get("recovery_episode_taxonomy")
    if not any(episode_present) and episode_taxonomy is None:
        scientific_episode_count = scientific_count
        scientific_episode_current = scientific_current
        scientific_episode_maximum = scientific_maximum
        source = "validated_typed_taxonomy_without_episode_grouping"
    else:
        if not all(episode_present) or not isinstance(episode_taxonomy, Mapping):
            raise ValueError("recovery episode MCP failure receipt is incomplete")
        if episode_taxonomy.get("schema_version") not in {
            MCP_TOOL_FAILURE_EPISODE_TAXONOMY_VERSION,
            *MCP_TOOL_FAILURE_EPISODE_TAXONOMY_LEGACY_VERSIONS,
        }:
            raise ValueError("MCP recovery episode taxonomy schema is unsupported")
        episode_maps: dict[str, dict[str, int]] = {}
        for map_name in (
            "counts_by_category",
            "current_consecutive_counts_by_category",
            "maximum_consecutive_counts_by_category",
        ):
            raw = episode_taxonomy.get(map_name)
            if not isinstance(raw, Mapping) or set(raw) != set(MCP_TOOL_FAILURE_CATEGORIES):
                raise ValueError(
                    f"recovery_episode_taxonomy.{map_name} must contain the complete "
                    "MCP failure taxonomy"
                )
            episode_maps[map_name] = {
                category: nonnegative_integer(
                    raw[category],
                    label=f"recovery_episode_taxonomy.{map_name}.{category}",
                )
                for category in MCP_TOOL_FAILURE_CATEGORIES
            }
        episode_counts = episode_maps["counts_by_category"]
        episode_currents = episode_maps["current_consecutive_counts_by_category"]
        episode_maximums = episode_maps["maximum_consecutive_counts_by_category"]
        for category in MCP_TOOL_FAILURE_CATEGORIES:
            if (
                episode_currents[category] > episode_maximums[category]
                or episode_maximums[category] > episode_counts[category]
                or episode_counts[category] > counts[category]
            ):
                raise ValueError(f"recovery episode counters are inconsistent for {category}")
        episode_total = nonnegative_integer(
            episode_taxonomy.get("recovery_episode_count"),
            label="recovery_episode_taxonomy.recovery_episode_count",
        )
        episode_current = nonnegative_integer(
            episode_taxonomy.get("current_consecutive_recovery_episode_count"),
            label="recovery_episode_taxonomy.current_consecutive_recovery_episode_count",
        )
        episode_maximum = nonnegative_integer(
            episode_taxonomy.get("maximum_consecutive_recovery_episode_count"),
            label="recovery_episode_taxonomy.maximum_consecutive_recovery_episode_count",
        )
        if (
            episode_total != sum(episode_counts.values())
            or episode_current > episode_maximum
            or episode_maximum > episode_total
        ):
            raise ValueError("recovery episode totals are inconsistent")
        scientific_episode_count = nonnegative_integer(
            receipt[episode_fields[0]], label=episode_fields[0]
        )
        scientific_episode_current = nonnegative_integer(
            receipt[episode_fields[1]], label=episode_fields[1]
        )
        scientific_episode_maximum = nonnegative_integer(
            receipt[episode_fields[2]], label=episode_fields[2]
        )
        if (
            scientific_episode_count != episode_counts["agent_invalid"]
            or scientific_episode_current != episode_currents["agent_invalid"]
            or scientific_episode_maximum != episode_maximums["agent_invalid"]
        ):
            raise ValueError("scientific MCP recovery episode counters disagree with taxonomy")
        source = "validated_typed_taxonomy_with_recovery_episodes"
    return {
        "source": source,
        "scientific_count": scientific_count,
        "scientific_current": scientific_current,
        "scientific_maximum": scientific_maximum,
        "scientific_episode_count": scientific_episode_count,
        "scientific_episode_current": scientific_episode_current,
        "scientific_episode_maximum": scientific_episode_maximum,
        "transport_count": counts["transport_ipc_os"],
        "transport_maximum": maximums["transport_ipc_os"],
        "provider_network_count": counts["provider_network"],
        "unclassified_count": counts["unclassified"],
    }


_FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["experiment_complete", "budget_exhausted", "stopped"],
        },
        "summary": {"type": "string", "maxLength": 2000},
    },
    "required": ["status", "summary"],
    "additionalProperties": False,
}

_CAMPAIGN_FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["campaign_complete", "budget_exhausted", "stopped"],
        },
        "summary": {"type": "string", "maxLength": 3000},
    },
    "required": ["status", "summary"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are the sole operation-level experimental agent in ChemWorld.
Complete one experiment by interacting with the environment only through the structured tools on
the required chemworld_lab MCP server. Call material_information once at the start. Submit every
physical operation with step, using the current expected_step and a complete action object. Use
the returned public outcome and state before selecting another operation. status exposes the
latest bounded state. history pages through the complete public operation stream: use offset=0
for the first event and next_offset to continue; without offset it returns recent events.
The response reports cache truncation and the persistent source path. inspect_artifact
retrieves a bounded public characterization fragment only when useful.

The current working directory is agent/ and persists within this benchmark cell. You may create
any concise notes or data files here, but doing so is optional. The MCP server is host-owned and
outside the writable workspace; it exposes only bounded public evidence and operation outcomes.
The parent files ../.ipc/, ../public/, and ../reference/ are host-owned and outside your writable
root. The environment, resource limits, validation, and authoritative trajectory remain
host-owned. Do not
fabricate a tool result, repair an invalid action silently, or ask the host to terminate or assay
on your behalf. When the tool reports experiment_ended or budget_exhausted, stop calling lab
operations and return the requested final JSON. When the outcome is a discarded batch,
report that explicitly. Do not provide private chain-of-thought.

Campaign lifecycle completion is part of the task. When more than one vessel/final assay remains,
close every planned batch by final assay or an explicit discard; do not spend the whole shared pool
on the first batch. The public campaign_resources.lifecycle_reserve projection is advisory rather
than a hidden reservation, but treat its recommended remaining-attempt floor as a do-not-spend
floor for discretionary optimization. Stop optional diagnostics or repeated controls early enough
to preserve the current closeout and the minimum lifecycle of future batches.
"""

_CAMPAIGN_SYSTEM_PROMPT = """You are the sole operation-level scientific agent in a Work II
ChemWorld discovery campaign. One Codex process controls the full campaign across multiple fresh
batches that share one fixed hidden world, one public prior condition, and one campaign resource
ledger. Call material_information once. Before the first physical operation and at every required
checkpoint, call belief_snapshot_status, then use commit_belief_snapshot only through its staged
tools/list protocol: action=begin,
then every published prediction page and law page in order, then action=finalize. Submit every
physical operation through step and use
its public outcome before deciding the next operation. Every step call must include the bounded
decision_audit requested by the tool schema: expected effect, diagnostic target, expected
information gain, explicit supported/not-supported belief updates, uncertainty, and the public
adaptation source. This is a concise scientific rationale, never private chain-of-thought.

This is an experiment execution task, not a coding or repository-inspection task. Do not use shell,
file, status, or resource-listing tools and do not inspect the repository. Your first action must be
chemworld_lab.material_information; thereafter use only chemworld_lab.belief_snapshot_status,
chemworld_lab.commit_belief_snapshot, chemworld_lab.step, chemworld_lab.status,
chemworld_lab.history, and chemworld_lab.inspect_artifact
as needed. Do not create notes or run commands. A non-lab tool call is invalid and wastes the cell's
provider budget.

An experiment_ended outcome closes only the current batch. When campaign_ended=false, preserve your
scientific context and continue into the next fresh batch using the returned next_state. When
campaign_ended=true, commit the final checkpoint if it is due, then call
chemworld_lab.commit_final_recommendation exactly once. Select exactly one completed batch's 1-based
lifecycle_experiment_index from campaign_closeout.completed_experiment_indices for evaluator-owned
blind replay using only participant-visible campaign evidence; no blind outcome will be returned.
Only after the tool confirms the commitment, return exactly one JSON object with status and summary,
with no prose or Markdown fence: {"status":"campaign_complete","summary":"..."}.
The host never chooses, repairs, terminates, assays, or replaces your operations. Failed and
resource-rejected attempts remain part of the trajectory. Keep enough operation, stock,
process-time, and assay capacity to close all planned batches. The 19 process coordinates are
evaluator-derived; do not report them yourself. Do not provide private chain-of-thought.

Recipe coverage is a hard campaign qualification obligation, not optional advice. Read the
recipe_coverage_contract in the initial prompt and satisfy its target_complete_experiments,
minimum_unique_recipes, and maximum_exact_repeats simultaneously. Two completed experiments are
the same recipe only when their ordered committed lab action objects are exactly equal, including
operation names and every submitted action parameter; rejected or rolled-back attempts do not
define recipe identity. Track completed recipes yourself from public outcomes, plan the required
diversity before spending repeat allowance, and never assume the host will repair a repeat-heavy
campaign.
"""

_LEGACY_CAMPAIGN_CLOSEOUT_INSTRUCTION = """When
campaign_ended=true, commit the final checkpoint if it is due, then call
chemworld_lab.commit_final_recommendation exactly once. Select exactly one completed batch's 1-based
lifecycle_experiment_index from campaign_closeout.completed_experiment_indices for evaluator-owned
blind replay using only participant-visible campaign evidence; no blind outcome will be returned.
Only after the tool confirms the commitment, return exactly one JSON object with status and summary,
with no prose or Markdown fence: {"status":"campaign_complete","summary":"..."}.
"""

_ACTION_READOUT_CAMPAIGN_CLOSEOUT_INSTRUCTION = """When
campaign_ended=true, commit the final checkpoint if it is due. The terminal held-out action packet
is intentionally unavailable before every required checkpoint is committed. After the final
checkpoint, call chemworld_lab.terminal_action_readout to reveal the fixed unseen candidate
operations without outcomes. Predict every required metric for every candidate, rank all
candidates, and commit the top-ranked candidate with chemworld_lab.commit_final_recommendation.
Do not revise or replace the already committed final belief checkpoint after seeing candidates.
Only after the tool confirms the commitment, return exactly one JSON object with status and summary,
with no prose or Markdown fence: {"status":"campaign_complete","summary":"..."}.
"""


_RANKING_ONLY_ACTION_READOUT_CAMPAIGN_CLOSEOUT_INSTRUCTION = """When
campaign_ended=true, commit the final checkpoint if it is due. The terminal held-out action packet
is intentionally unavailable before every required checkpoint is committed. After the final
checkpoint, call chemworld_lab.terminal_action_readout to reveal the fixed unseen candidate
operations without outcomes. Do not submit per-candidate numeric outcome predictions. Rank all
candidates exactly once, and commit the top-ranked candidate with
chemworld_lab.commit_final_recommendation. Do not revise or replace the already committed final
belief checkpoint after seeing candidates. Only after the tool confirms the commitment, return
exactly one JSON object with status and summary, with no prose or Markdown fence:
{"status":"campaign_complete","summary":"..."}.
"""


def _campaign_system_prompt(
    *, terminal_action_readout: bool, terminal_prediction_mode: str = "full_metrics"
) -> str:
    if not terminal_action_readout:
        return _CAMPAIGN_SYSTEM_PROMPT
    if terminal_prediction_mode not in {"full_metrics", "ranking_only"}:
        raise ValueError("unsupported terminal action prediction mode")
    if _LEGACY_CAMPAIGN_CLOSEOUT_INSTRUCTION not in _CAMPAIGN_SYSTEM_PROMPT:
        raise RuntimeError("campaign closeout instruction drifted")
    return _CAMPAIGN_SYSTEM_PROMPT.replace(
        _LEGACY_CAMPAIGN_CLOSEOUT_INSTRUCTION,
        (
            _RANKING_ONLY_ACTION_READOUT_CAMPAIGN_CLOSEOUT_INSTRUCTION
            if terminal_prediction_mode == "ranking_only"
            else _ACTION_READOUT_CAMPAIGN_CLOSEOUT_INSTRUCTION
        ),
    )


class InteractiveCodexExperimentError(RuntimeError):
    """The Codex process failed before producing an executable operation."""


class ProcessLike(Protocol):
    stdin: IO[str] | None
    stdout: IO[str] | None
    stderr: IO[str] | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[[Sequence[str], str, Path], ProcessLike]
ProviderAuthMode = Literal["none", "env_key", "experimental_bearer_token"]


def _default_process_factory(
    command: Sequence[str],
    prompt: str,
    cwd: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> ProcessLike:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        text=True,
        env=(dict(environment) if environment is not None else None),
        **kwargs,
    )
    if process.stdin is None:
        process.kill()
        raise OSError("Codex subprocess stdin was not created")
    process.stdin.write(prompt)
    process.stdin.close()
    return process


class _CodexEventMonitor:
    """Drain JSONL concurrently and retain no reasoning or tool-result bodies."""

    def __init__(self, process: ProcessLike) -> None:
        self.process = process
        self._lock = threading.RLock()
        self._tool_events: list[dict[str, Any]] = []
        self._event_counts: dict[str, int] = {}
        self._usage = _empty_usage()
        self._usage_observed = False
        self._usage_observation_event_type: str | None = None
        self._thread_id: str | None = None
        self._final_message: str | None = None
        self._stderr_sha256 = hashlib.sha256()
        self._stderr_byte_count = 0
        self._provider_errors: list[dict[str, Any]] = []
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="chemworld-codex-jsonl",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="chemworld-codex-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def alive(self) -> bool:
        return self.process.poll() is None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "thread_id": self._thread_id,
                "usage": dict(self._usage),
                "usage_observed": self._usage_observed,
                "usage_observation_event_type": self._usage_observation_event_type,
                "event_counts": dict(self._event_counts),
                "tool_events": deepcopy(self._tool_events),
                "stderr_byte_count": self._stderr_byte_count,
                "stderr_sha256": self._stderr_sha256.hexdigest(),
                "provider_errors": deepcopy(self._provider_errors),
            }

    def wait(self, timeout_s: float) -> dict[str, Any]:
        try:
            return_code = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return_code = self.process.wait(timeout=5.0)
            status = "timeout"
        else:
            status = "completed" if return_code == 0 else "failed"
        self._stdout_thread.join(timeout=5.0)
        self._stderr_thread.join(timeout=5.0)
        snapshot = self.snapshot()
        final_payload: dict[str, Any] | None = None
        final_payload_encoding: str | None = None
        with self._lock:
            message = self._final_message
        if message is not None:
            final_payload, final_payload_encoding = _parse_final_payload(message)
        return {
            **snapshot,
            "status": status,
            "return_code": int(return_code),
            "final_payload": final_payload,
            "final_payload_encoding": final_payload_encoding,
        }

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self._stdout_thread.join(timeout=2.0)
        self._stderr_thread.join(timeout=2.0)

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            self._consume_event(event)

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        for chunk in iter(lambda: stream.read(4096), ""):
            data = chunk.encode("utf-8", errors="replace")
            with self._lock:
                self._stderr_byte_count += len(data)
                self._stderr_sha256.update(data)

    def _consume_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", "unknown"))
        with self._lock:
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
            if event_type == "error":
                self._record_provider_error(event.get("message"))
            elif event_type == "turn.failed":
                error = event.get("error")
                if isinstance(error, Mapping):
                    self._record_provider_error(error.get("message"))
            if event_type == "thread.started":
                value = event.get("thread_id")
                if isinstance(value, str):
                    self._thread_id = value
            if isinstance(event.get("usage"), Mapping):
                self._usage = _normalize_usage(event["usage"])
                self._usage_observed = True
                self._usage_observation_event_type = event_type
            if event_type != "item.completed" or not isinstance(event.get("item"), Mapping):
                return
            item = event["item"]
            item_type = item.get("type")
            if item_type == "agent_message" and isinstance(item.get("text"), str):
                self._final_message = str(item["text"])
            elif item_type == "command_execution":
                self._tool_events.append(_sanitize_command_event(item))
            elif item_type == "file_change":
                self._tool_events.append(_sanitize_file_change_event(item))
            elif isinstance(item_type, str) and (
                "mcp" in item_type or item_type == "dynamic_tool_call"
            ):
                self._tool_events.append(_sanitize_mcp_tool_event(item))
            elif item_type == "reasoning":
                self._event_counts["reasoning_body_discarded"] = (
                    self._event_counts.get("reasoning_body_discarded", 0) + 1
                )

    def _record_provider_error(self, message: Any) -> None:
        """Retain only bounded metadata for provider errors, never their body."""

        if not isinstance(message, str):
            return
        encoded = message.encode("utf-8", errors="replace")
        entry = {
            "byte_count": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "category": _provider_error_category(message),
            "http_status_codes": sorted(
                {
                    int(value)
                    for value in re.findall(
                        r"(?:status|http(?:\s+status)?|code)\D{0,8}([1-5]\d{2})",
                        message,
                        flags=re.IGNORECASE,
                    )
                }
            ),
        }
        if entry not in self._provider_errors:
            self._provider_errors.append(entry)


class InteractiveCodexExperimentAgent(BaseAgent):
    """Keep one native Codex context alive for every complete experiment."""

    name = "interactive_codex_experiment"

    def __init__(
        self,
        *,
        workspace: str | Path,
        role_id: str,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "medium",
        model_provider: str = HTTPS_PROVIDER_ID,
        model_provider_name: str = "OpenAI",
        model_provider_base_url: str | None = None,
        model_provider_env_key: str | None = None,
        model_provider_wire_api: str = "responses",
        model_provider_auth_mode: ProviderAuthMode = "env_key",
        model_provider_api_key_file: str | Path | None = None,
        model_provider_model_catalog_json: str | Path | None = None,
        model_provider_preferred_auth_method: str | None = None,
        model_provider_forced_login_method: str | None = None,
        codex_executable: str | None = None,
        process_factory: ProcessFactory | None = None,
        request_timeout_s: float = 600.0,
        finalization_timeout_s: float = DEFAULT_FINALIZATION_TIMEOUT_S,
        session_wall_time_limit_s: float | None = None,
        max_recovered_mcp_tool_failures: int | None = None,
        max_consecutive_mcp_tool_failures: int | None = None,
        max_provider_error_events: int | None = None,
        session_progress_callback: Callable[[dict[str, Any]], None] | None = None,
        session_progress_interval_s: float = 30.0,
        pre_action_restart_limit: int = 1,
        accepted_turn_continuation_limit: int = 0,
        provider_process_attempt_limit: int | None = None,
        max_initial_prompt_bytes: int = 65_536,
        max_tool_output_bytes: int = 32_768,
        history_event_limit: int = 64,
        history_byte_limit: int = 131_072,
        session_scope: str = "experiment",
        belief_checkpoint_contract: Mapping[str, Any] | None = None,
        initial_world_model: Mapping[str, Any] | None = None,
        terminal_action_readout_contract: Mapping[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError(f"unsupported Codex model: {model}")
        if model_provider == HTTPS_PROVIDER_ID and model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported Codex model: {model}")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported Codex reasoning effort")
        if not model_provider or not model_provider_name:
            raise ValueError("model provider id and name must be non-empty")
        if model_provider_wire_api != "responses":
            raise ValueError("interactive Codex runner currently requires wire_api=responses")
        if model_provider_auth_mode not in {"none", "env_key", "experimental_bearer_token"}:
            raise ValueError("unsupported model provider auth mode")
        if (
            model_provider_auth_mode == "experimental_bearer_token"
            and not model_provider_api_key_file
        ):
            raise ValueError("experimental_bearer_token auth requires model_provider_api_key_file")
        if request_timeout_s <= 0 or finalization_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if session_wall_time_limit_s is not None and session_wall_time_limit_s <= 0:
            raise ValueError("session_wall_time_limit_s must be positive")
        for name, value in (
            ("max_recovered_mcp_tool_failures", max_recovered_mcp_tool_failures),
            ("max_consecutive_mcp_tool_failures", max_consecutive_mcp_tool_failures),
            ("max_provider_error_events", max_provider_error_events),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if pre_action_restart_limit < 0:
            raise ValueError("pre_action_restart_limit must be non-negative")
        if accepted_turn_continuation_limit < 0:
            raise ValueError("accepted_turn_continuation_limit must be non-negative")
        if provider_process_attempt_limit is not None and provider_process_attempt_limit < 1:
            raise ValueError("provider_process_attempt_limit must be positive")
        if session_progress_interval_s <= 0:
            raise ValueError("session_progress_interval_s must be positive")
        if max_initial_prompt_bytes < 4_096:
            raise ValueError("max_initial_prompt_bytes must be at least 4096")
        if session_scope not in {"experiment", "campaign"}:
            raise ValueError("session_scope must be experiment or campaign")
        if session_scope == "campaign" and not isinstance(belief_checkpoint_contract, Mapping):
            raise ValueError("campaign scope requires a belief checkpoint contract")
        if session_scope == "experiment" and belief_checkpoint_contract is not None:
            raise ValueError("belief checkpoint contract requires campaign scope")
        if session_scope == "experiment" and terminal_action_readout_contract is not None:
            raise ValueError("terminal action readout contract requires campaign scope")
        self._process_factory: ProcessFactory
        self._uses_default_process_factory = process_factory is None
        if process_factory is None:
            executable = codex_executable or shutil.which("codex")
            if not executable:
                raise InteractiveCodexExperimentError(
                    "Codex CLI is not installed or not available on PATH"
                )
            self.codex_executable = str(executable)
            self._process_factory = lambda command, prompt, cwd: _default_process_factory(
                command,
                prompt,
                cwd,
                environment=self._session_process_environment,
            )
        else:
            self.codex_executable = codex_executable or "codex"
            self._process_factory = process_factory
        self.role_id = role_id
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.model_provider = str(model_provider)
        self.model_provider_name = str(model_provider_name)
        self.model_provider_base_url = (
            None if model_provider_base_url is None else str(model_provider_base_url)
        )
        self.model_provider_env_key = (
            None if model_provider_env_key is None else str(model_provider_env_key)
        )
        self.model_provider_wire_api = str(model_provider_wire_api)
        self.model_provider_auth_mode: ProviderAuthMode = model_provider_auth_mode
        self.model_provider_api_key_file = (
            None if model_provider_api_key_file is None else Path(model_provider_api_key_file)
        )
        self.model_provider_model_catalog_json = (
            None
            if model_provider_model_catalog_json is None
            else Path(model_provider_model_catalog_json)
        )
        self.model_provider_preferred_auth_method = (
            None
            if model_provider_preferred_auth_method is None
            else str(model_provider_preferred_auth_method)
        )
        self.model_provider_forced_login_method = (
            None
            if model_provider_forced_login_method is None
            else str(model_provider_forced_login_method)
        )
        self._session_process_environment: dict[str, str] | None = None
        self._use_isolated_codex_home = False
        self.request_timeout_s = float(request_timeout_s)
        self.finalization_timeout_s = float(finalization_timeout_s)
        self.session_wall_time_limit_s = (
            None if session_wall_time_limit_s is None else float(session_wall_time_limit_s)
        )
        self.max_recovered_mcp_tool_failures = (
            None
            if max_recovered_mcp_tool_failures is None
            else int(max_recovered_mcp_tool_failures)
        )
        self.max_consecutive_mcp_tool_failures = (
            None
            if max_consecutive_mcp_tool_failures is None
            else int(max_consecutive_mcp_tool_failures)
        )
        self.max_provider_error_events = (
            None if max_provider_error_events is None else int(max_provider_error_events)
        )
        self.session_progress_callback = session_progress_callback
        self.session_progress_interval_s = float(session_progress_interval_s)
        self.pre_action_restart_limit = int(pre_action_restart_limit)
        self.accepted_turn_continuation_limit = int(accepted_turn_continuation_limit)
        self.provider_process_attempt_limit = (
            None if provider_process_attempt_limit is None else int(provider_process_attempt_limit)
        )
        self.max_initial_prompt_bytes = int(max_initial_prompt_bytes)
        self.session_scope = session_scope
        self.belief_checkpoint_contract = (
            deepcopy(dict(belief_checkpoint_contract))
            if belief_checkpoint_contract is not None
            else None
        )
        self.initial_world_model = (
            deepcopy(dict(initial_world_model)) if initial_world_model is not None else None
        )
        self.terminal_action_readout_contract = (
            deepcopy(dict(terminal_action_readout_contract))
            if terminal_action_readout_contract is not None
            else None
        )
        self.workspace = ExperimentCodexWorkspace(
            workspace,
            max_tool_output_bytes=max_tool_output_bytes,
            history_event_limit=history_event_limit,
            history_byte_limit=history_byte_limit,
        )
        self._mcp_server_path = Path(__file__).with_name("experiment_codex_mcp.py").resolve()
        self._source_project_root = Path(__file__).resolve().parents[3]
        self._mcp_server_sha256 = _file_sha256(self._mcp_server_path)

        # Keep post-startup failure accounting total.  A provider/process failure can occur
        # before BaseAgent.reset() is reached; the runner still needs to emit a retained,
        # machine-readable failure record rather than raising while collecting receipts.
        self._session = None
        self._sessions_started = 0
        self._sessions_completed = 0
        self._all_session_usage_complete = False
        self._all_session_usage_observed = False
        self._unattributed_pre_action_process_attempt_count = 0
        self._pre_action_restarts = 0
        self._accepted_turn_continuations = 0
        self._provider_process_attempt_count = 0
        self._logical_codex_turn_count = 0
        self._session_receipts = []
        self._cumulative_usage = _empty_usage()
        self._completed_session_elapsed_s = 0.0
        self._recovered_mcp_tool_failure_count = 0
        self._maximum_consecutive_mcp_tool_failure_count = 0
        self._maximum_consecutive_mcp_tool_failure_episode_count = 0
        self._mcp_tool_failure_counts_by_category = _empty_mcp_tool_failure_counts()
        self._maximum_consecutive_mcp_tool_failure_counts_by_category = (
            _empty_mcp_tool_failure_counts()
        )
        self._mcp_tool_failure_episode_counts_by_category = _empty_mcp_tool_failure_counts()
        self._maximum_consecutive_mcp_tool_failure_episode_counts_by_category = (
            _empty_mcp_tool_failure_counts()
        )
        self._provider_error_event_count = 0

    def _verify_experiment_tool(self) -> None:
        """Fail closed if either active MCP source or legacy fallback changes."""

        if _file_sha256(self._mcp_server_path) != self._mcp_server_sha256:
            raise ExperimentCodexIPCError("host-owned MCP server changed during the run")
        self.workspace.verify_lab_tool()

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.workspace.initialize_fresh()
        material_manifest = self.workspace.publish_material_information(
            _material_information_payload(
                task_info,
                initial_world_model=self.initial_world_model,
            )
        )
        self._material_manifest = material_manifest
        self._task_contract = _public_task_contract(task_info)
        self._task_contract_manifest = self.workspace.publish_task_contract(self._task_contract)
        self._belief_checkpoint_manifest = (
            self.workspace.publish_belief_checkpoint_contract(self.belief_checkpoint_contract or {})
            if self.session_scope == "campaign"
            else None
        )
        self._terminal_action_readout_manifest = (
            self.workspace.publish_terminal_action_readout_contract(
                self.terminal_action_readout_contract
            )
            if self.terminal_action_readout_contract is not None
            else None
        )
        self._session: dict[str, Any] | None = None
        self._sessions_started = 0
        self._sessions_completed = 0
        self._all_session_usage_complete = True
        self._all_session_usage_observed = True
        self._unattributed_pre_action_process_attempt_count = 0
        self._pre_action_restarts = 0
        self._accepted_turn_continuations = 0
        self._provider_process_attempt_count = 0
        self._logical_codex_turn_count = 0
        self._handled_request_ids: set[str] = set()
        self._pending_request: IPCRequest | None = None
        self._pending_outcome: dict[str, Any] | None = None
        self._last_decision: dict[str, Any] | None = None
        self._last_context_remaining = 0
        self._global_runner_action_count = 0
        self._experiment_action_count = 0
        self._session_action_count = 0
        self._campaign_closed_batch_count = 0
        self._campaign_completed_experiment_count = 0
        self._belief_snapshots: list[dict[str, Any]] = []
        self._cumulative_usage = _empty_usage()
        self._completed_session_elapsed_s = 0.0
        self._recovered_mcp_tool_failure_count = 0
        self._maximum_consecutive_mcp_tool_failure_count = 0
        self._maximum_consecutive_mcp_tool_failure_episode_count = 0
        self._mcp_tool_failure_counts_by_category = _empty_mcp_tool_failure_counts()
        self._maximum_consecutive_mcp_tool_failure_counts_by_category = (
            _empty_mcp_tool_failure_counts()
        )
        self._mcp_tool_failure_episode_counts_by_category = _empty_mcp_tool_failure_counts()
        self._maximum_consecutive_mcp_tool_failure_episode_counts_by_category = (
            _empty_mcp_tool_failure_counts()
        )
        self._provider_error_event_count = 0
        self._session_receipts: list[dict[str, Any]] = []
        self._completed_tool_events: list[dict[str, Any]] = []
        self._last_agent_snapshot = self.workspace.snapshot_agent_files()
        self._session_temp_directories: list[tempfile.TemporaryDirectory[str]] = []

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        raise RuntimeError(
            "InteractiveCodexExperimentAgent requires the official public-view runner"
        )

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        self._verify_experiment_tool()
        artifact = self._publish_latest_artifact(public_view)
        current_packet = _bounded_current_packet(
            context,
            public_view,
            artifact=artifact,
            recipe_coverage_contract=_participant_recipe_coverage_contract(
                self._task_contract
            ),
        )
        current_manifest = self.workspace.publish_current(current_packet)
        if self._session is None:
            self._start_session(current_packet)
        elif self._pending_request is not None and self._pending_outcome is not None:
            response = {
                **self._pending_outcome,
                "next_state": current_packet,
            }
            self.workspace.write_response(
                session_id=self._pending_request.session_id,
                request_id=self._pending_request.request_id,
                response=response,
            )
            self._pending_request = None
            self._pending_outcome = None

        self._last_context_remaining = context.remaining_operations
        request = self._wait_for_next_request(current_packet)
        try:
            self._verify_experiment_tool()
            self.workspace.verify_file(
                relative_path=str(self._material_manifest["relative_path"]),
                expected_sha256=str(self._material_manifest["sha256"]),
            )
            self.workspace.verify_file(
                relative_path=str(self._task_contract_manifest["relative_path"]),
                expected_sha256=str(self._task_contract_manifest["sha256"]),
            )
            if self._belief_checkpoint_manifest is not None:
                self.workspace.verify_file(
                    relative_path=str(self._belief_checkpoint_manifest["relative_path"]),
                    expected_sha256=str(self._belief_checkpoint_manifest["sha256"]),
                )
            self.workspace.verify_file(
                relative_path="public/current.json",
                expected_sha256=str(current_manifest["sha256"]),
            )
        except ExperimentCodexIPCError as error:
            self.close()
            raise InteractiveCodexExperimentError(
                "Codex modified a host-owned bridge/reference file; "
                "no environment action was emitted"
            ) from error
        self._handled_request_ids.add(request.request_id)
        if self._session is None:
            raise InteractiveCodexExperimentError("Codex session ended before action acceptance")
        self._session["accepted_action_count"] = int(self._session["accepted_action_count"]) + 1
        self._pending_request = request
        self._experiment_action_count += 1
        self._session_action_count += 1
        self._global_runner_action_count += 1
        self._last_decision = {
            "schema_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
            "status": "codex_lab_tool_action",
            "action": to_builtin(request.action),
            "session_id": request.session_id,
            "request_id": request.request_id,
            "expected_step": request.expected_step,
            "action_payload_sha256": request.payload_sha256,
            "current_public_state": current_manifest,
            "authoritative_ledger_in_workspace": False,
            "decision_audit_status": (
                "provided" if request.decision_audit is not None else "not_provided"
            ),
            "decision_audit": deepcopy(request.decision_audit),
        }
        return dict(request.action)

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        if self._pending_request is None:
            raise InteractiveCodexExperimentError(
                "runner returned an outcome without a pending Codex request"
            )
        if to_builtin(action) != self._pending_request.action:
            raise InteractiveCodexExperimentError(
                "runner action differs from the accepted Codex request"
            )
        self._verify_experiment_tool()
        event_id = f"runner-operation-{self._global_runner_action_count:04d}"
        artifact = self._publish_outcome_artifact(
            event_id=event_id,
            action=action,
            info=info,
        )
        outcome = _compact_outcome(
            event_id=event_id,
            action=action,
            observation=observation,
            reward=reward,
            info=info,
            artifact=artifact,
        )
        self.workspace.update_expected_step(
            session_id=self._pending_request.session_id,
            expected_step=self._session_action_count + 1,
        )
        successful_final_assay = bool(
            action.get("operation") == "measure"
            and action.get("instrument") == "final_assay"
            and info.get("transaction_status") == "committed"
            and info.get("leaderboard_score") is not None
        )
        ended = bool(info.get("experiment_ended", False) or successful_final_assay)
        batch_discarded = bool(info.get("batch_discarded", False))
        experiment_completed = bool(info.get("experiment_completed", not batch_discarded))
        budget_exhausted = self._last_context_remaining <= 1 and not ended
        closed_count = self._campaign_closed_batch_count
        completed_count = self._campaign_completed_experiment_count
        lifecycle_experiment_index: int | None = None
        if self.session_scope == "campaign" and ended:
            summaries = info.get("experiment_summaries")
            if not isinstance(summaries, list) or len(summaries) != closed_count + 1:
                raise InteractiveCodexExperimentError(
                    "campaign terminal summary count did not advance by exactly one"
                )
            closed_count = len(summaries)
            terminal_summary = info.get("last_terminal_summary")
            if not isinstance(terminal_summary, Mapping):
                terminal_summary = summaries[-1] if summaries else None
            raw_lifecycle_index = (
                terminal_summary.get("experiment_index")
                if isinstance(terminal_summary, Mapping)
                else None
            )
            lifecycle_experiment_index = (
                raw_lifecycle_index + 1
                if isinstance(raw_lifecycle_index, int)
                and not isinstance(raw_lifecycle_index, bool)
                and raw_lifecycle_index >= 0
                else closed_count
            )
            if lifecycle_experiment_index != closed_count:
                raise InteractiveCodexExperimentError(
                    "campaign lifecycle experiment index does not match closed batch count"
                )
            if experiment_completed:
                completed_count += 1
        evidence_id = (
            f"experiment-{completed_count}-final-assay"
            if ended and experiment_completed and completed_count > 0
            else None
        )
        campaign_ended = bool(
            (ended or budget_exhausted)
            if self.session_scope == "experiment"
            else info.get("campaign_terminal", False) or budget_exhausted
        )
        if self.session_scope == "campaign":
            self.workspace.publish_campaign_progress(
                session_id=self._pending_request.session_id,
                closed_batch_count=closed_count,
                completed_experiment_count=completed_count,
                completed_experiment_index=(
                    lifecycle_experiment_index if ended and experiment_completed else None
                ),
                observed_evidence_id=evidence_id,
                campaign_ended=campaign_ended,
            )
            self._campaign_closed_batch_count = closed_count
            self._campaign_completed_experiment_count = completed_count
            if self._session is not None:
                self._session["campaign_terminal"] = campaign_ended
        next_experiment_index = info.get("next_experiment_index")
        if (
            info.get("next_experiment_ready") is True
            and isinstance(next_experiment_index, int)
            and not isinstance(next_experiment_index, bool)
        ):
            next_experiment_index += 1
        else:
            next_experiment_index = None
        self._pending_outcome = {
            "ok": True,
            "schema_version": EXPERIMENT_CODEX_IPC_VERSION,
            **outcome,
            "experiment_ended": ended,
            "experiment_completed": experiment_completed,
            "batch_discarded": batch_discarded,
            "budget_exhausted": budget_exhausted,
            "campaign_ended": campaign_ended,
            "campaign_terminal_reason": info.get("campaign_terminal_reason"),
            "closed_batch_count": closed_count,
            "completed_experiment_count": completed_count,
            "lifecycle_experiment_index": lifecycle_experiment_index,
            "batch_id": (
                f"batch-{lifecycle_experiment_index:04d}"
                if lifecycle_experiment_index is not None
                else None
            ),
            "completed_ordinal": (
                completed_count if ended and experiment_completed else None
            ),
            "next_experiment_ready": info.get("next_experiment_ready"),
            "next_experiment_index": next_experiment_index,
            "experiment_index_base": 1,
            "evidence_id": evidence_id,
        }
        if "batch_budget_exhausted" in info:
            self._pending_outcome["batch_budget_exhausted"] = info["batch_budget_exhausted"]
        self.workspace.append_public_history(
            {
                "schema_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
                **self._pending_outcome,
            }
        )
        if self._last_decision is not None:
            self._last_decision["outcome"] = deepcopy(self._pending_outcome)

        before = self._last_agent_snapshot
        after = self.workspace.snapshot_agent_files()
        memory_diff = diff_agent_snapshots(before, after)
        self._last_agent_snapshot = after
        if self._last_decision is not None:
            self._last_decision["agent_workspace_changes"] = memory_diff

        if ended:
            self._experiment_action_count = 0

        if campaign_ended:
            self.workspace.write_response(
                session_id=self._pending_request.session_id,
                request_id=self._pending_request.request_id,
                response={**self._pending_outcome, "next_state": None},
            )
            self._finalize_session(
                terminal_reason=(
                    "campaign_complete"
                    if self.session_scope == "campaign" and ended
                    else "batch_discarded"
                    if batch_discarded
                    else "experiment_complete"
                    if ended
                    else "budget_exhausted"
                )
            )
            self._pending_request = None
            self._pending_outcome = None

    def decision_audit(self) -> dict[str, Any] | None:
        """Return the participant-authored bounded rationale for the accepted operation."""

        if self._last_decision is None:
            return None
        value = self._last_decision.get("decision_audit")
        return deepcopy(value) if isinstance(value, dict) else None

    def agent_trace(self) -> list[dict[str, Any]]:
        if self._last_decision is None:
            return []
        trace = deepcopy(self._last_decision)
        if self._session is not None:
            monitor = self._session["monitor"]
            if isinstance(monitor, _CodexEventMonitor):
                snapshot = monitor.snapshot()
                trace["live_session_audit"] = {
                    "event_counts": snapshot["event_counts"],
                    "tool_event_count": len(snapshot["tool_events"]),
                    "usage_finalized": False,
                }
                trace["artifact_access_count"] = len(
                    self.workspace.artifact_access_audit(str(self._session["session_id"]))
                )
        return [to_builtin(trace)]

    def interaction_capabilities(self) -> InteractionCapabilities:
        return InteractionCapabilities(
            decision_scope="operation",
            consumes_intermediate_observations=True,
            consumes_spectra=True,
            adapts_within_experiment=True,
            adapts_across_experiments=True,
            emits_structured_decision_audit=self.session_scope == "campaign",
        )

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "role_id": self.role_id,
                "requires_online_model": True,
                "provider": self.model_provider_name,
                "provider_id": self.model_provider,
                "provider_base_url": self.model_provider_base_url,
                "provider_env_key": self.model_provider_env_key,
                "provider_model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "interaction_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
                "session_scope": self.session_scope,
                "one_codex_exec_per_complete_experiment": self.session_scope == "experiment",
                "one_codex_thread_per_campaign_cell": self.session_scope == "campaign",
                "bounded_same_thread_continuation": self.session_scope == "campaign",
                "accepted_turn_continuation_limit": self.accepted_turn_continuation_limit,
                "codex_exec_ephemeral": False,
                "experiment_tool_transport": "host_owned_stdio_mcp",
                "mcp_server": {
                    "name": "chemworld_lab",
                    "version": MCP_SERVER_VERSION,
                    "required": True,
                    "enabled_tools": list(SUPPORTED_TOOLS),
                    "source_sha256": self._mcp_server_sha256,
                    "model_generated_shell_required": False,
                },
                "automatic_action_repair": False,
                "automatic_closeout": False,
                "authoritative_trajectory_in_workspace": False,
                "agent_workspace_optional": True,
                "forced_notebook": False,
                "material_information_reference": deepcopy(self._material_manifest),
                "initial_world_model_sha256": (
                    hashlib.sha256(
                        _canonical_json(self.initial_world_model).encode("utf-8")
                    ).hexdigest()
                    if self.initial_world_model is not None
                    else None
                ),
                "task_contract_reference": deepcopy(self._task_contract_manifest),
                "belief_checkpoint_contract_reference": deepcopy(self._belief_checkpoint_manifest),
                "terminal_action_readout_contract_reference": deepcopy(
                    self._terminal_action_readout_manifest
                ),
                "workspace": self.workspace.manifest(),
                "usage_accounting_granularity": (
                    "complete_codex_campaign_turn"
                    if self.session_scope == "campaign"
                    else "complete_codex_experiment_turn"
                ),
                "provider_session_count": self._sessions_started,
                "logical_codex_turn_count": self._logical_codex_turn_count,
                "backend_model_response_count": None,
                "backend_model_response_accounting_complete": False,
                "input_token_accounting_semantics": (
                    "cumulative_input_across_the_complete_codex_turn"
                ),
            }
        )
        return payload

    def method_resource_usage(self) -> dict[str, Any]:
        active = self._session is not None
        provider_usage_complete = bool(not active and self._all_session_usage_complete)
        provider_usage_observed = bool(not active and self._all_session_usage_observed)
        usage = dict(self._cumulative_usage)
        if self._session is not None:
            completed_turn_usage = self._session.get("completed_turn_usage")
            if isinstance(completed_turn_usage, Mapping):
                _merge_usage(usage, completed_turn_usage)
        operational = self._active_session_operational_audit()
        session_elapsed_s = self._completed_session_elapsed_s + float(
            operational["session_elapsed_s"]
        )
        recovered_mcp_tool_failure_count = self._recovered_mcp_tool_failure_count + int(
            operational["recovered_mcp_tool_failure_count"]
        )
        maximum_consecutive_mcp_tool_failure_count = max(
            self._maximum_consecutive_mcp_tool_failure_count,
            int(operational["maximum_consecutive_mcp_tool_failure_count"]),
        )
        active_taxonomy = operational["mcp_tool_failure_taxonomy"]
        counts_by_category = {
            category: self._mcp_tool_failure_counts_by_category[category]
            + int(active_taxonomy["counts_by_category"][category])
            for category in MCP_TOOL_FAILURE_CATEGORIES
        }
        maximum_by_category = {
            category: max(
                self._maximum_consecutive_mcp_tool_failure_counts_by_category[category],
                int(active_taxonomy["maximum_consecutive_counts_by_category"][category]),
            )
            for category in MCP_TOOL_FAILURE_CATEGORIES
        }
        active_episode_taxonomy = active_taxonomy["recovery_episode_taxonomy"]
        episode_counts_by_category = {
            category: self._mcp_tool_failure_episode_counts_by_category[category]
            + int(active_episode_taxonomy["counts_by_category"][category])
            for category in MCP_TOOL_FAILURE_CATEGORIES
        }
        maximum_episodes_by_category = {
            category: max(
                self._maximum_consecutive_mcp_tool_failure_episode_counts_by_category[category],
                int(active_episode_taxonomy["maximum_consecutive_counts_by_category"][category]),
            )
            for category in MCP_TOOL_FAILURE_CATEGORIES
        }
        maximum_episode_total = max(
            self._maximum_consecutive_mcp_tool_failure_episode_count,
            int(active_episode_taxonomy["maximum_consecutive_recovery_episode_count"]),
        )
        provider_error_event_count = self._provider_error_event_count + int(
            operational["provider_error_event_count"]
        )
        accepted_provider_session_count = sum(
            receipt.get("pre_action_retry_classification")
            == TERMINAL_ACCEPTED_PRE_ACTION_RETRY_CLASSIFICATION
            for receipt in self._session_receipts
        ) + int(
            active
            and self._session is not None
            and int(self._session.get("accepted_action_count", 0)) > 0
        )
        accepted_participant_model_call_count = (
            accepted_provider_session_count + self._accepted_turn_continuations
        )
        return {
            "schema_version": "chemworld-method-resource-usage-0.1",
            "accounting_complete": False,
            "provider_usage_pending": active,
            "in_flight_model_call_count": 1 if active else 0,
            "provider_usage_accounting_complete": provider_usage_complete,
            "provider_usage_observed": provider_usage_observed,
            "token_counts_observed": provider_usage_observed,
            "provider_call_accounting_complete": True,
            "provider_token_accounting_complete": provider_usage_complete,
            "provider_cache_accounting_complete": provider_usage_complete,
            "monetary_accounting_complete": False,
            "usage_source": (
                "active_codex_turn_usage_pending"
                if active
                else (
                    "codex_cli_completed_experiment_turns"
                    if provider_usage_complete
                    else (
                        "codex_cli_incomplete_interrupted_turns_with_partial_usage"
                        if provider_usage_observed
                        else "codex_cli_interrupted_before_usage_observation"
                    )
                )
            ),
            "model_call_count": accepted_participant_model_call_count,
            "provider_session_count": accepted_provider_session_count,
            "provider_process_attempt_count": self._provider_process_attempt_count,
            "accepted_provider_session_count": accepted_provider_session_count,
            "accepted_participant_model_call_count": (
                accepted_participant_model_call_count
            ),
            "unattributed_pre_action_process_attempt_count": (
                self._unattributed_pre_action_process_attempt_count
            ),
            "logical_codex_turn_count": self._logical_codex_turn_count,
            "accepted_turn_continuation_count": self._accepted_turn_continuations,
            "backend_model_response_count": None,
            "backend_model_response_accounting_complete": False,
            "input_token_count": usage["prompt_tokens"],
            "cached_input_token_count": usage["prompt_cache_hit_tokens"],
            "uncached_input_token_count": usage["prompt_cache_miss_tokens"],
            "input_cache_hit_ratio": (
                usage["prompt_cache_hit_tokens"] / usage["prompt_tokens"]
                if usage["prompt_tokens"]
                else 0.0
            ),
            "output_token_count": usage["completion_tokens"],
            "monetary_cost_usd": 0.0,
            "training_environment_step_count": 0,
            "cpu_time_s": 0.0,
            "gpu_time_s": 0.0,
            "active_session_elapsed_s": operational["session_elapsed_s"],
            "session_elapsed_s": round(session_elapsed_s, 3),
            "recovered_mcp_tool_failure_count": recovered_mcp_tool_failure_count,
            "maximum_consecutive_mcp_tool_failure_count": (
                maximum_consecutive_mcp_tool_failure_count
            ),
            "mcp_tool_failure_taxonomy": {
                "schema_version": MCP_TOOL_FAILURE_TAXONOMY_VERSION,
                "recovered_mcp_tool_failure_count": recovered_mcp_tool_failure_count,
                "current_consecutive_mcp_tool_failure_count": active_taxonomy[
                    "current_consecutive_mcp_tool_failure_count"
                ],
                "maximum_consecutive_mcp_tool_failure_count": (
                    maximum_consecutive_mcp_tool_failure_count
                ),
                "counts_by_category": counts_by_category,
                "current_consecutive_counts_by_category": active_taxonomy[
                    "current_consecutive_counts_by_category"
                ],
                "maximum_consecutive_counts_by_category": maximum_by_category,
                "recovery_episode_taxonomy": {
                    **active_episode_taxonomy,
                    "recovery_episode_count": sum(episode_counts_by_category.values()),
                    "maximum_consecutive_recovery_episode_count": maximum_episode_total,
                    "counts_by_category": episode_counts_by_category,
                    "maximum_consecutive_counts_by_category": (maximum_episodes_by_category),
                },
            },
            "scientific_compliance_mcp_tool_failure_count": counts_by_category["agent_invalid"],
            "current_consecutive_scientific_compliance_mcp_tool_failure_count": (
                active_taxonomy["current_consecutive_counts_by_category"]["agent_invalid"]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_count": (
                maximum_by_category["agent_invalid"]
            ),
            "scientific_compliance_mcp_tool_failure_episode_count": (
                episode_counts_by_category["agent_invalid"]
            ),
            "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                active_episode_taxonomy["current_consecutive_counts_by_category"]["agent_invalid"]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                maximum_episodes_by_category["agent_invalid"]
            ),
            "provider_error_event_count": provider_error_event_count,
            "session_operational_limits": operational["limits"],
            "model_provenance": {
                "provider": self.model_provider_name,
                "provider_id": self.model_provider,
                "provider_base_url": self.model_provider_base_url,
                "model_id": self.model,
                "model_snapshot_or_access_date": MODEL_ACCESS_DATE,
                "prompt_hash": _system_prompt_hash(),
                "request_parameters": {
                    "reasoning_effort": self.reasoning_effort,
                    "session_scope": self.session_scope,
                    "one_turn_per_experiment": self.session_scope == "experiment",
                    "one_thread_per_campaign_cell": self.session_scope == "campaign",
                    "accepted_turn_continuation_limit": self.accepted_turn_continuation_limit,
                    "provider_process_attempt_limit": self.provider_process_attempt_limit,
                    "workspace_tools": True,
                    "experiment_tool_transport": "host_owned_stdio_mcp",
                    "forced_notebook": False,
                },
                "tokenizer_or_provider_usage_source": "codex exec turn.completed",
                "input_token_accounting_semantics": (
                    "cumulative_input_across_the_complete_codex_turn; cache hits are "
                    "reused input tokens, not generated output"
                ),
                "pricing": {
                    "accounting_complete": False,
                    "model_source": MODEL_SOURCE,
                    "auth_source": AUTH_SOURCE,
                    "pricing_unavailable_reason": (
                        "ChatGPT subscription usage has no attributable per-run USD price"
                    ),
                },
                "private_reasoning_retained": False,
                "completed_session_count": self._sessions_completed,
            },
        }

    def provider_receipts(self) -> list[dict[str, Any]]:
        return deepcopy(self._session_receipts)

    def _active_session_remaining_wall_time_s(self) -> float | None:
        if self._session is None or self.session_wall_time_limit_s is None:
            return None
        started_at = self._session.get("started_at")
        if not isinstance(started_at, (int, float)):
            return 0.0
        return max(0.0, self.session_wall_time_limit_s - (perf_counter() - float(started_at)))

    def _active_session_operational_audit(
        self,
        *,
        monitor_snapshot: Mapping[str, Any] | None = None,
        mcp_tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._session is None:
            empty_taxonomy = _mcp_tool_failure_audit([])
            return {
                "session_elapsed_s": 0.0,
                "recovered_mcp_tool_failure_count": 0,
                "current_consecutive_mcp_tool_failure_count": 0,
                "maximum_consecutive_mcp_tool_failure_count": 0,
                "mcp_tool_failure_taxonomy": empty_taxonomy,
                "scientific_compliance_mcp_tool_failure_count": 0,
                "current_consecutive_scientific_compliance_mcp_tool_failure_count": 0,
                "maximum_consecutive_scientific_compliance_mcp_tool_failure_count": 0,
                "scientific_compliance_mcp_tool_failure_episode_count": 0,
                "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count": 0,
                "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count": 0,
                "provider_error_event_count": 0,
                "limits": {
                    "session_wall_time_limit_s": self.session_wall_time_limit_s,
                    "max_recovered_mcp_tool_failures": self.max_recovered_mcp_tool_failures,
                    "max_consecutive_mcp_tool_failures": (self.max_consecutive_mcp_tool_failures),
                    "max_provider_error_events": self.max_provider_error_events,
                    "mcp_tool_failure_limit_category": "agent_invalid_recovery_episode",
                },
            }
        snapshot = (
            dict(monitor_snapshot)
            if isinstance(monitor_snapshot, Mapping)
            else self._session["monitor"].snapshot()
        )
        calls = (
            mcp_tool_calls
            if mcp_tool_calls is not None
            else self.workspace.mcp_tool_call_audit(str(self._session["session_id"]))
        )
        started_at = self._session.get("started_at")
        elapsed = (
            max(0.0, perf_counter() - float(started_at))
            if isinstance(started_at, (int, float))
            else 0.0
        )
        taxonomy = _mcp_tool_failure_audit([item for item in calls if isinstance(item, Mapping)])
        agent_invalid_count = int(taxonomy["counts_by_category"]["agent_invalid"])
        current_agent_invalid = int(
            taxonomy["current_consecutive_counts_by_category"]["agent_invalid"]
        )
        maximum_agent_invalid = int(
            taxonomy["maximum_consecutive_counts_by_category"]["agent_invalid"]
        )
        episode_taxonomy = taxonomy["recovery_episode_taxonomy"]
        agent_invalid_episode_count = int(episode_taxonomy["counts_by_category"]["agent_invalid"])
        current_agent_invalid_episode = int(
            episode_taxonomy["current_consecutive_counts_by_category"]["agent_invalid"]
        )
        maximum_agent_invalid_episode = int(
            episode_taxonomy["maximum_consecutive_counts_by_category"]["agent_invalid"]
        )
        event_counts = snapshot.get("event_counts", {})
        provider_error_count = (
            int(event_counts.get("error", 0)) + int(event_counts.get("turn.failed", 0))
            if isinstance(event_counts, Mapping)
            else 0
        )
        if provider_error_count == 0:
            provider_errors = snapshot.get("provider_errors", [])
            provider_error_count = len(provider_errors) if isinstance(provider_errors, list) else 0
        return {
            "session_elapsed_s": elapsed,
            "recovered_mcp_tool_failure_count": taxonomy["recovered_mcp_tool_failure_count"],
            "current_consecutive_mcp_tool_failure_count": taxonomy[
                "current_consecutive_mcp_tool_failure_count"
            ],
            "maximum_consecutive_mcp_tool_failure_count": taxonomy[
                "maximum_consecutive_mcp_tool_failure_count"
            ],
            "mcp_tool_failure_taxonomy": taxonomy,
            "scientific_compliance_mcp_tool_failure_count": agent_invalid_count,
            "current_consecutive_scientific_compliance_mcp_tool_failure_count": (
                current_agent_invalid
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_count": (
                maximum_agent_invalid
            ),
            "scientific_compliance_mcp_tool_failure_episode_count": (agent_invalid_episode_count),
            "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                current_agent_invalid_episode
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                maximum_agent_invalid_episode
            ),
            "provider_error_event_count": provider_error_count,
            "limits": {
                "session_wall_time_limit_s": self.session_wall_time_limit_s,
                "max_recovered_mcp_tool_failures": self.max_recovered_mcp_tool_failures,
                "max_consecutive_mcp_tool_failures": (self.max_consecutive_mcp_tool_failures),
                "max_provider_error_events": self.max_provider_error_events,
                "mcp_tool_failure_limit_category": "agent_invalid_recovery_episode",
            },
        }

    def _operational_limit_failure(self, audit: Mapping[str, Any]) -> str | None:
        elapsed = float(audit.get("session_elapsed_s", 0.0))
        failed_calls = int(
            audit.get(
                "scientific_compliance_mcp_tool_failure_episode_count",
                audit.get(
                    "scientific_compliance_mcp_tool_failure_count",
                    audit.get("recovered_mcp_tool_failure_count", 0),
                ),
            )
        )
        consecutive_failed_calls = int(
            audit.get(
                "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count",
                audit.get(
                    "current_consecutive_scientific_compliance_mcp_tool_failure_count",
                    audit.get("current_consecutive_mcp_tool_failure_count", 0),
                ),
            )
        )
        provider_errors = int(audit.get("provider_error_event_count", 0))
        if self.session_wall_time_limit_s is not None and elapsed > self.session_wall_time_limit_s:
            return "session_wall_time_limit"
        if (
            self.max_recovered_mcp_tool_failures is not None
            and failed_calls > self.max_recovered_mcp_tool_failures
        ):
            return "max_recovered_mcp_tool_failures"
        if (
            self.max_consecutive_mcp_tool_failures is not None
            and consecutive_failed_calls > self.max_consecutive_mcp_tool_failures
        ):
            return "max_consecutive_mcp_tool_failures"
        if (
            self.max_provider_error_events is not None
            and provider_errors > self.max_provider_error_events
        ):
            return "max_provider_error_events"
        return None

    def _active_session_limit_failure(self) -> str | None:
        return self._operational_limit_failure(self._active_session_operational_audit())

    def close(self) -> None:
        """Stop an unfinished process; the official runner may call this on interruption."""

        if self._session is not None:
            self._record_interrupted_session(reason="agent_closed")

    def _start_session(self, current_packet: Mapping[str, Any]) -> None:
        session_id = f"{self.session_scope}-{self._sessions_started + 1:04d}-{uuid4().hex[:12]}"
        self.workspace.start_session(
            session_id=session_id,
            expected_step=1,
            response_timeout_s=self.request_timeout_s,
            session_scope=self.session_scope,
            terminal_action_readout_required=(
                self.terminal_action_readout_contract is not None
            ),
        )
        temporary = tempfile.TemporaryDirectory(prefix="chemworld-interactive-codex-")
        self._session_temp_directories.append(temporary)
        temp_root = Path(temporary.name)
        instructions_path = temp_root / "instructions.md"
        schema_path = temp_root / "final-schema.json"
        instructions_path.write_text(
            _campaign_system_prompt(
                terminal_action_readout=(
                    self.terminal_action_readout_contract is not None
                ),
                terminal_prediction_mode=str(
                    (self.terminal_action_readout_contract or {}).get(
                        "prediction_mode", "full_metrics"
                    )
                ),
            )
            if self.session_scope == "campaign"
            else _SYSTEM_PROMPT,
            encoding="utf-8",
        )
        schema_path.write_text(
            json.dumps(
                _CAMPAIGN_FINAL_OUTPUT_SCHEMA
                if self.session_scope == "campaign"
                else _FINAL_OUTPUT_SCHEMA,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._prepare_provider_launch(temp_root=temp_root)
        prompt = _initial_prompt(
            task_contract=self._task_contract,
            task_contract_manifest=self._task_contract_manifest,
            current_packet=current_packet,
            material_manifest=self._material_manifest,
            session_scope=self.session_scope,
            belief_checkpoint_manifest=self._belief_checkpoint_manifest,
            terminal_action_readout_manifest=self._terminal_action_readout_manifest,
        )
        if len(prompt.encode("utf-8")) > self.max_initial_prompt_bytes:
            raise InteractiveCodexExperimentError(
                "initial experiment prompt exceeds its hard byte cap"
            )
        command = self._command(
            instructions_path=instructions_path,
            schema_path=schema_path,
        )
        session_started_at = perf_counter()
        process = self._process_factory(
            command,
            prompt,
            self.workspace.agent_directory,
        )
        monitor = _CodexEventMonitor(process)
        self._sessions_started += 1
        self._provider_process_attempt_count += 1
        self._logical_codex_turn_count += 1
        self._session = {
            "session_id": session_id,
            "monitor": monitor,
            "temp": temporary,
            "accepted_action_count": 0,
            "prompt_byte_count": len(prompt.encode("utf-8")),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "instructions_path": str(instructions_path),
            "schema_path": str(schema_path),
            "started_at": session_started_at,
            "turn_started_at": session_started_at,
            "turn_receipts": [],
            "completed_turn_usage": _empty_usage(),
            "completed_turn_provider_error_event_count": 0,
        }
        self._session_action_count = 0

    def _emit_session_progress(self) -> None:
        if self.session_progress_callback is None or self._session is None:
            return
        monitor = self._session.get("monitor")
        if not isinstance(monitor, _CodexEventMonitor):
            return
        snapshot = monitor.snapshot()
        audit = self._active_session_operational_audit(monitor_snapshot=snapshot)
        usage_observed = snapshot.get("usage_observed") is True
        usage = snapshot.get("usage") if usage_observed else None
        self.session_progress_callback(
            to_builtin(
                {
                    "event": "provider_session_liveness",
                    "session_id": self._session.get("session_id"),
                    "thread_id": snapshot.get("thread_id"),
                    "session_elapsed_s": round(float(audit["session_elapsed_s"]), 3),
                    "accepted_action_count": int(self._session.get("accepted_action_count", 0)),
                    "next_expected_step": self._session_action_count + 1,
                    "event_counts": snapshot.get("event_counts", {}),
                    "mcp_tool_call_count": len(
                        self.workspace.mcp_tool_call_audit(str(self._session.get("session_id")))
                    ),
                    "recovered_mcp_tool_failure_count": audit["recovered_mcp_tool_failure_count"],
                    "current_consecutive_mcp_tool_failure_count": audit[
                        "current_consecutive_mcp_tool_failure_count"
                    ],
                    "maximum_consecutive_mcp_tool_failure_count": audit[
                        "maximum_consecutive_mcp_tool_failure_count"
                    ],
                    "provider_error_event_count": audit["provider_error_event_count"],
                    "usage_observed": usage_observed,
                    "usage_observation_event_type": snapshot.get("usage_observation_event_type"),
                    "usage": usage,
                }
            )
        )

    def _wait_for_next_request(
        self,
        current_packet: Mapping[str, Any],
    ) -> IPCRequest:
        while True:
            if self._session is None:
                raise InteractiveCodexExperimentError("Codex session is not active")
            limit_failure = self._active_session_limit_failure()
            if limit_failure is not None:
                self._record_interrupted_session(reason=limit_failure)
                raise InteractiveCodexExperimentError(
                    "Codex session crossed a frozen operational limit before the next "
                    "executable operation; no fallback action was emitted: " + limit_failure
                )
            session_id = str(self._session["session_id"])
            monitor = self._session["monitor"]
            if not isinstance(monitor, _CodexEventMonitor):
                raise TypeError("invalid Codex monitor")
            wait_timeout_s = self.request_timeout_s
            remaining_wall_time_s = self._active_session_remaining_wall_time_s()
            if remaining_wall_time_s is not None:
                wait_timeout_s = min(wait_timeout_s, max(0.001, remaining_wall_time_s))
            try:
                request = self.workspace.wait_for_request(
                    session_id=session_id,
                    expected_step=self._session_action_count + 1,
                    timeout_s=wait_timeout_s,
                    process_alive=monitor.alive,
                    handled_request_ids=self._handled_request_ids,
                    progress_callback=self._emit_session_progress,
                    progress_interval_s=self.session_progress_interval_s,
                )
            except (ExperimentCodexIPCError, TimeoutError) as error:
                limit_failure = self._active_session_limit_failure()
                retry_failure_type = self._pre_action_retry_failure_type(error)
                failure_reason = limit_failure or retry_failure_type or type(error).__name__
                accepted = int(self._session.get("accepted_action_count", 0))
                operational = self._active_session_operational_audit()
                session_id = str(self._session["session_id"])
                bridge_untouched = True
                try:
                    self._verify_experiment_tool()
                except ExperimentCodexIPCError:
                    bridge_untouched = False
                pre_action_retry_candidate = (
                    accepted == 0
                    and self._pre_action_restarts < self.pre_action_restart_limit
                    and (
                        self.provider_process_attempt_limit is None
                        or self._provider_process_attempt_count
                        < self.provider_process_attempt_limit
                    )
                    and limit_failure is None
                    and retry_failure_type is not None
                    and bridge_untouched
                    and int(operational["provider_error_event_count"]) == 0
                    and not self.workspace.mcp_tool_call_audit(session_id)
                )
                if self._continue_completed_accepted_turn(
                    error=error,
                    current_packet=current_packet,
                    limit_failure=limit_failure,
                    bridge_untouched=bridge_untouched,
                ):
                    continue
                receipt = self._record_interrupted_session(
                    reason=failure_reason,
                    pre_action_retry_candidate_failure_type=(
                        retry_failure_type if pre_action_retry_candidate else None
                    ),
                )
                can_restart = (
                    receipt.get("pre_action_retry_classification")
                    == ELIGIBLE_ZERO_ACTION_INFRASTRUCTURE_PREDECESSOR
                )
                if can_restart:
                    self._pre_action_restarts += 1
                    self._start_session(current_packet)
                    continue
                message = (
                    "Codex session crossed a frozen operational limit before the next "
                    "executable operation; no fallback action was emitted: " + limit_failure
                    if limit_failure is not None
                    else "Codex failed before the next executable operation; "
                    "no fallback action was emitted"
                )
                raise InteractiveCodexExperimentError(message) from error
            limit_failure = self._active_session_limit_failure()
            if limit_failure is not None:
                self._record_interrupted_session(reason=limit_failure)
                raise InteractiveCodexExperimentError(
                    "Codex session crossed a frozen operational limit before action "
                    "acceptance; no fallback action was emitted: " + limit_failure
                )
            return request

    @staticmethod
    def _pre_action_retry_failure_type(error: BaseException) -> str | None:
        """Name only transient request-wait failures eligible before any MCP call."""

        if isinstance(error, TimeoutError):
            return _REQUEST_WAIT_TIMEOUT
        if isinstance(error, ExperimentCodexIPCError) and str(error) == (
            "Codex process exited before submitting the next operation"
        ):
            return _PROCESS_EXITED_BEFORE_FIRST_REQUEST
        return None

    def _continue_completed_accepted_turn(
        self,
        *,
        error: BaseException,
        current_packet: Mapping[str, Any],
        limit_failure: str | None,
        bridge_untouched: bool,
    ) -> bool:
        """Resume one clean early-completed campaign turn on its original Codex thread."""

        if self._session is None:
            return False
        if (
            self.session_scope != "campaign"
            or self._accepted_turn_continuations >= self.accepted_turn_continuation_limit
            or (
                self.provider_process_attempt_limit is not None
                and self._provider_process_attempt_count >= self.provider_process_attempt_limit
            )
            or limit_failure is not None
            or not bridge_untouched
            or self._pre_action_retry_failure_type(error) != _PROCESS_EXITED_BEFORE_FIRST_REQUEST
            or int(self._session.get("accepted_action_count", 0)) <= 0
            or self._session.get("campaign_terminal") is True
        ):
            return False
        monitor = self._session.get("monitor")
        if not isinstance(monitor, _CodexEventMonitor):
            return False
        # A completed turn can emit its terminal event before the provider
        # process has published its return code.  One second was too short for
        # long DeepSeek campaign turns and misclassified clean early turns as
        # process interruptions, preventing same-thread continuation.
        result = monitor.wait(min(self.finalization_timeout_s, 30.0))
        usage = result.get("usage")
        turn_usage = usage if isinstance(usage, dict) else _empty_usage()
        normalized_usage = _combined_session_turn_usage(self._session, turn_usage)
        event_counts = result.get("event_counts")
        event_counts = event_counts if isinstance(event_counts, Mapping) else {}
        session_id = str(self._session["session_id"])
        mcp_tool_calls = self.workspace.mcp_tool_call_audit(session_id)
        taxonomy = _mcp_tool_failure_audit(
            [item for item in mcp_tool_calls if isinstance(item, Mapping)]
        )
        taxonomy_counts = taxonomy["counts_by_category"]
        provider_error_count = int(event_counts.get("error", 0)) + int(
            event_counts.get("turn.failed", 0)
        )
        if provider_error_count == 0:
            provider_errors = result.get("provider_errors")
            provider_error_count = len(provider_errors) if isinstance(provider_errors, list) else 0
        thread_id = result.get("thread_id")
        campaign_thread_id = self._session.get("thread_id")
        clean_provider_turn_completed = (
            int(event_counts.get("turn.completed", 0)) > 0
            and result.get("usage_observed") is True
            and _usage_complete(normalized_usage)
            and provider_error_count == 0
        )
        provider_process_finalized_cleanly_or_was_reaped = (
            (
                result.get("status") == "completed"
                and result.get("return_code") == 0
            )
            or result.get("status") == "timeout"
        )
        eligible = (
            clean_provider_turn_completed
            and provider_process_finalized_cleanly_or_was_reaped
            and isinstance(thread_id, str)
            and bool(thread_id)
            and (campaign_thread_id is None or thread_id == campaign_thread_id)
            and int(taxonomy_counts["provider_network"]) == 0
            and int(taxonomy_counts["transport_ipc_os"]) == 0
            and int(taxonomy_counts["unclassified"]) == 0
            and self.workspace.final_recommendation_audit(session_id) is None
        )
        if not eligible:
            return False
        turn_receipts = self._session.get("turn_receipts")
        completed_usage = self._session.get("completed_turn_usage")
        if not isinstance(turn_receipts, list) or not isinstance(completed_usage, dict):
            raise InteractiveCodexExperimentError("invalid accepted-turn continuation ledger")
        turn_receipts.append(
            to_builtin(
                {
                    "turn_index": len(turn_receipts) + 1,
                    "thread_id": thread_id,
                    "status": (
                        "completed_before_campaign_terminal"
                        if result.get("status") == "completed"
                        else "completed_before_campaign_terminal_process_reaped"
                    ),
                    "return_code": result.get("return_code"),
                    "provider_process_reaped_after_clean_turn": (
                        result.get("status") == "timeout"
                    ),
                    "accepted_action_count_after_turn": int(
                        self._session["accepted_action_count"]
                    ),
                    "continuation_classification": "eligible_same_thread_campaign_continuation",
                    "usage": normalized_usage,
                    "usage_complete": True,
                    "provider_error_event_count": 0,
                    "event_counts": dict(event_counts),
                    "mcp_tool_failure_taxonomy": taxonomy,
                    "experiment_tool_integrity_verified_after_turn": True,
                }
            )
        )
        # ``normalized_usage`` is the session total (completed turns plus this
        # turn).  Adding that total back into the completed-turn accumulator
        # counts every earlier turn again on each continuation.  Only the
        # provider usage reported for the newly completed turn belongs here.
        _merge_usage(completed_usage, turn_usage)
        self._session["thread_id"] = thread_id
        tool_events = result.get("tool_events")
        if isinstance(tool_events, list):
            self._completed_tool_events.extend(
                item for item in tool_events if isinstance(item, dict)
            )
        prompt = _accepted_turn_continuation_prompt(
            session_id=session_id,
            expected_step=self._session_action_count + 1,
            current_packet=current_packet,
        )
        command = self._resume_command(
            thread_id=thread_id,
            instructions_path=Path(self._session["instructions_path"]),
            schema_path=Path(self._session["schema_path"]),
        )
        process = self._process_factory(command, prompt, self.workspace.agent_directory)
        self._session["monitor"] = _CodexEventMonitor(process)
        self._session["turn_started_at"] = perf_counter()
        self._session["continuation_prompt_byte_count"] = len(prompt.encode("utf-8"))
        self._session["continuation_prompt_sha256"] = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        self._accepted_turn_continuations += 1
        self._provider_process_attempt_count += 1
        self._logical_codex_turn_count += 1
        return True

    def _finalize_session(self, *, terminal_reason: str) -> None:
        if self._session is None:
            return
        monitor = self._session["monitor"]
        if not isinstance(monitor, _CodexEventMonitor):
            raise TypeError("invalid Codex monitor")
        finalization_timeout_s = self.finalization_timeout_s
        remaining_wall_time_s = self._active_session_remaining_wall_time_s()
        if remaining_wall_time_s is not None:
            if remaining_wall_time_s <= 0.0:
                self._record_interrupted_session(reason="session_wall_time_limit")
                return
            finalization_timeout_s = min(finalization_timeout_s, remaining_wall_time_s)
        result = monitor.wait(finalization_timeout_s)
        integrity_error: ExperimentCodexIPCError | None = None
        try:
            self._verify_experiment_tool()
        except ExperimentCodexIPCError as error:
            integrity_error = error
        usage = result.get("usage")
        turn_usage = usage if isinstance(usage, dict) else _empty_usage()
        normalized_usage = _combined_session_turn_usage(self._session, turn_usage)
        usage_observed = result.get("usage_observed") is True
        _merge_usage(self._cumulative_usage, normalized_usage)
        tool_events = result.get("tool_events")
        if isinstance(tool_events, list):
            self._completed_tool_events.extend(
                item for item in tool_events if isinstance(item, dict)
            )
        session_id = str(self._session["session_id"])
        mcp_tool_calls = self.workspace.mcp_tool_call_audit(session_id)
        operational = self._active_session_operational_audit(
            monitor_snapshot=result,
            mcp_tool_calls=mcp_tool_calls,
        )
        belief_snapshots = self.workspace.belief_snapshot_audit(session_id)
        belief_snapshot_drafts = self.workspace.belief_snapshot_draft_audit(session_id)
        committed_recommendation = self.workspace.final_recommendation_audit(session_id)
        if belief_snapshots:
            self._belief_snapshots.extend(deepcopy(belief_snapshots))
        receipt_tool_events = (
            [item for item in tool_events if isinstance(item, dict)]
            if isinstance(tool_events, list)
            else []
        )
        if mcp_tool_calls and not any(
            event.get("event_type") == "mcp_tool_call" for event in receipt_tool_events
        ):
            receipt_tool_events.extend(_host_mcp_audit_events(mcp_tool_calls))
        final_payload = result.get("final_payload")
        final_valid = _valid_final_payload(final_payload)
        host_recommendation = (
            committed_recommendation.get("recommendation")
            if isinstance(committed_recommendation, dict)
            else None
        )
        final_recommendation = (
            deepcopy(host_recommendation)
            if isinstance(host_recommendation, dict)
            else _final_recommendation_from_payload(final_payload)
        )
        final_recommendation_source = (
            "host_mcp_commit"
            if isinstance(host_recommendation, dict)
            else "legacy_final_payload"
            if final_recommendation is not None
            else None
        )
        usage_complete = _usage_complete(turn_usage) and all(
            receipt.get("usage_complete") is True
            for receipt in self._session.get("turn_receipts", [])
            if isinstance(receipt, Mapping)
        )
        self._all_session_usage_complete = self._all_session_usage_complete and usage_complete
        self._all_session_usage_observed = self._all_session_usage_observed and usage_observed
        receipt = {
            "schema_version": "chemworld-interactive-codex-session-receipt-0.3",
            "session_id": self._session["session_id"],
            "thread_id": result.get("thread_id"),
            "status": result.get("status"),
            "return_code": result.get("return_code"),
            "terminal_reason": terminal_reason,
            "accepted_action_count": int(self._session.get("accepted_action_count", 0)),
            "pre_action_retry_classification": (
                TERMINAL_ACCEPTED_PRE_ACTION_RETRY_CLASSIFICATION
                if int(self._session.get("accepted_action_count", 0)) > 0
                else None
            ),
            "model_id": self.model,
            "reasoning_effort": self.reasoning_effort,
            "usage": normalized_usage,
            "turn_usage": turn_usage,
            "usage_observed": usage_observed,
            "usage_observation_event_type": result.get("usage_observation_event_type"),
            "usage_unavailable_reason": (
                None if usage_observed else "codex_cli_completed_without_usage_event"
            ),
            "usage_complete": usage_complete,
            "codex_turn_count": len(self._session.get("turn_receipts", [])) + 1,
            "accepted_turn_continuation_count": len(
                self._session.get("turn_receipts", [])
            ),
            "turn_receipts": [
                *deepcopy(self._session.get("turn_receipts", [])),
                _terminal_turn_receipt(
                    turn_index=len(self._session.get("turn_receipts", [])) + 1,
                    result=result,
                    usage=turn_usage,
                    accepted_action_count=int(self._session.get("accepted_action_count", 0)),
                    thread_id=result.get("thread_id"),
                ),
            ],
            "prompt_byte_count": self._session["prompt_byte_count"],
            "prompt_sha256": self._session["prompt_sha256"],
            "tool_events": receipt_tool_events,
            "event_counts": result.get("event_counts", {}),
            "provider_errors": result.get("provider_errors", []),
            "final_payload_valid": final_valid,
            "final_payload_encoding": result.get("final_payload_encoding"),
            "final_payload_status": (
                final_payload.get("status")
                if final_valid and isinstance(final_payload, dict)
                else None
            ),
            "final_payload_summary": (
                str(final_payload["summary"])
                if final_valid and isinstance(final_payload, dict)
                else None
            ),
            "final_recommendation": final_recommendation,
            "final_recommendation_source": final_recommendation_source,
            "final_recommendation_commit": committed_recommendation,
            "final_recommendation_sha256": (
                hashlib.sha256(_canonical_json(final_recommendation).encode("utf-8")).hexdigest()
                if final_recommendation is not None
                else None
            ),
            "stderr_byte_count": result.get("stderr_byte_count", 0),
            "stderr_sha256": result.get("stderr_sha256"),
            "artifact_access": self.workspace.artifact_access_audit(
                str(self._session["session_id"])
            ),
            "mcp_tool_calls": mcp_tool_calls,
            "session_scope": self.session_scope,
            "belief_snapshots": belief_snapshots,
            "belief_snapshot_count": len(belief_snapshots),
            "belief_snapshot_drafts": belief_snapshot_drafts,
            "belief_snapshot_draft_count": belief_snapshot_drafts["draft_count"],
            "belief_snapshot_fragment_count": belief_snapshot_drafts["fragment_count"],
            "experiment_tool_transport": "host_owned_stdio_mcp",
            "mcp_tool_integrity_verified_after_session": integrity_error is None,
            "experiment_tool_integrity_verified_after_session": integrity_error is None,
            "lab_tool_integrity_verified_after_session": integrity_error is None,
            "private_reasoning_retained": False,
            "session_elapsed_s": operational["session_elapsed_s"],
            "recovered_mcp_tool_failure_count": operational["recovered_mcp_tool_failure_count"],
            "current_consecutive_mcp_tool_failure_count": operational[
                "current_consecutive_mcp_tool_failure_count"
            ],
            "maximum_consecutive_mcp_tool_failure_count": operational[
                "maximum_consecutive_mcp_tool_failure_count"
            ],
            "mcp_tool_failure_taxonomy": operational["mcp_tool_failure_taxonomy"],
            "scientific_compliance_mcp_tool_failure_count": operational[
                "scientific_compliance_mcp_tool_failure_count"
            ],
            "current_consecutive_scientific_compliance_mcp_tool_failure_count": (
                operational["current_consecutive_scientific_compliance_mcp_tool_failure_count"]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_count": (
                operational["maximum_consecutive_scientific_compliance_mcp_tool_failure_count"]
            ),
            "scientific_compliance_mcp_tool_failure_episode_count": operational[
                "scientific_compliance_mcp_tool_failure_episode_count"
            ],
            "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                operational[
                    "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count"
                ]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                operational[
                    "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count"
                ]
            ),
            "provider_error_event_count": operational["provider_error_event_count"],
            "session_operational_limits": operational["limits"],
        }
        self._session_receipts.append(to_builtin(receipt))
        self._sessions_completed += 1
        self._completed_session_elapsed_s += float(operational["session_elapsed_s"])
        self._recovered_mcp_tool_failure_count += int(
            operational["recovered_mcp_tool_failure_count"]
        )
        self._maximum_consecutive_mcp_tool_failure_count = max(
            self._maximum_consecutive_mcp_tool_failure_count,
            int(operational["maximum_consecutive_mcp_tool_failure_count"]),
        )
        self._maximum_consecutive_mcp_tool_failure_episode_count = max(
            self._maximum_consecutive_mcp_tool_failure_episode_count,
            int(
                operational["mcp_tool_failure_taxonomy"]["recovery_episode_taxonomy"][
                    "maximum_consecutive_recovery_episode_count"
                ]
            ),
        )
        self._accumulate_mcp_tool_failure_taxonomy(operational)
        self._provider_error_event_count += int(operational["provider_error_event_count"])
        if self._last_decision is not None:
            self._last_decision["completed_session_receipt"] = to_builtin(receipt)
        self._retire_active_session()
        self._session = None
        if integrity_error is not None:
            raise InteractiveCodexExperimentError(
                "the experiment tool bridge failed its post-session integrity check"
            ) from integrity_error

    def _record_interrupted_session(
        self,
        *,
        reason: str,
        pre_action_retry_candidate_failure_type: str | None = None,
    ) -> dict[str, Any]:
        """Retain an attempt receipt even when no environment action was emitted."""

        if self._session is None:
            return {}
        monitor = self._session.get("monitor")
        if not isinstance(monitor, _CodexEventMonitor):
            self._retire_active_session()
            self._session = None
            return {}
        monitor.stop()
        snapshot = monitor.snapshot()
        try:
            self._verify_experiment_tool()
        except ExperimentCodexIPCError:
            experiment_tool_integrity_verified = False
        else:
            experiment_tool_integrity_verified = True
        usage = snapshot.get("usage")
        turn_usage = usage if isinstance(usage, dict) else _empty_usage()
        normalized_usage = _combined_session_turn_usage(self._session, turn_usage)
        usage_observed = snapshot.get("usage_observed") is True
        _merge_usage(self._cumulative_usage, normalized_usage)
        usage_complete = _usage_complete(turn_usage) and all(
            receipt.get("usage_complete") is True
            for receipt in self._session.get("turn_receipts", [])
            if isinstance(receipt, Mapping)
        )
        tool_events = snapshot.get("tool_events")
        session_id = str(self._session["session_id"])
        mcp_tool_calls = self.workspace.mcp_tool_call_audit(session_id)
        belief_snapshots = self.workspace.belief_snapshot_audit(session_id)
        belief_snapshot_drafts = self.workspace.belief_snapshot_draft_audit(session_id)
        committed_recommendation = self.workspace.final_recommendation_audit(session_id)
        if belief_snapshots:
            self._belief_snapshots.extend(deepcopy(belief_snapshots))
        host_recommendation = (
            committed_recommendation.get("recommendation")
            if isinstance(committed_recommendation, dict)
            else None
        )
        operational = self._active_session_operational_audit(
            monitor_snapshot=snapshot,
            mcp_tool_calls=mcp_tool_calls,
        )
        accepted_action_count = int(self._session.get("accepted_action_count", 0))
        effective_reason = (
            _PROCESS_EXITED_BEFORE_NEXT_ACTION
            if accepted_action_count > 0
            and reason == _PROCESS_EXITED_BEFORE_FIRST_REQUEST
            else reason
        )
        no_observed_provider_activity = (
            pre_action_retry_candidate_failure_type
            == _PROCESS_EXITED_BEFORE_FIRST_REQUEST
            and not usage_observed
            and snapshot.get("thread_id") is None
            and not snapshot.get("event_counts")
            and not snapshot.get("tool_events")
            and not snapshot.get("provider_errors")
        )
        eligible_predecessor = (
            pre_action_retry_candidate_failure_type
            in {_PROCESS_EXITED_BEFORE_FIRST_REQUEST, _REQUEST_WAIT_TIMEOUT}
            and (usage_complete or no_observed_provider_activity)
            and accepted_action_count == 0
            and int(operational["provider_error_event_count"]) == 0
            and not mcp_tool_calls
            and not belief_snapshots
            and belief_snapshot_drafts.get("draft_count") == 0
            and belief_snapshot_drafts.get("fragment_count") == 0
            and committed_recommendation is None
            and experiment_tool_integrity_verified
        )
        pre_action_retry_classification = (
            ELIGIBLE_ZERO_ACTION_INFRASTRUCTURE_PREDECESSOR
            if eligible_predecessor
            else TERMINAL_ACCEPTED_PRE_ACTION_RETRY_CLASSIFICATION
            if accepted_action_count > 0
            else None
        )
        unobserved_pre_action_attempt = (
            eligible_predecessor and no_observed_provider_activity
        )
        if not unobserved_pre_action_attempt:
            self._all_session_usage_complete = self._all_session_usage_complete and usage_complete
            self._all_session_usage_observed = self._all_session_usage_observed and usage_observed
        else:
            self._unattributed_pre_action_process_attempt_count += 1
        receipt_tool_events = (
            [item for item in tool_events if isinstance(item, dict)]
            if isinstance(tool_events, list)
            else []
        )
        if mcp_tool_calls and not any(
            event.get("event_type") == "mcp_tool_call" for event in receipt_tool_events
        ):
            receipt_tool_events.extend(_host_mcp_audit_events(mcp_tool_calls))
        receipt = {
            "schema_version": "chemworld-interactive-codex-session-receipt-0.3",
            "session_id": self._session["session_id"],
            "thread_id": snapshot.get("thread_id"),
            "status": "interrupted_before_next_action",
            "failure_type": effective_reason,
            "accepted_action_count": accepted_action_count,
            "pre_action_retry_classification": pre_action_retry_classification,
            "model_id": self.model,
            "reasoning_effort": self.reasoning_effort,
            "usage": normalized_usage,
            "turn_usage": turn_usage,
            "usage_observed": usage_observed,
            "usage_observation_event_type": snapshot.get("usage_observation_event_type"),
            "usage_unavailable_reason": (
                None
                if usage_observed
                else "no_observed_provider_activity_before_process_exit_usage_unattributed"
                if unobserved_pre_action_attempt
                else "codex_cli_emitted_no_usage_before_forced_termination"
            ),
            "usage_complete": usage_complete,
            "codex_turn_count": len(self._session.get("turn_receipts", [])) + 1,
            "accepted_turn_continuation_count": len(
                self._session.get("turn_receipts", [])
            ),
            "turn_receipts": [
                *deepcopy(self._session.get("turn_receipts", [])),
                _terminal_turn_receipt(
                    turn_index=len(self._session.get("turn_receipts", [])) + 1,
                    result=snapshot,
                    usage=turn_usage,
                    accepted_action_count=accepted_action_count,
                    thread_id=snapshot.get("thread_id"),
                    status="interrupted_before_next_action",
                ),
            ],
            "usage_accounting_scope": (
                "unobserved_not_attributable_pre_action_process_attempt"
                if unobserved_pre_action_attempt
                else "provider_process_attempt"
            ),
            "prompt_byte_count": self._session["prompt_byte_count"],
            "prompt_sha256": self._session["prompt_sha256"],
            "tool_events": receipt_tool_events,
            "event_counts": snapshot.get("event_counts", {}),
            "provider_errors": snapshot.get("provider_errors", []),
            "stderr_byte_count": snapshot.get("stderr_byte_count", 0),
            "stderr_sha256": snapshot.get("stderr_sha256"),
            "final_recommendation": (
                deepcopy(host_recommendation) if isinstance(host_recommendation, dict) else None
            ),
            "final_recommendation_source": (
                "host_mcp_commit" if isinstance(host_recommendation, dict) else None
            ),
            "final_recommendation_commit": committed_recommendation,
            "final_recommendation_sha256": (
                hashlib.sha256(_canonical_json(host_recommendation).encode("utf-8")).hexdigest()
                if isinstance(host_recommendation, dict)
                else None
            ),
            "artifact_access": self.workspace.artifact_access_audit(session_id),
            "mcp_tool_calls": mcp_tool_calls,
            "session_scope": self.session_scope,
            "belief_snapshots": belief_snapshots,
            "belief_snapshot_count": len(belief_snapshots),
            "belief_snapshot_drafts": belief_snapshot_drafts,
            "belief_snapshot_draft_count": belief_snapshot_drafts["draft_count"],
            "belief_snapshot_fragment_count": belief_snapshot_drafts["fragment_count"],
            "experiment_tool_transport": "host_owned_stdio_mcp",
            "mcp_tool_integrity_verified_after_session": (experiment_tool_integrity_verified),
            "experiment_tool_integrity_verified_after_session": (
                experiment_tool_integrity_verified
            ),
            "lab_tool_integrity_verified_after_session": (experiment_tool_integrity_verified),
            "private_reasoning_retained": False,
            "session_elapsed_s": operational["session_elapsed_s"],
            "recovered_mcp_tool_failure_count": operational["recovered_mcp_tool_failure_count"],
            "current_consecutive_mcp_tool_failure_count": operational[
                "current_consecutive_mcp_tool_failure_count"
            ],
            "maximum_consecutive_mcp_tool_failure_count": operational[
                "maximum_consecutive_mcp_tool_failure_count"
            ],
            "mcp_tool_failure_taxonomy": operational["mcp_tool_failure_taxonomy"],
            "scientific_compliance_mcp_tool_failure_count": operational[
                "scientific_compliance_mcp_tool_failure_count"
            ],
            "current_consecutive_scientific_compliance_mcp_tool_failure_count": (
                operational["current_consecutive_scientific_compliance_mcp_tool_failure_count"]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_count": (
                operational["maximum_consecutive_scientific_compliance_mcp_tool_failure_count"]
            ),
            "scientific_compliance_mcp_tool_failure_episode_count": operational[
                "scientific_compliance_mcp_tool_failure_episode_count"
            ],
            "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                operational[
                    "current_consecutive_scientific_compliance_mcp_tool_failure_episode_count"
                ]
            ),
            "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count": (
                operational[
                    "maximum_consecutive_scientific_compliance_mcp_tool_failure_episode_count"
                ]
            ),
            "provider_error_event_count": operational["provider_error_event_count"],
            "session_operational_limits": operational["limits"],
        }
        self._session_receipts.append(to_builtin(receipt))
        self._sessions_completed += 1
        self._completed_session_elapsed_s += float(operational["session_elapsed_s"])
        self._recovered_mcp_tool_failure_count += int(
            operational["recovered_mcp_tool_failure_count"]
        )
        self._maximum_consecutive_mcp_tool_failure_count = max(
            self._maximum_consecutive_mcp_tool_failure_count,
            int(operational["maximum_consecutive_mcp_tool_failure_count"]),
        )
        self._maximum_consecutive_mcp_tool_failure_episode_count = max(
            self._maximum_consecutive_mcp_tool_failure_episode_count,
            int(
                operational["mcp_tool_failure_taxonomy"]["recovery_episode_taxonomy"][
                    "maximum_consecutive_recovery_episode_count"
                ]
            ),
        )
        self._accumulate_mcp_tool_failure_taxonomy(operational)
        self._provider_error_event_count += int(operational["provider_error_event_count"])
        self._retire_active_session()
        self._session = None
        return to_builtin(receipt)

    def _accumulate_mcp_tool_failure_taxonomy(
        self,
        operational: Mapping[str, Any],
    ) -> None:
        taxonomy = operational.get("mcp_tool_failure_taxonomy", {})
        counts = taxonomy.get("counts_by_category", {}) if isinstance(taxonomy, Mapping) else {}
        maximums = (
            taxonomy.get("maximum_consecutive_counts_by_category", {})
            if isinstance(taxonomy, Mapping)
            else {}
        )
        episode_taxonomy = (
            taxonomy.get("recovery_episode_taxonomy", {}) if isinstance(taxonomy, Mapping) else {}
        )
        episode_counts = (
            episode_taxonomy.get("counts_by_category", {})
            if isinstance(episode_taxonomy, Mapping)
            else {}
        )
        episode_maximums = (
            episode_taxonomy.get("maximum_consecutive_counts_by_category", {})
            if isinstance(episode_taxonomy, Mapping)
            else {}
        )
        for category in MCP_TOOL_FAILURE_CATEGORIES:
            self._mcp_tool_failure_counts_by_category[category] += int(counts.get(category, 0))
            self._maximum_consecutive_mcp_tool_failure_counts_by_category[category] = max(
                self._maximum_consecutive_mcp_tool_failure_counts_by_category[category],
                int(maximums.get(category, 0)),
            )
            self._mcp_tool_failure_episode_counts_by_category[category] += int(
                episode_counts.get(category, 0)
            )
            self._maximum_consecutive_mcp_tool_failure_episode_counts_by_category[category] = max(
                self._maximum_consecutive_mcp_tool_failure_episode_counts_by_category[category],
                int(episode_maximums.get(category, 0)),
            )

    def _retire_active_session(self) -> None:
        """Drop protocol files and temporary prompts after retaining their audit."""

        if self._session is None:
            return
        session_id = str(self._session["session_id"])
        monitor = self._session.get("monitor")
        if isinstance(monitor, _CodexEventMonitor):
            # A Codex process owns the MCP subprocess. Reap it before removing
            # active_session.json so no late MCP request can race retirement.
            monitor.stop()
        self.workspace.retire_session(session_id)
        temporary = self._session.get("temp")
        if isinstance(temporary, tempfile.TemporaryDirectory):
            for attempt in range(TEMP_DIRECTORY_CLEANUP_RETRY_LIMIT):
                try:
                    temporary.cleanup()
                    break
                except OSError:
                    if attempt + 1 < TEMP_DIRECTORY_CLEANUP_RETRY_LIMIT:
                        sleep(TEMP_DIRECTORY_CLEANUP_RETRY_INTERVAL_S)
            if temporary in self._session_temp_directories:
                self._session_temp_directories.remove(temporary)
        self._session_process_environment = None
        self._use_isolated_codex_home = False

    def _publish_outcome_artifact(
        self,
        *,
        event_id: str,
        action: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if action.get("operation") != "measure":
            return None
        raw = info.get("raw_signal")
        if not isinstance(raw, Mapping) or not raw:
            return None
        artifact_id = f"characterization-{event_id}"
        return self.workspace.publish_artifact(
            artifact_id=artifact_id,
            payload={
                "schema_version": "chemworld-public-characterization-artifact-0.1",
                "event_id": event_id,
                "instrument": action.get("instrument"),
                "raw_signal": to_builtin(raw),
            },
        )

    def _publish_latest_artifact(
        self,
        public_view: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if self._pending_outcome is None:
            return None
        existing = self._pending_outcome.get("artifact")
        if isinstance(existing, Mapping):
            return deepcopy(dict(existing))
        action = self._pending_outcome.get("action")
        if not isinstance(action, Mapping) or action.get("operation") != "measure":
            return None
        tool = public_view.get("tool_json")
        raw = tool.get("raw_signal") if isinstance(tool, Mapping) else None
        if not isinstance(raw, Mapping) or not raw:
            return None
        event_id = str(self._pending_outcome["event_id"])
        artifact_id = f"characterization-{event_id}"
        return self.workspace.publish_artifact(
            artifact_id=artifact_id,
            payload={
                "schema_version": "chemworld-public-characterization-artifact-0.1",
                "event_id": event_id,
                "instrument": action.get("instrument"),
                "raw_signal": to_builtin(raw),
            },
        )

    def _command(self, *, instructions_path: Path, schema_path: Path) -> list[str]:
        command = [
            self.codex_executable,
            "exec",
            "--json",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--disable",
            "apps",
            "--disable",
            "multi_agent",
            "--disable",
            "plugins",
            "--sandbox",
            "workspace-write",
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            f"mcp_servers.chemworld_lab.command={json.dumps(sys.executable)}",
            "-c",
            (
                "mcp_servers.chemworld_lab.args="
                + json.dumps(
                    [
                        "-m",
                        "chemworld.agents.experiment_codex_mcp",
                        "--workspace",
                        str(self.workspace.root),
                    ]
                )
            ),
            "-c",
            ("mcp_servers.chemworld_lab.cwd=" + json.dumps(str(self._source_project_root))),
            "-c",
            "mcp_servers.chemworld_lab.required=true",
            "-c",
            "mcp_servers.chemworld_lab.enabled=true",
            "-c",
            "mcp_servers.chemworld_lab.supports_parallel_tool_calls=false",
            "-c",
            ("mcp_servers.chemworld_lab.enabled_tools=" + json.dumps(list(SUPPORTED_TOOLS))),
            "-c",
            'mcp_servers.chemworld_lab.default_tools_approval_mode="approve"',
            "-c",
            'mcp_servers.chemworld_lab.tools.step.approval_mode="approve"',
            "-c",
            "mcp_servers.chemworld_lab.startup_timeout_sec=30",
            "-c",
            (
                "mcp_servers.chemworld_lab.tool_timeout_sec="
                + str(int(self.request_timeout_s + 31.0))
            ),
            "-c",
            f'model_provider="{self.model_provider}"',
            *self._model_provider_config_overrides(),
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            f"model_instructions_file={json.dumps(instructions_path.as_posix())}",
            "-m",
            self.model,
            "-C",
            str(self.workspace.agent_directory),
        ]
        if not self._use_isolated_codex_home:
            command.insert(5, "--ignore-user-config")
        return command

    def _resume_command(
        self,
        *,
        thread_id: str,
        instructions_path: Path,
        schema_path: Path,
    ) -> list[str]:
        initial = self._command(instructions_path=instructions_path, schema_path=schema_path)
        if initial and initial[-1] == "-":
            initial = initial[:-1]
        command = [*initial[:2], "resume", *initial[2:]]
        for option in ("--sandbox", "-C"):
            index = command.index(option)
            del command[index : index + 2]
        command.extend([thread_id, "-"])
        return command

    def _model_provider_config_overrides(self) -> list[str]:
        """Render bounded Codex CLI provider overrides without exposing secrets."""

        if self.model_provider == HTTPS_PROVIDER_ID:
            config = (
                '{name="OpenAI",wire_api="responses",requires_openai_auth=true,'
                "supports_websockets=false}"
            )
            return ["-c", f"model_providers.{self.model_provider}={config}"]
        else:
            fields = [
                f"model_providers.{self.model_provider}.name={json.dumps(self.model_provider_name)}",
                f"model_providers.{self.model_provider}.wire_api={json.dumps(self.model_provider_wire_api)}",
            ]
            if self.model_provider_base_url:
                fields.append(
                    f"model_providers.{self.model_provider}.base_url={json.dumps(self.model_provider_base_url)}"
                )
            if self.model_provider_env_key:
                fields.append(
                    f"model_providers.{self.model_provider}.env_key={json.dumps(self.model_provider_env_key)}"
                )
            if self.model_provider_model_catalog_json:
                fields.append(
                    "model_catalog_json="
                    + json.dumps(self.model_provider_model_catalog_json.resolve().as_posix())
                )
            if self.model_provider_preferred_auth_method:
                fields.append(
                    "preferred_auth_method=" + json.dumps(self.model_provider_preferred_auth_method)
                )
            if self.model_provider_forced_login_method:
                fields.append(
                    "forced_login_method=" + json.dumps(self.model_provider_forced_login_method)
                )
            fields.append(f"model_providers.{self.model_provider}.supports_websockets=false")
            rendered: list[str] = []
            for field in fields:
                rendered.extend(["-c", field])
            return rendered

    def _prepare_provider_launch(self, *, temp_root: Path) -> None:
        """Prepare a cell-isolated persistent Codex home for exec + resume."""

        if (
            self.model_provider_auth_mode == "experimental_bearer_token"
            and self.model_provider_api_key_file is None
        ):
            raise InteractiveCodexExperimentError(
                "DeepSeek-compatible bearer-token launch requires an API-key file"
            )
        if (
            self.model_provider_auth_mode == "experimental_bearer_token"
            and self.model_provider_model_catalog_json is None
        ):
            raise InteractiveCodexExperimentError(
                "bearer-token launch requires an explicit model_catalog_json"
            )
        if (
            self.model_provider_auth_mode == "experimental_bearer_token"
            and self.model_provider_model_catalog_json is not None
            and not self.model_provider_model_catalog_json.is_file()
        ):
            raise InteractiveCodexExperimentError("configured model_catalog_json does not exist")

        codex_home = temp_root / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        if self.model_provider_auth_mode == "experimental_bearer_token":
            assert self.model_provider_api_key_file is not None
            assert self.model_provider_model_catalog_json is not None
            try:
                api_key = self.model_provider_api_key_file.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise InteractiveCodexExperimentError(
                    "unable to read the configured provider API-key file"
                ) from error
            if not api_key:
                raise InteractiveCodexExperimentError("configured provider API-key file is empty")
            config_lines = [
                f"model = {json.dumps(self.model)}",
                f"model_provider = {json.dumps(self.model_provider)}",
                f"model_reasoning_effort = {json.dumps(self.reasoning_effort)}",
                'preferred_auth_method = "apikey"',
                'forced_login_method = "api"',
                "model_catalog_json = "
                + json.dumps(self.model_provider_model_catalog_json.resolve().as_posix()),
                "",
                f"[model_providers.{self.model_provider}]",
                f"name = {json.dumps(self.model_provider_name)}",
                f"base_url = {json.dumps((self.model_provider_base_url or '').rstrip('/') + '/')}",
                f"wire_api = {json.dumps(self.model_provider_wire_api)}",
                f"experimental_bearer_token = {json.dumps(api_key)}",
            ]
            (codex_home / "config.toml").write_text(
                "\n".join(config_lines) + "\n", encoding="utf-8"
            )
        elif self.model_provider == HTTPS_PROVIDER_ID:
            source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            source_auth = source_home / "auth.json"
            if not source_auth.is_file():
                raise InteractiveCodexExperimentError(
                    "OpenAI Codex auth is unavailable for isolated session persistence"
                )
            shutil.copyfile(source_auth, codex_home / "auth.json")
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(codex_home)
        self._session_process_environment = environment
        self._use_isolated_codex_home = True


def _initial_prompt(
    *,
    task_contract: Mapping[str, Any],
    task_contract_manifest: Mapping[str, Any],
    current_packet: Mapping[str, Any],
    material_manifest: Mapping[str, Any],
    session_scope: str = "experiment",
    belief_checkpoint_manifest: Mapping[str, Any] | None = None,
    terminal_action_readout_manifest: Mapping[str, Any] | None = None,
) -> str:
    campaign = session_scope == "campaign"
    recipe_coverage_contract = _participant_recipe_coverage_contract(task_contract)
    payload = {
        "schema_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
        "instruction": (
            "Run the complete multi-experiment discovery campaign autonomously. Commit the "
            "required typed belief checkpoints, choose every operation after its public "
            "outcome, preserve the shared resource ledger, close every planned batch, and "
            "satisfy the recipe_coverage_contract as a hard qualification obligation."
            if campaign
            else "Complete one experiment autonomously and optimize the declared leaderboard "
            "score under the scoring contract's component weights, gates, safety, and "
            "resource limits. Current state is authoritative; submit every operation "
            "through chemworld_lab.step. No host closeout is available."
        ),
        "session_scope": session_scope,
        "recipe_coverage_contract": (
            to_builtin(recipe_coverage_contract) if campaign else None
        ),
        "task": to_builtin(dict(task_contract)),
        "task_contract_reference": {
            "contents_in_prompt": True,
            **to_builtin(dict(task_contract_manifest)),
            "relative_path": "../reference/task_contract.json",
        },
        "material_information": {
            "contents_in_prompt": False,
            **to_builtin(dict(material_manifest)),
            "relative_path": "../reference/material_information.json",
        },
        "belief_checkpoint_contract": (
            {
                "contents_in_prompt": False,
                **to_builtin(dict(belief_checkpoint_manifest or {})),
                "relative_path": "../reference/belief_checkpoint_contract.json",
                "submit_with": "chemworld_lab.commit_belief_snapshot",
                "submission_protocol": [
                    "action=begin",
                    "append every host-published prediction page in order",
                    "append every host-published law page in order",
                    "action=finalize",
                ],
            }
            if campaign
            else None
        ),
        "terminal_action_readout": (
            {
                "required": True,
                "contents_in_prompt": False,
                **to_builtin(dict(terminal_action_readout_manifest or {})),
                "reveal_gate": "campaign_terminal_and_all_belief_checkpoints_committed",
                "reveal_with": "chemworld_lab.terminal_action_readout",
                "outcomes_revealed": False,
                "participant_must_rank_all_candidates": True,
            }
            if terminal_action_readout_manifest is not None
            else None
        ),
        "initial_public_state": to_builtin(dict(current_packet)),
        "workspace": {
            "writable_root": "agent/ (current working directory; optional memory)",
            "transport": "host-owned STDIO MCP; no shell command is required",
            "mcp_server": "chemworld_lab (required, bounded, host-owned)",
            "material_reference": "chemworld_lab.material_information",
            "task_contract_reference": "../reference/task_contract.json",
            "public_history": "complete public operation stream with bounded pages; recent cache",
            "authoritative_trajectory_available": False,
        },
        "mcp_step_example": {
            "expected_step": 1,
            "action": {"operation": "one currently legal operation"},
            "submit_with": "chemworld_lab.step",
        },
    }
    if campaign and task_contract.get("free_research_campaign") is True:
        payload["instruction"] = (
            "Complete the research campaign for the stated research_goal using public lab tools. "
            "Choose experiments freely; there are no typed belief checkpoints or required law "
            "forms. Preserve context across batches and close the experiments specified "
            "by the published campaign budget. "
            + (
                "After campaign terminal, submit the short research handoff; no operating "
                "recommendation is required. "
                if task_contract.get("final_recommendation_required") is False
                else "After campaign terminal, commit one recommended completed experiment "
                "and submit the short final handoff. "
            )
            + "Free mechanism reporting and blind prediction follow "
            "in later turns without further lab access."
        )
        payload["belief_checkpoint_contract"] = None
        payload["recipe_coverage_contract"] = None
    return _canonical_json(payload)


def _participant_recipe_coverage_contract(
    task_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve the single campaign-card coverage contract for participant views."""

    campaign_resources = task_contract.get("campaign_resources")
    if not isinstance(campaign_resources, Mapping):
        return None
    card = campaign_resources.get("card")
    if not isinstance(card, Mapping):
        return None
    metadata = card.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    contract = metadata.get("recipe_coverage_contract")
    if not isinstance(contract, Mapping):
        return None
    return deepcopy(dict(contract))


def _bounded_current_packet(
    context: AgentDecisionContext,
    public_view: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None,
    recipe_coverage_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tool = public_view.get("tool_json")
    tool_view = tool if isinstance(tool, Mapping) else {}
    available = tool_view.get("available_actions")
    actions = (
        [
            _compact_legal_action(item)
            for item in available
            if isinstance(item, Mapping) and item.get("valid") is not False
        ]
        if isinstance(available, list)
        else []
    )
    state = context.to_dict()
    campaign = state.get("campaign_state")
    campaign_mapping = campaign if isinstance(campaign, Mapping) else {}
    tool_campaign = tool_view.get("campaign_state")
    tool_campaign_mapping = tool_campaign if isinstance(tool_campaign, Mapping) else {}
    resource_snapshot: Any = None
    for source in (campaign_mapping, tool_campaign_mapping, tool_view):
        for key in (
            "campaign_resources",
            "campaign_resource_state",
            "resource_state",
            "resource_ledger",
        ):
            candidate = source.get(key)
            if isinstance(candidate, Mapping):
                resource_snapshot = candidate
                break
        if resource_snapshot is not None:
            break
    participant_campaign = _participant_visible_campaign(campaign_mapping)
    compact_campaign = _compact_scalars(participant_campaign)
    for key in ("completed_batches", "discarded_batches", "experiment_summaries"):
        if key in participant_campaign:
            compact_campaign[key] = _compact_nested(participant_campaign.get(key))
    resource_snapshot = _participant_visible_resource_state(resource_snapshot)
    return {
        "schema_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
        "step": context.step,
        "stage": context.decision_stage,
        "remaining_operations": context.remaining_operations,
        "campaign_state": compact_campaign,
        "campaign_resources": _compact_nested(resource_snapshot),
        "recipe_coverage_contract": _compact_nested(recipe_coverage_contract),
        "visible_metrics": _compact_scalars(context.visible_metrics),
        **(
            {"operational_state": _compact_nested(tool_view["operational_state"])}
            if isinstance(tool_view.get("operational_state"), Mapping)
            else {}
        ),
        "latest_measurement": summarize_measurement(context.latest_spectra),
        "characterization_artifact": to_builtin(artifact) if artifact else None,
        "active_constraint_flags": {
            str(key): to_builtin(value)
            for key, value in context.constraint_flags.items()
            if value not in (None, False, 0, "", [], {})
        },
        "uncertainty": _compact_scalars(context.uncertainty),
        "legal_actions": actions,
    }


def _one_based_experiment_summary(value: object) -> object:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    result = deepcopy(dict(value))
    index = result.get("experiment_index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        lifecycle_index = index + 1
        result["experiment_index"] = lifecycle_index
        result.setdefault("lifecycle_experiment_index", lifecycle_index)
        result.setdefault("batch_id", f"batch-{lifecycle_index:04d}")
    result["experiment_index_base"] = 1
    return result


def _participant_visible_resource_state(value: object) -> object:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    result = deepcopy(dict(value))
    current = result.get("current_experiment")
    if isinstance(current, Mapping):
        current_result = deepcopy(dict(current))
        raw_index = current_result.get("experiment_index")
        terminal = result.get("campaign_terminal") is True
        if isinstance(raw_index, int) and not isinstance(raw_index, bool) and raw_index >= 0:
            current_result["experiment_index"] = None if terminal else raw_index + 1
        current_result["experiment_index_base"] = 1
        result["current_experiment"] = current_result
    return result


def _participant_visible_campaign(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose one unambiguous 1-based experiment namespace to Codex participants."""

    result = deepcopy(dict(value))
    summaries = result.get("experiment_summaries")
    closed_count = len(summaries) if isinstance(summaries, list) else 0
    completed_count = 0
    visible_summaries: list[object] = []
    if isinstance(summaries, list):
        for row in summaries:
            visible = _one_based_experiment_summary(row)
            if isinstance(row, Mapping):
                completed = row.get("final_assay") is True or row.get("outcome") == "completed"
                if row.get("outcome") is None and row.get("final_assay") is None:
                    completed = row.get("batch_discarded") is not True
                if completed:
                    completed_count += 1
                    if isinstance(visible, dict):
                        visible["completed_ordinal"] = completed_count
            visible_summaries.append(visible)
    result["closed_batch_count"] = closed_count
    result["completed_experiment_count"] = completed_count
    result["completed_experiment_indices"] = [
        int(row["lifecycle_experiment_index"])
        for row in visible_summaries
        if isinstance(row, Mapping) and row.get("completed_ordinal") is not None
    ]
    result["experiment_index_base"] = 1
    raw_index = result.get("experiment_index")
    terminal = result.get("done") is True or _mapping_flag(
        result.get("campaign_resources"), "campaign_terminal"
    )
    if isinstance(raw_index, int) and not isinstance(raw_index, bool) and raw_index >= 0:
        result["experiment_index"] = None if terminal else raw_index + 1
    result["experiment_summaries"] = visible_summaries
    for key in ("completed_batches", "discarded_batches"):
        rows = result.get(key)
        if isinstance(rows, list):
            result[key] = [_one_based_experiment_summary(row) for row in rows]
    completed_batches = result.get("completed_batches")
    if isinstance(completed_batches, list):
        for ordinal, row in enumerate(completed_batches, start=1):
            if isinstance(row, dict):
                row["completed_ordinal"] = ordinal
    if result.get("last_terminal_summary") is not None:
        result["last_terminal_summary"] = _one_based_experiment_summary(
            result["last_terminal_summary"]
        )
    result["campaign_resources"] = _participant_visible_resource_state(
        result.get("campaign_resources")
    )
    return result


def _mapping_flag(value: object, key: str) -> bool:
    return isinstance(value, Mapping) and value.get(key) is True


def _compact_outcome(
    *,
    event_id: str,
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    reward: float,
    info: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "action": to_builtin(dict(action)),
        "transaction_status": info.get("transaction_status"),
        "operation_type": info.get("operation_type", action.get("operation")),
        "reward": float(reward),
        "observation": {
            str(key): to_builtin(value) for key, value in observation.items() if value is not None
        },
        "processed_estimate": _compact_nested(info.get("processed_estimate")),
        "observed_keys": to_builtin(info.get("observed_keys", [])),
        "constraint_flags": {
            str(key): to_builtin(value)
            for key, value in (
                info.get("constraint_flags", {}).items()
                if isinstance(info.get("constraint_flags"), Mapping)
                else ()
            )
            if value not in (None, False, 0, "", [], {})
        },
        "error_message": info.get("error_message"),
        "leaderboard_score": info.get("leaderboard_score"),
        "measurement_cost": info.get("measurement_cost"),
        "sample_consumed": info.get("sample_consumed"),
        "state_delta_summary": _compact_nested(info.get("state_delta_summary")),
        "artifact": to_builtin(artifact) if artifact else None,
    }


def _material_information_payload(
    task_info: Mapping[str, Any],
    *,
    initial_world_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = task_info.get("material_information")
    material = raw if isinstance(raw, Mapping) else {}
    dossier = material.get("dossier")
    if isinstance(dossier, Mapping):
        public_material_information: dict[str, Any] = {
            "availability": "nominal_property_dossier",
            "dossier": to_builtin(dict(dossier)),
            "interpretation": (
                "This dossier is incomplete nominal prior information. Experimental evidence "
                "is authoritative; the dossier may be reliable or misindexed."
            ),
        }
    else:
        public_material_information = {
            "availability": "opaque_identifiers_only",
            "dossier": None,
            "interpretation": (
                "No task-specific nominal property dossier is supplied. Experimental evidence "
                "is authoritative."
            ),
        }
    payload = {
        "schema_version": "chemworld-env-owned-material-information-reference-0.1",
        "material_information": public_material_information,
        "material_catalog": to_builtin(task_info.get("material_catalog", {})),
    }
    if initial_world_model is not None:
        forbidden = {"arm", "prior_arm", "world_seed", "condition_id", "quality"}
        leaked: list[str] = []

        def collect_identity_fields(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    key = str(raw_key)
                    child_path = f"{path}.{key}"
                    if key in forbidden:
                        leaked.append(child_path)
                    collect_identity_fields(child, child_path)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    collect_identity_fields(child, f"{path}[{index}]")

        collect_identity_fields(initial_world_model, "initial_world_model")
        if leaked:
            raise ValueError(
                "initial_world_model contains participant-visible identity fields: "
                f"{sorted(leaked)}"
            )
        payload["initial_world_model"] = to_builtin(dict(initial_world_model))
    return payload


def _public_task_contract(task_info: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "env_id",
        "full_process_contract_id",
        "research_brief",
        "task_id",
        "task_contract_hash",
        "composition",
        "task_goal",
        "objective",
        "description",
        "budget",
        "official_budget",
        "episode_mode",
        "contract_profile",
        "safety_limit",
        "success_metrics",
        "constraints",
        "termination_policy",
        "observation_policy",
        "measurement_policy",
        "observation_contract",
        "observation_contract_hash",
        "experiment_lifecycle",
        "autonomous_workflow",
        "electrochemical_workflow_mode",
        "allowed_operations",
        "allowed_instruments",
        "method_budget_contract",
        "campaign_resource_card",
        "campaign_resources",
        "scoring_contract",
        "scoring_contract_hash",
        "scoring_contract_id",
        "scoring_weights",
        "objective_weights",
        "submission_requirements",
    )
    contract = {
        "schema_version": "chemworld-public-interactive-task-contract-0.1",
        **{key: to_builtin(task_info[key]) for key in keys if key in task_info},
    }
    if not isinstance(contract.get("experiment_lifecycle"), Mapping):
        allowed_operations = {str(value) for value in task_info.get("allowed_operations", [])}
        allowed_instruments = {str(value) for value in task_info.get("allowed_instruments", [])}
        if "terminate" in allowed_operations and "final_assay" in allowed_instruments:
            method_budget = task_info.get("method_budget_contract")
            planned_experiments = (
                method_budget.get("complete_experiment_limit", 1)
                if isinstance(method_budget, Mapping)
                else 1
            )
            contract["experiment_lifecycle"] = {
                "schema_version": "chemworld-public-experiment-lifecycle-0.1",
                "planned_complete_experiments": int(planned_experiments),
                "explicit_terminate_required": True,
                "final_assay_required": True,
                "final_assay_after_terminate": True,
                "automatic_closeout": False,
                "terminate_action_template": '{"operation":"terminate"}',
                "final_assay_action_template": (
                    '{"operation":"measure","instrument":"final_assay"}'
                ),
            }
    return contract


def _compact_legal_action(item: Mapping[str, Any]) -> dict[str, Any]:
    compact = compact_action(item)
    schema = item.get("schema")
    fields = schema.get("fields") if isinstance(schema, Mapping) else None
    if not isinstance(fields, list):
        return compact
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        name = field.get("field", field.get("name"))
        labels = field.get("choice_labels")
        if not isinstance(name, str) or not isinstance(labels, Mapping):
            continue
        target = compact.get(name)
        if isinstance(target, dict):
            target["choice_labels"] = {str(key): str(value) for key, value in labels.items()}
    return compact


def _compact_scalars(value: Any, *, limit: int = 32) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in sorted(value):
        item = value[key]
        if item is None or isinstance(item, bool | int | float):
            result[str(key)] = to_builtin(item)
        elif isinstance(item, str) and len(item) <= 256:
            result[str(key)] = item
        if len(result) >= limit:
            break
    return result


def _compact_nested(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if value is None or isinstance(value, bool | int | float):
        return to_builtin(value)
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, Mapping):
        return {
            str(key): compact
            for key, item in list(value.items())[:32]
            if (compact := _compact_nested(item, depth=depth + 1)) is not None
        }
    if isinstance(value, list | tuple) and len(value) <= 16:
        return [
            compact
            for item in value
            if (compact := _compact_nested(item, depth=depth + 1)) is not None
        ]
    return None


def _sanitize_command_event(item: Mapping[str, Any]) -> dict[str, Any]:
    command = str(item.get("command", ""))
    output = str(item.get("aggregated_output", ""))
    lower = command.lower()
    if "lab_tool.py step" in lower:
        classification = "lab_step"
    elif "lab_tool.py inspect" in lower:
        classification = "artifact_inspect"
    elif "lab_tool.py history" in lower:
        classification = "history_read"
    elif "lab_tool.py status" in lower:
        classification = "status_read"
    elif any(marker in lower for marker in ("get-content", "type ", " cat ")):
        classification = "file_read"
    elif any(marker in lower for marker in ("set-content", "out-file", "apply_patch")):
        classification = "file_write"
    else:
        classification = "other"
    paths = sorted(
        {
            match.replace("\\", "/")
            for match in re.findall(
                r"(?:agent|public|reference)/[A-Za-z0-9_.\-/]+",
                command.replace("\\", "/"),
            )
        }
    )
    data = output.encode("utf-8", errors="replace")
    return {
        "event_type": "command_execution",
        "classification": classification,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "referenced_relative_paths": paths[:16],
        "output_byte_count": len(data),
        "output_line_count": len(output.splitlines()),
        "output_sha256": hashlib.sha256(data).hexdigest(),
        "exit_code": item.get("exit_code"),
        "status": item.get("status"),
        "command_body_retained": False,
        "output_body_retained": False,
    }


def _sanitize_file_change_event(item: Mapping[str, Any]) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    raw = item.get("changes")
    for change in raw if isinstance(raw, list) else ():
        if not isinstance(change, Mapping):
            continue
        path = str(change.get("path", "")).replace("\\", "/")
        changes.append({"path": path, "kind": change.get("kind")})
    return {
        "event_type": "file_change",
        "changes": changes[:64],
        "content_retained": False,
    }


def _mcp_tool_classification(tool_name: str) -> str:
    return {
        "step": "lab_step",
        "status": "status_read",
        "history": "history_read",
        "inspect_artifact": "artifact_inspect",
        "material_information": "material_information_read",
        "commit_belief_snapshot": "belief_checkpoint_commit",
        "commit_final_recommendation": "final_recommendation_commit",
    }.get(tool_name, "other")


def _sanitize_mcp_tool_event(item: Mapping[str, Any]) -> dict[str, Any]:
    raw_tool = item.get("tool") or item.get("name") or item.get("tool_name") or ""
    tool_name = str(raw_tool).rsplit("__", maxsplit=1)[-1]
    arguments = item.get("arguments") or item.get("input") or {}
    result = item.get("result") or item.get("output") or {}
    return {
        "event_type": "mcp_tool_call",
        "classification": _mcp_tool_classification(tool_name),
        "server": "chemworld_lab",
        "tool": tool_name,
        "arguments_sha256": hashlib.sha256(_canonical_json(arguments).encode("utf-8")).hexdigest(),
        "result_sha256": hashlib.sha256(_canonical_json(result).encode("utf-8")).hexdigest(),
        "status": item.get("status"),
        "started_at": item.get("started_at") or item.get("timestamp"),
        "duration_ms": item.get("duration_ms"),
        "arguments_body_retained": False,
        "result_body_retained": False,
    }


def _host_mcp_audit_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "host_mcp_tool_call_audit",
            "classification": _mcp_tool_classification(str(row.get("tool", ""))),
            "server": "chemworld_lab",
            "tool": row.get("tool"),
            "arguments_sha256": row.get("arguments_sha256"),
            "result_sha256": row.get("result_sha256"),
            "argument_keys": row.get("argument_keys", []),
            "started_at": row.get("started_at"),
            "duration_ms": row.get("duration_ms"),
            "status": row.get("status", "called"),
            "error_type": row.get("error_type"),
            "error_code": row.get("error_code"),
            "error_field_path": row.get("error_field_path"),
            "error_detail": row.get("error_detail"),
            "error_detail_byte_count": row.get("error_detail_byte_count", 0),
            "error_detail_sha256": row.get("error_detail_sha256"),
            "arguments_body_retained": False,
            "result_body_retained": False,
        }
        for row in rows
    ]


def _normalize_usage(raw: Mapping[str, Any]) -> dict[str, int]:
    prompt = _nonnegative_int(raw.get("input_tokens"))
    completion = _nonnegative_int(raw.get("output_tokens"))
    cached = min(_nonnegative_int(raw.get("cached_input_tokens")), prompt)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_cache_hit_tokens": cached,
        "prompt_cache_miss_tokens": prompt - cached,
        "prompt_cache_write_tokens": _nonnegative_int(raw.get("cache_write_input_tokens")),
        "reasoning_output_tokens": _nonnegative_int(raw.get("reasoning_output_tokens")),
    }


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "prompt_cache_write_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def _merge_usage(total: dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in total:
        total[key] += _nonnegative_int(usage.get(key))


def _combined_session_turn_usage(
    session: Mapping[str, Any],
    current_turn_usage: Mapping[str, Any],
) -> dict[str, int]:
    total = _empty_usage()
    completed = session.get("completed_turn_usage")
    if isinstance(completed, Mapping):
        _merge_usage(total, completed)
    _merge_usage(total, current_turn_usage)
    return total


def _terminal_turn_receipt(
    *,
    turn_index: int,
    result: Mapping[str, Any],
    usage: Mapping[str, Any],
    accepted_action_count: int,
    thread_id: Any,
    status: str = "campaign_terminal_turn",
) -> dict[str, Any]:
    event_counts = result.get("event_counts")
    provider_errors = result.get("provider_errors")
    provider_error_count = (
        int(event_counts.get("error", 0)) + int(event_counts.get("turn.failed", 0))
        if isinstance(event_counts, Mapping)
        else 0
    )
    if provider_error_count == 0 and isinstance(provider_errors, list):
        provider_error_count = len(provider_errors)
    return {
        "turn_index": turn_index,
        "thread_id": thread_id,
        "status": status,
        "return_code": result.get("return_code"),
        "accepted_action_count_after_turn": accepted_action_count,
        "continuation_classification": "terminal_turn",
        "usage": dict(usage),
        "usage_complete": _usage_complete(usage),
        "provider_error_event_count": provider_error_count,
        "event_counts": dict(event_counts) if isinstance(event_counts, Mapping) else {},
    }


def _accepted_turn_continuation_prompt(
    *,
    session_id: str,
    expected_step: int,
    current_packet: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": INTERACTIVE_CODEX_EXPERIMENT_VERSION,
        "instruction": (
            "Continue the same unfinished campaign from the host-owned append-only boundary. "
            "Do not repeat or replace any accepted operation. Read the current public state, "
            "submit the next required lab call at exactly expected_step, and continue until the "
            "campaign terminal protocol is complete."
        ),
        "session_id": session_id,
        "expected_step": expected_step,
        "current_state": to_builtin(dict(current_packet)),
    }
    return _canonical_json(payload)


def _usage_complete(usage: Mapping[str, Any]) -> bool:
    prompt = _nonnegative_int(usage.get("prompt_tokens"))
    return (
        prompt > 0
        and _nonnegative_int(usage.get("total_tokens"))
        == prompt + _nonnegative_int(usage.get("completion_tokens"))
        and _nonnegative_int(usage.get("prompt_cache_hit_tokens"))
        + _nonnegative_int(usage.get("prompt_cache_miss_tokens"))
        == prompt
    )


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _valid_final_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status")
        in {
            "experiment_complete",
            "campaign_complete",
            "batch_discarded",
            "budget_exhausted",
            "stopped",
        }
        and isinstance(value.get("summary"), str)
        and len(value["summary"]) <= 3000
    )


def _final_recommendation_from_payload(value: Any) -> dict[str, Any] | None:
    """Normalize nested and flat campaign recommendation encodings."""

    if not isinstance(value, dict):
        return None
    nested = value.get("final_recommendation")
    if isinstance(nested, dict):
        return deepcopy(nested)
    index = value.get("selected_experiment_index")
    rationale = value.get("selection_rationale")
    if isinstance(index, int) and not isinstance(index, bool) and isinstance(rationale, str):
        return {
            "selected_experiment_index": index,
            "selection_rationale": rationale,
        }
    return None


def _parse_final_payload(message: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse exact JSON or one whole-message JSON fence without retaining raw text."""

    stripped = message.strip()
    candidates = [(stripped, "json")]
    fenced = re.fullmatch(
        r"```(?:json)?\s*(?P<body>\{.*\})\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        candidates.append((fenced.group("body"), "markdown_json_fence"))
    for candidate_text, encoding in candidates:
        try:
            candidate = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate, encoding
    return None, None


def _system_prompt_hash() -> str:
    return hashlib.sha256(
        (
            _SYSTEM_PROMPT
            + "|"
            + _CAMPAIGN_SYSTEM_PROMPT
            + "|"
            + INTERACTIVE_CODEX_EXPERIMENT_VERSION
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "DEFAULT_FINALIZATION_TIMEOUT_S",
    "INTERACTIVE_CODEX_EXPERIMENT_VERSION",
    "InteractiveCodexExperimentAgent",
    "InteractiveCodexExperimentError",
    "ProcessFactory",
]
