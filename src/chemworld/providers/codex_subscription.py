"""Auditable structured-output client backed by a ChatGPT Codex subscription."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from chemworld.providers.deepseek import DeepSeekAPIError, JsonCompletion

DEFAULT_MODEL = "gpt-5.6-sol"
MODEL_ACCESS_DATE = "2026-09-27"
MODEL_SOURCE = "https://developers.openai.com/api/docs/guides/latest-model"
AUTH_SOURCE = "https://learn.chatgpt.com/docs/auth"
SUPPORTED_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-6-astra")
HTTPS_PROVIDER_ID = "chemworld_openai_https"
ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
CommandRunner = Callable[[Sequence[str], str | None, float], subprocess.CompletedProcess[str]]


class CodexSubscriptionError(DeepSeekAPIError):
    """Redacted Codex CLI authentication, transport, or structured-output failure."""


def _default_command_runner(
    command: Sequence[str],
    stdin: str | None,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        list(command),
        input=stdin,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
        timeout=timeout_s,
        **kwargs,
    )


class CodexSubscriptionClient:
    """Call ``codex exec`` through an existing ChatGPT subscription login."""

    def __init__(
        self,
        *,
        codex_executable: str | None = None,
        model: str | None = None,
        timeout_s: float = 600.0,
        reasoning_effort: ReasoningEffort = "medium",
        max_attempts: int = 2,
        retry_backoff_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        command_runner: CommandRunner = _default_command_runner,
        persistent_workspace: Path | None = None,
        allow_document_tools: bool = False,
    ) -> None:
        resolved_executable = codex_executable or shutil.which("codex")
        if not resolved_executable:
            raise CodexSubscriptionError("Codex CLI is not installed or is not available on PATH.")
        resolved_model = model or DEFAULT_MODEL
        if resolved_model not in SUPPORTED_MODELS:
            raise CodexSubscriptionError(
                f"Codex subscription model {resolved_model!r} is not supported by this adapter."
            )
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported Codex subscription reasoning effort")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be non-negative")
        resolved_workspace = (
            None if persistent_workspace is None else persistent_workspace.resolve()
        )
        if resolved_workspace is not None and not resolved_workspace.is_dir():
            raise ValueError("persistent_workspace must be an existing directory")
        if allow_document_tools and resolved_workspace is None:
            raise ValueError("allow_document_tools requires an isolated persistent_workspace")
        self.codex_executable = str(resolved_executable)
        self.model = resolved_model
        self.thinking = True
        self.timeout_s = float(timeout_s)
        self.reasoning_effort = reasoning_effort
        self.max_attempts = int(max_attempts)
        self.retry_backoff_s = float(retry_backoff_s)
        self.persistent_workspace = resolved_workspace
        self.allow_document_tools = bool(allow_document_tools)
        self._sleep = sleep
        self._command_runner = command_runner
        self.cli_version = self._read_cli_version()
        self.auth_mode = self._verify_chatgpt_login()

    def pricing_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-provider-subscription-accounting-0.1",
            "provider": "OpenAI Codex",
            "requested_model_id": self.model,
            "transport": "codex_exec_jsonl_https_sse",
            "websocket_enabled": False,
            "authentication": self.auth_mode,
            "codex_cli_version": self.cli_version,
            "model_source": MODEL_SOURCE,
            "model_access_date": MODEL_ACCESS_DATE,
            "auth_source": AUTH_SOURCE,
            "accounting_complete": False,
            "pricing_unavailable_reason": (
                "ChatGPT subscription usage cannot be attributed to a per-run USD token price."
            ),
        }

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        output_schema: Mapping[str, Any] | None = None,
    ) -> JsonCompletion:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        resolved_output_schema = (
            _validated_explicit_output_schema(output_schema)
            if output_schema is not None
            else _output_schema_from_prompt(user_prompt)
        )
        aggregate_usage = _empty_usage()
        attempt_records: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._execute_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=resolved_output_schema,
                    max_tokens=max_tokens,
                )
            except subprocess.TimeoutExpired as error:
                last_error = error
                attempt_records.append(
                    _attempt_record(
                        attempt_index=attempt,
                        status="failed",
                        model_id=self.model,
                        failure_type="codex_cli_timeout",
                    )
                )
            except OSError as error:
                last_error = error
                attempt_records.append(
                    _attempt_record(
                        attempt_index=attempt,
                        status="failed",
                        model_id=self.model,
                        failure_type="codex_cli_process_error",
                    )
                )
            else:
                events = _parse_jsonl_events(result.stdout)
                usage = _normalized_usage(events)
                _merge_usage(aggregate_usage, usage)
                thread_id = _thread_id(events)
                error_events = _error_messages(events)
                if result.returncode != 0:
                    last_error = CodexSubscriptionError(
                        _failure_summary(result.returncode, error_events)
                    )
                    attempt_records.append(
                        _attempt_record(
                            attempt_index=attempt,
                            status="failed",
                            model_id=self.model,
                            request_id=thread_id,
                            usage=usage,
                            usage_complete=_usage_complete(usage),
                            failure_type="codex_cli_nonzero_exit",
                            internal_error_event_count=len(error_events),
                        )
                    )
                    if _nonretryable_cli_failure(error_events, result.stderr):
                        break
                else:
                    try:
                        payload = _final_payload(events)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        last_error = error
                        attempt_records.append(
                            _attempt_record(
                                attempt_index=attempt,
                                status="failed",
                                model_id=self.model,
                                request_id=thread_id,
                                usage=usage,
                                usage_complete=_usage_complete(usage),
                                failure_type="invalid_structured_output",
                                internal_error_event_count=len(error_events),
                            )
                        )
                    else:
                        attempt_records.append(
                            _attempt_record(
                                attempt_index=attempt,
                                status="succeeded",
                                model_id=self.model,
                                request_id=thread_id,
                                usage=usage,
                                usage_complete=_usage_complete(usage),
                                finish_reason="turn_completed",
                                internal_error_event_count=len(error_events),
                            )
                        )
                        return JsonCompletion(
                            payload=payload,
                            model=self.model,
                            usage=dict(aggregate_usage),
                            request_id=thread_id,
                            attempts=attempt,
                            finish_reason="turn_completed",
                            reasoning_content_present=False,
                            reasoning_character_count=0,
                            attempt_records=tuple(attempt_records),
                        )
            if attempt < self.max_attempts:
                self._sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
        raise CodexSubscriptionError(
            f"Codex subscription call failed after {len(attempt_records)} attempt(s).",
            attempts=max(len(attempt_records), 1),
            usage=aggregate_usage,
            retryable=not _is_nonretryable_error(last_error),
            attempt_records=tuple(attempt_records),
        ) from last_error

    def _execute_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Mapping[str, Any],
        max_tokens: int,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="chemworld-codex-subscription-") as temp:
            temp_root = Path(temp)
            workspace_root = self.persistent_workspace or temp_root
            instructions_path = temp_root / "instructions.md"
            schema_path = temp_root / "output-schema.json"
            execution_envelope = (
                "- Use the shell tool only to inspect or update files inside the isolated "
                "workspace.\n"
                "- Do not use apps, multi-agent delegation, plugins, web search, or "
                "external context.\n"
                if self.allow_document_tools
                else "- Do not call tools, inspect files, or use external context.\n"
            )
            instructions_path.write_text(
                (
                    f"{system_prompt}\n\n"
                    "Execution envelope:\n"
                    f"{execution_envelope}"
                    "- Return only the requested JSON object.\n"
                    f"- Keep the final response within {max_tokens} output tokens.\n"
                ),
                encoding="utf-8",
            )
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            command = [
                self.codex_executable,
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
            ]
            if not self.allow_document_tools:
                command.extend(["--disable", "shell_tool"])
            command.extend(
                [
                    "--disable",
                    "apps",
                    "--disable",
                    "multi_agent",
                    "--disable",
                    "plugins",
                ]
            )
            if self.allow_document_tools:
                command.extend(["--sandbox", "workspace-write"])
            command.extend(
                [
                    "-c",
                    'approval_policy="never"',
                    "-c",
                    'web_search="disabled"',
                    "-c",
                    f'model_provider="{HTTPS_PROVIDER_ID}"',
                    "-c",
                    (
                        f"model_providers.{HTTPS_PROVIDER_ID}="
                        '{name="OpenAI",wire_api="responses",'
                        "requires_openai_auth=true,supports_websockets=false}"
                    ),
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-c",
                    (f"model_instructions_file={json.dumps(instructions_path.as_posix())}"),
                    "-m",
                    self.model,
                    "-C",
                    str(workspace_root),
                ]
            )
            return self._command_runner(command, user_prompt, self.timeout_s)

    def _read_cli_version(self) -> str:
        try:
            result = self._command_runner(
                [self.codex_executable, "--version"],
                None,
                min(self.timeout_s, 30.0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CodexSubscriptionError("Unable to query the Codex CLI version.") from error
        version = (result.stdout or result.stderr).strip()
        if result.returncode != 0 or not version:
            raise CodexSubscriptionError("Unable to query the Codex CLI version.")
        return version

    def _verify_chatgpt_login(self) -> str:
        try:
            result = self._command_runner(
                [self.codex_executable, "login", "status"],
                None,
                min(self.timeout_s, 30.0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CodexSubscriptionError("Unable to query Codex login status.") from error
        status = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0:
            raise CodexSubscriptionError(
                "Codex CLI is not logged in. Run `codex login` with the ChatGPT account."
            )
        if "logged in using chatgpt" not in status.lower():
            raise CodexSubscriptionError(
                "Codex CLI is not using the required ChatGPT subscription login."
            )
        return "chatgpt_subscription_cached_login"


def _output_schema_from_prompt(user_prompt: str) -> dict[str, Any]:
    prompt = json.loads(user_prompt)
    if not isinstance(prompt, Mapping):
        raise ValueError("Codex subscription prompt must be a JSON object")
    shape = prompt.get("required_json_shape")
    if not isinstance(shape, Mapping) or not shape:
        raise ValueError("Codex subscription prompt lacks required_json_shape")
    schema = _schema_from_shape(shape)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("required_json_shape must describe a JSON object")
    return schema


def _validated_explicit_output_schema(
    output_schema: Mapping[str, Any],
) -> dict[str, Any]:
    schema = dict(output_schema)
    if not schema or schema.get("type") != "object":
        raise ValueError("explicit output_schema must describe a JSON object")
    try:
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("explicit output_schema must be JSON serializable") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("explicit output_schema must describe a JSON object")
    return decoded


def _schema_from_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        declared_type = value.get("type")
        if declared_type in {"string", "number", "integer", "boolean"}:
            schema: dict[str, Any] = {"type": declared_type}
            for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                constraint = value.get(keyword)
                if isinstance(constraint, int | float) and not isinstance(constraint, bool):
                    schema[keyword] = constraint
            if isinstance(value.get("enum"), list):
                schema["enum"] = list(value["enum"])
            return schema
        properties = {str(key): _schema_from_shape(item) for key, item in value.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "items": _schema_from_shape(value[0]) if value else {},
        }
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith("integer"):
            return {"type": "integer"}
        if lowered.startswith("number"):
            return {"type": "number"}
        if lowered.startswith("boolean"):
            return {"type": "boolean"}
        return {"type": "string"}
    return {}


def _parse_jsonl_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _thread_id(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "thread.started":
            value = event.get("thread_id")
            return value if isinstance(value, str) and value else None
    return None


def _final_payload(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    messages: list[str] = []
    completed = False
    for event in events:
        if event.get("type") == "turn.completed":
            completed = True
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            messages.append(str(item["text"]))
    if not completed or not messages:
        raise ValueError("Codex CLI did not produce a completed agent message")
    payload = json.loads(messages[-1])
    if not isinstance(payload, dict):
        raise TypeError("Codex CLI structured output is not an object")
    return payload


def _normalized_usage(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    usage = _empty_usage()
    for event in events:
        raw = event.get("usage")
        if event.get("type") != "turn.completed" or not isinstance(raw, Mapping):
            continue
        prompt = _nonnegative_int(raw.get("input_tokens"))
        completion = _nonnegative_int(raw.get("output_tokens"))
        cache_hit = min(_nonnegative_int(raw.get("cached_input_tokens")), prompt)
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_cache_hit_tokens": cache_hit,
            "prompt_cache_miss_tokens": prompt - cache_hit,
        }
    return usage


def _error_messages(events: Sequence[Mapping[str, Any]]) -> list[str]:
    messages: list[str] = []
    for event in events:
        if event.get("type") == "error" and isinstance(event.get("message"), str):
            messages.append(str(event["message"]))
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "error"
            and isinstance(item.get("message"), str)
        ):
            messages.append(str(item["message"]))
    return messages


def _failure_summary(returncode: int, messages: Sequence[str]) -> str:
    if messages:
        return f"Codex CLI failed with exit code {returncode}: {messages[-1][:300]}"
    return f"Codex CLI failed with exit code {returncode}."


def _nonretryable_cli_failure(messages: Sequence[str], stderr: str) -> bool:
    text = "\n".join([*messages, stderr]).lower()
    return any(
        marker in text
        for marker in (
            "requires a newer version of codex",
            "not logged in",
            "unexpected argument",
            "invalid value",
            "invalid json schema",
            "model metadata",
        )
    )


def _is_nonretryable_error(error: Exception | None) -> bool:
    return isinstance(error, CodexSubscriptionError) and not error.retryable


def _attempt_record(
    *,
    attempt_index: int,
    status: str,
    model_id: str,
    request_id: str | None = None,
    usage: Mapping[str, int] | None = None,
    usage_complete: bool = False,
    failure_type: str | None = None,
    finish_reason: str | None = None,
    internal_error_event_count: int = 0,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_index": int(attempt_index),
        "status": status,
        "request_id": request_id,
        "model_id": model_id,
        "model_identity_source": "explicit_codex_cli_argument",
        "transport": "https_sse",
        "websocket_enabled": False,
        "usage": dict(usage or {}),
        "usage_complete": bool(usage_complete),
        "billable": False,
        "usage_source": ("codex_cli_turn_completed" if usage_complete else "unavailable"),
        "finish_reason": finish_reason,
        "content_character_count": 0,
        "reasoning_character_count": 0,
        "internal_error_event_count": int(internal_error_event_count),
    }
    if failure_type:
        record["failure_type"] = failure_type
    return record


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }


def _merge_usage(total: dict[str, int], usage: Mapping[str, int]) -> None:
    for key in total:
        total[key] += _nonnegative_int(usage.get(key))


def _usage_complete(usage: Mapping[str, int]) -> bool:
    prompt = _nonnegative_int(usage.get("prompt_tokens"))
    completion = _nonnegative_int(usage.get("completion_tokens"))
    return (
        prompt > 0
        and _nonnegative_int(usage.get("total_tokens")) == prompt + completion
        and _nonnegative_int(usage.get("prompt_cache_hit_tokens"))
        + _nonnegative_int(usage.get("prompt_cache_miss_tokens"))
        == prompt
    )


def _nonnegative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


__all__ = [
    "AUTH_SOURCE",
    "DEFAULT_MODEL",
    "HTTPS_PROVIDER_ID",
    "MODEL_ACCESS_DATE",
    "MODEL_SOURCE",
    "SUPPORTED_MODELS",
    "CodexSubscriptionClient",
    "CodexSubscriptionError",
    "ReasoningEffort",
]
