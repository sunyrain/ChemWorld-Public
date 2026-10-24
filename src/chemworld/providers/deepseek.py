"""Auditable client for DeepSeek's OpenAI-compatible chat API."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

from chemworld._serialization import canonical_json_sha256 as _canonical_sha256

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
PRICING_ACCESS_DATE = "2026-07-13"
PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
MODEL_SOURCE = "https://api-docs.deepseek.com/api/list-models"
SUPPORTED_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
ReasoningEffort = Literal["high", "max"]
StoredReasoningEffort = Literal["medium", "high", "max"]


@dataclass(frozen=True)
class DeepSeekPricing:
    model_id: str
    input_cache_hit_per_million_usd: float
    input_cache_miss_per_million_usd: float
    output_per_million_usd: float
    access_date: str = PRICING_ACCESS_DATE
    source: str = PRICING_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_cache_hit_per_million_usd": self.input_cache_hit_per_million_usd,
            "input_cache_miss_per_million_usd": self.input_cache_miss_per_million_usd,
            "output_per_million_usd": self.output_per_million_usd,
            "access_date": self.access_date,
            "source": self.source,
        }


_PRICING = {
    "deepseek-v4-flash": DeepSeekPricing("deepseek-v4-flash", 0.0028, 0.14, 0.28),
    "deepseek-v4-pro": DeepSeekPricing("deepseek-v4-pro", 0.003625, 0.435, 0.87),
}


@dataclass(frozen=True)
class JsonCompletion:
    payload: dict[str, Any]
    model: str
    usage: dict[str, Any]
    request_id: str | None = None
    attempts: int = 1
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    reasoning_content_present: bool = False
    reasoning_character_count: int = 0
    attempt_records: tuple[dict[str, Any], ...] = ()


class JsonPlannerClient(Protocol):
    model: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        output_schema: Mapping[str, Any] | None = None,
    ) -> JsonCompletion: ...


class DeepSeekAPIError(RuntimeError):
    """Redacted provider, transport, identity, or structured-output failure."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        usage: dict[str, int] | None = None,
        retryable: bool = False,
        status_code: int | None = None,
        attempt_records: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = max(int(attempts), 1)
        self.usage = dict(usage or {})
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.attempt_records = tuple(dict(item) for item in attempt_records)


class DeepSeekClient:
    """Call a frozen DeepSeek model without provider SDK or silent fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
        thinking: bool = False,
        reasoning_effort: ReasoningEffort = "max",
        strict_tool_calls: bool = False,
        max_attempts: int = 3,
        retry_backoff_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not self._api_key:
            raise DeepSeekAPIError(
                "DEEPSEEK_API_KEY is not set. Keep the key in an environment variable."
            )
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be high or max")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be non-negative")
        self.timeout_s = float(timeout_s)
        self.thinking = bool(thinking)
        self.reasoning_effort: StoredReasoningEffort = reasoning_effort
        self.strict_tool_calls = bool(strict_tool_calls)
        if self.strict_tool_calls and not self.base_url.endswith("/beta"):
            raise ValueError("DeepSeek strict tool calls require a /beta base URL")
        self.max_attempts = int(max_attempts)
        self.retry_backoff_s = float(retry_backoff_s)
        self._sleep = sleep

    def pricing_snapshot(self) -> dict[str, Any]:
        try:
            pricing = _PRICING[self.model]
        except KeyError as exc:
            raise DeepSeekAPIError(
                f"No frozen pricing or formal identity for model {self.model!r}"
            ) from exc
        payload = pricing.to_dict()
        payload.update(
            {
                "schema_version": "chemworld-provider-pricing-0.4",
                "provider": "DeepSeek",
                "currency": "USD",
                "requested_model_id": self.model,
                "legacy_alias": False,
                "model_source": MODEL_SOURCE,
                "base_url": self.base_url,
            }
        )
        payload["pricing_version_sha256"] = _canonical_sha256(payload)
        return payload

    def estimate_cost_usd(self, usage: dict[str, Any]) -> float:
        pricing = self.pricing_snapshot()
        prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"))
        cache_hit = _nonnegative_int(usage.get("prompt_cache_hit_tokens"))
        cache_miss = _nonnegative_int(usage.get("prompt_cache_miss_tokens"))
        accounted_prompt = cache_hit + cache_miss
        if accounted_prompt < prompt_tokens:
            cache_miss += prompt_tokens - accounted_prompt
        completion = _nonnegative_int(usage.get("completion_tokens"))
        cost = (
            Decimal(cache_hit) * Decimal(str(pricing["input_cache_hit_per_million_usd"]))
            + Decimal(cache_miss) * Decimal(str(pricing["input_cache_miss_per_million_usd"]))
            + Decimal(completion) * Decimal(str(pricing["output_per_million_usd"]))
        ) / Decimal(1_000_000)
        return float(cost)

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
        resolved_output_schema = _validated_output_schema(output_schema)
        aggregate_usage = _empty_usage()
        attempt_records: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            envelope: dict[str, Any] | None = None
            attempt_usage: dict[str, int] = {}
            finish_reason: str | None = None
            content_character_count = 0
            reasoning_character_count = 0
            body = self._request_body(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                retry=attempt > 1,
                output_schema=resolved_output_schema,
            )
            try:
                raw, header_request_id = self._send(body)
            except DeepSeekAPIError as exc:
                last_error = exc
                attempt_records.append(
                    _attempt_record(
                        attempt_index=attempt,
                        status="failed",
                        request_id=None,
                        model_id=self.model,
                        usage={},
                        usage_complete=False,
                        billable=False,
                        failure_type=type(exc).__name__,
                    )
                )
                if not exc.retryable or attempt >= self.max_attempts:
                    raise DeepSeekAPIError(
                        str(exc),
                        attempts=attempt,
                        usage=aggregate_usage,
                        retryable=exc.retryable,
                        status_code=exc.status_code,
                        attempt_records=tuple(attempt_records),
                    ) from exc
                self._sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
                continue
            try:
                envelope = _parse_envelope(raw)
                attempt_usage = _normalized_usage(envelope.get("usage"))
                _merge_usage(aggregate_usage, attempt_usage)
                returned_model = str(envelope.get("model") or "")
                request_id = header_request_id or _optional_text(envelope.get("id"))
                if returned_model != self.model:
                    attempt_records.append(
                        _attempt_record(
                            attempt_index=attempt,
                            status="failed",
                            request_id=request_id,
                            model_id=returned_model or self.model,
                            usage=attempt_usage,
                            usage_complete=_usage_complete(attempt_usage),
                            billable=True,
                            failure_type="model_identity_mismatch",
                        )
                    )
                    raise DeepSeekAPIError(
                        "DeepSeek returned a model identity different from the frozen request",
                        attempts=attempt,
                        usage=aggregate_usage,
                        attempt_records=tuple(attempt_records),
                    )
                choice = envelope["choices"][0]
                message = choice["message"]
                finish_reason = _optional_text(choice.get("finish_reason"))
                if self.strict_tool_calls and resolved_output_schema is not None:
                    content = _strict_tool_call_arguments(message)
                    if finish_reason != "tool_calls":
                        raise ValueError(
                            "strict structured response did not finish with tool_calls"
                        )
                else:
                    content = message["content"]
                content_character_count = len(content) if isinstance(content, str) else 0
                reasoning = message.get("reasoning_content")
                reasoning_character_count = len(reasoning) if isinstance(reasoning, str) else 0
                payload = _parse_json_content(content)
                if not isinstance(payload, dict):
                    raise TypeError("JSON output is not an object")
                if resolved_output_schema is not None:
                    _validate_schema_instance(payload, resolved_output_schema)
                attempt_records.append(
                    _attempt_record(
                        attempt_index=attempt,
                        status="succeeded",
                        request_id=request_id,
                        model_id=returned_model,
                        usage=attempt_usage,
                        usage_complete=_usage_complete(attempt_usage),
                        billable=True,
                        finish_reason=finish_reason,
                        content_character_count=content_character_count,
                        reasoning_character_count=reasoning_character_count,
                    )
                )
                return JsonCompletion(
                    payload=payload,
                    model=returned_model,
                    usage=dict(aggregate_usage),
                    request_id=request_id,
                    attempts=attempt,
                    system_fingerprint=_optional_text(envelope.get("system_fingerprint")),
                    finish_reason=finish_reason,
                    reasoning_content_present=isinstance(reasoning, str) and bool(reasoning),
                    reasoning_character_count=(len(reasoning) if isinstance(reasoning, str) else 0),
                    attempt_records=tuple(attempt_records),
                )
            except DeepSeekAPIError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                attempt_records.append(
                    _attempt_record(
                        attempt_index=attempt,
                        status="failed",
                        request_id=(
                            header_request_id
                            or (
                                _optional_text(envelope.get("id")) if envelope is not None else None
                            )
                        ),
                        model_id=(
                            str(envelope.get("model") or self.model)
                            if envelope is not None
                            else self.model
                        ),
                        usage=attempt_usage,
                        usage_complete=(_usage_complete(attempt_usage) if attempt_usage else False),
                        billable=envelope is not None,
                        failure_type="invalid_structured_output",
                        finish_reason=finish_reason,
                        content_character_count=content_character_count,
                        reasoning_character_count=reasoning_character_count,
                        parse_error_type=type(exc).__name__,
                    )
                )
        raise DeepSeekAPIError(
            f"DeepSeek returned invalid JSON output after {self.max_attempts} attempts",
            attempts=self.max_attempts,
            usage=aggregate_usage,
            attempt_records=tuple(attempt_records),
        ) from last_error

    def _request_body(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        retry: bool,
        output_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        retry_note = (
            "\nThe previous response was empty or invalid. Return the required JSON object."
            if retry
            else ""
        )
        schema_note = (
            "\nReturn a JSON object conforming to this schema: "
            + json.dumps(
                output_schema,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if output_schema is not None and not self.strict_tool_calls
            else ""
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + schema_note + retry_note},
            ],
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "stream": False,
            "max_tokens": int(max_tokens),
        }
        if output_schema is not None and self.strict_tool_calls:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "chemworld_decision",
                        "description": "Submit exactly one validated ChemWorld decision.",
                        "strict": True,
                        "parameters": _deepseek_strict_schema(output_schema),
                    },
                }
            ]
            # DeepSeek thinking mode rejects every explicit tool_choice value.
            # The system prompt requires this single tool; the response parser
            # fails closed unless the model calls it exactly once.
            if not self.thinking:
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "chemworld_decision"},
                }
        else:
            body["response_format"] = {"type": "json_object"}
        if self.thinking:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _send(self, body: dict[str, Any]) -> tuple[str, str | None]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ChemWorld-Formal/0.4",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return (
                    response.read().decode("utf-8"),
                    response.headers.get("x-request-id"),
                )
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
            raise DeepSeekAPIError(
                f"DeepSeek HTTP {exc.code}",
                retryable=retryable,
                status_code=int(exc.code),
            ) from exc
        except urllib.error.URLError as exc:
            raise DeepSeekAPIError("DeepSeek connection failed", retryable=True) from exc
        except TimeoutError as exc:
            raise DeepSeekAPIError("DeepSeek request timed out", retryable=True) from exc


def _validated_output_schema(
    output_schema: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if output_schema is None:
        return None
    try:
        normalized = json.loads(
            json.dumps(
                dict(output_schema),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_schema must be finite JSON") from exc
    if not isinstance(normalized, dict) or normalized.get("type") != "object":
        raise ValueError("output_schema must describe a JSON object")
    return normalized


def _parse_envelope(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("response envelope is not an object")
    return payload


def _parse_json_content(content: object) -> Any:
    if not isinstance(content, str) or not content.strip():
        raise json.JSONDecodeError("empty model content", "", 0)
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _strict_tool_call_arguments(message: object) -> str:
    if not isinstance(message, Mapping):
        raise TypeError("strict tool response message is not an object")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("strict tool response must contain exactly one tool call")
    call = calls[0]
    if not isinstance(call, Mapping) or call.get("type") != "function":
        raise TypeError("strict tool response is not a function call")
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("strict tool response lacks its function object")
    if function.get("name") != "chemworld_decision":
        raise ValueError("strict tool response used an unexpected function")
    arguments = function.get("arguments")
    if not isinstance(arguments, str) or not arguments.strip():
        raise ValueError("strict tool response lacks JSON arguments")
    return arguments


def _deepseek_strict_schema(value: Any) -> Any:
    """Remove JSON-Schema keywords unsupported by DeepSeek strict tools.

    The full schema remains authoritative in the local decision validator. This
    transport projection preserves object shape, required fields, enums and
    numeric bounds while omitting provider-unsupported string/array length hints.
    """

    if isinstance(value, Mapping):
        projected = {
            str(key): _deepseek_strict_schema(item)
            for key, item in value.items()
            if key not in {"minLength", "maxLength", "minItems", "maxItems"}
        }
        if value.get("type") == "string" and "pattern" not in projected:
            minimum = value.get("minLength")
            maximum = value.get("maxLength")
            if isinstance(minimum, int) and minimum >= 0:
                upper = str(maximum) if isinstance(maximum, int) else ""
                projected["pattern"] = f"^.{{{minimum},{upper}}}$"
            elif isinstance(maximum, int) and maximum >= 0:
                projected["pattern"] = f"^.{{0,{maximum}}}$"
        return projected
    if isinstance(value, list):
        return [_deepseek_strict_schema(item) for item in value]
    return value


def _validate_schema_instance(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> None:
    """Validate the JSON-Schema subset used by ChemWorld decision contracts."""

    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                continue
            try:
                _validate_schema_instance(value, alternative, path=path)
            except (TypeError, ValueError):
                continue
            break
        else:
            raise ValueError(f"{path} does not match any allowed schema")
        return

    raw_type = schema.get("type")
    allowed_types = (
        tuple(str(item) for item in raw_type)
        if isinstance(raw_type, list)
        else ((str(raw_type),) if raw_type is not None else ())
    )
    if allowed_types and not any(_schema_type_matches(value, item) for item in allowed_types):
        raise TypeError(f"{path} has the wrong JSON type")

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ValueError(f"{path} is outside its enum")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not match its constant")

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        property_map = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        if isinstance(required, list):
            missing = [str(key) for key in required if key not in value]
            if missing:
                raise ValueError(f"{path} lacks required fields")
        if schema.get("additionalProperties") is False and any(
            key not in property_map for key in value
        ):
            raise ValueError(f"{path} contains undeclared fields")
        for key, item in value.items():
            child_schema = property_map.get(key)
            if isinstance(child_schema, Mapping):
                _validate_schema_instance(item, child_schema, path=f"{path}.{key}")
    elif isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            raise ValueError(f"{path} contains too few items")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            raise ValueError(f"{path} contains too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_instance(item, item_schema, path=f"{path}[{index}]")
    elif isinstance(value, str):
        minimum_length = schema.get("minLength")
        maximum_length = schema.get("maxLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise ValueError(f"{path} is shorter than allowed")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            raise ValueError(f"{path} is longer than allowed")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise ValueError(f"{path} does not match its pattern")
    elif isinstance(value, int | float) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, int | float) and value < minimum:
            raise ValueError(f"{path} is below its minimum")
        if isinstance(maximum, int | float) and value > maximum:
            raise ValueError(f"{path} is above its maximum")
        if isinstance(exclusive_minimum, int | float) and value <= exclusive_minimum:
            raise ValueError(f"{path} is not above its exclusive minimum")
        if isinstance(exclusive_maximum, int | float) and value >= exclusive_maximum:
            raise ValueError(f"{path} is not below its exclusive maximum")


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return False


def _merge_usage(total: dict[str, int], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        value = usage.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total[key] += value


def _normalized_usage(usage: object) -> dict[str, int]:
    normalized = _empty_usage()
    if isinstance(usage, dict):
        for key in normalized:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized[key] = value
    return normalized


def _usage_complete(usage: dict[str, int]) -> bool:
    return (
        usage["prompt_tokens"] > 0
        and usage["total_tokens"] > 0
        and (
            usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
            == usage["prompt_tokens"]
        )
        and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    )


def _attempt_record(
    *,
    attempt_index: int,
    status: str,
    request_id: str | None,
    model_id: str,
    usage: dict[str, int],
    usage_complete: bool,
    billable: bool,
    failure_type: str | None = None,
    finish_reason: str | None = None,
    content_character_count: int = 0,
    reasoning_character_count: int = 0,
    parse_error_type: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_index": int(attempt_index),
        "status": status,
        "request_id": request_id,
        "model_id": model_id,
        "usage": dict(usage),
        "usage_complete": bool(usage_complete),
        "billable": bool(billable),
        "usage_source": "provider_response" if billable else "unavailable",
        "finish_reason": finish_reason,
        "content_character_count": int(content_character_count),
        "reasoning_character_count": int(reasoning_character_count),
    }
    if failure_type:
        record["failure_type"] = failure_type
    if parse_error_type:
        record["parse_error_type"] = parse_error_type
    return record


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }


def _nonnegative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_SOURCE",
    "PRICING_ACCESS_DATE",
    "PRICING_SOURCE",
    "SUPPORTED_MODELS",
    "DeepSeekAPIError",
    "DeepSeekClient",
    "DeepSeekPricing",
    "JsonCompletion",
    "JsonPlannerClient",
    "ReasoningEffort",
]
