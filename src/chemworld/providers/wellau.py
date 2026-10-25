"""Auditable WellAU client for OpenAI-compatible Responses requests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Literal

from chemworld._serialization import canonical_json_sha256 as _canonical_sha256
from chemworld.providers.deepseek import DeepSeekAPIError, DeepSeekClient

DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.6-sol"
MODEL_ACCESS_DATE = "2026-07-25"
MODEL_SOURCE = "https://api.wellau.com/v1/models"
SUPPORTED_MODELS = (DEFAULT_MODEL,)
ReasoningEffort = Literal["medium", "high"]


class WellAUAPIError(DeepSeekAPIError):
    """Redacted WellAU transport, identity, or structured-output failure."""


class WellAUClient(DeepSeekClient):
    """Call the discovered WellAU Codex model through one Responses request."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
        reasoning_effort: ReasoningEffort = "high",
        max_attempts: int = 3,
        retry_backoff_s: float = 0.25,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        resolved_key = (api_key or os.environ.get("WELLAU_API_KEY", "")).strip()
        if not resolved_key:
            raise WellAUAPIError(
                "WELLAU_API_KEY is not set. Keep the key in an environment variable."
            )
        resolved_model = model or os.environ.get("WELLAU_MODEL") or DEFAULT_MODEL
        if resolved_model not in SUPPORTED_MODELS:
            raise WellAUAPIError(
                f"WellAU model {resolved_model!r} was not frozen by the discovery preflight"
            )
        if reasoning_effort not in {"medium", "high"}:
            raise ValueError(
                "WellAU gpt-5.6-sol requires reasoning_effort=medium or high"
            )
        kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "base_url": base_url or os.environ.get("WELLAU_BASE_URL") or DEFAULT_BASE_URL,
            "model": resolved_model,
            "timeout_s": timeout_s,
            "thinking": True,
            # DeepSeekClient validates its own provider-specific high/max enum.
            # WellAU's overridden request body uses the requested effort below.
            "reasoning_effort": "high",
            "max_attempts": max_attempts,
            "retry_backoff_s": retry_backoff_s,
        }
        if sleep is not None:
            kwargs["sleep"] = sleep
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort
        self.wire_api = "responses"

    def pricing_snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "chemworld-provider-pricing-unknown-0.1",
            "provider": "WellAU",
            "currency": "USD",
            "requested_model_id": self.model,
            "base_url": self.base_url,
            "wire_api": self.wire_api,
            "model_source": MODEL_SOURCE,
            "model_access_date": MODEL_ACCESS_DATE,
            "accounting_complete": False,
            "pricing_unavailable_reason": (
                "The authenticated model catalog and key file supplied no verifiable pricing."
            ),
        }
        payload["pricing_version_sha256"] = _canonical_sha256(payload)
        return payload

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
        text_format: dict[str, Any] = (
            {
                "type": "json_schema",
                "name": "chemworld_decision",
                "strict": True,
                "schema": dict(output_schema),
            }
            if output_schema is not None
            else {"type": "json_object"}
        )
        return {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt + retry_note,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"format": text_format},
            "max_output_tokens": int(max_tokens),
        }

    def _send(self, body: dict[str, Any]) -> tuple[str, str | None]:
        request = urllib.request.Request(
            f"{self.base_url}/responses",
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
                raw = response.read().decode("utf-8")
                return (
                    _responses_to_chat_envelope(raw, requested_model=self.model),
                    response.headers.get("x-request-id"),
                )
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
            raise WellAUAPIError(
                f"WellAU HTTP {exc.code}",
                retryable=retryable,
                status_code=int(exc.code),
            ) from exc
        except urllib.error.URLError as exc:
            raise WellAUAPIError("WellAU connection failed", retryable=True) from exc
        except TimeoutError as exc:
            raise WellAUAPIError("WellAU request timed out", retryable=True) from exc


def _responses_to_chat_envelope(raw: str, *, requested_model: str) -> str:
    """Adapt a Responses envelope to the audited parser shared with DeepSeek."""

    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise WellAUAPIError("WellAU Responses envelope is not an object")
    output_text = payload.get("output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        fragments: list[str] = []
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping) or item.get("type") != "message":
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, Mapping) or part.get("type") != "output_text":
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
        output_text = "".join(fragments)
    if not output_text.strip():
        raise WellAUAPIError("WellAU Responses output contains no output_text")

    usage = payload.get("usage")
    input_tokens = _response_usage_int(usage, "input_tokens")
    output_tokens = _response_usage_int(usage, "output_tokens")
    total_tokens = _response_usage_int(usage, "total_tokens")
    cached_tokens = 0
    if isinstance(usage, Mapping):
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, Mapping):
            cached_tokens = _nonnegative_int(input_details.get("cached_tokens"))
    cached_tokens = min(cached_tokens, input_tokens)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    status = str(payload.get("status") or "completed")
    incomplete = payload.get("incomplete_details")
    incomplete_reason = (
        str(incomplete.get("reason"))
        if isinstance(incomplete, Mapping) and incomplete.get("reason") is not None
        else None
    )
    envelope = {
        "id": payload.get("id"),
        "model": payload.get("model") or requested_model,
        "choices": [
            {
                "message": {"content": output_text},
                "finish_reason": "stop" if status == "completed" else incomplete_reason or status,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "prompt_cache_hit_tokens": cached_tokens,
            "prompt_cache_miss_tokens": max(input_tokens - cached_tokens, 0),
        },
    }
    return json.dumps(envelope, ensure_ascii=False)


def _response_usage_int(usage: object, key: str) -> int:
    return _nonnegative_int(usage.get(key)) if isinstance(usage, Mapping) else 0


def _nonnegative_int(value: object) -> int:
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_ACCESS_DATE",
    "MODEL_SOURCE",
    "SUPPORTED_MODELS",
    "ReasoningEffort",
    "WellAUAPIError",
    "WellAUClient",
]
