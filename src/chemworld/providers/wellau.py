"""Auditable WellAU client for OpenAI-compatible structured chat completions."""

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
    """Call the discovered WellAU Codex model without a silent model fallback."""

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

    def pricing_snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "chemworld-provider-pricing-unknown-0.1",
            "provider": "WellAU",
            "currency": "USD",
            "requested_model_id": self.model,
            "base_url": self.base_url,
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
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + retry_note},
            ],
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "chemworld_decision",
                        "strict": True,
                        "schema": output_schema,
                    },
                }
                if output_schema is not None
                else {"type": "json_object"}
            ),
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
            "max_tokens": int(max_tokens),
        }

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
            raise WellAUAPIError(
                f"WellAU HTTP {exc.code}",
                retryable=retryable,
                status_code=int(exc.code),
            ) from exc
        except urllib.error.URLError as exc:
            raise WellAUAPIError("WellAU connection failed", retryable=True) from exc
        except TimeoutError as exc:
            raise WellAUAPIError("WellAU request timed out", retryable=True) from exc


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
