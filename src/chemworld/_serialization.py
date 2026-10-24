"""Strict JSON encoding shared by physics, providers, and predictive evaluation.

Frozen provenance retains its source-bound implementation. This low-level module
avoids importing the evaluation package from physics or provider clients.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(payload: Any, *, ensure_ascii: bool = False) -> bytes:
    """Encode sorted compact JSON, rejecting nonfinite numbers.

    Most artifacts retain Unicode; older electrochemical digests escape it.
    Callers must select the original encoding when preserving an existing hash.
    """

    return json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any, *, ensure_ascii: bool = False) -> str:
    """Return the SHA-256 of strict canonical JSON with the selected encoding."""

    return hashlib.sha256(canonical_json_bytes(payload, ensure_ascii=ensure_ascii)).hexdigest()
