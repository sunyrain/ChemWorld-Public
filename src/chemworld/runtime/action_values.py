"""Shared numeric decoding for runtime operation services.

Schema and operation validation own admissibility. These helpers preserve the
existing scalar/array decoding and the closed bounds used by thermal and flow
services; they do not add clipping or a second action-validation policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def bounded_action_float(
    action: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = action_float(action, key, default)
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be finite and in [{minimum}, {maximum}]")
    return value
