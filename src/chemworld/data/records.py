"""Read trajectory values through their canonical outcome layers.

Historical aliases are interpreted here, without mutating the input or
materializing duplicate fields. The v0.3 format never falls back to an alias.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.data.schema import TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION

_MISSING = object()
_OUTCOME_PATHS = {
    "observation": ("environment_outcome", "observation"),
    "reward": ("evaluation_outcome", "online_transition_reward"),
    "agent_view": ("agent_visible_observation", "views"),
    "leaderboard_score": ("evaluation_outcome", "leaderboard_score"),
}
_LEGACY_VERSIONS = {None, "chemworld-trajectory-0.1", "chemworld-trajectory-0.2"}


def trajectory_value(
    record: Mapping[str, Any],
    field: str,
    default: Any = _MISSING,
) -> Any:
    """Read one migrated field with dict-style required/default semantics.

    Unversioned partial records remain supported for existing analysis callers.
    v0.1 uses its original top-level fields. v0.2 prefers the canonical layers,
    with alias fallback for historical partial records. Conflicting v0.2
    observation, reward, and score aliases are rejected even by partial readers.
    Full trajectory validation still checks the required record structure.

    In v0.3, a task_id equal to env_id denotes the logger's ad-hoc environment
    fallback, so there is no registered benchmark task to restore during replay.
    """

    if field not in _OUTCOME_PATHS and field != "benchmark_task_id":
        raise KeyError(f"not a migrated trajectory field: {field}")
    version = record.get("schema_version")
    canonical_only = version == TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION
    if not canonical_only and version not in _LEGACY_VERSIONS:
        raise ValueError(f"unsupported trajectory schema_version={version!r}")

    value = _MISSING
    if field == "benchmark_task_id":
        if canonical_only:
            value = record.get("task_id", _MISSING)
            if value is not _MISSING and value == record.get("env_id"):
                value = None
        else:
            value = record.get(field, _MISSING)
    else:
        if version != "chemworld-trajectory-0.1":
            layer, key = _OUTCOME_PATHS[field]
            if layer in record:
                outcome = record[layer]
                if not isinstance(outcome, Mapping):
                    raise ValueError(f"{layer} must be an object")
                value = outcome.get(key, _MISSING)
                if (
                    version == "chemworld-trajectory-0.2"
                    and field != "agent_view"
                    and value is not _MISSING
                    and field in record
                    and record[field] != value
                ):
                    raise ValueError(
                        f"trajectory v0.2 {field} alias does not match canonical outcome"
                    )
        if value is _MISSING and not canonical_only:
            value = record.get(field, _MISSING)

    if value is not _MISSING:
        return value
    if default is not _MISSING:
        return default
    raise KeyError(field)
