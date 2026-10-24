"""Descriptive statistics for audited static optimization campaigns."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class ScoreSummary(TypedDict):
    count: int
    mean: float | None
    median: float | None
    sample_standard_deviation: float | None
    minimum: float | None
    maximum: float | None


def score_summary(values: Sequence[float]) -> ScoreSummary:
    """Describe every supplied value, retaining the sample standard deviation."""

    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "sample_standard_deviation": (
            statistics.stdev(normalized) if len(normalized) > 1 else None
        ),
        "minimum": min(normalized) if normalized else None,
        "maximum": max(normalized) if normalized else None,
    }


def mean_mapping(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average matching metric fields without dropping rows or missing fields."""

    if not rows:
        return {}
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("metric rows do not share the same fields")
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in sorted(keys)
    }
