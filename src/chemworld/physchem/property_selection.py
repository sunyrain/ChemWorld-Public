"""Explicit policies over existing correlations; no property cache or source ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, TypeVar, cast

from chemworld.foundation.units import convert_value
from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
from chemworld.physchem.specs import PropertyCorrelation

_Correlation = TypeVar("_Correlation", bound=PropertyCorrelation | PiecewiseHeatCapacity)


@dataclass(frozen=True)
class PropertySelectionPolicy:
    mode: str = "declared_order"
    correlation_id: str | None = None
    source_priority: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(cast(object, self.source_priority), str):
            raise TypeError("source_priority must be a sequence of source IDs, not a string")
        priorities = tuple(self.source_priority)
        if self.mode not in {"declared_order", "correlation", "source_priority"}:
            raise ValueError("unknown property selection mode")
        if any(not isinstance(s, str) or not s.strip() for s in priorities):
            raise ValueError("source priority needs nonempty source IDs")
        if len(set(priorities)) != len(priorities):
            raise ValueError("duplicate source priorities")
        if self.mode == "correlation":
            if not isinstance(self.correlation_id, str) or not self.correlation_id.strip():
                raise ValueError("explicit selection needs a correlation_id")
        elif self.correlation_id is not None:
            raise ValueError("correlation_id requires explicit correlation selection")
        if (self.mode == "source_priority") != bool(priorities):
            raise ValueError("source_priority mode requires an explicit nonempty priority list")
        object.__setattr__(self, "source_priority", priorities)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "correlation_id": self.correlation_id,
            "source_priority": list(self.source_priority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PropertySelectionPolicy:
        if set(payload) - {"mode", "correlation_id", "source_priority"}:
            raise ValueError("unknown property selection fields")
        return cls(**payload)


@dataclass(frozen=True)
class CorrelationCandidate:
    correlation_id: str
    equation_id: str
    property_id: str
    source_id: str | None
    source_note: str
    output_unit: str
    validity_ranges: dict[str, tuple[float, float]]
    matches_policy: bool
    status: str
    diagnostics: tuple[str, ...]
    selected: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PropertySelectionReport:
    property_id: str
    policy: PropertySelectionPolicy
    temperature_K: float
    pressure_Pa: float | None
    candidates: tuple[CorrelationCandidate, ...]
    selected_id: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "property_id": self.property_id,
            "policy": self.policy.to_dict(),
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_id": self.selected_id,
            "reason": self.reason,
            "scope": "Declaration and input-domain selection; formulas are not evaluated",
        }


class PropertySelectionError(ValueError):
    def __init__(self, report: PropertySelectionReport) -> None:
        self.report = report
        diagnostics = "; ".join(d for c in report.candidates for d in c.diagnostics)
        super().__init__(
            f"No correlation selected for property {report.property_id!r}: "
            f"{report.reason}" + (f"; {diagnostics}" if diagnostics else "")
        )


def correlation_source_id(correlation: PropertyCorrelation | PiecewiseHeatCapacity) -> str | None:
    source = correlation.metadata.get("source", {})
    value = source.get("source_id") if isinstance(source, Mapping) else None
    return value if isinstance(value, str) and value else None


def inspect_property_selection(
    correlations: Sequence[PropertyCorrelation | PiecewiseHeatCapacity], *, property_id: str,
    temperature_K: float, pressure_Pa: float | None = None,
    policy: PropertySelectionPolicy | None = None,
) -> PropertySelectionReport:
    """Explain selection without trying formulas and concealing failed alternatives.

    Among candidates allowed by the policy, prefer the first declared-domain
    match, otherwise the first complete-input candidate. The evaluator retains
    ownership of raise/warn/ignore and the equation's hard mathematical domain.
    A source-priority list is an explicit shortlist: unlisted sources do not win.
    """
    from dataclasses import replace

    policy = policy or PropertySelectionPolicy()
    if not isinstance(policy, PropertySelectionPolicy):
        raise TypeError("selection requires a PropertySelectionPolicy")
    if (
        not isfinite(temperature_K)
        or temperature_K <= 0
        or (pressure_Pa is not None and (not isfinite(pressure_Pa) or pressure_Pa <= 0))
    ):
        raise ValueError("property selection needs finite positive temperature_K/pressure_Pa")
    candidates = []
    for curve in correlations:
        if curve.property_id != property_id:
            continue
        source = correlation_source_id(curve)
        matches = (
            policy.mode == "declared_order"
            or (policy.mode == "correlation" and curve.correlation_id == policy.correlation_id)
            or (policy.mode == "source_priority" and source in policy.source_priority)
        )
        issues, outside, missing = [], False, False
        if "pressure" in curve.input_units and pressure_Pa is None:
            missing = True
            issues.append("pressure_Pa is required by this correlation")
        for field, value, unit in (
            ("temperature", temperature_K, "K"),
            ("pressure", pressure_Pa, "Pa"),
        ):
            if value is None:
                continue
            converted = convert_value(value, unit, curve.input_units.get(field, unit))
            bounds = curve.validity_ranges.get(field)
            if bounds is not None and not bounds[0] <= converted <= bounds[1]:
                outside = True
                issues.append(
                    f"{field}={converted:g} outside validity range "
                    f"[{bounds[0]:g}, {bounds[1]:g}] for {curve.correlation_id}"
                )
        if policy.mode == "source_priority" and source is None:
            issues.append("no declared source_id for source-priority selection")
        candidates.append(
            CorrelationCandidate(
                curve.correlation_id,
                curve.equation_id,
                curve.property_id,
                source,
                curve.source_note,
                curve.output_unit,
                dict(curve.validity_ranges),
                matches,
                "missing_input"
                if missing
                else "outside_declared_range"
                if outside
                else "within_declared_range",
                tuple(issues),
            )
        )
    allowed = [
        i
        for i, row in enumerate(candidates)
        if row.matches_policy and row.status != "missing_input"
    ]
    if policy.mode == "source_priority":
        allowed.sort(key=lambda i: policy.source_priority.index(cast(str, candidates[i].source_id)))
    within = [i for i in allowed if candidates[i].status == "within_declared_range"]
    chosen = next(iter(within or allowed), None)
    if chosen is not None:
        candidates[chosen] = replace(candidates[chosen], selected=True)
        selected = candidates[chosen]
        reason = policy.mode + ":" + selected.status
        identity = selected.correlation_id
    else:
        identity = None
        reason = (
            "missing_property"
            if not candidates
            else "no_matching_policy"
            if not any(c.matches_policy for c in candidates)
            else "missing_required_input"
        )
    return PropertySelectionReport(
        property_id, policy, temperature_K, pressure_Pa, tuple(candidates), identity, reason
    )


def select_property_correlation(
    correlations: Sequence[_Correlation], *, property_id: str, temperature_K: float,
    pressure_Pa: float | None = None, policy: PropertySelectionPolicy | None = None,
) -> _Correlation:
    report = inspect_property_selection(
        correlations,
        property_id=property_id,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        policy=policy,
    )
    if report.selected_id is None:
        raise PropertySelectionError(report)
    return next(c for c in correlations if c.correlation_id == report.selected_id)
