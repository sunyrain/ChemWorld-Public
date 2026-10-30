"""Valve coefficient units, choking boundaries and sourced opening curves."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite, sqrt
from typing import Any

from scipy.interpolate import BSpline, make_interp_spline

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.fluids_devices import (
    IEC_REFERENCE_DENSITY,
    LiquidControlValveSpec,
    valve_critical_pressure_factor,
)
from chemworld.physchem.gas_control_valves import GasControlValveSpec
from chemworld.physchem.tabulated_functions import TabulatedScalar

KV_TO_CV = 1.1560992283536566


def convert_valve_coefficient(value: float, *, from_scale: str, to_scale: str) -> float:
    """Kv [m³/h water at 1 bar], Cv [US gal/min at 1 psi], Av [m²]."""
    if not isfinite(value) or value < 0:
        raise ValueError("valve coefficient must be finite and nonnegative")
    kv_to_scale = {"Kv": 1.0, "Cv": KV_TO_CV, "Av": sqrt(IEC_REFERENCE_DENSITY / 1e5) / 3600}
    if from_scale not in kv_to_scale or to_scale not in kv_to_scale:
        raise ValueError("valve coefficient scale must be Kv, Cv or Av")
    return value / kv_to_scale[from_scale] * kv_to_scale[to_scale]


def valve_cavitation_index(
    *,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
    saturation_pressure_Pa: float,
) -> float:
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    if (
        not isfinite(saturation_pressure_Pa)
        or not 0 <= saturation_pressure_Pa < upstream_pressure_Pa
    ):
        raise ValueError("saturation pressure must be nonnegative and below upstream")
    if downstream_pressure_Pa >= upstream_pressure_Pa:
        raise ValueError("cavitation index needs a positive pressure difference")
    return (upstream_pressure_Pa - saturation_pressure_Pa) / (
        upstream_pressure_Pa - downstream_pressure_Pa
    )


def gas_valve_choked(
    *,
    pressure_drop_ratio: float,
    heat_capacity_ratio: float,
    pressure_ratio_factor: float,
) -> bool:
    if not isfinite(pressure_drop_ratio) or not 0 <= pressure_drop_ratio <= 1:
        raise ValueError("gas pressure-drop ratio must lie in [0, 1]")
    _positive(heat_capacity_ratio, "heat capacity ratio")
    _positive(pressure_ratio_factor, "bare or installed pressure ratio factor")
    return pressure_drop_ratio >= heat_capacity_ratio / 1.4 * pressure_ratio_factor


def liquid_valve_choked(
    *,
    pressure_drop_Pa: float,
    upstream_pressure_Pa: float,
    saturation_pressure_Pa: float,
    critical_pressure_factor: float,
    effective_recovery_factor: float,
) -> bool:
    """Effective recovery means bare FL or the explicitly supplied installed FLP/FP."""
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    if not isfinite(pressure_drop_Pa) or not 0 <= pressure_drop_Pa <= upstream_pressure_Pa:
        raise ValueError("liquid pressure drop must lie between zero and upstream pressure")
    if not isfinite(saturation_pressure_Pa) or saturation_pressure_Pa < 0:
        raise ValueError("saturation pressure must be finite and nonnegative")
    _positive(critical_pressure_factor, "critical pressure factor")
    _positive(effective_recovery_factor, "effective recovery factor")
    driving = upstream_pressure_Pa - critical_pressure_factor * saturation_pressure_Pa
    if driving <= 0:
        raise ValueError("inlet cannot reach a declared liquid choking boundary")
    return pressure_drop_Pa >= effective_recovery_factor**2 * driving


def gas_valve_choke_pressure(
    *,
    pressure_ratio_factor: float,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float | None = None,
    downstream_pressure_Pa: float | None = None,
) -> float:
    if (upstream_pressure_Pa is None) == (downstream_pressure_Pa is None):
        raise ValueError("declare exactly one upstream or downstream absolute pressure")
    _positive(heat_capacity_ratio, "heat capacity ratio")
    _positive(pressure_ratio_factor, "bare or installed pressure ratio factor")
    fraction = heat_capacity_ratio / 1.4 * pressure_ratio_factor
    if fraction >= 1:
        raise ValueError(
            "no positive outlet choking pressure for this gamma and pressure ratio factor"
        )
    if upstream_pressure_Pa is not None:
        _positive(upstream_pressure_Pa, "upstream absolute pressure")
        return upstream_pressure_Pa * (1 - fraction)
    if downstream_pressure_Pa is None:
        raise AssertionError("invariant violated: downstream_pressure_Pa is None")
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    return downstream_pressure_Pa / (1 - fraction)


def liquid_valve_choke_pressure(
    *,
    saturation_pressure_Pa: float,
    critical_pressure_Pa: float,
    effective_recovery_factor: float,
    upstream_pressure_Pa: float | None = None,
    downstream_pressure_Pa: float | None = None,
) -> float:
    if (upstream_pressure_Pa is None) == (downstream_pressure_Pa is None):
        raise ValueError("declare exactly one upstream or downstream absolute pressure")
    if not isfinite(effective_recovery_factor) or not 0 < effective_recovery_factor <= 1:
        raise ValueError("effective recovery factor must lie in (0, 1]")
    ff = valve_critical_pressure_factor(saturation_pressure_Pa, critical_pressure_Pa)
    squared = effective_recovery_factor**2
    if upstream_pressure_Pa is not None:
        if (
            not isfinite(upstream_pressure_Pa)
            or upstream_pressure_Pa <= ff * saturation_pressure_Pa
        ):
            raise ValueError("inlet pressure is too low for liquid choking")
        return upstream_pressure_Pa * (1 - squared) + squared * ff * saturation_pressure_Pa
    if downstream_pressure_Pa is None:
        raise AssertionError("invariant violated: downstream_pressure_Pa is None")
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    if squared == 1:
        raise ValueError("upstream pressure is indeterminate at unit recovery")
    result = (downstream_pressure_Pa - squared * ff * saturation_pressure_Pa) / (1 - squared)
    if result <= downstream_pressure_Pa:
        raise ValueError("pressure pair cannot reach liquid choking")
    return result


@dataclass(frozen=True)
class ValveOpeningCurve:
    """Inherent normalized Cv curve; quick/equal curves require licensed source data.

    Cubic interpolation reproduces a declared source curve, not every valve of
    that nominal class. Manufacturer/source identity stays in ScientificTable.
    """

    characteristic: str
    source_curve: TabulatedScalar | None = None
    interpolation: str = "cubic"
    _spline: BSpline | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.characteristic not in {"linear", "quick_opening", "equal_percentage"}:
            raise ValueError("unknown valve characteristic")
        if self.interpolation not in {"linear", "cubic"}:
            raise ValueError("opening interpolation must be linear or cubic")
        if self.characteristic == "linear":
            if self.source_curve is not None:
                raise ValueError("linear characteristic has an exact equation, not a source table")
            return
        curve = self.source_curve
        if curve is None:
            raise ValueError("quick/equal characteristic requires an explicit sourced curve")
        if (
            len(curve.axes) != 1
            or curve.axes[0][0] != "opening"
            or (curve.axes[0][2] != "dimensionless" or curve.output_unit != "dimensionless")
        ):
            raise ValueError("opening and normalized coefficient must be dimensionless")
        if (
            curve.domain["opening"] != (0, 1)
            or not 0 <= curve.value_range[0] <= curve.value_range[1] <= 1
        ):
            raise ValueError("opening curve must cover [0, 1] with normalized coefficients")
        if self.interpolation == "cubic":
            column = curve.axes[0][1]
            rows = {row.row_id: row for row in curve.table.rows}
            points = sorted(
                (
                    float(rows[key].values[column].to("dimensionless").value),
                    float(rows[key].values[curve.output_column].to("dimensionless").value),
                )
                for key in curve.row_ids
            )
            if len(points) < 4:
                raise ValueError("cubic opening curve needs at least four source points")
            spline = make_interp_spline([p[0] for p in points], [p[1] for p in points], k=3)
            object.__setattr__(self, "_spline", spline)

    def value(self, opening: float) -> float:
        if not isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("opening must lie in [0, 1]")
        if self.characteristic == "linear":
            return opening
        if self.source_curve is None:
            raise AssertionError("invariant violated: self.source_curve is not None")
        result = (
            float(self._spline(opening))
            if self._spline is not None
            else float(self.source_curve.value({"opening": opening}))
        )
        if not -1e-12 <= result <= 1 + 1e-12:
            raise ValueError("declared source interpolation leaves physical normalized range")
        return min(1.0, max(0.0, result))

    def apply(
        self, full_open_spec: LiquidControlValveSpec | GasControlValveSpec, opening: float
    ) -> LiquidControlValveSpec | GasControlValveSpec:
        """Build a spec from the full-open rating, never cumulatively scale an old setting."""
        fraction = self.value(opening)
        return replace(
            full_open_spec,
            is_open=bool(fraction),
            kv_m3_h=full_open_spec.kv_m3_h * fraction if fraction else full_open_spec.kv_m3_h,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "interpolation": self.interpolation,
            "source_curve": self.source_curve.to_dict() if self.source_curve is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValveOpeningCurve:
        data = dict(payload)
        if data.get("source_curve") is not None:
            data["source_curve"] = TabulatedScalar.from_dict(data["source_curve"])
        return cls(**data)
