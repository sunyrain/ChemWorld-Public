"""Sourced screen and grill losses, referenced to the upstream approach velocity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, isfinite, radians
from typing import Any

from scipy.interpolate import BSpline, make_interp_spline

from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState

FILTER_METHODS = (
    "round_screen", "open_mesh", "square_screen", "square_grill", "round_grill",
)
MESH_PATTERNS = {
    "round bar screen": (.95, .2), "diamond pattern wire": (.67, 1.3),
    "knotted net": (.70, 4.9), "knotless net": (.72, 2.1),
}


@dataclass(frozen=True)
class FilterCurve:
    """One explicit source table, interpolation order and endpoint policy.

All coordinates are dimensionless; the ``angle_degrees`` name declares the
numerical angle convention. Source uncertainties are not prediction intervals.
"""
    table_function: TabulatedScalar
    interpolation: str = "linear"
    endpoint_policy: str = "raise"
    _spline: BSpline | None = field(init=False, default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        table = self.table_function
        if (len(table.axes) != 1 or table.axes[0][2] != "dimensionless"
                or table.output_unit != "dimensionless" or table.value_range[0] < 0):
            raise ValueError("filter curve needs one dimensionless axis and nonnegative output")
        if self.interpolation not in {"linear", "quadratic"}:
            raise ValueError("filter interpolation must be linear or quadratic")
        if self.endpoint_policy not in {"raise", "clip"}:
            raise ValueError("filter endpoint policy must be raise or clip")
        if self.interpolation == "quadratic":
            rows = {r.row_id: r for r in table.table.rows}
            xcol = table.axes[0][1]
            points = sorted((rows[k].values[xcol].to("dimensionless").value,
                             rows[k].values[table.output_column].to("dimensionless").value)
                            for k in table.row_ids)
            if len(points) < 3:
                raise ValueError("quadratic filter curve needs at least three points")
            object.__setattr__(self, "_spline", make_interp_spline(
                [p[0] for p in points], [p[1] for p in points], k=2))

    @property
    def input_name(self) -> str:
        return self.table_function.axes[0][0]

    def value(self, coordinate: float) -> float:
        if isinstance(coordinate, bool) or not isfinite(coordinate):
            raise ValueError("filter coordinate must be finite")
        lo, hi = self.table_function.domain[self.input_name]
        if self.endpoint_policy == "clip":
            coordinate = min(hi, max(lo, coordinate))
        elif not lo <= coordinate <= hi:
            raise ValueError("filter coordinate outside source domain")
        value = (float(self._spline(coordinate)) if self._spline is not None else
                 float(self.table_function.value({self.input_name: coordinate})))
        if not isfinite(value) or value < 0:
            raise ValueError("source interpolation predicts nonphysical filter loss")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"table_function": self.table_function.to_dict(),
                "interpolation": self.interpolation, "endpoint_policy": self.endpoint_policy}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilterCurve:
        return cls(TabulatedScalar.from_dict(payload["table_function"]),
                   payload.get("interpolation", "linear"), payload.get("endpoint_policy", "raise"))


def _curve_value(curve: FilterCurve | None, coordinate: float, name: str) -> float:
    if curve is None or curve.input_name != name:
        raise ValueError(f"explicit sourced filter curve on {name} required")
    return curve.value(coordinate)


def filter_loss_coefficient(
    method: str, *, open_fraction: float, reynolds: float | None = None,
    inclination_degrees: float = 0., mesh_pattern: str = "diamond pattern wire",
    primary_curve: FilterCurve | None = None, angle_curve: FilterCurve | None = None,
    thickness_m: float | None = None, hydraulic_diameter_m: float | None = None,
    darcy_friction_factor: float | None = None,
) -> float:
    """Return K on approach velocity; supplied data are never silently ignored."""
    if method not in FILTER_METHODS:
        raise ValueError("unknown filter loss method")
    alpha, angle = open_fraction, inclination_degrees
    if isinstance(alpha, bool) or not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("filter open fraction must lie in (0,1]")
    if not isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError("filter inclination must lie in [0,90] degrees")
    if reynolds is not None and (not isfinite(reynolds) or reynolds <= 0):
        raise ValueError("filter Reynolds number must be positive")
    if method not in {"round_screen", "open_mesh"} and (angle or angle_curve is not None):
        raise ValueError("selected filter law does not consume inclination")
    if method not in {"round_screen", "open_mesh"} and reynolds is not None:
        raise ValueError("selected filter law does not consume Reynolds number")
    geometry = (thickness_m, hydraulic_diameter_m, darcy_friction_factor)
    if any(v is not None for v in geometry):
        if method not in {"round_grill", "square_grill"}:
            raise ValueError("only grill laws consume passage-friction geometry")
        if any(v is None or isinstance(v, bool) or not isfinite(v) or v <= 0 for v in geometry):
            raise ValueError("grill thickness, hydraulic diameter and Darcy factor needed together")
    if method in {"open_mesh", "square_grill"} and primary_curve is not None:
        raise ValueError("this filter method has a direct equation, not a primary source curve")
    if method == "open_mesh":
        if alpha < .85 or reynolds is None or reynolds <= 500:
            raise ValueError("open mesh requires open fraction >=.85 and Reynolds >500")
        if mesh_pattern not in MESH_PATTERNS:
            raise ValueError("unknown mesh pattern")
        linear, quadratic = MESH_PATTERNS[mesh_pattern]
        value = linear*(1-alpha)+quadratic*(1-alpha)**2
    elif method == "round_screen":
        if not .05 <= alpha <= .8 or reynolds is None:
            raise ValueError("round screen requires alpha in [.05,.8] and Reynolds number")
        value = _curve_value(primary_curve, reynolds, "reynolds")*(1-alpha*alpha)/(alpha*alpha)
    elif method == "square_screen":
        value = _curve_value(primary_curve, alpha, "open_fraction")
    elif method == "square_grill":
        value = (.5*(1-alpha)+1-alpha*alpha)/(alpha*alpha)
    else:
        if not .3 <= alpha <= .7:
            raise ValueError("round grill requires open fraction in [.3,.7]")
        value = _curve_value(primary_curve, alpha, "open_fraction")
    if method in {"round_screen", "open_mesh"}:
        if angle < 45 or (method == "round_screen" and angle == 45):
            value *= cos(radians(angle))**2
        else:
            value *= _curve_value(angle_curve, angle, "angle_degrees")
    if thickness_m is not None:
        if hydraulic_diameter_m is None or darcy_friction_factor is None:
            raise AssertionError(
                "invariant violated: hydraulic_diameter_m or darcy_friction_factor is None"
            )
        if thickness_m > 50*hydraulic_diameter_m:
            value += darcy_friction_factor*thickness_m/(alpha*alpha*hydraulic_diameter_m)
    return value


@dataclass(frozen=True)
class FilterPressureReport:
    method: str
    pressure_drop_Pa: float
    loss_coefficient: float
    approach_velocity_m_s: float
    reynolds: float | None
    source_tables: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def filter_pressure_drop(
    method: str, fluid: FluidState, *, approach_velocity_m_s: float,
    open_fraction: float, characteristic_length_m: float | None = None,
    **loss_options: Any,
) -> FilterPressureReport:
    """Engineering loss from current rho/mu; no particle capture or filter inventory.

For round screens the Reynolds length is the space between rods. For mesh,
the caller declares its tested Reynolds length. Passage hydraulic diameter
for long-grill friction is a separate geometry input.
"""
    velocity = approach_velocity_m_s
    if isinstance(velocity, bool) or not isfinite(velocity) or velocity <= 0:
        raise ValueError("filter pressure evaluation needs a finite positive approach velocity")
    if "reynolds" in loss_options:
        raise ValueError("pressure consumer computes Reynolds from current fluid and geometry")
    re = None
    if method in {"round_screen", "open_mesh"}:
        if (characteristic_length_m is None or not isfinite(characteristic_length_m)
                or characteristic_length_m <= 0):
            raise ValueError("screen Reynolds geometry requires positive characteristic length")
        re = fluid.density_kg_m3*velocity*characteristic_length_m/fluid.viscosity_Pa_s
    elif characteristic_length_m is not None:
        raise ValueError("selected filter law does not consume Reynolds geometry")
    coefficient = filter_loss_coefficient(method, open_fraction=open_fraction,
                                          reynolds=re, **loss_options)
    sources = tuple(curve.table_function.table.dataset_id
                    for name in ("primary_curve", "angle_curve")
                    if (curve := loss_options.get(name)) is not None)
    return FilterPressureReport(method, .5*fluid.density_kg_m3*velocity**2*coefficient,
                                coefficient, velocity, re, sources)
