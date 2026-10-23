"""Explicit gravity-separator design capacities, without a dynamic phase split."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, isfinite, log, sqrt
from typing import Any

from scipy.interpolate import BSpline, make_interp_spline

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import GRAVITY_M_S2, FluidState

PSI_PA = 6894.757293168361
WATKINS_METHODS = ("spline", "blackwell", "branan")


def _density_ratio(liquid: FluidState, gas: FluidState) -> float:
    if liquid.density_kg_m3 <= gas.density_kg_m3:
        raise ValueError("separator requires liquid denser than gas")
    return gas.density_kg_m3 / liquid.density_kg_m3


@dataclass(frozen=True)
class SeparatorCapacityCurve:
    """Caller-owned Watkins cubic chart: ln(factor) versus ln(K/(1 ft/s)).

    Both logged coordinates are dimensionless. No digitized chart is bundled.
    A not-a-knot cubic is evaluated only inside its declared source domain.
    """

    source_curve: TabulatedScalar
    _spline: BSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        curve = self.source_curve
        if (
            len(curve.axes) != 1
            or curve.axes[0][0] != "ln_factor"
            or curve.axes[0][2] != "dimensionless"
            or curve.output_unit != "dimensionless"
        ):
            raise ValueError("Watkins chart needs dimensionless ln_factor and ln(K/(1 ft/s))")
        rows = {r.row_id: r for r in curve.table.rows}
        pairs = sorted(
            (
                rows[k].values[curve.axes[0][1]].to("dimensionless").value,
                rows[k].values[curve.output_column].to("dimensionless").value,
            )
            for k in curve.row_ids
        )
        if len(pairs) < 4:
            raise ValueError("cubic separator chart requires at least four points")
        object.__setattr__(
            self, "_spline", make_interp_spline([v[0] for v in pairs], [v[1] for v in pairs], k=3)
        )

    def value(self, factor: float) -> float:
        _positive(factor, "Watkins factor")
        coordinate = log(factor)
        low, high = self.source_curve.domain["ln_factor"]
        if not low <= coordinate <= high:
            raise ValueError("separator point outside declared chart domain")
        result = 0.3048 * exp(float(self._spline(coordinate)))
        _positive(result, "separator chart capacity")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"source_curve": self.source_curve.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SeparatorCapacityCurve:
        return cls(TabulatedScalar.from_dict(value["source_curve"]))


def separator_theoretical_capacity(
    droplet_diameter_m: float,
    drag_coefficient: float,
    *,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> float:
    """Souders-Brown K in m/s from a specified droplet drag balance."""
    for name, value in (
        ("droplet diameter", droplet_diameter_m),
        ("drag coefficient", drag_coefficient),
        ("gravity", gravity_m_s2),
    ):
        _positive(value, name)
    return sqrt(4 * gravity_m_s2 * droplet_diameter_m / (3 * drag_coefficient))


def separator_watkins_capacity(
    liquid: FluidState,
    gas: FluidState,
    *,
    vapor_mass_fraction: float,
    method: str = "spline",
    horizontal: bool = False,
    chart: SeparatorCapacityCurve | None = None,
) -> float:
    """K in m/s; published .006..5.4 flow-factor domain is enforced."""
    ratio = _density_ratio(liquid, gas)
    quality = vapor_mass_fraction
    if isinstance(quality, bool) or not isfinite(quality) or not 0 < quality < 1:
        raise ValueError("separator vapor mass fraction must lie in (0,1)")
    if method not in WATKINS_METHODS:
        raise ValueError("unknown Watkins method")
    factor = (1 - quality) / quality * sqrt(ratio)
    if not 0.006 <= factor <= 5.4:
        raise ValueError("Watkins flow factor outside published .006..5.4 domain")
    if method == "spline":
        if chart is None:
            raise ValueError("explicit source chart required for spline method")
        capacity = chart.value(factor)
    else:
        if chart is not None:
            raise ValueError("polynomial Watkins method does not consume a chart")
        coefficients = (
            (-1.877478097, -0.81145804597, -0.1870744085, -0.0145228667, -0.00101148518)
            if method == "blackwell"
            else (-1.942936, -0.814894, -0.179390, -0.0123790, 0.000386235, 0.00025955)
        )
        coordinate = log(factor)
        capacity = 0.3048 * exp(sum(c * coordinate**i for i, c in enumerate(coefficients)))
    return capacity * (1.25 if horizontal else 1.0)


def separator_demister_capacity(
    pressure_Pa: float,
    *,
    horizontal: bool = False,
    endpoint_policy: str = "raise",
) -> float:
    """York demister K in m/s; source clipping requires explicit opt-in."""
    _positive(pressure_Pa, "absolute pressure")
    if endpoint_policy not in {"raise", "clip"}:
        raise ValueError("endpoint policy must be raise or clip")
    pressure = pressure_Pa / PSI_PA
    if endpoint_policy == "clip":
        pressure = min(5500.0, max(1.0, pressure))
    elif not 1 <= pressure <= 5500:
        raise ValueError("York pressure outside 1..5500 psia domain")
    if pressure < 15:
        capacity = 0.1821 + 0.0029 * pressure + 0.046 * log(pressure)
    elif pressure < 40:
        capacity = 0.35
    else:
        capacity = 0.430 - 0.023 * log(pressure)
    return capacity * 0.3048 * (1.25 if horizontal else 1.0)


@dataclass(frozen=True)
class SeparatorCapacityResult:
    maximum_gas_velocity_m_s: float
    capacity_factor_m_s: float
    gas_volume_flow_m3_s: float | None
    required_gas_area_m2: float | None
    design_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def separator_gas_capacity(
    liquid: FluidState,
    gas: FluidState,
    *,
    capacity_factor_m_s: float,
    gas_mass_flow_kg_s: float | None = None,
    design_fraction: float = 1.0,
) -> SeparatorCapacityResult:
    """Souders-Brown velocity and required gas-flow area at the current density.

    Area is free gas-flow area, not a vessel cross section or liquid hold-up.
    This calculation does not transfer mass, determine efficiency or split phases.
    """
    ratio = _density_ratio(liquid, gas)
    _positive(capacity_factor_m_s, "capacity factor")
    _positive(design_fraction, "design fraction")
    if design_fraction > 1:
        raise ValueError("design fraction cannot exceed capacity")
    speed = capacity_factor_m_s * sqrt((1 - ratio) / ratio)
    flow = None
    area = None
    if gas_mass_flow_kg_s is not None:
        _positive(gas_mass_flow_kg_s, "gas mass flow")
        flow = gas_mass_flow_kg_s / gas.density_kg_m3
        area = flow / (design_fraction * speed)
    return SeparatorCapacityResult(speed, capacity_factor_m_s, flow, area, design_fraction)
