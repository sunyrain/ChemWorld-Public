"""Packed-tower hydraulics on explicit geometry and current fluid properties."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, log1p, pi, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.transport import GRAVITY_M_S2, FluidState


def _voidage(value: float) -> None:
    if isinstance(value, bool) or not isfinite(value) or not 0 < value < 1:
        raise ValueError("voidage must lie in (0,1)")


def mesh_voidage_from_mass(
    mass_kg: float,
    material_density_kg_m3: float,
    diameter_m: float,
    height_m: float,
) -> float:
    for name, value in (
        ("mass", mass_kg),
        ("material density", material_density_kg_m3),
        ("diameter", diameter_m),
        ("height", height_m),
    ):
        _positive(value, name)
    voidage = 1 - mass_kg / (material_density_kg_m3 * pi * diameter_m**2 * height_m / 4)
    _voidage(voidage)
    return voidage


def mesh_specific_area(voidage: float, wire_diameter_m: float) -> float:
    """Area/bed volume in 1/m, ideal non-touching cylindrical wires."""
    _voidage(voidage)
    _positive(wire_diameter_m, "wire diameter")
    return 4 * (1 - voidage) / wire_diameter_m


def demister_dry_pressure_drop(
    fluid: FluidState,
    *,
    superficial_velocity_m_s: float,
    voidage: float,
    specific_area_m_inv: float,
    depth_m: float,
    include_literature_fit: bool = False,
) -> float:
    """Setekleiv-Svendsen dry-pad empirical loss, not an irrigated-bed model.

    The rational fit has singularities and no globally qualified domain; invalid
    poles and negative predictions reject rather than clipping to a valid loss.
    """
    _voidage(voidage)
    for name, value in (
        ("velocity", superficial_velocity_m_s),
        ("specific area", specific_area_m_inv),
        ("depth", depth_m),
    ):
        _positive(value, name)
    product = specific_area_m_inv * depth_m
    a, b, c, d, e = (
        (7.3, 320.0, 52.4, 161.0, 27.2)
        if include_literature_fit
        else (10.29, 565.0, 74.9, 160.9, 45.33)
    )
    denominator1, denominator2 = 69.6 * product - product**2 - 779, d - 4.85 * product
    if denominator1 == 0 or denominator2 == 0:
        raise ValueError("demister dry fit at a rational singularity")
    viscous = fluid.viscosity_Pa_s * voidage * specific_area_m_inv**2 * depth_m
    viscous /= fluid.density_kg_m3 * superficial_velocity_m_s
    coefficient = a - b / denominator1 - c / denominator2 + e * viscous**0.75
    pressure = coefficient * fluid.density_kg_m3 * superficial_velocity_m_s**2 / voidage**2
    _positive(pressure, "predicted demister pressure loss")
    return pressure


def demister_wet_pressure_drop(
    *,
    superficial_velocity_m_s: float,
    voidage: float,
    wire_diameter_m: float,
    depth_m: float,
) -> float:
    """El-Dessouky wet-pad Pa; stainless-mesh source conversion is fixed at 7999kg/m3.

    This empirical expression does not consume liquid properties. Source ranges:
    speed .98..7.5m/s, wire .2...32mm, depth .1...2m, bulk mesh80.317..208.16kg/m3.
    Caller is responsible for applicability; it is not a general dry/wet maximum.
    """
    _voidage(voidage)
    for name, value in (
        ("velocity", superficial_velocity_m_s),
        ("wire diameter", wire_diameter_m),
        ("depth", depth_m),
    ):
        _positive(value, name)
    return (
        depth_m
        * 0.002356999643727531
        * (1 - voidage) ** 0.375798
        * superficial_velocity_m_s**0.81317
        * wire_diameter_m**-1.56114147
    )


@dataclass(frozen=True)
class DemisterEfficiencyResult:
    removal_fraction: float
    raw_prediction: float
    clipped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def demister_removal_efficiency(
    *,
    superficial_velocity_m_s: float,
    voidage: float,
    wire_diameter_m: float,
    droplet_diameter_m: float,
    endpoint_policy: str = "raise",
) -> DemisterEfficiencyResult:
    """El-Dessouky removal fraction; source's cap at one is explicit and reported."""
    _voidage(voidage)
    for name, value in (
        ("velocity", superficial_velocity_m_s),
        ("wire diameter", wire_diameter_m),
        ("droplet diameter", droplet_diameter_m),
    ):
        _positive(value, name)
    if endpoint_policy not in {"raise", "clip"}:
        raise ValueError("efficiency endpoint policy must be raise or clip")
    raw = (
        0.858352355761947
        * wire_diameter_m**-0.28264
        * (1 - voidage) ** 0.099625
        * superficial_velocity_m_s**0.106878
        * droplet_diameter_m**0.383197
    )
    if not isfinite(raw):
        raise ValueError("nonfinite demister efficiency")
    if raw > 1 and endpoint_policy == "raise":
        raise ValueError("demister model predicts removal above one; extrapolation unphysical")
    return DemisterEfficiencyResult(min(raw, 1.0), raw, raw > 1)


@dataclass(frozen=True)
class TowerPackingSpec:
    voidage: float
    specific_area_m_inv: float
    friction_coefficients: tuple[float, float, float]
    source_note: str

    def __post_init__(self) -> None:
        _voidage(self.voidage)
        _positive(self.specific_area_m_inv, "specific packing area")
        coefficients = tuple(self.friction_coefficients)
        if len(coefficients) != 3 or any(not isfinite(c) or c < 0 for c in coefficients):
            raise ValueError("three finite nonnegative packing friction coefficients required")
        if not any(coefficients) or not self.source_note.strip():
            raise ValueError("nonzero packing friction and its source note required")
        object.__setattr__(self, "friction_coefficients", coefficients)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TowerPackingSpec:
        return cls(
            value["voidage"],
            value["specific_area_m_inv"],
            tuple(value["friction_coefficients"]),
            value["source_note"],
        )


@dataclass(frozen=True)
class TowerPressureResult:
    pressure_drop_Pa: float
    dry_pressure_drop_Pa: float
    liquid_holdup_fraction: float
    gas_superficial_velocity_m_s: float
    branch: str
    dimensionless_residual: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dry_terms(spec: TowerPackingSpec, gas: FluidState, velocity: float) -> tuple[float, float]:
    _positive(velocity, "gas velocity")
    diameter = 6 * (1 - spec.voidage) / spec.specific_area_m_inv
    reynolds = velocity * gas.density_kg_m3 * diameter / gas.viscosity_Pa_s
    c1, c2, c3 = spec.friction_coefficients
    inverse, half = c1 / reynolds, c2 / sqrt(reynolds)
    factor = inverse + half + c3
    gradient = 0.75 * factor * (1 - spec.voidage) / spec.voidage**4.65
    gradient *= gas.density_kg_m3 / diameter * velocity**2
    return gradient, (-inverse - 0.5 * half) / factor


def _irrigation(spec: TowerPackingSpec, velocity: float) -> float:
    _positive(velocity, "liquid velocity")
    h0 = 0.555 * (velocity**2 * spec.specific_area_m_inv / (GRAVITY_M_S2 * spec.voidage**4.65)) ** (
        1 / 3
    )
    if h0 >= spec.voidage:
        raise ValueError("static irrigation holdup fills packing voidage")
    return h0


def _wet_peak(spec: TowerPackingSpec, h0: float, exponent: float) -> tuple[float, float]:
    # Maximize y/A(y), y=(pressure gradient)/(rho_l*g), holdup=h0+20*h0*y^2.
    # Its logarithmic derivative gives a quadratic in z=holdup-h0. The positive
    # root is evaluated without cancellation. This selects the hydraulic fold.
    free = spec.voidage - h0
    occupied = 1 - free
    a = 10.3 - 2 * exponent
    b = (2 * exponent - 1) * free + 10.3 * occupied
    discriminant = sqrt(b * b + 4 * a * free * occupied)
    z = 2 * free * occupied / (b + discriminant) if b >= 0 else (discriminant - b) / (2 * a)
    peak = sqrt(z / (20 * h0))
    return peak, peak / exp(_log_wet_multiplier(spec, h0 + z, exponent))


def _log_wet_multiplier(spec: TowerPackingSpec, holdup: float, exponent: float) -> float:
    return exponent * log1p(holdup / (1 - spec.voidage)) + 4.65 * log(
        spec.voidage / (spec.voidage - holdup)
    )


def stichlmair_tower_pressure(
    packing: TowerPackingSpec,
    gas: FluidState,
    *,
    gas_velocity_m_s: float,
    height_m: float = 1.0,
    liquid: FluidState | None = None,
    liquid_velocity_m_s: float = 0.0,
) -> TowerPressureResult:
    """Dry or irrigated pressure; wet solve returns the lower, pre-flood branch.

    A fold check rejects flows without a pre-flood solution. The shared root
    operates on dimensionless pressure; the model is linear in packing height.
    """
    _positive(height_m, "packing height")
    if not isfinite(liquid_velocity_m_s) or liquid_velocity_m_s < 0:
        raise ValueError("liquid velocity must be nonnegative")
    dry, c = _dry_terms(packing, gas, gas_velocity_m_s)
    if liquid_velocity_m_s == 0:
        return TowerPressureResult(
            dry * height_m, dry * height_m, 0.0, gas_velocity_m_s, "dry", 0.0
        )
    if liquid is None or liquid.density_kg_m3 <= gas.density_kg_m3:
        raise ValueError("irrigated packing needs an explicit liquid denser than gas")
    h0 = _irrigation(packing, liquid_velocity_m_s)
    exponent = (2 + c) / 3
    peak, maximum = _wet_peak(packing, h0, exponent)
    normalized_dry = dry / (liquid.density_kg_m3 * GRAVITY_M_S2)
    if normalized_dry > maximum * (1 + 2e-11):
        raise ValueError("gas flow exceeds the Stichlmair flooding fold")

    def residual(value: float) -> float:
        return log(value / normalized_dry) - _log_wet_multiplier(
            packing, h0 * (1 + 20 * value**2), exponent
        )

    at_fold = abs(residual(peak)) <= 2e-11
    value = (
        peak
        if at_fold
        else _engineering_root(
            residual,
            min(2 * normalized_dry, (normalized_dry + peak) / 2),
            normalized_dry,
            peak,
            name="tower_dimensionless_pressure",
            unit="dimensionless",
            scale=peak,
        )
    )
    return TowerPressureResult(
        value * liquid.density_kg_m3 * GRAVITY_M_S2 * height_m,
        dry * height_m,
        h0 * (1 + 20 * value**2),
        gas_velocity_m_s,
        "flooding fold" if at_fold else "pre-flood lower pressure",
        residual(value),
    )


def stichlmair_flooding_velocity(
    packing: TowerPackingSpec,
    gas: FluidState,
    liquid: FluidState,
    *,
    liquid_velocity_m_s: float,
    height_m: float = 1.0,
    gas_velocity_bounds_m_s: tuple[float, float] = (1e-8, 1e3),
) -> TowerPressureResult:
    """Solve the hydraulic fold with the common scalar root, not source Newton code."""
    _positive(height_m, "packing height")
    if liquid.density_kg_m3 <= gas.density_kg_m3:
        raise ValueError("flooding requires liquid denser than gas")
    h0 = _irrigation(packing, liquid_velocity_m_s)
    low, high = gas_velocity_bounds_m_s
    _positive(low, "lower gas velocity")
    if not isfinite(high) or high <= low:
        raise ValueError("ordered finite gas velocity bounds required")

    def residual(log_velocity: float) -> float:
        dry, c = _dry_terms(packing, gas, exp(log_velocity))
        _, maximum = _wet_peak(packing, h0, (2 + c) / 3)
        return log(dry / (liquid.density_kg_m3 * GRAVITY_M_S2 * maximum))

    if residual(log(low)) * residual(log(high)) > 0:
        raise ValueError("gas velocity bounds do not contain a flooding fold")
    value = _engineering_root(
        residual,
        (log(low) + log(high)) / 2,
        log(low),
        log(high),
        name="log_tower_flooding_velocity",
        unit="dimensionless",
        scale=1.0,
    )
    velocity = exp(value)
    dry, c = _dry_terms(packing, gas, velocity)
    peak, _ = _wet_peak(packing, h0, (2 + c) / 3)
    return TowerPressureResult(
        peak * liquid.density_kg_m3 * GRAVITY_M_S2 * height_m,
        dry * height_m,
        h0 * (1 + 20 * peak**2),
        velocity,
        "flooding fold",
        residual(value),
    )


def robbins_tower_pressure_drop(
    liquid: FluidState,
    gas: FluidState,
    *,
    liquid_mass_flux_kg_m2_s: float,
    gas_mass_flux_kg_m2_s: float,
    packing_factor_per_ft: float,
    height_m: float = 1.0,
) -> float:
    """Robbins pressure loss, retaining the source's rounded engineering conversions.

    Packing factor is explicitly per foot, not per metre. Empirical numerical
    constants use lb/(h*ft²), lb/ft³, cP, inH2O/ft; output pressure is Pa.
    """
    for name, value in (
        ("liquid mass flux", liquid_mass_flux_kg_m2_s),
        ("gas mass flux", gas_mass_flux_kg_m2_s),
        ("packing factor per ft", packing_factor_per_ft),
        ("height", height_m),
    ):
        _positive(value, name)
    root_factor = sqrt(packing_factor_per_ft / 20)
    liquid_loading = (
        liquid_mass_flux_kg_m2_s * 737.33812 * 62.4 / (liquid.density_kg_m3 * 0.062427961)
    )
    liquid_loading *= root_factor * (liquid.viscosity_Pa_s * 1000) ** 0.1
    gas_loading = (
        gas_mass_flux_kg_m2_s * 737.33812 * sqrt(0.075 / (gas.density_kg_m3 * 0.062427961))
    )
    gas_loading *= root_factor
    dry_term = 7.4e-8 * gas_loading**2 * 10 ** (2.7e-5 * liquid_loading)
    pressure = (
        (dry_term + 0.4 * (liquid_loading / 20000) ** 0.1 * dry_term**4) * 817.22083 * height_m
    )
    _positive(pressure, "Robbins pressure loss")
    return pressure
