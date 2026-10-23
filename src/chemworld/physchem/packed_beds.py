"""Sourced packed-bed pressure laws on the original single-fluid state.

Correlation ranges describe fitted evidence, not universal fluid validity.
Ergun uses the existing native implementation. Other laws share SI pressure
gradient variables; no reference package or new evolving inventory is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, pi, sqrt
from typing import Any

from chemworld.physchem.transport import FluidState, packed_bed_pressure_drop_ergun

PACKED_BED_METHODS = (
    "Ergun", "Tallmadge", "Kuo_Nydegger", "Jones_Krier", "Carman", "Hicks",
    "Brauer", "KTA", "Erdim_Akgiray_Demir", "Fahien_Schriver", "Idelchik",
    "Harrison_Brunner_Hecker", "Montillet_Akkari_Comiti", "Guo_Sun",
)
_WALL_METHODS = {"Harrison_Brunner_Hecker", "Montillet_Akkari_Comiti", "Guo_Sun"}


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PackedBedSpec:
    method: str
    length_m: float
    particle_diameter_m: float
    void_fraction: float
    sphericity: float = 1.0
    tube_diameter_m: float | None = None

    def __post_init__(self) -> None:
        if self.method not in PACKED_BED_METHODS:
            raise ValueError("unknown packed-bed method")
        _positive(self.length_m, "bed length")
        _positive(self.particle_diameter_m, "particle diameter")
        if not 0.05 < self.void_fraction < 0.95 or not 0.1 <= self.sphericity <= 1:
            raise ValueError("bed voidage must be in (.05,.95) and sphericity in [.1,1]")
        if self.tube_diameter_m is not None:
            _positive(self.tube_diameter_m, "tube diameter")
            if self.tube_diameter_m <= self.effective_particle_diameter_m:
                raise ValueError("tube must be wider than the effective particle")
            if self.method not in _WALL_METHODS:
                raise ValueError("selected bed law does not consume a tube diameter")
        if self.method == "Guo_Sun" and self.tube_diameter_m is None:
            raise ValueError("Guo Sun requires an explicit tube diameter")

    @property
    def effective_particle_diameter_m(self) -> float:
        return self.particle_diameter_m * self.sphericity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PackedBedSpec:
        return cls(**payload)


@dataclass(frozen=True)
class PackedBedPressureReport:
    method: str
    pressure_drop_Pa: float
    reynolds_particle: float
    reynolds_modified: float
    range_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_packed_bed_methods(*, tube_diameter_m: float | None = None) -> tuple[str, ...]:
    """Input availability only; the caller explicitly selects its physical law."""
    if tube_diameter_m is not None:
        _positive(tube_diameter_m, "tube diameter")
    return tuple(m for m in PACKED_BED_METHODS if m != "Guo_Sun" or tube_diameter_m is not None)


def _range_notes(spec: PackedBedSpec, re: float, rem: float) -> tuple[str, ...]:
    e, method = spec.void_fraction, spec.method
    ratio = (spec.tube_diameter_m / spec.effective_particle_diameter_m
             if spec.tube_diameter_m is not None else None)
    ranges = {
        "Ergun": (rem, 1., 2300., "modified Reynolds"),
        "Tallmadge": (re, .1, 100000., "particle Reynolds"),
        "Kuo_Nydegger": (re, 460., 14600., "particle Reynolds"),
        "Jones_Krier": (re, 733., 126670., "particle Reynolds"),
        "Carman": (rem, 300., 60000., "modified Reynolds"),
        "Hicks": (rem, 300., 60000., "modified Reynolds"),
        "KTA": (rem, 1., 100000., "modified Reynolds"),
        "Erdim_Akgiray_Demir": (rem, 2., 3582., "modified Reynolds"),
        "Idelchik": (rem, .001, 1000., "modified Reynolds"),
        "Harrison_Brunner_Hecker": (re, .72, 7700., "particle Reynolds"),
        "Montillet_Akkari_Comiti": (re, 10., 2500., "particle Reynolds"),
        "Guo_Sun": (rem, 100., 33000., "modified Reynolds"),
    }
    notes = []
    if re and method in ranges:
        value, lo, hi, name = ranges[method]
        if not lo < value < hi:
            notes.append(f"{name} outside source recommendation ({lo:g}, {hi:g})")
    porosities = {"KTA": (.36, .42), "Erdim_Akgiray_Demir": (.377, .470),
                  "Idelchik": (.3, .8), "Jones_Krier": (.3804, .4304),
                  "Kuo_Nydegger": (.3760, .3901), "Guo_Sun": (.476, .492),
                  "Harrison_Brunner_Hecker": (.33, .88)}
    if method in porosities and not porosities[method][0] < e < porosities[method][1]:
        notes.append(f"voidage outside source recommendation {porosities[method]}")
    tube_ranges = {"Erdim_Akgiray_Demir": (4., 34.1),
                   "Harrison_Brunner_Hecker": (8.3, 50.), "Guo_Sun": (2., 3.)}
    if method in tube_ranges:
        lo, hi = tube_ranges[method]
        if ratio is None:
            notes.append("tube-ratio source qualification not established from these inputs")
        elif not lo < ratio < hi:
            notes.append(f"tube/particle ratio outside source recommendation ({lo:g}, {hi:g})")
    if method == "Brauer":
        notes.append("source Reynolds ranges conflict: (2,20000) versus (.01,40000)")
    if method == "Fahien_Schriver":
        notes.append("source validity range not supplied")
    if method == "Ergun" and rem > 700:
        notes.append("source notes overprediction above modified Reynolds 700")
    return tuple(notes)


def packed_bed_pressure(
    spec: PackedBedSpec, fluid: FluidState, *, superficial_velocity_m_s: float,
    range_policy: str = "report",
) -> PackedBedPressureReport:
    """Pressure loss [Pa]; finite nonnegative velocity is the superficial velocity.

``report`` exposes source-range notes; ``raise`` requires all available source
qualifications. Neither policy establishes measured accuracy for a material.
"""
    v = superficial_velocity_m_s
    if isinstance(v, bool) or not isfinite(v) or v < 0:
        raise ValueError("superficial velocity must be finite and nonnegative")
    if range_policy not in {"report", "raise"}:
        raise ValueError("range_policy must be report or raise")
    d, e = spec.effective_particle_diameter_m, spec.void_fraction
    solid, rho, mu = 1-e, fluid.density_kg_m3, fluid.viscosity_Pa_s
    re = rho*v*d/mu
    rem = re/solid
    notes = _range_notes(spec, re, rem)
    if v and range_policy == "raise" and notes:
        raise ValueError("; ".join(notes))
    pressure = 0.
    if v and spec.method == "Ergun":
        pressure = packed_bed_pressure_drop_ergun(
            fluid=fluid, superficial_velocity_m_s=v, bed_length_m=spec.length_m,
            particle_diameter_m=spec.particle_diameter_m, void_fraction=e,
            sphericity=spec.sphericity,
        ).pressure_drop_Pa
    elif v:
        # Written as a dimensionless multiplier of viscous SI pressure gradient.
        viscous_scale = mu*v*solid**2/(d*d*e**3)
        coefficients = {
            "Tallmadge": (150., 4.2, 5/6), "Kuo_Nydegger": (276.23, 5.05, .87),
            "Jones_Krier": (150., 3.89, .87), "Carman": (180., 2.871, .9),
            "Brauer": (160., 3.1, .9), "KTA": (160., 3., .9),
            "Erdim_Akgiray_Demir": (160., 2.81, .904),
        }
        if spec.method in coefficients:
            a, b, power = coefficients[spec.method]
            multiplier = a + b*rem**power
        elif spec.method == "Hicks":
            multiplier = 6.8*rem**.8
        elif spec.method == "Fahien_Schriver":
            q = exp(-e*e*solid*rem/12.6)
            multiplier = q*136/solid**.38 + (1-q)*(
                1.87*e**.75/solid**.26*rem + 29/(solid**1.45*e*e))
        elif spec.method == "Idelchik":
            scaled = .45*rem/sqrt(e)
            multiplier = .765*e**-1.2*re/solid**2*(30/scaled + 3/scaled**.7 + .3)
        elif spec.method == "Harrison_Brunner_Hecker":
            ratio = 0. if spec.tube_diameter_m is None else d/spec.tube_diameter_m
            a = (1+pi*ratio/(6*solid))**2
            b = 1-pi*pi*ratio/24*(1-ratio/2)
            multiplier = 119.8*a+4.63*b*rem**(5/6)
        elif spec.method == "Montillet_Akkari_Comiti":
            diameter_factor = (2.2 if spec.tube_diameter_m is None or
                               spec.tube_diameter_m/d > 50 else (spec.tube_diameter_m/d)**.2)
            multiplier = (.061 if e < .4 else .05)*diameter_factor*(
                1000+60*sqrt(re)+12*re)/solid
        else:
            assert spec.tube_diameter_m is not None
            multiplier = 180+(9.5374*d/spec.tube_diameter_m-2.8054)*rem**.97
        pressure = multiplier*viscous_scale*spec.length_m
    if not isfinite(pressure) or pressure < 0:
        raise ValueError(f"{spec.method} predicts nonphysical loss {pressure}; no clipping")
    return PackedBedPressureReport(spec.method, pressure, re, rem, notes)


def packed_bed_voidage(
    *, particle_diameter_m: float, tube_diameter_m: float,
    particle_shape: str, sphericity: float | None = None, check_range: bool = True,
) -> float:
    """Benyahia-O'Neil correlations; cylindrical law is independent of sphericity."""
    _positive(particle_diameter_m, "particle diameter")
    _positive(tube_diameter_m, "tube diameter")
    ratio = tube_diameter_m/particle_diameter_m
    if ratio <= 1:
        raise ValueError("tube must be wider than particle")
    if type(check_range) is not bool:
        raise ValueError("check_range must be Boolean")
    if particle_shape == "general":
        if sphericity is None or not 0 < sphericity <= 1:
            raise ValueError("general particles need sphericity in (0,1]")
        if check_range and not (1.5 < ratio < 50 and .42 < sphericity <= 1):
            raise ValueError("general voidage leaves source range")
        value = .1504+.2024/sphericity+1.0814/(ratio+.1226)**2
    elif particle_shape in {"spherical", "cylindrical"}:
        if sphericity is not None:
            raise ValueError("this shape law does not consume sphericity")
        lo, hi = (1.5, 50) if particle_shape == "spherical" else (1.7, 26.3)
        if check_range and not lo < ratio < hi:
            raise ValueError("shape voidage leaves source range")
        a, b, c = ((.39, 1.74, 1.14) if particle_shape == "spherical" else (.373, 1.703, .611))
        value = a+b/(ratio+c)**2
    else:
        raise ValueError("particle shape must be general, spherical or cylindrical")
    if not 0 < value < 1:
        raise ValueError("nonphysical predicted bed voidage; no clipping")
    return value
