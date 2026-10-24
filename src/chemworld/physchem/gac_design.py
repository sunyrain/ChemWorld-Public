"""Sourced, mass-basis CPHSDM design of a fresh GAC bed under constant feed.

This empirical steady design owns no material inventory, clock or regeneration
history. C is kg solute/m3 liquid, q is kg solute/kg carbon; consequently the
numerical Freundlich k carries (m3/kg)**p. No material parameters are inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isfinite, pi, sqrt
from typing import Any

SCHEMA = "fresh-gac-cphsdm-design-v1"


def _positive(*values: float) -> bool:
    return all(not isinstance(v, bool) and isfinite(v) and v > 0 for v in values)


def _source(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bounds(value: tuple[float, float]) -> bool:
    return len(value) == 2 and _positive(*value) and value[0] < value[1]


@dataclass(frozen=True)
class GACEmpiricalParameters:
    freundlich_k: float
    freundlich_exponent: float
    stanton_coefficients: tuple[float, float]
    throughput_coefficients: tuple[float, float, float, float, float]
    source: str
    parameter_kind: str
    concentration_bounds_kg_m3: tuple[float, float]
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    biot_bounds: tuple[float, float]
    ratio_bounds: tuple[float, float] = (0.01, 0.99)
    concentration_unit: str = "kg/m^3"
    loading_unit: str = "kg/kg"

    def __post_init__(self) -> None:
        for name in (
            "stanton_coefficients",
            "throughput_coefficients",
            "concentration_bounds_kg_m3",
            "temperature_bounds_K",
            "pressure_bounds_Pa",
            "biot_bounds",
            "ratio_bounds",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not _positive(self.freundlich_k, self.freundlich_exponent)
            or self.freundlich_exponent >= 1
            or not _source(self.source)
            or self.parameter_kind not in {"synthetic", "published", "measured"}
            or self.concentration_unit != "kg/m^3"
            or self.loading_unit != "kg/kg"
        ):
            raise ValueError("Freundlich requires sourced mass-basis k, 0<p<1 and parameter kind")
        if (
            any(
                not _bounds(getattr(self, key))
                for key in (
                    "concentration_bounds_kg_m3",
                    "temperature_bounds_K",
                    "pressure_bounds_Pa",
                    "biot_bounds",
                    "ratio_bounds",
                )
            )
            or self.ratio_bounds[0] > 0.01
            or self.ratio_bounds[1] >= 1
        ):
            raise ValueError("CPHSDM requires positive explicit domains including ratio 0.01")
        if (
            len(self.stanton_coefficients) != 2
            or len(self.throughput_coefficients) != 5
            or any(
                isinstance(v, bool) or not isfinite(v) or v < 0
                for v in (*self.stanton_coefficients, *self.throughput_coefficients)
            )
            or sum(self.stanton_coefficients) <= 0
            or not _positive(self.throughput_coefficients[2], self.throughput_coefficients[4])
            or self.throughput_coefficients[1] + self.throughput_coefficients[3] <= 0
        ):
            raise ValueError("CPHSDM needs two Stanton and five monotone throughput coefficients")


@dataclass(frozen=True)
class GACBedSpec:
    apparent_density_kg_m3: float
    particle_diameter_m: float
    void_fraction: float
    empty_bed_contact_time_s: float
    superficial_velocity_m_s: float
    source: str

    def __post_init__(self) -> None:
        if (
            not _positive(
                self.apparent_density_kg_m3,
                self.particle_diameter_m,
                self.void_fraction,
                self.empty_bed_contact_time_s,
                self.superficial_velocity_m_s,
            )
            or self.void_fraction >= 1
            or not _source(self.source)
        ):
            raise ValueError("GAC bed needs sourced positive geometry and 0<void fraction<1")


@dataclass(frozen=True)
class GACTransportSpec:
    """None selects the correlation, independently for film and surface diffusion.

    The caller supplies D_l in m2/s when either correlation is selected. No
    diffusivity or temperature adjustment is silently supplied by this model.
    """

    source: str
    film_transfer_m_s: float | None = None
    surface_diffusion_m2_s: float | None = None
    liquid_diffusivity_m2_s: float | None = None
    shape_factor: float | None = None
    particle_porosity: float | None = None
    tortuosity: float | None = None
    surface_to_pore_flux_ratio: float | None = None

    def __post_init__(self) -> None:
        if not _source(self.source) or any(
            not _positive(value)
            for name in self.__dataclass_fields__
            if name != "source" and (value := getattr(self, name)) is not None
        ):
            raise ValueError("GAC transport requires sourced finite positive parameters")
        if self.film_transfer_m_s is None:
            if self.liquid_diffusivity_m2_s is None or self.shape_factor is None:
                raise ValueError("calculated film transfer needs diffusivity and shape factor")
        elif self.shape_factor is not None:
            raise ValueError("fixed film transfer must not carry an ignored shape factor")
        surface = (self.particle_porosity, self.tortuosity, self.surface_to_pore_flux_ratio)
        if self.surface_diffusion_m2_s is None:
            if self.liquid_diffusivity_m2_s is None or any(v is None for v in surface):
                raise ValueError(
                    "calculated surface diffusion needs diffusivity/porosity/tortuosity/ratio"
                )
            if self.particle_porosity is None or self.tortuosity is None:
                raise ValueError("missing surface diffusion declaration")
            if self.particle_porosity >= 1 or self.tortuosity < 1:
                raise ValueError("particle porosity must be <1 and tortuosity >=1")
        elif any(v is not None for v in surface):
            raise ValueError(
                "fixed surface diffusion must not carry ignored correlation parameters"
            )
        if (
            self.film_transfer_m_s is not None
            and self.surface_diffusion_m2_s is not None
            and self.liquid_diffusivity_m2_s is not None
        ):
            raise ValueError("two fixed coefficients must not carry ignored liquid diffusivity")


@dataclass(frozen=True)
class GACFeedConditions:
    volumetric_flow_m3_s: float
    concentration_kg_m3: float
    temperature_K: float
    pressure_Pa: float
    liquid_density_kg_m3: float
    dynamic_viscosity_Pa_s: float
    source: str

    def __post_init__(self) -> None:
        if (
            not _source(self.source)
            or not _positive(
                self.volumetric_flow_m3_s,
                self.concentration_kg_m3,
                self.temperature_K,
                self.pressure_Pa,
                self.liquid_density_kg_m3,
                self.dynamic_viscosity_Pa_s,
            )
            or self.concentration_kg_m3 >= self.liquid_density_kg_m3
        ):
            raise ValueError("GAC feed requires sourced positive flow/C/T/P/density/viscosity")


@dataclass(frozen=True)
class GACDesignSpec:
    target_species: str
    empirical: GACEmpiricalParameters
    bed: GACBedSpec
    transport: GACTransportSpec
    replacement_concentration_ratio: float = 0.5
    breakthrough_elements: int = 7

    def __post_init__(self) -> None:
        if not _source(self.target_species):
            raise ValueError("GAC design requires one target species")
        x = self.replacement_concentration_ratio
        if (
            isinstance(x, bool)
            or not isfinite(x)
            or not 0.01 < x < 1
            or not self.empirical.ratio_bounds[0] <= x <= self.empirical.ratio_bounds[1]
        ):
            raise ValueError("replacement ratio must be >0.01 and inside declared ratio domain")
        if (
            type(self.breakthrough_elements) is not int
            or not 2 <= self.breakthrough_elements <= 10000
        ):
            raise ValueError("breakthrough elements must be an integer in [2,10000]")

    def to_dict(self) -> dict[str, Any]:
        result = {"schema": SCHEMA, **asdict(self)}
        result["empirical"] = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in result["empirical"].items()
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GACDesignSpec:
        if set(value) != {"schema", *cls.__dataclass_fields__} or value["schema"] != SCHEMA:
            raise ValueError("unknown or incomplete GAC design declaration")
        contracts = {
            "empirical": set(GACEmpiricalParameters.__dataclass_fields__),
            "bed": set(GACBedSpec.__dataclass_fields__),
            "transport": set(GACTransportSpec.__dataclass_fields__),
        }
        for key, expected in contracts.items():
            if not isinstance(value[key], Mapping) or set(value[key]) != expected:
                raise ValueError("unknown or incomplete GAC parameter declaration")
        return cls(
            value["target_species"],
            GACEmpiricalParameters(**value["empirical"]),
            GACBedSpec(**value["bed"]),
            GACTransportSpec(**value["transport"]),
            value["replacement_concentration_ratio"],
            value["breakthrough_elements"],
        )


def evaluate_gac_design(spec: GACDesignSpec, feed: GACFeedConditions) -> dict[str, Any]:
    """Evaluate a fresh bed, never a used bed's remaining life or future ledger."""
    p, bed, transport = spec.empirical, spec.bed, spec.transport
    for name, value, bounds in (
        ("concentration", feed.concentration_kg_m3, p.concentration_bounds_kg_m3),
        ("temperature", feed.temperature_K, p.temperature_bounds_K),
        ("pressure", feed.pressure_Pa, p.pressure_bounds_Pa),
    ):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"GAC feed {name} outside declared domain")
    c, eps, dp = feed.concentration_kg_m3, bed.void_fraction, bed.particle_diameter_m
    qeq = p.freundlich_k * c**p.freundlich_exponent
    dg = bed.apparent_density_kg_m3 * qeq * (1 - eps) / (eps * c)
    ui = bed.superficial_velocity_m_s / eps
    kf, ds = transport.film_transfer_m_s, transport.surface_diffusion_m2_s
    diagnostics: dict[str, Any] = {}
    if kf is None:
        diffusivity, shape = transport.liquid_diffusivity_m2_s, transport.shape_factor
        if diffusivity is None or shape is None:
            raise ValueError("missing film transport inputs")
        reynolds = feed.liquid_density_kg_m3 * dp * ui / feed.dynamic_viscosity_Pa_s
        schmidt = feed.dynamic_viscosity_Pa_s / (feed.liquid_density_kg_m3 * diffusivity)
        kf = (
            shape
            * (1 + 1.5 * (1 - eps))
            * diffusivity
            / dp
            * (2 + 0.644 * sqrt(reynolds) * schmidt ** (1 / 3))
        )
        diagnostics.update(reynolds_number=reynolds, schmidt_number=schmidt)
    if ds is None:
        diffusivity = transport.liquid_diffusivity_m2_s
        porosity, tort = transport.particle_porosity, transport.tortuosity
        ratio = transport.surface_to_pore_flux_ratio
        if diffusivity is None or porosity is None or tort is None or ratio is None:
            raise ValueError("missing surface diffusion inputs")
        ds = ratio * porosity * c * diffusivity / (bed.apparent_density_kg_m3 * qeq * tort)
    biot = kf * dp * (1 - eps) / (2 * ds * dg * eps)
    if not p.biot_bounds[0] <= biot <= p.biot_bounds[1]:
        raise ValueError("GAC Biot number outside declared empirical domain")
    stanton = p.stanton_coefficients[0] * biot + p.stanton_coefficients[1]
    min_ebct = stanton * dp / (2 * kf * (1 - eps))
    ebct = bed.empty_bed_contact_time_s
    if ebct < min_ebct:
        raise ValueError("GAC bed below minimum EBCT for this constant-pattern design")
    tau, tau_min = ebct * eps, min_ebct * eps
    b0, b1, b2, b3, b4 = p.throughput_coefficients

    def throughput(x: float) -> float:
        return b0 + b1 * x**b2 + b3 / (1.01 - x**b4)

    def time_at(x: float) -> float:
        return (tau_min * throughput(x) + tau - tau_min) * (dg + 1)

    x_replace, elements = spec.replacement_concentration_ratio, spec.breakthrough_elements
    ratios = [0.0, *(0.01 + (x_replace - 0.01) * i / (elements - 1) for i in range(elements))]
    times = [0.0, *(time_at(x) for x in ratios[1:])]
    if not all(isfinite(t) and t > prev for prev, t in pairwise(times)):
        raise ValueError("GAC constant-pattern breakthrough times must be finite and increasing")
    operation_time = time_at(x_replace)
    average = (
        sum(
            (b - a) * (x + y) / 2
            for a, b, x, y in zip(
                times[:-1],
                times[1:],
                ratios[:-1],
                ratios[1:],
                strict=True,
            )
        )
        / operation_time
    )
    area = feed.volumetric_flow_m3_s / bed.superficial_velocity_m_s
    length = ebct * bed.superficial_velocity_m_s
    volume = area * length
    bulk_density = bed.apparent_density_kg_m3 * (1 - eps)
    mass = volume * bulk_density
    metrics = {
        "equilibrium_loading_kg_kg": qeq,
        "distribution_parameter": dg,
        "biot_number": biot,
        "minimum_stanton_number": stanton,
        "throughput_parameter": throughput(x_replace),
        "bulk_density_kg_m3": bulk_density,
        "interstitial_velocity_m_s": ui,
        "bed_length_m": length,
        "bed_diameter_m": sqrt(4 * area / pi),
        "bed_area_m2": area,
        "bed_volume_m3": volume,
        "carbon_mass_kg": mass,
        "residence_time_s": tau,
        "minimum_ebct_s": min_ebct,
        "minimum_residence_time_s": tau_min,
        "minimum_operational_time_s": tau_min * (dg + 1) * throughput(x_replace),
        "operational_time_s": operation_time,
        "bed_volumes_treated": operation_time / ebct,
        "carbon_usage_kg_s": mass / operation_time,
        "average_concentration_ratio": average,
        "adsorbed_mass_kg": feed.volumetric_flow_m3_s * c * (1 - average) * operation_time,
        "film_transfer_m_s": kf,
        "surface_diffusion_m2_s": ds,
        "average_outlet_target_kg_s": feed.volumetric_flow_m3_s * c * average,
    }
    if not _positive(*metrics.values()) or not 0 < average < x_replace:
        raise ValueError("GAC design produced nonphysical or nonfinite output")
    return {
        "model": SCHEMA,
        "specification": spec.to_dict(),
        "feed": asdict(feed),
        "metrics": metrics,
        "breakthrough": [
            {"time_s": t, "concentration_ratio": x} for t, x in zip(times, ratios, strict=True)
        ],
        "diagnostics": {
            **diagnostics,
            "film_transfer_mode": "fixed"
            if transport.film_transfer_m_s is not None
            else "calculated",
            "surface_diffusion_mode": "fixed"
            if transport.surface_diffusion_m2_s is not None
            else "calculated",
            "concentration_basis": "kg target / m^3 liquid; loading kg target / kg carbon",
            "freundlich_k_unit": "(m^3/kg)**freundlich_exponent",
            "fresh_bed": True,
            "fixed_influent": True,
            "isothermal": True,
            "isobaric": True,
            "zero_time_point": "(0,0); first modeled breakthrough ratio is 0.01",
            "boundary": "single target CPHSDM design; other solutes inert; no dynamic history",
            "material_calibration_claimed": False,
        },
    }
