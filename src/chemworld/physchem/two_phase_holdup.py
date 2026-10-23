"""Explicit gas/liquid holdup and mixture-viscosity correlation families.

Mass quality and volumetric void fraction are different quantities. Models
are selected by name; no empirical out-of-range result is silently clipped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, exp, isfinite, log, radians, sin, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _positive


@dataclass(frozen=True)
class TwoPhaseProperties:
    liquid_density_kg_m3: float
    vapor_density_kg_m3: float
    liquid_viscosity_Pa_s: float | None = None
    vapor_viscosity_Pa_s: float | None = None
    surface_tension_N_m: float | None = None
    pressure_Pa: float | None = None
    critical_pressure_Pa: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                _positive(value, name)
        if self.liquid_density_kg_m3 <= self.vapor_density_kg_m3:
            raise ValueError("gas/liquid correlations require liquid density above vapor density")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TwoPhaseProperties:
        return cls(**payload)


_VOIDAGE_GROUPS = {
    (): (
        "homogeneous",
        "Zivi",
        "Smith",
        "Fauske",
        "Chisholm_voidage",
        "Chisholm Armand",
        "Armand",
        "Nishino Yamazaki",
        "Huq_Loth",
        "Gregory_Scott",
    ),
    ("viscosity",): ("Thom", "Turner Wallis", "Baroczy", "Domanski Didion"),
    ("diameter",): ("Kawahara",),
    ("flux", "diameter"): (
        "Guzhov",
        "Graham",
        "Kopte_Newell_Chato",
        "Nicklin Wilkes Davidson",
        "Xu Fang voidage",
    ),
    ("viscosity", "flux", "diameter"): ("Tandon Varma Gupta", "Harms", "Yashar"),
    ("surface_tension", "flux", "diameter"): ("Steiner", "Rouhani 1", "Rouhani 2", "Dix"),
    ("surface_tension", "flux", "diameter", "reduced_pressure"): ("Sun Duffey Peng",),
    ("surface_tension", "flux", "diameter", "pressure"): ("Woldesemayat Ghajar",),
}
VOIDAGE_REQUIREMENTS = {
    name: requirements for requirements, names in _VOIDAGE_GROUPS.items() for name in names
}
TWO_PHASE_VISCOSITY_METHODS = (
    "McAdams",
    "Cicchitti",
    "Lin Kwok",
    "Beattie Whalley",
    "Duckler",
    "Fourar Bories",
)


def _quality(value: float, *, interior: bool = False) -> None:
    if not isfinite(value) or not 0 <= value <= 1 or (interior and value in {0, 1}):
        raise ValueError(
            "vapor quality must lie in (0, 1)" if interior else "vapor quality must lie in [0, 1]"
        )


def _viscosities(properties: TwoPhaseProperties) -> tuple[float, float]:
    if properties.liquid_viscosity_Pa_s is None or properties.vapor_viscosity_Pa_s is None:
        raise ValueError("selected correlation requires both phase viscosities")
    return properties.liquid_viscosity_Pa_s, properties.vapor_viscosity_Pa_s


def martinelli_parameter(
    properties: TwoPhaseProperties,
    vapor_quality: float,
    *,
    quality_exponent: float = 0.9,
    density_exponent: float = 0.5,
    viscosity_exponent: float = 0.1,
    friction_exponent: float | None = None,
) -> float:
    _quality(vapor_quality, interior=True)
    liquid_mu, vapor_mu = _viscosities(properties)
    if friction_exponent is not None:
        if not isfinite(friction_exponent) or not 0 <= friction_exponent < 2:
            raise ValueError("Martinelli friction exponent must lie in [0, 2)")
        quality_exponent, viscosity_exponent = (2 - friction_exponent) / 2, friction_exponent / 2
    if any(
        not isfinite(value) for value in (quality_exponent, density_exponent, viscosity_exponent)
    ):
        raise ValueError("Martinelli powers must be finite")
    return (
        ((1 - vapor_quality) / vapor_quality) ** quality_exponent
        * (properties.vapor_density_kg_m3 / properties.liquid_density_kg_m3) ** density_exponent
        * (liquid_mu / vapor_mu) ** viscosity_exponent
    )


def available_voidage_methods(
    properties: TwoPhaseProperties,
    *,
    mass_flux_kg_m2_s: float | None = None,
    diameter_m: float | None = None,
) -> tuple[str, ...]:
    """Input availability, not assurance a correlation is valid for this material."""
    available = {
        "viscosity": properties.liquid_viscosity_Pa_s is not None
        and properties.vapor_viscosity_Pa_s is not None,
        "surface_tension": properties.surface_tension_N_m is not None,
        "pressure": properties.pressure_Pa is not None,
        "reduced_pressure": properties.pressure_Pa is not None
        and properties.critical_pressure_Pa is not None,
        "flux": mass_flux_kg_m2_s is not None,
        "diameter": diameter_m is not None,
    }
    return tuple(
        name
        for name, required in VOIDAGE_REQUIREMENTS.items()
        if all(available[r] for r in required)
    )


def two_phase_void_fraction(
    properties: TwoPhaseProperties,
    vapor_quality: float,
    *,
    method: str,
    mass_flux_kg_m2_s: float | None = None,
    diameter_m: float | None = None,
    gravity_m_s2: float = 9.80665,
    inclination_degrees: float = 0.0,
    check_recommended_ranges: bool = True,
) -> float:
    """29 named engineering correlations; pure phases use their exact phase limits.

    Requiring inputs is not a claim of universal validity. A fitted correlation
    that predicts alpha outside [0, 1] raises with its unmodified prediction.
    """
    _quality(vapor_quality)
    if method not in VOIDAGE_REQUIREMENTS:
        raise ValueError(f"unknown voidage correlation {method!r}")
    _positive(gravity_m_s2, "gravity")
    if not isfinite(inclination_degrees) or not -90 <= inclination_degrees <= 90:
        raise ValueError("inclination must lie between -90 and +90 degrees")
    if type(check_recommended_ranges) is not bool:
        raise ValueError("range-check choice must be explicitly Boolean")
    if mass_flux_kg_m2_s is not None:
        _positive(mass_flux_kg_m2_s, "mass flux")
    if diameter_m is not None:
        _positive(diameter_m, "diameter")
    if vapor_quality in {0, 1}:
        return float(vapor_quality)
    if method not in available_voidage_methods(
        properties, mass_flux_kg_m2_s=mass_flux_kg_m2_s, diameter_m=diameter_m
    ):
        raise ValueError(f"{method} requires {VOIDAGE_REQUIREMENTS[method]}")
    x, liquid, vapor = (
        vapor_quality,
        properties.liquid_density_kg_m3,
        properties.vapor_density_kg_m3,
    )
    density_ratio, quality_ratio = vapor / liquid, (1 - x) / x
    homogeneous = 1 / (1 + quality_ratio * density_ratio)
    xtt = None
    if "viscosity" in VOIDAGE_REQUIREMENTS[method]:
        xtt = martinelli_parameter(properties, x)
    if method in {"homogeneous", "Armand", "Gregory_Scott"}:
        value = homogeneous * {"homogeneous": 1, "Armand": 0.833, "Gregory_Scott": 1 / 1.19}[method]
    elif method in {"Zivi", "Fauske"}:
        value = 1 / (1 + quality_ratio * density_ratio ** ({"Zivi": 2 / 3, "Fauske": 0.5}[method]))
    elif method == "Smith":
        slip = 0.4 + 0.6 * sqrt(
            (1 / density_ratio + 0.4 * quality_ratio) / (1 + 0.4 * quality_ratio)
        )
        value = 1 / (1 + quality_ratio * density_ratio * slip)
    elif method == "Chisholm_voidage":
        value = 1 / (1 + quality_ratio * density_ratio * sqrt(1 - x * (1 - 1 / density_ratio)))
    elif method == "Chisholm Armand":
        value = homogeneous / (homogeneous + sqrt(1 - homogeneous))
    elif method == "Nishino Yamazaki":
        value = 1 - sqrt(quality_ratio * density_ratio * homogeneous)
    elif method == "Huq_Loth":
        radical = sqrt(1 + 4 * x * (1 - x) * (1 / density_ratio - 1))
        value = 1 - 2 * (1 - x) ** 2 / (1 - 2 * x + radical)
    elif method in {"Thom", "Turner Wallis", "Baroczy"}:
        powers = {
            "Thom": (1.0, 0.89, 0.18),
            "Turner Wallis": (0.72, 0.4, 0.08),
            "Baroczy": (0.74, 0.65, 0.13),
        }[method]
        value = 1 / (
            1
            + martinelli_parameter(
                properties,
                x,
                quality_exponent=powers[0],
                density_exponent=powers[1],
                viscosity_exponent=powers[2],
            )
        )
    elif method == "Domanski Didion":
        assert xtt is not None
        value = (1 + xtt**0.8) ** -0.378 if xtt < 10 else 0.823 - 0.157 * log(xtt)
    elif method == "Kawahara":
        assert diameter_m is not None
        if diameter_m > 0.00025:
            value = 0.833 * homogeneous
        else:
            coefficient = 0.03 if diameter_m > 7.5e-5 else 0.02
            value = coefficient * sqrt(homogeneous) / (1 - (1 - coefficient) * sqrt(homogeneous))
    else:
        assert mass_flux_kg_m2_s is not None and diameter_m is not None
        flux, diameter, gravity = mass_flux_kg_m2_s, diameter_m, gravity_m_s2
        gas_superficial, liquid_superficial = flux * x / vapor, flux * (1 - x) / liquid
        mixture_volume = x / vapor + (1 - x) / liquid
        gas_froude = sqrt(flux**2 * x**3 / ((1 - x) * vapor**2 * gravity * diameter))
        if method == "Guzhov":
            mixture_froude = (flux * mixture_volume) ** 2 / (gravity * diameter)
            value = 0.81 * (1 - exp(-2.2 * sqrt(mixture_froude))) * homogeneous
        elif method in {"Graham", "Kopte_Newell_Chato"}:
            if method == "Kopte_Newell_Chato" and check_recommended_ranges and gas_froude > 454:
                raise ValueError("Kopte Newell Chato recommended domain requires Ft <= 454")
            threshold = 0.01032 if method == "Graham" else 0.044
            if gas_froude < threshold:
                value = 0.0 if method == "Graham" else homogeneous
            else:
                y = log(gas_froude)
                value = (
                    1 - exp(-1 - 0.3 * y - 0.0328 * y * y)
                    if method == "Graham"
                    else 1.045 - exp(-1 - 0.342 * y - 0.0268 * y * y + 0.00597 * y**3)
                )
        elif method == "Xu Fang voidage":
            froude_liquid = flux**2 / (gravity * diameter * liquid**2)
            if check_recommended_ranges and not (
                0.02 <= froude_liquid <= 145 and 0.004 <= density_ratio <= 0.153
            ):
                raise ValueError(
                    "Xu Fang requires Frlo in [.02, 145] and rho_g/rho_l in [.004, .153]"
                )
            value = 1 / (
                1 + (1 + 2 * froude_liquid**-0.2 * homogeneous**3.5) * quality_ratio * density_ratio
            )
        elif method in {"Tandon Varma Gupta", "Harms", "Yashar"}:
            assert xtt is not None
            liquid_mu, _ = _viscosities(properties)
            reynolds = flux * diameter / liquid_mu
            if method == "Yashar":
                value = (1 + 1 / gas_froude + xtt) ** -0.321
            elif method == "Harms":
                reynolds *= 1 - x
                value = (
                    1
                    - 10.06
                    * reynolds**-0.875
                    * (1.74 + 0.104 * sqrt(reynolds)) ** 2
                    / sqrt(1.376 + 7.242 * xtt**-1.655)
                ) ** 2
            else:
                if check_recommended_ranges and reynolds <= 50:
                    raise ValueError("Tandon Varma Gupta recommended domain requires Re_l > 50")
                f = 0.15 * (1 / xtt + 2.85 * xtt**-0.476)
                a, b, power = (1.928, 0.9293, -0.315) if reynolds < 1125 else (0.38, 0.0361, -0.088)
                term = reynolds**power / f
                value = 1 - a * term + b * term**2
        else:
            distribution, drift = 1.2, 0.35 * sqrt(gravity * diameter)
            if method != "Nicklin Wilkes Davidson":
                assert properties.surface_tension_N_m is not None
                sigma = properties.surface_tension_N_m
                capillary_speed = (gravity * sigma * (liquid - vapor) / liquid**2) ** 0.25
                if method in {"Rouhani 1", "Rouhani 2", "Steiner"}:
                    distribution = 1 + (0.12 if method == "Steiner" else 0.2) * (1 - x)
                    if method == "Rouhani 2":
                        distribution = 1 + 0.2 * (1 - x) * (gravity * diameter) ** 0.25 * sqrt(
                            liquid / flux
                        )
                    drift = 1.18 * (1 - x) * capillary_speed
                elif method == "Sun Duffey Peng":
                    assert (
                        properties.pressure_Pa is not None
                        and properties.critical_pressure_Pa is not None
                    )
                    reduced = properties.pressure_Pa / properties.critical_pressure_Pa
                    if reduced > 1:
                        raise ValueError("Sun Duffey Peng requires subcritical pressure")
                    distribution, drift = 1 / (0.82 + 0.18 * reduced), 1.41 * capillary_speed
                else:
                    distribution = homogeneous * (
                        1 + (liquid_superficial / gas_superficial) ** density_ratio**0.1
                    )
                    drift = 2.9 * capillary_speed
                    if method == "Woldesemayat Ghajar":
                        assert properties.pressure_Pa is not None
                        angle = radians(inclination_degrees)
                        drift *= (diameter * (1 + cos(angle))) ** 0.25 * (
                            1.22 * (1 + sin(angle))
                        ) ** (101325 / properties.pressure_Pa)
            value = gas_superficial / (distribution * flux * mixture_volume + drift)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(
            f"{method} predicts nonphysical void fraction {value!r}; no clipping applied"
        )
    return value


def two_phase_mixture_density(properties: TwoPhaseProperties, void_fraction: float) -> float:
    _quality(void_fraction)
    return (
        1 - void_fraction
    ) * properties.liquid_density_kg_m3 + void_fraction * properties.vapor_density_kg_m3


def void_fraction_from_density(
    properties: TwoPhaseProperties, mixture_density_kg_m3: float
) -> float:
    if not isfinite(mixture_density_kg_m3) or not (
        properties.vapor_density_kg_m3 <= mixture_density_kg_m3 <= properties.liquid_density_kg_m3
    ):
        raise ValueError("mixture density must lie between declared phase densities")
    return (properties.liquid_density_kg_m3 - mixture_density_kg_m3) / (
        properties.liquid_density_kg_m3 - properties.vapor_density_kg_m3
    )


def available_two_phase_viscosity_methods(properties: TwoPhaseProperties) -> tuple[str, ...]:
    """List methods with declared inputs, without asserting empirical validity."""
    if properties.liquid_viscosity_Pa_s is None or properties.vapor_viscosity_Pa_s is None:
        return ()
    return TWO_PHASE_VISCOSITY_METHODS


def two_phase_mixture_viscosity(
    properties: TwoPhaseProperties,
    vapor_quality: float,
    *,
    method: str,
) -> float:
    _quality(vapor_quality)
    liquid_mu, vapor_mu = _viscosities(properties)
    if method not in TWO_PHASE_VISCOSITY_METHODS:
        raise ValueError(f"unknown two-phase viscosity correlation {method!r}")
    x, liquid, vapor = (
        vapor_quality,
        properties.liquid_density_kg_m3,
        properties.vapor_density_kg_m3,
    )
    if method == "McAdams":
        return 1 / (x / vapor_mu + (1 - x) / liquid_mu)
    if method == "Cicchitti":
        return x * vapor_mu + (1 - x) * liquid_mu
    if method == "Lin Kwok":
        return liquid_mu * vapor_mu / (vapor_mu + x**1.4 * (liquid_mu - vapor_mu))
    volume = x / vapor + (1 - x) / liquid
    if method == "Beattie Whalley":
        alpha = x / vapor / volume
        return liquid_mu * (1 - alpha) * (1 + 2.5 * alpha) + vapor_mu * alpha
    if method == "Duckler":
        return (x * vapor_mu / vapor + (1 - x) * liquid_mu / liquid) / volume
    return (sqrt(x * vapor_mu / vapor) + sqrt((1 - x) * liquid_mu / liquid)) ** 2 / volume
