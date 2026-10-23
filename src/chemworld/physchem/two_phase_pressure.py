"""Named two-phase pipe pressure correlations with explicit model boundaries.

These are engineering correlations, not a dynamic multiphase inventory. All
except Beggs-Brill predict friction alone. Beggs-Brill returns a coupled total
with its own holdup and optional empirical acceleration correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, pi, radians, sin, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.two_phase_flow import lockhart_martinelli_pressure_drop
from chemworld.physchem.two_phase_holdup import (
    TwoPhaseProperties,
    _quality,
    martinelli_parameter,
)

_VISCOSITY_METHODS = (
    "Lockhart_Martinelli",
    "Bankoff",
    "Baroczy_Chisholm",
    "Chisholm",
    "Gronnerud",
    "Jung_Radermacher",
    "Muller_Steinhagen_Heck",
    "Theissing",
    "Wang_Chiang_Lu",
    "Yu_France",
    "Chisholm rough",
)
_SURFACE_METHODS = (
    "Kim_Mudawar",
    "Friedel",
    "Hwang_Kim",
    "Mishima_Hibiki",
    "Tran",
    "Xu_Fang",
    "Zhang_Hibiki_Mishima",
    "Chen_Friedel",
    "Zhang_Hibiki_Mishima adiabatic gas",
    "Zhang_Hibiki_Mishima flow boiling",
)
TWO_PHASE_PRESSURE_REQUIREMENTS = {
    **dict.fromkeys(_VISCOSITY_METHODS, ("liquid_viscosity", "vapor_viscosity")),
    **dict.fromkeys(
        _SURFACE_METHODS,
        (
            "liquid_viscosity",
            "vapor_viscosity",
            "surface_tension",
        ),
    ),
    "Lombardi_Pedrocchi": ("surface_tension",),
    "Zhang_Webb": ("liquid_viscosity", "pressure", "critical_pressure"),
    "Beggs-Brill": ("liquid_viscosity", "vapor_viscosity", "surface_tension", "pressure"),
}
_PROPERTY_NAMES = {
    "liquid_viscosity": "liquid_viscosity_Pa_s",
    "vapor_viscosity": "vapor_viscosity_Pa_s",
    "surface_tension": "surface_tension_N_m",
    "pressure": "pressure_Pa",
    "critical_pressure": "critical_pressure_Pa",
}
TWO_PHASE_PRESSURE_GUIDANCE = {
    "Lockhart_Martinelli": "Horizontal separated flow; original 2000-Re friction convention.",
    "Bankoff": "Source origin unknown and upstream author advises against use; reference only.",
    "Baroczy_Chisholm": "Empirical separated-flow pressure multiplier.",
    "Chisholm": "Smooth-pipe multiplier; n=.25. Select Chisholm rough for rough correction.",
    "Chisholm rough": "Explicit Chisholm 1968 roughness correction; limited empirical scope.",
    "Gronnerud": "Developed for evaporators; empirical liquid Froude correction.",
    "Jung_Radermacher": "Annular, turbulent liquid and vapor flow.",
    "Muller_Steinhagen_Heck": "Empirical blend of all-liquid and all-vapor pressure losses.",
    "Theissing": "Separated-flow blend; source also reports liquid-liquid use.",
    "Wang_Chiang_Lu": "Distinct mass flux branches below/above 200 kg/(m2 s).",
    "Yu_France": "Empirical Reynolds-dependent separated-flow multiplier.",
    "Kim_Mudawar": "Microchannels: D=.0695-6.22 mm, G=4-8528; smooth friction only.",
    "Friedel": "Horizontal or vertical upflow; poor for mu_l/mu_g > 1000; about 40% mean error.",
    "Hwang_Kim": "Developed only for .244 and .792 mm microtubes.",
    "Mishima_Hibiki": "Diameter-dependent small-channel separated-flow multiplier.",
    "Tran": "Boiling refrigerants, developed for hydraulic D=2.4-2.92 mm.",
    "Xu_Fang": "Empirical homogeneous Froude/Weber pressure multiplier.",
    "Zhang_Hibiki_Mishima": "Confinement correction for adiabatic vapor.",
    "Zhang_Hibiki_Mishima adiabatic gas": "Confinement correction for adiabatic gas.",
    "Zhang_Hibiki_Mishima flow boiling": "Confinement correction for flow boiling.",
    "Chen_Friedel": "Mini/microchannels only; quarter-Bond convention; not for large pipes.",
    "Lombardi_Pedrocchi": "Purely empirical homogeneous-density relationship.",
    "Zhang_Webb": "R134a/R22/R404a corresponding states; recommended D=1-7 mm, P/Pc>.2.",
    "Beggs-Brill": "Coupled friction/holdup/gravity; smooth-pipe origin; primitive acceleration.",
}


@dataclass(frozen=True)
class TwoPhasePressureResult:
    method: str
    pressure_drop_Pa: float
    pressure_kind: str
    liquid_reynolds: float | None
    vapor_reynolds: float | None
    liquid_holdup: float | None = None
    flow_regime: str | None = None
    friction_component_Pa: float | None = None
    gravity_component_Pa: float | None = None
    acceleration_fraction: float = 0.0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_two_phase_pressure_methods(properties: TwoPhaseProperties) -> tuple[str, ...]:
    """Required-input availability only; consult model guidance and result warnings."""
    return tuple(
        method
        for method, requirements in TWO_PHASE_PRESSURE_REQUIREMENTS.items()
        if all(getattr(properties, _PROPERTY_NAMES[name]) is not None for name in requirements)
    )


def _kim_friction(reynolds: float) -> float:
    if reynolds < 2000:
        return 64 / reynolds
    return 0.316 * reynolds**-0.25 if reynolds < 20000 else 0.184 * reynolds**-0.2


def _domain_warnings(
    method: str,
    properties: TwoPhaseProperties,
    diameter: float,
    flux: float,
    reynolds_l: float | None,
    reynolds_g: float | None,
) -> list[str]:
    warnings = []
    if method == "Kim_Mudawar":
        if not 0.0000695 <= diameter <= 0.00622 or not 4 <= flux <= 8528:
            warnings.append("Kim_Mudawar diameter/mass_flux outside reported development range")
        reduced = (
            properties.pressure_Pa / properties.critical_pressure_Pa
            if properties.pressure_Pa is not None and properties.critical_pressure_Pa is not None
            else None
        )
        if reduced is not None and not 0.0052 <= reduced <= 0.91:
            warnings.append("Kim_Mudawar reduced pressure outside reported development range")
        if (
            reynolds_l is not None
            and reynolds_g is not None
            and (
                reynolds_l > 79202
                or reynolds_g > 253810
                or flux * diameter / float(properties.liquid_viscosity_Pa_s or 1) > 89798
            )
        ):
            warnings.append("Kim_Mudawar Reynolds number above reported development range")
    if method == "Zhang_Webb":
        assert properties.pressure_Pa is not None and properties.critical_pressure_Pa is not None
        if (
            not 0.001 <= diameter <= 0.007
            or properties.pressure_Pa / properties.critical_pressure_Pa <= 0.2
        ):
            warnings.append("Zhang_Webb requires D=1-7 mm and reduced pressure above .2")
    if (
        method == "Hwang_Kim"
        and min(abs(diameter - value) for value in (0.000244, 0.000792)) > 1e-12
    ):
        warnings.append("Hwang_Kim diameter differs from the two experimental microtubes")
    if method == "Tran" and not 0.0024 <= diameter <= 0.00292:
        warnings.append("Tran diameter outside 2.4-2.92 mm development range")
    if method == "Jung_Radermacher" and (
        reynolds_l is None or reynolds_g is None or min(reynolds_l, reynolds_g) < 2040
    ):
        warnings.append("Jung_Radermacher requires turbulent liquid and vapor")
    return warnings


def two_phase_pressure_drop(
    properties: TwoPhaseProperties,
    *,
    method: str,
    mass_flow_kg_s: float,
    diameter_m: float,
    vapor_quality: float,
    length_m: float = 1.0,
    roughness_m: float = 0.0,
    inclination_degrees: float = 0.0,
    gravity_m_s2: float = 9.80665,
    beggs_brill_acceleration: bool = False,
    check_recommended_ranges: bool = True,
    reference_only: bool = False,
) -> TwoPhasePressureResult:
    """Evaluate a named correlation, preserving its meaning and known limitations.

    Nonzero inclination is accepted only by Beggs-Brill or vertical-up Friedel.
    Other correlations are evaluated as horizontal friction. No gravity term
    is silently added. Bankoff requires explicit reference-only permission.
    """
    for name, value in (
        ("mass flow", mass_flow_kg_s),
        ("diameter", diameter_m),
        ("length", length_m),
        ("gravity", gravity_m_s2),
    ):
        _positive(value, name)
    _quality(vapor_quality, interior=method != "Theissing")
    if not isfinite(roughness_m) or not 0 <= roughness_m < diameter_m:
        raise ValueError("roughness must be finite and in [0, diameter)")
    if not isfinite(inclination_degrees) or not -90 <= inclination_degrees <= 90:
        raise ValueError("inclination must lie in [-90, 90] degrees")
    if method not in TWO_PHASE_PRESSURE_REQUIREMENTS:
        raise ValueError(f"unknown two-phase pressure method {method!r}")
    if method not in available_two_phase_pressure_methods(properties):
        raise ValueError(f"{method} requires {TWO_PHASE_PRESSURE_REQUIREMENTS[method]}")
    if method == "Bankoff" and not reference_only:
        raise ValueError("Bankoff source origin is unknown; explicit reference_only required")
    if (
        inclination_degrees != 0
        and method != "Beggs-Brill"
        and not (method == "Friedel" and inclination_degrees == 90)
    ):
        raise ValueError("selected model is horizontal friction; use a separate gravity relation")
    if method in {"Kim_Mudawar", "Lockhart_Martinelli", "Lombardi_Pedrocchi"} and roughness_m:
        raise ValueError("selected relation has no roughness correction")
    p, x, flux, d, length = (
        properties,
        vapor_quality,
        mass_flow_kg_s / (pi * diameter_m * diameter_m / 4),
        diameter_m,
        length_m,
    )
    liquid, gas = p.liquid_density_kg_m3, p.vapor_density_kg_m3
    mu_l, mu_g = p.liquid_viscosity_Pa_s, p.vapor_viscosity_Pa_s
    re_l = flux * (1 - x) * d / mu_l if mu_l is not None else None
    re_g = flux * x * d / mu_g if mu_g is not None else None
    warnings = _domain_warnings(method, p, d, flux, re_l, re_g)
    if check_recommended_ranges and warnings:
        raise ValueError("; ".join(warnings))
    if method == "Bankoff":
        warnings.append("reference_only: unknown origin; upstream advises against using Bankoff")
    if not check_recommended_ranges:
        warnings.append("recommended range checks disabled: formula analysis, not qualification")
    if method == "Lockhart_Martinelli":
        assert mu_l is not None and mu_g is not None
        original = lockhart_martinelli_pressure_drop(
            mass_flow_kg_s=mass_flow_kg_s,
            vapor_quality=x,
            liquid_density_kg_m3=liquid,
            vapor_density_kg_m3=gas,
            liquid_viscosity_Pa_s=mu_l,
            vapor_viscosity_Pa_s=mu_g,
            diameter_m=d,
            length_m=length,
        )
        return TwoPhasePressureResult(
            method,
            original.pressure_drop_Pa,
            "friction",
            re_l,
            re_g,
            warnings=tuple(warnings) + original.warnings,
        )
    density_h = 1 / ((1 - x) / liquid + x / gas)
    if method == "Lombardi_Pedrocchi":
        assert p.surface_tension_N_m is not None
        drop = 0.83 * flux**1.4 * p.surface_tension_N_m**0.4 * length / (d**1.2 * density_h**0.866)
        return TwoPhasePressureResult(
            method, drop, "friction", re_l, re_g, warnings=tuple(warnings)
        )
    assert mu_l is not None
    rel0 = flux * d / mu_l
    f_l0 = pipe_friction_factor(rel0, roughness_m / d)
    dp_l0 = f_l0 * length / d * flux**2 / (2 * liquid)
    if method == "Zhang_Webb":
        assert p.pressure_Pa is not None and p.critical_pressure_Pa is not None
        reduced = p.pressure_Pa / p.critical_pressure_Pa
        if reduced >= 1:
            raise ValueError("Zhang_Webb requires subcritical pressure")
        multiplier = (
            (1 - x) ** 2 + 2.87 * x**2 / reduced + 1.68 * x**0.8 * (1 - x) ** 0.25 * reduced**-1.64
        )
        return TwoPhasePressureResult(
            method, dp_l0 * multiplier, "friction", re_l, re_g, warnings=tuple(warnings)
        )
    assert mu_g is not None and re_l is not None and re_g is not None
    if method == "Beggs-Brill":
        return _beggs_brill(
            p,
            flux,
            d,
            length,
            x,
            roughness_m,
            inclination_degrees,
            gravity_m_s2,
            beggs_brill_acceleration,
            tuple(warnings),
        )
    reg0 = flux * d / mu_g
    f_g0 = pipe_friction_factor(reg0, roughness_m / d)
    dp_g0 = f_g0 * length / d * flux**2 / (2 * gas)
    if method == "Theissing" and x in {0, 1}:
        return TwoPhasePressureResult(
            method,
            dp_l0 if x == 0 else dp_g0,
            "friction",
            re_l,
            re_g,
            warnings=tuple(warnings),
        )
    fl = (
        _kim_friction(re_l)
        if method == "Kim_Mudawar"
        else pipe_friction_factor(re_l, roughness_m / d)
    )
    fg = (
        _kim_friction(re_g)
        if method == "Kim_Mudawar"
        else pipe_friction_factor(re_g, roughness_m / d)
    )
    dp_l = fl * length / d * (flux * (1 - x)) ** 2 / (2 * liquid)
    dp_g = fg * length / d * (flux * x) ** 2 / (2 * gas)
    martinelli = sqrt(dp_l / dp_g)
    ratio = dp_g0 / dp_l0
    gravity = gravity_m_s2
    if method in {"Chisholm", "Chisholm rough", "Baroczy_Chisholm"}:
        gamma, power = sqrt(ratio), 0.25
        if gamma > 28:
            coefficient = 15000 / (gamma**2 * sqrt(flux))
        elif gamma > 9.5:
            coefficient = (
                520 / (sqrt(flux) * gamma)
                if flux <= 600 or method == "Baroczy_Chisholm"
                else 21 / gamma
            )
        elif method == "Baroczy_Chisholm" or flux >= 1900:
            coefficient = 55 / sqrt(flux)
        else:
            coefficient = 4.8 if flux <= 500 else 2400 / flux
        if method == "Chisholm rough":
            if mu_l == mu_g:
                raise ValueError(
                    "rough Chisholm friction exponent is undefined at equal viscosities"
                )
            power = log(f_l0 / f_g0) / log(reg0 / rel0)
            coefficient *= (0.5 * (1 + (mu_g / mu_l) ** 2 + 10 ** (-600 * roughness_m / d))) ** (
                4 * (0.25 - power)
            )
        drop = dp_l0 * (
            1 + (ratio - 1) * (coefficient * (x * (1 - x)) ** (1 - power / 2) + x ** (2 - power))
        )
    elif method == "Muller_Steinhagen_Heck":
        drop = (dp_l0 + 2 * (dp_g0 - dp_l0) * x) * (1 - x) ** (1 / 3) + dp_g0 * x**3
    elif method == "Gronnerud":
        froude_l = flux**2 / (liquid**2 * gravity * d)
        correction = 1.0 if froude_l >= 1 else froude_l**0.3 + 0.0055 * log(1 / froude_l) ** 2
        multiplier = 1 + correction * (x + 4 * (x**1.8 - x**10 * sqrt(correction))) * (
            liquid / gas * (mu_g / mu_l) ** 0.25 - 1
        )
        drop = dp_l0 * multiplier
    elif method == "Bankoff":
        alpha = (0.71 + 2.35 * gas / liquid) / (1 + (1 - x) * gas / (x * liquid))
        if 1 - alpha * (1 - gas / liquid) < 0:
            raise ValueError("Bankoff predicts a non-real pressure multiplier")
        multiplier = (
            (1 - alpha * (1 - gas / liquid)) ** (3 / 7) * (1 + x * (liquid / gas - 1)) / (1 - x)
        )
        drop = dp_l0 * multiplier**1.75
    elif method == "Jung_Radermacher":
        drop = dp_l0 * 12.82 * martinelli_parameter(p, x) ** -1.47 * (1 - x) ** 1.8
    elif method == "Theissing":
        n_l, n_g = log(dp_l / dp_l0) / log(1 - x), log(dp_g / dp_g0) / log(x)
        weight = (dp_g / dp_l) ** 0.1
        exponent = (n_l + weight * n_g) / (1 + weight)
        epsilon = 3 - 2 * (2 * sqrt(liquid / gas) / (1 + liquid / gas)) ** (0.7 / exponent)
        drop = (
            dp_l0 ** (1 / (exponent * epsilon)) * (1 - x) ** (1 / epsilon)
            + dp_g0 ** (1 / (exponent * epsilon)) * x ** (1 / epsilon)
        ) ** (exponent * epsilon)
    elif method == "Wang_Chiang_Lu":
        if flux >= 200:
            drop = dp_g * (1 + 9.397 * martinelli**0.62 + 0.564 * martinelli**2.45)
        else:
            coefficient = (
                4.566e-6
                * martinelli**0.128
                * rel0**0.938
                * (liquid / gas) ** -2.15
                * (mu_l / mu_g) ** 5.1
            )
            drop = dp_l + dp_g + coefficient * sqrt(dp_l * dp_g)
    elif method == "Yu_France":
        parameter = 18.65 * sqrt(gas / liquid) * (1 - x) / x * re_g**0.1 / sqrt(re_l)
        drop = dp_l * parameter**-1.9
    else:
        assert p.surface_tension_N_m is not None
        sigma = p.surface_tension_N_m
        confinement = sqrt(sigma / (gravity * (liquid - gas))) / d
        froude = flux**2 / (density_h**2 * gravity * d)
        weber = flux**2 * d / (density_h * sigma)
        if method in {"Friedel", "Chen_Friedel"}:
            if mu_g > mu_l:
                raise ValueError("Friedel requires liquid viscosity >= vapor viscosity")
            correction = (liquid / gas) ** 0.91 * (mu_g / mu_l) ** 0.19 * (1 - mu_g / mu_l) ** 0.7
            drop = dp_l0 * (
                (1 - x) ** 2
                + x**2 * ratio
                + 3.24 * x**0.78 * (1 - x) ** 0.224 * correction / (froude**0.0454 * weber**0.035)
            )
            if mu_l / mu_g > 1000:
                warnings.append("Friedel is known to perform poorly for mu_l/mu_g > 1000")
            if method == "Chen_Friedel":
                bond_quarter = 1 / (4 * confinement**2)
                omega = (
                    0.0333 * rel0**0.45 / (re_g**0.09 * (1 + 0.4 * exp(-bond_quarter)))
                    if bond_quarter < 2.5
                    else weber**0.2 / (2.5 + 0.06 * bond_quarter)
                )
                drop *= omega
        elif method == "Tran":
            drop = dp_l0 * (
                1 + (4.3 * ratio - 1) * (confinement * (x * (1 - x)) ** 0.875 + x**1.75)
            )
        elif method == "Xu_Fang":
            drop = dp_l0 * (
                ratio * x**3
                + (1 - x**2.59) ** 0.632
                * (
                    1
                    + 2 * x**1.17 * (ratio - 1)
                    + 0.00775 * x**-0.475 * froude**0.535 * weber**0.188
                )
            )
        else:
            if method == "Hwang_Kim":
                coefficient = 0.227 * rel0**0.452 * martinelli**-0.32 * confinement**-0.82
            elif method == "Mishima_Hibiki":
                coefficient = 21 * (1 - exp(-319 * d))
            elif method == "Kim_Mudawar":
                suratman = gas * sigma * d / mu_g**2
                if re_l < 2000 and re_g < 2000:
                    a, b, c, e = 3.5e-5, 0.44, 0.5, 0.48
                elif re_l < 2000:
                    a, b, c, e = 0.0015, 0.59, 0.19, 0.36
                elif re_g < 2000:
                    a, b, c, e = 0.00087, 0.17, 0.5, 0.14
                else:
                    a, b, c, e = 0.39, 0.03, 0.1, 0.35
                coefficient = a * rel0**b * suratman**c * (liquid / gas) ** e
            else:
                constant = (
                    0.674
                    if method.endswith("adiabatic gas")
                    else 0.358
                    if method.endswith("flow boiling")
                    else 0.142
                )
                coefficient = 21 * (1 - exp(-constant / confinement))
            # Algebraic separated-flow sum avoids a second pressure multiplier implementation.
            drop = dp_l + dp_g + coefficient * sqrt(dp_l * dp_g)
    if not isfinite(drop) or drop < 0:
        raise ValueError(f"{method} predicts nonphysical frictional pressure drop {drop!r}")
    return TwoPhasePressureResult(method, drop, "friction", re_l, re_g, warnings=tuple(warnings))


def _beggs_holdup(
    regime: str, fraction: float, froude: float, angle: float, velocity_number: float
) -> float:
    coefficients = {
        "segregated": (0.98, 0.4846, 0.0868),
        "intermittent": (0.845, 0.5351, 0.0173),
        "distributed": (1.065, 0.5824, 0.0609),
    }
    a, b, c = coefficients[regime]
    horizontal = max(fraction, a * fraction**b / froude**c)
    if angle == 0 or (angle > 0 and regime == "distributed"):
        return horizontal
    if angle <= 0:
        d, e, f, h = 4.7, -0.3692, 0.1244, -0.5056
    elif regime == "segregated":
        d, e, f, h = 0.011, -3.768, 3.539, -1.614
    else:
        d, e, f, h = 2.96, 0.305, -0.4473, 0.0978
    correction = max(0.0, (1 - fraction) * log(d * fraction**e * velocity_number**f * froude**h))
    sine = sin(1.8 * angle)
    return horizontal * (1 + correction * sine * (1 - sine**2 / 3))


def _beggs_brill(
    p: TwoPhaseProperties,
    flux: float,
    diameter: float,
    length: float,
    quality: float,
    roughness: float,
    inclination: float,
    gravity: float,
    acceleration: bool,
    warnings: tuple[str, ...],
) -> TwoPhasePressureResult:
    assert p.surface_tension_N_m is not None and p.pressure_Pa is not None
    assert p.liquid_viscosity_Pa_s is not None and p.vapor_viscosity_Pa_s is not None
    liquid, gas = p.liquid_density_kg_m3, p.vapor_density_kg_m3
    v_l, v_g = flux * (1 - quality) / liquid, flux * quality / gas
    speed, angle = v_l + v_g, radians(inclination)
    fraction = v_l / speed
    froude = speed**2 / (gravity * diameter)
    l1, l2, l3, l4 = (
        316 * fraction**0.302,
        0.0009252 * fraction**-2.4684,
        0.1 * fraction**-1.4516,
        0.5 * fraction**-6.738,
    )
    if (fraction < 0.01 and froude < l1) or (fraction >= 0.01 and froude < l2):
        regime = "segregated"
    elif fraction >= 0.01 and l2 <= froude <= l3:
        regime = "transition"
    elif (0.01 <= fraction < 0.4 and l3 < froude <= l1) or (fraction >= 0.4 and l3 < froude <= l4):
        regime = "intermittent"
    elif (fraction < 0.4 and froude >= l1) or (fraction >= 0.4 and froude > l4):
        regime = "distributed"
    else:
        raise ValueError("outside Beggs-Brill flow-regime map")
    liquid_number = v_l * (liquid / (gravity * p.surface_tension_N_m)) ** 0.25
    if regime == "transition":
        blend = (l3 - froude) / (l3 - l2)
        holdup = blend * _beggs_holdup("segregated", fraction, froude, angle, liquid_number) + (
            1 - blend
        ) * _beggs_holdup("intermittent", fraction, froude, angle, liquid_number)
    else:
        holdup = _beggs_holdup(regime, fraction, froude, angle, liquid_number)
    if not 0 < holdup <= 1:
        raise ValueError(f"Beggs-Brill predicts nonphysical liquid holdup {holdup!r}; no clipping")
    density_slip = liquid * holdup + gas * (1 - holdup)
    density_no_slip = liquid * fraction + gas * (1 - fraction)
    viscosity_no_slip = p.liquid_viscosity_Pa_s * fraction + p.vapor_viscosity_Pa_s * (1 - fraction)
    friction = pipe_friction_factor(
        density_no_slip * speed * diameter / viscosity_no_slip, roughness / diameter
    )
    ratio = fraction / holdup**2
    if 1 < ratio < 1.2:
        correction = log(2.2 * ratio - 1.2)
    else:
        logarithm = log(ratio)
        correction = logarithm / (
            -0.0523 + 3.182 * logarithm - 0.8725 * logarithm**2 + 0.01853 * logarithm**4
        )
    # The source's explicit correlation saturation is part of this named model.
    if correction > 7:
        warnings += ("Beggs-Brill friction exponent saturated at model limit 7",)
    friction *= exp(min(correction, 7))
    friction_drop = friction * length / diameter * density_no_slip * speed**2 / 2
    gravity_drop = gravity * sin(angle) * density_slip * length
    kinetic = v_g * speed * density_slip / p.pressure_Pa if acceleration else 0.0
    if kinetic >= 1:
        raise ValueError("Beggs-Brill acceleration fraction >=1; correlation reaches singularity")
    return TwoPhasePressureResult(
        "Beggs-Brill",
        (friction_drop + gravity_drop) / (1 - kinetic),
        "coupled_total",
        flux * (1 - quality) * diameter / p.liquid_viscosity_Pa_s,
        flux * quality * diameter / p.vapor_viscosity_Pa_s,
        holdup,
        regime,
        friction_drop,
        gravity_drop,
        kinetic,
        warnings,
    )
