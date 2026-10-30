"""Named straight-pipe friction equations with declared empirical ranges.

The correlations return Darcy factors by default. Clamond and Colebrook are
the same constitutive equation here, solved by the existing common root; an
upstream implementation-specific fast approximation is not a second model.
"""

from __future__ import annotations

from math import exp, isfinite, log, log10, sqrt

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import colebrook_friction_factor

# (Re min, Re max, relative roughness min, relative roughness max).
# None means unspecified by the source, not physically unlimited validity.
STRAIGHT_FRICTION_RANGES: dict[str, tuple[float | None, ...]] = {
    "Moody": (4000, 1e8, 0, 0.01),
    "Alshul_1952": (None, None, None, None),
    "Wood_1966": (4000, 5e7, 1e-5, 0.04),
    "Churchill_1973": (None, None, None, None),
    "Eck_1973": (None, None, None, None),
    "Jain_1976": (5000, 1e7, 4e-5, 0.05),
    "Swamee_Jain_1976": (5000, 1e8, 1e-6, 0.05),
    "Churchill_1977": (None, None, None, None),
    "Chen_1979": (4000, 4e8, 1e-7, 0.05),
    "Round_1980": (4000, 4e8, 0, 0.05),
    "Shacham_1980": (4000, 4e8, None, None),
    "Barr_1981": (None, None, None, None),
    "Zigrang_Sylvester_1": (4000, 1e8, 4e-5, 0.05),
    "Zigrang_Sylvester_2": (4000, 1e8, 4e-5, 0.05),
    "Haaland": (4000, 1e8, 1e-6, 0.05),
    "Serghides_1": (None, None, None, None),
    "Serghides_2": (None, None, None, None),
    "Tsal_1989": (4000, 1e8, 0, 0.05),
    "Manadilli_1997": (5245, 1e8, 0, 0.05),
    "Romeo_2002": (3000, 1.5e8, 0, 0.05),
    "Sonnad_Goudar_2006": (4000, 1e8, 1e-6, 0.05),
    "Rao_Kumar_2007": (None, None, None, None),
    "Buzzelli_2008": (None, None, None, None),
    "Avci_Karagoz_2009": (None, None, None, None),
    "Papaevangelo_2010": (1e4, 1e7, 1e-5, 0.001),
    "Brkic_2011_1": (None, None, None, None),
    "Brkic_2011_2": (None, None, None, None),
    "Fang_2011": (3000, 1e8, 0, 0.05),
    "Clamond": (0, None, 0, None),
    "Colebrook": (0, None, 0, None),
    "laminar": (0, 2040, 0, None),
    "Blasius": (4000, 1e5, 0, 0),
    "Prandtl_von_Karman_Nikuradse": (0, None, 0, 0),
    "von_Karman": (0, None, 0, None),
}


def _inside_range(name: str, reynolds: float, roughness: float) -> bool:
    low_r, high_r, low_e, high_e = STRAIGHT_FRICTION_RANGES[name]
    return (
        (low_r is None or reynolds >= low_r)
        and (high_r is None or reynolds <= high_r)
        and (low_e is None or roughness >= low_e)
        and (high_e is None or roughness <= high_e)
    )


def available_straight_friction_methods(
    reynolds: float,
    relative_roughness: float = 0.0,
    *,
    check_ranges: bool = True,
) -> tuple[str, ...]:
    _positive(reynolds, "Reynolds number")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("relative roughness must lie in [0, 1)")
    if check_ranges and reynolds < 2040:
        return ("laminar",)
    return tuple(
        name
        for name in STRAIGHT_FRICTION_RANGES
        if not check_ranges or _inside_range(name, reynolds, relative_roughness)
    )


def straight_friction_correlation(
    reynolds: float,
    relative_roughness: float = 0.0,
    *,
    method: str = "Colebrook",
    darcy: bool = True,
    check_recommended_ranges: bool = True,
) -> float:
    """Evaluate the named equation itself, without an automatic laminar override.

    Use ``pipe_friction_factor`` to choose laminar versus a named turbulent law.
    Disabling recommendations enables formula analysis, not model qualification.
    """
    _positive(reynolds, "Reynolds number")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("relative roughness must lie in [0, 1)")
    if method not in STRAIGHT_FRICTION_RANGES:
        raise ValueError(f"unknown friction method {method!r}")
    if check_recommended_ranges and not _inside_range(method, reynolds, relative_roughness):
        raise ValueError(f"outside declared {method} recommended range")
    r, e = reynolds, relative_roughness
    base = e / 3.7
    if method in {"Colebrook", "Clamond"}:
        factor = colebrook_friction_factor(r, e)
    elif method == "laminar":
        factor = 64 / r
    elif method == "Blasius":
        factor = 0.3164 * r**-0.25
    elif method == "Prandtl_von_Karman_Nikuradse":
        if e:
            raise ValueError("smooth-pipe relation requires zero roughness")
        factor = colebrook_friction_factor(r, 0)
    elif method == "von_Karman":
        if e == 0:
            raise ValueError("fully rough asymptote requires positive roughness")
        factor = 0.25 / log10(base) ** 2
    elif method == "Moody":
        factor = 0.0055 * (1 + (20000 * e + 1e6 / r) ** (1 / 3))
    elif method in {"Alshul_1952", "Tsal_1989"}:
        factor = 0.11 * (68 / r + e) ** 0.25
        if method == "Tsal_1989" and factor < 0.018:
            factor = 0.0028 + 0.85 * factor
    elif method == "Wood_1966":
        factor = 0.094 * e**0.225 + 0.53 * e + 88 * e**0.4 / r ** (1.62 * e**0.134)
    elif method == "Churchill_1973":
        factor = 0.25 / log10(base + (7 / r) ** 0.9) ** 2
    elif method == "Eck_1973":
        factor = 0.25 / log10(e / 3.715 + 15 / r) ** 2
    elif method == "Jain_1976":
        factor = (1.14 - 2 * log10(e + (29.843 / r) ** 0.9)) ** -2
    elif method == "Swamee_Jain_1976":
        factor = 0.25 / log10((6.97 / r) ** 0.9 + base) ** 2
    elif method == "Churchill_1977":
        term = (2.457 * log((7 / r) ** 0.9 + 0.27 * e)) ** 16 + (37530 / r) ** 16
        factor = 8 * ((8 / r) ** 12 + term**-1.5) ** (1 / 12)
    elif method == "Chen_1979":
        inner = e**1.1098 / 2.8257 + (7.149 / r) ** 0.8981
        factor = 0.25 / log10(e / 3.7065 - 5.0452 * log10(inner) / r) ** 2
    elif method == "Round_1980":
        factor = (1.8 * log10(r / (0.135 * r * e + 6.5))) ** -2
    elif method in {"Shacham_1980", "Zigrang_Sylvester_1", "Zigrang_Sylvester_2"}:
        inner = base + (14.5 if method == "Shacham_1980" else 13) / r
        inner = base - 5.02 * log10(inner) / r
        if method == "Zigrang_Sylvester_2":
            inner = base - 5.02 * log10(inner) / r
        factor = 0.25 / log10(inner) ** 2
    elif method == "Barr_1981":
        inner = base + 4.518 * log10(r / 7) / (r * (1 + r**0.52 * e**0.7 / 29))
        factor = 0.25 / log10(inner) ** 2
    elif method == "Haaland":
        factor = (-1.8 * log10(6.9 / r + base**1.11)) ** -2
    elif method in {"Serghides_1", "Serghides_2"}:
        estimates = [-2 * log10(base + 12 / r)]
        for _ in range(2 if method == "Serghides_1" else 1):
            estimates.append(-2 * log10(base + 2.51 * estimates[-1] / r))
        if method == "Serghides_2":
            estimates.insert(0, 4.781)
        a, b, c = estimates
        denominator = c - 2 * b + a
        # If successive estimates coincide, their common value is the limit.
        estimate = a if denominator == 0 and b == a else a - (b - a) ** 2 / denominator
        factor = estimate**-2
    elif method == "Manadilli_1997":
        factor = 0.25 / log10(base + 95 * r**-0.983 - 96.82 / r) ** 2
    elif method == "Romeo_2002":
        inner = (e / 7.7918) ** 0.9924 + (5.3326 / (208.815 + r)) ** 0.9345
        inner = e / 3.827 - 4.567 * log10(inner) / r
        factor = 0.25 / log10(e / 3.7065 - 5.0272 * log10(inner) / r) ** 2
    elif method == "Sonnad_Goudar_2006":
        term = 0.124 * e * r + log(0.4587 * r)
        factor = (0.8686 * (log(0.4587 * r) - term / (term + 1) * log(term))) ** -2
    elif method == "Rao_Kumar_2007":
        if e == 0:
            raise ValueError("Rao-Kumar expression requires positive roughness")
        beta = 1 - 0.55 * exp(-0.33 * log(r / 6.5) ** 2)
        factor = 0.25 / log10(1 / (2 * e * beta * (0.444 / r + 0.135))) ** 2
    elif method == "Buzzelli_2008":
        guess = (0.774 * log(r) - 1.41) / (1 + 1.32 * sqrt(e))
        denominator = base * r + 2.51 * guess
        estimate = guess - (guess + 2 * log10(denominator / r)) / (1 + 2.18 / denominator)
        factor = estimate**-2
    elif method == "Avci_Karagoz_2009":
        factor = 6.4 / log(r / (1 + 0.01 * r * e * (1 + 10 * sqrt(e)))) ** 2.4
    elif method == "Papaevangelo_2010":
        factor = (0.2479 - 9.47e-5 * (7 - log(r)) ** 4) / log10(e / 3.615 + 7.366 * r**-0.9142) ** 2
    elif method in {"Brkic_2011_1", "Brkic_2011_2"}:
        beta = log(r / (1.816 * log(1.1 * r / log(1 + 1.1 * r))))
        inner = 10 ** (-0.4343 * beta) if method.endswith("_1") else 2.18 * beta / r
        factor = 0.25 / log10(inner + e / 3.71) ** 2
    else:
        factor = 1.613 / log(0.234 * e**1.1007 - 60.525 * r**-1.1105 + 56.291 * r**-1.0712) ** 2
    if not isfinite(factor) or factor <= 0:
        raise ValueError(f"{method} predicts nonphysical friction factor {factor!r}")
    return factor if darcy else factor / 4


def friction_transmission_factor(
    *,
    darcy_friction_factor: float | None = None,
    transmission_factor: float | None = None,
) -> float:
    """F=2/sqrt(f_D); return the exactly one unspecified quantity."""
    if (darcy_friction_factor is None) == (transmission_factor is None):
        raise ValueError("provide exactly one Darcy factor or transmission factor")
    if darcy_friction_factor is not None:
        _positive(darcy_friction_factor, "Darcy friction factor")
        return 2 / sqrt(darcy_friction_factor)
    if transmission_factor is None:
        raise AssertionError("invariant violated: transmission_factor is None")
    _positive(transmission_factor, "transmission factor")
    return 4 / transmission_factor**2


def crane_fully_turbulent_friction(diameter_m: float) -> float:
    """Crane nominal-commercial-steel fit, using the full shared Colebrook solve.

    Its effective Re=7.5e6 D is a model convention, not measured flow Reynolds.
    """
    _positive(diameter_m, "diameter")
    return colebrook_friction_factor(7.5e6 * diameter_m, 3.4126825352925e-5 * diameter_m**-1.0112)
