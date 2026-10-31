"""IAPWS ordinary-water transport relations, independent of any water EOS.

Scientific coefficients transcribed from IAPWS R12-08 Tables 1-3 and R15-11
Table 6. https://iapws.org/technical-guidance/release/viscosity.download
https://iapws.org/documents/release/ThCond.download
The caller owns phase stability and the EOS state/derivatives. This module
does not turn arbitrary T/rho inputs into a certified stable water state.
"""

from math import atan, exp, isfinite, log1p, sin, sqrt, tan
from typing import Any, TypeVar, cast

from chemworld.physchem.caloric_math import FirstDerivative, exponential
from chemworld.physchem.conductivity_critical import critical_conductivity_shape
from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.second_order_pair import SecondOrderPair

_Temperature = TypeVar(
    "_Temperature", float, PreciseScalar, SecondOrderPair, FirstDerivative[Any, Any],
)

# IAPWS R15-11 Tables 1 and 2 (i rows, j columns), not reference code.
WATER_CONDUCTIVITY_DILUTE = (.002443221, .01323095, .006770357, -.003454586, .0004096266)
WATER_CONDUCTIVITY_RESIDUAL = (
    (1.60397357, -.646013523, .111443906, .102997357, -.0504123634, .00609859258),
    (2.33771842, -2.78843778, 1.53616167, -.463045512, .0832827019, -.00719201245),
    (2.19650529, -4.54580785, 3.55777244, -1.40944978, .275418278, -.0205938816),
    (-1.21051378, 1.60812989, -.621178141, .0716373224, 0, 0),
    (-2.7203370, 4.57586331, -3.18369245, 1.1168348, -.19268305, .012913842),
)

WATER_VISCOSITY_DILUTE = (1.67752, 2.20462, .6366564, -.241605)
WATER_VISCOSITY_RESIDUAL = (
    (0, 0, .520094), (1, 0, .0850895), (2, 0, -1.08374), (3, 0, -.289555),
    (0, 1, .222531), (1, 1, .999115), (2, 1, 1.88797), (3, 1, 1.26613), (5, 1, .120573),
    (0, 2, -.281378), (1, 2, -.906851), (2, 2, -.772479), (3, 2, -.489837), (4, 2, -.257040),
    (0, 3, .161913), (1, 3, .257399), (0, 4, -.0325372), (3, 4, .0698452),
    (4, 5, .00872102), (3, 6, -.00435673), (5, 6, -.000593264),
)
_REFERENCE_SUSCEPTIBILITY = (
    (6.53786807199516, -5.61149954923348, 3.39624167361325, -2.27492629730878,
     10.2631854662709, 1.97815050331519),
    (6.52717759281799, -6.30816983387575, 8.08379285492595, -9.82240510197603,
     12.1358413791395, -5.54349664571295),
    (5.35500529896124, -3.96415689925446, 8.91990208918795, -12.0338729505790,
     9.19494865194302, -2.16866274479712),
    (1.55225959906681, .464621290821181, 8.93237374861479, -11.0321960061126,
     6.16780999933360, -.965458722086812),
    (1.11999926419994, .595748562571649, 9.88952565078920, -10.3255051147040,
     4.66861294457414, -.503243546373828),
)


def water_viscosity_background(T: _Temperature, rho: float) -> _Temperature:
    """IAPWS R12-08 industrial/background equation, Pa s; supports analytic jets.

    Internal equation only; the public wrapper or correlation validates input.
    Density is kg/m³, T kelvin. No critical enhancement is included here.
    """
    tr, delta = T / 647.096, rho / 322
    dilute = 100 * tr**.5 / sum(h / tr**i for i, h in enumerate(WATER_VISCOSITY_DILUTE))
    residual = exponential(delta * sum(h * (1 / tr - 1)**i * (delta - 1)**j
                                       for i, j, h in WATER_VISCOSITY_RESIDUAL))
    return dilute * residual * 1e-6


def water_conductivity_background(T: _Temperature, rho: float) -> _Temperature:
    """IAPWS R15-11 eqs. 16-17, W/(m K); supports fixed-rho analytic jets."""
    tr, delta = T / 647.096, rho / 322
    dilute = tr**.5 / sum(c / tr**i for i, c in enumerate(WATER_CONDUCTIVITY_DILUTE))
    residual = exponential(delta * sum(c * (1/tr - 1)**i * (delta - 1)**j
                                      for i, row in enumerate(WATER_CONDUCTIVITY_RESIDUAL)
                                      for j, c in enumerate(row)))
    return .001 * dilute * residual


def water_conductivity_iapws(
    T: float, rho: float, *, Cp: float | None = None, Cv: float | None = None,
    mu: float | None = None, drho_dP: float | None = None,
    drho_dP_reference: float | None = None, reference_approximation: bool = False,
) -> float:
    """IAPWS R15-11 W/(m K); mass rho kg/m³, Cp/Cv J/(kg K), mu Pa s.

    Background alone is explicit by omitting all critical inputs. Otherwise
    Cp/Cv/mu/current compressibility are required, together with the reference
    derivative at 970.644 K at the same density or an explicit approximation
    opt-in. No EOS, implicit Cp cap or fabricated critical state is used.
    """
    from chemworld.physchem.viscosity_kernels import _positive

    _positive(T, "temperature")
    _positive(rho, "mass density", zero=True)
    if not isinstance(reference_approximation, bool):
        raise ValueError("reference_approximation must be boolean")
    critical = (Cp, Cv, mu, drho_dP, drho_dP_reference)
    enhancement = 0.0
    if any(v is not None for v in critical) or reference_approximation:
        if any(v is None for v in (Cp, Cv, mu, drho_dP)):
            raise ValueError("water critical enhancement requires Cp, Cv, mu and current drho_dP")
        Cp, Cv = cast(float, Cp), cast(float, Cv)
        mu, drho_dP = cast(float, mu), cast(float, drho_dP)
        for value, label in ((Cp, "mass Cp"), (Cv, "mass Cv"), (mu, "viscosity")):
            _positive(value, label)
        if Cp < Cv:
            raise ValueError("stable-fluid heat capacities require Cp >= Cv")
        if reference_approximation:
            if drho_dP_reference is not None:
                raise ValueError("supply reference derivative or select approximation, not both")
            drho_dP_reference = water_reference_compressibility(rho)
        elif drho_dP_reference is None:
            raise ValueError(
                "supply reference drho_dP or explicitly select reference_approximation")
        xi_nm = water_correlation_length_nm(T, rho, drho_dP, drho_dP_reference)
        y = xi_nm / .4
        # Official release eq. 21; rho=0 also has zero enhancement.
        shape = 0 if y < 1.2e-7 else critical_conductivity_shape(y, rho / 322, Cv / Cp)
        enhancement = .001 * 177.8514 * (rho/322) * (Cp/461.51805) * (T/647.096) / (mu/1e-6) * shape
    try:
        return _positive(water_conductivity_background(T, rho) + enhancement, "water conductivity")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("water conductivity is outside its numerical domain") from exc


def _finite_nonnegative(value: float, label: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, int | float)
            or not isfinite(value) or value < 0):
        raise ValueError(f"{label} must be finite and nonnegative")


def water_reference_compressibility(rho_kg_m3: float) -> float:
    """R15-11 approximate (drho/dP)_T at 970.644 K, kg/(m³ Pa).

    This is the conductivity release's reference-state polynomial, not a
    replacement EOS and not the scientific IAPWS-95 derivative. Selection in
    viscosity requires an explicit reference_approximation=True option.
    """
    _finite_nonnegative(rho_kg_m3, "density")
    delta = rho_kg_m3 / 322
    index = sum(delta > limit for limit in (.310559006, .776397516, 1.242236025, 1.863354037))
    denominator = sum(a * delta**i for i, a in enumerate(_REFERENCE_SUSCEPTIBILITY[index]))
    if denominator <= 0 or not isfinite(denominator):
        raise ValueError("water reference susceptibility polynomial is outside its domain")
    return 322 / 22.064e6 / denominator


def water_correlation_length_nm(
    T: float, rho: float, drho_dP: float, drho_dP_reference: float,
) -> float:
    """Shared viscosity/conductivity correlation length, R12-08 eqs. 20-21.

    Both compressibility inputs are in kg/(m³ Pa); reference T=970.644 K at
    the same density. The nonnegative susceptibility prescription is part
    of the IAPWS standard, not a numerical replacement for invalid inputs.
    """
    for value, label in ((T, "temperature"), (rho, "density"), (drho_dP, "drho_dP"),
                         (drho_dP_reference, "reference drho_dP")):
        _finite_nonnegative(value, label)
    if T == 0:
        raise ValueError("temperature must be positive")
    susceptibility = rho / 322 * (drho_dP - drho_dP_reference * 970.644 / T) * 22.064e6 / 322
    return .13 * (max(0, susceptibility) / .06)**(.630 / 1.239)


def water_viscosity_enhancement(correlation_length_nm: float) -> float:
    """Dimensionless critical multiplier, R12-08 eqs. 14-19; both Y branches."""
    _finite_nonnegative(correlation_length_nm, "correlation length")
    xi = correlation_length_nm
    c, d = xi / 1.9, xi / 1.1
    if xi <= .3817016416:
        y = c * d**5 * (1 - c + c*c - 765 / 504 * d*d) / 5
    else:
        psi = atan(d)  # equivalent acos(1/sqrt(1+d²)), stable at small d
        w = sqrt(abs((c - 1) / (c + 1))) * tan(psi / 2)
        lw = log1p(w) - log1p(-w) if c > 1 else 2 * atan(w)
        y = (sin(3*psi) / 12 - sin(2*psi) / (4*c)
             + (1 - 1.25*c*c) * sin(psi) / (c*c)
             - ((1 - 1.5*c*c) * psi - abs(c*c - 1)**1.5 * lw) / c**3)
    return exp(.068 * y)


def water_viscosity_iapws(
    T: float, rho: float, *, drho_dP: float | None = None,
    drho_dP_reference: float | None = None, reference_approximation: bool = False,
) -> float:
    """IAPWS viscosity, Pa s, from T K and mass density kg/m³.

    No derivatives selects the industrial/background formula. Supplying both
    derivatives selects the scientific critical enhancement. Supplying only
    the current derivative requires explicit approximation opt-in; this last
    choice reproduces the reference library's nonstandard convenience mode.
    """
    _finite_nonnegative(T, "temperature")
    _finite_nonnegative(rho, "density")
    if T == 0:
        raise ValueError("temperature must be positive")
    if not isinstance(reference_approximation, bool):
        raise ValueError("reference_approximation must be boolean")
    if drho_dP is None:
        if drho_dP_reference is not None or reference_approximation:
            raise ValueError("critical enhancement requires the current drho_dP")
        enhancement = 1.0
    else:
        if reference_approximation:
            if drho_dP_reference is not None:
                raise ValueError("supply reference derivative or select approximation, not both")
            drho_dP_reference = water_reference_compressibility(rho)
        elif drho_dP_reference is None:
            raise ValueError(
                "supply reference drho_dP or explicitly select reference_approximation")
        enhancement = water_viscosity_enhancement(
            water_correlation_length_nm(T, rho, drho_dP, drho_dP_reference))
    try:
        result = water_viscosity_background(T, rho) * enhancement
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("IAPWS viscosity outside numerical domain") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError("IAPWS viscosity must be finite positive")
    return result
