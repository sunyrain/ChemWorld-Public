"""Lemmon-Jacobsen (2004) air transport; molar density, no private air EOS.

doi:10.1023/B:IJOT.0000022327.04529.f3. Temperature K, density mol/m³,
Cp/Cv J/(mol K), compressibilities mol/(m³ Pa), viscosity Pa s.
"""

from math import pi

from chemworld.physchem.caloric_math import exponential, logarithm
from chemworld.physchem.conductivity_critical import critical_conductivity_shape


def air_viscosity_background(T, rho):
    """Shared existing Lemmon viscosity equation, supporting analytic jets."""
    tr, delta = 132.6312 / T, rho / 10447.7
    lt = logarithm(T / 103.3)
    collision = exponential(
        sum(b * lt**i for i, b in enumerate((0.431, -0.4623, 0.08406, 0.005341, -0.00331)))
    )
    dilute = 0.0266958 * (28.9586 * T) ** 0.5 / (0.36**2 * collision)
    residual = sum(
        n * tr**t * delta**d * exponential(-g * delta**ell)
        for n, t, d, g, ell in (
            (10.72, 0.2, 1, 0, 0),
            (1.122, 0.05, 4, 0, 0),
            (0.002019, 2.4, 9, 0, 0),
            (-8.876, 0.6, 1, 1, 1),
            (-0.02916, 3.6, 8, 1, 1),
        )
    )
    return (dilute + residual) * 1e-6


def air_conductivity_background(T, rho):
    """Dilute/residual terms W/(m K); fixed-rho analytic jets supported."""
    tau, delta = 132.6312 / T, rho / 10447.7
    dilute = 1.308 * air_viscosity_background(T, 0) / 1e-6 + 1.405 * tau**-1.1 - 1.036 * tau**-0.3
    residual = sum(
        n * tau**t * delta**d * exponential(-g * delta**ell)
        for n, t, d, g, ell in (
            (8.743, 0.1, 1, 0, 0),
            (14.76, 0, 2, 0, 0),
            (-16.62, 0.5, 3, 1, 2),
            (3.793, 2.7, 7, 1, 2),
            (-6.142, 0.3, 7, 1, 2),
            (-0.3778, 1.3, 11, 1, 2),
        )
    )
    return (dilute + residual) * 0.001


def air_conductivity_lemmon(
    T,
    rho,
    *,
    Cp=None,
    Cv=None,
    mu=None,
    drho_dP=None,
    drho_dP_reference=None,
):
    """W/(m K); critical enhancement requires all five explicit state inputs.

    Reference compressibility is at 265.262 K and the SAME molar density.
    Historical model constants kB=1.380658e-23, Gamma=.055 are retained; this
    is not an updated fit using a new Boltzmann constant. Negative excess
    susceptibility selects zero enhancement; invalid state inputs reject.
    """
    from chemworld.physchem.viscosity_kernels import _positive

    _positive(T, "temperature")
    _positive(rho, "molar density", zero=True)
    critical = (Cp, Cv, mu, drho_dP, drho_dP_reference)
    enhancement = 0.0
    if any(v is not None for v in critical):
        if any(v is None for v in critical):
            raise ValueError("air critical enhancement requires Cp, Cv, mu and both drho_dP inputs")
        for value, label in ((Cp, "molar Cp"), (Cv, "molar Cv"), (mu, "viscosity")):
            _positive(value, label)
        for value in (drho_dP, drho_dP_reference):
            _positive(value, "molar density-pressure derivative", zero=True)
        if Cp < Cv:
            raise ValueError("stable-fluid heat capacities require Cp >= Cv")
        excess = 3.78502e6 * rho / 10447.7**2 * (drho_dP - drho_dP_reference * 265.262 / T)
        xi = 0.11e-9 * (max(0, excess) / 0.055) ** (0.63 / 1.2415)
        shape = critical_conductivity_shape(xi / 0.31e-9, rho / 10447.7, Cv / Cp)
        enhancement = rho * Cp * 1.380658e-23 * 1.01 * T / (6 * pi * 0.31e-9 * mu) * shape
    try:
        return _positive(air_conductivity_background(T, rho) + enhancement, "air conductivity")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("air conductivity is outside its numerical domain") from exc
