"""Published critical-property estimates, in K, Pa and m³/mol.

These empirical estimates do not locate a thermodynamic critical point.
Equations come from public documentation and cited literature. For
Hekayati-Raeissi, mathematical expressions and scientific coefficients were
read from chemicals 1.5.2's public implementation (paper not directly checked).
The native algebra and validation below are independently implemented.
"""

from math import fsum, isclose, isfinite, log, log10
from types import MappingProxyType

from chemworld.foundation.gas import GAS_R
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.vapor_pressure_kernels import pressure_logarithm

BOILING_CRITICAL_FITS = MappingProxyType({
    "Perry8E": 1.64,
    "Chemicals2021FitInorganic": 1.606,
    "Chemicals2021FitOrganic": 1.453,
    "Chemicals2021FitElements": 1.748,
    "Chemicals2021FitBinary": 1.568,
    "Chemicals2021FitTernary": 1.601,
    "Chemicals2021FitCl": 1.558,
    "Chemicals2021FitN": 1.616,
    "Chemicals2021FitC": 1.611,
    "Chemicals2021FitF": 1.573,
    "Chemicals2021FitO": 1.613,
    "Chemicals2021FitI": 1.452,
    "Chemicals2021FitBr": 1.564,
    "Chemicals2021FitSi": 1.526,
})
MERSMANN_KIND_RADII_ANGSTROM = MappingProxyType({
    "C": .77, "Cl": .99, "I": 1.33, "H": .37, "F": .71,
    "S": 1.04, "O": .6, "N": .71, "Si": 1.17, "Br": 1.14,
})
MERSMANN_KIND_REGRESSED_RADII_ANGSTROM = MappingProxyType({
    "Nb": .5139380605234125, "Ne": .7708216694154189,
    "Al": 1.004994775098707, "Re": 1.1164444694484814,
    "Rb": 2.9910506044828837, "Rn": 1.9283158156480653,
    "Xe": 1.694221043013319, "Ta": 1.1185133195453156,
    "Bi": 1.8436438207262267, "Br": 1.3081458724155532,
    "Hf": .8829545460486594, "Mo": .740396259301556,
    "He": .9808144122544257, "C": .6068586007600608,
    "B": .7039677272439753, "F": .5409105884533288,
    "I": 1.7262432419406561, "H": .33296601702348533,
    "K": .7384112258842432, "O": .5883254088243008,
    "N": .5467979701131293, "P": 1.0444655158949694,
    "Si": 1.4181434041348049, "U": 1.5530287578073485,
    "Sn": 1.3339487990207999, "W": .8355335838735266,
    "V": .6714619384794069, "Sb": .8840680681215854,
    "Se": 1.5747549515496795, "Ge": 1.0730584829731715,
    "Kr": 1.393999829252709, "Cl": 1.0957835025011224,
    "S": 1.0364452121761167, "Hg": .6750818243474633,
    "As": .6750687692915264, "Ar": 1.2008872952022298,
    "Cs": 3.433699060142929, "Zr": .9346554283483623,
})


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _positive(value, name):
    if _finite(value, name) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _saturation_ratio(pressure, critical_pressure, reduced_temperature, required):
    _positive(pressure, "saturation pressure")
    _positive(critical_pressure, "critical pressure")
    if pressure >= critical_pressure:
        raise ValueError("saturation pressure must be below critical pressure")
    _finite(reduced_temperature, "reduced temperature")
    if not isclose(reduced_temperature, required, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"this definition requires saturation pressure at {required} Tc")
    return log10(pressure) - log10(critical_pressure)


def acentric_factor_from_pressure(Psat, Pc, *, reduced_temperature):
    """Pitzer definition; the caller must identify the 0.7 Tc pressure point."""
    return -_saturation_ratio(Psat, Pc, reduced_temperature, .7) - 1


def lee_kesler_acentric_factor(Tb, Tc, Pc):
    """Infer omega at the normal boiling point using the existing LK pressure kernel."""
    _positive(Tb, "normal boiling temperature")
    _positive(Tc, "critical temperature")
    _positive(Pc, "critical pressure")
    if Tb >= Tc or Pc <= 101325:
        raise ValueError("Lee-Kesler normal boiling estimate requires Tb < Tc and Pc > 1 atm")
    intercept, slope = derivative_values(
        lambda omega: pressure_logarithm(
            "lee_kesler_pressure", Tb, {"Tc": Tc, "Pc": Pc, "omega": omega}
        ), 0.0, order=1,
    )
    if slope == 0:
        raise ValueError("Lee-Kesler estimate has a singular omega coefficient")
    return _finite((log(101325) - intercept) / slope, "acentric estimate")


def stiel_polar_factor(Psat, Pc, omega, *, reduced_temperature):
    """Stiel factor using the explicitly identified saturation point at 0.6 Tc."""
    _finite(omega, "acentric factor")
    return _saturation_ratio(Psat, Pc, reduced_temperature, .6) + 1.70 * omega + 1.552


def critical_surface(*, method, Tc=None, Pc=None, Vc=None):
    """Solve the missing one of Tc/Pc/Vc; exactly two positive inputs required.

    Meissner uses the chemicals convention 20.8 atm cm³/(mol K), matching
    its public numerical example. The separately selectable Meissner_MPa
    uses 2.08 MPa cm³/(mol K), as in Martínez-Vitela & Gracia-Fadrique
    (2019), eq. 3. These differ by 1.325%; no unit label hides that choice.
    """
    surfaces = {
        "Ihmels": (2.215, -25000.0, 0.0),
        "Grigoras": (2.02, 290000.0, 0.0),
        "Meissner": (20.8 * 101325e-6, 0.0, 8e-6),
        "Meissner_MPa": (2.08, 0.0, 8e-6),
    }
    if method not in surfaces:
        raise ValueError(f"unknown critical surface {method!r}")
    values = {"Tc": Tc, "Pc": Pc, "Vc": Vc}
    if sum(value is not None for value in values.values()) != 2:
        raise ValueError("critical surface requires exactly two of Tc, Pc, Vc")
    for key, value in values.items():
        if value is not None:
            _positive(value, key)
    slope, pressure_offset, volume_offset = surfaces[method]
    # A common affine surface (Pc-p0)*(Vc-v0)=a*Tc; all units SI here.
    if Vc is not None and Vc <= volume_offset:
        raise ValueError("critical volume is outside this surface's positive-temperature domain")
    if Pc is not None and Pc <= pressure_offset:
        raise ValueError("critical pressure is outside this surface's positive-temperature domain")
    if Tc is None:
        Tc = (Pc - pressure_offset) * (Vc - volume_offset) / slope
    elif Pc is None:
        Pc = pressure_offset + slope * Tc / (Vc - volume_offset)
    else:
        Vc = volume_offset + slope * Tc / (Pc - pressure_offset)
    for key, value in (("Tc", Tc), ("Pc", Pc), ("Vc", Vc)):
        _positive(value, key)
    return Tc, Pc, Vc


def critical_compressibility(Tc, Pc, Vc):
    """Definition Zc=Pc*Vc/(R*Tc), with explicit SI inputs, not a table value."""
    for name, value in (("Tc", Tc), ("Pc", Pc), ("Vc", Vc)):
        _positive(value, name)
    return _positive(Pc * Vc / (GAS_R * Tc), "critical compressibility")


def hekayati_raeissi_critical(*, MW, V_sat=None, Tc=None, Pc=None, Vc=None,
                             saturation_temperature_K=298.15):
    """Return (Tc [K], Pc [Pa], Vc [m³/mol]); MW is in g/mol.

    Hekayati & Raeissi, doi:10.1002/aic.17004, equations 21--33 as
    transcribed in chemicals 1.5.2. V_sat denotes saturated *liquid*
    volume at 298.15 K. Supplied Vc selects the critical-volume model;
    otherwise V_sat selects both the volume and Tc/Pc fits. These are
    different fits: feeding the estimated Vc back in changes the model.

    At least one of Tc/Pc and one of Vc/V_sat is required. All supplied
    critical values are preserved, including when both Tc and Pc are
    given; this is not a consistency fit of redundant inputs. All given
    values must be finite and positive, even an unused V_sat. Positive
    algebraic results do not establish a compound's fitted validity range.
    """
    _positive(MW, "MW")
    for name, value in (("Tc", Tc), ("Pc", Pc), ("Vc", Vc), ("V_sat", V_sat)):
        if value is not None:
            _positive(value, name)
    if Tc is None and Pc is None:
        raise ValueError("Hekayati-Raeissi requires Tc or Pc")
    if Vc is None and V_sat is None:
        raise ValueError("Hekayati-Raeissi requires Vc or V_sat")
    if V_sat is not None:
        _finite(saturation_temperature_K, "saturation temperature")
        if not isclose(saturation_temperature_K, 298.15, rel_tol=0, abs_tol=1e-10):
            raise ValueError("V_sat must be saturated liquid volume at 298.15 K")

    use_saturation = Vc is None
    # The fitted volume coordinate is m³/kmol (numerically L/mol).
    # Expand into monomials instead of reproducing upstream intermediate
    # coefficient vectors/branch logic. fsum limits cancellation on summation.
    try:
        if use_saturation:
            v = 1000 * V_sat
            Vc = 1e-3 * fsum((
                -2.636518 * v**3, .01048744 * MW * v, 1.222756 * v,
                6.924349e-9 * MW**3, -1.102846e-5 * MW**2, .001871249 * MW,
            ))
            _positive(Vc, "estimated critical volume")
        if Tc is None or Pc is None:
            if use_saturation:
                factor = fsum((
                    -143.4646 * v**3, .1781802 * MW * v**2, 41.96576 * v**2,
                    -3.443955e-5 * MW**2, .009812262 * MW, .1106396,
                ))
            else:
                v = 1000 * Vc
                factor = fsum((
                    -.001822904 * MW * v**2, .00001947108 * MW**2 * v,
                    3.794091 * v, -2.472981e-8 * MW**3,
                ))
            _positive(factor, "critical temperature/pressure fit factor")
            if Tc is None:
                Tc = Pc * factor / (1000 * GAS_R)
            else:
                Pc = 1000 * GAS_R * Tc / factor
    except OverflowError as exc:
        raise ValueError("Hekayati-Raeissi inputs exceed the numeric fit domain") from exc
    for name, value in (("Tc", Tc), ("Pc", Pc), ("Vc", Vc)):
        _positive(value, name)
    if use_saturation and Tc <= 298.15:
        raise ValueError("saturated liquid at 298.15 K requires Tc above 298.15 K")
    return Tc, Pc, Vc


def boiling_critical_temperature(*, fit, Tb=None, Tc=None):
    """Return (Tb, Tc) using an explicitly selected public linear fit."""
    if fit not in BOILING_CRITICAL_FITS:
        raise ValueError(f"unknown boiling/critical fit {fit!r}")
    if (Tb is None) == (Tc is None):
        raise ValueError("provide exactly one of Tb and Tc")
    factor = BOILING_CRITICAL_FITS[fit]
    if Tb is None:
        Tb = _positive(Tc, "Tc") / factor
    else:
        Tc = _positive(Tb, "Tb") * factor
    return _positive(Tb, "Tb"), _positive(Tc, "Tc")


def mersmann_kind_volume(atoms, *, coefficient=3.645, power=.5, radii=None):
    """Mersmann-Kind volume from atom counts; covalent radii are in angstrom.

    Caller-selected radii must include H, which defines the reference length.
    The regressed radius table requires explicitly selecting its coefficient
    4.261206523632586 and power .5597281770786228 as well.
    """
    radii = MERSMANN_KIND_RADII_ANGSTROM if radii is None else radii
    _positive(coefficient, "coefficient")
    _finite(power, "power")
    if not atoms:
        raise ValueError("molecular composition cannot be empty")
    if "H" not in radii:
        raise ValueError("Mersmann-Kind requires the hydrogen reference radius")
    _positive(radii["H"], "hydrogen radius")
    count, weighted_radius = 0.0, 0.0
    for element, number in atoms.items():
        _positive(number, "atom count")
        if element not in radii:
            raise ValueError(f"missing covalent radius for {element}")
        count += number
        weighted_radius += number * _positive(radii[element], f"radius {element}")
    mean = weighted_radius / count
    diameter_m = 2e-10 * mean * coefficient / (mean / radii["H"]) ** power
    return _positive(count * 6.02214076e23 * diameter_m**3, "critical volume estimate")


def _vector(values, n, name):
    if values is None or len(values) != n:
        raise ValueError(f"{name} must have one value per component")
    return tuple(_positive(v, name) for v in values)


def _matrix(values, n):
    if values is None or len(values) != n or any(len(row) != n for row in values):
        raise ValueError("a complete explicit interaction matrix is required")
    return tuple(tuple(_finite(v, "interaction") for v in row) for row in values)


def mixture_critical_property(method, fractions, *, Tcs=None, Vcs=None, interactions=None):
    """Six pseudocritical rules; fractions must already sum to one.

    Chueh-Prausnitz Tc interactions are temperature increments in K
    (equivalently published dimensionless tau times Tref=1 K); its Vc
    interactions are cm³/mol. Wilson/Grieves interactions are dimensionless,
    with no symmetry assumption. Wilson's fitted correction is -2500 K
    or -2500 cm³/mol, respectively. Every matrix entry must be provided.
    """
    methods = {"Li", "Chueh_Prausnitz_Tc", "Chueh_Prausnitz_Vc",
               "Grieves_Thodos", "modified_Wilson_Tc", "modified_Wilson_Vc"}
    if method not in methods:
        raise ValueError(f"unknown mixture critical method {method!r}")
    x = tuple(_finite(v, "mole fraction") for v in fractions)
    n = len(x)
    if not n or min(x) < 0 or not isclose(sum(x), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("mole fractions must be nonnegative and sum to one")
    is_volume = method.endswith("_Vc")
    props = _vector(Vcs if is_volume else Tcs, n, "Vc" if is_volume else "Tc")
    if method == "Li" or method.startswith("Chueh_Prausnitz"):
        volumes = _vector(Vcs, n, "Vc")
        exponent = 1 if method == "Li" else 2 / 3
        weights = tuple(xi * vi**exponent for xi, vi in zip(x, volumes, strict=True))
        normalizer = sum(weights)
        weights = tuple(w / normalizer for w in weights)
    else:
        weights = x
    if method == "Li":
        if interactions is not None:
            raise ValueError("Li has no interaction parameter")
        result = sum(w * p for w, p in zip(weights, props, strict=True))
    else:
        matrix = _matrix(interactions, n)
        if method.startswith("Chueh_Prausnitz"):
            correction = sum(weights[i] * weights[j] * matrix[i][j]
                             for i in range(n) for j in range(n))
            result = sum(w * p for w, p in zip(weights, props, strict=True))
            result += correction * (1e-6 if is_volume else 1)
        else:
            # Zero-composition terms have zero weight, even if their unused
            # log/quotient would be singular. No clipping of active terms.
            result = 0.0
            for i in range(n):
                if x[i] == 0:
                    continue
                denominator = x[i] + sum(matrix[i][j] * x[j] for j in range(n))
                _positive(denominator, "mixture interaction denominator/log argument")
                if method == "Grieves_Thodos":
                    result += x[i] * props[i] / denominator
                else:
                    result += x[i] * (props[i] - 2500 * (1e-6 if is_volume else 1)
                                      * log(denominator))
    return _positive(result, "pseudocritical estimate")
