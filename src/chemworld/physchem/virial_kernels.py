"""Pure density-form B/C virial equations on the shared derivative arithmetic.

Public mathematical documentation: chemicals 1.5.2 virial; original scientific
references are recorded in NATIVE_CONV22D3A_EXPERIMENT.md. Xiang/Liu constants
were recovered from public calls in the documented basis, then independently
checked on holdouts; they are not claimed to be original-paper verified.
T, Tc: K; Pc: Pa; Vc: m3/mol; dipole: debye. No reference runtime dependency.
"""

from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType

from chemworld.physchem.caloric_math import derivative_values, exponential
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class VirialMethod:
    required: str
    third: bool = False
    integrals: bool = False

    @property
    def property_id(self):
        return ("third" if self.third else "second") + "_virial_coefficient"

    @property
    def unit(self):
        return "m^6/mol^2" if self.third else "m^3/mol"


VIRIAL_METHODS = MappingProxyType(
    {
        "Pitzer_Curl": VirialMethod("Tc Pc omega", integrals=True),
        "Abbott": VirialMethod("Tc Pc omega", integrals=True),
        "Tsonopoulos": VirialMethod("Tc Pc omega", integrals=True),
        "Tsonopoulos_extended": VirialMethod("Tc Pc omega a b", integrals=True),
        "Oconnell_Prausnitz": VirialMethod("Tc Pc omega"),
        "Xiang": VirialMethod("Tc Pc Vc omega"),
        "Meng": VirialMethod("Tc Pc omega a"),
        "Orbey_Vera": VirialMethod("Tc Pc omega", third=True),
        "Liu_Xiang": VirialMethod("Tc Pc Vc omega", third=True),
    }
)

# Published inverse-temperature powers; scientific constants only.
_BASE = ((0, 0.1445), (-1, -0.33), (-2, -0.1385), (-3, -0.0121))
_PITZER_W = ((0, 0.073), (-1, 0.46), (-2, -0.5), (-3, -0.097), (-8, -0.0073))
_TSON_W = ((0, 0.0637), (-2, 0.331), (-3, -0.423), (-8, -0.008))
_XIANG = ((4.553, 4.172, 0.0), (0.02644, 0.075, 16.5), (3.53, 4.297, 0.0))
_LIU = (
    (0.1623538, -0.5390344, 34.22804),
    (0.308744, 1.783526, -74.76559),
    (-0.01790184, -1.055391, 279.922),
    (-0.02789157, 0.09955867, -62.85431),
)


def _real(value, name, *, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or (positive and value <= 0)
        or (nonnegative and value < 0)
    ):
        raise ValueError(f"{name} must be a finite real number in its physical domain")
    return value


def _parameters(method, parameters):
    if method not in VIRIAL_METHODS:
        raise ValueError("unknown pure virial method")
    required = set(VIRIAL_METHODS[method].required.split())
    if set(parameters) != required:
        raise ValueError(f"{method} requires exactly {sorted(required)}")
    return {
        key: _real(value, key, positive=key in {"Tc", "Pc", "Vc"})
        for key, value in parameters.items()
    }


def _terms(method, p):
    w = p["omega"]
    if method == "Abbott":
        return ((0, 0.083 + 0.139 * w), (-1.6, -0.422), (-4.2, -0.172 * w))
    if method in {"Pitzer_Curl", "Oconnell_Prausnitz"}:
        # O'Connell public displayed formula uses four omega terms; its fifth
        # listed coefficient is unused. Do not silently substitute Pitzer.
        ws = (
            _PITZER_W
            if method == "Pitzer_Curl"
            else ((0, 0.073), (-2, 0.46), (-3, -0.5), (-8, -0.097))
        )
        return (*_BASE, *((power, w * c) for power, c in ws))
    if method in {"Tsonopoulos", "Tsonopoulos_extended"}:
        polar = ((-6, p["a"]), (-8, -p["b"])) if method.endswith("extended") else ()
        return (*_BASE, (-8, -0.000607), *((power, w * c) for power, c in _TSON_W), *polar)
    if method == "Meng":
        powers = (0, -1, -2, -3, -8)
        f0 = (0.13356, -0.30252, -0.15668, -0.00724, -0.00022)
        f1 = (0.17404, -0.15581, 0.38183, -0.44044, -0.00541)
        return (*((n, a + w * b) for n, a, b in zip(powers, f0, f1, strict=True)), (-6, p["a"]))
    if method == "Orbey_Vera":
        return (
            (0, 0.01407 - 0.02676 * w),
            (-2.8, 0.02432 + 0.01770 * w),
            (-3, 0.040 * w),
            (-6, -0.003 * w),
            (-10.5, -0.00313 - 0.00228 * w),
        )
    if method == "Liu_Xiang":
        theta = (p["Pc"] * p["Vc"] / (R * p["Tc"]) - 0.29) ** 2
        return tuple(
            (n, a + w * b + theta * c) for n, (a, b, c) in zip((0, -3, -6, -11), _LIU, strict=True)
        )
    raise ValueError("method does not have a power-series representation")


def _formula(method, T, p):
    tr = T / p["Tc"]
    if method == "Xiang":
        theta = (p["Pc"] * p["Vc"] / (R * p["Tc"]) - 0.29) ** 2
        b0, b1, b2 = (a + p["omega"] * b + theta * c for a, b, c in _XIANG)
        # Dimensions require multiplication by Vc; the public rendered
        # fraction is malformed. Characterization confirms this convention.
        return p["Vc"] * (-b0 * tr ** (-0.75) * exponential(b1 * tr**-3) + b2 * tr ** (-0.5))
    scale = (
        p["Vc"] ** 2
        if method == "Liu_Xiang"
        else ((R * p["Tc"] / p["Pc"]) ** (2 if VIRIAL_METHODS[method].third else 1))
    )
    return scale * sum(c * tr**n for n, c in _terms(method, p))


def virial_derivatives(method, temperature_K, parameters, *, order=3):
    """Value and first through requested (at most third) analytic T derivatives.

    All supplied parameters, including polar fits, remain fixed. Negative and
    zero B/C are valid; this is a correlation, not a gas stability certificate.
    """
    _real(temperature_K, "temperature_K", positive=True)
    if isinstance(order, bool) or not isinstance(order, int) or order not in (0, 1, 2, 3):
        raise ValueError("virial derivative order must be 0, 1, 2 or 3")
    p = _parameters(method, parameters)
    values = derivative_values(lambda t: _formula(method, t, p), temperature_K, order=order)
    if any(not isfinite(v) for v in values):
        raise ValueError("nonfinite virial value or derivative")
    return values


def virial_value(method, temperature_K, *, order=0, **parameters):
    """One value/derivative or documented -1/-2 indefinite integral.

    Integral constants are zero in the dimensional-T basis, with log(T/1 K).
    The second primitive contains T*log(T/1 K)-T for a T^-1 term.
    """
    if order in (0, 1, 2, 3):
        return virial_derivatives(method, temperature_K, parameters, order=order)[order]
    p = _parameters(method, parameters)
    _real(temperature_K, "temperature_K", positive=True)
    if order not in (-1, -2) or not VIRIAL_METHODS[method].integrals:
        raise ValueError("method does not declare this integral order")
    T = temperature_K
    value = 0.0
    for n, c in _terms(method, p):
        coefficient = c * p["Tc"] ** (-n)
        if order == -1:
            primitive = log(T) if n == -1 else T ** (n + 1) / (n + 1)
        elif n == -1:
            primitive = T * (log(T) - 1)
        elif n == -2:
            primitive = -log(T)
        else:
            primitive = T ** (n + 2) / ((n + 1) * (n + 2))
        value += coefficient * primitive
    value *= R * p["Tc"] / p["Pc"]
    if not isfinite(value):
        raise ValueError("nonfinite virial integral")
    return value


def virial_fast(method, temperature_K, parameters):
    """Fused value/three-derivative result from the same analytic kernel."""
    return virial_derivatives(method, temperature_K, parameters)


def virial_vector(method, temperature_K, parameter_sets):
    """One tuple of four outputs per explicit parameter set; no broadcasting."""
    return tuple(virial_fast(method, temperature_K, p) for p in parameter_sets)


def virial_matrix(method, temperature_K, parameter_rows):
    """Rectangular grid of explicit pure/pseudo-component parameters.

    This does not infer cross parameters or perform a mixture rule.
    """
    rows = tuple(tuple(row) for row in parameter_rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("virial matrix requires nonempty rectangular parameter rows")
    return tuple(virial_vector(method, temperature_K, row) for row in rows)


TSONOPOULOS_CLASSES = (
    "simple",
    "normal",
    "ketone",
    "aldehyde",
    "alkyl nitrile",
    "ether",
    "carboxylic acid",
    "ester",
    "alkyl halide",
    "mercaptan",
    "sulfide",
    "disulfide",
    "alkanol",
    "methanol",
    "water",
)


def tsonopoulos_polar_parameters(*, Tc, Pc, species_type, dipole=None, convention="documented"):
    """Explicit class, no chemical-name inference; dipole is in debye.

    Documented methanol constants and the alkanol zero-dipole limit differ
    from chemicals 1.5.2 calls. Select that API convention explicitly when
    compatibility is required. The documented halide mu_r^4 is retained;
    large values are not an assertion of accuracy for a material.
    """
    _real(Tc, "Tc", positive=True)
    _real(Pc, "Pc", positive=True)
    if species_type not in TSONOPOULOS_CLASSES:
        raise ValueError("unknown Tsonopoulos species_type")
    if convention not in {"documented", "chemicals_1_5_2"}:
        raise ValueError("unknown polar convention")
    if dipole is not None:
        _real(dipole, "dipole", nonnegative=True)
    if species_type in {"simple", "normal"}:
        return {"a": 0.0, "b": 0.0}
    if species_type == "water":
        return {"a": -0.0109, "b": 0.0}
    if species_type == "methanol":
        return {"a": 0.0878, "b": 0.0525} if convention == "documented" else {"a": 0.0, "b": 0.0}
    if dipole is None:
        raise ValueError("this polar class requires an explicit dipole")
    if dipole == 0 and convention == "chemicals_1_5_2":
        return {"a": 0.0, "b": 0.0}
    mu = 1e5 * dipole**2 * (Pc / 101325) / Tc**2
    if species_type == "alkanol":
        return {"a": 0.0878, "b": 0.00908 + 0.0006957 * mu}
    a = (
        -2.188e-4 * mu**4 - 7.831e-21 * mu**8
        if species_type in {"alkyl halide", "mercaptan", "sulfide", "disulfide"}
        else -2.14e-4 * mu - 4.308e-21 * mu**8
    )
    return {"a": a, "b": 0.0}


def meng_polar_parameter(*, Tc, Pc, dipole, haloalkane):
    _real(Tc, "Tc", positive=True)
    _real(Pc, "Pc", positive=True)
    _real(dipole, "dipole", nonnegative=True)
    if not isinstance(haloalkane, bool):
        raise ValueError("haloalkane requires an explicit boolean")
    mu = 1e5 * dipole**2 * (Pc / 101325) / Tc**2
    c = (
        (-1.1524e-6, 7.2238e-11, -1.8701e-15)
        if haloalkane
        else (-3.0309e-6, 9.503e-11, -1.2469e-15)
    )
    return sum(a * mu**n for a, n in zip(c, (2, 4, 6), strict=True))
