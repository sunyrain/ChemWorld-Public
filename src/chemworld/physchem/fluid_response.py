"""Pure fluid response identities and explicit mass/molar/property scales.

SI inputs except molecular weight (g/mol) and API/Baume scales. Derivative
arguments are derivatives of the same declared thermodynamic state; these
identities neither construct an EOS nor establish phase stability.
"""

from math import pi, sqrt
from typing import Literal, cast, overload

from chemworld.physchem.composition_math import _finite
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R

REFERENCE_WATER_DENSITY = 999.0170824078306  # kg/m3 at 60 F, public scale convention
AVOGADRO = 6.02214076e23


@overload
def mass_molar_property(value: None, molecular_weight: float, *, to: str = "molar") -> None: ...


@overload
def mass_molar_property(value: float, molecular_weight: float, *, to: str = "molar") -> float: ...


def mass_molar_property(
    value: float | None, molecular_weight: float, *, to: str = "molar"
) -> float | None:
    """Convert a property per kg <-> per mol; missing input stays missing."""
    mw = _finite(molecular_weight, "MW", positive=True) / 1000
    if to not in {"molar", "mass"}:
        raise ValueError("property target must be molar or mass")
    if value is None:
        return None
    value = _finite(value, "property")
    return value * mw if to == "molar" else value / mw


def density_molar_volume(value: float, molecular_weight: float) -> float:
    """The involution rho[kg/m3] <-> Vm[m3/mol], MW in g/mol."""
    return _finite(molecular_weight, "MW", positive=True) / (
        1000 * _finite(value, "density or molar volume", positive=True)
    )


def velocity_basis(value: float, molecular_weight: float, *, to: str = "molar") -> float:
    """m/s <-> m*sqrt(kg/mol)/s; this is not a molar flow."""
    value = _finite(value, "velocity")
    factor = sqrt(_finite(molecular_weight, "MW", positive=True) / 1000)
    if to not in {"molar", "mass"}:
        raise ValueError("velocity target must be molar or mass")
    return value * factor if to == "molar" else value / factor


def gravity_scale(
    value: float,
    *,
    source: str = "density",
    target: str = "specific_gravity",
    reference_density: float = REFERENCE_WATER_DENSITY,
) -> float:
    """Density, specific gravity, API, heavy/light Baume on one declared reference."""
    value = _finite(value, "gravity scale")
    ref = _finite(reference_density, "reference density", positive=True)
    if source == "density":
        sg = value / ref
    elif source == "specific_gravity":
        sg = value
    elif source == "API":
        sg = 141.5 / _finite(value + 131.5, "API denominator", positive=True)
    elif source == "Baume_heavy":
        sg = 145 / _finite(145 - value, "heavy Baume denominator", positive=True)
    elif source == "Baume_light":
        sg = 140 / _finite(130 + value, "light Baume denominator", positive=True)
    else:
        raise ValueError("unknown source gravity scale")
    _finite(sg, "specific gravity", positive=True)
    if target == "density":
        return sg * ref
    if target == "specific_gravity":
        return sg
    if target == "API":
        return 141.5 / sg - 131.5
    if target == "Baume_heavy":
        return 145 - 145 / sg
    if target == "Baume_light":
        return 140 / sg - 130
    raise ValueError("unknown target gravity scale")


def watson_characterization(boiling_temperature_K: float, specific_gravity: float) -> float:
    """Watson factor uses the cube root of absolute Rankine temperature."""
    return (_finite(boiling_temperature_K, "boiling temperature", positive=True) * 1.8) ** (
        1 / 3
    ) / (_finite(specific_gravity, "specific gravity", positive=True))


def vapor_mass_fraction(vapor_mole_fraction: float, liquid_mw: float, vapor_mw: float) -> float:
    vf = _finite(vapor_mole_fraction, "vapor fraction", nonnegative=True)
    if vf > 1:
        raise ValueError("vapor mole fraction must be at most one")
    liquid = _finite(liquid_mw, "liquid MW", positive=True)
    vapor = _finite(vapor_mw, "vapor MW", positive=True)
    return vf * vapor / ((1 - vf) * liquid + vf * vapor)


def parachor(
    molecular_weight: float, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """Parachor in N**0.25 m**2.75/mol, not its traditional scaled tabulation."""
    mw = _finite(molecular_weight, "MW", positive=True) / 1000
    liquid = _finite(liquid_density, "liquid density", positive=True)
    gas = _finite(gas_density, "gas density", nonnegative=True)
    sigma = _finite(surface_tension, "surface tension", nonnegative=True)
    return mw * sigma**0.25 / _finite(liquid - gas, "density contrast", positive=True)


def molecular_radius_of_gyration(
    molecular_weight: float,
    moment_a: float,
    moment_b: float,
    moment_c: float = 0.0,
    *,
    planar: bool = False,
) -> float:
    """DIPPR convention from principal moments kg*m2, output m (not inertial rms)."""
    mw = _finite(molecular_weight, "MW", positive=True) / 1000
    a = _finite(moment_a, "moment A", positive=True)
    b = _finite(moment_b, "moment B", positive=True)
    if not isinstance(planar, bool):
        raise ValueError("planar must be a boolean physical selector")
    moment = (
        sqrt(a) * sqrt(b)
        if planar
        else 2
        * pi
        * (a ** (1 / 3) * b ** (1 / 3) * _finite(moment_c, "moment C", positive=True) ** (1 / 3))
    )
    return sqrt(moment * AVOGADRO / mw)


def compressibility_factor(temperature_K: float, pressure_Pa: float, molar_volume: float) -> float:
    return (
        _finite(pressure_Pa, "pressure", positive=True)
        * _finite(molar_volume, "molar volume", positive=True)
        / (R * _finite(temperature_K, "temperature", positive=True))
    )


def isobaric_expansion(molar_volume: float, dV_dT: float) -> float:
    return _finite(dV_dT, "dV/dT") / _finite(molar_volume, "molar volume", positive=True)


def isothermal_compressibility(molar_volume: float, dV_dP: float) -> float:
    return -_finite(dV_dP, "dV/dP") / _finite(molar_volume, "molar volume", positive=True)


def heat_capacity_difference(temperature_K: float, dP_dT: float, dP_dV: float) -> float:
    """Cp-Cv = -T*(dP/dT at V)**2/(dP/dV at T); singular stiffness is undefined."""
    t, pt, pv = (
        _finite(temperature_K, "temperature", positive=True),
        _finite(dP_dT, "dP/dT"),
        _finite(dP_dV, "dP/dV"),
    )
    if pv == 0:
        raise ValueError("heat capacity response has singular mechanical stiffness")
    return -t * pt * pt / pv


def joule_thomson(
    temperature_K: float,
    molar_volume: float,
    cp: float,
    *,
    dV_dT: float | None = None,
    expansion: float | None = None,
) -> float:
    """K/Pa from exactly one declared thermal expansion representation."""
    t = _finite(temperature_K, "temperature", positive=True)
    v = _finite(molar_volume, "molar volume", positive=True)
    cp = _finite(cp, "Cp", positive=True)
    if (dV_dT is None) == (expansion is None):
        raise ValueError("supply exactly one of dV_dT or expansion")
    vt = (
        _finite(dV_dT, "dV/dT")
        if dV_dT is not None
        else v * _finite(cast(float, expansion), "expansion")
    )
    return (t * vt - v) / cp


def acoustic_velocity(
    molar_volume: float,
    dP_dV: float,
    cp: float,
    cv: float,
    *,
    molecular_weight: float | None = None,
) -> float:
    """Sound speed in m/s with MW, otherwise m*sqrt(kg/mol)/s."""
    v = _finite(molar_volume, "molar volume", positive=True)
    radicand = -v * v * _finite(dP_dV, "dP/dV") * isentropic_exponent(cp, cv)
    _finite(radicand, "sound speed squared", nonnegative=True)
    return (
        sqrt(radicand)
        if molecular_weight is None
        else velocity_basis(sqrt(radicand), molecular_weight, to="mass")
    )


def isentropic_exponent(cp: float, cv: float) -> float:
    return _finite(cp, "Cp", positive=True) / _finite(cv, "Cv", positive=True)


def isentropic_exponent_pv(
    cp: float, cv: float, molar_volume: float, pressure_Pa: float, dP_dV: float
) -> float:
    return (
        -_finite(molar_volume, "molar volume", positive=True)
        / _finite(pressure_Pa, "pressure", positive=True)
        * isentropic_exponent(cp, cv)
        * _finite(dP_dV, "dP/dV")
    )


def isentropic_exponent_pt(cp: float, pressure_Pa: float, dV_dT: float) -> float:
    denominator = 1 - _finite(pressure_Pa, "pressure", positive=True) * _finite(dV_dT, "dV/dT") / (
        _finite(cp, "Cp", positive=True)
    )
    if denominator == 0:
        raise ValueError("singular PT isentropic exponent")
    return 1 / denominator


def isentropic_exponent_tv(cv: float, molar_volume: float, dP_dT: float) -> float:
    return 1 + _finite(molar_volume, "molar volume", positive=True) * _finite(dP_dT, "dP/dT") / (
        _finite(cv, "Cv", positive=True)
    )


def phase_identification_parameter(
    molar_volume: float, dP_dT: float, dP_dV: float, d2P_dV2: float, d2P_dVdT: float
) -> float:
    v = _finite(molar_volume, "molar volume", positive=True)
    pt, pv = _finite(dP_dT, "dP/dT"), _finite(dP_dV, "dP/dV")
    if pt == 0 or pv == 0:
        raise ValueError("phase identification requires nonsingular first derivatives")
    return v * (_finite(d2P_dVdT, "d2P/dVdT") / pt - _finite(d2P_dV2, "d2P/dV2") / pv)


def phase_from_response(
    d2P_dVdT: float,
    *,
    molar_volume: float | None = None,
    dP_dT: float | None = None,
    dP_dV: float | None = None,
    d2P_dV2: float | None = None,
) -> Literal["s", "l", "g"]:
    """Derivative-based s/l/g classification; not a phase-equilibrium solver."""
    if _finite(d2P_dVdT, "d2P/dVdT") > 0:
        return "s"
    if any(v is None for v in (molar_volume, dP_dT, dP_dV, d2P_dV2)):
        raise ValueError("fluid classification requires all volume and pressure derivatives")
    return (
        "l"
        if phase_identification_parameter(
            cast(float, molar_volume), cast(float, dP_dT), cast(float, dP_dV),
            cast(float, d2P_dV2), d2P_dVdT,
        ) > 1
        else "g"
    )
