"""Engineering similarity groups and SI conversions with explicit inputs.

The group selector retains conventional engineering symbols. All inputs are
SI: length m, time s, velocity m/s, density kg/m3, dynamic viscosity Pa s,
kinematic viscosity/diffusivity m2/s, pressure Pa, conductivity W/(m K),
specific heat J/(kg K), surface tension N/m, heat flux W/m2 and power W.
K denotes permeability for Bejan_p, mass transfer velocity for Sherwood.
These algebraic groups do not certify the correlation that consumes them.
"""

from __future__ import annotations

from math import isclose, isfinite, pi, sin, sqrt
from typing import Any

from chemworld.physchem.transport import (
    GRAVITY_M_S2,
    FluidState,
    peclet_heat_number,
    prandtl_number,
    reynolds_number,
)

ENGINEERING_GROUP_PARAMETERS = {
    "Archimedes": "L rhof rhop mu g",
    "Bejan_L": "dP L mu alpha",
    "Bejan_p": "dP K mu alpha",
    "Biot": "h L k",
    "Boiling": "G q Hvap",
    "Bond": "rhol rhog sigma L g",
    "Capillary": "V mu sigma",
    "Cavitation": "P Psat rho V",
    "Confinement": "D rhol rhog sigma g",
    "Dean": "Re Di D",
    "Drag": "F A V rho",
    "Eckert": "V Cp dT",
    "Eotvos": "rhol rhog sigma L g",
    "Euler": "dP rho V",
    "Fourier_heat": "t L rho Cp k alpha",
    "Fourier_mass": "t L D",
    "Froude": "V L g squared",
    "Froude_densimetric": "V L rho1 rho2 heavy g",
    "Graetz_heat": "V D x rho Cp k alpha",
    "Grashof": "L beta T1 T2 rho mu nu g",
    "Hagen": "Re fd",
    "Jakob": "Cp Hvap Te",
    "Knudsen": "path L",
    "Lewis": "D alpha Cp k rho",
    "Mach": "V c",
    "Morton": "rhol rhog mul sigma g",
    "Nusselt": "h L k",
    "Ohnesorge": "L rho mu sigma",
    "Peclet_heat": "V L rho Cp k alpha",
    "Peclet_mass": "V L D",
    "Power_number": "P L N rho",
    "Prandtl": "Cp k mu nu rho alpha",
    "Rayleigh": "Pr Gr",
    "Reynolds": "V D rho mu nu",
    "Schmidt": "D mu nu rho",
    "Sherwood": "K L D",
    "Stanton": "h V rho Cp",
    "Stokes_number": "V Dp D rhop mu",
    "Strouhal": "f L V",
    "Suratman": "L rho mu sigma",
    "Weber": "V L rho sigma",
}


def _value(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"required engineering parameter: {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"finite numeric parameter required: {key}")
    return float(value)


def _positive(data: dict[str, Any], key: str) -> float:
    value = _value(data, key)
    if value <= 0:
        raise ValueError(f"positive parameter required: {key}")
    return value


def thermal_diffusivity_si(
    conductivity_W_m_K: float, density_kg_m3: float, heat_capacity_J_kg_K: float
) -> float:
    values = {"k": conductivity_W_m_K, "rho": density_kg_m3, "Cp": heat_capacity_J_kg_K}
    return _positive(values, "k") / (_positive(values, "rho") * _positive(values, "Cp"))


def convert_viscosity(
    *, density_kg_m3: float, dynamic_Pa_s: float | None = None, kinematic_m2_s: float | None = None
) -> float:
    """Exactly one viscosity input; return the other using explicit current density."""
    rho = _positive({"rho": density_kg_m3}, "rho")
    if (dynamic_Pa_s is None) == (kinematic_m2_s is None):
        raise ValueError("exactly one dynamic or kinematic viscosity required")
    if dynamic_Pa_s is not None:
        return _positive({"mu": dynamic_Pa_s}, "mu") / rho
    return _positive({"nu": kinematic_m2_s}, "nu") * rho


def _kinematic(values: dict[str, Any]) -> float:
    if "mu" in values and "rho" in values:
        result = convert_viscosity(density_kg_m3=values["rho"], dynamic_Pa_s=values["mu"])
        if "nu" in values and not isclose(result, _positive(values, "nu"), rel_tol=1e-12):
            raise ValueError("inconsistent dynamic and kinematic viscosity inputs")
        return result
    return _positive(values, "nu")


def _thermal(values: dict[str, Any]) -> float:
    if {"rho", "Cp", "k"}.issubset(values):
        result = thermal_diffusivity_si(values["k"], values["rho"], values["Cp"])
        if "alpha" in values and not isclose(result, _positive(values, "alpha"), rel_tol=1e-12):
            raise ValueError("inconsistent thermal diffusivity inputs")
        return result
    return _positive(values, "alpha")


def engineering_group(method: str, **parameters: Any) -> float:
    """Evaluate a named group; allowed conventional symbols are publicly declared.

    Velocity is a nonnegative magnitude. Pressure differences, expansivity and
    density contrasts retain signs. Euler uses rho*V2, drag/cavitation use
    rho*V2/2. Grashof uses absolute temperature difference, signed beta.
    """
    if method not in ENGINEERING_GROUP_PARAMETERS:
        raise ValueError("unknown engineering similarity group")
    if not parameters.keys() <= set(ENGINEERING_GROUP_PARAMETERS[method].split()):
        raise ValueError("unknown parameter for this engineering group")
    p = {k: v for k, v in parameters.items() if v is not None}
    for key in p:
        if key in {"heavy", "squared"}:
            if not isinstance(p[key], bool):
                raise ValueError("heavy/squared choices must be boolean")
        else:
            _value(p, key)
    for key in {"V", "t", "Re", "fd"} & p.keys():
        if p[key] < 0:
            raise ValueError(f"nonnegative engineering magnitude required: {key}")
    def v(key: str) -> float:
        return _value(p, key)

    def pos(key: str) -> float:
        return _positive(p, key)
    gravity = _positive({"g": p.get("g", GRAVITY_M_S2)}, "g")
    if method == "Reynolds":
        return reynolds_number(
            density_kg_m3=1, velocity_m_s=v("V"), diameter_m=pos("D"), viscosity_Pa_s=_kinematic(p)
        )
    if method == "Prandtl":
        if {"Cp", "k", "mu"}.issubset(p):
            return prandtl_number(
                heat_capacity_J_kg_K=pos("Cp"),
                viscosity_Pa_s=pos("mu"),
                thermal_conductivity_W_m_K=pos("k"),
            )
        return _kinematic(p) / _thermal(p)
    if method in ("Fourier_heat", "Fourier_mass"):
        diffusivity = _thermal(p) if method == "Fourier_heat" else pos("D")
        return v("t") * diffusivity / pos("L") ** 2
    if method in ("Peclet_heat", "Peclet_mass"):
        diffusivity = _thermal(p) if method == "Peclet_heat" else pos("D")
        return v("V") * pos("L") / diffusivity
    if method == "Graetz_heat":
        return v("V") * pos("D") ** 2 / (pos("x") * _thermal(p))
    if method == "Lewis":
        return _thermal(p) / pos("D")
    if method == "Schmidt":
        return _kinematic(p) / pos("D")
    if method == "Grashof":
        return (
            gravity * v("beta") * abs(v("T1") - p.get("T2", 0)) * pos("L") ** 3 / _kinematic(p) ** 2
        )
    if method in ("Biot", "Nusselt"):
        return v("h") * pos("L") / pos("k")
    if method in ("Bond", "Eotvos"):
        return gravity * (pos("rhol") - pos("rhog")) * pos("L") ** 2 / pos("sigma")
    if method == "Archimedes":
        return gravity * pos("L") ** 3 * pos("rhof") * (pos("rhop") - pos("rhof")) / pos("mu") ** 2
    if method in ("Bejan_L", "Bejan_p"):
        area = pos("L") ** 2 if method == "Bejan_L" else pos("K")
        return v("dP") * area / (pos("mu") * pos("alpha"))
    if method == "Boiling":
        return v("q") / (pos("G") * pos("Hvap"))
    if method == "Capillary":
        return v("V") * pos("mu") / pos("sigma")
    if method == "Cavitation":
        return 2 * (v("P") - v("Psat")) / (pos("rho") * pos("V") ** 2)
    if method in ("Confinement", "Froude_densimetric"):
        difference = (
            pos("rhol") - pos("rhog") if method == "Confinement" else pos("rho1") - pos("rho2")
        )
        if difference <= 0:
            raise ValueError("positive heavy-minus-light density contrast required")
        if method == "Confinement":
            return sqrt(pos("sigma") / (gravity * difference)) / pos("D")
        density = pos("rho1") if p.get("heavy", True) else pos("rho2")
        return v("V") * sqrt(density / (gravity * pos("L") * difference))
    if method == "Dean":
        return v("Re") * sqrt(pos("Di") / pos("D"))
    if method == "Drag":
        return 2 * v("F") / (pos("A") * pos("rho") * pos("V") ** 2)
    if method == "Eckert":
        if v("dT") == 0:
            raise ValueError("nonzero temperature difference required")
        return v("V") ** 2 / (pos("Cp") * v("dT"))
    if method == "Euler":
        return v("dP") / (pos("rho") * pos("V") ** 2)
    if method == "Froude":
        result = v("V") / sqrt(pos("L") * gravity)
        return result**2 if p.get("squared", False) else result
    if method == "Hagen":
        return v("fd") * v("Re") ** 2 / 2
    if method == "Jakob":
        return pos("Cp") * v("Te") / pos("Hvap")
    if method == "Knudsen":
        return pos("path") / pos("L")
    if method == "Mach":
        return v("V") / pos("c")
    if method == "Morton":
        return (
            gravity
            * pos("mul") ** 4
            * (pos("rhol") - pos("rhog"))
            / (pos("rhol") ** 2 * pos("sigma") ** 3)
        )
    if method == "Ohnesorge":
        return pos("mu") / sqrt(pos("L") * pos("rho") * pos("sigma"))
    if method == "Power_number":
        return v("P") / (pos("rho") * pos("N") ** 3 * pos("L") ** 5)
    if method == "Rayleigh":
        return pos("Pr") * v("Gr")
    if method == "Sherwood":
        return v("K") * pos("L") / pos("D")
    if method == "Stanton":
        return v("h") / (pos("V") * pos("rho") * pos("Cp"))
    if method == "Stokes_number":
        return pos("rhop") * v("V") * pos("Dp") ** 2 / (18 * pos("mu") * pos("D"))
    if method == "Strouhal":
        return v("f") * pos("L") / pos("V")
    if method == "Suratman":
        return pos("rho") * pos("sigma") * pos("L") / pos("mu") ** 2
    if method == "Weber":
        return pos("rho") * v("V") ** 2 * pos("L") / pos("sigma")
    raise AssertionError(method)


def current_fluid_similarity(
    fluid: FluidState,
    *,
    velocity_m_s: float,
    length_m: float,
    mass_diffusivity_m2_s: float | None = None,
) -> dict[str, float]:
    """Original Re/Pr/Pe consumers plus optional mass groups from one current state."""
    reynolds = reynolds_number(
        density_kg_m3=fluid.density_kg_m3,
        velocity_m_s=velocity_m_s,
        diameter_m=length_m,
        viscosity_Pa_s=fluid.viscosity_Pa_s,
    )
    prandtl = prandtl_number(
        heat_capacity_J_kg_K=fluid.heat_capacity_J_kg_K,
        viscosity_Pa_s=fluid.viscosity_Pa_s,
        thermal_conductivity_W_m_K=fluid.thermal_conductivity_W_m_K,
    )
    alpha = thermal_diffusivity_si(
        fluid.thermal_conductivity_W_m_K, fluid.density_kg_m3, fluid.heat_capacity_J_kg_K
    )
    result = {
        "Reynolds": reynolds,
        "Prandtl": prandtl,
        "Peclet_heat": peclet_heat_number(reynolds=reynolds, prandtl=prandtl),
        "thermal_diffusivity_m2_s": alpha,
    }
    if mass_diffusivity_m2_s is not None:
        diffusivity = _positive({"D": mass_diffusivity_m2_s}, "D")
        result.update(
            {
                "Schmidt": fluid.viscosity_Pa_s / (fluid.density_kg_m3 * diffusivity),
                "Lewis": alpha / diffusivity,
                "Peclet_mass": velocity_m_s * length_m / diffusivity,
            }
        )
    return result


def engineering_resistance_conversion(method: str, **values: float) -> float:
    """Darcy f, K, length, pressure and head in SI; no Fanning-factor substitution."""
    supported = {
        "K_from_L_equiv": "L_D fd",
        "K_from_f": "fd L D",
        "L_equiv_from_K": "K fd",
        "L_from_K": "K D fd",
        "f_from_K": "K L D",
        "dP_from_K": "K rho V",
        "head_from_K": "K V g",
        "head_from_P": "P rho g",
        "P_from_head": "head rho g",
        "relative_roughness": "D roughness",
    }
    if method not in supported or not values.keys() <= set(supported[method].split()):
        raise ValueError("known resistance conversion and its declared parameters required")
    p = {"fd": 0.015, "g": GRAVITY_M_S2, "roughness": 1.52e-6, **values}
    for key in values:
        if _value(p, key) < 0:
            raise ValueError("nonnegative resistance/head magnitude required")
    if method == "K_from_L_equiv":
        return _value(p, "L_D") * _positive(p, "fd")
    if method == "K_from_f":
        return _value(p, "fd") * _positive(p, "L") / _positive(p, "D")
    if method == "L_equiv_from_K":
        return _value(p, "K") / _positive(p, "fd")
    if method == "L_from_K":
        return _value(p, "K") * _positive(p, "D") / _positive(p, "fd")
    if method == "f_from_K":
        return _value(p, "K") * _positive(p, "D") / _positive(p, "L")
    if method == "dP_from_K":
        return _value(p, "K") * _positive(p, "rho") * _value(p, "V") ** 2 / 2
    if method == "head_from_K":
        return _value(p, "K") * _value(p, "V") ** 2 / (2 * _positive(p, "g"))
    if method == "head_from_P":
        return _value(p, "P") / (_positive(p, "rho") * _positive(p, "g"))
    if method == "P_from_head":
        return _value(p, "head") * _positive(p, "rho") * _positive(p, "g")
    return _value(p, "roughness") / _positive(p, "D")


def local_gravity(latitude_degrees: float, elevation_m: float) -> float:
    """Conventional latitude/height engineering gravity, not a geoid solution."""
    if (
        not isfinite(latitude_degrees)
        or not -90 <= latitude_degrees <= 90
        or not isfinite(elevation_m)
    ):
        raise ValueError("latitude in [-90,90] and finite elevation required")
    angle = latitude_degrees * pi / 180
    value = (
        9.780356 * (1 + 0.0052885 * sin(angle) ** 2 - 0.0000059 * sin(2 * angle) ** 2)
        - 3.086e-6 * elevation_m
    )
    return _positive({"gravity": value}, "gravity")


def ideal_gas_sound_speed(
    temperature_K: float, heat_capacity_ratio: float, molar_mass_kg_mol: float
) -> float:
    """Ideal calorically perfect gas sound speed; molar mass is explicitly kg/mol."""
    from chemworld.physchem.property_reports import R_J_PER_MOL_K

    p = {"T": temperature_K, "gamma": heat_capacity_ratio, "M": molar_mass_kg_mol}
    if _positive(p, "gamma") <= 1:
        raise ValueError("ideal gas heat capacity ratio must exceed one")
    return sqrt(_positive(p, "gamma") * R_J_PER_MOL_K * _positive(p, "T") / _positive(p, "M"))
