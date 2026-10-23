"""Helical-pipe friction and transition families with explicit regime choices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import pipe_friction_factor

HELICAL_TRANSITION_RANGES = {
    "Schmidt": (0.0, 0.14),
    "Seth Stahel": (0.0, 1.0),
    "Ito": (0.00116, 0.067),
    "Kubair Kuloor": (0.0005, 0.103),
    "Kutateladze Borishanskii": (0.0417, 0.1667),
    "Srinivasan": (0.004, 0.1),
}
HELICAL_FRICTION_METHODS = (
    "Schmidt turbulent",
    "Mori Nakayama turbulent",
    "Prasad",
    "Czop",
    "Guo",
    "Ju",
    "Mandal Nigam",
    "Srinivasan turbulent",
    "White",
    "Mori Nakayama laminar",
    "Schmidt laminar",
)


def _curvature(tube_diameter: float, coil_diameter: float) -> float:
    _positive(tube_diameter, "tube diameter")
    _positive(coil_diameter, "coil centerline diameter")
    if tube_diameter >= coil_diameter:
        raise ValueError("coil centerline diameter must exceed tube diameter")
    return tube_diameter / coil_diameter


def helical_transition_reynolds(
    tube_diameter_m: float,
    coil_diameter_m: float,
    *,
    method: str = "Schmidt",
    check_recommended_ranges: bool = True,
) -> float:
    ratio = _curvature(tube_diameter_m, coil_diameter_m)
    if method not in HELICAL_TRANSITION_RANGES:
        raise ValueError(f"unknown helical transition {method!r}")
    lo, hi = HELICAL_TRANSITION_RANGES[method]
    if check_recommended_ranges and not lo < ratio < hi:
        raise ValueError(f"{method} curvature outside recommended range")
    if method == "Schmidt":
        return 2300 * (1 + 8.6 * ratio**0.45)
    if method == "Seth Stahel":
        return 1900 * (1 + 8 * sqrt(ratio))
    if method == "Ito":
        return 20000 * ratio**0.32
    if method == "Kubair Kuloor":
        return 12730 * ratio**0.2
    if method == "Kutateladze Borishanskii":
        return 2300 + 10500 * ratio**0.3
    return 2100 * (1 + 12 * sqrt(ratio))


def available_helical_friction_methods(
    reynolds: float,
    tube_diameter_m: float,
    coil_diameter_m: float,
    *,
    transition_method: str = "Schmidt",
    check_regime: bool = True,
) -> tuple[str, ...]:
    _positive(reynolds, "Reynolds number")
    critical = helical_transition_reynolds(
        tube_diameter_m,
        coil_diameter_m,
        method=transition_method,
        check_recommended_ranges=False,
    )
    if not check_regime:
        return HELICAL_FRICTION_METHODS
    laminar = reynolds < critical
    return tuple(
        name
        for name in HELICAL_FRICTION_METHODS
        if (name.endswith("laminar") or name == "White") == laminar
    )


@dataclass(frozen=True)
class HelicalFrictionResult:
    method: str
    friction_factor: float
    darcy: bool
    dean_number: float
    critical_reynolds: float
    declared_regime: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def helical_friction_factor(
    reynolds: float,
    tube_diameter_m: float,
    coil_diameter_m: float,
    *,
    roughness_m: float = 0.0,
    method: str | None = None,
    transition_method: str = "Schmidt",
    laminar_method: str = "Schmidt laminar",
    turbulent_method: str = "Schmidt turbulent",
    darcy: bool = True,
    check_recommended_ranges: bool = True,
) -> HelicalFrictionResult:
    _positive(reynolds, "Reynolds number")
    ratio = _curvature(tube_diameter_m, coil_diameter_m)
    critical = helical_transition_reynolds(
        tube_diameter_m,
        coil_diameter_m,
        method=transition_method,
        check_recommended_ranges=check_recommended_ranges,
    )
    if not isfinite(roughness_m) or not 0 <= roughness_m < tube_diameter_m:
        raise ValueError("roughness must lie in [0, tube diameter)")
    laminar = reynolds < critical
    selected = method or (laminar_method if laminar else turbulent_method)
    if selected not in HELICAL_FRICTION_METHODS:
        raise ValueError(f"unknown helical friction method {selected!r}")
    method_laminar = selected.endswith("laminar") or selected == "White"
    dean = reynolds * sqrt(ratio)
    warnings = []
    if laminar != method_laminar:
        warnings.append("selected correlation conflicts with declared transition regime")
    if selected == "White" and not (11.6 < dean < 2000 and 3.878e-4 < ratio < 0.066):
        warnings.append("White recommended Dean/curvature range exceeded")
    if selected == "Mori Nakayama laminar" and not 100 < dean < 2000:
        warnings.append("Mori-Nakayama laminar recommended Dean range is100-2000")
    if selected == "Schmidt laminar" and not 100 < reynolds < critical:
        warnings.append("Schmidt laminar recommended100<Re<critical")
    if selected in {"Czop", "Guo"} and not 2e4 < reynolds < 1.5e5:
        warnings.append("Czop/Guo recommended20000<Re<150000")
    if selected == "Mandal Nigam" and not 2500 < dean < 15000:
        warnings.append("Mandal-Nigam recommended2500<Dean<15000")
    if selected == "Ju" and dean <= 11.6:
        warnings.append("Ju requires Dean>11.6")
    if selected == "Srinivasan turbulent" and not 0.01 < ratio < 0.15:
        warnings.append("Srinivasan recommended curvature .01-.15")
    if selected == "Mori Nakayama turbulent" and reynolds > 6.5e5 * sqrt(ratio):
        warnings.append("Mori-Nakayama turbulent Reynolds upper range exceeded")
    if selected == "Schmidt turbulent" and reynolds > 1.5e5:
        warnings.append("Schmidt turbulent recommendedRe<=150000")
    if check_recommended_ranges and warnings:
        raise ValueError("; ".join(warnings))
    if not check_recommended_ranges:
        warnings.append("recommended-domain checks disabled: formula analysis")
    supported_roughness = {"Ju", "Mandal Nigam", "Prasad", "Schmidt turbulent"}
    if roughness_m and selected not in supported_roughness:
        raise ValueError("selected helical equation has no roughness correction")
    r = reynolds
    straight = pipe_friction_factor(r, roughness_m / tube_diameter_m)
    if selected == "White":
        factor = 64 / r
        if dean >= 11.6:
            factor /= 1 - (1 - (11.6 / dean) ** 0.45) ** (1 / 0.45)
    elif selected == "Mori Nakayama laminar":
        factor = (
            64
            / r
            * (1.405296 if dean < 42.328036 else 0.108 * sqrt(dean) / (1 - 3.253 / sqrt(dean)))
        )
        if dean < 42.328036:
            warnings.append("source low-De extension uses constant1.405296 multiplier")
    elif selected == "Schmidt laminar":
        factor = 64 / r * (1 + 0.14 * ratio**0.97 * r ** (1 - 0.644 * ratio**0.312))
    elif selected == "Czop":
        factor = 0.096 * dean**-0.1517
    elif selected == "Guo":
        factor = 0.638 * r**-0.15 * ratio**0.51
    elif selected == "Ju":
        factor = straight * (1 + 0.11 * r**0.23 * ratio**0.14)
    elif selected == "Mandal Nigam":
        factor = straight * (1 + 0.03 * dean**0.27)
    elif selected == "Mori Nakayama turbulent":
        term = (r * ratio**2) ** -0.2
        factor = 0.3 * sqrt(ratio) * term * (1 + 0.112 * term)
    elif selected == "Prasad":
        factor = straight * (1 + 0.18 * (r * ratio**2) ** 0.25)
        warnings.append("Prasad experiment scope: hot water, coil/tube17.24 or34.9, Re1780-59500")
    elif selected == "Srinivasan turbulent":
        factor = 0.336 * dean**-0.2
    else:
        correction = (
            28800 / r * ratio**0.62 if r < 22000 else 0.0823 * (1 + ratio) * ratio**0.53 * r**0.25
        )
        factor = straight * (1 + correction)
    _positive(factor, "helical friction result")
    return HelicalFrictionResult(
        selected,
        factor if darcy else factor / 4,
        darcy,
        dean,
        critical,
        "laminar" if laminar else "turbulent",
        tuple(warnings),
    )
