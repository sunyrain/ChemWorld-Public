"""Expanded declarative gas kinetics consumed by the existing reaction network.

Source semantics are opt-in through ``mechanism_schema=cantera-gas-v2``. Existing
hand-authored liquid/concentration laws keep their original validation rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from math import copysign, exp, isfinite, log10
from typing import Any

from numpy.polynomial.chebyshev import chebval2d

from chemworld.physchem.reaction_network_specs import ReactionSpec
from chemworld.physchem.reaction_rate_laws import (
    R_J_PER_MOL_K as R,
)
from chemworld.physchem.reaction_rate_laws import (
    arrhenius_k,
    effective_third_body_concentration,
    mass_action_rate,
    prefixed_arrhenius_params,
    rate_driving_quantities,
)

MECHANISM_SCHEMA = "cantera-gas-v2"
FALLOFF_LAWS = {"lindemann_falloff", "troe_falloff", "sri_falloff", "tsang_falloff"}
MECHANISM_LAWS = FALLOFF_LAWS | {
    "arrhenius",
    "third_body_arrhenius",
    "pressure_log",
    "chebyshev",
    "blowers_masel",
}


def signed_arrhenius(params: Mapping[str, Any], temperature_K: float) -> float:
    """Reuse the shared nonnegative kernel while retaining explicitly signed fits."""
    a = float(params["A"])
    return copysign(arrhenius_k({**params, "A": abs(a)}, temperature_K), a)


def _troe_factor(fc: float, reduced_pressure: float) -> float:
    logfc = log10(max(fc, 1e-300))
    x = log10(max(reduced_pressure, 1e-300)) - 0.4 - 0.67 * logfc
    denominator = 0.75 - 1.27 * logfc - 0.14 * x
    # Stable at the removable zero-denominator limit of the usual f1 formula.
    return 10 ** (logfc * denominator**2 / (denominator**2 + x**2))


def falloff_broadening(params: Mapping[str, Any], law: str, t: float, pr: float) -> float:
    if law == "lindemann_falloff":
        return 1.0
    if law == "troe_falloff":
        a = float(params["troe_a"])
        t1, t3 = float(params["troe_T1"]), float(params["troe_T3"])
        fc = (1 - a) * (exp(-t / t3) if t3 else 0) + a * (exp(-t / t1) if t1 else 0)
        t2 = float(params.get("troe_T2", 0))
        return _troe_factor(fc + (exp(-t2 / t) if t2 else 0), pr)
    if law == "tsang_falloff":
        return _troe_factor(float(params["tsang_A"]) + float(params.get("tsang_B", 0)) * t, pr)
    a, b, c = (float(params[f"sri_{key}"]) for key in ("A", "B", "C"))
    base = a * exp(-b / t) + (exp(-t / c) if c else 0)
    if base < 0:
        raise ValueError("SRI broadening base is negative at this temperature")
    return base ** (1 / (1 + log10(max(pr, 1e-300)) ** 2)) * (
        float(params.get("sri_D", 1)) * t ** float(params.get("sri_E", 0))
    )


def mechanism_pressure_rate_constant(
    params: Mapping[str, Any], law: str, temperature_K: float, pressure_Pa: float
) -> float:
    """Source P-log clamps endpoints; source Chebyshev retains polynomial extrapolation."""
    t, p = temperature_K, pressure_Pa
    if t <= 0 or p <= 0 or not isfinite(t) or not isfinite(p):
        raise ValueError("gas rate validity requires finite positive temperature and pressure")
    if law == "pressure_log":
        grouped: dict[float, float] = {}
        for term in params["rate_constants"]:
            point = float(term["pressure_Pa"])
            grouped[point] = grouped.get(point, 0) + signed_arrhenius(term, t)
        if any(value <= 0 for value in grouped.values()):
            raise ValueError("P-log grouped pressure rate must be positive at this temperature")
        pressures = sorted(grouped)
        if p <= pressures[0]:
            return grouped[pressures[0]]
        if p >= pressures[-1]:
            return grouped[pressures[-1]]
        for lo, hi in pairwise(pressures):
            if lo <= p <= hi:
                weight = log10(p / lo) / log10(hi / lo)
                return 10 ** ((1 - weight) * log10(grouped[lo]) + weight * log10(grouped[hi]))
        raise ValueError("P-log pressure not bracketed")
    tmin, tmax = params["temperature_range_K"]
    pmin, pmax = params["pressure_range_Pa"]
    tr = (2 / t - 1 / tmin - 1 / tmax) / (1 / tmax - 1 / tmin)
    pr = (2 * log10(p) - log10(pmin) - log10(pmax)) / log10(pmax / pmin)
    value = float(10 ** chebval2d(tr, pr, params["coefficients"]))
    if not isfinite(value) or value <= 0:
        raise ValueError("nonfinite Chebyshev rate")
    return value


def mechanism_forward_rate_constant(
    reaction: ReactionSpec,
    concentrations: Mapping[str, float],
    temperature_K: float,
    *,
    species_thermo: Mapping[str, Any] | None = None,
) -> float:
    params, law, t = reaction.rate_law.parameters, reaction.rate_law.equation_id, temperature_K
    if t <= 0 or not isfinite(t):
        raise ValueError("temperature must be finite and positive")
    if law in {"pressure_log", "chebyshev"}:
        return mechanism_pressure_rate_constant(
            params, law, t, sum(concentrations.values()) * 1000 * R * t
        )
    if law in FALLOFF_LAWS or law == "third_body_arrhenius":
        collider = effective_third_body_concentration(
            concentrations,
            efficiencies=params.get("third_body_efficiencies", {}),
            default_efficiency=float(params.get("default_efficiency", 1)),
        )
        if law == "third_body_arrhenius":
            return signed_arrhenius(params, t) * collider
        low = signed_arrhenius(prefixed_arrhenius_params(params, "low"), t)
        high = signed_arrhenius(prefixed_arrhenius_params(params, "high"), t)
        if high == 0:
            return 0.0
        pr = low * collider / high
        broadening = falloff_broadening(params, law, t, pr)
        return (low if params.get("chemically_activated") else high * pr) / (1 + pr) * broadening
    if law == "blowers_masel":
        if species_thermo is None:
            raise ValueError("Blowers-Masel requires species thermochemistry")
        delta_h = sum(
            n * species_thermo[s].evaluate(t).enthalpy_J_mol
            for s, n in reaction.stoichiometry.items()
        )
        intrinsic, w = float(params["Ea0_J_per_mol"]), float(params["w_J_per_mol"])
        if delta_h <= -4 * intrinsic:
            activation = 0.0
        elif delta_h > 4 * intrinsic:
            activation = delta_h
        else:
            vp = 2 * w * (w + intrinsic) / (w - intrinsic)
            activation = (
                (w + delta_h / 2) * (vp - 2 * w + delta_h) ** 2 / (vp**2 - 4 * w * w + delta_h**2)
            )
        return signed_arrhenius({**params, "Ea_J_per_mol": activation}, t)
    if law != "arrhenius":
        raise ValueError(f"unsupported imported gas rate: {law}")
    return signed_arrhenius(params, t)


def _signed_mass_action(
    orders: Mapping[str, float], concentrations: Mapping[str, float], k: float
) -> float:
    if any(order < 0 and concentrations.get(name, 0) <= 0 for name, order in orders.items()):
        raise ValueError("negative reaction order is singular at zero concentration")
    return copysign(mass_action_rate(orders, concentrations, abs(k)), k)


def evaluate_mechanism_rate(
    reaction: ReactionSpec,
    *,
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
    species_thermo: Mapping[str, Any] | None = None,
) -> float:
    """Native net rate; thermochemical reverse uses the same equilibrium kernel."""
    driving = rate_driving_quantities(reaction, concentrations_mol_L)
    k = mechanism_forward_rate_constant(
        reaction, driving, temperature_K, species_thermo=species_thermo
    )
    forward = _signed_mass_action(reaction.kinetic_forward_orders, driving, k)
    if not reaction.reversible:
        return forward
    if species_thermo is None:
        raise ValueError("reversible gas kinetics require declared species thermochemistry")
    from chemworld.physchem.reaction_network import (
        thermochemical_concentration_equilibrium_constant,
    )

    pressure = float(reaction.rate_law.parameters["reference_pressure_Pa"])
    if any(species_thermo[s].reference_pressure_Pa != pressure for s in reaction.stoichiometry):
        raise ValueError("gas thermochemical reference pressures must agree")
    kc, _, _ = thermochemical_concentration_equilibrium_constant(
        reaction,
        species_thermo=species_thermo,
        temperature_K=temperature_K,
        standard_concentration_mol_L=pressure / (1000 * R * temperature_K),
    )
    return forward - _signed_mass_action(reaction.kinetic_reverse_orders, driving, k / kc)


def validate_mechanism_rate(reaction: ReactionSpec) -> None:
    p, law = reaction.rate_law.parameters, reaction.rate_law.equation_id
    if law not in MECHANISM_LAWS or p.get("mechanism_schema") != MECHANISM_SCHEMA:
        raise ValueError("unsupported gas mechanism schema or rate law")
    if reaction.rate_law.uses_activities:
        raise ValueError("gas mechanisms use concentrations")
    if reaction.reversible and p.get("reverse_basis") != "ideal_gas_thermo":
        raise ValueError("gas reverse rate requires ideal_gas_thermo")
    if reaction.reversible and reaction.metadata.get("custom_orders"):
        raise ValueError("custom reaction orders are only allowed for irreversible reactions")
    if reaction.metadata.get("custom_reverse_orders") and not reaction.reversible:
        raise ValueError("custom reverse reaction orders require a reversible reaction")
    groups = []
    if law in {"arrhenius", "third_body_arrhenius", "blowers_masel"}:
        groups = [p]
    elif law in FALLOFF_LAWS:
        groups = [prefixed_arrhenius_params(p, prefix) for prefix in ("low", "high")]
        if float(groups[0]["A"]) * float(groups[1]["A"]) < 0:
            raise ValueError("low/high Arrhenius factors must have the same sign")
    elif law == "pressure_log":
        groups = p["rate_constants"]
        if not groups or any(
            float(term["pressure_Pa"]) <= 0 or not isfinite(float(term["pressure_Pa"]))
            for term in groups
        ):
            raise ValueError("P-log needs positive pressure terms")
    elif law == "chebyshev":
        rows = p["coefficients"]
        if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
            raise ValueError("Chebyshev coefficients must be rectangular and nonempty")
        if any(not isfinite(float(v)) for row in rows for v in row):
            raise ValueError("nonfinite Chebyshev coefficient")
        for field in ("temperature_range_K", "pressure_range_Pa"):
            bounds = p[field]
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] or not isfinite(bounds[1]):
                raise ValueError(f"invalid {field}")
    for group in groups:
        if any(not isfinite(float(group.get(key, 0))) for key in ("A", "b", "Ea_J_per_mol")):
            raise ValueError("nonfinite Arrhenius parameters")
        if float(group["A"]) < 0 and not (p.get("negative_A") or law == "pressure_log"):
            raise ValueError("negative A requires explicit negative-A permission")
    efficiencies = p.get("third_body_efficiencies", {})
    if any(
        not isfinite(float(v)) or float(v) < 0
        for v in [p.get("default_efficiency", 1), *efficiencies.values()]
    ):
        raise ValueError("third-body efficiencies must be finite and nonnegative")
    if law == "sri_falloff" and (p["sri_C"] < 0 or p.get("sri_D", 1) < 0):
        raise ValueError("SRI C and D must be nonnegative")
    for field in (
        {"troe_a", "troe_T1", "troe_T3"}
        if law == "troe_falloff"
        else {"sri_A", "sri_B", "sri_C"}
        if law == "sri_falloff"
        else {"tsang_A"}
        if law == "tsang_falloff"
        else set()
    ):
        if not isfinite(float(p[field])):
            raise ValueError(f"nonfinite falloff parameter: {field}")
    if law == "blowers_masel":
        ea, w = float(p["Ea0_J_per_mol"]), float(p["w_J_per_mol"])
        if not (isfinite(ea) and isfinite(w) and 0 < ea < w):
            raise ValueError("Blowers-Masel requires 0 < Ea0 < w")


def audit_mechanism_rate_contract(reaction: ReactionSpec) -> Any:
    """Return the existing contract report, including zero/negative custom orders."""
    from chemworld.physchem.reaction_rate_contracts import ReactionRateContractReport

    violations = []
    try:
        validate_mechanism_rate(reaction)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        violations.append(str(error))
    p, law = reaction.rate_law.parameters, reaction.rate_law.equation_id
    order = sum(reaction.kinetic_forward_orders.values())
    reverse = sum(reaction.kinetic_reverse_orders.values()) if reaction.reversible else None
    effective = order + (law == "third_body_arrhenius")

    def unit(n: float, b: float = 0) -> str:
        return f"(mol/L)^{1 - n:g} s^-1" + (f" K^{-b:g}" if b else "")

    low = order if p.get("chemically_activated") else order + 1
    high = low - 1
    return ReactionRateContractReport(
        reaction.reaction_id,
        law,
        "mol/L",
        "concentration",
        1.0,
        "mol/(L*s)",
        order,
        effective,
        reverse,
        unit(effective, float(p.get("b", 0))),
        None if reverse is None else unit(reverse),
        None if reverse is None else f"(mol/L)^{reverse - order:g}",
        unit(low, float(p.get("low_b", 0))) if law in FALLOFF_LAWS else None,
        unit(high, float(p.get("high_b", 0))) if law in FALLOFF_LAWS else None,
        "J/mol",
        ("Source gas semantics; P-log clamps; Chebyshev extrapolates; thermo validates T.",),
        tuple(violations),
        ("Cantera public specification v3.2.0; native implementation",),
    )
