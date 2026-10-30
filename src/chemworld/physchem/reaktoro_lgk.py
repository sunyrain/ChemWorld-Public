"""Auditable, bounded evaluators for Reaktoro PHREEQC/GEM ``lg K(T)``.

The source audit intentionally preserves the coefficient records without
evaluating them.  This module provides the next, smaller contract: evaluate
one of the two published temperature expressions when the caller supplies an
explicit positive temperature domain and pressure.  The evaluator does not
construct an ``AqueousDatabase`` and does not apply pressure corrections;
the requested pressure must equal the model's explicit reference pressure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .reaktoro_source_audit import _LGK_COEFFICIENT_UNITS

LGK_EVALUATOR_SCHEMA = "chemworld-reaktoro-lgk-evaluator-v1"
LGK_VARIANTS = tuple(_LGK_COEFFICIENT_UNITS)

_FORMULAS = {
    "PhreeqcLgK": "A1 + A2*T + A3*T^-1 + A4*log10(T) + A5*T^-2 + A6*T^2",
    "GemsLgK": "A0 + A1*T + A2*T^-1 + A3*ln(T) + A4*T^-2 + A5*T^2 + A6*T^-0.5",
}


def _scalar(value: Any, label: str, unit: str) -> dict[str, float | str]:
    if not isinstance(value, Mapping) or set(value) != {"value", "unit"}:
        raise ValueError(f"{label} requires exactly value and unit")
    number = value["value"]
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError(f"{label}.value must be a finite number")
    number = float(number)
    if not math.isfinite(number):
        raise ValueError(f"{label}.value must be a finite number")
    if value["unit"] != unit:
        raise ValueError(f"{label} requires unit {unit}")
    return {"value": number, "unit": unit}


def _parameters(variant: str, params: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    if variant not in _LGK_COEFFICIENT_UNITS:
        raise ValueError(f"unsupported lgK variant {variant!r}")
    if not isinstance(params, Mapping):
        raise ValueError("lgK parameters must be a mapping")
    coefficient_units = _LGK_COEFFICIENT_UNITS[variant]
    required = set(coefficient_units) | {"Pr"}
    if set(params) != required:
        missing = sorted(required - set(params), key=str)
        extra = sorted(set(params) - required, key=str)
        details = []
        if missing:
            details.append("missing " + ", ".join(map(str, missing)))
        if extra:
            details.append("unknown " + ", ".join(map(str, extra)))
        raise ValueError(f"{variant} fields invalid ({'; '.join(details)})")
    coefficients: dict[str, float] = {}
    for key, unit in coefficient_units.items():
        coefficients[key] = float(_scalar(params[key], f"{variant}.{key}", unit)["value"])
    reference_pressure = float(_scalar(params["Pr"], f"{variant}.Pr", "Pa")["value"])
    return coefficients, reference_pressure


def _temperature_domain(value: Any) -> tuple[dict[str, float | str], dict[str, float | str]]:
    if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
        raise ValueError("temperature_domain requires exactly min and max")
    lower = _scalar(value["min"], "temperature_domain.min", "K")
    upper = _scalar(value["max"], "temperature_domain.max", "K")
    if lower["value"] <= 0 or upper["value"] <= 0:
        raise ValueError("temperature_domain bounds must be greater than zero")
    if lower["value"] > upper["value"]:
        raise ValueError("temperature_domain.min must not exceed max")
    return lower, upper


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite for the supplied inputs")
    return value


def evaluate_lgk(
    variant: str,
    params: Mapping[str, Any],
    *,
    temperature: Mapping[str, Any],
    temperature_domain: Mapping[str, Any],
    pressure: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a bounded PHREEQC or GEM ``lg K(T)`` expression.

    Every scalar is required in explicit SI units.  ``temperature`` must be
    inside the inclusive ``temperature_domain``; an out-of-domain value is
    rejected instead of silently extrapolated.  Since the published
    coefficient records do not include a reaction volume, ``pressure`` must
    equal the explicit ``Pr`` and no pressure correction is applied.
    """

    coefficients, reference_pressure = _parameters(variant, params)
    lower, upper = _temperature_domain(temperature_domain)
    t = float(_scalar(temperature, "temperature", "K")["value"])
    if t <= 0:
        raise ValueError("temperature must be greater than zero")
    if t < lower["value"] or t > upper["value"]:
        raise ValueError(
            "temperature is outside the explicit temperature_domain; extrapolation is rejected"
        )
    p = float(_scalar(pressure, "pressure", "Pa")["value"])
    if p != reference_pressure:
        raise ValueError(
            "pressure must equal the explicit reference pressure; "
            "pressure correction is unavailable"
        )

    try:
        if variant == "PhreeqcLgK":
            terms = {
                "A1": coefficients["A1"],
                "A2*T": coefficients["A2"] * t,
                "A3*T^-1": coefficients["A3"] / t,
                "A4*log10(T)": coefficients["A4"] * math.log10(t),
                "A5*T^-2": coefficients["A5"] / (t * t),
                "A6*T^2": coefficients["A6"] * (t * t),
            }
        else:
            terms = {
                "A0": coefficients["A0"],
                "A1*T": coefficients["A1"] * t,
                "A2*T^-1": coefficients["A2"] / t,
                "A3*ln(T)": coefficients["A3"] * math.log(t),
                "A4*T^-2": coefficients["A4"] / (t * t),
                "A5*T^2": coefficients["A5"] * (t * t),
                "A6*T^-0.5": coefficients["A6"] / math.sqrt(t),
            }
        for name, term in terms.items():
            _finite(term, f"term {name}")
        lg_k = _finite(sum(terms.values()), "lgK")
    except (OverflowError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("term ", "lgK")):
            raise
        raise ValueError("lgK expression cannot be evaluated for the supplied inputs") from exc

    return {
        "schema_version": LGK_EVALUATOR_SCHEMA,
        "provider": "reaktoro",
        "variant": variant,
        "formula": _FORMULAS[variant],
        "temperature": {"value": t, "unit": "K"},
        "temperature_domain": {"min": lower, "max": upper},
        "pressure": {"value": p, "unit": "Pa"},
        "reference_pressure": {"value": reference_pressure, "unit": "Pa"},
        "terms": {name: {"value": value, "unit": "1"} for name, value in terms.items()},
        "lgK": {"value": lg_k, "unit": "1"},
        "extrapolated": False,
        "pressure_correction": {
            "applied": False,
            "reason": "reaction volume data is outside this evaluator contract",
        },
    }


__all__ = ["LGK_EVALUATOR_SCHEMA", "LGK_VARIANTS", "evaluate_lgk"]
