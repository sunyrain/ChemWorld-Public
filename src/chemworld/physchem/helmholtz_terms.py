"""Public Helmholtz expressions on ChemWorld's shared analytic arithmetic.

The JSON terminology follows CoolProp 8.0.0. Equations were independently
expressed after reading its mathematical documentation and scalar expressions;
none of its derivative implementation or runtime is used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType
from typing import Any

from chemworld.physchem.caloric_math import (
    derivative_depth,
    exponential,
    exponential_minus_one,
    logarithm,
    nonnegative_power,
    scalar_value,
)

IDEAL_SCHEMAS = {
    "Lead": ("a1 a2", ""),
    "EnthalpyEntropyOffset": ("a1 a2", ""),
    "LogTau": ("a", ""),
    "Power": ("", "n t"),
    "PlanckEinstein": ("", "n t"),
    "PlanckEinsteinGeneralized": ("", "n t c d"),
    "PlanckEinsteinFunctionT": ("Tcrit", "n v"),
    "GERG2004Cosh": ("Tcrit", "n theta"),
    "GERG2004Sinh": ("Tcrit", "n theta"),
    "CP0Constant": ("cp_over_R Tc T0", ""),
    "CP0PolyT": ("Tc T0", "c t"),
    "CP0AlyLee": ("Tc T0", "c"),
}
RESIDUAL_SCHEMAS = {
    "Power": ("", "n d t l"),
    "Gaussian": ("", "n d t eta epsilon beta gamma"),
    "GaoB": ("", "n d t eta epsilon beta gamma b"),
    "NonAnalytic": ("", "n a b beta A B C D"),
    "Lemmon2005": ("", "n d t l m"),
    "DoubleExponential": ("", "n d t gd ld gt lt"),
    "Exponential": ("", "n d t g l"),
    "Associating": ("a m epsilonbar vbarn kappabar", ""),
}
TERM_SCHEMAS = {
    **{"IdealGasHelmholtz" + k: v for k, v in IDEAL_SCHEMAS.items()},
    **{"ResidualHelmholtz" + k: v for k, v in RESIDUAL_SCHEMAS.items()},
}


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("Helmholtz coefficients must be finite real numbers")
    return float(value)


@dataclass(frozen=True)
class HelmholtzTerm:
    type: str
    coefficients: Mapping[str, Any]
    reference: str = ""

    def __post_init__(self) -> None:
        if self.type not in TERM_SCHEMAS:
            raise ValueError(f"unsupported Helmholtz term: {self.type}")
        scalars, arrays = (part.split() for part in TERM_SCHEMAS[self.type])
        if set(self.coefficients) != set(scalars + arrays):
            raise ValueError(f"{self.type} requires exactly {scalars + arrays}")
        normalized: dict[str, Any] = {key: _number(self.coefficients[key]) for key in scalars}
        for key in arrays:
            values = self.coefficients[key]
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError("Helmholtz coefficient vectors must be nonempty")
            normalized[key] = tuple(_number(v) for v in values)
        if arrays and len({len(normalized[k]) for k in arrays}) != 1:
            raise ValueError("Helmholtz coefficient vectors must have equal lengths")
        if self.type.endswith("CP0AlyLee") and len(normalized["c"]) != 5:
            raise ValueError("Aly-Lee requires five coefficients")
        for key in ("Tc", "Tcrit", "T0"):
            if key in normalized and normalized[key] <= 0:
                raise ValueError("Helmholtz reference temperatures must be positive")
        if self.type.endswith("NonAnalytic") and any(v <= 0 for v in normalized["beta"]):
            raise ValueError("nonanalytic beta must be positive")
        object.__setattr__(self, "coefficients", MappingProxyType(normalized))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HelmholtzTerm:
        kind = value.get("type")
        if kind not in TERM_SCHEMAS:
            raise ValueError(f"unsupported Helmholtz term: {kind}")
        keys = " ".join(TERM_SCHEMAS[kind]).split()
        missing = set(keys) - set(value)
        if missing:
            raise ValueError(f"{kind} missing coefficients: {sorted(missing)}")
        return cls(kind, {k: value[k] for k in keys}, str(value.get("reference", "")))

    def to_dict(self) -> dict[str, Any]:
        result = {"type": self.type, **{k: list(v) if isinstance(v, tuple) else v
                                        for k, v in self.coefficients.items()}}
        if self.reference:
            result["reference"] = self.reference
        return result

    def evaluate(self, delta: Any, tau: Any, reducing_temperature_K: float) -> Any:
        if self.type.startswith("IdealGasHelmholtz"):
            return _ideal(self, delta, tau, reducing_temperature_K)
        return _residual(self, delta, tau)


def _planck(n: float, exponent: Any, constant: float = 1.0, multiplier: float = -1.0) -> Any:
    # Factoring exp(x) avoids overflowing the generalized oscillator.
    if scalar_value(exponent) > 0:
        return n * (exponent + logarithm(multiplier + constant * exponential(-exponent)))
    if constant == 1 and multiplier == -1:
        return n * logarithm(-exponential_minus_one(exponent))
    return n * logarithm(constant + multiplier * exponential(exponent))


def _cp_polynomial(c: float, power: float, tau: Any, Tc: float, T0: float) -> Any:
    tau0 = Tc / T0
    if power == 0:
        return c * (1 - tau / tau0 + logarithm(tau / tau0))
    if power == -1:
        return c / Tc * (tau * logarithm(tau0 / tau) + tau - tau0)
    return (-c * Tc**power * tau**-power / (power * (power + 1))
            - c * T0**(power + 1) * tau / (Tc * (power + 1)) + c * T0**power / power)


def _ideal(term: HelmholtzTerm, delta: Any, tau: Any, reducing_temperature_K: float) -> Any:
    c = term.coefficients
    kind = term.type.removeprefix("IdealGasHelmholtz")
    if kind in {"Lead", "EnthalpyEntropyOffset"}:
        return c["a1"] + c["a2"] * tau + (logarithm(delta) if kind == "Lead" else 0)
    if kind == "LogTau":
        return c["a"] * logarithm(tau)
    if kind == "Power":
        return sum(n * tau**t for n, t in zip(c["n"], c["t"], strict=True))
    if kind == "PlanckEinstein":
        return sum(_planck(n, -t * tau) for n, t in zip(c["n"], c["t"], strict=True))
    if kind == "PlanckEinsteinFunctionT":
        return sum(_planck(n, -v * tau / c["Tcrit"])
                   for n, v in zip(c["n"], c["v"], strict=True))
    if kind == "PlanckEinsteinGeneralized":
        return sum(_planck(n, t * tau, k, d)
                   for n, t, k, d in zip(c["n"], c["t"], c["c"], c["d"], strict=True))
    if kind in {"GERG2004Cosh", "GERG2004Sinh"}:
        value = 0.0
        for n, theta in zip(c["n"], c["theta"], strict=True):
            x = abs(theta) * c["Tcrit"] / reducing_temperature_K * tau
            if kind.endswith("Cosh"):
                value -= n * (x + logarithm(1 + exponential(-2*x)) - log(2))
            else:
                value += n * (x + logarithm(-exponential_minus_one(-2*x)) - log(2))
        return value
    if kind == "CP0Constant":
        return _cp_polynomial(c["cp_over_R"], 0, tau, c["Tc"], c["T0"])
    if kind == "CP0PolyT":
        return sum(_cp_polynomial(a, t, tau, c["Tc"], c["T0"])
                   for a, t in zip(c["c"], c["t"], strict=True))
    a, b, d, e, f = c["c"]
    value = _cp_polynomial(a, 0, tau, c["Tc"], c["T0"])
    if b:
        value += _planck(b, -2*d*tau/c["Tc"])
    if e:
        value += _planck(-e, -2*f*tau/c["Tc"], 1, 1)
    return value


def _absolute_power(value: Any, power: float) -> Any:
    # At delta=1 the composite |delta-1|^p can have finite derivatives even
    # when the intermediate z^(p/2) at z=0 does not. Use its actual limit.
    if scalar_value(value) == 0:
        depth = derivative_depth(value)
        if power > depth:
            return 0.0
        if power == int(power) and int(power) % 2 == 0:
            return value**int(power)
        raise ValueError("requested Helmholtz critical derivative does not exist")
    return nonnegative_power(value * value, power / 2)


def _residual(term: HelmholtzTerm, delta: Any, tau: Any) -> Any:
    c = term.coefficients
    kind = term.type.removeprefix("ResidualHelmholtz")
    if kind == "Associating":
        eta = c["vbarn"] * delta
        if not 0 <= scalar_value(eta) < 1:
            raise ValueError("association packing fraction must be in [0,1)")
        contact = (2 - eta) / (2 * (1 - eta)**3)
        strength = contact * exponential_minus_one(c["epsilonbar"] * tau) * c["kappabar"]
        fraction = 2 / (1 + (1 + 4 * strength * delta)**0.5)
        return c["a"] * c["m"] * (logarithm(fraction) - fraction / 2 + 0.5)
    value = 0.0
    for i, n in enumerate(c["n"]):
        p = {key: values[i] for key, values in c.items()}
        if kind == "NonAnalytic":
            theta = 1 - tau + p["A"] * _absolute_power(delta - 1, 1 / p["beta"])
            discriminant = theta**2 + p["B"] * _absolute_power(delta - 1, 2 * p["a"])
            value += n * nonnegative_power(discriminant, p["b"]) * delta * exponential(
                -p["C"] * (delta - 1)**2 - p["D"] * (tau - 1)**2)
            continue
        exponent = 0.0
        if kind in {"Power", "Lemmon2005"}:
            if p["l"]:
                exponent -= delta**p["l"]
            if kind == "Lemmon2005" and p["m"]:
                exponent -= tau**p["m"]
        elif kind == "Exponential":
            exponent = -p["g"] * delta**p["l"]
        elif kind == "DoubleExponential":
            exponent = -p["gd"] * delta**p["ld"] - p["gt"] * tau**p["lt"]
        elif kind == "Gaussian":
            exponent = -p["eta"] * (delta - p["epsilon"])**2 - p["beta"] * (tau-p["gamma"])**2
        elif kind == "GaoB":
            exponent = p["eta"] * (delta-p["epsilon"])**2 + 1/(p["b"]+p["beta"]*(tau-p["gamma"])**2)
        value += n * delta**p["d"] * tau**p["t"] * exponential(exponent)
    return value
