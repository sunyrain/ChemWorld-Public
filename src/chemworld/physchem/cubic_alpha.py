"""Explicit Soave/Twu alpha and analytic T responses on the existing EOS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import exponential, partial_derivative


@dataclass(frozen=True)
class CubicAlphaResponse:
    model: str
    temperature_K: float
    critical_temperature_K: float
    alpha: float
    d_alpha_dT: float
    d2_alpha_dT2: float
    d3_alpha_dT3: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cubic_alpha_value(
    temperature: Any,
    critical_temperature: float,
    *,
    model: str,
    coefficients: tuple[float, ...],
) -> Any:
    """Same analytic arithmetic as GE and the other native property equations."""
    reduced = temperature / critical_temperature
    if model == "soave":
        return (1 + coefficients[0] * (1 - reduced**0.5)) ** 2
    if model == "twu91":
        ell, m, n = coefficients
        return reduced ** (n * (m - 1)) * exponential(ell * (1 - reduced ** (m * n)))
    raise ValueError("cubic alpha model must be soave or twu91")


def cubic_alpha_response(
    *,
    temperature_K: float,
    critical_temperature_K: float,
    model: str,
    coefficients: tuple[float, ...],
) -> CubicAlphaResponse:
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (temperature_K, critical_temperature_K)
    ):
        raise ValueError("alpha temperatures must be positive and finite")
    if model not in {"soave", "twu91"} or len(coefficients) != (1 if model == "soave" else 3):
        raise ValueError("alpha model requires its complete coefficient vector")
    if any(isinstance(v, bool) or not isfinite(v) for v in coefficients):
        raise ValueError("alpha coefficients must be finite")

    def equation(values: Any) -> Any:
        return cubic_alpha_value(
            values["T"], critical_temperature_K, model=model, coefficients=coefficients
        )

    derivatives = tuple(
        partial_derivative(equation, {"T": temperature_K}, ("T",) * order) for order in range(1, 4)
    )
    alpha = equation({"T": temperature_K})
    if not isfinite(alpha) or alpha < 0 or any(not isfinite(v) for v in derivatives):
        raise ValueError("alpha response lies outside numerical range")
    return CubicAlphaResponse(model, temperature_K, critical_temperature_K, alpha, *derivatives)
