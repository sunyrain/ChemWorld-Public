"""IAPWS R14-08(2011), melting Eqs. 1--5 and sublimation Eq. 6, in SI.

Copyright 2011 IAPWS; published scientific coefficients with attribution:
https://iapws.org/technical-guidance/release/MeltSub
These are phase boundaries, not equations of state for ice III/V/VI/VII.
"""

from collections.abc import Sequence
from math import isclose, isfinite, log, ulp
from typing import Any, Protocol, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.caloric_math import FirstDerivative, exponential, scalar_value
from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_ice import ICE_TRIPLE_PRESSURE_PA, ICE_TRIPLE_TEMPERATURE_K

_MELTING_RANGES = {
    "Ih": (251.165, 273.16), "III": (251.165, 256.164),
    "V": (256.164, 273.31), "VI": (273.31, 355.0), "VII": (355.0, 715.0),
}

_Temperature = TypeVar(
    "_Temperature", float, PreciseScalar, SecondOrderPair, FirstDerivative[Any, Any],
)


class _Boundary(Protocol):
    @overload
    def __call__(self, temperature: float, /) -> float: ...

    @overload
    def __call__(self, temperature: SecondOrderPair, /) -> SecondOrderPair: ...


def _check_temperature(temperature: object, lower: float, upper: float, name: str) -> None:
    value = scalar_value(temperature)
    if isinstance(value, bool) or not isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} requires {lower} <= T <= {upper} K")


def _sublimation_log_ratio(temperature: _Temperature) -> _Temperature:
    theta = temperature / ICE_TRIPLE_TEMPERATURE_K
    return cast(_Temperature, sum(a*theta**(b-1) for a, b in (
        (-.212144006e2, .333333333e-2), (.273203819e2, .120666667e1),
        (-.610598130e1, .170333333e1),
    )))


def ice_sublimation_pressure(temperature_K: _Temperature) -> _Temperature:
    """Ice-Ih vapor pressure, 50--273.16 K; native analytic AD accepted."""
    _check_temperature(temperature_K, 50, ICE_TRIPLE_TEMPERATURE_K, "ice sublimation")
    return ICE_TRIPLE_PRESSURE_PA * exponential(_sublimation_log_ratio(temperature_K))


def water_melting_pressure(temperature_K: _Temperature, *, ice_phase: str = "Ih") -> _Temperature:
    """Liquid/solid boundary for an explicit ice phase within its valid range."""
    if ice_phase not in _MELTING_RANGES:
        raise ValueError("ice_phase must be Ih, III, V, VI or VII")
    low, high = _MELTING_RANGES[ice_phase]
    _check_temperature(temperature_K, low, high, f"ice {ice_phase} melting")
    if ice_phase == "Ih":
        theta = temperature_K / ICE_TRIPLE_TEMPERATURE_K
        return ICE_TRIPLE_PRESSURE_PA * (1 + cast(_Temperature, sum(a*(1-theta**b) for a, b in (
            (.119539337e7, 3), (.808183159e5, 25.75), (.333826860e4, 103.75)))))
    theta = temperature_K / low
    if ice_phase == "VII":
        return 2216e6 * exponential(1.73683*(1-1/theta) - .0544606*(1-theta**5)
                                   + .806106e-7*(1-theta**22))
    reference, coefficient, exponent = {
        "III": (208.566e6, .299948, 60), "V": (350.1e6, 1.18721, 8),
        "VI": (632.4e6, 1.07476, 4.6),
    }[ice_phase]
    return reference * (1 - coefficient*(1-theta**exponent))


def _inverse(
    pressure: float, low: float, high: float, evaluator: _Boundary, *, logarithmic: bool,
) -> float:
    if isinstance(pressure, bool) or not isfinite(pressure) or pressure <= 0:
        raise ValueError("pressure must be finite, positive and non-boolean")
    left, right = evaluator(low), evaluator(high)
    target = log(pressure / ICE_TRIPLE_PRESSURE_PA) if logarithmic else pressure
    # Endpoints must use the same floating-point forward mapping, especially
    # for sublimation's exponential/log round trip. No physical extrapolation.
    for T, value in ((low, left), (high, right)):
        original = ICE_TRIPLE_PRESSURE_PA * exponential(value) if logarithmic else value
        if pressure == original:
            return T
    if not min(left, right) <= target <= max(left, right):
        raise ValueError("pressure is outside the declared ice boundary range")
    initial = low + (high-low)*(target-left)/(right-left)

    def jacobian(point: Sequence[float] | NDArray[np.float64]) -> list[list[float]]:
        return [[cast(float, evaluator(SecondOrderPair(point[0], dx=1)).dx)]]

    result = solve_root(
        lambda point: [evaluator(point[0])-target], [initial], jacobian=jacobian,
        variables=[ResidualVariable("T", "K", 273.16, "algebraic", low, high)],
        residual_scales=[1 if logarithmic else max(pressure, 1e6)],
        policy=RootPolicy(tolerance=2e-12, maximum_iterations=100),
    )
    if not result.success:
        raise ValueError(f"ice boundary inversion failed: {result.failure_reason}")
    T = float(result.value[0])
    # Near the Ih triple point, a single T ULP moves p by ~0.8 micro-Pa.
    # Test the forward residual against that representable T resolution,
    # not an unattainable pressure precision. Domain bounds remain strict.
    resolution = 8 * ulp(T) * abs(jacobian([T])[0][0])
    if not isclose(evaluator(T), target, rel_tol=2e-11, abs_tol=max(2e-11, resolution)):
        raise ValueError("ice boundary inversion residual exceeds acceptance")
    return T


def ice_sublimation_temperature(pressure_Pa: float) -> float:
    """Sublimation inverse with log-pressure residual, including very low p."""
    return _inverse(pressure_Pa, 50, ICE_TRIPLE_TEMPERATURE_K,
                    _sublimation_log_ratio, logarithmic=True)


def water_melting_temperature(pressure_Pa: float, *, ice_phase: str = "Ih") -> float:
    """Shared-solver inverse for the caller-selected solid/liquid branch."""
    if ice_phase not in _MELTING_RANGES:
        raise ValueError("ice_phase must be Ih, III, V, VI or VII")
    return _inverse(pressure_Pa, *_MELTING_RANGES[ice_phase],
                    lambda T: water_melting_pressure(T, ice_phase=ice_phase), logarithmic=False)
