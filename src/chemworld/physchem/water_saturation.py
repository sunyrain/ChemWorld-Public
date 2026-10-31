"""One native IF97 region-4 saturation equation, shared with future water EOS.

Equations 30/31 of the IAPWS Revised Release on IF97 (2007), section 8.
Constants are published scientific parameters, not an imported implementation.
This module is only the finite saturation curve, not the complete IF97 EOS.
"""

from math import inf, isfinite, nextafter
from typing import Any, TypeVar

from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.second_order_pair import SecondOrderPair

_Temperature = TypeVar(
    "_Temperature", float, PreciseScalar, SecondOrderPair, FirstDerivative[Any, Any],
)

IF97_SATURATION_COEFFICIENTS = (
    0.11670521452767e4,
    -0.72421316703206e6,
    -0.17073846940092e2,
    0.12020824702470e5,
    -0.32325550322333e7,
    0.14915108613530e2,
    -0.48232657361591e4,
    0.40511340542057e6,
    -0.23855557567849,
    0.65017534844798e3,
)
IF97_SATURATION_RANGE_K = (273.15, 647.096)


def water_saturation_pressure(temperature_K: _Temperature) -> _Temperature:
    """IF97 region-4 pressure in Pa; supports the native derivative arithmetic."""
    T = scalar_value(temperature_K)
    if isinstance(T, bool) or not isfinite(T) or not 273.15 <= T <= 647.096:
        raise ValueError("IF97 saturation requires 273.15 <= T <= 647.096 K")
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = IF97_SATURATION_COEFFICIENTS
    theta = temperature_K + n9 / (temperature_K - n10)
    a = theta**2 + n1 * theta + n2
    b = n3 * theta**2 + n4 * theta + n5
    c = n6 * theta**2 + n7 * theta + n8
    return 1e6 * (2 * c / (-b + (b * b - 4 * a * c) ** 0.5)) ** 4


def water_saturation_temperature(pressure_Pa: float) -> float:
    """Explicit inverse, allowing one input ULP of endpoint rounding only."""
    low, high = map(water_saturation_pressure, IF97_SATURATION_RANGE_K)
    if (isinstance(pressure_Pa, bool) or not isfinite(pressure_Pa)
            or not nextafter(low, -inf) <= pressure_Pa <= nextafter(high, inf)):
        raise ValueError("pressure lies outside the declared IF97 saturation range")
    if pressure_Pa < low:
        return IF97_SATURATION_RANGE_K[0]
    if pressure_Pa > high:
        return IF97_SATURATION_RANGE_K[1]
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = IF97_SATURATION_COEFFICIENTS
    beta = (pressure_Pa / 1e6) ** 0.25
    e = beta * beta + n3 * beta + n6
    f = n1 * beta * beta + n4 * beta + n7
    g = n2 * beta * beta + n5 * beta + n8
    d = 2 * g / (-f - (f * f - 4 * e * g) ** 0.5)
    return 0.5 * (n10 + d - ((n10 + d) ** 2 - 4 * (n9 + n10 * d)) ** 0.5)
