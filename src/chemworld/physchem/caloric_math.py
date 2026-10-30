"""Pure derivative arithmetic and conservative ranges for property formulas.

Jets apply analytic chain rules, not finite differences. Intervals enclose
algebraic terms during capacity validation; they own no model or runtime state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, expm1, fsum, inf, isfinite, log, nextafter, tanh
from typing import Any, Generic, TypeAlias, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.second_order_pair import SecondOrderPair

JetCoefficient: TypeAlias = (
    "float | PreciseScalar | SecondOrderPair | NDArray[np.float64] | FirstDerivative[Any, Any]"
)
_Value = TypeVar("_Value", bound=JetCoefficient)
_Slope = TypeVar("_Slope", bound=JetCoefficient)
_Base = TypeVar("_Base", float, PreciseScalar)
_Sum = TypeVar("_Sum")


@dataclass(frozen=True)
class FirstDerivative(Generic[_Value, _Slope]):
    value: _Value
    slope: _Slope | float = 0.0

    def __add__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        # Coefficient promotion/broadcasting is owned by its numeric type. Keep
        # this dispatch dynamic while preserving each seed's value/slope types.
        left: Any = self
        other = _jet(other)
        return FirstDerivative(left.value + other.value, left.slope + other.slope)

    __radd__ = __add__

    def __neg__(self) -> FirstDerivative[Any, Any]:
        return FirstDerivative(-self.value, -self.slope)

    def __sub__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        return self + -_jet(other)

    def __rsub__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        return _jet(other) + -self

    def __mul__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        left: Any = self
        other = _jet(other)
        return FirstDerivative(left.value * other.value,
                               left.slope * other.value + left.value * other.slope)

    __rmul__ = __mul__

    def __truediv__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        left: Any = self
        other = _jet(other)
        value = left.value / other.value
        return FirstDerivative(value, (left.slope - value * other.slope) / other.value)

    def __rtruediv__(self, other: JetCoefficient) -> FirstDerivative[Any, Any]:
        return _jet(other) / self

    def __pow__(self, power: float) -> FirstDerivative[Any, Any]:
        if power == 0:
            return FirstDerivative(1)
        left: Any = self
        return FirstDerivative(left.value**power, power * left.value**(power - 1) * left.slope)


def _jet(value: JetCoefficient) -> FirstDerivative[Any, Any]:
    return value if isinstance(value, FirstDerivative) else FirstDerivative(value)


@overload
def compensated_sum(values: Iterable[float]) -> float: ...


@overload
def compensated_sum(values: Iterable[_Sum]) -> _Sum | float: ...


def compensated_sum(values: Iterable[Any]) -> Any:
    """Accurately sum a shared jet's scalar coefficients without losing derivatives.

    Dense Helmholtz pressure subtracts large residual terms. Compensating only
    the primal value leaves its pressure and response derivatives inaccurate.
    Nested first derivatives and SecondOrderPair use the same summation rule.
    Other arithmetic types retain their own addition contract.
    """
    values = tuple(values)
    if any(isinstance(value, FirstDerivative) for value in values):
        jets = tuple(_jet(value) for value in values)
        return FirstDerivative(compensated_sum(v.value for v in jets),
                               compensated_sum(v.slope for v in jets))
    if any(isinstance(value, SecondOrderPair) for value in values):
        pairs = tuple(value if isinstance(value, SecondOrderPair) else SecondOrderPair(value)
                     for value in values)
        return SecondOrderPair(*(compensated_sum(getattr(v, field) for v in pairs)
                                 for field in ("value", "dx", "dy", "dxx", "dxy", "dyy")))
    if all(isinstance(value, (int, float)) for value in values):
        return fsum(values)
    return sum(values)


def scalar_value(value: object) -> Any:
    """Unwrap any nesting depth, retaining the leaf's numeric type and precision."""
    while isinstance(value, (FirstDerivative, SecondOrderPair)):
        value = value.value
    return value


def derivative_depth(value: object) -> int:
    """Number of nested analytic derivative levels."""
    depth = 0
    while isinstance(value, FirstDerivative):
        depth += 1
        value = value.value
    return depth + (2 if isinstance(value, SecondOrderPair) else 0)


_Power = TypeVar("_Power", float, PreciseScalar, FirstDerivative[Any, Any], SecondOrderPair)


@overload
def nonnegative_power(value: _Power, exponent: float) -> _Power | float: ...


@overload
def nonnegative_power(
    value: float | PreciseScalar | FirstDerivative[Any, Any] | SecondOrderPair,
    exponent: FirstDerivative[Any, Any] | SecondOrderPair,
) -> float | PreciseScalar | FirstDerivative[Any, Any] | SecondOrderPair: ...


def nonnegative_power(
    value: float | PreciseScalar | FirstDerivative[Any, Any] | SecondOrderPair,
    exponent: float | FirstDerivative[Any, Any] | SecondOrderPair,
) -> float | PreciseScalar | FirstDerivative[Any, Any] | SecondOrderPair:
    """Real power with explicit finite, one-sided derivatives at zero.

    A variable exponent uses the same chain rules through exp/log. At zero
    its parameter derivative is intentionally rejected rather than guessing
    a continuation across negative bases.
    """
    if scalar_value(value) < 0:
        raise ValueError("fractional power requires a nonnegative base")
    if scalar_value(value) == 0:
        if isinstance(exponent, (FirstDerivative, SecondOrderPair)):
            raise ValueError("variable-exponent derivative at zero is undefined")
        depth = derivative_depth(value)
        if exponent < 0 or (exponent < depth and exponent != int(exponent)):
            raise ValueError("requested critical derivative does not exist")
        if exponent > depth:
            return 0.0
        result: float | PreciseScalar | FirstDerivative[Any, Any] | SecondOrderPair = 1.0
        for _ in range(int(exponent)):
            result *= value
        return result
    if isinstance(exponent, (FirstDerivative, SecondOrderPair)):
        return exponential(exponent * logarithm(value))
    return value**exponent


def derivative_values(
    function: Callable[[Any], Any], value: _Base, *, order: int = 1
) -> tuple[_Base | float, ...]:
    """Value through fourth derivative from nested first-order chain rules.

    The small fixed order is sufficient for pressure/Arrhenius derivatives;
    this is neither a symbolic expression interpreter nor a finite difference.
    Existing caloric consumers retain their original first-order arithmetic.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 4:
        raise ValueError("analytic derivative order must be an integer from zero through four")
    if not isfinite(value):
        raise ValueError("analytic differentiation requires a finite input")

    result = function(_derivative_seed(value, (True,) * order))
    derivatives = []
    for _ in range(order+1):
        derivatives.append(scalar_value(result))
        result = result.slope if isinstance(result, FirstDerivative) else 0.0
    if any(not isfinite(v) for v in derivatives):
        raise ValueError("property or requested analytic derivative is nonfinite")
    return tuple(derivatives)


def _derivative_constant(number: _Base, depth: int) -> _Base | FirstDerivative[Any, Any]:
    if not depth:
        return number
    return FirstDerivative(_derivative_constant(number, depth-1),
                           _derivative_constant(0, depth-1))


def _derivative_seed(
    number: _Base, active_levels: Sequence[bool]
) -> _Base | FirstDerivative[Any, Any]:
    if not active_levels:
        return number
    return FirstDerivative(_derivative_seed(number, active_levels[:-1]),
                           _derivative_constant(int(active_levels[-1]), len(active_levels)-1))


def partial_derivative(
    function: Callable[[dict[str, Any]], Any],
    values: Mapping[str, _Base],
    coordinates: Iterable[str],
) -> _Base | float:
    """A named mixed partial through total order three, using the same chain rules.

    ``function`` consumes a mapping. Repeated coordinates request repeated
    derivatives; each input is seeded at the same depth, including constants.
    The caller owns the model's physical domain and differentiability checks.
    """
    coordinates = tuple(coordinates)
    if not 1 <= len(coordinates) <= 3 or any(name not in values for name in coordinates):
        raise ValueError("partial derivative needs one to three known coordinates")
    if any(isinstance(v, bool) or not isfinite(v) for v in values.values()):
        raise ValueError("partial derivative inputs must be finite real numbers")
    seeded = {name: _derivative_seed(value, tuple(name == c for c in coordinates))
              for name, value in values.items()}
    result = function(seeded)
    for _ in coordinates:
        result = result.slope if isinstance(result, FirstDerivative) else 0.0
    result = scalar_value(result)
    if not isfinite(result):
        raise ValueError("requested analytic partial derivative is nonfinite")
    return result


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isfinite(self.lower) or not isfinite(self.upper) or self.lower > self.upper:
            raise ValueError("caloric interval must have ordered finite bounds")

    def __add__(self, other: float | Interval) -> Interval:
        other = _interval(other)
        return _outward(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self) -> Interval:
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: float | Interval) -> Interval:
        return self + -_interval(other)

    def __rsub__(self, other: float | Interval) -> Interval:
        return _interval(other) + -self

    def __mul__(self, other: float | Interval) -> Interval:
        other = _interval(other)
        if self.lower == self.upper == 0 or other.lower == other.upper == 0:
            return Interval(0, 0)
        values = [a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper)]
        return _outward(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other: float | Interval) -> Interval:
        other = _interval(other)
        if other.lower <= 0 <= other.upper:
            raise ValueError("caloric interval crosses a denominator zero")
        return self * _outward(1 / other.upper, 1 / other.lower)

    def __rtruediv__(self, other: float | Interval) -> Interval:
        return _interval(other) / self

    def __pow__(self, power: float) -> Interval:
        if power == 0:
            return Interval(1, 1)
        if power < 0:
            return 1 / self**(-power)
        if self.lower < 0 and power != int(power):
            raise ValueError("fractional power requires nonnegative interval")
        values = [self.lower**power, self.upper**power]
        if power == int(power) and int(power) % 2 == 0 and self.lower <= 0 <= self.upper:
            return Interval(0, nextafter(max(values), inf))
        return _outward(min(values), max(values))


def _outward(lower: float, upper: float) -> Interval:
    return Interval(nextafter(lower, -inf), nextafter(upper, inf))


def _interval(value: float | Interval) -> Interval:
    return value if isinstance(value, Interval) else Interval(value, value)


_MathValue: TypeAlias = (
    float | PreciseScalar | SecondOrderPair | FirstDerivative[Any, Any] | Interval
)
_Math = TypeVar("_Math", bound=_MathValue)


@overload
def exponential(value: float) -> float: ...


@overload
def exponential(value: _Math) -> _Math: ...


def exponential(value: _MathValue) -> _MathValue:
    if isinstance(value, PreciseScalar):
        return value.exponential()
    if isinstance(value, SecondOrderPair):
        result = exponential(value.value)
        return value.compose(result, result, result)
    if isinstance(value, FirstDerivative):
        result = exponential(value.value)
        return FirstDerivative(result, result * value.slope)
    if isinstance(value, Interval):
        return _outward(exp(value.lower), exp(value.upper))
    return exp(value)


@overload
def exponential_minus_one(value: float) -> float: ...


@overload
def exponential_minus_one(value: _Math) -> _Math: ...


def exponential_minus_one(value: _MathValue) -> _MathValue:
    """Accurate near-zero exp(x)-1 on the same nested chain-rule arithmetic."""
    if isinstance(value, PreciseScalar):
        return value.exponential_minus_one()
    if isinstance(value, SecondOrderPair):
        result = exponential(value.value)
        return value.compose(exponential_minus_one(value.value), result, result)
    if isinstance(value, FirstDerivative):
        return FirstDerivative(exponential_minus_one(value.value),
                               exponential(value.value)*value.slope)
    if isinstance(value, Interval):
        return _outward(expm1(value.lower), expm1(value.upper))
    return expm1(value)


@overload
def hyperbolic_tangent(value: float) -> float: ...


@overload
def hyperbolic_tangent(value: _Math) -> _Math: ...


def hyperbolic_tangent(value: _MathValue) -> _MathValue:
    """Stable tanh and analytic chain rule, including saturated derivatives."""
    if isinstance(value, Interval):
        return _outward(tanh(value.lower), tanh(value.upper))
    if isinstance(value, (FirstDerivative, SecondOrderPair)):
        primal = value.value
        magnitude = primal if scalar_value(primal) >= 0 else -primal
        q = exponential(-2*magnitude)
        first = 4*q/(1+q)**2
        result = hyperbolic_tangent(primal)
        if isinstance(value, SecondOrderPair):
            return value.compose(result, first, -2*result*first)
        return FirstDerivative(result, first*value.slope)
    if isinstance(value, PreciseScalar):
        sign = 1 if scalar_value(value) >= 0 else -1
        q = exponential(-2*sign*value)
        return -sign*exponential_minus_one(-2*sign*value)/(1+q)
    return tanh(value)


@overload
def logarithm(value: float) -> float: ...


@overload
def logarithm(value: _Math) -> _Math: ...


def logarithm(value: _MathValue) -> _MathValue:
    if isinstance(value, PreciseScalar):
        return value.logarithm()
    if isinstance(value, SecondOrderPair):
        inverse = 1/value.value
        return value.compose(logarithm(value.value), inverse, -inverse*inverse)
    if isinstance(value, FirstDerivative):
        return FirstDerivative(logarithm(value.value), value.slope / value.value)
    if isinstance(value, Interval):
        return _outward(log(value.lower), log(value.upper))
    return log(value)


@overload
def positive_part(value: float) -> float: ...


@overload
def positive_part(value: _Math) -> _Math | float: ...


def positive_part(value: _MathValue) -> _MathValue:
    if isinstance(value, SecondOrderPair):
        return value if value.value > 0 else SecondOrderPair(0.0)
    if isinstance(value, FirstDerivative):
        return value if scalar_value(value) > 0 else FirstDerivative(0)
    if isinstance(value, Interval):
        return Interval(max(0, value.lower), max(0, value.upper))
    return cast(float | PreciseScalar, max(0, value))


def oscillator_expression(x: _Power, *, cosh: bool = False) -> _Power | float:
    """Stable squared hyperbolic ratio, including analytic parameter derivatives."""
    if scalar_value(x) < 0:
        x = -x
    if scalar_value(x) > 745:
        return 0.0
    if not cosh and scalar_value(x) < 1e-4:
        square = x * x
        return 1 - square / 3 + square**2 / 15 - 2 * square**3 / 189
    denominator = (1 + exponential(-2*x)) if cosh else -exponential_minus_one(-2*x)
    return (2*x*exponential(-x)/denominator)**2


def oscillator_value_slope(
    x: float, temperature: float, *, cosh: bool = False
) -> tuple[float, float]:
    """(x/sinh(x))² or (x/cosh(x))², with x=theta/T; slope is d/dT."""
    x = abs(x)
    if x > 745:
        return 0.0, 0.0
    if not cosh and x < 1e-4:
        square = x * x
        return (1 - square / 3 + square**2 / 15 - 2 * square**3 / 189,
                (2 * square / 3 - 4 * square**2 / 15 + 12 * square**3 / 189) / temperature)
    value = oscillator_expression(x, cosh=cosh)
    logarithmic_slope = x * tanh(x) if cosh else x / tanh(x)
    return value, 2 * value * (logarithmic_slope - 1) / temperature


_Einstein = TypeVar("_Einstein", float, FirstDerivative[Any, Any], Interval)


def einstein(theta: float, temperature: _Einstein) -> _Einstein:
    """Einstein vibration contribution and its monotone temperature bounds."""
    def evaluate(T: float) -> tuple[float, float]:
        return oscillator_value_slope(theta / (2 * T), T)

    if isinstance(temperature, FirstDerivative):
        value, slope = evaluate(temperature.value)
        return FirstDerivative(value, slope * temperature.slope)
    if isinstance(temperature, Interval):
        return _outward(evaluate(temperature.lower)[0], evaluate(temperature.upper)[0])
    return evaluate(temperature)[0]
