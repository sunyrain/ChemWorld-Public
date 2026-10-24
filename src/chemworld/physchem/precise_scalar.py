"""Opt-in decimal scalar for cancellation-sensitive shared model arithmetic.

This carries no equations or solver. Published model coefficients remain the
same binary floats, converted exactly; shared jets and expressions operate on
50-digit scalars. Ordinary consumers keep ordinary float arithmetic.
"""

from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from functools import total_ordering

_CONTEXT = Context(prec=50)


def _decimal(value):
    if isinstance(value, PreciseScalar):
        return value.value
    if isinstance(value, (int, float, Decimal)):
        return Decimal(value)
    return NotImplemented


@total_ordering
@dataclass(frozen=True, slots=True, init=False)
class PreciseScalar:
    value: Decimal

    def __init__(self, value):
        number = _decimal(value)
        if number is NotImplemented:
            raise TypeError("precise scalar requires a real scalar")
        object.__setattr__(self, "value", number)

    def __float__(self):
        return float(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        number = _decimal(other)
        return NotImplemented if number is NotImplemented else self.value == number

    def __lt__(self, other):
        number = _decimal(other)
        return NotImplemented if number is NotImplemented else self.value < number

    def _binary(self, other, operation):
        number = _decimal(other)
        if number is NotImplemented:
            return NotImplemented
        with localcontext(_CONTEXT) as context:
            return PreciseScalar(getattr(context, operation)(self.value, number))

    def __add__(self, other):
        return self._binary(other, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, "subtract")

    def __rsub__(self, other):
        return -self+other

    def __neg__(self):
        return PreciseScalar(self.value.copy_negate())

    def __abs__(self):
        return PreciseScalar(self.value.copy_abs())

    def __mul__(self, other):
        return self._binary(other, "multiply")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, "divide")

    def __rtruediv__(self, other):
        number = _decimal(other)
        return NotImplemented if number is NotImplemented else PreciseScalar(number)/self

    def __pow__(self, other):
        return self._binary(other, "power")

    def exponential(self):
        with localcontext(_CONTEXT):
            return PreciseScalar(self.value.exp())

    def exponential_minus_one(self):
        with localcontext(_CONTEXT) as context:
            # Retain significant digits when the exponential rounds near one.
            if self.value:
                context.prec += max(0, -self.value.adjusted())
            return PreciseScalar(self.value.exp()-1)

    def logarithm(self):
        with localcontext(_CONTEXT):
            return PreciseScalar(self.value.ln())
