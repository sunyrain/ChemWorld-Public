"""Two-coordinate second-order chain rules, with a fixed six-scalar footprint.

The Hessian stores actual derivatives, not Taylor coefficients. This arithmetic
shares scalar model expressions with the existing nested derivative evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from chemworld.physchem.precise_scalar import PreciseScalar

PairScalar: TypeAlias = float | PreciseScalar


@dataclass(frozen=True, slots=True)
class SecondOrderPair:
    value: PairScalar
    dx: PairScalar = 0.0
    dy: PairScalar = 0.0
    dxx: PairScalar = 0.0
    dxy: PairScalar = 0.0
    dyy: PairScalar = 0.0

    def __add__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        if not isinstance(other, SecondOrderPair):
            return SecondOrderPair(
                self.value + other, self.dx, self.dy, self.dxx, self.dxy, self.dyy
            )
        return SecondOrderPair(
            self.value + other.value,
            self.dx + other.dx,
            self.dy + other.dy,
            self.dxx + other.dxx,
            self.dxy + other.dxy,
            self.dyy + other.dyy,
        )

    __radd__ = __add__

    def __neg__(self) -> SecondOrderPair:
        return self * -1

    def __sub__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        return self + -other

    def __rsub__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        return -self + other

    def __mul__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        if not isinstance(other, SecondOrderPair):
            return SecondOrderPair(
                self.value * other,
                self.dx * other,
                self.dy * other,
                self.dxx * other,
                self.dxy * other,
                self.dyy * other,
            )
        a, b = self, other
        return SecondOrderPair(
            a.value * b.value,
            a.dx * b.value + a.value * b.dx,
            a.dy * b.value + a.value * b.dy,
            a.dxx * b.value + 2 * a.dx * b.dx + a.value * b.dxx,
            a.dxy * b.value + a.dx * b.dy + a.dy * b.dx + a.value * b.dxy,
            a.dyy * b.value + 2 * a.dy * b.dy + a.value * b.dyy,
        )

    __rmul__ = __mul__

    def compose(self, value: PairScalar, first: PairScalar, second: PairScalar) -> SecondOrderPair:
        """Apply the scalar chain rule f(self) to its gradient and Hessian."""
        return SecondOrderPair(
            value,
            first * self.dx,
            first * self.dy,
            first * self.dxx + second * self.dx * self.dx,
            first * self.dxy + second * self.dx * self.dy,
            first * self.dyy + second * self.dy * self.dy,
        )

    def __pow__(self, power: float) -> SecondOrderPair:
        if power == 0:
            return SecondOrderPair(1.0)
        if power == 1:
            return self
        return self.compose(
            self.value**power,
            power * self.value ** (power - 1),
            power * (power - 1) * self.value ** (power - 2),
        )

    def __truediv__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        if isinstance(other, SecondOrderPair):
            return self * other**-1
        return self * (1 / other)

    def __rtruediv__(self, other: PairScalar | SecondOrderPair) -> SecondOrderPair:
        return self**-1 * other
