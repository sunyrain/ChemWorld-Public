"""SBML unit-kind adapter onto the existing physical dimension algebra.

Scale factors are coherent SI; native compartment volumes convert to litres
only at the state boundary. Source terms remain unchanged in declarations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import cast

from chemworld.foundation.dimensions import DimensionVector, dimension_definition
from chemworld.foundation.units import convert_value

_M = dimension_definition("mass").vector
_L = dimension_definition("length").vector
_T = dimension_definition("time").vector
_N = dimension_definition("amount").vector
_K = dimension_definition("temperature").vector
_I = dimension_definition("electrical_current").vector
_ONE = DimensionVector()
_ENERGY = _M*_L**2/_T**2
_CHARGE = _I*_T
_VOLT = _ENERGY/_CHARGE
_RESISTANCE = _VOLT/_I

# SBML's closed unit-kind vocabulary, not another general-purpose unit system.
_KINDS = {
    "ampere": (_I, 1.), "becquerel": (_ONE/_T, 1.), "coulomb": (_CHARGE, 1.),
    "dimensionless": (_ONE, 1.), "farad": (_CHARGE/_VOLT, 1.),
    "gram": (_M, convert_value(1., "g", "kg")), "gray": (_L**2/_T**2, 1.),
    "henry": (_RESISTANCE*_T, 1.), "hertz": (_ONE/_T, 1.),
    "joule": (_ENERGY, 1.), "katal": (_N/_T, 1.), "kelvin": (_K, 1.),
    "kilogram": (_M, 1.), "litre": (_L**3, convert_value(1., "L", "m^3")),
    "metre": (_L, 1.), "mole": (_N, 1.), "newton": (_M*_L/_T**2, 1.),
    "ohm": (_RESISTANCE, 1.), "pascal": (_M/_L/_T**2, 1.),
    "radian": (_ONE, 1.), "second": (_T, 1.), "siemens": (_ONE/_RESISTANCE, 1.),
    "sievert": (_L**2/_T**2, 1.), "steradian": (_ONE, 1.),
    "tesla": (_VOLT*_T/_L**2, 1.), "volt": (_VOLT, 1.), "watt": (_ENERGY/_T, 1.),
    "weber": (_VOLT*_T, 1.),
}
SBML_UNIT_KINDS = frozenset((*_KINDS, "item", "candela", "lumen", "lux"))


@dataclass(frozen=True)
class SBMLUnitSignature:
    dimension: DimensionVector = _ONE
    si_factor: float = 1.

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, DimensionVector):
            raise ValueError("SBML unit must use the common DimensionVector")
        if (not isinstance(cast(object, self.si_factor), Real) or isinstance(self.si_factor, bool)
                or not isfinite(self.si_factor) or self.si_factor <= 0):
            raise ValueError("SBML unit scale must be positive finite and representable")

    def __mul__(self, other: SBMLUnitSignature) -> SBMLUnitSignature:
        return SBMLUnitSignature(self.dimension*other.dimension, self.si_factor*other.si_factor)

    def __pow__(self, exponent: float) -> SBMLUnitSignature:
        try:
            return SBMLUnitSignature(self.dimension**exponent, self.si_factor**exponent)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("SBML unit power exceeds the finite numerical domain") from exc

    def _normalized(self, value: float, expected: DimensionVector, factor: float = 1.) -> float:
        if (not isinstance(cast(object, value), Real)
                or isinstance(value, bool) or not isfinite(value)):
            raise ValueError("SBML value to normalize must be a finite real number")
        if self.dimension != expected:
            raise ValueError("SBML source units do not match the requested native dimension")
        result = value*self.si_factor*factor
        if isinstance(result, bool) or not isfinite(result):
            raise ValueError("SBML normalized value must be finite")
        return result

    def amount_mol(self, value: float) -> float:
        return self._normalized(value, _N)

    def time_s(self, value: float) -> float:
        return self._normalized(value, _T)

    def volume_l(self, value: float) -> float:
        return self._normalized(value, _L**3, convert_value(1., "m^3", "L"))


def base_unit(kind: str, *, items_per_mole: float | None = None) -> SBMLUnitSignature:
    if kind == "item":
        if (not isinstance(cast(object, items_per_mole), Real) or isinstance(items_per_mole, bool)
                or not isfinite(cast(float, items_per_mole)) or cast(float, items_per_mole) <= 0):
            raise ValueError("SBML item-to-mol conversion requires explicit items_per_mole policy")
        return SBMLUnitSignature(_N, 1/cast(float, items_per_mole))
    if kind not in _KINDS:
        raise ValueError("unsupported SBML unit kind "+str(kind))
    return SBMLUnitSignature(*_KINDS[kind])


@dataclass(frozen=True)
class SBMLUnitTerm:
    kind: str
    exponent: float
    scale: int
    multiplier: float

    def __post_init__(self) -> None:
        if self.kind not in SBML_UNIT_KINDS:
            raise ValueError("unsupported SBML unit kind "+str(self.kind))
        if (not isinstance(cast(object, self.exponent), Real) or isinstance(self.exponent, bool)
                or not isfinite(self.exponent)
                or type(self.scale) is not int or isinstance(self.multiplier, bool)
                or not isinstance(cast(object, self.multiplier), Real)
                or not isfinite(self.multiplier) or self.multiplier <= 0):
            raise ValueError("invalid SBML unit exponent, decimal scale or multiplier")

    def signature(self, *, items_per_mole: float | None = None) -> SBMLUnitSignature:
        try:
            base = base_unit(self.kind, items_per_mole=items_per_mole)
            return SBMLUnitSignature(base.dimension,
                base.si_factor*self.multiplier*10.**self.scale)**self.exponent
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("SBML unit scale exceeds the finite numerical domain") from exc


def unit_definition(
    terms: Iterable[SBMLUnitTerm], *, items_per_mole: float | None = None,
) -> SBMLUnitSignature:
    signature = SBMLUnitSignature()
    for term in terms:
        if not isinstance(term, SBMLUnitTerm):
            raise ValueError("SBML unit definition requires validated source terms")
        signature *= term.signature(items_per_mole=items_per_mole)
    return signature
