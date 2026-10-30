"""Compile declarative ChemSage G/Cp data to the existing safe Gibbs evaluator.

The Cp primitive is data normalization. It adds no caloric state, numerical
quadrature, differentiation kernel, or separate thermodynamic evaluator.
"""

from collections.abc import Sequence
from math import isfinite
from typing import ClassVar

from chemworld.data.chem_sage_records import ChemSageEndmember, ChemSageHeader, ChemSageInterval
from chemworld.data.tdb import TdbFunction, TdbSegment
from chemworld.physchem.caloric_math import FirstDerivative, exponential_minus_one
from chemworld.physchem.tdb_expressions import TdbExpression

MINIMUM_TEMPERATURE_K = .01
_GIBBS_BASIS = ("1", "T", "T*LN(T)", "T**2", "T**3", "1/T", "P", "P**2")


class ChemSageExpression(TdbExpression):
    # Compiler-only stable primitive. Raw TDB's allowed functions do not change.
    _functions: ClassVar[dict] = {**TdbExpression._functions, "EXPM1": exponential_minus_one}


def _literal(value: float) -> str:
    number = float(value)
    if not isfinite(number):
        raise ValueError("ChemSage expression constants must be finite")
    return repr(number)


def basis_expression(
    indices: Sequence[int], coefficients: Sequence[float], *,
    excess_pressure_unit: str | None = None,
) -> str:
    if len(indices) != len(coefficients):
        raise ValueError("ChemSage basis indices and coefficients must match")
    if excess_pressure_unit not in {None, "Pa", "bar"}:
        raise ValueError("ChemSage excess pressure unit must be explicitly Pa or bar")
    terms = []
    for index, coefficient in zip(indices, coefficients, strict=True):
        if not 1 <= index <= 8:
            raise ValueError("unknown ChemSage basis index")
        if coefficient == 0:
            continue
        basis = _GIBBS_BASIS[index-1]
        if index >= 7:
            if excess_pressure_unit is None:
                raise ValueError("nonzero ChemSage pressure terms require an explicit source unit")
            if excess_pressure_unit == "bar":
                basis = "P/100000" if index == 7 else "(P/100000)**2"
        terms.append(f"({_literal(coefficient)})*({basis})")
    return "+".join(terms) or "0"


def _gibbs_expression(header: ChemSageHeader, interval: ChemSageInterval) -> ChemSageExpression:
    parts = [basis_expression(header.gibbs_indices, interval.coefficients)]
    for coefficient, power in interval.extra_powers:
        if coefficient != 0:
            basis = "LN(T)" if power == 99 else f"T**({_literal(power)})"
            parts.append(f"({_literal(coefficient)})*({basis})")
    return ChemSageExpression("+".join(parts))


def _power_thermal_expression(coefficient: float, power: float, reference: float) -> str:
    """Cp=c*T**p -> delta H - T*delta S, referenced at positive Tr."""
    c, tr = _literal(coefficient), _literal(reference)
    ratio = f"(T/({tr}))"
    log_ratio = f"LN({ratio})"
    if power == 0:
        return f"({c})*(T-({tr})-T*{log_ratio})"
    if power == -1:
        return f"({c})*({log_ratio}-{ratio}+1)"
    if power == 99:
        # Integral of c*ln(T), not the power T**99.
        return (f"({c})*(T*LN(T)-T-(({tr})*LN({tr})-({tr}))"
                f"-T*(LN(T)**2-LN({tr})**2)/2)")
    # Expm1 avoids huge cancelling affine constants for powers near 0/-1.
    p, q = _literal(power), _literal(power+1)
    scale = _literal(coefficient*reference**(power+1))
    return (f"({scale})*(EXPM1(({q})*{log_ratio})/({q})"
            f"-{ratio}*EXPM1(({p})*{log_ratio})/({p}))")


def _cp_expression(
    interval: ChemSageInterval, reference: float, enthalpy: float, entropy: float,
) -> ChemSageExpression:
    if len(interval.coefficients) != 4:
        raise ValueError("ChemSage Cp requires four standard coefficients")
    terms = [f"({_literal(enthalpy)})-T*({_literal(entropy)})"]
    powers = (*zip(interval.coefficients, (0, 1, 2, -2), strict=True), *interval.extra_powers)
    if len(powers) > 24:
        raise ValueError("ChemSage Cp expression exceeds the finite shared compiler bound")
    terms.extend(_power_thermal_expression(c, p, reference) for c, p in powers if c != 0)
    return ChemSageExpression("+".join(terms))


def _enthalpy_entropy(expression: TdbExpression, temperature: float) -> tuple[float, float]:
    response = expression.evaluate({"T": FirstDerivative(temperature, 1), "P": 101325.})
    value = response.value if isinstance(response, FirstDerivative) else response
    slope = response.slope if isinstance(response, FirstDerivative) else 0.
    return float(value-temperature*slope), float(-slope)


def endmember_function(
    member: ChemSageEndmember, header: ChemSageHeader, name: str,
) -> TdbFunction:
    if not member.intervals:
        raise ValueError("ChemSage endmember requires at least one interval")
    reduced = (member.option-1) % 12+1
    if not 1 <= member.option <= 24 or reduced not in {1, 4, 7, 10}:
        raise ValueError("ChemSage volume options require their declared mechanical model")
    if any(interval.pressure_volume_terms for interval in member.intervals):
        raise ValueError("ChemSage P,T-dependent volume terms need their declared mechanical model")
    heat = reduced >= 7
    segments = []
    lower, reference = MINIMUM_TEMPERATURE_K, 298.15
    enthalpy, entropy = member.enthalpy_298_J_mol, member.entropy_298_J_mol_K
    for interval in member.intervals:
        upper = 10000. if lower == interval.upper_K else interval.upper_K
        if not lower < upper:
            raise ValueError("ChemSage interval has no supported positive temperature domain")
        if heat:
            if enthalpy is None or entropy is None:
                raise ValueError("ChemSage Cp compilation requires its actual H/S reference")
            expression = _cp_expression(interval, reference, enthalpy, entropy)
        else:
            expression = _gibbs_expression(header, interval)
        segments.append(TdbSegment(lower, upper, expression))
        lower = interval.upper_K
        if heat:
            enthalpy, entropy = _enthalpy_entropy(expression, lower)
            reference = lower
            if len(segments) < len(member.intervals):
                latent = member.intervals[len(segments)].transition_enthalpy_J_mol
                enthalpy += latent
                entropy += latent/reference
    return TdbFunction(name, tuple(segments))
