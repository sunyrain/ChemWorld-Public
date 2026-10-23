"""DIPPR expressions from the public equation documentation, without source data.

The existing family registry owns coefficient declarations. These expressions
accept the shared analytic derivative numbers for T and all source coefficients.
"""

from math import isfinite

from chemworld.physchem.caloric_math import (
    exponential,
    nonnegative_power,
    oscillator_expression,
    scalar_value,
)
from chemworld.physchem.vapor_pressure_kernels import pressure_logarithm


def dippr_expression(equation_id, temperature, coefficients):
    """Property in declared source units; critical formulas remain subcritical."""
    T, c = temperature, coefficients
    if isinstance(scalar_value(T), bool) or not isfinite(scalar_value(T)) or scalar_value(T) <= 0:
        raise ValueError("DIPPR requires finite positive absolute temperature")
    equation = equation_id.removesuffix("_reciprocal")
    A, B, C, D, E, F, G = (c.get(key, 0.0) for key in "ABCDEFG")
    Tc = c.get("Tc")
    if Tc is not None and not scalar_value(Tc) > scalar_value(T):
        raise ValueError(f"{equation_id} requires T < Tc")
    try:
        if equation == "dippr100":
            value = (((((G*T + F)*T + E)*T + D)*T + C)*T + B)*T + A
        elif equation in {"dippr101", "dippr115"}:
            value = exponential(pressure_logarithm(equation, T, c))
        elif equation == "dippr102":
            value = A * nonnegative_power(T, B) / (1 + C/T + D/T**2)
        elif equation == "dippr104":
            value = A + B/T + C/T**3 + D/T**8 + E/T**9
        elif equation == "dippr105":
            if scalar_value(C) <= scalar_value(T):
                raise ValueError("DIPPR105 requires T < C")
            value = A / nonnegative_power(B, 1 + nonnegative_power(1-T/C, D))
        elif equation == "dippr106":
            Tr = T/Tc
            value = A * nonnegative_power(1-Tr, B + Tr*(C + Tr*(D + Tr*E)))
        elif equation == "dippr107":
            value = (A + B*oscillator_expression(C/T)
                     + D*oscillator_expression(E/T, cosh=True))
        elif equation == "dippr114":
            tau = 1-T/Tc
            value = (A*A/tau + B - 2*A*C*tau - A*D*tau**2 - C*C*tau**3/3
                     - C*D*tau**4/2 - D*D*tau**5/5)
        elif equation == "dippr116":
            tau = 1-T/Tc
            value = A + B*tau**.35 + C*tau**(2/3) + D*tau + E*tau**(4/3)
        elif equation == "dippr127":
            value = (A + B*oscillator_expression(C/(2*T))
                     + D*oscillator_expression(E/(2*T))
                     + F*oscillator_expression(G/(2*T)))
        else:
            raise ValueError(f"Unsupported DIPPR family {equation_id}")
        if equation_id.endswith("_reciprocal"):
            value = 1/value
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{equation_id} exceeds its finite domain or meets a pole") from exc
    return value
