"""Elementary smooth functions on the existing nested FirstDerivative arithmetic.

This extends the shared vocabulary; it does not implement another jet or AD engine.
"""

import math

from chemworld.physchem.caloric_math import FirstDerivative, exponential, logarithm, scalar_value

SMOOTH_FUNCTIONS = frozenset((
    "sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "arcsin", "arccos", "arctan", "arcsec", "arccsc", "arccot", "arcsinh", "arccosh", "arctanh",
    "arcsech", "arccsch", "arccoth", "exp", "ln",
))
_SCALAR_FUNCTIONS = {
    **{name: getattr(math, name) for name in ("sin", "cos", "tan", "sinh", "cosh", "tanh")},
    **{name: getattr(math, name.replace("arc", "a")) for name in (
        "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh")},
    "sec": lambda x: 1/math.cos(x), "csc": lambda x: 1/math.sin(x),
    "cot": lambda x: 1/math.tan(x), "sech": lambda x: 1/math.cosh(x),
    "csch": lambda x: 1/math.sinh(x), "coth": lambda x: 1/math.tanh(x),
    "arcsec": lambda x: math.acos(1/x), "arccsc": lambda x: math.asin(1/x),
    "arccot": lambda x: math.atan(1/x) if x else math.pi/2,
    "arcsech": lambda x: math.acosh(1/x),
    "arccsch": lambda x: math.asinh(1/x), "arccoth": lambda x: math.atanh(1/x),
}


def absolute(value):
    primal = scalar_value(value)
    if isinstance(value, FirstDerivative) and primal == 0:
        raise ValueError("absolute-value derivative is undefined at zero")
    return value if primal >= 0 else -value


def real_power(base, exponent):
    if isinstance(exponent, FirstDerivative):
        if scalar_value(base) <= 0:
            raise ValueError("variable real exponent requires positive base")
        return exponential(exponent*logarithm(base))
    return base**exponent


def smooth_function(name, value):
    if name == "exp":
        return exponential(value)
    if name == "ln":
        return logarithm(value)
    if name not in _SCALAR_FUNCTIONS:
        raise ValueError("unknown elementary smooth function "+name)
    if isinstance(value, FirstDerivative):
        if name == "arccot" and scalar_value(value) == 0:
            raise ValueError("principal arccot derivative is undefined at its zero branch cut")
        x = value.value
        def f(name):
            return smooth_function(name, x)

        slopes = {
            "sin": lambda: f("cos"), "cos": lambda: -f("sin"),
            "tan": lambda: 1+f("tan")**2, "sec": lambda: f("sec")*f("tan"),
            "csc": lambda: -f("csc")*f("cot"), "cot": lambda: -1-f("cot")**2,
            "sinh": lambda: f("cosh"), "cosh": lambda: f("sinh"),
            "tanh": lambda: 1-f("tanh")**2, "sech": lambda: -f("sech")*f("tanh"),
            "csch": lambda: -f("csch")*f("coth"), "coth": lambda: 1-f("coth")**2,
            "arcsin": lambda: (1-x*x)**(-.5), "arccos": lambda: -(1-x*x)**(-.5),
            "arctan": lambda: 1/(1+x*x), "arccot": lambda: -1/(1+x*x),
            "arcsec": lambda: 1/(absolute(x)*(x*x-1)**.5),
            "arccsc": lambda: -1/(absolute(x)*(x*x-1)**.5),
            "arcsinh": lambda: (1+x*x)**(-.5), "arccosh": lambda: (x*x-1)**(-.5),
            "arctanh": lambda: 1/(1-x*x), "arccoth": lambda: 1/(1-x*x),
            "arcsech": lambda: -1/(x*(1-x*x)**.5),
            "arccsch": lambda: -1/(absolute(x)*(1+x*x)**.5),
        }
        return FirstDerivative(smooth_function(name, x), slopes[name]()*value.slope)
    return _SCALAR_FUNCTIONS[name](value)
