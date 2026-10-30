"""DIPPR derivatives and local coefficient recovery for declared property consumers.

No fit, method selection or material-accuracy claim is implied by a derivative or
a local Taylor match. All parameter derivatives use the declared source units.
"""

from dataclasses import asdict, dataclass
from functools import partial
from math import exp, isfinite, log, ulp

from chemworld.foundation.units import convert_value
from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.dippr_kernels import dippr_expression
from chemworld.physchem.property_equations import evaluate_correlation


def _point(correlation, temperature_K, validity_policy):
    if not correlation.equation_id.startswith("dippr"):
        raise ValueError("DIPPR response requires a declared DIPPR correlation")
    equation = correlation.equation_id
    if equation == "dippr101_vapor_pressure":
        equation = "dippr101"
    result = evaluate_correlation(correlation, temperature_K=temperature_K,
                                  validity_policy=validity_policy)
    return equation, result


@dataclass(frozen=True)
class DIPPRTemperatureResponse:
    """Value followed by dY/dT, d²Y/dT², d³Y/dT³ as requested.

    Entry n is in ``output_unit / K**n``; the base unit is never silently
    changed to SI and coefficient units are never rescaled independently.
    """

    correlation_id: str
    temperature_K: float
    values: tuple[float, ...]
    output_unit: str
    warnings: tuple[str, ...]

    def to(self, output_unit):
        scale = convert_value(1, self.output_unit, output_unit)
        return DIPPRTemperatureResponse(self.correlation_id, self.temperature_K,
            tuple(v*scale for v in self.values), output_unit, self.warnings)

    def to_dict(self):
        return {**asdict(self), "temperature_basis": "K",
                "orders": list(range(len(self.values))),
                "value_units": [self.output_unit if i == 0 else f"({self.output_unit})/K^{i}"
                                for i in range(len(self.values))]}


def dippr_temperature_response(correlation, *, temperature_K, order=3, validity_policy="raise"):
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 3:
        raise ValueError("DIPPR derivative order must be an integer from zero through three")
    equation, result = _point(correlation, temperature_K, validity_policy)
    values = derivative_values(
        lambda t: dippr_expression(equation, t, correlation.coefficients),
        temperature_K, order=order,
    )
    return DIPPRTemperatureResponse(correlation.correlation_id, temperature_K, values,
                                    correlation.output_unit, result.warnings)


@dataclass(frozen=True)
class DIPPRParameterJacobian:
    temperatures_K: tuple[float, ...]
    parameters: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    output_unit: str
    warnings: tuple[tuple[str, ...], ...]

    def to_dict(self):
        return {**asdict(self),
                "parameter_basis": "declared source-coefficient coordinates, no normalization"}


def dippr_parameter_jacobian(correlation, temperatures_K, *, parameters=None,
                             validity_policy="raise"):
    """Rows follow temperatures, columns the explicit source coefficients, including Tc.

    To match the five published fitting Jacobians, select A-E (A-D for 102/105).
    An omitted optional coefficient must be explicitly supplied before fitting it.
    """
    names = tuple(correlation.coefficients) if parameters is None else tuple(parameters)
    if (not names or len(set(names)) != len(names)
            or any(name not in correlation.coefficients for name in names)):
        raise ValueError("Jacobian needs distinct explicitly supplied coefficient names")
    temperatures, rows, warnings = tuple(temperatures_K), [], []
    if not temperatures:
        raise ValueError("Jacobian needs at least one temperature")
    for T in temperatures:
        equation, result = _point(correlation, T, validity_policy)
        row = tuple(partial_derivative(partial(dippr_expression, equation, T),
                    correlation.coefficients, (name,)) for name in names)
        rows.append(row)
        warnings.append(result.warnings)
    return DIPPRParameterJacobian(temperatures, names, tuple(rows), correlation.output_unit,
                                  tuple(warnings))


@dataclass(frozen=True)
class DIPPRMixedDerivative:
    temperature_K: float
    coordinates: tuple[str, ...]
    value: float
    output_unit: str
    warnings: tuple[str, ...]

    def to_dict(self):
        return {**asdict(self), "coordinate_basis":
                "temperature in K; coefficients in declared source coordinates; "
                "value unit is output_unit divided by every requested coordinate unit"}


def dippr_mixed_derivative(correlation, *, temperature_K, coordinates, validity_policy="raise"):
    """Temperature/parameter mixed partial through total order three, without differencing."""
    equation, result = _point(correlation, temperature_K, validity_policy)
    names = tuple(coordinates)
    values = {"temperature": temperature_K, **correlation.coefficients}

    def expression(p):
        return dippr_expression(equation, p["temperature"],
                                 {key: p[key] for key in correlation.coefficients})

    derivative = partial_derivative(expression, values, names)
    return DIPPRMixedDerivative(temperature_K, names, derivative, correlation.output_unit,
                                result.warnings)


def dippr106_coefficients_from_point(*, temperature_K, critical_temperature_K, value,
                                      first_derivative, second_derivative=None):
    """Recover A/B or A/B/C, matching one subcritical point in caller-supplied output units.

    D=E=0. With a second derivative, log(tau)+2=0 is a rank-deficient local
    system and has no unique ABC solution. This does not fit a temperature range.
    """
    T, Tc = temperature_K, critical_temperature_K
    supplied = (T, Tc, value, first_derivative)
    if second_derivative is not None:
        supplied += (second_derivative,)
    if any(isinstance(v, bool) or not isfinite(v) for v in supplied):
        raise ValueError("local DIPPR data must be finite real numbers")
    if not 0 < T < Tc or value <= 0:
        raise ValueError("local DIPPR recovery requires 0 < T < Tc and positive value")
    tau, distance = 1-T/Tc, Tc-T
    log_tau, slope = log(tau), first_derivative/value
    C = 0.0
    exponent = -distance*slope
    if second_derivative is not None:
        determinant = log_tau+2
        if abs(determinant) <= 32*ulp(max(1.0, abs(log_tau))):
            raise ValueError("local DIPPR106 ABC recovery is rank deficient at log(tau)+2=0")
        curvature = second_derivative/value - slope*slope
        C = Tc*(slope-distance*curvature)/determinant
        exponent = distance*(C/Tc*log_tau-slope)
    try:
        A = exp(log(value)-exponent*log_tau)
    except OverflowError as exc:
        raise ValueError("local DIPPR recovery exceeds finite coefficient range") from exc
    B = exponent-C*T/Tc
    if not all(isfinite(v) for v in (A, B, C)) or A <= 0:
        raise ValueError("local DIPPR recovery has no finite positive amplitude")
    return {"Tc": Tc, "A": A, "B": B, "C": C, "D": 0.0, "E": 0.0}
