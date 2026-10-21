"""Vapor-pressure property reports and derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

from chemworld.physchem.caloric_math import derivative_values, exponential
from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.property_equations import (
    _convert_input,
    _vapor_pressure_method_family,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import (
    PropertyEvaluation,
    ValidityPolicy,
    _pressure_unit_to_pa_factor,
)
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.vapor_pressure_kernels import pressure_derivatives, pressure_logarithm


@dataclass(frozen=True)
class VaporPressureReport:
    """Vapor/sublimation pressure with derivative and method provenance."""

    pressure: PropertyEvaluation
    temperature_derivative_value: float
    temperature_derivative_unit: str
    dln_pressure_dT_1_K: float
    method_family: str
    validity_status: str
    reference_reading: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pressure.property_id not in {"vapor_pressure", "sublimation_pressure"}:
            raise ValueError("VaporPressureReport requires vapor or sublimation pressure")
        if not isfinite(self.temperature_derivative_value):
            raise ValueError("temperature derivative must be finite")
        if self.pressure.value <= 0:
            raise ValueError("pressure value must be positive")
        if not isfinite(self.dln_pressure_dT_1_K):
            raise ValueError("log-pressure derivative must be finite")
        if self.validity_status not in {"valid", "out_of_range"}:
            raise ValueError("validity_status must be valid or out_of_range")
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    @property
    def pressure_pa(self) -> float:
        return self.pressure.to("Pa").value

    @property
    def dp_dt_pa_per_k(self) -> float:
        return self.temperature_derivative_value * _pressure_unit_to_pa_factor(
            self.pressure.unit
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pressure": self.pressure.to_dict(),
            "temperature_derivative_value": self.temperature_derivative_value,
            "temperature_derivative_unit": self.temperature_derivative_unit,
            "dP_dT_Pa_per_K": self.dp_dt_pa_per_k,
            "dln_pressure_dT_1_K": self.dln_pressure_dT_1_K,
            "method_family": self.method_family,
            "validity_status": self.validity_status,
            "reference_reading": list(self.reference_reading),
        }


def vapor_pressure_report(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    validity_policy: ValidityPolicy = "warn",
) -> VaporPressureReport:
    """Evaluate pressure and analytic temperature derivative for vapor models."""

    if correlation.property_id not in {"vapor_pressure", "sublimation_pressure"}:
        raise ValueError("vapor_pressure_report requires vapor or sublimation pressure")
    pressure = evaluate_correlation(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    derivative = vapor_pressure_temperature_derivative(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    pressure_value = pressure.value
    dln = derivative.value / pressure_value
    return VaporPressureReport(
        pressure=pressure,
        temperature_derivative_value=derivative.value,
        temperature_derivative_unit=derivative.unit,
        dln_pressure_dT_1_K=dln,
        method_family=_vapor_pressure_method_family(correlation.equation_id),
        validity_status="valid" if not pressure.warnings else "out_of_range",
        reference_reading=(
            "reference_repos/chemicals/chemicals/vapor_pressure.py: "
            "Antoine, Wagner, dWagner_dT",
            "reference_repos/chemicals/chemicals/dippr.py: EQ101 order=1 "
            "derivative",
            "reference_repos/thermo/thermo/vapor_pressure.py: "
            "VaporPressure ranked methods and validity limits",
        ),
    )


def vapor_pressure_temperature_derivative(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    validity_policy: ValidityPolicy = "warn",
    order: int = 1,
) -> PropertyEvaluation:
    """Analytic first through third derivative in output units per kelvin power."""

    if correlation.property_id not in {"vapor_pressure", "sublimation_pressure"}:
        raise ValueError(
            "vapor_pressure_temperature_derivative requires vapor or "
            "sublimation pressure"
        )
    pressure = evaluate_correlation(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    inputs = {
        "temperature": _convert_input(temperature_K, "K", correlation, "temperature")
    }
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 3:
        raise ValueError("pressure derivative order must be an integer from one through three")
    equation, T = _pressure_kernel(correlation, temperature_K)
    derivative = pressure_derivatives(equation, T, correlation.coefficients, order=order)[order]
    if not isfinite(derivative):
        raise ValueError(
            f"Correlation derivative returned non-finite value: "
            f"{correlation.correlation_id}"
        )
    return PropertyEvaluation(
        property_id=f"{correlation.property_id}_temperature_derivative"
                    + (f"_{order}" if order > 1 else ""),
        correlation_id=f"{correlation.correlation_id}:dP_dT" + (str(order) if order > 1 else ""),
        equation_id=f"{correlation.equation_id}_temperature_derivative",
        value=derivative,
        unit=f"{correlation.output_unit}/K" + (f"^{order}" if order > 1 else ""),
        inputs=inputs,
        warnings=pressure.warnings,
    )


def _pressure_kernel(correlation, temperature_K):
    if correlation.property_id not in {"vapor_pressure", "sublimation_pressure"}:
        raise ValueError("a vapor or sublimation pressure correlation is required")
    equation = correlation.equation_id
    if equation == "dippr101_vapor_pressure":
        equation = "dippr101"
    family = CORRELATION_FAMILIES.get(equation)
    if equation not in {"antoine", "wagner", "dippr101", "dippr115"} and not (
            family is not None and family.pressure_method):
        raise ValueError(f"unsupported pressure derivative equation {equation!r}")
    T = (temperature_K if equation == "wagner" else
         _convert_input(temperature_K, "K", correlation, "temperature"))
    return equation, T


@dataclass(frozen=True)
class VaporPressureJacobian:
    """Rows follow temperatures; columns follow named source-coefficient coordinates."""

    temperatures_K: tuple[float, ...]
    parameters: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    pressure_unit: str
    warnings: tuple[tuple[str, ...], ...]

    def to_dict(self):
        return {"temperatures_K": list(self.temperatures_K), "parameters": list(self.parameters),
                "values": [list(row) for row in self.values], "pressure_unit": self.pressure_unit,
                "parameter_basis": "declared source-coefficient coordinates, no normalization",
                "warnings": [list(row) for row in self.warnings]}


def vapor_pressure_parameter_jacobian(correlation, temperatures_K, *, parameters=None,
                                      validity_policy="raise"):
    """Analytic Jacobian of the actual pressure, including source log/unit conventions."""
    names = tuple(correlation.coefficients) if parameters is None else tuple(parameters)
    if len(set(names)) != len(names) or any(name not in correlation.coefficients for name in names):
        raise ValueError("Jacobian columns must be distinct explicitly supplied coefficients")
    temperatures, rows, warnings = tuple(temperatures_K), [], []
    if not temperatures:
        raise ValueError("a pressure Jacobian needs at least one temperature")
    for temperature in temperatures:
        evaluation = evaluate_correlation(correlation, temperature_K=temperature,
                                          validity_policy=validity_policy)
        equation, T = _pressure_kernel(correlation, temperature)
        row = []
        for name in names:
            if (equation == "trc_antoine_extended" and name == "to"
                    and correlation.coefficients["to"]+273.15 == T
                    and correlation.coefficients["n"] <= 1):
                raise ValueError("TRC parameter derivative does not exist at the branch")
            def function(value, name=name, equation=equation, T=T):
                return exponential(pressure_logarithm(equation, T,
                    {**correlation.coefficients, name: value}))
            row.append(derivative_values(function, correlation.coefficients[name], order=1)[1])
        rows.append(tuple(row))
        warnings.append(evaluation.warnings)
    return VaporPressureJacobian(temperatures, names, tuple(rows), correlation.output_unit,
                                 tuple(warnings))


def _local_pressure_point(temperature_K, pressure_Pa, first_derivative_Pa_K):
    if any(isinstance(v, bool) or not isfinite(v) for v in
           (temperature_K, pressure_Pa, first_derivative_Pa_K)):
        raise ValueError("local pressure data must be finite numbers")
    if temperature_K <= 0 or pressure_Pa <= 0:
        raise ValueError("local pressure data require positive absolute T and P")
    return log(pressure_Pa), first_derivative_Pa_K/pressure_Pa


def antoine_coefficients_from_point(*, temperature_K, pressure_Pa, first_derivative_Pa_K,
                                    second_derivative_Pa_K2=None, base=10.0):
    """Local A/B (C=0) or A/B/C matching; this is not a global data regression."""
    L, slope = _local_pressure_point(temperature_K, pressure_Pa, first_derivative_Pa_K)
    if isinstance(base, bool) or not isfinite(base) or base <= 0 or base == 1:
        raise ValueError("Antoine base must be positive and unequal to one")
    denominator = temperature_K
    if second_derivative_Pa_K2 is not None:
        if isinstance(second_derivative_Pa_K2, bool) or not isfinite(second_derivative_Pa_K2):
            raise ValueError("local Antoine fit requires finite second derivative")
        curvature = second_derivative_Pa_K2/pressure_Pa-slope*slope
        if not curvature or not slope:
            raise ValueError("local A/B/C point is degenerate; no unique finite Antoine fit")
        denominator = -2*slope/curvature
        if denominator <= 0:
            raise ValueError("local data imply an invalid Antoine T+C denominator")
    return {"A": (L+slope*denominator)/log(base),
            "B": slope*denominator**2/log(base), "C": denominator-temperature_K, "base": base}


def dippr101_coefficients_from_point(*, temperature_K, pressure_Pa, first_derivative_Pa_K,
                                     second_derivative_Pa_K2):
    """Local A/B/C matching; infinite second derivative explicitly selects the A/C limit.

    For an infinite target curvature, B=0 and only value and first derivative
    are matched, as in the documented critical-point convention. The result
    must not be described as reproducing the infinite curvature.
    """
    L, slope = _local_pressure_point(temperature_K, pressure_Pa, first_derivative_Pa_K)
    T, second = temperature_K, second_derivative_Pa_K2
    if isinstance(second, bool) or second != second:
        raise ValueError("second derivative must be finite or an explicit infinity")
    if isfinite(second):
        curvature = second/pressure_Pa-slope*slope
        B, C = T*T*(slope+T*curvature), 2*T*slope+T*T*curvature
    else:
        B, C = 0.0, T*slope
    return {"A": L-B/T-C*log(T), "B": B, "C": C, "D": 0.0, "E": 0.0}


def arrhenius_pressure_extrapolation(correlation, *, anchor_temperature_K, temperature_bounds_K):
    """Explicit tangent in ln(P) versus 1/T; never a failed-method fallback."""
    pressure = vapor_pressure_report(correlation, temperature_K=anchor_temperature_K,
                                     validity_policy="raise")
    lower, upper = temperature_bounds_K
    if any(isinstance(v, bool) or not isfinite(v) for v in (lower, upper)) or not 0 < lower < upper:
        raise ValueError("extrapolation needs an explicit finite positive temperature interval")
    return PropertyCorrelation(correlation.correlation_id+":arrhenius", "arrhenius_pressure",
        {"T_ref": anchor_temperature_K, "P_ref": pressure.pressure.value,
         "slope": -anchor_temperature_K**2*pressure.dln_pressure_dT_1_K},
        {"temperature": "K"}, correlation.output_unit, correlation.property_id,
        {"temperature": (lower, upper)},
        correlation.source_note+"; explicit Arrhenius extrapolation",
        {"source_correlation": correlation.to_dict(), "extrapolation": "explicit_lnP_inverseT",
         "anchor_temperature_K": anchor_temperature_K,
         "matched_at_anchor": ["pressure", "first_temperature_derivative"]})

