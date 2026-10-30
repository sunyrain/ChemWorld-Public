"""Pure-fluid saturation reports and inverse vapor-pressure solves."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, nextafter
from typing import cast

from chemworld.foundation.units import convert_value
from chemworld.physchem.property_reports import (
    STANDARD_PRESSURE_PA,
    ValidityPolicy,
)
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.vapor_pressure import VaporPressureReport, vapor_pressure_report

_NEAR_CRITICAL_REDUCED_LIMIT = 0.98


@dataclass(frozen=True)
class PureSaturationReport:
    """Auditable pure-fluid saturation point from a vapor-pressure correlation."""

    correlation_id: str
    equation_id: str
    method_family: str
    solve_mode: str
    saturation_temperature_K: float
    saturation_pressure_Pa: float
    target_pressure_Pa: float | None
    pressure_residual_Pa: float
    log_pressure_residual: float
    converged: bool
    iterations: int
    bracket_temperature_K: tuple[float, float] | None
    critical_temperature_K: float | None
    critical_pressure_Pa: float | None
    warnings: tuple[str, ...]
    pressure_report: VaporPressureReport
    reference_reading: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.solve_mode not in {"pressure_at_temperature", "temperature_at_pressure"}:
            raise ValueError("Unsupported pure saturation solve mode")
        if self.saturation_temperature_K <= 0 or not isfinite(
            self.saturation_temperature_K
        ):
            raise ValueError("saturation_temperature_K must be positive and finite")
        if self.saturation_pressure_Pa <= 0 or not isfinite(self.saturation_pressure_Pa):
            raise ValueError("saturation_pressure_Pa must be positive and finite")
        if not isfinite(self.pressure_residual_Pa):
            raise ValueError("pressure_residual_Pa must be finite")
        if not isfinite(self.log_pressure_residual):
            raise ValueError("log_pressure_residual must be finite")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        if self.bracket_temperature_K is not None:
            lower, upper = self.bracket_temperature_K
            if lower <= 0 or upper <= lower:
                raise ValueError("bracket_temperature_K must be positive and increasing")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    @property
    def near_critical(self) -> bool:
        return any("near_critical" in warning for warning in self.warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "correlation_id": self.correlation_id,
            "equation_id": self.equation_id,
            "method_family": self.method_family,
            "solve_mode": self.solve_mode,
            "saturation_temperature_K": self.saturation_temperature_K,
            "saturation_pressure_Pa": self.saturation_pressure_Pa,
            "target_pressure_Pa": self.target_pressure_Pa,
            "pressure_residual_Pa": self.pressure_residual_Pa,
            "log_pressure_residual": self.log_pressure_residual,
            "converged": self.converged,
            "iterations": self.iterations,
            "bracket_temperature_K": list(self.bracket_temperature_K)
            if self.bracket_temperature_K is not None
            else None,
            "critical_temperature_K": self.critical_temperature_K,
            "critical_pressure_Pa": self.critical_pressure_Pa,
            "near_critical": self.near_critical,
            "warnings": list(self.warnings),
            "pressure_report": self.pressure_report.to_dict(),
            "reference_reading": list(self.reference_reading),
        }


def pure_saturation_pressure_report(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    critical_temperature_K: float | None = None,
    critical_pressure_Pa: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> PureSaturationReport:
    """Evaluate a pure-fluid saturation pressure at a temperature."""

    _check_critical_bounds(
        temperature_K=temperature_K,
        pressure_Pa=None,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=None,
    )
    pressure_report = vapor_pressure_report(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    pressure_pa = pressure_report.pressure_pa
    warnings = _critical_warnings(
        temperature_K=temperature_K,
        pressure_Pa=pressure_pa,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
    )
    return _build_report(
        correlation,
        pressure_report=pressure_report,
        solve_mode="pressure_at_temperature",
        saturation_temperature_K=temperature_K,
        target_pressure_Pa=None,
        pressure_residual_Pa=0.0,
        log_pressure_residual=0.0,
        converged=True,
        iterations=0,
        bracket_temperature_K=None,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        warnings=warnings,
    )


def pure_saturation_temperature_report(
    correlation: PropertyCorrelation,
    *,
    pressure_Pa: float,
    temperature_bounds_K: tuple[float, float] | None = None,
    critical_temperature_K: float | None = None,
    critical_pressure_Pa: float | None = None,
    validity_policy: ValidityPolicy = "raise",
    tolerance_K: float = 1e-7,
    tolerance_log_pressure: float = 1e-10,
    max_iterations: int = 100,
) -> PureSaturationReport:
    """Find a root of P_sat(T)=P in the selected bracket, without uniqueness claims.

    A source interval is a permissible default bracket. Sanjari's known
    nonmonotonic behavior requires an explicit caller-selected bracket.
    """

    if isinstance(pressure_Pa, bool) or pressure_Pa <= 0 or not isfinite(pressure_Pa):
        raise ValueError("pressure_Pa must be positive and finite")
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0
           for v in (tolerance_K, tolerance_log_pressure)):
        raise ValueError("saturation tolerances must be positive")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations <= 0):
        raise ValueError("max_iterations must be positive")
    if correlation.equation_id == "sanjari_pressure" and temperature_bounds_K is None:
        raise ValueError("Sanjari inversion requires an explicit temperature bracket")
    _check_critical_bounds(
        temperature_K=None,
        pressure_Pa=pressure_Pa,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
    )
    lower, upper = _temperature_bounds_k(
        correlation,
        temperature_bounds_K=temperature_bounds_K,
        critical_temperature_K=critical_temperature_K,
    )
    f_lower, lower_report = _log_pressure_residual(
        correlation,
        temperature_K=lower,
        pressure_Pa=pressure_Pa,
        validity_policy=validity_policy,
    )
    f_upper, upper_report = _log_pressure_residual(
        correlation,
        temperature_K=upper,
        pressure_Pa=pressure_Pa,
        validity_policy=validity_policy,
    )
    if abs(f_lower) <= tolerance_log_pressure:
        return _temperature_solve_report(
            correlation,
            pressure_report=lower_report,
            saturation_temperature_K=lower,
            pressure_Pa=pressure_Pa,
            log_pressure_residual=f_lower,
            iterations=0,
            bracket_temperature_K=(lower, upper),
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=critical_pressure_Pa,
        )
    if abs(f_upper) <= tolerance_log_pressure:
        return _temperature_solve_report(
            correlation,
            pressure_report=upper_report,
            saturation_temperature_K=upper,
            pressure_Pa=pressure_Pa,
            log_pressure_residual=f_upper,
            iterations=0,
            bracket_temperature_K=(lower, upper),
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=critical_pressure_Pa,
        )
    if f_lower * f_upper > 0:
        raise ValueError(
            "pressure_Pa is outside the saturation bracket for "
            f"{correlation.correlation_id}: residuals=({f_lower:g}, {f_upper:g})"
        )

    left = lower
    right = upper
    f_left = f_lower
    final_report = lower_report
    final_residual = f_lower
    final_temperature_K = lower
    iterations = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        mid = 0.5 * (left + right)
        f_mid, mid_report = _log_pressure_residual(
            correlation,
            temperature_K=mid,
            pressure_Pa=pressure_Pa,
            validity_policy=validity_policy,
        )
        final_report = mid_report
        final_residual = f_mid
        final_temperature_K = mid
        iterations = iteration
        if abs(f_mid) <= tolerance_log_pressure or (right - left) <= tolerance_K:
            converged = True
            break
        if f_left * f_mid <= 0:
            right = mid
        else:
            left = mid
            f_left = f_mid
    if not converged:
        raise ValueError(
            "pure saturation temperature solve did not converge within "
            f"{max_iterations} iterations"
        )
    return _temperature_solve_report(
        correlation,
        pressure_report=final_report,
        saturation_temperature_K=final_temperature_K,
        pressure_Pa=pressure_Pa,
        log_pressure_residual=final_residual,
        iterations=iterations,
        bracket_temperature_K=(lower, upper),
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
    )


def normal_boiling_point_report(
    correlation: PropertyCorrelation,
    *,
    pressure_Pa: float = STANDARD_PRESSURE_PA,
    temperature_bounds_K: tuple[float, float] | None = None,
    critical_temperature_K: float | None = None,
    critical_pressure_Pa: float | None = None,
    validity_policy: ValidityPolicy = "raise",
) -> PureSaturationReport:
    """Solve a normal or caller-specified boiling point from a vapor-pressure curve."""

    return pure_saturation_temperature_report(
        correlation,
        pressure_Pa=pressure_Pa,
        temperature_bounds_K=temperature_bounds_K,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        validity_policy=validity_policy,
    )


def _temperature_solve_report(
    correlation: PropertyCorrelation,
    *,
    pressure_report: VaporPressureReport,
    saturation_temperature_K: float,
    pressure_Pa: float,
    log_pressure_residual: float,
    iterations: int,
    bracket_temperature_K: tuple[float, float],
    critical_temperature_K: float | None,
    critical_pressure_Pa: float | None,
) -> PureSaturationReport:
    pressure_residual = pressure_report.pressure_pa - pressure_Pa
    warnings = _critical_warnings(
        temperature_K=saturation_temperature_K,
        pressure_Pa=pressure_report.pressure_pa,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
    )
    return _build_report(
        correlation,
        pressure_report=pressure_report,
        solve_mode="temperature_at_pressure",
        saturation_temperature_K=saturation_temperature_K,
        target_pressure_Pa=pressure_Pa,
        pressure_residual_Pa=pressure_residual,
        log_pressure_residual=log_pressure_residual,
        converged=True,
        iterations=iterations,
        bracket_temperature_K=bracket_temperature_K,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        warnings=tuple(dict.fromkeys((*warnings, "bracketed_root_only_uniqueness_not_certified",
            *(("open_critical_endpoint_uses_next_representable_temperature",)
              if ((correlation.equation_id == "wagner"
                   and bracket_temperature_K[1] == nextafter(correlation.coefficients["Tc"], 0))
                  or (critical_temperature_K is not None
                      and bracket_temperature_K[1] == nextafter(critical_temperature_K, 0)))
              else ())))),
    )


def _build_report(
    correlation: PropertyCorrelation,
    *,
    pressure_report: VaporPressureReport,
    solve_mode: str,
    saturation_temperature_K: float,
    target_pressure_Pa: float | None,
    pressure_residual_Pa: float,
    log_pressure_residual: float,
    converged: bool,
    iterations: int,
    bracket_temperature_K: tuple[float, float] | None,
    critical_temperature_K: float | None,
    critical_pressure_Pa: float | None,
    warnings: tuple[str, ...],
) -> PureSaturationReport:
    return PureSaturationReport(
        correlation_id=correlation.correlation_id,
        equation_id=correlation.equation_id,
        method_family=pressure_report.method_family,
        solve_mode=solve_mode,
        saturation_temperature_K=saturation_temperature_K,
        saturation_pressure_Pa=pressure_report.pressure_pa,
        target_pressure_Pa=target_pressure_Pa,
        pressure_residual_Pa=pressure_residual_Pa,
        log_pressure_residual=log_pressure_residual,
        converged=converged,
        iterations=iterations,
        bracket_temperature_K=bracket_temperature_K,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        warnings=tuple(dict.fromkeys((*pressure_report.pressure.warnings, *warnings))),
        pressure_report=pressure_report,
        reference_reading=(
            "reference_repos/thermo/thermo/vapor_pressure.py: VaporPressure "
            "method governance and validity limits",
            "reference_repos/coolprop/Web/coolprop/HighLevelAPI.rst: "
            "saturation inputs with quality and critical-region cautions",
            "reference_repos/phasepy/phasepy/mixtures.py: Psat/Tsat inverse "
            "API shape for Antoine-like correlations",
        ),
    )


def _log_pressure_residual(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    pressure_Pa: float,
    validity_policy: ValidityPolicy,
) -> tuple[float, VaporPressureReport]:
    report = vapor_pressure_report(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    return log(report.pressure_pa / pressure_Pa), report


def _temperature_bounds_k(
    correlation: PropertyCorrelation,
    *,
    temperature_bounds_K: tuple[float, float] | None,
    critical_temperature_K: float | None,
) -> tuple[float, float]:
    source_bounds = correlation.validity_ranges.get("temperature")
    if source_bounds is not None:
        unit = correlation.input_units.get("temperature", "K")
        source_bounds = cast(tuple[float, float],
                             tuple(convert_value(v, unit, "K") for v in source_bounds))
    if temperature_bounds_K is None:
        if source_bounds is None:
            raise ValueError(
                "temperature_bounds_K or correlation temperature validity range "
                "is required for saturation temperature solves")
        lower, upper = source_bounds
        if correlation.equation_id == "wagner" and upper == correlation.coefficients["Tc"]:
            upper = nextafter(upper, 0)
        if critical_temperature_K is not None and upper >= critical_temperature_K:
            upper = nextafter(critical_temperature_K, 0)
    else:
        lower, upper = temperature_bounds_K
    if (any(isinstance(v, bool) or not isfinite(v) for v in (lower, upper))
            or lower <= 0 or upper <= lower):
        raise ValueError("temperature_bounds_K must be positive and increasing")
    if source_bounds is not None and (lower < source_bounds[0] or upper > source_bounds[1]):
        raise ValueError(
            "saturation bracket exceeds source domain; select an explicit extrapolation")
    if critical_temperature_K is not None:
        _check_critical_bounds(
            temperature_K=lower,
            pressure_Pa=None,
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=None,
        )
        if upper >= critical_temperature_K:
            raise ValueError("select a temperature bracket below critical_temperature_K")
    return lower, upper


def _critical_warnings(
    *,
    temperature_K: float,
    pressure_Pa: float,
    critical_temperature_K: float | None,
    critical_pressure_Pa: float | None,
) -> tuple[str, ...]:
    _check_critical_bounds(
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
    )
    warnings: list[str] = []
    if (
        critical_temperature_K is not None
        and temperature_K / critical_temperature_K >= _NEAR_CRITICAL_REDUCED_LIMIT
    ):
        warnings.append("near_critical_temperature")
    if (
        critical_pressure_Pa is not None
        and pressure_Pa / critical_pressure_Pa >= _NEAR_CRITICAL_REDUCED_LIMIT
    ):
        warnings.append("near_critical_pressure")
    return tuple(warnings)


def _check_critical_bounds(
    *,
    temperature_K: float | None,
    pressure_Pa: float | None,
    critical_temperature_K: float | None,
    critical_pressure_Pa: float | None,
) -> None:
    if critical_temperature_K is not None:
        if critical_temperature_K <= 0 or not isfinite(critical_temperature_K):
            raise ValueError("critical_temperature_K must be positive and finite")
        if temperature_K is not None and temperature_K >= critical_temperature_K:
            raise ValueError("saturation temperature must be below critical_temperature_K")
    if critical_pressure_Pa is not None:
        if critical_pressure_Pa <= 0 or not isfinite(critical_pressure_Pa):
            raise ValueError("critical_pressure_Pa must be positive and finite")
        if pressure_Pa is not None and pressure_Pa >= critical_pressure_Pa:
            raise ValueError("saturation pressure must be below critical_pressure_Pa")


__all__ = [
    "PureSaturationReport",
    "normal_boiling_point_report",
    "pure_saturation_pressure_report",
    "pure_saturation_temperature_report",
]
