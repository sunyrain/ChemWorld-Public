"""Local frequency response of the shared residual, without advancing a world.

The supplied point must be stationary. Coefficients are frozen at its declared
time; this is not a stability certificate or a nonlinear periodic simulation.
Inputs are the residual's declared parameters. Direct output feedthrough can be
represented by algebraic rows, exactly as in the time-domain residual.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy import sparse

from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    ResidualNumericalError,
    in_bounds,
    linear_solve,
    vector,
)


@dataclass(frozen=True)
class FrequencyPoint:
    frequency_Hz: float
    response: np.ndarray | None
    relative_linear_residual: float | None
    failure_reason: str | None = None

    def __post_init__(self):
        if self.response is not None:
            owned = np.array(self.response, dtype=complex, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, "response", owned)

    @property
    def success(self):
        return self.response is not None and self.failure_reason is None

    def to_dict(self):
        return {
            "frequency_Hz": self.frequency_Hz,
            "success": self.success,
            "response_real": None if self.response is None else self.response.real.tolist(),
            "response_imag": None if self.response is None else self.response.imag.tolist(),
            "relative_linear_residual": self.relative_linear_residual,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class FrequencyResponse:
    system_id: str
    variables: tuple
    parameters: tuple
    points: tuple[FrequencyPoint, ...]
    diagnostics: dict

    @property
    def success(self):
        return all(point.success for point in self.points)

    def to_dict(self):
        return {
            "system_id": self.system_id,
            "success": self.success,
            "variables": [{"name": v.name, "unit": v.unit} for v in self.variables],
            "parameters": [{"name": p.name, "unit": p.unit} for p in self.parameters],
            "response_units": "variable unit / parameter unit",
            "phasor_convention": "exp(+i*2*pi*f*t)",
            "points": [point.to_dict() for point in self.points],
            "diagnostics": dict(self.diagnostics),
        }


def residual_frequency_response(
    system,
    state,
    frequencies_Hz,
    *,
    time_s=0.0,
    stationary_tolerance=1e-9,
    linear_tolerance=1e-10,
):
    """Solve (F_y + i omega F_ydot) dy/dp = -F_p at a stationary point.

    Each frequency has an independent success/failure record, in input order.
    Invalid inputs or a nonstationary point reject the entire analysis before
    any spectrum is computed. Model callback programming errors propagate.
    """
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    if (
        frequencies.ndim != 1
        or not frequencies.size
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0)
        or np.any(frequencies > np.finfo(float).max / (2 * np.pi))
    ):
        raise ValueError("frequencies must be a nonempty vector of finite positive Hz")
    if any(not isfinite(v) or v <= 0 for v in (stationary_tolerance, linear_tolerance)):
        raise ValueError("frequency response tolerances must be finite and positive")
    if not isfinite(time_s):
        raise ValueError("linearization time must be finite")
    if not system.parameters:
        raise ValueError("frequency response requires declared input parameters")
    state = vector(state, len(system.variables), "linearization state")
    if not in_bounds(state, system.variables):
        raise ResidualDomainError("linearization state outside declared bounds")
    evaluator = ResidualEvaluator(system)
    zero = np.zeros(len(state))
    residual = evaluator.value(time_s, state, zero)
    maximum = float(np.max(np.abs(residual / system.residual_scales)))
    if maximum > stationary_tolerance:
        raise ResidualDomainError(f"nonstationary linearization point: scaled residual {maximum}")
    jy = evaluator.derivative("state", time_s, state, zero)
    mass = evaluator.derivative("rate", time_s, state, zero)
    jp = evaluator.derivative("parameter", time_s, state, zero)
    rows, columns = system.residual_scales, system.scales
    inputs = np.array([p.scale for p in system.parameters])
    left, right = sparse.diags(1 / rows), sparse.diags(columns)
    jy = (left @ sparse.csc_matrix(jy) @ right).tocsc()
    mass = (left @ sparse.csc_matrix(mass) @ right).tocsc()
    forcing = -(left @ sparse.csc_matrix(jp) @ sparse.diags(inputs)).toarray()
    points = []
    for frequency in frequencies:
        error = None
        try:
            operator = jy.astype(complex) + (2j * np.pi * frequency) * mass
            scaled = linear_solve(operator, forcing)
            # Componentwise scale-aware backward error, per input column.
            difference = np.abs(operator @ scaled - forcing)
            magnitude = np.abs(operator) @ np.abs(scaled) + np.abs(forcing)
            denominator = np.max(magnitude, axis=0)
            numerator = np.max(difference, axis=0)
            errors = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
            )
            errors[(denominator == 0) & (numerator != 0)] = np.inf
            error = float(np.max(errors))
            if not isfinite(error) or error > linear_tolerance:
                raise ResidualNumericalError("frequency_linear_residual_exceeded")
            response = columns[:, None] * scaled / inputs[None, :]
            if not np.all(np.isfinite(response)):
                raise ResidualNumericalError("nonfinite_frequency_response")
            points.append(FrequencyPoint(float(frequency), response, error))
        except ResidualNumericalError as exc:
            points.append(
                FrequencyPoint(
                    float(frequency),
                    None,
                    error if error is not None and isfinite(error) else None,
                    str(exc),
                )
            )
    return FrequencyResponse(
        system.system_id,
        system.variables,
        system.parameters,
        tuple(points),
        {
            **evaluator.diagnostics(),
            "time_s": float(time_s),
            "maximum_scaled_stationary_residual": maximum,
            "stationary_tolerance": stationary_tolerance,
            "linear_tolerance": linear_tolerance,
            "completed": len(points),
            "passed": sum(point.success for point in points),
            "total": len(frequencies),
            "physical_time_advanced_s": 0.0,
            "scope": "local frozen-time linearization; not a stability certificate",
        },
    )


__all__ = ["FrequencyPoint", "FrequencyResponse", "residual_frequency_response"]
