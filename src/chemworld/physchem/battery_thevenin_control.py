"""Sourced, persistent external-circuit controls on the shared residual solver.

Named equations follow PyBaMM 25.4.2's actual external-circuit submodels.
In particular differential resistance keeps the upstream negative sign; its
positive-current fixed point is generally unstable, not silently stabilised.
These declarations own neither a clock nor an integrator.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite
from types import MappingProxyType

import numpy as np

from chemworld.data.scientific_inputs import ScientificTable, _keys, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.caloric_math import Interval
from chemworld.physchem.coupled_system import CoupledCoordinate
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualDomainError, ResidualVariable
from chemworld.physchem.root_isolation import isolate_roots

CONTROLLER_CURRENT = "thevenin:controller_current_A"
CONTROL_UNITS = {
    "initial_current_A": "A",
    "current_scale_A": "A",
    "power_scale_W": "W",
    "resistance_scale_Ohm": "Ohm",
    "voltage_scale_V": "V",
    "power_response_time_s": "s",
    "resistance_response_time_s": "s",
    "voltage_response_time_s": "s",
    "antiwindup_time_s": "s",
}
TARGET_UNITS = {
    "current": "A",
    "voltage": "V",
    "power": "W",
    "resistance": "Ohm",
    "differential_power": "W",
    "differential_resistance": "Ohm",
    "cccv": "V",
}
DYNAMIC_MODES = frozenset(("differential_power", "differential_resistance", "cccv"))


@dataclass(frozen=True)
class TheveninControlSpec:
    parameters: ScientificTable
    parameter_row_id: str
    convention: str = "pybamm-25.4.2-named"
    _native: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("Thevenin controller requires a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {row.row_id: row for row in self.parameters.rows}
        if self.parameter_row_id not in rows:
            raise ValueError("Thevenin controller parameter row does not exist")
        row = rows[self.parameter_row_id]
        if row.values.keys() != CONTROL_UNITS.keys():
            raise ValueError("Thevenin controller constants must match its declared equations")
        values = {
            key: convert_value(row.values[key].value, row.values[key].unit, unit)
            for key, unit in CONTROL_UNITS.items()
        }
        if any(
            not isfinite(v) or (key != "initial_current_A" and v <= 0) for key, v in values.items()
        ):
            raise ValueError("controller scales and response times must be positive and finite")
        if self.convention != "pybamm-25.4.2-named":
            raise ValueError("unknown Thevenin controller equation convention")
        gains = (values["current_scale_A"] / values[scale] / values[time]
                 for scale, time in (("power_scale_W", "power_response_time_s"),
                                     ("resistance_scale_Ohm", "resistance_response_time_s"),
                                     ("voltage_scale_V", "voltage_response_time_s")))
        if any(not isfinite(gain) or gain <= 0 for gain in (*gains, 1/values["antiwindup_time_s"])):
            raise ValueError("converted controller gains must remain positive and finite")
        object.__setattr__(self, "_native", MappingProxyType(values))

    @property
    def native_parameters(self):
        return self._native

    @property
    def coordinate_declarations(self):
        # Internal windup q is not the terminal current: CCCV q can legitimately
        # lie below the imposed Icc. Actual-current stopping belongs to the port.
        return (CoupledCoordinate(CONTROLLER_CURRENT, "A", self._native["current_scale_A"]),)

    def rate(self, mode, q, current, voltage, target):
        p = self._native
        if mode == "differential_power":
            return (
                -p["current_scale_A"]
                / p["power_scale_W"] / p["power_response_time_s"]
                * (current * voltage - target)
            )
        if mode == "differential_resistance":
            if current == 0:
                raise ResidualDomainError("differential resistance is singular at zero current")
            return (
                -p["current_scale_A"]
                / p["resistance_scale_Ohm"] / p["resistance_response_time_s"]
                * (voltage / current - target)
            )
        if mode == "cccv":
            return (-(q - current) / p["antiwindup_time_s"] + p["current_scale_A"]
                    / p["voltage_scale_V"] / p["voltage_response_time_s"] * (voltage - target))
        return 0.0

    def to_dict(self):
        return {
            "parameters": self.parameters.to_dict(),
            "parameter_row_id": self.parameter_row_id,
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, ("parameters", "parameter_row_id", "convention"))
        return cls(
            ScientificTable.from_dict(payload["parameters"]),
            payload["parameter_row_id"],
            payload["convention"],
        )


@dataclass(frozen=True)
class TheveninControlProgram:
    control_mode: str
    target: ScalarControlProgram
    current_limit: ScalarControlProgram | None = None
    current_branch_A: tuple[float, float] | None = None

    def __post_init__(self):
        if self.control_mode not in TARGET_UNITS or not isinstance(
            self.target, ScalarControlProgram
        ):
            raise ValueError("Thevenin requires a named mode and scalar target program")
        if self.target.unit != TARGET_UNITS[self.control_mode]:
            raise ValueError("named Thevenin target has the wrong explicit native unit")
        if self.control_mode in {"resistance", "differential_resistance"} and any(
            value <= 0 for _, value in self.target.setpoints
        ):
            raise ValueError("load resistance targets must be strictly positive")
        if self.control_mode == "cccv":
            if (
                not isinstance(self.current_limit, ScalarControlProgram)
                or self.current_limit.unit != "A"
                or any(value >= 0 for _, value in self.current_limit.setpoints)
            ):
                raise ValueError("CCCV requires an explicit negative charging-current program")
        elif self.current_limit is not None:
            raise ValueError("only CCCV accepts an independent current-limit program")
        if self.control_mode in {"power", "differential_resistance"}:
            branch = self.current_branch_A
            if (
                not isinstance(branch, (tuple, list))
                or len(branch) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, (float, int)) or not isfinite(v)
                    for v in branch
                )
                or branch[0] >= branch[1]
            ):
                raise ValueError("this mode requires an explicit finite current branch")
            if self.control_mode == "differential_resistance" and branch[0] <= 0 <= branch[1]:
                raise ValueError("differential-resistance current branch cannot contain zero")
            object.__setattr__(self, "current_branch_A", tuple(float(v) for v in branch))
        elif self.current_branch_A is not None:
            raise ValueError("this control mode does not accept a current-branch declaration")

    @property
    def unit(self):
        return self.target.unit

    @property
    def setpoints(self):
        return self.target.setpoints

    def value_at(self, time_s):
        return self.target.value_at(time_s)

    def to_dict(self):
        payload = {"control_mode": self.control_mode, "target": self.target.to_dict()}
        if self.current_limit is not None:
            payload["current_limit"] = self.current_limit.to_dict()
        if self.current_branch_A is not None:
            payload["current_branch_A"] = list(self.current_branch_A)
        return payload

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, ("control_mode", "target"), optional=("current_limit", "current_branch_A"))
        for key in ("target", "current_limit"):
            if key in payload:
                _keys(payload[key], ("setpoints", "unit", "mode"))
        return cls(
            payload["control_mode"],
            ScalarControlProgram(**payload["target"]),
            ScalarControlProgram(**payload["current_limit"])
            if "current_limit" in payload
            else None,
            payload.get("current_branch_A"),
        )


def thevenin_control_program(program):
    """Parse both historical scalar CC/CV and explicit named declarations."""
    if isinstance(program, TheveninControlProgram):
        return program
    if isinstance(program, Mapping):
        if "control_mode" in program:
            return TheveninControlProgram.from_dict(program)
        _keys(program, ("setpoints", "unit", "mode"))
        program = ScalarControlProgram(**program)
    if isinstance(program, ScalarControlProgram) and program.unit in {"A", "V"}:
        return TheveninControlProgram("current" if program.unit == "A" else "voltage", program)
    raise ValueError("legacy Thevenin scalar programs support only CC or CV")


def isolate_control_current(resistance, inputs, open_voltage, target, branch, mode, scale):
    """Certify every instantaneous root on the declared branch with common isolation.

    At fixed SOC/T, rectilinear R(I) is piecewise affine. The power residual
    is piecewise cubic; the load residual is piecewise quadratic. Both the
    function and one-sided Jacobians are enclosed across every table segment.
    Nonunique, missing and unresolved roots are distinct failures, never a
    Newton guess selecting an unreported branch.
    """
    lo, hi = branch
    grid = next(
        (resistance._points[k] for k, a in enumerate(resistance.axes) if a[0] == "current_A"), ()
    )
    nodes = (lo, *(v for v in grid if lo < v < hi), hi)

    def rvalue(current):
        supplied = {**inputs, "current_A": current}
        return resistance.value({key: supplied[key] for key, _, _ in resistance.axes}, trial=True)

    segments = []
    for a, b in pairwise(nodes):
        ra, rb = rvalue(a), rvalue(b)
        slope = (rb - ra) / (b - a)
        segments.append((a, b, ra - slope * a, slope))

    def coefficients(current):
        return next(
            ((a, b) for low, high, a, b in segments if low <= current <= high),
            segments[0][2:] if current < lo else segments[-1][2:],
        )

    def function(values):
        i = values[0]
        voltage = open_voltage - i * rvalue(i)
        return np.array([i * voltage - target if mode == "power" else voltage - i * target])

    def jacobian(values):
        i = values[0]
        a, b = coefficients(i)
        return np.array(
            [
                [
                    open_voltage - 2 * a * i - 3 * b * i * i
                    if mode == "power"
                    else -a - 2 * b * i - target
                ]
            ]
        )

    def enclosure(lower, upper):
        values, slopes = [], []
        for a, b, intercept, slope in segments:
            left, right = max(a, lower[0]), min(b, upper[0])
            if left > right:
                continue
            i = Interval(float(left), float(right))
            if mode == "power":
                values.append(open_voltage * i - intercept * i**2 - slope * i**3 - target)
                slopes.append(open_voltage - 2 * intercept * i - 3 * slope * i**2)
            else:
                values.append(open_voltage - (intercept + target) * i - slope * i**2)
                slopes.append(-intercept - 2 * slope * i - target)

        def hull(xs):
            return Interval(min(v.lower for v in xs), max(v.upper for v in xs))

        return (hull(values),), ((hull(slopes),),)

    result = isolate_roots(
        function,
        jacobian,
        enclosure,
        variables=(ResidualVariable("current_A", "A", scale, lower=lo, upper=hi),),
        residual_scales=(max(1.0, abs(target), abs(open_voltage) * scale),),
    )
    if result.status != "unique":
        raise ResidualDomainError(f"{mode} current branch: {result.status}")
    current = float(result.roots[0]["value"][0])
    if mode == "resistance" and current == 0:
        raise ResidualDomainError("resistance control is singular at zero current")
    # A fold is not an admissible smooth voltage/power control even if a point
    # solve happens to find it. The common isolator normally leaves it unresolved.
    derivative = float(jacobian((current,))[0, 0])
    if abs(derivative) <= 1e-12 * max(1.0, abs(open_voltage), abs(target) / scale):
        raise ResidualDomainError(f"{mode} current branch: singular fold")
    return current
