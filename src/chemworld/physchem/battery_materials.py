"""Declarative electrode OCV curves with the shared parameter provenance contract.

Terms are mathematical data, not executable callbacks. One expression supplies
voltages and analytic derivatives in all native battery models.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, expm1, inf, isfinite, log, log1p, nextafter, tanh
from numbers import Real

from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.physchem.analytic_functions import smooth_function
from chemworld.physchem.caloric_math import (
    Interval,
    derivative_depth,
    derivative_values,
    exponential,
    scalar_value,
)

OCV_VERSION = "polynomial-exp-tanh-ocv-v1"
OCV_ENDPOINT_VERSION = "polynomial-exp-tanh-clipped-endpoint-ocv-v2"
OCV_ENDPOINT_POLICY_VERSION = "clipped-rational-ocv-endpoint-v1"


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("OCV coefficients must be finite real numbers")
    return float(value)


@dataclass(frozen=True)
class ElectrodeOCVEndpointPolicy:
    """Sourced model convention: x=clip(theta, delta, 1-delta), U=Uraw(x)+b/x+b/(x-1).

    PyBaMM 25.4.2's lithium_ion_parameters.py ``U`` applies the same clipped
    coordinate to the material curve and the rational barrier. This is a model
    convention with its own provenance, not a measurement of an electrode.
    """

    source: InputSource
    barrier_V: float = 1e-6
    stoichiometry_clip: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.source, InputSource) or self.source.data_kind == "measured":
            raise ValueError("OCV endpoint policy requires an explicit model InputSource")
        barrier, clip = _number(self.barrier_V), _number(self.stoichiometry_clip)
        if barrier <= 0 or not 0 < clip < 1 - clip < 1:
            raise ValueError("OCV endpoint policy requires a positive barrier and clip in (0, 0.5)")
        object.__setattr__(self, "barrier_V", barrier)
        object.__setattr__(self, "stoichiometry_clip", clip)

    def to_dict(self):
        return {
            "model_version": OCV_ENDPOINT_POLICY_VERSION,
            "source": self.source.to_dict(),
            "barrier_V": self.barrier_V,
            "stoichiometry_clip": self.stoichiometry_clip,
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, Mapping)
            or set(value) != {"model_version", "source", "barrier_V", "stoichiometry_clip"}
            or value["model_version"] != OCV_ENDPOINT_POLICY_VERSION
        ):
            raise ValueError("unsupported OCV endpoint policy version or fields")
        return cls(InputSource.from_dict(value["source"]), value["barrier_V"],
                   value["stoichiometry_clip"])


@dataclass(frozen=True)
class ElectrodeOCV:
    """Polynomial/exp/tanh material curve with an optional sourced endpoint convention."""

    material_id: str
    source: InputSource
    polynomial_V: tuple[float, ...] = ()
    exponential_terms: tuple[tuple[float, float, float], ...] = ()
    tanh_terms: tuple[tuple[float, float, float], ...] = ()
    stoichiometry_domain: tuple[float, float] = (0.01, 0.99)
    supported_models: tuple[str, ...] = ("SPM", "SPMe", "DFN")
    source_row_id: str = ""
    parameter_inputs: tuple[tuple[str, InputValue], ...] = ()
    endpoint_policy: ElectrodeOCVEndpointPolicy | None = None

    def __post_init__(self):
        if not isinstance(self.material_id, str) or not self.material_id.strip():
            raise ValueError("OCV requires a material identity")
        if not isinstance(self.source, InputSource):
            raise ValueError("OCV requires an explicit InputSource")
        polynomial = tuple(_number(v) for v in self.polynomial_V)
        object.__setattr__(self, "polynomial_V", polynomial)
        for name in ("exponential_terms", "tanh_terms"):
            terms = tuple(tuple(_number(v) for v in term) for term in getattr(self, name))
            if any(len(t) != 3 for t in terms):
                raise ValueError("OCV terms require amplitude_V, scale and shift")
            object.__setattr__(self, name, terms)
        if (
            len(polynomial) > 21
            or not 1 <= sum(map(len, (polynomial, self.exponential_terms, self.tanh_terms))) <= 128
        ):
            raise ValueError("OCV requires a nonempty bounded expression")
        domain = tuple(_number(v) for v in self.stoichiometry_domain)
        if len(domain) != 2 or not 0 <= domain[0] < domain[1] <= 1:
            raise ValueError("OCV stoichiometry domain must be ordered inside [0, 1]")
        object.__setattr__(self, "stoichiometry_domain", domain)
        models = tuple(self.supported_models)
        if (
            not models
            or len(set(models)) != len(models)
            or not set(models) <= {"SPM", "SPMe", "DFN", "MPM", "NewmanTobias", "SplitOCVR"}
        ):
            raise ValueError("OCV requires explicit supported battery models")
        object.__setattr__(self, "supported_models", models)
        inputs = tuple(self.parameter_inputs)
        if (
            any(not isinstance(k, str) or not k or not isinstance(v, InputValue) for k, v in inputs)
            or len(dict(inputs)) != len(inputs)
            or (inputs and (not isinstance(self.source_row_id, str) or not self.source_row_id))
        ):
            raise ValueError("OCV parameter inputs require unique names and a source row")
        object.__setattr__(self, "parameter_inputs", inputs)
        if self.endpoint_policy is not None and not isinstance(
            self.endpoint_policy, ElectrodeOCVEndpointPolicy
        ):
            raise ValueError("OCV endpoint policy requires an explicit declaration")

    def _check(self, theta, trial):
        if not isfinite(scalar_value(theta)):
            raise ValueError("OCV stoichiometry must be finite")
        if (
            not trial
            and not self.stoichiometry_domain[0]
            <= scalar_value(theta)
            <= (self.stoichiometry_domain[1])
        ):
            raise ValueError("OCV stoichiometry is outside its declared domain")

    def _expression(self, theta):
        if self.endpoint_policy is None:
            return self._raw_expression(theta)
        policy = self.endpoint_policy
        low, high = policy.stoichiometry_clip, 1 - policy.stoichiometry_clip
        primal = scalar_value(theta)
        if primal in (low, high) and derivative_depth(theta):
            raise ValueError("OCV derivative is undefined at an endpoint clip breakpoint")
        # Detach a clamped expression from the original derivative seed. This
        # changes only the constitutive coordinate, never a particle inventory.
        x = low if primal < low else high if primal > high else theta
        return self._raw_expression(x) + policy.barrier_V * (1 / x + 1 / (x - 1))

    def _raw_expression(self, theta):
        value = 0.0
        for coefficient in reversed(self.polynomial_V):
            value = value * theta + coefficient
        for amplitude, scale, shift in self.exponential_terms:
            value += amplitude * exponential(scale * (theta - shift))
        for amplitude, scale, shift in self.tanh_terms:
            value += amplitude * smooth_function("tanh", scale * (theta - shift))
        return value

    def value(self, theta, *, trial=False):
        self._check(theta, trial)
        value = self._expression(theta)
        if not isfinite(scalar_value(value)):
            raise ValueError("nonfinite OCV")
        return value

    def derivatives(self, theta, *, order=1, trial=False):
        """Classical analytic derivatives; clip breakpoints reject positive orders."""
        self._check(theta, trial)
        return derivative_values(self._expression, theta, order=order)

    def integral(self, lower, upper, *, trial=False):
        """Exact integral in V; theta is dimensionless. No sampled fit or quadrature."""
        self._check(lower, trial)
        self._check(upper, trial)
        if self.endpoint_policy is None:
            return self._raw_integral(lower, upper)
        if lower > upper:
            return -self.integral(upper, lower, trial=trial)
        policy = self.endpoint_policy
        low, high = policy.stoichiometry_clip, 1 - policy.stoichiometry_clip
        a, b = max(lower, low), min(upper, high)
        result = 0.0
        if lower < low:
            result += (min(upper, low) - lower) * self._expression(low)
        if a < b:
            width = b - a
            result += self._raw_integral(a, b) + policy.barrier_V * (
                log1p(width / a) - log1p(width / (1 - b))
            )
        if upper > high:
            result += (upper - max(lower, high)) * self._expression(high)
        if not isfinite(result):
            raise ValueError("nonfinite OCV integral")
        return result

    def _raw_integral(self, lower, upper):
        def logcosh(x):
            return abs(x) + log1p(exp(-2 * abs(x))) - log(2)

        result = sum(
            a * (upper ** (i + 1) - lower ** (i + 1)) / (i + 1)
            for i, a in enumerate(self.polynomial_V)
        )
        for amplitude, scale, shift in self.exponential_terms:
            result += (
                amplitude * (upper - lower)
                if scale == 0
                else amplitude
                * exp(scale * (lower - shift))
                * expm1(scale * (upper - lower))
                / scale
            )
        for amplitude, scale, shift in self.tanh_terms:
            if scale:
                result += (
                    amplitude
                    / scale
                    * (logcosh(scale * (upper - shift)) - logcosh(scale * (lower - shift)))
                )
        if not isfinite(result):
            raise ValueError("nonfinite OCV integral")
        return result

    def enclosure(self, lower, upper, *, derivative=False, trial=False):
        """Outward interval on a closed declared sub-domain.

        With a policy, derivative bounds enclose both one-sided slopes at a
        clip breakpoint, including zero on the constant extension. They can
        certify monotonicity across a kink without claiming its derivative.
        """
        self._check(lower, trial)
        self._check(upper, trial)
        theta = Interval(lower, upper)
        if self.endpoint_policy is None:
            return self._raw_enclosure(theta, derivative=derivative)
        policy = self.endpoint_policy
        low, high = policy.stoichiometry_clip, 1 - policy.stoichiometry_clip
        if derivative and (upper < low or lower > high):
            return Interval(0.0, 0.0)
        clipped = Interval(max(low, min(high, lower)), max(low, min(high, upper)))
        result = self._raw_enclosure(clipped, derivative=derivative)
        if derivative:
            result -= policy.barrier_V * (1 / clipped**2 + 1 / (clipped - 1)**2)
            if lower <= low or upper >= high:
                result = Interval(min(result.lower, 0.0), max(result.upper, 0.0))
        else:
            result += policy.barrier_V * (1 / clipped + 1 / (clipped - 1))
        return result

    def _raw_enclosure(self, theta, *, derivative):
        result = Interval(0.0, 0.0)
        polynomial = (
            tuple(i * c for i, c in enumerate(self.polynomial_V) if i)
            if derivative
            else self.polynomial_V
        )
        for coefficient in reversed(polynomial):
            result = result * theta + coefficient
        for amplitude, scale, shift in self.exponential_terms:
            result += (
                amplitude * (scale if derivative else 1) * exponential(scale * (theta - shift))
            )
        for amplitude, scale, shift in self.tanh_terms:
            argument = scale * (theta - shift)
            if derivative:
                # sech² evaluated via exp(-2|z|) retains its positive tail even
                # where floating-point tanh(z) has rounded to exactly +/-1.
                closest = (
                    0.0
                    if argument.lower <= 0 <= argument.upper
                    else min(abs(argument.lower), abs(argument.upper))
                )
                furthest = max(abs(argument.lower), abs(argument.upper))

                def sech_squared(x):
                    q = exp(-2 * x)
                    return 4 * q / (1 + q) ** 2

                term = Interval(
                    max(0.0, nextafter(sech_squared(furthest), -inf)),
                    min(1.0, nextafter(sech_squared(closest), inf)),
                )
                result += amplitude * scale * term
            else:
                result += amplitude * Interval(
                    nextafter(tanh(argument.lower), -inf), nextafter(tanh(argument.upper), inf)
                )
        return result

    def to_dict(self):
        result = {
            "model_version": OCV_VERSION,
            "material_id": self.material_id,
            "source": self.source.to_dict(),
            "polynomial_V": list(self.polynomial_V),
            "exponential_terms": [list(t) for t in self.exponential_terms],
            "tanh_terms": [list(t) for t in self.tanh_terms],
            "stoichiometry_domain": list(self.stoichiometry_domain),
            "supported_models": list(self.supported_models),
            "source_row_id": self.source_row_id,
            "parameter_inputs": {k: v.to_dict() for k, v in self.parameter_inputs},
        }
        if self.endpoint_policy is not None:
            result["model_version"] = OCV_ENDPOINT_VERSION
            result["endpoint_policy"] = self.endpoint_policy.to_dict()
        return result

    @classmethod
    def from_dict(cls, value):
        keys = {
            "model_version",
            "material_id",
            "source",
            "polynomial_V",
            "exponential_terms",
            "tanh_terms",
            "stoichiometry_domain",
            "supported_models",
            "source_row_id",
            "parameter_inputs",
        }
        if not isinstance(value, Mapping):
            raise ValueError("unsupported OCV model version or fields")
        version = value.get("model_version")
        if version == OCV_ENDPOINT_VERSION:
            keys.add("endpoint_policy")
        if set(value) != keys or version not in (OCV_VERSION, OCV_ENDPOINT_VERSION):
            raise ValueError("unsupported OCV model version or fields")
        endpoint = (
            {} if version == OCV_VERSION else
            {"endpoint_policy": ElectrodeOCVEndpointPolicy.from_dict(value["endpoint_policy"])}
        )
        return cls(
            **{
                k: v
                for k, v in value.items()
                if k not in {"model_version", "source", "parameter_inputs", "endpoint_policy"}
            },
            source=InputSource.from_dict(value["source"]),
            parameter_inputs=tuple(
                (k, InputValue.from_dict(v)) for k, v in value["parameter_inputs"].items()
            ),
            **endpoint,
        )

    @classmethod
    def from_parameter_row(
        cls,
        table,
        row_id,
        *,
        material_id,
        polynomial=(),
        exponential_terms=(),
        tanh_terms=(),
        **kwargs,
    ):
        """Bind explicitly named, unit-checked columns from an existing parameter table."""
        if table.kind != "parameters":
            raise ValueError("OCV requires a parameter table")
        rows = [row for row in table.rows if row.row_id == row_id]
        if len(rows) != 1:
            raise ValueError("OCV requires one unambiguous parameter row")
        row = rows[0]
        used = {}

        def coefficient(name, unit):
            if name not in row.values:
                raise ValueError("missing OCV parameter column: " + name)
            cell = row.values[name]
            if cell.quality != "valid":
                raise ValueError("OCV coefficients must be valid parameters")
            used[name] = cell
            return cell.to(unit).value

        def terms(names):
            return tuple(
                tuple(
                    coefficient(name, unit)
                    for name, unit in zip(
                        term, ("V", "dimensionless", "dimensionless"), strict=True
                    )
                )
                for term in names
            )

        return cls(
            material_id,
            next(s for s in table.sources if s.provenance.source_id == row.source_id),
            polynomial_V=tuple(coefficient(name, "V") for name in polynomial),
            exponential_terms=terms(exponential_terms),
            tanh_terms=terms(tanh_terms),
            source_row_id=row.row_id,
            parameter_inputs=tuple(used.items()),
            **kwargs,
        )
