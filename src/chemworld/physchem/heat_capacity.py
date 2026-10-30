"""Same-phase piecewise heat capacity using the existing correlation contract."""

from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType

from scipy.optimize import brentq

from chemworld.foundation.units import convert_value
from chemworld.physchem.correlation_families import (
    CORRELATION_FAMILIES,
    validate_family_interval,
    validate_heat_capacity_interval,
)
from chemworld.physchem.specs import PropertyCorrelation

_CAPACITIES = {
    "heat_capacity", "ideal_gas_heat_capacity", "liquid_heat_capacity",
    "solid_heat_capacity", "isochoric_heat_capacity",
}


def capacity_phases(curve):
    """Intersect declared phase tags with the existing formula registry."""
    if isinstance(curve, PiecewiseHeatCapacity):
        return set.intersection(*(capacity_phases(segment) for segment in curve.segments))
    phases = {"gas", "liquid", "solid"}
    phase_properties = {"ideal_gas_heat_capacity": "gas", "liquid_heat_capacity": "liquid",
                        "solid_heat_capacity": "solid"}
    family = CORRELATION_FAMILIES.get(curve.equation_id)
    if family is not None and family.reference_method:
        phases &= {phase for key, phase in phase_properties.items()
                   if key in family.property_dimensions}
    for phase in (phase_properties.get(curve.property_id), curve.metadata.get("phase")):
        if phase is not None:
            phases &= {phase}
    return phases


def validate_capacity_curve(curve, lower, upper, *, minimum=0, allow_zero=False):
    """Validate a T-only curve and bound its molar capacity over the whole interval."""
    if isinstance(curve, PiecewiseHeatCapacity):
        for segment, low, high in curve.intervals(lower, upper):
            validate_capacity_curve(segment, low, high, minimum=minimum, allow_zero=allow_zero)
        return
    if (curve.property_id not in _CAPACITIES or curve.input_units != {"temperature": "K"}
            or not isfinite(lower) or not isfinite(upper) or not 0 < lower <= upper):
        raise ValueError("caloric curve needs a declared T-only Kelvin heat capacity")
    if not isfinite(minimum) or minimum < 0:
        raise ValueError("capacity minimum must be finite and nonnegative")
    equation, coefficients = curve.equation_id, curve.coefficients
    if equation == "cp_polynomial":
        equation = "dippr100"
        coefficients = {key.upper(): value for key, value in coefficients.items()}
    if equation not in CORRELATION_FAMILIES:
        raise ValueError("unsupported caloric curve")
    validate_family_interval(equation, coefficients, lower, upper)
    validate_heat_capacity_interval(
        equation, coefficients, lower, upper, allow_zero=allow_zero,
        minimum=convert_value(minimum, "J/(mol*K)", curve.output_unit))


@dataclass(frozen=True)
class PiecewiseHeatCapacity:
    """Contiguous same-phase ranges; H/S stay continuous at Cp breakpoints.

    The right segment owns an internal boundary. Its derivative is explicitly
    one-sided there; phase transitions and latent heat need a separate model.
    """

    correlation_id: str
    segments: tuple[PropertyCorrelation, ...]
    source_note: str

    def __post_init__(self) -> None:
        if not self.correlation_id or not self.source_note or not self.segments:
            raise ValueError("piecewise capacity requires identity, source note and segments")
        segments = tuple(PropertyCorrelation.from_dict(c.to_dict()) for c in self.segments)
        property_id = segments[0].property_id
        phase = segments[0].metadata.get("phase")
        possible_phases = {"gas", "liquid", "solid"}
        previous = None
        for segment in segments:
            possible_phases &= capacity_phases(segment)
            if (segment.property_id != property_id or segment.metadata.get("phase") != phase
                    or set(segment.validity_ranges) != {"temperature"} or not possible_phases):
                raise ValueError("piecewise segments need the same phase/basis and finite T ranges")
            low, high = segment.validity_ranges["temperature"]
            if not isfinite(low) or not isfinite(high) or not 0 < low < high:
                raise ValueError("piecewise segments need increasing finite positive T ranges")
            if previous is not None and low != previous:
                raise ValueError(
                    "piecewise capacity cannot contain gaps, overlaps or reversed ranges")
            validate_capacity_curve(segment, low, high,
                                    allow_zero=property_id == "isochoric_heat_capacity")
            previous = high
        object.__setattr__(self, "segments", segments)

    @property
    def equation_id(self) -> str:
        return "piecewise_heat_capacity"

    @property
    def property_id(self):
        return self.segments[0].property_id

    @property
    def input_units(self):
        return {"temperature": "K"}

    @property
    def output_unit(self) -> str:
        return "J/(mol*K)"

    @property
    def metadata(self):
        return MappingProxyType(dict(self.segments[0].metadata))

    @property
    def validity_ranges(self):
        return {"temperature": (self.segments[0].validity_ranges["temperature"][0],
                                self.segments[-1].validity_ranges["temperature"][1])}

    def segment_at(self, temperature):
        low, high = self.validity_ranges["temperature"]
        if (isinstance(temperature, bool) or not isfinite(temperature)
                or not low <= temperature <= high):
            raise ValueError("temperature outside piecewise domain; no implicit extension")
        for segment in self.segments[:-1]:
            if temperature < segment.validity_ranges["temperature"][1]:
                return segment
        return self.segments[-1]

    def intervals(self, lower, upper):
        self.segment_at(lower)
        self.segment_at(upper)
        if lower > upper:
            raise ValueError("piecewise interval bounds must be ordered")
        if lower == upper:
            return ((self.segment_at(lower), lower, upper),)
        return tuple((segment, max(lower, low), min(upper, high))
                     for segment in self.segments
                     for low, high in (segment.validity_ranges["temperature"],)
                     if min(upper, high) > max(lower, low))

    def integral(self, initial, final, *, over_temperature=False, epsabs=1e-10, epsrel=1e-10):
        from chemworld.physchem.enthalpy import caloric_polynomial_change
        from chemworld.physchem.property_equations import CorrelationIntegral, integrate_correlation

        if not all(isfinite(value) and value > 0 for value in (epsabs, epsrel)):
            raise ValueError("quadrature tolerances must be finite and positive")
        intervals = self.intervals(*sorted((initial, final)))
        values, errors, evaluations = [], [], 0
        for segment, lower, upper in intervals:
            if segment.equation_id == "cp_polynomial":
                if over_temperature and segment.property_id == "isochoric_heat_capacity":
                    # This is the mathematical Cv/T integral, not yet an entropy model.
                    from dataclasses import replace
                    segment = replace(segment, equation_id="dippr100", coefficients={
                        key.upper(): value for key, value in segment.coefficients.items()})
                else:
                    result = caloric_polynomial_change(segment, initial_temperature_K=lower,
                        final_temperature_K=upper, entropy=over_temperature)
                    values.append(result.value)
                    errors.append(0.0)
                    continue
            if segment.equation_id != "cp_polynomial":
                result = integrate_correlation(segment, initial_temperature_K=lower,
                    final_temperature_K=upper, over_temperature=over_temperature,
                    epsabs=epsabs / len(intervals), epsrel=epsrel)
                factor = convert_value(1, segment.output_unit, self.output_unit)
                values.append(result.value * factor)
                errors.append(result.absolute_error_estimate * factor)
                evaluations += result.evaluations
        value = fsum(values) * (1 if final >= initial else -1)
        if not isfinite(value):
            raise ValueError("nonfinite piecewise caloric integral")
        return CorrelationIntegral(self.correlation_id, self.equation_id, value,
            self.output_unit if over_temperature else "J/mol", initial, final, over_temperature,
            fsum(errors), evaluations, (), method="piecewise_declared_correlation_integrals")

    def to_dict(self):
        return {"model": "piecewise-heat-capacity-v1", "correlation_id": self.correlation_id,
                "source_note": self.source_note, "segments": [c.to_dict() for c in self.segments]}

    @classmethod
    def from_dict(cls, payload):
        if (set(payload) != {"model", "correlation_id", "source_note", "segments"}
                or payload["model"] != "piecewise-heat-capacity-v1"):
            raise ValueError("unsupported piecewise heat-capacity record")
        return cls(payload["correlation_id"],
                   tuple(PropertyCorrelation(**c) for c in payload["segments"]),
                   payload["source_note"])


def heat_capacity_from_dict(payload):
    return (PiecewiseHeatCapacity.from_dict(payload) if "model" in payload
            else PropertyCorrelation(**payload))


def temperature_from_caloric_change(curve, change, *, reference_temperature_K,
                                    temperature_bounds_K=None, entropy=False):
    """Invert same-phase delta-H/delta-U or delta-S on an explicitly bounded curve.

    This reuses scipy's bracketed scalar root implementation, as do the existing
    saturation and caloric closures. An entropy integral never masquerades as U/H.
    """
    from chemworld.physchem.enthalpy import caloric_correlation_change

    bounds = (curve.validity_ranges.get("temperature") if temperature_bounds_K is None
              else temperature_bounds_K)
    if bounds is None or len(bounds) != 2 or isinstance(change, bool) or not isfinite(change):
        raise ValueError("caloric inversion requires finite target and explicit temperature bounds")
    lower, upper = bounds
    if lower >= upper:
        raise ValueError("caloric inversion requires increasing temperature bounds")
    if entropy and curve.property_id == "isochoric_heat_capacity":
        raise ValueError("entropy inversion requires Cp or an explicit gas/EOS conversion")
    validate_capacity_curve(curve, lower, upper)

    def residual(T):
        return caloric_correlation_change(curve, initial_temperature_K=reference_temperature_K,
            final_temperature_K=T, entropy=entropy, validity_policy="raise").value - change

    if residual(lower) > 0 or residual(upper) < 0:
        raise ValueError("target caloric change is outside the declared temperature domain")
    return float(brentq(residual, lower, upper, xtol=1e-10))
