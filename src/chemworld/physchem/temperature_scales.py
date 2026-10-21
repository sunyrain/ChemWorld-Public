"""Historical temperature calibration on one reversible cubic mapping per scale.

Kelvin/Celsius units and historical temperature scales are separate concepts.
These functions never alter world state, raw observations or caloric reference
conventions; every source and destination scale must be explicit.
"""

from dataclasses import replace
from functools import lru_cache
from math import isfinite, ulp

import numpy as np
from scipy.interpolate import make_interp_spline

from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.temperature_scale_data import TEMPERATURE_SCALE_TABLES
from chemworld.physchem.viscosity_kernels import _positive

TEMPERATURE_SCALES = ("ITS-90", "ITS-68", "ITS-48", "ITS-76", "ITS-27")
_NAMES = {
    "IPTS-68": "ITS-68",
    "IPTS-48": "ITS-48",
    "EPT-76": "ITS-76",
    "ITS-27": "ITS-27",
    "IPTS-27": "ITS-27",
}


def _scale(name):
    result = _NAMES.get(name, name)
    if result not in TEMPERATURE_SCALES:
        raise ValueError(f"unknown temperature scale {name!r}")
    return result


@lru_cache(maxsize=4)
def _spline(scale):
    nodes, differences = TEMPERATURE_SCALE_TABLES[scale]
    # Interpolating the small difference avoids cancellation. Default not-a-knot
    # cubic gives the same linear-space interpolant as interpolating T+delta.
    return make_interp_spline(nodes, differences, k=3)


def temperature_scale_bounds(scale, *, basis="native"):
    """Finite empirical range; ITS-90 has no tabulated calibration restriction."""
    scale = _scale(scale)
    if basis not in {"native", "ITS-90"}:
        raise ValueError("bounds basis must be native or ITS-90")
    if scale == "ITS-90":
        return (0.0, float("inf"))
    nodes, differences = TEMPERATURE_SCALE_TABLES[scale]
    if basis == "native":
        return (nodes[0], nodes[-1])
    return (nodes[0] + differences[0], nodes[-1] + differences[-1])


def _in_range(value, bounds, scale):
    low, high = bounds
    # Only endpoint arithmetic roundoff is admitted, not physical extrapolation.
    eps_low = 8 * max(ulp(value), ulp(low))
    eps_high = 8 * max(ulp(value), ulp(high)) if isfinite(high) else 0
    if value < low - eps_low or value > high + eps_high:
        raise ValueError(f"{scale} temperature is outside [{low}, {high}] K")
    return min(max(value, low), high)


def _to_90(value, scale):
    if scale == "ITS-90":
        return value, 1.0
    value = _in_range(value, temperature_scale_bounds(scale), scale)
    spline = _spline(scale)
    return value + float(spline(value)), 1 + float(spline(value, nu=1))


def _from_90(value, scale):
    if scale == "ITS-90":
        return value, 1.0
    value = _in_range(value, temperature_scale_bounds(scale, basis="ITS-90"), scale)
    nodes, differences = TEMPERATURE_SCALE_TABLES[scale]
    mapped = np.asarray(nodes) + np.asarray(differences)
    # Exact node inversion also preserves every original endpoint.
    match = np.flatnonzero(mapped == value)
    spline = _spline(scale)
    if len(match):
        native = nodes[int(match[0])]
    else:
        i = int(np.searchsorted(mapped, value)) - 1
        low, high = nodes[i], nodes[i + 1]
        initial = low + (high - low) * (value - mapped[i]) / (mapped[i + 1] - mapped[i])
        result = solve_root(
            lambda x: [(x[0] - value) + float(spline(x[0]))],
            [initial],
            jacobian=lambda x: [[1 + float(spline(x[0], nu=1))]],
            variables=(
                ResidualVariable(
                    "historical_temperature", "K", max(value, 1), lower=low, upper=high
                ),
            ),
            residual_scales=[max(value, 1)],
            policy=RootPolicy(tolerance=4e-15),
        )
        if not result.success:
            raise ValueError(f"temperature-scale inverse failed: {result.failure_reason}")
        native = float(result.value[0])
    return native, 1 / (1 + float(spline(native, nu=1)))


def temperature_scale_conversion(value_K, current, desired):
    """Value and first derivative of the same monotone calibration mapping.

    Domain intersections use the actual ITS-90 images of historical endpoints.
    Identical historical scales still require the declared calibration range.
    No nominal 273.15 offset is used to convert between temperature scales.
    """
    _positive(value_K, "temperature", zero=True)
    old, new = _scale(current), _scale(desired)
    if old == new:
        return _in_range(value_K, temperature_scale_bounds(old), old), 1.0
    intermediate, first = _to_90(value_K, old)
    result, second = _from_90(intermediate, new)
    return result, first * second


def temperature_convert(value_K, current, desired):
    return temperature_scale_conversion(value_K, current, desired)[0]


def its90_68_difference(temperature_K, *, convention="published_overlap"):
    """Approximate T90-T68 in K, with an explicit historical convention.

    This is a polynomial approximation evaluated at T90, NOT the inverse of
    the table converter. Below 13.8 K the historical API returns zero (no
    correction, not a validated IPTS-68 calibration). The middle polynomial
    applies from 73.15 K, including the gap in the old IUPAC description:
    Harvey (2021), section 3.5.1, documents the overlapping polynomial ranges.
    Rusby's recommended 77 K switch is a different convention. Above the gold
    point, -1.398e-7*T**2 is the rounded radiation approximation; no wavelength
    correction or claim of thermodynamic-temperature accuracy is implied.

    ``published_overlap`` uses the middle polynomial in the undocumented gap,
    with the upper branch at each junction. ``chemicals_1_5_2`` reproduces the
    historical zero in 73.15 < T < 83.8 and the lower branch at each junction.
    Its zero in that gap means no correction, NOT physical equality of scales.
    """
    t = _positive(temperature_K, "temperature", zero=True)
    if convention not in {"published_overlap", "chemicals_1_5_2"}:
        raise ValueError("unknown temperature difference convention")
    legacy = convention == "chemicals_1_5_2"
    if t < 13.8:
        return 0.0
    if legacy and 73.15 < t < 83.8:
        return 0.0
    if t < 73.15 or (legacy and t == 73.15):
        x = (t - 40) / 40
        coefficients = (
            -0.005903,
            0.008174,
            -0.061924,
            -0.193388,
            1.490793,
            1.252347,
            -9.835868,
            1.411912,
            25.277595,
            -19.183815,
            -18.437089,
            27.000895,
            -8.716324,
        )
    elif t < 903.75 or (legacy and t == 903.75):
        x = (t - 273.15) / 630
        coefficients = (
            0.0,
            -0.148759,
            -0.267408,
            1.08076,
            1.269056,
            -4.089591,
            -1.871251,
            7.438081,
            -3.536296,
        )
    elif t < 1337.33 or (legacy and t == 1337.33):
        x = t - 273.15
        coefficients = (
            78.687209,
            -0.47135991,
            0.0010954715,
            -1.2357884e-6,
            6.7736583e-10,
            -1.4458081e-13,
        )
    else:
        return -1.398e-7 * t * t
    return float(np.polynomial.polynomial.polyval(x, coefficients))


def temperature_quantity(value, current, desired, *, unit="K"):
    """Explicit scale conversion of an immutable native Quantity."""
    from chemworld.foundation.units import Quantity

    return Quantity(temperature_convert(value.to("K").value, current, desired), "K").to(unit)


def temperature_input(value, current, desired):
    """Return an InputValue in K, preserving quality and source in its note.

    First-order uncertainty propagation covers the supplied measurement only;
    calibration-table uncertainty is unknown and is not silently set to zero.
    Missing values remain missing; invalid numeric values cannot be calibrated.
    """
    import json

    old, new = _scale(current), _scale(desired)
    normalized = value.to("K")
    if value.quality not in {"valid", "missing", "invalid"}:
        raise ValueError("temperature input requires valid, missing or invalid quality")
    if value.value is not None and value.quality != "valid":
        raise ValueError("cannot calibrate an invalid numeric temperature")
    result, derivative = (
        (None, None)
        if value.value is None
        else temperature_scale_conversion(normalized.value, old, new)
    )
    note = (
        f"Temperature scale {old} -> {new}; reversible not-a-knot cubic; "
        f"native source bounds {temperature_scale_bounds(old)} K; "
        f"native target bounds {temperature_scale_bounds(new)} K; "
        "first-order measurement uncertainty only; calibration uncertainty unknown. "
        f"Original input: {json.dumps(value.to_dict(), ensure_ascii=False)}"
    )
    return replace(
        normalized,
        value=result,
        uncertainty=None
        if normalized.uncertainty is None or derivative is None
        else abs(derivative) * normalized.uncertainty,
        uncertainty_note=note,
    )


def temperature_observation(channel, current, desired, *, channel_id="converted_temperature"):
    """Calibrate a scalar raw channel into the existing Observation container.

    The immutable raw channel, sample identity and both timestamps are retained.
    A curve needs explicit per-point uncertainty support and is rejected here.
    The conversion's provenance uses existing acquisition metadata, not state.
    """
    from chemworld.foundation.observations import Observation
    from chemworld.foundation.units import convert_value

    old, new = _scale(current), _scale(desired)
    if channel.errors() or channel.role != "raw":
        raise ValueError("temperature conversion requires a valid raw channel contract")
    if isinstance(channel.value, tuple):
        raise ValueError("temperature observation conversion requires a scalar channel")
    if not channel_id or channel_id == channel.channel_id:
        raise ValueError("derived temperature channel requires a distinct identity")
    if channel.value is not None and channel.quality != "valid":
        raise ValueError("only valid numeric temperatures can be calibrated")
    # Check the dimension even for a missing reading.
    convert_value(0.0, channel.unit, "K")

    def convert(t):
        return temperature_convert(convert_value(t, channel.unit, "K"), old, new)

    result, derivative = (
        (None, None)
        if channel.value is None
        else temperature_scale_conversion(convert_value(channel.value, channel.unit, "K"), old, new)
    )
    sigma = (
        None
        if channel.uncertainty is None or derivative is None
        else (
            abs(derivative)
            * channel.uncertainty
            * abs(convert_value(1.0, channel.unit, "K") - convert_value(0.0, channel.unit, "K"))
        )
    )
    processed = replace(
        channel,
        channel_id=channel_id,
        unit="K",
        value=result,
        uncertainty=sigma,
        valid_range=None
        if channel.valid_range is None
        else tuple(convert(v) for v in channel.valid_range),
        detection_limit=None
        if channel.detection_limit is None
        else convert(channel.detection_limit),
        role="processed",
        derived_from=(channel.channel_id,),
        acquisition_relation="temperature_scale_conversion",
    )
    metadata = {
        "source_scale": old,
        "target_scale": new,
        "method": "reversible_not_a_knot_cubic",
        "source_channel": channel.channel_id,
        "derived_channel": channel_id,
        "source_value": {"$channel_value": channel.channel_id},
        "source_bounds_K": list(temperature_scale_bounds(old)) if old != "ITS-90" else [0.0, None],
        "target_bounds_K": list(temperature_scale_bounds(new)) if new != "ITS-90" else [0.0, None],
        "derivative": derivative,
        "uncertainty": "first-order measurement only; calibration uncertainty unknown",
    }
    observation = Observation(
        channels=(channel, processed), raw_signal={"temperature_conversion": metadata}
    )
    if observation.errors():
        raise ValueError(f"invalid calibrated observation: {observation.errors()}")
    return observation
