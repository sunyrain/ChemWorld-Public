"""Constitutive endpoint convention, exact legacy arithmetic and strict codecs."""

import json
from dataclasses import replace
from math import exp, factorial, log, tanh

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_materials import (
    OCV_ENDPOINT_POLICY_VERSION,
    OCV_ENDPOINT_VERSION,
    OCV_VERSION,
    ElectrodeOCV,
    ElectrodeOCVEndpointPolicy,
)
from chemworld.physchem.caloric_math import FirstDerivative
from chemworld.physchem.data_conflicts import DataSourceProvenance


def source(name, kind="literature"):
    return InputSource(
        DataSourceProvenance(name, 0, name, "25.4.2", "BSD-3-Clause"), kind
    )


def policy(**kwargs):
    return ElectrodeOCVEndpointPolicy(source("PyBaMM model endpoint convention"), **kwargs)


def curve(*, endpoint=None, **kwargs):
    return ElectrodeOCV(
        "electrode", source("independent material coefficients", "measured"),
        polynomial_V=(0.2, -0.1, 0.03), stoichiometry_domain=(0.0, 1.0),
        endpoint_policy=endpoint, **kwargs,
    )


@pytest.mark.parametrize("x", [-0.2, 0.0, 5e-11, 0.2, 0.5, 0.9, 1.0, 1.2])
def test_pybamm_2542_uses_same_clipped_coordinate_for_raw_curve_and_barrier(x):
    p = policy()
    c = curve(endpoint=p)
    clipped = max(p.stoichiometry_clip, min(1 - p.stoichiometry_clip, x))
    expected = curve().value(clipped) + 1e-6 * (1 / clipped + 1 / (clipped - 1))
    assert c.value(x, trial=True) == expected
    assert c.endpoint_policy.source != c.source
    assert c.endpoint_policy.source.data_kind == "literature"
    assert c.stoichiometry_domain == (0.0, 1.0)
    with pytest.raises(ValueError, match="declared domain"):
        c.value(-0.1)


@pytest.mark.parametrize("x", [2e-10, 0.2, 0.5, 0.9, 1 - 2e-10])
def test_all_four_analytic_derivatives_include_the_barrier(x):
    p = policy()
    raw = curve(exponential_terms=((0.01, -3.0, 0.0),),
                tanh_terms=((-0.02, 5.0, 0.4),))
    c = replace(raw, endpoint_policy=p)
    base = raw.derivatives(x, order=4)
    actual = c.derivatives(x, order=4)
    for n in range(5):
        barrier = p.barrier_V * (-1)**n * factorial(n) * (
            x**(-n - 1) + (x - 1)**(-n - 1)
        )
        assert actual[n] == pytest.approx(base[n] + barrier, rel=2e-14, abs=1e-14)
    jet = c.value(FirstDerivative(x, 2.0))
    assert jet.value == actual[0]
    assert jet.slope == pytest.approx(2 * actual[1], rel=2e-14)


@pytest.mark.parametrize("x", [-1.0, 0.0, 0.05, 0.95, 1.0, 2.0])
def test_clamped_extension_has_zero_derivatives_without_changing_input(x):
    c = curve(endpoint=policy(stoichiometry_clip=0.1))
    assert c.derivatives(x, order=4, trial=True) == (c.value(x, trial=True), 0, 0, 0, 0)
    seed = FirstDerivative(x, 2.0)
    assert c.value(seed, trial=True) == c.value(x, trial=True)
    assert seed == FirstDerivative(x, 2.0)


@pytest.mark.parametrize("x", [0.1, 0.9])
def test_clip_breakpoints_have_values_and_integrals_but_no_classical_derivative(x):
    c = curve(endpoint=policy(stoichiometry_clip=0.1))
    assert c.derivatives(x, order=0) == (c.value(x),)
    assert c.integral(x, x) == 0.0
    for order in range(1, 5):
        with pytest.raises(ValueError, match="clip breakpoint"):
            c.derivatives(x, order=order)
    with pytest.raises(ValueError, match="clip breakpoint"):
        c.value(FirstDerivative(x, 1.0))


def primitive(x, barrier):
    return 0.2*x - 0.05*x*x + 0.01*x**3 + barrier*(log(x) + log(1 - x))


@pytest.mark.parametrize("a,b", [
    (0.2, 0.8), (0.0, 1.0), (-0.4, 1.3), (-0.4, 0.05), (0.95, 1.3),
    (0.1, 0.9), (0.0, 0.1), (0.9, 1.0), (0.2, 0.2),
])
def test_closed_form_integral_includes_barrier_and_both_constant_extensions(a, b):
    p = policy(barrier_V=0.03, stoichiometry_clip=0.1)
    c = curve(endpoint=p)
    low, high = 0.1, 0.9
    expected = 0.0
    if a < low:
        expected += (min(b, low) - a) * c.value(low)
    if max(a, low) < min(b, high):
        expected += primitive(min(b, high), p.barrier_V) - primitive(max(a, low), p.barrier_V)
    if b > high:
        expected += (b - max(a, high)) * c.value(high)
    actual = c.integral(a, b, trial=True)
    assert actual == pytest.approx(expected, rel=2e-14, abs=2e-16)
    assert actual == -c.integral(b, a, trial=True)
    assert actual == pytest.approx(c.integral(a, (a+b)/2, trial=True)
                                   + c.integral((a+b)/2, b, trial=True), abs=2e-16)


def test_integral_of_full_expression_and_default_near_endpoint_barrier():
    raw = curve(exponential_terms=((0.01, -3.0, 0.0),),
                tanh_terms=((-0.02, 5.0, 0.4),))
    c = replace(raw, endpoint_policy=policy())
    a, b = 1e-10, 0.4
    barrier = 1e-6 * log(b * (1 - b) / (a * (1 - a)))
    assert c.integral(a, b) == pytest.approx(raw.integral(a, b) + barrier, abs=2e-17)
    for x in (0.1, 0.4, 0.8):
        h = 1e-6
        assert (c.integral(a, x+h) - c.integral(a, x-h)) / (2*h) == pytest.approx(
            c.value(x), rel=1e-9
        )


def test_near_complete_window_barrier_integral_keeps_endpoint_log_precision():
    p = policy()
    c = replace(curve(endpoint=p), polynomial_V=(0.0,))
    low, high = p.stoichiometry_clip, 1 - p.stoichiometry_clip
    expected = p.barrier_V * (log(high / low) + log((1 - high) / (1 - low)))
    assert c.integral(low, high) == pytest.approx(expected, abs=5e-21)


@pytest.mark.parametrize("a,b", [(-1, 2), (-0.2, 0.05), (0.1, 0.1), (0.2, 0.8),
                                  (0.9, 0.9), (0.95, 1.3), (0.05, 0.2), (0.8, 0.95)])
def test_outward_bounds_enclose_analytic_extrema_and_both_kink_slopes(a, b):
    p = policy(barrier_V=0.03, stoichiometry_clip=0.1)
    c = replace(curve(endpoint=p), polynomial_V=(0.2,))
    low, high = max(0.1, min(0.9, a)), max(0.1, min(0.9, b))
    bounds = c.enclosure(a, b, trial=True)
    # The rational barrier is strictly decreasing between clips, so these are
    # its exact extrema over the whole interval, including constant extensions.
    assert bounds.lower <= c.value(high) <= c.value(low) <= bounds.upper
    slopes = c.enclosure(a, b, derivative=True, trial=True)
    if b < 0.1 or a > 0.9:
        assert (slopes.lower, slopes.upper) == (0.0, 0.0)
    else:
        def derivative(x):
            return -p.barrier_V * (1/x**2 + 1/(x-1)**2)

        exact_low = min(derivative(low), derivative(high))
        exact_high = derivative(max(low, min(high, 0.5)))
        assert slopes.lower <= exact_low <= exact_high <= slopes.upper
        if a <= 0.1 or b >= 0.9:
            assert slopes.lower <= 0.0 <= slopes.upper
    assert slopes.upper <= 0.0


def test_bounds_keep_true_positive_slope_and_validate_interval_domain():
    c = replace(curve(endpoint=policy(stoichiometry_clip=0.1)), polynomial_V=(0.2, 10.0))
    bounds = c.enclosure(0.4, 0.6, derivative=True)
    assert bounds.upper >= 10 - 8e-6 > 0
    assert bounds.lower <= c.derivatives(0.5)[1] <= bounds.upper
    with pytest.raises(ValueError, match="ordered finite"):
        c.enclosure(0.8, 0.2)
    with pytest.raises(ValueError, match="declared domain"):
        c.enclosure(-0.1, 0.3)


def test_new_codec_roundtrip_preserves_separate_policy_provenance_and_arithmetic():
    c = curve(endpoint=policy())
    raw = json.loads(json.dumps(c.to_dict()))
    assert raw["model_version"] == OCV_ENDPOINT_VERSION
    assert raw["endpoint_policy"]["model_version"] == OCV_ENDPOINT_POLICY_VERSION
    assert raw["source"] != raw["endpoint_policy"]["source"]
    restored = ElectrodeOCV.from_dict(raw)
    assert restored == c
    for x in (0.0, 0.2, 0.5, 1.0):
        assert restored.derivatives(x, order=4) == c.derivatives(x, order=4)
    assert restored.integral(0, 1) == c.integral(0, 1)
    assert restored.enclosure(0, 1, derivative=True) == c.enclosure(0, 1, derivative=True)


@pytest.mark.parametrize("mutation", [
    "old-version-with-policy", "new-version-missing-policy", "null-policy", "unknown-version",
    "nonstr-version",
    "policy-unknown-version", "policy-extra-field", "policy-missing-source", "policy-source-extra",
])
def test_codecs_fail_closed(mutation):
    raw = curve(endpoint=policy()).to_dict()
    if mutation == "old-version-with-policy":
        raw["model_version"] = OCV_VERSION
    elif mutation == "new-version-missing-policy":
        del raw["endpoint_policy"]
    elif mutation == "null-policy":
        raw["endpoint_policy"] = None
    elif mutation == "unknown-version":
        raw["model_version"] = "unknown"
    elif mutation == "nonstr-version":
        raw["model_version"] = []
    elif mutation == "policy-unknown-version":
        raw["endpoint_policy"]["model_version"] = "unknown"
    elif mutation == "policy-extra-field":
        raw["endpoint_policy"]["callback"] = "unknown"
    elif mutation == "policy-missing-source":
        del raw["endpoint_policy"]["source"]
    elif mutation == "policy-source-extra":
        raw["endpoint_policy"]["source"]["extra"] = "unknown"
    with pytest.raises(ValueError):
        ElectrodeOCV.from_dict(raw)


@pytest.mark.parametrize("kwargs", [
    {"barrier_V": 0}, {"barrier_V": -1}, {"barrier_V": True}, {"barrier_V": float("nan")},
    {"stoichiometry_clip": 0}, {"stoichiometry_clip": 0.5}, {"stoichiometry_clip": 1e-100},
])
def test_invalid_policies_reject(kwargs):
    with pytest.raises(ValueError):
        policy(**kwargs)
    with pytest.raises(ValueError, match="model InputSource"):
        ElectrodeOCVEndpointPolicy(source("measured material", "measured"))
    with pytest.raises(ValueError, match="explicit declaration"):
        replace(curve(), endpoint_policy={})


def test_none_policy_preserves_legacy_codec_and_exact_roundtrip_arithmetic():
    c = ElectrodeOCV(
        "legacy-exact",
        InputSource(DataSourceProvenance(
            "legacy", 0, "Synthetic OCV arithmetic fixture", "1", "CC0-1.0"
        ), "synthetic"),
        (0.2, -0.1), ((0.01, -3.0, 0.0),), ((-0.02, 5.0, 0.4),),
    )
    expected = {
        "model_version": OCV_VERSION, "material_id": "legacy-exact", "source": c.source.to_dict(),
        "polynomial_V": [0.2, -0.1], "exponential_terms": [[0.01, -3.0, 0.0]],
        "tanh_terms": [[-0.02, 5.0, 0.4]], "stoichiometry_domain": [0.01, 0.99],
        "supported_models": ["SPM", "SPMe", "DFN"], "source_row_id": "", "parameter_inputs": {},
    }
    assert c.to_dict() == expected
    assert list(c.to_dict()) == list(expected)
    restored = ElectrodeOCV.from_dict(json.loads(json.dumps(expected)))
    assert restored == c and restored.endpoint_policy is None
    # Closed derivatives check the mathematics independently of nested jets.
    # A fixed hexadecimal snapshot of exp/tanh-derived values is not a portable
    # accuracy reference; serialization on the same runtime must remain exact.
    x = 0.3
    e = 0.01 * exp(-3 * x)
    t = tanh(5 * (x - 0.4))
    expected_derivatives = (
        0.2 - 0.1 * x + e - 0.02 * t,
        -0.1 - 3 * e - 0.02 * 5 * (1 - t * t),
        9 * e + 0.02 * 25 * 2 * t * (1 - t * t),
        -27 * e - 0.02 * 125 * (-2 + 8 * t**2 - 6 * t**4),
        81 * e - 0.02 * 625 * (16 * t - 40 * t**3 + 24 * t**5),
    )
    actual_derivatives = c.derivatives(x, order=4)
    assert actual_derivatives == pytest.approx(expected_derivatives, rel=2e-14, abs=1e-14)
    assert restored.derivatives(x, order=4) == actual_derivatives
    assert c.integral(0.1, 0.9).hex() == "0x1.e504e310a6d9fp-4"
    assert c.integral(0.9, 0.1).hex() == "-0x1.e504e310a6d9fp-4"
    bounds = c.enclosure(0.1, 0.9)
    assert (bounds.lower.hex(), bounds.upper.hex()) == (
        "0x1.747d426777d25p-4", "0x1.b95de88beb236p-3"
    )
    slopes = c.enclosure(0.1, 0.9, derivative=True)
    assert (slopes.lower.hex(), slopes.upper.hex()) == (
        "-0x1.c71da9c0d37dbp-3", "-0x1.acc01985f7fc3p-4"
    )
