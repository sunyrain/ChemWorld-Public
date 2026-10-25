"""Algebraic Itô moment/invariant checks, not independent SDK qualification."""

import json
from copy import deepcopy
from itertools import product
from math import sqrt

import numpy as np
import pytest

from chemworld.physchem import RI5Draw, RI5Step, draw_ri5, ri5_step


def test_explicit_random_variables_round_trip_and_empty_noise_preserves_rng():
    rng = np.random.default_rng(17)
    before = deepcopy(rng.bit_generator.state)
    assert draw_ri5(rng, 0) == RI5Draw((), ())
    assert rng.bit_generator.state == before
    drawn = draw_ri5(rng, 12)
    assert RI5Draw.from_dict(json.loads(json.dumps(drawn.to_dict()))) == drawn
    assert set(drawn.hat) <= {-sqrt(3), 0, sqrt(3)}
    assert set(drawn.tilde) <= {-1, 1}


@pytest.mark.parametrize("change", ["schema", "unknown", "hat", "tilde", "dimension"])
def test_unknown_or_invalid_saved_random_variables_are_rejected(change):
    saved = RI5Draw((0.0,), (1,)).to_dict()
    if change == "schema":
        saved["schema"] = "other-method"
    elif change == "unknown":
        saved["code"] = "execute"
    elif change == "hat":
        saved["hat"] = [True]
    elif change == "tilde":
        saved["tilde"] = [0]
    else:
        saved["hat"] = []
    with pytest.raises(ValueError):
        RI5Draw.from_dict(saved)


def test_deterministic_limit_has_cubic_linear_stability_and_exact_time_quadrature():
    h, t, initial = 0.125, 0.75, 2.0
    z = -0.4 * h
    result = ri5_step(lambda _, y: -0.4 * y, None, t, [initial], h, RI5Draw((), ()))
    assert isinstance(result, RI5Step)
    assert result.values[0] == pytest.approx(initial * (1 + z + z * z / 2 + z**3 / 6), abs=1e-14)
    result = ri5_step(lambda at, _: [at * at], None, t, [initial], h, RI5Draw((), ()))
    assert result.values[0] == pytest.approx(initial + ((t + h) ** 3 - t**3) / 3, abs=1e-14)
    assert result.diffusion_evaluations == 0


def test_constant_additive_noise_and_linear_material_invariant():
    y = np.array([2.0, 3.0, 5.0])
    original = y.copy()
    drift = np.array([-0.5, 0.2, 0.3])
    matrix = np.array([[-1.0, 0.0], [1.0, -2.0], [0.0, 2.0]])
    draw = RI5Draw((-sqrt(3), sqrt(3)), (-1, 1))
    result = ri5_step(lambda *_: drift, lambda *_: matrix, 0, y, 0.04, draw)
    expected = y + 0.04 * drift - 0.2 * (matrix @ np.array(draw.hat))
    assert result.values == pytest.approx(expected, abs=1e-14)
    assert sum(result.values) == pytest.approx(sum(y), abs=1e-14)
    assert np.array_equal(y, original)


def test_geometric_brownian_ito_moments_through_second_order_exactly():
    h = 0.125
    values = [
        (
            weight,
            ri5_step(
                lambda *_: [0.0], lambda _, y: [[y[0]]], 0, [1.0], h, RI5Draw((hat,), (1,))
            ).values[0],
        )
        for hat, weight in ((-sqrt(3), 1 / 6), (0.0, 2 / 3), (sqrt(3), 1 / 6))
    ]
    assert sum(p * y for p, y in values) == pytest.approx(1.0, abs=1e-14)
    assert sum(p * y * y for p, y in values) == pytest.approx(1 + h + h * h / 2, abs=1e-14)


def test_noncommuting_diffusions_preserve_ito_product_and_cross_integral_variance():
    # dX=dW1, dY=dW2, dZ=X dW2, dU=Y dW1. Thus Z+U=XY,
    # E[Z]=E[U]=0 and E[Z^2]=E[U^2]=h^2/2 from the Itô isometry.
    h, samples = 0.125, []
    lattice = ((-sqrt(3), 1 / 6), (0.0, 2 / 3), (sqrt(3), 1 / 6))
    for (x1, p1), (x2, p2), eta1, eta2 in product(lattice, lattice, (-1, 1), (-1, 1)):
        result = ri5_step(
            lambda *_: np.zeros(4),
            lambda _, y: [[1.0, 0.0], [0.0, 1.0], [0.0, y[0]], [y[1], 0.0]],
            0,
            [0.0, 0.0, 0.0, 0.0],
            h,
            RI5Draw((x1, x2), (eta1, eta2)),
        )
        x, y, z, u = result.values
        assert z + u == pytest.approx(x * y, abs=1e-14)
        samples.append((p1 * p2 / 4, z, u))
    for index in (1, 2):
        assert sum(row[0] * row[index] for row in samples) == pytest.approx(0.0, abs=1e-14)
        assert sum(row[0] * row[index] ** 2 for row in samples) == pytest.approx(
            h * h / 2, abs=1e-14
        )


def test_zero_step_calls_no_model_and_negative_candidates_are_not_clipped():
    def unavailable(*_):
        raise AssertionError("zero interval must not evaluate physics")

    assert ri5_step(unavailable, unavailable, 3, [2.0], 0, RI5Draw((0.0,), (1,))).values == (2.0,)
    result = ri5_step(lambda *_: [-2.0], None, 0, [1.0], 1, RI5Draw((), ()))
    assert result.values[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("callback", ["drift-shape", "drift-nonfinite", "diffusion-shape"])
def test_invalid_callback_outputs_fail_before_returning_a_candidate(callback):
    drift = (
        (lambda *_: [0.0, 0.0])
        if callback == "drift-shape"
        else ((lambda *_: [float("nan")]) if callback == "drift-nonfinite" else (lambda *_: [0.0]))
    )

    def diffusion(*_):
        return [1.0]

    with pytest.raises(ValueError):
        ri5_step(drift, diffusion, 0, [1.0], 0.1, RI5Draw((0.0,), (1,)))


@pytest.mark.parametrize("invalid", [[1 + 2j], [True], ["1"], [], [float("inf")]])
def test_non_real_or_nonfinite_state_is_rejected_without_conversion(invalid):
    with pytest.raises(ValueError):
        ri5_step(lambda *_: [0.0], None, 0, invalid, 0.1, RI5Draw((), ()))


@pytest.mark.parametrize(
    "time,duration", [(-1, 0.1), (0, -0.1), (0, True), (float("inf"), 0.1), (1e20, 1e-5)]
)
def test_invalid_or_nonadvancing_clock_is_rejected(time, duration):
    with pytest.raises(ValueError):
        ri5_step(lambda *_: [0.0], None, time, [1.0], duration, RI5Draw((), ()))


@pytest.mark.parametrize("invalid", [np.array([1 + 2j]), np.array([True])])
def test_complex_and_boolean_callback_values_are_not_cast_to_real_physics(invalid):
    with pytest.raises(ValueError):
        ri5_step(lambda *_: invalid, None, 0, [1.0], 0.1, RI5Draw((), ()))
