import json

import numpy as np
import pytest

from chemworld.physchem import (
    VirialRootError,
    density_to_pressure_virial,
    density_virial_state,
    meng_duan_interactions,
    pressure_to_density_virial,
    pressure_virial_state,
    pressure_virial_volume,
    virial_cross_parameters,
    virial_mixture_response,
    virial_pressure,
    virial_volume_composition_response,
)


def constant(matrix):
    n = len(matrix)
    return (matrix, *([[0.0] * n for _ in range(n)] for _ in range(3)))


def test_homogeneous_composition_identities_and_simplex_projection():
    z = np.array([0.2, 0.3, 0.5])
    for kind, matrix, degree in [
        ("B", [[-1e-4, -2e-4, 1e-5], [-2e-4, -3e-4, 0], [1e-5, 0, 2e-4]], 2),
        ("C", [[1e-8, 2e-8, 3e-8], [2e-8, 4e-8, 5e-8], [3e-8, 5e-8, 6e-8]], 3),
    ]:
        r = virial_mixture_response(z, constant(matrix), kind=kind)
        assert z @ r.gradient == pytest.approx(degree * r.value)
        np.testing.assert_allclose(np.array(r.hessian) @ z, (degree - 1) * np.array(r.gradient))
        s = virial_mixture_response(z, constant(matrix), kind=kind, coordinates="independent")
        projection = np.array([[1, 0], [0, 1], [-1, -1]])
        np.testing.assert_allclose(s.gradient, projection.T @ r.gradient)
        np.testing.assert_allclose(s.hessian, projection.T @ r.hessian @ projection)
        np.testing.assert_allclose(
            s.third,
            np.einsum("ia,jb,kc,ijk->abc", projection, projection, projection, r.third),
            atol=1e-18,
        )
        assert json.loads(json.dumps(s.to_dict()))["coordinates"] == "independent"


def test_signed_and_zero_third_coefficients_do_not_get_clipped():
    r = virial_mixture_response([0.3, 0.7], constant([[-1e-8] * 2] * 2), kind="C")
    assert r.value == pytest.approx(-1e-8)
    assert r.gradient == pytest.approx([-3e-8, -3e-8])
    z = virial_mixture_response([0.3, 0.7], constant([[0.0] * 2] * 2), kind="C")
    assert z.value == 0
    assert z.temperature_derivatives == (0, 0, 0, 0)
    # A pure coefficient crossing zero is Ciii = Cii, not a singular cube root.
    p = virial_mixture_response([1.0], ([[0]], [[1e-9]], [[0]], [[0]]), kind="C")
    assert p.temperature_derivatives == (0, 1e-9, 0, 0)
    with pytest.raises(ValueError, match="continuation"):
        virial_mixture_response(
            [0.3, 0.7],
            ([[1e-8, 0], [0, 1e-8]], [[0, 1e-9], [1e-9, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]),
            kind="C",
        )


@pytest.mark.parametrize(
    "coefficients", [(), (1e-4,), (-1e-4, 2e-8), (1e-4, 1e-8, 1e-12), (1e-4, 1e-8, 1e-12, 1e-16)]
)
def test_series_conversion_and_both_roundtrips(coefficients):
    t, p = 500.0, 1e5
    converted = density_to_pressure_virial(t, coefficients)
    np.testing.assert_allclose(
        pressure_to_density_virial(t, converted), coefficients, rtol=1e-12, atol=1e-25
    )
    state = density_virial_state(t, p, coefficients)
    assert virial_pressure(t, state.molar_volume_m3_mol, coefficients) == pytest.approx(p)
    assert state.pressure_volume_slope < 0
    assert any(c["accepted"] for c in state.candidates)
    v = pressure_virial_volume(t, p, converted)
    pressure = pressure_virial_state(t, v, converted)
    assert pressure.pressure_Pa == pytest.approx(p)
    assert pressure.pressure_volume_slope < 0
    json.dumps(state.to_dict(), allow_nan=False)


def test_two_density_roots_and_spinodal_rejection():
    # In units where ideal volume is one, Z^2-Z-B=0.
    t, p = 1 / 8.31446261815324, 1.0
    state = density_virial_state(t, p, (-0.2,))
    assert state.compressibility_factor == pytest.approx((1 + np.sqrt(0.2)) / 2)
    assert len(state.candidates) == 2
    assert sum(c["accepted"] for c in state.candidates) == 1
    for b in (-0.25, -0.3):
        with pytest.raises(VirialRootError) as exc:
            density_virial_state(t, p, (b,))
        assert exc.value.candidates


def test_volume_responses_and_expansion_distinction():
    z = [0.4, 0.6]
    b = virial_mixture_response(z, constant([[-1e-4, 2e-5], [2e-5, 3e-4]]))
    c = virial_mixture_response(z, constant([[1e-8, 3e-8], [3e-8, 5e-8]]), kind="C")
    for expansion in ("density", "pressure"):
        r = virial_volume_composition_response(500, 1e6, b, c, expansion=expansion)
        np.testing.assert_allclose(r["hessian_m3_mol"], np.array(r["hessian_m3_mol"]).T)
        assert r["state"]["pressure_volume_slope"] < 0
    assert density_virial_state(
        500, 1e6, (b.value, c.value)
    ).molar_volume_m3_mol != pressure_virial_volume(
        500, 1e6, density_to_pressure_virial(500, (b.value, c.value))
    )


def test_explicit_cross_inputs_and_unknown_policy():
    cross = virial_cross_parameters([200, 400], [4e6, 5e6], [1e-4, 2e-4], [0.01, 0.2])
    assert cross[0][1] == cross[1][0]
    assert cross[0][0]["Tc"] == pytest.approx(200)
    assert cross[1][1]["Pc"] == pytest.approx(5e6)
    with pytest.raises(ValueError, match="symmetric"):
        virial_cross_parameters(
            [200, 400], [4e6, 5e6], [1e-4, 2e-4], [0.01, 0.2], interactions=[[0, 0.1], [0.2, 0]]
        )
    with pytest.raises(ValueError, match="no declared"):
        meng_duan_interactions([("alkane", 2), ("other", 0)])
    assert meng_duan_interactions([("alkane", 2), ("other", 0)], unknown="zero") == ((0, 0), (0, 0))
    with pytest.raises(ValueError):
        meng_duan_interactions([("alkane", 0)])


@pytest.mark.parametrize(
    "fractions,matrices",
    [
        ([0.2, 0.2], constant([[1, 1], [1, 1]])),
        ([1.0], ([[1]],)),
        ([1.0], constant([[float("nan")]])),
    ],
)
def test_invalid_mixture_inputs(fractions, matrices):
    with pytest.raises(ValueError):
        virial_mixture_response(fractions, matrices)
