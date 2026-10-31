"""Shared Gibbs packed seeds against explicit multivariate analytic derivatives."""

import numpy as np
import pytest

from chemworld.physchem.caloric_math import exponential, logarithm
from chemworld.physchem.calphad_response import analytic_response
from chemworld.physchem.gibbs_phase import GibbsPhaseModel


def test_packed_nested_seeds_preserve_cross_derivatives_and_disconnected_outputs():
    calls = []

    def function(z):
        calls.append(1)
        x, y, w = z
        return (x*x*y+exponential(w)*logarithm(y)+x/y, 42., w)

    x, y, w = 1.2, 2.3, -.4
    values, jacobian, hessian = analytic_response(function, (x, y, w))
    e = np.exp(w)
    assert len(calls) == 1
    assert values == pytest.approx((x*x*y+e*np.log(y)+x/y, 42, w))
    assert jacobian[0] == pytest.approx((2*x*y+1/y, x*x+e/y-x/y**2, e*np.log(y)))
    assert hessian[0] == pytest.approx(np.array([
        [2*y, 2*x-1/y**2, 0],
        [2*x-1/y**2, -e/y**2+2*x/y**3, e/y],
        [0, e/y, e*np.log(y)],
    ]))
    assert np.array_equal(jacobian[1], np.zeros(3))
    assert np.array_equal(jacobian[2], [0, 0, 1])
    assert np.array_equal(hessian[1:], np.zeros((2, 3, 3)))


def test_shared_phase_response_uses_declared_branches_without_cef_private_fields():
    class AnalyticPhase(GibbsPhaseModel):
        _diagnostics = ()
        constituents = (("FE", "NI"),)
        constraint_labels = ("site_sum:0",)

        def _constraints(self, fractions):
            return (sum(fractions[0].values()) - 1,)

        def _composition(self, fractions):
            total = sum(fractions[0].values())
            return total, {name: value / total for name, value in fractions[0].items()}

        def _energies(self, fractions, temperature, pressure):
            return {"polynomial": -2 * temperature**2 + 3 * pressure
                    + 5 * fractions[0]["FE"]**2}

        def _derivative_branches(self, fractions, temperature, pressure, **kwargs):
            return ("declared test boundary",) if temperature == 12 else ()

    phase = AnalyticPhase()
    sites = [{"FE": .5, "NI": .5}]
    thermal = phase.temperature_response(sites, 10., 20.)
    assert thermal.gibbs_J_mol == -138.75
    assert thermal.enthalpy_J_mol == 261.25
    assert thermal.entropy_J_mol_K == 40.
    assert thermal.fixed_constitution_heat_capacity_J_mol_K == 40.
    response = phase.derivatives(sites, 10., 20.)
    np.testing.assert_array_equal(response.jacobian[0], [-40., 3., 5., 0.])
    np.testing.assert_array_equal(response.hessian[0], np.diag([-4., 0., 10., 0.]))
    for readout in (phase.temperature_response, phase.derivatives):
        with pytest.raises(ValueError, match="one-sided"):
            readout(sites, 12., 20.)
        accepted = readout(sites, 12., 20., allow_one_sided=True)
        assert accepted.diagnostics == ("declared test boundary",)
