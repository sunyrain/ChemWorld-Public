"""Original CEF tangent search resolves dilute chemical potentials at fixed gates."""

from dataclasses import replace

import numpy as np
import pytest
from tests.reference.calphad_equilibrium_models import ELEMENTS, native_spec, solution

from chemworld.physchem.calphad_equilibrium_search import PhaseSurface, search_phase


@pytest.mark.parametrize("fractions", ((1 - 1e-8, 1e-8), (1e-8, 1 - 1e-8), (0.3, 0.7 - 1e-8, 1e-8)))
def test_original_tangent_gradient_is_polished_without_relaxing_site_or_stationarity(fractions):
    elements = ("FE", "NI", "CR")[: len(fractions)]
    spec = native_spec(
        {"tdb": ELEMENTS + solution("IDEAL", elements, (0,) * len(elements)), "phases": ["IDEAL"]}
    )
    rt = 8.3145 * 300.0
    mu = rt * np.log(fractions)
    surface = PhaseSurface(spec.models[0], elements, 300.0, 101325.0, spec.policy.site_floor)
    minima, attempts = search_phase(surface, surface.samples(8), mu, rt, spec.policy)
    assert minima and all(a["success"] for a in attempts), attempts
    assert all(a["stationarity_RT"] <= 1e-5 for a in attempts)
    for q, distance in minima:
        observed = surface.model._composition(surface.sites(q))[1]
        assert [observed[e] for e in elements] == pytest.approx(fractions, abs=2e-9)
        assert abs(distance) < 2e-5
        assert min(surface.constraints(q)) >= 0
    refined = [a["stationary_refinement"] for a in attempts if "stationary_refinement" in a]
    assert refined and all(r["accepted"] and r["original_stationarity_RT"] > 1e-5 for r in refined)
    limited = replace(spec.policy, maximum_iterations=1)
    minima, attempts = search_phase(surface, surface.samples(8), mu, rt, limited)
    assert not minima and not any(a["success"] for a in attempts)
    assert not any("stationary_refinement" in a for a in attempts)
