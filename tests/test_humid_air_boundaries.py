"""Full homogeneous model T/P endpoint block, declared before execution."""

import pytest

from chemworld.physchem.humid_air import humid_air_state, humid_air_tp


@pytest.mark.parametrize("T", [193, 300, 473])
@pytest.mark.parametrize("A", [1, 1-1e-10])
@pytest.mark.parametrize("p", [1, 101325, 5000000])
def test_model_domain_endpoints(A, T, p):
    state = humid_air_tp(A, T, p)
    assert state.pressure_Pa == pytest.approx(p, rel=5e-11)
    assert state.mass_density_kg_m3 > 0
    assert state.cp_J_kg_K >= state.cv_J_kg_K > 0
    assert A == 1 or state.composition_stiffness_J_kg > 0
    assert humid_air_state(A, T, state.mass_density_kg_m3) == state
