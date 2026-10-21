"""Source-time grids and explicit algorithms; no implicit CVODE substitution."""

import pytest

from chemworld.data.sedml_simulations import resolve_simulations
from test_sedml_declarations import SEDML, document


def test_source_grid_has_steps_plus_one_and_algorithm_max_step_converts_once():
    simulation = resolve_simulations(document())["s1"]
    assert simulation.source_times == pytest.approx(tuple(i*.4 for i in range(11)))
    assert simulation.algorithm.kisao_id == "KISAO:0000288"
    policy = simulation.algorithm.native_policy(seconds_per_source_time=60.)
    assert policy.method == "BDF" and policy.max_step_s == 1.2
    assert policy.rtol == 1e-10 and policy.atol == 1e-12


def test_lsoda_maps_to_the_existing_lsoda_backend_and_nonzero_source_time_is_preserved():
    text = SEDML.replace("KISAO:0000288", "KISAO:0000088").replace(
        'initialTime="0"', 'initialTime="5"').replace('outputStartTime="0"',
        'outputStartTime="6"').replace('outputEndTime="4"', 'outputEndTime="9"')
    simulation = resolve_simulations(document(text))["s1"]
    assert simulation.initial_time == 5
    assert simulation.source_times[0] == 6 and simulation.source_times[-1] == 9
    assert simulation.algorithm.native_policy(seconds_per_source_time=1).method == "LSODA"


def test_local_numerical_options_override_l1v4_global_options_with_the_same_id():
    global_options = '''<listOfAlgorithmParameters>
<algorithmParameter kisaoID="KISAO:0000209" value="1e-5"/>
<algorithmParameter kisaoID="KISAO:0000211" value="2e-12"/></listOfAlgorithmParameters>'''
    text = SEDML.replace('</sedML>', global_options+'</sedML>')
    policy = resolve_simulations(document(text))["s1"].algorithm
    assert policy.rtol == 1e-10 and policy.atol == 2e-12


@pytest.mark.parametrize("old,new", [("KISAO:0000288", "KISAO:0000019"),
    ("KISAO:0000288", "KISAO:0000029"), ("KISAO:0000467", "KISAO:9999999"),
    ('value="0.02"', 'value="0"'), ('numberOfSteps="10"', 'numberOfSteps="10001"')])
def test_unsupported_algorithms_options_or_expansion_are_rejected_before_native_runs(old, new):
    with pytest.raises(ValueError):
        resolve_simulations(document(SEDML.replace(old, new)))
