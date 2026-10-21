"""Fixed independent trajectory inputs are complete and exercise distinct semantics."""

import pytest

from chemworld.physchem.sbml_events import SBMLTriggerRoots
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from reference.run_sbml_trajectory_block import QUANTITIES, TIMES, cases, jumps
from test_sbml import model


@pytest.mark.parametrize("name,scales,index,text", list(cases()))
def test_all_eighteen_complete_sources_compile_in_the_declared_basis(name, scales, index, text):
    frame = SBMLKinetics(model(text))
    SBMLTriggerRoots(frame)
    assert frame.model.declaration.get("id") == f"trajectory_{index+1:02d}"
    assert frame.model.model_unit("timeUnits").time_s(1.) == scales["time_s"]
    assert frame.model.model_unit("substanceUnits").amount_mol(1.) == scales["amount_mol"]
    assert frame.model.model_unit("volumeUnits").volume_l(1.) == scales["volume_L"]
    assert name == f"trajectory-{index+1:02d}.xml"


def test_fixed_coverage_and_same_time_jump_groups_keep_priority_substeps():
    assert len(list(cases())) == 18 and len(TIMES) == 11 and len(QUANTITIES) == 7
    assert len({text for _, _, _, text in cases()}) == 18
    rows = [{"time_source": t, "values": [v]} for t, v in
            ((0., 0.), (1., 1.), (1., 2.), (1., 3.), (2., 4.), (3., 4.))]
    assert jumps(rows) == [1.]
