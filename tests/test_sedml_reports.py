"""Reports preserve multidimensional coordinates and do not invent derived units."""

import json

import pytest

from chemworld.data.sedml_models import resolve_models
from chemworld.data.sedml_reports import SEDMLArray, SEDMLReports
from chemworld.data.sedml_tasks import SEDMLTasks
from test_sbml import model
from test_sedml_declarations import SEDML, document


def plan(text=SEDML):
    parsed = document(text)
    return SEDMLReports(SEDMLTasks(parsed), resolve_models(parsed, {"model.xml": model()}))


def array(values=(0., 1., 2., 3., 4., 5.), keys=None, unit="second"):
    return SEDMLArray(values, (2, 3), ("scan:grid", "time"),
        tuple((repeat, float(time)) for repeat in range(2) for time in range(3))
        if keys is None else keys, unit)


def test_report_identity_preserves_repeat_axes_coordinates_labels_and_source_units():
    reports = plan().evaluate({("dg", "time"): array()})
    column = reports["report"][0]
    assert column["label"] == "time" and column["data_reference"] == "dg"
    assert column["data"] == array().to_dict()
    assert json.loads(json.dumps(reports, allow_nan=False))["report"][0] == column


def test_derived_math_is_shared_and_its_physical_unit_remains_explicitly_unknown():
    text = SEDML.replace('<ci>time</ci>',
        '<apply><times/><cn>2</cn><ci>time</ci></apply>')
    output = plan(text).evaluate({("dg", "time"): array()})["report"][0]["data"]
    assert output["values"] == [0., 2., 4., 6., 8., 10.]
    assert output["unit"] is None and output["shape"] == [2, 3]


def test_equal_lengths_do_not_align_different_actual_time_coordinates():
    variable = '''<variable id="other_time" taskReference="scan" symbol="KISAO:0000832"/>'''
    text = SEDML.replace('</listOfVariables>', variable+'</listOfVariables>').replace(
        '<ci>time</ci>', '<apply><minus/><ci>time</ci><ci>other_time</ci></apply>')
    changed = tuple((repeat, float(time)+.1) for repeat in range(2) for time in range(3))
    with pytest.raises(ValueError, match="time grids"):
        plan(text).evaluate({("dg", "time"): array(), ("dg", "other_time"): array(keys=changed)})


@pytest.mark.parametrize("old,new", [
    ('taskReference="scan"', 'taskReference="unknown"'),
    ('dataReference="dg"', 'dataReference="unknown"'),
    ('<ci>time</ci>', '<ci>undeclared</ci>'),
    ('symbol="urn:sedml:symbol:time"', 'symbol="urn:unknown"'),
])
def test_all_references_and_selectors_are_rejected_before_any_world_exists(old, new):
    with pytest.raises(ValueError):
        plan(SEDML.replace(old, new))


def test_wrong_unit_and_missing_observation_are_not_accepted_as_valid_data():
    reports = plan()
    with pytest.raises(ValueError, match="source unit"):
        reports.evaluate({("dg", "time"): array(unit="minute")})
    with pytest.raises(ValueError, match="exactly"):
        reports.evaluate({})


@pytest.mark.parametrize("values,shape,axes,keys", [
    ((float("nan"),), (1,), ("time",), ((0.,),)),
    ((True,), (1,), ("time",), ((0.,),)),
    ((1.,), (2,), ("time",), ((0.,),)),
    ((1.,), (1,), (), ((0.,),)),
    ((1.,), (1,), ("time",), (([],),)),
])
def test_malformed_or_mutable_report_arrays_are_rejected(values, shape, axes, keys):
    with pytest.raises(ValueError):
        SEDMLArray(values, shape, axes, keys)
