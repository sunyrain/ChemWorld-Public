"""Finite nested scans preserve their axes and reject unknown work before execution."""

import pytest

from chemworld.data.sedml_declarations import records
from chemworld.data.sedml_ranges import SEDMLRanges
from chemworld.data.sedml_tasks import SEDMLTasks
from test_sedml_declarations import SEDML, document


def scan(text=SEDML):
    return records(document(text).declaration, "listOfTasks")[1]


def test_master_grid_and_model_dependent_function_are_evaluated_at_each_boundary():
    functional = '''<functionalRange id="scaled" range="grid"><listOfVariables>
<variable id="live" modelReference="m1"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']"/>
<variable id="other" target="#grid"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>
<ci>live</ci><ci>other</ci><ci>grid</ci></apply></math></functionalRange>'''
    ranges = SEDMLRanges(scan(SEDML.replace('</listOfRanges>', functional+'</listOfRanges>')))
    calls = []

    def live(variable):
        calls.append(variable.attributes["id"])
        return len(calls)

    assert ranges.count == 3
    assert dict(ranges.values_at(0, model_value=live)) == pytest.approx({"grid": .1, "scaled": 1.2})
    assert dict(ranges.values_at(1, model_value=live)) == pytest.approx({"grid": .2, "scaled": 2.4})
    assert calls == ["live", "live"]
    with pytest.raises(TypeError):
        ranges.values_at(0, model_value=live)["grid"] = 10
    with pytest.raises(ValueError, match="current model"):
        ranges.values_at(0)


@pytest.mark.parametrize("spacing,start,end,expected", [
    ("linear", "2", "8", (2., 5., 8.)), ("linear", "8", "2", (8., 5., 2.)),
    ("log", "1", "100", (1., 10., 100.)), ("log", "100", "1", (100., 10., 1.)),
])
def test_uniform_ranges_include_both_endpoints(spacing, start, end, expected):
    text = SEDML.replace('<vectorRange id="grid"><value>0.1</value><value>0.2</value>'
                        '<value>0.4</value></vectorRange>',
        f'<uniformRange id="grid" start="{start}" end="{end}" numberOfSteps="2" '
        f'type="{spacing}"/>')
    ranges = SEDMLRanges(scan(text))
    actual = tuple(ranges.values_at(i)["grid"] for i in range(ranges.count))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("extra", [
    '<vectorRange id="short"><value>1</value></vectorRange>',
    '<uniformRange id="bad" start="0" end="1" numberOfSteps="2" type="log"/>',
    '<uniformRange id="huge" start="1" end="2" numberOfSteps="10000" type="linear"/>',
    '''<functionalRange id="cycle" range="cycle"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>cycle</ci></math></functionalRange>''',
    '''<functionalRange id="unknown" range="missing"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>missing</ci></math></functionalRange>''',
    '''<functionalRange id="undeclared" range="grid"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>missing</ci></math></functionalRange>''',
])
def test_every_auxiliary_range_is_validated_even_if_it_is_not_used(extra):
    with pytest.raises(ValueError):
        SEDMLRanges(scan(SEDML.replace('</listOfRanges>', extra+'</listOfRanges>')))


def test_nested_invocation_and_sample_counts_are_fixed_before_execution():
    outer = '''<repeatedTask id="outer" range="outer_grid" resetModel="true">
<listOfRanges><vectorRange id="outer_grid"><value>1</value><value>2</value></vectorRange>
</listOfRanges><listOfSubTasks><subTask task="scan"/></listOfSubTasks></repeatedTask>'''
    tasks = SEDMLTasks(document(SEDML.replace('</listOfTasks>', outer+'</listOfTasks>')))
    assert tasks.roots == ("outer",)
    assert dict(tasks.invocations) == {"t1": 1, "scan": 3, "outer": 6}
    assert dict(tasks.samples) == {"t1": 11, "scan": 33, "outer": 66}
    assert tasks.model_ids["outer"] == frozenset(("m1",))
    with pytest.raises(ValueError, match="budget"):
        SEDMLTasks(tasks.document, maximum_invocations=5)


@pytest.mark.parametrize("old,new", [
    ('<subTask task="t1"/>', '<subTask task="missing"/>'),
    ('<subTask task="t1"/>', '<subTask task="scan"/>'),
    ('modelReference="m1" simulationReference="s1"',
     'modelReference="missing" simulationReference="s1"'),
    ('simulationReference="s1"', 'simulationReference="missing"'),
    ('<subTask task="t1"/>', '<subTask task="t1"/><subTask task="t1"/>'),
    ('<setValue modelReference="m1"', '<setValue modelReference="m0"'),
])
def test_invalid_tasks_and_ambiguous_shared_model_order_reject_the_whole_graph(old, new):
    with pytest.raises(ValueError):
        SEDMLTasks(document(SEDML.replace(old, new)))


def test_explicit_subtask_order_counts_duplicate_invocations_and_preserves_order():
    text = SEDML.replace('<subTask task="t1"/>',
        '<subTask task="t1" order="2"/><subTask task="t1" order="1"/>')
    tasks = SEDMLTasks(document(text))
    assert tasks.invocations["scan"] == 6 and tasks.samples["scan"] == 66
    assert [n.attributes["order"] for n in tasks.sub_tasks["scan"]] == ["1", "2"]


def test_task_nesting_budget_is_checked_even_when_every_extra_range_has_only_one_value():
    text, previous = SEDML, "scan"
    for index in range(15):
        identity = "nested_"+str(index)
        extra = f'''<repeatedTask id="{identity}" range="r_{index}" resetModel="false">
<listOfRanges><vectorRange id="r_{index}"><value>0</value></vectorRange></listOfRanges>
<listOfSubTasks><subTask task="{previous}"/></listOfSubTasks></repeatedTask>'''
        text = text.replace('</listOfTasks>', extra+'</listOfTasks>')
        previous = identity
    with pytest.raises(ValueError, match="sixteen-level"):
        SEDMLTasks(document(text))


def test_unordered_subtask_changes_cannot_cross_into_the_other_subtasks_model():
    text = SEDML.replace('</listOfTasks>',
        '<task id="t2" modelReference="m0" simulationReference="s1"/></listOfTasks>')
    children = '''<subTask task="t1"><listOfChanges><setValue modelReference="m0" range="grid"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>grid</ci></math></setValue></listOfChanges>
</subTask><subTask task="t2"/>'''
    with pytest.raises(ValueError, match="unordered subtasks"):
        SEDMLTasks(document(text.replace('<subTask task="t1"/>', children)))
