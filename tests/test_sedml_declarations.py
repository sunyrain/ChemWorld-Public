"""Original declarations and closed structural profile; no experiment is run."""

import json

import pytest

from chemworld.data.sedml_declarations import SEDMLDocument, mathematics, records, step_count
from test_sbml import model

SEDML = '''<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1" version="4"
xmlns:s="http://www.sbml.org/sbml/level3/version2/core">
<notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Self-authored finite experiment.</p>
</body></notes><listOfModels><model id="m0" language="urn:sedml:language:sbml"
source="model.xml"/><model id="m1" language="urn:sedml:language:sbml" source="#m0">
<listOfChanges><changeAttribute
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value"
newValue="0.4"/></listOfChanges></model></listOfModels>
<listOfSimulations><uniformTimeCourse id="s1" initialTime="0" outputStartTime="0"
outputEndTime="4" numberOfSteps="10"><algorithm kisaoID="KISAO:0000288">
<listOfAlgorithmParameters><algorithmParameter kisaoID="KISAO:0000209" value="1e-10"/>
<algorithmParameter kisaoID="KISAO:0000467" value="0.02"/></listOfAlgorithmParameters>
</algorithm></uniformTimeCourse></listOfSimulations>
<listOfTasks><task id="t1" modelReference="m1" simulationReference="s1"/>
<repeatedTask id="scan" range="grid" resetModel="true" concatenate="false"><listOfRanges>
<vectorRange id="grid"><value>0.1</value><value>0.2</value><value>0.4</value></vectorRange>
</listOfRanges><listOfChanges><setValue modelReference="m1" range="grid"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value">
<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>grid</ci></math></setValue>
</listOfChanges><listOfSubTasks><subTask task="t1"/></listOfSubTasks></repeatedTask></listOfTasks>
<listOfDataGenerators><dataGenerator id="dg"><listOfVariables><variable id="time"
taskReference="scan" symbol="urn:sedml:symbol:time"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>time</ci></math></dataGenerator>
</listOfDataGenerators><listOfOutputs><report id="report"><listOfDataSets><dataSet
id="dataset" label="time" dataReference="dg"/></listOfDataSets></report></listOfOutputs></sedML>'''


def document(text=SEDML):
    return SEDMLDocument(text, model().source)


def test_original_experiment_text_namespace_scopes_math_and_provenance_roundtrip():
    original = document()
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    restored = SEDMLDocument.from_dict(payload)
    assert restored.to_xml() == SEDML and restored.source == original.source
    assert restored.to_dict() == payload
    simulation = records(restored.declaration, "listOfSimulations")[0]
    assert step_count(simulation.attributes) == 10
    generator = records(restored.declaration, "listOfDataGenerators")[0]
    assert mathematics(generator).evaluate({"time": 3.}) == 3.
    derived = records(restored.declaration, "listOfModels")[1]
    change = records(derived, "listOfChanges")[0]
    assert change.namespaces["s"] == "http://www.sbml.org/sbml/level3/version2/core"


def test_deprecated_point_count_keeps_the_identical_step_count_meaning():
    simulation = records(document(SEDML.replace("numberOfSteps", "numberOfPoints")).declaration,
                         "listOfSimulations")[0]
    assert step_count(simulation.attributes)+1 == 11


@pytest.mark.parametrize("old,new", [
    ('numberOfSteps="10"', 'numberOfSteps="10" numberOfPoints="10"'),
    ('numberOfSteps="10"', 'numberOfSteps="0"'),
    ('numberOfSteps="10"', 'numberOfSteps="10.0"'),
    ('outputStartTime="0"', 'outputStartTime="-1"'),
    ('outputEndTime="4"', 'outputEndTime="NaN"'),
    ('resetModel="true"', 'resetModel="yes"'),
    ('version="4"', 'version="5"'),
    ('outputEndTime="4"', 'outputEndTime="4" silentlyIgnored="true"'),
    ('<task id="t1"', '<task xmlns="urn:other" id="t1"'),
    ('<algorithm kisaoID="KISAO:0000288">', '<algorithm kisaoID="KISAO:0000288"><executable/>'),
    ('<ci>time</ci>', '<apply><ci>load_code</ci><ci>time</ci></apply>'),
    ('<value>0.1</value>', '<value>1_000</value>'),
    ('<value>0.1</value>', '<value>0.1</value><notes/><notes/>'),
    ('id="dataset"', 'id="time"'),
])
def test_unknown_or_malformed_declarations_are_refused_as_a_whole(old, new):
    assert old in SEDML
    with pytest.raises(ValueError):
        document(SEDML.replace(old, new))


def test_json_cannot_request_arbitrary_input_types_or_extra_semantics():
    payload = document().to_dict()
    payload["factory"] = "some.module:execute"
    with pytest.raises(ValueError):
        SEDMLDocument.from_dict(payload)
