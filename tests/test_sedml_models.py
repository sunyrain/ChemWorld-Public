"""Ordered model preprocessing, source-basis variables and whole-graph refusal."""

import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.data.sedml_models import resolve_models, source_quantity
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from test_sbml import model
from test_sedml_declarations import SEDML, document

PARAMETER = "/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value"


def test_model_inheritance_changes_only_the_derived_input_and_keeps_sources_and_units():
    supplied = model()
    resolved = resolve_models(document(), {"./model.xml": supplied})
    assert resolved["m0"].to_xml() == supplied.to_xml()
    assert SBMLKinetics(resolved["m0"]).context(SBMLKinetics(resolved["m0"]).initial_values, 0)[
        "k"] == .2
    assert source_quantity(resolved["m1"], PARAMETER,
        {"s": "http://www.sbml.org/sbml/level3/version2/core"}) == .4
    assert supplied.to_xml() == model().to_xml()
    assert resolved["m1"].source.provenance.license == supplied.source.provenance.license
    assert resolved["m1"].unit_defaults == supplied.unit_defaults
    assert resolved["m1"].source.provenance.checksum
    with pytest.raises(TypeError):
        resolved["m0"] = resolved["m1"]


def test_compute_change_reads_current_self_after_previous_changes_and_resolved_other_model():
    change = '''<computeChange target="'''+PARAMETER+'''">
<listOfVariables><variable id="x" modelReference="m0" target="'''+PARAMETER+'''"/>
<variable id="y" modelReference="m1" target="'''+PARAMETER+'''"/></listOfVariables>
<listOfParameters><parameter id="increment" value="0.1"/></listOfParameters>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>x</ci><ci>y</ci>
<ci>increment</ci></apply></math></computeChange>'''
    text = SEDML.replace('</listOfChanges></model>', change+'</listOfChanges></model>')
    resolved = resolve_models(document(text), {"model.xml": model()})
    value = source_quantity(resolved["m1"], PARAMETER,
        {"s": "http://www.sbml.org/sbml/level3/version2/core"})
    assert value == pytest.approx(.7)


@pytest.mark.parametrize("old,new", [
    ('source="model.xml"', 'source="#m1"'),
    ('source="#m0"', 'source="#unknown"'),
    ('source="model.xml"', 'source="https://example.org/model.xml"'),
    ('source="model.xml"', 'source="missing.xml"'),
    ('language="urn:sedml:language:sbml"', 'language="urn:sedml:language:cellml"'),
    ('newValue="0.4"', 'newValue="not_a_number"'),
    ("parameter[@id='k']/@value", "parameter[@id='absent']/@value"),
    ('xmlns:s="http://www.sbml.org/sbml/level3/version2/core"', 'xmlns:s="urn:wrong"'),
])
def test_unknown_model_sources_dependencies_targets_or_values_are_refused(old, new):
    with pytest.raises(ValueError):
        resolve_models(document(SEDML.replace(old, new)), {"model.xml": model()})


def test_closed_xpath_cannot_turn_annotation_metadata_into_a_physical_model_target():
    original = model()
    text = original.to_xml().replace('</annotation>',
        '<parameter xmlns="urn:test" id="k" value="99"/></annotation>')
    supplied = SBMLModel(text, original.source)
    target = "/s:sbml/s:model/s:annotation/a:parameter[@id='k']/@value"
    experiment = SEDML.replace('version="4"', 'version="4" xmlns:a="urn:test"').replace(
        PARAMETER, target)
    with pytest.raises(ValueError, match="actual declared core"):
        resolve_models(document(experiment), {"model.xml": supplied})


def test_resource_aliases_do_not_silently_choose_one_supplied_model():
    with pytest.raises(ValueError, match="duplicate local"):
        resolve_models(document(), {"model.xml": model(), "./model.xml": model()})


def test_preprocessing_does_not_ignore_a_quantity_selector_on_an_xml_literal():
    change = '''<computeChange target="'''+PARAMETER+'''"><listOfVariables>
<variable id="selected" modelReference="m0" target="'''+PARAMETER+'''"
symbol="KISAO:0000836"/></listOfVariables><math xmlns="http://www.w3.org/1998/Math/MathML">
<ci>selected</ci></math></computeChange>'''
    text = SEDML.replace('</listOfChanges></model>', change+'</listOfChanges></model>')
    with pytest.raises(ValueError, match="explicit model targets"):
        resolve_models(document(text), {"model.xml": model()})
