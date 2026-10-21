"""Original SBML declaration semantics and refusal boundaries, not a trajectory test."""

import json
from dataclasses import FrozenInstanceError

import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance

MODEL = '''<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="transport" timeUnits="second" substanceUnits="mole" extentUnits="mole"
volumeUnits="litre">
<notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Self-authored formal pools;
chemical formulas and reaction energies are unknown.</p></body></notes>
<annotation><source xmlns="urn:chemworld:test" code="not executable"/></annotation>
<listOfFunctionDefinitions><functionDefinition id="f"><math
xmlns="http://www.w3.org/1998/Math/MathML"><lambda><bvar><ci>x</ci></bvar><bvar><ci>y</ci>
</bvar><apply><times/><ci>x</ci><ci>y</ci></apply></lambda></math></functionDefinition>
</listOfFunctionDefinitions>
<listOfUnitDefinitions><unitDefinition id="per_second"><listOfUnits><unit kind="second"
exponent="-1" scale="0" multiplier="1"/></listOfUnits></unitDefinition>
<unitDefinition id="molar_rate"><listOfUnits><unit kind="mole" exponent="1" scale="0"
multiplier="1"/><unit kind="litre" exponent="-1" scale="0" multiplier="1"/>
<unit kind="second" exponent="-1" scale="0" multiplier="1"/></listOfUnits></unitDefinition>
</listOfUnitDefinitions>
<listOfCompartments><compartment id="c1" spatialDimensions="3" size="0.01" units="litre"
constant="false"/><compartment id="c2" spatialDimensions="3" size="0.02" units="litre"
constant="true"/></listOfCompartments>
<listOfSpecies><species id="A" compartment="c1" initialConcentration="1"
hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
<species id="B" compartment="c2" initialAmount="0" hasOnlySubstanceUnits="true"
boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="0.2" units="per_second" constant="true"/>
<parameter id="p" value="2" units="dimensionless" constant="false"/>
<parameter id="rp" value="0.1" units="per_second" constant="true"/>
<parameter id="v" units="molar_rate" constant="false"/></listOfParameters>
<listOfInitialAssignments><initialAssignment symbol="p"><math
xmlns="http://www.w3.org/1998/Math/MathML"><cn>3</cn></math></initialAssignment>
</listOfInitialAssignments>
<listOfRules><assignmentRule variable="v"><math xmlns="http://www.w3.org/1998/Math/MathML">
<apply><ci>f</ci><ci>A</ci><ci>k</ci></apply></math></assignmentRule>
<rateRule variable="p"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>rp</ci></math>
</rateRule></listOfRules>
<listOfConstraints><constraint><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/>
<ci>p</ci><cn>0</cn></apply></math><message><p xmlns="http://www.w3.org/1999/xhtml">p bound
</p></message></constraint></listOfConstraints>
<listOfReactions><reaction id="J" reversible="false"><listOfReactants><speciesReference
species="A" stoichiometry="1" constant="true"/></listOfReactants><listOfProducts>
<speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts><kineticLaw>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c1</ci>
<apply><ci>f</ci><ci>A</ci><ci>k</ci></apply></apply></math><listOfLocalParameters>
<localParameter id="k" value="0.3" units="per_second"/></listOfLocalParameters></kineticLaw>
</reaction></listOfReactions>
<listOfEvents><event id="switch" useValuesFromTriggerTime="true"><trigger initialValue="false"
persistent="false"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol
definitionURL="http://www.sbml.org/sbml/symbols/time" encoding="text">time</csymbol><cn>2</cn>
</apply></math></trigger><delay><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.5</cn>
</math></delay><priority><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>p</ci></math>
</priority><listOfEventAssignments><eventAssignment variable="A"><math
xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math></eventAssignment>
<eventAssignment variable="p"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>5</cn>
</math></eventAssignment></listOfEventAssignments></event></listOfEvents></model></sbml>'''


def model(text=MODEL, **options):
    return SBMLModel(text, InputSource(DataSourceProvenance(
        "synthetic-sbml", 0, "Self-authored declaration fixture", "1", "CC0-1.0"), "synthetic"),
        **options)


def test_complete_declarations_original_xml_json_and_frozen_provenance_roundtrip(tmp_path):
    parsed = model()
    assert parsed.to_xml() == MODEL
    assert len(parsed.declaration.items("listOfSpecies")) == 2
    assert len(parsed.declaration.items("listOfRules")) == 2
    assert len(parsed.declaration.metadata_xml) == 2
    assert len(parsed.declaration.items("listOfConstraints")[0].metadata_xml) == 1
    clone = SBMLModel.from_dict(json.loads(json.dumps(parsed.to_dict())))
    assert clone.to_dict() == parsed.to_dict()
    target = tmp_path/"model.xml"
    target.write_text(MODEL, encoding="utf-8")
    assert SBMLModel.from_file(target, source=parsed.source).to_xml() == MODEL
    event = parsed.declaration.items("listOfEvents")[0]
    assert event.get("useValuesFromTriggerTime") is True
    assert event.of("trigger")[0].get("persistent") is False
    assert event.of("delay")[0].mathematics.evaluate({}) == .5
    assert parsed.model_unit("volumeUnits").volume_l(.02) == .02
    assert parsed.unit_signature("molar_rate").si_factor == 1000
    with pytest.raises(FrozenInstanceError):
        parsed.raw_text = "modified"
    with pytest.raises(TypeError):
        parsed.unit_defaults["timeUnits"] = "second"


def test_local_parameters_shadow_global_values_without_changing_source_records():
    parsed = model()
    reaction = parsed.declaration.items("listOfReactions")[0]
    law = reaction.of("kineticLaw")[0]
    functions = {r.get("id"): r.mathematics
                 for r in parsed.declaration.items("listOfFunctionDefinitions")}
    assert law.items("listOfLocalParameters")[0].get("value") == .3
    scope = {"c1": .01, "A": 1, "k": .3}
    assert law.mathematics.evaluate(scope, functions=functions) == .003
    assert parsed.declaration.items("listOfParameters")[0].get("value") == .2


@pytest.mark.parametrize(("before", "after", "message"), [
    ('version="2"', 'version="1"', "Level 3 Version 2"),
    ('<ci>rp</ci>', '<ci>unknown</ci>', "undefined.*symbol"),
    ('<ci>f</ci><ci>A</ci><ci>k</ci>', '<ci>missing</ci><ci>A</ci><ci>k</ci>',
     "undefined.*function"),
    ('<ci>f</ci><ci>A</ci><ci>k</ci>', '<ci>f</ci><ci>A</ci>', "argument count"),
    ('variable="A"', 'variable="v"', "rule-owned"),
    ('symbol="p"', 'symbol="v"', "conflicts"),
    ('<ci>A</ci><ci>k</ci></apply></math></assignmentRule>',
     '<ci>v</ci><ci>k</ci></apply></math></assignmentRule>', "cyclic"),
    ('units="per_second"', 'units="unknown"', "undefined.*unit"),
    ('species="A" stoichiometry="1" constant="true"',
     'species="A" stoichiometry="1" constant="false"', "dynamic"),
    ('reversible="false"', 'reversible="false" fast="true"', "fast reaction"),
    ('initialConcentration="1"', 'initialConcentration="1" initialAmount="1"', "both initial"),
    ('id="B" compartment="c2"', 'id="A" compartment="c2"', "duplicate.*global"),
    ('species="B"', 'species="missing"', "undefined.*species"),
    ('compartment="c2" initialAmount', 'compartment="missing" initialAmount',
     "undefined.*compartment"),
    ('constant="false"/></listOfParameters>', 'constant="true"/></listOfParameters>', "constant"),
    ('value="0.2"', 'value="NaN"', "invalid SBML attribute"),
    ('<model id="transport"', '<model id="transport" arbitrary="unknown"', "unknown.*attributes"),
])
def test_unknown_conflicting_or_unsupported_model_rejects(before, after, message):
    assert before in MODEL
    with pytest.raises(ValueError, match=message):
        model(MODEL.replace(before, after))


def test_algebraic_rule_is_refused_without_losing_well_formed_xml():
    text = MODEL.replace('rateRule variable="p"', 'algebraicRule').replace(
        '</rateRule>', '</algebraicRule>')
    with pytest.raises(ValueError, match=r"unsupported.*child"):
        model(text)


def test_missing_source_unit_requires_explicit_author_policy_and_unknown_json_is_not_loaded():
    text = MODEL.replace(' timeUnits="second"', '')
    parsed = model(text)
    with pytest.raises(ValueError, match="explicit author default"):
        parsed.model_unit("timeUnits")
    assert model(text, unit_defaults={"timeUnits": "second"}).model_unit("timeUnits").time_s(3) == 3
    payload = parsed.to_dict()
    payload["loader"] = "arbitrary.module"
    with pytest.raises(ValueError, match="unknown SBML"):
        SBMLModel.from_dict(payload)


@pytest.mark.parametrize("invalid", ["0_2", "\uff10.\uff12", "1e99999"])
def test_xml_number_attributes_do_not_use_python_extended_float_syntax(invalid):
    with pytest.raises(ValueError, match="invalid SBML attribute"):
        model(MODEL.replace('value="0.2"', f'value="{invalid}"'))
