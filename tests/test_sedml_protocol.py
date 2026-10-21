"""Experiment compilation yields native declarations without executing a session."""

import json

import pytest

from chemworld.data.combine_archive import COMBINE_URI, ArchiveEntry, CombineArchive
from chemworld.world.sedml_protocol import SEDMLProtocol
from reference.run_sedml_initialization_diagnostic import model_xml
from test_sbml import model
from test_sedml_declarations import SEDML, document


def protocol(text=SEDML):
    return SEDMLProtocol(document(text), {"model.xml": model()})


def test_whole_experiment_compiles_to_existing_world_and_parameter_actions_without_live_state():
    plan = protocol()
    assert set(plan.worlds) == {"m1"}
    world = plan.worlds["m1"]
    declaration = world.physics.models_of("sbml")["formal_workspace"]
    assert declaration.controlled_parameters == ("k",)
    assert declaration.frame._initial["k"] == .4
    assert declaration.solver_method == "BDF" and declaration.maximum_step_s == .02
    assert "set_model_parameters" in world.task.operations
    assert plan.tasks.invocations["scan"] == 3
    payload = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    restored = SEDMLProtocol.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.worlds["m1"].to_dict() == world.to_dict()
    with pytest.raises(TypeError):
        plan.worlds["m1"] = None


def test_source_species_selectors_become_native_gauge_declarations():
    text = SEDML.replace('symbol="urn:sedml:symbol:time"',
        '''target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']" symbol="KISAO:0000836"''')
    plan = protocol(text)
    spec = plan.worlds["m1"].physics.models_of("sbml")["formal_workspace"]
    assert [(o.kind, o.source_id) for o in spec.observables] == [("amount", "A")]
    assert spec.readout_relative_sd == 0.


def test_archive_relative_parent_reference_preserves_original_sedml_and_source_provenance():
    source = model().source
    text = SEDML.replace('source="model.xml"', 'source="../models/model.xml"')
    original = CombineArchive((
        ArchiveEntry("models/model.xml", COMBINE_URI+"sbml", model().to_xml().encode()),
        ArchiveEntry("experiments/scan.sedml", COMBINE_URI+"sed-ml", text.encode(), True)), source)
    archive = CombineArchive.from_bytes(original.to_bytes(), source=source)
    plan = SEDMLProtocol.from_archive(archive)
    assert plan.document.to_xml() == text
    assert plan.document_location == "experiments/scan.sedml"
    assert plan.models["m0"].source.provenance.checksum
    assert plan.models["m0"].to_xml() == model().to_xml()
    assert SEDMLProtocol.from_dict(plan.to_dict()).worlds["m1"].to_dict() == (
        plan.worlds["m1"].to_dict())


@pytest.mark.parametrize("old,new", [
    ("parameter[@id='k']/@value", "parameter[@id='missing']/@value"),
    ('<setValue modelReference="m1" range="grid"',
     '<setValue modelReference="m1" range="missing"'),
    ('outputEndTime="4"', 'outputEndTime="0.000001"'),
    ('outputEndTime="4"', 'outputEndTime="200000"'),
    ('KISAO:0000288', 'KISAO:0000019'),
])
def test_unknown_target_algorithm_range_or_unadmitted_wait_rejects_whole_protocol(old, new):
    assert old in SEDML
    with pytest.raises(ValueError):
        protocol(SEDML.replace(old, new))


def test_configuration_decoder_rejects_unknown_execution_fields():
    payload = protocol().to_dict()
    payload["execute"] = "arbitrary code"
    with pytest.raises(ValueError):
        SEDMLProtocol.from_dict(payload)


def test_reset_recipe_initializes_constraints_after_all_new_xml_inputs_are_installed():
    constraint = '''<listOfConstraints><constraint><math xmlns="http://www.w3.org/1998/Math/MathML">
<apply><eq/><ci>A</ci><apply><times/><cn>2</cn><ci>k</ci></apply></apply>
</math></constraint></listOfConstraints>'''
    source = model(model_xml("I").replace('</model>', constraint+'</model>'))
    plan = SEDMLProtocol(document(), {"model.xml": source})
    world = plan.world_with_parameters("m1", {"k": 0.})
    frame = world.physics.models_of("sbml")["formal_workspace"].frame
    assert frame.initial_values["N:A"] == 0. and frame.initial_values["P:k"] == 0.


def test_total_channel_value_budget_is_checked_before_any_native_world_recipe(monkeypatch):
    values = ''.join('<value>0.1</value>' for _ in range(100))
    text = SEDML.replace('numberOfSteps="10"', 'numberOfSteps="10000"').replace(
        '<value>0.1</value><value>0.2</value><value>0.4</value>', values).replace(
        'symbol="urn:sedml:symbol:time"',
        '''target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']"''')

    def forbidden(*args, **kwargs):
        pytest.fail("world recipe ran before complete output budget validation")

    monkeypatch.setattr('chemworld.world.sedml_protocol.sbml_world', forbidden)
    with pytest.raises(ValueError, match="output budget"):
        protocol(text)


def test_mixed_xml_and_mathematical_targets_at_one_reset_boundary_are_explicitly_refused():
    change = '''<setValue modelReference="m1" range="grid"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>grid</ci></math></setValue>'''
    text = SEDML.replace('</setValue>\n</listOfChanges>', '</setValue>'+change+'</listOfChanges>')
    with pytest.raises(ValueError, match="cannot mix"):
        protocol(text)


def test_sequential_read_of_a_changed_derived_parameter_is_rejected_before_execution():
    source = model_xml("").replace('id="k" value="2" units="molar"',
        'id="k" value="0.2" units="per_second"').replace('</listOfParameters>',
        '<parameter id="x" units="per_second" constant="false"/></listOfParameters>')
    rules = '''<listOfRules><assignmentRule variable="x"><math
xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>k</ci></apply>
</math></assignmentRule></listOfRules>'''
    source = source.replace('<listOfReactions>', rules+'<listOfReactions>')
    change = '''<setValue modelReference="m1" target="/s:sbml/s:model/s:listOfParameters/
s:parameter[@id='rho']/@value"><listOfVariables><variable id="derived_read" modelReference="m1"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='x']"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>derived_read</ci></math></setValue>'''
    change = change.replace('/\ns:', '/s:')
    text = SEDML.replace('</setValue>\n</listOfChanges>', '</setValue>'+change+'</listOfChanges>')
    with pytest.raises(ValueError, match="candidate observations"):
        SEDMLProtocol(document(text), {"model.xml": model(source)})
