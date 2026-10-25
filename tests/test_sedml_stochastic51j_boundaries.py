"""Same-value old recipes and independently seeded original experiment branches."""

import pytest

from chemworld.eval.sedml_continuation import SEDMLExperimentSession
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol
from test_sbml import model
from test_sedml_declarations import SEDML, document
from test_sedml_stochastic51j import METHODS, native_state, plain, protocol, text


@pytest.mark.parametrize("method", ("KISAO:0000288", "KISAO:0000088"))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_original_deterministic_same_value_reset_recipe_restores_at_every_course(method, noise):
    xml = SEDML.replace("KISAO:0000288", method).replace(
        "<value>0.1</value><value>0.2</value><value>0.4</value>",
        "<value>0.4</value><value>0.4</value><value>0.4</value>",
    )
    plan = SEDMLProtocol(document(xml), {"model.xml": model()}, readout_relative_sd=0.01)
    direct = SEDMLExperimentSession(plan, seed=37, observation_noise_mode=noise)
    direct.advance(max_courses=1)
    resumed = SEDMLExperimentSession.from_dict(plain(direct.to_dict()))
    direct.advance()
    while not resumed.complete:
        resumed.advance(max_courses=1)
        resumed = SEDMLExperimentSession.from_dict(plain(resumed.to_dict()))
    assert plain(resumed.to_dict()) == plain(direct.to_dict())


def second_simulation(xml):
    start = xml.index("<uniformTimeCourse ")
    end = xml.index("</uniformTimeCourse>", start) + len("</uniformTimeCourse>")
    extra = xml[start:end].replace('id="s1"', 'id="s2"')
    extra = extra.replace('value="17"', 'value="37"')
    return xml.replace("</listOfSimulations>", extra + "</listOfSimulations>")


@pytest.mark.parametrize("method", METHODS)
def test_same_model_cannot_be_reused_with_conflicting_process_seeds(method):
    xml = second_simulation(text(method))
    task = '<task id="t2" modelReference="m1" simulationReference="s2"/>'
    xml = xml.replace("</listOfTasks>", task + "</listOfTasks>")
    with pytest.raises(ValueError, match="one origin and numerical policy"):
        protocol(xml=xml)


@pytest.mark.parametrize("method", METHODS)
def test_distinct_model_seeded_branches_resume_the_original_ordered_task_graph(method):
    xml = second_simulation(text(method))
    xml = xml.replace(
        "</listOfModels>",
        '<model id="m2" language="urn:sedml:language:sbml" source="#m0"/></listOfModels>',
    )
    xml = xml.replace(
        "</listOfTasks>",
        '<task id="t2" modelReference="m2" simulationReference="s2"/></listOfTasks>',
    )
    xml = xml.replace(
        '<subTask task="t1"/>', '<subTask task="t1" order="1"/><subTask task="t2" order="2"/>'
    )
    xml = xml.replace(
        'taskReference="scan" target=', 'taskReference="scan" modelReference="m1" target='
    )
    generator = """<dataGenerator id="dg2"><listOfVariables><variable id="a2"
taskReference="scan" modelReference="m2" target="/s:sbml/s:model/s:listOfSpecies/
s:species[@id='A']" symbol="KISAO:0000836"/></listOfVariables><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>a2</ci></math></dataGenerator>""".replace(
        "/\ns:", "/s:"
    )
    xml = xml.replace("</listOfDataGenerators>", generator + "</listOfDataGenerators>")
    xml = xml.replace(
        "</listOfDataSets>",
        '<dataSet id="dataset2" label="A2" dataReference="dg2"/></listOfDataSets>',
    )
    session = SEDMLExperimentSession(protocol(xml=xml), seed=37)
    session.advance(max_courses=1)
    restored = SEDMLExperimentSession.from_dict(plain(session.to_dict()))
    session.advance()
    while not restored.complete:
        restored.advance(max_courses=1)
        restored = SEDMLExperimentSession.from_dict(plain(restored.to_dict()))
    assert plain(restored.to_dict()) == plain(session.to_dict())
    assert session.progress()["completed_courses"] == 6
    assert all(column["data"]["shape"] == [3, 11] for column in session.result().reports["report"])
    snapshots = session.result().snapshots["scan"]
    first, second = (ChemWorldRuntime.from_snapshot(snapshots[k]) for k in ("m1", "m2"))
    assert (
        native_state(first)["continuation"]["process"]["rng_state"]
        != native_state(second)["continuation"]["process"]["rng_state"]
    )
    assert first._state.ledger.time_s == second._state.ledger.time_s == 4
    first.close()
    second.close()
