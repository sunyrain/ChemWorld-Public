"""Guard dependencies respect original quantity, lexical and full-reactant semantics."""

from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest

from chemworld.physchem.sbml_integer_boundary_policy import (
    guard_critical_reactions,
    guard_species,
)
from chemworld.physchem.sbml_integer_rules import SBMLIntegerRules
from chemworld.physchem.sbml_jump_triggers import SBMLJumpTriggers
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from chemworld.physchem.sbml_system import SBMLSystemSpec
from test_sbml import model
from test_sbml51k_events import clock, event, parsed
from test_sbml_stochastic51 import CORE, DIRECT, MATH, source

S = "{" + CORE + "}"
M = "{" + MATH + "}"


def compiled(original, guard):
    spec = SBMLSystemSpec(parsed(original=original, events=event(
        guard=guard, values=(("k", "<cn>0</cn>"),))),
        numerical_form="stochastic", solver_method=DIRECT)
    return spec.roots, spec.integer_kinetics.compile(spec.frame.initial_values, 0)


@pytest.mark.parametrize(("guard", "species", "critical"), (
    ("<apply><lt/><ci>A</ci><cn>79.5</cn></apply>", {"A"}, {"r"}),
    ("<apply><gt/><ci>B</ci><cn>0</cn></apply>", {"B"}, {"r"}),
    ("<apply><eq/><ci>C</ci><cn>3</cn></apply>", {"C"}, set()),
    (clock(0.5), set(), set()),
    ("<apply><gt/><ci>k</ci><cn>0</cn></apply>", set(), set()),
    ("<apply><gt/><ci>c</ci><cn>0</cn></apply>", set(), set()),
    ("<apply><and/>" + clock(0.5)
     + "<apply><gt/><ci>B</ci><cn>0</cn></apply></apply>", {"B"}, {"r"}),
))
def test_direct_and_time_guards(guard, species, critical):
    triggers, channels = compiled(source(), guard)
    assert guard_species(triggers) == frozenset(species)
    assert guard_critical_reactions(triggers, channels) == frozenset(critical)


@pytest.mark.parametrize(("name", "critical"), (
    ("A", {"r"}), ("B", {"r", "second"}), ("C", {"second"}),
))
def test_only_channels_that_change_guard_stocks_are_selected(name, critical):
    original = source()
    tree = ET.fromstring(original.to_xml())
    reactions = tree.find(".//" + S + "listOfReactions")
    second = deepcopy(reactions[0])
    second.set("id", "second")
    second.find(S + "listOfReactants")[0].set("species", "B")
    second.find(S + "listOfProducts")[0].set("species", "C")
    law = second.find(S + "kineticLaw")
    law.find(".//" + M + "ci[.='A']").text = "B"
    reactions.append(second)
    original = model(ET.tostring(tree, encoding="unicode"),
                     items_per_mole=original.items_per_mole)
    triggers, channels = compiled(original, "<apply><gt/><ci>" + name + "</ci><cn>0</cn></apply>")
    assert guard_critical_reactions(triggers, channels) == frozenset(critical)


def test_regenerated_catalyst_guard_does_not_make_the_channel_critical():
    triggers, channels = compiled(source(
        reactants={"A": 1, "B": 1}, products={"A": 1, "C": 1},
        counts={"A": 3, "B": 5, "C": 0}), "<apply><gt/><ci>A</ci><cn>0</cn></apply>")
    assert dict(channels[0].reactants) == {"A": 1, "B": 1}
    assert guard_species(triggers) == {"A"}
    assert guard_critical_reactions(triggers, channels) == frozenset()


def test_assignment_alias_is_expanded_before_inventory_dependencies():
    original = source()
    tree = ET.fromstring(original.to_xml())
    declaration = tree.find(S + "model")
    ET.SubElement(declaration.find(S + "listOfParameters"), S + "parameter",
                  id="q", value="0", constant="false")
    original = model(ET.tostring(tree, encoding="unicode"),
                     items_per_mole=original.items_per_mole)
    original = parsed(original=original, rules=(
        '<assignmentRule variable="q"><math xmlns="' + MATH + '">'
        '<ci>B</ci></math></assignmentRule>'))
    triggers, channels = compiled(original, "<apply><gt/><ci>q</ci><cn>0</cn></apply>")
    assert guard_species(triggers) == {"B"}
    assert guard_critical_reactions(triggers, channels) == {"r"}


def test_function_argument_shadows_same_named_global_species():
    original = source()
    tree = ET.fromstring(original.to_xml())
    declaration = tree.find(S + "model")
    functions = ET.fromstring('<listOfFunctionDefinitions xmlns="' + CORE + '">'
        '<functionDefinition id="twice"><math xmlns="' + MATH + '"><lambda>'
        '<bvar><ci>A</ci></bvar><apply><times/><ci>A</ci><cn>2</cn></apply>'
        '</lambda></math></functionDefinition></listOfFunctionDefinitions>')
    declaration.insert(0, functions)
    original = model(ET.tostring(tree, encoding="unicode"),
                     items_per_mole=original.items_per_mole)
    triggers, channels = compiled(original,
        "<apply><gt/><apply><ci>twice</ci><ci>B</ci></apply><cn>0</cn></apply>")
    assert guard_species(triggers) == {"B"}
    assert guard_critical_reactions(triggers, channels) == {"r"}


def test_reaction_rate_dependencies_respect_local_parameter_shadowing():
    original = source(reactants={"C": 1}, products={"B": 1},
        expression="<apply><times/><ci>A</ci><ci>C</ci></apply>")
    tree = ET.fromstring(original.to_xml())
    local = ET.SubElement(tree.find(".//" + S + "kineticLaw"), S + "listOfLocalParameters")
    ET.SubElement(local, S + "localParameter", id="A", value="0.2")
    original = model(ET.tostring(tree, encoding="unicode"),
                     items_per_mole=original.items_per_mole)
    triggers, channels = compiled(original, "<apply><gt/><ci>r</ci><cn>0</cn></apply>")
    assert guard_species(triggers) == {"C"}
    assert guard_critical_reactions(triggers, channels) == {"r"}


def test_source_held_species_has_no_net_channel_update():
    triggers, channels = compiled(source(reactants={"C": 1}, products={"B": 1}, boundary=("C",)),
                                  "<apply><gt/><ci>C</ci><cn>0</cn></apply>")
    assert guard_species(triggers) == {"C"}
    assert guard_critical_reactions(triggers, channels) == frozenset()


def test_empty_compiled_jump_guards_need_no_critical_channels():
    frame = SBMLKinetics(source())
    triggers = SBMLJumpTriggers(frame, SBMLIntegerRules(frame))
    assert guard_species(triggers) == frozenset()
    assert guard_critical_reactions(triggers, ()) == frozenset()


def test_other_guard_profiles_are_rejected_explicitly():
    with pytest.raises(ValueError, match="compiled jump guards"):
        guard_species(object())
