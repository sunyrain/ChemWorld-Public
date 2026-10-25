"""Original SBML units, formal owner, integer protocols and refusal boundaries."""

import copy
import json
import math
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation.discrete_inventory import AVOGADRO, mol_to_counts
from chemworld.physchem.sbml_stochastic import ALGORITHMS, initial_memory
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.sbml_world import sbml_world
from test_sbml import model

CORE = "http://www.sbml.org/sbml/level3/version2/core"
MATH = "http://www.w3.org/1998/Math/MathML"
DIRECT = "gillespie-direct-v1"
TAU = "adaptive-explicit-tau-leap-v1"


def source(
    reactants=None,
    products=None,
    *,
    counts=None,
    units="item",
    time="second",
    concentration=False,
    boundary=(),
    modifiers=(),
    expression=None,
    coefficient=0.2,
):
    reactants = {"A": 1} if reactants is None else reactants
    products = {"B": 1} if products is None else products
    counts = {"A": 80, "B": 0, "C": 3} if counts is None else counts
    root = ET.Element("sbml", {"xmlns": CORE, "level": "3", "version": "2"})
    declaration = ET.SubElement(
        root,
        "model",
        {
            "id": "integer_source",
            "timeUnits": time,
            "substanceUnits": units,
            "extentUnits": units,
            "volumeUnits": "litre",
        },
    )
    definitions = ET.SubElement(declaration, "listOfUnitDefinitions")
    for identity, kind, scale, multiplier in (
        ("mmol", "mole", -3, "1"),
        ("minute", "second", 0, "60"),
    ):
        definition = ET.SubElement(definitions, "unitDefinition", {"id": identity})
        group = ET.SubElement(definition, "listOfUnits")
        ET.SubElement(
            group,
            "unit",
            {"kind": kind, "exponent": "1", "scale": str(scale), "multiplier": multiplier},
        )
    group = ET.SubElement(declaration, "listOfCompartments")
    ET.SubElement(group, "compartment", {"id": "c", "constant": "true", "size": "1e-15"})
    ET.SubElement(group, "compartment", {"id": "d", "constant": "true", "size": "2e-15"})
    group = ET.SubElement(declaration, "listOfSpecies")
    scale = 1 / AVOGADRO if units == "item" else 1.0 if units == "mole" else 0.001
    for name, n in counts.items():
        ET.SubElement(
            group,
            "species",
            {
                "id": name,
                "compartment": "d" if name == "B" else "c",
                "initialAmount": repr(n / AVOGADRO / scale),
                "hasOnlySubstanceUnits": str(not concentration).lower(),
                "boundaryCondition": str(name in boundary).lower(),
                "constant": "false",
            },
        )
    group = ET.SubElement(declaration, "listOfParameters")
    ET.SubElement(group, "parameter", {"id": "k", "value": repr(coefficient), "constant": "true"})
    group = ET.SubElement(declaration, "listOfReactions")
    reaction = ET.SubElement(group, "reaction", {"id": "r", "reversible": "false"})
    for label, values in (("listOfReactants", reactants), ("listOfProducts", products)):
        if values:
            group = ET.SubElement(reaction, label)
            for name, n in values.items():
                ET.SubElement(
                    group,
                    "speciesReference",
                    {"species": name, "stoichiometry": str(n), "constant": "true"},
                )
    if modifiers:
        group = ET.SubElement(reaction, "listOfModifiers")
        for name in modifiers:
            ET.SubElement(group, "modifierSpeciesReference", {"species": name})
    law = ET.SubElement(reaction, "kineticLaw")
    if expression is None:
        powers = {**reactants, **dict.fromkeys(modifiers, 1)}
        factors = "<ci>k</ci>" + "".join(
            ("<ci>" + name + "</ci>") * n for name, n in powers.items()
        )
        if concentration:
            factors = "<ci>c</ci>" + factors
        expression = "<apply><times/>" + factors + "</apply>"
    law.append(ET.fromstring('<math xmlns="' + MATH + '">' + expression + "</math>"))
    return model(ET.tostring(root, encoding="unicode"), items_per_mole=AVOGADRO)


def spec(parsed=None, method=DIRECT, **kwargs):
    return SBMLSystemSpec(
        source() if parsed is None else parsed,
        solver_method=method,
        numerical_form="stochastic",
        tau_policy=TauLeapPolicy(max_tau_s=0.01, ssa_threshold=0.01) if method == TAU else None,
        **kwargs,
    )


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def owned(session):
    return normalize(model_record(session._state).settings)


def execute(session, action):
    info = (
        session.execute(action).info
        if isinstance(session, ChemWorldRuntime)
        else session.step(action)[4]
    )
    assert info["transaction_status"] == "committed", info
    return normalize({"state": session._state.to_dict(), "info": info})


@pytest.mark.parametrize(
    "units,time,concentration,expected",
    (
        ("item", "second", False, 0.2),
        ("mole", "second", False, 0.2),
        ("mmol", "minute", False, 0.2 / 60),
        ("item", "second", True, 0.2),
        ("mmol", "minute", True, 0.2 / 60),
    ),
)
def test_source_amount_time_volume_units_without_fake_formulas(
    units, time, concentration, expected
):
    declaration = spec(source(units=units, time=time, concentration=concentration))
    integer = declaration.integer_kinetics
    counts = integer.counts(declaration.frame.initial_values, 0)
    assert counts == {"A": 80, "B": 0, "C": 3}
    (channel,) = integer.compile(declaration.frame.initial_values, 0)
    assert channel.propensity(counts) == pytest.approx(80 * expected, rel=2e-14)
    assert dict(channel.changes) == {"A": -1, "B": 1}
    assert declaration.model.to_xml() == integer.frame.model.to_xml()


@pytest.mark.parametrize(
    "reactants,products,modifiers,boundary",
    (
        ({"A": 1, "B": 1}, {"A": 2}, (), ()),
        ({"A": 2}, {"A": 1, "B": 1}, (), ()),
        ({"A": 2, "B": 1}, {"A": 3}, (), ()),
        ({"A": 1, "B": 2}, {"A": 2, "B": 1}, (), ()),
        ({"A": 1, "B": 1}, {"A": 1, "C": 1}, (), ()),
        ({"A": 1}, {"B": 1}, ("C",), ()),
        ({"A": 1, "C": 1}, {"B": 1}, (), ("C",)),
        ({}, {"A": 1}, (), ()),
    ),
)
def test_full_molecular_propensities_held_boundary_and_zero_order(
    reactants, products, modifiers, boundary
):
    initial = {"A": 20, "B": 10, "C": 3}
    declaration = spec(
        source(reactants, products, counts=initial, modifiers=modifiers, boundary=boundary)
    )
    (channel,) = declaration.integer_kinetics.compile(declaration.frame.initial_values, 0)
    orders = {**reactants, **dict.fromkeys(modifiers, 1)}
    expected = 0.2 * math.prod(
        math.prod(initial[k] - j for j in range(n)) for k, n in orders.items()
    )
    assert channel.propensity(initial) == pytest.approx(expected, rel=2e-14)
    assert dict(channel.reactants) == orders
    for name in boundary:
        assert name not in dict(channel.changes)
    for name, n in orders.items():
        assert channel.propensity({**initial, name: n - 1}) == 0


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_zero_duration_and_split_observation_preserve_complete_numerical_path(method):
    declaration = spec(
        source({"A": 1, "B": 1}, {"A": 2}, counts={"A": 20, "B": 180, "C": 0}, coefficient=0.002),
        method,
    )
    integer, values = declaration.integer_kinetics, declaration.frame.initial_values
    args = (method, 100000, declaration.tau_policy)
    memory = initial_memory(17, method)
    zero_values, zero_memory, zero = integer.advance(values, memory, 0, 0, *args)
    assert (
        zero.work["random_draws"] == 0
        and zero_memory["process"]["rng_state"] == memory["process"]["rng_state"]
    )
    whole = integer.advance(zero_values, zero_memory, 0, 2, *args)
    a = integer.advance(zero_values, zero_memory, 0, 0.125, *args)
    b = integer.advance(a[0], a[1], 0.125, 1.875, *args)
    assert b[:2] == whole[:2]
    assert sum(integer.counts(b[0], 2).values()) == 200


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("entry", ("native", "gym"))
def test_original_world_control_meter_replay_json_and_new_process(method, noise, entry, tmp_path):
    cls = ChemWorldRuntime if entry == "native" else ChemWorldEnv
    declaration = spec(method=method, controlled_parameters=("k",))
    composition = sbml_world(
        declaration.model,
        numerical_form="stochastic",
        solver_method=method,
        tau_policy=declaration.tau_policy,
        seed=17,
        controlled_parameters=("k",),
        budget=100,
        observables=(
            SBMLObservable("amount", "A"),
            SBMLObservable("concentration", "B"),
            SBMLObservable("volume", "c"),
            SBMLObservable("reaction_rate", "r"),
        ),
    )
    session = cls(composition=composition, observation_noise_mode=noise)
    session.reset(seed=7)
    initial = normalize(session.snapshot().to_dict())
    actions = (
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        {"operation": "set_model_parameters", "values": {"k": 0.35}},
        {"operation": "wait", "duration_s": 0.875},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
    )
    results = []
    try:
        execute(session, actions[0])
        prefix = owned(session)
        results.append(execute(session, actions[1]))
        assert owned(session) == prefix  # A detector draw is not a process draw.
        before_clock = session._state.ledger.time_s
        results.append(execute(session, actions[2]))
        assert session._state.ledger.time_s == before_clock
        assert owned(session)["coordinates"]["N:A"] == prefix["coordinates"]["N:A"]
        assert owned(session)["continuation"]["process"]["pending"] is None
        checkpoint = normalize(session.snapshot().to_dict())
        restored = cls.from_snapshot(checkpoint)
        try:
            suffix = [execute(session, action) for action in actions[3:]]
            assert [execute(restored, action) for action in actions[3:]] == suffix
        finally:
            restored.close()
        assert session._state.ledger.time_s == 1
        assert owned(session)["continuation"]["process"]["time_s"] == 1
        assert session._state.species_amounts == {} and session._state.volume_L == 0
        counts = mol_to_counts(
            {k[2:]: v for k, v in owned(session)["coordinates"].items() if k.startswith("N:")}
        )
        assert sum(counts.values()) == 83
        replay = cls.from_snapshot(initial)
        try:
            execute(replay, actions[0])
            assert [execute(replay, action) for action in actions[1:3]] == results
            assert [execute(replay, action) for action in actions[3:]] == suffix
        finally:
            replay.close()
        input_path, output_path = tmp_path / "checkpoint.json", tmp_path / "actual.json"
        input_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        program = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')));"
            "rows=[];"
            'exec("for a in json.loads(sys.argv[3]):\\n v=r.execute(a);'
            "rows.append({'state':r._state.to_dict(),'info':v.info})\");"
            "Path(sys.argv[2]).write_text(json.dumps(rows),encoding='utf-8');r.close()"
        )
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                "utf8",
                "-c",
                program,
                str(input_path),
                str(output_path),
                json.dumps(actions[3:]),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert json.loads(output_path.read_text(encoding="utf-8")) == suffix
    finally:
        session.close()


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_bad_input_and_budget_failure_roll_back_original_owner_and_clock(method):
    r = ChemWorldRuntime(
        composition=sbml_world(
            source(coefficient=10),
            numerical_form="stochastic",
            solver_method=method,
            maximum_events=1,
            seed=17,
            controlled_parameters=("k",),
            tau_policy=TauLeapPolicy(max_tau_s=0.01) if method == TAU else None,
        )
    )
    r.reset(seed=1)
    try:
        before = owned(r)
        for action in (
            {"operation": "set_model_parameters", "values": {"k": -1}},
            {"operation": "wait", "duration_s": 10},
        ):
            info = r.execute(action).info
            assert info["transaction_status"] != "committed"
            assert owned(r) == before and r._state.ledger.time_s == 0
    finally:
        r.close()


@pytest.mark.parametrize(
    "expression",
    (
        "<apply><minus/><apply><times/><ci>k</ci><ci>A</ci></apply><ci>B</ci></apply>",
        "<apply><divide/><ci>A</ci><apply><plus/><cn>1</cn><ci>A</ci></apply></apply>",
        "<apply><power/><ci>A</ci><cn>0.5</cn></apply>",
        '<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>',
    ),
)
def test_unknown_net_saturating_fractional_or_time_rates_rejected(expression):
    with pytest.raises(ValueError):
        spec(source(expression=expression))


@pytest.mark.parametrize(
    "change", ("noninteger", "order4", "reversible", "event", "rule", "constraint")
)
def test_source_outside_integer_admission_rejected(change):
    parsed = source()
    tree = ET.fromstring(parsed.to_xml())
    declaration = tree[0]
    if change == "noninteger":
        next(tree.iter("{" + CORE + "}species")).set("initialAmount", "0.5")
    elif change == "order4":
        parsed = source({"A": 4}, {"B": 4})
        tree = ET.fromstring(parsed.to_xml())
    elif change == "reversible":
        next(tree.iter("{" + CORE + "}reaction")).set("reversible", "true")
    else:
        elements = {
            "event": '<listOfEvents><event useValuesFromTriggerTime="true">'
            '<trigger initialValue="false" '
            'persistent="true"><math xmlns="'
            + MATH
            + '"><apply><eq/><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
            '<cn>1</cn></apply></math></trigger></event></listOfEvents>',
            "rule": '<listOfRules><assignmentRule variable="A"><math xmlns="'
            + MATH
            + '"><cn>2</cn>'
            "</math></assignmentRule></listOfRules>",
            "constraint": '<listOfConstraints><constraint><math xmlns="' + MATH
            + '"><apply><geq/><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
            '<cn>0</cn></apply></math>'
            "</constraint></listOfConstraints>",
        }
        declaration.append(
            ET.fromstring('<wrap xmlns="' + CORE + '">' + elements[change] + "</wrap>")[0]
        )
    with pytest.raises(ValueError):
        spec(model(ET.tostring(tree, encoding="unicode"), items_per_mole=AVOGADRO))


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_v3_json_rejects_old_mode_codec_and_corrupted_state(method):
    declaration = spec(method=method)
    payload = normalize(declaration.to_dict())
    assert payload["format"] == "chemworld_sbml_system_v3"
    assert SBMLSystemSpec.from_dict(payload).to_dict() == payload
    payload["format"] = "chemworld_sbml_system_v2"
    payload.pop("tau_policy")
    with pytest.raises(ValueError):
        SBMLSystemSpec.from_dict(payload)
    r = ChemWorldRuntime(
        composition=sbml_world(
            declaration.model,
            numerical_form="stochastic",
            solver_method=method,
            tau_policy=declaration.tau_policy,
        )
    )
    r.reset(seed=1)
    try:
        execute(r, {"operation": "wait", "duration_s": 0.125})
        original = model_record(r._state)
        for key, value in (("signature", "different-source"), ("time_s", 123), ("extra", 1)):
            payload = copy.deepcopy(original.settings)
            payload["continuation"]["process"][key] = value
            from chemworld.foundation import upsert_equipment_record
            from chemworld.runtime.sbml_models import validate_model_state

            bad = r._state.replace(
                equipment=upsert_equipment_record(
                    r._state.equipment,
                    equipment_id=original.equipment_id,
                    equipment_type=original.equipment_type,
                    attached_vessel_id=r._state.vessel_id,
                    settings=payload,
                    batch_reset="preserve",
                    private_settings=("coordinates", "continuation"),
                )
            )
            with pytest.raises(ValueError):
                validate_model_state(bad, declaration)
    finally:
        r.close()


def test_lexical_function_local_parameter_shadowing_and_source_rate_readout():
    parsed = source({"A": 2}, {"A": 1, "B": 1})
    tree = ET.fromstring(parsed.to_xml())
    function = '<listOfFunctionDefinitions xmlns="' + CORE + '"><functionDefinition id="f">'
    function += '<math xmlns="' + MATH + '"><lambda><bvar><ci>x</ci></bvar><bvar><ci>y</ci>'
    function += "</bvar><apply><times/><ci>x</ci><apply><power/><ci>y</ci><cn>2</cn>"
    function += "</apply></apply></lambda></math></functionDefinition></listOfFunctionDefinitions>"
    # Add the function before parsing a model which calls it.
    tree[0].insert(0, ET.fromstring(function))
    law = next(tree.iter("{" + CORE + "}kineticLaw"))
    law.remove(law[0])
    law.append(
        ET.fromstring(
            '<math xmlns="' + MATH + '"><apply><ci>f</ci><ci>k</ci><ci>A</ci></apply></math>'
        )
    )
    law.append(
        ET.fromstring(
            '<listOfLocalParameters xmlns="' + CORE + '">'
            '<localParameter id="k" value="0.7"/></listOfLocalParameters>'
        )
    )
    parsed = model(ET.tostring(tree, encoding="unicode"), items_per_mole=AVOGADRO)
    declaration = spec(
        parsed, controlled_parameters=("k",), observables=(SBMLObservable("reaction_rate", "r"),)
    )
    values = declaration.frame.controlled_values(declaration.frame.initial_values, 0, {"k": 10})
    (channel,) = declaration.integer_kinetics.compile(values, 0)
    assert channel.propensity({"A": 80, "B": 0, "C": 3}) == pytest.approx(0.7 * 80 * 79)
    # The public source-law readout retains deterministic extent units. It is
    # deliberately distinct from the falling-factorial stochastic propensity.
    assert declaration.readout(values, 0)["reaction_rate_r"] == pytest.approx(
        0.7 * 80**2 / AVOGADRO
    )


def test_cubic_metre_compartments_and_initial_concentrations_keep_source_basis():
    tree = ET.fromstring(source(concentration=True).to_xml())
    definition = ET.fromstring(
        '<unitDefinition xmlns="' + CORE + '" id="cubic_metre">'
        '<listOfUnits><unit kind="metre" exponent="3" scale="0" multiplier="1"/>'
        "</listOfUnits></unitDefinition>"
    )
    next(tree.iter("{" + CORE + "}listOfUnitDefinitions")).append(definition)
    tree[0].set("volumeUnits", "cubic_metre")
    for c in tree.iter("{" + CORE + "}compartment"):
        c.set("size", "1e-18" if c.get("id") == "c" else "2e-18")
    for s in tree.iter("{" + CORE + "}species"):
        volume = 2e-18 if s.get("id") == "B" else 1e-18
        s.set("initialConcentration", repr(float(s.attrib.pop("initialAmount")) / volume))
    declaration = spec(model(ET.tostring(tree, encoding="unicode"), items_per_mole=AVOGADRO))
    values = declaration.frame.initial_values
    assert values["V:c"] == pytest.approx(1e-15)
    assert declaration.integer_kinetics.counts(values, 0) == {"A": 80, "B": 0, "C": 3}
    (channel,) = declaration.integer_kinetics.compile(values, 0)
    assert channel.propensity({"A": 80, "B": 0, "C": 3}) == pytest.approx(16)


def test_explicit_conversion_factors_define_integer_event_quantum():
    tree = ET.fromstring(source().to_xml())
    tree[0].set("conversionFactor", "conversion")
    next(tree.iter("{" + CORE + "}listOfParameters")).append(
        ET.fromstring('<parameter xmlns="' + CORE + '" id="conversion" value="2" constant="true"/>')
    )
    for r in tree.iter("{" + CORE + "}speciesReference"):
        r.set("stoichiometry", "0.5")
    parsed = model(ET.tostring(tree, encoding="unicode"), items_per_mole=AVOGADRO)
    declaration = spec(parsed)
    (channel,) = declaration.integer_kinetics.compile(declaration.frame.initial_values, 0)
    assert dict(channel.changes) == {"A": -1, "B": 1}
    assert dict(channel.reactants) == {"A": 1}
    assert channel.propensity({"A": 80, "B": 0, "C": 3}) == pytest.approx(16)
    with pytest.raises(ValueError, match="conversion factors"):
        spec(parsed, controlled_parameters=("conversion",))


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_two_original_formal_workspaces_advance_under_one_lab_clock(method):
    def make(seed):
        return sbml_world(
            source(),
            solver_method=method,
            numerical_form="stochastic",
            seed=seed,
            tau_policy=TauLeapPolicy(max_tau_s=0.01) if method == TAU else None,
        ).to_dict()

    payload = make(17)

    # Relabel one complete authored vessel declaration, including its unique
    # owner and binding, before composing it with the first native workspace.
    def rename(value):
        names = {"formal_workspace": "formal_other", "formal_model": "other_model"}
        if isinstance(value, str):
            return names.get(value, value)
        if isinstance(value, dict):
            return {names.get(k, k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return value

    other = rename(make(29))
    for component in other["components"]:
        component["id"] = "other_" + component["id"]
    payload["vessels"].extend(other["vessels"])
    payload["components"].extend(other["components"])
    for field in ("initial_states", "advance_bindings"):
        payload["physics"][field].update(other["physics"][field])
    payload["physics"]["models"]["sbml"]["bindings"].update(
        other["physics"]["models"]["sbml"]["bindings"]
    )
    r = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    r.reset(seed=7)
    try:
        execute(r, {"operation": "wait", "duration_s": 0.5})
        states = (r._state, *r._state.inactive_vessels.values())
        assert {s.vessel_id for s in states} == {"formal_workspace", "formal_other"}
        for state in states:
            assert state.ledger.time_s == 0.5
            assert model_record(state).settings["continuation"]["process"]["time_s"] == 0.5
            assert state.species_amounts == {} and state.process.stochastic_continuation is None
    finally:
        r.close()


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
@pytest.mark.parametrize("mutation", ("pending_extra", "rng_extra", "boolean_clock"))
def test_closed_pending_and_process_rng_snapshot_refusals(method, mutation):
    declaration = spec(method=method)
    integer = declaration.integer_kinetics
    values, memory, _ = integer.advance(
        declaration.frame.initial_values,
        initial_memory(17, method),
        0,
        1.0,
        method,
        10000,
        declaration.tau_policy,
    )
    assert memory["process"]["pending"] is not None
    if mutation == "pending_extra":
        memory["process"]["pending"]["unknown"] = 1
    elif mutation == "rng_extra":
        memory["process"]["rng_state"]["unknown"] = 1
    else:
        memory["process"]["time_s"] = True
    with pytest.raises(ValueError):
        integer.advance(values, memory, 1.0, 0, method, 10000, declaration.tau_policy)


@pytest.mark.parametrize("method", sorted(ALGORITHMS))
def test_detector_draws_do_not_change_process_path_and_boundary_pool_is_held(method):
    declaration = spec(source({"A": 1, "C": 1}, {"B": 1}, boundary=("C",)), method)
    composition = sbml_world(
        declaration.model,
        numerical_form="stochastic",
        solver_method=method,
        tau_policy=declaration.tau_policy,
        seed=17,
        budget=100,
        observables=(SBMLObservable("amount", "A"),),
    )
    a, b = (ChemWorldRuntime(composition=composition) for _ in range(2))
    a.reset(seed=7)
    b.reset(seed=99)
    try:
        execute(a, {"operation": "measure", "instrument": "formal_quantity_meter"})
        for session in (a, b):
            execute(session, {"operation": "wait", "duration_s": 1})
        assert owned(a) == owned(b)
        counts = declaration.integer_kinetics.counts(owned(a)["coordinates"], 1)
        assert counts["C"] == 3 and counts["A"] + counts["B"] == 80
    finally:
        a.close()
        b.close()
