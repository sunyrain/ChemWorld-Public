"""Real finite CEF material, shared heater, conservation and atomic rollback."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_material import CefMaterialSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.solid_solutions import solid_enthalpy, solid_model, validate_solid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.solid_world import calphad_world_spec


def material(elements=("FE", "NI"), *, phase_type="solid", sublattices=1):
    symbols = {"FE": "BCC_A2", "NI": "FCC_A1", "CR": "BCC_A2"}
    text = "\n".join(f"ELEMENT {e} {symbols[e]} 55 0 0 !" for e in elements)
    text += "\nTYPE_DEF % SEQ * !\n"
    text += f"PHASE ALPHA % {sublattices} " + " ".join(["1"]*sublattices) + " !\n"
    text += "CONSTITUENT ALPHA :"+ ":".join([",".join(elements)]*sublattices) + ": !\n"
    from itertools import product

    # Every endmember gives exactly 30 J/(mol atoms K); entropy changes no H.
    for entry in product(elements, repeat=sublattices):
        text += (f"PARAMETER G(ALPHA,{':'.join(entry)};0) 100 "
                 f"-{30*sublattices}*T*LN(T); 2000 N !\n")
    source = InputSource(DataSourceProvenance("cef-material-test", 0, "Analytic material", "1",
                                            "CC0"), "synthetic")
    phase = CefPhaseModel(TdbDatabase.from_string(text, source=source), "ALPHA")
    return CefMaterialSpec(phase, tuple({e: 1/len(elements) for e in elements}
                                      for _ in range(sublattices)),
                           500, 900, 600, .01, phase_type=phase_type)


def session(model=None):
    model = model or material()
    amounts = {e: .001/len(model.component_ids) for e in model.component_ids}
    composition = calphad_world_spec(model, element_amounts_mol=amounts, temperature_K=600)
    runtime = ChemWorldRuntime(composition=composition)
    runtime.reset(seed=40)
    return runtime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def physical_state(state):
    payload = state.to_dict()
    # Rejected/numerically failed operations still charge the declared attempt.
    return {k: v for k, v in payload.items() if k not in {"ledger", "process"}}


@pytest.mark.parametrize("elements,sublattices,phase_type", [
    (("FE", "NI"), 1, "solid"), (("FE", "NI", "CR"), 1, "solid"),
    (("FE", "NI"), 2, "solid"), (("FE", "NI"), 1, "liquid"),
])
def test_actual_elements_constitution_and_caloric_roundtrip(elements, sublattices, phase_type):
    model = material(elements, sublattices=sublattices, phase_type=phase_type)
    stock = {e: .003/len(elements) for e in elements}
    p = model.at_tp(stock, 713)
    assert p.heat_capacity_J_K == pytest.approx(.003*30+.2)
    assert p.enthalpy_J == pytest.approx(.003*30*713+.2*(713-600))
    assert model.at_ph(0, stock, p.enthalpy_J).temperature_K == pytest.approx(713, abs=1e-7)
    clone = CefMaterialSpec.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.to_dict() == model.to_dict()
    runtime = session(model)
    validate_solid(runtime._state, model)
    assert set(runtime._state.species_amounts) == set(elements)
    assert runtime._state.phase == phase_type
    assert len(runtime._state.phases.phases) == 1
    assert "physical_model" not in runtime._state.equipment.equipment["solid_material"].settings
    spec = runtime.compiled_composition.spec
    assert WorldCompositionSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_finite_power_analytic_temperature_energy_and_material_initialization():
    runtime = session()
    before = runtime._state
    model = solid_model(before, runtime.compiled_composition.spec)
    enthalpy = solid_enthalpy(before, model)
    execute(runtime, "heat", target_temperature_K=610, duration_s=10)
    state = runtime._state
    expected = 610-10*exp(-.05*10/.23)
    assert state.temperature_K == pytest.approx(expected, abs=2e-6)
    assert state.species_amounts == before.species_amounts
    assert solid_enthalpy(state, model)-enthalpy == pytest.approx(
        state.ledger.energy_jacket_J, abs=1e-7)
    measurement = execute(runtime, "measure", instrument="calphad_phase_meter").observation
    assert measurement.channel("amount:FE").value == .0005
    assert measurement.channel("site:0:NI").value == pytest.approx(.5)
    execute(runtime, "pause_control")
    execute(runtime, "wait", duration_s=1)
    assert runtime._state.temperature_K == pytest.approx(state.temperature_K, abs=1e-7)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert solid_enthalpy(runtime._state, model) == pytest.approx(
        enthalpy+.2*(state.temperature_K-600), abs=1e-7)
    runtime.reset(options={"scope": "vessel", "vessel_ids": ["reactor"]})
    assert runtime._state.temperature_K == 600


def test_invalid_capacity_inventory_model_and_rollback(monkeypatch):
    import chemworld.runtime.solid_solutions as module

    runtime = session()
    state = runtime._state
    model = material()
    for invalid in ({"A": .001, "B": .001}, {"FE": 0, "NI": .001},
                    {"FE": float("nan"), "NI": .001}):
        with pytest.raises(ValueError):
            model.at_tp(invalid, 600)
    with pytest.raises(ValueError):
        model.at_ph(0, state.species_amounts, -1e6)
    with pytest.raises(ValueError):
        replace(model, support_heat_capacity_J_K=-1)
    with pytest.raises(ValueError):
        replace(model, phase_type="gas")
    rejected = runtime.execute({"operation": "heat", "target_temperature_K": 901,
                                "duration_s": 1})
    assert rejected.info["operation_outcome"] == "rejected"
    assert physical_state(runtime._state) == physical_state(state)
    assert runtime._state.ledger.time_s == state.ledger.time_s
    state = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected CEF integrator failure")

    monkeypatch.setattr(module, "integrate_phase_inventory", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 1})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert physical_state(runtime._state) == physical_state(state)
    assert runtime._state.ledger.time_s == state.ledger.time_s
    assert runtime._state.ledger.energy_jacket_J == state.ledger.energy_jacket_J


@pytest.mark.parametrize("ionic", [False, True])
def test_stoichiometric_underdetermined_potentials_do_not_invent_atom_inventory(ionic):
    text = """
    ELEMENT FE BCC_A2 55.845 0 0 !
    ELEMENT O GAS 15.999 0 0 !
    TYPE_DEF % SEQ * !
    """
    if ionic:
        text += """
        SPECIES FE+2 FE1/+2 !
        SPECIES O-2 O1/-2 !
        PHASE ALPHA:Y % 2 1 1 !
        CONSTITUENT ALPHA :FE+2:O-2: !
        PARAMETER G(ALPHA,FE+2:O-2;0) 100 -120*T*LN(T); 2000 N !
        """
        sites = ({"FE+2": 1}, {"O-2": 1})
    else:
        text += """
        PHASE ALPHA % 2 1 1 !
        CONSTITUENT ALPHA :FE:O: !
        PARAMETER G(ALPHA,FE:O;0) 100 -60*T*LN(T); 2000 N !
        """
        sites = ({"FE": 1}, {"O": 1})
    source = InputSource(DataSourceProvenance("stoichiometric-test", 0, "FeO model", "1", "CC0"),
                         "synthetic")
    phase = CefPhaseModel(TdbDatabase.from_string(text, source=source), "ALPHA")
    model = CefMaterialSpec(phase, sites, 500, 900, 600, .01,
                           phase_type="liquid" if ionic else "solid")
    result = model.at_tp({"FE": .001, "O": .001}, 600)
    assert result.constitution.chemical_potentials.status == "underdetermined"
    assert result.constitution.chemical_potentials.values_J_mol is None
    assert result.heat_capacity_J_K == pytest.approx(.002*30+.2)
    runtime = session(model)
    execute(runtime, "heat", target_temperature_K=610, duration_s=1)
    assert set(runtime._state.species_amounts) == {"FE", "O"}
    assert sum(runtime._state.species_amounts.values()) == .001


def test_nonpositive_caloric_capacity_and_corrupted_internal_state_are_rejected():
    model = material()
    payload = model.to_dict()
    payload["phase_model"]["database"]["text"] = payload["phase_model"]["database"]["text"].replace(
        "-30*T*LN(T)", "30*T*LN(T)")
    payload["support_heat_capacity_J_K"] = 0
    with pytest.raises(ValueError, match="nonpositive"):
        CefMaterialSpec.from_dict(payload).at_tp({"FE": .001, "NI": .001}, 600)
    runtime = session()
    state = runtime._state
    phase = next(iter(state.phases.phases.values()))
    corrupted = replace(phase, metadata={**phase.metadata,
                                       "site_fractions": [{"FE": .4, "NI": .6}]})
    bad = state.replace(phases=replace(state.phases, phases={phase.phase_id: corrupted}))
    with pytest.raises(ValueError, match="stationary"):
        validate_solid(bad, model)


def test_constant_inventory_common_integrator_rejects_material_source():
    import numpy as np

    from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
    from chemworld.physchem.transport_devices import TransportInventory

    model = material()
    stock = {"FE": .001, "NI": .001}
    initial = TransportInventory((stock,), (model.at_tp(stock, 600).enthalpy_J,))
    with pytest.raises(ValueError, match="zero matter rates"):
        integrate_phase_inventory(
            model, initial, duration_s=1, closure_indices=(0,), stop_functions=(),
            constant_amounts=True,
            enthalpy_bounds=lambda i, n: (model.at_tp(n, 500).enthalpy_J,
                                           model.at_tp(n, 900).enthalpy_J),
            rates=lambda candidate, equilibria: PhaseInventoryRates(
                np.array([[1e-9, 0]]), np.array([0.]), 0, 0, ()),
        )


@pytest.mark.parametrize("case", ["duplicate_owner", "wrong_binding", "unknown_version"])
def test_material_configuration_rejects_conflicts_before_execution(case):
    payload = session().compiled_composition.spec.to_dict()
    model = payload["physics"]["models"]["calphad_material"]
    if case == "duplicate_owner":
        payload["components"].append({"id": "other-material", "kind": "solid_solution",
                                      "vessel": "reactor", "parameters": {}})
    elif case == "wrong_binding":
        model["bindings"]["undeclared-vessel"] = model["bindings"].pop("reactor")
    else:
        model["version"] = "unknown"
    with pytest.raises(ValueError):
        ChemWorldRuntime(composition=payload)


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_native_gym_cli_snapshot_and_agent_history(tmp_path, capsys, noise):
    check_native_entrypoints(session().compiled_composition.spec, tmp_path, capsys, noise)


def check_native_entrypoints(spec, tmp_path, capsys, noise):
    import subprocess
    import sys

    from chemworld.agents.base import BaseAgent
    from chemworld.cli import main
    from chemworld.data.logging import load_jsonl
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records

    runtime = ChemWorldRuntime(composition=spec, observation_noise_mode=noise)
    runtime.reset(seed=40)
    gym = ChemWorldEnv(composition=spec, observation_noise_mode=noise)
    gym.reset(seed=40)
    actions = [{"operation": "heat", "target_temperature_K": 610, "duration_s": 1},
               {"operation": "measure", "instrument": "calphad_phase_meter"},
               {"operation": "terminate"}]
    paths = []
    for i, action in enumerate(actions[:2]):
        execute(runtime, **action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path/f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.to_dict() == runtime._state.to_dict()
    world, snapshot = tmp_path/"world.json", tmp_path/"snapshot.json"
    world.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    runtime.snapshot().save(snapshot)
    output = tmp_path/"continued.json"
    code = """import json, sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
v=r.execute({'operation':'wait','duration_s':1})
assert v.info['transaction_status']=='committed',v.info
with open(sys.argv[2],'w') as f:json.dump(r._state.to_dict(),f)
"""
    subprocess.run([sys.executable, "-c", code, str(snapshot), str(output)], check=True,
                   capture_output=True, text=True)
    execute(runtime, "wait", duration_s=1)
    assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
    cli_history = tmp_path/"cef-cli.jsonl"
    main(["render", "--composition", str(world), "--seed", "40", "--actions", *paths,
          "--format", "json", "--noise-mode", noise, "--trajectory", str(cli_history)])
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 2

    class Sequence(BaseAgent):
        def act(self, history):
            return actions[len(history)]

    history = tmp_path/"cef.jsonl"
    run_agent(env_id="ChemWorld", agent=Sequence(), composition=spec.to_dict(),
              world_split="public-dev", budget=100, objective="balanced", seed=40,
              observation_noise_mode=noise, output_path=history)
    records = load_jsonl(history)
    report = verify_records(records, tolerance=1e-8)
    assert len(records) == 3
    assert report.verified, report.mismatches
    replay = """import sys
from chemworld.data.logging import load_jsonl
from chemworld.eval.verify import verify_records
records=load_jsonl(sys.argv[1])
report=verify_records(records,tolerance=1e-8)
assert report.verified,report.mismatches
print(len(records))
"""
    for path, count in ((history, 3), (cli_history, 2)):
        completed = subprocess.run([sys.executable, "-c", replay, str(path)], check=True,
                                   capture_output=True, text=True)
        assert completed.stdout.strip() == str(count)
