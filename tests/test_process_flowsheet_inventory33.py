"""A graph advances actual finite stocks on the original local/shared runtime."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.physchem.process_flowsheet import FlowsheetGraph, ProcessSource, ProcessUnit
from chemworld.physchem.process_flowsheet_inventory import (
    FlowsheetTank,
    ProcessFlowsheetInventorySpec,
    advance_process_flowsheet,
    process_flowsheet_inventory,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.process_flowsheet import flowsheet_record
from chemworld.world.process_flowsheet_world import build_process_flowsheet_world
from reference.run_molecular_processes import domain, stream


def example(*, cycle=False, real=True, capacity=0.2, reserve=1e-8, power=.01):
    d = domain(real)
    feed = stream(d, temperature=400, rate=.0001)
    units = [ProcessUnit("heater", MolecularHeatDuty(d, power), ("feed",), ("heated",))]
    output = "heated"
    if cycle:
        units.extend((ProcessUnit("compressor", MolecularPressureStage(d, "compressor", 4e5, .8),
                                  ("heated",), ("compressed",)),
                      ProcessUnit("expander", MolecularPressureStage(d, "expander", 2e5, .85),
                                  ("compressed",), ("product",))))
        output = "product"
    graph = FlowsheetGraph((ProcessSource("feed", d, feed),), tuple(units), {"product": output})
    spec = ProcessFlowsheetInventorySpec(
        graph, (FlowsheetTank("source", d, 2e5, "gas", .3, 10., reserve),
                FlowsheetTank("receiver", d, 2e5, "gas", capacity, 10.)),
        {"feed": "source"}, {"product": "receiver"})
    amounts = ({"A": .025, "B": .075}, {"A": .002, "B": .008})
    return spec, amounts, (400., 350.)


def runtime(*, shared=True, **kwargs):
    spec, amounts, temperatures = example(**kwargs)
    world = build_process_flowsheet_world(spec, amounts, temperatures,
                                          formulas={"A": "Ar", "B": "C3H8"},
                                          shared_clock=shared)
    result = ChemWorldRuntime(composition=world)
    result.reset(seed=33)
    return result


def test_public_inventory_factory_survives_owning_submodule_import():
    import importlib

    import chemworld.physchem as api

    module = importlib.import_module("chemworld.physchem.process_flowsheet_inventory")
    assert api.initialize_process_flowsheet_inventory is module.process_flowsheet_inventory
    spec, amounts, temperatures = example()
    initial = api.initialize_process_flowsheet_inventory(spec, amounts, temperatures)
    assert initial.amounts_mol == amounts


def test_local_world_exposes_and_executes_subsecond_flow():
    r = runtime(shared=False)
    try:
        step = r.execute({"operation": "run_flow", "duration_s": .2})
        assert step.info["transaction_status"] == "committed", step.info
        assert r._state.ledger.time_s == pytest.approx(.2)
        assert sum(r._state.species_amounts.values()) == pytest.approx(.1-.0001*.2)
    finally:
        r.close()


@pytest.mark.parametrize("real", [False, True])
def test_finite_graph_conserves_components_h_u_and_separate_heat_work(real):
    spec, amounts, temperatures = example(cycle=True, real=real)
    initial = process_flowsheet_inventory(spec, amounts, temperatures)
    solved = spec.graph.solve()
    assert solved.success, solved.to_dict()
    result, diagnostic = advance_process_flowsheet(spec, initial, 1.)
    assert result.actual_duration_s == pytest.approx(1.)
    assert result.final_state.amounts_mol[0] == pytest.approx({"A": .024975, "B": .074925})
    assert result.final_state.amounts_mol[1] == pytest.approx({"A": .002025, "B": .008075})
    assert diagnostic["heater_energy_J"] == pytest.approx(.01, abs=1e-8)
    assert diagnostic["shaft_import_energy_J"] > 0
    assert diagnostic["shaft_export_energy_J"] > 0
    assert diagnostic["shaft_work_J"] == pytest.approx(solved.shaft_work_W, abs=1e-6)
    assert abs(result.enthalpy_residual_J) < 1e-6
    assert abs(diagnostic["internal_energy_residual_J"]) < 1e-5
    assert diagnostic["graph_solves"] >= 1
    assert diagnostic["unit_evaluations"] >= 3*diagnostic["graph_solves"]
    assert initial.amounts_mol == amounts


@pytest.mark.parametrize("shared", [False, True])
def test_real_world_clocks_phases_and_utility_ports_commit_together(shared):
    r = runtime(shared=shared, cycle=True)
    try:
        action = {"operation": "wait" if shared else "run_flow", "duration_s": 1.}
        step = r.execute(action)
        assert step.info["transaction_status"] == "committed", step.info
        state = r._state
        assert state.ledger.time_s == pytest.approx(1)
        assert state.inactive_vessels["receiver"].ledger.time_s == pytest.approx(1)
        assert state.species_amounts == pytest.approx({"A": .024975, "B": .074925})
        record = flowsheet_record(state)
        d = record.settings["last_flowsheet"]
        assert state.ledger.flow_heat_J == pytest.approx(.01)
        assert state.ledger.flow_shaft_work_J == pytest.approx(d["shaft_work_J"])
        assert record.settings["utility_integrals"]["shaft_import_energy_J"] > 0
        assert record.settings["utility_integrals"]["shaft_export_energy_J"] > 0
    finally:
        r.close()


@pytest.mark.parametrize("kwargs,event", [({"capacity": .01015}, "capacity:receiver"),
                                          ({"reserve": .09985}, "reserve:source")])
def test_finite_event_books_only_actual_time(kwargs, event):
    r = runtime(**kwargs)
    try:
        step = r.execute({"operation": "wait", "duration_s": 2.})
        assert step.info["transaction_status"] == "committed", step.info
        assert step.info["operation_outcome"] == "stopped"
        assert r._state.ledger.time_s == pytest.approx(1.5, abs=1e-6)
        assert r._state.ledger.flow_heat_J == pytest.approx(.015, abs=1e-8)
        assert event in str(step.info)
    finally:
        r.close()


def test_declaration_and_full_session_json_continue_exactly():
    spec, _, _ = example()
    encoded = json.loads(json.dumps(spec.to_dict()))
    assert ProcessFlowsheetInventorySpec.from_dict(encoded).to_dict() == spec.to_dict()
    r = runtime()
    restored = None
    try:
        r.execute({"operation": "wait", "duration_s": .5})
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(r.snapshot().to_dict())))
        for current in (r, restored):
            step = current.execute({"operation": "wait", "duration_s": .5})
            assert step.info["transaction_status"] == "committed", step.info
        assert json.loads(json.dumps(r.snapshot().to_dict())) == json.loads(
            json.dumps(restored.snapshot().to_dict()))
    finally:
        r.close()
        if restored is not None:
            restored.close()


def test_flow_control_changes_future_rate_and_zero_stops_material_and_utilities():
    r = runtime()
    try:
        step = r.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 0.})
        assert step.info["transaction_status"] == "committed", step.info
        before = dict(r._state.species_amounts)
        step = r.execute({"operation": "wait", "duration_s": 1.})
        assert step.info["transaction_status"] == "committed", step.info
        assert r._state.species_amounts == before
        assert r._state.ledger.flow_heat_J == r._state.ledger.flow_shaft_work_J == 0
        step = r.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 10.})
        assert step.info["transaction_status"] == "committed", step.info
        scale = flowsheet_record(r._state).settings["flow_control"]["rate_scale"]
        step = r.execute({"operation": "wait", "duration_s": 1.})
        assert step.info["transaction_status"] == "committed", step.info
        assert sum(r._state.species_amounts.values()) == pytest.approx(.1-.0001*scale)
    finally:
        r.close()


def test_invalid_device_candidate_does_not_partially_transfer_finite_stocks():
    r = runtime(power=1e10)
    try:
        original = r._state.to_dict()
        step = r.execute({"operation": "wait", "duration_s": 1.})
        assert step.info["transaction_status"] != "committed"
        after = r._state.to_dict()
        # The original runtime charges a failed action's cost/risk; physical
        # material, phases, energy and time still remain uncommitted.
        for section in ("ledger", "process"):
            for key in ("cost", "risk"):
                assert after[section][key] >= original[section][key]
                after[section][key] = original[section][key]
        assert after == original
    finally:
        r.close()


def test_boundary_ownership_and_reference_mismatches_rejected():
    spec, _, _ = example()
    with pytest.raises(ValueError, match="owner"):
        replace(spec, outlet_tanks={"product": "missing"})
    with pytest.raises(ValueError, match="owner"):
        replace(spec, source_tanks={})
