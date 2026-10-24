"""Finite reactive gas networks use the original solver, phases and shared clock."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    gas_inventory_key,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import GAS_ACCOUNTING_TYPE, GasReactorNetworkRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world

module_spec = importlib.util.spec_from_file_location(
    "gas_network35_cases", Path(__file__).parent / "reference/gas_network35_cases.py"
)
cases = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cases)


def advance(case="molar", duration=1):
    network, spec = convert_mechanism(cases.document()), cases.network_spec(case)
    state = pack_gas_reactor_network_state(spec, network, cases.AMOUNTS)
    return spec, network, advance_gas_reactor_network(spec, network, state, duration)


@pytest.mark.parametrize("case", ("molar", "mass", "valve", "closed"))
def test_finite_flow_thermal_balances_and_original_species(case):
    spec, network, (result, diagnostic) = advance(case)
    assert result.actual_duration_s == 1
    assert set(network.species_ids) == {"A", "B"}
    assert sum(result.final_state.amounts_mol.values()) == pytest.approx(0.035, abs=1e-12)
    assert result.conservation_residuals["H"] == pytest.approx(0, abs=1e-12)
    assert diagnostic["energy_residual_J"] == pytest.approx(0, abs=1e-6)
    assert (
        diagnostic["nodes"]["source"]["enthalpy_out_J"]
        == diagnostic["nodes"]["receiver"]["enthalpy_in_J"]
    )
    for node in diagnostic["nodes"].values():
        assert node["energy_residual_J"] == pytest.approx(0, abs=1e-6)
    transferred = sum(diagnostic["connections"]["feed"]["amounts_mol"].values())
    assert transferred == pytest.approx(
        0 if case == "closed" else 0.003 if case == "molar" else transferred
    )
    assert diagnostic["nodes"]["receiver"]["amounts_mol"]["B"] > 0.002
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_each_reaction_reads_its_actual_local_volume():
    network = convert_mechanism(cases.document(second_order=True))
    spec = replace(cases.network_spec("closed"), connections=())
    amounts = {n.vessel_id: {"A": 0.02, "B": 0.01} for n in spec.nodes}
    initial = pack_gas_reactor_network_state(spec, network, amounts)
    rates = gas_reactor_network_system(spec, network).processes[0].evaluate(initial)
    assert rates.amount_rates_mol_s[gas_inventory_key("source", "A")] == pytest.approx(
        3 * rates.amount_rates_mol_s[gas_inventory_key("receiver", "A")]
    )
    assert len(rates.reaction_progress_mol_s) == 2


@pytest.mark.parametrize("case,time", (("source-cutoff", 5), ("pressure-limit", None)))
def test_events_stop_the_whole_network_at_the_actual_boundary(case, time):
    _, _, (result, diagnostic) = advance(case, 10)
    assert result.stopping_event is not None
    assert 0 < result.actual_duration_s < 10
    if time is not None:
        assert result.actual_duration_s == pytest.approx(time, abs=1e-7)
        assert sum(diagnostic["nodes"]["source"]["amounts_mol"].values()) == pytest.approx(0.015)
    else:
        assert diagnostic["nodes"]["receiver"]["pressure_Pa"] == pytest.approx(20000, abs=1e-6)


def test_three_node_branch_and_circulation_conserves_inventory_and_enthalpy():
    network = convert_mechanism(cases.document())
    spec = GasReactorNetworkSpec(
        tuple(GasReactorNodeSpec(v, i + 1) for i, v in enumerate(("a", "b", "c"))),
        (
            GasFlowConnectionSpec("ab", "a", "b", coefficient=0.001),
            GasFlowConnectionSpec("ac", "a", "c", coefficient=0.002),
            GasFlowConnectionSpec("ca", "c", "a", coefficient=0.0015),
        ),
        700,
        "a",
    )
    amounts = {n.vessel_id: {"A": 0.02, "B": 0.01} for n in spec.nodes}
    result, diagnostic = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1
    )
    assert sum(result.final_state.amounts_mol.values()) == pytest.approx(0.09)
    assert set(diagnostic["connections"]) == {"ab", "ac", "ca"}
    assert sum(
        n["enthalpy_in_J"] - n["enthalpy_out_J"] for n in diagnostic["nodes"].values()
    ) == pytest.approx(0, abs=1e-12)


def world(case="molar"):
    return gas_reactor_network_world(
        cases.network_spec(case), convert_mechanism(cases.document()), cases.AMOUNTS
    )


def test_real_shared_world_wait_changes_both_original_ledgers():
    runtime = ChemWorldRuntime(composition=world())
    try:
        runtime.reset(seed=7)
        outcome = runtime.execute({"operation": "wait", "duration_s": 1})
        assert outcome.info["transaction_status"] == "committed", outcome.info
        assert runtime._state.ledger.time_s == 1
        assert runtime._state.vessel_elapsed_s == {"source": 1, "receiver": 1}
        locals_ = local_states(runtime._state)
        assert sum(locals_["source"].species_amounts.values()) == pytest.approx(0.027)
        assert sum(locals_["receiver"].species_amounts.values()) == pytest.approx(0.008)
        for local in locals_.values():
            assert local.phases.phases["gas"].species_amounts_mol == local.species_amounts
        counters = [
            s
            for s in locals_["source"].state_slots.slots
            if s.spec.state_type == GAS_ACCOUNTING_TYPE
        ]
        assert len(counters) == 1
        assert all(
            f.species_id is None and f.conserved_quantity is None for f in counters[0].spec.fields
        )
    finally:
        runtime.close()


def test_solver_failure_returns_no_state_or_time_and_leaves_original_unchanged(monkeypatch):
    composition = world()
    states = composition.physics.initial_states
    state = states["source"].replace(inactive_vessels={"receiver": states["receiver"]})
    before = state.to_dict()

    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr("chemworld.runtime.gas_reactor_network.advance_gas_reactor_network", fail)
    outcome = GasReactorNetworkRuntime(composition).advance(state, {"duration_s": 1})
    assert outcome.outcome == "numerical_failure"
    assert outcome.state is None
    assert outcome.actual_duration_s == 0
    assert state.to_dict() == before


def test_near_vacuum_trial_extension_stops_before_empty_inventory():
    _, _, (result, diagnostic) = advance("molar", 11)
    amount = sum(diagnostic["nodes"]["source"]["amounts_mol"].values())
    assert result.stopping_event == "gas:source:minimum_amount"
    assert result.actual_duration_s == pytest.approx(10, abs=1e-8)
    assert 0 < amount <= 1.01e-12


def test_nonreacting_finite_storage_uses_opposite_signed_thermostat_heat():
    from chemworld.physchem.reaction_network import R_J_PER_MOL_K

    network = convert_mechanism(cases.document())
    spec = cases.network_spec()
    spec = replace(spec, nodes=tuple(replace(n, chemistry_enabled=False) for n in spec.nodes))
    result, diagnostic = advance_gas_reactor_network(
        spec,
        network,
        pack_gas_reactor_network_state(spec, network, cases.AMOUNTS),
        1,
    )
    expected = R_J_PER_MOL_K * 700 * 0.003
    assert diagnostic["nodes"]["source"]["bath_heat_J"] == pytest.approx(expected)
    assert diagnostic["nodes"]["receiver"]["bath_heat_J"] == pytest.approx(-expected)
    assert result.final_state.boundary_heat_J == pytest.approx(0, abs=1e-12)


def test_single_node_world_waits_without_declaring_vessel_selection():
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    spec = cases.network_spec("molar")
    spec = replace(spec, nodes=tuple(n for n in spec.nodes if n.vessel_id == "source"),
                   connections=(), anchor_vessel_id="source")
    network = convert_mechanism(cases.document())
    composition = gas_reactor_network_world(spec, network, {"source": cases.AMOUNTS["source"]})
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=7)
        before = runtime._state.species_amounts["A"]
        result = runtime.execute({"operation": "wait", "duration_s": 0.2})
        assert result.info["transaction_status"] == "committed"
        assert result.info["actual_duration_s"] == pytest.approx(0.2)
        assert runtime._state.species_amounts["A"] < before
    finally:
        runtime.close()


def test_pressure_valve_is_closed_against_adverse_gradient():
    spec, network = cases.network_spec("valve"), convert_mechanism(cases.document())
    amounts = {"source": {"A": 0.001, "B": 0.001}, "receiver": {"A": 0.02, "B": 0.01}}
    _, diagnostic = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1
    )
    assert sum(diagnostic["connections"]["feed"]["amounts_mol"].values()) == 0


def test_mass_flow_uses_current_composition_when_reaction_changes_molecular_weight():
    from chemworld.physchem.elements import molecular_weight

    payload = cases.document()
    payload["species"][1]["composition"] = {"H": 4}
    payload["reactions"][0]["equation"] = "2 A => B"
    network, spec = convert_mechanism(payload), cases.network_spec("mass")
    _, diagnostic = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, cases.AMOUNTS), 1
    )
    masses = {s.species_id: molecular_weight(s.composition) / 1000 for s in network.species}
    for node, sign in (("source", -1), ("receiver", 1)):
        delta = sum(
            (diagnostic["nodes"][node]["amounts_mol"][s] - cases.AMOUNTS[node][s]) * masses[s]
            for s in masses
        )
        assert delta == pytest.approx(sign * 6e-6, abs=1e-12)


@pytest.mark.parametrize(
    "field,value",
    (("phase_id", "bad"), ("vessel_id", "bad"), ("phase_type", "liquid"), ("legacy", None)),
)
def test_runtime_rejects_wrong_phase_ownership_and_legacy_carrier(field, value):
    from chemworld.foundation.solvents import SolventInventory
    from chemworld.foundation.state_ledgers import PhaseLedger
    from chemworld.runtime.gas_reactor_network import unpack_gas_network

    composition = world()
    states = composition.physics.initial_states
    receiver = states["receiver"]
    phase = receiver.phases.phases["gas"]
    if field == "legacy":
        phase = replace(
            phase,
            metadata={"carrier_representation": "legacy_volume"},
            solvents=SolventInventory.pure(0, receiver.volume_L),
        )
    else:
        phase = replace(phase, **{field: value})
    receiver = receiver.replace(phases=PhaseLedger({"gas": phase}))
    state = states["source"].replace(inactive_vessels={"receiver": receiver})
    with pytest.raises(ValueError, match="phase inventory"):
        unpack_gas_network(state, composition.physics.network)


def test_shared_runtime_failure_rolls_back_all_original_inventory_and_counters(monkeypatch):
    runtime = ChemWorldRuntime(composition=world())
    try:
        runtime.reset(seed=7)
        assert (
            runtime.execute({"operation": "wait", "duration_s": 0.2}).info["transaction_status"]
            == "committed"
        )
        before = runtime._state.to_dict()

        def fail(*args, **kwargs):
            raise RuntimeError("injected network failure")

        monkeypatch.setattr(
            "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network", fail
        )
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert result.info["actual_duration_s"] == 0
        after = runtime._state.to_dict()
        assert after["ledger"]["cost"] == pytest.approx(before["ledger"]["cost"] + 0.03)
        for key in ("ledger", "process"):
            for field in ("cost", "risk"):
                before[key].pop(field)
                after[key].pop(field)
        assert after == before
    finally:
        runtime.close()


@pytest.mark.parametrize("case", ("molar", "source-cutoff"))
def test_shared_snapshot_continues_in_new_process_without_inventory_or_clock_copy(tmp_path, case):
    runtime = ChemWorldRuntime(composition=world(case))
    try:
        runtime.reset(seed=7)
        first = runtime.execute({"operation": "wait", "duration_s": 0.25})
        assert first.info["transaction_status"] == "committed"
        source, output = tmp_path / "snapshot.json", tmp_path / "result.json"
        runtime.snapshot().save(source)
        actions = [
            {"operation": "select_vessel", "vessel": "source"},
            {"operation": "wait", "duration_s": 0.75 if case == "molar" else 10},
        ]
        for action in actions:
            assert runtime.execute(action).info["transaction_status"] == "committed"
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            f"results=[r.execute(a) for a in {actions!r}]; "
            "assert all(x.info['transaction_status']=='committed' for x in results); "
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict(),sort_keys=True))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
        if case == "source-cutoff":
            assert runtime._state.ledger.time_s == pytest.approx(5)
            assert runtime._state.lab_clock.stopped
            assert all(t == pytest.approx(5) for t in runtime._state.vessel_elapsed_s.values())
    finally:
        runtime.close()


def test_shared_connection_identity_cannot_silently_bind_a_different_endpoint():
    composition = world()
    payload = composition.to_dict()
    payload["connections"][0]["id"] = "unbound"
    from chemworld.world.composition import WorldCompositionSpec

    with pytest.raises(ValueError, match="material connection"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


def test_true_vacuum_is_explicitly_outside_positive_inventory_model():
    network, spec = convert_mechanism(cases.document()), cases.network_spec()
    amounts = {**cases.AMOUNTS, "receiver": {"A": 0, "B": 0}}
    with pytest.raises(ValueError, match="minimum amount"):
        advance_gas_reactor_network(
            spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1
        )


def test_shared_thermal_ledgers_keep_node_duties_separate_from_global_total():
    runtime = ChemWorldRuntime(composition=world())
    try:
        runtime.reset(seed=7)
        spec, network = cases.network_spec(), convert_mechanism(cases.document())
        physical = pack_gas_reactor_network_state(spec, network, cases.AMOUNTS)
        expected = {
            n.vessel_id: {"bath_heat_J": 0.0, "chemical_energy_change_J": 0.0} for n in spec.nodes
        }
        # Use the same accepted intervals to isolate allocation/selection from
        # the BDF's ordinary split-versus-one-shot truncation error.
        for _ in range(2):
            result, diagnostic = advance_gas_reactor_network(spec, network, physical, 0.5)
            physical = result.final_state
            for node in expected:
                for field in expected[node]:
                    expected[node][field] += diagnostic["nodes"][node][field]
        for index in range(2):
            if index:
                assert (
                    runtime.execute({"operation": "select_vessel", "vessel": "source"}).info[
                        "transaction_status"
                    ]
                    == "committed"
                )
            result = runtime.execute({"operation": "wait", "duration_s": 0.5})
            assert result.info["transaction_status"] == "committed", result.info.get(
                "rollback_reason"
            )
        for node, state in local_states(runtime._state).items():
            thermal = state.thermal.vessels[node]
            assert thermal.energy_jacket_J == pytest.approx(expected[node]["bath_heat_J"], abs=1e-7)
            assert thermal.heat_reaction_J == pytest.approx(
                -expected[node]["chemical_energy_change_J"], abs=1e-7
            )
            assert state.ledger.energy_jacket_J == thermal.energy_jacket_J
        assert runtime._state.ledger.energy_jacket_J == pytest.approx(
            sum(row["bath_heat_J"] for row in expected.values()), abs=1e-7
        )
        assert runtime._state.ledger.heat_reaction_J == pytest.approx(
            -sum(row["chemical_energy_change_J"] for row in expected.values()), abs=1e-7
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("defect", ("ownership", "total"))
def test_inconsistent_group_thermal_allocation_rolls_back_physical_slice(monkeypatch, defect):
    original = GasReactorNetworkRuntime.advance

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        deltas = {key: dict(value) for key, value in result.vessel_thermal_deltas.items()}
        if defect == "ownership":
            deltas.pop("receiver")
        else:
            deltas["receiver"]["energy_jacket_J"] += 1
        return replace(result, vessel_thermal_deltas=deltas)

    monkeypatch.setattr(GasReactorNetworkRuntime, "advance", corrupt)
    runtime = ChemWorldRuntime(composition=world())
    try:
        runtime.reset(seed=7)
        before = runtime._state.to_dict()
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["operation_outcome"] == "numerical_failure"
        after = runtime._state.to_dict()
        for key in ("ledger", "process"):
            for field in ("cost", "risk"):
                before[key].pop(field)
                after[key].pop(field)
        assert after == before
    finally:
        runtime.close()
