"""Original-world piston ownership, signs, event limits and actual-volume kinetics."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.gas_flow_column import example_inputs as spatial_inputs
from chemworld.examples.gas_piston import example_inputs, mechanism_document
from chemworld.examples.gas_surface_network import example_inputs as surface_inputs
from chemworld.physchem.gas_network_pressure import GasPressureControlSpec, gas_work_key
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_inventory_key,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import unpack_gas_network
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def internal_energy(states, thermal):
    # Independent authored constant-Cp source, original finite support included once.
    return sum(
        (0.5 * (s.temperature_K - 298.15) if thermal else 0.0)
        + sum(
            n * (h + cp * (s.temperature_K - 298.15) - 8.31446261815324 * s.temperature_K)
            for species, n in s.species_amounts.items()
            for h, cp in [(0.0, 30.0) if species == "A" else (-9000.0, 44.0)]
        )
        for s in states.values()
    )


@pytest.mark.parametrize("thermal", (False, True))
@pytest.mark.parametrize("compress", (False, True))
def test_actual_world_first_law_separates_input_work_and_original_output_work(thermal, compress):
    spec, network, amounts = example_inputs(thermal=thermal, connected=compress)
    if compress:
        spec = replace(
            spec,
            nodes=tuple(replace(n, chemistry_enabled=False) for n in spec.nodes),
            connections=tuple(
                replace(e, coefficient=0.0 if e.connection_id == "inlet" else 0.002)
                for e in spec.connections
            ),
        )
    runtime = ChemWorldRuntime(composition=gas_reactor_network_world(spec, network, amounts))
    try:
        runtime.reset(seed=35)
        initial = internal_energy(local_states(runtime._state), thermal)
        result = runtime.execute({"operation": "wait", "duration_s": 0.2})
        assert result.info["transaction_status"] == "committed"
        states = local_states(runtime._state)
        piston = states["piston"]
        delta_u = internal_energy(states, thermal) - initial
        by = runtime._state.ledger.boundary_work_J
        assert delta_u == pytest.approx(runtime._state.ledger.energy_jacket_J - by, abs=3e-6)
        assert by == pytest.approx(100 * (piston.volume_L - 1.0), abs=3e-6)
        assert (by < 0) if compress else (by > 0)
        account = next(
            s for s in piston.state_slots.slots if s.spec.state_type == "gas_network_accounting_v1"
        )
        assert account.values[gas_work_key("piston")] == pytest.approx(-by, abs=3e-6)
        assert piston.phases.phases["gas"].volume_L == piston.volume_L
        assert all(
            f.unit not in {"L", "K"} for s in piston.state_slots.slots for f in s.spec.fields
        )
        assert piston.vessels.vessels["piston"].max_volume_L == 2.0
    finally:
        runtime.close()


@pytest.mark.parametrize("thermal", (False, True))
def test_codecs_preserve_distinct_piston_contract_without_changing_old_contract(thermal):
    spec, _, _ = example_inputs(thermal=thermal)
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    original = replace(spec, pressure_control=None)
    assert "pressure_control" not in original.to_dict()
    assert GasReactorNetworkSpec.from_dict(original.to_dict()) == original
    data = spec.to_dict()
    data["model"] = original.to_dict()["model"]
    with pytest.raises(ValueError, match="model version"):
        GasReactorNetworkSpec.from_dict(data)


def test_second_order_reaction_reads_current_actual_piston_volume():
    spec, _, amounts = example_inputs()
    doc = mechanism_document()
    doc["reactions"][0]["equation"] = "2 A => B"
    network = convert_mechanism(doc)
    system = gas_reactor_network_system(spec, network)
    before = pack_gas_reactor_network_state(spec, network, amounts)
    more = pack_gas_reactor_network_state(
        spec, network, {"piston": {s: n * 1.5 for s, n in amounts["piston"].items()}}
    )
    first = (
        system.processes[0].evaluate(before).amount_rates_mol_s[gas_inventory_key("piston", "A")]
    )
    second = system.processes[0].evaluate(more).amount_rates_mol_s[gas_inventory_key("piston", "A")]
    # Identical concentrations, 1.5 times the actual reacting volume; fixed V gives 2.25.
    assert second == pytest.approx(1.5 * first, rel=1e-12)


def test_world_stops_at_finite_stroke_and_projects_same_original_phase_volume():
    spec, network, amounts = example_inputs()
    spec = replace(spec, pressure_control=replace(spec.pressure_control, maximum_volume_L=1.08))
    runtime = ChemWorldRuntime(composition=gas_reactor_network_world(spec, network, amounts))
    try:
        runtime.reset(seed=35)
        result = runtime.execute({"operation": "wait", "duration_s": 4.0})
        assert result.info["transaction_status"] == "committed"
        assert 0 < result.info["actual_duration_s"] < 4.0
        assert runtime._state.volume_L == pytest.approx(1.08, abs=2e-6)
        assert runtime._state.phases.phases["gas"].volume_L == runtime._state.volume_L
        assert runtime._state.ledger.boundary_work_J == pytest.approx(8.0, abs=1e-7)
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("surface", "spatial"))
def test_piston_does_not_silently_attach_unsupported_surface_or_spatial_kernel(kind):
    if kind == "surface":
        _, spec, _, _, _ = surface_inputs()
    else:
        spec, _, _ = spatial_inputs()
    node = spec.nodes[0]
    with pytest.raises(ValueError, match="without surfaces or spatial"):
        replace(
            spec,
            pressure_control=GasPressureControlSpec(
                node.vessel_id, 1e5, node.volume_L * 0.2, node.volume_L * 3
            ),
        )


@pytest.mark.parametrize("field", ("volume", "pressure", "phase"))
def test_restore_rejects_conflicting_physical_stock(field):
    spec, network, amounts = example_inputs(connected=True)
    composition = gas_reactor_network_world(spec, network, amounts)
    states = dict(composition.physics.initial_states)
    piston = states.pop("piston")
    if field == "volume":
        piston = piston.replace(volume_L=1.1)
    elif field == "pressure":
        piston = piston.replace(pressure_Pa=1.1e5)
    else:
        piston = piston.replace(species_amounts={"A": 0.01, "B": 0.01})
    with pytest.raises(ValueError):
        unpack_gas_network(piston.replace(inactive_vessels=states), network)


def test_public_trajectory_is_actual_volume_without_integrated_volume_coordinate():
    spec, network, amounts = example_inputs(thermal=True, connected=True)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    result, _ = advance_gas_reactor_network(spec, network, state, 0.2)
    for point in result.trajectory:
        total = sum(point.amounts_mol[gas_inventory_key("piston", s)] for s in network.species_ids)
        assert point.volume_L == pytest.approx(
            3.0 + total * 8.31446261815324 * point.temperature_K * 0.01
        )
    assert not any("volume" in key for key in result.final_state.coordinates)


def test_nonanchor_piston_can_resume_original_world_json():
    spec, network, amounts = example_inputs(thermal=True, connected=True)
    spec = replace(spec, anchor_vessel_id="feed", temperature_K=700.0)
    runtime = ChemWorldRuntime(composition=gas_reactor_network_world(spec, network, amounts))
    restored = None
    try:
        runtime.reset(seed=35)
        assert (
            runtime.execute({"operation": "wait", "duration_s": 0.2}).info["transaction_status"]
            == "committed"
        )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for r in (runtime, restored):
            assert (
                r.execute({"operation": "wait", "duration_s": 0.2}).info["transaction_status"]
                == "committed"
            )
        assert restored._state.to_dict() == runtime._state.to_dict()
        states = local_states(runtime._state)
        assert states["piston"].volume_L != 1.0
        assert states["piston"].pressure_Pa == 1e5
    finally:
        runtime.close()
        if restored:
            restored.close()
