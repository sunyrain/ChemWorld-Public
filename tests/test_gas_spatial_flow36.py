"""Actual finite feed/grid/product workflows and their physical boundaries."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.gas_flow_column import example_inputs, example_world
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_spatial_reactor import spatial_rates
from chemworld.runtime import ChemWorldRuntime


@pytest.mark.parametrize("non_equimolar", (False, True))
@pytest.mark.parametrize("model", ("stefan-maxwell", "mixture-averaged"))
def test_finite_tanks_grid_pressure_atoms_and_first_law(non_equimolar, model):
    spec, network, amounts = example_inputs(non_equimolar=non_equimolar, flux_model=model)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    result, detail = advance_gas_reactor_network(spec, network, state, 0.2)
    assert result.actual_duration_s == 0.2
    assert all(abs(r) < 1e-12 for r in result.conservation_residuals.values())
    for row in detail["nodes"].values():
        assert abs(row["energy_residual_J"]) < 1e-8
    p = [detail["nodes"][name]["pressure_Pa"] for name in spec.spatial_transport.cell_vessel_ids]
    assert max(p) - min(p) < 1e-6
    assert detail["nodes"]["feed"]["pressure_Pa"] > max(p)
    assert detail["nodes"]["receiver"]["pressure_Pa"] < min(p)
    assert sum(detail["nodes"]["feed"]["amounts_mol"].values()) == pytest.approx(
        sum(amounts["feed"].values()) - 0.0004 * 0.2, abs=1e-12
    )
    assert sum(detail["nodes"]["receiver"]["amounts_mol"].values()) == pytest.approx(
        sum(amounts["receiver"].values()) + 0.0004 * 0.2, abs=1e-12
    )
    assert detail["connections"]["inlet"]["diffusion_mol"] == {"A": 0, "B": 0}
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert state.time_s == 0


def test_unequal_boundaries_and_molar_production_have_one_continuity_solution():
    spec, network, amounts = example_inputs(non_equimolar=True)
    spec = replace(
        spec,
        connections=tuple(
            replace(e, coefficient=0 if e.connection_id == "inlet" else 0.0008)
            for e in spec.connections
        ),
    )
    state = pack_gas_reactor_network_state(spec, network, amounts)
    chemistry, packet = spatial_rates(spec, network, state)
    cells = spec.spatial_transport.cell_vessel_ids
    for i, name in enumerate(cells):
        incoming = packet["faces"]["inlet" if i == 0 else f"spatial-face:{i}"]["bulk_mol_s"]
        outgoing = packet["faces"]["outlet" if i == len(cells) - 1 else f"spatial-face:{i + 1}"][
            "bulk_mol_s"
        ]
        actual = sum(chemistry[name].amount_rates_mol_s.values()) + incoming - outgoing
        assert actual == pytest.approx(packet["cell_total_rates_mol_s"][name], abs=1e-17)
    assert packet["pressure_rate_Pa_s"] < 0


def test_reaction_induced_internal_reverse_flow_uses_its_actual_upstream_cell():
    spec, network, amounts = example_inputs(non_equimolar=True)
    spec = replace(spec, connections=tuple(replace(e, coefficient=0) for e in spec.connections))
    for i, name in enumerate(spec.spatial_transport.cell_vessel_ids):
        total = sum(amounts[name].values())
        amounts[name] = {"A": total * (0.9 if i < 2 else 0.1), "B": total * (0.1 if i < 2 else 0.9)}
    _, packet = spatial_rates(spec, network, pack_gas_reactor_network_state(spec, network, amounts))
    face = packet["faces"]["spatial-face:2"]
    assert face["bulk_mol_s"] < 0
    assert face["convection_mol_s"]["A"] / face["bulk_mol_s"] == pytest.approx(0.1)


def test_one_cell_boundary_mach_is_checked_without_an_interior_face():
    spec, network, amounts = example_inputs(cells=1)
    spec = replace(spec, spatial_transport=replace(spec.spatial_transport, maximum_mach=1e-9))
    with pytest.raises(ValueError, match="mass-average"):
        gas_reactor_network_system(spec, network).validate_state(
            pack_gas_reactor_network_state(spec, network, amounts)
        )


@pytest.mark.parametrize(
    "defect",
    (
        "tank-chemistry",
        "wrong-port",
        "extra-node",
        "nonmolar",
        "face-name",
        "sealed-mode",
        "codec",
        "cell-pressure",
    ),
)
def test_inconsistent_boundary_or_inventory_declarations_refused(defect):
    spec, network, amounts = example_inputs()
    with pytest.raises(ValueError):
        if defect == "tank-chemistry":
            replace(
                spec,
                nodes=tuple(
                    replace(n, chemistry_enabled=True) if n.vessel_id == "feed" else n
                    for n in spec.nodes
                ),
            )
        elif defect == "wrong-port":
            replace(
                spec,
                connections=(
                    replace(spec.connections[0], target_vessel_id="cell-001"),
                    spec.connections[1],
                ),
            )
        elif defect == "extra-node":
            replace(spec, nodes=(*spec.nodes, replace(spec.nodes[0], vessel_id="extra")))
        elif defect == "nonmolar":
            replace(
                spec,
                connections=(replace(spec.connections[0], mode="mass-flow"), spec.connections[1]),
            )
        elif defect == "face-name":
            replace(
                spec,
                connections=(
                    replace(spec.connections[0], connection_id="spatial-face:1"),
                    spec.connections[1],
                ),
            )
        elif defect == "sealed-mode":
            replace(spec, spatial_transport=replace(spec.spatial_transport, boundary_mode="sealed"))
        elif defect == "codec":
            raw = spec.to_dict()
            raw["model"] = "sealed-spatial-gas-network-v1"
            GasReactorNetworkSpec.from_dict(raw)
        else:
            amounts["cell-000"]["A"] *= 2
            gas_reactor_network_system(spec, network).validate_state(
                pack_gas_reactor_network_state(spec, network, amounts)
            )


def test_actual_world_nonanchor_selection_and_exact_json_continuation():
    runtime = ChemWorldRuntime(composition=example_world(non_equimolar=True))
    restored = None
    try:
        runtime.reset(seed=36)
        assert (
            runtime.execute({"operation": "wait", "duration_s": 0.1}).info["transaction_status"]
            == "committed"
        )
        assert (
            runtime.execute({"operation": "select_vessel", "vessel": "receiver"}).info[
                "transaction_status"
            ]
            == "committed"
        )
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        for item in (runtime, restored):
            assert (
                item.execute({"operation": "wait", "duration_s": 0.2}).info["transaction_status"]
                == "committed"
            )
        assert json.loads(json.dumps(runtime._state.to_dict())) == json.loads(
            json.dumps(restored._state.to_dict())
        )
        assert runtime._state.ledger.time_s == pytest.approx(0.3)
    finally:
        runtime.close()
        if restored is not None:
            restored.close()
