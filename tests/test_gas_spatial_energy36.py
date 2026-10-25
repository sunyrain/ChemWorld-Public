"""Thermal spatial edge cases and original runtime transaction consumers."""

import importlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from chemworld.examples.gas_thermal_column import example_inputs, example_world
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_spatial_energy import spatial_energy_rates
from chemworld.runtime import ChemWorldRuntime


@pytest.fixture(scope="module")
def cases():
    directory = str(Path(__file__).parent / "reference")
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("gas_spatial_energy36_cases")
    finally:
        sys.path.remove(directory)


@pytest.mark.parametrize("support", [0, 0.1])
def test_zero_or_finite_support_homogeneous_reacting_cells_have_no_spurious_flow(cases, support):
    card = cases.case(3)
    card.update(temperatures=[600] * 4, support=support, loss=0, link=0)
    card["fractions"] = [card["fractions"][0]] * 4
    spec, network, amounts = cases.native(card)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    result, detail = advance_gas_reactor_network(spec, network, state, 0.1)
    assert all(abs(row["bulk_mol"]) < 1e-12 for row in detail["connections"].values())
    assert result.final_state.temperature_K != 600
    assert result.final_state.work_J == 0
    assert abs(result.energy_balance_residual_J) < 1e-8


@pytest.mark.parametrize("bad", [-1, True, float("inf"), float("nan")])
def test_support_requires_finite_nonnegative_number(bad):
    with pytest.raises(ValueError):
        GasNodeEnergySpec("gas", 600, bad)


@pytest.mark.parametrize("defect", ["false-energy", "missing-energy", "codec", "unequal-P"])
def test_explicit_mode_and_pressure_manifold_refuse_invalid_authoring(defect):
    spec, network, amounts = example_inputs()
    with pytest.raises(ValueError):
        if defect == "false-energy":
            replace(spec, spatial_transport=replace(spec.spatial_transport, energy_equation=False))
        elif defect == "missing-energy":
            replace(spec, energy_nodes=(), heat_links=())
        elif defect == "codec":
            raw = spec.to_dict()
            raw["model"] = "finite-flow-spatial-gas-network-v1"
            GasReactorNetworkSpec.from_dict(raw)
        else:
            amounts["cell-001"]["A"] *= 1.01
            gas_reactor_network_system(spec, network).validate_state(
                pack_gas_reactor_network_state(spec, network, amounts)
            )


def test_uniform_thermochemical_reference_shift_changes_receipts_but_not_temperature(cases):
    card = cases.case(3)
    shifted = deepcopy(card)
    for species in shifted["species"].values():
        species["h"] += 7000 * species["atoms"].get("H", 0) + 3000 * species["atoms"].get("Ar", 0)
    outcomes = []
    for item in (card, shifted):
        spec, network, amounts = cases.native(item)
        state = pack_gas_reactor_network_state(spec, network, amounts)
        result, detail = advance_gas_reactor_network(spec, network, state, 0.03)
        outcomes.append((result, detail))
    assert outcomes[0][0].final_state.amounts_mol == pytest.approx(
        outcomes[1][0].final_state.amounts_mol, abs=2e-11
    )
    for name in outcomes[0][1]["nodes"]:
        assert outcomes[0][1]["nodes"][name]["temperature_K"] == pytest.approx(
            outcomes[1][1]["nodes"][name]["temperature_K"], abs=2e-7
        )
    assert outcomes[0][1]["connections"]["spatial-face:1"]["enthalpy_J"] != pytest.approx(
        outcomes[1][1]["connections"]["spatial-face:1"]["enthalpy_J"]
    )


def test_mirrored_gradient_reverses_bulk_and_fourier_without_inventing_chemical_heat(cases):
    card = cases.pure(4, 50)
    results = []
    for temperatures in (card["temperatures"], card["temperatures"][::-1]):
        spec, network, amounts = cases.native({**card, "temperatures": temperatures})
        state = pack_gas_reactor_network_state(spec, network, amounts)
        rates = spatial_energy_rates(spec, network, state)
        _, detail = advance_gas_reactor_network(spec, network, state, 0.03)
        assert all(r["chemical_energy_change_J"] == 0 for r in detail["nodes"].values())
        results.append(rates)
    for i in range(1, 4):
        for key in ("bulk_mol_s", "fourier_heat_W"):
            assert results[0]["faces"][f"spatial-face:{i}"][key] == pytest.approx(
                -results[1]["faces"][f"spatial-face:{4 - i}"][key], abs=1e-12
            )


def test_nonuniform_original_mesh_preserves_pressure_and_local_energy(cases):
    card = cases.case(3)
    spec, network, amounts = cases.native(card)
    mesh = DiffusionMesh("planar", [0, 0.002, 0.006, 0.012, 0.02], 0.01)
    original = {n.vessel_id: n.volume_L for n in spec.nodes}
    nodes = tuple(
        replace(n, volume_L=float(v * 1000))
        for n, v in zip(spec.nodes, mesh.cell_volumes_m3, strict=True)
    )
    spec = replace(spec, nodes=nodes, spatial_transport=replace(spec.spatial_transport, mesh=mesh))
    amounts = {
        n.vessel_id: {
            s: value * n.volume_L / original[n.vessel_id]
            for s, value in amounts[n.vessel_id].items()
        }
        for n in nodes
    }
    _, detail = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 0.03
    )
    assert max(abs(v) for v in detail["spatial"]["common_pressure_residual_Pa"].values()) < 1e-5
    assert all(abs(row["energy_residual_J"]) < 1e-7 for row in detail["nodes"].values())


def test_actual_upwind_temperature_used_at_finite_feed_boundary():
    spec, network, amounts = example_inputs()
    state = pack_gas_reactor_network_state(spec, network, amounts)
    rates = spatial_energy_rates(spec, network, state)
    inlet = rates["faces"]["inlet"]
    thermo = network.imported_gas_thermochemistry
    expected = sum(
        rate * thermo[s].evaluate(650).enthalpy_J_mol
        for s, rate in inlet["amount_rates_mol_s"].items()
    )
    assert inlet["enthalpy_W"] == pytest.approx(expected)
    assert inlet["fourier_heat_W"] == 0


def test_actual_mach_crossing_stops_before_depleting_the_grid(cases):
    card = cases.case(5)
    card.update(k=0, support=0, loss=0, link=0, Fin=0, Fout=0.0008, temperatures=[600] * 6)
    spec, network, amounts = cases.native(card)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    initial = spatial_energy_rates(spec, network, state)["maximum_mach"]
    spec = replace(
        spec, spatial_transport=replace(spec.spatial_transport, maximum_mach=initial * 1.02)
    )
    result, detail = advance_gas_reactor_network(spec, network, state, 1)
    assert result.stopping_event == "spatial:maximum_mach"
    assert 0 < result.actual_duration_s < 1
    assert detail["spatial"]["maximum_mach"] == pytest.approx(initial * 1.02, rel=1e-8)
    assert all(sum(row["amounts_mol"].values()) > 1e-8 for row in detail["nodes"].values())


def test_cooling_environment_stops_at_original_temperature_floor(cases):
    card = cases.pure(1, 0)
    spec, network, amounts = cases.native(card)
    spec = replace(
        spec,
        energy_nodes=(
            replace(
                spec.energy_nodes[0],
                minimum_temperature_K=599.9,
                loss_conductance_W_K=0.1,
                environment_temperature_K=300,
            ),
        ),
    )
    result, detail = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 0.1
    )
    assert result.stopping_event == "gas:cell-000:minimum_temperature"
    assert result.final_state.temperature_K == pytest.approx(599.9, abs=1e-7)
    assert detail["heat_loss_J"] > 0
    assert abs(result.energy_balance_residual_J) < 1e-8


def test_original_runtime_solver_failure_keeps_physical_state_and_queue():
    runtime = ChemWorldRuntime(composition=example_world())
    try:
        runtime.reset(seed=36)
        runtime.execute(
            {"operation": "queue_lab_heat", "target_temperature_K": 800, "duration_s": 1}
        )
        before = json.loads(json.dumps(runtime._state.to_dict()))
        with patch(
            "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
            side_effect=RuntimeError("36D injected solver failure"),
        ):
            result = runtime.execute({"operation": "wait", "duration_s": 0.03})
        after = json.loads(json.dumps(runtime._state.to_dict()))
        assert result.info["operation_outcome"] == "numerical_failure"
        assert result.info["actual_duration_s"] == 0
        # Existing operation penalties are retained, never confused with physics.
        for value in (before, after):
            for key in ("ledger", "process"):
                value[key].pop("cost")
                value[key].pop("risk")
        assert after == before
    finally:
        runtime.close()
