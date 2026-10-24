"""Same-model stationary surfaces, explicit failed points, controls and native use."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.catalysis_analysis import (
    catalytic_rate_control,
    catalytic_steady_states,
    map_catalytic_descriptors,
    solve_catalytic_steady_state,
    with_catalytic_descriptors,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world
from reference.catalysis54_fixtures import conditions, mechanism


def test_cycle_steady_root_has_nonzero_turnover_and_control_sum():
    spec, boundary = mechanism(), conditions()
    row = solve_catalytic_steady_state(spec, boundary, {"bound": 0.1}, target_species="B")
    assert row["status"] == "converged"
    assert 0 < row["coverages"]["bound"] < 1
    assert row["turnover_rate_s_inv"] > 0
    assert row["stability"] == "stable"
    assert abs(row["surface_residual_s_inv"][0]) < 1e-9
    control = catalytic_rate_control(spec, boundary, {"bound": 0.1}, target_species="B")
    assert control["status"] == "available"
    assert sum(r["value"] for r in control["rate_control"].values()) == pytest.approx(1, abs=1e-5)
    assert control["reaction_orders"]["A"]["value"] > 0
    assert control["apparent_activation_energy"]["unit"] == "J/mol"


def test_three_roots_and_every_seed_are_reported():
    result = catalytic_steady_states(
        mechanism(multistable=True),
        conditions(multistable=True),
        [{"bound": t} for t in (0.01, 0.5, 0.99, 0.2)],
        target_species="A",
    )
    assert result["seed_count"] == len(result["attempts"]) == 4
    assert result["distinct_roots"] == 3
    assert [r["stability"] for r in result["roots"]] == ["stable", "unstable", "stable"]
    assert result["roots"][1]["coverages"]["bound"] == pytest.approx(0.5, abs=1e-7)


def test_descriptor_map_keeps_the_invalid_hole_and_does_not_mutate_input():
    spec = mechanism()
    before = spec.to_dict()
    result = map_catalytic_descriptors(
        spec,
        conditions(),
        [
            {"binding": -3000, "strain": 0},
            {"binding": 7000, "strain": 0},
            {"binding": 3000, "strain": 0},
        ],
        [{"bound": 0.3}],
        target_species="B",
    )
    assert result["points_total"] == 3 and result["points_converged"] == 2
    assert result["points"][1]["status"] == "rejected"
    assert result["points"][1]["roots"] == []
    assert spec.to_dict() == before
    changed = with_catalytic_descriptors(spec, {"binding": 100, "strain": 200})
    assert changed.species_by_id["bound"].adsorbate.descriptors["binding"] == 100
    assert CatalyticMechanismSpec.from_dict(changed.to_dict()).to_dict() == changed.to_dict()


def test_failed_roots_and_unavailable_stencils_remain_visible():
    failed = solve_catalytic_steady_state(
        mechanism(multistable=True),
        conditions(multistable=True),
        {"bound": 0.2},
        target_species="A",
        policy=RootPolicy(maximum_iterations=1),
    )
    assert failed["status"] == "not_converged" and failed["coverages"] is None
    reverse = catalytic_rate_control(mechanism(), conditions(), {"bound": 0.2}, target_species="A")
    assert reverse["status"] == "unavailable"
    boundary = catalytic_rate_control(
        mechanism(), conditions(temperature=300), {"bound": 0.2}, target_species="B"
    )
    assert boundary["status"] == "partial"
    assert boundary["apparent_activation_energy"]["status"] == "unavailable"
    assert len(boundary["apparent_activation_energy"]["solutions"]) == 2
    equilibrium = catalytic_rate_control(
        mechanism(multistable=True),
        conditions(multistable=True),
        {"bound": 0.5},
        target_species="A",
    )
    assert equilibrium["status"] == "unavailable"


def test_analysis_boundary_rejects_unknown_gas_sites_and_bad_seed():
    spec, boundary = mechanism(), conditions()
    with pytest.raises(ValueError, match="match"):
        solve_catalytic_steady_state(
            spec,
            replace(boundary, gas_partial_pressures_Pa={"unknown": 1}),
            {"bound": 0.1},
            target_species="B",
        )
    with pytest.raises(ValueError):
        solve_catalytic_steady_state(spec, boundary, {"bound": 1.1}, target_species="B")


def test_selected_surface_state_reaches_original_finite_world_and_exact_snapshot():
    spec, boundary = mechanism(), conditions(capacity=1e-8)
    steady = solve_catalytic_steady_state(spec, boundary, {"bound": 0.1}, target_species="B")
    world = descriptor_catalysis_world(
        spec,
        steady["gas_amounts_mol"],
        capacities_mol=boundary.capacities_mol,
        initial_adsorbates_mol=steady["adsorbate_amounts_mol"],
        volume_L=1,
        temperature_K=350,
        support_heat_capacity_J_K=5,
        stop_temperature_K=440,
        maximum_pressure_Pa=2e5,
    )
    original, resumed = ChemWorldRuntime(composition=world), None
    try:
        original.reset(seed=54)
        first = original.execute({"operation": "wait", "duration_s": 2})
        assert first.info["transaction_status"] == "committed", first.info
        resumed = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(original.snapshot().to_dict()))
        )
        for runtime in (original, resumed):
            result = runtime.execute({"operation": "wait", "duration_s": 3})
            assert result.info["transaction_status"] == "committed", result.info
            assert runtime._state.species_amounts["A"] < steady["gas_amounts_mol"]["A"]
            assert runtime._state.species_amounts["B"] > steady["gas_amounts_mol"]["B"]
        assert original._state.to_dict() == resumed._state.to_dict()
    finally:
        original.close()
        if resumed is not None:
            resumed.close()
