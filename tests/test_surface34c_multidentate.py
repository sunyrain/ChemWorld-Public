"""Weighted site ownership without duplicating material or caloric inventory."""

import copy
import json
from dataclasses import replace

import pytest

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    solve_catalytic_steady_state,
)
from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    catalytic_inventory_key,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.catalysis_steady_chart import build_catalytic_steady_chart
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.catalysis_analysis import analyze_catalytic_world

THETA = {"Pt": 0.55, "HA": 0.2, "HB": 0.15, "HAA": 0.1}


def mechanism(size=2):
    return convert_surface_mechanism(
        authored_declaration(site_occupancy=size),
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    ).surface


def physical(spec):
    scales = spec.coverage_amount_scales_mol({"surface": 1e-6})
    slot = make_catalytic_slot(
        "test",
        "reactor",
        spec,
        capacities_mol={"surface": 1e-6},
        initial_amounts_mol={k: t * scales[k] for k, t in THETA.items()},
    )
    state = pack_catalytic_state(
        spec, (slot,), {"A": 0.01, "B": 0.004}, temperature_K=700, volume_L=1.0
    )
    return slot, state


@pytest.mark.parametrize("size", [2, 3])
def test_weighted_sites_are_distinct_from_molecular_inventory(size):
    spec = mechanism(size)
    slot, state = physical(spec)
    assert sum(slot.values[k] * spec.site_occupancies[k] for k in spec.adsorbates) == pytest.approx(
        1e-6
    )
    assert sum(slot.values[k] for k in spec.adsorbates) < 1e-6
    rates = evaluate_catalytic_rates(state, slot, spec)
    assert {k: rates.activities[k] for k in spec.adsorbates} == pytest.approx(THETA)
    assert sum(
        rates.amount_rates_mol_s[catalytic_inventory_key(slot, k)] * spec.site_occupancies[k]
        for k in spec.adsorbates
    ) == pytest.approx(0.0, abs=1e-16)
    for s in spec.species:
        if s.species.phase == "surface":
            assert s.internal_energy(700) == s.evaluate(700).enthalpy_J_mol
    assert dict(slot.spec.capacities[0].occupied_fields) == dict(spec.site_occupancies)


@pytest.mark.parametrize("size", [2, 3])
def test_multidentate_codec_and_input_ownership(size):
    spec = mechanism(size)
    raw = json.loads(json.dumps(spec.to_dict()))
    restored = CatalyticMechanismSpec.from_dict(raw)
    assert restored.to_dict() == raw
    raw["site_occupancies"]["HA"] = 1
    assert restored.site_occupancies["HA"] == size
    slot, state = physical(spec)
    assert evaluate_catalytic_rates(state, slot, restored) == evaluate_catalytic_rates(
        state, slot, spec
    )


def test_old_default_codec_omits_new_field():
    spec = mechanism(1)
    assert set(spec.site_occupancies.values()) == {1}
    assert "site_occupancies" not in spec.to_dict()
    declared = replace(spec, site_occupancies={"HA": 1})
    assert declared.to_dict() == spec.to_dict()
    assert CatalyticMechanismSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


@pytest.mark.parametrize("bad", [0, -1, 1.5, 4, True])
def test_outside_integer_one_two_three_sites_rejected(bad):
    with pytest.raises(ValueError, match="occupancies"):
        replace(mechanism(), site_occupancies={"HA": bad})


def test_changed_occupancy_without_changed_stoichiometry_rejected():
    with pytest.raises(ValueError, match="site class"):
        replace(mechanism(), site_occupancies={"HA": 3, "HB": 2, "HAA": 2})
    with pytest.raises(ValueError, match="occupancies"):
        replace(mechanism(), site_occupancies={"A": 2})


def test_gas_sites_and_noninteger_import_are_rejected():
    for index, value in ((0, 2), (3, 2.5)):
        payload = authored_declaration(site_occupancy=2)
        payload["species"][index]["sites"] = value
        with pytest.raises(ValueError, match="integer 1/2/3"):
            convert_surface_mechanism(
                payload,
                surface_phase="surface",
                source=SOURCE,
                minimum_temperature_K=300,
                maximum_temperature_K=1900,
            )


def test_undeclared_bep_hindered_and_nonideal_standard_states_are_rejected():
    from chemworld.examples.native_hindered_adsorbate import example_mechanism

    harmonic = example_mechanism()
    with pytest.raises(ValueError, match="standard states"):
        replace(harmonic, site_occupancies={harmonic.adsorbates[0]: 2})
    spec = mechanism()
    matrix = [list(row) for row in spec.interaction_matrix_J_mol]
    matrix[0][0] = 1.0
    with pytest.raises(ValueError, match="nonideal"):
        replace(spec, interaction_matrix_J_mol=tuple(tuple(row) for row in matrix))
    with pytest.raises(ValueError, match="complete imported"):
        replace(spec, site_species_complete=False)


@pytest.mark.parametrize("size", [2, 3])
def test_weighted_steady_chart_preserves_molecular_invariants(size):
    spec = mechanism(size)
    boundary = CatalyticAnalysisConditions(700, {"A": 7e4, "B": 3e4}, {"surface": 1e-6})
    chart = build_catalytic_steady_chart(spec, boundary.capacities_mol, THETA)
    root = solve_catalytic_steady_state(spec, boundary, THETA, target_species="B")
    assert root["status"] == "converged", root
    assert chart.rank == 3
    assert chart.same_class(root["coverages"])
    assert sum(root["coverages"].values()) == pytest.approx(1.0)
    assert sum(
        root["adsorbate_amounts_mol"][k] * spec.site_occupancies[k] for k in spec.adsorbates
    ) == pytest.approx(1e-6)


@pytest.mark.parametrize("size", [2, 3])
def test_world_restore_validates_weighted_original_capacity_and_query_is_read_only(size):
    composition = example_world(site_occupancy=size)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=34)
        reply = runtime.execute({"operation": "wait", "duration_s": 1})
        assert reply.info["transaction_status"] == "committed"
        before = copy.deepcopy(runtime.snapshot().to_dict())
        root = analyze_catalytic_world(composition, runtime._state, target_species="B")
        assert root["status"] == "converged"
        assert runtime.snapshot().to_dict() == before
        bad = copy.deepcopy(before)
        slot = next(iter(bad["world_state"]["state_slots"].values()))
        slot["values"]["HA"] *= 0.9
        with pytest.raises(ValueError):
            broken = ChemWorldRuntime.from_snapshot(bad)
            broken.close()
    finally:
        runtime.close()
