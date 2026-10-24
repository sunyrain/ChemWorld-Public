"""Finite water/solute/enthalpy throughflow consumes the original aqueous world."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.aqueous_flow_column import example_inputs
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    advance_aqueous_reactive_transport,
    aqueous_transport_faces,
    prepare_aqueous_transport_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_reactive_transport_world import build_aqueous_reactive_transport_world


def members(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def test_mechanical_dispersion_does_not_modify_ionic_electric_mobility():
    spec, n = example_inputs("salt", 1, 4)
    for key in ("M+", "A-"):
        n["cell-00"][key] *= 2
    base = aqueous_transport_faces(replace(spec, flow=replace(spec.flow, dispersion_m2_s=0.0)), n)
    changed = aqueous_transport_faces(spec, n)
    first, second = base["faces"]["aqueous-face:1"], changed["faces"]["aqueous-face:1"]
    assert first["potential_difference_V"] == second["potential_difference_V"] != 0
    assert (
        first["electrodiffusion_amount_rates_mol_s"]
        == second["electrodiffusion_amount_rates_mol_s"]
    )
    assert first["amount_rates_mol_s"]["M+"] != second["amount_rates_mol_s"]["M+"]
    assert abs(second["current_molar_charge_rate_mol_s"]) < 1e-17
    for row in changed["faces"].values():
        assert row["amount_rates_mol_s"]["H2O"] == pytest.approx(spec.flow.flow_m3_s / 18e-6)
        assert row["enthalpy_W"] == pytest.approx(row["amount_rates_mol_s"]["H2O"] * 1000)


@pytest.mark.parametrize("kind", ["salt", "fast", "slow"])
def test_finite_empty_receiver_gains_actual_water_and_atoms(kind):
    spec, n = example_inputs(kind, 0, 4)
    initial = prepare_aqueous_transport_state(spec, n)
    result = advance_aqueous_reactive_transport(spec, initial, 1.0)
    final = result.final
    assert not any(initial["collector"].values())
    assert final["amounts_by_vessel"]["collector"]["H2O"] > 0
    assert final["amounts_by_vessel"]["collector"]["A-"] > 0
    for v in spec.cell_vessel_ids:
        assert final["amounts_by_vessel"][v]["H2O"] == initial[v]["H2O"]

    def total(stocks, element):
        return sum(
            s.composition.get(element, 0) * stocks[v][s.species_id]
            for s in spec.equilibrium.database.species
            for v in spec.member_vessel_ids
        )

    for element in {e for s in spec.equilibrium.database.species for e in s.composition}:
        assert total(final["amounts_by_vessel"], element) == pytest.approx(
            total(initial, element), abs=2e-10
        )
    for row in final["cells"].values():
        assert row["internal_energy_change_J"] == pytest.approx(
            row["bath_heat_J"] + row["transported_enthalpy_J"] - row["boundary_work_J"], abs=3e-6
        )
    assert sum(row["transported_enthalpy_J"] for row in final["cells"].values()) == pytest.approx(
        0.0, abs=3e-6
    )
    assert sum(row["boundary_work_J"] for row in final["cells"].values()) == pytest.approx(
        0.0, abs=3e-6
    )


@pytest.mark.parametrize("boundary", ["feed", "collector"])
def test_finite_boundary_stops_entire_original_world(boundary):
    spec, n = example_inputs("salt", 1, 4)
    if boundary == "feed":
        spec = replace(spec, flow=replace(spec.flow, minimum_feed_volume_m3=1e-7))
        n["feed"] = {k: v * 3e-7 / 5e-6 for k, v in n["feed"].items()}
        expected = 40.0
    else:
        spec = replace(spec, flow=replace(spec.flow, collector_capacity_m3=1e-7))
        expected = 20.0
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    try:
        runtime.reset(seed=40)
        result = runtime.execute({"operation": "wait", "duration_s": 60.0})
        assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.ledger.time_s == pytest.approx(expected, abs=1e-10)
        assert all(
            runtime._state.vessel_elapsed_s[v] == pytest.approx(expected)
            for v in spec.member_vessel_ids
        )
        before = {v: s.species_amounts for v, s in members(runtime).items()}
        runtime.execute({"operation": "wait", "duration_s": 2.0})
        assert runtime._state.ledger.time_s == pytest.approx(expected, abs=1e-10)
        assert {v: s.species_amounts for v, s in members(runtime).items()} == before
        assert runtime._state.ledger.flow_shaft_work_J == 0
    finally:
        runtime.close()


def test_full_world_nonanchor_json_restores_current_finite_reservoirs():
    spec, n = example_inputs("slow")
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    restored = None
    try:
        runtime.reset(seed=40)
        runtime.execute({"operation": "select_vessel", "vessel_id": "cell-02"})
        first = runtime.execute({"operation": "wait", "duration_s": 1.0})
        assert first.info["transaction_status"] == "committed"
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            result = world.execute({"operation": "wait", "duration_s": 1.0})
            assert result.info["transaction_status"] == "committed", result.info
            assert world._state.ledger.time_s == 2
            for local in members(world).values():
                assert local.species_amounts == local.phases.phases["aqueous"].species_amounts_mol
                assert local.volume_L == pytest.approx(local.species_amounts["H2O"] * 18e-6 * 1000)
        assert restored.snapshot().to_dict() == runtime.snapshot().to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("kind", ["charge", "dry_solute", "capacity", "feed"])
def test_invalid_reservoirs_never_receive_implicit_water_or_charge(kind):
    spec, n = example_inputs("salt")
    if kind == "charge":
        n["feed"]["M+"] *= 2
    elif kind == "dry_solute":
        n["collector"]["M+"] = n["collector"]["A-"] = 1e-8
    elif kind == "capacity":
        n["collector"] = {k: 2 * v for k, v in n["feed"].items()}
    else:
        n["feed"] = {k: 0.01 * v for k, v in n["feed"].items()}
    before = json.dumps(n, sort_keys=True)
    with pytest.raises(ValueError):
        prepare_aqueous_transport_state(spec, n)
    assert json.dumps(n, sort_keys=True) == before


def test_versioned_flow_codec_and_closed_v1_remain_distinct():
    spec, _ = example_inputs("salt")
    assert spec.to_dict()["model"].endswith("v2")
    assert (
        AqueousReactiveTransportSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict()
        == spec.to_dict()
    )
    closed = replace(spec, flow=None)
    assert closed.to_dict()["model"].endswith("v1") and "flow" not in closed.to_dict()
    bad = spec.to_dict()
    bad["model"] = "finite-aqueous-reactive-transport-v1"
    with pytest.raises(ValueError, match="schema"):
        AqueousReactiveTransportSpec.from_dict(bad)


def test_fast_failure_rolls_back_all_finite_members_and_heat(monkeypatch):
    import chemworld.physchem.aqueous_reactive_transport as transport

    spec, n = example_inputs("slow")
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    try:
        runtime.reset(seed=40)
        before = {v: s.species_amounts for v, s in members(runtime).items()}

        def fail(*args, **kwargs):
            raise ValueError("injected fast closure failure")

        monkeypatch.setattr(transport, "solve_aqueous_reaction_equilibrium", fail)
        reply = runtime.execute({"operation": "wait", "duration_s": 1.0})
        assert reply.info["transaction_status"] != "committed"
        assert {v: s.species_amounts for v, s in members(runtime).items()} == before
        assert runtime._state.ledger.time_s == runtime._state.ledger.energy_jacket_J == 0
    finally:
        runtime.close()
