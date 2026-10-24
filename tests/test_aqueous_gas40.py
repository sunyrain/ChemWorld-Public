"""Finite gas/liquid inventory, thermodynamic reference and original world checks."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.examples.aqueous_headspace import example_inputs
from chemworld.physchem.aqueous_gas_transfer import (
    AqueousHeadspaceSpec,
    advance_aqueous_headspace,
    headspace_pressure,
    headspace_properties,
    headspace_transfer_rates,
    prepare_aqueous_headspace_state,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_gas_transfer import AqueousGasTransferRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


@pytest.mark.parametrize("condition", [0, 1])
def test_actual_gas_pressure_changes_liquid_enthalpy_but_not_fixed_water_internal_energy(condition):
    spec, n = example_inputs("neutral", condition)
    before = headspace_properties(spec, n)
    changed = {**n, "I(g)": 2 * n["I(g)"]}
    after = headspace_properties(spec, changed)
    delta_pressure = after["pressure_Pa"] - before["pressure_Pa"]
    assert delta_pressure == pytest.approx(8e4)
    assert after["liquid"]["enthalpy_J"] - before["liquid"]["enthalpy_J"] == pytest.approx(
        spec.liquid_volume_m3 * delta_pressure, abs=1e-10
    )
    assert after["liquid"]["internal_energy_J"] == pytest.approx(
        before["liquid"]["internal_energy_J"], abs=1e-10
    )
    assert after["gas"]["internal_energy_J"] == pytest.approx(
        after["gas"]["enthalpy_J"] - after["pressure_Pa"] * spec.gas_volume_m3, abs=1e-10
    )


@pytest.mark.parametrize("condition", [0, 1])
def test_forward_reverse_transfer_uses_same_gibbs_difference(condition):
    spec, n = example_inputs("double", condition)
    for row in headspace_transfer_rates(spec, n).values():
        assert row["forward_mol_s"] / row["reverse_mol_s"] == pytest.approx(
            exp(-row["reaction_gibbs_J_mol"] / (R * spec.temperature_K)), rel=1e-13
        )
        assert -row["net_mol_s"] * row["reaction_gibbs_J_mol"] >= 0


def test_zero_individual_gas_desorbs_without_adding_a_seed():
    spec, n = example_inputs("desorption")
    assert n["A(g)"] == 0
    result = advance_aqueous_headspace(spec, n, 5.0)
    assert result.trajectory[0]["amounts_mol"]["A(g)"] == 0
    assert result.final["amounts_mol"]["A(g)"] > 0
    assert result.final["amounts_mol"]["A(g)"] + result.final["amounts_mol"][
        "A(aq)"
    ] == pytest.approx(0.002, abs=1e-15)
    assert n["A(g)"] == 0


def test_fixed_water_is_exact_through_fast_logarithmic_closure():
    spec, n = example_inputs("double", 1)
    result = advance_aqueous_headspace(spec, n, 120.0, evaluation_times_s=(1.0, 30.0, 100.0))
    assert all(p["amounts_mol"]["H2O"] == n["H2O"] for p in result.trajectory)
    assert all(p["amounts_mol"]["I(g)"] == n["I(g)"] for p in result.trajectory)


def test_all_gases_zero_is_an_explicit_pressure_domain_failure():
    spec, n = example_inputs("neutral")
    for gas in spec.gas_species_ids:
        n[gas] = 0.0
    with pytest.raises(ValueError, match="no gas seeding"):
        prepare_aqueous_headspace_state(spec, n)


def test_water_cannot_be_replaced_by_declared_geometry():
    spec, n = example_inputs("neutral")
    n["H2O"] *= 0.9
    with pytest.raises(ValueError, match="actual water volume"):
        prepare_aqueous_headspace_state(spec, n)


def test_full_world_restores_current_pressure_and_phase_stocks():
    spec, n = example_inputs("acid")
    runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, n))
    restored = None
    try:
        runtime.reset(seed=40)
        initial = runtime._state.pressure_Pa
        first = runtime.execute({"operation": "wait", "duration_s": 5.0})
        assert first.info["transaction_status"] == "committed", first.info
        assert runtime._state.pressure_Pa < initial
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            result = world.execute({"operation": "wait", "duration_s": 5.0})
            assert result.info["transaction_status"] == "committed", result.info
            assert world._state.phases.total_amounts_mol() == world._state.species_amounts
            assert world._state.pressure_Pa == headspace_pressure(
                spec, world._state.species_amounts
            )
            assert world._state.vessels.vessels["reactor"].pressure_Pa == world._state.pressure_Pa
        assert runtime.snapshot().to_dict() == restored.snapshot().to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("bad", ["phase", "temperature"])
def test_current_owner_validation_does_not_reconstruct_historical_inventory(bad):
    spec, n = example_inputs("acid")
    composition = build_aqueous_headspace_world(spec, n)
    provider = AqueousGasTransferRuntime(composition)
    state = composition.physics.initial_states["reactor"]
    if bad == "phase":
        phase = replace(
            state.phases.phases["headspace"], species_amounts_mol={"I(g)": n["I(g)"], "A(g)": 0.0}
        )
        state = state.replace(
            phases=replace(state.phases, phases={**state.phases.phases, "headspace": phase})
        )
    else:
        state = state.replace(temperature_K=spec.temperature_K + 1)
    with pytest.raises(ValueError):
        provider.validate_state(state)


def test_invalid_transfer_cannot_change_the_declared_reaction():
    spec, _ = example_inputs("acid")
    with pytest.raises(ValueError, match="neutral gas"):
        replace(spec, transfers=(replace(spec.transfers[0], reaction_id="acid"),))


def test_negative_transfer_and_wrong_schema_are_rejected():
    spec, _ = example_inputs("neutral")
    with pytest.raises(ValueError):
        replace(spec.transfers[0], coefficient_mol_s=-1)
    payload = spec.to_dict()
    payload["model"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        AqueousHeadspaceSpec.from_dict(payload)


def test_failed_fast_closure_rolls_back_phases_pressure_time_and_heat(monkeypatch):
    import chemworld.physchem.aqueous_gas_transfer as physical

    spec, n = example_inputs("acid")
    runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, n))
    try:
        runtime.reset(seed=40)
        before = runtime._state

        def fail(*args, **kwargs):
            raise ValueError("injected fast failure")

        monkeypatch.setattr(physical, "solve_aqueous_reaction_equilibrium", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 5.0})
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.phases == before.phases
        assert runtime._state.pressure_Pa == before.pressure_Pa
        assert runtime._state.ledger.time_s == runtime._state.ledger.energy_jacket_J == 0
    finally:
        runtime.close()
