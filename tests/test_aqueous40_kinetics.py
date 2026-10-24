"""Finite database minerals through the real session, phase ledger and clock."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state import Ledger
from chemworld.foundation.state_ledgers import PhaseLedger, VesselLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.aqueous_mineral_equilibrium import aqueous_mineral_properties
from chemworld.physchem.aqueous_mineral_kinetics import (
    AqueousMineralKineticsSpec,
    advance_aqueous_minerals,
)
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral_kinetics import DatabaseMineralKineticsRuntime
from chemworld.world.aqueous_mineral_kinetics_world import build_aqueous_mineral_kinetics_world
from reference.aqueous40_kinetics_cases import cases, inputs


def world(case=None):
    case = case or cases()[0]
    spec, amounts, geometries = inputs(case)
    return spec, build_aqueous_mineral_kinetics_world(
        spec,
        amounts,
        geometries,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )


def execute(session, duration):
    result = session.execute({"operation": "wait", "duration_s": duration})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("case_index", [0, 2, 3, 8])
def test_real_wait_updates_original_stocks_heat_work_and_preserves_geometry(case_index):
    case = cases()[case_index]
    spec, composition = world(case)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=19)
    try:
        before = session._state
        props0 = aqueous_mineral_properties(
            spec.equilibrium,
            before.species_amounts,
            temperature_K=before.temperature_K,
            pressure_Pa=before.pressure_Pa,
        )
        execute(session, 100)
        after = session._state
        props1 = aqueous_mineral_properties(
            spec.equilibrium,
            after.species_amounts,
            temperature_K=after.temperature_K,
            pressure_Pa=after.pressure_Pa,
        )
        heat = after.ledger.energy_jacket_J - after.ledger.heat_loss_J
        assert after.ledger.time_s == 100
        assert after.state_slots == before.state_slots
        assert set(after.phases.phases) == set(before.phases.phases)
        assert props1.enthalpy_J - props0.enthalpy_J == pytest.approx(heat, abs=1e-8)
        assert props1.internal_energy_J - props0.internal_energy_J == pytest.approx(
            heat - after.ledger.boundary_work_J,
            abs=1e-8,
        )
        assert after.ledger.heat_reaction_J == 0
        mineral = spec.equilibrium.mineral_ids[0]
        if case_index == 2:
            assert after.species_amounts[mineral] > before.species_amounts[mineral]
            assert after.ledger.boundary_work_J < 0
        elif case_index == 3:
            assert after.species_amounts == before.species_amounts
            assert heat == 0 and after.ledger.boundary_work_J == 0
        else:
            assert after.species_amounts[mineral] < before.species_amounts[mineral]
            assert after.ledger.boundary_work_J > 0
    finally:
        session.close()


def test_full_exhaustion_and_following_wait_keep_exact_zero_and_full_clock():
    _, composition = world(cases()[1])
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=11)
    try:
        geometry = session._state.state_slots
        events = []
        for duration in (10000, 10000, 5000):
            execute(session, duration)
            events.extend(session._state.metadata["last_declared_mineral_advance"]["events"])
        assert session._state.species_amounts["NaCl(s)"] == 0
        assert len(events) == 1
        assert events[0]["time_s"] < 10000
        energy = session._state.ledger.energy_jacket_J
        execute(session, 100)
        assert session._state.ledger.time_s == 25100
        assert session._state.ledger.energy_jacket_J == energy
        assert session._state.state_slots == geometry
    finally:
        session.close()


def test_exact_json_snapshot_continuation_and_legacy_missing_work_default():
    spec, composition = world(cases()[6])
    clone = AqueousMineralKineticsSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert json.loads(json.dumps(clone.to_dict())) == json.loads(json.dumps(spec.to_dict()))
    assert Ledger(**{}).boundary_work_J == 0
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=23)
    replay = None
    try:
        execute(session, 20)
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        before = session._state.ledger
        left, right = execute(session, 40), execute(replay, 40)
        assert left.info == right.info
        assert session._state.to_dict() == replay._state.to_dict()
        receipt = session._state.metadata["last_declared_mineral_advance"]["trajectory"][-1]
        after = session._state.ledger
        assert after.boundary_work_J - before.boundary_work_J == pytest.approx(
            receipt["boundary_work_J"], abs=1e-12
        )
        assert after.time_s - before.time_s == 40
    finally:
        session.close()
        if replay is not None:
            replay.close()


@pytest.mark.parametrize("defect", ["geometry", "capacity", "temperature", "solver"])
def test_failed_real_wait_preserves_inventory_geometry_clock_and_energy(defect, monkeypatch):
    _, composition = world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=3)
    try:
        state = session._state
        if defect == "geometry":
            session._state = state.replace(state_slots=StateSlots(()))
        elif defect == "capacity":
            vessel = state.vessels.vessels[state.vessel_id]
            session._state = state.replace(
                vessels=VesselLedger({state.vessel_id: replace(vessel, max_volume_L=0.001)})
            )
        elif defect == "temperature":
            session._state = state.replace(temperature_K=310)
        else:
            from chemworld.physchem import solver_backend

            def fail(*args, **kwargs):
                raise RuntimeError("injected common ODE failure")

            monkeypatch.setattr(solver_backend, "solve_ode", fail)
        before = session._state
        result = session.execute({"operation": "wait", "duration_s": 100})
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        for key in (
            "time_s",
            "energy_jacket_J",
            "heat_loss_J",
            "heat_reaction_J",
            "boundary_work_J",
        ):
            assert getattr(session._state.ledger, key) == getattr(before.ledger, key)
    finally:
        session.close()


def test_unsupported_heat_transfer_and_declared_no_surface_hold():
    spec, composition = world()
    state = composition.physics.initial_states["reactor"]
    adapter = DatabaseMineralKineticsRuntime()
    assert not adapter.advance(state, {"duration_s": 10}, heat=True).success
    assert not adapter.advance(state, {"operation": "route_material", "duration_s": 10}).success
    assert not adapter.advance(state, {"duration_s": 10}, jacket_program={}).success
    _, amounts, geometries = inputs(cases()[0])
    geometries = {k: MineralGeometry(0, 0.01) for k in geometries}
    result = advance_aqueous_minerals(
        spec, amounts, geometries, 10, temperature_K=298.15, pressure_Pa=1e5
    )
    assert result.final.amounts_mol == amounts
    assert result.actual_duration_s == 10


def test_kinetic_fast_equilibrium_does_not_erase_slow_mineral_history():
    case = cases()[6]
    spec, amounts, geometry = inputs(case)
    result = advance_aqueous_minerals(
        spec, amounts, geometry, 0.001, temperature_K=298.15, pressure_Pa=1e5
    )
    assert result.final.amounts_mol["NaCl(s)"] == pytest.approx(amounts["NaCl(s)"], abs=1e-7)
    water_kg = result.final.amounts_mol["H2O"] * 0.018015
    final = result.final.amounts_mol
    assert final["NaCl(aq)"] == pytest.approx(
        10 * final["Na+"] * final["Cl-"] / water_kg, abs=1e-10
    )


def test_mismatched_original_phase_owner_is_rejected():
    _, composition = world()
    state = composition.physics.initial_states["reactor"]
    phases = dict(state.phases.phases)
    key = "mineral:NaCl(s)"
    phases[key] = replace(phases[key], phase_type="liquid")
    with pytest.raises(ValueError, match="pure mineral"):
        DatabaseMineralKineticsRuntime().validate_state(state.replace(phases=PhaseLedger(phases)))


def test_precipitation_draws_from_fast_complex_without_clipping_free_ions():
    spec, amounts, geometry = inputs(cases()[6])
    result = advance_aqueous_minerals(
        spec,
        amounts,
        geometry,
        50000,
        temperature_K=298.15,
        pressure_Pa=1e5,
    )
    final = result.final.amounts_mol
    assert result.diagnostics["feasible_liquid_initializations"] > 0
    assert final["NaCl(s)"] - amounts["NaCl(s)"] > amounts["Na+"]
    assert all(n >= 0 for n in final.values())
    assert final["Na+"] + final["NaCl(aq)"] + final["NaCl(s)"] == pytest.approx(0.055, abs=1e-10)
