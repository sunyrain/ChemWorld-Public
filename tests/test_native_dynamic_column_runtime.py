"""Actual native column actions, continuation, ownership and rollback."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_distillation_research import (
    initial_column_state,
    synthetic_column_spec,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.dynamic_column import read_column, validate_column_state
from chemworld.world.dynamic_column_world import dynamic_column_world_spec


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.fixture(scope="module")
def warm_snapshot():
    spec = synthetic_column_spec()
    session = ChemWorldRuntime(
        composition=dynamic_column_world_spec(spec, initial_column_state(spec))
    )
    try:
        session.reset(seed=20)
        execute(
            session, operation="distill", target_temperature_K=430, duration_s=120, reflux_ratio=1
        )
        return json.loads(json.dumps(session.snapshot().to_dict()))
    finally:
        session.close()


@pytest.fixture
def warm_session(warm_snapshot):
    session = ChemWorldRuntime.from_snapshot(warm_snapshot)
    yield session
    session.close()


def energy(state):
    return sum(s.values["enthalpy_J"] for s in state.state_slots.slots)


def test_actual_distill_has_timed_fractions_and_one_material_inventory(warm_session):
    state = warm_session._state
    fractions = state.metadata["column_timed_fractions"]
    assert state.ledger.time_s == 120
    assert len(fractions) == 60
    assert [f["end_s"] for f in fractions] == list(range(2, 121, 2))
    for key in ("light", "heavy"):
        assert sum(f["amounts_mol"][key] for f in fractions) == pytest.approx(
            state.phases.phases["distillate"].species_amounts_mol[key], abs=1e-10
        )
        assert state.species_amounts[key] == pytest.approx(
            sum(p.species_amounts_mol[key] for p in state.phases.phases.values()), abs=1e-12
        )
    assert all(set(s.values) == {"enthalpy_J"} for s in state.state_slots.slots)
    assert "column_timed_fractions" not in state.equipment.equipment["distillation_column"].settings


def test_measure_and_collection_settle_real_clock_and_phase_enthalpy(warm_session):
    session = warm_session
    before = session._state
    result = execute(session, operation="measure", instrument="phase_state")
    assert result.info["actual_duration_s"] == 1
    assert session._state.ledger.time_s == 121
    assert session.observe().channels[0].sampled_at.time_s == 121
    actual_heat = (
        session._state.ledger.energy_jacket_J
        - before.ledger.energy_jacket_J
        - session._state.ledger.heat_loss_J
        + before.ledger.heat_loss_J
    )
    assert energy(session._state) - energy(before) == pytest.approx(actual_heat, abs=1e-6)
    before = session._state
    execute(session, operation="collect_fraction", transfer_fraction=0.5)
    after = session._state
    assert after.ledger.time_s == before.ledger.time_s
    assert energy(after) == pytest.approx(energy(before), abs=1e-8)
    assert after.species_amounts == pytest.approx(before.species_amounts, abs=1e-12)
    for key in ("light", "heavy"):
        assert after.phases.phases["collected_fraction"].species_amounts_mol[key] == pytest.approx(
            0.5 * before.phases.phases["distillate"].species_amounts_mol[key], abs=1e-12
        )


def test_reflux_changes_and_exact_snapshot_replay(warm_session):
    session = warm_session
    other = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(session.snapshot().to_dict())))
    try:
        action = {
            "operation": "distill",
            "target_temperature_K": 430,
            "duration_s": 3,
            "reflux_ratio": 3,
        }
        execute(session, **action)
        execute(other, **action)
        assert session._state.to_dict() == other._state.to_dict()
        assert (
            session._state.equipment.equipment["distillation_column"].settings["reflux_ratio"] == 3
        )
        before = session._state.phases.phases["distillate"].species_amounts_mol.copy()
        execute(
            session,
            operation="distill",
            target_temperature_K=430,
            duration_s=2,
            reflux_ratio=0,
            full_reflux=1,
        )
        assert session._state.phases.phases["distillate"].species_amounts_mol == before
    finally:
        other.close()


def test_material_reset_replaces_fluid_keeps_body_heat_and_device_settings(warm_session):
    session = warm_session
    before = session._state
    body_h = sum(
        s.values["enthalpy_J"] for s in before.state_slots.slots if s.spec.owner_kind == "vessel"
    )
    initial = synthetic_column_spec()
    template = dynamic_column_world_spec(
        initial, initial_column_state(initial)
    ).physics.initial_states["column"]
    incoming_fluid_h = sum(
        s.values["enthalpy_J"]
        for s in template.state_slots.slots
        if s.spec.owner_kind == "material"
    )
    session.reset(options={"scope": "material", "vessel_ids": ["column"]})
    state = session._state
    assert state.ledger == before.ledger
    assert state.equipment == before.equipment
    assert state.species_amounts == pytest.approx(template.species_amounts, abs=1e-12)
    assert energy(state) == pytest.approx(body_h + incoming_fluid_h, abs=1e-6)
    assert state.temperature_K > 300
    assert "column_timed_fractions" not in state.metadata
    validate_column_state(state, initial)
    execute(session, operation="wait", duration_s=1, stirring_speed_rpm=600)


def test_numerical_failure_preserves_material_enthalpy_and_clock(warm_session, monkeypatch):
    import chemworld.runtime.dynamic_column as column

    session = warm_session
    before = session._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected coupled column failure")

    monkeypatch.setattr(column, "advance_dynamic_column", fail)
    result = session.execute(
        {"operation": "distill", "target_temperature_K": 430, "duration_s": 5, "reflux_ratio": 1}
    )
    assert result.info["operation_outcome"] == "numerical_failure"
    assert result.info["transaction_status"] == "rolled_back"
    assert session._state.phases == before.phases
    assert session._state.state_slots == before.state_slots
    assert session._state.ledger.time_s == before.ledger.time_s
    assert session._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J


def test_finite_collector_stop_commits_only_actual_interval():
    spec = replace(synthetic_column_spec(), collector_capacity_mol=1e-6)
    session = ChemWorldRuntime(
        composition=dynamic_column_world_spec(spec, initial_column_state(spec))
    )
    try:
        session.reset(seed=20)
        result = execute(
            session, operation="distill", target_temperature_K=430, duration_s=120, reflux_ratio=0
        )
        assert result.info["operation_outcome"] == "stopped"
        assert result.info["outcome_diagnostics"]["stopping_event"] == "collector_capacity"
        assert 0 < result.info["actual_duration_s"] < 120
        assert session._state.ledger.time_s == result.info["actual_duration_s"]
        assert session._state.ledger.energy_jacket_J == pytest.approx(
            5 * result.info["actual_duration_s"]
        )
        column = read_column(session._state, spec)
        assert sum(column.amounts_mol[-1].values()) == pytest.approx(1e-6, abs=1e-12)
    finally:
        session.close()


def test_native_vessel_volume_stops_at_actual_phase_volume():
    spec = synthetic_column_spec()
    session = ChemWorldRuntime(
        composition=dynamic_column_world_spec(spec, initial_column_state(spec), capacity_L=0.0015)
    )
    try:
        session.reset(seed=20)
        result = execute(
            session, operation="distill", target_temperature_K=430, duration_s=120, reflux_ratio=0
        )
        assert result.info["operation_outcome"] == "stopped"
        assert result.info["outcome_diagnostics"]["stopping_event"] == "vessel_volume"
        assert session._state.volume_L == pytest.approx(0.0015, abs=1e-9)
        assert 0 < session._state.ledger.time_s < 120
    finally:
        session.close()
