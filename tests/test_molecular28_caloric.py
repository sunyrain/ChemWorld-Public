"""Joint caloric closure and actual original finite-device/session semantics."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("thermo", reason="optional independent reference package")

from chemworld.foundation.state_ledgers import VesselLedger
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_caloric_flash import (
    MolecularCaloricFlashSeed,
    solve_molecular_caloric_flash,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_flash_thermal import (
    MolecularFlashThermalRuntime,
    MolecularFlashThermalSpec,
)
from chemworld.world.molecular_flash_thermal_world import build_molecular_flash_thermal_world
from reference.molecular28_caloric_cases import caloric_spec, seed, thermal_spec
from reference.molecular28_caloric_reference import lle_state, pr_state


def device(mode="heater"):
    return ThermalDeviceSpec(
        conductance_W_K=0.001 if mode == "proportional" else 1.0,
        maximum_power_W=0.02,
        maximum_cooling_power_W=0.02,
        setpoint_K=301.0 if mode == "proportional" else 285.0 if mode == "cooler" else 315.0,
        minimum_setpoint_K=280.0,
        maximum_setpoint_K=320.0,
    )


def world(mode="heater", method="BDF", spec=None):
    spec = spec or thermal_spec(method)
    ref = pr_state(0.6, temperature=300.0, total=0.01, support=0.1)
    feed = {"propane": 0.006, "butane": 0.004}
    return spec, build_molecular_flash_thermal_world(
        spec,
        feed,
        enthalpy_J=ref["H"],
        seed=seed(spec.equilibrium, feed),
        formulas={"propane": "C3H8", "butane": "C4H10"},
        device=device(mode),
    )


def execute(session, action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("kind", ["PR", "LLE"])
@pytest.mark.parametrize("coordinate", ["H", "S"])
def test_joint_caloric_reference(kind, coordinate):
    spec = caloric_spec(kind)
    ref = pr_state(0.6, temperature=300) if kind == "PR" else lle_state(0.4, 300)
    z = 0.6 if kind == "PR" else 0.4
    feed = dict(zip(spec.model.component_ids, (z, 1 - z), strict=True))
    result = solve_molecular_caloric_flash(
        spec, feed, coordinate=coordinate, value=ref[coordinate], seed=seed(spec, feed)
    )
    assert result.success, result.to_dict()
    assert result.temperature_K == pytest.approx(300, abs=2e-4)
    assert result.enthalpy_J == pytest.approx(ref["H"], abs=0.02)
    assert result.entropy_J_K == pytest.approx(ref["S"], abs=2e-4)
    assert all(s.complete and not s.unstable for s in result.post_split_stability)
    json.dumps(result.to_dict(), allow_nan=False)


def test_unresolved_roots_and_stability_remain_failure():
    spec = caloric_spec()
    feed = {"propane": 0.6, "butane": 0.4}
    ref = pr_state(0.6, temperature=300)
    bad = replace(spec, root_policy=replace(spec.root_policy, maximum_iterations=1))
    result = solve_molecular_caloric_flash(
        bad, feed, coordinate="H", value=ref["H"], seed=seed(spec, feed)
    )
    assert not result.success and result.diagnostics["root"]["iterations"] <= 1
    bad = replace(spec, stability_max_iterations=1)
    result = solve_molecular_caloric_flash(
        bad, feed, coordinate="S", value=ref["S"], seed=seed(spec, feed)
    )
    assert not result.success and result.post_split_stability
    assert any(not s.complete for s in result.post_split_stability)


def test_invalid_support_and_pure_branch_rejected():
    spec = caloric_spec()
    feed = {"propane": 1.0, "butane": 0.0}
    with pytest.raises(ValueError, match="two active"):
        solve_molecular_caloric_flash(spec, feed, coordinate="H", value=0.0, seed=seed(spec, feed))
    feed = {"propane": 0.6, "butane": 0.4}
    bad = MolecularCaloricFlashSeed(
        300.0, ({"propane": 0.0, "butane": 0.2}, feed), (250.0, 11500.0)
    )
    with pytest.raises(ValueError, match="positive support"):
        solve_molecular_caloric_flash(spec, feed, coordinate="H", value=0.0, seed=bad)


@pytest.mark.parametrize("mode", ["heater", "cooler"])
def test_real_wait_original_phases_energy_work_and_codec(mode):
    spec, composition = world(mode)
    assert MolecularFlashThermalSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=8)
    try:
        before = session._state
        execute(session, {"operation": "wait", "duration_s": 40})
        after = session._state
        assert set(after.phases.phases) == set(before.phases.phases) == {"liquid", "vapor"}
        assert after.species_amounts == pytest.approx(before.species_amounts, abs=1e-14)
        assert after.state_slots == before.state_slots
        assert after.ledger.time_s == 40
        heat = 0.8 if mode == "heater" else -0.8
        assert after.ledger.energy_jacket_J == pytest.approx(heat, abs=2e-7)
        assert after.ledger.boundary_work_J == pytest.approx(
            spec.pressure_Pa * (after.volume_L - before.volume_L) * 1e-3, abs=1e-9
        )
        assert (after.temperature_K > before.temperature_K) == (mode == "heater")
        result = execute(session, {"operation": "measure", "instrument": "phase_state"})
        assert result.observation is not None
    finally:
        session.close()


def test_pause_resume_exact_snapshot_continuation_and_heat_operation():
    _, composition = world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=8)
    replay = None
    try:
        execute(session, {"operation": "heat", "duration_s": 40, "target_temperature_K": 315})
        execute(session, {"operation": "pause_control"})
        before = session._state
        execute(session, {"operation": "wait", "duration_s": 40})
        assert session._state.temperature_K == pytest.approx(before.temperature_K, abs=1e-8)
        execute(session, {"operation": "resume_control"})
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for runtime in (session, replay):
            execute(runtime, {"operation": "wait", "duration_s": 40})
        assert session._state.to_dict() == replay._state.to_dict()
        assert session._state.ledger.time_s == 120
    finally:
        session.close()
        if replay is not None:
            replay.close()


@pytest.mark.parametrize("boundary", ["capacity", "temperature", "phase_fraction"])
def test_operating_boundary_stops_at_real_time(boundary):
    spec = thermal_spec()
    if boundary == "temperature":
        spec = replace(spec, stop_temperature_K=300.1)
    if boundary == "phase_fraction":
        spec = replace(spec, stop_phase_fraction=0.409)
    spec, composition = world("cooler" if boundary == "phase_fraction" else "heater", spec=spec)
    state = composition.physics.initial_states["reactor"]
    if boundary == "capacity":
        record = state.vessels.vessels["reactor"]
        state = state.replace(
            vessels=VesselLedger({"reactor": replace(record, max_volume_L=state.volume_L * 1.001)})
        )
    adapter = MolecularFlashThermalRuntime(composition)
    result = adapter.advance(state, {"duration_s": 200})
    assert result.success and result.outcome == "stopped", result.diagnostics
    assert 0 < result.actual_duration_s < 200
    assert result.state.ledger.time_s == result.actual_duration_s
    assert (
        result.diagnostics["stopping_event"]
        == {
            "capacity": "vessel_capacity",
            "temperature": "temperature_limit",
            "phase_fraction": "two_phase_operating_limit",
        }[boundary]
    )


def test_unsupported_transfer_and_unresolved_constitutive_domain_preserve_state():
    _, composition = world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=8)
    try:
        before = session._state.to_dict()
        result = session.execute(
            {"operation": "add_component", "component": "propane", "amount": 0.1}
        )
        assert result.info["transaction_status"] != "committed"
        for key in ("species", "phases", "volume_L", "temperature_K", "pressure_Pa"):
            assert session._state.to_dict()[key] == before[key]
        assert session._state.ledger.time_s == 0
        before = session._state.to_dict()
        adapter = MolecularFlashThermalRuntime(composition)
        result = adapter.advance(session._state, {"duration_s": 1}, jacket_program={})
        assert not result.success and result.state is None
        assert session._state.to_dict() == before
    finally:
        session.close()
