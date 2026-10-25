"""Empirical TDS stays a conserved mass; source seawater laws remain explicit."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.seawater_ro import seawater_ro_profile
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.seawater_ro import unpack_seawater_ro
from chemworld.world.seawater_ro_world import seawater_ro_world_spec
from reference.ro47close_fixtures import seawater_spec


def test_seawater_actual_profile_canary():
    model = seawater_spec()
    feed = {"H2O": 0.965, "TDS": 0.035}
    volume = model.properties.volume(feed, model.temperature_K, model.reservoir_pressure_Pa)
    result = seawater_ro_profile(model, feed, volume)
    assert 0 < result["water_recovery"] < 1
    assert result["pressure_drop_Pa"] > 0
    assert len(result["stations"]) == 3
    for station in result["stations"]:
        assert station["maximum_scaled_residual"] < 1e-8


def test_zero_salt_limit_preserves_exact_zero_tds():
    model = seawater_spec()
    feed = {"H2O": 1.0, "TDS": 0.0}
    result = seawater_ro_profile(model, feed, model.properties.volume(feed, 298.15, 101325.0))
    assert result["permeate_kg_s"]["TDS"] == 0
    assert result["retentate_kg_s"]["TDS"] == 0
    for row in result["stations"]:
        assert row["tds_flux_kg_m2_s"] == 0
        assert row["interface_salinity_kg_kg"] == 0
        assert row["permeate_salinity_kg_kg"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_seawater_original_slots_world_and_resume(shared):
    spec = replace(seawater_spec(), membrane_area_m2=0.001, membrane_length_m=0.05)
    runtime = ChemWorldRuntime(
        composition=seawater_ro_world_spec(spec, {"H2O": 0.965, "TDS": 0.035}, shared_clock=shared)
    )
    restored = None
    try:
        runtime.reset(seed=47)
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "wait", "duration_s": 2},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for current in (runtime, restored):
            result = current.execute({"operation": "wait", "duration_s": 2})
            assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.to_dict() == restored._state.to_dict()
        _, _, inventory = unpack_seawater_ro(runtime._state)
        assert sum(n["TDS"] for n in inventory.masses_kg) == pytest.approx(0.035)
        assert sum(n["H2O"] for n in inventory.masses_kg) == pytest.approx(0.965)
        assert runtime._state.ledger.flow_pump_work_J > 0
    finally:
        runtime.close()
        if restored:
            restored.close()
