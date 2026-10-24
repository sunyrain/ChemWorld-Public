"""Steady equipment actually feeds original inventory, event clock and utility ledger."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger
from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.runtime.molecular_process_path import (
    MolecularProcessPath,
    advance_molecular_process_receiver,
)
from reference.run_molecular_processes import path_and_world


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("form", ["ode", "residual"])
def test_native_finite_receiver_counts_heat_work_once_and_stops(real, form):
    path, state = path_and_world(real)
    original = state.to_dict(include_hidden=True)
    result = advance_molecular_process_receiver(
        state, path, 10, phase_id="bulk", maximum_amount_mol=1.1, numerical_form=form
    )
    assert result.success, result.diagnostics
    assert result.outcome == "stopped" and result.actual_duration_s == pytest.approx(5, abs=1e-7)
    assert result.state.species_amounts == pytest.approx({"A": 0.275, "B": 0.825}, abs=1e-8)
    assert result.state.phases.phases["bulk"].species_amounts_mol == result.state.species_amounts
    assert result.state.ledger.flow_heat_J == pytest.approx(50, abs=2e-5)
    assert result.state.ledger.flow_shaft_work_J > 0
    assert result.state.ledger.energy_jacket_J == 0
    assert result.state.thermal.vessels["receiver"].energy_jacket_J == 0
    assert result.diagnostics["receiver_energy_residual_J"] == pytest.approx(0, abs=2e-5)
    assert result.diagnostics["path_energy_residual_J"] == pytest.approx(0, abs=2e-5)
    assert sum(row["amount"] for row in result.outputs["actual_exchanges"]) == pytest.approx(0.1)
    assert result.state.ledger.time_s == pytest.approx(5)
    assert state.to_dict(include_hidden=True) == original
    json.dumps(result.diagnostics, allow_nan=False)


def test_backpressure_event_shortens_all_resources_and_failure_preserves_world():
    path, state = path_and_world(pressure_limit=135000.0)
    result = advance_molecular_process_receiver(
        state, path, 10, phase_id="bulk", maximum_amount_mol=2.0
    )
    assert result.success, result.diagnostics
    assert result.diagnostics["stopping_event"] == "receiver_pressure_limit"
    assert 0 < result.actual_duration_s < 5
    assert result.state.pressure_Pa == pytest.approx(135000.0, rel=1e-8)
    assert result.state.ledger.flow_heat_J == pytest.approx(10 * result.actual_duration_s)
    old = state.to_dict(include_hidden=True)
    bad_path = replace(path, devices=(MolecularHeatDuty(path.domain, 1e9),))
    failed = advance_molecular_process_receiver(
        state, bad_path, 10, phase_id="bulk", maximum_amount_mol=2.0
    )
    assert not failed.success and failed.state is None and failed.actual_duration_s == 0
    assert "root" in failed.diagnostics["steady_path"]["diagnostics"]["devices"][0]["diagnostics"]
    assert state.to_dict(include_hidden=True) == old


def test_native_snapshot_and_path_continuation_reconstruct_current_inventory():
    path, state = path_and_world(True)
    first = advance_molecular_process_receiver(
        state, path, 2, phase_id="bulk", maximum_amount_mol=1.1
    )
    assert first.success, first.diagnostics
    restored = WorldState.from_dict(
        json.loads(json.dumps(first.state.to_dict(include_hidden=True)))
    )
    decoded = MolecularProcessPath.from_dict(json.loads(json.dumps(path.to_dict())))
    resumed = advance_molecular_process_receiver(
        restored, decoded, 8, phase_id="bulk", maximum_amount_mol=1.1
    )
    full = advance_molecular_process_receiver(
        state, path, 10, phase_id="bulk", maximum_amount_mol=1.1
    )
    assert resumed.success and full.success
    assert resumed.actual_duration_s == pytest.approx(3, abs=1e-7)
    assert resumed.state.species_amounts == pytest.approx(full.state.species_amounts, abs=1e-8)
    assert resumed.state.temperature_K == pytest.approx(full.state.temperature_K, abs=2e-6)
    for key in ("time_s", "flow_heat_J", "flow_shaft_work_J"):
        assert getattr(resumed.state.ledger, key) == pytest.approx(getattr(full.state.ledger, key))
    assert decoded == path


def test_receiver_temperature_event_and_signed_upstream_utilities():
    path, state = path_and_world()
    hardware = state.vessels.vessels["receiver"]
    limited = state.replace(
        vessels=VesselLedger({"receiver": replace(hardware, max_temperature_K=305.0)})
    )
    result = advance_molecular_process_receiver(
        limited, path, 10, phase_id="bulk", maximum_amount_mol=2.0
    )
    assert result.success, result.diagnostics
    assert result.diagnostics["stopping_event"] == "receiver_temperature_max"
    assert result.state.temperature_K == pytest.approx(305.0, abs=1e-7)
    assert 0 < result.actual_duration_s < 5
    recovery = replace(
        path,
        devices=(
            MolecularPressureStage(path.domain, "expander", 1.5e5, 0.8),
            MolecularHeatDuty(path.domain, -5.0, boundary_temperature_K=250.0),
        ),
    )
    result = advance_molecular_process_receiver(
        state, recovery, 2.0, phase_id="bulk", maximum_amount_mol=2.0
    )
    assert result.success, result.diagnostics
    assert result.state.ledger.flow_heat_J == pytest.approx(-10.0)
    assert result.state.ledger.flow_shaft_work_J < 0
    assert result.state.thermal.vessels["receiver"].energy_jacket_J == 0
    assert result.diagnostics["receiver_energy_residual_J"] == pytest.approx(0, abs=2e-5)
