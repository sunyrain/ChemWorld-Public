"""The existing finite redox layer consumes the common residual advance."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualNumericalError
from chemworld.physchem.transient_electrode import (
    TransientElectrodeSpec,
    advance_electrode,
)
from chemworld.runtime import ChemWorldRuntime, SessionSnapshot
from chemworld.runtime.transient_electrode import unpack_electrode
from chemworld.world.electrode_world import electrode_world_spec


def test_redox_diffusion_residual_uses_same_inventory_charge_and_work_as_explicit_ode():
    spec = TransientElectrodeSpec(cells=6, numerical_form="residual")
    initial = spec.initial_state()
    program = ScalarControlProgram(((0, 0.05),), "V")
    result = advance_electrode(spec, initial, program, 0.2, evaluation_times_s=[0, 0.2])
    expected = advance_electrode(
        replace(spec, numerical_form="ode"), initial, program, 0.2, evaluation_times_s=[0, 0.2]
    )
    assert result.final_state.amounts_mol == pytest.approx(
        expected.final_state.amounts_mol, rel=1e-7, abs=1e-15
    )
    assert result.final_state.coordinates == pytest.approx(
        expected.final_state.coordinates, rel=1e-7, abs=1e-10
    )
    assert result.solver_diagnostic["numerical_form"] == "residual"
    assert all(abs(v) < 1e-9 for v in result.conservation_residuals.values())
    assert result.energy_balance_residual_J is None


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_residual_redox_runtime_keeps_real_units_resources_and_exact_snapshot_resume(noise):
    original = ChemWorldRuntime(
        composition=electrode_world_spec(
            TransientElectrodeSpec(cells=6, numerical_form="residual")
        ),
        observation_noise_mode=noise,
    )
    restored = None
    try:
        original.reset(seed=19)
        for action in (
            {"operation": "configure_electrical_control", "potential_V": 0.05},
            {"operation": "wait", "duration_s": 1},
        ):
            assert original.execute(action).info["transaction_status"] == "committed"
        restored = ChemWorldRuntime.from_snapshot(
            SessionSnapshot.from_dict(
                json.loads(json.dumps(original.snapshot().to_dict(), sort_keys=True))
            )
        )
        for action in (
            {"operation": "wait", "duration_s": 2},
            {"operation": "measure", "instrument": "electrical_meter"},
        ):
            a, b = original.execute(action), restored.execute(action)
            assert a.info["transaction_status"] == b.info["transaction_status"] == "committed"
            assert a.observation.to_dict() == b.observation.to_dict()
        assert original._state.ledger == restored._state.ledger
        assert original._state.ledger.time_s == pytest.approx(4)
        assert original._state.phases == restored._state.phases
        assert original._state.state_slots == restored._state.state_slots
        assert unpack_electrode(restored._state)[2].numerical_form == "residual"
        assert a.observation.channel("electrical_current").unit == "A"
        assert a.observation.channel("terminal_potential").unit == "V"
    finally:
        original.close()
        if restored is not None:
            restored.close()


def test_residual_redox_numerical_failure_preserves_phases_equipment_and_elapsed_time(monkeypatch):
    import chemworld.physchem.coupled_residual as adapter

    session = ChemWorldRuntime(
        composition=electrode_world_spec(TransientElectrodeSpec(cells=6, numerical_form="residual"))
    )
    try:
        session.reset(seed=19)
        session.execute({"operation": "configure_electrical_control", "potential_V": 0.05})
        before = session._state

        def fail(*args, **kwargs):
            raise ResidualNumericalError("injected spatial residual failure")

        monkeypatch.setattr(adapter, "solve_residual", fail)
        result = session.execute({"operation": "wait", "duration_s": 1})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.equipment == before.equipment
        assert session._state.ledger.time_s == before.ledger.time_s
    finally:
        session.close()
