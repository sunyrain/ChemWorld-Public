"""Boundary returns preserve physical state and the provider result contract."""

import pytest

from chemworld.examples.native_liquid_research import synthetic_liquid_world
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.control_program import (
    advance_electrical_program,
    configure_electrical_control,
    electrical_device,
)
from chemworld.runtime.distillation_services import ChemWorldDistillationServices
from chemworld.runtime.liquid_equilibrium import LiquidEquilibriumRuntime
from chemworld.runtime.transient_electrode import TransientElectrodeRuntime
from chemworld.world.electrode_world import electrode_world_spec


@pytest.mark.parametrize("options", [{"heat": True}, {"jacket_program": {}}])
@pytest.mark.parametrize("model", ["liquid", "electrode"])
def test_isothermal_strategy_rejects_heat_without_advancing(model, options):
    if model == "liquid":
        world = synthetic_liquid_world()
        provider = LiquidEquilibriumRuntime(world)
        reason = "liquid_model_is_isothermal"
    else:
        world = electrode_world_spec()
        provider = TransientElectrodeRuntime(world)
        reason = "electrode_isothermal_boundary"
    state = world.physics.initial_states["reactor"]
    before = state.to_dict()

    result = provider.advance(state, {"operation": "wait", "duration_s": 1}, **options)

    assert not result.success and result.outcome == "rejected"
    assert result.failure_reason == reason
    assert result.state is None and result.actual_duration_s == 0
    assert result.to_dict()["diagnostics"] == {}
    assert state.to_dict() == before


def test_shortcut_distillation_rejects_full_reflux_without_advancing():
    session = ChemWorldRuntime(task_id="reaction-to-distillation")
    try:
        session.reset(seed=4)
        domains = session.runtime.domain_services
        provider = ChemWorldDistillationServices(domains.world, domains.species_view)
        state = session._state
        before = state.to_dict()

        result = provider.distill(state, {"operation": "distill", "full_reflux": 1})

        assert not result.success and result.outcome == "rejected"
        assert result.failure_reason == "shortcut_full_reflux_unsupported"
        assert result.state is None and result.actual_duration_s == 0
        assert result.to_dict()["diagnostics"] == {}
        assert state.to_dict() == before
    finally:
        session.close()


@pytest.mark.parametrize("duration", [0, 1e-9])
def test_electrical_no_interval_returns_success_without_running_plant(duration):
    world = electrode_world_spec()
    state = configure_electrical_control(
        world.physics.initial_states["reactor"], {"potential_V": 0.1}
    )
    before = state.to_dict()

    def unexpected_advance(*args):
        pytest.fail("a request below the interval threshold must not advance the plant")

    result = advance_electrical_program(state, {"duration_s": duration}, unexpected_advance)

    assert result.success and result.outcome == "success"
    assert result.failure_reason is None and result.actual_duration_s == 0
    assert result.to_dict()["diagnostics"]["control_transitions"] == []
    assert state.to_dict() == before
    assert result.state.ledger == state.ledger
    assert result.state.species_amounts == state.species_amounts
    assert result.state.state_slots == state.state_slots
    settings = electrical_device(result.state).settings["electrical_control"]
    assert settings["last_requested_s"] == duration
    assert settings["last_executed_s"] == 0
