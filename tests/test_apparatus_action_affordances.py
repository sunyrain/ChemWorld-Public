"""Public actions must describe the installed apparatus, including rejections."""

import gymnasium as gym
import pytest

import chemworld  # noqa: F401 -- registers the preset interface
from chemworld.action_codec import ActionCodec
from chemworld.examples.native_distillation_research import (
    initial_column_state,
    synthetic_column_spec,
)
from chemworld.operation_validator import OperationValidator
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.constitution_factory import make_chemworld_constitution
from chemworld.world.dynamic_column_world import dynamic_column_world_spec
from chemworld.world.state_factory import initial_chemworld_state


@pytest.mark.parametrize("operation", ["heat", "wait", "run_flow"])
def test_default_registry_resolves_explicit_short_duration_bounds(operation: str) -> None:
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={operation},
        authored_field_bounds={(operation, "duration_s"): (0.001, 2.0)},
    )
    assert validator.public_field_bounds(
        operation, "duration_s", initial_chemworld_state(), low=1.0, high=14400.0,
    ) == (0.001, 2.0)


def test_named_feed_choices_require_their_routing_physics() -> None:
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={"add_component"},
        action_codec=ActionCodec(feed_ids=("feed",)),
    )
    with pytest.raises(ValueError, match="material-routing physics"):
        validator.public_field_choices(
            "add_component", "component", initial_chemworld_state(), choices=("feed",),
        )


def test_shortcut_column_does_not_advertise_or_accept_total_reflux():
    with gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0) as env:
        env.reset(seed=0)
        base = env.unwrapped
        fields = {f["field"]: f for f in base.action_schema("distill")["fields"]}
        assert fields["full_reflux"]["choices"] == [0]
        before = base._state
        _, _, _, _, info = env.step({
            "operation": "distill", "target_temperature_K": 360., "duration_s": 1.,
            "reflux_ratio": 1., "full_reflux": 1,
        })
        assert info["transaction_status"] == "validation_failed"
        assert info["preconditions"]["payload_choice:full_reflux"] is False
        assert base._state.species_amounts == before.species_amounts
        assert base._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("path", ["rinse", "left", "right"])
def test_fixed_volume_flow_reactor_rejects_uninstalled_paths(path):
    with gym.make("ChemWorld", task_id="flow-reaction-optimization", seed=0) as env:
        env.reset(seed=0)
        base = env.unwrapped
        fields = {f["field"]: f for f in base.action_schema("set_flow_rate")["fields"]}
        assert fields["flow_path"]["choices"] == ["process"]
        assert fields["flow_rate_mL_min"]["bounds"] == {"low": 0.01, "high": 20.0}
        for flow, selected_path, reason in ((0., "process", "payload_bounds:flow_rate_mL_min"),
                                             (1., path, "payload_choice:flow_path")):
            result = base.validate_action({
                "operation": "set_flow_rate", "flow_rate_mL_min": flow,
                "flow_path": selected_path,
            })
            assert not result["valid"]
            assert not result["dispatchable_to_runtime"]
            assert reason in result["invalid_reasons"]


def test_native_column_keeps_total_reflux_available_and_executes_it():
    spec = synthetic_column_spec()
    session = ChemWorldRuntime(
        composition=dynamic_column_world_spec(spec, initial_column_state(spec))
    )
    try:
        session.reset(seed=20)
        choices = session.operation_validator.public_field_choices(
            "distill", "full_reflux", session._state, choices=(0, 1),
        )
        assert choices == (0, 1)
        result = session.execute({
            "operation": "distill", "target_temperature_K": 430., "duration_s": 1.,
            "reflux_ratio": 1., "full_reflux": 1,
        })
        assert result.info["transaction_status"] == "committed", result.info
        assert session._state.ledger.time_s == pytest.approx(1.)
    finally:
        session.close()


@pytest.mark.parametrize("retained,removed", [("organic", "aqueous"), ("aqueous", "organic")])
def test_separation_choices_follow_retained_phase_inventory(retained, removed):
    with gym.make("ChemWorld", task_id="partition-discovery", budget_override=16) as env:
        env.reset(seed=0)
        base = env.unwrapped
        for action in (
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.01},
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.01},
            {"operation": "add_extractant", "extractant": 2, "volume_L": 0.018},
            {"operation": "mix", "duration_s": 600., "stirring_speed_rpm": 600.},
            {"operation": "settle", "duration_s": 300.},
            {"operation": "separate_phase", "target_phase": retained},
        ):
            *_, info = env.step(action)
            assert info["transaction_status"] == "committed", (action, info)

        fields = {f["field"]: f for f in base.action_schema("separate_phase")["fields"]}
        assert fields["target_phase"]["choices"] == [retained]
        assert base.validate_action({
            "operation": "separate_phase", "target_phase": retained,
        })["valid"]
        before = base._state
        *_, info = env.step({"operation": "separate_phase", "target_phase": removed})
        assert info["transaction_status"] == "validation_failed"
        assert info["preconditions"]["payload_choice:target_phase"] is False
        assert base._state.phases == before.phases
        assert base._state.species_amounts == before.species_amounts
        assert base._state.ledger.time_s == before.ledger.time_s
