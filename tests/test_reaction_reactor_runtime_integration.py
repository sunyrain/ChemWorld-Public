from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.foundation import equipment_settings
from chemworld.physchem.batch_reactors import DynamicBatchReactorModel
from chemworld.physchem.reaction_adapter_manifest import (
    reaction_reactor_runtime_adapter_manifest,
    reaction_reactor_runtime_provider_contract,
)
from chemworld.world.reaction_kernel import integrate_compiled_reaction_ode


def _charge(
    env: gym.Env[Any, Any],
    *,
    solvent: int = 2,
    catalyst: int = 1,
    amount_mol: float = 0.010,
    volume_L: float = 0.030,
) -> None:
    for action in (
        {"operation": "add_solvent", "volume_L": volume_L, "solvent": solvent},
        {"operation": "add_reagent", "amount_mol": amount_mol},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": 0.0002,
            "catalyst": catalyst,
        },
    ):
        _observation, _reward, _terminated, _truncated, info = env.step(action)
        assert info["transaction_status"] == "committed"


def _reaction_response(
    *,
    temperature_K: float = 385.0,
    duration_s: float = 1200.0,
    catalyst: int = 1,
    solvent: int = 2,
    amount_mol: float = 0.010,
) -> dict[str, float]:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=4, debug_truth=True)
    try:
        env.reset(seed=4)
        _charge(env, solvent=solvent, catalyst=catalyst, amount_mol=amount_mol)
        _observation, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": temperature_K,
                "duration_s": duration_s,
                "stirring_speed_rpm": 700.0,
            }
        )
        assert info["transaction_status"] == "committed"
        state = env.unwrapped._state
        return {
            **env.unwrapped.runtime.domain_services.species_view.truth_values(state),
            "temperature_K": state.temperature_K,
            "energy_jacket_J": state.ledger.energy_jacket_J,
        }
    finally:
        env.close()


def test_formal_heat_wait_use_validated_dynamic_batch_once_per_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = DynamicBatchReactorModel.simulate

    def recording_simulate(self: DynamicBatchReactorModel, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.reactor_id)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DynamicBatchReactorModel, "simulate", recording_simulate)
    env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=4, debug_truth=True)
    try:
        env.reset(seed=4)
        _charge(env)
        configured = env.unwrapped._state
        initial_charge_ledger = configured.species.initial_amounts_mol.copy()
        assert configured.ledger.time_s == 0.0

        _observation, _reward, _terminated, _truncated, heat_info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1200.0,
                "stirring_speed_rpm": 700.0,
            }
        )
        assert heat_info["transaction_status"] == "committed"
        after_heat = env.unwrapped._state
        settings = equipment_settings(after_heat.equipment, "batch_reactor")
        assert calls == ["chemworld_runtime_dynamic_batch_v1"]
        assert settings["runtime_provider_id"] == (
            "chemworld_validated_reaction_reactor_runtime_v1"
        )
        assert settings["reactor_model_id"] == (
            "dynamic_batch_heat_release_jacket_sampling"
        )
        assert settings["reaction_network_id"] == (
            env.unwrapped.runtime.compiled_mechanism.network.network_id
        )
        assert settings["mechanism_hash"] == (
            env.unwrapped.runtime.compiled_mechanism.mechanism_hash
        )
        assert settings["last_operation_semantic"] == "advance"
        assert settings["reaction_advance_index"] == 1
        assert settings["termination_reason"] == "duration_reached"
        assert settings["solver_diagnostic"]["success"] is True
        assert settings["solver_diagnostic"]["nonnegative_passed"] is True
        assert settings["reactor_diagnostic"]["material_balance_closed"] is True
        assert settings["material_balance_error_mol"] < 1.0e-8
        assert settings["maximum_conservation_drift_mol"] < 1.0e-7
        assert max(
            abs(value) for value in settings["element_inventory_residuals_mol"].values()
        ) < 1.0e-8
        assert abs(settings["charge_inventory_residual_mol"]) < 1.0e-8
        assert abs(settings["energy_balance_residual_J"]) < 1.0e-5 * max(
            abs(after_heat.ledger.energy_jacket_J)
            + abs(after_heat.ledger.heat_reaction_J)
            + abs(after_heat.ledger.heat_loss_J),
            1.0,
        )
        assert len(settings["trajectory_digest"]) == 64
        serialized_settings = json.dumps(settings)
        assert "chemworld_reaction_network_lite" not in serialized_settings
        assert "chemworld_reactor_lite" not in serialized_settings
        assert after_heat.species.initial_amounts_mol == initial_charge_ledger
        assert set(after_heat.species_amounts) == set(
            env.unwrapped.runtime.compiled_mechanism.network.species_ids
        )
        assert "A" not in after_heat.species_amounts
        assert "P" not in after_heat.species_amounts
        assert "solver_diagnostic" not in after_heat.metadata

        _observation, _reward, _terminated, _truncated, wait_info = env.step(
            {
                "operation": "wait",
                "duration_s": 300.0,
                "stirring_speed_rpm": 700.0,
            }
        )
        assert wait_info["transaction_status"] == "committed"
        after_wait = env.unwrapped._state
        wait_settings = equipment_settings(after_wait.equipment, "batch_reactor")
        assert calls == [
            "chemworld_runtime_dynamic_batch_v1",
            "chemworld_runtime_dynamic_batch_v1",
        ]
        assert wait_settings["reaction_advance_index"] == 2
        assert wait_settings["last_operation"] == "wait"
        assert after_wait.ledger.time_s == pytest.approx(1500.0)
        assert after_wait.process.metrics["reaction_cumulative_time_s"] == pytest.approx(
            1500.0
        )
        assert after_wait.species.initial_amounts_mol == initial_charge_ledger
    finally:
        env.close()


def test_repeated_wait_is_an_additional_advance_not_duplicate_configuration() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=4)
    try:
        env.reset(seed=4)
        _charge(env)
        action = {
            "operation": "wait",
            "duration_s": 120.0,
            "stirring_speed_rpm": 700.0,
        }
        env.step(action)
        first = env.unwrapped._state
        env.step(action)
        second = env.unwrapped._state
        settings = equipment_settings(second.equipment, "batch_reactor")

        assert first.ledger.time_s == pytest.approx(120.0)
        assert second.ledger.time_s == pytest.approx(240.0)
        assert settings["reaction_advance_index"] == 2
        assert settings["repeat_semantic"].startswith("each committed repeat")
        assert second.ledger.cost > first.ledger.cost
    finally:
        env.close()


def test_temperature_time_catalyst_solvent_and_composition_are_non_degenerate() -> None:
    baseline = _reaction_response()
    cold = _reaction_response(temperature_K=330.0)
    short = _reaction_response(duration_s=300.0)
    catalyst_zero = _reaction_response(catalyst=0)
    solvent_zero = _reaction_response(solvent=0)
    lean = _reaction_response(amount_mol=0.005)
    rich = _reaction_response(amount_mol=0.015)

    assert baseline["conversion"] > cold["conversion"]
    assert baseline["conversion"] > short["conversion"]
    assert baseline["conversion"] != pytest.approx(catalyst_zero["conversion"])
    assert baseline["selectivity"] != pytest.approx(solvent_zero["selectivity"])
    assert lean["conversion"] != pytest.approx(rich["conversion"])
    assert baseline["temperature_K"] != pytest.approx(cold["temperature_K"])
    assert baseline["energy_jacket_J"] != pytest.approx(short["energy_jacket_J"])


def test_invalid_and_runaway_domains_fail_closed() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=4)
    try:
        env.reset(seed=4)
        _charge(env)
        state = env.unwrapped._state
        compiled = env.unwrapped.runtime.compiled_mechanism
        with pytest.raises(ValueError, match="duration_s"):
            integrate_compiled_reaction_ode(
                state=state,
                world=env.unwrapped.world,
                compiled_mechanism=compiled,
                duration_s=float("nan"),
                target_temperature_K=385.0,
                heat=True,
                stirring_speed_rpm=700.0,
            )

        runaway_reactions = tuple(
            replace(reaction, delta_h_J_per_mol=reaction.delta_h_J_per_mol * 1000.0)
            for reaction in compiled.network.reactions
        )
        runaway_network = replace(compiled.network, reactions=runaway_reactions)
        runaway_mechanism = replace(
            compiled,
            network=runaway_network,
            reaction_enthalpies={
                reaction.reaction_id: reaction.delta_h_J_per_mol
                for reaction in runaway_reactions
            },
        )
        with pytest.raises(RuntimeError, match="validity domain"):
            integrate_compiled_reaction_ode(
                state=state,
                world=env.unwrapped.world,
                compiled_mechanism=runaway_mechanism,
                duration_s=1200.0,
                target_temperature_K=385.0,
                heat=True,
                stirring_speed_rpm=700.0,
            )
    finally:
        env.close()


def test_model_failure_rolls_back_without_physical_or_time_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=4)
    try:
        env.reset(seed=4)
        _charge(env)
        before = env.unwrapped._state

        def fail_simulation(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("injected solver nonconvergence")

        monkeypatch.setattr(DynamicBatchReactorModel, "simulate", fail_simulation)
        _observation, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1200.0,
                "stirring_speed_rpm": 700.0,
            }
        )
        after = env.unwrapped._state

        assert info["transaction_status"] == "rolled_back"
        assert info["rollback_reason"] == "numerical_failure"
        assert info["operation_outcome"] == "numerical_failure"
        assert info["actual_duration_s"] == 0.0
        assert after.species_amounts == before.species_amounts
        assert after.temperature_K == before.temperature_K
        assert after.ledger.time_s == before.ledger.time_s
        assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        assert after.ledger.heat_reaction_J == before.ledger.heat_reaction_J
        assert after.ledger.heat_loss_J == before.ledger.heat_loss_J
        assert after.ledger.cost > before.ledger.cost
        assert after.ledger.risk > before.ledger.risk
    finally:
        env.close()


def test_runtime_adapter_explicitly_replaces_both_lite_model_ids() -> None:
    contract = reaction_reactor_runtime_provider_contract()
    manifest = reaction_reactor_runtime_adapter_manifest()

    assert contract.model_id == "chemworld_validated_reaction_reactor_runtime_v1"
    assert contract.role.value == "runtime"
    assert contract.maturity.value == "reference_validated"
    assert contract.intended_operations == ("heat", "wait")
    assert manifest.status == "integrated"
    assert set(manifest.replaces_model_ids) == {
        "chemworld_reaction_network_lite",
        "chemworld_reactor_lite",
    }
