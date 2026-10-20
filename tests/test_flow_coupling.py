from __future__ import annotations

from copy import deepcopy
from itertools import pairwise
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import chemworld  # noqa: F401
from chemworld.foundation import equipment_settings, equipment_status
from chemworld.physchem.maturity import MaturityLevel, validate_model_card
from chemworld.physchem.pfr_reactors import (
    PFRGeometrySpec,
    PFRModel,
    geometry_resolved_pfr_model_card,
)


def _charged_flow_env(*, seed: int = 0) -> gym.Env:
    env = gym.make(
        "ChemWorld",
        task_id="flow-reaction-optimization",
        seed=seed,
        budget_override=12,
    )
    env.reset(seed=seed)
    for action in (
        {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": 0.00022,
            "catalyst": 1,
        },
    ):
        _, _, _, _, info = env.step(action)
        assert info["transaction_status"] == "committed"
    return env


def _configure(
    env: gym.Env,
    *,
    flow_rate_mL_min: float = 1.2,
    residence_time_s: float = 900.0,
) -> dict[str, Any]:
    _, _, _, _, info = env.step(
        {
            "operation": "set_flow_rate",
            "flow_rate_mL_min": flow_rate_mL_min,
            "residence_time_s": residence_time_s,
        }
    )
    assert info["transaction_status"] == "committed"
    return info


def _run(
    env: gym.Env,
    *,
    target_temperature_K: float = 382.0,
    duration_s: float = 1800.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, _, _, info = env.step(
        {
            "operation": "run_flow",
            "target_temperature_K": target_temperature_K,
            "duration_s": duration_s,
        }
    )
    state = env.unwrapped._state
    return info, equipment_settings(state.equipment, "flow_reactor")


def _physical_snapshot(state: Any) -> dict[str, Any]:
    return {
        "species": deepcopy(state.species_amounts),
        "volume_L": state.volume_L,
        "temperature_K": state.temperature_K,
        "pressure_Pa": state.pressure_Pa,
        "time_s": state.ledger.time_s,
        "energy_jacket_J": state.ledger.energy_jacket_J,
        "heat_reaction_J": state.ledger.heat_reaction_J,
        "heat_loss_J": state.ledger.heat_loss_J,
    }


def test_flow_configuration_is_not_a_hidden_experiment() -> None:
    env = _charged_flow_env()
    try:
        before = env.unwrapped._state
        physical_before = _physical_snapshot(before)
        initial_charge_before = deepcopy(before.species.initial_amounts_mol)
        cost_before = before.ledger.cost

        info = _configure(env)
        after = env.unwrapped._state
        settings = equipment_settings(after.equipment, "flow_reactor")
        configuration = equipment_settings(
            after.equipment,
            "flow_reactor:configuration",
        )

        assert _physical_snapshot(after) == physical_before
        assert after.species.initial_amounts_mol == initial_charge_before
        assert after.ledger.cost > cost_before
        assert "equipment" in info["affected_ledgers"]
        assert settings == {
            "flow_rate_mL_min": 1.2,
            "residence_time_s": 900.0,
            "requested_residence_time_s": 900.0,
            "minimum_run_duration_s": 900.0,
        }
        assert configuration["configuration_revision"] == 1
        assert configuration["configuration_semantic"] == ("configure_only_no_physical_advance")
        assert len(configuration["configured_feed_signature"]) == 64
        assert configuration["runtime_provider_id"] == ("chemworld_geometry_resolved_pfr_v2")
        assert configuration["hardware_contract"] == ("fixed_volume_flow_derived_residence_v1")
        assert configuration["reactor_volume_L"] == pytest.approx(0.018)
    finally:
        env.close()


def test_formal_flow_model_card_is_reference_validated_and_bounded() -> None:
    card = geometry_resolved_pfr_model_card()

    assert card.model_id == "chemworld_geometry_resolved_pfr_v2"
    assert card.maturity is MaturityLevel.REFERENCE_VALIDATED
    assert validate_model_card(card) == []
    assert any("single-phase" in note for note in card.model_limit_notes)


def test_configuration_can_precede_charge_but_feed_changes_require_reconfiguration() -> None:
    empty = gym.make(
        "ChemWorld",
        task_id="flow-reaction-optimization",
        seed=0,
        budget_override=12,
    )
    empty.reset(seed=0)
    try:
        info = _configure(empty)
        assert info["transaction_status"] == "committed"
    finally:
        empty.close()

    env = _charged_flow_env()
    try:
        _configure(env)
        _, _, _, _, add_info = env.step({"operation": "add_reagent", "amount_mol": 0.001})
        assert add_info["transaction_status"] == "committed"
        before = _physical_snapshot(env.unwrapped._state)

        run_info, _ = _run(env)

        assert run_info["transaction_status"] == "rolled_back"
        assert _physical_snapshot(env.unwrapped._state) == before
    finally:
        env.close()


def test_official_flow_runtime_calls_geometry_resolved_pfr_once(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    original = PFRModel.simulate

    def recording_simulate(self: PFRModel, *args: Any, **kwargs: Any) -> Any:
        calls.append(
            {
                "geometry": self.geometry,
                "temperature_K": kwargs["temperature_K"],
                "axial_positions_m": kwargs["axial_positions_m"],
            }
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PFRModel, "simulate", recording_simulate)
    env = _charged_flow_env()
    try:
        inlet_temperature = env.unwrapped._state.temperature_K
        initial_charge = deepcopy(env.unwrapped._state.species.initial_amounts_mol)
        _configure(env)
        info, settings = _run(env)
        state = env.unwrapped._state

        assert info["transaction_status"] == "committed"
        assert len(calls) == 1
        assert calls[0]["geometry"] is not None
        assert calls[0]["temperature_K"] == pytest.approx(inlet_temperature)
        assert inlet_temperature != pytest.approx(382.0)
        assert len(calls[0]["axial_positions_m"]) == 9
        assert settings["runtime_provider_id"] == ("chemworld_geometry_resolved_pfr_v2")
        assert settings["runtime_adapter_id"] == ("chemworld_geometry_resolved_pfr_v1")
        assert settings["inlet_temperature_K"] == pytest.approx(inlet_temperature)
        assert settings["boundary_temperature_K"] == pytest.approx(382.0)
        assert settings["geometry"]["volume_L"] == pytest.approx(1.2 / 1000.0 / 60.0 * 900.0)
        assert settings["solver_diagnostic"]["success"] is True
        assert settings["reactor_diagnostic"]["material_balance_closed"] is True
        assert settings["material_balance_error_mol"] < 1.0e-8
        assert len(settings["axial_profile"]) == 9
        pressures = [point["pressure_Pa"] for point in settings["axial_profile"]]
        assert all(left >= right for left, right in pairwise(pressures))
        assert settings["pressure_drop_Pa"] > 0.0
        assert settings["hydraulic_ledger"]["metadata"]["model_id"] == (
            "single_phase_straight_tube_darcy_weisbach_v1"
        )
        assert settings["provenance"]["model_id"] == "geometry_resolved_pfr_v2"
        assert settings["provenance"]["reaction_network_id"]
        assert len(settings["result_digest"]) == 64
        assert "lite" not in repr(settings).lower()
        assert state.species.initial_amounts_mol == initial_charge
        assert state.process.metrics["flow_experiment_count"] == 1.0
        assert state.process.metrics["flow_campaign_time_s"] == pytest.approx(1800.0)
        assert state.process.metrics["flow_throughput_mL"] == pytest.approx(36.0)
        assert state.process.metrics["flow_hydraulic_energy_J"] > 0.0
        thermal = settings["thermal_ledger"]
        scale = max(
            abs(thermal["energy_jacket_J"])
            + abs(thermal["heat_reaction_J"])
            + abs(thermal["heat_loss_J"]),
            1.0,
        )
        assert abs(thermal["energy_balance_residual_J"]) <= 1.0e-5 * scale
        assert env.unwrapped.constitution.check_state(state).passed
    finally:
        env.close()


def test_flow_run_requires_fresh_configuration_and_rolls_back_atomically() -> None:
    env = _charged_flow_env()
    try:
        _configure(env)
        committed, _ = _run(env)
        assert committed["transaction_status"] == "committed"
        before = env.unwrapped._state
        physical_before = _physical_snapshot(before)
        cost_before = before.ledger.cost
        risk_before = before.ledger.risk

        rejected, _ = _run(env)
        after = env.unwrapped._state

        assert rejected["transaction_status"] == "rolled_back"
        assert rejected["rollback_reason"] == "rejected"
        assert rejected["operation_outcome"] == "rejected"
        assert _physical_snapshot(after) == physical_before
        assert after.ledger.cost > cost_before
        assert after.ledger.risk >= risk_before
        assert equipment_status(after.equipment, "flow_reactor") == "completed"
    finally:
        env.close()


def test_reconfiguration_creates_a_second_traceable_experiment() -> None:
    env = _charged_flow_env()
    try:
        initial_charge = deepcopy(env.unwrapped._state.species.initial_amounts_mol)
        _configure(env, residence_time_s=600.0)
        first, _ = _run(env, duration_s=900.0)
        assert first["transaction_status"] == "committed"
        _configure(env, flow_rate_mL_min=0.8, residence_time_s=600.0)
        second, settings = _run(env, target_temperature_K=370.0, duration_s=1350.0)
        state = env.unwrapped._state

        assert second["transaction_status"] == "committed"
        assert settings["configuration_revision"] == 2
        assert settings["flow_experiment_index"] == 2
        assert state.process.metrics["flow_experiment_count"] == 2.0
        assert state.process.metrics["flow_campaign_time_s"] == pytest.approx(2250.0)
        assert state.process.metrics["flow_throughput_mL"] == pytest.approx(36.0)
        assert state.species.initial_amounts_mol == initial_charge
    finally:
        env.close()


def test_invalid_runtime_model_result_fails_closed(monkeypatch: Any) -> None:
    env = _charged_flow_env()
    try:
        _configure(env)
        before = env.unwrapped._state
        physical_before = _physical_snapshot(before)

        def fail_model(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected solver failure")

        monkeypatch.setattr(PFRModel, "simulate", fail_model)
        info, _ = _run(env)
        after = env.unwrapped._state

        assert info["transaction_status"] == "rolled_back"
        assert _physical_snapshot(after) == physical_before
    finally:
        env.close()


def test_flow_runtime_rejects_incomplete_residence_time() -> None:
    env = _charged_flow_env()
    try:
        _configure(env, residence_time_s=900.0)
        before = _physical_snapshot(env.unwrapped._state)
        info, _ = _run(env, duration_s=899.0)

        assert info["transaction_status"] == "validation_failed"
        assert "payload_bounds:duration_s" in info["world_events"][0]["payload"]["invalid_reasons"]
        assert _physical_snapshot(env.unwrapped._state) == before
    finally:
        env.close()


def test_flow_schema_publishes_effective_minimum_run_duration() -> None:
    env = _charged_flow_env()
    try:
        _configure(env, residence_time_s=900.0)
        schema = env.unwrapped.action_schema("run_flow")
        duration = next(field for field in schema["fields"] if field["field"] == "duration_s")

        assert duration["bounds"]["low"] == pytest.approx(900.0)
        assert duration["state_dependent_bounds"] is True
        accepted = env.unwrapped.validate_action(
            {
                "operation": "run_flow",
                "target_temperature_K": 382.0,
                "duration_s": 900.0,
            }
        )
        assert accepted["valid"] is True
    finally:
        env.close()


def test_flow_controls_are_physically_identifiable() -> None:
    def experiment(
        flow_rate_mL_min: float,
        residence_time_s: float,
        target_temperature_K: float,
    ) -> tuple[dict[str, Any], float]:
        env = _charged_flow_env()
        try:
            _configure(
                env,
                flow_rate_mL_min=flow_rate_mL_min,
                residence_time_s=residence_time_s,
            )
            derived_residence = 0.018 * 1000.0 * 60.0 / flow_rate_mL_min
            info, settings = _run(
                env,
                target_temperature_K=target_temperature_K,
                duration_s=max(2400.0, derived_residence),
            )
            assert info["transaction_status"] == "committed"
            conversion = env.unwrapped._state.process.metrics["flow_conversion"]
            return settings, float(conversion)
        finally:
            env.close()

    short, short_conversion = experiment(1.2, 300.0, 382.0)
    long, long_conversion = experiment(1.2, 1200.0, 382.0)
    cool, _ = experiment(1.2, 900.0, 330.0)
    hot, _ = experiment(1.2, 900.0, 410.0)
    slow, _ = experiment(0.5, 900.0, 382.0)
    fast, _ = experiment(3.0, 900.0, 382.0)

    assert long_conversion == pytest.approx(short_conversion)
    assert long["geometry"]["volume_L"] == pytest.approx(short["geometry"]["volume_L"])
    assert long["geometry"]["length_m"] == pytest.approx(short["geometry"]["length_m"])
    assert hot["outlet_temperature_K"] > cool["outlet_temperature_K"]
    assert hot["thermal_ledger"]["energy_jacket_J"] != pytest.approx(
        cool["thermal_ledger"]["energy_jacket_J"]
    )
    assert fast["pressure_drop_Pa"] > slow["pressure_drop_Pa"]
    assert fast["reynolds_number"] > slow["reynolds_number"]
    assert slow["geometry"]["volume_L"] == pytest.approx(fast["geometry"]["volume_L"])
    assert slow["geometry"]["length_m"] == pytest.approx(fast["geometry"]["length_m"])


def test_pfr_fails_on_invalid_feed_and_nonpositive_outlet_pressure() -> None:
    env = _charged_flow_env()
    try:
        network = env.unwrapped.runtime.domain_services.species_view.mechanism.network
        concentrations = dict.fromkeys(network.species_ids, 0.0)
        concentrations[network.species_ids[0]] = 1.0
        model = PFRModel(
            network,
            reactor_volume_L=1.0,
            volumetric_flow_L_s=0.01,
        )
        with pytest.raises(ValueError, match="nonnegative"):
            model.simulate(
                {**concentrations, network.species_ids[0]: -1.0},
                temperature_K=350.0,
            )
        with pytest.raises(ValueError, match="finite"):
            model.simulate(
                {**concentrations, network.species_ids[0]: np.nan},
                temperature_K=350.0,
            )

        geometry = PFRGeometrySpec(length_m=1.0, inner_diameter_m=0.004)
        pressure_model = PFRModel(
            network,
            reactor_volume_L=geometry.volume_l,
            volumetric_flow_L_s=0.001,
            geometry=geometry,
            inlet_pressure_Pa=1.0,
        )
        with pytest.raises(RuntimeError, match="nonpositive outlet"):
            pressure_model.simulate(concentrations, temperature_K=350.0)
    finally:
        env.close()
