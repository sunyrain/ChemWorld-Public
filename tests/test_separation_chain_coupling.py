from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import pytest

import chemworld.runtime.phase_separation_services as separation_services
from chemworld.foundation import equipment_settings


def _reaction_and_contact_actions(
    *,
    extractant: int = 3,
    extractant_volume_L: float = 0.018,
    mixing_duration_s: float = 240.0,
) -> tuple[dict[str, Any], ...]:
    return (
        {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {
            "operation": "add_catalyst",
            "catalyst_amount_mol": 0.00025,
            "catalyst": 1,
        },
        {
            "operation": "heat",
            "target_temperature_K": 385.0,
            "duration_s": 1500.0,
            "stirring_speed_rpm": 720.0,
        },
        {
            "operation": "wait",
            "duration_s": 900.0,
            "stirring_speed_rpm": 720.0,
        },
        {"operation": "quench"},
        {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
        {
            "operation": "add_extractant",
            "extractant": extractant,
            "volume_L": extractant_volume_L,
        },
        {
            "operation": "mix",
            "duration_s": mixing_duration_s,
            "stirring_speed_rpm": 850.0,
        },
        {"operation": "settle", "duration_s": 420.0},
        {"operation": "separate_phase", "target_phase": "organic"},
    )


def _step_committed(env: gym.Env, action: dict[str, Any]) -> None:
    _, _, _, _, info = env.step(action)
    assert info["transaction_status"] == "committed", (action, info)
    assert not info["constraint_flags"]["precondition_failed"], (action, info)


def _runtime(env: gym.Env) -> Any:
    return cast(Any, env.unwrapped)


def _run_to_selected_phase(
    *,
    extractant: int = 3,
    extractant_volume_L: float = 0.018,
    mixing_duration_s: float = 240.0,
) -> gym.Env:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    env.reset(seed=0)
    for action in _reaction_and_contact_actions(
        extractant=extractant,
        extractant_volume_L=extractant_volume_L,
        mixing_duration_s=mixing_duration_s,
    ):
        _step_committed(env, action)
    return env


def _track_call(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    calls: dict[str, int],
) -> None:
    original = getattr(separation_services, name)

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls[name] = calls.get(name, 0) + 1
        return original(*args, **kwargs)

    monkeypatch.setattr(separation_services, name, tracked)


def test_phase_contact_preserves_species_and_control_volume_inventory() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=17)
    env.reset(seed=17)
    try:
        actions = _reaction_and_contact_actions()
        for action in actions[:6]:
            _step_committed(env, action)

        before_contact = _runtime(env)._state
        expected_amounts = before_contact.species_amounts.copy()
        expected_volume = before_contact.volume_L + 0.012 + 0.018

        for action in actions[6:9]:
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed", (action, info)
            assert not info["constraint_flags"]["constitution_failed"], (action, info)

        after_mix = _runtime(env)._state
        for species_id in ("A", "D", "E"):
            assert after_mix.species_amounts[species_id] == pytest.approx(
                expected_amounts[species_id],
                abs=1.0e-12,
            )
        assert after_mix.species_amounts["P_aq"] + after_mix.species_amounts[
            "P_org"
        ] == pytest.approx(
            expected_amounts["P_aq"] + expected_amounts["P_org"],
            abs=1.0e-12,
        )
        assert after_mix.species_amounts["B_aq"] + after_mix.species_amounts[
            "B_org"
        ] == pytest.approx(
            expected_amounts["B_aq"] + expected_amounts["B_org"],
            abs=1.0e-12,
        )
        assert _runtime(env).constitution.element_totals(
            after_mix.species_amounts
        ) == pytest.approx(
            _runtime(env).constitution.element_totals(expected_amounts),
            abs=1.0e-12,
        )
        assert after_mix.volume_L == pytest.approx(expected_volume, abs=1.0e-12)
        assert sum(
            phase.volume_L for phase in after_mix.phases.phases.values()
        ) == pytest.approx(expected_volume, abs=1.0e-12)
        assert _runtime(env).constitution.check_state(after_mix).passed
    finally:
        env.close()


def test_formal_runtime_executes_each_declared_provider_and_preserves_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {}
    for name in (
        "partition_split",
        "run_sorbent_drying",
        "run_vacuum_concentration",
        "run_bounded_transfer",
    ):
        _track_call(monkeypatch, name, calls)

    env = _run_to_selected_phase()
    try:
        state = _runtime(env)._state
        assert state.metadata["extraction_model_id"] == (
            "chemworld_stability_aware_lle_vnext"
        )
        assert state.metadata["extraction_provenance"]
        assert state.process.metrics["process_mass_balance_error"] < 1.0e-10

        for action in (
            {"operation": "wash", "wash_volume_L": 0.008},
            {"operation": "dry"},
            {"operation": "concentrate", "duration_s": 600.0},
            {"operation": "transfer", "transfer_fraction": 0.92},
        ):
            _step_committed(env, action)
            state = _runtime(env)._state
            assert state.process.metrics["process_mass_balance_error"] < 1.0e-10
            assert state.phases.total_amounts_mol() == pytest.approx(
                state.species_amounts,
                abs=1.0e-12,
            )
            assert _runtime(env).constitution.check_state(state).passed

        assert calls == {
            "partition_split": 1,
            "run_sorbent_drying": 1,
            "run_vacuum_concentration": 1,
            "run_bounded_transfer": 1,
        }
        assert state.metadata["wash_model_id"] == (
            "chemworld_stability_aware_lle_vnext"
        )
        assert state.metadata["wash_provenance"]

        expected = {
            "sorbent_dryer": (
                "drying_model_id",
                "chemworld_sorbent_drying_vnext",
                "drying_provider_path",
            ),
            "vacuum_concentrator": (
                "concentration_model_id",
                "chemworld_vacuum_concentration_vnext",
                "concentration_provider_path",
            ),
            "transfer_line": (
                "transfer_model_id",
                "chemworld_transfer_holdup_vnext",
                "transfer_provider_path",
            ),
        }
        for equipment_id, (model_key, model_id, provider_key) in expected.items():
            settings = equipment_settings(state.equipment, equipment_id)
            assert settings[model_key] == model_id
            assert settings[provider_key].endswith("Provider")
            provenance_key = model_key.replace("model_id", "provenance")
            assert settings[provenance_key]
    finally:
        env.close()


def test_disposable_drying_debits_actual_finite_media_and_never_refills_on_material_reset():
    from chemworld.foundation.state_ownership import StateSlots
    from chemworld.runtime.disposable_sorbent import make_media_supply, media_slot

    env = _run_to_selected_phase()
    try:
        session = _runtime(env)
        before = media_slot(session._state).values["available_mass_kg"]
        _step_committed(env,{"operation":"dry"})
        after = media_slot(session._state)
        assert after.values["available_mass_kg"] < before
        packet = session._state.metadata["removed_phase_inventory_history"][-1]["media_packet"]
        assert packet["used_mass_kg"] == pytest.approx(before-after.values["available_mass_kg"])
        exchanges = session._last_operation_record.material_balance.exchanges
        assert any(x.inventory_kind == "entity" and x.unit == "kg" for x in exchanges)
        replacement,slot = make_media_supply(session._state.vessel_id,mass_kg=0.,
                                             owner_id="explicitly-empty-media-kit")
        session.reset(options={"scope":"equipment","equipment_id":after.spec.owner_id,
            "replacement":replacement,"slots":StateSlots((slot,))})
        current = session._state
        result = session.execute({"operation":"dry"})
        # The physical-domain preflight rejects an exhausted source before
        # execution; the candidate never acquires replacement medium.
        assert result.info["transaction_status"] == "validation_failed"
        assert not result.info["preconditions"]["runtime_domain_valid"]
        assert session._state.phases == current.phases
        assert session._state.ledger.time_s == current.ledger.time_s
        session.reset(options={"scope":"material"})
        assert media_slot(session._state).values["available_mass_kg"] == 0
        assert media_slot(session._state).spec.owner_id == "explicitly-empty-media-kit"
    finally:
        env.close()


def test_extractant_identity_volume_and_contact_time_change_partition_outcome() -> None:
    settings = (
        (0, 0.012, 60.0),
        (3, 0.024, 480.0),
    )
    outcomes: list[tuple[float, float, float]] = []
    for extractant, volume, duration in settings:
        env = _run_to_selected_phase(
            extractant=extractant,
            extractant_volume_L=volume,
            mixing_duration_s=duration,
        )
        try:
            state = _runtime(env)._state
            outcomes.append(
                (
                    float(state.metadata["partition_coefficient"]),
                    float(state.process.metrics["purity"]),
                    float(state.process.metrics["recovery"]),
                )
            )
        finally:
            env.close()

    assert outcomes[1][0] > outcomes[0][0]
    assert outcomes[1][2] > outcomes[0][2]
    assert outcomes[1] != pytest.approx(outcomes[0])


def test_wash_exposes_purity_recovery_tradeoff_and_records_removed_outlet() -> None:
    env = _run_to_selected_phase()
    try:
        before = _runtime(env)._state
        before_purity = float(before.process.metrics["purity"])
        before_recovery = float(before.process.metrics["recovery"])
        _step_committed(env, {"operation": "wash", "wash_volume_L": 0.012})
        after = _runtime(env)._state

        assert after.process.metrics["purity"] >= before_purity
        assert after.process.metrics["recovery"] < before_recovery
        assert "wash_aqueous" not in after.phases.phases
        removed = after.metadata["removed_phase_inventory_history"][-1]
        assert removed["operation"] == "wash"
        assert "wash_aqueous" in removed["inventories"]
        assert after.process.waste_L > before.process.waste_L
        assert after.metadata["wash_material_balance_error_mol"] < 1.0e-10
        assert after.process.metrics["process_mass_balance_error"] < 1.0e-10
        assert after.phases.total_amounts_mol() == pytest.approx(
            after.species_amounts,
            abs=1.0e-12,
        )
    finally:
        env.close()


def test_separated_phase_is_the_only_feed_for_a_later_recontact() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    env.reset(seed=0)
    try:
        actions = _reaction_and_contact_actions()
        for action in actions[:-1]:
            _step_committed(env, action)
        before_separation = _runtime(env)._state
        phase_ledgers = (
            _runtime(env).runtime.domain_services.phase_separation.phase_ledgers
        )
        contact_target = phase_ledgers.phase_product_amount(before_separation)

        _step_committed(env, actions[-1])
        selected = _runtime(env)._state
        selected_target = phase_ledgers.phase_product_amount(selected)
        assert set(selected.phases.phases) == {"organic"}
        assert selected_target < contact_target
        assert selected.phases.total_amounts_mol() == pytest.approx(
            selected.species_amounts,
            abs=1.0e-12,
        )

        _step_committed(
            env,
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.010},
        )
        _step_committed(
            env,
            {
                "operation": "mix",
                "duration_s": 180.0,
                "stirring_speed_rpm": 700.0,
            },
        )
        recontacted = _runtime(env)._state
        assert phase_ledgers.phase_product_amount(recontacted) == pytest.approx(
            selected_target,
            abs=1.0e-12,
        )
        assert recontacted.phases.total_amounts_mol() == pytest.approx(
            recontacted.species_amounts,
            abs=1.0e-12,
        )
        assert _runtime(env).constitution.check_state(recontacted).passed
    finally:
        env.close()


def test_material_addition_after_phase_creation_updates_the_phase_source_of_truth() -> None:
    env = gym.make("ChemWorld", task_id="partition-discovery", seed=0)
    try:
        env.reset(seed=0)
        _step_committed(
            env,
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
        )
        _step_committed(
            env,
            {"operation": "add_extractant", "extractant": 3, "volume_L": 0.018},
        )
        _step_committed(env, {"operation": "add_reagent", "amount_mol": 0.010})
        state = _runtime(env)._state
        assert state.phases.total_amounts_mol() == pytest.approx(
            state.species_amounts,
            abs=1.0e-12,
        )
        assert _runtime(env).constitution.check_state(state).passed
    finally:
        env.close()


def test_downstream_operations_fail_closed_before_a_selected_material_phase() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        env.reset(seed=0)
        actions = (
            {"operation": "wash", "wash_volume_L": 0.008},
            {"operation": "dry"},
            {"operation": "concentrate", "duration_s": 600.0},
            {"operation": "transfer", "transfer_fraction": 0.90},
        )
        for action in actions:
            before = _runtime(env)._state
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] in {"validation_failed", "rolled_back"}
            assert info["constraint_flags"]["precondition_failed"] is True
            after = _runtime(env)._state
            assert after.species_amounts == before.species_amounts
            assert after.volume_L == before.volume_L
            assert after.temperature_K == before.temperature_K
            assert after.pressure_Pa == before.pressure_Pa
            assert after.phases == before.phases
            assert after.equipment == before.equipment
            assert after.ledger.cost >= before.ledger.cost
    finally:
        env.close()
