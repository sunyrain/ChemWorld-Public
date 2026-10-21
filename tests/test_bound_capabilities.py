"""Bound capabilities must agree with actual validation without advancing a world."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.agent_interface import resource_blocked_actions, tool_json_view
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.examples.native_capabilities import bound_capabilities
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec
from chemworld.world.surface_world import surface_world_spec


def instrument_field(session):
    return session.action_schema("measure")["fields"][0]


def run(session, action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_installed_task_and_termination_are_distinct_and_executable():
    spec = surface_world_spec(terminal_assay=True)
    spec = replace(spec, task=replace(
        spec.task, instruments=("gas_pressure", "gas_final_assay")
    ))
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=7)
        field = instrument_field(session)
        assert set(field["installed_choices"]) == {
            "gas_pressure", "gas_composition", "gas_final_assay"
        }
        assert field["task_allowed_choices"] == ["gas_final_assay", "gas_pressure"]
        assert field["choices"] == ["gas_pressure"]
        statuses = {r["instrument"]: r for r in field["choice_status"]}
        assert not statuses["gas_composition"]["task_allowed"]
        assert "instrument_allowed_by_task" in statuses["gas_composition"]["invalid_reasons"]
        assert "measure_final_requires_terminated" in statuses["gas_final_assay"]["invalid_reasons"]
        run(session, {"operation": "measure", "instrument": "gas_pressure"})
        run(session, {"operation": "terminate"})
        assert instrument_field(session)["choices"] == ["gas_final_assay"]
        action = session.sample_valid_action(np.random.default_rng(1), operation="measure")
        assert action == {"operation": "measure", "instrument": "gas_final_assay"}
        run(session, action)
        assert instrument_field(session)["choices"] == []
    finally:
        session.close()


def test_vessel_switch_uses_its_instrument_instead_of_a_global_default():
    spec = surface_world_spec()
    spec = replace(spec, components=tuple(
        replace(c, parameters={
            "instruments": ["gas_pressure" if c.vessel == "reactor" else "gas_composition"]
        }) if c.kind == "observation" else c
        for c in spec.components
    ))
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=7)
        assert instrument_field(session)["choices"] == ["gas_pressure"]
        run(session, {"operation": "select_vessel", "vessel": "receiver"})
        assert instrument_field(session)["choices"] == ["gas_composition"]
        rows = {r["operation"]: r for r in session.available_actions()}
        assert rows["measure"]["valid"]
        assert "measure" in session.operation_validator.valid_operations(session._state)
        action = session.sample_valid_action(np.random.default_rng(2), operation="measure")
        assert action["instrument"] == "gas_composition"
        run(session, action)
    finally:
        session.close()


def test_resource_exhaustion_preserves_installation_and_reports_exact_rejection():
    card = CampaignResourceCard(
        card_id="bound-instruments", operation_attempt_limit=12, vessel_start_limit=1,
        final_assay_limit=1, nonfinal_instrument_use_limit=3, stock_limits={},
        per_instrument_limits={"hplc": 1, "uvvis": 0, "gc": 0, "nmr": 0, "ir": 0, "ms": 0},
    )
    session = ChemWorldRuntime(
        task_id="reaction-to-assay", campaign_resource_card=card,
        episode_mode_override="campaign", budget_override=20,
    )
    try:
        session.reset(seed=0)
        run(session, {"operation": "add_solvent", "volume_L": 0.02, "solvent": 0})
        run(session, {"operation": "add_reagent", "amount_mol": 0.001})
        assert instrument_field(session)["choices"] == ["hplc"]
        run(session, {"operation": "measure", "instrument": "hplc"})
        field = instrument_field(session)
        assert field["choices"] == []
        row = next(r for r in field["choice_status"] if r["instrument"] == "hplc")
        assert row["installed"] and row["task_allowed"] and not row["valid"]
        assert "campaign_resource:per_instrument_limit:hplc" in row["invalid_reasons"]
        assert not session.validate_action({"operation": "measure", "instrument": "hplc"})["valid"]
        assert "measure" not in {r["operation"] for r in session.available_actions()}
        blocked = next(r for r in resource_blocked_actions(session) if r["operation"] == "measure")
        assert "per_instrument_limit:hplc" in blocked["reason_codes"]
        with pytest.raises(ValueError, match="No currently available"):
            session.sample_valid_action(np.random.default_rng(3), operation="measure")
    finally:
        session.close()


def test_queries_and_sampling_preserve_snapshot_and_native_battery_contract():
    session = ChemWorldRuntime(composition=battery_world_spec())
    try:
        _, initial_info = session.reset(seed=7)
        before = session.snapshot().to_dict()
        schema = session.action_schema("configure_electrical_control")
        current = next(f for f in schema["fields"] if f["field"] == "current_A")
        assert current["unit"] == "A"
        from chemworld.runtime.control_program import electrical_device

        settings = electrical_device(session._state).settings["electrical_control"]
        assert current["bounds"] == {
            "low": settings["limits_A"][0], "high": settings["limits_A"][1]
        }
        assert schema["installed"] and schema["task_allowed"]
        for payload in (
            {}, {"current_A": current["bounds"]["high"] + 0.01},
            {"current_A": 0.1, "potential_V": 4},
        ):
            assert not session.validate_action({
                "operation": "configure_electrical_control", **payload
            })["valid"]
        rng = np.random.default_rng(4)
        for _ in range(6):
            action = session.sample_valid_action(rng, operation="configure_electrical_control")
            if "current_A" in action:
                assert current["bounds"]["low"] <= action["current_A"] <= current["bounds"]["high"]
            assert session.validate_action(action)["valid"]
        capabilities = bound_capabilities(session)
        assert [r["instrument"] for r in capabilities["instruments"]] == ["battery_meter"]
        view = tool_json_view(session, {}, initial_info)
        assert "measure" in {r["operation"] for r in view["available_actions"]}
        assert session.snapshot().to_dict() == before
        run(session, action)
        reading = run(session, {"operation": "measure", "instrument": "battery_meter"})
        assert reading.info["native_observation"]["instrument_id"] == "battery_meter"
    finally:
        session.close()


def test_operation_permissions_do_not_relabel_installed_hardware():
    spec = battery_world_spec()
    spec = replace(spec, task=replace(spec.task, operations=("measure", "terminate")))
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=7)
        schema = session.action_schema("configure_electrical_control")
        assert schema["installed"] and not schema["task_allowed"]
        row = next(
            r for r in session.available_actions(include_invalid=True)
            if r["operation"] == "configure_electrical_control"
        )
        assert row["installed"] and not row["task_allowed"] and not row["valid"]
        assert "operation_allowed_by_task" in row["invalid_reasons"]
        assert not session.validate_action({
            "operation": "configure_electrical_control", "current_A": 0.01
        })["valid"]
        unknown = session.action_schema("not_a_registered_operation")
        assert not unknown["installed"] and not unknown["valid_operation_type"]
    finally:
        session.close()


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_sampler_rejects_invalid_search_limits_without_an_attempt(attempts):
    session = ChemWorldRuntime(composition=battery_world_spec())
    try:
        session.reset(seed=7)
        before = session.snapshot().to_dict()
        with pytest.raises(ValueError, match="positive integer"):
            session.sample_valid_action(np.random.default_rng(0), max_attempts=attempts)
        assert session.snapshot().to_dict() == before
    finally:
        session.close()
