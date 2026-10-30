from __future__ import annotations

import gymnasium as gym
import pytest

import chemworld
from chemworld.agent_interface import agent_view_bundle
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.eval.verify import verify_records
from chemworld.world.composition import WorldCompositionError


def _request() -> dict[str, object]:
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "composed-reaction-assay-test",
        "world_split": "public-dev",
        "components": [
            {"kind": "reaction", "role": "transformation", "parameters": {}},
            {"kind": "thermal", "role": "temperature-and-energy", "parameters": {}},
            {"kind": "observation", "role": "public-measurement", "parameters": {}},
        ],
        "task": {
            "budget": 8,
            "resources": {"operation_budget": 8},
            "description": "Declarative reaction-world test.",
        },
    }


def _diagnostic_codes(error: WorldCompositionError) -> set[str]:
    return {item.code for item in error.diagnostics}


def test_compile_world_composition_exposes_complete_public_surface() -> None:
    compiled = chemworld.compile_world_composition(_request())
    surface = compiled.to_public_dict()

    assert compiled.task_spec.task_id == "composed-reaction-assay-test"
    assert compiled.scenario_spec.scenario_id == "reaction-to-assay"
    assert [item["kind"] for item in surface["world"]["components"]] == [
        "reaction",
        "thermal",
        "observation",
    ]
    assert set(surface["task"]["operations"]) == set(
        compiled.task_spec.allowed_operations
    )
    assert set(surface["task"]["instruments"]) == set(
        compiled.task_spec.allowed_instruments
    )
    assert surface["task"]["resources"]["operation_budget"] == 8
    assert surface["task"]["termination"] == "final-assay-or-budget"
    assert surface["task"]["evaluation"]["metrics"] == [
        "final_assay_score",
        "trajectory_validity",
    ]


@pytest.mark.parametrize(
    ("component_kinds", "scenario_id"),
    (
        (("reaction", "thermal", "observation"), "reaction-to-assay"),
        (
            ("reaction", "thermal", "phase", "separation", "observation"),
            "reaction-to-purification",
        ),
        (("phase", "separation", "observation"), "partition-discovery"),
        (
            ("reaction", "thermal", "crystallization", "observation"),
            "reaction-to-crystallization",
        ),
        (
            ("reaction", "thermal", "distillation", "observation"),
            "reaction-to-distillation",
        ),
        (
            ("reaction", "thermal", "continuous_flow", "observation"),
            "flow-reaction-optimization",
        ),
        (
            ("reaction", "electrochemistry", "observation"),
            "electrochemical-conversion",
        ),
        (("phase", "observation"), "equilibrium-characterization"),
    ),
)
def test_registered_component_patterns_compile(
    component_kinds: tuple[str, ...],
    scenario_id: str,
) -> None:
    request = _request()
    request["composition_id"] = f"composed-{scenario_id}-test"
    request["components"] = [
        {"kind": kind, "role": kind, "parameters": {}} for kind in component_kinds
    ]
    request["task"] = {"description": "Registered composition-pattern test."}

    compiled = chemworld.compile_world_composition(request)

    assert compiled.scenario_spec.scenario_id == scenario_id
    assert set(compiled.spec.component_kinds) == set(component_kinds)
    assert "terminate" in compiled.task_spec.allowed_operations
    assert "measure" in compiled.task_spec.allowed_operations
    assert "final_assay" in compiled.task_spec.allowed_instruments


def test_composed_world_runs_complete_lifecycle() -> None:
    env = gym.make("ChemWorld", composition=_request(), seed=4)
    try:
        _, info = env.reset(seed=4)
        assert info["task_id"] == "composed-reaction-assay-test"
        assert info["composition"] == env.unwrapped.compiled_composition.to_public_dict()
        actions = (
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.01},
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.0002,
                "catalyst": 1,
            },
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1200.0,
                "stirring_speed_rpm": 700.0,
            },
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        )
        for action in actions:
            _, _, terminated, truncated, step_info = env.step(action)
            assert step_info["transaction_status"] == "committed"
        assert terminated
        assert not truncated
        assert step_info["leaderboard_score"] is not None
    finally:
        env.close()


def test_composed_world_trajectory_replays_from_logged_request(tmp_path) -> None:
    env = gym.make("ChemWorld", composition=_request(), seed=4)
    trajectory_path = tmp_path / "composed-world.jsonl"
    actions = (
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.01},
        {
            "operation": "heat",
            "target_temperature_K": 385.0,
            "duration_s": 1200.0,
            "stirring_speed_rpm": 700.0,
        },
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    )
    try:
        observation, _ = env.reset(seed=4)
        task_info = {
            **env.unwrapped.task_info(),
            **env.unwrapped.evaluator_provenance(),
        }
        assert task_info["composition_request"] == env.unwrapped.compiled_composition.spec.to_dict()
        with TrajectoryLogger(trajectory_path) as logger:
            for step, action in enumerate(actions, start=1):
                observation, reward, terminated, truncated, info = env.step(action)
                logger.log(
                    task_info=task_info,
                    step=step,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={"agent_id": "composition-replay-test"},
                    agent_view=agent_view_bundle(env, observation, info),
                )
    finally:
        env.close()

    records = load_jsonl(trajectory_path)
    expected_request = chemworld.compile_world_composition(_request()).spec.to_dict()
    assert records[0]["composition_request"] == expected_request
    assert verify_records(records, tolerance=0.0).verified


def test_unregistered_component_combination_fails_before_execution() -> None:
    request = _request()
    request["components"] = [
        {"kind": "distillation", "role": "fractionation", "parameters": {}},
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]

    with pytest.raises(WorldCompositionError, match="missing_dependency") as exc:
        chemworld.compile_world_composition(request)
    assert "missing_dependency" in _diagnostic_codes(exc.value)


def test_fixed_task_and_composition_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        gym.make(
            "ChemWorld",
            task_id="reaction-to-assay",
            composition=_request(),
        )


def test_composed_world_rejects_reset_scenario_override() -> None:
    env = gym.make("ChemWorld", composition=_request(), seed=0)
    try:
        with pytest.raises(ValueError, match="unavailable for composed worlds"):
            env.reset(options={"scenario_id": "reaction-to-distillation"})
    finally:
        env.close()


def test_compatibility_report_is_public_and_preexecution() -> None:
    report = chemworld.check_world_composition_compatibility(_request())
    compiled = chemworld.compile_world_composition(_request())

    assert report.compatible
    assert report.pattern == "component-network"
    assert report.minimum_resources["operation_budget"] == 4
    assert report.state_owners["reactor:temperature_control"] == "thermal"
    assert compiled.to_public_dict()["compatibility"] == report.to_dict()


def test_missing_dependency_is_reported_before_pattern_resolution() -> None:
    request = _request()
    request["components"] = [
        {"kind": "reaction", "role": "reaction", "parameters": {}},
        {"kind": "crystallization", "role": "crystal", "parameters": {}},
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]

    with pytest.raises(WorldCompositionError) as exc:
        chemworld.compile_world_composition(request)

    assert "missing_dependency" in _diagnostic_codes(exc.value)
    assert any(item.components == ("crystallization", "thermal") for item in exc.value.diagnostics)


def test_conflicting_state_owners_are_reported() -> None:
    request = _request()
    request["components"] = [
        {"kind": "reaction", "role": "reaction", "parameters": {}},
        {"kind": "thermal", "role": "thermal", "parameters": {}},
        {"kind": "phase", "role": "phase", "parameters": {}},
        {"kind": "crystallization", "role": "crystal", "parameters": {}},
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]

    with pytest.raises(WorldCompositionError) as exc:
        chemworld.compile_world_composition(request)

    assert "conflicting_state_owner" in _diagnostic_codes(exc.value)
    assert any("phase_transition" in item.message for item in exc.value.diagnostics)


def test_duplicate_component_kind_is_a_structured_conflict() -> None:
    request = _request()
    request["components"] = [
        {"kind": "reaction", "role": "first", "parameters": {}},
        {"kind": "reaction", "role": "second", "parameters": {}},
        {"kind": "thermal", "role": "thermal", "parameters": {}},
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]

    with pytest.raises(WorldCompositionError) as exc:
        chemworld.compile_world_composition(request)

    assert _diagnostic_codes(exc.value) == {"conflicting_state_owner"}


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value", "expected_code"),
    (
        (
            "temperature_range_K",
            {"value": [280.0, 320.0], "unit": "L"},
            "unit_mismatch",
        ),
        ("potential_range_V", [-1.0, 1.0], "unsupported_parameter"),
        ("temperature_range_K", [240.0, 320.0], "invalid_parameter"),
    ),
)
def test_component_parameter_failures_are_location_aware(
    parameter_name: str,
    parameter_value: object,
    expected_code: str,
) -> None:
    request = _request()
    request["components"] = [
        {"kind": "reaction", "role": "reaction", "parameters": {}},
        {
            "kind": "thermal",
            "role": "thermal",
            "parameters": {parameter_name: parameter_value},
        },
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]

    with pytest.raises(WorldCompositionError) as exc:
        chemworld.compile_world_composition(request)

    matching = [item for item in exc.value.diagnostics if item.code == expected_code]
    assert matching
    assert matching[0].path.endswith(parameter_name)


def test_resource_and_lifecycle_impossibilities_fail_before_execution() -> None:
    request = _request()
    request["task"] = {
        "budget": 3,
        "operations": ["add_solvent", "add_reagent", "measure"],
        "instruments": ["final_assay"],
        "resources": {
            "operation_budget": 3,
            "sample_volume_L": 0.0001,
            "instrument_uses": 0,
            "final_assays": 0,
        },
    }

    with pytest.raises(WorldCompositionError) as exc:
        chemworld.compile_world_composition(request)

    assert {"lifecycle_hole", "resource_impossibility"}.issubset(
        _diagnostic_codes(exc.value)
    )


def test_explicit_empty_instrument_subset_is_rejected_by_preflight() -> None:
    request = _request()
    request["task"] = {
        "description": "An explicit empty instrument surface must not regain defaults.",
        "instruments": [],
    }

    report = chemworld.check_world_composition_compatibility(request)

    assert not report.compatible
    assert any(
        item.code == "lifecycle_hole" and "final_assay" in item.message
        for item in report.diagnostics
    )
    with pytest.raises(WorldCompositionError, match="lifecycle_hole"):
        chemworld.compile_world_composition(request)


def test_multi_vessel_instrument_surface_must_have_a_provider() -> None:
    request = _request()
    request["components"] = [
        {
            "id": f"{vessel}-{kind}",
            "kind": kind,
            "vessel": vessel,
            "parameters": (
                {"instruments": ["hplc"]}
                if vessel == "a" and kind == "observation"
                else {"instruments": ["gc"]}
                if vessel == "b" and kind == "observation"
                else {}
            ),
        }
        for vessel in ("a", "b")
        for kind in ("reaction", "thermal", "observation")
    ]
    request["connections"] = [
        {
            "id": "a-to-b",
            "source": {"component": "a-reaction", "port": "out"},
            "target": {"component": "b-reaction", "port": "in"},
            "unit": "mol",
        }
    ]

    report = chemworld.check_world_composition_compatibility(request)

    assert not report.compatible
    assert any(
        item.code == "interface_mismatch"
        and item.path == "task.instruments"
        and "final_assay" in item.message
        for item in report.diagnostics
    )
    with pytest.raises(WorldCompositionError, match="interface_mismatch"):
        chemworld.compile_world_composition(request)


def test_multi_vessel_instrument_union_binds_local_runtime_surfaces() -> None:
    request = _request()
    request["components"] = [
        {
            "id": f"{vessel}-{kind}",
            "kind": kind,
            "vessel": vessel,
            "parameters": (
                {"instruments": ["hplc", "final_assay"]}
                if vessel == "a" and kind == "observation"
                else {"instruments": ["gc"]}
                if vessel == "b" and kind == "observation"
                else {}
            ),
        }
        for vessel in ("a", "b")
        for kind in ("reaction", "thermal", "observation")
    ]
    request["connections"] = [
        {
            "id": "a-to-b",
            "source": {"component": "a-reaction", "port": "out"},
            "target": {"component": "b-reaction", "port": "in"},
            "unit": "mol",
        }
    ]
    request["task"] = {
        "budget": 40,
        "instruments": ["hplc", "gc", "final_assay"],
    }

    compiled = chemworld.compile_world_composition(request)
    env = gym.make("ChemWorld", composition=compiled, seed=0)
    try:
        env.reset(seed=0)
        state = env.unwrapped._state
        assert state.metadata["component_network"]["instruments"] == {
            "a": ["hplc", "final_assay"],
            "b": ["gc"],
        }
    finally:
        env.close()


def test_specialized_membrane_requires_native_physics_binding() -> None:
    request = _request()
    request["components"] = [
        {"kind": kind, "role": kind, "parameters": {}}
        for kind in ("reaction", "thermal", "continuous_flow", "ro_membrane", "observation")
    ]

    report = chemworld.check_world_composition_compatibility(request)

    assert not report.compatible
    assert any(
        item.code == "missing_physics_binding" and item.components == ("ro_membrane",)
        for item in report.diagnostics
    )
    with pytest.raises(WorldCompositionError, match="missing_physics_binding"):
        chemworld.compile_world_composition(request)


@pytest.mark.parametrize("component_kind", ("battery_spm", "battery_cell", "battery_thevenin"))
def test_specialized_battery_requires_native_physics_binding(component_kind: str) -> None:
    request = _request()
    request["components"] = [
        {"kind": component_kind, "role": component_kind, "parameters": {}},
        {"kind": "observation", "role": "observation", "parameters": {}},
    ]

    report = chemworld.check_world_composition_compatibility(request)

    assert not report.compatible
    assert any(
        item.code == "missing_physics_binding" and item.components == (component_kind,)
        for item in report.diagnostics
    )
    with pytest.raises(WorldCompositionError, match="missing_physics_binding"):
        chemworld.compile_world_composition(request)


def test_authored_units_are_converted_and_bounds_reach_runtime_validator() -> None:
    request = _request()
    request["components"] = [
        {"kind": "reaction", "role": "reaction", "parameters": {}},
        {
            "kind": "thermal",
            "role": "thermal",
            "parameters": {
                "temperature_range_K": {
                    "value": [26.85, 46.85],
                    "unit": "degC",
                }
            },
        },
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]
    env = gym.make("ChemWorld", composition=request, seed=0)
    try:
        env.reset(seed=0)
        accepted = env.unwrapped.validate_action(
            {
                "operation": "heat",
                "target_temperature_K": 310.0,
                "duration_s": 10.0,
                "stirring_speed_rpm": 600.0,
            }
        )
        rejected = env.unwrapped.validate_action(
            {
                "operation": "heat",
                "target_temperature_K": 350.0,
                "duration_s": 10.0,
                "stirring_speed_rpm": 600.0,
            }
        )

        assert accepted["preconditions"]["payload_bounds:target_temperature_K"] is True
        assert rejected["preconditions"]["payload_bounds:target_temperature_K"] is False
    finally:
        env.close()


def test_composed_flagship_keeps_runtime_workflow_profile() -> None:
    request = _request()
    request["composition_id"] = "composed-crystallization-workflow-test"
    request["components"] = [
        {"kind": "reaction", "role": "reaction", "parameters": {}},
        {"kind": "thermal", "role": "thermal", "parameters": {}},
        {"kind": "crystallization", "role": "crystal", "parameters": {}},
        {"kind": "observation", "role": "measurement", "parameters": {}},
    ]
    request["task"] = {"description": "Composed crystallization workflow test."}
    env = gym.make("ChemWorld", composition=request, seed=0)
    try:
        env.reset(seed=0)
        result = env.unwrapped.validate_action(
            {"operation": "seed_crystals", "seed_mass_g": 0.0001}
        )

        assert env.unwrapped.task_id == "composed-crystallization-workflow-test"
        assert env.unwrapped.runtime_task_profile_id == "reaction-to-crystallization"
        assert result["preconditions"]["seed_crystals_requires_reaction_advance"] is False
    finally:
        env.close()
