from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

import chemworld
from chemworld.tasks import get_task, list_tasks

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "world-authoring"
REFERENCE_PATHS = EXAMPLE_ROOT / "use-case-reference-paths-v0.1.json"
COMPOSITION_EXAMPLES = (
    (
        "composed-equilibrium-characterization-v0.1.json",
        "phase-observation",
        {"phase", "observation"},
    ),
    (
        "composed-reaction-assay-v0.1.json",
        "reaction-thermal-observation",
        {"reaction", "thermal", "observation"},
    ),
    (
        "composed-reaction-purification-v0.1.json",
        "reaction-phase-separation-observation",
        {"reaction", "thermal", "phase", "separation", "observation"},
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("filename", "expected_pattern", "expected_components"),
    COMPOSITION_EXAMPLES,
)
def test_public_composition_examples_compile(
    filename: str,
    expected_pattern: str,
    expected_components: set[str],
) -> None:
    request = _read_json(EXAMPLE_ROOT / filename)
    report = chemworld.check_world_composition_compatibility(request)
    compiled = chemworld.compile_world_composition(request)

    assert report.compatible
    assert report.pattern == "component-network"
    assert compiled.compatibility == report
    assert set(compiled.spec.component_kinds) == expected_components
    assert compiled.task_spec.task_id == request["composition_id"]
    assert "terminate" in compiled.task_spec.allowed_operations
    assert "measure" in compiled.task_spec.allowed_operations
    assert "final_assay" in compiled.task_spec.allowed_instruments
    if expected_pattern == "phase-observation":
        assert set(compiled.task_spec.allowed_operations) == {
            "add_solvent",
            "add_reagent",
            "measure",
            "terminate",
        }
    json.dumps(compiled.to_public_dict(), allow_nan=False)


def _physical_state(env: gym.Env[Any, Any]) -> dict[str, Any]:
    state = env.unwrapped._state.to_dict(include_hidden=True)
    state.pop("ledger", None)
    state.pop("process", None)
    return state


def test_prelaunch_reference_paths_execute_and_fail_closed() -> None:
    specification = _read_json(REFERENCE_PATHS)

    assert specification["status"] == "frozen_prelaunch_specification"
    assert [case["use_case_id"] for case in specification["cases"]] == ["U02", "U03"]

    for case in specification["cases"]:
        request = _read_json(ROOT / case["composition_request"])
        compatibility = chemworld.check_world_composition_compatibility(request)
        compiled = chemworld.compile_world_composition(request)

        assert compatibility.compatible
        assert compatibility.pattern == "component-network"
        assert request["composition_id"] == case["composition_id"]
        assert compiled.task_spec.objective == case["objective"]
        assert len(case["actions"]) == case["submitted_action_count"]
        assert len(case["actions"]) <= compiled.task_spec.budget

        env = gym.make("ChemWorld", composition=request, seed=case["seed"])
        env.reset(seed=case["seed"])
        try:
            observed_validation: list[bool] = []
            observed_transactions: list[str] = []
            final_terminated = False
            final_truncated = False
            first_physical_state = _physical_state(env)

            for step, action in enumerate(case["actions"], start=1):
                validation = env.unwrapped.validate_action(action)
                _, _, terminated, truncated, info = env.step(action)
                observed_validation.append(bool(validation["valid"]))
                observed_transactions.append(str(info["transaction_status"]))
                final_terminated = bool(terminated)
                final_truncated = bool(truncated)

                expected_failure = case["expected_failure"]
                if expected_failure is not None and step == expected_failure["step"]:
                    assert info["transaction_status"] == expected_failure[
                        "transaction_status"
                    ]
                    assert info["rollback_reason"] == expected_failure["rollback_reason"]
                    assert _physical_state(env) == first_physical_state

            assert observed_validation == case["expected_validation"]
            assert observed_transactions == case["expected_transactions"]
            assert observed_transactions[-1] == case["expected_final"][
                "transaction_status"
            ]
            assert final_terminated is case["expected_final"]["terminated"]
            assert final_truncated is case["expected_final"]["truncated"]
        finally:
            env.close()


def test_reference_task_map_covers_registry_once_and_uses_registered_patterns() -> None:
    mapping = _read_json(EXAMPLE_ROOT / "reference-task-contract-map-v0.1.json")
    patterns = mapping["patterns"]
    assert isinstance(patterns, list)

    mapped_task_ids: list[str] = []
    for pattern in patterns:
        component_kinds = pattern["component_kinds"]
        request = {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"mapping-check-{pattern['pattern_id']}",
            "world_split": "public-test",
            "components": [
                {"kind": kind, "role": kind, "parameters": {}}
                for kind in component_kinds
            ],
            "task": {},
        }
        report = chemworld.check_world_composition_compatibility(request)
        compiled = chemworld.compile_world_composition(request)
        anchor = get_task(compiled.runtime_task_profile_id)

        assert report.compatible
        assert report.pattern == "component-network"
        for task_id in pattern["reference_task_ids"]:
            task = get_task(task_id)
            assert task.allowed_operations == anchor.allowed_operations
            assert task.allowed_instruments == anchor.allowed_instruments
        mapped_task_ids.extend(pattern["reference_task_ids"])

    registered_task_ids = {task.task_id for task in list_tasks()}
    assert len(mapped_task_ids) == len(set(mapped_task_ids)) == 15
    assert set(mapped_task_ids) == registered_task_ids


def test_reference_task_map_preserves_claim_boundaries() -> None:
    mapping = _read_json(EXAMPLE_ROOT / "reference-task-contract-map-v0.1.json")

    assert mapping["mapping_boundary"] == {
        "component_pattern_explains_world_surface": True,
        "registered_task_contracts_remain_frozen": True,
        "byte_for_byte_reconstruction_claim": False,
        "qualification_claim": False,
        "exhaustive_task_space_claim": False,
    }
    assert set(mapping["task_overlay_fields"]) == {
        "objective",
        "budget",
        "episode_mode",
        "allowed_operations",
        "allowed_instruments",
        "observation_policy",
        "termination_policy",
        "success_metrics",
        "world_split",
        "seeds",
    }
