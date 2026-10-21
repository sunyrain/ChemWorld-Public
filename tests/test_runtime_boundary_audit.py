from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.world.operations import OPERATION_TYPES


def test_chemworld_env_has_no_operation_specific_step_branching() -> None:
    source = Path("src/chemworld/envs/chemworld_env.py").read_text(encoding="utf-8")

    assert "self.execute(action)" in source
    assert "def execute(" not in source
    assert ".domain_services" not in source
    assert "operation_record.operation_type" not in source
    for operation in OPERATION_TYPES:
        assert f'== "{operation}"' not in source
        assert f"== '{operation}'" not in source


def test_invalid_action_path_is_runtime_owned() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    env.reset(seed=0)
    before = env.unwrapped._state

    _obs, _reward, _terminated, _truncated, info = env.step(
        {"operation": "heat", "target_temperature_K": 360.0, "duration_s": 60.0}
    )
    after = env.unwrapped._state

    assert info["kernel_id"] == "chemworld.operation.heat"
    assert info["transaction_status"] == "rolled_back"
    assert not info["preconditions"]["has_volume"]
    assert not info["preconditions"]["has_material"]
    assert info["affected_ledgers"] == ["process"]
    assert after.phases == before.phases
    assert after.species == before.species
    assert after.equipment == before.equipment
    assert after.temperature_K == before.temperature_K
    assert after.ledger.time_s == before.ledger.time_s
    assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
    assert after.ledger.heat_reaction_J == before.ledger.heat_reaction_J
    assert after.ledger.sample_consumed_L == before.ledger.sample_consumed_L
    assert after.ledger.cost > before.ledger.cost
    env.close()


def test_optimized_python_keeps_runtime_validation_and_physical_execution() -> None:
    # Assertions in this child would vanish under -O. Inspect its real results
    # in the parent instead, including rejected attempts and a committed history.
    script = """
import json
from chemworld.runtime import ChemWorldRuntime
r = ChemWorldRuntime(task_id="reaction-to-assay", seed=0)
r.reset(seed=0)
initial = r._state
unknown = r.execute({"operation": "unknown-operation"}).info
empty = r.execute({"operation": "heat", "target_temperature_K": 360., "duration_s": 60.}).info
rejected = r._state
actions = [
    {"operation": "add_solvent", "volume_L": .028, "solvent": 2},
    {"operation": "add_reagent", "amount_mol": .01},
    {"operation": "heat", "target_temperature_K": 340., "duration_s": 60.},
]
statuses = [r.execute(action).info["transaction_status"] for action in actions]
print(json.dumps({
    "debug": __debug__,
    "unknown": [unknown["kernel_id"], unknown["transaction_status"]],
    "empty": [empty["kernel_id"], empty["transaction_status"]],
    "empty_preconditions": [empty["preconditions"]["has_volume"],
                            empty["preconditions"]["has_material"]],
    "rejected_physics_unchanged": all([
        rejected.phases == initial.phases, rejected.species == initial.species,
        rejected.temperature_K == initial.temperature_K,
        rejected.ledger.time_s == initial.ledger.time_s,
        rejected.ledger.energy_jacket_J == initial.ledger.energy_jacket_J,
    ]),
    "rejection_charged": rejected.ledger.cost > initial.ledger.cost,
    "statuses": statuses,
    "heated": r._state.temperature_K > initial.temperature_K,
    "elapsed": r._state.ledger.time_s - rejected.ledger.time_s,
    "constitution_passed": r.constitution.check_state(r._state).passed,
}))
r.close()
"""
    child = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-O", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(child.stdout)
    assert result == {
        "debug": False,
        "unknown": ["validation:invalid_action", "validation_failed"],
        "empty": ["chemworld.operation.heat", "rolled_back"],
        "empty_preconditions": [False, False],
        "rejected_physics_unchanged": True,
        "rejection_charged": True,
        "statuses": ["committed"] * 3,
        "heated": True,
        "elapsed": 60.0,
        "constitution_passed": True,
    }
