"""Exercise the environment entry independently of archived agent applications."""

import json
import subprocess
import sys

import pytest

from chemworld.cli import main
from chemworld.examples.user_reward import run_example
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def test_environment_and_research_imports_do_not_load_legacy_agents():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "from chemworld.cli import build_parser",
                    "from chemworld.eval.runner import run_agent",
                    "import chemworld.agents.experiment_documents",
                    "assert not any(m.startswith('chemworld.legacy') for m in sys.modules)",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_caller_reward_does_not_change_acquired_observations():
    without = run_example(reward_fn=None)
    custom = run_example(reward_fn=lambda action, observation: 17.0 if observation else None)
    assert [r["observation"] for r in custom["steps"]] == [
        r["observation"] for r in without["steps"]
    ]
    assert [r["user_reward"] for r in custom["steps"]] == [None, None, 17.0]
    assert all(r["user_reward"] is None for r in without["steps"])
    assert custom["steps"][-1]["observation"]["native"]["channels"][0]["unit"] == "Pa"


def test_default_cli_exports_environment_capabilities(capsys):
    main(["capabilities", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert any(row["module"] == "surface_kinetics" for row in rows)


@pytest.mark.parametrize("terminal_assay", [False, True])
@pytest.mark.parametrize("composition_id", ["my-instrument-study", "reaction-to-assay"])
def test_unscored_environment_prompt_uses_user_goal_without_benchmark_submission(
    terminal_assay, composition_id,
):
    spec = surface_world_spec(terminal_assay=terminal_assay).to_dict()
    spec["composition_id"] = composition_id
    spec["task"]["objective"] = "Compare instrument response across vessels"
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=12)
        before = session.snapshot().to_dict()
        prompt = session.task_prompt()
        assert prompt["task_goal"] == spec["task"]["objective"]
        assert prompt["submission_requirements"] == []
        assert prompt["threshold"] is None
        assert prompt["prompt_version"] == "chemworld-environment-prompt-0.1"
        assert "Reward and evaluation: user-defined" in prompt["text"]
        assert "native observation channels" in prompt["measurement_policy"]
        for obsolete in (
            "Optimize the public task score", "Balance score", "Submission requirements",
            "agent manifest", "Success metrics:",
        ):
            assert obsolete not in prompt["text"]
        assert prompt["allowed_operations"] == session.task_info()["allowed_operations"]
        if terminal_assay:
            template = prompt["experiment_lifecycle"]["final_assay_action_template"]
            assert "gas_final_assay" in template
        else:
            assert "No terminal assay is required" in prompt["text"]
        assert session.snapshot().to_dict() == before
    finally:
        session.close()


@pytest.mark.parametrize("command", ["run", "evaluate", "leaderboard", "suite", "baselines"])
def test_archived_commands_are_not_default_environment_commands(command):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2
