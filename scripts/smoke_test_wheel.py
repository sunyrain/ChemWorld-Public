"""Install wheel or sdist in a fresh, locked environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = """
import json, sys, platform
from pathlib import Path
from importlib.metadata import version, distribution
from importlib.resources import files
import chemworld, gymnasium as gym
from chemworld.tasks import list_tasks
from chemworld.task_design import serious_task_readiness_manifest
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.eval.risk_policy import load_risk_cost_protocol
from chemworld.eval.resource_accounting import load_resource_protocol
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID

assert Path(chemworld.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert version('chemworld-bench') == chemworld.__version__
tasks = []
for task in list_tasks():
    env = gym.make('ChemWorld', task_id=task.task_id, seed=0)
    try:
        _, info = env.reset(seed=0)
        assert info['task_id'] == task.task_id
        assert env.unwrapped.available_actions()
        tasks.append(task.task_id)
    finally:
        env.close()
readiness = serious_task_readiness_manifest()
assert readiness['contract_ready_count'] == len(readiness['task_ids'])
assert load_risk_cost_protocol()
assert load_resource_protocol()
root = configuration_root()
assert root.resolve().is_relative_to(Path(sys.prefix).resolve())
assert not (root / 'current.json').exists()
assert not (root / 'private_eval.placeholder.json').exists()
assert not (root / 'providers').exists()
for asset in ('index.html', 'lab.css', 'lab.js'):
    assert files('chemworld.lab').joinpath('static', asset).is_file()
assert any(e.name == 'chemworld-lab' for e in distribution('chemworld-bench').entry_points)
from chemworld.lab.session import LabSessionManager
manager = LabSessionManager()
session = manager.create('reaction-to-assay', 0)
for action in (
    {'operation': 'add_solvent', 'volume_L': 0.025, 'solvent': 0},
    {'operation': 'add_reagent', 'amount_mol': 0.012},
    {'operation': 'terminate'},
    {'operation': 'measure', 'instrument': 'final_assay'},
):
    assert session.step(action)['accepted']
assert session.replay()['verified']
assert manager.close(session.session_id)['final_assay_count'] == 1
print(json.dumps({'package': chemworld.__file__, 'version': chemworld.__version__,
                  'tasks': tasks, 'contracts': readiness['contract_ready_count'],
                  'platform': {'system': platform.system(), 'machine': platform.machine(),
                               'python_version': platform.python_version()},
                  'runtime_semantics_id': RUNTIME_SEMANTICS_ID}))
"""


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def captured_run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(
            f"Installed command failed ({result.returncode}): {result.stdout}\n{result.stderr}"
        )
    return result


def check_install(archive: Path, *, python: str, root: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="chemworld-installed-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        child_env = clean_environment()
        print(f"Installing {archive.name} on {python}", flush=True)
        subprocess.run(["uv", "venv", "--python", python, str(environment)], check=True)
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = subprocess.check_output(
            ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes"],
            cwd=root,
            text=True,
        )
        lock = work / "requirements.txt"
        lock.write_text(requirements, encoding="utf-8")
        subprocess.run(
            ["uv", "pip", "sync", "--python", str(interpreter), str(lock)],
            cwd=work,
            env=child_env,
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(interpreter), "--no-deps", str(archive)],
            cwd=work,
            env=child_env,
            check=True,
        )
        probe = captured_run(
            [str(interpreter), "-I", "-c", PROBE],
            cwd=work,
            env=child_env,
        )
        payload = json.loads(probe.stdout)
        example = root / "examples/demo_manual_event_sequence.py"
        completed = captured_run(
            [str(interpreter), "-I", str(example)],
            cwd=work,
            env=child_env,
        )
        assert json.loads(completed.stdout.splitlines()[-1])["final_assay_completed"]
        captured_run(
            [
                str(interpreter),
                "-I",
                str(root / "examples/demo_minimal_agent.py"),
                "--output",
                str(work / "minimal-agent.jsonl"),
            ],
            cwd=work,
            env=child_env,
        )
        research = work / "offline-research"
        captured_run(
            [
                str(interpreter),
                "-I",
                str(root / "examples/demo_offline_research.py"),
                "--output",
                str(research),
            ],
            cwd=work,
            env=child_env,
        )
        research_summary = json.loads((research / "summary.json").read_text())
        assert research_summary["accepted"]
        trajectory = work / "trajectory.jsonl"
        for args in (
            ["tasks", "list"],
            [
                "run",
                "--task",
                "reaction-to-assay",
                "--agent",
                "random",
                "--seed",
                "0",
                "--output",
                str(trajectory),
            ],
            ["verify", "--submission", str(trajectory)],
            ["evaluate", "--submission", str(trajectory), "--output", str(work / "result.json")],
        ):
            result = captured_run(
                [str(interpreter), "-I", "-c", "from chemworld.cli import main; main()", *args],
                cwd=work,
                env=child_env,
            )
            if args[0] == "verify":
                assert json.loads(result.stdout)["verified"] is True
        return {
            "archive": archive.name,
            "python": python,
            "installed_check": "passed",
            "offline_research": {
                "accepted": research_summary["accepted"],
                "new_process_replays": len(research_summary["new_process_replay"]),
            },
            **payload,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Existing wheel or sdist; otherwise build both"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, help="Optional JSON result, outside the repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="chemworld-build-") as temporary:
        if args.archive:
            archives = [args.archive.resolve()]
        else:
            subprocess.run(["uv", "build", "--out-dir", temporary], cwd=root, check=True)
            archives = sorted([*Path(temporary).glob("*.whl"), *Path(temporary).glob("*.tar.gz")])
        rows = [check_install(archive, python=args.python, root=root) for archive in archives]
    report = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
