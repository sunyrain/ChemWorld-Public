"""Install wheel or sdist in a fresh, locked environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
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

NATIVE_PROBE = """
import importlib.util, json, platform, sys, tempfile
from pathlib import Path
import chemworld
from chemworld.examples.native_surface_research import run_example
from chemworld.examples.reuse_native_equipment import run_example as reuse_example
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.envs.spaces import to_observation
from chemworld.eval.verify import verify_records

assert Path(chemworld.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert importlib.util.find_spec('cantera') is None
cases = []
for surface, bulk in ((False,False), (False,True), (True,False), (True,True)):
    for assay in (False, True):
        result = run_example(include_surface=surface, include_bulk=bulk,
                             mobile=surface and assay, terminal_assay=assay)
        assert result['ended']
        assert all(row['transaction'] == 'committed' for row in result['steps'])
        cases.append({'surface':surface, 'bulk':bulk, 'assay':assay,
                      'physical_time_s':result['physical_time_s']})
reuse = reuse_example()
assert len(reuse['configurations']) == 4
assert all(row['same_heater_after_material_initialization'] for row in reuse['configurations'])
assert all(value for key,value in reuse['task_binding'].items() if key != 'assay_readings')
session = ChemWorldRuntime(composition=surface_world_spec(), observation_noise_mode='keyed')
session.reset(seed=12)
session.execute({'operation':'add_component', 'component':'CO-feed', 'amount_mol':2e-5})
snapshot = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
restored = ChemWorldRuntime.from_snapshot(snapshot)
action = {'operation':'measure', 'instrument':'gas_pressure'}
expected, actual = session.execute(action), restored.execute(action)
assert actual == expected
assert restored._state.to_dict() == session._state.to_dict()
session.close()
restored.close()
with tempfile.TemporaryDirectory(prefix='native-replay-') as temporary:
    trajectory = Path(temporary) / 'history.jsonl'
    session = ChemWorldRuntime(composition=surface_world_spec(), observation_noise_mode='keyed')
    session.reset(seed=12)
    actions = [
        {'operation':'add_component', 'component':'CO-feed', 'amount_mol':2e-5},
        {'operation':'add_component', 'component':'O2-feed', 'amount_mol':1e-5},
        {'operation':'heat', 'target_temperature_K':380, 'duration_s':2,
         'stirring_speed_rpm':600},
        {'operation':'measure', 'instrument':'gas_pressure'},
        {'operation':'terminate'},
    ]
    with TrajectoryLogger(trajectory) as logger:
        for step, action in enumerate(actions, 1):
            result = session.execute(action)
            logger.log(task_info={**session.task_info(), **session.evaluator_provenance()},
                       step=step, action=action,
                       observation=to_observation(result.observation.values),
                       reward=result.reward, terminated=result.terminated,
                       truncated=result.truncated, info=result.info,
                       agent_metadata={'agent_id':'installed-native-probe'})
    session.close()
    replay = verify_records(load_jsonl(trajectory), tolerance=1e-8)
    assert replay.verified and replay.checked_steps == 5
print(json.dumps({'package':chemworld.__file__, 'cantera_installed':False,
                  'platform':{'system':platform.system(), 'python':platform.python_version()},
                  'surface_cases':cases, 'reuse_configurations':4,
                  'task_binding':True, 'serialized_continuation':True,
                  'trajectory_replay_steps':replay.checked_steps}))
"""


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    for key in tuple(env):
        if key.upper().endswith("_API_KEY") or key.upper() in {
            "OPENAI_KEY", "ANTHROPIC_KEY", "DEEPSEEK_KEY",
        }:
            env.pop(key)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def captured_run(command, **kwargs):
    started = time.monotonic()
    with subprocess.Popen(command, text=True, encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kwargs) as process:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"Install check alive: pid={process.pid}, "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
        if process.returncode:
            raise RuntimeError(
                f"Installed command failed ({process.returncode}): {stdout}\n{stderr}"
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def check_install(archive: Path, *, python: str, root: Path, native_only: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix="chemworld-installed-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        child_env = clean_environment()
        print(f"Installing {archive.name} on {python}", flush=True)
        captured_run(["uv", "venv", "--python", python, str(environment)])
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = captured_run(
            ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes"],
            cwd=root,
        ).stdout
        lock = work / "requirements.txt"
        lock.write_text(requirements, encoding="utf-8")
        captured_run(
            ["uv", "pip", "sync", "--python", str(interpreter), str(lock)],
            cwd=work,
            env=child_env,
        )
        captured_run(
            ["uv", "pip", "install", "--python", str(interpreter), "--no-deps", str(archive)],
            cwd=work,
            env=child_env,
        )
        if native_only:
            print("Checking native examples, retained equipment and serialized continuation",
                  flush=True)
            probe = captured_run(
                [str(interpreter), "-I", "-c", NATIVE_PROBE], cwd=work, env=child_env,
            )
            return {"archive": archive.name, "requested_python": python,
                    "installed_check": "passed", **json.loads(probe.stdout)}
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
    parser.add_argument("--native-foundation-only", action="store_true",
                        help="Check native examples and continuation without optional libraries.")
    parser.add_argument("--output", type=Path, help="Optional JSON result, outside the repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="chemworld-build-") as temporary:
        if args.archive:
            archives = [args.archive.resolve()]
        else:
            print("Building wheel and sdist", flush=True)
            captured_run(["uv", "build", "--out-dir", temporary], cwd=root)
            archives = sorted([*Path(temporary).glob("*.whl"), *Path(temporary).glob("*.tar.gz")])
        rows = [check_install(archive, python=args.python, root=root,
                              native_only=args.native_foundation_only) for archive in archives]
    report = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
