"""Install wheel or sdist in a fresh, locked environment outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

PROBE = """
import json, sys, platform
from pathlib import Path
from importlib.metadata import version, distribution
from importlib.util import find_spec
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
assert find_spec('chemworld.lab') is None
assert find_spec('chemworld.workbench') is None
assert {e.name for e in distribution('chemworld-bench').entry_points
        if e.group == 'console_scripts'} == {'chemworld'}
from chemworld.examples.user_reward import run_example
native = run_example(reward_fn=None)
assert len(native['steps']) == 3
assert native['steps'][-1]['observation']['native']['channels'][0]['unit'] == 'Pa'
from chemworld.eval.runner import run_agent
import chemworld.agents.interactive_codex_experiment
print(json.dumps({'package': chemworld.__file__, 'version': chemworld.__version__,
                  'tasks': tasks, 'contracts': readiness['contract_ready_count'],
                  'ui_packages_absent': True, 'console_commands': ['chemworld'],
                  'native_user_actions': len(native['steps']), 'agent_imports': True,
                  'platform': {'system': platform.system(), 'machine': platform.machine(),
                               'python_version': platform.python_version()},
                  'runtime_semantics_id': RUNTIME_SEMANTICS_ID}))
"""

NATIVE_PROBE = """
import importlib.util, json, os, platform, subprocess, sys, time
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
upstreams = ('cantera', 'CoolProp', 'pybamm', 'watertap', 'idaes', 'phasepy',
             'thermo', 'copasi', 'COPASI', 'catmap', 'pycalphad', 'phreeqpython',
             'chemicals', 'fluids', 'periodictable', 'teqp', 'thermopack', 'rmgpy', 'libsbml')
assert all(importlib.util.find_spec(name) is None for name in upstreams)
assert not any(key.upper().endswith('_API_KEY') or key.upper() in
               {'OPENAI_KEY', 'ANTHROPIC_KEY', 'DEEPSEEK_KEY'} for key in os.environ)
evidence = Path(os.environ['CHEMWORLD_INSTALL_EVIDENCE'])
phase_started = {}
def progress(stage, completed, total, **fields):
    now = time.monotonic()
    elapsed = now - phase_started.setdefault(stage, now)
    rate = completed / elapsed if elapsed else 0
    print('CHEMWORLD_PROGRESS ' + json.dumps({'stage':stage, 'completed':completed,
          'total':total, 'units_per_s':rate, 'eta_s':(total-completed)/rate if rate else None,
          **fields}), file=sys.stderr, flush=True)
cases = []
progress('surface', 0, 8)
for surface, bulk in ((False,False), (False,True), (True,False), (True,True)):
    for assay in (False, True):
        result = run_example(include_surface=surface, include_bulk=bulk,
                             mobile=surface and assay, terminal_assay=assay)
        assert result['ended']
        assert all(row['transaction'] == 'committed' for row in result['steps'])
        cases.append({'surface':surface, 'bulk':bulk, 'assay':assay,
                      'physical_time_s':result['physical_time_s']})
        progress('surface', len(cases), 8)
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
trajectory = evidence / 'surface-history.jsonl'
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
from chemworld.examples.native_domains import WORKFLOWS, run_workflow
from chemworld.examples.native_capabilities import capability_rows
assert any(row['module'] == 'lab_schedule' for row in capability_rows())
domain_cases = []
domain_output = evidence / 'domains'
progress('domains', 0, len(WORKFLOWS))
for index, name in enumerate(WORKFLOWS, 1):
    progress('domains', index - 1, len(WORKFLOWS), current=name)
    row = run_workflow(name, domain_output)
    completed = subprocess.run(
        [sys.executable, '-I', '-c', 'from chemworld.cli import main; main()',
         'verify', '--submission', row['trajectory']],
        text=True, capture_output=True, check=True)
    replay_result = json.loads(completed.stdout)
    (domain_output / (name + '-fresh-replay.json')).write_text(
        json.dumps(replay_result, indent=2), encoding='utf-8')
    assert replay_result['verified'] and replay_result['checked_steps'] == row['actions']
    row['trajectory_file'] = Path(row.pop('trajectory')).name
    row['new_process_replay'] = True
    domain_cases.append(row)
    progress('domains', index, len(WORKFLOWS), current=name)
documentation_commands = [
    ['-m', 'chemworld.examples.native_domains', '--workflow', 'heater',
     '--noise-mode', 'keyed', '--output', str(evidence / 'documented-keyed')],
    ['-m', 'chemworld.examples.native_capabilities', '--format', 'json'],
    ['-m', 'chemworld.examples.native_capabilities', '--composition',
     str(domain_output / 'heater-recipe.json')],
    ['-c', 'from chemworld.cli import main; main()', 'verify', '--submission',
     str(evidence / 'documented-keyed' / 'heater.jsonl')],
    ['-m', 'chemworld.examples.native_surface_research', '--bulk', '--mobile', '--terminal-assay'],
    ['-m', 'chemworld.examples.reuse_native_equipment'],
]
for index, command in enumerate(documentation_commands, 1):
    progress('documentation', index - 1, len(documentation_commands))
    completed = subprocess.run([sys.executable, '-I', *command],
                               text=True, capture_output=True)
    (evidence / ('documentation-' + str(index) + '.stdout')).write_text(
        completed.stdout, encoding='utf-8')
    (evidence / ('documentation-' + str(index) + '.stderr')).write_text(
        completed.stderr, encoding='utf-8')
    completed.check_returncode()
    if index == 2:
        assert json.loads(completed.stdout) == json.loads(json.dumps(capability_rows()))
    if index == 4:
        assert json.loads(completed.stdout)['verified']
    progress('documentation', index, len(documentation_commands))
assert all(name not in sys.modules for name in upstreams)
print(json.dumps({'package':chemworld.__file__, 'cantera_installed':False,
                  'platform':{'system':platform.system(), 'python':platform.python_version()},
                  'surface_cases':cases, 'reuse_configurations':4,
                  'task_binding':True, 'serialized_continuation':True,
                  'trajectory_replay_steps':replay.checked_steps,
                  'upstream_libraries_absent':list(upstreams),
                  'documentation_commands_passed':len(documentation_commands),
                  'domain_workflows':domain_cases}))
"""


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    for key in tuple(env):
        if key.upper().endswith("_API_KEY") or key.upper() in {
            "OPENAI_KEY",
            "ANTHROPIC_KEY",
            "DEEPSEEK_KEY",
        }:
            env.pop(key)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def captured_run(command, *, log_prefix=None, poll_interval_s=30, **kwargs):
    started = time.monotonic()
    shown = 0

    def show_progress(stderr):
        nonlocal shown
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        lines = [
            line for line in (stderr or "").splitlines() if line.startswith("CHEMWORLD_PROGRESS ")
        ]
        for line in lines[shown:]:
            print(line, flush=True)
        shown = len(lines)

    with ExitStack() as stack:
        if log_prefix is None:
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="chemworld-command-"),
            )
            log_prefix = Path(temporary) / "command"
        stdout_path, stderr_path = (
            log_prefix.with_suffix(".stdout.log"),
            log_prefix.with_suffix(".stderr.log"),
        )
        if stdout_path.exists() or stderr_path.exists():
            raise ValueError("Command evidence already exists; choose a new attempt")
        stdout_file = stack.enter_context(stdout_path.open("w", encoding="utf-8"))
        stderr_file = stack.enter_context(stderr_path.open("w", encoding="utf-8"))
        # Windows communicate() does not return partial pipe output on timeout.
        # Durable files expose flushed progress while preserving both streams.
        with subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, **kwargs) as process:
            while True:
                try:
                    process.wait(timeout=poll_interval_s)
                    break
                except subprocess.TimeoutExpired:
                    show_progress(stderr_path.read_text(encoding="utf-8", errors="replace"))
                    print(
                        f"Install check alive: pid={process.pid}, "
                        f"elapsed={time.monotonic() - started:.0f}s",
                        flush=True,
                    )
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        show_progress(stderr)
        if process.returncode:
            raise RuntimeError(
                f"Installed command failed ({process.returncode}): {stdout}\n{stderr}"
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def check_install(
    archive: Path, *, python: str, root: Path, evidence: Path, native_only: bool = False
) -> dict:
    evidence.mkdir(exist_ok=False)
    command_index = 0

    def run(command, **kwargs):
        nonlocal command_index
        command_index += 1
        return captured_run(command, log_prefix=evidence / f"command-{command_index:02}", **kwargs)

    with tempfile.TemporaryDirectory(prefix="chemworld-installed-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        child_env = clean_environment()
        child_env["CHEMWORLD_INSTALL_EVIDENCE"] = str(evidence.resolve())
        print(f"Installing {archive.name} on {python}", flush=True)
        run(["uv", "venv", "--python", python, str(environment)])
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = run(
            ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes"],
            cwd=root,
        ).stdout
        lock = work / "requirements.txt"
        lock.write_text(requirements, encoding="utf-8")
        run(
            ["uv", "pip", "sync", "--python", str(interpreter), str(lock)],
            cwd=work,
            env=child_env,
        )
        run(
            ["uv", "pip", "install", "--python", str(interpreter), "--no-deps", str(archive)],
            cwd=work,
            env=child_env,
        )
        if native_only:
            print(
                "Checking native examples, retained equipment and serialized continuation",
                flush=True,
            )
            probe = run(
                [str(interpreter), "-I", "-c", NATIVE_PROBE],
                cwd=work,
                env=child_env,
            )
            return {
                "archive": archive.name,
                "requested_python": python,
                "installed_check": "passed",
                **json.loads(probe.stdout),
            }
        probe = run(
            [str(interpreter), "-I", "-c", PROBE],
            cwd=work,
            env=child_env,
        )
        payload = json.loads(probe.stdout)
        example = root / "examples/demo_manual_event_sequence.py"
        completed = run(
            [str(interpreter), "-I", str(example)],
            cwd=work,
            env=child_env,
        )
        assert json.loads(completed.stdout.splitlines()[-1])["final_assay_completed"]
        run(
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
        research = evidence / "offline-research"
        run(
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
        trajectory = evidence / "legacy-trajectory.jsonl"
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
            [
                "evaluate",
                "--submission",
                str(trajectory),
                "--output",
                str(evidence / "legacy-result.json"),
            ],
        ):
            result = run(
                [
                    str(interpreter),
                    "-I",
                    "-c",
                    "from chemworld.legacy.cli import main; main()",
                    *args,
                ],
                cwd=work,
                env=child_env,
            )
            if args[0] == "verify":
                assert json.loads(result.stdout)["verified"] is True
        native = run([str(interpreter), "-I", "-c", NATIVE_PROBE], cwd=work, env=child_env)
        return {
            "archive": archive.name,
            "python": python,
            "installed_check": "passed",
            "offline_research": {
                "accepted": research_summary["accepted"],
                "new_process_replays": len(research_summary["new_process_replay"]),
            },
            **payload,
            "native": json.loads(native.stdout),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        help="Existing wheel or sdist (repeatable); otherwise build both",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--native-foundation-only",
        action="store_true",
        help="Check native examples and continuation without optional libraries.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result, outside the repository")
    parser.add_argument("--evidence", type=Path, help="New external directory for logs and replays")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def write_result(rows):
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    if args.output and args.output.exists():
        raise ValueError("Installation report already exists; choose a new attempt path")
    evidence = args.evidence or (args.output.with_suffix("") if args.output else None)
    if evidence is None:
        evidence = Path(tempfile.mkdtemp(prefix="chemworld-install-evidence-"))
    else:
        if evidence.resolve().is_relative_to(root) or (
            args.output and args.output.resolve().is_relative_to(root)
        ):
            raise ValueError("Installation evidence must be outside the checkout")
        evidence.mkdir(parents=True, exist_ok=False)
    if args.archive:
        archives = [archive.resolve() for archive in args.archive]
    else:
        destination = evidence / "archives"
        destination.mkdir()
        print("Building wheel and sdist", flush=True)
        try:
            captured_run(
                ["uv", "build", "--out-dir", str(destination)],
                cwd=root,
                log_prefix=evidence / "build",
            )
            archives = sorted([*destination.glob("*.whl"), *destination.glob("*.tar.gz")])
            if len(archives) != 2:
                raise RuntimeError("Build must produce one wheel and one sdist")
        except Exception:
            failure = traceback.format_exc()
            rows = [
                {
                    "archive_kind": kind,
                    "requested_python": args.python,
                    "installed_check": "not_executed",
                    "failure": failure,
                    "evidence": str(evidence.resolve()),
                }
                for kind in ("wheel", "sdist")
            ]
            write_result(rows)
            print(json.dumps(rows, indent=2))
            return 1
    revision = dirty = None
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    rows = []
    started = time.monotonic()
    for index, archive in enumerate(archives, 1):
        unit = evidence / f"unit-{index}"
        identity = {
            "archive": archive.name,
            "requested_python": args.python,
            "archive_kind": "wheel" if archive.suffix == ".whl" else "sdist",
            "source_commit": revision,
            "source_worktree_dirty": dirty,
            "evidence": str(unit.resolve()),
        }
        try:
            identity["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            result = check_install(
                archive,
                python=args.python,
                root=root,
                evidence=unit,
                native_only=args.native_foundation_only,
            )
            row = {**result, **identity}
        except Exception:
            row = {**identity, "installed_check": "failed", "failure": traceback.format_exc()}
        rows.append(row)
        write_result(rows)
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed > 0 else None
        print(
            json.dumps(
                {
                    "stage": "install",
                    "completed": index,
                    "total": len(archives),
                    "units_per_s": rate,
                    "eta_s": (len(archives) - index) / rate if rate else None,
                    "status": row["installed_check"],
                }
            ),
            flush=True,
        )
    report = json.dumps(rows, indent=2) + "\n"
    write_result(rows)
    print(report)
    return 0 if all(row["installed_check"] == "passed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
