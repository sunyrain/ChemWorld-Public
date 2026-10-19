"""A fixed, offline research loop; see docs/offline_research.md before execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.agents.experiment_documents import ExperimentDocumentWorkspace
from chemworld.data.datasets import dataset_card, export_dataset
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID

DESIGN: dict[str, Any] = {
    "formal_result": False,
    "question": "Does local linear yield prediction transfer across jacket setpoints?",
    "task": "reaction-to-assay",
    "seed": 0,
    "train_K": [330.0, 350.0, 370.0],
    "test_K": [360.0, 400.0],
    "expected_episodes": 7,
    "expected_operations": 45,
    "expected_final_assays": 5,
    "expected_rejections": 1,
    "rule": "Piecewise linear; nearest segment extrapolation; clip to [0,1].",
    "baseline": "Arithmetic mean of the three observed training yields.",
    "pass_rule": "Integrity and exact replay, not prediction accuracy or score.",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def actions_at(temperature: float) -> list[dict[str, Any]]:
    return [
        {"operation": "add_solvent", "volume_L": 0.030, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
        {
            "operation": "heat",
            "target_temperature_K": temperature,
            "duration_s": 1500.0,
            "stirring_speed_rpm": 720.0,
        },
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "wait", "duration_s": 600.0, "stirring_speed_rpm": 720.0},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def run_unit(
    output: Path,
    documents: ExperimentDocumentWorkspace,
    name: str,
    temperature: float,
    *,
    ending: str = "assay",
    invalid_probe: bool = False,
) -> dict:
    path = output / "trajectories" / f"{name}.jsonl"
    sequence = actions_at(temperature)
    if ending != "assay":
        sequence = sequence[:2]
    if invalid_probe:
        sequence.insert(0, {"operation": "measure", "instrument": "hplc"})
    env = gym.make(
        "ChemWorld",
        task_id=DESIGN["task"],
        seed=0,
        budget_override=2 if ending == "truncated" else 18,
        episode_mode_override="single_experiment",
    )
    failures = []
    try:
        env.reset(seed=0)
        core = cast(ChemWorldEnv, env.unwrapped)
        # Host-only provenance is required by native replay; never give it to an agent.
        provenance = {**core.task_info(), **core.evaluator_provenance()}
        with TrajectoryLogger(path) as logger:
            for step, action in enumerate(sequence, 1):
                obs, reward, terminated, truncated, info = env.step(action)
                logger.log(
                    task_info=provenance,
                    step=step,
                    action=action,
                    observation=obs,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={"agent_name": "offline-fixed-design"},
                )
                documents.append_operation(
                    {
                        "event_id": f"{name}-{step:03d}",
                        "action": action,
                        "transaction_status": info["transaction_status"],
                        "observation": observation_to_json(obs),
                        "observed_mask": info.get("observed_mask", {}),
                        "trajectory_reference": {
                            "path": f"trajectories/{name}.jsonl",
                            "step": step,
                        },
                    }
                )
                expected_rejection = invalid_probe and step == 1
                committed = info["transaction_status"] == "committed"
                if committed == expected_rejection:
                    failures.append({"step": step, "reason": "unexpected transaction status"})
                if terminated or truncated:
                    break
    except Exception as exc:
        failures.append({"reason": f"{type(exc).__name__}: {exc}"})
    finally:
        env.close()
    records = load_jsonl(path) if path.exists() else []
    assays = [
        r
        for r in records
        if r["instrument"] == "final_assay" and r["transaction_status"] == "committed"
    ]
    if len(assays) != int(ending == "assay"):
        failures.append({"reason": "unexpected final assay count"})
    return {
        "name": name,
        "setpoint_K": temperature,
        "planned_ending": ending,
        "operation_count": len(records),
        "final_assay_count": len(assays),
        "yield": assays[-1]["observation"].get("yield") if assays else None,
        "final_score": assays[-1].get("leaderboard_score") if assays else None,
        "failures": failures,
    }


def forecast(temperature: float, training: list[dict]) -> float:
    values = [(row["setpoint_K"], row["yield"]) for row in training]
    if any(y is None for _, y in values):
        raise ValueError("Training assay missing; cannot fabricate a prediction")
    left, right = (values[0], values[1]) if temperature <= values[1][0] else values[1:]
    value = left[1] + (temperature - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
    return float(max(0.0, min(1.0, value)))


def replay_in_new_process(path: Path) -> dict:
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from chemworld.data.logging import load_jsonl; "
            "from chemworld.eval.verify import verify_records; "
            "print(json.dumps(verify_records(load_jsonl(sys.argv[1]), tolerance=0.0).to_dict()))",
            str(path),
        ],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if child.returncode:
        return {"verified": False, "error": child.stderr[-2000:]}
    return json.loads(child.stdout)


def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "design.json", {**DESIGN, "runtime_semantics_id": RUNTIME_SEMANTICS_ID})
    documents = ExperimentDocumentWorkspace(output)
    documents.initialize()
    documents.write_notebook(
        "# Offline research notebook\n\nQuestion: how far does local interpolation transfer?\n"
        "Hypothesis: a local linear rule may interpolate; extrapolation is uncertain.\n"
        "Fixed design.json: same world seed, independent vessels, no provider.\n"
        "First HPLC in train-01 is deliberately attempted on an empty vessel.\n"
    )
    training = []
    for index, temperature in enumerate(DESIGN["train_K"], 1):
        training.append(
            run_unit(output, documents, f"train-{index:02d}", temperature, invalid_probe=index == 1)
        )
        print(f"Research episodes {index}/7 recorded", flush=True)
    predictions = []
    if all(row["yield"] is not None for row in training):
        baseline = sum(row["yield"] for row in training) / len(training)
        predictions = [
            {
                "setpoint_K": t,
                "yield_prediction": forecast(t, training),
                "observed_mean_prediction": baseline,
            }
            for t in DESIGN["test_K"]
        ]
    # Seal predictions before either held-out run; never rewrite this file later.
    write_json(output / "predictions.json", predictions)
    documents.write_notebook(
        documents.read_notebook()
        + "\n## Evidence and sealed forecasts\n"
        + json.dumps(training, indent=2)
        + "\n\n"
        + json.dumps(predictions, indent=2)
        + "\n"
    )
    rows = list(training)
    for index, temperature in enumerate(DESIGN["test_K"], 1):
        rows.append(run_unit(output, documents, f"test-{index:02d}", temperature))
        print(f"Research episodes {len(rows)}/7 recorded", flush=True)
    for ending in ("open", "truncated"):
        rows.append(run_unit(output, documents, ending, 350.0, ending=ending))
        print(f"Research episodes {len(rows)}/7 recorded", flush=True)
    errors = [
        {
            **p,
            "observed_yield": r["yield"],
            "absolute_error": abs(p["yield_prediction"] - r["yield"]),
            "mean_baseline_absolute_error": abs(p["observed_mean_prediction"] - r["yield"]),
        }
        for p, r in zip(predictions, rows[3:5], strict=False)
        if r["yield"] is not None
    ]
    export = export_dataset(
        output / "trajectories", output=output / "dataset.jsonl", format="jsonl"
    )
    card = dataset_card(output / "trajectories")
    write_json(output / "dataset_card.json", card)
    replay = {}
    for path in sorted((output / "trajectories").glob("*.jsonl")):
        replay[path.name] = replay_in_new_process(path)
        print(f"New-process replay {len(replay)}/7: {path.name}", flush=True)
    write_json(output / "replay.json", replay)
    ledger_before = documents.manifest()["authoritative_ledger"]
    documents.write_notebook(
        documents.read_notebook()
        + "\n## Held-out observations and errors\n"
        + json.dumps(errors, indent=2)
        + "\n\n"
        "Evidence: test-01/test-02 final records; open/truncated have no assay.\n"
        "The linear rule is descriptive, not an identified kinetic mechanism.\n"
        "Thermal lag, reaction networks and measurement depletion are competing explanations.\n"
        "Two forecast errors do not calibrate an interval or establish generalization.\n"
        "Next experiment: independently vary duration at fixed setpoint; not run here.\n"
    )
    assert documents.manifest()["authoritative_ledger"] == ledger_before
    counts = card["outcome_counts"]
    accepted = (
        len(rows) == 7
        and len(errors) == 2
        and not any(r["failures"] for r in rows)
        and counts["operation_count"] == 45
        and counts["final_assay_experiment_count"] == 5
        and counts["rejected_action_count"] == 1
        and counts["open_experiment_count"] == 1
        and counts["truncated_open_experiment_count"] == 1
        and len(replay) == 7
        and all(r["verified"] for r in replay.values())
    )
    summary = {
        "formal_result": False,
        "accepted": accepted,
        "provider_calls": 0,
        "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
        "episodes": rows,
        "prediction_errors": errors,
        "outcome_counts": counts,
        "export": export.to_dict(),
        "new_process_replay": replay,
        "documents": documents.manifest(),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
