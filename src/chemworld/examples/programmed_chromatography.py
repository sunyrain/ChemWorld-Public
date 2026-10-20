"""Offline instrument-history demonstration; no model provider or API key.

Run from an installed package with ``python -m chemworld.examples.programmed_chromatography
--output /path/to/new-directory``. Each of six finite workflows has its own full
JSONL history, exact replay check and machine-readable summary, including failures.
"""

import argparse
import json
from math import exp
from pathlib import Path

from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.envs.spaces import to_observation
from chemworld.eval.verify import verify_records
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.chromatography import SAMPLE_L, memory, settings
from chemworld.world.chromatography_world import chromatography_world_spec


def sequence(instrument, program):
    name = "programmed_" + instrument
    actions = [
        {"operation": "configure_chromatography", "instrument": name, "program_index": program}
    ]
    for bottle in ("standard", "sample", "blank") * 2:
        actions.extend(
            (
                {"operation": "select_vessel", "vessel": "reactor"},
                {
                    "operation": "route_material",
                    "connection": "reactor-to-waste",
                    "transfer_fraction": 1,
                    "mixing": "allow",
                },
                {"operation": "select_vessel", "vessel": bottle},
                {
                    "operation": "route_material",
                    "connection": bottle + "-to-reactor",
                    "transfer_fraction": 0.1,
                    "mixing": "allow",
                },
                {"operation": "select_vessel", "vessel": "reactor"},
                {"operation": "measure", "instrument": name},
            )
        )
    return [*actions, {"operation": "terminate"}]


def run_case(instrument, program, path):
    name = "programmed_" + instrument
    session = ChemWorldRuntime(composition=chromatography_world_spec(instrument))
    session.reset(seed=12)
    measurements = []
    with TrajectoryLogger(path) as logger:
        for step, action in enumerate(sequence(instrument, program), 1):
            before = session._state
            result = session.execute(action)
            logger.log(
                task_info={**session.task_info(), **session.evaluator_provenance()},
                step=step,
                action=action,
                observation=to_observation(result.observation.values),
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=result.info,
                agent_metadata={"agent_id": "fixed-development-sequence"},
            )
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(f"step {step}: {result.info['transaction_status']}")
            if action["operation"] == "measure":
                config = settings(session._state, name)
                old = memory(before, name).values
                new = memory(session._state, name).values
                injected = {
                    k: v * SAMPLE_L / before.volume_L for k, v in before.species_amounts.items()
                }
                waste = config["chromatography_acquisition"]["effluent_mol"]
                residual = max(abs(new[k] + waste[k] - old[k] - injected[k]) for k in new)
                if residual > 1e-14:
                    raise RuntimeError("independent residue/waste balance failed")
                if len(measurements) % 3 == 2 and any(
                    abs(new[k] - old[k] * exp(-6)) > 1e-14 for k in new
                ):
                    raise RuntimeError("blank did not wash the actual retained inventory")
                measurements.append(
                    {
                        "bottle": ("standard", "sample", "blank")[len(measurements) % 3],
                        "use_count": config["use_count"],
                        "residual_mol": residual,
                        "injected_mol": injected,
                        "retained_mol": new,
                        "effluent_mol": waste,
                        "retention_s": config["chromatography_acquisition"]["retention_s"],
                        "areas_AU_s": [
                            result.observation.channel(f"peak_{i}_area").value for i in range(1, 4)
                        ],
                    }
                )
    replay = verify_records(load_jsonl(path), tolerance=1e-8)
    return {
        "instrument": instrument,
        "program": program,
        "steps": step,
        "measurements": measurements,
        "replay_verified": replay.verified,
        "passed": replay.verified,
        "sample_consumed_L": session._state.ledger.sample_consumed_L,
        "analysis_time_s": session._state.ledger.analysis_time_s,
        "maximum_balance_residual_mol": max(m["residual_mol"] for m in measurements),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for instrument in ("hplc", "gc"):
        for program in range(3):
            try:
                row = run_case(instrument, program, args.output / f"{instrument}-{program}.jsonl")
            except Exception as error:
                row = {
                    "instrument": instrument,
                    "program": program,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            rows.append(row)
            print(f"{len(rows)}/6 {instrument} program {program}: {row['passed']}", flush=True)
    summary = {
        "status": "development",
        "planned_workflows": 6,
        "passed": sum(r["passed"] for r in rows),
        "failed": sum(not r["passed"] for r in rows),
        "planned_acquisitions": 36,
        "workflows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
