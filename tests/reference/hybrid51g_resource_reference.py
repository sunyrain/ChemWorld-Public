"""Independent analytic paths for the fixed resource seeds; no native imports."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
AVOGADRO = 6.02214076e23


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    if unit["kind"] == "feedback":
        from stochastic_hybrid51_reference import simulate
    else:
        from cross_partition51_reference import simulate
    assert not any(k == "chemworld" or k.startswith("chemworld.") for k in sys.modules)
    trials = (
        [(997, "warmup")]
        + [(s, "timed") for s in (1, 7, 17, 37, 73, 101, 251, 509)]
        + [(17, "allocation")]
    )
    with (args.output / "trajectories.jsonl").open("x", encoding="utf-8") as stream:
        for index, (seed, purpose) in enumerate(trials):
            row = {
                "id": unit["id"] + "-" + str(index),
                "seed": seed,
                "purpose": purpose,
                "status": "error",
            }
            try:
                result = simulate(unit["parameters"], seed, unit["duration_s"])
                if unit["kind"] == "feedback":
                    total = unit["parameters"]["A0_mol"] * AVOGADRO
                    values = {
                        "A": result["fraction_A"] * total,
                        "B": (1 - result["fraction_A"]) * total,
                        "E0": 1 - result["E1"],
                        "E1": result["E1"],
                    }
                else:
                    values = {k: result[k] for k in ("D", "A", "B")}
                row.update(
                    status="ok",
                    populations=values,
                    events=len(result["events"]),
                    event_path=result["events"],
                    time_s=unit["duration_s"],
                    time_basis="declared_analytic_horizon",
                )
            except Exception as exc:
                row.update(error=repr(exc), traceback=traceback.format_exc())
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    print(
        json.dumps({"unit": unit["id"], "analytic_evaluations_completed": len(trials)}), flush=True
    )


if __name__ == "__main__":
    main()
