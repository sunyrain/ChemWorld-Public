"""Reduce retained 35-close results without running or reselecting a scientific unit."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--canaries", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing summaries")
    block = json.loads(args.block.read_text(encoding="utf-8"))
    summary = {k: v for k, v in block.items() if k != "units"}
    summary["raw_report"] = str(args.block)
    summary["units"] = []
    for unit in block["units"]:
        entry = {k: v for k, v in unit.items() if k != "artifacts"}
        for check in entry["checks"]:
            if "actual" in check:
                a, b = np.asarray(check["actual"]), np.asarray(check["expected"])
                delta = np.abs(a - b)
                limit = check["atol"] + check["rtol"] * np.abs(b)
                check["maximum_absolute_error"] = float(np.max(delta))
                check["maximum_tolerance_fraction"] = float(
                    np.max(
                        np.divide(
                            delta, limit, out=np.zeros_like(delta, dtype=float), where=limit > 0
                        )
                    )
                )
        artifact = unit["artifacts"]
        native = artifact.get("native")
        if isinstance(native, dict) and "original_rate_evaluations" in native:
            entry["native"] = {
                k: native[k]
                for k in (
                    "scope",
                    "status",
                    "amounts_mol",
                    "temperature_K",
                    "coordinates",
                    "held_vessel_ids",
                    "reachable_dimension",
                    "original_rate_evaluations",
                    "material_residual_mol_s",
                    "temperature_residual_K_s",
                    "energy_residual_W",
                    "energy_invariant_errors_J",
                    "conservation_error",
                    "root_diagnostic",
                    "derivative_diagnostic",
                )
            }
        elif isinstance(native, dict):
            entry["work"] = native["work"]
            entry["jacobian_rhs_evaluations"] = sum(
                a["solver"].get("feedback_jacobian_rhs_evaluations", 0)
                for c in native["trajectories"]
                for a in c["advances"]
                if a["solver"]
            )
        elif isinstance(native, list):
            entry["native"] = native
            entry["mechanism_species"] = artifact["mechanism_species"]
            entry["mechanism_reactions"] = artifact["mechanism_reactions"]
        summary["units"].append(entry)
    summary["canaries"] = []
    for path in args.canaries:
        attempt = json.loads(path.read_text(encoding="utf-8"))
        summary["canaries"].append(
            {
                "path": str(path),
                **{k: v for k, v in attempt.items() if k != "units"},
                "units": [
                    {k: v for k, v in unit.items() if k != "artifacts"} for unit in attempt["units"]
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "units": summary["passed_units"],
                "comparison_groups": summary["passed_comparison_groups"],
                "canaries": len(summary["canaries"]),
            }
        )
    )


if __name__ == "__main__":
    main()
