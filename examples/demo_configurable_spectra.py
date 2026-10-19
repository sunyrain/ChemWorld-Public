"""Acquire a configurable synthetic NMR signal without a model provider."""

from __future__ import annotations

import json

from chemworld.envs.chemworld_env import ChemWorldEnv


def main() -> None:
    composition = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "public-configurable-spectrum",
        "components": [
            {"kind": "reaction"},
            {"kind": "thermal"},
            {"kind": "observation", "parameters": {"instruments": ["nmr", "final_assay"]}},
        ],
        "task": {"budget": 20, "instruments": ["nmr", "final_assay"]},
    }
    env = ChemWorldEnv(composition=composition, seed=0)
    try:
        env.reset(seed=0)
        actions = [
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.015},
            {
                "operation": "heat",
                "duration_s": 600,
                "target_temperature_K": 365,
                "stirring_speed_rpm": 600,
            },
            {
                "operation": "configure_instrument",
                "instrument": "nmr",
                "scan_count": 16,
                "resolution_factor": 1.0,
                "dilution_factor": 2.0,
            },
            {"operation": "measure", "instrument": "nmr"},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        for action in actions:
            *_, info = env.step(action)
            if info["transaction_status"] != "committed":
                raise RuntimeError(f"Operation failed: {action['operation']}")
            if action.get("instrument") == "nmr" and action["operation"] == "measure":
                packet = info["raw_signal"]
                print(
                    json.dumps(
                        {
                            "instrument": "nmr",
                            "settings": packet["settings"],
                            "points": len(packet["axis"]["values"]),
                            "scope": "synthetic assay reporters",
                        }
                    )
                )
        print(json.dumps({"committed_operations": len(actions), "final_assay_completed": True}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
