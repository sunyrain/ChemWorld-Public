"""Import a finite gas mechanism and execute the common heat/wait/measure path.

No upstream package or model key is required. Initial inventory is an explicit
JSON object in mol; the authored vessel is 0.01 L at 700 K.
"""

import argparse
import json

from chemworld.physchem.mechanism_import import import_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_mechanism_world import gas_mechanism_world


def run_example(path, amounts_mol, *, phase_name=None):
    network = import_mechanism(path, phase_name=phase_name)
    runtime = ChemWorldRuntime(composition=gas_mechanism_world(network, amounts_mol))
    runtime.reset(seed=7)
    records = []
    try:
        for action in (
            {"operation": "heat", "target_temperature_K": 750, "duration_s": 1},
            {"operation": "measure", "instrument": "gas_pressure"},
            {"operation": "wait", "duration_s": 1},
            {"operation": "measure", "instrument": "gas_pressure"},
            {"operation": "terminate"},
        ):
            result = runtime.execute(action)
            info = result.info
            records.append(
                {
                    "action": action,
                    "outcome": info["operation_outcome"],
                    "transaction": info["transaction_status"],
                    "observation": info.get("native_observation")
                    if action["operation"] == "measure"
                    else None,
                    "physical_duration_s": info["actual_duration_s"],
                }
            )
            if info["transaction_status"] != "committed":
                raise RuntimeError(f"operation failed: {action}: {info.get('rollback_reason')}")
        return {
            "source": network.metadata["source"],
            "species": len(network.species),
            "reactions": len(network.reactions),
            "steps": records,
            "physical_time_s": runtime._state.ledger.time_s,
            "analysis_time_s": runtime._state.ledger.analysis_time_s,
            "final_temperature_K": runtime._state.temperature_K,
        }
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism")
    parser.add_argument("--phase")
    parser.add_argument("--amounts", required=True, help="JSON inventory in mol")
    args = parser.parse_args()
    print(
        json.dumps(
            run_example(args.mechanism, json.loads(args.amounts), phase_name=args.phase), indent=2
        )
    )


if __name__ == "__main__":
    main()
