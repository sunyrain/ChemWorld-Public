"""Run a native surface research sequence with shared equipment and no model API.

python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
Use --no-surface, optionally with --bulk, to run the same apparatus and actions
as an inert thermal body or a bulk-only reactor.
"""

import argparse
import json

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def run_example(*, include_surface=True, include_bulk=False, mobile=False, terminal_assay=False):
    env = ChemWorldRuntime(
        composition=surface_world_spec(
            include_surface=include_surface,
            include_bulk=include_bulk,
            mobile=mobile,
            terminal_assay=terminal_assay,
        )
    )
    env.reset(seed=12)
    actions = [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5},
        {
            "operation": "heat",
            "target_temperature_K": 380,
            "duration_s": 10,
            "stirring_speed_rpm": 600,
        },
        {"operation": "measure", "instrument": "gas_pressure"},
        {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "gas_composition"},
        {
            "operation": "route_material",
            "connection": "reactor-to-receiver",
            "transfer_fraction": 0.5,
            "mixing": "allow",
        },
        {"operation": "select_vessel", "vessel": "receiver"},
        {"operation": "wait", "duration_s": 5, "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "gas_composition"},
        {"operation": "terminate"},
    ]
    if terminal_assay:
        actions.append({"operation": "measure", "instrument": "gas_final_assay"})
    history = []
    try:
        for action in actions:
            result = env.execute(action)
            info = result.info
            history.append(
                {
                    "action": action,
                    "transaction": info["transaction_status"],
                    "outcome": info["operation_outcome"],
                    "physical_duration_s": info["actual_duration_s"],
                    "measurement_status": info["measurement_status"],
                    "observation": info["native_observation"]
                    if action["operation"] == "measure"
                    else None,
                    "ended": result.terminated,
                    "truncated": result.truncated,
                }
            )
            if info["transaction_status"] != "committed":
                raise RuntimeError(
                    f"Example action failed: {action}: {info.get('rollback_reason')}"
                )
        return {
            "configuration": {"surface": include_surface, "bulk": include_bulk, "mobile": mobile},
            "steps": history,
            "ended": history[-1]["ended"],
            "physical_time_s": env._state.ledger.time_s,
            "analysis_time_s": env._state.ledger.analysis_time_s,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-surface", action="store_true")
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--terminal-assay", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_example(
                include_surface=not args.no_surface,
                include_bulk=args.bulk,
                mobile=args.mobile,
                terminal_assay=args.terminal_assay,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
