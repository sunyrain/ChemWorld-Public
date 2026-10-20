"""Run potential steps and ramps using shared control, wait and measurement.

uv run --no-sync python -m chemworld.examples.native_electrode_research
No model credentials or third-party electrochemical runtime is needed.
"""

import json

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.electrode_world import electrode_world_spec


def run_example():
    session = ChemWorldRuntime(composition=electrode_world_spec())
    session.reset(seed=28)
    steps = []
    try:
        for potential, ramp_s in ((0.1, 0), (-0.1, 5), (0, 5)):
            for action in (
                {
                    "operation": "configure_electrical_control",
                    "potential_V": potential,
                    "ramp_duration_s": ramp_s,
                },
                {"operation": "wait", "duration_s": 5},
                {"operation": "measure", "instrument": "electrical_meter"},
            ):
                result = session.execute(action)
                if result.info["transaction_status"] != "committed":
                    raise RuntimeError(
                        f"electrode operation failed: {result.info['rollback_reason']}"
                    )
                if action["operation"] == "measure":
                    steps.append(
                        {
                            "time_s": session._state.ledger.time_s,
                            "readings": {
                                c.channel_id: {"value": c.value, "unit": c.unit}
                                for c in result.observation.channels
                            },
                        }
                    )
        result = session.execute({"operation": "terminate"})
        if not result.terminated:
            raise RuntimeError("electrode study did not terminate")
        return {
            "scope": "isothermal supported-electrolyte working electrode",
            "measurements": steps,
            "operations": 10,
            "terminated": True,
        }
    finally:
        session.close()


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, allow_nan=False))
