"""Run finite dilute RO without an external flowsheet dependency."""

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.membrane_world import membrane_world_spec


def main() -> None:
    runtime = ChemWorldRuntime(composition=membrane_world_spec())
    runtime.reset(seed=34)
    for action in (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
        {"operation": "run_flow", "duration_s": 1800},
        {"operation": "measure", "instrument": "membrane_meter"},
        {"operation": "wait", "duration_s": 1800},
        {"operation": "measure", "instrument": "membrane_meter"},
    ):
        result = runtime.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        print(
            {
                "action": action["operation"],
                "outcome": result.info["operation_outcome"],
                "time_s": runtime._state.ledger.time_s,
                "readings": {
                    c.channel_id: {"value": c.value, "unit": c.unit}
                    for c in result.observation.channels
                    if c.quality == "valid"
                },
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
