"""Run a finite pump/pipe/exchanger experiment using native actions."""

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.transport_world import transport_world_spec


def main():
    runtime = ChemWorldRuntime(composition=transport_world_spec())
    runtime.reset(seed=24)
    for action in (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
        {"operation": "run_flow", "duration_s": 60},
        {"operation": "measure", "instrument": "flow_meter"},
        {"operation": "wait", "duration_s": 120},
        {"operation": "measure", "instrument": "flow_meter"},
    ):
        result = runtime.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        print(
            {
                "action": action["operation"],
                "outcome": result.info["operation_outcome"],
                "time_s": runtime._state.ledger.time_s,
                "source_L": runtime._state.inactive_vessels["feed"].volume_L,
                "receiver_K": runtime._state.temperature_K,
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
