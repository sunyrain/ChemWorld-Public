"""Shared heater/analyzer example; no model key or external chemistry backend."""

import json

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.lab_world import shared_lab_world


def main():
    runtime = ChemWorldRuntime(composition=shared_lab_world())
    runtime.reset(seed=17)
    for action in (
        {"operation": "queue_lab_heat", "duration_s": 30, "target_temperature_K": 350},
        {"operation": "queue_lab_measure"},
        {"operation": "select_vessel", "vessel": "sample"},
        {"operation": "queue_lab_heat", "duration_s": 30, "target_temperature_K": 360},
        {"operation": "queue_lab_measure"},
        *({"operation": "wait", "duration_s": 120} for _ in range(3)),
    ):
        result = runtime.execute(action)
        assert result.info["transaction_status"] == "committed", result.info
        print(
            json.dumps(
                {
                    "action": action,
                    "time_s": runtime._state.ledger.time_s,
                    "completed": [job["sequence"] for job in runtime._state.lab_clock.completed],
                }
            ),
            flush=True,
        )
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for instance in (runtime, restored):
        instance.execute({"operation": "wait", "duration_s": 10})
    assert runtime._state.to_dict() == restored._state.to_dict()
    print("Exact resumed continuation; final ledger:", runtime._state.ledger)


if __name__ == "__main__":
    main()
