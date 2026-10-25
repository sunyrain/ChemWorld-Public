"""Fresh-process continuation of complete original three-tank RO snapshots."""

from __future__ import annotations

import json
import sys

from chemworld.runtime import ChemWorldRuntime


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def continue_world(runtime, short_wait_s):
    suffix = (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 7.5},
        {"operation": "wait", "duration_s": short_wait_s},
        {"operation": "wait", "duration_s": 1.25},
    )
    rows = []
    for action in suffix:
        result = runtime.execute(action)
        rows.append({"action": action, "info": result.info})
        if result.info["transaction_status"] != "committed":
            break
    return normalized({"actions": rows, "snapshot": runtime.snapshot().to_dict()})


if __name__ == "__main__":
    request = json.load(sys.stdin)
    runtime = ChemWorldRuntime.from_snapshot(request["snapshot"])
    try:
        json.dump(continue_world(runtime, request["short_wait_s"]), sys.stdout, allow_nan=False)
    finally:
        runtime.close()
