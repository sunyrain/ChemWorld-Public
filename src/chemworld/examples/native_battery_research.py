"""Run a finite battery discharge/rest/recharge without optional reference libraries."""

import json

from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec


def run_example():
    spec = BatterySPMSpec()
    session = ChemWorldRuntime(composition=battery_world_spec(spec))
    session.reset(seed=7)
    measurements = []
    for rate, duration in ((1, 600), (0, 300), (-0.5, 300)):
        for action in (
            {
                "operation": "configure_electrical_control",
                "current_A": spec.current_at_c_rate(rate),
            },
            {"operation": "wait", "duration_s": duration},
            {"operation": "measure", "instrument": "battery_meter"},
        ):
            result = session.execute(action)
            if result.info.get("transaction_status") != "committed":
                raise RuntimeError(f"battery example action failed: {result.info}")
        measurements.append(
            {
                "time_s": session._state.ledger.time_s,
                "readings": {
                    c.channel_id: {"value": c.value, "unit": c.unit}
                    for c in result.observation.channels
                },
            }
        )
    final = session.execute({"operation": "terminate"})
    return {
        "scope": "synthetic isothermal single-particle battery",
        "operations": 10,
        "terminated": final.terminated,
        "measurements": measurements,
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
