"""Run three native RO/finite-rinse cycles without resetting membrane history."""

from chemworld.physchem.membranes import MembraneFoulingSpec, MembraneSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.membrane_world import membrane_world_spec


def main() -> None:
    runtime = ChemWorldRuntime(
        composition=membrane_world_spec(MembraneSpec(fouling=MembraneFoulingSpec()))
    )
    runtime.reset(seed=35)
    for cycle in range(1, 4):
        for path, duration in (("process", 900), ("rinse", 60)):
            for action in (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20, "flow_path": path},
                {"operation": "run_flow", "duration_s": duration},
                {"operation": "measure", "instrument": "membrane_meter"},
            ):
                result = runtime.execute(action)
                if result.info["transaction_status"] != "committed":
                    raise RuntimeError(result.info)
            print(
                {
                    "cycle": cycle,
                    "path": path,
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
