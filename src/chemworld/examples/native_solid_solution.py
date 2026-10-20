"""Finite native solid demixing, heating and cooling on the shared heater."""

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.solid_world import solid_world_spec


def main():
    runtime = ChemWorldRuntime(composition=solid_world_spec())
    runtime.reset(seed=40)
    for target in (800, 300):
        result = runtime.execute(
            {"operation": "heat", "target_temperature_K": target, "duration_s": 120}
        )
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        observed = runtime.execute({"operation": "measure", "instrument": "solid_phase_meter"})
        if observed.info["transaction_status"] != "committed":
            raise RuntimeError(observed.info)
        print({c.channel_id: c.value for c in observed.observation.channels}, flush=True)


if __name__ == "__main__":
    main()
