"""Finite carbonate preparation and pH reads without an upstream library."""

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral import committed_speciation
from chemworld.world.aqueous_world import aqueous_world_spec


def run_example():
    session = ChemWorldRuntime(composition=aqueous_world_spec())
    session.reset(seed=32)
    trace = []
    try:
        for action in (
            {"operation": "wait", "duration_s": 2},
            {"operation": "measure", "instrument": "aqueous_ph"},
            {"operation": "add_component", "component": "HCl-solution", "amount_mol": 0.01},
            {"operation": "wait", "duration_s": 3},
            {"operation": "add_component", "component": "NaOH-solution", "amount_mol": 0.008},
            {"operation": "sample", "sample_volume_L": 0.001},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            outcome = session.execute(action)
            if outcome.info["transaction_status"] != "committed":
                raise RuntimeError(outcome.info.get("error_message"))
            if action["operation"] == "measure" and outcome.info["measurement_status"] != "valid":
                raise RuntimeError("native aqueous pH readout failed")
            trace.append(
                {
                    "action": action,
                    "time_s": session._state.ledger.time_s,
                    "public_reading": {
                        c.channel_id: c.value for c in outcome.observation.channels if c.observed
                    },
                }
            )
        state = session._state
        return {
            "mode": "synthetic-development",
            "operations": trace,
            "final_speciation": committed_speciation(state).to_dict(),
            "sample_consumed_L": state.ledger.sample_consumed_L,
            "remaining_finite_sources": {
                k: v.species_amounts for k, v in state.inactive_vessels.items()
            },
            "not_covered": ["temperature_dependence", "caloric_closure", "mineral_kinetics"],
        }
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))
