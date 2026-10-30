"""Actual finite mixing/washing under an explicit synthetic liquid activity law."""

from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.phase_equilibrium_units import StrictLiquidSystemSpec
from chemworld.physchem.specs import ComponentSpec
from chemworld.world.liquid_world import liquid_world_spec


def synthetic_liquid_world(amplitude=3):
    model = StrictLiquidSystemSpec(
        ActivityModelSpec("synthetic-liquid", ("A", "B"), "margules",
                          {"A:A|B": amplitude, "A:B|A": amplitude}),
        {"A": .05, "B": .05}, wash_feed_id="wash-feed")
    return liquid_world_spec(model, (ComponentSpec("A", "C2H6O", metadata={"synthetic": True}),
                                     ComponentSpec("B", "C3H8O", metadata={"synthetic": True})))


def run_example(amplitude=3):
    from chemworld.runtime import ChemWorldRuntime

    session = ChemWorldRuntime(composition=synthetic_liquid_world(amplitude))
    session.reset(seed=12)
    history = []
    try:
        for action in (
            {"operation": "mix", "duration_s": 2, "stirring_speed_rpm": 600},
            {"operation": "wash", "wash_volume_L": 1e-5},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "route_material", "connection": "reactor-to-receiver",
             "transfer_fraction": .2, "mixing": 1},
            {"operation": "terminate"},
        ):
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(f"Liquid example failed: {result.info['error_message']}")
            history.append({"action": action, "outcome": result.info["operation_outcome"],
                            "actual_duration_s": result.info["actual_duration_s"]})
        return {"ended": result.terminated, "steps": history,
                "phases": session._state.phases.to_dict(),
                "reading": session.observe().to_dict(),
                "physical_time_s": session._state.ledger.time_s}
    finally:
        session.close()


def main() -> None:
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
