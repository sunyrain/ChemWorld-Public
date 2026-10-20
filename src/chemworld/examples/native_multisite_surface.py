"""One native carrier with two site classes and finite thermal research."""

from chemworld.physchem.surface_specs import MultisiteSurfaceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def main():
    model = MultisiteSurfaceSpec(
        interaction_matrix_J_mol=(
            (5000, 0, 2500, 0),
            (0, 5000, 0, 2500),
            (2500, 0, 5000, 0),
            (0, 2500, 0, 5000),
        )
    )
    runtime = ChemWorldRuntime(composition=surface_world_spec(surface_spec=model))
    runtime.reset(seed=38)
    for action in (
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 2e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 1e-5},
        {"operation": "heat", "target_temperature_K": 400, "duration_s": 60},
        {"operation": "measure", "instrument": "gas_composition"},
        {"operation": "wait", "duration_s": 60},
        {"operation": "measure", "instrument": "gas_composition"},
    ):
        result = runtime.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        print(action["operation"], runtime._state.ledger.time_s,
              {c.channel_id: c.value for c in result.observation.channels}
              if action["operation"] == "measure"
              else {"temperature_K": runtime._state.temperature_K}, flush=True)


if __name__ == "__main__":
    main()
