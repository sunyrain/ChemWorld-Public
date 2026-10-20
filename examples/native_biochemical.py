"""Offline low-copy experiment, JSON snapshot and exact resumed intervention."""

import json

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_world


def main():
    runtime = ChemWorldRuntime(composition=biochemical_world(seed=7))
    runtime.reset(seed=12)  # detector stream; process seed belongs to the world
    for action in (
        {"operation": "add_component", "component": 1, "amount_mol": 10 / AVOGADRO},
        {"operation": "wait", "duration_s": 5},
        {"operation": "measure", "instrument": "molecule_counter"},
        {"operation": "measure", "instrument": "molecule_capture"},
    ):
        result = runtime.execute(action)
        assert result.info["transaction_status"] == "committed"
        print(action, {c.channel_id: c.value for c in result.observation.channels})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for instance in (runtime, restored):
        instance.execute({"operation": "wait", "duration_s": 3})
    assert runtime._state.to_dict() == restored._state.to_dict()
    print("Exact serialized continuation verified (evaluator-only snapshot).")


if __name__ == "__main__":
    main()
