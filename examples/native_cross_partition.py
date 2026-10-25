"""Actual discrete/continuous molecular exchange in the original ChemWorld."""

import json

from chemworld.foundation.discrete_inventory import AVOGADRO, mol_to_counts
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_cross_partition import CrossPartitionHybridSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world


def author(case=None, *, max_events=10000):
    case = {"release": .7, "capture": .6, "forward": .5, "reverse": .3,
            "catalytic": .15, "D": 4, "A": 2.25, "B": 1.5, **(case or {})}
    volume = 1e-15
    rows = (("release", "D => A", case["release"], {"D": 1}),
            ("capture", "A => D", case["capture"], {"A": 1}),
            ("forward", "A => B", case["forward"], {"A": 1}),
            ("reverse", "B => A", case["reverse"], {"B": 1}),
            ("catalytic", "A => B", case["catalytic"] * AVOGADRO * volume, {"A": 1, "D": 1}))
    network = ReactionNetworkSpec("cross-partition-molecular-exchange", tuple(
        SpeciesSpec(k, "H2") for k in ("D", "A", "B")), tuple(
        ReactionSpec.from_equation(reaction_id=name, equation=eq,
            rate_law=RateLawSpec(name, "mass_action", {"k": k}), forward_orders=orders)
        for name, eq, k, orders in rows))
    spec = CrossPartitionHybridSpec(("D",), ("release", "capture"), volume,
                                    max_events_per_advance=max_events)
    return network, spec, {k: case[k] / AVOGADRO for k in ("D", "A", "B")}


def main():
    network, spec, amounts = author()
    runtime = ChemWorldRuntime(composition=stochastic_hybrid_world(
        network, spec, amounts, seed=17))
    runtime.reset(seed=3)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["transaction_status"] == "committed", result.info
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for instance in (runtime, restored):
        result = instance.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["transaction_status"] == "committed", result.info
    assert runtime._state.to_dict() == restored._state.to_dict()
    print(json.dumps({"discrete_counts": mol_to_counts(
          {"D": runtime._state.species_amounts["D"]}),
          "continuous_molecule_equivalents": {k: runtime._state.species_amounts[k] * AVOGADRO
          for k in ("A", "B")},
          "transfer": runtime._state.process.stochastic_continuation["partition_transfer"],
          "exact_json_resume": True}, indent=2))


if __name__ == "__main__":
    main()
