"""Prospectively fixed original-network authoring for 51B."""

from copy import deepcopy

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_hybrid import StochasticHybridSpec, initial_hybrid_continuation
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world

CASES = (
    {"k0": .15, "k1": .8, "alpha": 1.2, "beta": .7, "A0_mol": 1e-18},
    {"k0": .4, "k1": 1.5, "alpha": 2., "beta": 1.1, "A0_mol": 1e-15},
    {"k0": .05, "k1": .4, "alpha": .3, "beta": 1.8, "A0_mol": 1e-12},
    {"k0": .7, "k1": .2, "alpha": 3., "beta": .2, "A0_mol": 1e-9},
)
VOLUME = 1e-12


def author(case=None, *, max_events=10000):
    case = CASES[0] if case is None else case
    rows = (
        ("activate", "E0 => E1", case["alpha"] * VOLUME / case["A0_mol"], {"E0": 1, "A": 1}),
        ("deactivate", "E1 => E0", case["beta"], {"E1": 1}),
        ("basal", "A => B", case["k0"], {"A": 1}),
        ("catalysed", "A => B", case["k1"] * AVOGADRO * VOLUME, {"A": 1, "E1": 1}),
    )
    network = ReactionNetworkSpec("hybrid51-feedback", tuple(SpeciesSpec(k, f)
        for k, f in (("A", "H2"), ("B", "H2"), ("E0", "He"), ("E1", "He"))),
        tuple(ReactionSpec.from_equation(reaction_id=name, equation=eq,
            rate_law=RateLawSpec(name, "mass_action", {"k": k}), forward_orders=orders)
            for name, eq, k, orders in rows))
    spec = StochasticHybridSpec(("E0", "E1"), ("activate", "deactivate"), VOLUME,
                                max_events_per_advance=max_events)
    return network, spec, {"A": case["A0_mol"], "B": 0., "E0": 1 / AVOGADRO, "E1": 0.}


def shared_world(seed=17):
    network, spec, amounts = author()
    payload = stochastic_hybrid_world(network, spec, amounts, seed=seed).to_dict()
    payload["composition_id"] = "hybrid51-shared"
    payload["vessels"].append({"id": "second", "capacity_L": VOLUME})
    payload["components"][0]["vessel"] = "reactor"
    payload["components"].append({"id": "second:stochastic", "kind": "stochastic",
                                  "vessel": "second", "parameters": {}})
    def rename(value):
        if isinstance(value, dict):
            return {rename(k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        if isinstance(value, str) and (value == "reactor" or value.startswith("reactor:")):
            return "second" + value[len("reactor"):]
        return value

    second = rename(deepcopy(payload["physics"]["initial_states"]["reactor"]))
    second["process"]["stochastic_continuation"] = initial_hybrid_continuation(
        network, spec, amounts, seed=seed + 1)
    payload["physics"]["initial_states"]["second"] = second
    payload["physics"]["models"]["stochastic_hybrid"]["bindings"]["second"] = spec.to_dict()
    payload["physics"]["lab_clock"] = LabClockSpec().to_dict()
    return WorldCompositionSpec.from_dict(payload)
