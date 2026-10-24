"""Prospective 51C inputs; no outcome-dependent case or seed selection."""

from math import factorial

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import AVOGADRO, TAU_ALGORITHM, initial_continuation
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.world.composition import WorldCompositionSpec

VOLUME = 1e-15
DURATION = .973
REPLICATES = 2048
POLICIES = tuple(TauLeapPolicy(epsilon=e, max_tau_s=t, ssa_threshold=2.)
                 for e, t in ((.1, .1), (.03, .04), (.01, .015)))
CASES = (
    {"id": "low-depletion", "kind": "linear", "species": {"A": "H", "B": "H"},
     "initial": {"A": 3}, "rows": (("ab", "A => B", .7, (1,)),),
     "q": ((-.7, .7), (0., 0.))},
    {"id": "high-depletion", "kind": "linear", "species": {"A": "H", "B": "H"},
     "initial": {"A": 1000}, "rows": (("ab", "A => B", .7, (1,)),),
     "q": ((-.7, .7), (0., 0.))},
    {"id": "reversible", "kind": "linear", "species": {"A": "H", "B": "H"},
     "initial": {"A": 1000},
     "rows": (("ab", "A => B", .5, (1,)), ("ba", "B => A", 1.5, (1,))),
     "q": ((-.5, .5), (1.5, -1.5))},
    {"id": "competing", "kind": "linear", "species": dict.fromkeys(("A", "B", "C"), "H"),
     "initial": {"A": 1000},
     "rows": (("ab", "A => B", .4, (1,)), ("ac", "A => C", .3, (1,))),
     "q": ((-.7, .4, .3), (0., 0., 0.), (0., 0., 0.))},
    {"id": "finite-birth-death", "kind": "linear",
     "species": dict.fromkeys(("F", "A", "W"), "H"), "initial": {"F": 1000},
     "rows": (("fa", "F => A", .4, (1,)), ("aw", "A => W", .3, (1,))),
     "q": ((-.4, .4, 0.), (0., -.3, .3), (0., 0., 0.))},
    {"id": "multiscale", "kind": "linear", "species": dict.fromkeys(("A", "B", "C"), "H"),
     "initial": {"A": 200},
     "rows": (("ab", "A => B", 12., (1,)), ("ba", "B => A", 8., (1,)),
              ("bc", "B => C", .15, (1,))),
     "q": ((-12., 12., 0.), (8., -8.15, .15), (0., 0., 0.))},
    {"id": "dimer", "kind": "polymer", "order": 2,
     "species": {"A": "H", "B": "H2"}, "initial": {"A": 200},
     "rows": (("form", "2 A => B", .003, (2,)), ("split", "B => 2 A", .4, (1,)))},
    {"id": "trimer", "kind": "polymer", "order": 3,
     "species": {"A": "H", "C": "H3"}, "initial": {"A": 150},
     "rows": (("form", "3 A => C", .0001, (3,)), ("split", "C => 3 A", .4, (1,)))},
)


def make_network(case=CASES[1], policy=POLICIES[1], *, algorithm=TAU_ALGORITHM,
                 max_events=1000000):
    rows = []
    for identity, equation, coefficient, orders in case["rows"]:
        k = coefficient * (AVOGADRO * VOLUME) ** (sum(orders) - 1)
        for order in orders:
            k /= factorial(order)
        rows.append(ReactionSpec.from_equation(
            reaction_id=identity, equation=equation,
            rate_law=RateLawSpec(identity, "mass_action", {"k": k})))
    declaration = {"volume_L": VOLUME, "algorithm": algorithm,
                   "max_events_per_advance": max_events}
    if algorithm == TAU_ALGORITHM:
        declaration["tau_policy"] = policy.to_dict()
    return ReactionNetworkSpec(
        "tau51-" + case["id"], tuple(SpeciesSpec(k, f) for k, f in case["species"].items()),
        tuple(rows), stochastic=declaration)


def convert_world(composition, seed=513900020, policy=POLICIES[1]):
    payload = composition.to_dict()
    physics = payload["physics"]
    physics["network"]["stochastic"].update(algorithm=TAU_ALGORITHM, tau_policy=policy.to_dict())
    for state in physics["initial_states"].values():
        old = state["process"].get("stochastic_continuation")
        if old is not None:
            continuation = initial_continuation(seed, old["time_s"], algorithm=TAU_ALGORITHM)
            if "sampling_rng_state" in old:
                continuation["sampling_rng_state"] = old["sampling_rng_state"]
            state["process"]["stochastic_continuation"] = continuation
    return WorldCompositionSpec.from_dict(payload)
