"""Fixed synthetic networks and independent finite distribution references."""

from math import comb, exp

import numpy as np
from scipy.linalg import expm
from scipy.stats import binom, multinomial

from chemworld.examples.native_domains import mixed_clock_world
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import AVOGADRO, initial_continuation
from chemworld.world.composition import WorldCompositionSpec

NEXT = "gibson-bruck-next-reaction-v1"
DIRECT = "gillespie-direct-v1"
VOLUME = 1e-15


def make_network(kind="reversible", *, algorithm=NEXT, max_events=1_000_000, pairs=16):
    if kind in {"irreversible", "reversible"}:
        formulas = {"A": "H2", "B": "H2"}
        rows = [("forward", "A => B", .7 if kind == "irreversible" else .5)]
        if kind == "reversible":
            rows.append(("reverse", "B => A", 1.5))
    elif kind == "multiscale":
        formulas = dict.fromkeys(("A", "B", "C"), "H2")
        rows = [("ab", "A => B", 12), ("ba", "B => A", 8),
                ("bc", "B => C", .15), ("cb", "C => B", .02)]
    elif kind == "dimer":
        formulas = {"A": "H", "B": "H2"}
        rows = [("pair", "2 A => B", .03 * AVOGADRO * VOLUME / 2),
                ("split", "B => 2 A", .4)]
    elif kind == "sparse":
        formulas = {key: "H2" for i in range(pairs) for key in (f"A{i}", f"B{i}")}
        rows = [(f"{direction}{i}", equation, rate) for i in range(pairs)
                for direction, equation, rate in (
                    ("f", f"A{i} => B{i}", .3), ("r", f"B{i} => A{i}", .7),
                )]
    else:
        raise ValueError("unknown fixed case")
    return ReactionNetworkSpec(
        f"next51-{kind}", tuple(SpeciesSpec(k, v) for k, v in formulas.items()),
        tuple(ReactionSpec.from_equation(
            reaction_id=name, equation=equation,
            rate_law=RateLawSpec(name, "mass_action", {"k": rate}),
        ) for name, equation, rate in rows),
        stochastic={"volume_L": VOLUME, "algorithm": algorithm,
                    "max_events_per_advance": max_events},
    )


def fixed_cases():
    return [
        *({"kind": "irreversible", "copies": n, "duration": .8} for n in (1, 10, 100, 1000)),
        *({"kind": "reversible", "copies": n, "duration": 1.} for n in (2, 20, 200, 1000)),
        *({"kind": "multiscale", "copies": n, "duration": .25} for n in (10, 100)),
        *({"kind": "dimer", "copies": n, "duration": 1.} for n in (8, 24)),
    ]


def independent_law(kind, copies, duration):
    """No calls to native propensity/stoichiometry code in the reference law."""
    if kind in {"irreversible", "reversible"}:
        p = 1 - exp(-.7 * duration) if kind == "irreversible" else .25 * (1 - exp(-2 * duration))
        support = np.asarray([(copies - b, b) for b in range(copies + 1)], dtype=int)
        probability = binom.pmf(np.arange(copies + 1), copies, p)
    elif kind == "multiscale":
        q = np.asarray([[-12., 12., 0.], [8., -8.15, .15], [0., .02, -.02]])
        p = expm(q * duration)[0]
        support = np.asarray([(a, b, copies - a - b) for a in range(copies + 1)
                              for b in range(copies - a + 1)], dtype=int)
        probability = multinomial.pmf(support, copies, p)
    elif kind == "dimer":
        size = copies // 2 + 1
        q = np.zeros((size, size))
        for b in range(size):
            if b + 1 < size:
                q[b, b + 1] = .03 * comb(copies - 2 * b, 2)
            if b > 0:
                q[b, b - 1] = .4 * b
            q[b, b] = -sum(q[b])
        probability = expm(q * duration)[0]
        support = np.asarray([(copies - 2 * b, b) for b in range(size)], dtype=int)
    else:
        raise ValueError("unknown independent distribution")
    if abs(sum(probability) - 1) > 1e-11 or min(probability) < -1e-14:
        raise ArithmeticError("independent reference distribution is invalid")
    return support, probability


def shared_world(seed):
    payload = mixed_clock_world().to_dict()
    physics = payload["physics"]
    physics["network"]["stochastic"]["algorithm"] = NEXT
    for state in physics["initial_states"].values():
        old = state["process"].get("stochastic_continuation")
        if old is not None:
            continuation = initial_continuation(seed, algorithm=NEXT)
            if "sampling_rng_state" in old:
                continuation["sampling_rng_state"] = old["sampling_rng_state"]
            state["process"]["stochastic_continuation"] = continuation
    return WorldCompositionSpec.from_dict(payload)
