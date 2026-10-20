"""Finite synthetic mechanisms, not fitted biological or reduced MM models."""

import numpy as np

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_world import stochastic_world


def biochemical_network(kind="enzyme", *, scale=1):
    if type(scale) is not int or scale <= 0:
        raise ValueError("copy/volume scale must be a positive integer")
    if kind == "reversible":
        formulas = {"A": "CH2O", "B": "CH2O"}
        channels = (("forward", "A => B", 0.1), ("reverse", "B => A", 0.2))
    elif kind == "enzyme":
        formulas = {"E": "N2", "S": "CH2O", "ES": "CH2N2O", "P": "CH2O"}
        channels = (
            ("binding", "E + S => ES", 1e6),
            ("unbinding", "ES => E + S", 1.0),
            ("catalysis", "ES => E + P", 0.1),
        )
    else:
        raise ValueError("supported finite mechanisms: reversible, enzyme")
    return ReactionNetworkSpec(
        f"synthetic-{kind}",
        tuple(SpeciesSpec(key, formula) for key, formula in formulas.items()),
        tuple(
            ReactionSpec.from_equation(
                reaction_id=name,
                equation=equation,
                rate_law=RateLawSpec(name, "mass_action", {"k": k}),
            )
            for name, equation, k in channels
        ),
        stochastic={"volume_L": 1e-15 * scale, "max_events_per_advance": 10_000_000},
    )


def biochemical_world(kind="enzyme", *, counts=None, seed=0, budget=40):
    network = biochemical_network(kind)
    if counts is None:
        counts = {"A": 100} if kind == "reversible" else {"E": 10, "S": 100}
    payload = stochastic_world(network, counts, seed=seed, budget=budget).to_dict()
    continuation = payload["physics"]["initial_states"]["reactor"]["process"][
        "stochastic_continuation"
    ]
    continuation["sampling_rng_state"] = np.random.PCG64(np.random.SeedSequence([seed, 3710])).state
    payload["components"].extend(
        [
            {"kind": "material_input", "parameters": {}},
            {
                "kind": "observation",
                "parameters": {"instruments": ["molecule_counter", "molecule_capture"]},
            },
        ]
    )
    payload["task"].update(
        objective="characterize a finite low-copy synthetic network",
        operations=["add_component", "wait", "measure", "terminate"],
        instruments=["molecule_counter", "molecule_capture"],
    )
    return WorldCompositionSpec.from_dict(payload)
