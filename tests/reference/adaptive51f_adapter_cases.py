"""Fixed actual-method 29/30 wiring inputs; no material rounding in native inputs."""

from dataclasses import replace

from sde51e_cases import source_sbml

from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM, AdaptiveHybridSpec
from chemworld.world.biochemical_worlds import biochemical_network

PROFILES = (
    ("ode", {"A": 1200.0, "B": 1200.0}, 0.0, 0.0, [17]),
    ("ssa", {"A": 40.0, "B": 40.0}, 1e9, 1e9, [1, 17, 37, 17]),
    ("switch", {"A": 80.0, "B": 80.0}, 50.0, 75.0, [1, 17, 37, 17]),
    ("fractional", {"A": 50.25, "B": 50.75}, 45.0, 45.0, [1, 17, 37, 17]),
)
UNITS = tuple(
    {
        "id": f"F{index * 4 + offset:02d}",
        "method": method,
        "profile": profile,
        "initial": initial,
        "lower": lower,
        "upper": upper,
        "seeds": seeds,
    }
    for index, method in enumerate(("RK4", "LSODA"))
    for offset, (profile, initial, lower, upper, seeds) in enumerate(PROFILES, 1)
)
REACTIONS = [
    {"id": "forward", "stoichiometry": {"A": -1, "B": 1}, "orders": {"A": 1}, "k": 1.0},
    {"id": "reverse", "stoichiometry": {"A": 1, "B": -1}, "orders": {"B": 1}, "k": 0.2},
]


def author(unit):
    original = biochemical_network("reversible")
    reactions = tuple(
        replace(r, rate_law=replace(r.rate_law, parameters={"k": data["k"]}))
        for r, data in zip(original.reactions, REACTIONS, strict=True)
    )
    network = replace(original, reactions=reactions, stochastic=None)
    spec = AdaptiveHybridSpec(
        volume_L=1e-15,
        method=unit["method"],
        lower_limit=unit["lower"],
        upper_limit=unit["upper"],
        step_s=0.125,
        partition_steps=1,
        rtol=1e-10,
        atol_particles=1e-12,
    )
    return network, spec, {k: n * MOLECULAR_QUANTUM for k, n in unit["initial"].items()}


def jobs(unit):
    return [
        {
            **unit,
            "id": f"{unit['id']}-{index:02d}",
            "seed": seed,
            "duration_s": 1.0,
            "output_steps": 8,
            "volume_L": 1e-15,
            "initial_fluxes": {
                "forward": unit["initial"]["A"],
                "reverse": 0.2 * unit["initial"]["B"],
            },
            "sbml": source_sbml({**unit, "reactions": REACTIONS}),
        }
        for index, seed in enumerate(unit["seeds"], 1)
    ]
