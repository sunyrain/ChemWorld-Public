"""Original native world using declared approximate, integer Poisson batches."""

from typing import Any

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import TAU_ALGORITHM
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_world import stochastic_world


def tau_world(*, seed: int = 51, copies: int = 10000) -> WorldCompositionSpec:
    network = ReactionNetworkSpec(
        "finite-tau-conversion", (SpeciesSpec("A", "H2"), SpeciesSpec("B", "H2")),
        (ReactionSpec.from_equation(reaction_id="convert", equation="A => B",
                                   rate_law=RateLawSpec("k", "mass_action", {"k": .7})),),
        stochastic={"volume_L": 1e-15, "algorithm": TAU_ALGORITHM,
                    "tau_policy": TauLeapPolicy().to_dict()})
    return stochastic_world(network, {"A": copies}, seed=seed)


def run_example() -> dict[str, Any]:
    runtime = ChemWorldRuntime(composition=tau_world())
    runtime.reset(seed=5)
    first = runtime.execute({"operation": "wait", "duration_s": 1})
    restored = ChemWorldRuntime.from_snapshot(runtime.snapshot().to_dict())
    action = {"operation": "wait", "duration_s": 2}
    second = runtime.execute(action)
    restored.execute(action)
    return {"first": first.info["outcome_diagnostics"],
            "second": second.info["outcome_diagnostics"],
            "exact_resume": runtime._state.to_dict() == restored._state.to_dict(),
            "output": "Approximate step-held leap endpoints; no SSA event-time claim."}
