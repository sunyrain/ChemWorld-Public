"""Public authored integer worlds as JSON; installed consumers need no test imports."""

import argparse
import json
from pathlib import Path

import numpy as np

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import AVOGADRO, TAU_ALGORITHM
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_world import stochastic_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, algorithm in (
        ("direct", "gillespie-direct-v1"),
        ("next", "gibson-bruck-next-reaction-v1"),
        ("tau", TAU_ALGORITHM),
    ):
        declaration = {"volume_L": 1e-15, "algorithm": algorithm, "max_events_per_advance": 1000000}
        if name == "tau":
            declaration["tau_policy"] = TauLeapPolicy(max_tau_s=0.01).to_dict()
        network = ReactionNetworkSpec(
            "integer-regenerated51",
            tuple(SpeciesSpec(k, "CH2O") for k in ("A", "B", "C")),
            (
                ReactionSpec.from_equation(
                    reaction_id="reaction",
                    equation="A + B => 2 A",
                    forward_orders={"A": 1, "B": 1},
                    rate_law=RateLawSpec(
                        "reaction", "mass_action", {"k": 0.002 * AVOGADRO * 1e-15}
                    ),
                ),
            ),
            stochastic=declaration,
        )
        payload = stochastic_world(network, {"A": 39, "B": 261, "C": 0}, seed=17).to_dict()
        payload["physics"]["initial_states"]["reactor"]["process"]["stochastic_continuation"][
            "sampling_rng_state"
        ] = np.random.PCG64(np.random.SeedSequence([17, 3710])).state
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
            operations=["add_component", "wait", "measure", "terminate"],
            instruments=["molecule_counter", "molecule_capture"],
        )
        world = WorldCompositionSpec.from_dict(payload)
        (args.output / (name + ".json")).write_text(
            json.dumps(world.to_dict(), indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
