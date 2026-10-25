"""Author public world JSON as data; installed consumers never import checkout helpers."""

import argparse
import json
from dataclasses import replace

from cross_partition51_cases import shared_world as cross_world
from stochastic_hybrid51_cases import shared_world as feedback_world

from chemworld.physchem.stochastic_hybrid import initial_hybrid_continuation


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for kind, factory in (("feedback", feedback_world), ("cross", cross_world)):
        original = factory(seed=17)
        payload = original.to_dict()
        for index, identity in enumerate(("reactor", "second")):
            spec = replace(
                original.physics.models_of("stochastic_hybrid")[identity], ode_method="RK45"
            )
            payload["physics"]["models"]["stochastic_hybrid"]["bindings"][identity] = spec.to_dict()
            payload["physics"]["initial_states"][identity]["process"]["stochastic_continuation"] = (
                initial_hybrid_continuation(
                    original.physics.network,
                    spec,
                    original.physics.initial_states[identity].species_amounts,
                    seed=17 + index,
                )
            )
        (args.output / (kind + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
