"""Author original-source public world JSON before actual wheel consumption."""

import argparse
import json
from pathlib import Path

from integer51h_distribution_cases import units
from integer51h_source_cases import job

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.world.sbml_world import sbml_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xml = job(units()[1], "direct")["sbml"]
    model = SBMLModel(
        xml,
        InputSource(
            DataSourceProvenance(
                "sbml51i-package", 0, "Self-authored H S02 elementary source", "1", "CC0-1.0"
            ),
            "synthetic",
        ),
        items_per_mole=AVOGADRO,
    )
    for name, method in (
        ("direct", "gillespie-direct-v1"),
        ("next", "gibson-bruck-next-reaction-v1"),
        ("tau", "adaptive-explicit-tau-leap-v1"),
    ):
        world = sbml_world(
            model,
            numerical_form="stochastic",
            solver_method=method,
            tau_policy=TauLeapPolicy(max_tau_s=0.01, ssa_threshold=0.01) if name == "tau" else None,
            seed=17,
            maximum_events=100000,
            budget=100,
            controlled_parameters=("k0",),
            readout_relative_sd=0.02,
            observables=(
                SBMLObservable("amount", "A"),
                SBMLObservable("amount", "B"),
                SBMLObservable("volume", "cell"),
                SBMLObservable("reaction_rate", "reaction"),
            ),
        )
        (args.output / (name + ".json")).write_text(
            json.dumps(world.to_dict(), indent=2), encoding="utf-8"
        )
    (args.output / "source.xml").write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    main()
