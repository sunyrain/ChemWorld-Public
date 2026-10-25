"""51D fixed authoring and original shared-clock consumer."""

import importlib.util
from copy import deepcopy
from pathlib import Path

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.reaction_network import RateLawSpec, ReactionNetworkSpec, ReactionSpec
from chemworld.physchem.stochastic_cross_partition import CrossPartitionHybridSpec
from chemworld.physchem.stochastic_hybrid import initial_hybrid_continuation
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world

_source = Path(__file__).resolve().parents[2] / "examples/native_cross_partition.py"
_spec = importlib.util.spec_from_file_location("native_cross_partition", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
author = _module.author

BASE = {"release": .7, "capture": .6, "forward": .5, "reverse": .3,
        "catalytic": .15, "D": 4, "A": 2.25, "B": 1.5}
CASES = (
    BASE,
    {**BASE, "D": 0, "A": .25, "B": 3.5, "reverse": 1., "catalytic": 0.},
    {**BASE, "D": 0, "A": 2.25, "B": 0., "release": 0., "capture": .05,
     "forward": 2., "reverse": 0., "catalytic": 0.},
    {**BASE, "D": 2, "A": 0., "B": 0., "forward": .2, "reverse": .8},
    {**BASE, "D": 4, "A": 4., "B": 0., "forward": 0., "reverse": 0., "catalytic": 0.},
    {**BASE, "D": 0, "A": 1., "B": 0., "release": 0., "capture": 0.,
     "forward": 0., "reverse": 0., "catalytic": 0.},
)


def shared_world(seed=17):
    network, spec, amounts = author()
    payload = stochastic_hybrid_world(network, spec, amounts, seed=seed).to_dict()
    payload["composition_id"] = "cross-partition51-shared"
    payload["task"]["operations"].append("select_vessel")
    payload["vessels"].append({"id": "second", "capacity_L": spec.volume_L})
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


def quantum_author(quantum, *, fractional=False):
    base, _, _ = author()
    volume = 1e-15
    scale = AVOGADRO * volume
    rows = (("release", f"{quantum} D => {quantum} A",
             0. if fractional else .4 * scale ** (quantum - 1)),
            ("capture", f"{quantum} A => {quantum} D", .3 * scale ** (quantum - 1)),
            ("flow", "A => B", 0.))
    network = ReactionNetworkSpec(f"cross-partition-quantum-{quantum}", base.species,
        tuple(ReactionSpec.from_equation(reaction_id=name, equation=eq,
            rate_law=RateLawSpec(name, "mass_action", {"k": k})) for name, eq, k in rows))
    spec = CrossPartitionHybridSpec(("D",), ("release", "capture"), volume)
    amounts = {"D": (0 if fractional else quantum) / AVOGADRO,
               "A": (quantum - .25 if fractional else 0) / AVOGADRO, "B": 0.}
    return network, spec, amounts
