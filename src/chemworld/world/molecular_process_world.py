"""A declared steady path filling one actual homogeneous finite receiver."""

from collections.abc import Mapping
from math import isfinite
from typing import cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.molecular_process_runtime import MolecularProcessReceiverSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_molecular_process_world(
    spec: MolecularProcessReceiverSpec,
    initial_state: WorldState,
    *,
    formulas: Mapping[str, str],
    budget: int = 100,
    duration_range_s: tuple[float, float] = (1, 14400),
) -> WorldCompositionSpec:
    """Declare the original inventory, hardware limits and admissible wait duration."""
    if (
        len(duration_range_s) != 2
        or any(isinstance(v, bool) or not isfinite(v) for v in duration_range_s)
        or not 1e-6 <= duration_range_s[0] < duration_range_s[1] <= 14400
    ):
        raise ValueError(
            "process world requires finite ordered duration bounds within 1e-6..14400 s"
        )
    ids = spec.path.domain.model.component_ids
    if set(formulas) != set(ids):
        raise ValueError("molecular process species require their explicit formulas")
    vessel = initial_state.vessel_id
    network = ReactionNetworkSpec(
        "authored-molecular-process",
        tuple(SpeciesSpec(k, formulas[k], phase=initial_state.phase) for k in ids),
        (),
    )
    physics = CompositionPhysics(
        network,
        {vessel: initial_state},
        models={"molecular_process": {vessel: spec}},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "molecular-process-finite-receiver",
            "components": [
                {
                    "id": f"{vessel}:process",
                    "kind": "molecular_process",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": f"{vessel}:thermal",
                    "kind": "thermal",
                    "vessel": vessel,
                    "parameters": {
                        "temperature_range_K": list(spec.path.domain.temperature_range_K),
                        "duration_range_s": list(duration_range_s),
                    },
                },
            ],
            "vessels": [
                {
                    "id": vessel,
                    "capacity_L": cast(VesselLedger, initial_state.vessels)
                    .vessels[vessel]
                    .max_volume_L,
                }
            ],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "fill a finite receiver through a declared molecular process",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
