"""Read the original finite-heat CALPHAD world's actual ternary inventory."""

from collections.abc import Callable
from math import fsum
from typing import Any, cast

from chemworld.foundation import WorldState
from chemworld.physchem.calphad_ternary_diagram import (
    CalphadTernaryPath,
    CalphadTernarySpec,
    Composition,
    scan_calphad_ternary_diagram,
)
from chemworld.runtime.calphad_thermal import CalphadThermalSpec, validate_calphad_thermal_state
from chemworld.world.composition import WorldCompositionSpec


def analyze_calphad_ternary_world(
    composition: WorldCompositionSpec,
    state: WorldState,
    *,
    grid_intervals: int = 6,
    requested_compositions: tuple[Composition, ...] = (),
    paths: tuple[CalphadTernaryPath, ...] = (),
    maximum_evaluations: int = 256,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if composition.physics is None or state.phases is None:
        raise ValueError("ternary query requires original composition physics and phase inventory")
    original = composition.physics.models_of("calphad_material").get(state.vessel_id)
    if not isinstance(original, CalphadThermalSpec) or len(original.component_ids) != 3:
        raise ValueError("ternary query requires the original three-element CALPHAD thermal world")
    validate_calphad_thermal_state(state, original)
    components = cast(tuple[str, str, str], original.component_ids)
    amounts = {
        e: fsum(p.species_amounts_mol.get(e, 0.0) for p in state.phases.phases.values())
        for e in components
    }
    total = fsum(amounts.values())
    actual = cast(Composition, tuple(amounts[e] / total for e in components))
    spec = CalphadTernarySpec(
        original.caloric.equilibrium,
        components,
        state.temperature_K,
        state.pressure_Pa,
        grid_intervals,
        (*requested_compositions, actual),
        paths,
        maximum_evaluations,
    )
    diagram = scan_calphad_ternary_diagram(spec, progress=progress)
    index = next(i for i, p in enumerate(diagram.points) if tuple(p["composition"]) == actual)
    return {
        "diagram": diagram.to_dict(),
        "world_query": {
            "vessel_id": state.vessel_id,
            "state_time_s": state.ledger.time_s,
            "actual_element_amounts_mol": amounts,
            "actual_composition": list(actual),
            "actual_point_index": index,
            "world_advanced": False,
            "inventory_owner": "original_PhaseLedger",
            "physical_time_s": None,
            "heat_duty_J": None,
        },
    }
