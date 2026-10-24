"""Apply a resolved diagram point through the original zero-time TP ledger boundary."""

from collections.abc import Sequence
from math import fsum

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumPhase
from chemworld.physchem.calphad_phase_diagram import CalphadDiagramResult
from chemworld.runtime.calphad_phase_inventory import (
    CalphadPhaseInventoryResult,
    CalphadPhaseTarget,
    equilibrate_calphad_phase_inventory,
)


def select_calphad_diagram_inventory_point(
    ledger: PhaseLedger,
    diagram: CalphadDiagramResult,
    point_index: int,
    *,
    vessel_id: str,
    source_phase_ids: Sequence[str],
    phase_targets: Sequence[CalphadPhaseTarget],
) -> CalphadPhaseInventoryResult:
    if type(point_index) is not int or not 0 <= point_index < len(diagram.scan_points):
        raise ValueError("diagram selection requires an existing scan point")
    point, spec = diagram.scan_points[point_index], diagram.specification
    if not point["success"]:
        raise ValueError("cannot materialize an unresolved diagram point")
    if not source_phase_ids or any(key not in ledger.phases for key in source_phase_ids):
        raise ValueError("diagram materialization requires original selected phases")
    records = [ledger.phases[key] for key in source_phase_ids]
    if any(set(p.species_amounts_mol) - set(spec.components) for p in records):
        raise ValueError("selected inventory differs from diagram components")
    amounts = {e: fsum(p.species_amounts_mol.get(e, 0.0) for p in records) for e in spec.components}
    total = fsum(amounts.values())
    if total <= 0 or abs(amounts[spec.components[0]] / total - point["composition"]) > 1e-10:
        raise ValueError("diagram point composition differs from actual inventory")
    # Cached phase amounts are analysis outputs, never a source of material.
    # Only phase identity and constitutions can seed the original numerical search.
    seeds = tuple(CalphadEquilibriumPhase(
        p["phase_name"], tuple(dict(s) for s in p["site_fractions"]), {}, 0., 0., 0., 0.)
        for p in point["equilibrium"]["phases"])
    return equilibrate_calphad_phase_inventory(
        ledger,
        spec.equilibrium,
        vessel_id=vessel_id,
        source_phase_ids=source_phase_ids,
        phase_targets=phase_targets,
        temperature_K=point["temperature_K"],
        pressure_Pa=spec.pressure_Pa,
        initial_phases=seeds,
    )
