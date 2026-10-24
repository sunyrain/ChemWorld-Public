"""Select a ternary analysis point using only the original actual phase stock."""

from collections.abc import Sequence
from math import fsum

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumPhase
from chemworld.physchem.calphad_ternary_diagram import CalphadTernaryResult
from chemworld.runtime.calphad_phase_inventory import (
    CalphadPhaseInventoryResult,
    CalphadPhaseTarget,
    equilibrate_calphad_phase_inventory,
)


def select_calphad_ternary_inventory_point(
    ledger: PhaseLedger,
    diagram: CalphadTernaryResult,
    point_index: int,
    *,
    vessel_id: str,
    source_phase_ids: Sequence[str],
    phase_targets: Sequence[CalphadPhaseTarget],
) -> CalphadPhaseInventoryResult:
    if type(point_index) is not int or not 0 <= point_index < len(diagram.points):
        raise ValueError("ternary selection requires an existing point")
    point, spec = diagram.points[point_index], diagram.specification
    if not point["success"]:
        raise ValueError("cannot materialize an unresolved ternary point")
    if (
        not source_phase_ids
        or len(set(source_phase_ids)) != len(source_phase_ids)
        or any(key not in ledger.phases for key in source_phase_ids)
    ):
        raise ValueError("ternary selection requires distinct original source phases")
    records = [ledger.phases[key] for key in source_phase_ids]
    if any(
        p.vessel_id != vessel_id or set(p.species_amounts_mol) - set(spec.components)
        for p in records
    ):
        raise ValueError("ternary selected inventory has different owner or components")
    amounts = {e: fsum(p.species_amounts_mol.get(e, 0.0) for p in records) for e in spec.components}
    total = fsum(amounts.values())
    if total <= 0 or any(
        abs(amounts[e] / total - x) > 1e-10
        for e, x in zip(spec.components, point["composition"], strict=True)
    ):
        raise ValueError("ternary point composition differs from actual inventory")
    seeds = tuple(
        CalphadEquilibriumPhase(
            p["phase_name"], tuple(dict(s) for s in p["site_fractions"]), {}, 0.0, 0.0, 0.0, 0.0
        )
        for p in point["equilibrium"]["phases"]
    )
    return equilibrate_calphad_phase_inventory(
        ledger,
        spec.equilibrium,
        vessel_id=vessel_id,
        source_phase_ids=source_phase_ids,
        phase_targets=phase_targets,
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        initial_phases=seeds,
    )
