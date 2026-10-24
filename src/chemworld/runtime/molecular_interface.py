"""Read planar interface endpoints from the sole native phase inventory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel, molecular_eos_state
from chemworld.physchem.molecular_interface import (
    MolecularInterfaceResult,
    SquareGradientInfluence,
    molecular_planar_interface,
    pure_square_gradient_interface,
)


@dataclass(frozen=True)
class MolecularInventoryInterface:
    left_phase_id: str
    right_phase_id: str
    interface: MolecularInterfaceResult
    area_m2: float | None
    surface_excess_grand_potential_J: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"left_phase_id": self.left_phase_id, "right_phase_id": self.right_phase_id,
                "interface": self.interface.to_dict(), "area_m2": self.area_m2,
                "surface_excess_grand_potential_J": self.surface_excess_grand_potential_J,
                "state_effect": "read_only; no material, time, heat or internal-energy update"}


def molecular_interface_from_ledger(
    ledger: PhaseLedger, model: ResidualHelmholtzModel, influence: SquareGradientInfluence, *,
    vessel_id: str, left_phase_id: str, right_phase_id: str, temperature_K: float,
    area_m2: float | None = None, interior_density_grid: Sequence[float] | None = None,
    mesh: DiffusionMesh | None = None,
    component_density_bounds: Sequence[tuple[float, float]] | None = None,
    initial_profile: Sequence[Sequence[float]] | None = None, max_iterations: int = 500,
) -> MolecularInventoryInterface:
    """Rebuild both same-temperature endpoint states from their real n/V.

    An explicit area returns gamma*A, an excess grand potential. It is not a
    heat duty or an internal energy and is never written to a thermal ledger.
    """
    if left_phase_id == right_phase_id:
        raise ValueError("an interface requires two distinct phase identities")
    if area_m2 is not None and (isinstance(area_m2, bool) or not isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("interface area must be explicitly finite and positive")
    states = []
    for key in (left_phase_id, right_phase_id):
        if key not in ledger.phases:
            raise ValueError("interface phase is absent from the native ledger")
        phase = ledger.phases[key]
        if phase.vessel_id != vessel_id or phase.phase_id != key:
            raise ValueError("interface cannot cross vessel or phase ownership")
        if (phase.solvents.volume_L > 0 or set(phase.species_amounts_mol)-set(model.component_ids)
                or any(not isfinite(v) or v < 0 for v in phase.species_amounts_mol.values())
                or not isfinite(phase.volume_L) or phase.volume_L <= 0):
            raise ValueError("interface requires declared finite species and positive phase volume")
        amounts = tuple(float(phase.species_amounts_mol.get(k, 0)) for k in model.component_ids)
        total = fsum(amounts)
        if total <= 0:
            raise ValueError("interface cannot attach to an empty phase")
        states.append(molecular_eos_state(model, temperature_K, total/(phase.volume_L*1e-3),
                                          tuple(v/total for v in amounts)))
    if mesh is None:
        if interior_density_grid is None:
            raise ValueError("pure interface requires an explicit interior density grid")
        result = pure_square_gradient_interface(model, states[0], states[1], influence,
                                                 interior_density_grid=interior_density_grid)
    else:
        if component_density_bounds is None:
            raise ValueError("free interface profile requires explicit component density bounds")
        result = molecular_planar_interface(model, states[0], states[1], influence, mesh=mesh,
            component_density_bounds=component_density_bounds, initial_profile=initial_profile,
            max_iterations=max_iterations)
    excess = (area_m2*result.surface_tension_N_m
              if area_m2 is not None and result.success and result.surface_tension_N_m is not None
              else None)
    return MolecularInventoryInterface(left_phase_id, right_phase_id, result, area_m2, excess)
