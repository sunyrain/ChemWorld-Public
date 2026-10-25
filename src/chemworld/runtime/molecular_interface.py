"""Read planar interface endpoints from the sole native phase inventory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from typing import Any, cast

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel, molecular_eos_state
from chemworld.physchem.molecular_interface import (
    MolecularInfluence,
    MolecularInterfaceResult,
    molecular_planar_interface,
    pure_square_gradient_interface,
)
from chemworld.runtime.molecular_multiphase_inventory import MolecularMultiphaseInventorySpec
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseEquilibriumSpec, _amounts


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
    ledger: PhaseLedger, model: ResidualHelmholtzModel, influence: MolecularInfluence, *,
    vessel_id: str, left_phase_id: str, right_phase_id: str, temperature_K: float,
    area_m2: float | None = None, interior_density_grid: Sequence[float] | None = None,
    mesh: DiffusionMesh | None = None,
    component_density_bounds: Sequence[tuple[float, float]] | None = None,
    initial_profile: Sequence[Sequence[float]] | None = None, max_iterations: int = 500,
    quadrature_tolerance: float = 1e-9, residual_tolerance: float = 1e-7,
    first_integral_tolerance: float = .05, pressure_Pa: float | None = None,
) -> MolecularInventoryInterface:
    """Rebuild both same-temperature endpoint states from their real n/V.

    An explicit area returns gamma*A, an excess grand potential. It is not a
    heat duty or an internal energy and is never written to a thermal ledger.
    """
    if left_phase_id == right_phase_id:
        raise ValueError("an interface requires two distinct phase identities")
    if area_m2 is not None and (isinstance(area_m2, bool) or not isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("interface area must be explicitly finite and positive")
    if pressure_Pa is not None and (
        isinstance(pressure_Pa, bool) or not isfinite(pressure_Pa) or pressure_Pa <= 0
    ):
        raise ValueError("interface vessel pressure must be finite and positive")
    states = []
    for key in (left_phase_id, right_phase_id):
        if key not in ledger.phases:
            raise ValueError("interface phase is absent from the native ledger")
        phase = ledger.phases[key]
        if phase.vessel_id != vessel_id or phase.phase_id != key:
            raise ValueError("interface cannot cross vessel or phase ownership")
        if (isinstance(phase.volume_L, bool) or not isfinite(phase.volume_L)
                or phase.volume_L <= 0):
            raise ValueError("interface requires positive finite phase volume")
        inventory = _amounts(phase, model.component_ids)
        amounts = tuple(inventory[k] for k in model.component_ids)
        total = fsum(amounts)
        if total <= 0:
            raise ValueError("interface cannot attach to an empty phase")
        states.append(molecular_eos_state(model, temperature_K, total/(phase.volume_L*1e-3),
                                          tuple(v/total for v in amounts)))
        if pressure_Pa is not None and not isclose(
            states[-1].pressure_Pa, pressure_Pa, rel_tol=2e-7, abs_tol=1e-6
        ):
            raise ValueError("interface phase pressure disagrees with the current vessel")
    if mesh is None:
        if interior_density_grid is None:
            raise ValueError("pure interface requires an explicit interior density grid")
        result = pure_square_gradient_interface(model, states[0], states[1], influence,
            interior_density_grid=interior_density_grid, quadrature_tolerance=quadrature_tolerance)
    else:
        if component_density_bounds is None:
            raise ValueError("free interface profile requires explicit component density bounds")
        result = molecular_planar_interface(model, states[0], states[1], influence, mesh=mesh,
            component_density_bounds=component_density_bounds, initial_profile=initial_profile,
            max_iterations=max_iterations, residual_tolerance=residual_tolerance,
            first_integral_tolerance=first_integral_tolerance)
    excess = (area_m2*result.surface_tension_N_m
              if area_m2 is not None and result.success and result.surface_tension_N_m is not None
              else None)
    return MolecularInventoryInterface(left_phase_id, right_phase_id, result, area_m2, excess)


def molecular_interface_from_state(
    state: WorldState,
    specification: MolecularPhaseEquilibriumSpec | MolecularMultiphaseInventorySpec,
    influence: MolecularInfluence, *, left_phase_id: str, right_phase_id: str,
    area_m2: float | None = None, interior_density_grid: Sequence[float] | None = None,
    mesh: DiffusionMesh | None = None,
    component_density_bounds: Sequence[tuple[float, float]] | None = None,
    initial_profile: Sequence[Sequence[float]] | None = None, max_iterations: int = 500,
    quadrature_tolerance: float = 1e-9, residual_tolerance: float = 1e-7,
    first_integral_tolerance: float = .05,
) -> MolecularInventoryInterface:
    """Query a live vessel using its existing phase model, temperature and pressure.

    The two selected phases must still exist, match their declared identities,
    and occupy their declared density basins. No flash is performed to repair a
    stale or nonequilibrium inventory. A query leaves all world ledgers intact;
    its optional area product is an excess grand potential, never heat or U.
    """
    declaration = (specification.phase_declaration
                   if isinstance(specification, MolecularMultiphaseInventorySpec)
                   else specification)
    ledger = cast(PhaseLedger, state.phases)
    targets = {target.phase_id: target for target in declaration.phase_targets}
    basins = {basin.name: basin for basin in declaration.basins}
    for phase_id in (left_phase_id, right_phase_id):
        if phase_id not in targets or phase_id not in ledger.phases:
            raise ValueError("interface requires present, explicitly declared phase identities")
        record, target = ledger.phases[phase_id], targets[phase_id]
        if record.phase_type != target.phase_type:
            raise ValueError("interface phase type disagrees with the phase declaration")
        amounts = _amounts(record, declaration.residual_model.component_ids)
        if not isfinite(record.volume_L) or record.volume_L <= 0:
            raise ValueError("interface requires occupied phases with positive volume")
        rho = fsum(amounts.values()) / (record.volume_L * 1e-3)
        lo, hi = basins[target.basin].density_bounds_mol_m3
        if not lo <= rho <= hi:
            raise ValueError("interface phase density leaves its declared basin")
    return molecular_interface_from_ledger(
        ledger, declaration.residual_model, influence, vessel_id=state.vessel_id,
        left_phase_id=left_phase_id, right_phase_id=right_phase_id,
        temperature_K=state.temperature_K, pressure_Pa=state.pressure_Pa,
        area_m2=area_m2, interior_density_grid=interior_density_grid, mesh=mesh,
        component_density_bounds=component_density_bounds, initial_profile=initial_profile,
        max_iterations=max_iterations, quadrature_tolerance=quadrature_tolerance,
        residual_tolerance=residual_tolerance, first_integral_tolerance=first_integral_tolerance,
    )
