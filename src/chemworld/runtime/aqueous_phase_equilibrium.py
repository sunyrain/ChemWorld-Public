"""Commit-ready aqueous TP candidates using the existing owned phase ledger."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, isclose
from typing import Any

from chemworld.foundation.state_ledgers import PhaseLedger, VesselRecord
from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumResult,
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)


@dataclass(frozen=True)
class AqueousPhaseInventoryResult:
    success: bool
    ledger: PhaseLedger
    equilibrium: AqueousEquilibriumResult
    initial_properties: AqueousPhaseProperties
    final_properties: AqueousPhaseProperties | None = None
    heat_duty_J: float | None = None
    boundary_work_J: float | None = None
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "ledger": self.ledger.to_dict(),
            "equilibrium": self.equilibrium.to_dict(),
            "initial_properties": self.initial_properties.to_dict(),
            "final_properties": self.final_properties.to_dict() if self.final_properties else None,
            "heat_duty_J": self.heat_duty_J,
            "boundary_work_J": self.boundary_work_J,
            "failure_reason": self.failure_reason,
            "scope": "closed_single_aqueous_phase_TP; no elapsed_time_or_heat_rate",
        }


def equilibrate_aqueous_phase_inventory(
    ledger: PhaseLedger,
    specification: AqueousEquilibriumSpec,
    phase_id: str,
    *,
    vessel: VesselRecord,
    temperature_K: float,
    pressure_Pa: float,
    initial_guess_mol: Mapping[str, float] | None = None,
) -> AqueousPhaseInventoryResult:
    """Close one liquid phase at declared TP, preserving all other phase owners.

    TP is an external equilibrium boundary, not a fixed-volume trajectory.
    Q=delta H and work done by the material=P*delta V follow the same Gibbs
    model as the equilibrium. This function returns a candidate only: a session
    must perform its usual transaction and resource accounting to install it.
    """
    phase = ledger.phases.get(phase_id)
    if phase is None or phase.phase_id != phase_id or phase.vessel_id != vessel.vessel_id:
        raise ValueError("aqueous equilibrium requires an existing phase in the declared vessel")
    if phase.phase_type not in {"liquid", "aqueous"} or phase.solvents.volume_L > 0:
        raise ValueError("aqueous equilibrium requires explicit liquid species inventory")
    if temperature_K > vessel.max_temperature_K or pressure_Pa > vessel.max_pressure_Pa:
        raise ValueError("aqueous TP boundary exceeds declared vessel limits")
    if (not isclose(temperature_K, vessel.temperature_K, rel_tol=0, abs_tol=1e-8)
            or not isclose(pressure_Pa, vessel.pressure_Pa, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("aqueous closure requires the vessel's current temperature and pressure")
    before = aqueous_phase_properties(
        specification.database, phase.species_amounts_mol,
        temperature_K=temperature_K, pressure_Pa=pressure_Pa,
    )
    if not isclose(phase.volume_L, before.volume_m3 * 1000, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError("initial aqueous phase volume disagrees with its declared Gibbs model")
    other_volume = fsum(
        p.volume_L for key, p in ledger.phases.items()
        if key != phase_id and p.vessel_id == vessel.vessel_id
    )
    if other_volume + phase.volume_L > vessel.max_volume_L:
        raise ValueError("initial aqueous inventory exceeds vessel capacity")
    result = solve_aqueous_reaction_equilibrium(
        specification, phase.species_amounts_mol, temperature_K=temperature_K,
        pressure_Pa=pressure_Pa, initial_guess_mol=initial_guess_mol,
    )
    if not result.success:
        return AqueousPhaseInventoryResult(
            False, ledger, result, before, failure_reason=result.failure_reason,
        )
    if not (result.amounts_mol is not None):
        raise ValueError("successful aqueous equilibrium requires amounts")
    try:
        after = aqueous_phase_properties(
            specification.database, result.amounts_mol,
            temperature_K=temperature_K, pressure_Pa=pressure_Pa,
        )
        if other_volume + after.volume_m3 * 1000 > vessel.max_volume_L:
            raise ValueError("equilibrium aqueous inventory exceeds vessel capacity")
    except (ValueError, ArithmeticError) as exc:
        return AqueousPhaseInventoryResult(
            False, ledger, result, before, failure_reason=str(exc),
        )
    candidate = replace(
        phase, species_amounts_mol=result.amounts_mol, volume_L=after.volume_m3 * 1000,
        metadata={**phase.metadata, "carrier_representation": "species_inventory",
                  "aqueous_equilibrium_database": specification.database.database_id,
                  "aqueous_tp_conditions": {"temperature_K": temperature_K,
                                            "pressure_Pa": pressure_Pa}},
    )
    return AqueousPhaseInventoryResult(
        True, PhaseLedger({**ledger.phases, phase_id: candidate}), result, before, after,
        after.enthalpy_J - before.enthalpy_J,
        pressure_Pa * (after.volume_m3 - before.volume_m3),
    )
