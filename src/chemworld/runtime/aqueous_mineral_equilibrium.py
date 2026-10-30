"""Atomic mineral TP candidates over original finite PhaseLedger owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, isclose
from typing import Any, cast

from chemworld.foundation.state_ledgers import PhaseLedger, VesselRecord
from chemworld.physchem.aqueous_mineral_equilibrium import (
    AqueousMineralEquilibriumResult,
    AqueousMineralEquilibriumSpec,
    aqueous_mineral_properties,
    solve_aqueous_mineral_equilibrium,
)
from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)


@dataclass(frozen=True)
class AqueousMineralInventoryResult:
    success: bool
    ledger: PhaseLedger
    equilibrium: AqueousMineralEquilibriumResult
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
            "scope": "closed_TP_candidate; no_clock_or_resource_commit",
        }


def equilibrate_aqueous_mineral_inventory(
    ledger: PhaseLedger,
    specification: AqueousMineralEquilibriumSpec,
    liquid_phase_id: str,
    mineral_phase_ids: Mapping[str, str],
    *,
    vessel: VesselRecord,
    temperature_K: float,
    pressure_Pa: float,
) -> AqueousMineralInventoryResult:
    """Return finite inventory/Q/PdV for the original transaction, without advancing time.

    Every mineral has an explicitly authored original solid-phase identity, including
    an initially absent phase with zero amount/volume. No imaginary seed is installed.
    Heat is delta H and boundary work is P delta V, both positive into/away from the
    material respectively. The consumer must settle each utility exactly once.
    """
    if (
        set(mineral_phase_ids) != set(specification.mineral_ids)
        or len(set(mineral_phase_ids.values())) != len(mineral_phase_ids)
        or liquid_phase_id in mineral_phase_ids.values()
    ):
        raise ValueError("each mineral and the liquid require distinct original phase identities")
    selected = {liquid_phase_id, *mineral_phase_ids.values()}
    if any(
        key not in ledger.phases
        or ledger.phases[key].phase_id != key
        or ledger.phases[key].vessel_id != vessel.vessel_id
        for key in selected
    ):
        raise ValueError(
            "mineral equilibrium requires original phases owned by the declared vessel"
        )
    if temperature_K > vessel.max_temperature_K or pressure_Pa > vessel.max_pressure_Pa:
        raise ValueError("mineral TP boundary exceeds original hardware limits")
    if not isclose(temperature_K, vessel.temperature_K, abs_tol=1e-8, rel_tol=0) or not isclose(
        pressure_Pa, vessel.pressure_Pa, abs_tol=1e-6, rel_tol=0
    ):
        raise ValueError("mineral closure requires the vessel's current temperature and pressure")
    liquid = ledger.phases[liquid_phase_id]
    liquid_ids = {s.species_id for s in specification.database.species if s.phase != "mineral"}
    if (
        liquid.phase_type not in {"aqueous", "liquid"}
        or liquid.solvents.volume_L > 0
        or set(liquid.species_amounts_mol) - liquid_ids
    ):
        raise ValueError("liquid requires explicit finite declared water/solute inventory")
    amounts = dict(liquid.species_amounts_mol)
    mineral_volumes = {
        s.species_id: cast(float, s.standard_state.molar_volume_m3_mol)
        for s in specification.database.species
        if s.phase == "mineral"
    }
    for mineral, key in mineral_phase_ids.items():
        phase = ledger.phases[key]
        if (
            phase.phase_type != "solid"
            or set(phase.species_amounts_mol) != {mineral}
            or phase.solvents.volume_L > 0
        ):
            raise ValueError(
                "a pure-mineral phase must retain exactly its original mineral species"
            )
        amount = phase.species_amounts_mol[mineral]
        if not isclose(
            phase.volume_L, amount * mineral_volumes[mineral] * 1000, rel_tol=1e-8, abs_tol=1e-12
        ):
            raise ValueError("original mineral volume disagrees with its standard state")
        amounts[mineral] = amount
    before = aqueous_mineral_properties(
        specification, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    liquid_before = aqueous_phase_properties(
        specification.database,
        liquid.species_amounts_mol,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    if not isclose(liquid.volume_L, liquid_before.volume_m3 * 1000, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError("original liquid volume disagrees with its Gibbs model")
    other_volume = fsum(
        p.volume_L
        for key, p in ledger.phases.items()
        if key not in selected and p.vessel_id == vessel.vessel_id
    )
    if before.volume_m3 * 1000 + other_volume > vessel.max_volume_L:
        raise ValueError("original inventory exceeds vessel capacity")
    result = solve_aqueous_mineral_equilibrium(
        specification, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    if not result.success:
        return AqueousMineralInventoryResult(
            False, ledger, result, before, failure_reason=result.failure_reason
        )
    if not (result.amounts_mol is not None and result.properties is not None):
        raise ValueError("successful mineral equilibrium requires amounts and properties")
    after = result.properties
    if after.volume_m3 * 1000 + other_volume > vessel.max_volume_L:
        return AqueousMineralInventoryResult(
            False, ledger, result, before, failure_reason="candidate exceeds vessel capacity"
        )
    liquid_amounts = {k: n for k, n in result.amounts_mol.items() if k in liquid_ids}
    liquid_after = aqueous_phase_properties(
        specification.database, liquid_amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    updates = {
        liquid_phase_id: replace(
            liquid,
            species_amounts_mol=liquid_amounts,
            volume_L=liquid_after.volume_m3 * 1000,
            metadata={**liquid.metadata, "carrier_representation": "species_inventory"},
        )
    }
    for mineral, key in mineral_phase_ids.items():
        n = result.amounts_mol[mineral]
        updates[key] = replace(
            ledger.phases[key],
            species_amounts_mol={mineral: n},
            volume_L=n * mineral_volumes[mineral] * 1000,
        )
    return AqueousMineralInventoryResult(
        True,
        PhaseLedger({**ledger.phases, **updates}),
        result,
        before,
        after,
        after.enthalpy_J - before.enthalpy_J,
        pressure_Pa * (after.volume_m3 - before.volume_m3),
    )
