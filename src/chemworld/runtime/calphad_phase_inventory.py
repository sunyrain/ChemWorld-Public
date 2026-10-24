"""Zero-time CALPHAD TP material closure on the original PhaseLedger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite
from typing import Any, cast

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumPhase,
    CalphadEquilibriumResult,
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)


@dataclass(frozen=True)
class CalphadPhaseTarget:
    phase_id: str
    phase_name: str
    molar_volume_L_mol_atoms: float
    phase_type: str = "solid"
    selected: bool = False

    def __post_init__(self) -> None:
        if (any(not isinstance(s, str) or not s for s in
                (self.phase_id, self.phase_name, self.phase_type))
                or isinstance(self.molar_volume_L_mol_atoms, bool)
                or not isfinite(self.molar_volume_L_mol_atoms)
                or self.molar_volume_L_mol_atoms <= 0 or type(self.selected) is not bool
                or self.phase_type not in {"solid", "liquid"}):
            raise ValueError("CALPHAD target requires identity, type and positive authored volume")


@dataclass(frozen=True)
class CalphadPhaseInventoryResult:
    success: bool
    ledger: PhaseLedger
    equilibrium: CalphadEquilibriumResult
    installed_phase_ids: tuple[str, ...] = ()
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "ledger": self.ledger.to_dict(),
                "equilibrium": self.equilibrium.to_dict(),
                "installed_phase_ids": self.installed_phase_ids,
                "failure_reason": self.failure_reason,
                "volume_scope": "authored_positive_phase_volume; vessel_capacity_checked_by_caller",
                "energy_scope": "TP_material_boundary; no_heat_duty_or_time_advance"}


def equilibrate_calphad_phase_inventory(
    ledger: PhaseLedger, specification: CalphadEquilibriumSpec, *, vessel_id: str,
    source_phase_ids: Sequence[str], phase_targets: Sequence[CalphadPhaseTarget],
    temperature_K: float, pressure_Pa: float,
    initial_phases: Sequence[CalphadEquilibriumPhase] | None = None,
) -> CalphadPhaseInventoryResult:
    """Close only selected actual atomic stocks; failed results retain the ledger.

    Multiple explicit target slots may name one phase model. Constitutions are
    assigned deterministically in composition order. Volumes are author inputs,
    not derived from pressure-independent Gibbs terms. Legacy solvent stock is
    rejected instead of discarded. No persisted second material owner is made.
    """
    sources, targets = tuple(source_phase_ids), tuple(phase_targets)
    if (not vessel_id or not sources or len(set(sources)) != len(sources)
            or any(key not in ledger.phases for key in sources)):
        raise ValueError("CALPHAD closure requires existing distinct explicit source phases")
    names = {m.phase_name for m in specification.models}
    if (not targets or len({t.phase_id for t in targets}) != len(targets)
            or {t.phase_name for t in targets} != names):
        raise ValueError("CALPHAD targets require unique identities and cover all allowed phases")
    if any(t.phase_id in ledger.phases and t.phase_id not in sources for t in targets):
        raise ValueError("CALPHAD target would overwrite an unrelated phase")
    records = [ledger.phases[key] for key in sources]
    if any(p.vessel_id != vessel_id or p.phase_id != key
           for key, p in zip(sources, records, strict=True)):
        raise ValueError("CALPHAD closure cannot cross vessel ownership")
    declared = cast(Mapping[str, Any], specification.models[0].database.elements)
    elements = set(declared)-{"VA", "/-"}
    for p in records:
        if (set(p.species_amounts_mol)-elements or p.solvents.volume_L > 0
                or any(isinstance(n, bool) or not isfinite(n) or n < 0
                       for n in p.species_amounts_mol.values())):
            raise ValueError("CALPHAD closure requires atomic stock without legacy solvents")
    feed = {e: fsum(p.species_amounts_mol.get(e, 0.) for p in records) for e in elements}
    equilibrium = calphad_tp_equilibrium(
        specification, temperature_K, pressure_Pa, feed, initial_phases=initial_phases)

    def failure(reason: str) -> CalphadPhaseInventoryResult:
        return CalphadPhaseInventoryResult(False, ledger, equilibrium, failure_reason=reason)

    if not equilibrium.success:
        return failure(equilibrium.failure_reason)
    available = {name: [t for t in targets if t.phase_name == name] for name in names}
    installed = {}
    for phase in equilibrium.phases:
        if not available[phase.phase_name]:
            return failure("insufficient_distinct_phase_targets_for_model")
        target = available[phase.phase_name].pop(0)
        volume = phase.amount_mol_atoms*target.molar_volume_L_mol_atoms
        if not isfinite(volume) or volume <= 0:
            return failure("invalid_installed_phase_volume")
        installed[target.phase_id] = PhaseRecord(
            target.phase_id, vessel_id, target.phase_type, volume,
            phase.element_amounts_mol, selected=target.selected,
            metadata={"carrier_representation": "species_inventory",
                "tdb_phase": phase.phase_name, "site_fractions": phase.site_fractions,
                "molar_basis": "nonvacant_atoms", "calphad_phase_target": asdict(target),
                "calphad_tp_conditions": {"temperature_K": temperature_K,
                                           "pressure_Pa": pressure_Pa}})
    candidate = PhaseLedger({**{k: p for k, p in ledger.phases.items() if k not in sources},
                             **installed})
    before, after = ledger.total_amounts_mol(), candidate.total_amounts_mol()
    if any(not isclose(before.get(e, 0.), after.get(e, 0.), rel_tol=1e-9, abs_tol=0.)
           for e in set(before) | set(after)):
        return failure("phase_ledger_element_mismatch")
    return CalphadPhaseInventoryResult(True, candidate, equilibrium, tuple(installed))
