"""Molecular TP closure on the existing PhaseLedger and MolecularStream.

The ledger remains the sole material owner. A failed closure never installs a
candidate phase, and caloric values exist only with an explicit common reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite
from typing import TYPE_CHECKING, Any

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel
from chemworld.physchem.molecular_flash import (
    MolecularDensityBasin,
    MolecularTPFlashResult,
    molecular_tp_flash,
)
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream

if TYPE_CHECKING:
    from chemworld.physchem.molecular_multiphase import MolecularMultiphaseResult


@dataclass(frozen=True)
class MolecularPhaseTarget:
    phase_id: str
    basin: str
    phase_type: str
    selected: bool = False

    def __post_init__(self) -> None:
        if (any(not isinstance(v, str) or not v for v in
                (self.phase_id, self.basin, self.phase_type)) or type(self.selected) is not bool):
            raise ValueError("molecular phase target requires explicit identity, basin and type")


@dataclass(frozen=True)
class MolecularPhaseEquilibriumSpec:
    residual_model: ResidualHelmholtzModel
    basins: tuple[MolecularDensityBasin, ...]
    phase_targets: tuple[MolecularPhaseTarget, ...]
    caloric_model: MolecularCaloricModel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "basins", tuple(self.basins))
        object.__setattr__(self, "phase_targets", tuple(self.phase_targets))
        names = {b.name for b in self.basins}
        if (not names or len(names) != len(self.basins) or not self.phase_targets
                or len({t.phase_id for t in self.phase_targets}) != len(self.phase_targets)
                or any(t.basin not in names for t in self.phase_targets)
                or names != {t.basin for t in self.phase_targets}):
            raise ValueError("phase targets require unique IDs and cover all declared basins")
        if (self.caloric_model is not None
                and self.caloric_model.residual_model != self.residual_model):
            raise ValueError("caloric and phase closure must use the same residual model")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-molecular-phase-equilibrium-v1",
                "residual_model": molecular_model_to_dict(self.residual_model),
                "basins": [asdict(b) for b in self.basins],
                "phase_targets": [asdict(t) for t in self.phase_targets],
                # Reuse the ideal reference encoding without storing a second residual model.
                "ideal_references": None if self.caloric_model is None else
                    [r.to_dict() for r in self.caloric_model.ideal_references]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularPhaseEquilibriumSpec:
        from chemworld.physchem.helmholtz import HelmholtzFluid

        if data.get("model") != "chemworld-molecular-phase-equilibrium-v1":
            raise ValueError("unknown molecular phase-equilibrium declaration")
        residual = molecular_model_from_dict(data["residual_model"])
        ideal = data.get("ideal_references")
        return cls(residual, tuple(MolecularDensityBasin(**b) for b in data["basins"]),
                   tuple(MolecularPhaseTarget(**p) for p in data["phase_targets"]),
                   None if ideal is None else MolecularCaloricModel(
                       residual, tuple(HelmholtzFluid.from_dict(r) for r in ideal)))


@dataclass(frozen=True)
class MolecularPhaseInventoryResult:
    success: bool
    ledger: PhaseLedger
    flash: MolecularTPFlashResult | MolecularMultiphaseResult
    installed_phase_ids: tuple[str, ...] = ()
    phase_enthalpies_J: dict[str, float] | None = None
    total_enthalpy_J: float | None = None
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "ledger": self.ledger.to_dict(),
                "flash": self.flash.to_dict(), "installed_phase_ids": self.installed_phase_ids,
                "phase_enthalpies_J": self.phase_enthalpies_J,
                "total_enthalpy_J": self.total_enthalpy_J, "failure_reason": self.failure_reason,
                "energy_scope": "selected_phase_enthalpy_only; no heat-duty or time advance"}


def _amounts(record: PhaseRecord, component_ids: tuple[str, ...]) -> dict[str, float]:
    if (set(record.species_amounts_mol)-set(component_ids) or any(
        isinstance(n, bool) or not isfinite(n) or n < 0
        for n in record.species_amounts_mol.values()
    )):
        raise ValueError("selected phase contains undeclared or invalid molecular inventory")
    if record.solvents.volume_L > 0:
        raise ValueError("molecular phase closure cannot discard a legacy solvent inventory")
    return {key: float(record.species_amounts_mol.get(key, 0)) for key in component_ids}


def _molecular_inventory_feed(
    ledger: PhaseLedger, specification: MolecularPhaseEquilibriumSpec, *, vessel_id: str,
    source_phase_ids: Sequence[str],
) -> dict[str, float]:
    """Read real source stock and validate the common phase ownership boundary."""
    sources = tuple(source_phase_ids)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("molecular closure needs distinct explicit source phase IDs")
    if not vessel_id or any(k not in ledger.phases for k in sources):
        raise ValueError("molecular closure source phase is missing")
    records = [ledger.phases[k] for k in sources]
    if any(p.vessel_id != vessel_id or p.phase_id != key
           for key, p in zip(sources, records, strict=True)):
        raise ValueError("molecular phase closure cannot cross vessel ownership")
    if any(t.phase_id in ledger.phases and t.phase_id not in sources
           for t in specification.phase_targets):
        raise ValueError("molecular phase target would overwrite an unrelated phase")
    inventories = [_amounts(p, specification.residual_model.component_ids) for p in records]
    return {k: fsum(ns[k] for ns in inventories)
            for k in specification.residual_model.component_ids}


def _install_molecular_phase_result(
    ledger: PhaseLedger, specification: MolecularPhaseEquilibriumSpec,
    flash: MolecularTPFlashResult | MolecularMultiphaseResult, *, vessel_id: str,
    source_phase_ids: Sequence[str], temperature_K: float, pressure_Pa: float,
) -> MolecularPhaseInventoryResult:
    """The common material/caloric commit boundary for accepted equilibrium results."""
    sources = tuple(source_phase_ids)
    if not flash.success:
        return MolecularPhaseInventoryResult(False, ledger, flash,
                                               failure_reason=flash.failure_reason)
    available = {name: [t for t in specification.phase_targets if t.basin == name]
                 for name in {b.name for b in specification.basins}}
    installed: dict[str, PhaseRecord] = {}
    enthalpies: dict[str, float] | None = {} if specification.caloric_model is not None else None
    try:
        for phase in flash.phases:
            if not available[phase.basin]:
                return MolecularPhaseInventoryResult(False, ledger, flash,
                    failure_reason="insufficient_distinct_phase_targets_for_basin")
            target = available[phase.basin].pop(0)
            if specification.caloric_model is not None:
                caloric = specification.caloric_model.state(temperature_K,
                    phase.state.molar_density_mol_m3, phase.state.mole_fractions)
                assert enthalpies is not None
                enthalpies[target.phase_id] = phase.amount_mol*caloric.enthalpy_J_mol
            previous = ledger.phases.get(target.phase_id)
            installed[target.phase_id] = PhaseRecord(target.phase_id, vessel_id, target.phase_type,
                phase.volume_m3*1000, phase.component_amounts_mol, selected=target.selected,
                metadata={**(previous.metadata if previous is not None else {}),
                    "carrier_representation": "species_inventory",
                    "molecular_density_basin": phase.basin,
                    "molecular_tp_conditions": {"temperature_K": temperature_K,
                                                "pressure_Pa": pressure_Pa}})
    except (ValueError, ArithmeticError) as exc:
        return MolecularPhaseInventoryResult(False, ledger, flash,
                                               failure_reason="caloric_closure_failed: "+str(exc))
    candidate = PhaseLedger({**{key: p for key, p in ledger.phases.items() if key not in sources},
                              **installed})
    before, after = ledger.total_amounts_mol(), candidate.total_amounts_mol()
    if any(not isclose(before.get(k, 0), after.get(k, 0), rel_tol=5e-13, abs_tol=0)
           for k in set(before) | set(after)):
        return MolecularPhaseInventoryResult(False, ledger, flash,
                                               failure_reason="phase_ledger_material_mismatch")
    return MolecularPhaseInventoryResult(True, candidate, flash, tuple(installed), enthalpies,
                                          None if enthalpies is None else fsum(enthalpies.values()))


def equilibrate_molecular_phase_inventory(
    ledger: PhaseLedger, specification: MolecularPhaseEquilibriumSpec, *, vessel_id: str,
    source_phase_ids: Sequence[str], temperature_K: float, pressure_Pa: float,
    composition_starts: Sequence[Sequence[float]] | None = None,
    max_iterations: int = 200, tpd_tolerance: float = 1e-8,
) -> MolecularPhaseInventoryResult:
    """Replace only explicitly selected same-vessel phases after successful TP closure.

    Multiple targets may name one basin: their declaration order receives the
    returned phases in composition order, so an LLE split has distinct phase IDs.
    This is an isothermal/isobaric material closure, not a fixed-volume evolution
    or a heat-duty calculation. Unrelated phases retain their content and owners.
    """
    sources = tuple(source_phase_ids)
    feed = _molecular_inventory_feed(ledger, specification, vessel_id=vessel_id,
                                     source_phase_ids=sources)
    flash = molecular_tp_flash(specification.residual_model, temperature_K, pressure_Pa, feed,
        basins=specification.basins, composition_starts=composition_starts,
        max_iterations=max_iterations, tpd_tolerance=tpd_tolerance)
    return _install_molecular_phase_result(ledger, specification, flash, vessel_id=vessel_id,
        source_phase_ids=sources, temperature_K=temperature_K, pressure_Pa=pressure_Pa)


def molecular_phase_stream(
    ledger: PhaseLedger, specification: MolecularPhaseEquilibriumSpec, *, vessel_id: str,
    temperature_K: float, molar_rate_mol_s: float, phase_id: str | None = None,
) -> MolecularStream:
    """Build an existing bulk stream from the actual selected ledger phase.

    Recompute composition and density from current amounts and volume. Never use
    a cached flash inventory, or assign heat from an absent caloric reference.
    Rate is an instantaneous stream request; finite transfer/time remains with
    the existing material-port/integrator consumer.
    """
    if specification.caloric_model is None:
        raise ValueError("molecular stream requires explicit common ideal caloric references")
    declared = {t.phase_id for t in specification.phase_targets}
    if phase_id is None:
        selected = [p.phase_id for p in ledger.phases.values()
                    if p.vessel_id == vessel_id and p.selected and p.phase_id in declared]
        if len(selected) != 1:
            raise ValueError("molecular stream requires one selected declared phase or a phase ID")
        phase_id = selected[0]
    if phase_id not in ledger.phases or phase_id not in declared:
        raise ValueError("molecular stream phase is missing or undeclared")
    phase = ledger.phases[phase_id]
    if phase.vessel_id != vessel_id:
        raise ValueError("molecular stream cannot cross phase ownership")
    target = next(t for t in specification.phase_targets if t.phase_id == phase_id)
    if phase.phase_type != target.phase_type:
        raise ValueError("molecular stream phase type disagrees with its declaration")
    amounts = _amounts(phase, specification.residual_model.component_ids)
    total = fsum(amounts.values())
    if not isfinite(phase.volume_L) or phase.volume_L <= 0 or total <= 0:
        raise ValueError("molecular stream phase requires finite positive amount and volume")
    density = total/(phase.volume_L*1e-3)
    basin = next(b for b in specification.basins if b.name == target.basin)
    if not basin.density_bounds_mol_m3[0] <= density <= basin.density_bounds_mol_m3[1]:
        raise ValueError("molecular stream density is outside its declared phase basin")
    state = specification.caloric_model.state(temperature_K, density,
        tuple(amounts[k]/total for k in specification.residual_model.component_ids))
    return MolecularStream(state, molar_rate_mol_s)
