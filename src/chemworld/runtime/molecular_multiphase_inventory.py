"""General molecular state pairs consuming the original phase ledger.

The declaration contains one caloric/EOS model. The live PhaseLedger remains
the only material source; solver seeds and diagnostic phases are never stock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite
from typing import Any

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.molecular_flash import MolecularFlashPhase
from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    MolecularMultiphaseSpec,
    solve_molecular_multiphase,
)
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseInventoryResult,
    MolecularPhaseTarget,
    _amounts,
    _install_molecular_phase_result,
    _molecular_inventory_feed,
    molecular_phase_stream,
)


@dataclass(frozen=True)
class MolecularMultiphaseInventorySpec:
    equilibrium: MolecularMultiphaseSpec
    phase_targets: tuple[MolecularPhaseTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_targets", tuple(self.phase_targets))
        # Reuse the original target/basin/identity contract, including support
        # for distinct liquid phases sharing one constitutive density basin.
        _ = self.phase_declaration
        kinds = self.equilibrium.phase_kind_by_basin
        if any(t.basin in kinds and t.phase_type != kinds[t.basin] for t in self.phase_targets):
            raise ValueError("inventory phase types disagree with the equilibrium declaration")

    @property
    def phase_declaration(self) -> MolecularPhaseEquilibriumSpec:
        """A view over the same model for existing installation and live streams."""
        return MolecularPhaseEquilibriumSpec(
            self.equilibrium.model.residual_model,
            self.equilibrium.basins,
            self.phase_targets,
            self.equilibrium.model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_multiphase_inventory_v1",
            "equilibrium": self.equilibrium.to_dict(),
            "phase_targets": [asdict(target) for target in self.phase_targets],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularMultiphaseInventorySpec:
        if data.get("model_type") != "molecular_multiphase_inventory_v1":
            raise ValueError("unknown molecular multiphase inventory declaration")
        return cls(
            MolecularMultiphaseSpec.from_dict(data["equilibrium"]),
            tuple(MolecularPhaseTarget(**target) for target in data["phase_targets"]),
        )


def equilibrate_molecular_multiphase_inventory(
    ledger: PhaseLedger,
    specification: MolecularMultiphaseInventorySpec,
    *,
    vessel_id: str,
    source_phase_ids: Sequence[str],
    inputs: Mapping[str, float],
    seed: MolecularMultiphaseSeed | None = None,
    composition_starts: Sequence[Sequence[float]] | None = None,
) -> MolecularPhaseInventoryResult:
    """Solve a declared state pair and commit its actual phases atomically.

    H, S, U and V refer to the selected real material totals. No source phases
    are removed on an unresolved search, missing target or caloric failure.
    This closure does not itself charge heat/work or advance the shared clock.
    """
    sources = tuple(source_phase_ids)
    declaration = specification.phase_declaration
    feed = _molecular_inventory_feed(
        ledger, declaration, vessel_id=vessel_id, source_phase_ids=sources
    )
    result = solve_molecular_multiphase(
        specification.equilibrium, feed, inputs=inputs, seed=seed,
        composition_starts=composition_starts,
    )
    if not result.success:
        return MolecularPhaseInventoryResult(
            False, ledger, result, failure_reason=result.failure_reason
        )
    if not result.stability_checked:
        return MolecularPhaseInventoryResult(
            False, ledger, result, failure_reason="unqualified_multiphase_trial"
        )
    if result.temperature_K is None or result.pressure_Pa is None:
        raise ValueError("successful multiphase closure requires physical T and P")
    return _install_molecular_phase_result(
        ledger, declaration, result, vessel_id=vessel_id, source_phase_ids=sources,
        temperature_K=result.temperature_K, pressure_Pa=result.pressure_Pa,
    )


@dataclass(frozen=True)
class MolecularMultiphaseInventoryState:
    """Calorics read from current stock, without asserting phase equilibrium."""

    feed_amounts_mol: dict[str, float]
    seed: MolecularMultiphaseSeed
    enthalpy_J: float
    entropy_J_K: float
    internal_energy_J: float
    volume_m3: float


def molecular_multiphase_inventory_state(
    ledger: PhaseLedger,
    specification: MolecularMultiphaseInventorySpec,
    *,
    vessel_id: str,
    source_phase_ids: Sequence[str],
    temperature_K: float,
    pressure_Pa: float,
) -> MolecularMultiphaseInventoryState:
    """Read a warm seed and extensive calorics from the actual phase records.

    Missing/empty phases are absent, rather than invented at a positive floor.
    Every selected record must have a declared identity, type and density basin.
    Pressure/temperature compatibility is checked before constructing the seed.
    """
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0
           for v in (temperature_K, pressure_Pa)):
        raise ValueError("molecular inventory requires finite positive T and P")
    sources = tuple(source_phase_ids)
    feed = _molecular_inventory_feed(
        ledger, specification.phase_declaration, vessel_id=vessel_id, source_phase_ids=sources
    )
    targets = {t.phase_id: t for t in specification.phase_targets}
    basins = {b.name: b for b in specification.equilibrium.basins}
    model = specification.equilibrium.model
    phases, enthalpies, entropies, energies, volumes = [], [], [], [], []
    for phase_id in sources:
        if phase_id not in targets:
            raise ValueError("live molecular phase must have a declared target identity")
        record, target = ledger.phases[phase_id], targets[phase_id]
        if record.phase_type != target.phase_type:
            raise ValueError("live molecular phase type disagrees with its declaration")
        amounts = _amounts(record, model.component_ids)
        total = fsum(amounts.values())
        if total <= 0:
            if record.volume_L != 0:
                raise ValueError("empty molecular phase cannot retain finite material volume")
            continue
        if not isfinite(record.volume_L) or record.volume_L <= 0:
            raise ValueError("occupied molecular phase needs positive material volume")
        volume = record.volume_L * 1e-3
        rho = total / volume
        bounds = basins[target.basin].density_bounds_mol_m3
        if not bounds[0] <= rho <= bounds[1]:
            raise ValueError("live molecular phase density leaves its declared basin")
        state = model.state(
            temperature_K, rho, tuple(amounts[k] / total for k in model.component_ids)
        )
        if not isclose(state.pressure_Pa, pressure_Pa, rel_tol=2e-7, abs_tol=1e-6):
            raise ValueError("live molecular phase pressure disagrees with the vessel")
        phases.append(MolecularFlashPhase(target.basin, amounts, state))
        enthalpies.append(total * state.enthalpy_J_mol)
        entropies.append(total * state.entropy_J_mol_K)
        energies.append(total * state.internal_energy_J_mol)
        volumes.append(volume)
    if not phases:
        raise ValueError("molecular multiphase inventory cannot be empty")
    phases.sort(key=lambda phase: (phase.basin, phase.state.mole_fractions))
    return MolecularMultiphaseInventoryState(
        feed, MolecularMultiphaseSeed(temperature_K, pressure_Pa, tuple(phases)),
        fsum(enthalpies), fsum(entropies), fsum(energies), fsum(volumes),
    )


def molecular_multiphase_stream(
    ledger: PhaseLedger, specification: MolecularMultiphaseInventorySpec, *, vessel_id: str,
    temperature_K: float, molar_rate_mol_s: float, phase_id: str | None = None,
) -> MolecularStream:
    """Use the existing stream constructor against the current real inventory."""
    return molecular_phase_stream(
        ledger, specification.phase_declaration, vessel_id=vessel_id,
        temperature_K=temperature_K, molar_rate_mol_s=molar_rate_mol_s, phase_id=phase_id,
    )
