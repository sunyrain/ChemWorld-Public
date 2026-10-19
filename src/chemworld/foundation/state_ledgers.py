"""Typed ledger records used by ChemWorld hidden state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from chemworld.foundation.solvents import SolventInventory


@dataclass(frozen=True)
class SpeciesLedger:
    """Species definitions, roles, and initial amounts.

    Phase ledgers are the primary material-state source; global species totals
    are derived from phase contents.
    """

    species_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    initial_amounts_mol: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_roles", deepcopy(self.species_roles))
        object.__setattr__(
            self,
            "initial_amounts_mol",
            {
                str(species_id): max(float(amount_mol), 0.0)
                for species_id, amount_mol in self.initial_amounts_mol.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_roles": {
                species_id: list(roles) for species_id, roles in self.species_roles.items()
            },
            "initial_amounts_mol": deepcopy(self.initial_amounts_mol),
        }


@dataclass(frozen=True)
class PhaseRecord:
    phase_id: str
    vessel_id: str
    phase_type: str
    volume_L: float
    species_amounts_mol: dict[str, float]
    settled: bool = False
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    solvents: SolventInventory = field(default_factory=SolventInventory)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_amounts_mol", deepcopy(self.species_amounts_mol))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        if self.solvents.volume_L > 0 and abs(self.solvents.volume_L - self.volume_L) > 1e-14:
            object.__setattr__(self, "solvents", self.solvents.at_volume(self.volume_L))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "vessel_id": self.vessel_id,
            "phase_type": self.phase_type,
            "volume_L": self.volume_L,
            "species_amounts_mol": deepcopy(self.species_amounts_mol),
            "settled": self.settled,
            "selected": self.selected,
            "metadata": deepcopy(self.metadata),
            "solvents": self.solvents.to_dict(),
        }


@dataclass(frozen=True)
class PhaseLedger:
    phases: dict[str, PhaseRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", deepcopy(self.phases))

    def total_amounts_mol(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for phase in self.phases.values():
            for species_id, amount in phase.species_amounts_mol.items():
                totals[species_id] = totals.get(species_id, 0.0) + float(amount)
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {phase_id: phase.to_dict() for phase_id, phase in self.phases.items()}


@dataclass(frozen=True)
class VesselRecord:
    vessel_id: str
    vessel_type: str
    max_volume_L: float
    max_temperature_K: float
    max_pressure_Pa: float
    phase_ids: tuple[str, ...] = ()
    temperature_K: float = 298.15
    pressure_Pa: float = 101_325.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "vessel_type": self.vessel_type,
            "max_volume_L": self.max_volume_L,
            "max_temperature_K": self.max_temperature_K,
            "max_pressure_Pa": self.max_pressure_Pa,
            "phase_ids": list(self.phase_ids),
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
        }


@dataclass(frozen=True)
class VesselLedger:
    vessels: dict[str, VesselRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vessels", deepcopy(self.vessels))

    def to_dict(self) -> dict[str, Any]:
        return {vessel_id: vessel.to_dict() for vessel_id, vessel in self.vessels.items()}


@dataclass(frozen=True)
class EquipmentRecord:
    equipment_id: str
    equipment_type: str
    attached_vessel_id: str
    status: str = "idle"
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", deepcopy(self.settings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "attached_vessel_id": self.attached_vessel_id,
            "status": self.status,
            "settings": deepcopy(self.settings),
        }


@dataclass(frozen=True)
class EquipmentLedger:
    equipment: dict[str, EquipmentRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "equipment", deepcopy(self.equipment))

    def to_dict(self) -> dict[str, Any]:
        return {
            equipment_id: equipment.to_dict() for equipment_id, equipment in self.equipment.items()
        }


@dataclass(frozen=True)
class VesselThermalRecord:
    vessel_id: str
    energy_jacket_J: float = 0.0
    heat_reaction_J: float = 0.0
    heat_loss_J: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ThermalLedger:
    vessels: dict[str, VesselThermalRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vessels", deepcopy(self.vessels))

    def to_dict(self) -> dict[str, Any]:
        return {vessel_id: record.to_dict() for vessel_id, record in self.vessels.items()}


@dataclass(frozen=True)
class ProcessLedger:
    time_s: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    sample_consumed_L: float = 0.0
    waste_L: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    last_observation: dict[str, float | None] = field(default_factory=dict)
    last_observed_mask: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        object.__setattr__(
            self,
            "last_observation",
            {
                str(key): (None if value is None else float(value))
                for key, value in self.last_observation.items()
            },
        )
        object.__setattr__(
            self,
            "last_observed_mask",
            {str(key): bool(value) for key, value in self.last_observed_mask.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["metrics"] = deepcopy(self.metrics)
        payload["last_observation"] = deepcopy(self.last_observation)
        payload["last_observed_mask"] = deepcopy(self.last_observed_mask)
        return payload


def process_with_metrics(
    process: ProcessLedger | None,
    **metrics: float,
) -> ProcessLedger:
    """Return a process ledger with merged derived process metrics."""

    process = process or ProcessLedger()
    merged = process.metrics.copy()
    merged.update({str(key): float(value) for key, value in metrics.items()})
    return replace(process, metrics=merged)


def species_with_added_initial_amounts(
    species: SpeciesLedger | None,
    additions_mol: Mapping[str, float],
) -> SpeciesLedger:
    """Return a species ledger with accumulated campaign initial charges."""

    species = species or SpeciesLedger()
    merged = species.initial_amounts_mol.copy()
    for species_id, amount_mol in additions_mol.items():
        merged[str(species_id)] = merged.get(str(species_id), 0.0) + max(
            float(amount_mol),
            0.0,
        )
    return replace(species, initial_amounts_mol=merged)


def scale_species_initial_amounts(
    species: SpeciesLedger | None,
    factor: float,
) -> SpeciesLedger | None:
    """Scale the remaining sample's initial-charge basis after proportional withdrawal."""

    if species is None:
        return None
    return replace(
        species,
        initial_amounts_mol={
            species_id: float(amount_mol) * factor
            for species_id, amount_mol in species.initial_amounts_mol.items()
        },
    )


def process_with_last_observation(
    process: ProcessLedger | None,
    values: dict[str, float | None],
    observed_mask: dict[str, bool],
) -> ProcessLedger:
    """Return a process ledger with updated public observation cache."""

    process = process or ProcessLedger()
    return replace(
        process,
        last_observation=values.copy(),
        last_observed_mask=observed_mask.copy(),
    )


__all__ = [
    "EquipmentLedger",
    "EquipmentRecord",
    "PhaseLedger",
    "PhaseRecord",
    "ProcessLedger",
    "SpeciesLedger",
    "ThermalLedger",
    "VesselLedger",
    "VesselRecord",
    "VesselThermalRecord",
    "process_with_last_observation",
    "process_with_metrics",
    "scale_species_initial_amounts",
]
