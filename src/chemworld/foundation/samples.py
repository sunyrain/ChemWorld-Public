"""Material removed from the active vessel and retained as identified samples."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

from chemworld.foundation.material_properties import MATERIAL_FIELDS
from chemworld.foundation.state_ledgers import PhaseLedger, SpeciesLedger
from chemworld.foundation.state_ownership import StateSlots

# A finite, public address space also has a lossless numeric Gym encoding.
# Filtrate addresses refer to samples collected by resuspend_crystals.
CONTAINER_IDS = (
    "active",
    *(f"container-{i:02d}" for i in range(1, 17)),
    *(f"filtrate-{i:04d}" for i in range(1, 17)),
)


@dataclass(frozen=True)
class MaterialContents:
    """Material-local processing state; costs, clocks and instruments stay global."""

    phases: PhaseLedger
    species: SpeciesLedger
    properties: dict[str, dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)
    reference_sources: dict[str, dict[str, float]] = field(default_factory=dict)
    reference_shares: dict[str, float] = field(default_factory=dict)
    state_slots: StateSlots = field(default_factory=StateSlots)
    liquid_heat_capacity_J_L_K: float = 4180.0
    mechanism_id: str = ""

    def __post_init__(self) -> None:
        for key in (
            "phases",
            "species",
            "properties",
            "metrics",
            "reference_sources",
            "reference_shares",
            "state_slots",
        ):
            object.__setattr__(self, key, deepcopy(getattr(self, key)))
        if any(
            name not in MATERIAL_FIELDS or not set(values) <= MATERIAL_FIELDS[name].keys()
            for name, values in self.properties.items()
        ):
            raise ValueError("Undeclared material properties cannot travel with a sample")
        if self.state_slots.stationary().slots:
            raise ValueError("Stored material cannot contain vessel or equipment state")
        if not isfinite(self.liquid_heat_capacity_J_L_K) or self.liquid_heat_capacity_J_L_K <= 0:
            raise ValueError("Material heat capacity must be finite and positive")


@dataclass(frozen=True)
class StoredSample:
    sample_id: str
    source_vessel_id: str
    collection_operation: str
    collected_at_s: float
    volume_L: float
    species_amounts_mol: dict[str, float]
    temperature_K: float
    solvent: int
    quenched: bool
    seed_target_mol: float = 0.0
    capacity_L: float = 0.10
    retired: bool = False
    lineage: tuple[str, ...] = ()
    contents: MaterialContents | None = None

    def __post_init__(self) -> None:
        values = [
            self.collected_at_s,
            self.volume_L,
            self.temperature_K,
            self.seed_target_mol,
            self.capacity_L,
        ]
        values.extend(self.species_amounts_mol.values())
        if not self.sample_id or not all(isfinite(x) and x >= 0 for x in values):
            raise ValueError("Stored sample requires an identity and finite nonnegative inventory")
        object.__setattr__(self, "species_amounts_mol", deepcopy(self.species_amounts_mol))
        object.__setattr__(self, "contents", deepcopy(self.contents))
        if self.volume_L > self.capacity_L + 1e-12:
            raise ValueError("Sample exceeds container capacity")
        if self.retired and (self.volume_L > 0 or any(self.species_amounts_mol.values())):
            raise ValueError("Only empty containers can be retired")
        if self.contents is not None:
            if self.retired and self.contents.state_slots.has_inventory():
                raise ValueError("Retired containers cannot retain material-owned inventory")
            phases = self.contents.phases.phases.values()
            values = [x for p in phases for x in (p.volume_L, *p.species_amounts_mol.values())]
            if not all(isfinite(x) and x >= 0 for x in values):
                raise ValueError("Sample phases require finite nonnegative inventory")
            if (
                abs(sum(p.volume_L for p in self.contents.phases.phases.values()) - self.volume_L)
                > 1e-12
            ):
                raise ValueError("Sample phase volumes and aggregate volume disagree")
            totals = self.contents.phases.total_amounts_mol()
            if any(
                abs(totals.get(k, 0) - self.species_amounts_mol.get(k, 0)) > 1e-12
                for k in totals.keys() | self.species_amounts_mol.keys()
            ):
                raise ValueError("Sample phase and aggregate inventories disagree")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.contents:
            payload["contents"] = {
                "phases": self.contents.phases.to_dict(),
                "phase_order": list(self.contents.phases.phases),
                "species": self.contents.species.to_dict(),
                "properties": deepcopy(self.contents.properties),
                "metrics": dict(self.contents.metrics),
                "reference_sources": deepcopy(self.contents.reference_sources),
                "reference_shares": dict(self.contents.reference_shares),
                "state_slots": self.contents.state_slots.to_dict(),
                "state_slot_order": [slot.spec.slot_id
                                     for slot in self.contents.state_slots.slots],
                "liquid_heat_capacity_J_L_K": self.contents.liquid_heat_capacity_J_L_K,
                "mechanism_id": self.contents.mechanism_id,
            }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StoredSample:
        from chemworld.foundation.state_serialization import restore_sample

        return restore_sample(payload)

    def public_summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_vessel_id": self.source_vessel_id,
            "collection_operation": self.collection_operation,
            "collected_at_s": self.collected_at_s,
            "volume_L": self.volume_L,
            "quenched": self.quenched,
            "capacity_L": self.capacity_L,
            "retired": self.retired,
            "lineage": list(self.lineage),
            "storage_model": "sealed_isothermal_quenched_no_phase_evolution",
        }


@dataclass(frozen=True)
class SampleLedger:
    samples: dict[str, StoredSample] = field(default_factory=dict)
    transfers: tuple[dict[str, Any], ...] = ()
    containers_used: int = 0
    transfer_tools_used: int = 0
    active_lineage: tuple[str, ...] = ()
    active_reference_sources: dict[str, dict[str, float]] = field(default_factory=dict)
    active_reference_shares: dict[str, float] = field(default_factory=dict)
    batch_generation: int = 0

    def __post_init__(self) -> None:
        if any(key != sample.sample_id for key, sample in self.samples.items()):
            raise ValueError("Sample ledger keys must match sample identities")
        object.__setattr__(self, "samples", deepcopy(self.samples))
        object.__setattr__(self, "transfers", deepcopy(self.transfers))
        object.__setattr__(
            self, "active_reference_sources", deepcopy(self.active_reference_sources)
        )
        object.__setattr__(self, "active_reference_shares", deepcopy(self.active_reference_shares))

    def append(self, sample: StoredSample) -> SampleLedger:
        if sample.sample_id in self.samples:
            raise ValueError(f"Sample identity already exists: {sample.sample_id}")
        from dataclasses import replace

        return replace(self, samples={**self.samples, sample.sample_id: sample})

    def total_amounts_mol(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for sample in self.samples.values():
            for species, amount in sample.species_amounts_mol.items():
                totals[species] = totals.get(species, 0.0) + amount
            if sample.contents:
                for species, amount in sample.contents.state_slots.species_amounts().items():
                    totals[species] = totals.get(species, 0.0) + amount
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {key: sample.to_dict() for key, sample in self.samples.items()}

    def public_summary(self) -> list[dict[str, Any]]:
        return [sample.public_summary() for sample in self.samples.values()]

    def routing_summary(self) -> dict[str, Any]:
        return {
            "containers_used": self.containers_used,
            "transfer_tools_used": self.transfer_tools_used,
            "active_lineage": list(self.active_lineage),
            "batch_generation": self.batch_generation,
            "transfers": deepcopy(list(self.transfers)),
        }
