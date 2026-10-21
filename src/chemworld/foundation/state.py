"""Hidden state, public observations, and experiment ledger."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any

from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.lab_clock import LabClockState
from chemworld.foundation.observations import Observation
from chemworld.foundation.samples import SampleLedger
from chemworld.foundation.solvents import SolventAccounting, SolventInventory
from chemworld.foundation.state_helpers import (
    equipment_settings,
    equipment_status,
    has_phase_system,
    instrument_completed,
    instrument_equipment_id,
    phases_are_settled,
    scale_phase_ledger,
    selected_phase_id,
    upsert_equipment_record,
)
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    ProcessLedger,
    SpeciesLedger,
    ThermalLedger,
    VesselLedger,
    VesselRecord,
    VesselThermalRecord,
    process_with_last_observation,
    process_with_metrics,
    scale_species_initial_amounts,
    species_with_added_initial_amounts,
)
from chemworld.foundation.state_ownership import StateSlots


@dataclass(frozen=True)
class Ledger:
    time_s: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    sample_consumed_L: float = 0.0
    energy_jacket_J: float = 0.0
    heat_reaction_J: float = 0.0
    heat_loss_J: float = 0.0
    gas_pump_work_J: float = 0.0
    gas_bath_heat_J: float = 0.0
    flow_pump_work_J: float = 0.0
    flow_electrical_work_J: float = 0.0
    analysis_time_s: float = 0.0

    def with_updates(self, **updates: float) -> Ledger:
        return replace(self, **updates)

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class WorldState:
    """Owned physical state with ledger defaults normalized at construction.

    Optional ledger arguments are input conveniences. Every successfully
    constructed state (including ``replace``) has species, phases, vessels,
    equipment, thermal and process ledgers; consumers need no repeated None guard.
    """

    species_amounts: dict[str, float]
    volume_L: float
    temperature_K: float
    pressure_Pa: float
    phase: str
    vessel_id: str
    terminated: bool = False
    quenched: bool = False
    ledger: Ledger = field(default_factory=Ledger)
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    species: SpeciesLedger | None = None
    phases: PhaseLedger | None = None
    vessels: VesselLedger | None = None
    equipment: EquipmentLedger | None = None
    thermal: ThermalLedger | None = None
    process: ProcessLedger | None = None
    samples: SampleLedger = field(default_factory=SampleLedger)
    solvent_accounting: SolventAccounting = field(default_factory=SolventAccounting)
    inactive_vessels: dict[str, WorldState] = field(default_factory=dict)
    vessel_elapsed_s: dict[str, float] = field(default_factory=dict)
    state_slots: StateSlots = field(default_factory=StateSlots)
    liquid_heat_capacity_J_L_K: float | None = 4180.0
    lab_clock: LabClockState | None = None

    def __post_init__(self) -> None:
        if self.phase == "formal":
            if (self.liquid_heat_capacity_J_L_K is not None or self.volume_L != 0.
                    or self.species_amounts or self.state_slots.species_amounts()
                    or any(getattr(self.solvent_accounting, key).volume_L
                           for key in ("initial", "added", "removed"))
                    or (self.species is not None and self.species.initial_amounts_mol)
                    or self.samples.samples
                    or (self.phases is not None and any(
                        p.species_amounts_mol or p.solvents.volume_L or p.volume_L != 0.
                        for p in self.phases.phases.values()))):
                raise ValueError("formal workspaces cannot own physical material or heat capacity")
        elif (self.liquid_heat_capacity_J_L_K is None
              or not isfinite(self.liquid_heat_capacity_J_L_K)
              or self.liquid_heat_capacity_J_L_K <= 0):
            raise ValueError("Liquid heat capacity must be finite and positive")
        if self.vessel_id in self.inactive_vessels or any(
            key != value.vessel_id or value.inactive_vessels
            for key, value in self.inactive_vessels.items()
        ):
            raise ValueError("Inactive vessels must be distinct, flat local snapshots")
        object.__setattr__(self, "inactive_vessels", dict(self.inactive_vessels))
        object.__setattr__(self, "vessel_elapsed_s", dict(self.vessel_elapsed_s))
        object.__setattr__(self, "species_amounts", deepcopy(self.species_amounts))
        object.__setattr__(self, "units", deepcopy(self.units))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        object.__setattr__(self, "state_slots", deepcopy(self.state_slots))
        species = self.species or SpeciesLedger(
            initial_amounts_mol={
                key: float(value)
                for key, value in self.species_amounts.items()
                if float(value) > 0.0
            }
        )
        phase = PhaseRecord(
            phase_id="reactor_liquid",
            vessel_id=self.vessel_id,
            phase_type=self.phase,
            volume_L=self.volume_L,
            species_amounts_mol=self.species_amounts,
            solvents=(
                self.phases.phases["reactor_liquid"].solvents
                if self.phases is not None and "reactor_liquid" in self.phases.phases
                else SolventInventory()
            ),
        )
        phases = (
            PhaseLedger({"reactor_liquid": phase})
            if self.phases is None or set(self.phases.phases) == {"reactor_liquid"}
            else self.phases
        )
        if self.vessels is None:
            vessel = VesselRecord(
                vessel_id=self.vessel_id,
                vessel_type=self.vessel_id,
                max_volume_L=0.10,
                max_temperature_K=470.0,
                max_pressure_Pa=550_000.0,
                phase_ids=tuple(phases.phases),
                temperature_K=self.temperature_K,
                pressure_Pa=self.pressure_Pa,
            )
            vessels = VesselLedger({self.vessel_id: vessel})
        elif set(self.vessels.vessels) == {self.vessel_id}:
            current_vessel = self.vessels.vessels[self.vessel_id]
            vessels = VesselLedger(
                {
                    self.vessel_id: VesselRecord(
                        vessel_id=current_vessel.vessel_id,
                        vessel_type=current_vessel.vessel_type,
                        max_volume_L=current_vessel.max_volume_L,
                        max_temperature_K=current_vessel.max_temperature_K,
                        max_pressure_Pa=current_vessel.max_pressure_Pa,
                        phase_ids=tuple(phases.phases),
                        temperature_K=self.temperature_K,
                        pressure_Pa=self.pressure_Pa,
                    )
                }
            )
        else:
            vessels = self.vessels
        equipment = (
            EquipmentLedger(
                {
                    "batch_reactor": EquipmentRecord(
                        "batch_reactor",
                        "batch_reactor",
                        attached_vessel_id=self.vessel_id,
                        status="terminated" if self.terminated else "idle",
                        settings={},
                    )
                }
            )
            if self.equipment is None
            else self.equipment
        )
        thermal = (
            ThermalLedger(
                {
                    self.vessel_id: VesselThermalRecord(
                        self.vessel_id,
                        energy_jacket_J=self.ledger.energy_jacket_J,
                        heat_reaction_J=self.ledger.heat_reaction_J,
                        heat_loss_J=self.ledger.heat_loss_J,
                    )
                }
            )
            if self.thermal is None
            or (
                set(self.thermal.vessels) == {self.vessel_id}
                and "component_network" not in self.metadata
            )
            else self.thermal
        )
        process = ProcessLedger(
            time_s=self.ledger.time_s,
            cost=self.ledger.cost,
            risk=self.ledger.risk,
            sample_consumed_L=self.ledger.sample_consumed_L,
            waste_L=0.0 if self.process is None else self.process.waste_L,
            metrics={} if self.process is None else self.process.metrics,
            last_observation_record=(
                None if self.process is None else self.process.last_observation_record
            ),
            stochastic_continuation=(
                None if self.process is None else self.process.stochastic_continuation
            ),
        )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "vessels", vessels)
        object.__setattr__(self, "equipment", equipment)
        object.__setattr__(self, "thermal", thermal)
        object.__setattr__(self, "process", process)
        for slot in self.state_slots.slots:
            if slot.location_id != self.vessel_id:
                raise ValueError("State slot location must match its local vessel")
            if (
                slot.spec.owner_kind == "equipment"
                and slot.spec.owner_id not in equipment.equipment
            ):
                raise ValueError("State slot equipment owner is absent")
            if slot.spec.owner_kind == "vessel" and slot.spec.owner_id != self.vessel_id:
                raise ValueError("State slot vessel owner is absent")
            if slot.spec.phase_id is not None and slot.spec.phase_id not in phases.phases:
                raise ValueError("Material state refers to an absent phase")

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "volume_L": self.volume_L,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "phase": self.phase,
            "vessel_id": self.vessel_id,
            "inactive_vessels": {
                key: value.to_dict(include_hidden=include_hidden)
                for key, value in sorted(self.inactive_vessels.items())
            },
            "vessel_elapsed_s": dict(sorted(self.vessel_elapsed_s.items())),
            "terminated": self.terminated,
            "quenched": self.quenched,
            "ledger": self.ledger.to_dict(),
            "units": deepcopy(self.units),
            "metadata": deepcopy(self.metadata),
            "species": None if self.species is None else self.species.to_dict(),
            "phases": None if self.phases is None else self.phases.to_dict(),
            "vessels": None if self.vessels is None else self.vessels.to_dict(),
            "equipment": None if self.equipment is None else self.equipment.to_dict(
                include_hidden=include_hidden),
            "thermal": None if self.thermal is None else self.thermal.to_dict(),
            "process": None
            if self.process is None
            else self.process.to_dict(include_hidden=include_hidden),
            "samples": self.samples.to_dict() if include_hidden else self.samples.public_summary(),
            "material_routing": self.samples.routing_summary(),
            "solvent_accounting": self.solvent_accounting.to_dict(),
            "state_slots": self.state_slots.to_dict(include_hidden=include_hidden),
        }
        if include_hidden:
            payload["state_slot_order"] = [slot.spec.slot_id for slot in self.state_slots.slots]
            payload["liquid_heat_capacity_J_L_K"] = self.liquid_heat_capacity_J_L_K
            payload["species_amounts"] = deepcopy(self.species_amounts)
            payload["active_reference_sources"] = deepcopy(self.samples.active_reference_sources)
            payload["active_reference_shares"] = deepcopy(self.samples.active_reference_shares)
        if self.lab_clock is not None:
            payload["lab_clock"] = self.lab_clock.to_dict(include_hidden=include_hidden)
        return payload

    def replace(self, **updates: Any) -> WorldState:
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorldState:
        """Restore a full physical snapshot, not a public observation."""
        from chemworld.foundation.state_serialization import restore_state

        return restore_state(payload)


@dataclass(frozen=True)
class OperationRecord:
    operation_type: str
    preconditions: dict[str, bool]
    state_delta_summary: dict[str, float]
    constitution_checks: list[dict[str, object]]
    instrument: str | None = None
    measurement_cost: float = 0.0
    sample_consumed_L: float = 0.0
    material_balance: TransitionBalance = field(default_factory=TransitionBalance)
    terminal_assay: bool | None = None

    @property
    def is_instrument_measurement(self) -> bool:
        return self.instrument is not None

    @property
    def is_final_assay(self) -> bool:
        return (
            self.instrument == "final_assay" if self.terminal_assay is None else self.terminal_assay
        )

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, object]:
        return {
            "operation_type": self.operation_type,
            "preconditions": self.preconditions,
            "state_delta_summary": self.state_delta_summary,
            "constitution_checks": self.constitution_checks,
            "instrument": self.instrument,
            "terminal_assay": self.is_final_assay,
            "measurement_cost": self.measurement_cost,
            "sample_consumed": self.sample_consumed_L,
            "material_balance": self.material_balance.to_dict(include_hidden=include_hidden),
        }


__all__ = [
    "EquipmentLedger",
    "EquipmentRecord",
    "Ledger",
    "Observation",
    "OperationRecord",
    "PhaseLedger",
    "PhaseRecord",
    "ProcessLedger",
    "SpeciesLedger",
    "ThermalLedger",
    "VesselLedger",
    "VesselRecord",
    "VesselThermalRecord",
    "WorldState",
    "equipment_settings",
    "equipment_status",
    "has_phase_system",
    "instrument_completed",
    "instrument_equipment_id",
    "phases_are_settled",
    "process_with_last_observation",
    "process_with_metrics",
    "scale_phase_ledger",
    "scale_species_initial_amounts",
    "selected_phase_id",
    "species_with_added_initial_amounts",
    "upsert_equipment_record",
]
