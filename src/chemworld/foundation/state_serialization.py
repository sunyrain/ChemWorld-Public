"""Read full physical snapshots; legacy material settings convert only here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chemworld.foundation.lab_clock import LabClockState
from chemworld.foundation.material_properties import MATERIAL_METRICS, extract_material_properties
from chemworld.foundation.samples import MaterialContents, SampleLedger, StoredSample
from chemworld.foundation.solvents import SOLVENT_IDS, SolventAccounting, SolventInventory
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
)
from chemworld.foundation.state_ownership import StateSlots

if TYPE_CHECKING:
    from chemworld.foundation.state import WorldState


def _solvents(value: dict[str, Any]) -> SolventInventory:
    if "volumes_L" in value:  # previous dataclasses.asdict sample representation
        return SolventInventory(tuple(value["volumes_L"]))
    return SolventInventory(tuple(value.get(key, 0.0) for key in SOLVENT_IDS))


def _phases(
    value: dict[str, Any], *, order: list[str] | tuple[str, ...] | None = None,
) -> PhaseLedger:
    # Historical dataclasses.asdict used one wrapper. A phase itself may also
    # have the identity "phases"; its explicit record is not that wrapper.
    if set(value) == {"phases"} and "phase_id" not in value["phases"]:
        value = value["phases"]
    if order is not None and (
        not isinstance(order, list | tuple) or any(not isinstance(key, str) for key in order)
    ):
        raise ValueError("serialized phase order must be an identity sequence")
    keys = tuple(value) if order is None else tuple(order)
    if len(set(keys)) != len(keys) or set(keys) != set(value):
        raise ValueError("serialized phase order must match its declared identities")
    if any(value[key].get("phase_id") != key for key in keys):
        raise ValueError("serialized phase identities disagree with their mapping")
    return PhaseLedger(
        {
            key: PhaseRecord(**{
                **value[key], "solvents": _solvents(value[key].get("solvents", {}))
            })
            for key in keys
        }
    )


def _species(value: dict[str, Any]) -> SpeciesLedger:
    return SpeciesLedger(
        {k: tuple(v) for k, v in value.get("species_roles", {}).items()},
        value.get("initial_amounts_mol", {}),
    )


def _equipment(value: dict[str, Any]) -> EquipmentLedger:
    return EquipmentLedger(
        {key: EquipmentRecord(**record) for key, record in value.get("equipment", value).items()}
    )


def restore_sample(payload: dict[str, Any], *, require_order: bool = False) -> StoredSample:
    values = dict(payload)
    contents = values.get("contents")
    if contents is not None:
        if require_order and not {"phase_order", "state_slot_order"} <= contents.keys():
            raise ValueError("session sample is missing its declared inventory order")
        properties = contents.get("properties")
        if properties is None:
            properties = extract_material_properties(_equipment(contents.get("equipment", {})))
        values["contents"] = MaterialContents(
            _phases(contents["phases"], order=contents.get("phase_order")),
            _species(contents["species"]),
            properties,
            {k: v for k, v in contents.get("metrics", {}).items() if k in MATERIAL_METRICS},
            contents.get("reference_sources", {}),
            contents.get("reference_shares", {}),
            StateSlots.from_dict(
                contents.get("state_slots", {}), order=contents.get("state_slot_order")
            ),
            contents.get("liquid_heat_capacity_J_L_K", 4180.0),
            contents.get("mechanism_id", ""),
        )
    values["lineage"] = tuple(values.get("lineage", ()))
    return StoredSample(**values)


def restore_state(payload: dict[str, Any], *, require_order: bool = False) -> WorldState:
    from chemworld.foundation.state import Ledger, WorldState

    if "species_amounts" not in payload:
        raise ValueError("Restoring physical state requires a full hidden-state snapshot")
    if require_order and "state_slot_order" not in payload:
        raise ValueError("session state is missing its declared material slot order")
    vessel_id, vessels = payload["vessel_id"], payload.get("vessels")
    if (
        not isinstance(vessel_id, str) or not isinstance(vessels, dict)
        or vessel_id not in vessels or not isinstance(vessels[vessel_id], dict)
        or "phase_ids" not in vessels[vessel_id]
    ):
        raise ValueError("restored active vessel requires its declared identity and phase order")
    routing = payload.get("material_routing", {})
    samples = SampleLedger(
        samples={key: restore_sample(p, require_order=require_order)
                 for key, p in payload.get("samples", {}).items()},
        transfers=tuple(routing.get("transfers", ())),
        containers_used=routing.get("containers_used", 0),
        transfer_tools_used=routing.get("transfer_tools_used", 0),
        active_lineage=tuple(routing.get("active_lineage", ())),
        active_reference_sources=payload.get("active_reference_sources", {}),
        active_reference_shares=payload.get("active_reference_shares", {}),
        batch_generation=routing.get("batch_generation", 0),
    )
    return WorldState(
        **{
            key: payload[key]
            for key in (
                "species_amounts",
                "volume_L",
                "temperature_K",
                "pressure_Pa",
                "phase",
                "vessel_id",
                "terminated",
                "quenched",
                "units",
                "metadata",
            )
        },
        ledger=Ledger(**payload["ledger"]),
        species=_species(payload["species"]),
        phases=_phases(
            payload["phases"], order=vessels[vessel_id]["phase_ids"]
        ),
        equipment=_equipment(payload["equipment"]),
        vessels=VesselLedger(
            {
                key: VesselRecord(**{**p, "phase_ids": tuple(p["phase_ids"])})
                for key, p in payload["vessels"].items()
            }
        ),
        thermal=ThermalLedger(
            {key: VesselThermalRecord(**p) for key, p in payload["thermal"].items()}
        ),
        process=ProcessLedger(**payload["process"]),
        samples=samples,
        solvent_accounting=SolventAccounting(
            **{
                key: _solvents(value)
                for key, value in payload.get("solvent_accounting", {}).items()
            }
        ),
        inactive_vessels={
            key: restore_state(p, require_order=require_order)
            for key, p in payload.get("inactive_vessels", {}).items()
        },
        vessel_elapsed_s=payload.get("vessel_elapsed_s", {}),
        state_slots=StateSlots.from_dict(
            payload.get("state_slots", {}), order=payload.get("state_slot_order")
        ),
        liquid_heat_capacity_J_L_K=payload.get("liquid_heat_capacity_J_L_K", 4180.0),
        lab_clock=(LabClockState(**payload["lab_clock"]) if "lab_clock" in payload else None),
    )
