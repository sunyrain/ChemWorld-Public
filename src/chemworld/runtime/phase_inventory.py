"""Shared PH compartment projection onto the existing owned state ledgers."""

from dataclasses import dataclass, replace
from math import isclose

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K


@dataclass(frozen=True)
class CompartmentLayout:
    model_id: str
    state_type: str
    slot_prefix: str
    metadata_key: str
    phase_ids: tuple[tuple[str, str], ...]
    selected_phase_id: str
    preserved_phase_ids: tuple[str, ...] = ()
    preserved_slot_specs: tuple[StateSlotSpec, ...] = ()


def compartment_slot_specs(state, spec, layout):
    declarations = {}
    for i, (liquid, _) in enumerate(layout.phase_ids):
        declarations[f"{layout.slot_prefix}:cell:{i}"] = StateSlotSpec(
            f"{layout.slot_prefix}:cell:{i}",
            "material",
            f"{state.vessel_id}:cell:{i}",
            layout.state_type,
            (StateFieldSpec("enthalpy_J", "J", minimum=None),),
            phase_id=liquid,
            batch_reset="initial",
        )
        if spec.capacities_j_k[i] > 0:
            declarations[f"{layout.slot_prefix}:body:{i}"] = StateSlotSpec(
                f"{layout.slot_prefix}:body:{i}",
                "vessel",
                state.vessel_id,
                layout.state_type,
                (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                batch_reset="initial",
            )
    return declarations


def read_compartments(state, spec, layout):
    ids = spec.properties.component_ids
    amounts = [dict.fromkeys(ids, 0.0) for _ in layout.phase_ids]
    lookup = {
        key: (i, kind)
        for i, pair in enumerate(layout.phase_ids)
        for key, kind in zip(pair, ("liquid", "gas"), strict=True)
    }
    for key, phase in state.phases.phases.items():
        if key in layout.preserved_phase_ids:
            continue
        if key not in lookup:
            raise ValueError("phase inventory contains an undeclared compartment")
        index, kind = lookup[key]
        marker = phase.metadata.get(layout.metadata_key)
        if (
            isinstance(marker, bool)
            or not isinstance(marker, int)
            or marker != index
            or phase.phase_type != kind
            or set(phase.species_amounts_mol) != set(ids)
        ):
            raise ValueError("phase inventory needs its declared compartment and identities")
        for component, value in phase.species_amounts_mol.items():
            amounts[index][component] += value
    slots = {s.spec.slot_id: s for s in state.state_slots.slots}
    declarations = compartment_slot_specs(state, spec, layout)
    required = set(declarations)
    for declaration in layout.preserved_slot_specs:
        declarations[declaration.slot_id] = declaration
        if declaration.phase_id in state.phases.phases:
            required.add(declaration.slot_id)
    if not required <= slots.keys() or set(slots) - set(declarations):
        raise ValueError("phase inventory requires exactly its declared owned enthalpy slots")
    for key, slot in slots.items():
        if slot.spec != declarations[key] or slot.location_id != state.vessel_id:
            raise ValueError("compartment enthalpy ownership/units differ from their declaration")
    enthalpies = tuple(
        slots[f"{layout.slot_prefix}:cell:{i}"].values["enthalpy_J"]
        + (
            slots[f"{layout.slot_prefix}:body:{i}"].values["enthalpy_J"]
            if spec.capacities_j_k[i] > 0
            else 0
        )
        for i in range(len(layout.phase_ids))
    )
    return tuple(amounts), enthalpies


def install_compartments(state, spec, inventory, layout):
    phases = {k: p for k, p in state.phases.phases.items() if k in layout.preserved_phase_ids}
    previous = {s.spec.slot_id: s for s in state.state_slots.slots}
    extras = {s.slot_id for s in layout.preserved_slot_specs}
    slots = [s for s in state.state_slots.slots if s.spec.slot_id in extras]
    declarations = compartment_slot_specs(state, spec, layout)
    temperatures = []
    for i, (amounts, h) in enumerate(
        zip(inventory.amounts_mol, inventory.enthalpies_J, strict=True)
    ):
        report = spec.at_ph(i, amounts, h)
        temperatures.append(None if report is None else report.temperature_K)
        if (
            report is not None
            and not spec.properties.temperature_bounds_K[0] - 1e-7
            <= (report.temperature_K)
            <= spec.equipment.maximum_temperature_K + 1e-7
        ):
            raise ValueError("accepted compartment leaves its declared thermal envelope")
        allocations = (
            (("liquid", dict.fromkeys(spec.properties.component_ids, 0.0)),)
            if (report is None)
            else (("liquid", report.liquid_amounts_mol), ("gas", report.vapor_amounts_mol))
        )
        for kind, n in allocations:
            total = sum(n.values())
            if not total and kind == "gas":
                continue
            key = layout.phase_ids[i][0 if kind == "liquid" else 1]
            volume = (
                total * spec.liquid_molar_holdup_L_mol
                if kind == "liquid"
                else total * R_J_PER_MOL_K * report.temperature_K / spec.pressure_Pa * 1000
            )
            phases[key] = PhaseRecord(
                key,
                state.vessel_id,
                kind,
                volume,
                n,
                settled=True,
                selected=key == layout.selected_phase_id,
                metadata={
                    layout.metadata_key: i,
                    "model": layout.model_id,
                    "carrier_representation": "species_inventory",
                },
            )
        fluid_h = h if report is None else h - report.support_enthalpy_J
        energies = {f"{layout.slot_prefix}:cell:{i}": fluid_h}
        if spec.capacities_j_k[i] > 0:
            energies[f"{layout.slot_prefix}:body:{i}"] = report.support_enthalpy_J
        for slot_id, energy in energies.items():
            slot = (
                replace(previous[slot_id], values={"enthalpy_J": energy})
                if slot_id in previous
                else StateSlot(declarations[slot_id], state.vessel_id, {"enthalpy_J": energy})
            )
            slots.append(slot)
    volume = sum(p.volume_L for p in phases.values())
    if volume > state.vessels.vessels[state.vessel_id].max_volume_L + 1e-12:
        raise ValueError("compartment inventory exceeds its finite vessel capacity")
    amounts = {
        key: sum(p.species_amounts_mol[key] for p in phases.values())
        for key in spec.properties.component_ids
    }
    return state.replace(
        species_amounts=amounts,
        phases=PhaseLedger(phases),
        volume_L=volume,
        state_slots=StateSlots(tuple(slots)),
        temperature_K=temperatures[0],
        pressure_Pa=spec.pressure_Pa,
        process=replace(state.process, last_observation_record=None),
    )


def validate_compartments(state, spec, layout, inventory):
    if not isclose(state.pressure_Pa, spec.pressure_Pa, abs_tol=1e-6):
        raise ValueError("compartment pressure differs from its declared fixed boundary")
    expected = install_compartments(state, spec, inventory, layout)
    if not isclose(state.temperature_K, expected.temperature_K, abs_tol=1e-6):
        raise ValueError("representative temperature differs from the primary PH compartment")
    for key, phase in expected.phases.phases.items():
        actual = state.phases.phases.get(key)
        if (
            actual is None
            or abs(actual.volume_L - phase.volume_L) > 1e-10
            or any(
                abs(actual.species_amounts_mol[k] - v) > 1e-10
                for k, v in phase.species_amounts_mol.items()
            )
        ):
            raise ValueError("compartment phase inventories do not satisfy their PH closure")
