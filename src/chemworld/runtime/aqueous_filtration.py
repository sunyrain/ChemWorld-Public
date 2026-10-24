"""Two finite filtration packets and three phase owners, one original transaction."""

from __future__ import annotations

from dataclasses import replace
from json import dumps, loads
from math import fsum
from typing import Any, cast

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.aqueous_mineral_equilibrium import aqueous_mineral_properties
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.aqueous_precipitation import (
    PrecipitationClosureError,
    close_precipitation,
    precipitation_properties,
    precipitation_record,
    precipitation_specification,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult

AQUEOUS_FILTER_TYPE = "database_aqueous_solid_liquid_filter"


def aqueous_filter_record(state: Any) -> Any:
    if state is None:
        return None
    rows = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == AQUEOUS_FILTER_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(rows) > 1:
        raise ValueError("one filtration device is allowed per original source vessel")
    return rows[0] if rows else None


def validate_aqueous_filter_declaration(
    state: Any, physics: Any, *, check_connections: bool = True
) -> Any:
    """Validate actual/restored hardware configuration without another chemical solve."""
    from chemworld.world.aqueous_filtration_world import AqueousFilterSpec

    record = aqueous_filter_record(state)
    original = aqueous_filter_record(physics.initial_states.get(state.vessel_id))
    if record is None and original is None:
        return None
    if (
        record is None
        or original is None
        or loads(dumps(record.settings)) != loads(dumps(original.settings))
        or record.equipment_id != original.equipment_id
    ):
        raise ValueError("filter declaration differs from original world configuration")
    spec = AqueousFilterSpec.from_dict(record.settings["filter_specification"])
    if spec.source_vessel != state.vessel_id or spec.filter_id != record.equipment_id:
        raise ValueError("filter identity must belong to its original source vessel")
    names = (spec.source_vessel, spec.cake_vessel, spec.filtrate_vessel)
    for name in names:
        owner = physics.initial_states.get(name)
        if (
            owner is None
            or precipitation_record(owner) is None
            or precipitation_record(owner).settings.get("allow_empty_inventory") is not True
        ):
            raise ValueError("all filter endpoints require explicitly empty-capable aqueous owners")
        if precipitation_specification(owner) != precipitation_specification(state):
            raise ValueError("filter endpoints require identical declared aqueous chemistry")
    if check_connections:
        edges = [
            e
            for e in state.metadata["component_network"]["connections"]
            if e["source"]["component"] == spec.filter_id
        ]
        if len(edges) != 2:
            raise ValueError("filter requires exactly its two authorized output connections")
        for port, target in (("cake", spec.cake_vessel), ("filtrate", spec.filtrate_vessel)):
            matches = [e for e in edges if e["source"]["port"] == port]
            if len(matches) != 1 or (
                matches[0]["source_vessel"] != spec.source_vessel
                or matches[0]["target_vessel"] != target
                or matches[0]["target"]["port"] != "in"
                or matches[0]["target"]["component"] != target + "-solution"
                or matches[0]["id"] != spec.filter_id + ":" + port
                or target not in state.inactive_vessels
            ):
                raise ValueError(
                    "filter output does not match its authorized source/target connection"
                )
    return spec


def _install_amounts(state: Any, amounts: dict[str, float]) -> Any:
    record = precipitation_record(state)
    spec = precipitation_specification(state)
    liquid_id, mineral_ids = (
        record.settings["liquid_phase_id"],
        record.settings["mineral_phase_ids"],
    )
    liquid = {
        s.species_id: amounts[s.species_id] for s in spec.database.species if s.phase != "mineral"
    }
    volume = (
        1000
        * aqueous_phase_properties(
            spec.database, liquid, temperature_K=state.temperature_K, pressure_Pa=state.pressure_Pa
        ).volume_m3
        if any(amounts.values())
        else 0.0
    )
    phases = {
        liquid_id: replace(
            state.phases.phases[liquid_id], volume_L=volume, species_amounts_mol=liquid
        )
    }
    for species in spec.database.species:
        if species.phase == "mineral":
            n, key = amounts[species.species_id], mineral_ids[species.species_id]
            phases[key] = replace(
                state.phases.phases[key],
                species_amounts_mol={species.species_id: n},
                volume_L=n * cast(float, species.standard_state.molar_volume_m3_mol) * 1000,
            )
    candidate = state.replace(
        phases=PhaseLedger(phases),
        species_amounts=amounts,
        volume_L=fsum(p.volume_L for p in phases.values()),
        process=replace(state.process, last_observation_record=None),
    )
    return close_precipitation(candidate)[0]


def filter_aqueous_suspension(state: Any, physics: Any) -> ModelProviderResult:
    """Return numerical closure failures through the original transactional outcome."""
    try:
        return _filter_aqueous_suspension(state, physics)
    except (PrecipitationClosureError, RuntimeError, ArithmeticError) as exc:
        return ModelProviderResult(
            {},
            diagnostics={"solver_error": str(exc)},
            success=False,
            outcome="numerical_failure",
            failure_reason="aqueous_filtration_numerical_failure",
        )


def _filter_aqueous_suspension(state: Any, physics: Any) -> ModelProviderResult:
    spec = validate_aqueous_filter_declaration(state, physics)
    if spec is None or state.lab_clock is not None:
        raise ValueError("aqueous filtration requires its declared device and local clock")
    members = {state.vessel_id: state, **state.inactive_vessels}
    names = (spec.source_vessel, spec.cake_vessel, spec.filtrate_vessel)
    chemistry = precipitation_specification(state)
    mineral = chemistry.mineral_ids[0]
    if state.species_amounts[mineral] <= 0:
        raise ValueError("filtration requires actual suspended mineral inventory")
    for name in names:
        local = members[name]
        if (
            local.temperature_K != state.temperature_K
            or local.pressure_Pa != state.pressure_Pa
            or precipitation_specification(local) != chemistry
        ):
            raise ValueError("filtration endpoints require common declared chemistry and TP")
        close_precipitation(local)
        if name != state.vessel_id and any(local.species_amounts.values()):
            raise ValueError("filtration receivers must be actually empty before collection")
    cake_amounts = {
        k: n * (spec.solid_retention_fraction if k == mineral else spec.cake_liquid_fraction)
        for k, n in state.species_amounts.items()
    }
    cake_packet = MaterialPacket(state.vessel_id, spec.cake_vessel, tuple(cake_amounts.items()))
    remaining, cake = cake_packet.apply(
        state.species_amounts, members[spec.cake_vessel].species_amounts
    )
    filtrate_packet = MaterialPacket(
        state.vessel_id, spec.filtrate_vessel, tuple(remaining.items())
    )
    zero, filtrate = filtrate_packet.apply(remaining, members[spec.filtrate_vessel].species_amounts)
    packets = (cake_packet, filtrate_packet)
    candidates = {
        state.vessel_id: _install_amounts(state, zero),
        spec.cake_vessel: _install_amounts(members[spec.cake_vessel], cake),
        spec.filtrate_vessel: _install_amounts(members[spec.filtrate_vessel], filtrate),
    }
    before = {k: precipitation_properties(members[k]) for k in names}
    after = {k: precipitation_properties(candidates[k]) for k in names}
    packet_properties = {
        p.destination: aqueous_mineral_properties(
            chemistry,
            dict(p.amounts_mol),
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
        )
        for p in packets
    }
    heats = {
        name: after[name].enthalpy_J
        - before[name].enthalpy_J
        - (
            packet_properties[name].enthalpy_J
            if name != state.vessel_id
            else -fsum(p.enthalpy_J for p in packet_properties.values())
        )
        for name in names
    }
    work = state.pressure_Pa * fsum(after[k].volume_m3 - before[k].volume_m3 for k in names)
    local_heat = {
        k: {"energy_jacket_J": max(q, 0.0), "heat_loss_J": max(-q, 0.0)} for k, q in heats.items()
    }
    original_moles = fsum(state.species_amounts.values())
    for packet in packets:
        local = members[packet.destination]
        fraction = packet.total_mol / original_moles
        candidates[packet.destination] = candidates[packet.destination].replace(
            species=species_with_added_initial_amounts(
                local.species,
                {k: n * fraction for k, n in state.species.initial_amounts_mol.items()},
            )
        )
    source = candidates.pop(state.vessel_id)
    source = source.replace(
        species=replace(
            source.species,
            initial_amounts_mol=dict.fromkeys(source.species.initial_amounts_mol, 0.0),
        )
    )
    receipt = {
        "filter": spec.filter_id,
        "model": spec.to_dict()["model"],
        "packets": [p.to_dict() for p in packets],
        "vessel_heat_duty_J": heats,
        "heat_duty_J": fsum(heats.values()),
        "boundary_work_J": work,
        "initial_properties": {k: v.to_dict() for k, v in before.items()},
        "final_properties": {k: v.to_dict() for k, v in after.items()},
        "duration_s": 0.0,
        "assumption": "instantaneous_declared_phase_retention_no_hydraulics",
    }
    source = source.replace(
        inactive_vessels={**state.inactive_vessels, **candidates},
        ledger=advance_ledger(
            state.ledger,
            0,
            energy_deltas={
                **{
                    key: fsum(v[key] for v in local_heat.values())
                    for key in ("energy_jacket_J", "heat_loss_J")
                },
                "boundary_work_J": work,
            },
        ),
        metadata={
            **source.metadata,
            "aqueous_filter_history": [*state.metadata.get("aqueous_filter_history", []), receipt],
        },
    )
    return ModelProviderResult(
        {},
        state=source,
        diagnostics={"aqueous_filtration": receipt},
        vessel_thermal_deltas=local_heat,
    )
