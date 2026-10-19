"""Conservative, finite-address routing of quenched liquids and crystal slurries.

Storage is sealed/isothermal with no chemistry or phase evolution. Moving an
equal fraction of every phase models a representative homogenized aliquot.
Tools and emptied containers are single use; mixing must be explicitly requested.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from chemworld.foundation import WorldState, equipment_settings
from chemworld.foundation.samples import CONTAINER_IDS, MaterialContents, StoredSample
from chemworld.foundation.solvents import HEAT_CAPACITY_RATIOS, SolventInventory
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    ProcessLedger,
    SpeciesLedger,
)
from chemworld.runtime.full_process_contract import population_settings

MATERIAL_EQUIPMENT = frozenset({"batch_reactor", "crystallizer", "crystal_filter"})
ROUTING_OPERATIONS = ("create_container", "transfer_material")


def container_id(value: Any) -> str:
    if isinstance(value, str) and value in CONTAINER_IDS:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(CONTAINER_IDS):
        raise ValueError("Unknown container address")
    return CONTAINER_IDS[value]


def _extensive(key: str) -> bool:
    return key.endswith(("_mol", "_g", "_L"))


def _scale(contents: MaterialContents, fraction: float, vessel_id: str) -> MaterialContents:
    phases = PhaseLedger(
        {
            key: replace(
                p,
                vessel_id=vessel_id,
                volume_L=p.volume_L * fraction,
                species_amounts_mol={s: n * fraction for s, n in p.species_amounts_mol.items()},
            )
            for key, p in contents.phases.phases.items()
        }
    )
    records = {}
    for key, record in contents.equipment.equipment.items():
        settings = {
            k: v * fraction if _extensive(k) and isinstance(v, (int, float)) else v
            for k, v in record.settings.items()
        }
        if "population_cohorts" in settings:
            settings.update(
                population_settings(
                    tuple((n * fraction, d) for n, d in record.settings["population_cohorts"])
                )
            )
        records[key] = replace(record, attached_vessel_id=vessel_id, settings=settings)
    return MaterialContents(
        phases,
        replace(
            contents.species,
            initial_amounts_mol={
                s: n * fraction for s, n in contents.species.initial_amounts_mol.items()
            },
        ),
        EquipmentLedger(records),
        {k: v * fraction if _extensive(k) else v for k, v in contents.metrics.items()},
        contents.reference_sources,
        {k: v * fraction for k, v in contents.reference_shares.items()},
    )


def _active(state: WorldState) -> StoredSample:
    assert state.phases is not None and state.species is not None
    settings = equipment_settings(state.equipment, "batch_reactor")
    crystal = equipment_settings(state.equipment, "crystallizer")
    lineage = state.samples.active_lineage or (
        f"batch-{state.samples.batch_generation:04d}"
        f"-{state.vessel_id}"
        f"-material-{len(state.samples.transfers) + 1:04d}",
    )
    capacity = state.vessels.vessels[state.vessel_id].max_volume_L if state.vessels else 0.1
    sources = state.samples.active_reference_sources
    shares = state.samples.active_reference_shares
    reference = _reference_amounts(sources, shares)
    initial = state.species.initial_amounts_mol
    if not sources or any(
        abs(reference.get(k, 0) - initial.get(k, 0)) > 1e-12
        for k in reference.keys() | initial.keys()
    ):
        # A new charge (or sample-normalized analytical withdrawal) changes the basis.
        source_id = f"{lineage[0]}:charge-{len(state.samples.transfers) + 1}"
        sources, shares = {source_id: initial}, {source_id: 1.0}
    return StoredSample(
        "active",
        state.vessel_id,
        "working_vessel",
        state.ledger.time_s,
        state.volume_L,
        state.species_amounts,
        state.temperature_K,
        int(settings.get("solvent", 0)),
        state.quenched,
        float(crystal.get("seed_target_mol", 0))
        + float(crystal.get("dissolved_seed_target_mol", 0)),
        capacity_L=capacity,
        lineage=lineage,
        contents=MaterialContents(
            state.phases,
            state.species,
            EquipmentLedger(
                {
                    k: v
                    for k, v in (state.equipment or EquipmentLedger()).equipment.items()
                    if k in MATERIAL_EQUIPMENT
                }
            ),
            {} if state.process is None else state.process.metrics,
            sources,
            shares,
        ),
    )


def _contents(sample: StoredSample) -> MaterialContents:
    if sample.contents is not None:
        return sample.contents
    # Retained R26 filtrates carry a true liquid inventory and dissolved seed.
    return MaterialContents(
        PhaseLedger(
            {
                "reactor_liquid": PhaseRecord(
                    "reactor_liquid",
                    sample.sample_id,
                    "liquid",
                    sample.volume_L,
                    sample.species_amounts_mol,
                    solvents=SolventInventory.pure(sample.solvent, sample.volume_L),
                )
            }
        ),
        SpeciesLedger(),
        EquipmentLedger(
            {
                "batch_reactor": EquipmentRecord(
                    "batch_reactor",
                    "batch_reactor",
                    sample.sample_id,
                    settings={
                        "solvent": sample.solvent,
                        "solvent_volume_L": sample.volume_L,
                        "reaction_stopped": True,
                    },
                ),
                "crystallizer": EquipmentRecord(
                    "crystallizer",
                    "crystallizer",
                    sample.sample_id,
                    settings={"dissolved_seed_target_mol": sample.seed_target_mol},
                ),
            }
        ),
        {"reaction_chemistry_stopped": 1.0},
    )


def _portion(sample: StoredSample, fraction: float, target: str) -> StoredSample:
    contents = _scale(_contents(sample), fraction, target)
    return replace(
        sample,
        sample_id=target,
        volume_L=sample.volume_L * fraction,
        species_amounts_mol=contents.phases.total_amounts_mol(),
        seed_target_mol=sample.seed_target_mol * fraction,
        contents=contents,
        retired=False,
    )


def _sum(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left.get(key, 0.0) + right.get(key, 0.0) for key in left.keys() | right.keys()}


def _reference_amounts(
    sources: dict[str, dict[str, float]], shares: dict[str, float]
) -> dict[str, float]:
    total: dict[str, float] = {}
    for source, amounts in sources.items():
        total = _sum(total, {s: n * shares.get(source, 0) for s, n in amounts.items()})
    return total


def _occupied(sample: StoredSample) -> bool:
    return sample.volume_L > 0 or any(sample.species_amounts_mol.values())


def _merge(destination: StoredSample, incoming: StoredSample) -> StoredSample:
    if not _occupied(destination):
        return replace(
            incoming,
            capacity_L=destination.capacity_L,
            collection_operation="transfer_material",
        )
    left, right = _contents(destination), _contents(incoming)
    if set(left.phases.phases) != set(right.phases.phases):
        raise ValueError("Mixing requires matching phase topology; resuspend before mixing")
    # Liquid and slurry heat capacities are solvent dominated in this domain.
    volume = destination.volume_L + incoming.volume_L
    left_capacity = sum(
        p.solvents.volume_L * p.solvents.linear_property(HEAT_CAPACITY_RATIOS)
        for p in left.phases.phases.values()
    )
    right_capacity = sum(
        p.solvents.volume_L * p.solvents.linear_property(HEAT_CAPACITY_RATIOS)
        for p in right.phases.phases.values()
    )
    weight = (
        right_capacity / (left_capacity + right_capacity) if left_capacity + right_capacity else 0.5
    )
    phases = {}
    for key, a in left.phases.phases.items():
        b = right.phases.phases[key]
        if (a.phase_type, a.selected, a.settled) != (b.phase_type, b.selected, b.settled):
            raise ValueError("Material phases have incompatible separation state")
        phases[key] = replace(
            a,
            volume_L=a.volume_L + b.volume_L,
            species_amounts_mol=_sum(a.species_amounts_mol, b.species_amounts_mol),
            solvents=a.solvents + b.solvents,
        )
    equipment: dict[str, EquipmentRecord] = {}
    for key in left.equipment.equipment.keys() | right.equipment.equipment.keys():
        left_record = left.equipment.equipment.get(key)
        right_record = right.equipment.equipment.get(key)
        if left_record is None or right_record is None:
            record = left_record or right_record
            assert record is not None
            equipment[key] = record
            continue
        both_catalyzed = all(
            float(record.settings.get("catalyst_amount_mol", 0)) > 0
            for record in (left_record, right_record)
        )
        checked_settings = (
            ("catalyst", "crystals_filtered") if both_catalyzed else ("crystals_filtered",)
        )
        for setting in checked_settings:
            if left_record.settings.get(setting) != right_record.settings.get(setting):
                raise ValueError(f"Mixing requires matching {setting}")
        settings = dict(left_record.settings)
        if (
            float(right_record.settings.get("catalyst_amount_mol", 0)) > 0
            and "catalyst" in right_record.settings
        ):
            settings["catalyst"] = right_record.settings["catalyst"]
        for k, value in right_record.settings.items():
            if _extensive(k) and isinstance(value, (float, int)):
                settings[k] = float(settings.get(k, 0)) + value
            elif k not in settings:
                settings[k] = value
        cohorts = (
            *left_record.settings.get("population_cohorts", ()),
            *right_record.settings.get("population_cohorts", ()),
        )
        if cohorts:
            settings.update(population_settings(tuple(tuple(c) for c in cohorts)))
        equipment[key] = replace(left_record, settings=settings)
    # Different processed-material metrics must be remeasured, not averaged as truth.
    # Extensive reference amounts add. Equal intensive state survives splitting/reunion.
    metrics = {
        k: left.metrics.get(k, 0) + right.metrics.get(k, 0)
        for k in left.metrics.keys() | right.metrics.keys()
        if _extensive(k)
    }
    metrics.update(
        {k: v for k, v in left.metrics.items() if not _extensive(k) and right.metrics.get(k) == v}
    )
    sources = {**left.reference_sources, **right.reference_sources}
    shares = {
        k: min(1.0, left.reference_shares.get(k, 0) + right.reference_shares.get(k, 0))
        for k in sources
    }
    # Separate filtrate/cake products refer to the same original charge. A reunion
    # must not count that reference twice; physical inventories are always additive.
    reference_amounts = (
        _reference_amounts(sources, shares)
        if sources
        else _sum(left.species.initial_amounts_mol, right.species.initial_amounts_mol)
    )
    contents = MaterialContents(
        PhaseLedger(phases),
        SpeciesLedger(
            {**left.species.species_roles, **right.species.species_roles},
            reference_amounts,
        ),
        EquipmentLedger(equipment),
        metrics,
        sources,
        shares,
    )
    return replace(
        destination,
        contents=contents,
        volume_L=volume,
        species_amounts_mol=contents.phases.total_amounts_mol(),
        temperature_K=destination.temperature_K * (1 - weight) + incoming.temperature_K * weight,
        seed_target_mol=destination.seed_target_mol + incoming.seed_target_mol,
        lineage=tuple(dict.fromkeys((*destination.lineage, *incoming.lineage))),
    )


def _install(state: WorldState, sample: StoredSample) -> WorldState:
    contents = _scale(_contents(sample), 1.0, state.vessel_id)
    equipment = {
        k: replace(v, status="stale") if k.startswith("instrument:") else v
        for k, v in (state.equipment or EquipmentLedger()).equipment.items()
        if k not in MATERIAL_EQUIPMENT
    }
    equipment.update(contents.equipment.equipment)
    return state.replace(
        volume_L=sample.volume_L,
        species_amounts=sample.species_amounts_mol,
        phases=contents.phases,
        species=contents.species,
        temperature_K=sample.temperature_K,
        quenched=sample.quenched,
        equipment=EquipmentLedger(equipment),
        process=replace(
            state.process or ProcessLedger(),
            metrics=contents.metrics,
            last_observation={},
            last_observed_mask={},
        ),
    )


def _available_samples(state: WorldState) -> dict[str, StoredSample]:
    samples = state.samples.samples.copy()
    try:
        samples["active"] = _active(state)
    except ValueError:
        # Some process models expose a selected-receiver volume and a larger
        # multi-vessel phase ledger. Those need R25 ports, not whole-batch routing.
        return samples
    return samples


def routing_error(state: WorldState, operation: str, action: dict[str, Any]) -> str | None:
    """Shared public validation and runtime admission, without mutating inventories."""
    try:
        if state.terminated:
            raise ValueError("Cannot route material after termination")
        if operation == "create_container":
            name = container_id(action.get("container"))
            if name not in CONTAINER_IDS[1:17] or name in state.samples.samples:
                raise ValueError("Container address must be an unused disposable slot")
            volume = action.get("capacity_L")
            if isinstance(volume, bool) or not isinstance(volume, (int, float)):
                raise ValueError("Container capacity must be numeric")
            if not isfinite(volume) or not 0.0001 <= volume <= 0.1:
                raise ValueError("Container capacity outside [0.0001, 0.1] L")
        else:
            source = container_id(action.get("source_container"))
            target = container_id(action.get("destination_container"))
            if source == target:
                raise ValueError("Source and destination must differ")
            samples = _available_samples(state)
            if source not in samples or target not in samples:
                raise ValueError("Create the destination container before transferring")
            a, b = samples[source], samples[target]
            if a.retired or b.retired or not _occupied(a):
                raise ValueError(
                    "Source must contain material; retired containers cannot be reused"
                )
            if not a.quenched or (_occupied(b) and not b.quenched):
                raise ValueError("Stored material must be quenched")
            fraction = action.get("transfer_fraction")
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ValueError("Transfer fraction must be numeric")
            if not isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("Transfer fraction outside (0, 1]")
            mixing = action.get("mixing")
            if isinstance(mixing, bool) or mixing not in (0, 1):
                raise ValueError("Mixing must be 0 (empty only) or 1 (explicit mixture)")
            if _occupied(b) and mixing != 1:
                raise ValueError("Occupied destination requires explicit mixing consent")
            if b.volume_L + a.volume_L * fraction > b.capacity_L + 1e-12:
                raise ValueError("Transfer exceeds destination capacity")
            for sample in (a, b):
                if _occupied(sample) and not set(_contents(sample).phases.phases) <= {
                    "reactor_liquid",
                    "solid",
                    "mother_liquor",
                    "cake_liquor",
                }:
                    raise ValueError(
                        "Routing supports quenched reactor liquids and crystal slurries"
                    )
            _merge(b, _portion(a, fraction, target))
    except ValueError as exc:
        return str(exc)
    return None


def routing_available(state: WorldState, operation: str) -> bool:
    if state.terminated:
        return False
    if operation == "create_container":
        return any(name not in state.samples.samples for name in CONTAINER_IDS[1:17])
    return bool(material_routes(state, first_only=True))


def material_routes(state: WorldState, *, first_only: bool = False) -> list[dict[str, Any]]:
    """Public addresses, capacity bounds and mixing constraints, without compositions."""
    if state.terminated:
        return []
    routes = []
    samples = _available_samples(state)
    for source, a in samples.items():
        if not a.quenched or a.retired or not _occupied(a):
            continue
        for target, b in samples.items():
            if source == target or b.retired or (not b.quenched and _occupied(b)):
                continue
            available = b.capacity_L - b.volume_L
            if a.volume_L > 0 and available <= 0:
                continue
            fraction = min(1.0, available / a.volume_L) if a.volume_L else 1.0
            if fraction <= 0:
                continue
            if (
                routing_error(
                    state,
                    "transfer_material",
                    {
                        "source_container": source,
                        "destination_container": target,
                        "transfer_fraction": fraction,
                        "mixing": 1,
                    },
                )
                is None
            ):
                routes.append(
                    {
                        "source_container": CONTAINER_IDS.index(source),
                        "destination_container": CONTAINER_IDS.index(target),
                        "maximum_fraction": fraction,
                        "minimum_mixing": int(_occupied(b)),
                    }
                )
                if first_only:
                    return routes
    return routes


class ChemWorldMaterialRoutingServices:
    def create_container(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        error = routing_error(state, "create_container", action)
        if error:
            raise ValueError(error)
        name = container_id(action["container"])
        sample = StoredSample(
            name,
            name,
            "create_container",
            state.ledger.time_s,
            0,
            {},
            state.temperature_K,
            0,
            True,
            capacity_L=float(action["capacity_L"]),
        )
        return state.replace(
            samples=replace(
                state.samples.append(sample), containers_used=state.samples.containers_used + 1
            ),
            ledger=state.ledger.with_updates(cost=state.ledger.cost + 0.02),
        )

    def transfer_material(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        error = routing_error(state, "transfer_material", action)
        if error:
            raise ValueError(error)
        source = container_id(action["source_container"])
        target = container_id(action["destination_container"])
        fraction = float(action["transfer_fraction"])
        samples = _available_samples(state)
        a, b = samples[source], samples[target]
        incoming = _portion(a, fraction, target)
        incoming = replace(
            incoming,
            source_vessel_id=source,
            lineage=a.lineage or (a.sample_id,),
            collected_at_s=state.ledger.time_s,
        )
        remaining = _portion(a, 1 - fraction, source)
        samples[source] = replace(remaining, retired=(fraction == 1 and source != "active"))
        samples[target] = _merge(b, incoming)
        event = {
            "transfer_id": len(state.samples.transfers) + 1,
            "source": source,
            "destination": target,
            "fraction": fraction,
            "volume_L": incoming.volume_L,
            "time_s": state.ledger.time_s,
            "mixing": int(action["mixing"]),
            "lineage": list(incoming.lineage),
            "new_disposable_tool": True,
        }
        active = samples.pop("active", None)
        touches_active = "active" in (source, target)
        next_state = state
        active_lineage = state.samples.active_lineage
        reference_sources = state.samples.active_reference_sources
        reference_shares = state.samples.active_reference_shares
        if touches_active:
            assert active is not None
            next_state = _install(state, active)
            active_lineage = active.lineage if _occupied(active) else ()
            reference_sources = _contents(active).reference_sources
            reference_shares = _contents(active).reference_shares
        return next_state.replace(
            samples=replace(
                state.samples,
                samples=samples,
                transfers=(*state.samples.transfers, event),
                transfer_tools_used=state.samples.transfer_tools_used + 1,
                active_lineage=active_lineage,
                active_reference_sources=reference_sources,
                active_reference_shares=reference_shares,
            ),
            ledger=state.ledger.with_updates(cost=state.ledger.cost + 0.005),
        )
