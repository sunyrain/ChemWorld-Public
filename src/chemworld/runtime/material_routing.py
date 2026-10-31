"""Conservative, finite-address routing of quenched liquids and crystal slurries.

Storage is sealed/isothermal with no chemistry or phase evolution. Moving an
equal fraction of every phase models a representative homogenized aliquot.
Tools and emptied containers are single use; mixing must be explicitly requested.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState, equipment_settings
from chemworld.foundation.material_properties import (
    MATERIAL_FIELDS,
    MATERIAL_METRICS,
    extract_material_properties,
    install_material_properties,
    merge_fields,
    scale_fields,
)
from chemworld.foundation.samples import CONTAINER_IDS, MaterialContents, StoredSample
from chemworld.foundation.solvents import HEAT_CAPACITY_RATIOS, SolventInventory
from chemworld.foundation.state_ledgers import (
    PhaseLedger,
    PhaseRecord,
    ProcessLedger,
    SpeciesLedger,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime.full_process_contract import population_settings

if TYPE_CHECKING:
    from chemworld.foundation.balances import MaterialPacket
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from chemworld.world.physical_world import CompositionPhysics

ROUTING_OPERATIONS = ("create_container", "transfer_material")


def container_id(value: Any) -> str:
    if isinstance(value, str) and value in CONTAINER_IDS:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(CONTAINER_IDS):
        raise ValueError("Unknown container address")
    return CONTAINER_IDS[value]


def _population(properties: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    crystal = properties.get("crystallizer", {})
    if "population_cohorts" in crystal:
        crystal.update(population_settings(tuple(tuple(c) for c in crystal["population_cohorts"])))
    return properties


def _scale(contents: MaterialContents, fraction: float, vessel_id: str) -> MaterialContents:
    if not isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Material fraction must be in [0, 1]")
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
    return MaterialContents(
        phases,
        replace(
            contents.species,
            initial_amounts_mol={
                s: n * fraction for s, n in contents.species.initial_amounts_mol.items()
            },
        ),
        _population(
            {
                name: scale_fields(values, MATERIAL_FIELDS[name], fraction)
                for name, values in contents.properties.items()
            }
        ),
        scale_fields(contents.metrics, MATERIAL_METRICS, fraction),
        contents.reference_sources,
        {k: v * fraction for k, v in contents.reference_shares.items()},
        contents.state_slots.portion(fraction, vessel_id),
        contents.liquid_heat_capacity_J_L_K,
        contents.mechanism_id,
    )


def _active(state: WorldState) -> StoredSample:
    heat_capacity = state.liquid_heat_capacity_J_L_K
    if heat_capacity is None:
        raise ValueError("Formal workspaces cannot be routed as physical material")
    # Construction has already populated the optional ledger inputs.
    species = cast(SpeciesLedger, state.species)
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
    initial = species.initial_amounts_mol
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
            cast(PhaseLedger, state.phases),
            species,
            extract_material_properties(state.equipment),
            {}
            if state.process is None
            else {k: v for k, v in state.process.metrics.items() if k in MATERIAL_METRICS},
            sources,
            shares,
            state.state_slots.material(),
            heat_capacity,
            str(state.metadata.get("mechanism_hash", state.metadata.get("mechanism_id", ""))),
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
        {
            "batch_reactor": {
                "solvent": sample.solvent,
                "solvent_volume_L": sample.volume_L,
                "reaction_stopped": True,
            },
            "crystallizer": {"dissolved_seed_target_mol": sample.seed_target_mol},
        },
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
    return (
        sample.volume_L > 0
        or any(sample.species_amounts_mol.values())
        or (sample.contents is not None and sample.contents.state_slots.has_inventory())
    )


def _merge(destination: StoredSample, incoming: StoredSample) -> StoredSample:
    if not _occupied(destination):
        return replace(
            incoming,
            capacity_L=destination.capacity_L,
            collection_operation="transfer_material",
        )
    left = _scale(_contents(destination), 1.0, destination.sample_id)
    right = _scale(_contents(incoming), 1.0, destination.sample_id)
    if left.mechanism_id and right.mechanism_id and left.mechanism_id != right.mechanism_id:
        raise ValueError("Material mechanisms are incompatible")
    if set(left.phases.phases) != set(right.phases.phases):
        raise ValueError("Mixing requires matching phase topology; resuspend before mixing")
    if left.liquid_heat_capacity_J_L_K != right.liquid_heat_capacity_J_L_K:
        raise ValueError("Mixing requires a common liquid heat-capacity model")
    # U = C*T for the declared constant-C material model. Supports contribute
    # their own J/K; device heat capacity stays outside the transferred material.
    volume = destination.volume_L + incoming.volume_L
    left_capacity = sum(
        p.solvents.volume_L
        * p.solvents.linear_property(HEAT_CAPACITY_RATIOS)
        * left.liquid_heat_capacity_J_L_K
        for p in left.phases.phases.values()
    )
    left_capacity += left.state_slots.heat_capacity()
    right_capacity = (
        sum(
            p.solvents.volume_L
            * p.solvents.linear_property(HEAT_CAPACITY_RATIOS)
            * right.liquid_heat_capacity_J_L_K
            for p in right.phases.phases.values()
        )
        + right.state_slots.heat_capacity()
    )
    if not left_capacity + right_capacity and destination.temperature_K != incoming.temperature_K:
        raise ValueError("Thermal mixing requires a declared heat capacity")
    weight = (
        right_capacity / (left_capacity + right_capacity) if left_capacity + right_capacity else 0
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
    properties = _population(
        {
            name: merge_fields(
                left.properties.get(name, {}), right.properties.get(name, {}), MATERIAL_FIELDS[name]
            )
            for name in left.properties.keys() | right.properties.keys()
        }
    )
    metrics = merge_fields(left.metrics, right.metrics, MATERIAL_METRICS)
    slots = left.state_slots.merge(right.state_slots)
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
        properties,
        metrics,
        sources,
        shares,
        slots,
        left.liquid_heat_capacity_J_L_K,
        left.mechanism_id or right.mechanism_id,
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
    equipment = install_material_properties(state.equipment, contents.properties, state.vessel_id)
    slots = StateSlots((*state.state_slots.stationary().slots, *contents.state_slots.slots))
    stationary_capacity = state.state_slots.stationary().heat_capacity()
    material_capacity = contents.state_slots.heat_capacity() + sum(
        p.solvents.volume_L
        * p.solvents.linear_property(HEAT_CAPACITY_RATIOS)
        * contents.liquid_heat_capacity_J_L_K
        for p in contents.phases.phases.values()
    )
    temperature = sample.temperature_K
    if stationary_capacity:
        temperature = (
            stationary_capacity * state.temperature_K + material_capacity * temperature
        ) / (stationary_capacity + material_capacity)
    return state.replace(
        volume_L=sample.volume_L,
        species_amounts=sample.species_amounts_mol,
        phases=contents.phases,
        species=contents.species,
        temperature_K=temperature,
        quenched=sample.quenched,
        equipment=equipment,
        state_slots=slots,
        liquid_heat_capacity_J_L_K=contents.liquid_heat_capacity_J_L_K,
        process=replace(
            state.process or ProcessLedger(),
            metrics=contents.metrics,
            last_observation_record=None,
        ),
    )


def _available_samples(state: WorldState) -> dict[str, StoredSample]:
    samples = state.samples.samples.copy()
    if state.phase == "formal":
        return samples
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
    def __init__(self, physics: CompositionPhysics | None = None) -> None:
        self.physics = physics

    def uses_native_packets(self, state: WorldState) -> bool:
        from chemworld.runtime.aqueous_mineral import aqueous_record
        from chemworld.runtime.aqueous_precipitation import precipitation_record
        from chemworld.runtime.condensed_material_routing import background_model
        from chemworld.runtime.humid_air_material import humid_air_model

        return self.physics is not None and (
            state.phase == "gas"
            or humid_air_model(state, self.physics) is not None
            or aqueous_record(state) is not None
            or precipitation_record(state) is not None
            or state.vessel_id in self.physics.models_of("flash")
            or state.vessel_id in self.physics.models_of("liquid")
            or background_model(state, self.physics) is not None
        )

    def feed_additions(self, state: WorldState, action: dict[str, Any]) -> dict[str, float]:
        from chemworld.runtime.thermal_material_routing import feed_packet

        return feed_packet(state, self.physics, action)[1]

    def executed_feed(
        self, before: WorldState, after: WorldState, action: dict[str, Any],
    ) -> MaterialPacket:
        from chemworld.foundation.balances import MaterialPacket

        # Material accounting selects this branch through uses_native_packets.
        physics = cast("CompositionPhysics", self.physics)
        source_id = physics.feed_sources[action["component"]]
        receipt = after.metadata.get("last_material_packet")
        if not isinstance(receipt, dict):
            raise ValueError("finite feed requires its executed material packet")
        packet = MaterialPacket.from_dict(receipt)
        if (packet.source, packet.destination) != (source_id, before.vessel_id):
            raise ValueError("executed feed packet has the wrong endpoints")
        packet.validate_representative(
            before.inactive_vessels[source_id].species_amounts,
            requested_mol=action["amount_mol"],
        )
        packet.check_debit(
            before.inactive_vessels[source_id].species_amounts,
            after.inactive_vessels[source_id].species_amounts,
        )
        return packet

    def add_finite_feed(self, state: WorldState, action: dict[str, Any]) -> ModelProviderResult:
        from chemworld.runtime.kernel_contracts import ModelProviderResult
        from chemworld.runtime.thermal_material_routing import add_finite_feed

        candidate = add_finite_feed(state, self.physics, action)
        return ModelProviderResult(
            outputs={},
            state=candidate,
            diagnostics={"material_packet": candidate.metadata["last_material_packet"]},
        )

    def route_material(
        self, state: WorldState, action: dict[str, Any],
    ) -> WorldState | ModelProviderResult:
        from chemworld.runtime.aqueous_precipitation import (
            precipitation_record,
            route_precipitation_packet,
        )
        from chemworld.runtime.component_network import route_material
        from chemworld.runtime.kernel_contracts import ModelProviderResult
        from chemworld.runtime.thermal_material_routing import route_gas_packet

        if precipitation_record(state) is not None:
            return route_precipitation_packet(state, self.physics, action)
        if self.uses_native_packets(state):
            candidate = route_gas_packet(state, self.physics, action)
            return ModelProviderResult(
                outputs={},
                state=candidate,
                diagnostics={"material_packet": candidate.metadata["last_material_packet"]},
            )
        return route_material(state, action)

    def feed_error(self, state: WorldState, action: dict[str, Any]) -> str | None:
        try:
            self.add_finite_feed(state, action)
        except (ArithmeticError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            return str(exc)
        return None

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
        if (
            self.physics is not None
            and state.vessel_id in self.physics.models_of("crystal")
            and (_occupied(b) or target == "active")
            and abs(a.temperature_K - b.temperature_K) > 1e-9
        ):
            raise ValueError("crystal container routing currently requires equal temperatures")
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
        touches_active = "active" in (source, target)
        next_state = state
        active_lineage = state.samples.active_lineage
        reference_sources = state.samples.active_reference_sources
        reference_shares = state.samples.active_reference_shares
        if touches_active:
            # Both selected addresses were resolved above; the active one exists.
            active = samples["active"]
            next_state = _install(state, active)
            active_lineage = active.lineage if _occupied(active) else ()
            reference_sources = _contents(active).reference_sources
            reference_shares = _contents(active).reference_shares
        samples.pop("active", None)
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
