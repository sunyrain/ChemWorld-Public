"""Original fixed catalytic slots attached to the original finite gas network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    catalytic_inventory_key,
    catalytic_system,
    make_catalytic_slot,
    validate_catalytic_slot,
)
from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.surface_import_rates import ImportedSurfaceReaction


@dataclass(frozen=True)
class GasSurfaceAttachment:
    """One fixed complete imported ideal surface; dynamic amounts live in its slot."""

    vessel_id: str
    owner_id: str
    mechanism: CatalyticMechanismSpec
    area_m2: float
    heat_capacity_J_K: float = 0.0

    def __post_init__(self) -> None:
        if any(not isinstance(s, str) or not s for s in (self.vessel_id, self.owner_id)):
            raise ValueError("surface attachment requires vessel and original equipment identities")
        for value, positive in ((self.area_m2, True), (self.heat_capacity_J_K, False)):
            if (
                isinstance(value, bool)
                or not isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                raise ValueError("surface area must be positive and support capacity nonnegative")
        mechanism = self.mechanism
        if (
            not mechanism.site_species_complete
            or len(mechanism.site_types) != 1
            or set(mechanism.site_densities_mol_m2) != set(mechanism.site_types)
            or any(not isinstance(r, ImportedSurfaceReaction) for r in mechanism.reactions)
            or any(any(row) for row in mechanism.interaction_matrix_J_mol)
            or any(
                mechanism.species_by_id[s].surface_declaration is None for s in mechanism.adsorbates
            )
        ):
            raise ValueError("gas attachment requires one complete imported ideal surface")

    def make_slot(self, amounts_mol: Mapping[str, float]) -> StateSlot:
        return make_catalytic_slot(
            self.owner_id,
            self.vessel_id,
            self.mechanism,
            capacities_mol={
                site: density * self.area_m2
                for site, density in self.mechanism.site_densities_mol_m2.items()
            },
            initial_amounts_mol=amounts_mol,
            heat_capacity_J_K=self.heat_capacity_J_K,
        )

    def validate_slot(self, slot: StateSlot) -> None:
        validate_catalytic_slot(slot, self.mechanism)
        expected = self.make_slot({s: slot.values[s] for s in self.mechanism.adsorbates})
        if (
            slot.spec != expected.spec
            or slot.location_id != expected.location_id
            or slot.values != expected.values
        ):
            raise ValueError("surface slot must retain its declared owner, site stock and capacity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "owner_id": self.owner_id,
            "mechanism": self.mechanism.to_dict(),
            "area_m2": self.area_m2,
            "heat_capacity_J_K": self.heat_capacity_J_K,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GasSurfaceAttachment:
        values = dict(raw)
        values["mechanism"] = CatalyticMechanismSpec.from_dict(values["mechanism"])
        return cls(**values)


def gas_phase_network(network: ReactionNetworkSpec) -> ReactionNetworkSpec:
    """Project shared identity metadata onto explicit gas species, without new identities."""
    if all(s.phase == "gas" for s in network.species):
        return network
    if any(s.phase not in {"gas", "surface"} for s in network.species):
        raise ValueError("finite gas/surface network cannot consume other phases")
    gas = tuple(s for s in network.species if s.phase == "gas")
    names = {s.species_id for s in gas}
    if any(set(r.stoichiometry) - names for r in network.reactions):
        raise ValueError("bulk gas reactions cannot consume surface inventory")
    return replace(network, species=gas)


def network_with_gas_surfaces(
    network: ReactionNetworkSpec, attachments: Sequence[GasSurfaceAttachment]
) -> ReactionNetworkSpec:
    if not attachments:
        return network
    gas = gas_phase_network(network)
    table = {s.species_id: s for s in gas.species}
    thermochemistry = gas.imported_gas_thermochemistry
    for attachment in attachments:
        mechanism = attachment.mechanism
        if set(mechanism.gas_species) != set(gas.species_ids):
            raise ValueError("surface and bulk gas must share exactly the same gas species")
        for name in mechanism.gas_species:
            species = mechanism.species_by_id[name]
            if (
                table[name] != species.species
                or name not in thermochemistry
                or thermochemistry[name].to_dict() != species.gas_model.to_dict()
            ):
                raise ValueError(
                    "surface and bulk gas require identical identities and thermochemistry"
                )
        for name in mechanism.adsorbates:
            identity = mechanism.species_by_id[name].species
            if name in table and table[name] != identity:
                raise ValueError(
                    "surface species identity conflicts with the shared identity table"
                )
            table[name] = identity
    result = replace(gas, species=tuple(table.values()))
    if any(s.phase != "gas" for s in network.species) and network.species != result.species:
        raise ValueError("shared identity table differs from declared gas/surface species")
    return result


class GasSurfaceContext:
    """Transient numerical mapping of original owned slots, never persistent state."""

    def __init__(self, spec: Any, network: ReactionNetworkSpec, slots: Sequence[StateSlot]):
        from chemworld.physchem.gas_reactor_network import gas_inventory_key

        self.spec = spec
        self.network = network_with_gas_surfaces(network, spec.surfaces)
        self.gas = gas_phase_network(self.network)
        by_owner = {s.spec.owner_id: s for s in slots}
        if len(by_owner) != len(slots) or set(by_owner) != {a.owner_id for a in spec.surfaces}:
            raise ValueError("surface slots must cover each declared original owner exactly once")
        self.slots = {a.vessel_id: by_owner[a.owner_id] for a in spec.surfaces}
        self.attachments = {a.vessel_id: a for a in spec.surfaces}
        self.models: dict[str, Any] = {}
        self.processes: dict[str, Any] = {}
        self.identities: dict[str, dict[str, str]] = {}
        self.laws: dict[str, dict[str, float]] = {}
        energy = {e.vessel_id: e for e in spec.energy_nodes}
        for node in spec.nodes:
            name = node.vessel_id
            support = energy[name].support_heat_capacity_J_K if energy else 0.0
            reference = energy[name].reference_temperature_K if energy else 298.15
            identities = {s: gas_inventory_key(name, s) for s in self.gas.species_ids}
            if name in self.attachments:
                attachment, slot = self.attachments[name], self.slots[name]
                attachment.validate_slot(slot)
                local = catalytic_system(
                    attachment.mechanism, (slot,), support_heat_capacity_J_K=support
                )
                # Ideal surface has no excess interactions; preserve the original
                # thermal model and original support reference in both phases.
                model: Any = local.thermodynamics
                base = getattr(model, "base", model)
                shifted = replace(base, reference_temperature_K=reference)
                model = replace(model, base=shifted) if hasattr(model, "base") else shifted
                for s in attachment.mechanism.adsorbates:
                    key = catalytic_inventory_key(slot, s)
                    identities[key] = key
                self.models[name], self.processes[name] = model, local.processes
                for site in attachment.mechanism.site_types:
                    self.laws[f"sites:{slot.spec.slot_id}:{site}"] = {
                        catalytic_inventory_key(slot, s): attachment.mechanism.site_occupancies[s]
                        for s in attachment.mechanism.adsorbates
                        if attachment.mechanism.adsorbate_sites[s] == site
                    }
                local_species = {key: value.species for key, value in base.species.items()}
            else:
                self.models[name] = IdealGasThermodynamics(
                    self.gas.imported_gas_thermochemistry, support, reference
                )
                self.processes[name] = ()
                local_species = {s.species_id: s for s in self.gas.species}
            self.identities[name] = identities
            for local_key, species in local_species.items():
                for element, count in species.composition.items():
                    self.laws.setdefault(element, {})[identities[local_key]] = count
        self.inventory_keys = tuple(key for row in self.identities.values() for key in row.values())

    def view(self, state: Any, node: Any) -> CoupledStateView:
        from chemworld.physchem.gas_reactor_network import _key

        return CoupledStateView(
            {
                key: state.amounts_mol[global_key]
                for key, global_key in self.identities[node.vessel_id].items()
            },
            state.coordinates.get(_key("node_temperature_K", node.vessel_id), state.temperature_K),
            node.volume_L,
            state.time_s,
        )

    def chemistry(self, state: Any, node: Any, gas: ProcessContribution) -> ProcessContribution:
        amounts, progress = dict(gas.amount_rates_mol_s), dict(gas.reaction_progress_mol_s)
        view = self.view(state, node)
        for process in self.processes[node.vessel_id]:
            contribution = process.evaluate(view)
            for key, value in contribution.amount_rates_mol_s.items():
                amounts[key] = amounts.get(key, 0.0) + value
            progress.update(contribution.reaction_progress_mol_s)
        return ProcessContribution(amounts, progress)

    def validate(self, state: Any) -> None:
        for node in self.spec.nodes:
            view = self.view(state, node)
            self.models[node.vessel_id].validate_state(view)
            for process in self.processes[node.vessel_id]:
                if process.validate_state is not None:
                    process.validate_state(view)

    def amounts(self) -> dict[str, float]:
        return {
            catalytic_inventory_key(self.slots[name], s): self.slots[name].values[s]
            for name, attachment in self.attachments.items()
            for s in attachment.mechanism.adsorbates
        }

    def project(self, state: Any, slot: StateSlot) -> StateSlot:
        attachment = self.attachments[slot.location_id]
        return replace(
            slot,
            values={
                **slot.values,
                **{
                    s: state.amounts_mol[catalytic_inventory_key(slot, s)]
                    for s in attachment.mechanism.adsorbates
                },
            },
        )


__all__ = ["GasSurfaceAttachment"]
