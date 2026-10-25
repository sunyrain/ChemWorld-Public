"""Declared reversible finite-site chemistry on original StateSlot/CoupledProcess."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, fsum, isclose, prod
from types import MappingProxyType
from typing import Any

from chemworld.foundation.state_ownership import (
    CapacityConstraint,
    InventoryInteractionSpec,
    StateFieldSpec,
    StateSlot,
    StateSlotSpec,
)
from chemworld.physchem.catalysis_descriptors import FreeEnergyBEP, finite_number, validate_source
from chemworld.physchem.catalysis_thermochemistry import (
    CatalyticSpeciesThermo,
    CatalyticThermodynamics,
)
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.hindered_adsorbate import HinderedAdsorbateThermo
from chemworld.physchem.inventory_interactions import (
    InventoryInteraction,
    with_inventory_interactions,
)
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.surface_import_rates import ImportedSurfaceReaction, imported_surface_rates

CATALYTIC_STATE_TYPE = "native_declared_catalysis"


def vacancy_key(site: str) -> str:
    return "vacancy:" + site


def capacity_field(site: str) -> str:
    return "capacity:" + site


def catalytic_inventory_key(slot: StateSlot, species_id: str) -> str:
    return f"{slot.spec.slot_id}:{species_id}"


@dataclass(frozen=True)
class CatalyticReactionSpec:
    reaction_id: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate_site: str
    attempt_frequency_s_inv: float
    bep: FreeEnergyBEP
    source: DataSourceProvenance
    rate_unit: str = "s^-1"
    reservoir_reactants: Mapping[str, int] = field(default_factory=dict)
    reservoir_products: Mapping[str, int] = field(default_factory=dict)
    electron_stoichiometry: int = 0

    def __post_init__(self) -> None:
        for name in ("reactants", "products", "reservoir_reactants", "reservoir_products"):
            values = dict(getattr(self, name))
            if (not values and name in {"reactants", "products"}) or any(
                not key or type(v) is not int or v <= 0 for key, v in values.items()
            ):
                raise ValueError(
                    "elementary catalytic sides require positive integer stoichiometry"
                )
            object.__setattr__(self, name, MappingProxyType(values))
        if type(self.electron_stoichiometry) is not int:
            raise ValueError("external electron stoichiometry must be an explicit integer")
        if (set(self.reactants) | set(self.products)) & (
            set(self.reservoir_reactants) | set(self.reservoir_products)
        ):
            raise ValueError("a participant cannot also be an external reservoir participant")
        validate_source(self.source)
        finite_number(self.attempt_frequency_s_inv, "turnover prefactor")
        if (
            not self.reaction_id
            or not self.rate_site
            or self.attempt_frequency_s_inv <= 0
            or self.reactants == self.products
            or self.rate_unit != "s^-1"
        ):
            raise ValueError(
                "reaction requires net edits, identity and explicit positive s^-1 prefactor"
            )

    @property
    def stoichiometry(self) -> dict[str, int]:
        reactants = {**self.reactants, **self.reservoir_reactants}
        products = {**self.products, **self.reservoir_products}
        return {
            k: products.get(k, 0) - reactants.get(k, 0)
            for k in products.keys() | reactants.keys()
            if products.get(k, 0) != reactants.get(k, 0)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "rate_site": self.rate_site,
            "attempt_frequency_s_inv": self.attempt_frequency_s_inv,
            "rate_unit": self.rate_unit,
            **(
                {"reservoir_reactants": dict(self.reservoir_reactants)}
                if self.reservoir_reactants
                else {}
            ),
            **(
                {"reservoir_products": dict(self.reservoir_products)}
                if self.reservoir_products
                else {}
            ),
            **(
                {"electron_stoichiometry": self.electron_stoichiometry}
                if self.electron_stoichiometry
                else {}
            ),
            "bep": self.bep.to_dict(),
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CatalyticReactionSpec:
        values = dict(raw)
        values["bep"] = FreeEnergyBEP.from_dict(values["bep"])
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)


@dataclass(frozen=True)
class CatalyticMechanismSpec:
    mechanism_id: str
    species: tuple[CatalyticSpeciesThermo, ...]
    site_types: tuple[str, ...]
    adsorbate_sites: Mapping[str, str]
    reactions: tuple[CatalyticReactionSpec | ImportedSurfaceReaction, ...]
    interaction_matrix_J_mol: tuple[tuple[float, ...], ...]
    minimum_temperature_K: float
    maximum_temperature_K: float
    source: DataSourceProvenance
    site_species_complete: bool = False
    site_densities_mol_m2: Mapping[str, float] = field(default_factory=dict)
    site_occupancies: Mapping[str, int] = field(default_factory=dict)
    reservoir_species: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reservoir_species", tuple(self.reservoir_species))
        if len(set(self.reservoir_species)) != len(self.reservoir_species):
            raise ValueError("reservoir species must be unique")
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "site_types", tuple(self.site_types))
        object.__setattr__(self, "adsorbate_sites", MappingProxyType(dict(self.adsorbate_sites)))
        object.__setattr__(self, "reactions", tuple(self.reactions))
        object.__setattr__(
            self, "site_densities_mol_m2", MappingProxyType(dict(self.site_densities_mol_m2))
        )
        if type(self.site_species_complete) is not bool:
            raise ValueError("complete surface configuration mode must be boolean")
        validate_source(self.source)
        ids = [s.species.species_id for s in self.species]
        if (
            not self.mechanism_id
            or not ids
            or len(set(ids)) != len(ids)
            or any(k.startswith(("vacancy:", "capacity:")) or k == "heat_capacity_J_K" for k in ids)
            or not self.site_types
            or len(set(self.site_types)) != len(self.site_types)
            or any(not site for site in self.site_types)
        ):
            raise ValueError(
                "catalytic mechanism requires unique nonreserved species and site identities"
            )
        if {s.species.species_id for s in self.species if s.species.phase == "liquid"} != set(
            self.reservoir_species
        ):
            raise ValueError("real liquid species must be explicitly registered chemostats")
        if not self.gas_species or not self.adsorbates:
            raise ValueError("finite gas/surface profile requires both gas and adsorbed species")
        if set(self.site_occupancies) - set(self.adsorbates) or any(
            type(v) is not int or v not in (1, 2, 3) for v in self.site_occupancies.values()
        ):
            raise ValueError("surface site occupancies require declared species and integers 1/2/3")
        object.__setattr__(
            self,
            "site_occupancies",
            MappingProxyType({k: self.site_occupancies.get(k, 1) for k in self.adsorbates}),
        )
        multidentate = any(v != 1 for v in self.site_occupancies.values())
        if multidentate and (
            not self.site_species_complete
            or len(self.site_types) != 1
            or any(not isinstance(r, ImportedSurfaceReaction) for r in self.reactions)
            or any(
                s.surface_declaration is None for s in self.species if s.species.phase == "surface"
            )
        ):
            raise ValueError(
                "multidentate sites require one complete imported ideal surface; "
                "BEP/harmonic/hindered standard states are not declared"
            )
        if set(self.adsorbate_sites) != set(self.adsorbates) or set(
            self.adsorbate_sites.values()
        ) != set(self.site_types):
            raise ValueError("each adsorbate binds one declared populated site class")
        if set(self.site_densities_mol_m2) - set(self.site_types):
            raise ValueError("site density refers to an undeclared site class")
        for density in self.site_densities_mol_m2.values():
            finite_number(density, "site density")
            if density <= 0:
                raise ValueError("site density must be positive")
        for item in self.species:
            if isinstance(item.adsorbate, HinderedAdsorbateThermo):
                site = self.adsorbate_sites[item.species.species_id]
                if self.site_densities_mol_m2.get(site) != item.adsorbate.site_density_mol_m2:
                    raise ValueError("hindered thermochemistry and owned site density disagree")
        if len({s.reference_frame for s in self.species}) != 1:
            raise ValueError("all catalytic species must share a declared caloric reference frame")
        pressures = {
            s.gas_model.reference_pressure_Pa for s in self.species if s.species.phase == "gas"
        }
        if len(pressures) != 1:
            raise ValueError("gas standard pressures must agree")
        for value in (self.minimum_temperature_K, self.maximum_temperature_K):
            finite_number(value, "catalytic temperature domain")
        if not 0 < self.minimum_temperature_K < self.maximum_temperature_K:
            raise ValueError("catalytic temperature domain must be positive and ordered")
        for entry in self.species:
            entry.validate_temperature_interval(
                self.minimum_temperature_K, self.maximum_temperature_K
            )
        interaction = InventoryInteractionSpec(
            self.adsorbates,
            tuple(capacity_field(s) for s in self.site_types),
            self.interaction_matrix_J_mol,
        )
        object.__setattr__(self, "interaction_matrix_J_mol", interaction.coefficients_J_mol)
        if multidentate and any(v != 0 for row in self.interaction_matrix_J_mol for v in row):
            raise ValueError("multidentate nonideal interaction standard states are not declared")
        if not self.reactions or len({r.reaction_id for r in self.reactions}) != len(
            self.reactions
        ):
            raise ValueError("catalysis requires unique declared reaction identities")
        known = (set(ids) - set(self.reservoir_species)) | {
            vacancy_key(site) for site in self.site_types
        }
        species = self.species_by_id
        elements = {e for s in self.species for e in s.species.composition}
        for reaction in self.reactions:
            if isinstance(reaction, ImportedSurfaceReaction):
                if (
                    not self.site_species_complete
                    or set(self.site_densities_mol_m2) != set(self.site_types)
                    or any(v <= 0 for v in self.site_densities_mol_m2.values())
                ):
                    raise ValueError("imported kinetics needs explicit complete sites and density")
                for value in self.site_densities_mol_m2.values():
                    finite_number(value, "surface site density")
                if set(reaction.coverage_dependencies or {}) - set(self.adsorbates) or (
                    reaction.sticking_species is not None
                    and set(reaction.reactants) & set(self.gas_species)
                    != {reaction.sticking_species}
                ):
                    raise ValueError("coverage/sticking terms must refer to declared phase species")
            if (
                set(reaction.reactants) | set(reaction.products)
            ) - known or reaction.rate_site not in self.site_types:
                raise ValueError("catalytic reaction references undeclared species/site")
            if isinstance(reaction, CatalyticReactionSpec):
                external = set(reaction.reservoir_reactants) | set(reaction.reservoir_products)
                if external - set(self.reservoir_species) or (
                    reaction.electron_stoichiometry and not self.reservoir_species
                ):
                    raise ValueError(
                        "external reaction needs registered real reservoirs and electron boundary"
                    )
            elif self.reservoir_species:
                raise ValueError(
                    "imported closed surface kinetics cannot implicitly acquire reservoirs"
                )
            stoich = reaction.stoichiometry
            if self.site_species_complete and any(k.startswith("vacancy:") for k in stoich):
                raise ValueError("complete surface configurations cannot use an implicit vacancy")
            for element in elements:
                if (
                    fsum(
                        n * species[k].species.composition.get(element, 0)
                        for k, n in stoich.items()
                        if k in species
                    )
                    != 0
                ):
                    raise ValueError("catalytic reaction does not conserve elements")
            electron = (
                reaction.electron_stoichiometry
                if isinstance(reaction, CatalyticReactionSpec)
                else 0
            )
            if (
                sum(n * species[k].species.charge for k, n in stoich.items() if k in species)
                - electron
            ):
                raise ValueError("catalytic reaction does not conserve charge")
            for site in self.site_types:
                balance = stoich.get(vacancy_key(site), 0) + sum(
                    stoich.get(k, 0) * self.site_occupancies[k]
                    for k, occupied in self.adsorbate_sites.items()
                    if occupied == site
                )
                if balance:
                    raise ValueError(
                        "catalytic reaction does not conserve each declared site class"
                    )
            if not set(stoich) & set(self.adsorbates):
                raise ValueError("catalytic recipe must change a declared adsorbed inventory")

    @property
    def gas_species(self) -> tuple[str, ...]:
        return tuple(s.species.species_id for s in self.species if s.species.phase == "gas")

    @property
    def adsorbates(self) -> tuple[str, ...]:
        return tuple(s.species.species_id for s in self.species if s.species.phase == "surface")

    @property
    def species_by_id(self) -> dict[str, CatalyticSpeciesThermo]:
        return {s.species.species_id: s for s in self.species}

    @property
    def reference_pressure_Pa(self) -> float:  # noqa: N802 - shared unit-explicit contract
        return self.species_by_id[self.gas_species[0]].gas_model.reference_pressure_Pa

    def coverage_amount_scales_mol(self, capacities_mol: Mapping[str, float]) -> dict[str, float]:
        """Molecular mol per unit site coverage; physical capacity remains mol of sites."""
        if set(capacities_mol) != set(self.site_types):
            raise ValueError("coverage conversion requires the original declared site capacities")
        return {
            k: capacities_mol[self.adsorbate_sites[k]] / self.site_occupancies[k]
            for k in self.adsorbates
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "descriptor-catalysis-v1",
            **(
                {"reservoir_species": list(self.reservoir_species)}
                if self.reservoir_species
                else {}
            ),
            "mechanism_id": self.mechanism_id,
            "species": [s.to_dict() for s in self.species],
            "site_types": list(self.site_types),
            "adsorbate_sites": dict(self.adsorbate_sites),
            "reactions": [r.to_dict() for r in self.reactions],
            "interaction_matrix_J_mol": [list(row) for row in self.interaction_matrix_J_mol],
            "minimum_temperature_K": self.minimum_temperature_K,
            "maximum_temperature_K": self.maximum_temperature_K,
            "source": self.source.to_dict(),
            **({"site_species_complete": True} if self.site_species_complete else {}),
            **(
                {"site_occupancies": dict(self.site_occupancies)}
                if any(v != 1 for v in self.site_occupancies.values())
                else {}
            ),
            **(
                {"site_densities_mol_m2": dict(self.site_densities_mol_m2)}
                if self.site_densities_mol_m2 or self.site_species_complete
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CatalyticMechanismSpec:
        values = dict(raw)
        if values.pop("model") != "descriptor-catalysis-v1":
            raise ValueError("unknown declared catalytic mechanism")
        values["species"] = tuple(CatalyticSpeciesThermo.from_dict(s) for s in values["species"])
        values["reactions"] = tuple(
            ImportedSurfaceReaction.from_dict(r)
            if r.get("model") == "imported-ideal-surface-rate-v1"
            else CatalyticReactionSpec.from_dict(r)
            for r in values["reactions"]
        )
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)


def make_catalytic_slot(
    owner_id: str,
    location_id: str,
    spec: CatalyticMechanismSpec,
    *,
    capacities_mol: Mapping[str, float],
    initial_amounts_mol: Mapping[str, float],
    heat_capacity_J_K: float = 0.0,
    fixed: bool = True,
    phase_id: str | None = None,
) -> StateSlot:
    if set(capacities_mol) != set(spec.site_types) or set(initial_amounts_mol) - set(
        spec.adsorbates
    ):
        raise ValueError("catalytic slot requires declared capacities and adsorbed species")
    if type(fixed) is not bool or (not fixed and not phase_id):
        raise ValueError("movable catalyst needs an explicit material phase")
    if spec.site_species_complete:
        for site, capacity in capacities_mol.items():
            total = fsum(
                initial_amounts_mol.get(k, 0) * spec.site_occupancies[k]
                for k, s in spec.adsorbate_sites.items()
                if s == site
            )
            if not isclose(total, capacity, rel_tol=1e-10, abs_tol=1e-24):
                raise ValueError("explicit surface configurations must fill the declared capacity")
    fields = tuple(
        StateFieldSpec(
            capacity_field(site), "mol", conserved_quantity=f"{spec.mechanism_id}:sites:{site}"
        )
        for site in spec.site_types
    )
    fields += tuple(StateFieldSpec(key, "mol", species_id=key) for key in spec.adsorbates)
    fields += (
        StateFieldSpec("heat_capacity_J_K", "J/K", conserved_quantity="carrier_heat_capacity"),
    )
    capacities = tuple(
        CapacityConstraint(
            capacity_field(site),
            tuple(
                (key, spec.site_occupancies[key])
                for key in spec.adsorbates
                if spec.adsorbate_sites[key] == site
            ),
        )
        for site in spec.site_types
    )
    interaction = InventoryInteractionSpec(
        spec.adsorbates,
        tuple(capacity_field(s) for s in spec.site_types),
        spec.interaction_matrix_J_mol,
    )
    slot_spec = StateSlotSpec(
        f"catalyst:{owner_id}",
        "equipment" if fixed else "material",
        owner_id,
        CATALYTIC_STATE_TYPE,
        fields,
        mechanism_id=spec.mechanism_id,
        phase_id=None if fixed else phase_id,
        heat_capacity_field="heat_capacity_J_K",
        capacities=capacities,
        energy_interactions=(interaction,),
    )
    return StateSlot(
        slot_spec,
        location_id,
        {
            **{capacity_field(s): n for s, n in capacities_mol.items()},
            **{s: initial_amounts_mol.get(s, 0.0) for s in spec.adsorbates},
            "heat_capacity_J_K": heat_capacity_J_K,
        },
    )


def validate_catalytic_slot(slot: StateSlot, spec: CatalyticMechanismSpec) -> None:
    if slot.spec.state_type != CATALYTIC_STATE_TYPE or slot.spec.mechanism_id != spec.mechanism_id:
        raise ValueError("catalytic slot belongs to a different mechanism")
    expected = make_catalytic_slot(
        slot.spec.owner_id,
        slot.location_id,
        spec,
        capacities_mol={s: slot.values[capacity_field(s)] for s in spec.site_types},
        initial_amounts_mol={s: slot.values[s] for s in spec.adsorbates},
        heat_capacity_J_K=slot.values["heat_capacity_J_K"],
        fixed=slot.spec.owner_kind == "equipment",
        phase_id=slot.spec.phase_id,
    )
    if expected.spec != slot.spec:
        raise ValueError("catalytic slot fields/capacities/interactions disagree with mechanism")


def validate_catalytic_thermodynamics(spec: CatalyticMechanismSpec, thermodynamics: Any) -> None:
    if not isinstance(thermodynamics, CatalyticThermodynamics):
        raise ValueError("declared catalysis needs its shared gas/harmonic thermochemistry table")
    if any(
        key not in thermodynamics.species
        or value.to_dict() != thermodynamics.species[key].to_dict()
        for key, value in spec.species_by_id.items()
    ):
        raise ValueError("catalytic Gibbs and caloric U must use the same declared species data")


@dataclass(frozen=True)
class CatalyticRates:
    amount_rates_mol_s: dict[str, float]
    forward_rates_mol_s: dict[str, float]
    reverse_rates_mol_s: dict[str, float]
    activities: dict[str, float]
    delta_g_J_mol: dict[str, float]
    barriers_J_mol: dict[str, tuple[float, float] | None]


def evaluate_catalytic_rates(
    state: CoupledStateView,
    slot: StateSlot,
    spec: CatalyticMechanismSpec,
    *,
    solver_trial: bool = False,
    reservoir_activities: Mapping[str, float] | None = None,
    potential_V: float | None = None,
    gas_volume_L: float | None = None,
) -> CatalyticRates:
    validate_catalytic_slot(slot, spec)
    if not spec.minimum_temperature_K <= state.temperature_K <= spec.maximum_temperature_K:
        raise ValueError("catalytic state outside declared temperature domain")
    if spec.reservoir_species:
        if (
            reservoir_activities is None
            or set(reservoir_activities) != set(spec.reservoir_species)
            or potential_V is None
        ):
            raise ValueError(
                "open catalysis requires explicit activities and fixed potential boundary"
            )
        finite_number(potential_V, "electrode potential")
        for activity in reservoir_activities.values():
            finite_number(activity, "reservoir activity")
            if activity <= 0:
                raise ValueError("reservoir activities must be positive")
    elif reservoir_activities is not None or potential_V is not None:
        raise ValueError("closed catalysis cannot silently acquire electrochemical boundaries")
    if gas_volume_L is not None:
        finite_number(gas_volume_L, "gas phase volume")
        if gas_volume_L <= 0:
            raise ValueError("gas phase volume must be positive")
    keys = {
        k: k
        if k in (*spec.gas_species, *spec.reservoir_species)
        else catalytic_inventory_key(slot, k)
        for k in spec.species_by_id
    }
    amounts = {k: state.amounts_mol[packed] for k, packed in keys.items()}
    if any(v < 0 for v in amounts.values()):
        raise ValueError("catalytic physical amounts must be nonnegative")
    capacities = {site: slot.values[capacity_field(site)] for site in spec.site_types}
    activities = {
        k: amounts[k]
        * R
        * state.temperature_K
        / ((state.volume_L if gas_volume_L is None else gas_volume_L) / 1000)
        / spec.reference_pressure_Pa
        for k in spec.gas_species
    }
    activities.update(reservoir_activities or {})
    for site, capacity in capacities.items():
        occupied = sum(
            amounts[k] * spec.site_occupancies[k]
            for k, own in spec.adsorbate_sites.items()
            if own == site
        )
        if not solver_trial and occupied > capacity + 1e-12 * max(capacity, 1e-20):
            raise ValueError("catalytic occupied inventory exceeds finite sites")
        if (
            spec.site_species_complete
            and not solver_trial
            and not isclose(occupied, capacity, rel_tol=1e-10, abs_tol=1e-24)
        ):
            raise ValueError("explicit surface configuration inventory lost its conserved sites")
        for key, own in spec.adsorbate_sites.items():
            if own == site:
                activities[key] = (
                    amounts[key] / capacity * spec.site_occupancies[key] if capacity else 0.0
                )
        activities[vacancy_key(site)] = max(0, 1 - occupied / capacity) if capacity else 0.0
    potential = {
        k: s.evaluate(state.temperature_K).gibbs_J_mol for k, s in spec.species_by_id.items()
    }
    potential.update({vacancy_key(site): 0.0 for site in spec.site_types})
    # Same existing interaction, capacity normalization and gradient as U(n,T).
    term = InventoryInteraction.from_slot(slot, slot.spec.energy_interactions[0])
    _, gradient = term.energy_and_gradient(state.amounts_mol)
    for key in spec.adsorbates:
        potential[key] += gradient[keys[key]]
    barriers: dict[str, tuple[float, float] | None] = {}
    derivatives, forward, reverse, delta_g = (
        dict.fromkeys(keys.values(), 0.0),
        {},
        {},
        {},
    )
    thermal = R * state.temperature_K
    for reaction in spec.reactions:
        delta = fsum(nu * potential[k] for k, nu in reaction.stoichiometry.items())
        if isinstance(reaction, ImportedSurfaceReaction):
            f, r = imported_surface_rates(
                reaction,
                temperature_K=state.temperature_K,
                activities=activities,
                gas_species=spec.gas_species,
                reference_pressure_Pa=spec.reference_pressure_Pa,
                site_density_mol_m2=spec.site_densities_mol_m2[reaction.rate_site],
                capacity_mol=capacities[reaction.rate_site],
                delta_g_J_mol=delta,
                site_occupancies=spec.site_occupancies,
            )
            for key, nu in reaction.stoichiometry.items():
                derivatives[keys[key]] += nu * (f - r)
            forward[reaction.reaction_id], reverse[reaction.reaction_id] = f, r
            delta_g[reaction.reaction_id], barriers[reaction.reaction_id] = delta, None
            continue
        if reaction.electron_stoichiometry:
            if potential_V is None:
                raise ValueError("electronic reaction requires an explicit SHE boundary")
            delta -= reaction.electron_stoichiometry * F * potential_V
        barrier_f, barrier_r = reaction.bep.barriers(delta)
        scale = capacities[reaction.rate_site]
        f = (
            scale
            * reaction.attempt_frequency_s_inv
            * exp(-barrier_f / thermal)
            * prod(
                activities[k] ** n
                for k, n in {**reaction.reactants, **reaction.reservoir_reactants}.items()
            )
        )
        r = (
            scale
            * reaction.attempt_frequency_s_inv
            * exp(-barrier_r / thermal)
            * prod(
                activities[k] ** n
                for k, n in {**reaction.products, **reaction.reservoir_products}.items()
            )
        )
        for key, nu in reaction.stoichiometry.items():
            if key in keys:
                derivatives[keys[key]] += nu * (f - r)
        forward[reaction.reaction_id], reverse[reaction.reaction_id] = f, r
        delta_g[reaction.reaction_id], barriers[reaction.reaction_id] = (
            delta,
            (barrier_f, barrier_r),
        )
    return CatalyticRates(derivatives, forward, reverse, activities, delta_g, barriers)


def catalytic_process(
    slot: StateSlot,
    spec: CatalyticMechanismSpec,
    *,
    reservoir_activities: Mapping[str, float] | None = None,
    potential_V: float | None = None,
    gas_volume_L: float | None = None,
) -> CoupledProcess:
    validate_catalytic_slot(slot, spec)

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        values = evaluate_catalytic_rates(
            state,
            slot,
            spec,
            solver_trial=True,
            reservoir_activities=reservoir_activities,
            potential_V=potential_V,
            gas_volume_L=gas_volume_L,
        )
        progress = {
            f"{slot.spec.slot_id}:{key}:forward": value
            for key, value in values.forward_rates_mol_s.items()
        }
        progress.update(
            {
                f"{slot.spec.slot_id}:{key}:reverse": value
                for key, value in values.reverse_rates_mol_s.items()
            }
        )
        return ProcessContribution(values.amount_rates_mol_s, progress)

    def validate(state: CoupledStateView) -> None:
        evaluate_catalytic_rates(
            state,
            slot,
            spec,
            reservoir_activities=reservoir_activities,
            potential_V=potential_V,
            gas_volume_L=gas_volume_L,
        )

    return CoupledProcess(
        f"declared-catalysis:{slot.spec.slot_id}", evaluate, validate_state=validate
    )


def catalytic_system(
    spec: CatalyticMechanismSpec,
    slots: Sequence[StateSlot],
    *,
    support_heat_capacity_J_K: float = 0.0,
    isothermal: bool = False,
) -> CoupledSystem:
    if not slots or len({s.spec.slot_id for s in slots}) != len(slots):
        raise ValueError("catalytic system requires unique owned slots")
    species = {k: spec.species_by_id[k] for k in spec.gas_species}
    for slot in slots:
        validate_catalytic_slot(slot, spec)
        species.update(
            {catalytic_inventory_key(slot, k): spec.species_by_id[k] for k in spec.adsorbates}
        )
    base = CatalyticThermodynamics(
        species, support_heat_capacity_J_K + sum(s.values["heat_capacity_J_K"] for s in slots)
    )
    laws: dict[str, dict[str, float]] = {}
    for key, value in species.items():
        for element, count in value.species.composition.items():
            laws.setdefault(element, {})[key] = count
    return CoupledSystem(
        with_inventory_interactions(base, slots),
        tuple(catalytic_process(s, spec) for s in slots),
        isothermal=isothermal,
        conservation_laws=laws,
        inventory_scales_mol=dict.fromkeys(species, 0.0001),
    )


def pack_catalytic_state(
    spec: CatalyticMechanismSpec,
    slots: Sequence[StateSlot],
    gas_amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    volume_L: float,
) -> CoupledState:
    if set(gas_amounts_mol) - set(spec.gas_species):
        raise ValueError("unknown gas species in catalytic inventory")
    amounts = {k: gas_amounts_mol.get(k, 0.0) for k in spec.gas_species}
    for slot in slots:
        validate_catalytic_slot(slot, spec)
        for key in spec.adsorbates:
            packed = catalytic_inventory_key(slot, key)
            if packed in amounts:
                raise ValueError("duplicate catalytic inventory instance")
            amounts[packed] = slot.values[key]
    return CoupledState(amounts, temperature_K, volume_L)
