"""Competitive Langmuir/LDF contribution on caller-owned finite inventories.

Loading is an owned amount, not a second bulk stock or a fresh-bed initializer.
Standalone contact and native processes use this contribution in CoupledSystem.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import exp, isclose, isfinite
from types import MappingProxyType

from chemworld.foundation.state_ownership import (
    CapacityConstraint,
    StateFieldSpec,
    StateSlot,
    StateSlotSpec,
)
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K

SORBENT_STATE_TYPE = "competitive-sorbent-v1"


@dataclass(frozen=True)
class SorbentKineticsSpec:
    sorbent_id: str
    site_capacity_mol_per_kg: float
    reference_affinity_L_per_mol: Mapping[str, float]
    mass_transfer_rates_s_inv: Mapping[str, float]
    adsorption_energy_J_mol: Mapping[str, float]
    reference_temperature_K: float = 298.15
    temperature_bounds_K: tuple[float, float] = (280.0, 430.0)
    carrier_heat_capacity_J_kg_K: float = 1000.0

    def __post_init__(self):
        for name in (
            "reference_affinity_L_per_mol",
            "mass_transfer_rates_s_inv",
            "adsorption_energy_J_mol",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "temperature_bounds_K", tuple(self.temperature_bounds_K))
        ids = set(self.reference_affinity_L_per_mol)
        if (
            not self.sorbent_id
            or not ids
            or any(not isinstance(k, str) or not k for k in ids)
            or set(self.mass_transfer_rates_s_inv) != ids
            or set(self.adsorption_energy_J_mol) != ids
            or any(not isfinite(v) or v <= 0 for v in self.reference_affinity_L_per_mol.values())
            or any(not isfinite(v) or v < 0 for v in self.mass_transfer_rates_s_inv.values())
            or any(not isfinite(v) for v in self.adsorption_energy_J_mol.values())
            or any(
                not isfinite(v) or v < 0
                for v in (self.site_capacity_mol_per_kg, self.carrier_heat_capacity_J_kg_K)
            )
        ):
            raise ValueError("sorbent needs explicit finite capacity, affinity, rates and energies")
        lower, upper = self.temperature_bounds_K
        if not (
            0 < lower < upper and isfinite(upper) and lower <= self.reference_temperature_K <= upper
        ):
            raise ValueError("sorbent reference must lie inside its temperature domain")

    @property
    def component_ids(self):
        return tuple(self.reference_affinity_L_per_mol)

    def affinities(self, temperature_K):
        if not self.temperature_bounds_K[0] <= temperature_K <= self.temperature_bounds_K[1]:
            raise ValueError("sorbent temperature leaves its declared affinity domain")
        return {
            key: value
            * exp(
                -self.adsorption_energy_J_mol[key]
                / R_J_PER_MOL_K
                * (1 / temperature_K - 1 / self.reference_temperature_K)
            )
            for key, value in self.reference_affinity_L_per_mol.items()
        }

    def to_dict(self):
        return {
            "sorbent_id": self.sorbent_id,
            "site_capacity_mol_per_kg": self.site_capacity_mol_per_kg,
            "reference_affinity_L_per_mol": dict(self.reference_affinity_L_per_mol),
            "mass_transfer_rates_s_inv": dict(self.mass_transfer_rates_s_inv),
            "adsorption_energy_J_mol": dict(self.adsorption_energy_J_mol),
            "reference_temperature_K": self.reference_temperature_K,
            "temperature_bounds_K": self.temperature_bounds_K,
            "carrier_heat_capacity_J_kg_K": self.carrier_heat_capacity_J_kg_K,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


def loading_field(component):
    return f"loading:{component}"


def sorbent_key(slot, component):
    return f"{slot.spec.slot_id}:{loading_field(component)}"


def make_sorbent_slot(
    spec,
    *,
    owner_id,
    location_id,
    mass_kg,
    initial_loading_mol_per_kg=None,
    fixed=True,
    phase_id=None,
    bound_species_ids=None,
):
    loading = {} if initial_loading_mol_per_kg is None else dict(initial_loading_mol_per_kg)
    species = (
        dict(zip(spec.component_ids, spec.component_ids, strict=True))
        if (bound_species_ids is None)
        else dict(bound_species_ids)
    )
    if (
        not isfinite(mass_kg)
        or mass_kg < 0
        or set(loading) - set(spec.component_ids)
        or set(species) != set(spec.component_ids)
        or any(not isfinite(v) or v < 0 for v in loading.values())
        or sum(loading.values()) > spec.site_capacity_mol_per_kg + 1e-12
        or (fixed and phase_id is not None)
        or (not fixed and phase_id is None)
    ):
        raise ValueError(
            "sorbent lot needs finite mass/loading and explicit fixed/mobile ownership"
        )
    fields = (
        StateFieldSpec("mass_kg", "kg", conserved_quantity=f"sorbent:{spec.sorbent_id}:mass"),
        StateFieldSpec("sites_mol", "mol", conserved_quantity=f"sorbent:{spec.sorbent_id}:sites"),
        StateFieldSpec("heat_capacity_J_K", "J/K"),
        *(
            StateFieldSpec(loading_field(k), "mol", species_id=species[k])
            for k in spec.component_ids
        ),
    )
    declaration = StateSlotSpec(
        f"{owner_id}:sorbent",
        "equipment" if fixed else "material",
        owner_id,
        SORBENT_STATE_TYPE,
        fields,
        mechanism_id=spec.sorbent_id,
        phase_id=phase_id,
        batch_reset="preserve" if fixed else "clear",
        heat_capacity_field="heat_capacity_J_K",
        capacities=(
            CapacityConstraint(
                "sites_mol", tuple((loading_field(k), 1.0) for k in spec.component_ids)
            ),
        ),
    )
    return StateSlot(
        declaration,
        location_id,
        {
            "mass_kg": mass_kg,
            "sites_mol": mass_kg * spec.site_capacity_mol_per_kg,
            "heat_capacity_J_K": mass_kg * spec.carrier_heat_capacity_J_kg_K,
            **{loading_field(k): mass_kg * loading.get(k, 0) for k in spec.component_ids},
        },
    )


def validate_sorbent_slot(slot, spec):
    expected = {
        "mass_kg",
        "sites_mol",
        "heat_capacity_J_K",
        *(loading_field(k) for k in spec.component_ids),
    }
    declarations = {field.name: field for field in slot.spec.fields}
    valid_fields = set(declarations) == expected
    if valid_fields:
        for name, field in declarations.items():
            bound = name in {loading_field(k) for k in spec.component_ids}
            unit = "mol" if bound or name == "sites_mol" else ("kg" if name == "mass_kg" else "J/K")
            conserved = (
                f"sorbent:{spec.sorbent_id}:mass"
                if name == "mass_kg"
                else f"sorbent:{spec.sorbent_id}:sites"
                if name == "sites_mol"
                else None
            )
            valid_fields &= bool(
                (not bound or field.species_id)
                and field
                == StateFieldSpec(
                    name,
                    unit,
                    species_id=field.species_id if bound else None,
                    conserved_quantity=conserved,
                )
            )
    if (
        slot.spec.state_type != SORBENT_STATE_TYPE
        or slot.spec.mechanism_id != spec.sorbent_id
        or set(slot.values) != expected
        or not valid_fields
        or slot.spec.owner_kind not in {"material", "equipment"}
        or slot.spec.batch_reset != ("preserve" if slot.spec.owner_kind == "equipment" else "clear")
        or ((slot.spec.owner_kind == "material") != (slot.spec.phase_id is not None))
        or slot.spec.heat_capacity_field != "heat_capacity_J_K"
        or slot.spec.capacities
        != (
            CapacityConstraint(
                "sites_mol", tuple((loading_field(k), 1.0) for k in spec.component_ids)
            ),
        )
        or not isclose(
            slot.values["sites_mol"],
            slot.values["mass_kg"] * spec.site_capacity_mol_per_kg,
            abs_tol=1e-12,
        )
        or not isclose(
            slot.values["heat_capacity_J_K"],
            slot.values["mass_kg"] * spec.carrier_heat_capacity_J_kg_K,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("sorbent owned lot disagrees with its material declaration")


def sorbent_inventory(slot, spec):
    validate_sorbent_slot(slot, spec)
    return {sorbent_key(slot, k): slot.values[loading_field(k)] for k in spec.component_ids}


def accepted_sorbent_slot(slot, spec, state):
    return replace(
        slot,
        values={
            **slot.values,
            **{
                loading_field(k): state.amounts_mol[sorbent_key(slot, k)]
                for k in spec.component_ids
            },
        },
    )


def sorbent_process(slot, spec, *, fluid_keys=None, progress=None, thermodynamics=None):
    """Same candidate-state contribution for native and standalone contact."""
    validate_sorbent_slot(slot, spec)
    fluid = (
        dict(zip(spec.component_ids, spec.component_ids, strict=True))
        if (fluid_keys is None)
        else dict(fluid_keys)
    )
    if set(fluid) != set(spec.component_ids) or len(set(fluid.values())) != len(fluid):
        raise ValueError("sorbent component ports must identify distinct fluid inventories")
    bound = {k: sorbent_key(slot, k) for k in spec.component_ids}
    capacity = slot.values["sites_mol"]
    evaluations = [0]

    def validate(state):
        spec.affinities(state.temperature_K)
        if any(state.amounts_mol[key] < 0 for key in (*fluid.values(), *bound.values())):
            raise ValueError("sorbent contact inventory is negative")
        if sum(state.amounts_mol[key] for key in bound.values()) > capacity + 1e-10:
            raise ValueError("sorbent contact exceeds its shared finite site capacity")

    def evaluate(state):
        affinity = spec.affinities(state.temperature_K)
        if thermodynamics is not None:
            for key in spec.component_ids:
                difference = thermodynamics.species_energy_j_mol(
                    bound[key], state.temperature_K
                ) - (thermodynamics.species_energy_j_mol(fluid[key], state.temperature_K))
                if not isclose(
                    difference, spec.adsorption_energy_J_mol[key], rel_tol=1e-10, abs_tol=1e-6
                ):
                    raise ValueError("sorbent caloric offsets disagree with its van't Hoff energy")
        concentrations = {
            k: state.amounts_mol[fluid[k]] / state.volume_L for k in spec.component_ids
        }
        denominator = 1 + sum(affinity[k] * concentrations[k] for k in spec.component_ids)
        rates = {
            k: spec.mass_transfer_rates_s_inv[k]
            * (
                capacity * affinity[k] * concentrations[k] / denominator
                - state.amounts_mol[bound[k]]
            )
            for k in spec.component_ids
        }
        # Unequal LDF mobilities can request more shared sites than exist.
        # The finite-site closure admits released sites plus relaxation of
        # actual vacancies at the fastest declared mobility. Thus d(vacancy)/dt
        # >= -k_max*vacancy, including the full face, without a discontinuous
        # event or post-hoc loading repair. Uniform-k LDF is unchanged.
        adsorption, desorption = (
            sum(max(v, 0) for v in rates.values()),
            sum(max(-v, 0) for v in rates.values()),
        )
        vacant = max(capacity - sum(state.amounts_mol[key] for key in bound.values()), 0.0)
        available = desorption + max(spec.mass_transfer_rates_s_inv.values()) * vacant
        if adsorption > available:
            rates = {k: v * available / adsorption if v > 0 else v for k, v in rates.items()}
        transfers = tuple(
            InternalTransfer(
                f"{slot.spec.slot_id}:{k}",
                fluid[k] if v >= 0 else bound[k],
                bound[k] if v >= 0 else fluid[k],
                abs(v),
            )
            for k, v in rates.items()
        )
        evaluations[0] += 1
        if progress is not None:
            progress(state.time_s, evaluations[0])
        return ProcessContribution(internal_transfers=transfers)

    keys = (*fluid.values(), *bound.values())
    return CoupledProcess(
        f"{slot.spec.slot_id}:Langmuir-LDF",
        evaluate,
        validate_state=validate,
        inventory_dependencies=dict.fromkeys(keys, keys),
    )


def advance_sorbent_contact(
    spec,
    slot,
    fluid_amounts_mol,
    *,
    temperature_K,
    volume_L,
    duration_s,
    thermodynamics,
    isothermal=False,
    heat_flux=None,
    evaluation_times_s=None,
    time_s=0.0,
    progress=None,
):
    """A transient view through the existing integrator, without a new clock.

    The supplied caloric model excludes the carrier body; its declared owned
    capacity is added here, as in the native thermal inventory assembler.
    """
    amounts = {**fluid_amounts_mol, **sorbent_inventory(slot, spec)}
    initial = CoupledState(amounts, temperature_K, volume_L, time_s)
    laws = {k: {k: 1.0, sorbent_key(slot, k): 1.0} for k in spec.component_ids}
    thermodynamics = replace(
        thermodynamics,
        support_heat_capacity_J_K=(
            thermodynamics.support_heat_capacity_J_K + slot.values["heat_capacity_J_K"]
        ),
    )
    system = CoupledSystem(
        thermodynamics,
        (sorbent_process(slot, spec, progress=progress, thermodynamics=thermodynamics),),
        heat_flux=(lambda _: HeatFlux()) if heat_flux is None else heat_flux,
        isothermal=isothermal,
        conservation_laws=laws,
        inventory_scales_mol={key: max(v, 1e-6) for key, v in amounts.items()},
    )
    return system.advance(initial, duration_s, evaluation_times_s=evaluation_times_s)


def isothermal_sorbent_loading(bed, fluid_amounts_mol, *, volume_L, duration_s):
    """Legacy explicit contact request, through the same local LDF contribution.

    The caller explicitly supplies the lot's initial loading. This temperature-
    free API has no caloric data: its held dummy thermal coordinate certifies
    kinetics only, and must not be presented as an adsorption-energy result.
    """
    spec = SorbentKineticsSpec(
        bed.sorbent_id,
        bed.site_capacity_mol_per_kg,
        bed.affinity_L_per_mol,
        dict.fromkeys(bed.affinity_L_per_mol, bed.mass_transfer_rate_per_s),
        dict.fromkeys(bed.affinity_L_per_mol, 0.0),
        carrier_heat_capacity_J_kg_K=0.0,
    )
    slot = make_sorbent_slot(
        spec,
        owner_id=bed.sorbent_id,
        location_id="contact",
        mass_kg=bed.sorbent_mass_kg,
        initial_loading_mol_per_kg=bed.initial_loading_mol_per_kg,
    )
    fluid = {**dict.fromkeys(spec.component_ids, 0.0), **fluid_amounts_mol}
    inventory = {**fluid, **sorbent_inventory(slot, spec)}
    thermodynamics = ConstantCvThermodynamics(
        {k: SpeciesInternalEnergy(0.0, 0.0) for k in inventory}, 1.0, 298.15
    )
    result = advance_sorbent_contact(
        spec,
        slot,
        fluid,
        temperature_K=298.15,
        volume_L=volume_L,
        duration_s=duration_s,
        thermodynamics=thermodynamics,
        isothermal=True,
    )
    return {
        k: (
            result.final_state.amounts_mol[sorbent_key(slot, k)] / bed.sorbent_mass_kg
            if bed.sorbent_mass_kg
            else 0.0
        )
        for k in spec.component_ids
    }
