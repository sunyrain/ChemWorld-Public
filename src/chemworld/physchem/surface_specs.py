"""Finite synthetic single-site CO oxidation parameters and owned state layout.

These are the native development parameters in NATIVE_SURFACE_DESIGN.md, not a
fit to a real catalyst or a general upstream mechanism-file interpreter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from math import exp, isclose, isfinite
from types import MappingProxyType

from chemworld.foundation.state_ownership import (
    CapacityConstraint,
    InventoryInteractionSpec,
    StateFieldSpec,
    StateSlot,
    StateSlotSpec,
)
from chemworld.physchem.coupled_system import SpeciesInternalEnergy
from chemworld.physchem.reaction_network import (
    R_J_PER_MOL_K,
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)

GAS_SPECIES = ("CO", "O2", "CO2", "N2")
ADSORBATES = ("CO_ads", "O_ads")
SURFACE_REACTION_IDS = ("R1", "R2", "R3", "R4", "R5")
SURFACE_STATE_TYPE = "native_single_site_co_oxidation"
MULTISITE_STATE_TYPE = "native_reversible_multisite_co_oxidation"
MULTISITE_ADSORBATES = (*ADSORBATES, "CO_B", "O_B")
# One stoichiometry source for both process derivatives and per-step checks.
SURFACE_STOICHIOMETRY = MappingProxyType(
    {
        "CO": (-1, 1, 0, 0, 0),
        "O2": (0, 0, -1, 1, 0),
        "CO2": (0, 0, 0, 0, 1),
        "N2": (0, 0, 0, 0, 0),
        "CO_ads": (1, -1, 0, 0, -1),
        "O_ads": (0, 0, 2, -2, -1),
    }
)


@dataclass(frozen=True)
class SurfaceMechanismSpec:
    rate_constants_s_inv: tuple[float, ...] = (2.0, 0.2, 4.0, 0.05, 10.0)
    activation_energies_J_mol: tuple[float, ...] = (0, 50_000, 0, 60_000, 40_000)
    coverage_activation_energy_J_mol: float = 5000.0
    reference_temperature_K: float = 350.0
    reference_pressure_Pa: float = 100_000.0
    rate_unit: str = "s^-1"
    energy_unit: str = "J/mol"

    def __post_init__(self) -> None:
        for name in ("rate_constants_s_inv", "activation_energies_J_mol"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 5 or any(not isfinite(value) or value < 0.0 for value in values):
                raise ValueError("the five-step mechanism needs five finite nonnegative parameters")
            object.__setattr__(self, name, values)
        if not 300 <= self.reference_temperature_K <= 450:
            raise ValueError("kinetic reference temperature must be within 300-450 K")
        if not isfinite(self.reference_pressure_Pa) or self.reference_pressure_Pa <= 0:
            raise ValueError("reference pressure must be finite and positive in Pa")
        if (
            not isfinite(self.coverage_activation_energy_J_mol)
            or not 0 <= self.coverage_activation_energy_J_mol <= 10_000
        ):
            raise ValueError("coverage energy must be in [0, 10000] J/mol")
        if self.rate_unit != "s^-1" or self.energy_unit != "J/mol":
            raise ValueError("surface rates require s^-1 and activation energies require J/mol")

    @property
    def mechanism_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return (
            "synthetic-co-oxidation-single-site-v1:"
            + hashlib.sha256(payload.encode()).hexdigest()[:16]
        )

    def to_dict(self):
        return {
            "model": "legacy-five-step",
            **asdict(self),
            "rate_constants_s_inv": list(self.rate_constants_s_inv),
            "activation_energies_J_mol": list(self.activation_energies_J_mol),
        }


@dataclass(frozen=True)
class MultisiteSurfaceSpec:
    """Explicit reversible alternative to the historical five-step mechanism.

    Four surface amounts share one carrier with two separately conserved site
    classes. The symmetric, temperature-independent excess free energy is also
    internal energy. Standard enthalpy and entropy use the shared caloric data.
    """

    second_site_ratio: float = 1.0
    interaction_matrix_J_mol: tuple[tuple[float, ...], ...] = (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    gibbs_reference_J_mol: tuple[float, ...] = (-10000, 0, -20000, 0, -14000, -4000, -12000, -3000)
    attempt_frequency_s_inv: float = 100.0
    barrier_J_mol: float = 10000.0
    reference_temperature_K: float = 350.0
    reference_pressure_Pa: float = 100000.0

    def __post_init__(self):
        interaction = InventoryInteractionSpec(
            MULTISITE_ADSORBATES, ("sites_mol", "sites_B_mol"), self.interaction_matrix_J_mol
        )
        object.__setattr__(self, "interaction_matrix_J_mol", interaction.coefficients_J_mol)
        object.__setattr__(self, "gibbs_reference_J_mol", tuple(self.gibbs_reference_J_mol))
        if (
            not isfinite(self.second_site_ratio)
            or not 0 <= self.second_site_ratio <= 2
            or any(abs(v) > 10000 for row in self.interaction_matrix_J_mol for v in row)
            or len(self.gibbs_reference_J_mol) != 8
            or any(not isfinite(v) or abs(v) > 50000 for v in self.gibbs_reference_J_mol)
            or not isfinite(self.attempt_frequency_s_inv)
            or not 0 < self.attempt_frequency_s_inv <= 1000
            or not isfinite(self.barrier_J_mol)
            or not 0 <= self.barrier_J_mol <= 50000
            or self.reference_temperature_K != 350
            or self.reference_pressure_Pa != 100000
        ):
            raise ValueError("multisite parameters leave the finite synthetic model domain")

    @property
    def mechanism_id(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return (
            "synthetic-reversible-multisite-v1:" + hashlib.sha256(payload.encode()).hexdigest()[:16]
        )

    def to_dict(self):
        return {
            "model": "reversible-multisite",
            **asdict(self),
            "interaction_matrix_J_mol": [list(row) for row in self.interaction_matrix_J_mol],
            "gibbs_reference_J_mol": list(self.gibbs_reference_J_mol),
        }


def surface_spec_from_dict(payload):
    values = dict(payload)
    kind = values.pop("model")
    if kind == "legacy-five-step":
        return SurfaceMechanismSpec(**values)
    if kind == "reversible-multisite":
        return MultisiteSurfaceSpec(**values)
    raise ValueError("unknown native surface mechanism")


DEFAULT_SURFACE_MECHANISM = SurfaceMechanismSpec()


def surface_species(spec=DEFAULT_SURFACE_MECHANISM) -> tuple[SpeciesSpec, ...]:
    return tuple(
        SpeciesSpec(name, formula, phase=phase)
        for name, formula, phase in (
            ("CO", "CO", "gas"),
            ("O2", "O2", "gas"),
            ("CO2", "CO2", "gas"),
            ("N2", "N2", "gas"),
            ("CO_ads", "CO", "surface"),
            ("O_ads", "O", "surface"),
        )
        + (
            (("CO_B", "CO", "surface"), ("O_B", "O", "surface"))
            if isinstance(spec, MultisiteSurfaceSpec)
            else ()
        )
    )


def surface_species_energies(spec=DEFAULT_SURFACE_MECHANISM) -> dict[str, SpeciesInternalEnergy]:
    """The sole species U/Cv table, referenced to 350 K, including bound atoms."""
    energies = {
        "CO": SpeciesInternalEnergy(-110_000, 20.8),
        "O2": SpeciesInternalEnergy(0, 20.8),
        "CO2": SpeciesInternalEnergy(-390_000, 28.8),
        "N2": SpeciesInternalEnergy(0, 20.8),
        "CO_ads": SpeciesInternalEnergy(-210_000, 10),
        "O_ads": SpeciesInternalEnergy(-100_000, 5),
    }
    if isinstance(spec, MultisiteSurfaceSpec):
        energies.update(
            {"CO_B": SpeciesInternalEnergy(-205000, 10), "O_B": SpeciesInternalEnergy(-95000, 5)}
        )
    return energies


def make_surface_slot(
    owner_id: str,
    location_id: str,
    *,
    fixed: bool = True,
    area_m2: float = 0.01,
    site_density_mol_m2: float = 2.5e-5,
    initial_coverages: tuple[float, float] = (0.0, 0.0),
    second_coverages: tuple[float, float] = (0.0, 0.0),
    heat_capacity_J_K: float = 0.0,
    phase_id: str | None = None,
    spec: SurfaceMechanismSpec = DEFAULT_SURFACE_MECHANISM,
) -> StateSlot:
    if not isfinite(area_m2) or not 0.001 <= area_m2 <= 0.1:
        raise ValueError("initial catalyst area must be in [0.001, 0.1] m2")
    if not isfinite(site_density_mol_m2) or not 1e-6 <= site_density_mol_m2 <= 1e-4:
        raise ValueError("site density must be in [1e-6, 1e-4] mol/m2")
    if not isfinite(heat_capacity_J_K) or heat_capacity_J_K < 0:
        raise ValueError("carrier heat capacity must be finite and nonnegative in J/K")
    if (
        len(initial_coverages) != 2
        or any(not isfinite(value) or value < 0 for value in initial_coverages)
        or sum(initial_coverages) > 1
    ):
        raise ValueError("coverages must be nonnegative and sum to at most one")
    if not fixed and not phase_id:
        raise ValueError("movable catalytic material needs an explicit material phase")
    multi = isinstance(spec, MultisiteSurfaceSpec)
    if (
        len(second_coverages) != 2
        or any(not isfinite(v) or v < 0 for v in second_coverages)
        or sum(second_coverages) > 1
    ):
        raise ValueError("second-site coverages must be finite, nonnegative and sum to at most one")
    if any(second_coverages) and (not multi or spec.second_site_ratio == 0):
        raise ValueError("second-site adsorbates require a positive declared second site inventory")
    extra_fields = (
        (
            StateFieldSpec("sites_B_mol", "mol", conserved_quantity=f"{spec.mechanism_id}:sites_B"),
            StateFieldSpec("CO_B", "mol", species_id="CO_B"),
            StateFieldSpec("O_B", "mol", species_id="O_B"),
        )
        if multi
        else ()
    )
    slot_spec = StateSlotSpec(
        slot_id=f"surface:{owner_id}",
        owner_kind="equipment" if fixed else "material",
        owner_id=owner_id,
        state_type=MULTISITE_STATE_TYPE if multi else SURFACE_STATE_TYPE,
        mechanism_id=spec.mechanism_id,
        phase_id=None if fixed else phase_id,
        fields=(
            StateFieldSpec("area_m2", "m2", conserved_quantity="surface_area"),
            StateFieldSpec(
                "site_density_mol_m2", "mol/m2", mixing="weighted", weight_field="area_m2"
            ),
            StateFieldSpec("sites_mol", "mol", conserved_quantity=f"{spec.mechanism_id}:sites"),
            StateFieldSpec("CO_ads", "mol", species_id="CO_ads"),
            StateFieldSpec("O_ads", "mol", species_id="O_ads"),
            StateFieldSpec("heat_capacity_J_K", "J/K", conserved_quantity="carrier_heat_capacity"),
            *extra_fields,
        ),
        heat_capacity_field="heat_capacity_J_K",
        capacities=(CapacityConstraint("sites_mol", (("CO_ads", 1), ("O_ads", 1))),)
        + ((CapacityConstraint("sites_B_mol", (("CO_B", 1), ("O_B", 1))),) if multi else ()),
        energy_interactions=(
            InventoryInteractionSpec(
                MULTISITE_ADSORBATES, ("sites_mol", "sites_B_mol"), spec.interaction_matrix_J_mol
            ),
        )
        if multi
        else (),
    )
    sites = area_m2 * site_density_mol_m2
    return StateSlot(
        slot_spec,
        location_id,
        {
            "area_m2": area_m2,
            "site_density_mol_m2": site_density_mol_m2,
            "sites_mol": sites,
            "CO_ads": sites * initial_coverages[0],
            "O_ads": sites * initial_coverages[1],
            "heat_capacity_J_K": heat_capacity_J_K,
            **(
                {
                    "sites_B_mol": sites * spec.second_site_ratio,
                    "CO_B": sites * spec.second_site_ratio * second_coverages[0],
                    "O_B": sites * spec.second_site_ratio * second_coverages[1],
                }
                if multi
                else {}
            ),
        },
    )


def validate_surface_slot(
    slot: StateSlot,
    spec: SurfaceMechanismSpec = DEFAULT_SURFACE_MECHANISM,
) -> None:
    multi = isinstance(spec, MultisiteSurfaceSpec)
    state_type = MULTISITE_STATE_TYPE if multi else SURFACE_STATE_TYPE
    if slot.spec.state_type != state_type or slot.spec.mechanism_id != spec.mechanism_id:
        raise ValueError("surface state and kinetic mechanism are incompatible")
    expected_units = {
        "area_m2": "m2",
        "site_density_mol_m2": "mol/m2",
        "sites_mol": "mol",
        "CO_ads": "mol",
        "O_ads": "mol",
        "heat_capacity_J_K": "J/K",
    }
    if multi:
        expected_units.update(dict.fromkeys(("sites_B_mol", "CO_B", "O_B"), "mol"))
    if {field.name: field.unit for field in slot.spec.fields} != expected_units:
        raise ValueError("surface state layout must declare the native physical units")
    expected = make_surface_slot(
        slot.spec.owner_id,
        slot.location_id,
        fixed=slot.spec.owner_kind == "equipment",
        phase_id=slot.spec.phase_id,
        spec=spec,
    ).spec
    if (
        slot.spec.fields != expected.fields
        or slot.spec.capacities != expected.capacities
        or slot.spec.heat_capacity_field != expected.heat_capacity_field
        or slot.spec.energy_interactions != expected.energy_interactions
    ):
        raise ValueError("surface state must retain its declared species and ownership rules")
    if not isclose(
        slot.values["sites_mol"],
        slot.values["area_m2"] * slot.values["site_density_mol_m2"],
        rel_tol=1e-12,
        abs_tol=1e-24,
    ):
        raise ValueError("surface area, site density and inventory disagree")
    if multi and not isclose(
        slot.values["sites_B_mol"],
        spec.second_site_ratio * slot.values["sites_mol"],
        rel_tol=1e-12,
        abs_tol=1e-24,
    ):
        raise ValueError("second-site inventory disagrees with declared carrier geometry")


def bulk_co_oxidation_network() -> ReactionNetworkSpec:
    """Separate optional bulk process; surface rates never hide this reaction."""
    reference = 350.0
    activation = 40_000.0
    energies = surface_species_energies()
    reaction = ReactionSpec.from_equation(
        reaction_id="bulk_co_oxidation",
        equation="2 CO + O2 => 2 CO2",
        rate_law=RateLawSpec(
            "bulk-third-order",
            "arrhenius",
            {
                "A": 1000.0 * exp(activation / (R_J_PER_MOL_K * reference)),
                "Ea_J_per_mol": activation,
            },
        ),
    )
    delta_u = sum(
        coefficient * energies[key].u_ref_J_mol
        for key, coefficient in reaction.stoichiometry.items()
    )
    reference_delta_h = delta_u + sum(reaction.stoichiometry.values()) * R_J_PER_MOL_K * reference
    return ReactionNetworkSpec(
        network_id="synthetic-co-oxidation-bulk-v1",
        species=surface_species()[:4],
        reactions=(replace(reaction, delta_h_J_per_mol=reference_delta_h),),
        metadata={
            "synthetic_development_parameters": True,
            "native_energy_owner": "ConstantCvThermodynamics",
            "delta_h_reference_temperature_K": reference,
        },
    )
