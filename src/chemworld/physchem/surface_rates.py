"""Native surface rates shared by independent evaluation and coupled advance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import exp, isclose, isfinite, log, prod
from types import MappingProxyType

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledAdvanceResult,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.inventory_interactions import (
    InventoryInteraction,
    InventoryThermodynamics,
    with_inventory_interactions,
)
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.physchem.reactor_solvers import reaction_network_process
from chemworld.physchem.surface_specs import (
    ADSORBATES,
    DEFAULT_SURFACE_MECHANISM,
    GAS_SPECIES,
    MULTISITE_ADSORBATES,
    MULTISITE_STATE_TYPE,
    SURFACE_REACTION_IDS,
    SURFACE_STOICHIOMETRY,
    MultisiteSurfaceSpec,
    SyntheticSurfaceSpec,
    bulk_co_oxidation_network,
    surface_species_energies,
    validate_surface_slot,
)


def adsorbate_key(slot: StateSlot, species_id: str) -> str:
    if species_id not in surface_adsorbates(slot):
        raise ValueError("unknown adsorbed species")
    return f"{slot.spec.slot_id}:{species_id}"


def surface_adsorbates(slot: StateSlot) -> tuple[str, ...]:
    return MULTISITE_ADSORBATES if slot.spec.state_type == MULTISITE_STATE_TYPE else ADSORBATES


def surface_inventory(slot: StateSlot) -> dict[str, float]:
    return {
        adsorbate_key(slot, species): slot.values[species] for species in surface_adsorbates(slot)
    }


def gas_amounts_from_pressure(
    mole_fractions: Mapping[str, float],
    *,
    pressure_Pa: float,
    temperature_K: float,
    volume_L: float,
) -> dict[str, float]:
    if (
        set(mole_fractions) - set(GAS_SPECIES)
        or any(not isfinite(value) or value < 0 for value in mole_fractions.values())
        or not isclose(sum(mole_fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("gas fractions must be nonnegative, supported, and sum to one")
    if any(not isfinite(value) or value <= 0 for value in (temperature_K, volume_L)):
        raise ValueError("gas state requires positive finite K and L")
    if not isfinite(pressure_Pa) or pressure_Pa < 0:
        raise ValueError("pressure must be finite and nonnegative in Pa")
    total = pressure_Pa * (volume_L / 1000.0) / (R_J_PER_MOL_K * temperature_K)
    return {species: total * mole_fractions.get(species, 0.0) for species in GAS_SPECIES}


def surface_state(
    gas_amounts_mol: Mapping[str, float],
    slots: Sequence[StateSlot],
    *,
    temperature_K: float,
    volume_L: float,
    time_s: float = 0.0,
) -> CoupledState:
    if set(gas_amounts_mol) != set(GAS_SPECIES):
        raise ValueError("surface gas inventory must explicitly contain CO/O2/CO2/N2")
    if len({slot.spec.slot_id for slot in slots}) != len(slots):
        raise ValueError("duplicate surface instance")
    amounts = dict(gas_amounts_mol)
    for slot in slots:
        amounts.update(surface_inventory(slot))
    return CoupledState(amounts, temperature_K, volume_L, time_s)


@dataclass(frozen=True)
class SurfaceRates:
    per_site_rates_s_inv: tuple[float, ...]
    progress_rates_mol_s: tuple[float, ...]
    amount_rates_mol_s: Mapping[str, float]
    coverages: Mapping[str, float]
    partial_pressures_Pa: Mapping[str, float]


# Forward/reverse channels use these same stoichiometries for kinetics and
# material accounting. Vacancies are derived from each conserved site capacity.
MULTISITE_REACTIONS = (
    ({"CO": 1, "vac_A": 1}, {"CO_ads": 1}, "A"),
    ({"O2": 1, "vac_A": 2}, {"O_ads": 2}, "A"),
    ({"CO_ads": 1, "O_ads": 1}, {"CO2": 1, "vac_A": 2}, "A"),
    ({"CO": 1, "vac_B": 1}, {"CO_B": 1}, "B"),
    ({"O2": 1, "vac_B": 2}, {"O_B": 2}, "B"),
    ({"CO_B": 1, "O_B": 1}, {"CO2": 1, "vac_B": 2}, "B"),
    ({"CO_ads": 1, "vac_B": 1}, {"vac_A": 1, "CO_B": 1}, "exchange"),
)


def standard_surface_gibbs(spec: MultisiteSurfaceSpec, temperature: float) -> dict[str, float]:
    """Standard Gibbs potentials from the same U/Cv and reference G.

    Gas standard H=U+RT and Cp=Cv+R; bound species have no PV term. Ideal
    pressure and site-mixing activities are applied separately in the rates.
    """
    energies = surface_species_energies(spec)
    values = {}
    for key, reference_g in zip(
        (*GAS_SPECIES, *MULTISITE_ADSORBATES), spec.gibbs_reference_J_mol, strict=True
    ):
        caloric = energies[key]
        gas_R = R_J_PER_MOL_K if key in GAS_SPECIES else 0
        ref = spec.reference_temperature_K
        cp = caloric.cv_J_mol_K + gas_R
        h0 = caloric.u_ref_J_mol + gas_R * ref
        entropy = (h0 - reference_g) / ref + cp * log(temperature / ref)
        values[key] = h0 + cp * (temperature - ref) - temperature * entropy
    return {**values, "vac_A": 0.0, "vac_B": 0.0}


def _evaluate_multisite_rates(
    state: CoupledStateView, slot: StateSlot, spec: MultisiteSurfaceSpec, *, solver_trial: bool,
) -> SurfaceRates:
    if not 300 <= state.temperature_K <= 450:
        raise ValueError("synthetic surface temperature domain is 300-450 K")
    amounts = {k: state.amounts_mol[k] for k in GAS_SPECIES}
    ads = {k: state.amounts_mol[adsorbate_key(slot, k)] for k in MULTISITE_ADSORBATES}
    if any(not isfinite(v) or v < 0 for v in (*amounts.values(), *ads.values())):
        raise ValueError("multisite amounts must be finite and nonnegative")
    activities, coverages = {}, {}
    capacities = {"A": slot.values["sites_mol"], "B": slot.values["sites_B_mol"]}
    for site, species in (("A", ADSORBATES), ("B", ("CO_B", "O_B"))):
        capacity = capacities[site]
        if not solver_trial and sum(ads[k] for k in species) > capacity * (1 + 1e-12):
            raise ValueError("occupied inventory exceeds its own site class")
        coverages.update({k: ads[k] / capacity if capacity else 0 for k in species})
        coverages[f"vac_{site}"] = max(0, 1 - sum(coverages[k] for k in species)) if capacity else 0
    pressure = {
        k: n * R_J_PER_MOL_K * state.temperature_K / (state.volume_L / 1000)
        for k, n in amounts.items()
    }
    activities.update({k: p / spec.reference_pressure_Pa for k, p in pressure.items()})
    activities.update(coverages)
    potential = standard_surface_gibbs(spec, state.temperature_K)
    term = InventoryInteraction.from_slot(slot, slot.spec.energy_interactions[0])
    _, gradient = term.energy_and_gradient(state.amounts_mol)
    potential.update(
        {k: potential[k] + gradient[adsorbate_key(slot, k)] for k in MULTISITE_ADSORBATES}
    )
    thermal = R_J_PER_MOL_K * state.temperature_K
    derivatives = dict.fromkeys(
        (*GAS_SPECIES, *(adsorbate_key(slot, k) for k in MULTISITE_ADSORBATES)), 0.0
    )
    forward_reverse: list[float] = []
    per_site: list[float] = []
    for reactants, products, site in MULTISITE_REACTIONS:
        delta = sum(v * potential[k] for k, v in products.items()) - sum(
            v * potential[k] for k, v in reactants.items()
        )
        prefactor = spec.attempt_frequency_s_inv * exp(-spec.barrier_J_mol / thermal)
        f = (
            prefactor
            * exp(-max(delta, 0) / thermal)
            * prod(activities[k] ** v for k, v in reactants.items())
        )
        r = (
            prefactor
            * exp(-max(-delta, 0) / thermal)
            * prod(activities[k] ** v for k, v in products.items())
        )
        total = sum(capacities.values())
        scale = (
            capacities[site]
            if site in capacities
            else (capacities["A"] * capacities["B"] / total if total else 0)
        )
        per_site.extend((f, r))
        forward_reverse.extend((scale * f, scale * r))
        for key in reactants.keys() | products.keys():
            if key.startswith("vac_"):
                continue
            packed = key if key in GAS_SPECIES else adsorbate_key(slot, key)
            derivatives[packed] += scale * (f - r) * (products.get(key, 0) - reactants.get(key, 0))
    return SurfaceRates(
        tuple(per_site),
        tuple(forward_reverse),
        MappingProxyType(derivatives),
        MappingProxyType(coverages),
        MappingProxyType(pressure),
    )


def _evaluate_surface_rates(
    state: CoupledStateView,
    slot: StateSlot,
    spec: SyntheticSurfaceSpec,
    *,
    solver_trial: bool,
) -> SurfaceRates:
    if isinstance(spec, MultisiteSurfaceSpec):
        return _evaluate_multisite_rates(state, slot, spec, solver_trial=solver_trial)
    if not 300 <= state.temperature_K <= 450:
        raise ValueError("synthetic surface temperature domain is 300-450 K")
    sites = slot.values["sites_mol"]
    try:
        gas = {key: state.amounts_mol[key] for key in GAS_SPECIES}
        adsorbed = {key: state.amounts_mol[adsorbate_key(slot, key)] for key in ADSORBATES}
    except KeyError as exc:
        raise ValueError("surface contribution requires the complete shared inventory") from exc
    if any(value < 0 or not isfinite(value) for value in (*gas.values(), *adsorbed.values())):
        raise ValueError("surface inventory must be finite and nonnegative")
    co = adsorbed["CO_ads"] / sites if sites else 0.0
    oxygen = adsorbed["O_ads"] / sites if sites else 0.0
    if not solver_trial and (sum(adsorbed.values()) > sites * (1 + 1e-12)):
        raise ValueError("surface occupied inventory exceeds site capacity")
    vacant = max(0.0, 1 - co - oxygen) if sites else 0.0
    pressures = {
        key: amount * R_J_PER_MOL_K * state.temperature_K / (state.volume_L / 1000)
        for key, amount in gas.items()
    }
    k = tuple(
        reference
        * exp(
            -energy / R_J_PER_MOL_K * (1 / state.temperature_K - 1 / spec.reference_temperature_K)
        )
        for reference, energy in zip(
            spec.rate_constants_s_inv,
            spec.activation_energies_J_mol,
            strict=True,
        )
    )
    per_site = (
        k[0] * pressures["CO"] / spec.reference_pressure_Pa * vacant,
        k[1] * co,
        k[2] * pressures["O2"] / spec.reference_pressure_Pa * vacant**2,
        k[3] * oxygen**2,
        k[4]
        * co
        * oxygen
        * exp(-spec.coverage_activation_energy_J_mol * co / (R_J_PER_MOL_K * state.temperature_K)),
    )
    if any(not isfinite(value) or value < 0 for value in per_site):
        raise RuntimeError("surface rate evaluation produced invalid rates")
    progress = tuple(sites * value for value in per_site)
    derivatives = {}
    for species, stoichiometry in SURFACE_STOICHIOMETRY.items():
        key = species if species in GAS_SPECIES else adsorbate_key(slot, species)
        derivatives[key] = sum(
            coefficient * rate
            for coefficient, rate in zip(
                stoichiometry,
                progress,
                strict=True,
            )
        )
    return SurfaceRates(
        per_site,
        progress,
        MappingProxyType(derivatives),
        MappingProxyType({"CO_ads": co, "O_ads": oxygen, "vacant": vacant}),
        MappingProxyType(pressures),
    )


def evaluate_surface_rates(
    state: CoupledStateView,
    slot: StateSlot,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
) -> SurfaceRates:
    validate_surface_slot(slot, spec)
    return _evaluate_surface_rates(state, slot, spec, solver_trial=False)


def surface_process(
    slot: StateSlot,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
) -> CoupledProcess:
    validate_surface_slot(slot, spec)
    # Copy static carrier parameters at bind time. The evolving adsorbate and
    # gas amounts always come from the coordinator's current candidate.
    carrier = replace(slot)
    channels = (
        tuple(f"R{i + 1}:{direction}" for i in range(7) for direction in ("forward", "reverse"))
        if isinstance(spec, MultisiteSurfaceSpec)
        else SURFACE_REACTION_IDS
    )

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        rates = _evaluate_surface_rates(state, carrier, spec, solver_trial=True)
        return ProcessContribution(
            amount_rates_mol_s=rates.amount_rates_mol_s,
            reaction_progress_mol_s={
                f"{carrier.spec.slot_id}:{reaction}": rate
                for reaction, rate in zip(channels, rates.progress_rates_mol_s, strict=True)
            },
        )

    def validate(state: CoupledStateView) -> None:
        if not 300 <= state.temperature_K <= 450:
            raise ValueError("synthetic surface temperature domain is 300-450 K")
        updated_surface_slot(carrier, state, spec)

    return CoupledProcess(
        f"native-surface:{carrier.spec.slot_id}", evaluate, validate_state=validate
    )


def updated_surface_slot(
    slot: StateSlot,
    state: CoupledStateView,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
) -> StateSlot:
    validate_surface_slot(slot, spec)
    # StateSlot's own capacity/ownership rules validate accepted inventories;
    # numerical trial extension never renormalizes a committed coverage.
    return replace(
        slot,
        values={
            **slot.values,
            **{
                key: state.amounts_mol[adsorbate_key(slot, key)] for key in surface_adsorbates(slot)
            },
        },
    )


def surface_thermodynamics(
    slots: Sequence[StateSlot],
    *,
    support_heat_capacity_J_K: float = 20.0,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
) -> InventoryThermodynamics:
    if not isfinite(support_heat_capacity_J_K) or support_heat_capacity_J_K < 0:
        raise ValueError("fixed support heat capacity must be finite and nonnegative")
    energies = surface_species_energies(spec)
    species = {key: energies[key] for key in GAS_SPECIES}
    capacity = support_heat_capacity_J_K
    for slot in slots:
        validate_surface_slot(slot, spec)
        for key in surface_adsorbates(slot):
            qualified = adsorbate_key(slot, key)
            if qualified in species:
                raise ValueError("duplicate surface instance")
            species[qualified] = energies[key]
        capacity += slot.values["heat_capacity_J_K"]
    return with_inventory_interactions(
        ConstantCvThermodynamics(species, capacity, reference_temperature_K=350), slots
    )


def surface_conservation_laws(slots: Sequence[StateSlot]) -> dict[str, dict[str, float]]:
    carbon: dict[str, float] = {"CO": 1, "CO2": 1}
    oxygen: dict[str, float] = {"CO": 1, "O2": 2, "CO2": 2}
    for slot in slots:
        carbon[adsorbate_key(slot, "CO_ads")] = 1
        oxygen[adsorbate_key(slot, "CO_ads")] = 1
        oxygen[adsorbate_key(slot, "O_ads")] = 1
        if slot.spec.state_type == MULTISITE_STATE_TYPE:
            carbon[adsorbate_key(slot, "CO_B")] = 1
            oxygen[adsorbate_key(slot, "CO_B")] = 1
            oxygen[adsorbate_key(slot, "O_B")] = 1
    return {"C": carbon, "O": oxygen, "N": {"N2": 2}}


def make_surface_system(
    slots: Sequence[StateSlot],
    *,
    include_bulk: bool = False,
    isothermal: bool = False,
    support_heat_capacity_J_K: float = 20.0,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
) -> CoupledSystem:
    """Independent composition convenience; runtime uses the same contributions."""
    processes = tuple(surface_process(slot, spec) for slot in slots)
    if include_bulk:
        processes += (reaction_network_process(bulk_co_oxidation_network()),)
    scales = dict.fromkeys(GAS_SPECIES, 0.0001)
    for slot in slots:
        scales.update(
            {
                adsorbate_key(slot, key): max(slot.values["sites_mol"], 1e-20)
                for key in surface_adsorbates(slot)
            }
        )
    return CoupledSystem(
        surface_thermodynamics(
            slots, support_heat_capacity_J_K=support_heat_capacity_J_K, spec=spec
        ),
        processes,
        isothermal=isothermal,
        conservation_laws=surface_conservation_laws(slots),
        inventory_scales_mol=scales,
    )


def simulate_surface(
    state: CoupledState,
    slots: Sequence[StateSlot],
    duration_s: float,
    *,
    spec: SyntheticSurfaceSpec = DEFAULT_SURFACE_MECHANISM,
    isothermal: bool = True,
    include_bulk: bool = False,
    evaluation_times_s: Sequence[float] | None = None,
) -> CoupledAdvanceResult:
    """One call to the common solver; no surface-specific integration loop."""
    system = make_surface_system(slots, spec=spec, isothermal=isothermal, include_bulk=include_bulk)
    return system.advance(state, duration_s, evaluation_times_s=evaluation_times_s)
