"""Original finite-site catalysis with real maintained liquid stocks and C/J ports."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose, log

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.catalysis_descriptors import finite_number
from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    CatalyticRates,
    CatalyticReactionSpec,
    catalytic_inventory_key,
    catalytic_process,
    evaluate_catalytic_rates,
    validate_catalytic_slot,
)
from chemworld.physchem.catalysis_reservoirs import CatalyticReservoirBoundary
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.coupled_system import (
    DEFAULT_COUPLED_ODE_POLICY,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.inventory_interactions import with_inventory_interactions
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R

ELECTRICAL_FIELDS = ("faradaic_charge_C", "electrical_work_J")


def reservoir_ports(boundary: CatalyticReservoirBoundary) -> tuple[MaterialPort, ...]:
    return tuple(
        MaterialPort(
            f"reservoir:{key}:{direction}",
            key,
            "upstream:" + key if direction == "in" else "liquid",
            "liquid" if direction == "in" else "upstream:" + key,
            direction,
        )
        for key in boundary.solutes
        for direction in ("in", "out")
    )


def validate_boundary(
    spec: CatalyticMechanismSpec, boundary: CatalyticReservoirBoundary, potential_V: float
) -> None:
    if not spec.reservoir_species or set(boundary.solutes) != set(spec.reservoir_species):
        raise ValueError("open catalysis must bind every original registered aqueous chemostat")
    boundary.validate_species({k: spec.species_by_id[k].species for k in spec.reservoir_species})
    if (
        not boundary.minimum_potential_V <= potential_V <= boundary.maximum_potential_V
        or not spec.minimum_temperature_K <= boundary.bath.setpoint_K <= spec.maximum_temperature_K
    ):
        raise ValueError("fixed SHE potential or isotherm outside declared catalytic domain")


def pack_chemostatted_state(
    spec: CatalyticMechanismSpec,
    slot: StateSlot,
    boundary: CatalyticReservoirBoundary,
    gas_amounts_mol: Mapping[str, float],
    *,
    time_s: float = 0,
    liquid_amounts_mol: Mapping[str, float] | None = None,
    coordinates: Mapping[str, float] | None = None,
    exchanged_mol: Mapping[str, float] | None = None,
    exchange_energy_J: Mapping[str, float] | None = None,
) -> CoupledState:
    validate_catalytic_slot(slot, spec)
    if set(gas_amounts_mol) != set(spec.gas_species):
        raise ValueError("gas phase must retain exactly its real declared species")
    liquid = boundary.amounts_mol if liquid_amounts_mol is None else dict(liquid_amounts_mol)
    if set(liquid) != set(boundary.solutes) or any(
        not isclose(liquid[k], n, rel_tol=1e-10, abs_tol=1e-25)
        for k, n in boundary.amounts_mol.items()
    ):
        raise ValueError("aqueous inventory disagrees with its declared a/gamma/volume")
    return CoupledState(
        {
            **gas_amounts_mol,
            **liquid,
            **{catalytic_inventory_key(slot, k): slot.values[k] for k in spec.adsorbates},
        },
        boundary.bath.setpoint_K,
        boundary.gas_volume_L + boundary.liquid_volume_L,
        time_s,
        coordinates=dict.fromkeys(ELECTRICAL_FIELDS, 0.0) if coordinates is None else coordinates,
        exchanged_mol={} if exchanged_mol is None else exchanged_mol,
        exchange_energy_J={} if exchange_energy_J is None else exchange_energy_J,
    )


def open_catalytic_rates(
    spec: CatalyticMechanismSpec,
    slot: StateSlot,
    boundary: CatalyticReservoirBoundary,
    potential_V: float,
    state: CoupledStateView,
    *,
    solver_trial: bool = False,
) -> CatalyticRates:
    return evaluate_catalytic_rates(
        state,
        slot,
        spec,
        solver_trial=solver_trial,
        reservoir_activities=boundary.activities,
        potential_V=potential_V,
        gas_volume_L=boundary.gas_volume_L,
    )


def faradaic_current(spec: CatalyticMechanismSpec, rates: CatalyticRates) -> float:
    return F * sum(
        r.electron_stoichiometry
        * (rates.forward_rates_mol_s[r.reaction_id] - rates.reverse_rates_mol_s[r.reaction_id])
        for r in spec.reactions
        if isinstance(r, CatalyticReactionSpec)
    )


@dataclass(frozen=True)
class ChemostattedElectricalReadout:
    terminal_potential_V: float  # Working electrode vs SHE; no counterelectrode voltage claim.
    applied_current_A: float
    coverages: Mapping[str, float]
    forward_rates_mol_s: Mapping[str, float]
    reverse_rates_mol_s: Mapping[str, float]
    standard_delta_g_J_mol: Mapping[str, float]
    affinities_J_mol: Mapping[str, float | None]


def chemostatted_readout(
    spec: CatalyticMechanismSpec,
    slot: StateSlot,
    boundary: CatalyticReservoirBoundary,
    potential_V: float,
    state: CoupledStateView,
) -> ChemostattedElectricalReadout:
    rates = open_catalytic_rates(spec, slot, boundary, potential_V, state)
    affinities = {}
    for reaction in spec.reactions:
        stoich = reaction.stoichiometry
        affinities[reaction.reaction_id] = (
            rates.delta_g_J_mol[reaction.reaction_id]
            + R * state.temperature_K * sum(v * log(rates.activities[k]) for k, v in stoich.items())
            if all(rates.activities[k] > 0 for k in stoich)
            else None
        )
    return ChemostattedElectricalReadout(
        potential_V,
        faradaic_current(spec, rates),
        {k: rates.activities[k] for k in spec.adsorbates},
        rates.forward_rates_mol_s,
        rates.reverse_rates_mol_s,
        rates.delta_g_J_mol,
        affinities,
    )


def chemostatted_catalytic_system(
    spec: CatalyticMechanismSpec,
    slot: StateSlot,
    boundary: CatalyticReservoirBoundary,
    potential_V: float,
    *,
    support_heat_capacity_J_K: float = 0.0,
) -> CoupledSystem:
    finite_number(support_heat_capacity_J_K, "support heat capacity")
    if support_heat_capacity_J_K < 0:
        raise ValueError("support heat capacity cannot be negative")
    validate_boundary(spec, boundary, potential_V)
    validate_catalytic_slot(slot, spec)
    species = {
        **{k: spec.species_by_id[k] for k in (*spec.gas_species, *spec.reservoir_species)},
        **{catalytic_inventory_key(slot, k): spec.species_by_id[k] for k in spec.adsorbates},
    }
    base = CatalyticThermodynamics(
        species,
        support_heat_capacity_J_K
        + slot.values["heat_capacity_J_K"]
        + boundary.carrier.volumetric_cv_J_L_K * boundary.liquid_volume_L,
        boundary.carrier.feed_temperature_K,
    )
    thermo = with_inventory_interactions(base, (slot,))

    def rates(state: CoupledStateView) -> CatalyticRates:
        return open_catalytic_rates(spec, slot, boundary, potential_V, state, solver_trial=True)

    def compensate(state: CoupledStateView) -> ProcessContribution:
        result = rates(state)
        current = faradaic_current(spec, result)
        port_rates, enthalpies = {}, {}
        for key in boundary.solutes:
            dn = result.amount_rates_mol_s[key]
            for direction, rate in (("in", max(-dn, 0)), ("out", max(dn, 0))):
                name = f"reservoir:{key}:{direction}"
                port_rates[name] = rate
                enthalpies[name] = (
                    spec.species_by_id[key].evaluate(state.temperature_K).enthalpy_J_mol
                )
        return ProcessContribution(
            port_rates_mol_s=port_rates,
            port_enthalpies_J_mol=enthalpies,
            coordinate_rates={
                "faradaic_charge_C": current,
                "electrical_work_J": potential_V * current,
            },
        )

    def bath_power(state: CoupledStateView) -> float:
        # Trial extension is only for constitutive root evaluation. Accepted n
        # still goes through original CoupledSystem nonnegative/ownership checks.
        state = replace(state, amounts_mol={k: max(v, 0) for k, v in state.amounts_mol.items()})
        result = rates(state)
        derivatives = {**result.amount_rates_mol_s, **dict.fromkeys(boundary.solutes, 0.0)}
        flow = -sum(
            result.amount_rates_mol_s[k]
            * spec.species_by_id[k].evaluate(state.temperature_K).enthalpy_J_mol
            for k in boundary.solutes
        )
        return (
            thermo.chemical_energy_rate_w(state, derivatives, ())
            - flow
            - potential_V * faradaic_current(spec, result)
        )

    laws = {
        element: {k: s.species.composition.get(element, 0) for k, s in species.items()}
        for element in sorted({e for s in species.values() for e in s.species.composition})
    }
    laws["matter_minus_electrode_charge_C"] = {
        **{k: F * s.species.charge for k, s in species.items()},
        "faradaic_charge_C": -1.0,
    }

    def physical_state(state: CoupledStateView) -> None:
        if (
            set(state.coordinates) != set(ELECTRICAL_FIELDS)
            or not isclose(state.temperature_K, boundary.bath.setpoint_K, rel_tol=0, abs_tol=1e-9)
            or not isclose(
                state.volume_L, boundary.gas_volume_L + boundary.liquid_volume_L, rel_tol=1e-12
            )
            or any(
                not isclose(state.amounts_mol[k], n, rel_tol=1e-10, abs_tol=1e-25)
                for k, n in boundary.amounts_mol.items()
            )
        ):
            raise ValueError("chemostat original temperature/volume/amount/charge layout differs")

    return CoupledSystem(
        thermo,
        (
            catalytic_process(
                slot,
                spec,
                reservoir_activities=boundary.activities,
                potential_V=potential_V,
                gas_volume_L=boundary.gas_volume_L,
            ),
            CoupledProcess(
                "real-aqueous-reservoir-exchange", compensate, validate_state=physical_state
            ),
        ),
        ports=reservoir_ports(boundary),
        heat_flux=lambda state: HeatFlux(work_W=potential_V * faradaic_current(spec, rates(state))),
        isothermal=True,
        conservation_laws=laws,
        coordinates=(
            CoupledCoordinate("faradaic_charge_C", "C", 1.0),
            CoupledCoordinate("electrical_work_J", "J", 1.0),
        ),
        events=(
            CoupledEvent(
                "gas_pressure_limit",
                lambda state: (
                    boundary.maximum_pressure_Pa
                    - sum(state.amounts_mol[k] for k in spec.gas_species)
                    * R
                    * state.temperature_K
                    / (boundary.gas_volume_L / 1000)
                ),
            ),
            CoupledEvent(
                "electrode_current_limit",
                lambda state: (
                    boundary.maximum_abs_current_A
                    - abs(
                        faradaic_current(
                            spec,
                            rates(
                                replace(
                                    state,
                                    amounts_mol={
                                        k: max(v, 0) for k, v in state.amounts_mol.items()
                                    },
                                )
                            ),
                        )
                    )
                ),
            ),
            CoupledEvent(
                "bath_heating_limit",
                lambda state: boundary.bath.maximum_power_W - bath_power(state),
            ),
            CoupledEvent(
                "bath_cooling_limit",
                lambda state: boundary.bath.maximum_cooling_power_W + bath_power(state),
            ),
        ),
        inventory_scales_mol={
            k: max(boundary.amounts_mol[k], 1e-16) if k in boundary.solutes else 1e-6
            for k in species
        },
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10, atol=1e-12),
    )
