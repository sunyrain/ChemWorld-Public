"""Generic finite gas/site reactions and same-interaction energy/rate coupling."""

from dataclasses import replace
from math import log, prod

import pytest

from chemworld.physchem.catalysis_network import (
    catalytic_inventory_key,
    catalytic_system,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
    validate_catalytic_thermodynamics,
)
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.inventory_interactions import InventoryInteraction
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from reference.catalysis53_fixtures import mechanism, state_and_slot


@pytest.mark.parametrize("dissociative", [False, True])
def test_generic_nonco_mechanism_detailed_balance_and_stoichiometry(dissociative):
    spec, slot, state = state_and_slot(mechanism(dissociative=dissociative))
    rates = evaluate_catalytic_rates(state, slot, spec)
    for reaction in spec.reactions:
        f, r = (
            rates.forward_rates_mol_s[reaction.reaction_id],
            rates.reverse_rates_mol_s[reaction.reaction_id],
        )
        quotient = prod(rates.activities[k] ** n for k, n in reaction.products.items()) / prod(
            rates.activities[k] ** n for k, n in reaction.reactants.items()
        )
        assert log(f / r) == pytest.approx(
            -rates.delta_g_J_mol[reaction.reaction_id] / (R * state.temperature_K) - log(quotient)
        )
    gas = spec.gas_species[0]
    nuclear_rate = rates.amount_rates_mol_s[gas] * (2 if dissociative else 1)
    nuclear_rate += sum(
        rates.amount_rates_mol_s[catalytic_inventory_key(slot, k)] for k in spec.adsorbates
    )
    assert nuclear_rate == pytest.approx(0, abs=1e-15)
    assert all(rates.barriers_J_mol[r.reaction_id][0] >= 0 for r in spec.reactions)


def test_same_interaction_gradient_enters_gibbs_rates_and_shared_energy():
    spec, slot, state = state_and_slot()
    matrix = spec.interaction_matrix_J_mol
    values = [state.amounts_mol[catalytic_inventory_key(slot, k)] for k in spec.adsorbates]
    total = 3e-4
    gradient = [sum(a * b for a, b in zip(row, values, strict=True)) / total for row in matrix]
    expected_u = 0.5 * sum(n * mu for n, mu in zip(values, gradient, strict=True))
    interaction = InventoryInteraction.from_slot(slot, slot.spec.energy_interactions[0])
    energy, mus = interaction.energy_and_gradient(state.amounts_mol)
    assert energy == pytest.approx(expected_u)
    assert list(mus.values()) == pytest.approx(gradient)
    rates = evaluate_catalytic_rates(state, slot, spec)
    ideal = (
        spec.species_by_id["bound_A"].evaluate(350).gibbs_J_mol
        - spec.species_by_id["He"].evaluate(350).gibbs_J_mol
    )
    assert rates.delta_g_J_mol["adsorb"] == pytest.approx(ideal + gradient[0])
    system = catalytic_system(spec, (slot,), support_heat_capacity_J_K=5)
    base = system.thermodynamics.base.internal_energy_j(state)
    assert system.thermodynamics.internal_energy_j(state) == pytest.approx(base + expected_u)


def test_mobile_slot_split_preserves_extensive_interaction_energy():
    spec, slot, state = state_and_slot(mobile=True)
    whole = InventoryInteraction.from_slot(
        slot, slot.spec.energy_interactions[0]
    ).energy_and_gradient(state.amounts_mol)[0]
    energy = 0
    for fraction in (0.25, 0.75):
        part = slot.portion(fraction, "receiver")
        packed = pack_catalytic_state(spec, (part,), {"He": 0}, temperature_K=350, volume_L=1)
        energy += InventoryInteraction.from_slot(
            part, part.spec.energy_interactions[0]
        ).energy_and_gradient(packed.amounts_mol)[0]
    assert energy == pytest.approx(whole)


@pytest.mark.parametrize("isothermal", [True, False])
def test_original_coupled_solver_preserves_atoms_sites_and_energy(isothermal):
    spec, slot, state = state_and_slot()
    system = catalytic_system(spec, (slot,), support_heat_capacity_J_K=5, isothermal=isothermal)
    initial_u = system.thermodynamics.internal_energy_j(state)
    result = system.advance(state, 0.5)
    final = result.final_state
    assert final.amounts_mol["He"] != state.amounts_mol["He"]
    assert result.conservation_residuals["He"] == pytest.approx(0, abs=1e-11)
    assert system.thermodynamics.internal_energy_j(final) - initial_u == pytest.approx(
        final.boundary_heat_J, abs=2e-6
    )
    for key, site in spec.adsorbate_sites.items():
        assert (
            0
            <= final.amounts_mol[catalytic_inventory_key(slot, key)]
            <= slot.values["capacity:" + site]
        )
    assert (final.temperature_K == 350) == isothermal


def test_bad_atom_site_data_and_thermal_table_mismatch_refuse_binding():
    spec = mechanism()
    with pytest.raises(ValueError, match="elements"):
        replace(spec, reactions=(replace(spec.reactions[0], products={"bound_A": 2}),))
    with pytest.raises(ValueError, match="site class"):
        replace(spec, reactions=(replace(spec.reactions[0], reactants={"He": 1, "vacancy:B": 1}),))
    bad = replace(
        spec.species[1],
        adsorbate=replace(spec.species[1].adsorbate, descriptors={"binding": 1, "strain": 0}),
    )
    with pytest.raises(ValueError, match="same declared"):
        validate_catalytic_thermodynamics(
            spec, CatalyticThermodynamics({**spec.species_by_id, "bound_A": bad})
        )
    with pytest.raises(ValueError, match="capacity"):
        make_catalytic_slot(
            "c",
            "r",
            spec,
            capacities_mol={"A": 1e-4, "B": 1e-4},
            initial_amounts_mol={"bound_A": 2e-4},
        )
