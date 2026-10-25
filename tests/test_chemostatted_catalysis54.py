"""Generic external stoichiometry, real chemostats and independent one-step kinetics."""

from dataclasses import replace

import pytest

from chemworld.examples.chemostatted_catalysis import example_boundary, example_mechanism
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, make_catalytic_slot
from chemworld.physchem.chemostatted_catalysis import (
    chemostatted_catalytic_system,
    chemostatted_readout,
    pack_chemostatted_state,
)
from reference.chemostatted_catalysis54_reference import equilibrium_potential, independent


def physical(kind="her", *, temperature=298.15, pH=2.0, potential=-0.2):
    spec, boundary = example_mechanism(kind), example_boundary(temperature=temperature, pH=pH)
    ads = (
        {"H_ads": 1e-7}
        if kind != "hydroxylation"
        else {"oxygen_bound": 9e-7, "hydroxyl_bound": 1e-7}
    )
    slot = make_catalytic_slot(
        "catalyst", "reactor", spec, capacities_mol={"metal": 1e-6}, initial_amounts_mol=ads
    )
    initial = pack_chemostatted_state(spec, slot, boundary, {"hydrogen": 1e-6})
    system = chemostatted_catalytic_system(
        spec, slot, boundary, potential, support_heat_capacity_J_K=0.1
    )
    return spec, boundary, slot, initial, system


@pytest.mark.parametrize("kind", ["volmer", "her", "hydroxylation"])
@pytest.mark.parametrize("temperature", [288.15, 318.15])
@pytest.mark.parametrize("pH,potential", [(2.0, -0.25), (4.0, -0.05)])
def test_analytic_and_independent_dynamic_reference(kind, temperature, pH, potential):
    spec, boundary, slot, initial, system = physical(
        kind, temperature=temperature, pH=pH, potential=potential
    )
    result = system.advance(initial, 20)
    expected = independent(kind, temperature, pH, potential, 20)
    final = result.final_state
    reading = chemostatted_readout(spec, slot, boundary, potential, final)
    key = "hydroxyl_bound" if kind == "hydroxylation" else "H_ads"
    assert reading.coverages[key] == pytest.approx(expected["theta"], rel=3e-7, abs=3e-10)
    assert final.amounts_mol["hydrogen"] == pytest.approx(expected["gas_mol"], rel=3e-7, abs=3e-15)
    for name in ("faradaic_charge_C", "electrical_work_J"):
        assert final.coordinates[name] == pytest.approx(expected[name], rel=3e-7, abs=1e-10)
    assert final.boundary_heat_J == pytest.approx(expected["bath_heat_J"], rel=3e-7, abs=1e-9)
    assert abs(result.energy_balance_residual_J) < 3e-6
    assert final.amounts_mol["proton"] == initial.amounts_mol["proton"]
    assert final.amounts_mol["chloride"] == initial.amounts_mol["chloride"]
    assert abs(result.conservation_residuals["matter_minus_electrode_charge_C"]) < 1e-8


@pytest.mark.parametrize("pH", [1.0, 3.0, 5.0])
def test_nernst_and_mass_action_activity_are_not_counted_twice(pH):
    potential = equilibrium_potential("volmer", 298.15, pH)
    spec, boundary, slot, _, _ = physical("volmer", pH=pH, potential=potential)
    slot = replace(slot, values={**slot.values, "H_ads": 5e-7})
    state = pack_chemostatted_state(spec, slot, boundary, {"hydrogen": 1e-6})
    read = chemostatted_readout(spec, slot, boundary, potential, state)
    assert read.applied_current_A == pytest.approx(0, abs=1e-15)
    assert read.affinities_J_mol["electron-proton-capture"] == pytest.approx(0, abs=1e-8)
    f = read.forward_rates_mol_s["electron-proton-capture"]
    r = read.reverse_rates_mol_s["electron-proton-capture"]
    assert f == pytest.approx(r, rel=1e-10, abs=1e-25)


def test_external_charge_elements_and_source_remain_strict():
    spec = example_mechanism()
    with pytest.raises(ValueError, match="charge"):
        replace(
            spec,
            reactions=(replace(spec.reactions[0], electron_stoichiometry=0), *spec.reactions[1:]),
        )
    with pytest.raises(ValueError, match="elements"):
        replace(
            spec,
            reactions=(
                replace(spec.reactions[0], reservoir_reactants={"chloride": 1}),
                *spec.reactions[1:],
            ),
        )
    with pytest.raises(ValueError, match="registered"):
        replace(
            spec,
            reactions=(
                replace(spec.reactions[0], reservoir_reactants={"unregistered": 1}),
                *spec.reactions[1:],
            ),
        )
    with pytest.raises(ValueError, match="explicitly registered"):
        replace(spec, reservoir_species=())
    assert CatalyticMechanismSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_unbalanced_chemostat_and_wrong_real_proton_rejected():
    spec, boundary, slot, initial, _ = physical()
    broken = replace(
        boundary,
        solutes={
            **boundary.solutes,
            "chloride": replace(boundary.solutes["chloride"], activity=0.1),
        },
    )
    with pytest.raises(ValueError, match="electroneutral"):
        chemostatted_catalytic_system(spec, slot, broken, -0.2)
    with pytest.raises(ValueError, match="real aqueous H"):
        chemostatted_catalytic_system(
            spec, slot, replace(boundary, proton_species_id="chloride"), -0.2
        )
    with pytest.raises(ValueError, match="aqueous inventory"):
        pack_chemostatted_state(
            spec,
            slot,
            boundary,
            {"hydrogen": 1e-6},
            liquid_amounts_mol={"proton": 0, "chloride": initial.amounts_mol["chloride"]},
        )


def test_old_closed_codec_has_no_new_fields():
    from reference.catalysis53_fixtures import mechanism

    spec = mechanism()
    encoded = spec.to_dict()
    assert "reservoir_species" not in encoded
    assert all(
        not {"reservoir_reactants", "reservoir_products", "electron_stoichiometry"} & r.keys()
        for r in encoded["reactions"]
    )
    assert all("aqueous_standard" not in s for s in encoded["species"])


@pytest.mark.parametrize("support", [-1, float("inf"), float("nan")])
def test_nonphysical_support_rejected(support):
    spec, boundary, slot, _, _ = physical()
    with pytest.raises(ValueError, match="heat capacity"):
        chemostatted_catalytic_system(spec, slot, boundary, -0.2, support_heat_capacity_J_K=support)


@pytest.mark.parametrize("bad", ("gamma", "potential", "carrier", "bath"))
def test_declared_boundary_cannot_silently_change_model(bad):
    spec, boundary, slot, _, _ = physical()
    with pytest.raises(ValueError):
        if bad == "gamma":
            replace(boundary.solutes["proton"], activity_coefficient=0)
        elif bad == "potential":
            chemostatted_catalytic_system(spec, slot, boundary, 0.2)
        elif bad == "carrier":
            replace(boundary, carrier=replace(boundary.carrier, solvent_index=1))
        else:
            replace(boundary, bath=replace(boundary.bath, enabled=False))


def test_external_product_and_reversed_electron_declaration_retain_same_physics():
    spec, boundary, slot, initial, system = physical("volmer")
    reaction = spec.reactions[0]
    reverse = replace(
        spec,
        reactions=(
            replace(
                reaction,
                reactants=reaction.products,
                products=reaction.reactants,
                reservoir_reactants=reaction.reservoir_products,
                reservoir_products=reaction.reservoir_reactants,
                electron_stoichiometry=-reaction.electron_stoichiometry,
            ),
        ),
    )
    final = system.advance(initial, 20).final_state
    reversed_final = (
        chemostatted_catalytic_system(reverse, slot, boundary, -0.2, support_heat_capacity_J_K=0.1)
        .advance(initial, 20)
        .final_state
    )
    assert reversed_final.amounts_mol == pytest.approx(final.amounts_mol, rel=1e-10, abs=1e-20)
    assert reversed_final.coordinates == pytest.approx(final.coordinates, rel=1e-10, abs=1e-15)
    assert reversed_final.exchanged_mol == pytest.approx(final.exchanged_mol, rel=1e-10, abs=1e-20)


def test_declared_gamma_changes_real_stock_without_counting_activity_twice():
    spec, boundary, slot, initial, _ = physical("volmer")
    changed = replace(
        boundary,
        solutes={k: replace(v, activity_coefficient=2.0) for k, v in boundary.solutes.items()},
    )
    half_stock = pack_chemostatted_state(spec, slot, changed, {"hydrogen": 1e-6})
    original = chemostatted_readout(spec, slot, boundary, -0.2, initial)
    reading = chemostatted_readout(spec, slot, changed, -0.2, half_stock)
    assert half_stock.amounts_mol["proton"] == initial.amounts_mol["proton"] / 2
    assert reading.applied_current_A == original.applied_current_A
    assert reading.forward_rates_mol_s == original.forward_rates_mol_s
    assert reading.reverse_rates_mol_s == original.reverse_rates_mol_s
