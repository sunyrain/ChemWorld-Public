"""Finite fluid/bound inventory, shared thermal solve and carrier ownership."""

from dataclasses import replace
from math import sqrt

import pytest

from chemworld.physchem.property_packages import PropertyPackageThermodynamics, SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.sorbent_kinetics import (
    SorbentKineticsSpec,
    accepted_sorbent_slot,
    advance_sorbent_contact,
    loading_field,
    make_sorbent_slot,
    sorbent_inventory,
    sorbent_key,
    validate_sorbent_slot,
)
from chemworld.physchem.specs import PropertyCorrelation


def calorics(spec, slot):
    models = {}
    for component, cp, slope in (("A", 50, 0.02), ("B", 70, 0.03)):
        if component not in spec.component_ids:
            continue
        for bound in (False, True):
            key = sorbent_key(slot, component) if bound else component
            models[key] = SpeciesCaloricModel(
                PropertyCorrelation(
                    key + ":Cv",
                    "cp_polynomial",
                    {"a": cp - slope * 298.15, "b": slope},
                    {"temperature": "K"},
                    "J/(mol*K)",
                    property_id="isochoric_heat_capacity",
                    validity_ranges={"temperature": (280, 430)},
                    source_note="Synthetic condensed Cv, not converted from Cp",
                ),
                CaloricReference(
                    phase="solid" if bound else "liquid",
                    energy_basis="internal_energy",
                    energy_J_mol=(spec.adsorption_energy_J_mol[component] if bound else 0),
                ),
            )
    return PropertyPackageThermodynamics(models, 20, 298.15)


def fixture(*, preloaded=False, single=False):
    ids = ("A",) if single else ("A", "B")
    spec = SorbentKineticsSpec(
        "synthetic",
        1,
        dict(zip(ids, (1, 3), strict=False)),
        dict(zip(ids, (0.01, 0.02), strict=False)),
        dict(zip(ids, (-10000, -20000), strict=False)),
    )
    slot = make_sorbent_slot(
        spec,
        owner_id="bed",
        location_id="reactor",
        mass_kg=0.01,
        initial_loading_mol_per_kg=({"A": 0.3, "B": 0.2} if preloaded else None),
    )
    return spec, slot


def test_closed_contact_reaches_independent_single_component_langmuir_quadratic():
    spec, slot = fixture(single=True)
    result = advance_sorbent_contact(
        spec,
        slot,
        {"A": 0.1},
        temperature_K=298.15,
        volume_L=0.1,
        duration_s=3000,
        thermodynamics=calorics(spec, slot),
        isothermal=True,
    )
    # V*C + sites*b*C/(1+b*C) = initial molecules, independently solved.
    a, b, c = 0.1, 0.1 + 0.01 - 0.1, -0.1
    concentration = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)
    expected = 0.01 * concentration / (1 + concentration)
    assert result.final_state.amounts_mol[sorbent_key(slot, "A")] == pytest.approx(
        expected, abs=1e-10
    )
    assert result.final_state.amounts_mol["A"] + expected == pytest.approx(0.1, abs=1e-10)


def test_adsorption_heat_uses_the_same_caloric_energy_and_insulated_temperature():
    spec, slot = fixture()
    result = advance_sorbent_contact(
        spec,
        slot,
        {"A": 0.1, "B": 0.05},
        temperature_K=298.15,
        volume_L=0.1,
        duration_s=300,
        thermodynamics=calorics(spec, slot),
    )
    final = result.final_state
    assert final.temperature_K > 298.15
    assert abs(result.energy_balance_residual_J) < 1e-6
    rise = final.temperature_K - 298.15
    sensible = (20 + slot.values["heat_capacity_J_K"]) * rise
    for key, n, cv, slope in (("A", 0.1, 50, 0.02), ("B", 0.05, 70, 0.03)):
        sensible += n * (cv * rise + 0.5 * slope * rise**2)
        sensible += final.amounts_mol[sorbent_key(slot, key)] * spec.adsorption_energy_J_mol[key]
    assert abs(sensible) < 1e-6
    assert final.boundary_heat_J == 0


def test_actual_preloaded_carrier_desorbs_into_finite_fresh_wash_without_reinitialization():
    spec, slot = fixture(preloaded=True)
    original = sorbent_inventory(slot, spec)
    result = advance_sorbent_contact(
        spec,
        slot,
        {"A": 0.0, "B": 0.0},
        temperature_K=318.15,
        volume_L=0.1,
        duration_s=300,
        thermodynamics=calorics(spec, slot),
        isothermal=True,
    )
    updated = accepted_sorbent_slot(slot, spec, result.final_state)
    assert updated.spec == slot.spec
    assert updated.values["sites_mol"] == slot.values["sites_mol"]
    for key in spec.component_ids:
        assert result.final_state.amounts_mol[key] > 0
        assert updated.values[loading_field(key)] < slot.values[loading_field(key)]
        assert updated.values[loading_field(key)] + result.final_state.amounts_mol[key] == (
            pytest.approx(original[sorbent_key(slot, key)], abs=1e-10)
        )


def test_full_carrier_competition_stays_within_shared_sites_and_conserves_each_species():
    spec, slot = fixture()
    slot = replace(slot, values={**slot.values, loading_field("A"): 0.01})
    result = advance_sorbent_contact(
        spec,
        slot,
        {"A": 0.0, "B": 0.1},
        temperature_K=298.15,
        volume_L=0.1,
        duration_s=300,
        thermodynamics=calorics(spec, slot),
        isothermal=True,
        evaluation_times_s=(0, 1, 10, 30, 100, 300),
    )
    for state in result.trajectory:
        assert sum(state.amounts_mol[sorbent_key(slot, k)] for k in spec.component_ids) <= (
            slot.values["sites_mol"] + 1e-10
        )
    assert result.final_state.amounts_mol["A"] > 0
    assert result.final_state.amounts_mol[sorbent_key(slot, "B")] > 0


def test_mobile_carrier_portion_carries_load_capacity_and_heat_while_fixed_stays():
    spec, fixed = fixture(preloaded=True)
    mobile = make_sorbent_slot(
        spec,
        owner_id="mobile",
        location_id="reactor",
        mass_kg=0.01,
        initial_loading_mol_per_kg={"A": 0.3, "B": 0.2},
        fixed=False,
        phase_id="liquid",
    )
    moved = mobile.portion(0.4, "receiver")
    for name, value in mobile.values.items():
        assert moved.values[name] == pytest.approx(0.4 * value)
    assert moved.location_id == "receiver"
    assert moved.spec.owner_id == mobile.spec.owner_id
    with pytest.raises(ValueError):
        fixed.portion(0.4, "receiver")
    with pytest.raises(ValueError, match="declaration"):
        validate_sorbent_slot(replace(fixed, spec=replace(fixed.spec, batch_reset="clear")), spec)


def test_inconsistent_caloric_binding_energy_is_rejected_instead_of_double_heating():
    spec, slot = fixture()
    thermo = calorics(spec, slot)
    key = sorbent_key(slot, "A")
    wrong = replace(
        thermo.species[key], reference=replace(thermo.species[key].reference, energy_J_mol=-9000)
    )
    thermo = replace(thermo, species={**thermo.species, key: wrong})
    with pytest.raises(ValueError, match="caloric offsets"):
        advance_sorbent_contact(
            spec,
            slot,
            {"A": 0.1, "B": 0.05},
            temperature_K=298.15,
            volume_L=0.1,
            duration_s=10,
            thermodynamics=thermo,
        )
