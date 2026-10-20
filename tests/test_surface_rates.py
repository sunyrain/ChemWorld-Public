"""Native parameter, stoichiometry, ownership and coupled-physics checks.

These finite unit fixtures do not run the separate Cantera reference matrix or
claim empirical catalyst validation. Surface and bulk share actual finite gas.
"""

import json
from dataclasses import replace
from math import exp

import pytest

import chemworld.physchem.coupled_system as coupled
from chemworld.foundation.state_ownership import StateSlot, StateSlots
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.physchem.surface_rates import (
    adsorbate_key,
    evaluate_surface_rates,
    gas_amounts_from_pressure,
    make_surface_system,
    simulate_surface,
    surface_inventory,
    surface_process,
    surface_state,
    surface_thermodynamics,
    updated_surface_slot,
)
from chemworld.physchem.surface_specs import (
    DEFAULT_SURFACE_MECHANISM,
    GAS_SPECIES,
    SURFACE_STOICHIOMETRY,
    SurfaceMechanismSpec,
    bulk_co_oxidation_network,
    make_surface_slot,
    surface_species,
)


def state(slots, temperature=350, fractions=None):
    fractions = {"CO": 0.2, "O2": 0.1, "N2": 0.7} if fractions is None else fractions
    return surface_state(
        gas_amounts_from_pressure(
            fractions, pressure_Pa=100000, temperature_K=temperature, volume_L=0.01
        ),
        slots,
        temperature_K=temperature,
        volume_L=0.01,
    )


def test_each_step_conserves_elements_and_sites_with_two_vacancies_for_oxygen():
    definitions = {item.species_id: item for item in surface_species()}
    vacancy = (-1, 1, -2, 2, 2)
    for step in range(5):
        for element in ("C", "O", "N"):
            assert (
                sum(
                    coefficients[step] * definitions[species].composition.get(element, 0)
                    for species, coefficients in SURFACE_STOICHIOMETRY.items()
                )
                == 0
            )
        assert (
            vacancy[step]
            + sum(SURFACE_STOICHIOMETRY[species][step] for species in ("CO_ads", "O_ads"))
            == 0
        )
    assert SURFACE_STOICHIOMETRY["O_ads"][2] == 2
    assert vacancy[2] == -2


def test_reference_point_uses_pressure_activities_mol_and_correct_coverage_factor():
    slot = make_surface_slot("catalyst", "reactor", initial_coverages=(0.3, 0.2))
    initial = state((slot,))
    result = evaluate_surface_rates(initial, slot)
    expected = (0.2, 0.06, 0.1, 0.002, 0.6 * exp(-5000 * 0.3 / (R_J_PER_MOL_K * 350)))
    assert result.per_site_rates_s_inv == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert result.progress_rates_mol_s == pytest.approx(tuple(2.5e-7 * x for x in expected))
    assert result.partial_pressures_Pa["CO"] == pytest.approx(20000)
    assert sum(initial.amounts_mol[key] for key in GAS_SPECIES) == pytest.approx(
        100000 * 1e-5 / (R_J_PER_MOL_K * 350),
    )


@pytest.mark.parametrize("temperature", [300, 350, 430])
def test_arrhenius_temperature_law_is_reference_normalized(temperature):
    slot = make_surface_slot("catalyst", "reactor", initial_coverages=(0.25, 0.25))
    result = evaluate_surface_rates(state((slot,), temperature), slot)
    expected = 0.2 * exp(-50000 / R_J_PER_MOL_K * (1 / temperature - 1 / 350)) * 0.25
    assert result.per_site_rates_s_inv[1] == pytest.approx(expected, rel=1e-12)


def test_oxygen_adsorption_uses_vacancy_squared_and_two_bound_oxygen_atoms():
    spec = SurfaceMechanismSpec(rate_constants_s_inv=(0, 0, 4, 0, 0))
    slot = make_surface_slot("catalyst", "reactor", initial_coverages=(0.1, 0.2), spec=spec)
    result = evaluate_surface_rates(state((slot,)), slot, spec)
    progress = slot.values["sites_mol"] * 4 * 0.1 * 0.7**2
    assert result.amount_rates_mol_s["O2"] == pytest.approx(-progress)
    assert result.amount_rates_mol_s[adsorbate_key(slot, "O_ads")] == pytest.approx(2 * progress)


def test_missing_reactant_or_vacancy_removes_only_corresponding_forward_rates():
    slot = make_surface_slot("catalyst", "reactor")
    missing = evaluate_surface_rates(state((slot,), fractions={"N2": 1}), slot)
    assert missing.per_site_rates_s_inv == (0, 0, 0, 0, 0)
    covered = make_surface_slot("covered", "reactor", initial_coverages=(0, 1))
    rates = evaluate_surface_rates(state((covered,)), covered)
    assert rates.per_site_rates_s_inv[0] == rates.per_site_rates_s_inv[2] == 0
    assert rates.per_site_rates_s_inv[3] > 0  # Desorption can free occupied sites.


def test_area_and_site_density_scale_total_progress_without_changing_per_site_rate():
    baseline = make_surface_slot("a", "reactor", initial_coverages=(0.3, 0.2))
    original = evaluate_surface_rates(state((baseline,)), baseline)
    for slot in (
        make_surface_slot("b", "reactor", area_m2=0.02, initial_coverages=(0.3, 0.2)),
        make_surface_slot("c", "reactor", site_density_mol_m2=5e-5, initial_coverages=(0.3, 0.2)),
    ):
        result = evaluate_surface_rates(state((slot,)), slot)
        assert result.per_site_rates_s_inv == pytest.approx(original.per_site_rates_s_inv)
        assert result.progress_rates_mol_s == pytest.approx(
            tuple(2 * value for value in original.progress_rates_mol_s)
        )


def test_coverage_barrier_uses_co_coverage_and_is_applied_once():
    no_barrier = replace(DEFAULT_SURFACE_MECHANISM, coverage_activation_energy_J_mol=0)
    a = make_surface_slot("a", "reactor", initial_coverages=(0.3, 0.2), spec=no_barrier)
    b = make_surface_slot("b", "reactor", initial_coverages=(0.3, 0.2))
    ratio = (
        evaluate_surface_rates(state((b,)), b).per_site_rates_s_inv[4]
        / evaluate_surface_rates(
            state((a,)),
            a,
            no_barrier,
        ).per_site_rates_s_inv[4]
    )
    assert ratio == pytest.approx(exp(-5000 * 0.3 / (R_J_PER_MOL_K * 350)), rel=1e-12)


def test_single_step_desorption_matches_exponential_and_same_thermodynamic_heat():
    spec = SurfaceMechanismSpec(
        rate_constants_s_inv=(0, 0.2, 0, 0, 0),
        activation_energies_J_mol=(0, 0, 0, 0, 0),
    )
    slot = make_surface_slot("desorber", "reactor", initial_coverages=(0.5, 0), spec=spec)
    initial = state((slot,), fractions={"N2": 1})
    system = make_surface_system((slot,), spec=spec, isothermal=True)
    system = replace(system, policy=replace(system.policy, rtol=1e-10, atol=1e-14))
    result = system.advance(initial, 10)
    remaining = slot.values["CO_ads"] * exp(-0.2 * 10)
    released = slot.values["CO_ads"] - remaining
    assert result.final_state.amounts_mol[adsorbate_key(slot, "CO_ads")] == pytest.approx(
        remaining,
        rel=1e-6,
        abs=1e-15,
    )
    assert result.final_state.amounts_mol["CO"] == pytest.approx(released, abs=1e-15)
    assert result.final_state.boundary_heat_J == pytest.approx(100000 * released, abs=1e-9)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-9)


def test_moving_material_reuses_owned_area_sites_bound_species_and_heat_capacity():
    slot = make_surface_slot(
        "particles",
        "reactor",
        fixed=False,
        phase_id="gas_phase",
        initial_coverages=(0.6, 0.1),
        heat_capacity_J_K=2,
    )
    a = slot.portion(0.25, "vial")
    b = slot.portion(0.75, "vial")
    assert a.values["area_m2"] == pytest.approx(slot.values["area_m2"] * 0.25)
    assert a.values["heat_capacity_J_K"] == 0.5
    assert a.coverages("sites_mol") == pytest.approx(slot.coverages("sites_mol"))
    original_progress = evaluate_surface_rates(state((slot,)), slot).progress_rates_mol_s
    assert evaluate_surface_rates(state((a,)), a).progress_rates_mol_s == pytest.approx(
        tuple(0.25 * value for value in original_progress)
    )
    assert surface_thermodynamics((a,)).support_heat_capacity_J_K == 20.5
    joined = a.merge(b)
    assert joined.values == pytest.approx(slot.values)
    assert StateSlots((joined,)).species_amounts() == pytest.approx(
        {
            "CO_ads": slot.values["CO_ads"],
            "O_ads": slot.values["O_ads"],
        }
    )
    fixed = make_surface_slot("fixed", "reactor", initial_coverages=(0.2, 0.3))
    assert fixed.to_dict(include_hidden=False) == {}
    with pytest.raises(ValueError, match="Only material"):
        fixed.portion(0.5, "vial")
    assert StateSlots((fixed, slot)).stationary().slots == (fixed,)


def test_different_sized_carriers_mix_by_inventory_in_existing_state_handler():
    a = make_surface_slot(
        "a", "vial", fixed=False, phase_id="gas_phase", area_m2=0.01, initial_coverages=(0.8, 0)
    )
    b = make_surface_slot(
        "b", "vial", fixed=False, phase_id="gas_phase", area_m2=0.02, initial_coverages=(0.1, 0)
    )
    combined = StateSlots((a,)).merge(StateSlots((b,)))
    coverage = combined.coverages(a.spec.state_type, "sites_mol")
    assert coverage["CO_ads"] == pytest.approx((0.8 + 2 * 0.1) / 3)
    assert coverage["CO_ads"] != pytest.approx((0.8 + 0.1) / 2)


def test_multiple_surface_instances_share_gas_and_one_actual_solver_call(monkeypatch):
    a = make_surface_slot("a", "reactor")
    b = make_surface_slot("b", "reactor", initial_coverages=(0.8, 0))
    calls = []
    solve = coupled.solve_ode

    def counted(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(coupled, "solve_ode", counted)
    initial = state((a, b))
    result = simulate_surface(initial, (a, b), 30)
    assert calls == [1]
    assert result.final_state.amounts_mol["CO2"] > 0
    assert result.final_state.amounts_mol[adsorbate_key(a, "CO_ads")] != pytest.approx(
        result.final_state.amounts_mol[adsorbate_key(b, "CO_ads")],
        abs=1e-12,
    )
    for residual in result.conservation_residuals.values():
        assert abs(residual) < 1e-12


def test_optional_bulk_is_a_separate_network_contribution_with_shared_energy():
    slot = make_surface_slot("catalyst", "reactor")
    initial = state((slot,))
    surface_only = simulate_surface(initial, (slot,), 20, isothermal=False)
    mixed = simulate_surface(initial, (slot,), 20, include_bulk=True, isothermal=False)
    assert mixed.final_state.amounts_mol["CO2"] > surface_only.final_state.amounts_mol["CO2"]
    assert mixed.final_state.temperature_K > 350
    assert mixed.final_state.boundary_heat_J == 0
    assert mixed.energy_balance_residual_J == pytest.approx(0, abs=2e-6)
    bulk = bulk_co_oxidation_network()
    rate = bulk.reaction_rates(initial.amounts_mol, volume_L=0.01, temperature_K=350)[
        "bulk_co_oxidation"
    ]
    assert rate == pytest.approx(
        1000 * (initial.amounts_mol["CO"] / 0.01) ** 2 * initial.amounts_mol["O2"] / 0.01,
        rel=1e-12,
    )
    assert surface_thermodynamics((slot,)).internal_energy_j(mixed.final_state) == pytest.approx(
        surface_thermodynamics((slot,)).internal_energy_j(initial),
        abs=2e-6,
    )


def test_segmented_history_and_restored_owned_adsorbates_continue_without_reset():
    slot = make_surface_slot("history", "reactor", initial_coverages=(0.8, 0))
    initial = state((slot,))
    system = make_surface_system((slot,), isothermal=True)
    whole = system.advance(initial, 60)
    first = system.advance(initial, 20)
    saved = updated_surface_slot(slot, first.final_state)
    restored = StateSlot.from_dict(json.loads(json.dumps(saved.to_dict())))
    assert restored.spec.owner_id == slot.spec.owner_id
    amounts = dict(first.final_state.amounts_mol)
    amounts.update(surface_inventory(restored))
    resumed = make_surface_system((restored,), isothermal=True).advance(
        replace(first.final_state, amounts_mol=amounts),
        40,
    )
    assert dict(resumed.final_state.amounts_mol) == pytest.approx(
        dict(whole.final_state.amounts_mol),
        abs=1e-12,
        rel=1e-5,
    )
    assert resumed.final_state.boundary_heat_J == pytest.approx(
        whole.final_state.boundary_heat_J,
        abs=1e-6,
    )
    assert resumed.actual_duration_s == 40
    assert resumed.final_state.time_s == whole.final_state.time_s == 60


def test_empty_material_portion_has_no_fictitious_surface_inventory_or_activity():
    slot = make_surface_slot("empty", "vial", fixed=False, phase_id="gas_phase").portion(0, "vial")
    initial = state((slot,))
    rates = evaluate_surface_rates(initial, slot)
    assert rates.progress_rates_mol_s == (0, 0, 0, 0, 0)
    result = simulate_surface(initial, (slot,), 10)
    assert dict(result.final_state.amounts_mol) == dict(initial.amounts_mol)


@pytest.mark.parametrize(
    "changes",
    [
        {"rate_constants_s_inv": (1, 2)},
        {"activation_energies_J_mol": (0, 0, 0, 0, float("nan"))},
        {"coverage_activation_energy_J_mol": 10001},
        {"reference_temperature_K": 299},
        {"reference_pressure_Pa": 0},
        {"rate_unit": "kmol/(m2*s)"},
        {"energy_unit": "J/kmol"},
    ],
)
def test_parameter_shape_domain_and_units_are_explicitly_validated(changes):
    with pytest.raises(ValueError):
        SurfaceMechanismSpec(**changes)


def test_incompatible_mechanism_or_overfull_candidate_is_rejected_by_shared_advance():
    slot = make_surface_slot("catalyst", "reactor")
    other = replace(DEFAULT_SURFACE_MECHANISM, coverage_activation_energy_J_mol=0)
    with pytest.raises(ValueError, match="incompatible"):
        surface_process(slot, other)
    initial = state((slot,))
    bad = replace(
        initial,
        amounts_mol={
            **initial.amounts_mol,
            adsorbate_key(slot, "CO_ads"): 1.01 * slot.values["sites_mol"],
        },
    )
    with pytest.raises(ValueError, match="capacity"):
        make_surface_system((slot,)).advance(bad, 1)
    with pytest.raises(ValueError, match="capacity"):
        evaluate_surface_rates(bad, slot)
