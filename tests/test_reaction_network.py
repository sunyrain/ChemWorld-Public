from __future__ import annotations

import math
from pathlib import Path

import pytest

from chemworld.physchem import (
    NASA7SpeciesThermo,
    NASA7TemperatureSegment,
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
    cantera_comparable_reaction_cases,
    effective_third_body_concentration,
    evaluate_rate_law,
    evaluate_reaction_ode_reference_case,
    finite_difference_reaction_sensitivities,
    kinetic_sensitivity_parameter_candidates,
    lindemann_falloff_rate_constant,
    load_mechanism,
    parse_reaction_equation,
    perturb_network_parameters,
    reaction_kinetics_model_cards,
    reverse_rate_constant_from_equilibrium,
    thermochemical_concentration_equilibrium_constant,
    thermochemical_detailed_balance,
    troe_broadening_factor,
    troe_falloff_rate_constant,
    validate_model_card,
)


def _constant_cp_thermo(
    species_id: str,
    *,
    a5: float = 0.0,
    a6: float = 0.0,
) -> NASA7SpeciesThermo:
    return NASA7SpeciesThermo(
        species_id=species_id,
        segments=(
            NASA7TemperatureSegment(
                min_temperature_K=200.0,
                max_temperature_K=2500.0,
                coefficients=(3.5, 0.0, 0.0, 0.0, 0.0, a5, a6),
                label=f"{species_id}:constant-cp",
            ),
        ),
    )


def test_parse_reaction_equation_and_stoichiometric_matrix() -> None:
    stoich, reversible = parse_reaction_equation("CO2 + H2 <=> CO + H2O")
    assert reversible
    assert stoich == {"CO2": -1.0, "H2": -1.0, "CO": 1.0, "H2O": 1.0}

    network = ReactionNetworkSpec(
        network_id="water_gas_shift",
        species=(
            SpeciesSpec("CO2", "CO2", phase="gas"),
            SpeciesSpec("H2", "H2", phase="gas"),
            SpeciesSpec("CO", "CO", phase="gas"),
            SpeciesSpec("H2O", "H2O", phase="gas"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="wgs",
                equation="CO2 + H2 <=> CO + H2O",
                rate_law=RateLawSpec(
                    "wgs_reversible",
                    "reversible_arrhenius",
                    {"A": 0.1, "Ea_J_per_mol": 1000.0, "K_eq": 1.0},
                ),
            ),
        ),
    )

    assert network.species_ids == ("CO2", "H2", "CO", "H2O")
    assert network.reaction_ids == ("wgs",)
    assert network.stoichiometric_matrix() == ((-1.0,), (-1.0,), (1.0,), (1.0,))
    assert network.check_element_balance()


def test_element_balance_catches_impossible_reaction() -> None:
    with pytest.raises(ValueError, match="not element balanced"):
        ReactionNetworkSpec(
            network_id="bad_network",
            species=(
                SpeciesSpec("A", "H2"),
                SpeciesSpec("B", "O2"),
            ),
            reactions=(
                ReactionSpec.from_equation(
                    reaction_id="bad",
                    equation="A => B",
                    rate_law=RateLawSpec("bad_rate", "mass_action", {"k": 1.0}),
                ),
            ),
        )


def test_yaml_mechanism_loads_and_reproduces_batch_qualitative_behavior() -> None:
    network = load_mechanism("configs/mechanisms/simple_batch_reaction.yaml")
    assert network.network_id == "simple_batch_reaction"
    assert network.check_element_balance()
    assert len(network.species) == 7
    assert len(network.reactions) == 5

    short = network.integrate_batch(
        {"A": 1.0, "P": 0.0, "B": 0.0, "D": 0.0, "E": 0.0, "Cat_active": 0.01},
        volume_L=1.0,
        temperature_K=360.0,
        duration_s=1000.0,
    )
    long = network.integrate_batch(
        {"A": 1.0, "P": 0.0, "B": 0.0, "D": 0.0, "E": 0.0, "Cat_active": 0.01},
        volume_L=1.0,
        temperature_K=360.0,
        duration_s=20000.0,
    )

    assert short.final_amounts_mol["P"] > 0.0
    assert long.final_amounts_mol["D"] > short.final_amounts_mol["D"]
    assert long.final_amounts_mol["Cat_dead"] > short.final_amounts_mol["Cat_dead"]
    assert long.final_amounts_mol["A"] < short.final_amounts_mol["A"]


def test_rate_law_variants_return_finite_nonnegative_rates() -> None:
    concentrations = {"A": 1.0, "B": 0.5, "P": 0.2, "Cat": 0.01}
    reactions = (
        ReactionSpec.from_equation(
            reaction_id="mass_action",
            equation="A => B",
            rate_law=RateLawSpec("ma", "mass_action", {"k": 0.2}),
        ),
        ReactionSpec.from_equation(
            reaction_id="modified_arrhenius",
            equation="A => B",
            rate_law=RateLawSpec("arr", "modified_arrhenius", {"A": 0.2, "b": 0.5}),
        ),
        ReactionSpec.from_equation(
            reaction_id="catalytic",
            equation="A => B",
            rate_law=RateLawSpec(
                "cat",
                "catalytic_activity",
                {"A": 0.2, "catalyst_species": "Cat", "reference_concentration_mol_L": 0.01},
            ),
        ),
        ReactionSpec.from_equation(
            reaction_id="lh",
            equation="A + B => P",
            rate_law=RateLawSpec(
                "lh",
                "langmuir_hinshelwood",
                {"A": 0.2, "adsorption": {"A": 1.0, "B": 0.5}},
            ),
        ),
        ReactionSpec.from_equation(
            reaction_id="mm",
            equation="A => B",
            rate_law=RateLawSpec(
                "mm",
                "michaelis_menten",
                {"substrate": "A", "vmax": 0.4, "Km": 0.2},
            ),
        ),
    )

    rates = [
        evaluate_rate_law(reaction, concentrations_mol_L=concentrations, temperature_K=330.0)
        for reaction in reactions
    ]
    assert all(rate >= 0.0 for rate in rates)
    assert rates[-1] == pytest.approx(0.4 / 1.2)


def test_third_body_arrhenius_uses_collision_efficiencies() -> None:
    reaction = ReactionSpec.from_equation(
        reaction_id="third_body",
        equation="A => P",
        rate_law=RateLawSpec(
            "third_body_rate",
            "third_body_arrhenius",
            {
                "A": 2.0,
                "default_efficiency": 0.0,
                "third_body_efficiencies": {"N2": 1.0, "Ar": 0.2},
            },
        ),
    )
    concentrations = {"A": 0.5, "N2": 3.0, "Ar": 5.0}

    third_body = effective_third_body_concentration(
        concentrations,
        efficiencies={"N2": 1.0, "Ar": 0.2},
        default_efficiency=0.0,
    )
    rate = evaluate_rate_law(
        reaction,
        concentrations_mol_L=concentrations,
        temperature_K=900.0,
    )

    assert third_body == pytest.approx(4.0)
    assert rate == pytest.approx(2.0 * third_body * concentrations["A"])


def test_lindemann_falloff_matches_low_and_high_pressure_limits() -> None:
    params = {
        "low_A": 2.0,
        "high_A": 10.0,
        "default_efficiency": 0.0,
        "third_body_efficiencies": {"N2": 1.0},
    }
    low_concentrations = {"A": 1.0, "N2": 1.0e-8}
    high_concentrations = {"A": 1.0, "N2": 1.0e8}

    low_k = lindemann_falloff_rate_constant(params, low_concentrations, 800.0)
    high_k = lindemann_falloff_rate_constant(params, high_concentrations, 800.0)

    assert low_k == pytest.approx(2.0e-8, rel=1.0e-6)
    assert high_k == pytest.approx(10.0, rel=1.0e-6)


def test_troe_falloff_broadens_lindemann_rate() -> None:
    params = {
        "low_A": 30.0,
        "high_A": 10.0,
        "default_efficiency": 0.0,
        "third_body_efficiencies": {"N2": 1.0},
        "troe_a": 0.5,
        "troe_T3": 1000.0,
        "troe_T1": 10_000.0,
        "troe_T2": 5000.0,
    }
    concentrations = {"A": 1.0, "N2": 1.0}
    lindemann = lindemann_falloff_rate_constant(params, concentrations, 900.0)
    reduced_pressure = 30.0 * concentrations["N2"] / 10.0
    broadening = troe_broadening_factor(params, 900.0, reduced_pressure)
    troe = troe_falloff_rate_constant(params, concentrations, 900.0)

    assert 0.0 < broadening <= 1.0
    assert troe == pytest.approx(lindemann * broadening)
    assert troe < lindemann


def test_falloff_reaction_network_is_bath_gas_sensitive() -> None:
    network = ReactionNetworkSpec(
        network_id="falloff_bath_gas_network",
        species=(
            SpeciesSpec("A", "C2H4O2", phase="gas"),
            SpeciesSpec("P", "C2H4O2", phase="gas"),
            SpeciesSpec("N2", "N2", phase="gas"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="r1",
                equation="A => P",
                rate_law=RateLawSpec(
                    "r1_falloff",
                    "lindemann_falloff",
                    {
                        "low_A": 0.02,
                        "high_A": 0.2,
                        "default_efficiency": 0.0,
                        "third_body_efficiencies": {"N2": 1.0},
                    },
                ),
            ),
        ),
    )

    dilute_bath = network.integrate_batch(
        {"A": 1.0, "P": 0.0, "N2": 0.05},
        volume_L=1.0,
        temperature_K=850.0,
        duration_s=200.0,
    )
    dense_bath = network.integrate_batch(
        {"A": 1.0, "P": 0.0, "N2": 5.0},
        volume_L=1.0,
        temperature_K=850.0,
        duration_s=200.0,
    )

    assert dense_bath.final_amounts_mol["P"] > dilute_bath.final_amounts_mol["P"]
    assert dense_bath.final_amounts_mol["N2"] == pytest.approx(5.0)
    assert dilute_bath.final_amounts_mol["N2"] == pytest.approx(0.05)


def test_reversible_reaction_moves_toward_equilibrium_ratio() -> None:
    network = load_mechanism("configs/mechanisms/reversible_reaction.yaml")
    result = network.integrate_batch(
        {"A": 1.0, "B": 0.0},
        volume_L=1.0,
        temperature_K=350.0,
        duration_s=20000.0,
    )
    ratio = result.final_amounts_mol["B"] / result.final_amounts_mol["A"]
    assert ratio == pytest.approx(4.0, rel=0.05)


def test_thermochemical_detailed_balance_sets_reverse_rate_from_nasa7() -> None:
    temperature = 600.0
    target_k_eq = 4.0
    species_thermo = {
        "A": _constant_cp_thermo("A"),
        "B": _constant_cp_thermo("B", a6=math.log(target_k_eq)),
    }
    reaction = ReactionSpec.from_equation(
        reaction_id="thermo_reversible",
        equation="A <=> B",
        rate_law=RateLawSpec(
            "thermo_reversible_rate",
            "reversible_arrhenius",
            {"A": 0.02, "Ea_J_per_mol": 0.0, "K_eq_source": "nasa7"},
        ),
    )

    detailed_balance = thermochemical_detailed_balance(
        reaction,
        species_thermo=species_thermo,
        temperature_K=temperature,
    )

    assert detailed_balance.dimensionless_equilibrium_constant == pytest.approx(target_k_eq)
    assert detailed_balance.concentration_equilibrium_constant == pytest.approx(target_k_eq)
    assert detailed_balance.reverse_rate_constant == pytest.approx(0.02 / target_k_eq)
    assert reverse_rate_constant_from_equilibrium(
        forward_rate_constant=0.02,
        concentration_equilibrium_constant=target_k_eq,
    ) == pytest.approx(detailed_balance.reverse_rate_constant)

    net_rate = evaluate_rate_law(
        reaction,
        concentrations_mol_L={"A": 0.25, "B": 1.0},
        temperature_K=temperature,
        species_thermo=species_thermo,
    )
    assert net_rate == pytest.approx(0.0, abs=1e-14)


def test_thermochemical_reversible_network_moves_to_nasa7_equilibrium_ratio() -> None:
    temperature = 600.0
    target_k_eq = 4.0
    species_thermo = {
        "A": _constant_cp_thermo("A"),
        "B": _constant_cp_thermo("B", a6=math.log(target_k_eq)),
    }
    network = ReactionNetworkSpec(
        network_id="thermo_reversible_network",
        species=(
            SpeciesSpec("A", "C2H4O2", phase="gas"),
            SpeciesSpec("B", "C2H4O2", phase="gas"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="r1",
                equation="A <=> B",
                rate_law=RateLawSpec(
                    "r1_thermo_reversible",
                    "reversible_arrhenius",
                    {"A": 0.02, "Ea_J_per_mol": 0.0, "K_eq_source": "nasa7"},
                ),
            ),
        ),
    )

    result = network.integrate_batch(
        {"A": 1.0, "B": 0.0},
        volume_L=1.0,
        temperature_K=temperature,
        duration_s=2000.0,
        species_thermo=species_thermo,
    )
    ratio = result.final_amounts_mol["B"] / result.final_amounts_mol["A"]
    assert ratio == pytest.approx(target_k_eq, rel=5e-3)


def test_thermochemical_equilibrium_constant_uses_concentration_standard_state() -> None:
    temperature = 700.0
    target_dimensionless_k = 8.0
    standard_concentration = 2.0
    reaction = ReactionSpec.from_equation(
        reaction_id="association",
        equation="A + B <=> C",
        rate_law=RateLawSpec(
            "association_rate",
            "reversible_arrhenius",
            {
                "A": 0.1,
                "Ea_J_per_mol": 0.0,
                "K_eq_source": "nasa7",
                "standard_concentration_mol_L": standard_concentration,
            },
        ),
    )
    species_thermo = {
        "A": _constant_cp_thermo("A"),
        "B": _constant_cp_thermo("B"),
        "C": _constant_cp_thermo(
            "C",
            a5=3.5 * temperature,
            a6=math.log(target_dimensionless_k) + 3.5 * math.log(temperature),
        ),
    }

    concentration_k, dimensionless_k, delta_g = (
        thermochemical_concentration_equilibrium_constant(
            reaction,
            species_thermo=species_thermo,
            temperature_K=temperature,
            standard_concentration_mol_L=standard_concentration,
        )
    )

    assert dimensionless_k == pytest.approx(target_dimensionless_k)
    assert concentration_k == pytest.approx(target_dimensionless_k / standard_concentration)
    assert delta_g < 0.0


def test_thermochemical_reversible_rate_fails_without_species_thermo() -> None:
    reaction = ReactionSpec.from_equation(
        reaction_id="missing_thermo",
        equation="A <=> B",
        rate_law=RateLawSpec(
            "missing_thermo_rate",
            "reversible_arrhenius",
            {"A": 0.02, "K_eq_source": "nasa7"},
        ),
    )
    with pytest.raises(ValueError, match="species_thermo"):
        evaluate_rate_law(
            reaction,
            concentrations_mol_L={"A": 1.0, "B": 0.0},
            temperature_K=600.0,
        )


def test_cantera_comparable_reaction_ode_cases_match_analytical_solutions() -> None:
    cases = cantera_comparable_reaction_cases()
    assert {case.case_id for case in cases} == {
        "cantera_comparable_irreversible_first_order",
        "cantera_comparable_reversible_first_order",
    }

    for case in cases:
        comparison = evaluate_reaction_ode_reference_case(case)
        assert comparison.passed, comparison.to_dict()

        analytical_final = case.analytical_amounts_mol(case.duration_s)
        assert comparison.analytical_final_amounts_mol == pytest.approx(
            analytical_final,
            rel=1e-12,
            abs=1e-12,
        )
        initial_total = sum(case.initial_amounts_mol.values())
        final_total = sum(comparison.final_amounts_mol.values())
        assert final_total == pytest.approx(initial_total, rel=1e-9, abs=1e-9)


def test_reaction_kinetics_model_cards_are_auditable() -> None:
    cards = reaction_kinetics_model_cards()
    assert len(cards) == 1
    card = cards[0]
    assert card.maturity.value == "reference_validated"
    assert validate_model_card(card) == []
    assert any("Cantera" in note for note in card.reference_reading)
    assert any("RMG" in note for note in card.reference_reading)
    assert any(
        evidence.evidence_id == "kinetic-finite-difference-sensitivity-test"
        for evidence in card.validation_evidence
    )
    assert any(
        evidence.evidence_id == "pressure-dependent-falloff-rate-tests"
        for evidence in card.validation_evidence
    )


def test_finite_difference_reaction_sensitivity_matches_first_order_analytic() -> None:
    k = 0.04
    duration = 25.0
    network = ReactionNetworkSpec(
        network_id="sensitivity_first_order",
        species=(
            SpeciesSpec("A", "C2H4O2"),
            SpeciesSpec("P", "C2H4O2"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="r1",
                equation="A => P",
                rate_law=RateLawSpec("r1_rate", "mass_action", {"k": k}),
            ),
        ),
    )

    report = finite_difference_reaction_sensitivities(
        network,
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=350.0,
        duration_s=duration,
        observable_species_id="P",
        perturbation_log_step=1e-5,
        relative_parameter_uncertainty=0.2,
    )

    expected_product = 1.0 - math.exp(-k * duration)
    expected_sensitivity = k * duration * math.exp(-k * duration) / expected_product
    entry = report.ranked_entries()[0]
    assert report.baseline_observable_value == pytest.approx(expected_product)
    assert entry.parameter_id == "r1.k"
    assert entry.normalized_sensitivity == pytest.approx(expected_sensitivity, rel=5e-4)
    assert report.uncertainty_summary["normalized_std_estimate"] == pytest.approx(
        abs(expected_sensitivity) * 0.2,
        rel=5e-4,
    )
    assert report.explanation_ranking()[0]["parameter_id"] == "r1.k"
    assert report.to_dict()["entries"][0]["direction"] == "positive"


def test_reaction_sensitivity_handles_zero_baseline_observable() -> None:
    network = ReactionNetworkSpec(
        network_id="sensitivity_zero_baseline",
        species=(
            SpeciesSpec("A", "C2H4O2"),
            SpeciesSpec("P", "C2H4O2"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="r1",
                equation="A => P",
                rate_law=RateLawSpec("r1_rate", "mass_action", {"k": 0.04}),
            ),
        ),
    )

    report = finite_difference_reaction_sensitivities(
        network,
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=350.0,
        duration_s=0.0,
        observable_species_id="P",
    )

    assert report.baseline_observable_value == pytest.approx(0.0)
    assert report.entries[0].normalized_sensitivity is None
    assert report.uncertainty_summary["normalized_std_estimate"] is None


def test_kinetic_sensitivity_candidates_and_failures_are_explicit() -> None:
    network = ReactionNetworkSpec(
        network_id="sensitivity_candidates",
        species=(
            SpeciesSpec("A", "C2H4O2"),
            SpeciesSpec("P", "C2H4O2"),
        ),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="r1",
                equation="A => P",
                rate_law=RateLawSpec(
                    "r1_rate",
                    "arrhenius",
                    {"A": 1.2, "Ea_J_per_mol": 10_000.0},
                ),
            ),
        ),
    )

    assert kinetic_sensitivity_parameter_candidates(network) == (("r1", "A"),)
    with pytest.raises(ValueError, match="Unknown reaction id"):
        finite_difference_reaction_sensitivities(
            network,
            {"A": 1.0},
            volume_L=1.0,
            temperature_K=350.0,
            duration_s=10.0,
            observable_species_id="P",
            parameters=(("missing", "A"),),
        )
    with pytest.raises(ValueError, match="too large"):
        finite_difference_reaction_sensitivities(
            network,
            {"A": 1.0},
            volume_L=1.0,
            temperature_K=350.0,
            duration_s=10.0,
            observable_species_id="P",
            perturbation_log_step=0.5,
        )


def test_parameter_perturbation_is_deterministic() -> None:
    network = load_mechanism("configs/mechanisms/simple_batch_reaction.yaml")
    first = perturb_network_parameters(network, seed=123, relative_std=0.1)
    second = perturb_network_parameters(network, seed=123, relative_std=0.1)
    third = perturb_network_parameters(network, seed=456, relative_std=0.1)

    assert first.to_dict() == second.to_dict()
    assert first.to_dict() != third.to_dict()


def test_json_mechanism_loads(tmp_path: Path) -> None:
    mechanism_path = tmp_path / "simple.json"
    mechanism_path.write_text(
        """
        {
          "network_id": "json_isomerization",
          "species": [
            {"species_id": "A", "formula": "C2H4O2"},
            {"species_id": "B", "formula": "C2H4O2"}
          ],
          "reactions": [
            {
              "reaction_id": "r1",
              "equation": "A => B",
              "rate_law": {
                "rate_law_id": "r1_rate",
                "equation_id": "mass_action",
                "parameters": {"k": 0.1}
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    network = load_mechanism(mechanism_path)
    assert network.network_id == "json_isomerization"
    assert network.check_element_balance()


def test_large_network_runs_deterministically() -> None:
    species = tuple(
        SpeciesSpec(f"S{i}", "H2", phase="gas")
        for i in range(20)
    )
    reactions = tuple(
        ReactionSpec.from_equation(
            reaction_id=f"r{i}",
            equation=f"S{i % 20} => S{(i + 1) % 20}",
            rate_law=RateLawSpec(f"k{i}", "mass_action", {"k": 0.001 + i * 1e-5}),
        )
        for i in range(30)
    )
    network = ReactionNetworkSpec("large_deterministic", species, reactions)

    result_a = network.integrate_batch(
        {"S0": 1.0},
        volume_L=1.0,
        temperature_K=300.0,
        duration_s=1000.0,
        evaluation_times_s=(0.0, 500.0, 1000.0),
    )
    result_b = network.integrate_batch(
        {"S0": 1.0},
        volume_L=1.0,
        temperature_K=300.0,
        duration_s=1000.0,
        evaluation_times_s=(0.0, 500.0, 1000.0),
    )

    assert result_a.to_dict() == result_b.to_dict()
    assert len(network.stoichiometric_matrix()) == 20
    assert len(network.stoichiometric_matrix()[0]) == 30
