from __future__ import annotations

import pytest

from chemworld.physchem import (
    STANDARD_PRESSURE_PA,
    ActivityModelSpec,
    MaturityLevel,
    activity_coefficients,
    activity_model_cards,
    binary_azeotrope_diagnostic_report,
    bubble_pressure_pa,
    bubble_temperature_report,
    curated_property_package,
    dew_pressure_pa,
    dew_temperature_report,
    flash_isothermal,
    gamma_phi_k_value_report,
    liquid_liquid_split,
    lle_phase_stability_diagnostic,
    rachford_rice_diagnostic_report,
    rachford_rice_vapor_fraction,
    raoult_k_values,
    uniquac_activity_report,
    validate_model_card,
)


def test_ideal_and_margules_activity_coefficients() -> None:
    ideal = ActivityModelSpec("ideal_ab", ("A", "B"), "ideal")
    assert activity_coefficients(ideal, {"A": 0.2, "B": 0.8}, temperature_K=298.15) == {
        "A": 1.0,
        "B": 1.0,
    }

    margules = ActivityModelSpec(
        "margules_ab",
        ("A", "B"),
        "margules",
        {"A:A|B": 1.2, "A:B|A": 0.8},
    )
    gamma = activity_coefficients(margules, {"A": 0.2, "B": 0.8}, temperature_K=298.15)
    assert gamma["A"] > 1.0
    assert gamma["B"] > 1.0
    assert gamma["A"] > gamma["B"]


def test_wilson_activity_coefficients_match_reference_example() -> None:
    model = ActivityModelSpec(
        "wilson_ab",
        ("A", "B"),
        "wilson",
        {
            "lambda:A|B": 0.154,
            "lambda:B|A": 0.888,
        },
    )
    gamma = activity_coefficients(model, {"A": 0.252, "B": 0.748}, temperature_K=300.0)

    assert gamma["A"] == pytest.approx(1.881492608717, rel=1e-12)
    assert gamma["B"] == pytest.approx(1.165577493112, rel=1e-12)


def test_nrtl_activity_coefficients_match_reference_example() -> None:
    model = ActivityModelSpec(
        "nrtl_ab",
        ("A", "B"),
        "nrtl",
        {
            "tau:A|B": 0.1759,
            "tau:B|A": 0.7991,
            "alpha:A|B": 0.2,
            "alpha:B|A": 0.3,
        },
    )
    gamma = activity_coefficients(model, {"A": 0.1, "B": 0.9}, temperature_K=310.0)

    assert gamma["A"] == pytest.approx(2.121421, rel=5e-7)
    assert gamma["B"] == pytest.approx(1.011342, rel=5e-7)


def test_uniquac_activity_coefficients_match_reference_example() -> None:
    model = ActivityModelSpec(
        "uniquac_ab",
        ("A", "B"),
        "uniquac",
        {
            "r:A": 2.1055,
            "q:A": 1.972,
            "r:B": 0.9200,
            "q:B": 1.400,
            "tau:A|B": 1.0919744384510301,
            "tau:B|A": 0.37452902779205477,
        },
    )

    report = uniquac_activity_report(
        model,
        {"A": 0.252, "B": 0.748},
        temperature_K=343.15,
    )
    gamma = activity_coefficients(
        model,
        {"A": 0.252, "B": 0.748},
        temperature_K=343.15,
    )

    assert gamma == report.activity_coefficients
    assert report.volume_fractions["A"] + report.volume_fractions["B"] == pytest.approx(1.0)
    assert report.surface_fractions["A"] + report.surface_fractions["B"] == pytest.approx(1.0)
    assert report.tau_matrix["A"]["A"] == pytest.approx(1.0)
    assert report.tau_matrix["A"]["B"] == pytest.approx(1.0919744384510301)
    assert gamma["A"] == pytest.approx(2.35875137797083, rel=1e-12)
    assert gamma["B"] == pytest.approx(1.2442093415968987, rel=1e-12)


def test_wilson_and_nrtl_support_directional_ternary_parameters() -> None:
    components = ("A", "B", "C")
    composition = {"A": 0.2, "B": 0.3, "C": 0.5}
    wilson_parameters = {}
    nrtl_parameters = {}
    for left in components:
        for right in components:
            if left == right:
                continue
            offset = 0.05 * (components.index(left) + 1)
            offset += 0.02 * (components.index(right) + 1)
            wilson_parameters[f"lambda:{left}|{right}"] = 0.7 + offset
            nrtl_parameters[f"tau:{left}|{right}"] = 0.1 + offset
            nrtl_parameters[f"alpha:{left}|{right}"] = 0.25 + 0.01 * components.index(left)

    wilson = ActivityModelSpec("wilson_abc", components, "wilson", wilson_parameters)
    nrtl = ActivityModelSpec("nrtl_abc", components, "nrtl", nrtl_parameters)
    wilson_gamma = activity_coefficients(wilson, composition, temperature_K=320.0)
    nrtl_gamma = activity_coefficients(nrtl, composition, temperature_K=320.0)

    assert set(wilson_gamma) == set(components)
    assert set(nrtl_gamma) == set(components)
    assert all(value > 0.0 for value in wilson_gamma.values())
    assert all(value > 0.0 for value in nrtl_gamma.values())
    assert any(value != pytest.approx(1.0) for value in nrtl_gamma.values())


def test_uniquac_supports_directional_ternary_parameters() -> None:
    components = ("A", "B", "C")
    parameters = {
        "r:A": 0.92,
        "q:A": 1.4,
        "r:B": 2.1055,
        "q:B": 1.972,
        "r:C": 3.1878,
        "q:C": 2.4,
    }
    for left in components:
        for right in components:
            if left == right:
                continue
            offset = 0.02 * (components.index(left) + 1)
            offset -= 0.01 * (components.index(right) + 1)
            parameters[f"tau_a:{left}|{right}"] = offset
            parameters[f"tau_b:{left}|{right}"] = 5.0 * (components.index(right) + 1)

    model = ActivityModelSpec("uniquac_abc", components, "uniquac", parameters)
    report = uniquac_activity_report(
        model,
        {"A": 0.45, "B": 0.25, "C": 0.30},
        temperature_K=330.0,
    )

    assert set(report.activity_coefficients) == set(components)
    assert all(value > 0.0 for value in report.activity_coefficients.values())
    assert report.to_dict()["coordination_number"] == pytest.approx(10.0)
    assert any(
        value != pytest.approx(1.0)
        for value in report.activity_coefficients.values()
    )


def test_activity_model_parameter_contracts_fail_fast() -> None:
    with pytest.raises(ValueError, match="wilson requires"):
        ActivityModelSpec(
            "bad_wilson",
            ("A", "B"),
            "wilson",
            {"lambda:A|B": 0.4},
        )
    with pytest.raises(ValueError, match="NRTL alpha"):
        ActivityModelSpec(
            "bad_nrtl",
            ("A", "B"),
            "nrtl",
            {
                "tau:A|B": 0.2,
                "tau:B|A": 0.4,
                "alpha:A|B": 0.0,
                "alpha:B|A": 0.3,
            },
        )
    with pytest.raises(ValueError, match="UNIQUAC requires q:B"):
        ActivityModelSpec(
            "bad_uniquac_missing_q",
            ("A", "B"),
            "uniquac",
            {
                "r:A": 2.1,
                "q:A": 2.0,
                "r:B": 1.0,
                "tau:A|B": 1.1,
                "tau:B|A": 0.8,
            },
        )
    with pytest.raises(ValueError, match="UNIQUAC tau"):
        ActivityModelSpec(
            "bad_uniquac_tau",
            ("A", "B"),
            "uniquac",
            {
                "r:A": 2.1,
                "q:A": 2.0,
                "r:B": 1.0,
                "q:B": 1.4,
                "tau:A|B": 0.0,
                "tau:B|A": 0.8,
            },
        )


def test_activity_model_cards_are_auditable() -> None:
    cards = activity_model_cards()
    assert {card.model_id for card in cards} == {
        "activity_excess_thermodynamics",
        "wilson_activity_coefficients",
        "nrtl_activity_coefficients",
        "uniquac_activity_coefficients",
        "ideal_gamma_vle_temperature_reports",
        "gamma_phi_k_values_and_azeotrope_diagnostics",
    }
    for card in cards:
        assert card.maturity is MaturityLevel.REFERENCE_VALIDATED
        assert validate_model_card(card) == []
        assert card.validation_evidence


def test_lle_tpd_diagnostic_classifies_ideal_uniform_partition_as_single_liquid() -> None:
    report = lle_phase_stability_diagnostic(
        {"A": 1.0, "B": 1.0},
        partition_coefficients={"A": 1.0, "B": 1.0},
        aqueous_volume_L=1.0,
        organic_volume_L=1.0,
    )

    assert report.phase_status == "single_liquid"
    assert report.minimum_tpd_like >= -1e-12
    assert report.partition_log_spread == pytest.approx(0.0)
    assert report.organic_trial_composition == pytest.approx({"A": 0.5, "B": 0.5})
    assert report.aqueous_trial_composition == pytest.approx({"A": 0.5, "B": 0.5})


def test_lle_tpd_diagnostic_detects_partition_driven_two_liquid_split() -> None:
    report = lle_phase_stability_diagnostic(
        {"product": 1.0, "impurity": 0.25},
        partition_coefficients={"product": 8.0, "impurity": 0.2},
        aqueous_volume_L=1.0,
        organic_volume_L=0.4,
        stage_efficiency=0.9,
        initialization_policy="partition_weighted_test",
    )

    assert report.phase_status == "two_liquid"
    assert report.initialization_policy == "partition_weighted_test"
    assert report.minimum_tpd_like < 0.0
    assert report.partition_log_spread > 0.0
    assert report.organic_trial_composition["product"] > report.feed_composition["product"]
    assert report.aqueous_trial_composition["impurity"] > report.feed_composition["impurity"]
    assert report.to_dict()["model_id"] == "lle_tpd_style_phase_stability"


def test_liquid_liquid_split_records_stability_diagnostic_and_balances_material() -> None:
    split = liquid_liquid_split(
        {"product": 1.0, "impurity": 0.25},
        partition_coefficients={"product": 5.0, "impurity": 0.4},
        aqueous_volume_L=1.0,
        organic_volume_L=0.35,
        stage_efficiency=0.85,
        initialization_policy="unit_test_seed",
    )

    assert split.material_balance_error_mol < 1e-12
    assert split.stability_diagnostic is not None
    assert split.stability_diagnostic["initialization_policy"] == "unit_test_seed"
    assert split.stability_diagnostic["phase_status"] == "two_liquid"
    assert split.recovery_to_organic["product"] > split.recovery_to_organic["impurity"]


def test_rachford_rice_flash_balances_two_phase_split() -> None:
    z = {"light": 0.5, "heavy": 0.5}
    k_values = {"light": 2.0, "heavy": 0.5}
    beta = rachford_rice_vapor_fraction(z, k_values)
    flash = flash_isothermal(z, k_values)

    assert beta == pytest.approx(0.5)
    assert flash.vapor_fraction == pytest.approx(0.5)
    assert flash.vapor_composition["light"] > flash.liquid_composition["light"]
    assert sum(flash.liquid_composition.values()) == pytest.approx(1.0)
    assert sum(flash.vapor_composition.values()) == pytest.approx(1.0)


def test_rachford_rice_diagnostic_report_classifies_phase_regions() -> None:
    z = {"light": 0.5, "heavy": 0.5}
    two_phase = rachford_rice_diagnostic_report(
        z,
        {"light": 2.0, "heavy": 0.5},
    )
    liquid = rachford_rice_diagnostic_report(
        z,
        {"light": 0.8, "heavy": 0.3},
    )
    vapor = rachford_rice_diagnostic_report(
        z,
        {"light": 2.0, "heavy": 1.5},
    )

    assert two_phase.phase_status == "two_phase"
    assert two_phase.vapor_fraction == pytest.approx(0.5)
    assert abs(two_phase.residual) < 1e-10
    assert two_phase.to_dict()["phase_status"] == "two_phase"
    assert liquid.phase_status == "all_liquid"
    assert liquid.vapor_fraction == pytest.approx(0.0)
    assert liquid.warnings
    assert vapor.phase_status == "all_vapor"
    assert vapor.vapor_fraction == pytest.approx(1.0)
    assert vapor.warnings


def test_raoult_k_values_bubble_and_dew_pressure() -> None:
    model = ActivityModelSpec("ideal_ab", ("A", "B"), "ideal")
    psat = {"A": 100_000.0, "B": 20_000.0}
    liquid = {"A": 0.25, "B": 0.75}
    vapor = {"A": 0.6, "B": 0.4}

    bubble = bubble_pressure_pa(
        liquid,
        vapor_pressures_Pa=psat,
        activity_model=model,
        temperature_K=330.0,
    )
    dew = dew_pressure_pa(
        vapor,
        vapor_pressures_Pa=psat,
        activity_model=model,
        temperature_K=330.0,
    )
    k_values = raoult_k_values(
        model,
        liquid,
        vapor_pressures_Pa=psat,
        pressure_Pa=bubble,
        temperature_K=330.0,
    )

    assert 20_000.0 < bubble < 100_000.0
    assert 20_000.0 < dew < 100_000.0
    assert sum(liquid[key] * k_values[key] for key in liquid) == pytest.approx(1.0)


def test_gamma_phi_k_value_report_records_all_factors_and_rejects_bad_phi() -> None:
    model = ActivityModelSpec("ideal_ab", ("light", "heavy"), "ideal")
    report = gamma_phi_k_value_report(
        model,
        {"light": 0.4, "heavy": 0.6},
        vapor_pressures_Pa={"light": 120_000.0, "heavy": 40_000.0},
        pressure_Pa=100_000.0,
        temperature_K=330.0,
        vapor_fugacity_coefficients={"light": 0.8, "heavy": 1.2},
        liquid_reference_fugacity_coefficients={"light": 1.1, "heavy": 1.0},
        poynting_factors={"light": 1.02, "heavy": 1.0},
    )

    assert report.k_values["light"] == pytest.approx(
        1.0 * 120_000.0 * 1.1 * 1.02 / (0.8 * 100_000.0)
    )
    assert report.k_values["heavy"] == pytest.approx(1.0 * 40_000.0 / (1.2 * 100_000.0))
    assert report.relative_volatilities["light"] > report.relative_volatilities["heavy"]
    assert report.reference_component_id == "heavy"
    assert report.to_dict()["vapor_fugacity_coefficients"]["light"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="vapor_fugacity_coefficients"):
        gamma_phi_k_value_report(
            model,
            {"light": 0.4, "heavy": 0.6},
            vapor_pressures_Pa={"light": 120_000.0, "heavy": 40_000.0},
            pressure_Pa=100_000.0,
            temperature_K=330.0,
            vapor_fugacity_coefficients={"light": 0.0, "heavy": 1.0},
        )


def test_binary_azeotrope_diagnostic_reports_relative_volatility_crossing() -> None:
    model = ActivityModelSpec(
        "margules_crossing",
        ("light", "heavy"),
        "margules",
        {"A:light|heavy": 0.0, "A:heavy|light": 2.0},
    )
    crossing = binary_azeotrope_diagnostic_report(
        model,
        vapor_pressures_Pa={"light": 100_000.0, "heavy": 50_000.0},
        pressure_Pa=100_000.0,
        temperature_K=330.0,
        light_component_id="light",
        grid_size=51,
    )
    no_crossing = binary_azeotrope_diagnostic_report(
        ActivityModelSpec("ideal_ab", ("light", "heavy"), "ideal"),
        vapor_pressures_Pa={"light": 100_000.0, "heavy": 50_000.0},
        pressure_Pa=100_000.0,
        temperature_K=330.0,
        light_component_id="light",
        grid_size=11,
    )

    assert crossing.status == "relative_volatility_crossing"
    assert crossing.crossing_bracket is not None
    assert crossing.estimated_azeotrope_composition is not None
    assert 0.5 < crossing.estimated_azeotrope_composition["light"] < 1.0
    assert crossing.estimated_residual == pytest.approx(0.0)
    assert crossing.scan_points[0].residual > 0.0
    assert crossing.scan_points[-1].residual < 0.0
    assert crossing.to_dict()["status"] == "relative_volatility_crossing"
    assert no_crossing.status == "no_crossing"
    assert no_crossing.crossing_bracket is None
    assert no_crossing.warnings


def test_curated_binary_bubble_and_dew_temperature_reports_close() -> None:
    components = ("ethanol", "water")
    model = ActivityModelSpec("ideal_ethanol_water", components, "ideal")
    correlations = {
        component_id: curated_property_package(component_id).by_property(
            "vapor_pressure"
        )[0]
        for component_id in components
    }

    bubble = bubble_temperature_report(
        {"ethanol": 0.5, "water": 0.5},
        vapor_pressure_correlations=correlations,
        activity_model=model,
        pressure_Pa=STANDARD_PRESSURE_PA,
    )
    dew = dew_temperature_report(
        {"ethanol": 0.5, "water": 0.5},
        vapor_pressure_correlations=correlations,
        activity_model=model,
        pressure_Pa=STANDARD_PRESSURE_PA,
    )

    assert bubble.converged
    assert dew.converged
    assert bubble.temperature_K < dew.temperature_K
    assert abs(bubble.residual) < 1e-8
    assert abs(dew.residual) < 1e-8
    assert bubble.vapor_composition["ethanol"] > bubble.liquid_composition["ethanol"]
    assert dew.liquid_composition["ethanol"] < dew.vapor_composition["ethanol"]
    assert sum(
        bubble.liquid_composition[key] * bubble.k_values[key]
        for key in components
    ) == pytest.approx(1.0, rel=1e-8)
    assert sum(dew.vapor_composition[key] / dew.k_values[key] for key in components) == (
        pytest.approx(1.0, rel=1e-8)
    )
    assert set(bubble.saturation_reports) == set(components)
    assert bubble.to_dict()["saturation_reports"]["ethanol"]["converged"] is True


def test_vle_temperature_report_validation_failures_are_explicit() -> None:
    components = ("ethanol", "water")
    model = ActivityModelSpec("ideal_ethanol_water", components, "ideal")
    correlations = {
        component_id: curated_property_package(component_id).by_property(
            "vapor_pressure"
        )[0]
        for component_id in components
    }
    with pytest.raises(ValueError, match="missing"):
        bubble_temperature_report(
            {"ethanol": 0.5, "water": 0.5},
            vapor_pressure_correlations={"ethanol": correlations["ethanol"]},
            activity_model=model,
            pressure_Pa=STANDARD_PRESSURE_PA,
        )
    with pytest.raises(ValueError, match="outside the temperature bracket"):
        dew_temperature_report(
            {"ethanol": 0.5, "water": 0.5},
            vapor_pressure_correlations=correlations,
            activity_model=model,
            pressure_Pa=STANDARD_PRESSURE_PA,
            temperature_bounds_K=(280.0, 300.0),
        )


def test_lle_split_preserves_material_and_extractant_volume_improves_recovery() -> None:
    feed = {"product": 1.0, "impurity": 0.2}
    small = liquid_liquid_split(
        feed,
        partition_coefficients={"product": 4.0, "impurity": 0.5},
        aqueous_volume_L=1.0,
        organic_volume_L=0.25,
        stage_efficiency=0.9,
    )
    large = liquid_liquid_split(
        feed,
        partition_coefficients={"product": 4.0, "impurity": 0.5},
        aqueous_volume_L=1.0,
        organic_volume_L=1.0,
        stage_efficiency=0.9,
    )

    assert large.recovery_to_organic["product"] > small.recovery_to_organic["product"]
    assert large.recovery_to_organic["product"] > large.recovery_to_organic["impurity"]
    assert large.material_balance_error_mol < 1e-12


def test_phase_equilibrium_validation_fails_fast() -> None:
    model = ActivityModelSpec("ideal_ab", ("A", "B"), "ideal")
    with pytest.raises(ValueError, match="missing"):
        activity_coefficients(model, {"A": 1.0}, temperature_K=298.15)
    with pytest.raises(ValueError, match="K-value"):
        flash_isothermal({"A": 0.5, "B": 0.5}, {"A": 1.0})
    with pytest.raises(ValueError, match="phase volumes"):
        liquid_liquid_split(
            {"A": 1.0},
            partition_coefficients={"A": 1.0},
            aqueous_volume_L=0.0,
            organic_volume_L=1.0,
        )
