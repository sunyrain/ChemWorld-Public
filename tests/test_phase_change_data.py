"""Source identity, physical domains and real native latent-heat consumers."""

import json
from math import log

import pytest

from chemworld.data.phase_change import PhaseChangeDatabase
from chemworld.physchem.enthalpy import (
    PhaseTransitionSpec,
    phase_path_enthalpy_report,
    phase_transition_enthalpy,
)
from chemworld.physchem.phase_change_kernels import (
    boiling_vaporization_enthalpy,
    watson_exponent,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


def write(db, method, text):
    path = db.source_directory / db.source(method).relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def latent(equation="watson_hvap", **coefficients):
    return PropertyCorrelation(
        "latent", equation, coefficients, {"temperature": "K"}, "J/mol", "heat_of_vaporization"
    )


def test_methods_share_properties_derivatives_and_legacy_watson_boundary():
    curve = latent(Tc=600, T_ref=300, Hvap_ref=30000)
    assert evaluate_correlation(curve, temperature_K=300).value == 30000
    assert evaluate_correlation(curve, temperature_K=700).value == 0  # legacy behavior
    assert correlation_temperature_derivative(curve, temperature_K=300).value == pytest.approx(-38)
    with pytest.raises(ValueError, match="critical branch"):
        correlation_temperature_derivative(curve, temperature_K=600)
    for name, params in (
        ("pitzer", {"omega": 0.2}),
        ("smk", {"omega": 0.2}),
        ("mk", {"omega": 0.2}),
        ("velasco", {"omega": 0.2}),
        ("ppds12", {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}),
    ):
        curve = latent(name + "_hvap", Tc=600, **params)
        assert evaluate_correlation(curve, temperature_K=600).value == 0
        with pytest.raises(ValueError):
            correlation_temperature_derivative(curve, temperature_K=600)
        with pytest.raises(ValueError, match="supercritical"):
            evaluate_correlation(curve, temperature_K=601, validity_policy="ignore")
    with pytest.raises(ValueError, match="Psat"):
        evaluate_correlation(
            latent("clapeyron_hvap", Tc=600, Pc=1e6, Psat=2e6, dZ=1), temperature_K=300
        )


def test_anchor_estimators_are_explicit_and_exponent_is_identifiable():
    estimate = boiling_vaporization_enthalpy(
        method="Riedel",
        boiling_temperature_K=388.4,
        critical_temperature_K=620,
        critical_pressure_Pa=56.3e5,
    )
    assert estimate == pytest.approx(35089.80179000598)
    h1, h2 = 40000, 40000 * (200 / 300) ** 0.5
    assert watson_exponent(
        temperature_1_K=300,
        temperature_2_K=400,
        enthalpy_1_J_mol=h1,
        enthalpy_2_J_mol=h2,
        critical_temperature_K=600,
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="distinct"):
        watson_exponent(
            temperature_1_K=300,
            temperature_2_K=300,
            enthalpy_1_J_mol=h1,
            enthalpy_2_J_mol=h2,
            critical_temperature_K=600,
        )
    with pytest.raises(ValueError, match="below Tc"):
        boiling_vaporization_enthalpy(
            method="Chen",
            boiling_temperature_K=601,
            critical_temperature_K=600,
            critical_pressure_Pa=1e6,
        )


def test_full_catalog_lazy_identity_and_twu_is_not_latent_heat(tmp_path):
    db = PhaseChangeDatabase(tmp_path)
    assert len(db.catalog) == 10 and not db.loaded_sources
    key = "UEMGWPRHOOEKTA-UHFFFAOYSA-N"
    write(
        db,
        "BELL_TWU",
        f"Tc\tPc\tc0\tc1\tc2\tname\tinchikey\n548.4\t4203546\t0.2\t0.8\t2.0\tfluid\t{key}\n",
    )
    record = db.record(key, method="BELL_TWU")
    assert record.cas_number is None and record.raw("Tc") == "548.4"
    assert db.value(key, "Pc", method="BELL_TWU").quantity("bar").value == pytest.approx(42.03546)
    assert db.value(key, "c0", method="BELL_TWU").quantity().value == 0.2
    assert db.latent_heats(key, method="BELL_TWU") == ()


def test_points_require_temperature_and_do_not_extrapolate_even_with_ignore(tmp_path):
    db = PhaseChangeDatabase(tmp_path)
    write(db, "CRC_FUSION", "CAS\tChemical\tFormula\tHfus\n7732-18-5\twater\tH2O\t6010\n")
    with pytest.raises(ValueError, match="missing physical"):
        db.latent_heat("7732-18-5", method="CRC_FUSION")
    curve = db.latent_heat("7732-18-5", method="CRC_FUSION", parameters={"T_ref": 273.15})
    card = db.property_input(
        "7732-18-5",
        ComponentSpec("water", "H2O"),
        method="CRC_FUSION",
        parameters={"T_ref": 273.15},
    )
    assert PropertyInputCard.from_json(json.dumps(card.to_dict())).to_dict() == card.to_dict()
    assert evaluate_correlation(curve, temperature_K=273.15).value == 6010
    with pytest.raises(ValueError, match="extrapolated"):
        evaluate_correlation(curve, temperature_K=300, validity_policy="ignore")
    with pytest.raises(ValueError, match="no temperature derivative"):
        correlation_temperature_derivative(curve, temperature_K=273.15)
    with pytest.raises(ValueError, match="temperature interval"):
        db.latent_heat(
            "7732-18-5",
            method="CRC_FUSION",
            parameters={"T_ref": 273.15},
            temperature_bounds_K=(270, 280),
        )
    transition = PhaseTransitionSpec("melt", "solid", "liquid", 273.15, curve)
    assert (
        phase_transition_enthalpy(transition, from_phase="liquid", to_phase="solid").value == -6010
    )
    with pytest.raises(ValueError, match="positive phase direction"):
        PhaseTransitionSpec("wrong", "liquid", "gas", 273.15, curve)


def test_table_missing_zero_and_interval_derivatives_are_preserved(tmp_path):
    db = PhaseChangeDatabase(tmp_path)
    write(
        db,
        "VDI",
        json.dumps({"7732-18-5": {"T": [300, 400, 500, 600], "Hvap": [30000, None, 10000, 0]}}),
    )
    (data,) = db.latent_heats("7732-18-5", method="VDI")
    assert data.enthalpies_J_mol == (30000, None, 10000, 0)
    with pytest.raises(ValueError, match="missing"):
        data.build_segments(interpolation="linear")
    curve = data.build(interpolation="linear", temperature_bounds_K=(500, 600))
    assert evaluate_correlation(curve, temperature_K=600).value == 0
    assert evaluate_correlation(curve, temperature_K=550).value == 5000
    assert correlation_temperature_derivative(curve, temperature_K=550).value == -100
    with pytest.raises(ValueError, match="beyond source"):
        data.build(interpolation="linear", temperature_bounds_K=(500, 601))


def test_native_input_phase_path_and_flash_consume_source_latent_heat(tmp_path):
    from chemworld.physchem.flash_units import tp_flash_with_energy_balance

    db = PhaseChangeDatabase(tmp_path)
    write(
        db,
        "PERRY_VAPORIZATION",
        "CAS\tChemical\tTc\tC1\tC2\tC3\tC4\tTmin\tTmax\n"
        "7732-18-5\twater\t600\t40000\t0.5\t0\t0\t300\t590\n",
    )
    component = ComponentSpec("water", "H2O")
    card = db.property_input("7732-18-5", component, method="PERRY_VAPORIZATION")
    assert PropertyInputCard.from_json(json.dumps(card.to_dict())).to_dict() == card.to_dict()
    (curve,) = card.package.correlations
    h = evaluate_correlation(curve, temperature_K=350).value
    cp = PropertyCorrelation(
        "cp", "cp_polynomial", {"a": 30}, {"temperature": "K"}, "J/(mol*K)", "heat_capacity"
    )
    transition = PhaseTransitionSpec("boil", "liquid", "gas", 350, curve)
    path = phase_path_enthalpy_report(
        component_id="water",
        heat_capacity_correlations={"liquid": cp, "gas": cp},
        transitions=(transition,),
        initial_phase="liquid",
        final_phase="gas",
        initial_temperature_K=320,
        final_temperature_K=380,
    )
    assert path.total_enthalpy_J_mol == pytest.approx(1800 + h)
    flash = tp_flash_with_energy_balance(
        {"light": 0.5, "heavy": 0.5},
        temperature_K=350,
        pressure_Pa=1e5,
        vapor_pressures_Pa={"light": 2e5, "heavy": 5e4},
        feed_molar_enthalpies_J_mol={"light": 0, "heavy": 0},
        liquid_molar_enthalpies_J_mol={"light": 0, "heavy": 0},
        vapor_molar_enthalpies_J_mol={"light": h, "heavy": 1.5 * h},
    )
    assert flash.converged and flash.vapor_fraction == pytest.approx(0.5)
    assert flash.energy_balance_error_J < 1e-10 and flash.material_balance_error_mol < 1e-10
    assert flash.heat_duty_J == pytest.approx(h / 3 + 1.5 * h / 6)


def test_sublimation_has_explicit_direction_and_missing_source_stays_missing(tmp_path):
    db = PhaseChangeDatabase(tmp_path)
    write(
        db,
        "GHAZARATI_SUBLIMATION",
        "CAS\tChemical\tHsub\terror\n7732-18-5\twater\t51000\t1000\nmissing\tempty\t\t\n",
    )
    curve = db.latent_heat("7732-18-5", method="GHAZARATI_SUBLIMATION")
    transition = PhaseTransitionSpec("sublime", "solid", "gas", 298.15, curve)
    assert phase_transition_enthalpy(transition, from_phase="gas", to_phase="solid").value == -51000
    with pytest.raises(ValueError, match="missing or invalid"):
        db.latent_heat("missing", method="GHAZARATI_SUBLIMATION")
    assert db.value("missing", "Hsub", method="GHAZARATI_SUBLIMATION").quality == "missing"


def test_clapeyron_fixed_pressure_derivative_is_partial():
    curve = latent("clapeyron_hvap", Tc=600, Pc=3e6, Psat=1e5, dZ=0.9)
    derivative = correlation_temperature_derivative(curve, temperature_K=300).value
    assert derivative == pytest.approx(8.31446261815324 * 0.9 * log(30) / (1 - 300 / 600) ** 2)


def test_legacy_latent_scale_derivative_uses_convertible_native_unit():
    curve = PropertyCorrelation(
        "kj",
        "watson_hvap",
        {"Tc": 600, "T_ref": 300, "Hvap_ref": 30},
        {"temperature": "K"},
        "kJ/mol",
        "heat_of_vaporization",
    )
    result = correlation_temperature_derivative(curve, temperature_K=300)
    assert result.to("kJ/(mol*K)").value == pytest.approx(-0.038)


def test_imported_latent_heat_drives_equipment_plateau(tmp_path):
    from chemworld.physchem.heat_transfer_units import (
        HeatTransferEquipmentSpec,
        PhaseChangeBoundarySpec,
        equipment_heat_transfer,
    )

    db = PhaseChangeDatabase(tmp_path)
    write(
        db,
        "CRC_VAPORIZATION",
        "CAS\tChemical\tFormula\tTb\tHvapTb\tHvap298\n"
        "7732-18-5\twater\tH2O\t373.15\t40650\t44000\n",
    )
    data = db.latent_heats("7732-18-5", method="CRC_VAPORIZATION")[0]
    h = evaluate_correlation(data.build(), temperature_K=373.15).value
    equipment = HeatTransferEquipmentSpec(
        equipment_id="coil",
        surface_type="coil",
        area_m2=2,
        clean_overall_u_W_m2_K=300,
        provenance_id="synthetic-device",
    )
    result = equipment_heat_transfer(
        equipment,
        process_temperature_K=373.15,
        utility_temperature_K=400,
        process_heat_capacity_J_K=10000,
        duration_s=1,
        phase_change=PhaseChangeBoundarySpec("boiling", 373.15, h, 2, data.curve_id),
    )
    assert result.final_temperature_K == 373.15
    assert result.phase_changed_mol == pytest.approx(600 * (400 - 373.15) / h)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-10)
    assert result.provenance["phase_change"] == data.curve_id


def test_chemsep_units_and_absent_melting_point_require_explicit_condition(tmp_path):
    db = PhaseChangeDatabase(tmp_path)
    write(
        db,
        "CHEMSEP_XML",
        '<compounds><compound><CAS value="7732-18-5"/>'
        '<HeatOfFusionAtMeltingPoint units="J/kmol" value="6010000"/>'
        "</compound></compounds>",
    )
    (data,) = db.latent_heats("7732-18-5", method="CHEMSEP_XML")
    with pytest.raises(ValueError, match="missing physical"):
        data.build()
    assert (
        evaluate_correlation(data.build(parameters={"T_ref": 273.15}), temperature_K=273.15).value
        == 6010
    )
