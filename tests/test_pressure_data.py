"""Pressure sources preserve identity, incomplete domains and native consumers."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantSourceUnavailableError,
    UnknownConstantMethodError,
)
from chemworld.data.pressure import PressureDatabase
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.vapor_pressure import vapor_pressure_temperature_derivative


def write(db, method, content):
    path = db.source_directory / db.source(method).relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_pressure_catalog_is_lazy_and_shares_the_constant_cache(tmp_path):
    db = PressureDatabase(tmp_path)
    assert len(db.catalog) == 10 and db.loaded_sources == ()
    with pytest.raises(ConstantSourceUnavailableError):
        db.records("ANTOINE_POLING")
    with pytest.raises(UnknownConstantMethodError):
        db.records("guess")
    write(
        db,
        "WEBBOOK",
        "CAS\tAntoineA\tAntoineB\tAntoineC\tAntoineTmin\tAntoineTmax\n"
        "7732-18-5\t20\t3000\t0\t250\t450\n",
    )
    rows = db.records("WEBBOOK")
    assert db.records("WEBBOOK") is rows
    assert db.loaded_sources == ("WEBBOOK",)
    curve = db.pressure("7732-18-5", method="WEBBOOK")
    assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(exp(10))


def test_pressure_keeps_duplicate_identities_and_missing_bounds(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "LANDOLT_SUBLIMATION",
        "CAS\tName\tA\tB\tC\tTmin\tTmax\n"
        "7732-18-5\tfirst\t20\t3000\t0\t\t400\n"
        "7732-18-5\tsecond\t21\t3100\t0\t250\t400\n",
    )
    candidates = db.pressures("7732-18-5", method="LANDOLT_SUBLIMATION")
    assert len(candidates) == 2
    assert candidates[0].source_bounds_K == (None, 400)
    with pytest.raises(AmbiguousConstantIdentityError):
        db.pressure("7732-18-5", method="LANDOLT_SUBLIMATION")
    with pytest.raises(ValueError, match="explicit"):
        candidates[0].build()
    curve = candidates[0].build(temperature_bounds_K=(250, 400))
    assert curve.property_id == "sublimation_pressure"
    assert curve.metadata["source_bounds_K"] == (None, 400)
    with pytest.raises(ValueError, match="extrapolates"):
        candidates[0].build(temperature_bounds_K=(250, 401))
    with pytest.raises(ValueError, match="missing"):
        replace(candidates[0], coefficients={"A": None}).build(temperature_bounds_K=(250, 400))


def test_critical_source_limit_is_not_silently_shrunk(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "WAGNER_POLING",
        "CAS\tName\tA\tB\tC\tD\tTc\tPc\tTmin\tTmax\n"
        "7732-18-5\twater\t-7\t1\t-1\t-2\t600\t3000000\t250\t600\n",
    )
    curve = db.pressure("7732-18-5", method="WAGNER_POLING")
    assert curve.validity_ranges["temperature"] == (250, 600)
    assert evaluate_correlation(curve, temperature_K=600).value == pytest.approx(3e6)
    assert vapor_pressure_temperature_derivative(curve, temperature_K=600).value == pytest.approx(
        35000
    )
    with pytest.raises(ValueError):
        vapor_pressure_temperature_derivative(curve, temperature_K=600, order=2)
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=601, validity_policy="ignore")


def test_vdi_table_interpolation_is_explicit_and_reuses_native_arrhenius(tmp_path):
    db = PressureDatabase(tmp_path)
    write(db, "VDI", json.dumps({"7732-18-5": {"T": [300, 350, 400], "P": [1000, 10000, 40000]}}))
    data = db.pressures("7732-18-5", method="VDI")[0]
    with pytest.raises(ValueError, match="interpolation"):
        data.build()
    curves = data.build_segments(interpolation="log_pressure_inverse_temperature")
    assert len(curves) == 2
    for index, curve in enumerate(curves):
        low, high = curve.validity_ranges["temperature"]
        harmonic_mid = 2 / (1 / low + 1 / high)
        assert evaluate_correlation(curve, temperature_K=harmonic_mid).value == pytest.approx(
            (data.pressures_Pa[index] * data.pressures_Pa[index + 1]) ** 0.5
        )
    with pytest.raises(ValueError, match="missing"):
        replace(data, pressures_Pa=(1000, None, 40000)).build_segments(
            interpolation="log_pressure_inverse_temperature"
        )


def test_pressure_source_input_codec_and_existing_property_saturation_consumer(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "ANTOINE_POLING",
        "CAS\tChemical\tA\tB\tC\tTmin\tTmax\n7732-18-5\twater\t8\t1000\t0\t250\t450\n",
    )
    card = db.property_input("7732-18-5", ComponentSpec("water", "H2O"), method="ANTOINE_POLING")
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    report = restored.package.vapor_pressure_report(temperature_K=350)
    expected = 10 ** (8 - 1000 / 350)
    assert report.pressure_pa == pytest.approx(expected)
    assert report.dp_dt_pa_per_k == pytest.approx(expected * log(10) * 1000 / 350**2)
    inverse = restored.package.pure_saturation_temperature_report(pressure_Pa=expected)
    assert inverse.saturation_temperature_K == pytest.approx(350, abs=1e-6)
    assert "bracketed_root_only_uniqueness_not_certified" in inverse.warnings
    assert restored.reference is None


def test_chemsep_unknown_equation_is_retained_and_cannot_masquerade_as_antoine(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "CHEMSEP_XML",
        '<compounds><compound><CAS value="7732-18-5"/>'
        '<VaporPressure units="Pa"><eqno value="999"/><A value="1"/>'
        '<Tmin units="K" value="300"/><Tmax units="K" value="400"/>'
        "</VaporPressure></compound></compounds>",
    )
    data = db.pressures("7732-18-5", method="CHEMSEP_XML")[0]
    assert data.equation_id == "unresolved_chemsep_999"
    with pytest.raises(ValueError, match="unresolved"):
        data.build()


def test_pressure_underflow_is_unavailable_not_a_valid_zero(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "LANDOLT_PRESSURE",
        "CAS\tName\tA\tB\tC\tTmin\tTmax\n7732-18-5\ttest\t20\t10000000\t0\t250\t400\n",
    )
    curve = db.pressure("7732-18-5", method="LANDOLT_PRESSURE")
    with pytest.raises(ValueError, match="underflow"):
        evaluate_correlation(curve, temperature_K=300)


def test_chemsep_200_uses_documented_reduced_basis_and_its_own_pressure_scale(tmp_path):
    db = PressureDatabase(tmp_path)
    write(
        db,
        "CHEMSEP_XML",
        '<compounds><compound><CAS value="7732-18-5"/>'
        '<CriticalTemperature units="K" value="600"/>'
        '<CriticalPressure units="Pa" value="2999000"/>'
        '<VaporPressure units="Pa"><eqno value="200"/><A value="-7"/><B value="1"/>'
        '<C value="-1"/><D value="-2"/><E value="3000000"/>'
        '<Tmin units="K" value="250"/><Tmax units="K" value="600"/>'
        "</VaporPressure></compound></compounds>",
    )
    curve = db.pressure("7732-18-5", method="CHEMSEP_XML")
    tau, Tr = 0.5, 0.5
    expected = 3e6 * exp((-7 * tau + tau**1.5 - tau**2.5 - 2 * tau**5) / Tr)
    assert curve.equation_id == "wagner25"
    assert curve.coefficients["Pc"] == 3000000
    assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(expected)
    assert "Pc=E" in " ".join(curve.metadata["source_notes"])


def test_imported_pressure_drives_existing_two_phase_flash_and_material_closure(tmp_path):
    from chemworld.physchem.flash_units import tp_flash_with_energy_balance

    db = PressureDatabase(tmp_path)
    write(
        db,
        "ANTOINE_POLING",
        "CAS\tChemical\tA\tB\tC\tTmin\tTmax\n"
        f"a\tlight\t{log(200000, 10) + 1000 / 350}\t1000\t0\t300\t400\n"
        f"b\theavy\t{log(50000, 10) + 1000 / 350}\t1000\t0\t300\t400\n",
    )
    pressures = {
        name: evaluate_correlation(
            db.pressure(identity, method="ANTOINE_POLING"), temperature_K=350
        ).value
        for identity, name in (("a", "light"), ("b", "heavy"))
    }
    result = tp_flash_with_energy_balance(
        {"light": 0.5, "heavy": 0.5},
        temperature_K=350,
        pressure_Pa=100000,
        vapor_pressures_Pa=pressures,
        feed_molar_enthalpies_J_mol={"light": 0, "heavy": 0},
        liquid_molar_enthalpies_J_mol={"light": 0, "heavy": 0},
        vapor_molar_enthalpies_J_mol={"light": 30000, "heavy": 40000},
    )
    assert result.converged and result.phase_status == "two_phase"
    assert result.vapor_fraction == pytest.approx(0.5)
    assert result.material_balance_error_mol < 1e-12
    assert result.energy_balance_error_J < 1e-12
    assert result.heat_duty_J == pytest.approx(50000 / 3)


def test_vdi_pressure_scientific_table_preserves_missing_and_invalid_quality(tmp_path):
    from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable

    db = PressureDatabase(tmp_path)
    write(
        db, "VDI", json.dumps({"7732-18-5": {"T": [300, 350, 400], "P": [1000, None, "invalid"]}})
    )
    table = db.saturation_table(
        "7732-18-5",
        columns=(
            QuantityColumn("T", "K", "condition", (300, 400), "Source temperature"),
            QuantityColumn("P", "Pa", "observation", (0, 1e6), "Source pressure"),
        ),
        field_bindings={"T": "T", "P": "P"},
        dataset_id="pressure-table",
        group_id="source",
        split="unused",
    )
    assert [r.values["P"].quality for r in table.rows] == ["valid", "missing", "invalid"]
    restored = ScientificTable.from_json(json.dumps(table.to_dict()))
    assert restored.to_dict() == table.to_dict()
