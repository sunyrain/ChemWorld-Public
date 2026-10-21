"""Conditions, missing inputs, units and original electro-optical consumers."""

import json
import math
from dataclasses import replace

import pytest

from chemworld.data.electro_optical import ElectroOpticalDatabase, RefractiveIndexObservation
from chemworld.foundation.units import Quantity
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.electro_optics import (
    brix_reading,
    brix_to_refractive_index,
    molar_refractivity,
    parallel_plate_capacitance,
    polarizability_volume,
    refractive_index_from_molar_refractivity,
    refractive_index_tde,
    refractive_index_to_brix,
    relative_permittivity_crc,
    water_permittivity_iapws,
    water_refractive_index_iapws,
)
from chemworld.physchem.property_equations import correlation_temperature_derivative
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.viscosity_kernels import viscosity_value


@pytest.fixture
def database(tmp_path):
    sources = {
        "Electrolytes/Permittivity (Dielectric Constant) of Liquids.tsv":
        "CAS\tChemical\tT\tPermittivity\tA\tB\tC\tD\tTmin\tTmax\n"
        "7732-18-5\tsynthetic\t300\t80\t110\t-.1\t\t\t280\t350\n"
        "64-17-5\tpoint\t293.15\t25\t\t\t\t\t\t\n"
        "71-36-3\tmissing interior\t300\t20\t100\t\t.0001\t\t280\t350\n",
        "Misc/CRC Handbook Organic RI.csv": "CAS\tRI\tRIT\n64-17-5\t1.3611\t293.15\n"
        "71-36-3\t1.4\t\n",
        "Misc/wikidata_properties.tsv": "CAS\tRI\tRIT\n64-17-5\t1.36\t\n7732-18-5\t\t\n",
        "Misc/Poling Dipole.csv": "CAS\tChemical\tdipole_moment\n64-17-5\tsynthetic\t1.44\n"
        "56-23-5\tzero dipole\t0\n",
    }
    for filename, text in sources.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    return ElectroOpticalDatabase(tmp_path)


def test_source_fit_card_derivative_and_actual_capacitance(database):
    component = ComponentSpec("water", "H2O", cas_number="7732-18-5")
    card = database.property_input("7732-18-5", component, method="CRC_PERMITTIVITY")
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored.to_dict() == card.to_dict()
    capacitances = []
    for t in (280, 350):
        epsilon = restored.package.evaluate(
            "relative_permittivity", temperature_K=t, validity_policy="raise"
        )
        c = parallel_plate_capacitance(epsilon, area_m2=0.01, separation_m=0.001)
        capacitances.append(c.quantity().to("uF").value)
        assert c.inputs["temperature"] == t and c.valid
    assert capacitances[0] > capacitances[1] > 0
    curve = restored.package.correlations[0]
    assert curve.metadata["omitted_high_order_terms"] == ["C", "D"]
    assert correlation_temperature_derivative(curve, temperature_K=300).to("1/K").value == -0.1
    warning = restored.package.evaluate("relative_permittivity", temperature_K=400)
    assert parallel_plate_capacitance(warning, area_m2=0.01, separation_m=0.001).warnings


def test_dielectric_source_point_and_fit_are_distinct(database):
    point = database.property_input(
        "64-17-5",
        ComponentSpec("ethanol", "C2H6O", cas_number="64-17-5"),
        method="CRC_PERMITTIVITY",
        source_point=True,
    )
    assert point.package.evaluate("relative_permittivity", temperature_K=293.15).value == 25
    with pytest.raises(ValueError, match="extrapolated"):
        point.package.evaluate("relative_permittivity", temperature_K=300, validity_policy="ignore")
    with pytest.raises(ValueError, match="observation"):
        correlation_temperature_derivative(point.package.correlations[0], temperature_K=293.15)
    with pytest.raises(ValueError, match="field A"):
        database.record("64-17-5", method="CRC_PERMITTIVITY").correlation()
    with pytest.raises(ValueError, match="interior"):
        database.record("71-36-3", method="CRC_PERMITTIVITY").correlation()


def test_ri_unknown_conditions_remain_unknown_after_json_and_brix(database):
    observation = database.refractive_index("64-17-5", method="WIKIDATA")
    restored = RefractiveIndexObservation.from_dict(json.loads(json.dumps(observation.to_dict())))
    assert restored.temperature.value is None and restored.temperature.quality == "missing"
    report = restored.evaluate()
    assert report.inputs == {} and not report.valid
    brix = brix_reading(report)
    assert brix.warnings[:1] == report.warnings and brix.unit == "degBx"
    assert brix.inputs == {} and brix.value > 0
    with pytest.raises(ValueError, match="unknown measurement"):
        restored.correlation()
    with pytest.raises(ValueError, match="missing"):
        database.refractive_index("7732-18-5", method="WIKIDATA").evaluate()
    # Query preserves both sources; no silent CRC-only or last-source overwrite.
    assert set(database.methods("64-17-5", "RI", sources=("CRC_RI", "WIKIDATA"))) == {
        "CRC_RI",
        "WIKIDATA",
    }


def test_known_ri_point_uses_original_input_card(database):
    card = database.property_input(
        "64-17-5",
        ComponentSpec("ethanol", "C2H6O", cas_number="64-17-5"),
        method="CRC_RI",
        source_point=True,
    )
    card = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    ri = card.package.evaluate("refractive_index", temperature_K=293.15)
    assert brix_reading(ri).inputs["temperature"] == 293.15
    assert card.package.correlations[0].metadata["spectral_condition"].startswith("sodium D")
    with pytest.raises(ValueError, match="identity"):
        database.property_input(
            "64-17-5",
            ComponentSpec("water", "H2O", cas_number="7732-18-5"),
            method="CRC_RI",
            source_point=True,
        )


def test_dipole_source_zero_units_and_existing_lucas_consumer(database):
    dipole = database.value("64-17-5", "dipole_moment", method="POLING").input_value()
    assert dipole.to("C*m").to("Debye").value == pytest.approx(1.44)
    zero = database.value("56-23-5", "dipole_moment", method="POLING").input_value()
    assert zero.quality == "valid" and zero.value == 0
    with_dipole = viscosity_value(
        "Lucas_gas", 400, Tc=450, Pc=5e6, Zc=0.25, MW=80, dipole=dipole.to("Debye").value
    )
    without = viscosity_value("Lucas_gas", 400, Tc=450, Pc=5e6, Zc=0.25, MW=80, dipole=zero.value)
    assert with_dipole > without > 0


def test_water_vacuum_limits_partial_derivatives_and_mathematical_domains():
    assert water_permittivity_iapws(300, 0) == 1
    assert water_refractive_index_iapws(300, 0) == 1
    assert derivative_values(lambda t: water_permittivity_iapws(t, 1000), 300)[1] < 0
    assert derivative_values(lambda rho: water_refractive_index_iapws(300, rho), 1000)[1] > 0
    assert all(
        math.isfinite(x)
        for x in derivative_values(lambda t: water_refractive_index_iapws(t, 1000), 300, order=3)
    )
    with pytest.raises(ValueError):
        water_permittivity_iapws(228, 1000)
    with pytest.raises(ValueError):
        water_permittivity_iapws(300, -1)
    with pytest.raises(ValueError, match="range"):
        water_refractive_index_iapws(800, 100)
    assert water_refractive_index_iapws(800, 100, extrapolate=True) > 1
    with pytest.raises(ValueError):
        relative_permittivity_crc(300, -1)


def test_water_dielectric_constant_conventions_are_explicit():
    assert water_permittivity_iapws(300, 1000) == pytest.approx(78.03317817364899)
    assert water_permittivity_iapws(300, 1000, constants="chemicals_1_5_2") == pytest.approx(
        78.03377085566082
    )
    with pytest.raises(ValueError, match="convention"):
        water_permittivity_iapws(300, 1000, constants="automatic")


def test_invalid_source_cells_do_not_become_missing_or_zero(database):
    observation = database.refractive_index("64-17-5", method="WIKIDATA")
    invalid = replace(observation, temperature=replace(observation.temperature, quality="invalid"))
    with pytest.raises(ValueError, match="invalid"):
        invalid.evaluate()
    record = database.record("7732-18-5", method="CRC_PERMITTIVITY")
    raw = list(record.raw_fields)
    raw[record.field_names.index("C")] = "nan"
    with pytest.raises(ValueError, match="invalid"):
        replace(record, raw_fields=tuple(raw)).correlation()
    with pytest.raises(ValueError, match="boolean"):
        record.correlation(source_point="false")


def test_tde_explicit_polynomial_order_wavelength_and_empty_inputs():
    assert refractive_index_tde(298.15, [0.001, 1.5], [0.0001, 0]) == 1.5
    assert refractive_index_tde(300, [1.5], [0.0001, 0], 600e-9) > 1.5
    with pytest.raises(ValueError):
        refractive_index_tde(300, [], [0])
    with pytest.raises(ValueError):
        refractive_index_tde(300, [1.5], [0], -1)


@pytest.mark.parametrize("index", [0.8, 1, 1.4, 2])
def test_lorentz_inverse_signed_response_and_polarizability_volume_units(index):
    rm = molar_refractivity(index, 1e-4)
    assert refractive_index_from_molar_refractivity(rm, 1e-4) == pytest.approx(index)
    alpha = polarizability_volume(index, 1e-4)
    assert Quantity(alpha, "m^3").to("L").value == pytest.approx(alpha * 1000)
    with pytest.raises(ValueError):
        refractive_index_from_molar_refractivity(1e-4, 1e-4)


def test_brix_scale_is_explicit_reversible_and_distinct_from_arbitrary_mass_fraction():
    assert refractive_index_to_brix(brix_to_refractive_index(5.8)) == pytest.approx(5.8)
    for x in (-1, 96):
        with pytest.raises(ValueError):
            brix_to_refractive_index(x)
        assert refractive_index_to_brix(
            brix_to_refractive_index(x, extrapolate=True), extrapolate=True
        ) == pytest.approx(x)
    with pytest.raises(ValueError):
        Quantity(10, "degBx").to("dimensionless")
