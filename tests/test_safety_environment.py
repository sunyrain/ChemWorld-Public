"""Source uncertainty, categories and real finite source consumers."""

import csv
import gzip
import json
from dataclasses import replace

import pytest

from chemworld.data.assessment_catalog import ASSESSMENT_SOURCES
from chemworld.data.assessments import AssessmentDatabase, AssessmentValue
from chemworld.data.constants import AmbiguousConstantIdentityError, UnknownConstantMethodError
from chemworld.foundation.units import Quantity
from chemworld.physchem.safety_environment import (
    crowl_louvar_flammability_limit,
    emission_impact,
    fire_mixture_limit,
    flammability_estimate,
    gas_exposure_comparison,
    iso_10156_lower_limit,
    mass_concentration_to_ppmv,
    nfpa_30_classification,
    octanol_water_partition,
    ppmv_to_mass_concentration,
    source_flammable_mixture,
)
from chemworld.physchem.specs import ComponentSpec, MixtureSpec


@pytest.fixture
def database(tmp_path):
    rows = {
        "IPCC_2021": (
            ["CAS", "20yr GWP", "100yr GWP", "500yr GWP", "50yr GTP", "100yr GTP"],
            [["74-82-8", 80, 30, 10, 12, 4], ["124-38-9", 1, 1, 1, 1, 1]],
        ),
        "IPCC_2014": (
            ["CAS", "20yr GWP", "100yr GWP", "20yr GTP", "50yr GTP", "100yr GTP"],
            [["74-82-8", 84, 28, 64, 13, 4]],
        ),
        "CRC_LOGP": (["CAS", "logP"], [["64-17-5", -1], ["71-36-3", 1]]),
        "ODP": (
            [
                "CAS",
                "ODP1",
                "ODP1 Min",
                "ODP1 Max",
                "ODP1 Design",
                "ODP2",
                "ODP2 Min",
                "ODP2 Max",
                "ODP2 Design",
            ],
            [["75-69-4", "0.5-1", 0.5, 1, 0.75, "0.5-1", 0.5, 1, 0.75]],
        ),
        "IEC_2010": (
            ["CAS", "T_flash", "T_autoignition", "LFL", "UFL"],
            [
                ["74-82-8", 200, 800, 0.05, 0.15],
                ["74-84-0", 190, 780, 0.03, 0.125],
                ["111-87-5", 354.15, 543.15, -0.009, 0.07],
            ],
        ),
        "IARC": (["CAS", "group", "year", "description"], [["71-43-2", 1, 2010, "Synthetic"]]),
        "ONTARIO_RAW": (
            [
                "CASRN",
                "Agent [CAS No.]",
                "Time-Weighted Average Limit (TWA)",
                "Short-Term Exposure Limit (STEL)",
                "Ceiling Limit (C)",
                "Notations",
                "MW",
            ],
            [["7440-02-0", "Synthetic agent", "0.1 mg/m3", "", "", "as metal", 58.7]],
        ),
    }
    ontario = {
        "74-82-8": dict(
            Name="Synthetic gas",
            **{
                "TWA (ppm)": 10,
                "TWA (mg/m^3)": 6,
                "STEL (ppm)": 20,
                "STEL (mg/m^3)": 12,
                "Ceiling (ppm)": None,
                "Ceiling (mg/m^3)": None,
                "Skin": False,
                "MW": 16.04,
            },
        ),
        "7440-02-0": dict(
            Name="Synthetic agent",
            **{
                "TWA (ppm)": None,
                "TWA (mg/m^3)": 0.1,
                "STEL (ppm)": None,
                "STEL (mg/m^3)": None,
                "Ceiling (ppm)": None,
                "Ceiling (mg/m^3)": None,
                "Skin": False,
                "MW": 58.7,
            },
        ),
    }
    ontario[" 7440-02-0"] = ontario["7440-02-0"]
    for source in ASSESSMENT_SOURCES:
        path = tmp_path / source.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.kind == "mapping-json":
            path.write_text(json.dumps(ontario), encoding="utf8")
        else:
            header, data = rows.get(source.source_id, (["CAS"], []))
            opener = gzip.open if source.kind == "tsv-gzip" else open
            with opener(path, "wt", encoding="utf8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(header)
                writer.writerows(data)
    return AssessmentDatabase(tmp_path)


def test_climate_versions_horizons_mass_consumer_and_missing_are_explicit(database):
    sources = {
        cas: database.assessment(cas, "GWP", method="IPCC (2021) 20yr")
        for cas in ("74-82-8", "124-38-9")
    }
    sources = {
        k: AssessmentValue.from_dict(json.loads(json.dumps(v.to_dict())))
        for k, v in sources.items()
    }
    report = emission_impact(
        {"74-82-8": Quantity(2, "kg"), "124-38-9": Quantity(1000, "g")}, sources
    )
    assert report.value == 161 and report.unit == "kg" and report.warnings
    assert sources["74-82-8"].conditions["horizon_years"] == 20
    sources["124-38-9"] = database.assessment("124-38-9", "GTP", method="IPCC (2021) 50yr")
    with pytest.raises(ValueError, match="horizon"):
        emission_impact({k: Quantity(1, "kg") for k in sources}, sources)
    missing = database.assessment("64-17-5", "GWP", method="IPCC (2021) 20yr")
    assert missing.quality == "not_listed" and missing.value is None
    with pytest.raises(ValueError, match="valid"):
        emission_impact({"64-17-5": Quantity(1, "kg")}, {"64-17-5": missing})


def test_odp_text_and_intervals_do_not_become_precise_factors(database):
    value = database.assessment("75-69-4", "ODP", method="ODP2 string")
    assert value.value == "0.5-1" and value.unit is None
    assert value.conditions["source_interval"]["minimum"] == "0.5"
    with pytest.raises(ValueError, match="scalar"):
        emission_impact({"75-69-4": Quantity(1, "kg")}, {"75-69-4": value})
    with pytest.raises(ValueError, match="numeric"):
        value.evaluate()


def test_source_logp_enters_original_finite_lle_with_conditions(database):
    sources = {
        cas: database.assessment(cas, "logP", method="CRC") for cas in ("64-17-5", "71-36-3")
    }
    result = octanol_water_partition(
        {"64-17-5": 2, "71-36-3": 3},
        sources,
        water_volume_L=1,
        octanol_volume_L=0.25,
        temperature_K=300,
    )
    assert result.organic_amounts_mol["64-17-5"] == pytest.approx(2 * 0.025 / 1.025)
    assert result.organic_amounts_mol["71-36-3"] == pytest.approx(3 * 2.5 / 3.5)
    assert result.material_balance_error_mol < 1e-12
    assert "source temperature unknown" in result.stability_diagnostic["application"]
    with pytest.raises(ValueError, match="exactly"):
        octanol_water_partition(
            {"64-17-5": 2}, sources, water_volume_L=1, octanol_volume_L=0.25, temperature_K=300
        )


def test_ontario_original_units_duplicate_keys_annotations_and_false(database):
    assert database.assessment("74-82-8", "Skin", method="Ontario Limits").value is False
    absent = database.assessment("64-17-5", "Skin", method="Ontario Limits")
    assert absent.value is None and absent.quality == "not_listed"
    ppm = database.assessment("74-82-8", "TWA", method="Ontario Limits")
    mass = database.assessment("74-82-8", "TWA", method="Ontario Limits", unit="mg/m^3")
    assert (ppm.value, ppm.unit, mass.value, mass.unit) == (10, "ppmv", 6, "mg/m^3")
    assert ppm.conditions["source_values"] == {"ppmv": 10, "mg/m^3": 6}
    assert "derived" in ppm.conditions["numeric_view"]
    with pytest.raises(AmbiguousConstantIdentityError):
        database.assessment("7440-02-0", "TWA", method="Ontario Limits")
    assert database.assessment_methods("7440-02-0", "TWA") == ("Ontario Limits",)
    records = database.records_for("7440-02-0", method="ONTARIO")
    assert [r.source_key for r in records] == ["7440-02-0", " 7440-02-0"]
    for r in records:
        (annotation,) = database.exposure_source_records(r.record_id)
        assert annotation.raw("Notations") == "as metal"
        assert database.assessment(r.record_id, "TWA", method="Ontario Limits").value == 0.1


def test_unlisted_carcinogen_is_distinct_from_negative_classification(database):
    listed = database.assessment(
        "71-43-2", "Carcinogen", method="International Agency for Research on Cancer"
    )
    assert listed.value == "Carcinogenic to humans (1)" and listed.conditions["year"] == "2010"
    unlisted = database.assessment(
        "74-82-8", "Carcinogen", method="International Agency for Research on Cancer"
    )
    assert unlisted.quality == "not_listed" and unlisted.value is None
    assert len(database.assessment_methods("74-82-8", "Carcinogen")) == 2
    with pytest.raises(UnknownConstantMethodError):
        database.assessment("74-82-8", "Carcinogen", method="automatic")


def test_original_mixture_consumes_source_lfl_and_retains_bad_source(database):
    components = (
        ComponentSpec("methane", "CH4", default_phase="gas"),
        ComponentSpec("ethane", "C2H6", default_phase="gas"),
    )
    mixture = MixtureSpec.from_mole_fractions(
        components, (0.2, 0.8), phase_label="gas", temperature_K=300, pressure_Pa=101325
    )
    identities = {"methane": "74-82-8", "ethane": "74-84-0"}
    sources = {
        k: database.assessment(v, "LFL", method="IEC 60079-20-1 (2010)")
        for k, v in identities.items()
    }
    report = source_flammable_mixture(mixture, sources, source_identities=identities, limit="LFL")
    assert report.value == pytest.approx(1 / (0.2 / 0.05 + 0.8 / 0.03)) and report.warnings
    bad = database.assessment("111-87-5", "LFL", method="IEC 60079-20-1 (2010)")
    assert bad.value == -0.009 and any("outside" in w for w in bad.warnings)
    with pytest.raises(ValueError):
        fire_mixture_limit([1], [bad.value])


def test_gas_exposure_requires_declared_window_and_original_units(database):
    methane = ComponentSpec("methane", "CH4", default_phase="gas")
    nitrogen = ComponentSpec("nitrogen", "N2", default_phase="gas")
    mixture = MixtureSpec.from_mole_fractions(
        (methane, nitrogen),
        (5e-6, 1 - 5e-6),
        phase_label="gas",
        temperature_K=300,
        pressure_Pa=101325,
    )
    value = database.assessment("74-82-8", "TWA", method="Ontario Limits")
    report = gas_exposure_comparison(
        mixture, "methane", value, source_identity="74-82-8", averaging_window="TWA"
    )
    assert report.value == 0.5 and report.inputs["concentration"] == 5
    with pytest.raises(ValueError, match="valid STEL"):
        gas_exposure_comparison(
            mixture, "methane", value, source_identity="74-82-8", averaging_window="STEL"
        )
    with pytest.raises(ValueError, match="identity"):
        gas_exposure_comparison(
            mixture, "methane", value, source_identity="71-43-2", averaging_window="TWA"
        )


def test_ppmv_is_distinct_from_nmr_and_mass_conversions_are_physical():
    assert Quantity(10, "ppmv").to("gas_volume_fraction").value == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        Quantity(10, "ppmv").to("ppm")
    assert Quantity(1, "mg/m^3").to("kg/m^3").value == 1e-6
    mass = ppmv_to_mass_concentration(100, 40, temperature_K=300, pressure_Pa=101325)
    assert mass_concentration_to_ppmv(
        mass, 40, temperature_K=300, pressure_Pa=101325
    ) == pytest.approx(100)
    for bad in (True, -1, float("nan"), 1e6 + 1):
        with pytest.raises(ValueError):
            ppmv_to_mass_concentration(bad, 40, temperature_K=300, pressure_Pa=101325)


def test_estimates_unknown_elements_missing_inputs_and_nfpa_conflict():
    with pytest.raises(ValueError, match="CHO"):
        crowl_louvar_flammability_limit({"C": 1, "Cl": 4}, limit="LFL")
    with pytest.raises(ValueError, match="requires"):
        flammability_estimate(limit="UFL", method="Suzuki (1994)")
    assert (
        flammability_estimate(
            limit="LFL", method="Crowl and Louvar (2001)", atoms={"C": 6, "H": 14}
        )
        > 0
    )
    with pytest.raises(ValueError, match="sum"):
        fire_mixture_limit([0.2, 0.2], [0.05, 0.03])
    with pytest.raises(ValueError, match="needs"):
        nfpa_30_classification(250)
    with pytest.raises(ValueError, match="disagree"):
        nfpa_30_classification(250, boiling_temperature_K=290, vapor_pressure_at_100F_Pa=90000)


def test_assessment_schema_rejects_unknown_or_fabricated_values(database):
    value = database.assessment("74-82-8", "GWP", method="IPCC (2021) 20yr")
    with pytest.raises(ValueError, match="schema"):
        AssessmentValue.from_dict({**value.to_dict(), "extra": 1})
    with pytest.raises(ValueError, match="unavailable"):
        replace(value, quality="missing")


def test_iso_explicit_diluent_identity_and_physical_domain(database):
    # Unknown zero-weight species has no physical contribution; positive ones cannot vanish.
    assert iso_10156_lower_limit([1, 0], [0.05, None], ["74-82-8", "unknown"]) == 0.05
    for fractions, limits, identities, error in (
        ([1], [None], ["7727-37-9"], "no combustible"),
        ([0.5, 0.5], [0.05, None], ["74-82-8", "unknown"], "missing LFL"),
        ([0.5, 0.5], [0.05, 0.01], ["74-82-8", "7727-37-9"], "diluent"),
        ([0.5, 0.5], [1, None], ["74-82-8", "7727-37-9"], "below one"),
        ([0.1, 0.9], [0.5, None], ["74-82-8", "7440-37-1"], "correction"),
        ([0.01, 0.99], [0.05, None], ["74-82-8", "2551-62-4"], "exceeds unity"),
        ([0.4, 0.4], [0.05, None], ["74-82-8", "7727-37-9"], "sum"),
        ([True], [0.05], ["74-82-8"], "finite"),
    ):
        with pytest.raises(ValueError, match=error):
            iso_10156_lower_limit(fractions, limits, identities)
    gas = MixtureSpec.from_mole_fractions(
        (
            ComponentSpec("methane", "CH4", default_phase="gas"),
            ComponentSpec("nitrogen", "N2", default_phase="gas"),
        ),
        (0.5, 0.5),
        phase_label="gas",
        temperature_K=300,
        pressure_Pa=101325,
    )
    sources = {
        "methane": database.assessment("74-82-8", "LFL", method="IEC 60079-20-1 (2010)"),
        "nitrogen": None,
    }
    identities = {"methane": "74-82-8", "nitrogen": "7727-37-9"}
    result = source_flammable_mixture(
        gas, sources, source_identities=identities, limit="LFL", method="ISO_10156_2017"
    )
    assert result.value == 0.1
    with pytest.raises(ValueError, match="only LFL"):
        source_flammable_mixture(
            gas, sources, source_identities=identities, limit="UFL", method="ISO_10156_2017"
        )
    with pytest.raises(ValueError, match="every combustible"):
        source_flammable_mixture(gas, sources, source_identities=identities, limit="LFL")


def test_explicit_flammability_selection_never_falls_back(database):
    assert database.flammability_methods("74-82-8", limit="LFL") == ("IEC 60079-20-1 (2010)",)
    with pytest.raises(ValueError, match="requires"):
        database.flammability_value("74-82-8", limit="LFL", method="Suzuki (1994)")
    with pytest.raises(ValueError, match="not_listed"):
        database.flammability_value(
            "64-17-5", limit="LFL", method="IEC 60079-20-1 (2010)", atoms={"C": 2, "H": 6, "O": 1}
        )
    result = database.flammability_value(
        "74-82-8", limit="LFL", method="Suzuki (1994)", heat_of_combustion_J_mol=-900000
    )
    assert result.inputs == {"combustion_enthalpy_J_mol": -900000}
    assert "estimate" in result.warnings[0]
