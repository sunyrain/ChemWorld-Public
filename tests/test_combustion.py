"""Combustion scientific domains, source semantics and original consumers."""

import csv
import json
from dataclasses import replace

import pytest

from chemworld.data.assessments import AssessmentValue
from chemworld.data.fuels import (
    FUEL_SOURCES,
    FuelDatabase,
    fuel_quality_comparison,
    source_ignition_quality,
    source_octane_index,
)
from chemworld.foundation.units import Quantity
from chemworld.physchem.combustion import (
    combustion_data,
    combustion_products_mixture,
    combustion_reference_heat,
    combustion_stoichiometry,
    heating_value_mass,
    ignition_delay_quality,
)
from chemworld.physchem.combustion_streams import air_fuel_ratio_solver, combustion_spec_solver
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec


def test_halogen_water_reactant_and_oxygen_release_balance():
    assert combustion_stoichiometry("CCl4") == {"CO2": 1, "H2O": -2, "HCl": 4, "O2": 0}
    assert combustion_stoichiometry("H2O2") == {"H2O": 1, "O2": 0.5}
    with pytest.raises(ValueError, match="neutral"):
        combustion_stoichiometry("NH4+")
    with pytest.raises(ValueError):
        combustion_stoichiometry({"C": True})


def test_unresolved_elements_and_ash_are_not_missing_mass():
    assert combustion_stoichiometry("NaOH")["Na"] == 1
    result = combustion_stoichiometry("NaOH", missing_handling="ash")
    assert result["Ash"] == 22.98976928 and result["H2O"] == 0.5 and result["O2"] == 0.25
    data = combustion_data(
        {"C": 1, "Na": 1}, method="Specification", HHV=-1000, missing_handling="ash"
    )
    assert data.atoms["Na"] == 1 and data.stoichiometry["Ash"] == 22.98976928
    with pytest.raises(ValueError, match="missing product"):
        combustion_data("CNa", method="Specification", HHV=-1000)


def test_calorific_conflict_phase_and_mass_estimate_domain():
    with pytest.raises(ValueError, match="nonempty"):
        combustion_data(stoichiometry={}, MW=16, method="Specification", HHV=-900000)
    explicit = combustion_data(
        stoichiometry={"O2": -2, "H2O": 2, "CO2": 1}, MW=16.043, method="Stoichiometry", Hf=-74520
    )
    assert explicit.HHV == -890604
    with pytest.raises(ValueError, match="inconsistent"):
        combustion_data("CH4", method="Specification", HHV=-900000, LHV=-900000)
    with pytest.raises(ValueError, match="disagree"):
        combustion_data("CH4", stoichiometry={"CO2": 1}, method="Stoichiometry", Hf=-70000)
    with pytest.raises(ValueError, match="10 wt"):
        heating_value_mass({"C": 0.8, "O": 0.2}, method="Dulong")
    with pytest.raises(ValueError, match="sum"):
        heating_value_mass({"C": 80, "H": 20}, method="Boie")
    data = combustion_data({"C": 1, "H": 4, "O": 0}, method="Dulong")
    assert data.HHV < 0 and data.LHV > data.HHV


def test_source_phase_reference_consumption_changes_actual_finite_heat():
    fuel = ComponentSpec("methane", "CH4", default_phase="gas")
    hf = CaloricReference(phase="gas", energy_J_mol=-74520, source_note="Synthetic fuel datum")
    refs = {
        "CO2": replace(hf, energy_J_mol=-393474, source_note="Synthetic carbon dioxide"),
        "O2": replace(hf, energy_J_mol=0, source_note="Explicit oxygen reference"),
        "H2O": replace(
            hf, phase="liquid", energy_J_mol=-285825, source_note="Synthetic liquid water"
        ),
    }
    hhv = combustion_reference_heat(fuel, hf, refs, Quantity(2, "mol"))
    refs["H2O"] = replace(refs["H2O"], phase="gas", energy_J_mol=-285825 + 44011.496)
    lhv = combustion_reference_heat(fuel, hf, refs, Quantity(2000, "mmol"))
    assert lhv.value - hhv.value == pytest.approx(4 * 44011.496)
    assert hhv.value == -1781208 and "Synthetic fuel datum" in hhv.warnings
    refs["H2O"] = replace(refs["H2O"], temperature_K=350)
    with pytest.raises(ValueError, match="temperature and pressure"):
        combustion_reference_heat(fuel, hf, refs, Quantity(2, "mol"))


def test_mixture_masks_identity_and_precomputed_balance_are_not_ignored():
    assert combustion_products_mixture(
        [{"C": 1, "H": 4}], [2], identities=["fuel"], reactivities=[False]
    ) == {"fuel": 2}
    with pytest.raises(ValueError, match="preserved identity"):
        combustion_products_mixture([{"C": 1}], [1], reactivities=[False])
    with pytest.raises(ValueError, match="atoms disagree"):
        combustion_products_mixture([{"C": 1}], [1], identities=["7732-18-5"])
    with pytest.raises(ValueError, match="atom balance"):
        combustion_products_mixture([{"C": 1}], [1], stoichiometries=[{"CO2": 2, "O2": -2}])


def inputs():
    return dict(  # noqa: C408 -- explicit scientific input names
        zs_air=[0.79, 0.21, 0, 0, 0],
        zs_fuel=[0, 0, 1, 0, 0],
        CASs=["7727-37-9", "7782-44-7", "74-82-8", "7732-18-5", "124-38-9"],
        atomss=[{"N": 2}, {"O": 2}, {"C": 1, "H": 4}, {"H": 2, "O": 1}, {"C": 1, "O": 2}],
    )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"n_air": 1, "n_fuel": 1}, "insufficient"),
        ({"n_fuel": 1}, "two independent"),
        ({"ratio": 10, "ratio_basis": "mole", "O2_excess": 0.3}, "extensive"),
        ({"n_air": 20, "n_fuel": 1, "n_out": 1}, "inconsistent"),
        ({"n_fuel": 1, "ratio": 20, "ratio_basis": "volume"}, "basis factors"),
        ({"n_fuel": 1, "frac_out_O2": 1}, "below one"),
    ],
)
def test_stream_specs_reject_undefined_or_inconsistent_problems(options, error):
    with pytest.raises(ValueError, match=error):
        combustion_spec_solver(**inputs(), **options)


def test_all_mole_ratio_output_scale_spec_is_solved_without_reference_fallback():
    result = combustion_spec_solver(**inputs(), n_out=21, ratio=20, ratio_basis="mole")
    assert result["n_air"] == 20 and result["n_fuel"] == 1
    assert result["ns_out"] == pytest.approx([15.8, 2.2, 0, 2, 1])
    with pytest.raises(ValueError, match="inconsistent"):
        air_fuel_ratio_solver(ratio=5, basis="mole", n_air=2, n_fuel=2)


def test_delay_domain_and_exact_piecewise_endpoints():
    with pytest.raises(ValueError, match=r"1\.512"):
        ignition_delay_quality(0.0015, quantity="DCN")
    assert ignition_delay_quality(0.0031, quantity="DCN") == pytest.approx(65.50080058327184)
    assert ignition_delay_quality(0.0065, quantity="DCN") == pytest.approx(32.720731430334524)
    assert ignition_delay_quality(0.002, quantity="RON") < 0


def test_source_missing_prediction_pairing_and_temperature_criterion(tmp_path):
    for source in FUEL_SOURCES:
        path = tmp_path / source.relative_path
        path.parent.mkdir(exist_ok=True, parents=True)
        columns = (
            ["IGNITION_DELAY"]
            if source.source_id == "DAHMEN_MARQUARDT"
            else ["RON", "MON", "CETANE", "POUR_POINT", "CLOUD_POINT"]
        )
        values = [".004"] if columns == ["IGNITION_DELAY"] else ["100", "90", "45", "-20", ""]
        with path.open("w", encoding="utf8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["CAS", *columns])
            w.writerow(["64-17-5", *values])
    db = FuelDatabase(tmp_path)
    assert db.loaded_sources == ()
    ron = db.fuel_value("64-17-5", "RON", method="COMBUSTDB_PREDICTIONS")
    ron = AssessmentValue.from_dict(json.loads(json.dumps(ron.to_dict())))
    mon = db.fuel_value("64-17-5", "MON", method="COMBUSTDB_PREDICTIONS")
    assert source_octane_index(ron, mon).value == 95 and ron.conditions["data_kind"] == "prediction"
    with pytest.raises(ValueError, match="same identity"):
        source_octane_index(ron, replace(mon, method="COMBUSTDB"))
    pour = db.fuel_value("64-17-5", "POUR_POINT", method="COMBUSTDB_PREDICTIONS")
    assert fuel_quality_comparison(pour, Quantity(273.15, "K"), relation="maximum")[
        "meets_criterion"
    ]
    missing = db.fuel_value("64-17-5", "CLOUD_POINT", method="COMBUSTDB_PREDICTIONS")
    assert missing.quality == "missing" and missing.value is None
    absent = db.fuel_value("74-82-8", "RON", method="COMBUSTDB")
    assert absent.quality == "not_listed"
    with pytest.raises(ValueError, match="valid numeric"):
        fuel_quality_comparison(absent, Quantity(50, "dimensionless"), relation="minimum")
    delay = db.fuel_value("64-17-5", "ignition_delay", method="DAHMEN_MARQUARDT")
    assert source_ignition_quality(delay, quantity="DCN").value == pytest.approx(51.11)
