import json

import pytest

from chemworld.data.constants import AmbiguousConstantIdentityError
from chemworld.data.volume import VolumeDatabase
from chemworld.data.volume_curves import source_volumes
from chemworld.physchem import ComponentSpec, evaluate_correlation
from chemworld.physchem.property_inputs import PropertyInputCard


def table(tmp_path, source, text):
    db = VolumeDatabase(tmp_path)
    path = tmp_path / db.source(source).relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return db


def test_constant_is_an_unlocated_observation_with_explicit_assumption(tmp_path):
    db = table(tmp_path, "CRC_SOLID_VOLUME", "CAS\tChemical\tVm\n7732-18-5\twater\t0.00002\n")
    with pytest.raises(ValueError, match="application temperature"):
        db.volume("7732-18-5", method="CRC_SOLID_VOLUME")
    component = ComponentSpec(
        "water", "H2O", molecular_weight_g_mol=18.01528, cas_number="7732-18-5"
    )
    card = db.property_input(
        "7732-18-5",
        component,
        method="CRC_SOLID_VOLUME",
        application_temperature_K=300,
        application_note="Synthetic application assumption",
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    result = restored.evaluate_state(
        temperature_K=300, pressure_Pa=101325, phase="solid", quantities=("density", "molar_volume")
    )
    assert result.properties["density"].value == pytest.approx(900.764)
    assert result.properties["molar_volume"].value == pytest.approx(2e-5)
    curve = restored.package.correlations[0]
    assert curve.metadata["temperature_range_basis"] == "caller_point_assumption"
    with pytest.raises(ValueError, match="extrapolated"):
        evaluate_correlation(curve, temperature_K=301, validity_policy="ignore")


def test_explicit_parameters_identity_and_no_source_override(tmp_path):
    db = table(
        tmp_path,
        "COSTALD_PARAMETERS",
        "CAS\tChemical\tomega_SRK\tVchar\tZ_RA\n64-17-5\tethanol\t0.2\t0.0002\t0.27\n",
    )
    candidates = db.volumes("64-17-5", method="COSTALD_PARAMETERS")
    assert len(candidates) == 2
    with pytest.raises(AmbiguousConstantIdentityError, match="curve_id"):
        db.volume("64-17-5", method="COSTALD_PARAMETERS")
    with pytest.raises(ValueError, match="exactly"):
        candidates[0].build(parameters={"Tc": 600, "omega": 0.3}, temperature_bounds_K=(250, 400))
    c = candidates[0].build(parameters={"Tc": 600}, temperature_bounds_K=(250, 400))
    assert evaluate_correlation(c, temperature_K=300).value > 0
    with pytest.raises(ValueError, match="identity"):
        db.property_input(
            "64-17-5",
            ComponentSpec("water", "H2O", cas_number="7732-18-5"),
            method="COSTALD_PARAMETERS",
            curve_id=candidates[0].curve_id,
            parameters={"Tc": 600},
            temperature_bounds_K=(250, 400),
        )


def test_perry_volume_enters_original_density_state_consumer(tmp_path):
    db = table(
        tmp_path,
        "PERRY_VOLUME",
        "CAS\tChemical\tC1\tC2\tC3\tC4\tTmin\tTmax\n"
        "64-17-5\tethanol\t2000\t0.25\t600\t0.3\t200\t500\n",
    )
    card = db.property_input(
        "64-17-5",
        ComponentSpec("ethanol", "C2H6O", molecular_weight_g_mol=46, cas_number="64-17-5"),
        method="PERRY_VOLUME",
    )
    report = card.evaluate_state(
        temperature_K=300,
        pressure_Pa=101325,
        phase="liquid",
        quantities=("density", "molar_volume"),
    )
    expected = 0.25 ** (1 + 0.5**0.3) / 2000
    assert report.properties["molar_volume"].value == pytest.approx(expected)
    assert report.properties["density"].value == pytest.approx(0.046 / expected)


def test_missing_coefficients_and_unknown_temperature_not_guessed(tmp_path):
    db = table(tmp_path, "MCHAWEH_SN0", "CAS\tChemical\tdelta_SRK\n64-17-5\tethanol\t\n")
    assert db.record("64-17-5", method="MCHAWEH_SN0").value("delta_SRK").quality == "missing"
    with pytest.raises(ValueError, match="missing"):
        db.volume(
            "64-17-5",
            method="MCHAWEH_SN0",
            parameters={"Tc": 600, "Vc": 0.0002, "omega": 0.2},
            temperature_bounds_K=(250, 400),
        )


def test_chemsep_liquid_solid_virial_unit_mapping(tmp_path):
    path = tmp_path / "Misc/ChemSep8.32.xml"
    path.parent.mkdir()
    path.write_text(
        """<compounds><compound><CompoundID value="water"/>
    <CAS value="7732-18-5"/><CriticalTemperature units="K" value="600"/>
    <SolidDensity units="kmol/m3"><eqno value="2"/><A value="50"/><B value="-0.01"/>
    <Tmin units="K" value="0"/><Tmax units="K" value="500"/></SolidDensity>
    <SecondVirialCoefficient units="m3/kmol"><eqno value="104"/>
    <A value="1"/><B value="-600"/><C value="0"/><D value="0"/><E value="0"/>
    <Tmin units="K" value="200"/><Tmax units="K" value="500"/></SecondVirialCoefficient>
    </compound></compounds>""",
        encoding="utf-8",
    )
    db = VolumeDatabase(tmp_path)
    candidates = source_volumes(db.records("CHEMSEP_XML")[0])
    assert len(candidates) == 2
    with pytest.raises(ValueError, match="explicit"):
        candidates[0].build()
    c = candidates[0].build(temperature_bounds_K=(200, 400))
    assert evaluate_correlation(c, temperature_K=300).value == pytest.approx(1 / 47000)
    b = candidates[1].build()
    assert evaluate_correlation(b, temperature_K=300).value == pytest.approx(-0.001)


def test_tabulated_volume_keeps_missing_intervals_and_explicit_interpolation(tmp_path):
    path = tmp_path / "Misc/VDI Saturation Compounds Data.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "64-17-5": {
                    "Name": "ethanol",
                    "T": [200, 300, 400],
                    "Volume (l)": [0.0001, 0.0002, None],
                    "Volume (g)": [0.02, 0.03, 0.04],
                    "Density (l)": [460, 230, None],
                    "Density (g)": [2.3, 1.533333333, 1.15],
                }
            }
        )
    )
    db = VolumeDatabase(tmp_path)
    candidates = db.volumes("64-17-5", method="VDI")
    assert len(candidates) == 4
    with pytest.raises(ValueError, match="missing"):
        candidates[0].build_segments(interpolation="linear")
    c = candidates[0].build(interpolation="linear", temperature_bounds_K=(200, 300))
    assert evaluate_correlation(c, temperature_K=250).value == pytest.approx(0.00015)
    with pytest.raises(ValueError, match="exceeds"):
        candidates[2].build_segments(interpolation="linear", temperature_bounds_K=(150, 450))
