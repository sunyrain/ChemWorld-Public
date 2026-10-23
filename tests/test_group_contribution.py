"""Predicted groups enter original critical, caloric and transport inputs."""

import json

import pytest

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.critical_properties import eos_component_from_critical_inputs
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.eos import CubicEOSSpec, evaluate_cubic_eos
from chemworld.physchem.group_contribution import (
    JOBACK_FIELDS,
    GroupContributionTable,
    GroupEstimateReport,
    bondi_group_properties,
    fedors_critical_volume,
    joback_estimate,
    joback_property_correlations,
    wilson_jasperson_critical,
)
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec

SOURCE = DataSourceProvenance("published-groups", 0, "Joback public increments", version="1")


def joback_table():
    return GroupContributionTable(
        "Joback",
        {
            "1": dict(
                zip(
                    JOBACK_FIELDS,
                    (
                        0.0141,
                        -0.0012,
                        65.0,
                        23.58,
                        -5.1,
                        -76.45,
                        -43.96,
                        19.5,
                        -0.00808,
                        0.000153,
                        -9.67e-8,
                        0.908,
                        2.373,
                        548.29,
                        -1.719,
                    ),
                    strict=True,
                )
            ),
            "24": dict(
                zip(
                    JOBACK_FIELDS,
                    (
                        0.038,
                        0.0031,
                        62.0,
                        76.75,
                        61.2,
                        -133.22,
                        -120.5,
                        6.45,
                        0.067,
                        -3.57e-5,
                        2.86e-9,
                        4.189,
                        8.972,
                        340.35,
                        -0.35,
                    ),
                    strict=True,
                )
            ),
        },
        SOURCE,
    )


def test_joback_full_public_example_native_curves_and_critical_consumer():
    table = GroupContributionTable.from_dict(json.loads(json.dumps(joback_table().to_dict())))
    report = joback_estimate(
        "acetone", {"1": 2, "24": 1}, table, atom_count=10, molecular_weight_g_mol=58.041864812
    )
    assert report.values["Tb"] == pytest.approx(322.11)
    assert report.values["Tc"] == pytest.approx(500.5590049525365)
    assert report.values["Pc"] == pytest.approx(4802499.604994407)
    assert report.values["Hf"] == pytest.approx(-217830)
    assert report.values["Gf"] == pytest.approx(-154540)
    assert report.values["Hfus"] == pytest.approx(5125)
    restored = GroupEstimateReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.to_dict() == report.to_dict()
    with pytest.raises(TypeError):
        restored.inputs["counts"]["1"] = 9
    cp, mu = joback_property_correlations(report, temperature_range_K=(298, 500))
    assert evaluate_correlation(cp, temperature_K=350).value == pytest.approx(84.6910975)
    assert evaluate_correlation(mu, temperature_K=300).value == pytest.approx(0.0002940378347162687)
    component = ComponentSpec("acetone", "C3H6O")
    eos = eos_component_from_critical_inputs(
        component,
        Tc=report.critical_estimate("Tc"),
        Pc=report.critical_estimate("Pc"),
        omega=InputValue(0.3, "dimensionless", None, "authored omega"),
    )
    assert eos.critical_temperature_K == report.values["Tc"]
    gas = evaluate_cubic_eos(
        CubicEOSSpec("predicted-acetone", "peng_robinson", (eos,)),
        {"acetone": 1},
        temperature_K=400,
        pressure_Pa=1e5,
    )
    assert 0 < gas.compressibility_factor < 1
    state = evaluate_property_state(
        (ComponentPropertyPackage(component, (cp, mu)),),
        {"acetone": 1},
        temperature_K=350,
        pressure_Pa=101325,
        phase="gas",
        quantities=("cp", "enthalpy"),
        references={"acetone": CaloricReference(phase="gas", temperature_K=300)},
    )
    assert state.properties["cp"].value == pytest.approx(84.6910975)
    expected_delta = sum(
        value * (350 ** (power + 1) - 300 ** (power + 1)) / (power + 1)
        for power, value in enumerate(report.values["Cpig_coefficients"])
    )
    assert state.properties["enthalpy"].value == pytest.approx(expected_delta)


def test_joback_missing_coefficients_are_not_zero_filled():
    payload = joback_table().to_dict()
    payload["coefficients"]["24"]["mua"] = None
    table = GroupContributionTable.from_dict(payload)
    report = joback_estimate(
        "acetone", {"1": 2, "24": 1}, table, atom_count=10, molecular_weight_g_mol=58.04
    )
    assert report.values["viscosity_coefficients"] is None
    assert len(joback_property_correlations(report, temperature_range_K=(298, 500))) == 1
    with pytest.raises(ValueError, match="unavailable"):
        report.critical_estimate("unavailable")


def test_fedors_replaces_alcohol_atoms_and_rejects_unknown_structure():
    table = GroupContributionTable(
        "Fedors",
        {
            k: {"Vc": v}
            for k, v in {"C": 34.426, "H": 9.172, "O": 20.291, "O_alcohol": 18.0}.items()
        },
        SOURCE,
    )
    assert fedors_critical_volume(
        {"C": 4, "H": 10, "O": 1}, table, alcohol_oxygens=1
    ) == pytest.approx(0.000274024)
    with pytest.raises(ValueError, match="exceed"):
        fedors_critical_volume({"C": 4, "H": 10, "O": 1}, table, alcohol_oxygens=2)
    with pytest.raises(ValueError, match="ring sizes"):
        fedors_critical_volume({"C": 4, "H": 10, "O": 1}, table, rings_by_size={7: 1})


def test_wilson_jasperson_both_orders_and_missing_coefficients():
    table = GroupContributionTable(
        "Wilson-Jasperson",
        {
            "C": {"Tc": 0.008532, "Pc": 0.72983},
            "H": {"Tc": 0.002793, "Pc": 0.1266},
            "O": {"Tc": 0.020341, "Pc": 0.4336},
            "group:OH_large": {"Tc": 0.01, "Pc": 0},
            "V": {"Tc": 0.0186, "Pc": None},
        },
        SOURCE,
    )
    for second, tc, pc in ((False, 702.883, 3794106.49), (True, 693.567, 3743819.6667)):
        result = wilson_jasperson_critical(
            {"C": 8, "H": 10, "O": 1},
            table,
            boiling_temperature_K=477.67,
            ring_count=1,
            functional_groups={"OH_large": 1},
            second_order=second,
        )
        assert result["Tc"] == pytest.approx(tc, abs=0.001)
        assert result["Pc"] == pytest.approx(pc, abs=0.02)
    unavailable = wilson_jasperson_critical({"V": 1}, table, boiling_temperature_K=400)
    assert unavailable["Tc"] is not None and unavailable["Pc"] is None


def test_bondi_missing_area_does_not_become_known_zero():
    table = GroupContributionTable(
        "Bondi",
        {
            "CH3": {"Vw": 13.67, "Aw": 2.12},
            "CH2": {"Vw": 10.23, "Aw": 1.35},
            "S": {"Vw": 10.8, "Aw": None},
        },
        SOURCE,
    )
    result = bondi_group_properties({"CH3": 2, "CH2": 8}, table)
    assert result["V_vdw"] == pytest.approx(0.00010918)
    assert result["A_vdw"] == pytest.approx(1504000.0)
    assert result["R"] == pytest.approx(109.18 / 15.17)
    unavailable = bondi_group_properties({"CH3": 2, "S": 1}, table)
    assert unavailable["V_vdw"] > 0 and unavailable["A_vdw"] is None and unavailable["Q"] is None


@pytest.mark.parametrize("counts", [{"1": 0}, {"1": True}, {"1": 1.5}, {"missing": 1}])
def test_invalid_group_inputs_rejected(counts):
    with pytest.raises(ValueError):
        joback_estimate("bad", counts, joback_table(), atom_count=10, molecular_weight_g_mol=58.0)
