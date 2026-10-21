"""Scientific boundary, source and native consumer contracts for F1."""

import json
import math
from dataclasses import replace

import pytest

from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.henry import (
    HENRY_SCALES,
    henry_convert,
    henry_log_mixture,
    henry_matrix,
    henry_pressure_derivatives,
    henry_pressure_mixture,
    henry_raoult_flash,
)
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.solubility import (
    HansenParameters,
    freezing_point_depression,
    hansen_distance,
    hildebrand_parameter,
    solid_solubility,
)
from chemworld.physchem.specs import ComponentSpec, MixtureSpec, PropertyCorrelation


@pytest.mark.parametrize("scale", HENRY_SCALES)
def test_identity_conversion_does_not_need_cancelled_solvent_data(scale):
    assert henry_convert(1.2, scale, scale) == pytest.approx(1.2)


def test_basis_cancellation_and_explicit_reference_temperature():
    assert henry_convert(101325, "SI", "KHpx") == 1
    assert henry_convert(0.2, "Hcc", "KHcc") == 5
    assert henry_convert(2, "Hbp", "mol/(kg*atm)") == 2
    assert henry_convert(2, "mol/(kg*Pa)", "Hbp") == 202650
    assert (
        henry_convert(
            1,
            "mol/(kg*Pa)",
            "Hcp",
            solvent_molar_density_mol_m3=50000,
            solvent_molecular_weight_g_mol=20,
        )
        == 1000
    )
    assert henry_convert(1, "Hcc", "Hcp", gas_reference_temperature_K=400) == pytest.approx(
        0.5 * henry_convert(1, "Hcc", "Hcp", gas_reference_temperature_K=200)
    )
    assert (
        henry_convert(
            1, "alpha", "Hcc", gas_reference_temperature_K=300, bunsen_reference_temperature_K=300
        )
        == 1
    )
    with pytest.raises(ValueError, match="molar density"):
        henry_convert(1, "SI", "Hcp")
    with pytest.raises(ValueError, match="molecular weight"):
        henry_convert(1, "Hbp", "SI")


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True])
def test_invalid_henry_constant_rejects(value):
    with pytest.raises(ValueError):
        henry_convert(value, "Hcp", "SI")


def test_mixture_zero_is_real_and_missing_policy_retains_coverage():
    reports = henry_log_mixture(
        [0, math.log(100)], [1, 1], dlog_dT=[0, 0.01], d2log_dT2=[0, -0.0001], temperature_K=300
    )
    assert [r.value for r in reports] == pytest.approx([10, 0.05, -0.00025])
    assert reports[0].inputs["temperature"] == 300
    assert reports[2].unit == "Pa/K^2"
    assert reports[0].valid
    with pytest.raises(ValueError, match="missing Henry"):
        henry_log_mixture([0, None], [1, 1])
    ignored = henry_log_mixture([math.log(100), None], [1, 1], missing="ignore")[0]
    legacy = henry_log_mixture([math.log(100), None], [1, 1], missing="zero_log")[0]
    assert ignored.value == pytest.approx(100) and legacy.value == pytest.approx(10)
    assert ignored.inputs["known_solvent_fraction"] == 0.5 and ignored.warnings
    assert henry_log_mixture([0, None], [1, 0])[0].valid
    assert henry_pressure_mixture([10, 1000], weights=[2, 2]).value == pytest.approx(100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_pressures": [None], "weights": [1], "missing": "ignore"},
        {"log_pressures": [None], "weights": [1], "missing": "zero_log"},
        {"log_pressures": [0], "weights": [0]},
        {"log_pressures": [0], "weights": [1], "dlog_dT": [None]},
        {"log_pressures": [None, 0], "weights": [1, 1], "dlog_dT": [0, 0], "missing": "ignore"},
        {"log_pressures": [0], "weights": [1], "d2log_dT2": [0]},
        {"log_pressures": [math.inf], "weights": [1]},
        {"log_pressures": [-1000], "weights": [1]},
    ],
)
def test_unknown_or_impossible_mixture_does_not_invent_a_henry_value(kwargs):
    with pytest.raises(ValueError):
        henry_log_mixture(**kwargs)


def test_matrix_roles_and_explicit_legacy_adapter():
    matrix = [[None, None], [0, None]]
    assert henry_matrix(matrix, [0.8, 0.2], [False, True])[1][0].value == 1
    with pytest.raises(ValueError, match="missing Henry"):
        henry_matrix(matrix, [0.8, 0.2], [False, True], zero_is_missing=True)
    # An actual missing pair cannot silently discard a supplied derivative.
    with pytest.raises(ValueError, match="cannot have"):
        henry_matrix(
            [[None, None], [None, None]],
            [0.8, 0.2],
            [False, True],
            dlog_dT=[[None, None], [0, None]],
            missing="ignore",
        )
    with pytest.raises(ValueError, match="second Henry"):
        henry_matrix([[0]], [1], [False], d2log_dT2=[[0]])
    with pytest.raises(ValueError, match="policy"):
        henry_matrix([[0]], [1], [False], missing="guess")
    with pytest.raises(ValueError):
        henry_matrix([[0]], [1], [True])


def test_henry_correlations_have_solvent_semantics_and_common_third_derivative():
    curve = PropertyCorrelation(
        "gas-in-water",
        "henry_pressure",
        {"A": 20, "B": -1200},
        {"temperature": "K"},
        "Pa",
        "henry_pressure",
        {"temperature": (250, 400)},
        "Synthetic gas in water",
        {"solvent_id": "water"},
    )
    restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
    values = henry_pressure_derivatives(300, 20, -1200, order=3)
    assert evaluate_correlation(restored, temperature_K=300).value == values[0]
    assert all(math.isfinite(v) for v in values)
    with pytest.raises(ValueError):
        replace(curve, property_id="vapor_pressure")
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=450, validity_policy="raise")
    with pytest.raises(ValueError):
        henry_pressure_derivatives(300, -1000)


def test_original_flash_consumes_same_temperature_reports_and_preserves_feed():
    mixture = MixtureSpec.from_mole_fractions(
        (ComponentSpec("water", "H2O"), ComponentSpec("gas", "CO2")),
        (0.7, 0.3),
        phase_label="liquid",
        temperature_K=300,
        pressure_Pa=50000,
    )
    h = henry_pressure_mixture([200000], weights=[1], temperature_K=300)
    psat = PropertyEvaluation(
        "vapor_pressure", "synthetic", "constant", 20000, "Pa", {"temperature": 300}
    )
    result = henry_raoult_flash(
        mixture, henry_evaluations={"gas": h.to("bar")}, solvent_vapor_pressures={"water": psat}
    )
    assert 0 < result.vapor_fraction < 1
    for identity, z in zip(mixture.component_ids, mixture.mole_fractions, strict=True):
        assert (1 - result.vapor_fraction) * result.liquid_composition[identity] + (
            result.vapor_fraction * result.vapor_composition[identity]
        ) == pytest.approx(z, abs=1e-9)
    for bad in (
        replace(h, inputs={"temperature": 301}),
        replace(h, property_id="vapor_pressure"),
        replace(h, warnings=("missing solvent",)),
    ):
        with pytest.raises(ValueError, match="same-temperature"):
            henry_raoult_flash(
                mixture, henry_evaluations={"gas": bad}, solvent_vapor_pressures={"water": psat}
            )
    with pytest.raises(ValueError, match="partition"):
        henry_raoult_flash(
            mixture, henry_evaluations={"gas": h}, solvent_vapor_pressures={"gas": psat}
        )


def test_solid_and_hildebrand_domains_and_freezing_bases():
    assert solid_solubility(350, 350, 12000) == 1
    assert derivative_values(lambda t: solid_solubility(t, 350, 12000), 300)[1] > 0
    for t, gamma in ((351, 1), (350, 0.5)):
        with pytest.raises(ValueError):
            solid_solubility(t, 350, 12000, gamma=gamma)
    for h in (-1, 100):
        with pytest.raises(ValueError):
            hildebrand_parameter(
                300, vaporization_enthalpy_J_mol=h, liquid_molar_volume_m3_mol=0.001
            )
    assert freezing_point_depression(350, 10000, mole_fraction=0.01) == pytest.approx(
        freezing_point_depression(
            350, 10000, molality_mol_kg=0.5, solvent_molecular_weight_g_mol=20
        )
    )
    with pytest.raises(ValueError):
        freezing_point_depression(350, 10000, mole_fraction=0.01, molality_mol_kg=0.5)
    with pytest.raises(ValueError):
        freezing_point_depression(350, 10000, molality_mol_kg=0.5)


def test_hansen_actual_source_vector_and_missing_values(tmp_path):
    path = tmp_path / "Misc/hspipy_hansen_solubility_parameters.tsv"
    path.parent.mkdir()
    path.write_text(
        "CAS\tHANSEN_DELTA_D\tHANSEN_DELTA_P\tHANSEN_DELTA_H\n"
        "64-17-5\t15000\t8000\t19000\n71-36-3\t16000\t6000\t16000\n"
        "7732-18-5\t16000\t\t42000\n",
        encoding="utf8",
    )
    database = ConstantDatabase(tmp_path)
    a, b, missing = [HansenParameters.from_record(r) for r in database.records("HSPIPY_HANSEN")]
    a = HansenParameters.from_dict(json.loads(json.dumps(a.to_dict())))
    assert a.source == database.records("HSPIPY_HANSEN")[0].source.provenance
    assert a.dispersion.uncertainty is None and a.dispersion.uncertainty_note
    distance = hansen_distance(a, b)
    assert distance.value == pytest.approx(math.sqrt(17e6))
    assert distance.to("MPa^0.5").value == pytest.approx(math.sqrt(17))
    radius = InputValue(5, "MPa^0.5", None, "Explicit synthetic radius")
    assert hansen_distance(a, b, interaction_radius=radius).value < 1
    assert missing.polar.quality == "missing" and missing.polar.value is None
    with pytest.raises(ValueError, match="all three"):
        hansen_distance(a, missing)
    with pytest.raises(ValueError):
        hansen_distance(a, b, interaction_radius=replace(radius, value=0))
    with pytest.raises(ValueError):
        HansenParameters.from_dict({**a.to_dict(), "unknown": 0})
