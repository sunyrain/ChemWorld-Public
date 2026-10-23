"""Source-aware regular solutions use the existing activity and calibration path."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.activity_thermodynamics import (
    activity_parameter_jacobian,
    activity_thermodynamic_report,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.equilibrium import ActivityModelSpec, activity_coefficients
from chemworld.physchem.regular_solution import hansen_activity_model, regular_solution_model
from chemworld.physchem.solubility import HansenParameters


def vectors():
    source = DataSourceProvenance(
        "synthetic-Hansen", 0, "Authored source-bearing vectors", version="1", license="MIT"
    )
    return tuple(
        HansenParameters(
            key,
            *(InputValue(v, "MPa^0.5", None, "not reported") for v in row),
            source,
            "record-" + key,
        )
        for key, row in (("A", (16.6, 12.3, 5.5)), ("B", (14.5, 0.0, 0.0)))
    )


def hansen():
    return hansen_activity_model(
        "hansen-test",
        vectors(),
        alpha=0.6,
        molar_volumes={
            key: InputValue(value / 1000, "L/mol", None, "authored volume")
            for key, value in (("A", 89.0), ("B", 115.2))
        },
        volume_reference="explicit constant synthetic reference volumes",
    )


def test_hansen_real_source_shape_unit_conversion_json_and_original_gamma_consumer():
    model = hansen()
    report = activity_thermodynamic_report(model, {"A": 0.97, "B": 0.03}, temperature_K=298.15)
    assert report.activity_coefficients["B"] == pytest.approx(3.489578619082938, rel=2e-9)
    assert model.parameters["delta_d:A"] == 16600
    assert model.parameters["V:A"] == pytest.approx(89e-6)
    restored = ActivityModelSpec(**json.loads(json.dumps(model.to_dict())))
    assert restored.provenance == model.provenance
    assert restored.provenance["hansen_sources"][0] == vectors()[0].to_dict()
    assert activity_coefficients(
        restored, {"A": 0.97, "B": 0.03}, temperature_K=298.15
    ) == pytest.approx(report.activity_coefficients)
    assert report.parameter_contract["provenance"]["hansen_sources"][0]["record_id"] == "record-A"


@pytest.mark.parametrize("model_name", ("regular_solution", "flory_huggins"))
def test_regular_thermal_contract_and_directional_identifiability(model_name):
    model = regular_solution_model(
        "regular-test",
        ("A", "B"),
        model=model_name,
        molar_volumes_m3_mol={"A": 89e-6, "B": 109e-6},
        solubility_parameters_Pa_half={"A": 18000.0, "B": 16000.0},
        interaction_parameters={("A", "B"): 0.02, ("B", "A"): 0.04},
    )
    r = activity_thermodynamic_report(model, {"A": 0.3, "B": 0.7}, temperature_K=320)
    assert r.heat_capacity_excess_J_mol_K == pytest.approx(0, abs=1e-12)
    if model_name == "regular_solution":
        assert r.enthalpy_excess_J_mol == pytest.approx(r.gibbs_excess_J_mol)
        assert r.entropy_excess_J_mol_K == pytest.approx(0, abs=1e-12)
    else:
        vbar = 0.3 * 89e-6 + 0.7 * 109e-6
        assert r.entropy_excess_J_mol_K == pytest.approx(
            -8.31446261815324 * (0.3 * log(89e-6 / vbar) + 0.7 * log(109e-6 / vbar))
        )
    jac = activity_parameter_jacobian(
        model,
        {"A": 0.3, "B": 0.7},
        temperature_K=320,
        parameters=("lambda:A|B", "lambda:B|A"),
        quantities=("GE", "ln_gamma:A", "HE", "CpE"),
    )
    for row in jac.values():
        assert row["lambda:A|B"] == pytest.approx(row["lambda:B|A"], abs=1e-12)
    reversed_interactions = replace(
        model, parameters={**model.parameters, "lambda:A|B": -0.1, "lambda:B|A": 0.16}
    )
    assert activity_coefficients(
        reversed_interactions, {"A": 0.3, "B": 0.7}, temperature_K=320
    ) == pytest.approx(r.activity_coefficients)


def test_builder_zero_correction_is_explicit_but_low_level_missing_pairs_fail():
    model = regular_solution_model(
        "unadjusted",
        ("A", "B"),
        molar_volumes_m3_mol={"A": 1e-4, "B": 1e-4},
        solubility_parameters_Pa_half={"A": 18000, "B": 16000},
    )
    assert model.provenance["interaction_mode"] == "explicit_zero_corrections"
    assert model.parameters["lambda:A|B"] == model.parameters["lambda:B|A"] == 0
    assert activity_coefficients(model, {"A": 0.5, "B": 0.5}, temperature_K=300)[
        "A"
    ] == pytest.approx(exp(100 / (8.31446261815324 * 300)))
    p = dict(model.parameters)
    del p["lambda:A|B"]
    with pytest.raises(ValueError, match="missing"):
        replace(model, parameters=p)


def test_invalid_or_missing_source_values_and_volumes_do_not_become_zero():
    original = vectors()
    invalid = replace(
        original[0], polar=InputValue(None, "Pa^0.5", None, "source missing", "missing")
    )
    with pytest.raises(ValueError, match="all three"):
        hansen_activity_model(
            "missing",
            (invalid, original[1]),
            alpha=0.6,
            molar_volumes=dict.fromkeys("AB", InputValue(1e-4, "m^3/mol", None, "explicit")),
            volume_reference="declared synthetic",
        )
    model = hansen()
    for field in ("alpha", "V:A", "delta_p:A"):
        with pytest.raises(ValueError):
            replace(model, parameters={**model.parameters, field: -1.0})
    with pytest.raises(ValueError):
        replace(model, parameters={**model.parameters, "V:A": 0.0})
    zero = replace(model, parameters={**model.parameters, "alpha": 0.0})
    assert activity_thermodynamic_report(
        zero, {"A": 0.3, "B": 0.7}, temperature_K=300
    ).enthalpy_excess_J_mol == pytest.approx(0, abs=1e-10)


def test_old_activity_payload_and_detached_provenance_remain_compatible():
    old = {"model_id": "old", "component_ids": ("A", "B"), "model": "ideal", "parameters": {}}
    assert "provenance" not in ActivityModelSpec(**old).to_dict()
    source = {"record": {"id": "explicit"}}
    model = ActivityModelSpec(**old, provenance=source)
    source["record"]["id"] = "mutated"
    assert model.provenance["record"]["id"] == "explicit"
    with pytest.raises(ValueError):
        ActivityModelSpec(**old, provenance={"bad": float("nan")})
