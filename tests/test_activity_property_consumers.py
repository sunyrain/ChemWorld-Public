"""Activity energy reaches the existing property interface and declared references."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.physchem import ActivityModelSpec, activity_thermodynamic_report
from chemworld.physchem.curated_properties import curated_property_package
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation


def inputs():
    packages = tuple(
        ComponentPropertyPackage(
            replace(
                curated_property_package(key).component,
                allowed_property_correlations=("liquid_heat_capacity",),
            ),
            (
                PropertyCorrelation(
                    correlation_id=f"synthetic-{key}-cp",
                    property_id="liquid_heat_capacity",
                    equation_id="cp_polynomial",
                    coefficients={"a": cp},
                    input_units={"temperature": "K"},
                    output_unit="J/(mol*K)",
                    validity_ranges={"temperature": (275, 450)},
                    source_note="Synthetic consumer fixture",
                ),
            ),
        )
        for key, cp in (("water", 80), ("ethanol", 110))
    )
    references = {
        key: CaloricReference(
            phase="liquid",
            temperature_K=300,
            energy_J_mol=h,
            entropy_J_mol_K=s,
        )
        for key, h, s in (("water", -100, 30), ("ethanol", 200, 50))
    }
    model = ActivityModelSpec(
        model_id="synthetic-wilson-caloric",
        component_ids=("water", "ethanol"),
        model="wilson",
        parameters={"lambda_b:water|ethanol": 150, "lambda_b:ethanol|water": -90},
    )
    return packages, references, model


def evaluate(temperature=340, amounts=None, quantities=None, **kwargs):
    packages, references, model = inputs()
    options = {
        "temperature_K": temperature,
        "pressure_Pa": 101325,
        "phase": "liquid",
        "quantities": quantities
        or ("enthalpy", "cp", "entropy", "chemical_potential", "activity_coefficient"),
        "references": references,
        "activity_model": model,
    }
    options.update(kwargs)
    return evaluate_property_state(packages, amounts or {"water": 2, "ethanol": 3}, **options)


def independent_gibbs(temperature, amounts):
    """Direct binary Wilson GE plus pure Cp integrals, independent of derivative machinery."""
    n = sum(amounts.values())
    x, y = amounts["water"] / n, amounts["ethanol"] / n
    pure_h = x * (-100 + 80 * (temperature - 300)) + y * (200 + 110 * (temperature - 300))
    pure_s = x * (30 + 80 * log(temperature / 300)) + y * (50 + 110 * log(temperature / 300))
    mixing = R * temperature * (x * log(x) + y * log(y))
    excess = (
        -R
        * temperature
        * (x * log(x + y * exp(150 / temperature)) + y * log(y + x * exp(-90 / temperature)))
    )
    return n * (pure_h - temperature * pure_s + mixing + excess)


def test_original_state_consumer_matches_gibbs_and_partial_molar_derivatives():
    amounts = {"water": 2, "ethanol": 3}
    report = evaluate(amounts=amounts)
    props = report.properties
    assert 5 * (props["enthalpy"].value - 340 * props["entropy"].value) == pytest.approx(
        independent_gibbs(340, amounts), rel=1e-12
    )
    for key in amounts:
        step = 1e-4
        high, low = dict(amounts), dict(amounts)
        high[key] += step
        low[key] -= step
        expected = (independent_gibbs(340, high) - independent_gibbs(340, low)) / (2 * step)
        assert report.component_properties["chemical_potential"][key].value == pytest.approx(
            expected, rel=2e-9
        )
    assert report.model_id == "synthetic-wilson-caloric"
    assert (
        json.loads(json.dumps(report.to_dict()))["model_details"]["activity"]["model"] == "wilson"
    )


def test_temperature_response_and_pure_reference_offsets_are_consistent():
    center = evaluate()
    high, low = evaluate(340.001), evaluate(339.999)
    assert (high.properties["enthalpy"].value - low.properties["enthalpy"].value) / 0.002 == (
        pytest.approx(center.properties["cp"].value, rel=1e-8)
    )
    assert (high.properties["entropy"].value - low.properties["entropy"].value) / 0.002 == (
        pytest.approx(center.properties["cp"].value / 340, rel=1e-8)
    )
    _, references, _ = inputs()
    shifted = {
        key: replace(
            ref, energy_J_mol=ref.energy_J_mol + 500, entropy_J_mol_K=ref.entropy_J_mol_K + 2
        )
        for key, ref in references.items()
    }
    changed = evaluate(references=shifted)
    assert changed.properties["enthalpy"].value - center.properties["enthalpy"].value == (
        pytest.approx(500)
    )
    assert changed.properties["entropy"].value - center.properties[
        "entropy"
    ].value == pytest.approx(2)
    assert changed.properties["cp"].value == center.properties["cp"].value
    for key in references:
        assert (
            changed.component_properties["chemical_potential"][key].value
            - center.component_properties["chemical_potential"][key].value
        ) == pytest.approx(-180)


def test_pure_endpoint_absent_species_and_aliases_use_existing_identity_contract():
    report = evaluate(
        amounts={"7732-18-5": 2}, quantities=("enthalpy", "cp", "activity_coefficient")
    )
    assert report.properties["enthalpy"].value == pytest.approx(-100 + 80 * 40)
    assert report.properties["cp"].value == pytest.approx(80)
    assert report.component_properties["activity_coefficient"]["water"].value == pytest.approx(1)
    assert set(report.component_properties["activity_coefficient"]) == {"water", "ethanol"}
    assert dict(report.component_amounts_mol) == {"water": 2}


def test_gamma_only_does_not_require_unrequested_caloric_data():
    packages, _, model = inputs()
    bare = tuple(replace(package, correlations=()) for package in packages)
    report = evaluate_property_state(
        bare,
        {"water": 2, "ethanol": 3},
        temperature_K=340,
        pressure_Pa=101325,
        phase="liquid",
        quantities=("activity_coefficient",),
        activity_model=model,
    )
    expected = activity_thermodynamic_report(model, {"water": 2, "ethanol": 3}, temperature_K=340)
    for key, gamma in expected.activity_coefficients.items():
        assert report.component_properties["activity_coefficient"][key].value == gamma
    assert not report.reference_states


@pytest.mark.parametrize(
    "options,match",
    [
        ({"phase": "gas"}, "liquid phase"),
        ({"eos": object()}, "simultaneous EOS"),
        ({"quantities": ("cv",)}, "Cv requires"),
        ({"quantities": ("internal_energy",)}, "volume-mixing"),
        ({"quantities": ("fugacity_coefficient",)}, "fugacity requires"),
        ({"activity_model": None, "quantities": ("activity_coefficient",)}, "explicit liquid"),
    ],
)
def test_activity_model_cannot_invent_other_physics(options, match):
    with pytest.raises(ValueError, match=match):
        evaluate(**options)


def test_identity_and_zero_fraction_chemical_potential_rejections():
    mismatched = ActivityModelSpec("mismatch", ("water", "other"), "ideal")
    with pytest.raises(ValueError, match="identities"):
        evaluate(activity_model=mismatched)
    with pytest.raises(ValueError, match="zero mole fractions"):
        evaluate(amounts={"water": 1, "ethanol": 0})


def test_internal_energy_reference_cannot_be_reinterpreted_as_excess_enthalpy():
    _, references, _ = inputs()
    refs = {key: replace(ref, energy_basis="internal_energy") for key, ref in references.items()}
    with pytest.raises(ValueError, match="pure-liquid H references"):
        evaluate(references=refs, quantities=("internal_energy",))
