# ruff: noqa: C408 -- scientific parameter declarations
import json

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.foundation.units import convert_value
from chemworld.physchem import (
    VIRIAL_METHODS,
    ComponentPropertyPackage,
    ComponentSpec,
    ComponentUncertainty,
    correlation_temperature_derivative,
    evaluate_correlation,
    second_virial_coefficient_report,
    tsonopoulos_polar_parameters,
    virial_correlation,
    virial_derivatives,
    virial_matrix,
    virial_temperature_report,
    virial_value,
    virial_vector,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard


def parameters(method):
    all_values = dict(Tc=500.0, Pc=5e6, Vc=0.0002, omega=0.2, a=0.03, b=0.01)
    return {k: all_values[k] for k in VIRIAL_METHODS[method].required.split()}


def card_for(curve):
    component = ComponentSpec("synthetic", "CH4", molecular_weight_g_mol=16.0)
    return PropertyInputCard(
        ComponentPropertyPackage(component, (curve,)),
        (
            InputSource(
                DataSourceProvenance(
                    "synthetic", 0, "Synthetic test parameters", version="1", license="CC0"
                ),
                "synthetic",
            ),
        ),
        {
            curve.correlation_id: ComponentUncertainty(
                curve.correlation_id,
                unit=curve.output_unit,
                source_id="synthetic",
                coverage="unspecified",
                note="Synthetic, no uncertainty estimate",
            )
        },
        None,
        "Virial coefficients only, no caloric reference",
    )


@pytest.mark.parametrize("method", VIRIAL_METHODS)
def test_original_declaration_card_report_and_batch(method):
    p = parameters(method)
    curve = virial_correlation(
        method,
        parameters=p,
        temperature_range_K=(300, 1500),
        source_note="Synthetic validation inputs",
    )
    card = PropertyInputCard.from_json(json.dumps(card_for(curve).to_dict()))
    value = card.package.evaluate(curve.property_id, temperature_K=600)
    expected = virial_derivatives(method, 600, p)
    assert value.value == pytest.approx(expected[0])
    assert correlation_temperature_derivative(curve, temperature_K=600).value == pytest.approx(
        expected[1]
    )
    reports = virial_temperature_report(curve, temperature_K=600)
    assert tuple(r.to(r.unit).value for r in reports) == pytest.approx(expected)
    assert virial_vector(method, 600, [p, p]) == (expected, expected)
    assert virial_matrix(method, 600, [[p, p], [p, p]]) == (
        (expected, expected),
        (expected, expected),
    )
    if not VIRIAL_METHODS[method].third:
        assert second_virial_coefficient_report(curve, temperature_K=600).value == pytest.approx(
            expected[0]
        )
    with pytest.raises(ValueError, match="range"):
        virial_temperature_report(curve, temperature_K=200)
    if not VIRIAL_METHODS[method].integrals:
        with pytest.raises(ValueError, match="integral"):
            virial_value(method, 600, order=-1, **p)


def test_explicit_polar_conventions_survive_json_and_reject_conflicts():
    p = dict(Tc=500.0, Pc=5e6, omega=0.2)
    curves = [
        virial_correlation(
            "Tsonopoulos_extended",
            parameters=p,
            temperature_range_K=(300, 1500),
            source_note="Synthetic methanol class",
            polar=dict(species_type="methanol", convention=convention),
        )
        for convention in ("documented", "chemicals_1_5_2")
    ]
    assert curves[0].coefficients["a"] == 0.0878
    assert curves[1].coefficients["a"] == 0
    assert (
        evaluate_correlation(curves[0], temperature_K=400).value
        != evaluate_correlation(curves[1], temperature_K=400).value
    )
    restored = PropertyInputCard.from_json(json.dumps(card_for(curves[0]).to_dict()))
    assert (
        restored.package.correlations[0].metadata["polar_selection"]["convention"] == "documented"
    )
    with pytest.raises(ValueError, match="not both"):
        virial_correlation(
            "Meng",
            parameters={**p, "a": 0},
            temperature_range_K=(300, 600),
            source_note="Synthetic",
            polar=dict(dipole=1.0, haloalkane=False),
        )
    with pytest.raises(ValueError, match="explicit dipole"):
        tsonopoulos_polar_parameters(Tc=500, Pc=5e6, species_type="ketone")
    with pytest.raises(ValueError, match="unknown"):
        tsonopoulos_polar_parameters(Tc=500, Pc=5e6, species_type="guess")


@pytest.mark.parametrize("changed", [{"Tc": 0}, {"Pc": -1}, {"omega": float("nan")}, {"extra": 1}])
def test_invalid_parameter_domains(changed):
    with pytest.raises(ValueError):
        virial_value("Abbott", 500, **{**parameters("Abbott"), **changed})


def test_orders_matrix_and_units():
    p = parameters("Abbott")
    for order in (True, 4, -3):
        with pytest.raises(ValueError):
            virial_value("Abbott", 500, order=order, **p)
    for temperature in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            virial_value("Abbott", temperature, **p)
    with pytest.raises(ValueError, match="rectangular"):
        virial_matrix("Abbott", 500, [[p], [p, p]])
    assert convert_value(1, "L^2/mol^2", "m^6/mol^2") == 1e-6
    assert convert_value(1, "(L^2/mol^2)/K^3", "(m^6/mol^2)/K^3") == 1e-6
    with pytest.raises(ValueError):
        convert_value(1, "m^6/mol^2", "m^3/mol")
