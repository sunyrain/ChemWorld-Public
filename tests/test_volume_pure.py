"""Phase, basis, boundaries and real property-card consumption of pure volumes."""
# ruff: noqa: C408 -- scientific parameter declarations

import json
from dataclasses import replace

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.foundation.units import Quantity
from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    ComponentUncertainty,
    PropertyCorrelation,
    correlation_temperature_derivative,
    evaluate_correlation,
    integrate_correlation,
    molar_volume_report,
    volume_correlation,
    volume_derivatives,
    volume_parameter_jacobian,
    volume_value,
)
from chemworld.physchem.caloric_math import derivative_values, nonnegative_power
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.volume_kernels import BHIRUD_TABLE


def test_solid_and_molten_phases_are_not_interchangeable():
    solid = volume_correlation(
        "Goodman",
        parameters={"Tt": 243.225, "Vml": 0.00023585},
        temperature_range_K=(100, 243.225),
        source_note="Goodman public example",
    )
    report = molar_volume_report(solid, temperature_K=200, phase="solid")
    assert report.molar_volume_m3_mol == pytest.approx(0.0002053665090860923)
    assert report.density_kg_m3 is None
    with pytest.raises(ValueError, match="phase"):
        molar_volume_report(solid, temperature_K=200, phase="liquid")
    with pytest.raises(ValueError, match="Tt"):
        volume_value("Goodman", 244, Tt=243.225, Vml=0.00023585)
    with pytest.raises(ValueError, match="Tm"):
        volume_value("CRC_inorganic", 200, Tm=239.08, rho0=2370, k=2.687)
    with pytest.raises(ValueError, match="positive"):
        volume_value("CRC_inorganic", 1000, Tm=239.08, rho0=2370, k=20)


def test_rackett_fit_is_specific_volume_without_mw_not_mass_density():
    p = dict(Tc=425.18, rhoc=228.3, b=0.2724, n=0.2863)
    v = volume_value("Rackett_fit", 298.15, **p)
    assert v == pytest.approx(0.00174520519958, rel=1e-10)
    assert volume_value("Rackett_fit", 298.15, **p, output="density") == pytest.approx(1 / v)
    assert volume_value("Rackett_fit", 298.15, **p, MW=58.123) == pytest.approx(v * 0.058123)
    with pytest.raises(ValueError, match="MW"):
        volume_value("Rackett_fit", 298.15, **p, output="molar_volume")
    with pytest.raises(ValueError, match="MW"):
        volume_value("Rackett", 300, Tc=600, Pc=4e6, Zc=0.27, output="density")


@pytest.mark.parametrize(
    "method,parameters",
    (
        ("Rackett", dict(Tc=600, Pc=4e6, Zc=0.27)),
        ("Yen_Woods_saturation", dict(Tc=600, Vc=0.0002, Zc=0.27)),
        ("Yamada_Gunn", dict(Tc=600, Pc=4e6, omega=0.2)),
        ("Townsend_Hales", dict(Tc=600, Vc=0.0002, omega=0.2)),
        ("COSTALD", dict(Tc=600, Vc=0.0002, omega=0.2)),
        ("SNM0", dict(Tc=600, Vc=0.0002, omega=0.2)),
        ("Campbell_Thodos", dict(Tc=600, Tb=350, Pc=4e6, MW=78.11)),
        ("PPDS17", dict(Tc=600, a0=0.01, a1=0.3, a2=0.02)),
        ("TDE_VDNS_rho", dict(Tc=600, rhoc=300, a1=500, a2=100, a3=-20, a4=10)),
        ("Rackett_fit", dict(Tc=600, rhoc=300, b=0.27, n=0.28)),
        ("volume_VDI_PPDS", dict(Tc=600, rhoc=300, a=500, b=100, c=-20, d=10)),
    ),
)
def test_finite_critical_value_never_implies_a_finite_derivative(method, parameters):
    assert volume_value(method, 600, **parameters) > 0
    with pytest.raises(ValueError, match="derivative"):
        volume_derivatives(method, 600, parameters)
    with pytest.raises(ValueError, match="Tc"):
        volume_value(method, 601, **parameters)


def test_critical_polynomial_density_retains_derivatives_when_fractional_terms_are_absent():
    p = dict(Tc=600, rhoc=300, a1=0, a2=100, a3=20, a4=10)
    value, first, second, third = volume_derivatives("TDE_VDNS_rho", 600, p, order=3)
    assert (value, first, second, third) == pytest.approx(
        (300, -100 / 600, 40 / 600**2, -60 / 600**3)
    )
    assert derivative_values(lambda t: nonnegative_power(t, 4 / 3), 0, order=1) == (0, 0)
    with pytest.raises(ValueError, match="derivative"):
        derivative_values(lambda t: nonnegative_power(t, 4 / 3), 0, order=2)


def test_bhirud_knots_and_join_keep_value_and_derivative_boundaries_separate():
    p = dict(Tc=500.0, Pc=4e6, omega=0.2)
    assert len(BHIRUD_TABLE[0]) == 12
    assert volume_value("Bhirud_normal", 490, **p) > 0
    with pytest.raises(ValueError, match="join"):
        volume_derivatives("Bhirud_normal", 490, p)
    assert volume_derivatives("Bhirud_normal", 495, p, order=2)[0] > 0
    with pytest.raises(ValueError, match="knot"):
        volume_derivatives("Bhirud_normal", 495, p, order=3)
    assert volume_derivatives("Bhirud_normal", 500, p, order=3)[0] > 0
    with pytest.raises(ValueError, match="Tc"):
        volume_value("Bhirud_normal", 500.1, **p)


def test_ideal_gas_uses_actual_pressure_in_values_derivatives_and_integrals():
    curve = volume_correlation(
        "ideal_gas",
        parameters={},
        temperature_range_K=(200, 600),
        pressure_range_Pa=(1e5, 3e5),
        source_note="ideal gas definition",
    )
    low = evaluate_correlation(curve, temperature_K=300, pressure_Pa=1e5).value
    high = evaluate_correlation(curve, temperature_K=300, pressure_Pa=2e5).value
    assert low == pytest.approx(2 * high)
    d = correlation_temperature_derivative(curve, temperature_K=300, pressure_Pa=2e5)
    assert d.to("m^3/(mol*K)").value == pytest.approx(high / 300)
    integral = integrate_correlation(
        curve, initial_temperature_K=250, final_temperature_K=350, pressure_Pa=2e5
    )
    assert integral.value == pytest.approx(high * 100)
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=300)
    with pytest.raises(ValueError, match="pressure"):
        evaluate_correlation(curve, temperature_K=300, pressure_Pa=4e5, validity_policy="raise")


def test_card_json_source_units_and_actual_density_consumption():
    component = ComponentSpec("benzene", "C6H6")
    curve = volume_correlation(
        "PPDS17",
        parameters=dict(Tc=562.05, a0=0.0115508, a1=0.281004, a2=-0.00635447),
        output="density",
        temperature_range_K=(280, 500),
        source_note="NIST TDE public benzene example",
    )
    card = PropertyInputCard(
        ComponentPropertyPackage(component, (curve,)),
        (
            InputSource(
                DataSourceProvenance(
                    "tde", 1, "NIST TDE PPDS17", "public example", "public scientific coefficients"
                ),
                "literature",
            ),
        ),
        {
            curve.correlation_id: ComponentUncertainty(
                curve.correlation_id,
                unit="kg/m^3",
                coverage="example only",
                source_id="tde",
                note="uncertainty not supplied",
            )
        },
        None,
        "saturation density only; no caloric datum",
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    assert restored.to_dict() == card.to_dict()
    states = [
        restored.evaluate_state(
            temperature_K=t,
            pressure_Pa=101325,
            phase="liquid",
            quantities=("density", "molar_volume"),
        )
        for t in (300, 400)
    ]
    rho = states[0].properties["density"]
    assert rho.to("g/mL").value == pytest.approx(0.871520087707, rel=1e-10)
    assert states[1].properties["molar_volume"].value > states[0].properties["molar_volume"].value
    assert rho.value * states[0].properties["molar_volume"].value == pytest.approx(
        component.molecular_weight_g_mol / 1000
    )
    assert curve.metadata["physical_relation"] == "saturated_liquid"
    assert Quantity(rho.value, "kg/m^3").to("g/mL").value == rho.to("g/mL").value
    with pytest.raises(ValueError, match="EOS"):
        restored.evaluate_state(
            temperature_K=300, pressure_Pa=101325, phase="liquid", quantities=("dV_dP_T",)
        )
    with pytest.raises(ValueError):
        replace(curve, property_id="solid_density")


def test_old_rackett_path_uses_same_equation_and_preserves_its_strict_endpoint():
    p = dict(Tc=600.0, Pc=4e6, Zc=0.27)
    curve = PropertyCorrelation(
        correlation_id="old",
        property_id="liquid_molar_volume",
        equation_id="rackett_liquid_molar_volume",
        coefficients=p,
        input_units={"temperature": "K"},
        output_unit="m^3/mol",
        validity_ranges={"temperature": (200, 599)},
        source_note="legacy input",
    )
    assert evaluate_correlation(curve, temperature_K=300).value == volume_value("Rackett", 300, **p)
    with pytest.raises(ValueError, match="Tc"):
        evaluate_correlation(curve, temperature_K=600, validity_policy="ignore")


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1.0, 0.0, True))
def test_nonphysical_temperatures_and_critical_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        volume_value("Rackett", bad, Tc=600, Pc=4e6, Zc=0.27)
    with pytest.raises(ValueError):
        volume_value("Rackett", 300, Tc=bad, Pc=4e6, Zc=0.27)


def test_selector_missing_units_and_source_ranges_are_explicit():
    p = dict(Tc=600, Tb=350, Pc=4e6, MW=78.11, dipole=1, has_hydroxyl=1)
    with pytest.raises(ValueError, match="continuous"):
        volume_parameter_jacobian("Campbell_Thodos", 300, p, names=("has_hydroxyl",))
    with pytest.raises(ValueError, match="0 or 1"):
        volume_value("Campbell_Thodos", 300, **{**p, "has_hydroxyl": 2})
    with pytest.raises(ValueError, match="missing"):
        volume_value("Rackett", 300, Tc=600, Zc=0.27)
    with pytest.raises(ValueError, match="source"):
        volume_correlation(
            "Rackett",
            parameters=dict(Tc=600, Pc=4e6, Zc=0.27),
            temperature_range_K=(200, 500),
            source_note="",
        )
    with pytest.raises(ValueError, match="pressure"):
        volume_correlation(
            "Rackett",
            parameters=dict(Tc=600, Pc=4e6, Zc=0.27),
            temperature_range_K=(200, 500),
            pressure_range_Pa=(1e5, 1e6),
            source_note="test",
        )


def test_hydroxyl_convention_is_serialized_and_not_a_fit_parameter():
    p = dict(Tc=647.14, Tb=373.15, Pc=22048320, MW=18.01528, dipole=1.85, has_hydroxyl=True)
    curve = volume_correlation(
        "Campbell_Thodos",
        parameters=p,
        temperature_range_K=(300, 600),
        source_note="public API convention",
    )
    assert curve.coefficients["hydroxyl_polar_term"] == 1
    restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert restored.coefficients == curve.coefficients
    compatible = volume_value("Campbell_Thodos", 450, **p)
    documented = volume_value("Campbell_Thodos", 450, **p, hydroxyl_polar_term=0)
    assert compatible > documented
    with pytest.raises(ValueError, match="continuous"):
        volume_parameter_jacobian(
            "Campbell_Thodos", 450, {**p, "hydroxyl_polar_term": 1}, names=("hydroxyl_polar_term",)
        )


def test_correlation_mw_cannot_silently_disagree_with_its_component():
    component = ComponentSpec("benzene", "C6H6")
    curve = volume_correlation(
        "Rackett",
        parameters=dict(Tc=600, Pc=4e6, Zc=0.27, MW=44),
        output="density",
        temperature_range_K=(200, 500),
        source_note="wrong component fixture",
    )
    with pytest.raises(ValueError, match="MW must match"):
        ComponentPropertyPackage(component, (curve,)).evaluate("liquid_density", temperature_K=300)
