"""Physical units, critical limits and actual interface-property consumers."""
# ruff: noqa: C408 -- compact scientific parameters

import json
import math
from dataclasses import replace

import pytest

from chemworld.data.interface import INTERFACE_SOURCES, InterfaceDatabase
from chemworld.physchem.interface_kernels import interface_derivatives, interface_value
from chemworld.physchem.interfaces import (
    capillary_force,
    diguilio_teja,
    laplace_pressure,
    mixture_surface_tension,
    parachor_surface_tension,
    water_oil_interfacial_tension,
    weinaug_katz,
    winterfeld_scriven_davis,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec, MixtureSpec, PropertyCorrelation


@pytest.fixture
def database(tmp_path):
    samples = {
        "JASPER_LANGE": "CAS\tName\ta\tb\tTmin\tTmax\n"
        "71-36-3\tsynthetic\t24\t.0773\t250\t400\n"
        "74-82-8\tmissing\t20\t.05\t\t400\n"
        "7732-18-5\tmissing fit\t\t.05\t250\t400\n",
        "SOMAYAJULU": "CAS\tChemical\tTt\tTc\tA\tB\tC\n"
        "7732-18-5\tsynthetic\t273.16\t647.126\t232.713514\t-140.18645\t-4.890098\n",
    }
    for source in INTERFACE_SOURCES:
        if source.source_id in samples:
            path = tmp_path / source.relative_path
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(samples[source.source_id], encoding="utf8")
    return InterfaceDatabase(tmp_path)


def test_source_to_input_card_to_state_to_pressure_and_force(database):
    assert database.loaded_sources == () and len(database.catalog) == 5
    component = ComponentSpec("butanol", "C4H10O", cas_number="71-36-3")
    card = database.property_input("71-36-3", component, method="JASPER_LANGE")
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored.to_dict() == card.to_dict()
    pressures = []
    for t in (280, 350):
        state = restored.evaluate_state(
            temperature_K=t, pressure_Pa=101325, phase="liquid", quantities=("surface_tension",)
        )
        tension = state.properties["surface_tension"]
        pressure = laplace_pressure(
            tension.to("mN/m"), curvature_1_per_m=1000, curvature_2_per_m=1000
        )
        force = capillary_force(tension, wetted_perimeter_m=0.02, contact_angle_rad=math.pi / 3)
        assert pressure.value == pytest.approx(2000 * tension.value)
        assert force.value == pytest.approx(0.01 * tension.value)
        assert force.quantity().unit == "N"
        pressures.append(pressure.value)
    assert pressures[0] > pressures[1]
    curve = card.package.correlations[0]
    assert curve.metadata["interface"] == "liquid-gas"
    assert curve.metadata["source_record"]["fields"]["a"] == "24"
    assert correlation_temperature_derivative(curve, temperature_K=300).to(
        "mN/(m*K)"
    ).value == pytest.approx(-0.0773)
    with pytest.raises(ValueError, match="liquid"):
        restored.evaluate_state(
            temperature_K=300, pressure_Pa=101325, phase="gas", quantities=("surface_tension",)
        )
    with pytest.raises(ValueError, match="identity"):
        database.property_input(
            "71-36-3", replace(component, cas_number="74-82-8"), method="JASPER_LANGE"
        )


def test_missing_source_bounds_require_application_interval_and_missing_fit_is_never_filled(
    database,
):
    with pytest.raises(ValueError, match="explicit"):
        database.surface_tension("74-82-8", method="JASPER_LANGE")
    curve = database.surface_tension(
        "74-82-8", method="JASPER_LANGE", temperature_bounds_K=(250, 350)
    )
    assert curve.metadata["source_bounds_K"] == [None, 400]
    assert curve.metadata["temperature_range_basis"] == "caller_application"
    assert evaluate_correlation(curve, temperature_K=300).value > 0
    with pytest.raises(ValueError, match="exceeds"):
        database.surface_tension("74-82-8", method="JASPER_LANGE", temperature_bounds_K=(250, 450))
    with pytest.raises(ValueError, match="missing"):
        database.surface_tension("7732-18-5", method="JASPER_LANGE")
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=400, validity_policy="raise")


def test_source_critical_value_separate_from_unavailable_second_derivative(database):
    curve = database.surface_tension("7732-18-5", method="SOMAYAJULU")
    assert evaluate_correlation(curve, temperature_K=647.126).value == 0
    assert correlation_temperature_derivative(curve, temperature_K=647.126).value == 0
    with pytest.raises(ValueError, match="critical derivative"):
        interface_derivatives("Somayajulu", 647.126, dict(curve.coefficients), order=2)


@pytest.mark.parametrize("exponent,order", [(1.25, 1), (2.25, 2), (3.25, 3)])
def test_critical_zero_derivatives_only_through_the_existing_order(exponent, order):
    p = dict(Tc=500, sigma0=0.1, n0=exponent)
    assert interface_derivatives("REFPROP_sigma", 500, p, order=order) == (0,) * (order + 1)
    if order < 3:
        with pytest.raises(ValueError, match="critical derivative"):
            interface_derivatives("REFPROP_sigma", 500, p, order=order + 1)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_integer_powers_have_exact_endpoint_derivatives(n):
    p = dict(Tc=500, sigma0=0.1, n0=n)
    derivatives = interface_derivatives("REFPROP_sigma", 500, p, order=3)
    expected = [0.0] * 4
    expected[n] = 0.1 * math.factorial(n) * (-1 / 500) ** n
    assert derivatives == pytest.approx(expected)


def test_disabled_and_cancelling_fractional_terms_do_not_invent_singularities():
    assert interface_derivatives("REFPROP_sigma", 500, dict(Tc=500, sigma0=0, n0=-2), order=3) == (
        0,
        0,
        0,
        0,
    )
    assert interface_derivatives(
        "REFPROP_sigma", 500, dict(Tc=500, sigma0=0.1, n0=0.5, sigma1=-0.1, n1=0.5), order=3
    ) == (0, 0, 0, 0)


def test_watson_endpoint_variable_exponent_and_high_order_polynomial():
    p = dict(Tc=500, a1=math.log(0.1), a2=1, a3=1)
    assert interface_derivatives("Watson_sigma", 500, p, order=2) == pytest.approx(
        (0, 0, 2 * 0.1 / 500**2)
    )
    with pytest.raises(ValueError, match="logarithmic"):
        interface_derivatives("Watson_sigma", 500, p, order=3)
    p = dict(Tc=500, a1=0.1, a2=0.2, a3=0.3)
    t, tau = 300, 0.4
    assert interface_derivatives("ISTExpansion", t, p, order=3) == pytest.approx(
        (
            0.1 * tau + 0.2 * tau**2 + 0.3 * tau**3,
            -(0.1 + 0.4 * tau + 0.9 * tau**2) / 500,
            (0.4 + 1.8 * tau) / 500**2,
            -1.8 / 500**3,
        )
    )


def test_supercritical_is_an_explicit_computational_extension():
    with pytest.raises(ValueError, match="T <= Tc"):
        interface_value("sigma_IAPWS", 650)
    assert interface_derivatives("sigma_IAPWS", 650, {}, order=3, above_critical="zero") == (
        0,
        0,
        0,
        0,
    )
    assert interface_derivatives("sigma_IAPWS", 647.096, {}, order=1) == (0, 0)
    with pytest.raises(ValueError, match="critical derivative"):
        interface_derivatives("sigma_IAPWS", 647.096, {}, order=2, above_critical="zero")


@pytest.mark.parametrize(
    "method,p,first_exists",
    [
        ("Sastri_Rao_alcohol", dict(Tc=500, Tb=300, Pc=5e6), False),
        ("Mersmann_Kind_sigma", dict(Tc=500, Tb=300, Tm=200, Pc=5e6), True),
        ("PPDS14", dict(Tc=500, a0=0.1, a1=0.8, a2=0.2), False),
    ],
)
def test_sublinear_and_linear_critical_leading_terms(method, p, first_exists):
    assert interface_value(method, 500, **p) == 0
    if first_exists:
        assert interface_derivatives(method, 500, p)[1] < 0
    else:
        with pytest.raises(ValueError, match="critical derivative"):
            interface_derivatives(method, 500, p)
    with pytest.raises(ValueError, match="critical derivative"):
        interface_derivatives(method, 500, p, order=2)


def test_point_correlation_has_no_invented_temperature_dependence():
    curve = PropertyCorrelation(
        "point",
        "interface_source_point",
        dict(T_ref=300, sigma_ref=0.02),
        {"temperature": "K"},
        "N/m",
        "surface_tension",
        {"temperature": (300, 300)},
        "Explicit source observation",
    )
    assert evaluate_correlation(curve, temperature_K=300).value == 0.02
    with pytest.raises(ValueError, match="observation"):
        correlation_temperature_derivative(curve, temperature_K=300)
    with pytest.raises(ValueError, match="extrapolated"):
        evaluate_correlation(curve, temperature_K=301, validity_policy="ignore")


@pytest.mark.parametrize(
    "method,t,p",
    [
        ("Jasper", 500, dict(a=10, b=1)),
        ("Aleem", 500, dict(MW=20, Tb=100, rhol=500, Hvap_Tb=1e5, Cpl=1000)),
        ("sigma_Gharagheizi_1", 499.5, dict(Tc=500, omega=1, MW=100)),
        ("sigma_Gharagheizi_1", 300, dict(Tc=500, omega=-0.1, MW=100)),
        ("PPDS14", 300, dict(Tc=500, a0=-0.1, a1=1.2, a2=0)),
        ("Brock_Bird", 300, dict(Tc=500, Tb=500, Pc=5e6)),
        ("Mersmann_Kind_sigma", 300, dict(Tc=500, Tb=300, Tm=350, Pc=5e6)),
        ("API10A32", 300, dict(Tc=500, K_W=0)),
        ("Jasper", True, dict(a=20, b=0.1)),
        ("unknown", 300, {}),
        ("Jasper", 300, dict(a=20, b=0.1, extra=1)),
        ("Jasper", 300, dict(a=math.nan, b=0.1)),
    ],
)
def test_bad_inputs_and_negative_predictions_reject(method, t, p):
    with pytest.raises(ValueError):
        interface_value(method, t, **p)


def test_petroleum_unit_correction_and_aleem_mass_basis():
    assert interface_value(
        "API10A32", 288.7055555555555, Tc=741.1111111111111, K_W=12.4
    ) == pytest.approx(0.029577333312096968)
    assert interface_value(
        "Aleem", 90, MW=16.04246, Tb=111.6, rhol=458.7, Hvap_Tb=510870, Cpl=2465
    ) == pytest.approx(0.01669970230131523)


def mixtures():
    components = (
        ComponentSpec("a", "C4H10"),
        ComponentSpec("b", "C6H14"),
    )
    return tuple(
        MixtureSpec.from_mole_fractions(
            tuple(replace(c, default_phase=phase) for c in components),
            xs,
            phase_label=phase,
            temperature_K=298.15,
            pressure_Pa=101325,
        )
        for xs, phase in (((0.4, 0.6), "liquid"), ((0.6, 0.4), "gas"))
    )


def test_mixture_and_two_phase_consumers_use_existing_inventory():
    liquid, vapor = mixtures()
    liquid = MixtureSpec.from_dict(json.loads(json.dumps(liquid.to_dict())))
    a = mixture_surface_tension(
        liquid,
        method="Winterfeld_Scriven_Davis",
        component_tensions_N_m=(0.01547, 0.02877),
        molar_densities_mol_m3=(8610, 15530),
    )
    assert a.value == winterfeld_scriven_davis(
        liquid.mole_fractions, (0.01547, 0.02877), (8610, 15530)
    )
    b = mixture_surface_tension(
        liquid,
        method="Diguilio_Teja",
        component_tensions_N_m=(0.01424, 0.02530),
        boiling_temperatures_K=(309.21, 312.95),
        critical_temperatures_K=(469.7, 508),
    )
    assert b.inputs["sigma_at_Tb_N_m:a"] == 0.01424
    c = parachor_surface_tension(
        liquid,
        vapor,
        parachors_SI=(5.1e-5, 7.2e-5),
        liquid_molar_volume_m3_mol=0.000125,
        vapor_molar_volume_m3_mol=0.02011,
    )
    assert c.value == pytest.approx(0.06547479150776776)
    assert laplace_pressure(c, curvature_1_per_m=1000, curvature_2_per_m=0).value > 65
    with pytest.raises(ValueError, match="same T/P"):
        parachor_surface_tension(
            liquid,
            replace(vapor, temperature_K=300),
            parachors_SI=(5.1e-5, 7.2e-5),
            liquid_molar_volume_m3_mol=0.000125,
            vapor_molar_volume_m3_mol=0.02011,
        )
    with pytest.raises(ValueError, match="liquid"):
        mixture_surface_tension(
            vapor,
            method="Winterfeld_Scriven_Davis",
            component_tensions_N_m=(0.01, 0.02),
            molar_densities_mol_m3=(10000, 15000),
        )
    with pytest.raises(ValueError, match="same composition"):
        mixture_surface_tension(
            replace(liquid, mass_fractions=(0.5, 0.5)),
            method="Winterfeld_Scriven_Davis",
            component_tensions_N_m=(0.01, 0.02),
            molar_densities_mol_m3=(10000, 15000),
        )
    with pytest.raises(ValueError, match="exactly"):
        mixture_surface_tension(
            liquid,
            method="Winterfeld_Scriven_Davis",
            component_tensions_N_m=(0.01, 0.02),
            molar_densities_mol_m3=(10000, 15000),
            boiling_temperatures_K=(300, 310),
        )


def test_physical_zero_mixture_limits_and_no_zero_volume_gas_shortcut():
    assert winterfeld_scriven_davis((1,), (0,), (10000,)) == 0
    assert winterfeld_scriven_davis((0, 1), (0, 0.02), (10000, 15000)) == pytest.approx(0.02)
    assert diguilio_teja(500, (1,), (0.02,), (300,), (500,)) == 0
    assert weinaug_katz((1e-5,), 0.001, 0.001, (1,), (1,)) == 0
    assert weinaug_katz((1e-5,), 0.001, None, (1,), (1,)) == pytest.approx(1e-8)
    for call in (
        lambda: winterfeld_scriven_davis((0.5,), (0.02,), (10000,)),
        lambda: winterfeld_scriven_davis((1,), (-0.02,), (10000,)),
        lambda: winterfeld_scriven_davis((1,), (0.02,), (0,)),
        lambda: weinaug_katz((1e-5,), 0.001, 0, (1,), (1,)),
        lambda: diguilio_teja(501, (1,), (0.02,), (300,), (500,)),
        lambda: water_oil_interfacial_tension(
            300, water_density_kg_m3=-1, oil_density_kg_m3=750, oil_critical_temperature_K=900
        ),
    ):
        with pytest.raises(ValueError):
            call()


def test_water_oil_interface_remains_distinct_and_capillary_sign_is_physical():
    from chemworld.physchem.property_reports import PropertyEvaluation

    value = water_oil_interfacial_tension(
        580, water_density_kg_m3=980, oil_density_kg_m3=760, oil_critical_temperature_K=914
    )
    assert value == pytest.approx(0.02893598143089256)
    evaluation = PropertyEvaluation(
        "interfacial_tension", "water-oil", "Meybodi", value, "N/m", {"temperature": 580}
    )
    assert capillary_force(evaluation, wetted_perimeter_m=0.02, contact_angle_rad=math.pi).value < 0
    assert laplace_pressure(evaluation, curvature_1_per_m=10, curvature_2_per_m=-10).value == 0
    with pytest.raises(ValueError, match="interface-property"):
        laplace_pressure(
            replace(evaluation, property_id="thermal_conductivity"),
            curvature_1_per_m=1,
            curvature_2_per_m=1,
        )


def test_registry_property_curve_retains_critical_value_and_units():
    curve = PropertyCorrelation(
        "water",
        "interface_sigma_iapws",
        {},
        {"temperature": "K"},
        "N/m",
        "surface_tension",
        {"temperature": (273.16, 647.096)},
        "IAPWS",
    )
    assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(0.0716859625271)
    assert evaluate_correlation(curve, temperature_K=647.096).value == 0
