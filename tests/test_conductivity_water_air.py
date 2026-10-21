"""Water/air units, explicit critical inputs and shared heat-transfer paths."""

import json

import pytest

from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    air_conductivity_lemmon,
    conductivity_value,
    water_conductivity_iapws,
    water_reference_compressibility,
)
from chemworld.physchem.transport import internal_heat_transfer_coefficient, jacket_heat_transfer

WATER = {
    "Cp": 7634.337046792,
    "Cv": 3037.934412104,
    "mu": 70.905106751524e-6,
    "drho_dP": 5.209378197916e-6,
}
AIR = {
    "Cp": 2137.078854678728,
    "Cv": 35.24316159996235,
    "mu": 1.7762253265868595e-5,
    "drho_dP": 0.07417878614315769,
    "drho_dP_reference": 0.00035919027241528256,
}


def test_water_explicit_reference_approximation_and_mass_heat_capacity():
    t, rho = 620, 613.227777440324
    expected = water_conductivity_iapws(t, rho, **WATER, reference_approximation=True)
    assert expected == pytest.approx(0.48148519510200044)
    assert expected == water_conductivity_iapws(
        t,
        rho,
        **WATER,
        drho_dP_reference=water_reference_compressibility(rho),
    )
    with pytest.raises(ValueError, match="explicitly"):
        water_conductivity_iapws(t, rho, **WATER)
    with pytest.raises(ValueError, match="not both"):
        water_conductivity_iapws(
            t,
            rho,
            **WATER,
            reference_approximation=True,
            drho_dP_reference=water_reference_compressibility(rho),
        )
    with pytest.raises(ValueError, match="boolean"):
        water_conductivity_iapws(t, rho, **WATER, reference_approximation=1)


@pytest.mark.parametrize(
    "fluid,function,t,rho,phase,formula,critical",
    [
        (
            "water",
            water_conductivity_iapws,
            620,
            613.227777440324,
            "liquid",
            "H2O",
            {**WATER, "reference_approximation": True},
        ),
        ("air", air_conductivity_lemmon, 132.64, 10400, "gas", "N2", AIR),
    ],
)
def test_background_property_package_and_critical_heat_consumption(
    fluid,
    function,
    t,
    rho,
    phase,
    formula,
    critical,
):
    method = "k_IAPWS" if fluid == "water" else "k_air_lemmon"
    curve = PropertyCorrelation(
        fluid + "-background",
        "conductivity_" + method.lower(),
        {"rho": rho},
        {"temperature": "K"},
        "W/(m*K)",
        "thermal_conductivity",
        {"temperature": (100, 1000)},
        "Fixed-density mathematical transport path; no EOS phase inference",
        {"phase": phase},
    )
    restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
    package = ComponentPropertyPackage(
        ComponentSpec(fluid, formula, default_phase=phase), (restored,)
    )
    state = package.evaluate_state(
        temperature_K=t, pressure_Pa=1e5, phase=phase, quantities=("thermal_conductivity",)
    )
    background = state.properties["thermal_conductivity"].value
    assert background == function(t, rho) == conductivity_value(method, t, rho=rho)
    enhanced = function(t, rho, **critical)
    assert enhanced > background
    heats = []
    for k in (background, enhanced):
        h = internal_heat_transfer_coefficient(
            nusselt=3.66, thermal_conductivity_W_m_K=k, diameter_m=0.01
        )
        heats.append(
            jacket_heat_transfer(
                area_m2=0.1,
                overall_u_W_m2_K=h,
                reactor_temperature_K=t,
                jacket_temperature_K=t + 10,
                duration_s=5,
            ).heat_energy_J
        )
    assert heats[1] / heats[0] == pytest.approx(enhanced / background)


@pytest.mark.parametrize("key", ("Cp", "Cv", "mu", "drho_dP", "drho_dP_reference"))
@pytest.mark.parametrize("fluid", ("water", "air"))
def test_partial_critical_inputs_never_silently_disable_enhancement(fluid, key):
    function = water_conductivity_iapws if fluid == "water" else air_conductivity_lemmon
    kwargs = dict(AIR)
    del kwargs[key]
    with pytest.raises(ValueError):
        function(300, 100, **kwargs)


@pytest.mark.parametrize("fluid", ("water", "air"))
@pytest.mark.parametrize(
    "changes",
    (
        {"Cp": float("nan")},
        {"Cv": 3000},
        {"mu": 0},
        {"drho_dP": -1},
        {"drho_dP_reference": float("inf")},
        {"Cp": True},
    ),
)
def test_invalid_critical_states_reject(fluid, changes):
    function = water_conductivity_iapws if fluid == "water" else air_conductivity_lemmon
    with pytest.raises(ValueError):
        function(300, 100, **{**AIR, **changes})


@pytest.mark.parametrize(
    "function,t,rho",
    (
        (water_conductivity_iapws, 647.35, 322),
        (air_conductivity_lemmon, 132.64, 10400),
    ),
)
def test_zero_and_negative_excess_and_zero_density_are_finite_background(function, t, rho):
    for current, reference in ((0, 0), (0, 1e-6)):
        critical = {**AIR, "drho_dP": current, "drho_dP_reference": reference}
        assert function(t, rho, **critical) == function(t, rho)
        assert function(t, 0, **critical) == function(t, 0)


def test_air_tiny_critical_excess_has_continuous_zero_limit():
    rho, t = 10400, 132.64
    critical = {
        "Cp": 2000,
        "Cv": 35,
        "mu": 1.8e-5,
        "drho_dP": 1e-18 * 10447.7**2 / (3.78502e6 * rho),
        "drho_dP_reference": 0,
    }
    result = air_conductivity_lemmon(t, rho, **critical)
    smaller = air_conductivity_lemmon(t, rho, **{**critical, "drho_dP": critical["drho_dP"] * 1e-4})
    baseline = air_conductivity_lemmon(t, rho)
    assert 0 < smaller - baseline < 0.02 * (result - baseline)
    assert air_conductivity_lemmon(132.64, 10400, **AIR) == pytest.approx(0.07562307234760)
