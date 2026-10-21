"""Full conversion boundaries and real native consumers of viscosity models."""

import json
from dataclasses import replace

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.dense_viscosity import (
    lucas_liquid_viscosity_report,
    przedziecki_sridhar_temperature_viscosity,
)
from chemworld.physchem.petroleum_viscosity import (
    dynamic_viscosity,
    kinematic_viscosity,
    twu_1985_kinematic_viscosity,
    twu_1985_viscosity,
    viscosity_index,
)
from chemworld.physchem.property_equations import correlation_temperature_derivative
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import (
    ComponentSpec,
    ComponentUncertainty,
    MixtureSpec,
    PropertyCorrelation,
)
from chemworld.physchem.transport import FluidState, PipeSpec, pipe_pressure_drop
from chemworld.physchem.transport_properties import transport_property_report
from chemworld.physchem.viscosity_kernels import viscosity_derivatives, viscosity_value
from chemworld.physchem.viscosity_mixtures import (
    dense_mixture_viscosity_ledger,
    gas_mixture_viscosity_ledger,
)
from chemworld.physchem.viscosity_scale_data import VISCOSITY_LINEAR_SCALES, VISCOSITY_SCALE_TABLES
from chemworld.physchem.viscosity_scales import VISCOSITY_SCALES, viscosity_converter


@pytest.mark.parametrize("name", [n for n in VISCOSITY_SCALES if n != "kinematic viscosity"])
def test_full_scale_domains_and_explicit_extrapolation(name):
    low = (
        min(VISCOSITY_SCALE_TABLES[name][0])
        if name in VISCOSITY_SCALE_TABLES
        else VISCOSITY_LINEAR_SCALES[name][1]
    )
    with pytest.raises(ValueError, match="outside valid interval"):
        viscosity_converter(0.99 * low, name, "kinematic viscosity")
    outside_si = viscosity_converter(0.99 * low, name, "kinematic viscosity", extrapolate=True)
    assert outside_si > 0
    assert viscosity_converter(
        outside_si, "kinematic viscosity", name, extrapolate=True
    ) == pytest.approx(0.99 * low)
    if name in VISCOSITY_SCALE_TABLES:
        high = max(VISCOSITY_SCALE_TABLES[name][0])
        with pytest.raises(ValueError, match="outside valid interval"):
            viscosity_converter(1.01 * high, name, "kinematic viscosity")
        outside_si = viscosity_converter(1.01 * high, name, "kinematic viscosity", extrapolate=True)
        assert outside_si > 0
        assert viscosity_converter(
            outside_si, "kinematic viscosity", name, extrapolate=True
        ) == pytest.approx(1.01 * high)


def test_barbey_default_is_reversible_and_reference_curve_explicitly_retains_overshoot():
    value = (1.4 * 2.5) ** 0.5
    nu = viscosity_converter(value, "Barbey degrees", "kinematic viscosity")
    assert 3300e-6 < nu < 4400e-6
    assert viscosity_converter(nu, "kinematic viscosity", "barbey") == pytest.approx(value)
    historical = viscosity_converter(
        value, "barbey", "kinematic viscosity", interpolation="reference_cubic"
    )
    assert historical > 4400e-6
    with pytest.raises(ValueError, match="outside valid interval"):
        viscosity_converter(
            historical, "kinematic viscosity", "barbey", interpolation="reference_cubic"
        )
    assert viscosity_converter(8.79, "Engler degrees", "Parlin cup #7 seconds") == pytest.approx(
        52.5
    )
    for kwargs in (
        {"old_scale": "missing"},
        {"value": -1},
        {"interpolation": "linear"},
        {"extrapolate": 1},
        {"value": float("inf")},
    ):
        with pytest.raises(ValueError):
            viscosity_converter(
                **{"value": 10, "old_scale": "engler", "new_scale": "kinematic viscosity", **kwargs}
            )


def test_index_kinematic_units_and_standard_integer_rounding():
    assert viscosity_index(73.3e-6, 8.86e-6, rounding=True) == 92
    assert viscosity_index(6.394e-6, 2e-6) == pytest.approx(100)
    assert viscosity_index(6.394e-6, 2e-6, rounding=True) == 100
    assert viscosity_index(10e-6, 1.999e-6) is None
    with pytest.raises(ValueError):
        viscosity_index(1e-5, -1)


def test_twu_characterization_density_is_separate_from_state_density():
    t, tb, rho = 338.7055, 672.3166, 895.5189
    nu = twu_1985_kinematic_viscosity(t, tb, rho)
    assert twu_1985_viscosity(t, tb, rho) == pytest.approx(0.008235009644854494)
    assert twu_1985_viscosity(t, tb, rho, density_at_T_kg_m3=800) == nu * 800
    assert kinematic_viscosity(dynamic_viscosity(nu, 800), 800) == pytest.approx(nu)
    for density in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            dynamic_viscosity(nu, density)


def curve(method, coefficients, low=250, high=550):
    return PropertyCorrelation(
        "synthetic-" + method,
        "viscosity_" + method.lower(),
        coefficients,
        {"temperature": "K"},
        "Pa*s",
        "liquid_viscosity",
        {"temperature": (low, high)},
        "Synthetic equation / native consumer test",
    )


def pressure_drop(mu, rho=800):
    return pipe_pressure_drop(
        PipeSpec(length_m=2, diameter_m=0.01), FluidState(rho, mu), volumetric_flow_m3_s=1e-9
    ).pressure_drop_friction_Pa


@pytest.mark.parametrize(
    "method,p",
    [
        ("Twu_1985", {"Tb": 672.3166, "rho_60F": 895.5189, "rho_at_T": 800}),
        (
            "Przedziecki_Sridhar",
            {
                "Tm": 178,
                "Tc": 591.8,
                "Pc": 41e5,
                "Vc": 316e-6,
                "Vf": 95e-6,
                "omega": 0.263,
                "MW": 92.14,
            },
        ),
    ],
)
def test_temperature_estimates_use_original_property_card_json_package_and_pipe(method, p):
    from chemworld.physchem.property_packages import ComponentPropertyPackage

    correlation = curve(method, p)
    package = ComponentPropertyPackage(ComponentSpec("fraction", "C7H8"), (correlation,))
    source = InputSource(
        DataSourceProvenance("synthetic", 0, "Synthetic viscosity input", "1", "CC0-1.0"),
        "synthetic",
    )
    card = PropertyInputCard(
        package,
        (source,),
        {
            correlation.correlation_id: ComponentUncertainty(
                correlation.correlation_id,
                unit="Pa*s",
                source_id="synthetic",
                coverage="synthetic",
                note="Synthetic inputs; predictive uncertainty not established",
            )
        },
        None,
        "No caloric reference",
    )
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    mus = [
        restored.evaluate_state(
            temperature_K=t, pressure_Pa=101325, phase="liquid", quantities=("viscosity",)
        )
        .properties["viscosity"]
        .value
        for t in (300, 450)
    ]
    assert mus[0] > mus[1]
    assert pressure_drop(mus[0]) / pressure_drop(mus[1]) == pytest.approx(mus[0] / mus[1])
    assert correlation_temperature_derivative(correlation, temperature_K=383).value < 0


def test_compressed_lucas_consumes_pressure_on_original_report_and_pipe():
    correlation = curve("mu_TDE", {"A": -8, "B": 0, "C": 0, "D": 0})
    saturated = transport_property_report(
        correlation, temperature_K=300, pressure_Pa=2e5, phase="liquid"
    )
    kwargs = {
        "temperature_K": 300,
        "critical_temperature_K": 572.2,
        "critical_pressure_Pa": 3.47e6,
        "acentric_factor": 0.236,
        "saturation_pressure_Pa": 2e5,
    }
    low = lucas_liquid_viscosity_report(saturated, pressure_Pa=2e5, **kwargs)
    high = lucas_liquid_viscosity_report(saturated, pressure_Pa=5e7, **kwargs)
    assert low.evaluation.value == saturated.evaluation.value
    assert high.evaluation.value > low.evaluation.value
    assert pressure_drop(high.evaluation.value) > pressure_drop(low.evaluation.value)
    assert json.loads(json.dumps(high.to_dict()))["property"]["inputs"]["pressure"] == 5e7
    for changes in ({"temperature_K": 301}, {"saturation_pressure_Pa": 1e5}):
        with pytest.raises(ValueError):
            lucas_liquid_viscosity_report(saturated, pressure_Pa=5e7, **{**kwargs, **changes})


def test_mixture_methods_use_existing_state_ledger_and_flow():
    mixture = MixtureSpec.from_mole_fractions(
        (
            ComponentSpec("a", "SO2", default_phase="gas"),
            ComponentSpec("b", "C2H6O", default_phase="gas"),
        ),
        (0.05, 0.95),
        phase_label="gas",
        temperature_K=308.2,
        pressure_Pa=101325,
    )
    mus = (1.34e-5, 9.5029e-6)
    for method in ("Wilke", "Herning_Zipperer", "Brokaw"):
        kwargs = (
            {"polar_parameters": (0.42, 0.19), "epsilon_k_K": (347, 432)}
            if method == "Brokaw"
            else {}
        )
        ledger = gas_mixture_viscosity_ledger(mixture, mus, method=method, **kwargs)
        changed = gas_mixture_viscosity_ledger(
            replace(mixture, mole_fractions=(0.9, 0.1)), mus, method=method, **kwargs
        )
        assert ledger.mixture_value != changed.mixture_value
        assert pressure_drop(ledger.mixture_value, 2) != pressure_drop(changed.mixture_value, 2)
        assert json.loads(json.dumps(ledger.to_dict()))["method_family"] == method
    with pytest.raises(ValueError, match="unknown"):
        gas_mixture_viscosity_ledger(mixture, mus, method="automatic")
    with pytest.raises(ValueError, match="explicit"):
        gas_mixture_viscosity_ledger(mixture, mus, method="Brokaw")


def test_lbc_volume_and_mixture_state_are_real_inputs_to_original_ledger():
    components = tuple(
        ComponentSpec(k, f, default_phase="gas")
        for k, f in (("a", "CH4"), ("b", "C2H6"), ("c", "C3H8"))
    )
    mixture = MixtureSpec.from_mole_fractions(
        components, (0.4, 0.3, 0.3), phase_label="gas", temperature_K=300, pressure_Pa=1e6
    )
    kwargs = {
        "critical_temperatures_K": (190.564, 305.32, 369.83),
        "critical_pressures_Pa": (4599000, 4872000, 4248000),
        "critical_volumes_m3_mol": (9.86e-5, 0.0001455, 0.0002),
    }
    low, high = [dense_mixture_viscosity_ledger(mixture, v, **kwargs) for v in (0.0023025, 0.00015)]
    assert high.mixture_value > low.mixture_value
    assert pressure_drop(high.mixture_value) > pressure_drop(low.mixture_value)
    assert (
        json.loads(json.dumps(high.to_dict()))["contributions"]["a"]["molar_volume_m3_mol"]
        == 0.00015
    )
    with pytest.raises(ValueError):
        dense_mixture_viscosity_ledger(mixture, 0, **kwargs)


def test_explicit_limiting_policy_and_branch_derivatives():
    p = {"Tc": 500, "Pc": 5e6, "MW": 40}
    with pytest.raises(ValueError):
        viscosity_value("viscosity_gas_Gharagheizi", 50, **p)
    p["low_temperature_limit"] = 1
    assert viscosity_derivatives("viscosity_gas_Gharagheizi", 50, p)[1] == 0
    with pytest.raises(ValueError, match="branch"):
        viscosity_derivatives("viscosity_gas_Gharagheizi", 100, p)
    with pytest.raises(ValueError):
        viscosity_value("viscosity_gas_Gharagheizi", 50, **{**p, "low_temperature_limit": 2})
    with pytest.raises(ValueError, match="T >= Tm"):
        przedziecki_sridhar_temperature_viscosity(
            170,
            melting_volume_m3_mol=95e-6,
            melting_temperature_K=178,
            critical_temperature_K=591.8,
            critical_pressure_Pa=41e5,
            critical_volume_m3_mol=316e-6,
            acentric_factor=0.263,
            molecular_weight_g_mol=92.14,
        )
