"""Real mixture state, pressure, identity, source and boundary consumption."""
# ruff: noqa: C408 -- scientific inputs

import json
from dataclasses import replace

import pytest

from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    MixtureSpec,
    PropertyCorrelation,
    evaluate_correlation,
    evaluate_property_state,
    mixture_volume_composition_jacobian,
    mixture_volume_correlation,
    mixture_volume_derivatives,
    mixture_volume_evaluation,
    mixture_volume_value,
    volume_correlation,
    volume_derivatives,
    volume_parameter_jacobian,
    volume_value,
)

COMPONENTS = (ComponentSpec("methanol", "CH4O"), ComponentSpec("water", "H2O"))
DATA = {
    "methanol": dict(Tcs=512.58, Vcs=0.000117, omegas=0.559),
    "water": dict(Tcs=647.29, Vcs=5.6e-5, omegas=0.344),
}


def mixture(xs=(0.4576, 0.5424), pressure=1e7):
    return MixtureSpec.from_mole_fractions(COMPONENTS, xs, phase_label="liquid",
                                           temperature_K=298, pressure_Pa=pressure)


def curve(state=None, output="molar_volume"):
    return mixture_volume_correlation(
        state or mixture(), method="COSTALD_mixture_compressed", component_parameters=DATA,
        saturation_pressure_model="generalized_riedel", temperature_range_K=(280, 350),
        pressure_range_Pa=(1e7, 1e8), source_note="public synthetic qualification", output=output)


@pytest.mark.parametrize("output", ("molar_volume", "density"))
def test_fixed_composition_curve_json_and_real_multi_component_package_state(output):
    model = PropertyCorrelation.from_dict(json.loads(json.dumps(curve(output=output).to_dict())))
    packages = tuple(ComponentPropertyPackage(c, ()) for c in COMPONENTS)
    reports = [evaluate_property_state(
        packages, dict(zip(mixture().component_ids, mixture().mole_fractions, strict=True)),
        temperature_K=298, pressure_Pa=p, phase="liquid", quantities=("density", "molar_volume"),
        volume_model=model) for p in (1e7, 1e8)]
    assert reports[1].properties["density"].value > reports[0].properties["density"].value
    assert reports[1].properties["molar_volume"].value < reports[0].properties["molar_volume"].value
    assert reports[0].properties["density"].value * reports[0].properties["molar_volume"].value == (
        pytest.approx(mixture().average_molecular_weight_g_mol / 1000))
    with pytest.raises(ValueError, match="composition"):
        evaluate_property_state(packages, {"methanol": 0.2, "water": 0.8},
            temperature_K=298, pressure_Pa=1e7, phase="liquid", quantities=("density",),
            volume_model=model)
    with pytest.raises(ValueError, match="EOS"):
        evaluate_property_state(packages, {"methanol": 0.2, "water": 0.8},
            temperature_K=298, pressure_Pa=1e7, phase="gas", quantities=("density",),
            volume_model=model)


def test_composition_is_consumed_and_parameter_keys_are_bound_to_identities():
    values = [mixture_volume_evaluation(mixture(xs), method="COSTALD_mixture",
               component_parameters=DATA).value for xs in ((0.2, 0.8), (0.8, 0.2))]
    assert values[0] != pytest.approx(values[1])
    reversed_state = MixtureSpec.from_mole_fractions(
        COMPONENTS[::-1], (0.5424, 0.4576), phase_label="liquid", temperature_K=298,
        pressure_Pa=1e7)
    a = mixture_volume_evaluation(mixture(), method="COSTALD_mixture", component_parameters=DATA)
    b = mixture_volume_evaluation(reversed_state, method="COSTALD_mixture",
                                 component_parameters=DATA)
    assert a.value == pytest.approx(b.value)
    with pytest.raises(ValueError, match="identities"):
        mixture_volume_evaluation(mixture(), method="COSTALD_mixture",
                                 component_parameters={"methanol": DATA["methanol"]})
    with pytest.raises(ValueError, match="same composition"):
        mixture_volume_evaluation(replace(mixture(), mass_fractions=(0.5, 0.5)),
                                 method="COSTALD_mixture", component_parameters=DATA)


def test_amgat_point_inputs_reuse_existing_ledger_without_inventing_temperature_dependence():
    from chemworld.physchem import mixture_molar_volume_ledger

    state = mixture()
    params = {"methanol": {"Vms": 4.057e-5}, "water": {"Vms": 5.861e-5}}
    direct = mixture_volume_evaluation(state, method="Amgat", component_parameters=params)
    ledger = mixture_molar_volume_ledger(
        component_mole_fractions=dict(zip(state.component_ids, state.mole_fractions, strict=True)),
        component_molar_volumes_m3_mol={k: v["Vms"] for k, v in params.items()})
    assert direct.value == ledger.mixture_molar_volume_m3_mol
    with pytest.raises(ValueError, match="point volumes"):
        mixture_volume_correlation(state, method="Amgat", component_parameters=params,
                                   temperature_range_K=(280, 350), source_note="point values")


@pytest.mark.parametrize("method", ("Rackett_mixture", "COSTALD_mixture",
                                    "COSTALD_mixture_compressed"))
def test_single_component_limit_and_zero_fraction(method):
    p = dict(Tcs=(500, 600), Vcs=(0.0002, 0.0003), omegas=(0.2, 0.3))
    pure, pure_p = "COSTALD", dict(Tc=500, Vc=0.0002, omega=0.2)
    if method == "Rackett_mixture":
        p = dict(Tcs=(500, 600), Pcs=(4e6, 5e6), Zrs=(0.27, 0.28), MWs=(50, 80))
        pure, pure_p = "Rackett", dict(Tc=500, Pc=4e6, Zc=0.27)
    elif method.endswith("compressed"):
        p.update(P=1e7, Psat=1e5)
        pure = "COSTALD_compressed"
        pure_p.update(P=1e7, Psat=1e5, estimate_saturation=1,
                      Pc=(0.291 - 0.080 * 0.2) * 8.31446261815324 * 500 / 0.0002)
    assert mixture_volume_value(method, 300, (1, 0), **p) == pytest.approx(
        volume_value(pure, 300, **pure_p))


@pytest.mark.parametrize("method,parameters", (
    ("Tait", dict(P_ref=1e5, rho_ref=800, B=1e8, C=0.1)),
    ("Tait_molar", dict(P_ref=1e5, V_ref=1e-4, B=1e8, C=0.1)),
    ("COSTALD_compressed", dict(Psat=1e5, Vs=1e-4, Tc=500, Pc=4e6, omega=0.2)),
))
def test_pressure_is_not_clipped_and_derivatives_hold_reference_inputs(method, parameters):
    with pytest.raises(ValueError, match="P >="):
        volume_value(method, 300, P=9e4, **parameters)
    base = volume_value(method, 300, P=1e5, **parameters)
    high = volume_value(method, 300, P=1e8, **parameters)
    assert high > base if method == "Tait" else high < base
    slope = volume_parameter_jacobian(method, 300, dict(P=1e7, **parameters), names=("P",))["P"]
    assert slope > 0 if method == "Tait" else slope < 0
    if method.startswith("Tait"):
        assert volume_derivatives(method, 300, dict(P=1e7, **parameters), order=3)[1:] == (0, 0, 0)


def test_tait_log_domain_positive_volume_and_explicit_molecular_weight():
    for kwargs in (dict(B=-2e5, C=0.1), dict(B=1e8, C=10)):
        with pytest.raises(ValueError, match="positive"):
            volume_value("Tait_molar", 300, P=1e8, P_ref=1e5, V_ref=1e-4, **kwargs)
    with pytest.raises(ValueError, match="MW"):
        volume_value("Tait", 300, P=1e7, P_ref=1e5, rho_ref=800, B=1e8, C=0.1,
                     output="molar_volume")


def test_saturation_estimators_are_explicit_and_never_silently_replace_supplied_values():
    p = dict(Tcs=(500, 600), Vcs=(0.0002, 0.0003), omegas=(0.2, 0.3), P=1e7)
    with pytest.raises(ValueError, match="explicitly"):
        mixture_volume_value("COSTALD_mixture_compressed", 300, (0.5, 0.5), **p)
    with pytest.raises(ValueError, match="exclusively"):
        mixture_volume_value("COSTALD_mixture_compressed", 300, (0.5, 0.5), **p,
                             Psat=1e5, saturation_pressure_model="generalized_riedel")
    with pytest.raises(ValueError, match="explicit"):
        volume_value("COSTALD_compressed", 300, Tc=500, Pc=4e6, omega=0.2, P=1e7)
    with pytest.raises(ValueError, match="T < Tc"):
        volume_value("COSTALD_compressed", 500, Tc=500, Pc=4e6, omega=0.2, P=1e7,
                     Psat=1e5, Vs=1e-4)


@pytest.mark.parametrize("xs,params", (
    ((0.5, 0.4), dict(Vms=(1e-4, 2e-4))),
    ((1, 0), dict(Vms=(1e-4, float("nan")))),
    ((1, 0), dict(Vms=(1e-4,))),
    ((1, 0), dict(Vms=(1e-4, 2e-4), P=1e5)),
    ((1.1, -0.1), dict(Vms=(1e-4, 2e-4))),
))
def test_invalid_mixture_inputs_are_rejected(xs, params):
    with pytest.raises(ValueError):
        mixture_volume_value("Amgat", 300, xs, **params)


def test_pressure_derivatives_are_not_assigned_to_saturation_curves():
    with pytest.raises(ValueError, match="compressed"):
        mixture_volume_derivatives("Amgat", 300, (1,), dict(Vms=(1e-4,)), variable="pressure")
    assert mixture_volume_composition_jacobian("Amgat", 300, (0.4, 0.6),
                                              dict(Vms=(1e-4, 2e-4))) == (-1e-4,)


def test_pure_compressed_input_card_roundtrip_consumes_pressure():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem import ComponentUncertainty
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.physchem.property_inputs import PropertyInputCard

    c = volume_correlation("Tait", parameters=dict(P_ref=1e5, rho_ref=800, B=1e8, C=0.1),
        temperature_range_K=(280, 320), pressure_range_Pa=(1e5, 1e8), output="density",
        source_note="fixed synthetic Tait parameters")
    card = PropertyInputCard(ComponentPropertyPackage(COMPONENTS[0], (c,)),
        (InputSource(DataSourceProvenance("tait", 1, "synthetic", "equation", "public"),
                     "literature"),),
        {c.correlation_id: ComponentUncertainty(c.correlation_id, unit="kg/m^3",
           coverage="synthetic", source_id="tait", note="not supplied")}, None, "no caloric datum")
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    values = [restored.evaluate_state(temperature_K=300, pressure_Pa=p, phase="liquid",
              quantities=("density",)).properties["density"].value for p in (1e5, 1e8)]
    assert values[1] > values[0] == 800
    with pytest.raises(ValueError, match="pressure"):
        evaluate_correlation(c, temperature_K=300)
