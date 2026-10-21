"""Native input semantics, branch behavior and original consumer integration."""

import json
from dataclasses import replace

import pytest

from chemworld.data.viscosity import VISCOSITY_SOURCES, ViscosityDatabase
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import FluidState, PipeSpec, pipe_pressure_drop
from chemworld.physchem.transport_properties import wilke_gas_mixture_viscosity_ledger
from chemworld.physchem.viscosity_kernels import viscosity_derivatives, viscosity_value
from chemworld.physchem.viscosity_mixtures import wilke_prefactors, wilke_viscosity
from chemworld.physchem.water_transport import (
    water_correlation_length_nm,
    water_viscosity_enhancement,
    water_viscosity_iapws,
)


@pytest.fixture
def database(tmp_path):
    rows = {
        "VN2_VISCOSITY": ("CAS\tName\tA\tB\tTmin\tTmax\n"
                          "71-36-3\t1-butanol\t-5.9719\t1007\t220\t390\n"
                          "74-82-8\tmissing\t\t1007\t220\t390\n"),
        "VDI_LIQUID_VISCOSITY": ("CAS\tName\tA\tB\tC\tD\tE\n"
                                "71-36-3\tsynthetic butanol fit\t1\t1\t500\t40\t0.0001\n"),
    }
    for source in VISCOSITY_SOURCES:
        if source.source_id in rows:
            path = tmp_path / source.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rows[source.source_id], encoding="utf-8")
    return ViscosityDatabase(tmp_path)


def test_source_scale_identity_laziness_and_original_card_roundtrip(database):
    assert database.loaded_sources == () and len(database.catalog) == 8
    component = ComponentSpec("butanol", "C4H10O", cas_number="71-36-3")
    card = database.property_input("71-36-3", component, method="VN2_VISCOSITY")
    copied = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert card.to_dict() == copied.to_dict()
    state = copied.evaluate_state(temperature_K=348.15, pressure_Pa=101325,
                                 phase="liquid", quantities=("viscosity",))
    assert state.properties["viscosity"].value == pytest.approx(.000459836869568295)
    curve = copied.package.correlations[0]
    assert curve.metadata["source_record"]["fields"]["A"] == "-5.9719"
    assert correlation_temperature_derivative(curve, temperature_K=348.15).to("mPa*s/K").value < 0
    with pytest.raises(ValueError, match="identity"):
        database.property_input("71-36-3", replace(component, cas_number="74-82-8"),
                                method="VN2_VISCOSITY")
    with pytest.raises(ValueError, match="missing"):
        database.viscosity("74-82-8", method="VN2_VISCOSITY")


def test_unknown_source_interval_is_explicit_and_never_widens_known_range(database):
    with pytest.raises(ValueError, match="explicit"):
        database.viscosity("71-36-3", method="VDI_LIQUID_VISCOSITY")
    curve = database.viscosity("71-36-3", method="VDI_LIQUID_VISCOSITY",
                               temperature_bounds_K=(250, 350))
    assert curve.metadata["source_bounds_K"] is None
    assert curve.metadata["temperature_range_basis"] == "caller_application"
    with pytest.raises(ValueError, match="exceeds"):
        database.viscosity("71-36-3", method="VN2_VISCOSITY", temperature_bounds_K=(200, 400))
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=400, validity_policy="raise")


def test_source_viscosity_changes_existing_laminar_pipe_pressure_drop(database):
    curve = database.viscosity("71-36-3", method="VN2_VISCOSITY")
    pipe = PipeSpec(length_m=2, diameter_m=.01)
    results = []
    for temperature in (280, 350):
        mu = evaluate_correlation(curve, temperature_K=temperature).value
        flow = pipe_pressure_drop(pipe, FluidState(810, mu), volumetric_flow_m3_s=1e-7)
        results.append((mu, flow.pressure_drop_friction_Pa))
    assert results[0][1] > results[1][1]
    assert results[0][1] / results[1][1] == pytest.approx(results[0][0] / results[1][0])


def test_ppds9_real_cube_root_value_at_endpoint_and_derivative_rejection():
    p = {"A": 1, "B": 1, "C": 500, "D": 40, "E": .0001}
    curve = PropertyCorrelation("ppds9", "viscosity_ppds9", p, {"temperature": "K"},
                                "Pa*s", "liquid_viscosity", {"temperature": (250, 600)},
                                "Synthetic pole and real-root test")
    assert evaluate_correlation(curve, temperature_K=500).value == .0001
    assert evaluate_correlation(curve, temperature_K=550).value > 0
    with pytest.raises(ValueError, match="T=C"):
        correlation_temperature_derivative(curve, temperature_K=500)
    with pytest.raises(ValueError, match="pole"):
        viscosity_value("PPDS9", 40, **p)


@pytest.mark.parametrize("method,t,p", [
    ("PPDS5", 0, {"Tc": 400, "a0": 1e-5, "a1": .2, "a2": .8}),
    ("mu_air_lemmon", 300, {"rho": -1}),
    ("mu_Yaws", 300, {"A": 1, "B": 2, "C": 3}),
    ("Yoon_Thodos", 300, {"Tc": 400, "Pc": float("nan"), "MW": 40}),
    ("viscosity_gas_Gharagheizi", 50, {"Tc": 400, "Pc": 1e6, "MW": 40}),
])
def test_invalid_or_undeclared_inputs_are_not_silently_repaired(method, t, p):
    with pytest.raises(ValueError):
        viscosity_value(method, t, **p)


def test_lemmon_molar_density_is_used_and_stiel_branch_derivative_is_not_invented():
    dilute = viscosity_value("mu_air_lemmon", 300, rho=0)
    dense = viscosity_value("mu_air_lemmon", 300, rho=10000)
    assert dense > dilute
    with pytest.raises(ValueError, match="branch"):
        viscosity_derivatives("Stiel_Thodos", 600, {"Tc": 400, "Pc": 5e6, "MW": 40})


def test_streamed_and_precomputed_wilke_share_original_ledger_and_zero_fraction_limit():
    x, mus, mw = (.05, .95), (1.34e-5, 9.5029e-6), (64.06, 46.07)
    direct = wilke_viscosity(x, mus, mw)
    assert direct == pytest.approx(wilke_viscosity(x, mus, factors=wilke_prefactors(mw)))
    ledger = wilke_gas_mixture_viscosity_ledger(
        component_mole_fractions=dict(zip(("a", "b"), x, strict=True)),
        component_viscosities_Pa_s=dict(zip(("a", "b"), mus, strict=True)),
        component_molecular_weights_g_mol=dict(zip(("a", "b"), mw, strict=True)),
    )
    assert ledger.mixture_value == direct
    assert sum(c["partial_viscosity_Pa_s"] for c in ledger.contributions.values()) == direct
    assert wilke_viscosity((0, 1), mus, mw) == pytest.approx(mus[1])
    assert wilke_viscosity(x[::-1], mus[::-1], mw[::-1]) == pytest.approx(direct)
    for bad in ((.5, .6), (float("nan"), 0), (2, -1)):
        with pytest.raises(ValueError):
            wilke_viscosity(bad, mus, mw)


def test_water_critical_inputs_are_explicit_and_no_eos_is_invented():
    background = water_viscosity_iapws(647.35, 322)
    enhanced = water_viscosity_iapws(647.35, 322, drho_dP=.01213641949033,
                                    reference_approximation=True)
    assert enhanced > background
    assert viscosity_value("mu_IAPWS", 647.35, rho=322) == background
    assert water_viscosity_iapws(300, 0) > 0
    assert water_correlation_length_nm(300, 998, 1e-6, 1e-5) == 0
    assert water_viscosity_enhancement(0) == 1
    for kwargs in ({"drho_dP": .001}, {"drho_dP_reference": .001},
                   {"reference_approximation": True}, {"reference_approximation": 1},
                   {"drho_dP": -.001, "reference_approximation": True},
                   {"drho_dP": .001, "drho_dP_reference": .001,
                    "reference_approximation": True}):
        with pytest.raises(ValueError):
            water_viscosity_iapws(647.35, 322, **kwargs)


def test_water_enhancement_is_continuous_at_both_analytic_branches():
    for length in (.3817016416, 1.9):
        left, center, right = [water_viscosity_enhancement(length + d)
                               for d in (-1e-10, 0, 1e-10)]
        assert left == pytest.approx(center, abs=1e-10)
        assert right == pytest.approx(center, abs=1e-10)
