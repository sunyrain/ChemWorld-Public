"""Critical estimates must reach native consumers with identity and units intact."""

import json

import pytest

from chemworld.data.constant_catalog import CONSTANT_SOURCES
from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.critical_kernels import (
    acentric_factor_from_pressure,
    critical_surface,
    mersmann_kind_volume,
    mixture_critical_property,
    stiel_polar_factor,
)
from chemworld.physchem.critical_properties import (
    CriticalEstimate,
    acentric_estimate_from_correlation,
    eos_component_from_critical_inputs,
    estimate_critical_property,
    mixture_critical_estimate,
)
from chemworld.physchem.eos import CubicEOSSpec, evaluate_cubic_eos
from chemworld.physchem.specs import ComponentSpec, MixtureSpec, PropertyCorrelation


def supplied(value, unit="K"):
    return InputValue(value, unit, None, "explicit synthetic test input")


@pytest.mark.parametrize("model", ["peng_robinson", "srk"])
def test_estimated_inputs_drive_existing_eos_and_roundtrip(model):
    component = ComponentSpec("decane", "C10H22")
    Tc = estimate_critical_property(component, method="Tb_Tc_relationship",
                                    parameters={"Tb": 447.3, "fit": "Perry8E"})
    Vc = estimate_critical_property(component, method="Mersmann_Kind", parameters={})
    Pc = estimate_critical_property(component, method="Ihmels", parameters={"Tc": Tc, "Vc": Vc})
    omega = estimate_critical_property(component, method="LK_omega",
                                       parameters={"Tb": 447.3, "Tc": Tc, "Pc": Pc})
    recovered = [CriticalEstimate.from_dict(json.loads(json.dumps(r.to_dict())))
                 for r in (Tc, Vc, Pc, omega)]
    assert recovered == [Tc, Vc, Pc, omega]
    spec = eos_component_from_critical_inputs(component, Tc=Tc, Pc=Pc, omega=omega)
    eos = CubicEOSSpec("estimated", model=model, components=(spec,))
    state = evaluate_cubic_eos(eos, {"decane": 1}, temperature_K=800, pressure_Pa=1e6)
    changed = eos_component_from_critical_inputs(component, Tc=Tc, Pc=Pc,
                                                 omega=supplied(0, "dimensionless"))
    other = evaluate_cubic_eos(CubicEOSSpec("changed", model=model, components=(changed,)),
                               {"decane": 1},
                               temperature_K=800, pressure_Pa=1e6)
    assert state.compressibility_factor != pytest.approx(other.compressibility_factor)
    assert state.molar_volume_m3_mol > 0
    assert "estimate" in omega.input_value().uncertainty_note
    assert Pc.inputs["Tc"]["method"] == "Tb_Tc_relationship"


def test_real_constant_query_consumer_identity_units_and_missing(tmp_path):
    source = next(s for s in CONSTANT_SOURCES if s.source_id == "HEOS")
    path = tmp_path / source.relative_path
    path.parent.mkdir(parents=True)
    path.write_text("CAS\tname\tTc\tPc\tomega\n7732185\tWater\t647.096\t22064000\t0.3443\n"
                    "64-17-5\tEthanol\t514.71\t6268000\t\n", encoding="utf-8")
    db = ConstantDatabase(tmp_path)
    water = ComponentSpec("water", "H2O", cas_number="7732-18-5")
    values = {key: db.value(water.cas_number, key, method="HEOS") for key in ("Tc", "Pc", "omega")}
    spec = eos_component_from_critical_inputs(water, **values)
    assert spec.critical_pressure_Pa == 22064000
    with pytest.raises(ValueError, match="identity"):
        eos_component_from_critical_inputs(ComponentSpec("ethanol", "C2H6O"), **values)
    with pytest.raises(ValueError, match="field"):
        eos_component_from_critical_inputs(water, **{**values, "omega": values["Tc"]})
    ethanol = ComponentSpec("ethanol", "C2H6O", cas_number="64-17-5")
    missing = db.value("64-17-5", "omega", method="HEOS")
    with pytest.raises(ValueError, match="missing"):
        eos_component_from_critical_inputs(ethanol, Tc=supplied(500),
                                           Pc=supplied(50, "bar"), omega=missing)
    estimate = estimate_critical_property(water, method="Ihmels",
                                          parameters={"Tc": values["Tc"], "Pc": values["Pc"]})
    assert estimate.inputs["Tc"]["source"]["source_id"] == "chemicals-constants:HEOS"


@pytest.mark.parametrize("method", ["omega_definition", "Stiel_polar_factor"])
def test_psat_curve_evaluates_required_temperature_and_retains_sources(method):
    component = ComponentSpec("example", "C2H6")
    curve = PropertyCorrelation(
        "example-pressure", "lee_kesler_pressure", {"Tc": 600, "Pc": 3e6, "omega": .2},
        {"temperature": "K"}, "Pa", "vapor_pressure",
        validity_ranges={"temperature": (300, 590)},
    )
    params = {"omega": supplied(.2, "dimensionless")} if method == "Stiel_polar_factor" else {}
    report = acentric_estimate_from_correlation(component, curve, Tc=supplied(600),
                                               Pc=supplied(30, "bar"), method=method, **params)
    assert report.inputs["reduced_temperature"] == (.7 if method == "omega_definition" else .6)
    assert report.inputs["Pc_source"]["unit"] == "bar"
    assert report.inputs["curve"]["correlation_id"] == "example-pressure"
    assert report.unit == "dimensionless"
    with pytest.raises(ValueError):
        acentric_estimate_from_correlation(component, curve, Tc=supplied(1000),
                                           Pc=supplied(30, "bar"), method=method, **params)


@pytest.mark.parametrize("method", ["Li", "Chueh_Prausnitz_Tc", "Chueh_Prausnitz_Vc",
                                    "Grieves_Thodos", "modified_Wilson_Tc", "modified_Wilson_Vc"])
def test_mixture_spec_consumes_all_rules_and_order_is_preserved(method):
    components = (ComponentSpec("b", "C4H10"), ComponentSpec("a", "C5H12"))
    mix = MixtureSpec.from_mole_fractions(components, (.3, .7), phase_label="liquid",
                                         temperature_K=300, pressure_Pa=1e5)
    fields = ({"Tc", "Vc"} if method in {"Li", "Chueh_Prausnitz_Tc"}
              else {"Vc"} if method.endswith("_Vc") else {"Tc"})
    inputs = {c.identifier: {key: supplied((425.12, 469.7)[i], "K") if key == "Tc"
                             else supplied((.000255, .000313)[i], "m^3/mol") for key in fields}
              for i, c in enumerate(components)}
    matrix = None if method == "Li" else [[0, 1.1], [.9, 0]]
    report = mixture_critical_estimate(mix, method=method, critical_inputs=inputs,
                                       interactions=matrix)
    assert report.subject_ids == ("b", "a")
    assert report.kind == "pseudocritical_estimate"
    assert report.value > 0
    assert CriticalEstimate.from_dict(json.loads(json.dumps(report.to_dict()))) == report
    with pytest.raises(ValueError, match="identity"):
        eos_component_from_critical_inputs(components[0], Tc=report,
                                           Pc=supplied(3e6, "Pa"),
                                           omega=supplied(.2, "dimensionless"))
    with pytest.raises(ValueError, match="identities"):
        mixture_critical_estimate(mix, method=method, critical_inputs={"a": inputs["a"]},
                                  interactions=matrix)


@pytest.mark.parametrize("method", ["Grieves_Thodos", "modified_Wilson_Tc", "modified_Wilson_Vc"])
def test_zero_fraction_limit_and_singular_active_interaction(method):
    kwargs = {"Tcs": (400, 500), "Vcs": (.0002, .0003), "interactions": [[0, 0], [0, 0]]}
    result = mixture_critical_property(method, (0, 1), **kwargs)
    assert result == (0.0003 if method.endswith("_Vc") else 500)
    kwargs["interactions"] = [[0, -2], [-2, 0]]
    with pytest.raises(ValueError, match="denominator"):
        mixture_critical_property(method, (.5, .5), **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"Tc": 500}, {"Tc": 500, "Pc": 3e6, "Vc": .0002},
                                   {"Tc": float("nan"), "Pc": 3e6}, {"Tc": 500, "Pc": -2}])
def test_critical_surface_rejects_ambiguous_or_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        critical_surface(method="Ihmels", **kwargs)


def test_meissner_conventions_are_explicit_and_never_silently_equated():
    atm = critical_surface(method="Meissner", Tc=851, Vc=.000308)[1]
    mpa = critical_surface(method="Meissner_MPa", Tc=851, Vc=.000308)[1]
    assert atm == pytest.approx(5978445.2)
    assert atm / mpa == pytest.approx(1.01325)
    with pytest.raises(ValueError, match="volume"):
        critical_surface(method="Meissner", Tc=500, Vc=8e-6)
    with pytest.raises(ValueError, match="pressure"):
        critical_surface(method="Grigoras", Tc=500, Pc=290000)


def test_molecular_model_uses_existing_isotope_composition_and_missing_radii():
    light, heavy = ComponentSpec("light", "C2H6O"), ComponentSpec("heavy", "[13C]2D6O")
    reports = [estimate_critical_property(c, method="Mersmann_Kind", parameters={})
               for c in (light, heavy)]
    assert reports[0].value == reports[1].value
    assert reports[1].inputs["formula"] == "[13C]2D6O"
    with pytest.raises(ValueError, match="radius"):
        mersmann_kind_volume({"Xe": 1})
    with pytest.raises(ValueError, match="ionic"):
        estimate_critical_property(ComponentSpec("ion", "NH4+"),
                                    method="Mersmann_Kind", parameters={})


def test_reject_missing_matrix_nonunit_composition_and_wrong_saturation_point():
    with pytest.raises(ValueError, match="matrix"):
        mixture_critical_property("modified_Wilson_Tc", (.4, .6), Tcs=(400, 500))
    with pytest.raises(ValueError, match="sum"):
        mixture_critical_property("Li", (.4, .4), Tcs=(400, 500), Vcs=(.0002, .0003))
    for reduced in (.6, 1, float("nan")):
        with pytest.raises(ValueError):
            acentric_factor_from_pressure(1e5, 3e6, reduced_temperature=reduced)
    with pytest.raises(ValueError, match=r"0\.6"):
        stiel_polar_factor(1e5, 3e6, .2, reduced_temperature=.7)
