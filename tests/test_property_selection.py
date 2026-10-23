import json
from dataclasses import replace

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    PropertySelectionError,
    PropertySelectionPolicy,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentUncertainty


def curve(identity, cp, source, bounds=(250, 400)):
    return PropertyCorrelation(
        identity,
        "cp_polynomial",
        {"a": cp},
        {"temperature": "K"},
        "J/(mol*K)",
        "liquid_heat_capacity",
        {"temperature": bounds},
        "Explicit synthetic heat capacity",
        {"phase": "liquid", "source": {"source_id": source}},
    )


def package():
    return ComponentPropertyPackage(
        ComponentSpec("ethanol", "C2H6O"),
        (curve("first", 20, "a", (250, 350)), curve("second", 40, "b")),
    )


def test_declared_order_and_explicit_selection_change_actual_caloric_state():
    base = package()
    assert base.selection_report("liquid_heat_capacity", temperature_K=300).selected_id == "first"
    assert base.selection_report("liquid_heat_capacity", temperature_K=375).selected_id == "second"
    selected = base.with_selection(
        {"liquid_heat_capacity": PropertySelectionPolicy("correlation", "second")}
    )
    restored = ComponentPropertyPackage.from_dict(json.loads(json.dumps(selected.to_dict())))
    assert restored == selected
    ref = CaloricReference(phase="liquid", temperature_K=300, entropy_J_mol_K=0)
    result = restored.evaluate_state(
        temperature_K=320,
        pressure_Pa=101325,
        phase="liquid",
        quantities=("cp", "enthalpy", "entropy"),
        reference=ref,
    )
    from math import log

    assert result.properties["cp"].value == 40
    assert result.properties["enthalpy"].value == pytest.approx(800)
    assert result.properties["entropy"].value == pytest.approx(40 * log(320 / 300))
    assert restored.selection_report("liquid_heat_capacity", temperature_K=320).to_dict()[
        "candidates"
    ][1]["selected"]


def test_explicit_choice_is_not_replaced_when_outside_domain_or_formula_fails():
    selected = package().with_selection(
        {"liquid_heat_capacity": PropertySelectionPolicy("correlation", "first")}
    )
    with pytest.raises(ValueError, match="outside validity range"):
        selected.evaluate("liquid_heat_capacity", temperature_K=375, validity_policy="raise")
    assert selected.evaluate(
        "liquid_heat_capacity", temperature_K=375, validity_policy="warn"
    ).warnings
    assert (
        selected.evaluate(
            "liquid_heat_capacity", temperature_K=375, validity_policy="ignore"
        ).correlation_id
        == "first"
    )
    invalid = replace(package(), correlations=(curve("bad", -1, "a"), curve("good", 40, "b")))
    with pytest.raises(ValueError, match="negative"):
        invalid.evaluate("liquid_heat_capacity", temperature_K=300)


def test_source_priority_is_explicit_shortlist_with_complete_candidate_diagnostics():
    base = package()
    policy = PropertySelectionPolicy("source_priority", source_priority=("b", "a"))
    selected = base.with_selection({"liquid_heat_capacity": policy})
    report = selected.selection_report("liquid_heat_capacity", temperature_K=320)
    assert report.selected_id == "second" and len(report.candidates) == 2
    missing = PropertySelectionPolicy("source_priority", source_priority=("missing",))
    unbound = selected.with_selection({"liquid_heat_capacity": missing})
    with pytest.raises(PropertySelectionError) as exc:
        unbound.evaluate("liquid_heat_capacity", temperature_K=300)
    assert exc.value.report.reason == "no_matching_policy"
    assert len(exc.value.report.candidates) == 2
    assert base.selection_report("missing", temperature_K=300).reason == "missing_property"
    anonymous = replace(base, correlations=(replace(base.correlations[0], metadata={}),))
    report = anonymous.selection_report("liquid_heat_capacity", temperature_K=300, policy=policy)
    assert "no declared source_id" in report.candidates[0].diagnostics[0]


def card(identity, cp, source):
    c = replace(curve(identity, cp, source), metadata={"phase": "liquid"})
    origin = InputSource(
        DataSourceProvenance(source, 0, "Synthetic selection fixture", "1", "test"), "synthetic"
    )
    return PropertyInputCard(
        ComponentPropertyPackage(package().component, (c,)),
        (origin,),
        {
            identity: ComponentUncertainty(
                identity,
                unit=c.output_unit,
                source_id=source,
                coverage="unspecified",
                note="Synthetic, not measured",
            )
        },
        CaloricReference(phase="liquid", temperature_K=300),
        "Explicit test datum",
    )


def test_input_card_keeps_both_overlapping_sources_and_explicit_policy_through_json():
    first, second = card("one", 20, "source-a"), card("two", 40, "source-b")
    with pytest.raises(ValueError, match="overlapping"):
        PropertyInputCard.combine((first, second), reference_note="Synthetic alternatives")
    combined = PropertyInputCard.combine(
        (first, second),
        reference_note="Explicit selection",
        selection_policies={
            "liquid_heat_capacity": PropertySelectionPolicy(
                "source_priority", source_priority=("source-b", "source-a")
            )
        },
    )
    copy = PropertyInputCard.from_json(json.dumps(combined.to_dict()))
    assert copy.to_dict() == combined.to_dict()
    assert len(copy.package.correlations) == len(copy.sources) == 2
    assert copy.package.evaluate("liquid_heat_capacity", temperature_K=320).value == 40
    changed = copy.with_selection(
        {"liquid_heat_capacity": PropertySelectionPolicy("correlation", "one")}
    )
    assert changed.package.evaluate("liquid_heat_capacity", temperature_K=320).value == 20
    assert combined.package.evaluate("liquid_heat_capacity", temperature_K=320).value == 40


def test_missing_pressure_is_diagnosed_before_execution_and_unit_domains_are_converted():
    from chemworld.physchem import volume_correlation

    c = volume_correlation(
        "Tait",
        parameters={"P_ref": 1e5, "rho_ref": 800, "B": 1e8, "C": 0.1},
        temperature_range_K=(250, 400),
        pressure_range_Pa=(1e5, 1e7),
        output="density",
        source_note="Synthetic Tait fixture",
    )
    p = ComponentPropertyPackage(package().component, (c,))
    report = p.selection_report(c.property_id, temperature_K=300)
    assert report.reason == "missing_required_input"
    with pytest.raises(PropertySelectionError, match="pressure_Pa"):
        p.evaluate(c.property_id, temperature_K=300)
    assert p.evaluate(c.property_id, temperature_K=300, pressure_Pa=1e5).value == 800
    c = PropertyCorrelation(
        "celsius",
        "linear_liquid_density",
        {"rho_ref": 800, "T_ref": 20, "alpha": 0},
        {"temperature": "degC"},
        "kg/m^3",
        "liquid_density",
        {"temperature": (10, 30)},
        "Synthetic Celsius fixture",
    )
    report = ComponentPropertyPackage(p.component, (c,)).selection_report(
        c.property_id, temperature_K=293.15
    )
    assert report.candidates[0].status == "within_declared_range"


def test_virial_source_selection_changes_actual_pressure_and_thermal_json():
    from tests.test_virial_consumers import thermal_model

    from chemworld.physchem import VirialGasThermodynamics, virial_correlation

    c = ComponentSpec("gas", "C7H8", default_phase="gas")
    curves = tuple(
        virial_correlation(
            "Abbott",
            parameters={"Tc": tc, "Pc": 4e6, "omega": 0.2},
            temperature_range_K=(500, 1000),
            correlation_id=f"B-{tc}",
            source_note="Synthetic virial fixture",
        )
        for tc in (500, 600)
    )
    p = ComponentPropertyPackage(c, curves).with_selection(
        {"second_virial_coefficient": PropertySelectionPolicy("correlation", "B-600")}
    )
    model = VirialGasThermodynamics(p, thermal_model())
    restored = VirialGasThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.package == p
    actual = p.evaluate_state(
        temperature_K=800,
        pressure_Pa=2e5,
        phase="gas",
        quantities=("molar_volume",),
        gas_model="virial",
    )
    single = ComponentPropertyPackage(c, (curves[1],)).evaluate_state(
        temperature_K=800,
        pressure_Pa=2e5,
        phase="gas",
        quantities=("molar_volume",),
        gas_model="virial",
    )
    assert actual.properties == single.properties


def test_selection_policy_rejects_ambiguous_or_mutable_configuration():
    for args in (
        {"mode": "guess"},
        {"correlation_id": "first"},
        {"mode": "correlation"},
        {"mode": "source_priority"},
        {"mode": "source_priority", "source_priority": ("a", "a")},
        {"mode": "source_priority", "source_priority": "source"},
    ):
        with pytest.raises((TypeError, ValueError)):
            PropertySelectionPolicy(**args)
    p = package().with_selection({"liquid_heat_capacity": PropertySelectionPolicy()})
    with pytest.raises(TypeError):
        p.selection_policies["liquid_heat_capacity"] = PropertySelectionPolicy()


def test_source_binding_preserves_piecewise_record_details_and_rejects_conflicts():
    from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity

    base = card("piecewise", 20, "a")
    segments = tuple(
        replace(
            curve(f"segment-{i}", cp, "a", bounds),
            metadata={
                "phase": "liquid",
                "source": {
                    "source_id": "a",
                    "record_id": f"row-{i}",
                    "url": f"https://example.org/row/{i}",
                },
            },
        )
        for i, (cp, bounds) in enumerate(((20, (250, 300)), (40, (300, 400))))
    )
    piecewise = PiecewiseHeatCapacity("piecewise", segments, "Two source-backed segments")
    base = replace(base, package=replace(base.package, correlations=(piecewise,)))
    selected = base.with_selection(
        {"liquid_heat_capacity": PropertySelectionPolicy("source_priority", source_priority=("a",))}
    )
    copied = PropertyInputCard.from_json(json.dumps(selected.to_dict()))
    assert copied.package.correlations[0].to_dict() == piecewise.to_dict()
    actual = copied.evaluate_state(
        temperature_K=320, pressure_Pa=101325, phase="liquid", quantities=("cp", "enthalpy")
    )
    assert actual.properties["cp"].value == 40
    assert actual.properties["enthalpy"].value == pytest.approx(800)
    bad = replace(segments[1], metadata={"phase": "liquid", "source": {"source_id": "b"}})
    conflicting = replace(piecewise, segments=(segments[0], bad))
    with pytest.raises(ValueError, match="disagrees"):
        replace(base, package=replace(base.package, correlations=(conflicting,))).with_selection({})


def test_public_inspection_filters_other_properties_without_evaluating_them():
    from chemworld.physchem import inspect_property_selection

    irrelevant = replace(curve("other", 100, "a"), property_id="ideal_gas_heat_capacity")
    report = inspect_property_selection(
        (irrelevant, *package().correlations), property_id="liquid_heat_capacity", temperature_K=300
    )
    assert len(report.candidates) == 2 and report.selected_id == "first"


@pytest.mark.parametrize("temperature", (350, 360))
@pytest.mark.parametrize("order", (("a", "b"), ("b", "a")))
@pytest.mark.parametrize("mode", ("declared_order", "correlation", "source_priority"))
def test_selection_controls_flash_equilibrium_and_energy_after_json(mode, order, temperature):
    from math import exp, log

    from chemworld.examples.native_flash_research import synthetic_vle_properties
    from chemworld.physchem.flash_units import VLEPropertyPackage, tp_flash
    from chemworld.physchem.property_reports import R_J_PER_MOL_K

    base = synthetic_vle_properties()
    light = base.packages[0]
    alternatives = []
    policies = {}
    for c in light.correlations:
        prop = c.property_id
        for label in order:
            coefficients = dict(c.coefficients)
            if label == "b":
                if prop == "vapor_pressure":
                    coefficients["A"] = log(1e5) + 30000 / R_J_PER_MOL_K / 345
                else:
                    coefficients["a"] = 120
            alternatives.append(
                replace(
                    c,
                    correlation_id=f"{prop}-{label}",
                    coefficients=coefficients,
                    metadata={"source": {"source_id": label}},
                )
            )
        policies[prop] = (
            PropertySelectionPolicy()
            if mode == "declared_order"
            else PropertySelectionPolicy("correlation", f"{prop}-b")
            if mode == "correlation"
            else PropertySelectionPolicy("source_priority", source_priority=("b", "a"))
        )
    selected = replace(light, correlations=tuple(alternatives)).with_selection(policies)
    model = replace(base, packages=(selected, base.packages[1]))
    restored = VLEPropertyPackage.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.to_dict() == model.to_dict()
    label = order[0] if mode == "declared_order" else "b"
    cp, boiling = (80, 330) if label == "a" else (120, 345)
    k1 = exp(30000 / R_J_PER_MOL_K * (1 / boiling - 1 / temperature))
    k2 = exp(40000 / R_J_PER_MOL_K * (1 / 370 - 1 / temperature))
    beta = (
        0
        if 0.5 * (k1 + k2) <= 1
        else 1
        if 0.5 / k1 + 0.5 / k2 <= 1
        else -(0.5 * (k1 - 1) + 0.5 * (k2 - 1)) / ((k1 - 1) * (k2 - 1))
    )
    actual = tp_flash(
        restored,
        {"light": 0.5, "heavy": 0.5},
        temperature_K=temperature,
        pressure_Pa=1e5,
        feed_enthalpy_J=1000,
    )
    assert actual.vapor_fraction == pytest.approx(beta, rel=1e-8, abs=1e-10)
    enthalpy = 0
    for key, k, capacity, latent in (("light", k1, cp, 30000), ("heavy", k2, 100, 40000)):
        liquid = (1 - beta) * 0.5 / (1 + beta * (k - 1))
        vapor = 0.5 - liquid
        assert actual.k_values[key] == pytest.approx(k, rel=1e-8, abs=1e-10)
        assert actual.liquid_amounts_mol[key] == pytest.approx(liquid, rel=1e-8, abs=1e-10)
        assert actual.vapor_amounts_mol[key] == pytest.approx(vapor, rel=1e-8, abs=1e-10)
        enthalpy += 0.5 * capacity * (temperature - 298.15) - liquid * latent
    assert actual.total_enthalpy_j == pytest.approx(enthalpy, abs=1e-6)
    assert actual.heat_duty_J == pytest.approx(enthalpy - 1000, abs=1e-6)
    for prop in policies:
        assert restored.packages[0].selection_report(
            prop, temperature_K=temperature
        ).selected_id == (f"{prop}-{label}")
