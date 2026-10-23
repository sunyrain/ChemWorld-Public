import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.physchem.eos import CubicEOSSpec, EOSComponentSpec
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.phase_evaluation import (
    PhaseEvaluation,
    PhasePropertyModel,
    PhaseTransportSpec,
)
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.property_selection import PropertySelectionPolicy
from chemworld.physchem.specs import ComponentSpec, MixtureSpec, PropertyCorrelation


def packages():
    output = []
    for i, (key, formula) in enumerate((("A", "C2H6"), ("B", "C3H8"))):
        specifications = (
            ("ideal_gas_heat_capacity", "cp_polynomial", {"a": 80 + 20 * i}, "J/(mol*K)"),
            ("liquid_heat_capacity", "cp_polynomial", {"a": 100 + 20 * i}, "J/(mol*K)"),
            (
                "gas_viscosity",
                "sutherland_gas_viscosity",
                {"mu_ref": (1 + i) * 1e-5, "T_ref": 300, "S": 100},
                "Pa*s",
            ),
            ("liquid_viscosity", "andrade_viscosity", {"A": 1e-5, "B": 1000 + 200 * i}, "Pa*s"),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": 0.1 + 0.1 * i, "T_ref": 300},
                "W/(m*K)",
            ),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": 700 + 100 * i, "T_ref": 300},
                "kg/m^3",
            ),
            (
                "surface_tension",
                "surface_tension_power",
                {"Tc": 600 + 50 * i, "T_ref": 300, "sigma_ref": 0.03 + 0.01 * i},
                "N/m",
            ),
        )
        curves = tuple(
            PropertyCorrelation(
                f"{key}-{prop}",
                equation,
                coefficients,
                {"temperature": "K"},
                unit,
                property_id=prop,
                validity_ranges={"temperature": (250, 550)},
                source_note="Explicit synthetic phase consumer inputs",
            )
            for prop, equation, coefficients, unit in specifications
        )
        output.append(
            ComponentPropertyPackage(ComponentSpec(key, formula, aliases=(f"alias-{key}",)), curves)
        )
    return tuple(output)


def references(phase="gas"):
    return {
        key: CaloricReference(
            phase=phase,
            temperature_K=300,
            energy_J_mol=100 * i,
            entropy_J_mol_K=200 + 20 * i,
            source_note="Explicit synthetic datum",
        )
        for i, key in enumerate(("A", "B"))
    }


def test_phase_model_and_evaluation_are_immutable_and_share_original_state_consumer():
    ps = packages()
    model = PhasePropertyModel(
        ps,
        phase="gas",
        references=references(),
        transport_models={
            "viscosity": PhaseTransportSpec("viscosity", "Wilke", "Selected published gas mixing"),
            "thermal_conductivity": PhaseTransportSpec(
                "thermal_conductivity",
                "Wassiljewa_Herning_Zipperer",
                "Selected gas conductivity mixing",
            ),
        },
    )
    amounts = {"alias-A": 2.0, "B": 3.0}
    quantities = (
        "cp",
        "cv",
        "enthalpy",
        "internal_energy",
        "entropy",
        "chemical_potential",
        "density",
        "molar_volume",
        "fugacity_coefficient",
        "viscosity",
        "thermal_conductivity",
    )
    result = model.evaluate(amounts, temperature_K=340, pressure_Pa=2e5, quantities=quantities)
    original = evaluate_property_state(
        ps,
        amounts,
        phase="gas",
        temperature_K=340,
        pressure_Pa=2e5,
        references=references(),
        quantities=quantities[:-2],
    )
    assert result.value("cp") == original.properties["cp"].value
    assert result.value("enthalpy", unit="kJ/mol") == original.properties["enthalpy"].value / 1000
    assert result.value("chemical_potential", component="A") == (
        original.component_properties["chemical_potential"]["A"].value
    )
    assert result.value("enthalpy") - result.value("internal_energy") == pytest.approx(
        2e5 * result.value("molar_volume")
    )
    assert result.value("viscosity") > 0
    assert result.composition == {"A": 0.4, "B": 0.6}
    payload = model.to_dict()
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(payload)))
    assert (
        restored.evaluate(
            amounts, temperature_K=340, pressure_Pa=2e5, quantities=quantities
        ).to_dict()
        == result.to_dict()
    )
    assert PhaseEvaluation.from_dict(json.loads(json.dumps(result.to_dict()))).to_dict() == (
        result.to_dict()
    )
    ps[0].correlations[0].coefficients["a"] = 9000
    amounts["B"] = 999
    payload["packages"][0]["correlations"][0]["coefficients"]["a"] = -10
    assert model.evaluate(
        {"A": 2, "B": 3}, temperature_K=340, pressure_Pa=2e5, quantities=("cp",)
    ).value("cp") == result.value("cp")
    with pytest.raises(TypeError):
        result.report["properties"]["cp"]["value"] = 0


def test_liquid_activity_phase_retains_energy_entropy_and_reports_missing_closures():
    activity = ActivityModelSpec("liquid", ("A", "B"), "margules", {"A:A|B": 0.4, "A:B|A": 0.6})
    model = PhasePropertyModel(
        packages(),
        phase="liquid",
        references=references("liquid"),
        activity_model=activity,
        transport_models={
            "viscosity": PhaseTransportSpec(
                "viscosity", "log_mole_fraction", "Explicit liquid mixing"
            )
        },
    )
    result = model.evaluate(
        {"A": 0.4, "B": 0.6},
        temperature_K=340,
        pressure_Pa=101325,
        quantities=(
            "enthalpy",
            "cp",
            "entropy",
            "chemical_potential",
            "activity_coefficient",
            "viscosity",
        ),
    )
    pure_mu = [1e-5 * exp(1000 / 340), 1e-5 * exp(1200 / 340)]
    assert result.value("viscosity") == pytest.approx(
        exp(0.4 * log(pure_mu[0]) + 0.6 * log(pure_mu[1]))
    )
    assert result.value("enthalpy") - 340 * result.value("entropy") == pytest.approx(
        0.4 * result.value("chemical_potential", component="A")
        + 0.6 * result.value("chemical_potential", component="B")
    )
    assert not model.capabilities()["cv"]["supported"]
    assert not model.capabilities()["density"]["supported"]
    with pytest.raises(ValueError, match="unavailable"):
        model.evaluate(
            {"A": 0.4, "B": 0.6}, temperature_K=340, pressure_Pa=101325, quantities=("cv",)
        )
    without_ref = PhasePropertyModel(packages(), phase="gas")
    assert without_ref.capabilities()["cp"]["supported"]
    assert not without_ref.capabilities()["enthalpy"]["supported"]


TRANSPORTS = (
    ("gas", "viscosity", "Wilke", {}),
    ("gas", "viscosity", "Herning_Zipperer", {}),
    ("gas", "viscosity", "Brokaw", {"polar_parameters": [0.2, 0.3], "epsilon_k_K": [200, 250]}),
    (
        "gas",
        "viscosity",
        "LBC",
        {
            "critical_temperatures_K": [500, 580],
            "critical_pressures_Pa": [4e6, 3.5e6],
            "critical_volumes_m3_mol": [2e-4, 3e-4],
        },
    ),
    ("liquid", "viscosity", "log_mole_fraction", {}),
    ("liquid", "thermal_conductivity", "DIPPR9H", {}),
    ("liquid", "thermal_conductivity", "DIPPR9I", {}),
    ("liquid", "thermal_conductivity", "Filippov", {}),
    ("gas", "thermal_conductivity", "Wassiljewa_Herning_Zipperer", {}),
    ("gas", "thermal_conductivity", "Lindsay_Bromley", {"boiling_temperatures_K": [300, 350]}),
    ("liquid", "surface_tension", "Winterfeld_Scriven_Davis", {}),
    (
        "liquid",
        "surface_tension",
        "Diguilio_Teja",
        {"boiling_temperatures_K": [300, 350], "critical_temperatures_K": [600, 650]},
    ),
)


@pytest.mark.parametrize("phase,quantity,method,parameters", TRANSPORTS)
def test_original_mixture_transport_families_consume_current_phase_sources(
    phase,
    quantity,
    method,
    parameters,
):
    model = PhasePropertyModel(
        packages(),
        phase=phase,
        transport_models={
            quantity: PhaseTransportSpec(
                quantity, method, "Published explicit mixing selection", parameters
            )
        },
    )
    result = model.evaluate(
        {"A": 2, "B": 3}, temperature_K=340, pressure_Pa=1e5, quantities=(quantity,)
    )
    assert result.value(quantity) > 0
    evidence = result.report["model_details"]["transport"][quantity]
    assert evidence["mixture"]["temperature_K"] == 340
    assert evidence["mixture"]["mole_fractions"] == (0.4, 0.6)
    if method != "Diguilio_Teja":
        assert all(v["inputs"]["temperature"] == 340 for v in evidence["pure_evaluations"])
    else:
        assert [v["inputs"]["temperature"] for v in evidence["pure_evaluations"]] == [300, 350]


def test_missing_source_and_ambiguous_correlations_do_not_become_defaults():
    ps = packages()
    missing = replace(
        ps[0], correlations=tuple(c for c in ps[0].correlations if c.property_id != "gas_viscosity")
    )
    model = PhasePropertyModel(
        (missing, ps[1]),
        phase="gas",
        transport_models={"viscosity": PhaseTransportSpec("viscosity", "Wilke", "Explicit mixing")},
    )
    assert not model.capabilities()["viscosity"]["supported"]
    with pytest.raises(ValueError, match="unavailable"):
        model.evaluate(
            {"A": 0.4, "B": 0.6}, temperature_K=340, pressure_Pa=1e5, quantities=("viscosity",)
        )
    with pytest.raises(ValueError, match="current pure curves"):
        PhaseTransportSpec(
            "thermal_conductivity", "DIPPR9I", "Explicit source", {"molar_volumes_m3_mol": [1, 2]}
        )
    with pytest.raises(ValueError, match="composition binding"):
        PhasePropertyModel(
            ps,
            phase="gas",
            transport_models={
                "thermal_conductivity": next(
                    c for c in ps[0].correlations if c.property_id == "thermal_conductivity"
                )
            },
        )


def test_cubic_phase_model_consumes_original_departures_and_fixed_json():
    ps = packages()
    eos = CubicEOSSpec(
        "phase-pr",
        "peng_robinson",
        (EOSComponentSpec("A", 500, 4e6, 0.2), EOSComponentSpec("B", 580, 3.5e6, 0.4)),
    )
    model = PhasePropertyModel(ps, phase="liquid", references=references(), eos=eos)
    result = model.evaluate(
        {"A": 0.4, "B": 0.6},
        temperature_K=340,
        pressure_Pa=1e5,
        quantities=(
            "cp",
            "cv",
            "enthalpy",
            "entropy",
            "chemical_potential",
            "density",
            "molar_volume",
        ),
    )
    assert result.value("density") > 1
    assert result.value("cp") > result.value("cv")
    assert result.value("enthalpy") - 340 * result.value("entropy") == pytest.approx(
        0.4 * result.value("chemical_potential", component="A")
        + 0.6 * result.value("chemical_potential", component="B")
    )


def test_parallel_helmholtz_phase_model_uses_original_reference_and_caller_amounts():
    from test_helmholtz_property_consumers import fixtures

    package, helmholtz = fixtures()
    model = PhasePropertyModel((package,), phase="gas", helmholtz_model=helmholtz)
    quantities = ("cp", "cv", "enthalpy", "entropy", "molar_volume", "density", "speed_of_sound")
    result = model.evaluate(
        {package.component.identifier: 3}, temperature_K=450, pressure_Pa=1e5, quantities=quantities
    )
    direct = package.evaluate_state(
        temperature_K=450,
        pressure_Pa=1e5,
        phase="gas",
        quantities=quantities,
        helmholtz_model=helmholtz,
    )
    assert {q: result.value(q) for q in quantities} == {
        q: v.value for q, v in direct.properties.items()
    }
    assert sum(result.report["component_amounts_mol"].values()) == 3
    assert PhasePropertyModel.from_dict(model.to_dict()).to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="own phase/caloric"):
        PhasePropertyModel(
            (package,),
            phase="gas",
            helmholtz_model=helmholtz,
            references={package.component.identifier: CaloricReference(phase="gas")},
        )


def test_fixed_mixture_transport_curve_keeps_original_composition_binding():
    ps = packages()
    mixture = MixtureSpec.from_mole_fractions(
        tuple(p.component for p in ps),
        (0.4, 0.6),
        phase_label="liquid",
        temperature_K=340,
        pressure_Pa=1e5,
    )
    curve = replace(
        next(c for c in ps[0].correlations if c.property_id == "thermal_conductivity"),
        metadata={"mixture": mixture.to_dict(), "phase": "liquid"},
    )
    model = PhasePropertyModel(ps, phase="liquid", transport_models={"thermal_conductivity": curve})
    result = model.evaluate(
        {"A": 2, "B": 3}, temperature_K=340, pressure_Pa=1e5, quantities=("thermal_conductivity",)
    )
    assert result.value("thermal_conductivity") == 0.1
    with pytest.raises(ValueError, match="composition/MW"):
        model.evaluate(
            {"A": 1, "B": 1},
            temperature_K=340,
            pressure_Pa=1e5,
            quantities=("thermal_conductivity",),
        )


def test_transport_source_validity_and_explicit_pure_selection_do_not_fall_back():
    ps = packages()
    original = next(c for c in ps[0].correlations if c.property_id == "thermal_conductivity")
    selected_package = replace(
        ps[0],
        correlations=(
            *ps[0].correlations,
            replace(
                original,
                correlation_id="alternative-k",
                validity_ranges={"temperature": (250, 280)},
            ),
        ),
    ).with_selection(
        {
            "thermal_conductivity": PropertySelectionPolicy(
                mode="correlation", correlation_id="alternative-k"
            )
        }
    )
    selected = {
        "thermal_conductivity": PhaseTransportSpec(
            "thermal_conductivity",
            "Wassiljewa_Herning_Zipperer",
            "Explicit mixing and chosen pure alternative",
        )
    }
    model = PhasePropertyModel((selected_package, ps[1]), phase="gas", transport_models=selected)
    with pytest.raises(ValueError):
        model.evaluate(
            {"A": 0.4, "B": 0.6},
            temperature_K=340,
            pressure_Pa=1e5,
            quantities=("thermal_conductivity",),
        )
    model = PhasePropertyModel(ps, phase="gas", transport_models=selected)
    with pytest.raises(ValueError):
        model.evaluate(
            {"A": 0.4, "B": 0.6},
            temperature_K=600,
            pressure_Pa=1e5,
            quantities=("thermal_conductivity",),
        )
    result = model.evaluate(
        {"A": 0.4, "B": 0.6},
        temperature_K=600,
        pressure_Pa=1e5,
        quantities=("thermal_conductivity",),
        validity_policy="warn",
    )
    assert result.report["properties"]["thermal_conductivity"]["warnings"]


def test_phase_transport_selects_phase_specific_pure_conductivity_sources():
    ps = packages()
    changed = []
    for p in ps:
        base = next(c for c in p.correlations if c.property_id == "thermal_conductivity")
        changed.append(
            replace(
                p,
                correlations=(
                    *(c for c in p.correlations if c.property_id != "thermal_conductivity"),
                    replace(base, correlation_id="gas-k", metadata={"phase": "gas"}),
                    replace(
                        base,
                        correlation_id="liquid-k",
                        metadata={"phase": "liquid"},
                        coefficients={**base.coefficients, "k_ref": 9.0},
                    ),
                ),
            )
        )
    model = PhasePropertyModel(
        changed,
        phase="gas",
        transport_models={
            "thermal_conductivity": PhaseTransportSpec(
                "thermal_conductivity", "Wassiljewa_Herning_Zipperer", "Explicit mixing"
            )
        },
    )
    result = model.evaluate(
        {"A": 0.4, "B": 0.6},
        temperature_K=340,
        pressure_Pa=1e5,
        quantities=("thermal_conductivity",),
    )
    assert 0.1 <= result.value("thermal_conductivity") <= 0.2
    assert {
        e["correlation_id"]
        for e in result.report["model_details"]["transport"]["thermal_conductivity"][
            "pure_evaluations"
        ]
    } == {"gas-k"}
