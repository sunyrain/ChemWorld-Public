"""Original package/phase/transport consumption without surrogate identities."""

import json
from dataclasses import replace

import pytest
from tests.reference.incompressible_fixtures import amounts, material, transport_properties

from chemworld.data.incompressible_bindings import bind_incompressible_liquid
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
from chemworld.physchem.phase_evaluation import PhaseEvaluation, PhasePropertyModel
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties
from chemworld.physchem.water_if97_properties import IF97WaterModel
from chemworld.world.transport_world import synthetic_transport_properties


@pytest.mark.parametrize("name", ("MEG", "MEG2", "MPG", "MPG2", "MCA", "MCA2", "Water"))
def test_qualified_identity_json_and_original_phase_report(name):
    model, packages, phase = material(name)
    n = {"water": 2.0} if name == "Water" else amounts(model, 0.2)
    requested = (
        "cp",
        "source_cp",
        "enthalpy",
        "entropy",
        "density",
        "molar_volume",
        "partial_enthalpy",
        "partial_volume",
    )
    report = phase.evaluate(n, temperature_K=300.0, pressure_Pa=2e6, quantities=requested)
    restored_model = IncompressibleMaterialModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored_model.to_dict() == model.to_dict()
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(phase.to_dict())))
    assert (
        restored.evaluate(n, temperature_K=300.0, pressure_Pa=2e6, quantities=requested).to_dict()
        == report.to_dict()
    )
    assert PhaseEvaluation.from_dict(json.loads(json.dumps(report.to_dict()))).to_dict() == (
        report.to_dict()
    )
    assert sum(n[k] * report.value("partial_enthalpy", component=k) for k in n) == pytest.approx(
        sum(n.values()) * report.value("enthalpy"), rel=1e-12
    )
    for key in ("cv", "dU_dT_V", "chemical_potential", "speed_of_sound"):
        assert not phase.capabilities()[key]["supported"]
        with pytest.raises(ValueError, match="unavailable"):
            phase.evaluate(n, temperature_K=300.0, pressure_Pa=2e6, quantities=(key,))
    with pytest.raises(ValueError, match="strict"):
        phase.evaluate(
            n, temperature_K=300.0, pressure_Pa=2e6, quantities=("cp",), validity_policy="ignore"
        )
    direct = evaluate_property_state(
        packages,
        n,
        temperature_K=300.0,
        pressure_Pa=2e6,
        phase="liquid",
        quantities=requested,
        incompressible_model=model,
    )
    assert direct.properties["cp"].value == report.value("cp")
    if name == "Water":
        assert packages[0].evaluate_state(
            temperature_K=300.0,
            pressure_Pa=2e6,
            phase="liquid",
            quantities=("cp",),
            incompressible_model=model,
        ).properties["cp"].value == (report.value("cp"))
        assert packages[0].phase_model(
            phase="liquid", incompressible_model=model
        ).component_ids == ("water",)


@pytest.mark.parametrize("name", ("MEG", "MPG", "MCA"))
def test_original_transport_uses_current_solution_and_true_capacity(name):
    prop = transport_properties(name)
    model = prop.incompressible_model
    n = amounts(model, 0.2, 0.01)
    fluid, h, volume = prop.evaluate(n, 300.0)
    core = model.fluid.state(300.0, prop.pressure_Pa, 0.2)
    assert fluid.heat_capacity_J_kg_K == core.cp_J_kg_K
    assert prop.enthalpy(n, 300.0) == pytest.approx(sum(n.values()) * h)
    assert volume == pytest.approx(0.01 / core.mass_density_kg_m3)
    assert prop.heat_capacity(n, 300.0) == pytest.approx(0.01 * core.cp_J_kg_K)
    assert prop.enthalpy(dict.fromkeys(n, 0.0), 300.0) == 0
    assert LiquidTransportProperties.from_dict(
        json.loads(json.dumps(prop.to_dict()))
    ).to_dict() == (prop.to_dict())
    other = prop.evaluate(amounts(model, 0.05, 0.01), 300.0)[0]
    assert other.density_kg_m3 != fluid.density_kg_m3
    assert other.viscosity_Pa_s != fluid.viscosity_Pa_s


@pytest.mark.parametrize("basis", ("mole", "volume"))
def test_explicit_other_concentration_bases_and_extensive_derivatives(basis):
    original, packages, _ = material("MEG")
    # Synthetic axis declaration tests the general binding contract. These
    # coefficients are not represented as a qualified volume/mole EG fit.
    fluid = replace(original.fluid, fraction_basis=basis, fluid_id="synthetic-axis")
    options = (
        {}
        if basis == "mole"
        else {
            "preparation_molar_volumes_m3_mol": (18e-6, 55e-6),
            "preparation_reference_temperature_K": 293.15,
            "preparation_source": (
                "synthetic fixed preparation volumes, not current solution volume"
            ),
        }
    )
    if basis == "volume":
        with pytest.raises(ValueError, match="preparation"):
            replace(original, fluid=fluid)
    model = replace(
        original, fluid=fluid, binding_source="synthetic concentration-axis test", **options
    )
    n = {"water": 4.0, "glycol": 1.0}
    x = model.inventory_coordinates(n)[3]
    assert x == pytest.approx(0.2 if basis == "mole" else 55 / (4 * 18 + 55))
    phase = PhasePropertyModel(packages, phase="liquid", incompressible_model=model)
    quantities = ("enthalpy", "partial_enthalpy", "molar_volume", "partial_volume")
    report = phase.evaluate(n, temperature_K=300.0, pressure_Pa=1e5, quantities=quantities)
    for field, partial in (("enthalpy", "partial_enthalpy"), ("molar_volume", "partial_volume")):
        for key in n:
            h = 1e-5
            values = []
            for sign in (1, -1):
                perturbed = {**n, key: n[key] + sign * h}
                values.append(
                    sum(perturbed.values())
                    * phase.evaluate(
                        perturbed, temperature_K=300.0, pressure_Pa=1e5, quantities=(field,)
                    ).value(field)
                )
            assert report.value(partial, component=key) == pytest.approx(
                (values[0] - values[1]) / (2 * h), rel=1e-7, abs=1e-10
            )


def test_identity_component_weight_and_binding_fail_closed():
    model, packages, _ = material("MEG")
    with pytest.raises(ValueError, match="solute identity"):
        bind_incompressible_liquid("MPG", water=model.components[0], solute=model.components[1])
    with pytest.raises(ValueError, match="qualified"):
        bind_incompressible_liquid("DowQ", water=model.components[0])
    wrong = replace(
        packages[1],
        component=replace(
            packages[1].component,
            molecular_weight_g_mol=packages[1].component.molecular_weight_g_mol * 1.001,
        ),
    )
    with pytest.raises(ValueError, match="identities"):
        PhasePropertyModel((packages[0], wrong), phase="liquid", incompressible_model=model)
    with pytest.raises(ValueError, match="binding"):
        replace(model, binding_source="")
    with pytest.raises(ValueError, match="neutral pure or binary"):
        replace(model, solute_id="undeclared")
    with pytest.raises(ValueError):
        bind_incompressible_liquid("Water", water=ComponentSpec("water", "H2O"))
    with pytest.raises(KeyError):
        model.inventory_coordinates({"water": 1.0, "unknown": 1.0})
    with pytest.raises(ValueError):
        model.inventory_coordinates({"water": True, "glycol": 1.0})
    with pytest.raises(ValueError):
        model.inventory_coordinates({"water": 0.0, "glycol": 0.0})
    with pytest.raises(ValueError, match="concentration outside"):
        model.inventory_coordinates({"solvent": 2.0, "glycol": 1.0})
    assert model.inventory_coordinates({"solvent": 4.0, "glycol": 1.0})[0] == {
        "water": 4.0,
        "glycol": 1.0,
    }


@pytest.mark.parametrize("conflict", ("reference", "activity", "volume", "water", "transport"))
def test_mutually_exclusive_phase_sources(conflict):
    model, packages, _ = material("MEG")
    options = {
        "reference": {"references": {"water": CaloricReference(phase="liquid")}},
        "activity": {"activity_model": ActivityModelSpec("conflict", model.component_ids)},
        "volume": {"volume_model": object()},
        "water": {"water_model": IF97WaterModel()},
        "transport": {"transport_models": {"viscosity": object()}},
    }
    with pytest.raises(ValueError):
        PhasePropertyModel(
            packages, phase="liquid", incompressible_model=model, **options[conflict]
        )


def test_transport_conflicts_domain_and_legacy_json():
    prop = transport_properties("MEG")
    with pytest.raises(ValueError, match="exclusive"):
        replace(prop, volume_mixing_method="Amgat")
    with pytest.raises(ValueError, match="exclusive"):
        replace(prop, caloric_references={"water": CaloricReference(phase="liquid")})
    with pytest.raises(ValueError, match="exceeds"):
        replace(prop, temperature_bounds_K=(100.0, 400.0))
    legacy = synthetic_transport_properties()
    assert "incompressible_model" not in legacy.to_dict()
    assert LiquidTransportProperties.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()
    package = ComponentPropertyPackage(prop.packages[0].component, ())
    phase = package.phase_model(phase="liquid")
    assert "incompressible_model" not in phase.to_dict()
    assert PhasePropertyModel.from_dict(phase.to_dict()).to_dict() == phase.to_dict()


def test_public_exports():
    from chemworld import data, physchem

    assert data.bind_incompressible_liquid is bind_incompressible_liquid
    assert physchem.IncompressibleMaterialModel is IncompressibleMaterialModel
