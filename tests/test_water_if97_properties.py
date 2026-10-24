"""Water property consumption keeps identity, units, datum and caller stock."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem import IF97WaterModel
from chemworld.physchem.phase_evaluation import PhaseEvaluation, PhasePropertyModel
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.water_if97_solvers import water_if97_tp
from chemworld.physchem.water_saturation import water_saturation_pressure

POINTS = (
    (300, 3e6, "liquid"), (700, 3500, "gas"), (640, 22e6, "liquid"),
    (640, 19e6, "gas"), (650, 25.6e6, "gas"), (1500, .5e6, "gas"),
    *((T, water_saturation_pressure(T), phase)
      for T in (500, 623.15, 640) for phase in ("gas", "liquid")),
)


def water():
    return ComponentPropertyPackage(ComponentSpec(
        "water", "H2O", cas_number="7732-18-5", aliases=("H2O",)), ())


@pytest.mark.parametrize("temperature,pressure,phase", POINTS)
@pytest.mark.parametrize("amount", [.25, 1, 4])
def test_all_regions_share_one_mass_to_molar_conversion_and_caller_stock(
    temperature, pressure, phase, amount,
):
    package = water()
    model = package.phase_model(phase=phase, water_model=IF97WaterModel())
    native = water_if97_tp(temperature, pressure,
        phase=None if temperature >= 647.096 else "vapor" if phase == "gas" else "liquid")
    mass = package.component.molecular_weight_g_mol / 1000
    volume = mass / native.mass_density_kg_m3
    expected = {
        "pressure": native.pressure_Pa, "density": native.mass_density_kg_m3,
        "enthalpy": native.enthalpy_J_kg * mass,
        "internal_energy": native.internal_energy_J_kg * mass,
        "entropy": native.entropy_J_kg_K * mass, "cp": native.cp_J_kg_K * mass,
        "cv": native.cv_J_kg_K * mass, "dH_dT_P": native.cp_J_kg_K * mass,
        "dU_dT_V": native.cv_J_kg_K * mass, "speed_of_sound": native.speed_of_sound_m_s,
        "gibbs_energy": native.gibbs_energy_J_kg * mass,
        "isothermal_compressibility": native.isothermal_compressibility_Pa_inv,
        "thermal_expansion": native.thermal_expansion_K_inv,
        "molar_volume": volume, "dV_dT_P": native.thermal_expansion_K_inv * volume,
        "dV_dP_T": -native.isothermal_compressibility_Pa_inv * volume,
        "chemical_potential": native.gibbs_energy_J_kg * mass,
        "helmholtz_energy": (native.internal_energy_J_kg
                              - temperature * native.entropy_J_kg_K) * mass,
        "compressibility_factor": native.pressure_Pa * volume / (461.526 * mass * temperature),
    }
    inventory = {"H2O": amount}
    state = model.evaluate(inventory, temperature_K=temperature, pressure_Pa=pressure,
                           quantities=tuple(expected))
    assert inventory == {"H2O": amount}
    assert state.report["component_amounts_mol"] == {"water": amount}
    assert state.report["model_details"]["molar_conversion_kg_mol"] == mass
    assert state.valid
    for key, value in expected.items():
        assert state.value(key, component="water" if key == "chemical_potential" else None
                           ) == pytest.approx(value, rel=2e-12, abs=1e-9)
    assert state.value("enthalpy") == pytest.approx(
        state.value("internal_energy") + pressure * state.value("molar_volume"), rel=2e-12)
    assert state.value("gibbs_energy") == pytest.approx(
        state.value("enthalpy") - temperature * state.value("entropy"), rel=2e-12)
    assert state.value("enthalpy", unit="kJ/mol") == pytest.approx(expected["enthalpy"] / 1000)
    restored = PhaseEvaluation.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.to_dict() == state.to_dict()
    restored_model = PhasePropertyModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored_model.to_dict() == model.to_dict()


@pytest.mark.parametrize("change", [
    {"cas_number": ""}, {"cas_number": "64-17-5"}, {"charge": 1},
    {"formula": "H2O2", "molecular_weight_g_mol": None},
])
def test_water_identity_cannot_be_reassigned(change):
    package = water()
    package = replace(package, component=replace(package.component, **change))
    with pytest.raises(ValueError, match="neutral H2O"):
        package.phase_model(phase="gas", water_model=IF97WaterModel())


@pytest.mark.parametrize("options", [
    {"reference": CaloricReference(phase="gas")}, {"gas_model": "virial"},
    {"helmholtz_model": object()}, {"validity_policy": "ignore"},
    {"validity_policy": "warn"},
])
def test_other_references_and_equations_cannot_override_if97(options):
    with pytest.raises(ValueError, match="exclusive"):
        water().evaluate_state(temperature_K=500, pressure_Pa=1e5, phase="gas",
            quantities=("enthalpy",), water_model=IF97WaterModel(), **options)


def test_phase_declaration_and_domain_failures_are_explicit():
    package, model = water(), IF97WaterModel()
    for options in ({"helmholtz_model": object()}, {"reference": CaloricReference(phase="gas")}):
        with pytest.raises(ValueError, match="exclusively"):
            package.phase_model(phase="gas", water_model=model, **options)
    with pytest.raises(ValueError, match="exclusively"):
        PhasePropertyModel((package, replace(package, component=replace(
            package.component, identifier="second"))), phase="gas", water_model=model)
    with pytest.raises(ValueError, match="exclusively"):
        package.phase_model(phase="solid", water_model=model)
    declaration = package.phase_model(phase="liquid", water_model=model)
    for temperature, pressure, amounts in ((650, 25e6, {"water": 1}),
                                          (270, 1e6, {"water": 1}),
                                          (300, 3e6, {})):
        with pytest.raises(ValueError):
            declaration.evaluate(amounts, temperature_K=temperature,
                                 pressure_Pa=pressure, quantities=("density",))
    with pytest.raises(KeyError, match="unknown component"):
        declaration.evaluate({"Ar": 1}, temperature_K=300,
                             pressure_Pa=3e6, quantities=("density",))
    with pytest.raises(ValueError, match="strict"):
        declaration.evaluate({"water": 1}, temperature_K=300, pressure_Pa=3e6,
                             quantities=("density",), validity_policy="warn")
    with pytest.raises(ValueError, match="modified"):
        IF97WaterModel.from_dict({**model.to_dict(), "density_offset": 1})
    legacy = package.phase_model(phase="gas")
    assert "water_model" not in legacy.to_dict()
    assert PhasePropertyModel.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()


def test_declared_transport_reuses_original_selection_and_does_not_replace_energy():
    curve = PropertyCorrelation(
        correlation_id="explicit-water-conductivity", property_id="thermal_conductivity",
        equation_id="thermal_conductivity_polynomial", coefficients={"a": .02, "b": .0001},
        input_units={"temperature": "K"}, output_unit="W/(m*K)",
        validity_ranges={"temperature": (400, 600)}, metadata={"phase": "gas"},
        source_note="Synthetic integration fixture, not measured steam conductivity")
    package = replace(water(), correlations=(curve,))
    model = package.phase_model(phase="gas", water_model=IF97WaterModel())
    assert model.capabilities()["thermal_conductivity"]["supported"]
    assert not model.capabilities()["viscosity"]["supported"]
    original = water_if97_tp(500, 1e5)
    explicit = package.phase_model(phase="gas", water_model=IF97WaterModel(),
                                   transport_models={"thermal_conductivity": curve})
    for declaration in (model, explicit):
        result = declaration.evaluate({"water": 2}, temperature_K=500, pressure_Pa=1e5,
            quantities=("thermal_conductivity", "enthalpy", "density"))
        assert result.value("thermal_conductivity") == pytest.approx(.07)
        assert result.properties["thermal_conductivity"]["correlation_id"] == curve.correlation_id
        assert result.value("density") == original.mass_density_kg_m3
        assert result.value("enthalpy") == (original.enthalpy_J_kg
                                            * (package.component.molecular_weight_g_mol / 1000))
        with pytest.raises(ValueError):
            declaration.evaluate({"water": 2}, temperature_K=700, pressure_Pa=1e5,
                                 quantities=("thermal_conductivity",))
