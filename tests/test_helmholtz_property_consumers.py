"""Native EOS responses reach the original unit-bearing property interface."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_solvers import HelmholtzAncillary, HelmholtzPhaseModel
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


def fixtures():
    component = ComponentSpec("synthetic-argon", "Ar", default_phase="gas")
    # A mathematical single-phase EOS; the phase declarations are synthetic
    # domain/seed data, not an assertion of measured argon coexistence.
    fluid = HelmholtzFluid(
        component.identifier,
        "",
        0.039948,
        8.3,
        300,
        1000,
        100,
        1500,
        1e9,
        (
            HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
            HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5}),
        ),
        (HelmholtzTerm("ResidualHelmholtzPower", {"n": [0.05], "d": [1], "t": [1], "l": [0]}),),
        "Synthetic algebraic EOS consumer fixture",
    )
    ancillaries = {
        key: HelmholtzAncillary(
            "rhoLnoexp" if key == "rhoL" else "rhoV",
            125,
            1000,
            100,
            125,
            (0,),
            (1,),
        )
        for key in ("rhoL", "rhoV")
    }
    return ComponentPropertyPackage(component, ()), HelmholtzPhaseModel(
        fluid, 125, 1000, 1e6, ancillaries
    )


def test_original_package_uses_eos_without_a_second_cp_or_density_model():
    package, model = fixtures()
    temperature, density, gas_constant = 450, 20, 8.3
    pressure = density * gas_constant * temperature * (1 + 0.05 * (20 / 1000) * (300 / 450))
    keys = (
        "density",
        "enthalpy",
        "internal_energy",
        "entropy",
        "cp",
        "cv",
        "molar_volume",
        "speed_of_sound",
        "compressibility_factor",
        "chemical_potential",
        "dV_dT_P",
        "dV_dP_T",
        "isothermal_compressibility",
        "thermal_expansion",
    )
    report = package.evaluate_state(
        temperature_K=temperature,
        pressure_Pa=pressure,
        phase="gas",
        quantities=keys,
        helmholtz_model=model,
    )
    values = report.properties
    assert values["molar_volume"].value == pytest.approx(1 / density, rel=1e-9)
    assert values["density"].value == pytest.approx(density * 0.039948, rel=1e-9)
    assert values["cv"].value == pytest.approx(1.5 * gas_constant)
    expected_u = gas_constant * (1.5 * temperature + 0.05 * 300 * density / 1000)
    assert values["internal_energy"].value == pytest.approx(expected_u)
    assert values["enthalpy"].value == pytest.approx(expected_u + pressure / density)
    mu = report.component_properties["chemical_potential"][package.component.identifier]
    assert mu.value == pytest.approx(
        values["enthalpy"].value - temperature * values["entropy"].value
    )
    for value in (*values.values(), mu):
        assert value.quantity().to(value.unit).value == value.value
    saved = json.loads(json.dumps(report.to_dict()))
    assert saved["model_details"]["source_molar_mass_kg_mol"] == 0.039948
    assert saved["component_amounts_mol"] == {"synthetic-argon": 1}
    assert not saved["reference_states"]


def test_original_package_derivatives_use_the_same_density_root():
    package, model = fixtures()

    def at(temperature, pressure):
        return package.evaluate_state(
            temperature_K=temperature,
            pressure_Pa=pressure,
            phase="gas",
            quantities=("molar_volume", "dV_dT_P", "dV_dP_T"),
            helmholtz_model=model,
        ).properties

    center = at(450, 1e5)
    dt = (at(450.01, 1e5)["molar_volume"].value - at(449.99, 1e5)["molar_volume"].value) / 0.02
    dp = (at(450, 1e5 + 1)["molar_volume"].value - at(450, 1e5 - 1)["molar_volume"].value) / 2
    assert dt == pytest.approx(center["dV_dT_P"].value, rel=2e-7)
    assert dp == pytest.approx(center["dV_dP_T"].value, rel=2e-7)


@pytest.mark.parametrize(
    "change",
    [
        {"casrn": "7732-18-5"},
        {"fluid_id": "unbound-component"},
    ],
)
def test_source_identity_cannot_be_reassigned_to_existing_inventory(change):
    package, model = fixtures()
    model = replace(model, fluid=replace(model.fluid, **change))
    with pytest.raises(ValueError, match=r"identity|component name"):
        package.evaluate_state(
            temperature_K=450,
            pressure_Pa=1e5,
            phase="gas",
            quantities=("enthalpy",),
            helmholtz_model=model,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"reference": CaloricReference(phase="gas")}, "own caloric reference"),
        ({"gas_model": "virial"}, "own caloric reference"),
        ({"validity_policy": "warn"}, "strict source domain"),
        ({"phase": "solid"}, "fluid phase"),
        ({"quantities": ("viscosity",)}, "No gas_viscosity correlation"),
        ({"temperature_K": 2000}, "source validity"),
    ],
)
def test_other_models_domains_and_offsets_are_not_silently_combined(options, match):
    package, model = fixtures()
    args = {
        "temperature_K": 450,
        "pressure_Pa": 1e5,
        "phase": "gas",
        "quantities": ("enthalpy",),
        "helmholtz_model": model,
        **options,
    }
    with pytest.raises(ValueError, match=match):
        package.evaluate_state(**args)


def test_declared_transport_shares_current_state_without_overriding_eos():
    package, model = fixtures()
    conductivity = PropertyCorrelation(
        correlation_id="explicit-gas-conductivity", property_id="thermal_conductivity",
        equation_id="thermal_conductivity_polynomial", coefficients={"a": .02, "b": .0001},
        input_units={"temperature": "K"}, output_unit="W/(m*K)",
        validity_ranges={"temperature": (300, 500)}, metadata={"phase": "gas"},
        source_note="Synthetic transport relation; independent of EOS",
    )
    package = replace(package, correlations=(conductivity,))
    report = package.evaluate_state(temperature_K=450, pressure_Pa=1e5, phase="gas",
        quantities=("thermal_conductivity", "enthalpy", "density"), helmholtz_model=model)
    k = report.properties["thermal_conductivity"]
    assert k.value == pytest.approx(.065)
    assert k.correlation_id == conductivity.correlation_id
    eos = model.solve_tp(temperature_K=450, pressure_Pa=1e5, phase_hint="gas").state
    assert report.properties["density"].value == eos.mass_density_kg_m3
    assert report.properties["enthalpy"].value == eos.enthalpy_J_mol
    with pytest.raises(ValueError):
        package.evaluate_state(temperature_K=550, pressure_Pa=1e5, phase="gas",
            quantities=("thermal_conductivity",), helmholtz_model=model)
    package = replace(package, correlations=(replace(conductivity, metadata={"phase": "liquid"}),))
    with pytest.raises(ValueError, match="requested phase"):
        package.evaluate_state(temperature_K=450, pressure_Pa=1e5, phase="gas",
            quantities=("thermal_conductivity",), helmholtz_model=model)
