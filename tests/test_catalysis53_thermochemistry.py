"""Descriptor domains and one gas/harmonic caloric source for G, U and Cv."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.physchem.catalysis_descriptors import DescriptorAxis, FreeEnergyBEP
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from reference.catalysis53_fixtures import SOURCE, harmonic, mechanism


@pytest.mark.parametrize(
    "point,expected",
    [
        ({"binding": 0, "strain": 0}, -5000),
        ({"binding": 10000, "strain": 5000}, 1250),
        ({"binding": -10000, "strain": -5000}, -11250),
    ],
)
def test_descriptor_values_and_complete_codec(point, expected):
    spec = mechanism(descriptors=point)
    assert spec.species[1].adsorbate.electronic_energy.evaluate(point) == expected
    restored = CatalyticMechanismSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()


@pytest.mark.parametrize("temperature", [300, 350, 450])
def test_harmonic_partition_function_and_same_table_derivatives(temperature):
    model = harmonic(modes=(2000,))
    state = model.evaluate(temperature)
    q = exp(-2000 / (R * temperature))
    occupation = q / (1 - q)
    expected_u = -5000 + 2000 * (0.5 + occupation)
    expected_g = -5000 + 1000 + R * temperature * log(1 - q)
    expected_cv = R * (2000 / (R * temperature)) ** 2 * q / (1 - q) ** 2
    assert state.enthalpy_J_mol == pytest.approx(expected_u)
    assert state.gibbs_J_mol == pytest.approx(expected_g)
    assert state.cp_J_mol_K == pytest.approx(expected_cv)
    step = 1e-3
    left, right = model.evaluate(temperature - step), model.evaluate(temperature + step)
    assert (right.enthalpy_J_mol - left.enthalpy_J_mol) / (2 * step) == pytest.approx(
        state.cp_J_mol_K, rel=2e-6
    )
    assert -(right.gibbs_J_mol - left.gibbs_J_mol) / (2 * step) == pytest.approx(
        state.entropy_J_mol_K, rel=2e-6
    )


def test_gas_uses_original_thermo_and_surface_has_no_gas_pv_subtraction():
    spec = mechanism()
    gas, ads, _ = spec.species
    for temperature in (300, 350, 450):
        assert gas.internal_energy(temperature) == pytest.approx(
            gas.evaluate(temperature).enthalpy_J_mol - R * temperature
        )
        assert gas.cv(temperature) == pytest.approx(1.5 * R)
        assert ads.internal_energy(temperature) == ads.evaluate(temperature).enthalpy_J_mol
        assert ads.cv(temperature) == ads.evaluate(temperature).cp_J_mol_K
    thermal = CatalyticThermodynamics(spec.species_by_id, 5)
    assert (
        CatalyticThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict()))).to_dict()
        == thermal.to_dict()
    )


def test_bad_domains_missing_data_and_negative_bep_never_silently_clip():
    with pytest.raises(ValueError, match="outside"):
        harmonic(descriptors={"binding": 10001, "strain": 0})
    with pytest.raises(ValueError, match="positive"):
        harmonic(modes=(0,))
    with pytest.raises(ValueError, match="J/mol"):
        DescriptorAxis("x", 0, 1, "eV")
    with pytest.raises(ValueError, match="negative"):
        FreeEnergyBEP(1, 0.5, -100, 100, SOURCE).barriers(10)
    with pytest.raises(ValueError, match="domain"):
        FreeEnergyBEP(100, 0.5, -100, 100, SOURCE).barriers(101)
    spec = mechanism()
    with pytest.raises(ValueError, match="reference frame"):
        replace(
            spec, species=(replace(spec.species[0], reference_frame="unrelated"), *spec.species[1:])
        )


def test_nested_gas_declaration_and_snapshot_outputs_cannot_change_cached_physics():
    from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo

    original = mechanism().species[0]
    authored = original.to_dict()["gas_declaration"]
    bound = replace(original, gas_declaration=authored)
    initial = bound.to_dict()
    energy = bound.internal_energy(350)
    authored["thermo"]["data"][0][5] += 1000
    authored["composition"]["He"] = 99
    exported = bound.to_dict()
    exported["gas_declaration"]["thermo"]["data"][0][5] -= 2000
    exported["gas_declaration"]["composition"]["He"] = 77
    assert bound.to_dict() == initial
    assert bound.internal_energy(350) == energy
    restored = CatalyticSpeciesThermo.from_dict(json.loads(json.dumps(bound.to_dict())))
    assert restored.internal_energy(350) == energy
    with pytest.raises(TypeError):
        bound.gas_declaration["thermo"]["data"][0][5] = 0


def test_original_piecewise_gibbs_mapping_survives_frozen_declaration():
    from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo

    cp = 20.8
    declaration = {
        "thermo": {
            "model": "piecewise-Gibbs",
            "h0": 0,
            "T-min": 250,
            "T-max": 600,
            "data": {str(t): cp * ((t - 298.15) - t * log(t / 298.15)) for t in (250, 298.15, 600)},
        },
        "composition": {"He": 1},
        "reference_pressure_Pa": 100000,
        "units": {"energy": "J", "quantity": "mol"},
    }
    bound = replace(mechanism().species[0], gas_declaration=declaration)
    expected = cp * (350 - 298.15) - R * 350
    assert bound.internal_energy(350) == pytest.approx(expected)
    restored = CatalyticSpeciesThermo.from_dict(json.loads(json.dumps(bound.to_dict())))
    assert restored.internal_energy(350) == bound.internal_energy(350)
