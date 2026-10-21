"""Element/isotope/charge balance, formation references and real caloric consumers."""

import json
from dataclasses import replace
from fractions import Fraction
from math import log

import pytest

from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.stoichiometry import (
    balance_stoichiometry,
    standard_formation_reaction,
    stoichiometric_matrix,
    stoichiometry_mass_error,
    stoichiometry_mass_to_molar,
    stoichiometry_molar_to_mass,
)
from chemworld.physchem.thermochemistry import (
    CorrelationSpeciesThermo,
    formation_enthalpy_basis,
    formation_entropy,
    formation_gibbs_energy,
    formation_reference_phase_change,
    reaction_thermochemistry,
)


@pytest.mark.parametrize(
    "formulas,sides,charges,expected",
    [
        (("NH3", "O2", "NO", "H2O"), (True, True, False, False), None, (4, 5, 4, 6)),
        (("C2H6", "O2", "CO2", "H2O"), (True, True, False, False), None, (2, 7, 4, 6)),
        (
            ("Fe", "MnO4", "H", "Fe", "Mn", "H2O"),
            (True, True, True, False, False, False),
            (2, -1, 1, 3, 2, 0),
            (5, 1, 8, 5, 1, 4),
        ),
        (("H2", "D2", "HD"), (True, True, False), None, (1, 1, 2)),
        (("C0.5H1.5", "O2", "CO2", "H2O"), (True, True, False, False), None, (8, 7, 4, 6)),
    ],
)
def test_exact_balance_includes_charge_isotopes_and_fractional_compositions(
    formulas, sides, charges, expected
):
    atoms = [parse_formula(formula) for formula in formulas]
    matrix = stoichiometric_matrix(atoms, sides, charges=charges)
    result = balance_stoichiometry(matrix)
    assert result == expected
    assert all(sum(a * b for a, b in zip(row, result, strict=True)) == 0 for row in matrix)
    fractional = balance_stoichiometry(matrix, allow_fractional=True)
    assert fractional[0] == 1
    assert fractional == tuple(Fraction(value, expected[0]) for value in expected)


def test_exact_balance_does_not_erase_isotope_or_electron_charge():
    matrix = stoichiometric_matrix(
        ({"Fe": 1}, {"Fe": 1}, {}), (True, False, False), charges=(2, 3, -1)
    )
    assert balance_stoichiometry(matrix) == (1, 1, 1)
    with pytest.raises(ValueError, match="explicit nonzero charge"):
        stoichiometric_matrix(({"Fe": 1}, {}), (True, False))
    with pytest.raises(ValueError):
        balance_stoichiometry(stoichiometric_matrix(({"H": 2}, {"D": 2}), (True, False)))
    matrix = stoichiometric_matrix(({"C": Fraction(1, 3)}, {"C": 1}), (True, False))
    assert balance_stoichiometry(matrix) == (3, 1)


def test_underdetermined_and_infeasible_balances_require_explicit_constraints():
    matrix = stoichiometric_matrix(
        [parse_formula(s) for s in ("C", "O2", "CO", "CO2")], (True, True, False, False)
    )
    with pytest.raises(ValueError, match="underdetermined"):
        balance_stoichiometry(matrix)
    assert balance_stoichiometry(matrix, fixed_coefficients={0: 3, 2: 2}) == (3, 2, 2, 1)
    with pytest.raises(ValueError, match="positive balance"):
        balance_stoichiometry(matrix, fixed_coefficients={0: 1, 2: 2})
    with pytest.raises(ValueError, match="inconsistent"):
        balance_stoichiometry(matrix, fixed_coefficients={0: 3, 1: 1, 2: 2})


def test_standard_formation_and_mass_basis_are_explicit():
    compound, coefficients, forms = standard_formation_reaction(parse_formula("CO"))
    assert compound == 2 and coefficients == (2, 1)
    assert forms == ({"C": 1}, {"O": 2})
    compound, coefficients, _ = standard_formation_reaction(
        parse_formula("CO"), allow_fractional=True
    )
    assert compound == 1 and coefficients == (1, Fraction(1, 2))
    atoms = [parse_formula(s) for s in ("H2", "O2", "H2O")]
    weights = list(map(molecular_weight, atoms))
    masses = stoichiometry_molar_to_mass((2, 1, 2), weights)
    assert stoichiometry_mass_to_molar(masses, weights) == pytest.approx((2, 1, 2))
    assert stoichiometry_mass_error((2, 1, 2), weights, (True, True, False)) == pytest.approx(0)
    assert abs(stoichiometry_mass_error((1, 1, 1), weights, (True, True, False))) > 15
    with pytest.raises(ValueError):
        standard_formation_reaction({"D": 1}, elemental_forms={"[2H]": {"H": 2}})


def cp(phase="gas", value=30):
    return PropertyCorrelation(
        "Cp:" + phase,
        "cp_polynomial",
        {"a": value},
        {"temperature": "K"},
        "J/(mol*K)",
        "ideal_gas_heat_capacity" if phase == "gas" else phase + "_heat_capacity",
        {"temperature": (200, 500)},
        "Synthetic constant capacity",
    )


def reference(phase="gas", energy=-1000, entropy=100):
    return CaloricReference(
        phase=phase,
        temperature_K=300,
        pressure_Pa=100000,
        energy_J_mol=energy,
        entropy_J_mol_K=entropy,
    )


def test_formation_gibbs_uses_element_entropy_not_absolute_compound_entropy():
    compound = reference(energy=-1000, entropy=100)
    elements = (reference(energy=0, entropy=20), reference(energy=0, entropy=30))
    gibbs = formation_gibbs_energy(compound, elements, (1, 2))
    assert gibbs == -7000
    assert (
        formation_entropy(
            formation_enthalpy_J_mol=-1000, formation_gibbs_J_mol=gibbs, temperature_K=300
        )
        == 20
    )
    assert (
        formation_enthalpy_basis(
            vaporization_enthalpy_J_mol=44018, liquid_formation_enthalpy_J_mol=-285830
        )
        == -241812
    )
    assert (
        formation_enthalpy_basis(
            vaporization_enthalpy_J_mol=44018, gas_formation_enthalpy_J_mol=-241812
        )
        == -285830
    )
    with pytest.raises(ValueError, match="exactly one"):
        formation_enthalpy_basis(vaporization_enthalpy_J_mol=44018)
    for mismatch in (
        replace(elements[0], pressure_Pa=101325),
        replace(elements[0], temperature_K=298.15),
        replace(elements[0], entropy_J_mol_K=None),
    ):
        with pytest.raises(ValueError, match="matching T/P"):
            formation_gibbs_energy(compound, (mismatch, elements[1]), (1, 2))


def test_same_temperature_phase_reference_conversion_reuses_the_real_phase_path():
    from chemworld.physchem.enthalpy import PhaseTransitionSpec

    latent = PropertyCorrelation(
        "latent",
        "constant_phase_change_enthalpy",
        {"delta_h": 10000},
        {"temperature": "K"},
        "J/mol",
        "heat_of_vaporization",
        {"temperature": (200, 500)},
    )
    transition = PhaseTransitionSpec("boil", "liquid", "gas", 400, latent)
    source = reference()
    curves = {"gas": cp(), "liquid": cp("liquid", 60)}
    liquid = formation_reference_phase_change(
        source,
        component_id="test",
        target_phase="liquid",
        heat_capacity_correlations=curves,
        transitions=(transition,),
    )
    assert liquid.energy_J_mol == -14000
    assert liquid.temperature_K == 300 and liquid.pressure_Pa == 100000
    assert liquid.entropy_J_mol_K is None
    restored = formation_reference_phase_change(
        liquid,
        component_id="test",
        target_phase="gas",
        heat_capacity_correlations=curves,
        transitions=(transition,),
    )
    assert restored.energy_J_mol == source.energy_J_mol


def test_source_correlations_use_the_existing_reaction_thermochemistry_result():
    first = CorrelationSpeciesThermo("A", cp(), reference(energy=0, entropy=10))
    second = CorrelationSpeciesThermo("B", cp(), reference(energy=1000, entropy=20))
    result = reaction_thermochemistry(
        reaction_id="isomerize",
        stoichiometry={"A": -1, "B": 1},
        species_thermo={"A": first, "B": second},
        temperature_K=400,
    )
    assert result.delta_h_J_mol == 1000 and result.delta_s_J_mol_K == pytest.approx(10)
    assert result.delta_g_J_mol == pytest.approx(-3000)
    restored = CorrelationSpeciesThermo.from_dict(json.loads(json.dumps(second.to_dict())))
    assert restored.evaluate(400).entropy_J_mol_K == pytest.approx(20 + 30 * log(4 / 3))
    with pytest.raises(ValueError, match="reference pressure"):
        reaction_thermochemistry(
            reaction_id="invalid",
            stoichiometry={"A": -1, "B": 1},
            species_thermo={
                "A": first,
                "B": replace(second, reference=replace(second.reference, pressure_Pa=101325)),
            },
            temperature_K=400,
        )
