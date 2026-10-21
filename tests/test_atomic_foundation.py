"""Behavioral coverage for native atomic identities and formula inputs."""

import json
import subprocess
import sys

import pytest

from chemworld.physchem import (
    ComponentSpec,
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
    atom_fractions,
    atomic_data_provenance,
    cas_to_int,
    check_cas_number,
    element_matrix,
    formula_charge,
    get_element,
    get_isotope,
    hill_formula,
    index_hydrogen_deficiency,
    int_to_cas,
    isotope_data,
    mass_fractions_from_formula,
    mixture_atomic_composition,
    molecular_weight,
    parse_formula,
    serialize_formula,
    similarity_variable,
    sorted_cas_key,
    validate_cas_number,
)
from chemworld.physchem.atomic_data import AtomicEstimate, element_data


def test_full_periodic_table_preserves_missing_weights_and_source_intervals():
    assert {item.atomic_number for item in element_data().values()} == set(range(1, 119))
    assert get_element(118).symbol == "Og"
    assert get_element("aluminium") == get_element("Al")
    assert get_element("caesium").symbol == "Cs"
    assert get_element("sulphur").symbol == "S"
    assert get_element("26").symbol == "Fe"
    assert get_element("Tc").atomic_weight_g_mol is None
    with pytest.raises(ValueError, match="No standard atomic weight"):
        molecular_weight(parse_formula("TcO2"))
    assert molecular_weight(parse_formula("[99Tc]O2")) > 130
    assert get_element("Ar").atomic_weight_g_mol == 39.948
    assert get_element("Ar").standard_weight.value == 39.95
    assert get_element("Ar").weight_source == "chemworld-original-26"
    assert get_element("Xe").weight_source == "CIAAW2021"
    assert get_element("H").standard_weight.interval == (1.00784, 1.00811)
    assert get_element("Pb").standard_weight.standard_uncertainty == 1.1


@pytest.mark.parametrize("identity", [0, 119, True, 6.5, "unknown", None])
def test_invalid_element_identity(identity):
    with pytest.raises(ValueError, match="identity"):
        get_element(identity)


def test_isotope_uncertainty_abundance_and_immutability():
    assert len(isotope_data()) == 2939
    assert get_isotope("D") == get_isotope("H", 2)
    assert get_isotope("T") == get_isotope("[3H]")
    assert get_isotope("[13C]").mass.value == 13.00335483534
    assert get_isotope("[13C]").mass.standard_uncertainty == 2.5e-10
    hydrogen = get_isotope("[1H]").abundance
    assert hydrogen.value is None
    assert hydrogen.interval == (0.99972, 0.99999)
    assert hydrogen.available
    assert get_isotope("T").abundance == AtomicEstimate(None)
    assert not get_isotope("T").abundance.available
    assert get_isotope("[5Be]").estimated_mass
    nitrogen = get_isotope("[14N]").abundance
    assert nitrogen.value == 0.996337
    assert nitrogen.interval == (0.99578, 0.99663)
    assert nitrogen.standard_uncertainty == 0.000004
    with pytest.raises(TypeError):
        isotope_data()["[13C]"] = None
    provenance = atomic_data_provenance()
    assert provenance["abundance_records"] == 289
    provenance["abundance_records"] = 0
    assert atomic_data_provenance()["abundance_records"] == 289


@pytest.mark.parametrize("identity", ["[999C]", "13C", "[0C]", None, 13])
def test_invalid_isotope_identity(identity):
    with pytest.raises(ValueError):
        get_isotope(identity)


@pytest.mark.parametrize("mass_number", [True, 13.5, "13", 0])
def test_invalid_isotope_mass_number(mass_number):
    with pytest.raises(ValueError):
        get_isotope("C", mass_number)


@pytest.mark.parametrize(
    "formula",
    [
        "",
        "()",
        "H0",
        "H-0",
        "H+0",
        "H2(O",
        "H2)O",
        "[OH)",
        "{}",
        "H2 O",
        "H2..O",
        "H2O·",
        "·H2O",
        "H2O··H2O",
        "Xx",
        "[999C]",
        "H+-",
        "H2e3",
        "H(0+)",
        "H^0+",
        "H2O.",
    ],
)
def test_invalid_formula_is_rejected(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


def test_decimal_hydrate_and_charge_ambiguities_are_explicit():
    assert parse_formula("CuSO4.5H2O") == {"Cu": 1, "S": 1, "O": 5.5, "H": 2}
    assert parse_formula("CuSO4·5H2O") == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert parse_formula("Fe3+") == {"Fe": 3}
    assert formula_charge("Fe3+") == 1
    assert parse_formula("Fe^3+") == {"Fe": 1}
    assert formula_charge("Fe^3+") == 3
    assert parse_formula("2[Fe{CN}6]") == {"Fe": 2, "C": 12, "N": 12}
    assert serialize_formula("C0.00000001H0.00000002") == "C0.00000001H0.00000002"
    assert parse_formula(serialize_formula("Pd(NH3)4.0001+2")) == parse_formula("Pd(NH3)4.0001+2")


def test_component_charge_inference_and_json_roundtrip():
    component = ComponentSpec("heavy_ammonium", "[15N]D4+")
    assert component.charge == 1
    assert component.composition == {"[15N]": 1, "[2H]": 4}
    assert ComponentSpec.from_dict(json.loads(json.dumps(component.to_dict()))) == component
    assert ComponentSpec("legacy_sodium", "Na", charge=1).charge == 1
    assert ComponentSpec("water", "H2O").charge == 0
    with pytest.raises(ValueError, match="conflicts"):
        ComponentSpec("conflict", "SO4-2", charge=-1)
    for charge in [True, "1", 1.5]:
        with pytest.raises(ValueError, match="integer"):
            ComponentSpec.from_dict({"identifier": "bad", "formula": "Na+", "charge": charge})


def test_isotopic_conservation_and_composition_consumers():
    heavy, normal = parse_formula("D2O"), parse_formula("H2O")
    matrix, labels = element_matrix([normal, heavy])
    assert labels == ("H", "[2H]", "O")
    assert matrix == ((2, 0, 1), (0, 2, 1))
    matrix, labels = element_matrix([heavy], ("O", "D"))
    assert labels == ("O", "[2H]")
    assert matrix == ((1, 2),)
    with pytest.raises(ValueError, match="duplicate"):
        element_matrix([heavy], ("D", "[2H]"))
    mixed = mixture_atomic_composition([normal, heavy], [0.25, 0.75])
    assert mixed == {"H": 0.5, "[2H]": 1.5, "O": 1}
    assert molecular_weight(mixed) == pytest.approx(
        0.25 * molecular_weight(normal) + 0.75 * molecular_weight(heavy)
    )
    assert atom_fractions(mixed) == pytest.approx({"H": 1 / 6, "[2H]": 0.5, "O": 1 / 3})
    assert sum(mass_fractions_from_formula(mixed).values()) == pytest.approx(1)
    assert hill_formula(parse_formula("D2[13C]C2H4O")) == "C2[13C]H4[2H]2O"
    assert index_hydrogen_deficiency(parse_formula("[13C]6D6")) == 4
    assert index_hydrogen_deficiency(parse_formula("C2H4Cl2")) == 0
    assert similarity_variable(normal) == pytest.approx(3 / 18.015)
    with pytest.raises(ValueError, match="hydrogen deficiency"):
        index_hydrogen_deficiency(parse_formula("FeO"))
    with pytest.raises(ValueError, match="sum to one"):
        mixture_atomic_composition([normal, heavy], [0.25, 0.25])


def test_reaction_network_rejects_transmutation_but_accepts_isotope_exchange():
    rate = RateLawSpec("exchange", "mass_action", {"k": 0.1})
    species = (
        SpeciesSpec("light", "H2O"),
        SpeciesSpec("heavy", "D2O"),
        SpeciesSpec("mixed", "HDO"),
    )
    with pytest.raises(ValueError, match="not element balanced"):
        ReactionNetworkSpec(
            "transmutation",
            species,
            (
                ReactionSpec.from_equation(
                    reaction_id="bad", equation="light => heavy", rate_law=rate
                ),
            ),
        )
    network = ReactionNetworkSpec(
        "exchange",
        species,
        (
            ReactionSpec.from_equation(
                reaction_id="good", equation="light + heavy => 2 mixed", rate_law=rate
            ),
        ),
    )
    assert network.check_element_balance()
    assert network.element_matrix()[1] == ("H", "[2H]", "O")


@pytest.mark.parametrize("count", [-1, True, float("inf"), float("nan")])
def test_invalid_compositions(count):
    with pytest.raises(ValueError, match="atom counts"):
        molecular_weight({"C": count})


@pytest.mark.parametrize("cas", ["64-17-5", "7732-18-5", "1234567-89-5"])
def test_cas_conversion_and_checksum(cas):
    assert check_cas_number(cas)
    assert validate_cas_number(cas) == cas
    assert int_to_cas(cas_to_int(cas)) == cas


@pytest.mark.parametrize(
    "cas",
    [
        "7732-18-4",
        "064-17-5",
        "64-1-5",
        "64-17-55",
        " 64-17-5",
        "\uff16\uff14-\uff11\uff17-\uff15",
        "4-17-5",
        "12345678-90-1",
        64175,
        True,
        None,
    ],
)
def test_invalid_cas(cas):
    assert not check_cas_number(cas)
    with pytest.raises(ValueError, match="cas_number"):
        cas_to_int(cas)


@pytest.mark.parametrize("number", [True, -1, 64175.0, "64175", 1, 7732184])
def test_invalid_integer_cas(number):
    with pytest.raises(ValueError):
        int_to_cas(number)


def test_cas_ordering_is_numeric():
    assert sorted_cas_key(["7732-18-5", "64-17-5"]) == ("64-17-5", "7732-18-5")


def test_isotope_resources_are_lazy_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from chemworld.physchem import ComponentSpec, get_isotope
from chemworld.physchem.atomic_data import isotope_data
assert isotope_data.cache_info().currsize == 0
ComponentSpec('water', 'H2O')
assert isotope_data.cache_info().currsize == 0
get_isotope('D')
assert isotope_data.cache_info().currsize == 1
assert 'chemicals' not in sys.modules and 'periodictable' not in sys.modules
print('lazy atomic resources OK')
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy atomic resources OK" in result.stdout
