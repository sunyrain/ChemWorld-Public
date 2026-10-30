"""Fixed-width CHEMKIN subset fixtures and source-boundary checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.chemkin_import import (
    chemkin_import_capabilities,
    convert_chemkin,
    import_chemkin,
)
from chemworld.physchem.mechanism_import import _pre_exponential


def _nasa_record(name: str, composition: str) -> str:
    """Build one fixed-width NASA7 record with distinct low/high coefficients."""

    header = [" "] * 80
    header[:18] = name.ljust(18)
    header[24:44] = composition.ljust(20)
    # CHEMKIN stores Tlow, Thigh, Tmid in columns 46--75 (not ascending).
    for start, value in ((45, 200.0), (55, 3000.0), (65, 1000.0)):
        header[start : start + 10] = f"{value:10.3f}"
    header[79] = "1"
    low = [3.5, 1.0e-4, -2.0e-8, 3.0e-12, -4.0e-16, -1200.0, 2.0]
    high = [3.0, 2.0e-4, -3.0e-8, 4.0e-12, -5.0e-16, -1100.0, 2.5]

    def line(values: list[float], number: str) -> str:
        return "".join(f"{value:15.8E}" for value in values).ljust(79) + number

    return "\n".join(
        (
            "".join(header),
            line(high[:5], "2"),
            line(high[5:] + low[:3], "3"),
            line(low[3:], "4"),
        )
    )


def _thermo(*species: tuple[str, str]) -> str:
    return "\n".join(_nasa_record(name, composition) for name, composition in species)


def _mechanism(reactions: str, species: tuple[str, ...] = ("H2", "O2", "H2O")) -> str:
    return "\n".join(
        (
            "ELEMENTS",
            "H O",
            "END",
            "SPECIES",
            " ".join(species),
            "END",
            "REACTIONS CAL/MOLE MOLES",
            reactions,
            "END",
        )
    )


def _mechanism_with_transport(
    reactions: str,
    transport: str,
    species: tuple[str, ...] = ("H2", "O2", "H2O"),
) -> str:
    return "\n".join(
        (
            "ELEMENTS",
            "H O",
            "END",
            "SPECIES",
            " ".join(species),
            "END",
            "TRANSPORT",
            transport,
            "END",
            "REACTIONS CAL/MOLE MOLES",
            reactions,
            "END",
        )
    )


def test_transport_records_are_retained_with_source_units_and_provenance():
    mechanism = _mechanism_with_transport(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4",
        "H2 1 38.0 2.92 0.0 0.79 280.0\n"
        "O2 1 107.4 3.458 0.0 1.6 3.8\n"
        "H2O 2 572.4 2.605 1.844 0.0 4.0",
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert network.metadata["transport_section_present"] is True
    assert network.metadata["transport_records"] == [
        {
            "species": "H2",
            "geometry": 1,
            "geometry_name": "linear",
            "epsilon_over_k_K": 38.0,
            "sigma_A": 2.92,
            "dipole_D": 0.0,
            "polarizability_A3": 0.79,
            "rotational_relaxation": 280.0,
        },
        {
            "species": "O2",
            "geometry": 1,
            "geometry_name": "linear",
            "epsilon_over_k_K": 107.4,
            "sigma_A": 3.458,
            "dipole_D": 0.0,
            "polarizability_A3": 1.6,
            "rotational_relaxation": 3.8,
        },
        {
            "species": "H2O",
            "geometry": 2,
            "geometry_name": "nonlinear",
            "epsilon_over_k_K": 572.4,
            "sigma_A": 2.605,
            "dipole_D": 1.844,
            "polarizability_A3": 0.0,
            "rotational_relaxation": 4.0,
        },
    ]
    transport_header = next(
        item
        for item in network.metadata["source_section_provenance"]
        if item["name"] == "TRANSPORT"
    )
    assert transport_header["header"] == "TRANSPORT"


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        ("H2 1 38.0 2.92 0.0 0.79", "six fields"),
        ("NOPE 1 38.0 2.92 0.0 0.79 280.0", "not declared"),
        ("H2 3 38.0 2.92 0.0 0.79 280.0", "geometry must be 0, 1, or 2"),
        ("H2 1 -1.0 2.92 0.0 0.79 280.0", "epsilon and sigma must be positive"),
        (
            "H2 1 38.0 2.92 nan 0.79 280.0",
            "values must be finite",
        ),
        (
            "H2 1 38.0 2.92 0.0 -0.79 280.0",
            "optional properties must be nonnegative",
        ),
    ],
)
def test_transport_records_fail_closed_on_invalid_source(
    transport: str, message: str
):
    mechanism = _mechanism_with_transport(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4", transport
    )
    with pytest.raises(ValueError, match=message):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_transport_duplicate_species_is_rejected():
    mechanism = _mechanism_with_transport(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4",
        "H2 1 38.0 2.92 0.0 0.79 280.0\n"
        "H2 1 38.0 2.92 0.0 0.79 280.0",
    )
    with pytest.raises(ValueError, match="duplicate CHEMKIN TRANSPORT"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_transport_header_arguments_are_rejected():
    mechanism = _mechanism_with_transport(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4",
        "H2 1 38.0 2.92 0.0 0.79 280.0",
    ).replace("TRANSPORT\n", "TRANSPORT V1\n", 1)
    with pytest.raises(ValueError, match="TRANSPORT section does not accept arguments"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_empty_transport_section_is_recorded_as_present():
    mechanism = _mechanism_with_transport(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4", ""
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert network.metadata["transport_section_present"] is True
    assert network.metadata["transport_records"] == []


@pytest.mark.parametrize("quantity_token", ["MOLE", "MOLES"])
def test_reaction_units_accept_cantera_singular_and_chemkin_plural(quantity_token: str):
    mechanism = _mechanism(
        "2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4"
    ).replace("CAL/MOLE MOLES", f"CAL/MOLE {quantity_token}")
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert len(network.reactions) == 1


@pytest.mark.parametrize("order", [1.0, 2.0])
def test_molecules_header_converts_rate_constant_to_native_mol_l(order: float):
    if order == 1:
        species = ("H2", "H", "H2O")
        equation = "H2 => 2 H"
        thermo = _thermo(("H2", "H 2"), ("H", "H 1"), ("H2O", "H 2 O 1"))
    else:
        species = ("H2", "O2", "H2O", "H2O2")
        equation = "H2 + O2 => H2O2"
        thermo = _thermo(
            ("H2", "H 2"),
            ("O2", "O 2"),
            ("H2O", "H 2 O 1"),
            ("H2O2", "H 2 O 2"),
        )
    mechanism = _mechanism(equation + " 1.0E+0 0.0 0.0", species).replace(
        "CAL/MOLE MOLES", "CAL/MOLE MOLECULES"
    )
    network = convert_chemkin(mechanism, thermo)
    reaction = network.reactions[0]
    expected = 1.0 if order == 1 else AVOGADRO / 1000
    assert reaction.rate_law.parameters["A"] == pytest.approx(expected)
    assert network.metadata["source_quantity_basis"] == "molecule"
    assert network.metadata["source_avogadro_per_mol"] == AVOGADRO


def test_explicit_molecule_preexponential_units_use_exact_avogadro_scaling():
    units = {
        "volume": "cm^3",
        "quantity": "molecule",
        "time": "s",
    }
    assert _pre_exponential("1 cm^3/molecule/s", units, 2.0) == pytest.approx(
        AVOGADRO / 1000
    )


def test_molecules_header_requires_declared_quantity_and_rejects_unknown_mode():
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    missing = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0").replace(
        "CAL/MOLE MOLES", "CAL/MOLE"
    )
    with pytest.raises(ValueError, match="MOLE, MOLES, or MOLECULES"):
        convert_chemkin(missing, thermo)
    unknown = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0").replace(
        "CAL/MOLE MOLES", "CAL/MOLE PARTICLES"
    )
    with pytest.raises(ValueError, match="named reaction sections"):
        convert_chemkin(unknown, thermo)


def test_cantera_yaml2ck_export_roundtrips_real_h2o2_fixture(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )

    network = import_chemkin(mechanism_path, thermo_path)
    assert len(network.species) == 10
    assert len(network.reactions) == 29
    assert network.metadata["import_format"] == "chemkin-gas-mechanism-v1"


def _single_file_mechanism(
    reactions: str,
    species: tuple[str, ...] = ("H2", "O2", "H2O"),
) -> str:
    thermo = _thermo(
        ("H2", "H 2"),
        ("O2", "O 2"),
        ("H2O", "H 2 O 1"),
    )
    return "\n".join(
        (
            "ELEMENTS",
            "H O",
            "END",
            "SPECIES",
            " ".join(species),
            "END",
            "THERMO ALL",
            "200.000 1000.000 5000.000",
            thermo,
            "END",
            "REACTIONS CAL/MOLE MOLES",
            reactions,
            "END",
        )
    )


def test_fixed_width_nasa7_elementary_reaction_roundtrips_native_network():
    network = convert_chemkin(
        _mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4"),
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        source="synthetic.ckin",
    )

    assert len(network.species) == 3
    assert len(network.reactions) == 1
    assert network.reactions[0].reversible
    assert network.reactions[0].rate_law.equation_id == "arrhenius"
    assert network.metadata["import_format"] == "chemkin-gas-mechanism-v1"
    assert network.imported_gas_thermochemistry["H2O"].evaluate(700).cp_J_mol_K > 0


def test_rev_continuation_splits_explicit_forward_and_reverse_rates():
    mechanism = _mechanism(
        "2 H2 + O2 <=> 2 H2O 1.0E+13 0.0 5.0E+4\n"
        "REV / 2.0E+12 0.5 1.0E+3 /"
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert len(network.reactions) == 2
    assert [reaction.reversible for reaction in network.reactions] == [False, False]
    assert [reaction.rate_law.equation_id for reaction in network.reactions] == [
        "arrhenius",
        "arrhenius",
    ]
    assert network.reactions[0].equation.startswith("2 H2 + O2 =>")
    assert network.reactions[1].equation.startswith("2 H2O =>")
    assert network.reactions[1].rate_law.parameters["A"] == pytest.approx(2.0e9)


@pytest.mark.parametrize("reverse_first", [True, False])
def test_rev_third_body_efficiencies_are_mirrored_in_either_order(reverse_first: bool):
    continuations = [
        "REV / 2.0E+12 0.5 1.0E+3 /",
        "H2/2.0/ O2/0.5/",
    ]
    if not reverse_first:
        continuations.reverse()
    mechanism = _mechanism(
        "\n".join(
            (
                "2 H2 + O2 + M = 2 H2O + M 1.0E+13 0.0 5.0E+4",
                *continuations,
            )
        )
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert len(network.reactions) == 2
    assert all(
        reaction.rate_law.equation_id == "third_body_arrhenius" for reaction in network.reactions
    )
    assert all(
        reaction.rate_law.parameters["third_body_efficiencies"] == {"H2": 2.0, "O2": 0.5}
        for reaction in network.reactions
    )


def test_rev_rejects_malformed_and_pressure_dependent_sources():
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    malformed = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\nREV / 1.0 2.0 /")
    with pytest.raises(ValueError, match="REV continuation needs A, b, Ea"):
        convert_chemkin(malformed, thermo)
    falloff = _mechanism(
        "2 H2 (+M) = 2 H2O (+M) 1.0E+3 0.0 0.0\n"
        "LOW / 2.0E+3 0.0 0.0 /\nREV / 1.0E+3 0.0 0.0 /"
    )
    with pytest.raises(ValueError, match="unsupported for pressure-dependent or falloff"):
        convert_chemkin(falloff, thermo)
    for continuation in (
        "PLOG / 1.0 2.0 0.0 0.0 /",
        "CHEB / 2 2 /",
    ):
        mixed = _mechanism(
            "2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\n"
            f"REV / 1.0E+3 0.0 0.0 /\n{continuation}"
        )
        with pytest.raises(ValueError, match="REV cannot be combined"):
            convert_chemkin(mixed, thermo)


def test_ford_continuations_map_forward_custom_orders():
    species = ("A", "B", "C")
    mechanism = _mechanism(
        "\n".join(
            (
                "A => B 1.0E+3 0.0 0.0",
                "FORD / A 0.5 /",
                "FORD / C -0.25 /",
            )
        ),
        species,
    )
    network = convert_chemkin(
        mechanism,
        _thermo(*[(name, "H 1") for name in species]),
    )
    reaction = network.reactions[0]
    assert reaction.reversible is False
    assert reaction.forward_orders == {"A": 0.5, "C": -0.25}
    assert reaction.metadata["custom_orders"] is True
    assert reaction.metadata["negative_orders"] is True


def test_rord_continuations_map_reverse_custom_orders():
    mechanism = _mechanism(
        "2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\nRORD / H2O 0.5 /"
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    reaction = network.reactions[0]
    assert reaction.reversible is True
    assert reaction.reverse_orders == {"H2O": 0.5}
    assert reaction.metadata["custom_reverse_orders"] is True
    assert reaction.metadata["custom_orders"] is False
    rates = network.reaction_rates(
        {"H2": 0.2, "O2": 0.1, "H2O": 0.3}, volume_L=1, temperature_K=700
    )
    assert "reaction-001" in rates


def test_ford_rejects_reversible_duplicate_and_rord_sources():
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    reversible = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\nFORD / H2 0.5 /")
    with pytest.raises(ValueError, match="irreversible reaction"):
        convert_chemkin(reversible, thermo)
    duplicate = _mechanism(
        "2 H2 + O2 => 2 H2O 1.0E+3 0.0 0.0\nFORD / H2 0.5 /\nFORD / H2 0.6 /"
    )
    with pytest.raises(ValueError, match="duplicate CHEMKIN FORD"):
        convert_chemkin(duplicate, thermo)
    rord = _mechanism("2 H2 + O2 => 2 H2O 1.0E+3 0.0 0.0\nRORD / H2 0.5 /")
    with pytest.raises(ValueError, match="RORD"):
        convert_chemkin(rord, thermo)


@pytest.mark.parametrize(
    ("continuation", "message"),
    [
        ("RORD / NO_SUCH_SPECIES 0.5 /", "reverse orders"),
        ("RORD / H2 -0.5 /", "nonnegative"),
    ],
)
def test_rord_rejects_unknown_and_negative_orders(continuation: str, message: str):
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    mechanism = _mechanism(
        f"2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\n{continuation}"
    )
    with pytest.raises(ValueError, match=message):
        convert_chemkin(mechanism, thermo)


@pytest.mark.parametrize(
    "continuation",
    [
        "PLOG / 1.0 2.0E+3 0.0 0.0 /\nRORD / H2O 0.5 /",
        "RORD / H2O 0.5 /\nPLOG / 1.0 2.0E+3 0.0 0.0 /",
        "RORD / H2O 0.5 /\nLOW / 2.0E+3 0.0 0.0 /",
        "RORD / H2O 0.5 /\nTROE / 0.5 100.0 1000.0 3000.0 /",
        "REV / 1.0E+3 0.0 0.0 /\nRORD / H2O 0.5 /",
    ],
)
def test_rord_rejects_pressure_falloff_and_rev_combinations(continuation: str):
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    mechanism = _mechanism(
        "2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\n" + continuation
    )
    with pytest.raises(ValueError, match=r"RORD|REV"):
        convert_chemkin(mechanism, thermo)


def test_cantera_exported_elem_header_is_accepted():
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4").replace(
        "ELEMENTS", "ELEM", 1
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert network.metadata["import_format"] == "chemkin-gas-mechanism-v1"


def test_third_body_and_troe_falloff_are_converted_with_efficiencies():
    species = ("A", "B", "C")
    thermo = _thermo(("A", "H 1"), ("B", "H 1"), ("C", "H 2"))
    mechanism = _mechanism(
        "\n".join(
            (
                "A + B + M = C + M 2.0E+12 0.0 1.0E+3",
                "A + B (+M) = C (+M) 3.0E+10 0.0 1.0E+3",
                "LOW / 2.0E+15 0.0 2.0E+3 /",
                "TROE / 0.5 100.0 1000.0 3000.0 /",
                "A/2.0/ B/0.5/",
                "DUPLICATE",
            )
        ),
        species,
    )

    network = convert_chemkin(mechanism, thermo)
    assert [reaction.rate_law.equation_id for reaction in network.reactions] == [
        "third_body_arrhenius",
        "troe_falloff",
    ]
    assert network.reactions[0].rate_law.parameters["default_efficiency"] == 1
    assert network.reactions[1].metadata["source_declaration"]["duplicate"] is True
    assert network.reactions[1].rate_law.parameters["third_body_efficiencies"] == {
        "A": 2.0,
        "B": 0.5,
    }
    assert network.metadata["duplicate_reaction_count"] == 1


def test_duplicate_reactions_remain_distinct_native_entries():
    mechanism = _mechanism(
        "\n".join(
            (
                "2 H2 + O2 => 2 H2O 1.0E+3 0.0 0.0",
                "2 H2 + O2 => 2 H2O 2.0E+3 0.0 0.0",
                "DUPLICATE",
            )
        )
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert len(network.reactions) == 2
    assert network.reaction_ids == ("reaction-001", "reaction-002")
    assert network.metadata["duplicate_reaction_count"] == 1


def test_plog_continuations_convert_to_native_pressure_log():
    species = ("A", "B", "C")
    mechanism = _mechanism(
        "\n".join(
            (
                "A => B 1.0E+3 0.0 0.0",
                "PLOG / 0.1 1.0E+4 0.0 1.0E+3 /",
                "PLOG / 1.0 2.0E+4 0.0 2.0E+3 /",
            )
        ),
        species,
    )
    network = convert_chemkin(
        mechanism,
        _thermo(*[(name, "H 1") for name in species]),
    )
    reaction = network.reactions[0]
    assert reaction.rate_law.equation_id == "pressure_log"
    assert reaction.rate_law.parameters["pressure_range_Pa"] == pytest.approx(
        [10132.5, 101325.0]
    )
    assert [term["A"] for term in reaction.rate_law.parameters["rate_constants"]] == [
        1.0e4,
        2.0e4,
    ]


def test_real_cantera_h2o2_export_with_plog_replays_through_both_parsers(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    ck2yaml = pytest.importorskip("cantera.ck2yaml")
    cantera = pytest.importorskip("cantera")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )
    lines = mechanism_path.read_text(encoding="utf-8").splitlines()
    modified: list[str] = []
    found = False
    for line in lines:
        modified.append(line)
        if line.strip().startswith("H2 + O <=> H + OH"):
            modified.extend(
                (
                    "PLOG / 0.1 3.0E+10 0.0 1.0E+3 /",
                    "PLOG / 1.0 4.0E+10 0.0 1.0E+3 /",
                )
            )
            found = True
    assert found
    mechanism_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    native = import_chemkin(mechanism_path, thermo_path)
    plog = next(
        reaction for reaction in native.reactions if reaction.rate_law.equation_id == "pressure_log"
    )
    assert len(plog.rate_law.parameters["rate_constants"]) == 2

    yaml_path = tmp_path / "roundtrip.yaml"
    ck2yaml.convert(
        mechanism_path,
        thermo_file=thermo_path,
        out_name=yaml_path,
        quiet=True,
    )
    assert cantera.Solution(str(yaml_path)).n_reactions == 29


def test_real_cantera_h2o2_export_with_rev_replays_through_both_parsers(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    ck2yaml = pytest.importorskip("cantera.ck2yaml")
    cantera = pytest.importorskip("cantera")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )
    lines = mechanism_path.read_text(encoding="utf-8").splitlines()
    modified: list[str] = []
    found = False
    for line in lines:
        modified.append(line)
        if line.strip().startswith("H2 + O <=> H + OH"):
            modified.append("REV / 3.0E+10 0.0 1.0E+3 /")
            found = True
    assert found
    mechanism_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    native = import_chemkin(mechanism_path, thermo_path)
    assert len(native.reactions) == 30
    assert sum(not reaction.reversible for reaction in native.reactions) == 2

    yaml_path = tmp_path / "roundtrip.yaml"
    ck2yaml.convert(
        mechanism_path,
        thermo_file=thermo_path,
        out_name=yaml_path,
        quiet=True,
    )
    assert cantera.Solution(str(yaml_path)).n_reactions == 30


def test_real_cantera_h2o2_export_with_ford_replays_through_both_parsers(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    ck2yaml = pytest.importorskip("cantera.ck2yaml")
    cantera = pytest.importorskip("cantera")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )
    lines = mechanism_path.read_text(encoding="utf-8").splitlines()
    modified: list[str] = []
    found = False
    for line in lines:
        if line.strip().startswith("H2 + O <=> H + OH"):
            line = line.replace("<=>", "=>", 1)
            modified.append(line)
            modified.append("FORD / H2 0.5 /")
            found = True
        else:
            modified.append(line)
    assert found
    mechanism_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    native = import_chemkin(mechanism_path, thermo_path)
    custom = next(
        reaction for reaction in native.reactions if reaction.forward_orders.get("H2") == 0.5
    )
    assert custom.reversible is False

    yaml_path = tmp_path / "roundtrip.yaml"
    ck2yaml.convert(
        mechanism_path,
        thermo_file=thermo_path,
        out_name=yaml_path,
        quiet=True,
    )
    assert cantera.Solution(str(yaml_path)).n_reactions == 29


def test_real_cantera_h2o2_export_with_rord_replays_native_import(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    ck2yaml = pytest.importorskip("cantera.ck2yaml")
    cantera = pytest.importorskip("cantera")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )
    lines = mechanism_path.read_text(encoding="utf-8").splitlines()
    modified: list[str] = []
    found = False
    for line in lines:
        modified.append(line)
        if line.strip().startswith("H2 + O <=> H + OH"):
            modified.append("RORD / H 0.5 /")
            found = True
    assert found
    mechanism_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    native = import_chemkin(mechanism_path, thermo_path)
    custom = next(
        reaction
        for reaction in native.reactions
        if reaction.metadata.get("custom_reverse_orders")
    )
    assert custom.reversible is True
    assert custom.reverse_orders["H"] == pytest.approx(0.5)

    # Cantera's current ck2yaml parser does not implement RORD and treats the
    # slash payload as an Arrhenius triple; preserve that provider boundary.
    with pytest.raises(cantera.ck2yaml.InputError, match="could not convert string to float"):
        ck2yaml.convert(
            mechanism_path,
            thermo_file=thermo_path,
            out_name=tmp_path / "roundtrip.yaml",
            quiet=True,
        )


def test_cheb_continuations_convert_to_native_pressure_polynomial():
    species = ("A", "B", "C")
    mechanism = _mechanism(
        "\n".join(
            (
                "A => B 1.0E+3 0.0 0.0",
                "TCHEB / 300.0 2500.0 /",
                "PCHEB / 0.1 10.0 /",
                "CHEB / 2 2 /",
                "CHEB / 1.0 0.1 /",
                "CHEB / 0.2 -0.3 /",
            )
        ),
        species,
    )
    network = convert_chemkin(
        mechanism,
        _thermo(*[(name, "H 1") for name in species]),
    )
    reaction = network.reactions[0]
    assert reaction.rate_law.equation_id == "chebyshev"
    assert reaction.rate_law.parameters["temperature_range_K"] == [300.0, 2500.0]
    assert reaction.rate_law.parameters["pressure_range_Pa"] == pytest.approx(
        [10132.5, 1013250.0]
    )
    assert len(reaction.rate_law.parameters["coefficients"]) == 2


def test_real_cantera_h2o2_export_with_cheb_replays_through_both_parsers(tmp_path: Path):
    yaml2ck = pytest.importorskip("cantera.yaml2ck")
    ck2yaml = pytest.importorskip("cantera.ck2yaml")
    cantera = pytest.importorskip("cantera")
    fixture = Path(__file__).parent / "fixtures" / "mechanisms" / "cantera-3.1.0" / "h2o2.yaml"
    mechanism_path = tmp_path / "h2o2.inp"
    thermo_path = tmp_path / "thermo.dat"
    yaml2ck.convert(
        fixture,
        phase_name="ohmech",
        mechanism_path=mechanism_path,
        thermo_path=thermo_path,
        overwrite=True,
    )
    lines = mechanism_path.read_text(encoding="utf-8").splitlines()
    modified: list[str] = []
    found = False
    for line in lines:
        modified.append(line)
        if line.strip().startswith("H2 + O <=> H + OH"):
            modified.extend(
                (
                    "TCHEB / 300.0 2500.0 /",
                    "PCHEB / 0.1 10.0 /",
                    "CHEB / 2 2 /",
                    "CHEB / 1.0 0.1 /",
                    "CHEB / 0.2 -0.3 /",
                )
            )
            found = True
    assert found
    mechanism_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    native = import_chemkin(mechanism_path, thermo_path)
    cheb = next(
        reaction for reaction in native.reactions if reaction.rate_law.equation_id == "chebyshev"
    )
    assert len(cheb.rate_law.parameters["coefficients"]) == 2

    yaml_path = tmp_path / "roundtrip.yaml"
    ck2yaml.convert(
        mechanism_path,
        thermo_file=thermo_path,
        out_name=yaml_path,
        quiet=True,
    )
    assert cantera.Solution(str(yaml_path)).n_reactions == 29


def test_cheb_rejects_incomplete_or_malformed_matrix():
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    missing_row = _mechanism(
        "\n".join(
            (
                "2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0",
                "TCHEB / 300.0 2500.0 /",
                "PCHEB / 0.1 10.0 /",
                "CHEB / 2 2 /",
                "CHEB / 1.0 0.1 /",
            )
        )
    )
    with pytest.raises(ValueError, match="incomplete rows"):
        convert_chemkin(missing_row, thermo)
    wrong_width = missing_row.replace("CHEB / 1.0 0.1 /", "CHEB / 1.0 0.1 0.2 /")
    with pytest.raises(ValueError, match="wrong width"):
        convert_chemkin(wrong_width, thermo)


def test_plog_rejects_malformed_and_third_body_continuations():
    thermo = _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1"))
    malformed = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\nPLOG / 1.0 2.0 0.0 /")
    with pytest.raises(ValueError, match="PLOG continuation needs P, A, b, Ea"):
        convert_chemkin(malformed, thermo)
    collider = _mechanism("H2 + O2 + M = H2O + M 1.0E+3 0.0 0.0\nPLOG / 1.0 2.0 0.0 0.0 /")
    with pytest.raises(ValueError, match="third-body collider"):
        convert_chemkin(collider, thermo)


def test_embedded_thermo_single_file_is_supported():
    network = convert_chemkin(
        _single_file_mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4"),
    )
    assert network.metadata["thermo_source"] == "embedded THERMO section"
    assert len(network.species) == 3


def test_import_embedded_thermo_requires_no_companion_file(tmp_path: Path):
    mechanism_path = tmp_path / "embedded.ckin"
    mechanism_path.write_text(
        _single_file_mechanism("2 H2 + O2 = 2 H2O 1.0E+4 0.0 0.0"),
        encoding="utf-8",
    )
    network = import_chemkin(mechanism_path)
    assert network.metadata["source_mechanism"] == "embedded.ckin; embedded THERMO"


def test_import_expands_relative_nested_include_and_records_sources(tmp_path: Path):
    nested = tmp_path / "parts" / "mechanism.ckin"
    nested.parent.mkdir()
    nested.write_text(_single_file_mechanism("2 H2 + O2 = 2 H2O 1.0E+4 0.0 0.0"), encoding="utf-8")
    root = tmp_path / "root.ckin"
    root.write_text("INCLUDE 'parts/mechanism.ckin'\n", encoding="utf-8")

    network = import_chemkin(root)
    assert len(network.species) == 3
    assert network.metadata["included_source_files"] == (str(nested.resolve()),)


def test_import_expands_companion_thermo_include_and_records_nested_source(tmp_path: Path):
    mechanism_path = tmp_path / "mechanism.ckin"
    thermo_path = tmp_path / "thermo.dat"
    nested_thermo = tmp_path / "thermo" / "species.dat"
    nested_thermo.parent.mkdir()
    nested_thermo.write_text(
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        encoding="utf-8",
    )
    mechanism_path.write_text(
        _mechanism("2 H2 + O2 = 2 H2O 1.0E+4 0.0 0.0"), encoding="utf-8"
    )
    thermo_path.write_text("INCLUDE <thermo/species.dat>\n", encoding="utf-8")

    network = import_chemkin(mechanism_path, thermo_path)
    assert network.metadata["thermo_source"] == "companion NASA7 text"
    assert network.metadata["included_source_files"] == (str(nested_thermo.resolve()),)
    assert len(network.imported_gas_thermochemistry) == 3


def test_import_rejects_missing_and_cyclic_include(tmp_path: Path):
    missing = tmp_path / "missing.ckin"
    missing.write_text("INCLUDE no-such-file.ckin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        import_chemkin(missing)

    first = tmp_path / "first.ckin"
    second = tmp_path / "second.ckin"
    first.write_text("INCLUDE second.ckin\n", encoding="utf-8")
    second.write_text("INCLUDE first.ckin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic CHEMKIN INCLUDE"):
        import_chemkin(first)


def test_missing_companion_or_embedded_thermo_is_explicitly_rejected():
    with pytest.raises(ValueError, match="companion NASA7 thermo text or embedded THERMO"):
        convert_chemkin(_mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0"))


def test_single_named_reaction_section_header_is_consumed_and_recorded():
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0").replace(
        "REACTIONS CAL/MOLE MOLES", "REACTIONS combustion_section CAL/MOLE MOLES MWOFF"
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert network.metadata["reaction_section_name"] == "combustion_section"
    assert network.metadata["reaction_molecular_weight_mode"] == "MWOFF"
    provenance = network.metadata["source_section_provenance"][-1]
    assert provenance["section_name"] == "combustion_section"
    assert provenance["arguments"] == ["combustion_section", "CAL/MOLE", "MOLES", "MWOFF"]


def test_named_reaction_section_requires_explicit_units():
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0").replace(
        "REACTIONS CAL/MOLE MOLES", "REACTIONS combustion_section"
    )
    with pytest.raises(ValueError, match="activation-energy"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("REACTIONS combustion_section CAL/MOLE MOLES EXTRA", "unknown REACTIONS"),
        ("REACTIONS combustion_section CAL/MOLE MOLES CAL/MOLE", "repeat or mix"),
        ("REACTIONS combustion_section CAL/MOLE MOLES MOLES", "quantity units"),
        ("REACTIONS combustion_section CAL/MOLE MOLES MWOFF MWON", "MW modes"),
    ],
)
def test_named_reaction_section_header_ambiguity_fails_closed(
    header: str, message: str
):
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0").replace(
        "REACTIONS CAL/MOLE MOLES", header
    )
    with pytest.raises(ValueError, match=message):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_named_reaction_label_is_rejected_explicitly():
    mechanism = _mechanism("R1: 2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0")
    with pytest.raises(ValueError, match="named reaction sections"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_reaction_label_without_equation_is_rejected_as_named_source():
    mechanism = _mechanism("R1: a vendor-only reaction heading")
    with pytest.raises(ValueError, match="named reaction sections"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_duplicate_core_section_is_rejected_before_conversion():
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0")
    duplicate = mechanism + "\nREACTIONS CAL/MOLE MOLES\nEND\n"
    with pytest.raises(ValueError, match="duplicate REACTIONS section"):
        convert_chemkin(
            duplicate,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_accepted_source_records_section_provenance():
    network = convert_chemkin(
        _mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0"),
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    provenance = network.metadata["source_section_provenance"]
    assert [item["name"] for item in provenance] == ["ELEMENTS", "SPECIES", "REACTIONS"]
    assert provenance[-1]["arguments"] == ["CAL/MOLE", "MOLES"]
    assert provenance[-1]["line"] == 7


def test_malformed_rord_continuation_rejects_whole_import():
    mechanism = _mechanism(
        "2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0\nRORD / H2O /"
    )
    with pytest.raises(ValueError, match="RORD continuation"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_import_reads_local_mechanism_and_thermo_files(tmp_path: Path):
    mechanism_path = tmp_path / "minimal.ckin"
    thermo_path = tmp_path / "minimal.thermo"
    mechanism_path.write_text(
        _mechanism("2 H2 + O2 = 2 H2O 1.0D+4 0.0 0.0"), encoding="utf-8"
    )
    thermo_path.write_text(
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        encoding="utf-8",
    )
    assert import_chemkin(mechanism_path, thermo_path).metadata["import_format"] == (
        "chemkin-gas-mechanism-v1"
    )


def test_capabilities_are_json_ready_and_keep_explicit_boundaries():
    capabilities = chemkin_import_capabilities()
    assert "NASA7" in capabilities["thermo_models"]
    assert "RORD reverse custom reaction orders" in capabilities["options"]
    assert all("RORD" not in item for item in capabilities["unsupported"])
    assert capabilities["source_section_provenance"]["metadata_key"] == (
        "source_section_provenance"
    )
    assert capabilities["named_section_policy"]["reaction_labels"] == (
        "reject whole mechanism"
    )
    assert capabilities["whole_mechanism_policy"] == "all species and reactions converted or reject"
